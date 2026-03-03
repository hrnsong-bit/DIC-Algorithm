"""
ADSS-DIC v2 (Adaptive Subset-Subdivision) 래퍼 모듈

사분면(Q5~Q8) 복수 채택 방식:
    - 각 불량 POI에서 threshold를 넘는 모든 사분면에 IC-GN 수행
    - 결과를 ADSSResult로 반환 (sub-POI 복수 저장)
    - 대표값(최대 ZNCC)을 ICGNResult에 반영하여 기존 파이프라인 호환

파이프라인: FFT-CC → IC-GN → ADSS v2 재계산 → PLS 변형률

References:
    Zhao, J. & Pan, B. "Adaptive subset-subdivision for automatic
    digital image correlation calculation on discontinuous shape
    and deformation." Applied Optics, 2025.
"""

import numpy as np
import time
import logging

from .results import ADSSResult, ICGN_SUCCESS
from .shape_function_numba import AFFINE, QUADRATIC, get_num_params
from .variable_subset import _detect_grid_structure
from .adss_subset_numba import (
    Q5, Q6, Q7, Q8,
    process_bad_pois_adss_multi_parallel,
    allocate_adss_multi_batch_buffers,
)

logger = logging.getLogger(__name__)


def _build_neighbor_info_adss_v2(
    bad_indices, points_x, points_y,
    valid_mask, zncc_values, parameters,
    zncc_threshold, ny, nx, n_params
):
    """
    ADSS-DIC v2용 이웃 정보 구성 — 대각선 4방위만 (Q5~Q8).

    Returns
    -------
    all_neighbor_valid : (n_bad, 4) bool
    all_neighbor_params : (n_bad, 4, n_params) float64
    all_neighbor_x : (n_bad, 4) float64
    all_neighbor_y : (n_bad, 4) float64
    """
    n_bad = len(bad_indices)

    all_neighbor_valid = np.zeros((n_bad, 4), dtype=np.bool_)
    all_neighbor_params = np.zeros((n_bad, 4, n_params), dtype=np.float64)
    all_neighbor_x = np.zeros((n_bad, 4), dtype=np.float64)
    all_neighbor_y = np.zeros((n_bad, 4), dtype=np.float64)

    # Q5~Q8 대응 대각선 이웃 방향 (dy, dx)
    neighbor_dirs = [
        (-1, -1),  # Q5: 좌상 이웃
        (-1,  1),  # Q6: 우상 이웃
        ( 1, -1),  # Q7: 좌하 이웃
        ( 1,  1),  # Q8: 우하 이웃
    ]

    for k in range(n_bad):
        flat_idx = bad_indices[k]
        iy = flat_idx // nx
        ix = flat_idx % nx

        for d, (dy, dx) in enumerate(neighbor_dirs):
            niy = iy + dy
            nix = ix + dx

            if 0 <= niy < ny and 0 <= nix < nx:
                nf = niy * nx + nix
                if valid_mask[nf] and zncc_values[nf] >= zncc_threshold:
                    all_neighbor_valid[k, d] = True
                    all_neighbor_params[k, d, :n_params] = parameters[nf, :n_params]
                    all_neighbor_x[k, d] = float(points_x[nf])
                    all_neighbor_y[k, d] = float(points_y[nf])

    return all_neighbor_valid, all_neighbor_params, all_neighbor_x, all_neighbor_y


def compute_adss_recalc(
    ref_image, grad_x, grad_y,
    coeffs, order,
    points_x, points_y,
    valid_mask, zncc_values, parameters,
    convergence_flags, iteration_counts, failure_reasons,
    subset_size,
    max_iterations=50,
    convergence_threshold=0.001,
    shape_function='affine',
    zncc_threshold=0.9,
    zncc_pre_threshold=0.5,
):
    """
    ADSS-DIC v2 재계산 — 사분면 복수 채택.

    Returns
    -------
    ADSSResult
        모든 sub-POI가 포함된 결과 객체.
        대표값은 ICGNResult 배열에 in-place 반영됨.
    """
    t_start = time.time()
    n_poi = len(points_x)

    if shape_function == 'affine':
        shape_type = AFFINE
    else:
        shape_type = QUADRATIC
    n_params = get_num_params(shape_type)

    M = subset_size // 2

    # === 빈 결과 템플릿 ===
    def _empty_adss(n_bad=0, elapsed=0.0):
        return ADSSResult(
            parent_indices=np.array([], dtype=np.int64),
            quarter_types=np.array([], dtype=np.int32),
            points_x=np.array([], dtype=np.int64),
            points_y=np.array([], dtype=np.int64),
            parameters=np.zeros((0, n_params), dtype=np.float64),
            zncc_values=np.array([], dtype=np.float64),
            iterations=np.array([], dtype=np.int32),
            xsi_mins=np.array([], dtype=np.int32),
            xsi_maxs=np.array([], dtype=np.int32),
            eta_mins=np.array([], dtype=np.int32),
            eta_maxs=np.array([], dtype=np.int32),
            n_bad_original=n_bad,
            n_sub_total=0,
            n_parent_recovered=0,
            n_unrecoverable=n_bad,
            elapsed_time=elapsed,
        )

    # === 1. 불량 POI 식별 ===
    bad_mask = ~valid_mask | (zncc_values < zncc_threshold)
    bad_indices = np.where(bad_mask)[0].astype(np.int64)
    n_bad = len(bad_indices)

    if n_bad == 0:
        logger.info("ADSS-DIC v2: 불량 POI 없음 — 스킵 (%.3fs)", time.time() - t_start)
        return _empty_adss(0, time.time() - t_start)

    logger.info("ADSS-DIC v2: %d개 불량 POI 감지, 사분면 복수 채택 시작...", n_bad)

    # === 2. 격자 구조 감지 ===
    grid_info = _detect_grid_structure(points_x, points_y)
    if grid_info is None:
        logger.warning("ADSS-DIC v2: 격자 구조 감지 실패 — 스킵")
        return _empty_adss(n_bad, time.time() - t_start)

    ny, nx, spacing = grid_info
    logger.info("  격자 구조: %d×%d, 간격=%dpx", ny, nx, spacing)

    # === 3. 이웃 정보 구성 (대각선 4방위만) ===
    all_nv, all_np, all_nx, all_ny = _build_neighbor_info_adss_v2(
        bad_indices, points_x, points_y,
        valid_mask, zncc_values, parameters,
        zncc_threshold, ny, nx, n_params
    )

    has_candidate = np.any(all_nv, axis=1)
    n_candidates = int(np.sum(has_candidate))

    if n_candidates == 0:
        logger.info("ADSS-DIC v2: 시도 가능한 후보 없음")
        return _empty_adss(n_bad, time.time() - t_start)

    candidate_mask = has_candidate
    cand_indices = bad_indices[candidate_mask]
    cand_nv = all_nv[candidate_mask]
    cand_np_ = all_np[candidate_mask]
    cand_nx_ = all_nx[candidate_mask]
    cand_ny_ = all_ny[candidate_mask]
    n_cand = len(cand_indices)

    logger.info("  시도 가능 후보: %d / %d", n_cand, n_bad)

    # === 4. 버퍼 할당 ===
    n_pixels_max = (M + 1) * (M + 1)  # 사분면 크기
    bufs = allocate_adss_multi_batch_buffers(n_cand, n_pixels_max, n_params)

    max_sub = 4 * n_cand
    result_p = np.zeros((max_sub, n_params), dtype=np.float64)
    result_zncc = np.zeros(max_sub, dtype=np.float64)
    result_iter = np.zeros(max_sub, dtype=np.int32)
    result_qt = np.zeros(max_sub, dtype=np.int32)
    result_parent = np.full(max_sub, -1, dtype=np.int64)
    result_count = np.zeros(n_cand, dtype=np.int32)
    result_candidate_zncc = np.full((n_cand, 4), -1.0, dtype=np.float64)

    # === 5. 배치 병렬 처리 ===
    process_bad_pois_adss_multi_parallel(
        ref_image, grad_x, grad_y,
        coeffs, order,
        points_x, points_y,
        subset_size, max_iterations, convergence_threshold,
        shape_type,
        cand_indices,
        cand_nv, cand_np_, cand_nx_, cand_ny_,
        zncc_threshold, zncc_pre_threshold,
        result_p, result_zncc, result_iter, result_qt,
        result_parent, result_count, result_candidate_zncc,
        bufs['f'], bufs['dfdx'], bufs['dfdy'],
        bufs['J'], bufs['H'], bufs['H_inv'],
        bufs['p'], bufs['xsi_w'], bufs['eta_w'],
        bufs['x_def'], bufs['y_def'],
        bufs['g'], bufs['b'], bufs['dp'], bufs['p_new'],
        bufs['xsi_local'], bufs['eta_local'],
        bufs['init_params'],
        bufs['out_p'], bufs['out_zncc'], bufs['out_iter'],
        bufs['out_qt'], bufs['out_cand_zncc'],
    )

    # === 6. 유효 결과 추출 ===
    valid_sub_mask = result_parent >= 0
    n_sub_total = int(np.sum(valid_sub_mask))

    if n_sub_total == 0:
        logger.info("ADSS-DIC v2: 복원된 sub-POI 없음")
        return _empty_adss(n_bad, time.time() - t_start)

    sub_parent = result_parent[valid_sub_mask]
    sub_qt = result_qt[valid_sub_mask]
    sub_p = result_p[valid_sub_mask]
    sub_zncc = result_zncc[valid_sub_mask]
    sub_iter = result_iter[valid_sub_mask]

    # 사분면 영역 정보 생성
    sub_xsi_min = np.zeros(n_sub_total, dtype=np.int32)
    sub_xsi_max = np.zeros(n_sub_total, dtype=np.int32)
    sub_eta_min = np.zeros(n_sub_total, dtype=np.int32)
    sub_eta_max = np.zeros(n_sub_total, dtype=np.int32)

    qt_to_range = {
        5: (-M, 0, -M, 0),   # Q5: Upper-left
        6: (0, M, -M, 0),    # Q6: Upper-right
        7: (-M, 0, 0, M),    # Q7: Lower-left
        8: (0, M, 0, M),     # Q8: Lower-right
    }

    for i in range(n_sub_total):
        qt = int(sub_qt[i])
        if qt in qt_to_range:
            sub_xsi_min[i], sub_xsi_max[i], sub_eta_min[i], sub_eta_max[i] = qt_to_range[qt]

    sub_px = np.array([int(points_x[pi]) for pi in sub_parent], dtype=np.int64)
    sub_py = np.array([int(points_y[pi]) for pi in sub_parent], dtype=np.int64)

    # === 7. 대표값 ICGNResult 반영 (기존 파이프라인 호환) ===
    unique_parents = np.unique(sub_parent)
    n_parent_recovered = len(unique_parents)
    n_unrecoverable = n_bad - n_parent_recovered

    # === 8. candidate_zncc를 bad_indices 전체로 매핑 ===
    all_cand_zncc = np.full((n_bad, 4), -1.0, dtype=np.float64)
    for k in range(n_cand):
        orig_k = np.searchsorted(bad_indices, cand_indices[k])
        if orig_k < n_bad:
            all_cand_zncc[orig_k, :] = result_candidate_zncc[k, :]

    elapsed = time.time() - t_start

    # === 9. 로그 출력 ===
    quarter_names = {5: 'Q5:UL', 6: 'Q6:UR', 7: 'Q7:LL', 8: 'Q8:LR'}

    for k in range(n_cand):
        fidx = cand_indices[k]
        n_rec = result_count[k]
        cz = result_candidate_zncc[k]

        # 이 POI에서 복원된 사분면 목록
        offset = k * 4
        rec_qts = []
        for j in range(n_rec):
            rec_qts.append(quarter_names.get(result_qt[offset + j], '?'))

        zncc_strs = []
        qn = ['Q5:UL', 'Q6:UR', 'Q7:LL', 'Q8:LR']
        for i in range(4):
            if cz[i] < 0:
                zncc_strs.append(f"{qn[i]}:N/A")
            else:
                zncc_strs.append(f"{qn[i]}:{cz[i]:.4f}")

        logger.info(
            "  POI[%d] (%d,%d) → %d개 복원 %s | 1-warp: %s",
            fidx, points_x[fidx], points_y[fidx],
            n_rec, rec_qts,
            " | ".join(zncc_strs)
        )

    logger.info(
        "ADSS-DIC v2 완료: %d개 불량 → %d개 부모 복원 (%d sub-POI), "
        "%d개 복원불가 (%.3fs)",
        n_bad, n_parent_recovered, n_sub_total, n_unrecoverable, elapsed
    )

    return ADSSResult(
        parent_indices=sub_parent,
        quarter_types=sub_qt,
        points_x=sub_px,
        points_y=sub_py,
        parameters=sub_p,
        zncc_values=sub_zncc,
        iterations=sub_iter,
        xsi_mins=sub_xsi_min,
        xsi_maxs=sub_xsi_max,
        eta_mins=sub_eta_min,
        eta_maxs=sub_eta_max,
        candidate_zncc=all_cand_zncc,
        n_bad_original=n_bad,
        n_sub_total=n_sub_total,
        n_parent_recovered=n_parent_recovered,
        n_unrecoverable=n_unrecoverable,
        elapsed_time=elapsed,
    )
