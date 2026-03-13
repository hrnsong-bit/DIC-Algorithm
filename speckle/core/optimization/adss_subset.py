"""
ADSS-DIC v3.1 (Adaptive Subset-Subdivision) 래퍼 모듈

v3.0 대비 변경:
    - 2-pass rescue 지원: result_fail_info shape (n_cand, 8, 3) → (n_cand, 8, 4).
    - 로그에 pass 번호(P1/P2) 반영.

삼각형(Q1~Q4) + 사각형(Q5~Q8) 적응 선택 방식:
    - 각 불량 POI에서 8개 quarter의 1-warp ZNCC를 평가
    - 삼각형 세트 vs 사각형 세트 중 유효 quarter 수가 많은 쪽 선택
    - 선택된 세트에서 threshold를 넘는 quarter에 IC-GN 수행 (pass 1)
    - 외부 seed 부재로 미평가된 quarter를 내부 seed로 재평가 (pass 2)
    - 결과를 ADSSResult로 반환 (sub-POI 복수 저장)
    - 대표값(최대 ZNCC)을 ICGNResult에 반영하여 기존 파이프라인 호환

파이프라인: FFT-CC → IC-GN → ADSS v3.1 재계산 → PLS 변형률

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
    Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8,
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
    ADSS-DIC v3용 이웃 정보 구성 — quarter당 3이웃 후보.

    8개 quarter 각각에 대해 인접한 3방향 이웃을 후보로 저장한다.
      Q1(상):   상, 좌상, 우상
      Q2(하):   하, 좌하, 우하
      Q3(좌):   좌, 좌상, 좌하
      Q4(우):   우, 우상, 우하
      Q5(좌상): 좌상, 상, 좌
      Q6(우상): 우상, 상, 우
      Q7(좌하): 좌하, 하, 좌
      Q8(우하): 우하, 하, 우
    """

    n_bad = len(bad_indices)

    all_neighbor_valid  = np.zeros((n_bad, 8, 3), dtype=np.bool_)
    all_neighbor_params = np.zeros((n_bad, 8, 3, n_params), dtype=np.float64)
    all_neighbor_x      = np.zeros((n_bad, 8, 3), dtype=np.float64)
    all_neighbor_y      = np.zeros((n_bad, 8, 3), dtype=np.float64)

    # 8방위 이웃 방향 (dy, dx)
    #   0:좌상  1:상  2:우상
    #   3:좌         4:우
    #   5:좌하  6:하  7:우하
    all_dirs = [
        (-1, -1), (-1,  0), (-1, +1),
        ( 0, -1),           ( 0, +1),
        (+1, -1), (+1,  0), (+1, +1),
    ]

    # 각 사분면에 대한 3이웃 후보 인덱스 (all_dirs 기준)
    quarter_neighbor_indices = [
        # --- 삼각형 (Q1~Q4) ---
        [1, 0, 2],  # Q1(상)  ← 상, 좌상, 우상
        [6, 5, 7],  # Q2(하)  ← 하, 좌하, 우하
        [3, 0, 5],  # Q3(좌)  ← 좌, 좌상, 좌하
        [4, 2, 7],  # Q4(우)  ← 우, 우상, 우하
        # --- 사각형 (Q5~Q8) ---
        [0, 1, 3],  # Q5(좌상) ← 좌상, 상, 좌
        [2, 1, 4],  # Q6(우상) ← 우상, 상, 우
        [5, 6, 3],  # Q7(좌하) ← 좌하, 하, 좌
        [7, 6, 4],  # Q8(우하) ← 우하, 하, 우
    ]

    for k in range(n_bad):
        flat_idx = bad_indices[k]
        iy = flat_idx // nx
        ix = flat_idx % nx

        for q in range(8):
            for c, dir_idx in enumerate(quarter_neighbor_indices[q]):
                dy, dx = all_dirs[dir_idx]
                niy = iy + dy
                nix = ix + dx

                if 0 <= niy < ny and 0 <= nix < nx:
                    nf = niy * nx + nix
                    if valid_mask[nf] and zncc_values[nf] >= zncc_threshold:
                        all_neighbor_valid[k, q, c]  = True
                        all_neighbor_params[k, q, c, :n_params] = parameters[nf, :n_params]
                        all_neighbor_x[k, q, c] = float(points_x[nf])
                        all_neighbor_y[k, q, c] = float(points_y[nf])

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
):
    """
    ADSS-DIC v3.1 재계산 — 사분면 복수 채택 + 2-pass rescue + fail_info 디버깅 로그.
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
        logger.info("ADSS-DIC v3.1: 불량 POI 없음 — 스킵 (%.3fs)", time.time() - t_start)
        return _empty_adss(0, time.time() - t_start)

    logger.info("ADSS-DIC v3.1: %d개 불량 POI 감지, 2-pass 사분면 복수 채택 시작...", n_bad)

    # === 2. 격자 구조 감지 ===
    grid_info = _detect_grid_structure(points_x, points_y)
    if grid_info is None:
        logger.warning("ADSS-DIC v3.1: 격자 구조 감지 실패 — 스킵")
        return _empty_adss(n_bad, time.time() - t_start)

    ny, nx, spacing = grid_info
    logger.info("  격자 구조: %d×%d, 간격=%dpx", ny, nx, spacing)

    # === 3. 이웃 정보 구성 ===
    all_nv, all_np, all_nx, all_ny = _build_neighbor_info_adss_v2(
        bad_indices, points_x, points_y,
        valid_mask, zncc_values, parameters,
        zncc_threshold, ny, nx, n_params
    )

    has_candidate = np.any(all_nv.reshape(n_bad, -1), axis=1)
    n_candidates = int(np.sum(has_candidate))

    # 2-pass에서는 외부 이웃이 전혀 없는 POI도 내부 seed로 복원 가능하므로
    # 모든 불량 POI를 candidate로 포함시킨다.
    # (기존: has_candidate가 False인 POI는 제외했음)
    # 단, 최소한 해당 POI의 8개 quarter 중 어느 것도 외부 이웃이 없는 경우에도
    # prange 루프에는 포함시키되, process_poi_adss_multi 내부에서
    # Phase 2 세트 선택 시 count_tri == 0 && count_rect == 0 이면 즉시 return 0.
    # 따라서 전부 포함해도 안전하다.
    cand_indices = bad_indices
    cand_nv = all_nv
    cand_np_ = all_np
    cand_nx_ = all_nx
    cand_ny_ = all_ny
    n_cand = len(cand_indices)

    logger.info("  시도 후보: %d (외부이웃有: %d, 외부이웃無: %d)",
                n_cand, n_candidates, n_cand - n_candidates)

    # === 4. 버퍼 할당 ===
    n_pixels_max = (M + 1) * (M + 1)
    bufs = allocate_adss_multi_batch_buffers(n_cand, n_pixels_max, n_params)

    max_sub = 4 * n_cand
    result_p = np.zeros((max_sub, n_params), dtype=np.float64)
    result_zncc = np.zeros(max_sub, dtype=np.float64)
    result_iter = np.zeros(max_sub, dtype=np.int32)
    result_qt = np.zeros(max_sub, dtype=np.int32)
    result_parent = np.full(max_sub, -1, dtype=np.int64)
    result_count = np.zeros(n_cand, dtype=np.int32)
    result_candidate_zncc = np.full((n_cand, 8), -1.0, dtype=np.float64)
    result_fail_info = np.full((n_cand, 8, 4), -1.0, dtype=np.float64)

    # === 5. 배치 병렬 처리 ===
    process_bad_pois_adss_multi_parallel(
        ref_image, grad_x, grad_y,
        coeffs, order,
        points_x, points_y,
        subset_size, max_iterations, convergence_threshold,
        shape_type,
        cand_indices,
        cand_nv, cand_np_, cand_nx_, cand_ny_,
        zncc_threshold,
        result_p, result_zncc, result_iter, result_qt,
        result_parent, result_count, result_candidate_zncc,
        result_fail_info,
        bufs['f'], bufs['dfdx'], bufs['dfdy'],
        bufs['J'], bufs['H'], bufs['H_inv'],
        bufs['p'], bufs['xsi_w'], bufs['eta_w'],
        bufs['x_def'], bufs['y_def'],
        bufs['g'], bufs['b'], bufs['dp'], bufs['p_new'],
        bufs['xsi_local'], bufs['eta_local'],
        bufs['init_params'],
        bufs['out_p'], bufs['out_zncc'], bufs['out_iter'],
        bufs['out_qt'], bufs['out_cand_zncc'],
        bufs['out_fail_info'],
    )

    # === 6. 유효 결과 추출 ===
    valid_sub_mask = result_parent >= 0
    n_sub_total = int(np.sum(valid_sub_mask))

    if n_sub_total == 0:
        logger.info("ADSS-DIC v3.1: 복원된 sub-POI 없음")
        return _empty_adss(n_bad, time.time() - t_start)

    sub_parent = result_parent[valid_sub_mask]
    sub_qt = result_qt[valid_sub_mask]
    sub_p = result_p[valid_sub_mask]
    sub_zncc = result_zncc[valid_sub_mask]
    sub_iter = result_iter[valid_sub_mask]
    sub_slots = np.flatnonzero(valid_sub_mask)
    sub_recovery_passes = np.ones(n_sub_total, dtype=np.int32)

    for i, slot in enumerate(sub_slots):
        cand_k = int(slot // 4)
        qt = int(result_qt[slot])
        if 0 <= cand_k < n_cand and 1 <= qt <= 8:
            pass_num = int(result_fail_info[cand_k, qt - 1, 3])
            sub_recovery_passes[i] = pass_num if pass_num > 0 else 1

    sub_xsi_min = np.zeros(n_sub_total, dtype=np.int32)
    sub_xsi_max = np.zeros(n_sub_total, dtype=np.int32)
    sub_eta_min = np.zeros(n_sub_total, dtype=np.int32)
    sub_eta_max = np.zeros(n_sub_total, dtype=np.int32)

    qt_to_range = {
        1: (-M, M, -M, 0),    # Q1 bounding box
        2: (-M, M, 0, M),     # Q2
        3: (-M, 0, -M, M),    # Q3
        4: (0, M, -M, M),     # Q4
        5: (-M, 0, -M, 0),    # Q5
        6: (0, M, -M, 0),     # Q6
        7: (-M, 0, 0, M),     # Q7
        8: (0, M, 0, M),      # Q8
    }

    for i in range(n_sub_total):
        qt = int(sub_qt[i])
        if qt in qt_to_range:
            sub_xsi_min[i], sub_xsi_max[i], sub_eta_min[i], sub_eta_max[i] = qt_to_range[qt]

    sub_px = np.array([int(points_x[pi]) for pi in sub_parent], dtype=np.int64)
    sub_py = np.array([int(points_y[pi]) for pi in sub_parent], dtype=np.int64)

    # === 7. 통계 ===
    unique_parents = np.unique(sub_parent)
    n_parent_recovered = len(unique_parents)
    n_unrecoverable = n_bad - n_parent_recovered

    # === 8. candidate_zncc를 bad_indices 전체로 매핑 ===
    all_cand_zncc = np.full((n_bad, 8), -1.0, dtype=np.float64)
    for k in range(n_cand):
        orig_k = np.searchsorted(bad_indices, cand_indices[k])
        if orig_k < n_bad:
            all_cand_zncc[orig_k, :] = result_candidate_zncc[k, :]

    elapsed = time.time() - t_start

    # === 9. 로그 출력 (fail_info + pass 번호 포함) ===
    quarter_names = {
        1: 'Q1:UT', 2: 'Q2:LT', 3: 'Q3:LF', 4: 'Q4:RT',
        5: 'Q5:UL', 6: 'Q6:UR', 7: 'Q7:LL', 8: 'Q8:LR',
    }
    fail_code_names = {
        -1: 'NOT_TRIED', 0: 'SUCCESS', 1: 'LOW_ZNCC', 2: 'DIVERGED',
        3: 'OUT_OF_BOUNDS', 4: 'SINGULAR_H', 5: 'FLAT_SUBSET',
        6: 'MAX_DISP', 7: 'FLAT_TARGET', 99: 'REF_EXTRACT_FAIL'
    }
    qn = ['Q1:UT', 'Q2:LT', 'Q3:LF', 'Q4:RT', 'Q5:UL', 'Q6:UR', 'Q7:LL', 'Q8:LR']

    n_pass2_tried = 0
    n_pass2_ok = 0

    for k in range(n_cand):
        fidx = cand_indices[k]
        n_rec = result_count[k]
        cz = result_candidate_zncc[k]
        fi = result_fail_info[k]

        offset = k * 4
        rec_qts = []
        for j in range(n_rec):
            rec_qts.append(quarter_names.get(result_qt[offset + j], '?'))

        detail_strs = []
        for i in range(8):
            pass_num = int(fi[i, 3])
            pass_tag = f"P{pass_num}" if pass_num > 0 else ""

            if cz[i] < 0 and pass_num < 2:
                detail_strs.append(f"{qn[i]}:N/A")
            else:
                fc = int(fi[i, 2])
                fc_name = fail_code_names.get(fc, f'CODE_{fc}')
                final_z = fi[i, 0]
                n_it = int(fi[i, 1])

                if pass_num == 2:
                    n_pass2_tried += 1

                if fc == 0:
                    if pass_num == 2:
                        n_pass2_ok += 1
                    seed_tag = "ext" if pass_num == 1 else "int"
                    detail_strs.append(
                        f"{qn[i]}:{pass_tag}1w={cz[i]:.4f}→OK({seed_tag},z={final_z:.4f},{n_it}it)"
                    )
                else:
                    seed_tag = "ext" if pass_num == 1 else "int"
                    detail_strs.append(
                        f"{qn[i]}:{pass_tag}1w={cz[i]:.4f}→FAIL({seed_tag},{fc_name},z={final_z:.4f},{n_it}it)"
                    )

        logger.info(
            "  POI[%d] (%d,%d) → %d개 복원 %s | %s",
            fidx, points_x[fidx], points_y[fidx],
            n_rec, rec_qts,
            " | ".join(detail_strs)
        )

    logger.info(
        "ADSS-DIC v3.1 완료: %d개 불량 → %d개 부모 복원 (%d sub-POI), "
        "%d개 복원불가, pass2 시도=%d 성공=%d (%.3fs)",
        n_bad, n_parent_recovered, n_sub_total, n_unrecoverable,
        n_pass2_tried, n_pass2_ok, elapsed
    )

    return ADSSResult(
        parent_indices=sub_parent,
        quarter_types=sub_qt,
        points_x=sub_px,
        points_y=sub_py,
        parameters=sub_p,
        zncc_values=sub_zncc,
        iterations=sub_iter,
        recovery_passes=sub_recovery_passes,
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
