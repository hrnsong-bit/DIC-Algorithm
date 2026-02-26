"""
ADSS-DIC (Adaptive Subset-Subdivision) 래퍼 모듈

1단계 IC-GN 결과에서 불량 POI를 식별하고,
quarter-subset 분할 + IC-GN 재계산으로 복원한다.

파이프라인: FFT-CC → IC-GN → ADSS 재계산 → PLS 변형률

References:
    Zhao, J. & Pan, B. "Adaptive subset-subdivision for automatic
    digital image correlation calculation on discontinuous shape
    and deformation." Applied Optics, 2025.
"""

import numpy as np
import time
import logging

from .shape_function_numba import AFFINE, QUADRATIC, get_num_params
from .icgn_core_numba import ICGN_SUCCESS
from .variable_subset import _detect_grid_structure
from .adss_subset_numba import (
    process_bad_pois_adss_parallel,
    allocate_adss_batch_buffers,
)

logger = logging.getLogger(__name__)


def _build_neighbor_info_adss(
    bad_indices, points_x, points_y,
    valid_mask, zncc_values, parameters,
    zncc_threshold, ny, nx, n_params
):
    """
    ADSS-DIC용 이웃 정보 구성.

    각 불량 POI에 대해 Q1~Q8에 대응하는 8방위 이웃의
    well-matched 여부, IC-GN 파라미터, 좌표를 수집한다.

    Parameters
    ----------
    bad_indices : (n_bad,) int64
        불량 POI의 flat 인덱스.
    points_x, points_y : (n_poi,) int64
        전체 POI 좌표.
    valid_mask : (n_poi,) bool
        1단계 valid 마스크.
    zncc_values : (n_poi,) float64
        1단계 ZNCC 값.
    parameters : (n_poi, n_params) float64
        1단계 IC-GN 파라미터.
    zncc_threshold : float64
    ny, nx : int
        격자 행/열 수.
    n_params : int

    Returns
    -------
    all_neighbor_valid : (n_bad, 8) bool
    all_neighbor_params : (n_bad, 8, n_params) float64
    all_neighbor_x : (n_bad, 8) float64
    all_neighbor_y : (n_bad, 8) float64
    """
    n_bad = len(bad_indices)

    all_neighbor_valid = np.zeros((n_bad, 8), dtype=np.bool_)
    all_neighbor_params = np.zeros((n_bad, 8, n_params), dtype=np.float64)
    all_neighbor_x = np.zeros((n_bad, 8), dtype=np.float64)
    all_neighbor_y = np.zeros((n_bad, 8), dtype=np.float64)

    # Q1~Q8 대응 이웃 방향 (dy, dx)
    # Q1: Upper → 위쪽 이웃
    # Q2: Lower → 아래쪽 이웃
    # Q3: Left  → 왼쪽 이웃
    # Q4: Right → 오른쪽 이웃
    # Q5: UL    → 좌상 이웃
    # Q6: UR    → 우상 이웃
    # Q7: LL    → 좌하 이웃
    # Q8: LR    → 우하 이웃
    neighbor_dirs = [
        (-1,  0),  # Q1
        ( 1,  0),  # Q2
        ( 0, -1),  # Q3
        ( 0,  1),  # Q4
        (-1, -1),  # Q5
        (-1,  1),  # Q6
        ( 1, -1),  # Q7
        ( 1,  1),  # Q8
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
):
    """
    ADSS-DIC 재계산 — 메인 래퍼 함수.

    1단계 IC-GN 결과에서 불량 POI를 식별하고,
    quarter-subset 분할 + IC-GN으로 복원을 시도한다.

    Parameters
    ----------
    ref_image : (H, W) float64
        참조 이미지.
    grad_x, grad_y : (H, W) float64
        참조 이미지 gradient.
    coeffs : (H, W) float64
        deformed B-spline 계수.
    order : int
        보간 차수 (3 또는 5).
    points_x, points_y : (n_poi,) int64
        전체 POI 좌표.
    valid_mask : (n_poi,) bool
        1단계 valid 마스크 — in-place 업데이트.
    zncc_values : (n_poi,) float64
        1단계 ZNCC — in-place 업데이트.
    parameters : (n_poi, n_params) float64
        1단계 파라미터 — in-place 업데이트.
    convergence_flags : (n_poi,) bool
        — in-place 업데이트.
    iteration_counts : (n_poi,) int32
        — in-place 업데이트.
    failure_reasons : (n_poi,) int32
        — in-place 업데이트.
    subset_size : int
    max_iterations : int
    convergence_threshold : float64
    shape_function : str
        'affine' 또는 'quadratic'.
    zncc_threshold : float64
        C₀ (기본 0.9).

    Returns
    -------
    dict with keys:
        'n_bad', 'n_recovered', 'n_failed',
        'recovered_indices', 'failed_indices',
        'quarter_types', 'elapsed_time'
    """
    t_start = time.time()
    n_poi = len(points_x)

    # shape_type 변환
    if shape_function == 'affine':
        shape_type = AFFINE
    else:
        shape_type = QUADRATIC
    n_params = get_num_params(shape_type)

    # === 1. 불량 POI 식별 ===
    bad_mask = ~valid_mask | (zncc_values < zncc_threshold)
    bad_indices = np.where(bad_mask)[0].astype(np.int64)
    n_bad = len(bad_indices)

    empty_result = {
        'n_bad': n_bad,
        'n_recovered': 0,
        'n_failed': n_bad,
        'recovered_indices': np.array([], dtype=np.int64),
        'failed_indices': bad_indices.copy() if n_bad > 0 else np.array([], dtype=np.int64),
        'quarter_types': np.array([], dtype=np.int32),
        'elapsed_time': 0.0,
    }

    if n_bad == 0:
        empty_result['n_failed'] = 0
        empty_result['elapsed_time'] = time.time() - t_start
        logger.info("ADSS-DIC: 불량 POI 없음 — 스킵 (%.3fs)", empty_result['elapsed_time'])
        return empty_result

    logger.info("ADSS-DIC: %d개 불량 POI 감지, 재계산 시작...", n_bad)

    # === 2. 격자 구조 감지 ===
    grid_info = _detect_grid_structure(points_x, points_y)
    if grid_info is None:
        empty_result['elapsed_time'] = time.time() - t_start
        logger.warning("ADSS-DIC: 격자 구조 감지 실패 — 스킵")
        return empty_result

    ny, nx, spacing = grid_info
    logger.info("  격자 구조: %d×%d, 간격=%dpx", ny, nx, spacing)

    # === 3. 이웃 정보 구성 ===
    all_neighbor_valid, all_neighbor_params, all_neighbor_x, all_neighbor_y = \
        _build_neighbor_info_adss(
            bad_indices, points_x, points_y,
            valid_mask, zncc_values, parameters,
            zncc_threshold, ny, nx, n_params
        )

    # 시도 가능한 POI 필터 (이웃이 하나라도 있는 POI)
    has_candidate = np.any(all_neighbor_valid, axis=1)
    n_candidates = int(np.sum(has_candidate))

    if n_candidates == 0:
        empty_result['elapsed_time'] = time.time() - t_start
        logger.info("ADSS-DIC: 시도 가능한 후보 없음")
        return empty_result

    candidate_mask = has_candidate
    candidate_indices = bad_indices[candidate_mask]
    candidate_neighbor_valid = all_neighbor_valid[candidate_mask]
    candidate_neighbor_params = all_neighbor_params[candidate_mask]
    candidate_neighbor_x = all_neighbor_x[candidate_mask]
    candidate_neighbor_y = all_neighbor_y[candidate_mask]
    n_cand = len(candidate_indices)

    logger.info("  시도 가능 후보: %d / %d", n_cand, n_bad)

    # === 4. 버퍼 할당 ===
    M = subset_size // 2
    n_pixels_max = (2 * M + 1) * (M + 1)  # 십자형 최대 크기
    bufs = allocate_adss_batch_buffers(n_cand, n_pixels_max, n_params)

    result_p = np.zeros((n_cand, n_params), dtype=np.float64)
    result_zncc = np.zeros(n_cand, dtype=np.float64)
    result_iter = np.zeros(n_cand, dtype=np.int32)
    result_conv = np.zeros(n_cand, dtype=np.bool_)
    result_fail = np.zeros(n_cand, dtype=np.int32)
    result_qt = np.zeros(n_cand, dtype=np.int32)

    # === 5. 배치 처리 ===
    process_bad_pois_adss_parallel(
        ref_image, grad_x, grad_y,
        coeffs, order,
        points_x, points_y,
        subset_size, max_iterations, convergence_threshold,
        shape_type,
        candidate_indices,
        candidate_neighbor_valid,
        candidate_neighbor_params,
        candidate_neighbor_x, candidate_neighbor_y,
        zncc_threshold,
        result_p, result_zncc, result_iter, result_conv,
        result_fail, result_qt,
        bufs['f'], bufs['dfdx'], bufs['dfdy'],
        bufs['J'], bufs['H'], bufs['H_inv'],
        bufs['p'], bufs['xsi_w'], bufs['eta_w'],
        bufs['x_def'], bufs['y_def'],
        bufs['g'], bufs['b'], bufs['dp'], bufs['p_new'],
        bufs['xsi_local'], bufs['eta_local'],
        bufs['init_params'],
    )

    # === 6. 결과 병합 ===
    recovered_list = []
    failed_list = []
    quarter_types = np.full(n_bad, -1, dtype=np.int32)

    for k in range(n_cand):
        fidx = candidate_indices[k]
        # bad_indices 내 원래 위치
        orig_k = np.searchsorted(bad_indices, fidx)

        if result_conv[k] and result_zncc[k] >= zncc_threshold:
            valid_mask[fidx] = True
            zncc_values[fidx] = result_zncc[k]
            parameters[fidx, :n_params] = result_p[k, :]
            convergence_flags[fidx] = True
            iteration_counts[fidx] = result_iter[k]
            failure_reasons[fidx] = ICGN_SUCCESS
            recovered_list.append(fidx)
            if orig_k < n_bad:
                quarter_types[orig_k] = result_qt[k]
        else:
            failed_list.append(fidx)
            if orig_k < n_bad:
                quarter_types[orig_k] = result_qt[k]

    # 후보가 없었던 불량 POI도 failed에 추가
    no_candidate_indices = bad_indices[~candidate_mask]
    failed_list.extend(no_candidate_indices.tolist())

    recovered_indices = np.array(recovered_list, dtype=np.int64)
    failed_indices = np.array(failed_list, dtype=np.int64)
    n_recovered = len(recovered_list)
    n_failed = len(failed_list)

    elapsed = time.time() - t_start
    logger.info(
        "ADSS-DIC 완료: %d개 불량 → %d개 복원, %d개 실패 (%.3fs)",
        n_bad, n_recovered, n_failed, elapsed
    )

    return {
        'n_bad': n_bad,
        'n_recovered': n_recovered,
        'n_failed': n_failed,
        'recovered_indices': recovered_indices,
        'failed_indices': failed_indices,
        'quarter_types': quarter_types,
        'elapsed_time': elapsed,
    }
