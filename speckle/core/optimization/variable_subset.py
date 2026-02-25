"""
Variable Subset Recalculation 모듈

1단계 IC-GN에서 ZNCC < C₀인 불량 POI를 식별하고,
8방위 이웃 기반 서브셋 재배치 + IC-GN 재계산으로 복원한다.

기존 DIC 파이프라인(FFT-CC → IC-GN) 이후 자동 실행되며,
불량 POI가 없으면 비용 없이 스킵된다.

References:
    - Ma, Q., et al. Optics and Lasers in Engineering, 2021.
      (Variable subset DIC with pixel-level ZNCC map)
    - Zhao, J. & Pan, B. Optics and Lasers in Engineering, 2025.
      (ADSS-DIC: Adaptive Subset-Subdivision)
"""

import numpy as np
import time
import logging

from .shape_function_numba import get_num_params
from .icgn_core_numba import ICGN_SUCCESS
from .variable_subset_numba import (
    process_bad_pois_parallel,
    allocate_variable_batch_buffers,
)

logger = logging.getLogger(__name__)


def _detect_grid_structure(points_x, points_y):
    """
    POI 좌표에서 격자 구조(ny, nx, spacing)를 추론한다.

    Args:
        points_x: 전체 POI x좌표 (n_poi,) int64
        points_y: 전체 POI y좌표 (n_poi,) int64

    Returns:
        (ny, nx, spacing) if 격자 구조 감지
        None if 격자 구조가 아님
    """
    n_poi = len(points_x)
    if n_poi < 2:
        return None

    # unique한 x, y 값 추출
    unique_x = np.unique(points_x)
    unique_y = np.unique(points_y)
    nx = len(unique_x)
    ny = len(unique_y)

    # 격자 크기 일치 확인
    if nx * ny != n_poi:
        return None

    # 등간격 확인
    if nx > 1:
        dx = np.diff(unique_x)
        if not np.all(dx == dx[0]):
            return None
        spacing = int(dx[0])
    elif ny > 1:
        dy = np.diff(unique_y)
        spacing = int(dy[0])
    else:
        return None

    if ny > 1:
        dy = np.diff(unique_y)
        if not np.all(dy == dy[0]):
            return None
        if nx > 1 and int(dy[0]) != spacing:
            return None

    return ny, nx, spacing


def _build_neighbor_valid(
    bad_indices, points_x, points_y,
    valid_mask, zncc_values, zncc_threshold,
    ny, nx, min_valid_neighbors=2  # 하위 호환용, 실제로는 사용 안 함
):
    n_bad = len(bad_indices)
    all_neighbor_valid = np.zeros((n_bad, 8), dtype=np.bool_)

    # 8방위 오프셋 (dy, dx)
    offsets = [
        (-1, -1), (-1, 0), (-1, +1),
        ( 0, -1),          ( 0, +1),
        (+1, -1), (+1, 0), (+1, +1),
    ]

    # 연속 3이웃 그룹 → 대응 서브셋
    # (이웃 인덱스 3개, 서브셋 인덱스)
    triplet_to_subset = [
        ([0, 1, 2], 6),  # NW,N,NE  → S7 (N 확장)
        ([1, 2, 4], 5),  # N,NE,E   → S6 (NE 확장)
        ([2, 4, 7], 3),  # NE,E,SE  → S4 (E 확장)
        ([4, 7, 6], 0),  # E,SE,S   → S1 (SE 확장)
        ([7, 6, 5], 1),  # SE,S,SW  → S2 (S 확장)
        ([6, 5, 3], 2),  # S,SW,W   → S3 (SW 확장)
        ([5, 3, 0], 4),  # SW,W,NW  → S5 (W 확장)
        ([3, 0, 1], 7),  # W,NW,N   → S8 (NW 확장)
    ]

    for k in range(n_bad):
        flat_idx = bad_indices[k]
        iy = flat_idx // nx
        ix = flat_idx % nx

        # 8방위 이웃 ZNCC 수집
        neighbor_zncc = np.full(8, -1.0)
        neighbor_ok = np.zeros(8, dtype=np.bool_)

        for d in range(8):
            dy, dx = offsets[d]
            ny_new = iy + dy
            nx_new = ix + dx
            if 0 <= ny_new < ny and 0 <= nx_new < nx:
                nf = ny_new * nx + nx_new
                if valid_mask[nf] and zncc_values[nf] >= zncc_threshold:
                    neighbor_ok[d] = True
                    neighbor_zncc[d] = zncc_values[nf]

        # 연속 3이웃이 모두 valid한 그룹 찾기
        best_mean_zncc = -1.0
        best_subset_idx = -1

        for triplet, s_idx in triplet_to_subset:
            if all(neighbor_ok[d] for d in triplet):
                mean_z = np.mean([neighbor_zncc[d] for d in triplet])
                if mean_z > best_mean_zncc:
                    best_mean_zncc = mean_z
                    best_subset_idx = s_idx

        if best_subset_idx >= 0:
            all_neighbor_valid[k, best_subset_idx] = True

    return all_neighbor_valid


def compute_variable_subset_recalc(
    ref_image, grad_x, grad_y,
    coeffs, order,
    points_x, points_y,
    initial_u, initial_v,
    valid_mask, zncc_values, parameters,
    convergence_flags, iteration_counts, failure_reasons,
    subset_size,
    max_iterations=50,
    convergence_threshold=0.001,
    shape_function='affine',
    zncc_threshold=0.9,
):
    """
    Variable Subset 2단계 재계산 — 메인 래퍼 함수.

    1단계 IC-GN 결과에서 불량 POI를 식별하고,
    이웃 기반 서브셋 재배치 + IC-GN으로 복원을 시도한다.

    Args:
        ref_image: 참조 이미지 (H, W) float64
        grad_x, grad_y: gradient (H, W) float64
        coeffs: deformed B-spline 계수 (H, W) float64
        order: 보간 차수
        points_x, points_y: POI 좌표 (n_poi,) int64
        initial_u, initial_v: FFT-CC 초기 변위 (n_poi,) float64
        valid_mask: 1단계 valid 마스크 (n_poi,) bool — in-place 업데이트됨
        zncc_values: 1단계 ZNCC (n_poi,) float64 — in-place 업데이트됨
        parameters: 1단계 파라미터 (n_poi, n_params) float64 — in-place 업데이트됨
        convergence_flags: (n_poi,) bool — in-place 업데이트됨
        iteration_counts: (n_poi,) int32 — in-place 업데이트됨
        failure_reasons: (n_poi,) int32 — in-place 업데이트됨
        subset_size: 서브셋 크기
        max_iterations: 최대 반복 횟수
        convergence_threshold: 수렴 임계값
        shape_function: 'affine' 또는 'quadratic'
        zncc_threshold: C₀ (기본 0.9)

    Returns:
        dict with keys:
            'n_bad': 불량 POI 수
            'n_recovered': 복원 성공 수
            'n_failed': 복원 실패 수
            'recovered_indices': 복원된 POI의 flat 인덱스
            'failed_indices': 복원 실패 POI의 flat 인덱스
            'subset_types': 각 불량 POI가 선택한 서브셋 타입
            'elapsed_time': 소요 시간 (초)
    """
    from .shape_function_numba import AFFINE, QUADRATIC

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

    if n_bad == 0:
        elapsed = time.time() - t_start
        logger.info("Variable Subset: 불량 POI 없음 — 스킵 (%.3fs)", elapsed)
        return {
            'n_bad': 0,
            'n_recovered': 0,
            'n_failed': 0,
            'recovered_indices': np.array([], dtype=np.int64),
            'failed_indices': np.array([], dtype=np.int64),
            'subset_types': np.array([], dtype=np.int32),
            'elapsed_time': elapsed,
        }

    logger.info("Variable Subset: %d개 불량 POI 감지, 재계산 시작...", n_bad)

    # === 2. 격자 구조 감지 ===
    grid_info = _detect_grid_structure(points_x, points_y)
    if grid_info is None:
        elapsed = time.time() - t_start
        logger.warning("Variable Subset: 격자 구조 감지 실패 — 스킵")
        return {
            'n_bad': n_bad,
            'n_recovered': 0,
            'n_failed': n_bad,
            'recovered_indices': np.array([], dtype=np.int64),
            'failed_indices': bad_indices.copy(),
            'subset_types': np.array([], dtype=np.int32),
            'elapsed_time': elapsed,
        }
    ny, nx, spacing = grid_info
    logger.info("  격자 구조: %d×%d, 간격=%dpx", ny, nx, spacing)

    # === 3. 이웃 valid 판정 ===
    all_neighbor_valid = _build_neighbor_valid(
        bad_indices, points_x, points_y,
        valid_mask, zncc_values, zncc_threshold,
        ny, nx
    )

    # === 4. 버퍼 할당 ===
    n_pixels = subset_size * subset_size
    batch_bufs = allocate_variable_batch_buffers(n_bad, n_pixels, n_params)

    result_p = np.empty((n_bad, n_params), dtype=np.float64)
    result_zncc = np.empty(n_bad, dtype=np.float64)
    result_iter = np.empty(n_bad, dtype=np.int32)
    result_conv = np.empty(n_bad, dtype=np.bool_)
    result_fail = np.empty(n_bad, dtype=np.int32)
    result_subset = np.empty(n_bad, dtype=np.int32)

    # === 5. 배치 병렬 처리 ===
    process_bad_pois_parallel(
        ref_image, grad_x, grad_y,
        coeffs, order,
        points_x, points_y,
        initial_u, initial_v,
        subset_size, max_iterations, convergence_threshold,
        shape_type,
        bad_indices,
        all_neighbor_valid,
        zncc_threshold,
        result_p, result_zncc, result_iter, result_conv, result_fail, result_subset,
        batch_bufs['f'], batch_bufs['dfdx'], batch_bufs['dfdy'],
        batch_bufs['J'], batch_bufs['H'], batch_bufs['H_inv'],
        batch_bufs['p'], batch_bufs['xsi_w'], batch_bufs['eta_w'],
        batch_bufs['x_def'], batch_bufs['y_def'],
        batch_bufs['g'], batch_bufs['b'], batch_bufs['dp'], batch_bufs['p_new'],
        batch_bufs['xsi_local'], batch_bufs['eta_local'],
    )

    # === 6. 결과 병합 ===
    recovered_list = []
    failed_list = []

    for k in range(n_bad):
        flat_idx = bad_indices[k]

        if result_conv[k] and result_zncc[k] >= zncc_threshold:
            # 복원 성공 → 기존 결과 덮어쓰기
            valid_mask[flat_idx] = True
            zncc_values[flat_idx] = result_zncc[k]
            parameters[flat_idx, :n_params] = result_p[k, :]
            convergence_flags[flat_idx] = True
            iteration_counts[flat_idx] = result_iter[k]
            failure_reasons[flat_idx] = ICGN_SUCCESS
            recovered_list.append(flat_idx)
        else:
            failed_list.append(flat_idx)

    recovered_indices = np.array(recovered_list, dtype=np.int64)
    failed_indices = np.array(failed_list, dtype=np.int64)
    n_recovered = len(recovered_list)
    n_failed = len(failed_list)

    elapsed = time.time() - t_start
    logger.info(
        "Variable Subset 완료: %d개 불량 → %d개 복원, %d개 실패 (%.3fs)",
        n_bad, n_recovered, n_failed, elapsed
    )

    return {
        'n_bad': n_bad,
        'n_recovered': n_recovered,
        'n_failed': n_failed,
        'recovered_indices': recovered_indices,
        'failed_indices': failed_indices,
        'subset_types': result_subset.copy(),
        'elapsed_time': elapsed,
    }
