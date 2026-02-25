
import numpy as np
from numba import jit, float64, prange

from .interpolation_numba import (
    is_inside_batch,
    _interp2d_cubic,
    _interp2d_quintic,
)
from .shape_function_numba import (
    AFFINE, QUADRATIC,
    warp,
    compute_steepest_descent,
    compute_hessian,
    get_num_params,
)
from .icgn_core_numba import (
    compute_znssd,
    icgn_iterate,
    ICGN_SUCCESS,
    ICGN_FAIL_LOW_ZNCC,
    ICGN_FAIL_FLAT_SUBSET,
    ICGN_FAIL_SINGULAR_HESSIAN,
)


"""
Variable Subset 로컬 좌표 생성

S₀~S₈ 각 서브셋 타입에 따라 비대칭 로컬 좌표 (ξ, η)를 생성한다.
서브셋 크기는 모두 (2M+1)×(2M+1)로 동일하며, 제어점 위치만 다르다.

서브셋 타입별 제어점 위치 및 로컬 좌표 범위:
    S₀: 중심         → ξ ∈ [-M, +M],  η ∈ [-M, +M]   (표준)
    S₁: 좌상 꼭짓점   → ξ ∈ [0, +2M],  η ∈ [0, +2M]   (SE 확장)
    S₂: 상변 중점     → ξ ∈ [-M, +M],  η ∈ [0, +2M]   (S 확장)
    S₃: 우상 꼭짓점   → ξ ∈ [-2M, 0],  η ∈ [0, +2M]   (SW 확장)
    S₄: 좌변 중점     → ξ ∈ [0, +2M],  η ∈ [-M, +M]   (E 확장)
    S₅: 우변 중점     → ξ ∈ [-2M, 0],  η ∈ [-M, +M]   (W 확장)
    S₆: 좌하 꼭짓점   → ξ ∈ [0, +2M],  η ∈ [-2M, 0]   (NE 확장)
    S₇: 하변 중점     → ξ ∈ [-M, +M],  η ∈ [-2M, 0]   (N 확장)
    S₈: 우하 꼭짓점   → ξ ∈ [-2M, 0],  η ∈ [-2M, 0]   (NW 확장)

여기서 ξ는 x(col) 방향, η는 y(row) 방향이다.
기존 generate_local_coordinates와 동일한 convention:
    meshgrid(coords_x, coords_y, indexing='ij') → eta가 행, xsi가 열

Reference:
    Ma, C., et al. "Variable subset DIC algorithm for measuring 
    discontinuous displacement based on pixel-level ZNCC value 
    distribution map." Measurement, 180, 109583, 2021.
"""

# 서브셋 타입 상수
S0 = 0  # 표준 (중심)
S1 = 1  # 좌상 꼭짓점 → SE 확장
S2 = 2  # 상변 중점   → S 확장
S3 = 3  # 우상 꼭짓점 → SW 확장
S4 = 4  # 좌변 중점   → E 확장
S5 = 5  # 우변 중점   → W 확장
S6 = 6  # 좌하 꼭짓점 → NE 확장
S7 = 7  # 하변 중점   → N 확장
S8 = 8  # 우하 꼭짓점 → NW 확장

# 서브셋 타입 → 대응 이웃 방향 매핑
# (dy, dx) 형태: 이웃의 그리드 좌표 오프셋
SUBSET_TO_NEIGHBOR = {
    S1: ( 1,  1),  # SE 이웃
    S2: ( 1,  0),  # S 이웃
    S3: ( 1, -1),  # SW 이웃
    S4: ( 0,  1),  # E 이웃
    S5: ( 0, -1),  # W 이웃
    S6: (-1,  1),  # NE 이웃
    S7: (-1,  0),  # N 이웃
    S8: (-1, -1),  # NW 이웃
}

# 이웃 방향 → 서브셋 타입 역매핑
NEIGHBOR_TO_SUBSET = {v: k for k, v in SUBSET_TO_NEIGHBOR.items()}


def generate_variable_local_coordinates(subset_size, subset_type=S0):
    """
    Variable subset용 비대칭 로컬 좌표 (ξ, η) 생성

    Args:
        subset_size: 서브셋 크기 (홀수, 예: 19 → M=9)
        subset_type: 서브셋 타입 (S0~S8, 기본값 S0=표준)

    Returns:
        (xsi, eta) — 1D float64 배열, 각 길이 = subset_size²
    """
    M = subset_size // 2

    if subset_type == S0:
        xsi_range = np.arange(-M, M + 1, dtype=np.float64)
        eta_range = np.arange(-M, M + 1, dtype=np.float64)
    elif subset_type == S1:
        xsi_range = np.arange(0, 2 * M + 1, dtype=np.float64)
        eta_range = np.arange(0, 2 * M + 1, dtype=np.float64)
    elif subset_type == S2:
        xsi_range = np.arange(-M, M + 1, dtype=np.float64)
        eta_range = np.arange(0, 2 * M + 1, dtype=np.float64)
    elif subset_type == S3:
        xsi_range = np.arange(-2 * M, 1, dtype=np.float64)
        eta_range = np.arange(0, 2 * M + 1, dtype=np.float64)
    elif subset_type == S4:
        xsi_range = np.arange(0, 2 * M + 1, dtype=np.float64)
        eta_range = np.arange(-M, M + 1, dtype=np.float64)
    elif subset_type == S5:
        xsi_range = np.arange(-2 * M, 1, dtype=np.float64)
        eta_range = np.arange(-M, M + 1, dtype=np.float64)
    elif subset_type == S6:
        xsi_range = np.arange(0, 2 * M + 1, dtype=np.float64)
        eta_range = np.arange(-2 * M, 1, dtype=np.float64)
    elif subset_type == S7:
        xsi_range = np.arange(-M, M + 1, dtype=np.float64)
        eta_range = np.arange(-2 * M, 1, dtype=np.float64)
    elif subset_type == S8:
        xsi_range = np.arange(-2 * M, 1, dtype=np.float64)
        eta_range = np.arange(-2 * M, 1, dtype=np.float64)
    else:
        raise ValueError(f"Unknown subset_type: {subset_type}")

    # meshgrid: indexing='ij' → eta가 행(첫번째 축), xsi가 열(두번째 축)
    eta_2d, xsi_2d = np.meshgrid(eta_range, xsi_range, indexing='ij')

    return xsi_2d.ravel().copy(), eta_2d.ravel().copy()

@jit(nopython=True, cache=True)
def extract_reference_subset_variable(
    ref_image, grad_x, grad_y,
    cx, cy,
    xsi_min, xsi_max, eta_min, eta_max,
    f_out, dfdx_out, dfdy_out
):
    """
    비대칭 서브셋 영역에서 Reference subset 추출 — nopython 호환

    기존 extract_reference_subset과 동일한 로직이나,
    추출 범위가 cx ± half 대칭이 아닌 [cx+xsi_min, cx+xsi_max] × [cy+eta_min, cy+eta_max]
    비대칭 범위를 사용한다.

    S₀(중심 서브셋)에서 xsi_min=-M, xsi_max=+M, eta_min=-M, eta_max=+M 을 입력하면
    기존 extract_reference_subset과 수치적으로 동일한 결과를 생성한다.

    Args:
        ref_image: 참조 이미지 (H, W) float64
        grad_x, grad_y: gradient 이미지 (H, W) float64
        cx, cy: POI 중심 좌표 (정수)
        xsi_min, xsi_max: x(열) 방향 로컬 좌표 범위 (정수)
        eta_min, eta_max: y(행) 방향 로컬 좌표 범위 (정수)
        f_out: 출력 — reference 강도 벡터 (n_pixels,)
        dfdx_out: 출력 — x gradient 벡터 (n_pixels,)
        dfdy_out: 출력 — y gradient 벡터 (n_pixels,)

    Returns:
        (f_mean, f_tilde, valid)
        valid=False이면 flat subset 또는 경계 밖
    """
    h = ref_image.shape[0]
    w = ref_image.shape[1]

    # 절대 좌표 범위 계산
    row_start = cy + eta_min
    row_end = cy + eta_max
    col_start = cx + xsi_min
    col_end = cx + xsi_max

    # 경계 체크
    if row_start < 0 or row_end >= h or col_start < 0 or col_end >= w:
        return 0.0, 0.0, False

    # subset 추출 (row-major ravel: eta 외부, xsi 내부)
    idx = 0
    for row in range(row_start, row_end + 1):
        for col in range(col_start, col_end + 1):
            f_out[idx] = ref_image[row, col]
            dfdx_out[idx] = grad_x[row, col]
            dfdy_out[idx] = grad_y[row, col]
            idx += 1

    # mean, tilde 계산
    n = idx
    f_sum = 0.0
    for i in range(n):
        f_sum += f_out[i]
    f_mean = f_sum / n

    tilde_sq = 0.0
    for i in range(n):
        diff = f_out[i] - f_mean
        tilde_sq += diff * diff
    f_tilde = np.sqrt(tilde_sq)

    if f_tilde < 1e-10:
        return f_mean, f_tilde, False

    return f_mean, f_tilde, True

@jit(nopython=True, cache=True)
def evaluate_candidate_zncc(
    ref_image, grad_x, grad_y,
    coeffs, order,
    cx, cy,
    initial_u, initial_v,
    xsi_min, xsi_max, eta_min, eta_max,
    xsi, eta,
    shape_type,
    # 작업 버퍼
    f, dfdx, dfdy,
    xsi_w, eta_w,
    x_def, y_def,
    g
):
    """
    단일 후보 서브셋에 대해 1회 warp ZNCC를 계산.

    FFT-CC 초기값(u, v)만으로 warp를 1회 수행하고,
    reference subset과 deformed subset 간의 ZNCC를 반환한다.
    8방위 후보 중 최적 서브셋을 빠르게 선별하기 위한 함수.

    Args:
        ref_image: 참조 이미지 (H, W) float64
        grad_x, grad_y: gradient 이미지 (H, W) float64
        coeffs: deformed 이미지의 B-spline 계수 (H, W) float64
        order: 보간 차수 (3 또는 5)
        cx, cy: POI 중심 좌표 (정수)
        initial_u, initial_v: FFT-CC 초기 추정 변위
        xsi_min, xsi_max: x(열) 방향 로컬 좌표 범위 (정수)
        eta_min, eta_max: y(행) 방향 로컬 좌표 범위 (정수)
        xsi, eta: 비대칭 로컬 좌표 (n_pixels,) — generate_variable_local_coordinates로 생성
        shape_type: AFFINE(0) 또는 QUADRATIC(1)

        [작업 버퍼 — 사전 할당]
        f, dfdx, dfdy: (n_pixels,)
        xsi_w, eta_w: (n_pixels,)
        x_def, y_def: (n_pixels,)
        g: (n_pixels,)

    Returns:
        zncc: ZNCC 값 (0.0~1.0). 실패 시 -1.0 반환.
    """
    n_pixels = len(xsi)
    img_h = ref_image.shape[0]
    img_w = ref_image.shape[1]

    # 1. Reference subset 추출 (비대칭)
    f_mean, f_tilde, valid = extract_reference_subset_variable(
        ref_image, grad_x, grad_y, cx, cy,
        xsi_min, xsi_max, eta_min, eta_max,
        f, dfdx, dfdy
    )
    if not valid:
        return -1.0

    # 2. 초기 파라미터 설정 (u, v만 설정, 나머지 0)
    if shape_type == AFFINE:
        n_params = 6
    else:
        n_params = 12

    p = np.zeros(n_params, dtype=np.float64)
    if shape_type == AFFINE:
        p[0] = initial_u   # u
        p[3] = initial_v   # v
    else:
        p[0] = initial_u   # u
        p[6] = initial_v   # v

    # 3. Warp: 로컬 좌표 → warped 좌표
    warp(p, xsi, eta, xsi_w, eta_w, shape_type)

    # 4. Deformed 좌표 계산 (절대 좌표)
    for i in range(n_pixels):
        x_def[i] = float64(cx) + xsi_w[i]
        y_def[i] = float64(cy) + eta_w[i]

    # 5. 경계 체크
    if not is_inside_batch(y_def, x_def, img_h, img_w, order):
        return -1.0

    # 6. B-spline 보간으로 deformed subset 추출
    if order == 3:
        for i in range(n_pixels):
            g[i] = _interp2d_cubic(coeffs, y_def[i], x_def[i])
    else:
        for i in range(n_pixels):
            g[i] = _interp2d_quintic(coeffs, y_def[i], x_def[i])

    # 7. Deformed subset 통계
    g_sum = 0.0
    for i in range(n_pixels):
        g_sum += g[i]
    g_mean = g_sum / float64(n_pixels)

    g_tilde_sq = 0.0
    for i in range(n_pixels):
        diff = g[i] - g_mean
        g_tilde_sq += diff * diff
    g_tilde = np.sqrt(g_tilde_sq)

    if g_tilde < 1e-10:
        return -1.0

    # 8. ZNSSD → ZNCC
    znssd = compute_znssd(f, f_mean, f_tilde, g, g_mean, g_tilde, n_pixels)
    zncc = 1.0 - 0.5 * znssd

    return zncc

@jit(nopython=True, cache=True)
def process_poi_variable(
    ref_image, grad_x, grad_y,
    coeffs, order,
    cx, cy,
    initial_u, initial_v,
    subset_size,
    max_iterations,
    convergence_threshold,
    shape_type,
    neighbor_valid,
    zncc_threshold,
    # 작업 버퍼 (사전 할당 — POI별)
    f, dfdx, dfdy,
    J, H, H_inv_buf,
    p, xsi_w, eta_w,
    x_def, y_def,
    g, b, dp, p_new,
    # 추가 버퍼 (variable subset 전용)
    xsi_local, eta_local
):
    """
    단일 불량 POI에 대한 Variable Subset 재계산.

    8방위 이웃 중 valid한 방향의 후보 서브셋을 1회-warp ZNCC로 평가하고,
    최고 ZNCC 후보를 선택하여 IC-GN을 수행한다.

    neighbor_valid 인덱스 매핑:
        [0]=NW 이웃 valid → S1(SE 확장)
        [1]=N  이웃 valid → S2(S 확장)
        [2]=NE 이웃 valid → S3(SW 확장)
        [3]=W  이웃 valid → S4(E 확장)
        [4]=E  이웃 valid → S5(W 확장)
        [5]=SW 이웃 valid → S6(NE 확장)
        [6]=S  이웃 valid → S7(N 확장)
        [7]=SE 이웃 valid → S8(NW 확장)

    Args:
        ref_image: 참조 이미지 (H, W) float64
        grad_x, grad_y: gradient 이미지 (H, W) float64
        coeffs: deformed 이미지의 B-spline 계수 (H, W) float64
        order: 보간 차수 (3 또는 5)
        cx, cy: POI 중심 좌표 (정수)
        initial_u, initial_v: FFT-CC 초기 추정 변위
        subset_size: 서브셋 크기 (홀수)
        max_iterations: 최대 반복 횟수
        convergence_threshold: 수렴 임계값
        shape_type: AFFINE(0) 또는 QUADRATIC(1)
        neighbor_valid: (8,) boolean — 8방위 이웃의 IC-GN valid 여부
        zncc_threshold: 최종 ZNCC 합격 기준 (예: 0.9)

        [작업 버퍼]
        f, dfdx, dfdy: (n_pixels,)
        J: (n_pixels, n_params)
        H, H_inv_buf: (n_params, n_params)
        p: (n_pixels,)
        xsi_w, eta_w, x_def, y_def, g: (n_pixels,)
        b, dp, p_new: (n_params,)
        xsi_local, eta_local: (n_pixels,) — 비대칭 로컬 좌표 저장 전용

    Returns:
        (zncc, n_iter, converged, fail_code, best_subset_type)
        best_subset_type: 선택된 서브셋 타입 (S1=1 ~ S8=8), 후보 없으면 -1
        p는 최종 파라미터로 업데이트됨
    """
    M = subset_size // 2
    n_pixels = subset_size * subset_size
    n_params = get_num_params(shape_type)

    # --- 서브셋 타입별 좌표 범위 테이블 (S1~S8, 인덱스 0~7) ---
    xsi_mins = np.array([0, -M, -2*M, 0, -2*M, 0, -M, -2*M], dtype=np.int64)
    xsi_maxs = np.array([2*M, M, 0, 2*M, 0, 2*M, M, 0], dtype=np.int64)
    eta_mins = np.array([0, 0, 0, -M, -M, -2*M, -2*M, -2*M], dtype=np.int64)
    eta_maxs = np.array([2*M, 2*M, 2*M, M, M, 0, 0, 0], dtype=np.int64)

    # =========================================================
    #  Phase 1: 8방위 후보 평가 (1회-warp ZNCC)
    # =========================================================
    best_zncc_candidate = -1.0
    best_idx = -1

    for i in range(8):
        if not neighbor_valid[i]:
            continue

        xsi_min_i = xsi_mins[i]
        xsi_max_i = xsi_maxs[i]
        eta_min_i = eta_mins[i]
        eta_max_i = eta_maxs[i]

        # 비대칭 로컬 좌표 생성 (xsi_local, eta_local에 직접 기록)
        idx = 0
        for row_offset in range(eta_min_i, eta_max_i + 1):
            for col_offset in range(xsi_min_i, xsi_max_i + 1):
                xsi_local[idx] = float64(col_offset)
                eta_local[idx] = float64(row_offset)
                idx += 1

        # 1회-warp ZNCC 평가
        zncc_i = evaluate_candidate_zncc(
            ref_image, grad_x, grad_y,
            coeffs, order,
            cx, cy,
            initial_u, initial_v,
            xsi_min_i, xsi_max_i, eta_min_i, eta_max_i,
            xsi_local, eta_local,
            shape_type,
            f, dfdx, dfdy,
            xsi_w, eta_w,
            x_def, y_def,
            g
        )

        if zncc_i > best_zncc_candidate:
            best_zncc_candidate = zncc_i
            best_idx = i

    # 후보 없음 또는 모든 후보 실패
    if best_idx < 0 or best_zncc_candidate < 0.0:
        return 0.0, 0, False, ICGN_FAIL_LOW_ZNCC, -1

    # =========================================================
    #  Phase 2: 선택된 후보로 IC-GN 수행
    # =========================================================
    best_xsi_min = xsi_mins[best_idx]
    best_xsi_max = xsi_maxs[best_idx]
    best_eta_min = eta_mins[best_idx]
    best_eta_max = eta_maxs[best_idx]

    # 최종 비대칭 로컬 좌표 생성 (xsi_local, eta_local)
    idx = 0
    for row_offset in range(best_eta_min, best_eta_max + 1):
        for col_offset in range(best_xsi_min, best_xsi_max + 1):
            xsi_local[idx] = float64(col_offset)
            eta_local[idx] = float64(row_offset)
            idx += 1

    # 1. Reference subset 추출 (비대칭)
    f_mean, f_tilde, valid = extract_reference_subset_variable(
        ref_image, grad_x, grad_y, cx, cy,
        best_xsi_min, best_xsi_max, best_eta_min, best_eta_max,
        f, dfdx, dfdy
    )

    if not valid:
        return 0.0, 0, False, ICGN_FAIL_FLAT_SUBSET, best_idx + 1

    # 2. Steepest Descent Image (비대칭 좌표 사용)
    compute_steepest_descent(dfdx, dfdy, xsi_local, eta_local, J, shape_type)

    # 3. Hessian
    compute_hessian(J, H)

    # 4. Hessian 역행렬
    det = np.linalg.det(H)
    if abs(det) < 1e-30:
        return 0.0, 0, False, ICGN_FAIL_SINGULAR_HESSIAN, best_idx + 1

    H_inv = np.linalg.inv(H)
    for i in range(n_params):
        for j in range(n_params):
            H_inv_buf[i, j] = H_inv[i, j]

    # 5. 초기 파라미터 설정
    for k in range(n_params):
        p[k] = 0.0
    if shape_type == AFFINE:
        p[0] = initial_u
        p[3] = initial_v
    else:
        p[0] = initial_u
        p[6] = initial_v

    # 6. IC-GN 반복
    #    xsi=xsi_local, eta=eta_local (읽기 전용, 버퍼 분리됨)
    #    xsi_w, eta_w = warp 출력 버퍼
    #    x_def, y_def = deformed 좌표 버퍼
    zncc, n_iter, conv, fail_code = icgn_iterate(
        f, f_mean, f_tilde,
        J, H_inv_buf,
        coeffs, order,
        ref_image.shape[0], ref_image.shape[1],
        cx, cy,
        xsi_local, eta_local,
        p,
        subset_size,
        max_iterations,
        convergence_threshold,
        shape_type,
        n_pixels,
        n_params,
        xsi_w, eta_w,
        x_def, y_def,
        g, b, dp, p_new
    )

    # 7. ZNCC 임계값 확인
    if conv and zncc >= zncc_threshold:
        return zncc, n_iter, True, ICGN_SUCCESS, best_idx + 1
    else:
        return zncc, n_iter, False, fail_code, best_idx + 1

@jit(nopython=True, parallel=True, cache=True)
def process_bad_pois_parallel(
    ref_image, grad_x, grad_y,
    coeffs, order,
    points_x, points_y,
    initial_u, initial_v,
    subset_size,
    max_iterations,
    convergence_threshold,
    shape_type,
    bad_indices,
    all_neighbor_valid,
    zncc_threshold,
    # 결과 배열
    result_p,
    result_zncc,
    result_iter,
    result_conv,
    result_fail,
    result_subset_type,
    # 2D 작업 버퍼
    buf_f, buf_dfdx, buf_dfdy,
    buf_J, buf_H, buf_H_inv,
    buf_p, buf_xsi_w, buf_eta_w,
    buf_x_def, buf_y_def,
    buf_g, buf_b, buf_dp, buf_p_new,
    buf_xsi_local, buf_eta_local
):
    """
    불량 POI들을 prange로 병렬 처리하여 Variable Subset 재계산 수행.

    Args:
        ref_image, grad_x, grad_y: 참조 이미지 및 gradient (공유, 읽기전용)
        coeffs: deformed B-spline 계수 (공유, 읽기전용)
        order: 보간 차수
        points_x, points_y: 전체 POI 좌표 배열 (n_total_poi,)
        initial_u, initial_v: 전체 FFT-CC 초기 변위 배열 (n_total_poi,)
        subset_size, max_iterations, convergence_threshold: 설정
        shape_type: AFFINE 또는 QUADRATIC
        bad_indices: 불량 POI의 인덱스 배열 (n_bad,) int64
            points_x/y, initial_u/v에서의 인덱스
        all_neighbor_valid: 이웃 valid 정보 (n_bad, 8) boolean
            bad_indices[k]번 POI의 8방위 이웃 valid 여부
        zncc_threshold: ZNCC 합격 기준 (예: 0.9)

        [결과 배열 — n_bad 크기]
        result_p: (n_bad, n_params)
        result_zncc: (n_bad,)
        result_iter: (n_bad,) int32
        result_conv: (n_bad,) boolean
        result_fail: (n_bad,) int32
        result_subset_type: (n_bad,) int32 — 선택된 서브셋 타입

        [2D 작업 버퍼 — n_bad 크기, allocate_variable_batch_buffers()로 생성]
    """
    n_bad = len(bad_indices)
    n_params = buf_p.shape[1]

    for k in prange(n_bad):
        idx = bad_indices[k]
        cx = points_x[idx]
        cy = points_y[idx]

        zncc, n_iter, conv, fail_code, best_st = process_poi_variable(
            ref_image, grad_x, grad_y,
            coeffs, order,
            cx, cy,
            initial_u[idx], initial_v[idx],
            subset_size,
            max_iterations,
            convergence_threshold,
            shape_type,
            all_neighbor_valid[k],
            zncc_threshold,
            # 작업 버퍼 (k 행)
            buf_f[k], buf_dfdx[k], buf_dfdy[k],
            buf_J[k], buf_H[k], buf_H_inv[k],
            buf_p[k], buf_xsi_w[k], buf_eta_w[k],
            buf_x_def[k], buf_y_def[k],
            buf_g[k], buf_b[k], buf_dp[k], buf_p_new[k],
            buf_xsi_local[k], buf_eta_local[k]
        )

        result_zncc[k] = zncc
        result_iter[k] = n_iter
        result_conv[k] = conv
        result_fail[k] = fail_code
        result_subset_type[k] = best_st

        for j in range(n_params):
            result_p[k, j] = buf_p[k, j]

def allocate_variable_batch_buffers(n_bad, n_pixels, n_params):
    """
    Variable Subset 배치 처리용 2D 작업 버퍼 할당.

    기존 allocate_batch_buffers와 동일 구조에
    xsi_local, eta_local 버퍼가 추가됨.

    Args:
        n_bad: 불량 POI 수
        n_pixels: 서브셋 픽셀 수 (subset_size²)
        n_params: 파라미터 수 (6 또는 12)

    Returns:
        dict of 2D buffers
    """
    return {
        'f': np.empty((n_bad, n_pixels), dtype=np.float64),
        'dfdx': np.empty((n_bad, n_pixels), dtype=np.float64),
        'dfdy': np.empty((n_bad, n_pixels), dtype=np.float64),
        'J': np.empty((n_bad, n_pixels, n_params), dtype=np.float64),
        'H': np.empty((n_bad, n_params, n_params), dtype=np.float64),
        'H_inv': np.empty((n_bad, n_params, n_params), dtype=np.float64),
        'p': np.empty((n_bad, n_params), dtype=np.float64),
        'xsi_w': np.empty((n_bad, n_pixels), dtype=np.float64),
        'eta_w': np.empty((n_bad, n_pixels), dtype=np.float64),
        'x_def': np.empty((n_bad, n_pixels), dtype=np.float64),
        'y_def': np.empty((n_bad, n_pixels), dtype=np.float64),
        'g': np.empty((n_bad, n_pixels), dtype=np.float64),
        'b': np.empty((n_bad, n_params), dtype=np.float64),
        'dp': np.empty((n_bad, n_params), dtype=np.float64),
        'p_new': np.empty((n_bad, n_params), dtype=np.float64),
        'xsi_local': np.empty((n_bad, n_pixels), dtype=np.float64),
        'eta_local': np.empty((n_bad, n_pixels), dtype=np.float64),
    }
