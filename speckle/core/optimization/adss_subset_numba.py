import numpy as np
from numba import jit, float64, prange

from .variable_subset_numba import extract_reference_subset_variable
from .interpolation_numba import (
    is_inside_batch,
    _interp2d_cubic,
    _interp2d_quintic,
)
from .shape_function_numba import (
    AFFINE, QUADRATIC,
    warp,
    get_num_params,
    compute_steepest_descent,
    compute_hessian,
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
ADSS-DIC (Adaptive Subset-Subdivision) Numba 가속 모듈

서브셋을 8개의 quarter-subset으로 분할하여 불연속(균열)을 포함하는
POI에서 균열이 없는 영역만으로 IC-GN을 재계산한다.

Quarter-subset 정의:
    십자형 (half-subset):
        Q1: Upper half   ξ ∈ [-M, +M], η ∈ [-M, 0]   → (2M+1)×(M+1) 픽셀
        Q2: Lower half   ξ ∈ [-M, +M], η ∈ [0, +M]   → (2M+1)×(M+1) 픽셀
        Q3: Left half    ξ ∈ [-M, 0],  η ∈ [-M, +M]   → (M+1)×(2M+1) 픽셀
        Q4: Right half   ξ ∈ [0, +M],  η ∈ [-M, +M]   → (M+1)×(2M+1) 픽셀
    대각형 (quarter-subset):
        Q5: Upper-left   ξ ∈ [-M, 0],  η ∈ [-M, 0]   → (M+1)×(M+1) 픽셀
        Q6: Upper-right  ξ ∈ [0, +M],  η ∈ [-M, 0]   → (M+1)×(M+1) 픽셀
        Q7: Lower-left   ξ ∈ [-M, 0],  η ∈ [0, +M]   → (M+1)×(M+1) 픽셀
        Q8: Lower-right  ξ ∈ [0, +M],  η ∈ [0, +M]   → (M+1)×(M+1) 픽셀

    Q0: Full subset (참조용) ξ ∈ [-M, +M], η ∈ [-M, +M] → (2M+1)² 픽셀

여기서 ξ는 x(col) 방향, η는 y(row) 방향이다.

Reference:
    Zhao, J., & Pan, B. "Adaptive subset-subdivision for automatic
    digital image correlation calculation on discontinuous shape
    and deformation." Applied Optics, 2025.
"""

# === Quarter-subset 상수 ===
Q0 = 0  # Full subset (참조용, ADSS에서 미사용)
Q1 = 1  # Upper half
Q2 = 2  # Lower half
Q3 = 3  # Left half
Q4 = 4  # Right half
Q5 = 5  # Upper-left
Q6 = 6  # Upper-right
Q7 = 7  # Lower-left
Q8 = 8  # Lower-right

# 각 quarter-subset에 대응하는 이웃 방향 (dy, dx)
# 해당 방향의 well-matched 이웃 POI에서 초기값을 가져옴
QUARTER_TO_NEIGHBOR = {
    Q1: (-1,  0),  # upper half  → 위쪽 이웃
    Q2: ( 1,  0),  # lower half  → 아래쪽 이웃
    Q3: ( 0, -1),  # left half   → 왼쪽 이웃
    Q4: ( 0,  1),  # right half  → 오른쪽 이웃
    Q5: (-1, -1),  # upper-left  → 좌상 이웃
    Q6: (-1,  1),  # upper-right → 우상 이웃
    Q7: ( 1, -1),  # lower-left  → 좌하 이웃
    Q8: ( 1,  1),  # lower-right → 우하 이웃
}

def generate_quarter_local_coordinates(subset_size, subset_type):
    """
    ADSS-DIC quarter-subset 로컬 좌표 생성.

    Parameters
    ----------
    subset_size : int
        원래 서브셋 크기 (홀수, e.g., 21 → M=10).
    subset_type : int
        Q0~Q8 중 하나.

    Returns
    -------
    local_x : np.ndarray (1D, float64)
        ξ 좌표 (열 방향).
    local_y : np.ndarray (1D, float64)
        η 좌표 (행 방향).
    """
    M = subset_size // 2

    if subset_type == Q0:
        xsi_range = np.arange(-M, M + 1, dtype=np.float64)
        eta_range = np.arange(-M, M + 1, dtype=np.float64)
    elif subset_type == Q1:  # Upper half
        xsi_range = np.arange(-M, M + 1, dtype=np.float64)
        eta_range = np.arange(-M, 0 + 1, dtype=np.float64)
    elif subset_type == Q2:  # Lower half
        xsi_range = np.arange(-M, M + 1, dtype=np.float64)
        eta_range = np.arange(0, M + 1, dtype=np.float64)
    elif subset_type == Q3:  # Left half
        xsi_range = np.arange(-M, 0 + 1, dtype=np.float64)
        eta_range = np.arange(-M, M + 1, dtype=np.float64)
    elif subset_type == Q4:  # Right half
        xsi_range = np.arange(0, M + 1, dtype=np.float64)
        eta_range = np.arange(-M, M + 1, dtype=np.float64)
    elif subset_type == Q5:  # Upper-left
        xsi_range = np.arange(-M, 0 + 1, dtype=np.float64)
        eta_range = np.arange(-M, 0 + 1, dtype=np.float64)
    elif subset_type == Q6:  # Upper-right
        xsi_range = np.arange(0, M + 1, dtype=np.float64)
        eta_range = np.arange(-M, 0 + 1, dtype=np.float64)
    elif subset_type == Q7:  # Lower-left
        xsi_range = np.arange(-M, 0 + 1, dtype=np.float64)
        eta_range = np.arange(0, M + 1, dtype=np.float64)
    elif subset_type == Q8:  # Lower-right
        xsi_range = np.arange(0, M + 1, dtype=np.float64)
        eta_range = np.arange(0, M + 1, dtype=np.float64)
    else:
        raise ValueError(f"Invalid subset_type: {subset_type}")

    # meshgrid: eta → 행(첫번째 축), xsi → 열(두번째 축)
    eta_2d, xsi_2d = np.meshgrid(eta_range, xsi_range, indexing='ij')
    local_x = xsi_2d.ravel().copy()
    local_y = eta_2d.ravel().copy()

    return local_x, local_y

@jit(nopython=True, cache=True)
def predict_initial_params(
    neighbor_x, neighbor_y,
    neighbor_params,
    target_x, target_y,
    shape_type,
    out_params
):
    """
    ADSS-DIC Eq.(3): 이웃 POI의 IC-GN 결과로부터 불량 POI의 초기값을 예측.

    1차 Taylor 전개를 사용하여 well-matched 이웃의 변위·변형률 파라미터를
    대상 POI 위치로 외삽한다.

    Parameters
    ----------
    neighbor_x, neighbor_y : float64
        이웃 POI 좌표 (x⁽ᵏ⁾, y⁽ᵏ⁾).
    neighbor_params : np.ndarray (n_params,) float64
        이웃 POI의 IC-GN 수렴 파라미터.
        Affine:    [u, ux, uy, v, vx, vy]
        Quadratic: [u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy]
    target_x, target_y : float64
        대상 (불량) POI 좌표 (x, y).
    shape_type : int
        AFFINE(0) 또는 QUADRATIC(1).
    out_params : np.ndarray (n_params,) float64
        출력 — 예측된 초기 파라미터.

    Notes
    -----
    Eq.(3):
        u_init = u⁽ᵏ⁾ + ux⁽ᵏ⁾·(x⁽ᵏ⁾ - x) + uy⁽ᵏ⁾·(y⁽ᵏ⁾ - y)
        v_init = v⁽ᵏ⁾ + vx⁽ᵏ⁾·(x⁽ᵏ⁾ - x) + vy⁽ᵏ⁾·(y⁽ᵏ⁾ - y)
        1차 변형률 gradient는 이웃 값을 그대로 복사.
    """
    dx = neighbor_x - target_x
    dy = neighbor_y - target_y

    if shape_type == 0:  # AFFINE: [u, ux, uy, v, vx, vy]
        u_k = neighbor_params[0]
        ux_k = neighbor_params[1]
        uy_k = neighbor_params[2]
        v_k = neighbor_params[3]
        vx_k = neighbor_params[4]
        vy_k = neighbor_params[5]

        out_params[0] = u_k + ux_k * dx + uy_k * dy  # u_init
        out_params[1] = ux_k                           # ux_init
        out_params[2] = uy_k                           # uy_init
        out_params[3] = v_k + vx_k * dx + vy_k * dy  # v_init
        out_params[4] = vx_k                           # vx_init
        out_params[5] = vy_k                           # vy_init

    else:  # QUADRATIC: [u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy]
        u_k = neighbor_params[0]
        ux_k = neighbor_params[1]
        uy_k = neighbor_params[2]
        v_k = neighbor_params[6]
        vx_k = neighbor_params[7]
        vy_k = neighbor_params[8]

        out_params[0] = u_k + ux_k * dx + uy_k * dy  # u_init
        out_params[1] = ux_k                           # ux_init
        out_params[2] = uy_k                           # uy_init
        out_params[3] = neighbor_params[3]             # uxx
        out_params[4] = neighbor_params[4]             # uxy
        out_params[5] = neighbor_params[5]             # uyy
        out_params[6] = v_k + vx_k * dx + vy_k * dy  # v_init
        out_params[7] = vx_k                           # vx_init
        out_params[8] = vy_k                           # vy_init
        out_params[9] = neighbor_params[9]             # vxx
        out_params[10] = neighbor_params[10]           # vxy
        out_params[11] = neighbor_params[11]           # vyy
        
@jit(nopython=True, cache=True)
def evaluate_quarter_zncc(
    ref_image, grad_x, grad_y,
    coeffs, order,
    cx, cy,
    init_params,
    xsi_min, xsi_max, eta_min, eta_max,
    xsi, eta,
    shape_type,
    n_pixels,
    # 작업 버퍼
    f, dfdx, dfdy,
    xsi_w, eta_w,
    x_def, y_def,
    g
):
    """
    단일 quarter-subset에 대해 1회 warp ZNCC를 계산.

    predict_initial_params로 예측한 전체 파라미터(변위 + gradient)를
    사용하여 warp를 1회 수행하고, reference와 deformed 간 ZNCC를 반환한다.
    8개 후보 중 최적 quarter-subset을 선별하기 위한 함수.

    Parameters
    ----------
    ref_image : (H, W) float64
        참조 이미지.
    grad_x, grad_y : (H, W) float64
        참조 이미지 gradient.
    coeffs : (H, W) float64
        deformed 이미지의 B-spline 계수.
    order : int
        보간 차수 (3 또는 5).
    cx, cy : int
        POI 중심 좌표.
    init_params : (n_params,) float64
        predict_initial_params로 예측한 초기 파라미터.
        Affine: [u, ux, uy, v, vx, vy]
        Quadratic: [u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy]
    xsi_min, xsi_max : int
        ξ (열) 방향 로컬 좌표 범위.
    eta_min, eta_max : int
        η (행) 방향 로컬 좌표 범위.
    xsi, eta : (n_pixels,) float64
        quarter-subset 로컬 좌표.
    shape_type : int
        AFFINE(0) 또는 QUADRATIC(1).
    n_pixels : int
        quarter-subset 픽셀 수.
    f, dfdx, dfdy : (n_pixels_max,) float64
        작업 버퍼 — reference subset.
    xsi_w, eta_w : (n_pixels_max,) float64
        작업 버퍼 — warped 좌표.
    x_def, y_def : (n_pixels_max,) float64
        작업 버퍼 — deformed 절대 좌표.
    g : (n_pixels_max,) float64
        작업 버퍼 — deformed subset 강도.

    Returns
    -------
    zncc : float64
        ZNCC 값 (0.0~1.0). 실패 시 -1.0.
    """
    img_h = ref_image.shape[0]
    img_w = ref_image.shape[1]

    # 1. Reference subset 추출 (비대칭 영역)
    f_mean, f_tilde, valid = extract_reference_subset_variable(
        ref_image, grad_x, grad_y, cx, cy,
        xsi_min, xsi_max, eta_min, eta_max,
        f, dfdx, dfdy
    )
    if not valid:
        return -1.0

    # 2. 전체 초기 파라미터로 warp
    warp(init_params, xsi, eta, xsi_w, eta_w, shape_type)

    # 3. Deformed 절대 좌표 계산
    for i in range(n_pixels):
        x_def[i] = float64(cx) + xsi_w[i]
        y_def[i] = float64(cy) + eta_w[i]

    # 4. 경계 체크
    if not is_inside_batch(y_def[:n_pixels], x_def[:n_pixels], img_h, img_w, order):
        return -1.0

    # 5. B-spline 보간
    if order == 3:
        for i in range(n_pixels):
            g[i] = _interp2d_cubic(coeffs, y_def[i], x_def[i])
    else:
        for i in range(n_pixels):
            g[i] = _interp2d_quintic(coeffs, y_def[i], x_def[i])

    # 6. Deformed subset 통계
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

    # 7. ZNSSD → ZNCC
    znssd = compute_znssd(f, f_mean, f_tilde, g, g_mean, g_tilde, n_pixels)
    zncc = 1.0 - 0.5 * znssd

    return zncc

@jit(nopython=True, cache=True)
def process_poi_adss(
    ref_image, grad_x, grad_y,
    coeffs, order,
    cx, cy,
    subset_size,
    max_iterations,
    convergence_threshold,
    shape_type,
    neighbor_valid,       # (8,) boolean — Q1~Q8 순서로 대응 이웃의 well-matched 여부
    neighbor_params,      # (8, n_params) — 대응 이웃의 IC-GN 파라미터
    neighbor_x,           # (8,) — 대응 이웃의 x좌표
    neighbor_y,           # (8,) — 대응 이웃의 y좌표
    zncc_threshold,
    # 작업 버퍼 (사전 할당 — POI별)
    f, dfdx, dfdy,
    J, H, H_inv_buf,
    p, xsi_w, eta_w,
    x_def, y_def,
    g, b, dp, p_new,
    xsi_local, eta_local,
    init_params_buf       # (n_params,) — 초기 파라미터 임시 버퍼
):
    """
    단일 불량 POI에 대한 ADSS-DIC 재계산.

    십자 분할(Q1~Q4)과 대각선 분할(Q5~Q8) 두 세트를 각각 평가하고,
    더 높은 ZNCC를 가진 세트에서 최적 quarter-subset을 선택하여
    IC-GN을 수행한다.

    Parameters
    ----------
    ref_image, grad_x, grad_y : (H, W) float64
    coeffs : (H, W) float64 — deformed B-spline 계수
    order : int — 보간 차수 (3 또는 5)
    cx, cy : int — POI 중심 좌표
    subset_size : int — 원래 서브셋 크기 (홀수)
    max_iterations : int
    convergence_threshold : float64
    shape_type : int — AFFINE(0) 또는 QUADRATIC(1)
    neighbor_valid : (8,) boolean
        인덱스 0~7 = Q1~Q8에 대응하는 이웃의 well-matched 여부
    neighbor_params : (8, n_params) float64
        각 이웃의 IC-GN 수렴 파라미터
    neighbor_x, neighbor_y : (8,) float64
        각 이웃의 좌표
    zncc_threshold : float64

    Returns
    -------
    (zncc, n_iter, converged, fail_code, best_quarter_type)
        best_quarter_type: 선택된 quarter (1~8), 실패 시 -1
        p 배열은 최종 파라미터로 업데이트됨
    """
    M = subset_size // 2
    n_params = get_num_params(shape_type)

    # === Quarter-subset 좌표 범위 테이블 (Q1~Q8, 인덱스 0~7) ===
    # Q1: Upper half,  Q2: Lower half,  Q3: Left half,  Q4: Right half
    # Q5: Upper-left,  Q6: Upper-right, Q7: Lower-left, Q8: Lower-right
    xsi_mins = np.array([-M, -M, -M,  0, -M,  0, -M,  0], dtype=np.int64)
    xsi_maxs = np.array([ M,  M,  0,  M,  0,  M,  0,  M], dtype=np.int64)
    eta_mins = np.array([-M,  0, -M, -M, -M, -M,  0,  0], dtype=np.int64)
    eta_maxs = np.array([ 0,  M,  M,  M,  0,  0,  M,  M], dtype=np.int64)

    # === 8개 후보 1-warp ZNCC 계산 ===
    candidate_zncc = np.full(8, -1.0, dtype=np.float64)

    for i in range(8):
        if not neighbor_valid[i]:
            continue

        # 초기값 예측 (Taylor 전개)
        predict_initial_params(
            neighbor_x[i], neighbor_y[i],
            neighbor_params[i],
            float64(cx), float64(cy),
            shape_type,
            init_params_buf
        )

        # quarter-subset 로컬 좌표 생성 (인라인)
        xsi_min_i = xsi_mins[i]
        xsi_max_i = xsi_maxs[i]
        eta_min_i = eta_mins[i]
        eta_max_i = eta_maxs[i]

        idx = 0
        for row_offset in range(eta_min_i, eta_max_i + 1):
            for col_offset in range(xsi_min_i, xsi_max_i + 1):
                xsi_local[idx] = float64(col_offset)
                eta_local[idx] = float64(row_offset)
                idx += 1
        n_pixels_i = idx

        # 1-warp ZNCC 평가
        zncc_i = evaluate_quarter_zncc(
            ref_image, grad_x, grad_y,
            coeffs, order,
            cx, cy,
            init_params_buf,
            xsi_min_i, xsi_max_i, eta_min_i, eta_max_i,
            xsi_local, eta_local,
            shape_type,
            n_pixels_i,
            f, dfdx, dfdy,
            xsi_w, eta_w,
            x_def, y_def,
            g
        )
        candidate_zncc[i] = zncc_i

    # === 세트 비교: 십자(Q1~Q4) vs 대각선(Q5~Q8) ===
    best_cross = -1.0
    best_cross_idx = -1
    for i in range(4):  # Q1~Q4 → 인덱스 0~3
        if candidate_zncc[i] > best_cross:
            best_cross = candidate_zncc[i]
            best_cross_idx = i

    best_diag = -1.0
    best_diag_idx = -1
    for i in range(4, 8):  # Q5~Q8 → 인덱스 4~7
        if candidate_zncc[i] > best_diag:
            best_diag = candidate_zncc[i]
            best_diag_idx = i

    # 더 높은 세트에서 최적 후보 선택
    if best_cross >= best_diag:
        best_idx = best_cross_idx
    else:
        best_idx = best_diag_idx

    # 후보 없음
    if best_idx < 0 or candidate_zncc[best_idx] < 0.0:
        return 0.0, 0, False, ICGN_FAIL_LOW_ZNCC, -1

    # === 선택된 quarter-subset으로 IC-GN 수행 ===
    best_xsi_min = xsi_mins[best_idx]
    best_xsi_max = xsi_maxs[best_idx]
    best_eta_min = eta_mins[best_idx]
    best_eta_max = eta_maxs[best_idx]

    # 로컬 좌표 재생성
    idx = 0
    for row_offset in range(best_eta_min, best_eta_max + 1):
        for col_offset in range(best_xsi_min, best_xsi_max + 1):
            xsi_local[idx] = float64(col_offset)
            eta_local[idx] = float64(row_offset)
            idx += 1
    n_pixels = idx

    # 초기값 재계산
    predict_initial_params(
        neighbor_x[best_idx], neighbor_y[best_idx],
        neighbor_params[best_idx],
        float64(cx), float64(cy),
        shape_type,
        init_params_buf
    )

    # 1. Reference subset 추출
    f_mean, f_tilde, valid = extract_reference_subset_variable(
        ref_image, grad_x, grad_y, cx, cy,
        best_xsi_min, best_xsi_max, best_eta_min, best_eta_max,
        f, dfdx, dfdy
    )
    if not valid:
        return 0.0, 0, False, ICGN_FAIL_FLAT_SUBSET, best_idx + 1

    # 2. Steepest Descent Image
    compute_steepest_descent(dfdx[:n_pixels], dfdy[:n_pixels],
                             xsi_local[:n_pixels], eta_local[:n_pixels],
                             J[:n_pixels], shape_type)

    # 3. Hessian
    compute_hessian(J[:n_pixels], H)

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
        p[k] = init_params_buf[k]

    # 6. IC-GN 반복 — 버퍼를 n_pixels 크기로 슬라이스
    zncc, n_iter, conv, fail_code = icgn_iterate(
        f[:n_pixels], f_mean, f_tilde,
        J[:n_pixels], H_inv_buf,
        coeffs, order,
        ref_image.shape[0], ref_image.shape[1],
        cx, cy,
        xsi_local[:n_pixels], eta_local[:n_pixels],
        p,
        subset_size,
        max_iterations,
        convergence_threshold,
        shape_type,
        n_pixels,
        n_params,
        xsi_w[:n_pixels], eta_w[:n_pixels],
        x_def[:n_pixels], y_def[:n_pixels],
        g[:n_pixels], b, dp, p_new
    )

    # 7. ZNCC 임계값 확인
    if conv and zncc >= zncc_threshold:
        return zncc, n_iter, True, ICGN_SUCCESS, best_idx + 1
    else:
        return zncc, n_iter, False, fail_code, best_idx + 1

@jit(nopython=True, parallel=True, cache=True)
def process_bad_pois_adss_parallel(
    ref_image, grad_x, grad_y,
    coeffs, order,
    points_x, points_y,
    subset_size,
    max_iterations,
    convergence_threshold,
    shape_type,
    bad_indices,
    all_neighbor_valid,
    all_neighbor_params,
    all_neighbor_x,
    all_neighbor_y,
    zncc_threshold,
    # 결과 배열
    result_p,
    result_zncc,
    result_iter,
    result_conv,
    result_fail,
    result_quarter_type,
    # 2D 작업 버퍼
    buf_f, buf_dfdx, buf_dfdy,
    buf_J, buf_H, buf_H_inv,
    buf_p, buf_xsi_w, buf_eta_w,
    buf_x_def, buf_y_def,
    buf_g, buf_b, buf_dp, buf_p_new,
    buf_xsi_local, buf_eta_local,
    buf_init_params
):
    """
    불량 POI들을 prange로 병렬 처리하여 ADSS-DIC 재계산 수행.

    Parameters
    ----------
    ref_image, grad_x, grad_y : (H, W) float64
        참조 이미지 및 gradient (공유, 읽기전용).
    coeffs : (H, W) float64
        deformed B-spline 계수 (공유, 읽기전용).
    order : int
        보간 차수 (3 또는 5).
    points_x, points_y : (n_total_poi,) int64
        전체 POI 좌표 배열.
    subset_size : int
    max_iterations : int
    convergence_threshold : float64
    shape_type : int
        AFFINE(0) 또는 QUADRATIC(1).
    bad_indices : (n_bad,) int64
        불량 POI의 인덱스 (points_x/y에서의 인덱스).
    all_neighbor_valid : (n_bad, 8) boolean
        각 불량 POI의 8방위 이웃 well-matched 여부.
    all_neighbor_params : (n_bad, 8, n_params) float64
        각 이웃의 IC-GN 수렴 파라미터.
    all_neighbor_x, all_neighbor_y : (n_bad, 8) float64
        각 이웃의 좌표.
    zncc_threshold : float64

    결과 배열 (n_bad 크기)
    ----------
    result_p : (n_bad, n_params)
    result_zncc : (n_bad,)
    result_iter : (n_bad,) int32
    result_conv : (n_bad,) boolean
    result_fail : (n_bad,) int32
    result_quarter_type : (n_bad,) int32
        선택된 quarter-subset 타입 (1~8), 실패 시 -1.

    2D 작업 버퍼 (allocate_adss_batch_buffers로 생성)
    ----------
    buf_f ~ buf_init_params : 각 (n_bad, dim)
    """
    n_bad = len(bad_indices)
    n_params = buf_p.shape[1]

    for k in prange(n_bad):
        poi_idx = bad_indices[k]
        cx = points_x[poi_idx]
        cy = points_y[poi_idx]

        zncc, n_iter, conv, fail_code, best_qt = process_poi_adss(
            ref_image, grad_x, grad_y,
            coeffs, order,
            cx, cy,
            subset_size,
            max_iterations,
            convergence_threshold,
            shape_type,
            all_neighbor_valid[k],
            all_neighbor_params[k],
            all_neighbor_x[k],
            all_neighbor_y[k],
            zncc_threshold,
            # 작업 버퍼 (k 행)
            buf_f[k], buf_dfdx[k], buf_dfdy[k],
            buf_J[k], buf_H[k], buf_H_inv[k],
            buf_p[k], buf_xsi_w[k], buf_eta_w[k],
            buf_x_def[k], buf_y_def[k],
            buf_g[k], buf_b[k], buf_dp[k], buf_p_new[k],
            buf_xsi_local[k], buf_eta_local[k],
            buf_init_params[k]
        )

        result_zncc[k] = zncc
        result_iter[k] = n_iter
        result_conv[k] = conv
        result_fail[k] = fail_code
        result_quarter_type[k] = best_qt

        for j in range(n_params):
            result_p[k, j] = buf_p[k, j]


def allocate_adss_batch_buffers(n_bad, n_pixels_max, n_params):
    """
    ADSS-DIC 배치 처리용 2D 작업 버퍼 할당.

    Parameters
    ----------
    n_bad : int
        불량 POI 수.
    n_pixels_max : int
        quarter-subset 최대 픽셀 수.
        십자형: (2M+1)*(M+1), 대각형: (M+1)²
        → 최대값 = (2M+1)*(M+1).
    n_params : int
        파라미터 수 (6 또는 12).

    Returns
    -------
    dict of 2D numpy arrays.
    """
    return {
        'f': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'dfdx': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'dfdy': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'J': np.zeros((n_bad, n_pixels_max, n_params), dtype=np.float64),
        'H': np.zeros((n_bad, n_params, n_params), dtype=np.float64),
        'H_inv': np.zeros((n_bad, n_params, n_params), dtype=np.float64),
        'p': np.zeros((n_bad, n_params), dtype=np.float64),
        'xsi_w': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'eta_w': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'x_def': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'y_def': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'g': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'b': np.zeros((n_bad, n_params), dtype=np.float64),
        'dp': np.zeros((n_bad, n_params), dtype=np.float64),
        'p_new': np.zeros((n_bad, n_params), dtype=np.float64),
        'xsi_local': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'eta_local': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'init_params': np.zeros((n_bad, n_params), dtype=np.float64),
    }
