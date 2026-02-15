"""
Numba IC-GN 코어 모듈

IC-GN (Inverse Compositional Gauss-Newton) 반복 루프의 Numba JIT 구현.
기존 icgn.py의 _icgn_iterate, process_poi와 수치적으로 동일한 결과를 생성하면서,
Phase 4에서 prange 병렬 루프 안에서 nopython 모드로 직접 호출 가능.

설계 원칙:
    - 모든 함수가 @jit(nopython=True)로 컴파일
    - Python 객체 의존성 제거: ImageInterpolator → coeffs 배열 + order 정수
    - str 분기 제거: shape_function str → shape_type 정수 (AFFINE=0, QUADRATIC=1)
    - 메모리 사전 할당: 반복 루프 내에서 배열 재할당 최소화
    - leaf 함수들 (interpolation_numba, shape_function_numba) 직접 호출

함수 호출 구조:
    process_poi_numba
    ├── extract_reference_subset  (reference image에서 subset 추출)
    ├── compute_steepest_descent  (Jacobian: pre-loop, 1회)
    ├── compute_hessian           (Hessian: pre-loop, 1회)
    ├── np.linalg.inv(H)         (Hessian 역행렬: pre-loop, 1회)
    └── icgn_iterate              (핵심 반복 루프)
        ├── warp                  (좌표 변환)
        ├── is_inside_batch       (경계 체크)
        ├── interp2d              (보간)
        ├── compute_znssd         (ZNSSD/ZNCC 계산)
        ├── compute_residual_b    (잔차 → b = -J^T @ residual)
        ├── matvec                (dp = H_inv @ b)
        ├── check_convergence     (수렴 판정)
        └── update_warp           (파라미터 업데이트)

References:
    - Pan, B., et al. Experimental Mechanics, 2013.
    - Pan, B., et al. Optics and Lasers in Engineering, 2013.
    - Jiang, Z., et al. Optics and Lasers in Engineering, 2014.
"""

import numpy as np
from numba import jit, prange, float64, int64, int32, boolean, types
from scipy.ndimage import spline_filter

from .interpolation_numba import (
    prefilter_image,
    interp2d,
    is_inside_batch,
    _interp2d_cubic,
    _interp2d_quintic,
)
from .shape_function_numba import (
    AFFINE, QUADRATIC,
    NUM_PARAMS_AFFINE, NUM_PARAMS_QUADRATIC,
    warp,
    compute_steepest_descent,
    compute_hessian,
    check_convergence,
    update_warp,
    get_num_params,
)


# =============================================================================
#  실패 코드 상수 (results.py와 동일)
# =============================================================================
# Numba nopython에서 Python 모듈 상수를 import할 수 없으므로 여기서 재정의

ICGN_SUCCESS = 0
ICGN_FAIL_LOW_ZNCC = 1
ICGN_FAIL_DIVERGED = 2
ICGN_FAIL_OUT_OF_BOUNDS = 3
ICGN_FAIL_SINGULAR_HESSIAN = 4
ICGN_FAIL_FLAT_SUBSET = 5
ICGN_FAIL_MAX_DISPLACEMENT = 6
ICGN_FAIL_FLAT_TARGET = 7


# =============================================================================
#  1. Reference Subset 추출
# =============================================================================

@jit(nopython=True, cache=True)
def extract_reference_subset(
    ref_image, grad_x, grad_y, cx, cy, subset_size,
    f_out, dfdx_out, dfdy_out
):
    """
    Reference subset 추출 — nopython 호환

    Args:
        ref_image: 참조 이미지 (H, W) float64
        grad_x, grad_y: gradient 이미지 (H, W) float64
        cx, cy: POI 중심 좌표 (정수)
        subset_size: 서브셋 크기 (홀수)
        f_out: 출력 — reference 강도 벡터 (n_pixels,)
        dfdx_out: 출력 — x gradient 벡터 (n_pixels,)
        dfdy_out: 출력 — y gradient 벡터 (n_pixels,)

    Returns:
        (f_mean, f_tilde, valid)
        valid=False이면 flat subset 또는 경계 밖
    """
    half = subset_size // 2
    h = ref_image.shape[0]
    w = ref_image.shape[1]

    # 경계 체크
    if cy - half < 0 or cy + half >= h or cx - half < 0 or cx + half >= w:
        return 0.0, 0.0, False

    # subset 추출 (row-major ravel)
    idx = 0
    for row in range(cy - half, cy + half + 1):
        for col in range(cx - half, cx + half + 1):
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


# =============================================================================
#  2. ZNSSD 계산
# =============================================================================

@jit(nopython=True, cache=True)
def compute_znssd(f, f_mean, f_tilde, g, g_mean, g_tilde, n):
    """
    Zero-mean Normalized Sum of Squared Differences

    Args:
        f, g: 강도 벡터 (n,)
        f_mean, g_mean: 평균
        f_tilde, g_tilde: norm(f - f_mean)
        n: 유효 픽셀 수

    Returns:
        ZNSSD 값 (0 = 완벽 매칭, 4 = 완전 불일치)
    """
    s = 0.0
    inv_ft = 1.0 / f_tilde
    inv_gt = 1.0 / g_tilde
    for i in range(n):
        diff = (f[i] - f_mean) * inv_ft - (g[i] - g_mean) * inv_gt
        s += diff * diff
    return s


# =============================================================================
#  3. 잔차 벡터 및 b = -J^T @ residual 계산
# =============================================================================

@jit(nopython=True, cache=True)
def compute_b_vector(f, f_mean, f_tilde, g, g_mean, g_tilde,
                     J, b_out, n_pixels, n_params):
    """
    잔차 계산 후 b = -J^T @ residual 산출 (in-place)

    residual[i] = (f[i] - f_mean) - (f_tilde / g_tilde) * (g[i] - g_mean)
    b[j] = -sum_i( J[i,j] * residual[i] )

    J^T @ residual을 수동 루프로 계산 (nopython 호환)
    """
    scale = f_tilde / g_tilde

    for j in range(n_params):
        b_out[j] = 0.0

    for i in range(n_pixels):
        res_i = (f[i] - f_mean) - scale * (g[i] - g_mean)
        for j in range(n_params):
            b_out[j] -= J[i, j] * res_i


# =============================================================================
#  4. 행렬-벡터 곱 dp = H_inv @ b
# =============================================================================

@jit(nopython=True, cache=True)
def matvec(A, x, y, n):
    """
    행렬-벡터 곱: y = A @ x (in-place)

    Args:
        A: (n, n) 행렬
        x: (n,) 벡터
        y: (n,) 출력 벡터
        n: 차원
    """
    for i in range(n):
        s = 0.0
        for j in range(n):
            s += A[i, j] * x[j]
        y[i] = s


# =============================================================================
#  5. IC-GN 반복 루프 (핵심)
# =============================================================================

@jit(nopython=True, cache=True)
def icgn_iterate(
    f, f_mean, f_tilde,
    J, H_inv,
    coeffs, order,
    img_h, img_w,
    cx, cy,
    xsi, eta,
    p,
    subset_size,
    max_iterations,
    convergence_threshold,
    shape_type,
    n_pixels,
    n_params,
    # 작업 버퍼 (사전 할당)
    xsi_w, eta_w,
    x_def, y_def,
    g, b, dp, p_new
):
    """
    단일 POI에 대한 IC-GN 반복 — nopython JIT

    기존 icgn.py의 _icgn_iterate와 수치적으로 동일한 결과.

    Args:
        f: reference subset 강도 (n_pixels,)
        f_mean, f_tilde: reference subset 통계
        J: Steepest Descent Image (n_pixels, n_params)
        H_inv: 역 Hessian (n_params, n_params)
        coeffs: B-spline 계수 (img_h, img_w)
        order: 보간 차수 (3 또는 5)
        img_h, img_w: 이미지 크기
        cx, cy: POI 중심 좌표
        xsi, eta: 로컬 좌표 (n_pixels,)
        p: 초기 파라미터 (n_params,) — 수정됨
        subset_size: 서브셋 크기
        max_iterations: 최대 반복 횟수
        convergence_threshold: 수렴 임계값
        shape_type: AFFINE(0) 또는 QUADRATIC(1)
        n_pixels: 픽셀 수 (subset_size²)
        n_params: 파라미터 수 (6 또는 12)
        xsi_w, eta_w: warp된 좌표 버퍼 (n_pixels,)
        x_def, y_def: deformed 좌표 버퍼 (n_pixels,)
        g: target subset 버퍼 (n_pixels,)
        b: b 벡터 버퍼 (n_params,)
        dp: delta p 버퍼 (n_params,)
        p_new: 업데이트된 p 버퍼 (n_params,)

    Returns:
        (zncc, n_iter, converged, fail_code)
        p는 in-place로 업데이트됨
    """
    n_iter = 0
    conv = False
    zncc = 0.0
    fail_code = ICGN_SUCCESS

    if shape_type == AFFINE:
        u_idx = 0
        v_idx = 3
    else:
        u_idx = 0
        v_idx = 6

    p_initial_u = p[u_idx]
    p_initial_v = p[v_idx]

    half = subset_size // 2
    reference_half = 10
    scale_factor = float64(half) / float64(reference_half)
    divergence_threshold = 1.0 * scale_factor
    max_displacement_change = 5.0 * scale_factor

    for iteration in range(max_iterations):
        n_iter = iteration + 1

        # 1. Warp
        warp(p, xsi, eta, xsi_w, eta_w, shape_type)

        for i in range(n_pixels):
            x_def[i] = float64(cx) + xsi_w[i]
            y_def[i] = float64(cy) + eta_w[i]

        # 2. 경계 체크
        if not is_inside_batch(y_def, x_def, img_h, img_w, order):
            fail_code = ICGN_FAIL_OUT_OF_BOUNDS
            conv = False
            break

        # 3. 보간
        if order == 3:
            for i in range(n_pixels):
                g[i] = _interp2d_cubic(coeffs, y_def[i], x_def[i])
        else:
            for i in range(n_pixels):
                g[i] = _interp2d_quintic(coeffs, y_def[i], x_def[i])

        # 4. Target subset 통계
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
            fail_code = ICGN_FAIL_FLAT_TARGET
            break

        # 5. ZNSSD / ZNCC
        znssd = compute_znssd(f, f_mean, f_tilde, g, g_mean, g_tilde, n_pixels)
        zncc = 1.0 - 0.5 * znssd

        if zncc < 0.5:
            fail_code = ICGN_FAIL_LOW_ZNCC
            conv = False
            break

        # 6. b = -J^T @ residual
        compute_b_vector(f, f_mean, f_tilde, g, g_mean, g_tilde,
                         J, b, n_pixels, n_params)

        # 7. dp = H_inv @ b
        matvec(H_inv, b, dp, n_params)

        # 8. 수렴 체크
        conv, dp_norm = check_convergence(dp, half, convergence_threshold,
                                          shape_type)

        if conv:
            fail_code = ICGN_SUCCESS
            break

        # 9. 발산 판단
        if dp_norm > divergence_threshold:
            fail_code = ICGN_FAIL_DIVERGED
            conv = False
            break

        # 10. Warp 업데이트
        update_warp(p, dp, p_new, shape_type)

        # p_new → p 복사
        for k in range(n_params):
            p[k] = p_new[k]

        # 11. 최대 변위 변화 체크
        displacement_change_u = abs(p[u_idx] - p_initial_u)
        displacement_change_v = abs(p[v_idx] - p_initial_v)

        if (displacement_change_u > max_displacement_change
                or displacement_change_v > max_displacement_change):
            fail_code = ICGN_FAIL_MAX_DISPLACEMENT
            conv = False
            break

    return zncc, n_iter, conv, fail_code


# =============================================================================
#  6. 단일 POI 처리 (process_poi 전체)
# =============================================================================

@jit(nopython=True, cache=True)
def process_poi_numba(
    ref_image, grad_x, grad_y,
    coeffs, order,
    cx, cy,
    initial_u, initial_v,
    xsi, eta,
    subset_size,
    max_iterations,
    convergence_threshold,
    shape_type,
    # 작업 버퍼 (사전 할당 — POI별)
    f, dfdx, dfdy,
    J, H, H_inv_buf,
    p, xsi_w, eta_w,
    x_def, y_def,
    g, b, dp, p_new
):
    """
    단일 POI에 대한 전체 IC-GN 처리 — nopython JIT

    기존 icgn.py의 process_poi와 수치적으로 동일.
    Phase 4에서 prange 루프 안에서 호출될 예정.

    Args:
        ref_image: 참조 이미지 (H, W) float64
        grad_x, grad_y: gradient 이미지 (H, W) float64
        coeffs: deformed 이미지의 B-spline 계수 (H, W) float64
        order: 보간 차수 (3 또는 5)
        cx, cy: POI 중심 좌표 (정수)
        initial_u, initial_v: FFT-CC 초기 추정 변위
        xsi, eta: 로컬 좌표 (n_pixels,)
        subset_size: 서브셋 크기
        max_iterations: 최대 반복 횟수
        convergence_threshold: 수렴 임계값
        shape_type: AFFINE(0) 또는 QUADRATIC(1)

        [작업 버퍼 — 모두 사전 할당, POI별 독립]
        f, dfdx, dfdy: (n_pixels,)
        J: (n_pixels, n_params)
        H, H_inv_buf: (n_params, n_params)
        p: (n_params,)
        xsi_w, eta_w, x_def, y_def, g: (n_pixels,)
        b, dp, p_new: (n_params,)

    Returns:
        (zncc, n_iter, converged, fail_code)
        p는 최종 파라미터로 업데이트됨
    """
    n_pixels = len(xsi)
    n_params = get_num_params(shape_type)
    img_h = ref_image.shape[0]
    img_w = ref_image.shape[1]

    # 1. Reference subset 추출
    f_mean, f_tilde, valid = extract_reference_subset(
        ref_image, grad_x, grad_y, cx, cy, subset_size,
        f, dfdx, dfdy
    )

    if not valid:
        return 0.0, 0, False, ICGN_FAIL_FLAT_SUBSET

    # 2. Steepest Descent Image
    compute_steepest_descent(dfdx, dfdy, xsi, eta, J, shape_type)

    # 3. Hessian
    compute_hessian(J, H)

    # 4. Hessian 역행렬
    # np.linalg.inv는 Numba nopython에서 지원됨
    # 그러나 singular 체크를 위해 det를 확인
    # Numba에서 try/except 불가 → determinant 체크
    det = np.linalg.det(H)
    if abs(det) < 1e-30:
        return 0.0, 0, False, ICGN_FAIL_SINGULAR_HESSIAN

    H_inv = np.linalg.inv(H)
    # H_inv_buf에 복사 (외부에서 제공된 버퍼 활용)
    for i in range(n_params):
        for j in range(n_params):
            H_inv_buf[i, j] = H_inv[i, j]

    # 5. 초기 파라미터 설정
    for k in range(n_params):
        p[k] = 0.0

    if shape_type == AFFINE:
        p[0] = initial_u  # u
        p[3] = initial_v  # v
    else:
        p[0] = initial_u  # u
        p[6] = initial_v  # v

    # 6. IC-GN 반복
    zncc, n_iter, conv, fail_code = icgn_iterate(
        f, f_mean, f_tilde,
        J, H_inv_buf,
        coeffs, order,
        img_h, img_w,
        cx, cy,
        xsi, eta,
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

    return zncc, n_iter, conv, fail_code


# =============================================================================
#  7. 버퍼 할당 헬퍼 (Python — prange 전에 1회 호출)
# =============================================================================

def allocate_poi_buffers(n_pixels, n_params):
    """
    단일 POI 처리용 작업 버퍼 할당

    prange에서는 각 스레드가 독립 버퍼를 사용해야 한다.
    이 함수로 POI당 또는 스레드당 버퍼를 생성.

    Args:
        n_pixels: 서브셋 픽셀 수 (subset_size²)
        n_params: 파라미터 수 (6 또는 12)

    Returns:
        dict of buffers
    """
    return {
        'f': np.empty(n_pixels, dtype=np.float64),
        'dfdx': np.empty(n_pixels, dtype=np.float64),
        'dfdy': np.empty(n_pixels, dtype=np.float64),
        'J': np.empty((n_pixels, n_params), dtype=np.float64),
        'H': np.empty((n_params, n_params), dtype=np.float64),
        'H_inv': np.empty((n_params, n_params), dtype=np.float64),
        'p': np.empty(n_params, dtype=np.float64),
        'xsi_w': np.empty(n_pixels, dtype=np.float64),
        'eta_w': np.empty(n_pixels, dtype=np.float64),
        'x_def': np.empty(n_pixels, dtype=np.float64),
        'y_def': np.empty(n_pixels, dtype=np.float64),
        'g': np.empty(n_pixels, dtype=np.float64),
        'b': np.empty(n_params, dtype=np.float64),
        'dp': np.empty(n_params, dtype=np.float64),
        'p_new': np.empty(n_params, dtype=np.float64),
    }


def allocate_batch_buffers(n_poi, n_pixels, n_params):
    """
    다수 POI 처리용 2D 작업 버퍼 할당

    prange에서 각 POI가 독립 행을 사용하도록 2D 배열로 할당.
    이렇게 하면 prange에서 인덱스로 슬라이싱하여 thread-safe 접근 가능.

    Args:
        n_poi: POI 수
        n_pixels: 서브셋 픽셀 수
        n_params: 파라미터 수

    Returns:
        dict of 2D buffers (n_poi, dim)
    """
    return {
        'f': np.empty((n_poi, n_pixels), dtype=np.float64),
        'dfdx': np.empty((n_poi, n_pixels), dtype=np.float64),
        'dfdy': np.empty((n_poi, n_pixels), dtype=np.float64),
        'J': np.empty((n_poi, n_pixels, n_params), dtype=np.float64),
        'H': np.empty((n_poi, n_params, n_params), dtype=np.float64),
        'H_inv': np.empty((n_poi, n_params, n_params), dtype=np.float64),
        'p': np.empty((n_poi, n_params), dtype=np.float64),
        'xsi_w': np.empty((n_poi, n_pixels), dtype=np.float64),
        'eta_w': np.empty((n_poi, n_pixels), dtype=np.float64),
        'x_def': np.empty((n_poi, n_pixels), dtype=np.float64),
        'y_def': np.empty((n_poi, n_pixels), dtype=np.float64),
        'g': np.empty((n_poi, n_pixels), dtype=np.float64),
        'b': np.empty((n_poi, n_params), dtype=np.float64),
        'dp': np.empty((n_poi, n_params), dtype=np.float64),
        'p_new': np.empty((n_poi, n_params), dtype=np.float64),
    }


# =============================================================================
#  8. 배치 POI 처리 (prange 병렬)
# =============================================================================

@jit(nopython=True, parallel=True, cache=True)
def process_all_pois_parallel(
    ref_image, grad_x, grad_y,
    coeffs, order,
    points_x, points_y,
    initial_u, initial_v,
    xsi, eta,
    subset_size,
    max_iterations,
    convergence_threshold,
    shape_type,
    # 결과 배열
    result_p,       # (n_poi, n_params)
    result_zncc,    # (n_poi,)
    result_iter,    # (n_poi,)
    result_conv,    # (n_poi,)  boolean
    result_fail,    # (n_poi,)  int32
    # 2D 작업 버퍼
    buf_f, buf_dfdx, buf_dfdy,
    buf_J, buf_H, buf_H_inv,
    buf_p, buf_xsi_w, buf_eta_w,
    buf_x_def, buf_y_def,
    buf_g, buf_b, buf_dp, buf_p_new
):
    """
    모든 POI를 prange로 병렬 처리

    Phase 4의 최종 형태. 각 POI가 독립 버퍼를 사용하여
    thread-safe하게 처리됨.

    Args:
        ref_image, grad_x, grad_y: 참조 이미지 및 gradient (공유, 읽기전용)
        coeffs: deformed B-spline 계수 (공유, 읽기전용)
        order: 보간 차수
        points_x, points_y: POI 좌표 배열 (n_poi,)
        initial_u, initial_v: 초기 변위 배열 (n_poi,)
        xsi, eta: 로컬 좌표 (공유, 읽기전용)
        subset_size, max_iterations, convergence_threshold: 설정
        shape_type: AFFINE 또는 QUADRATIC

        [결과 배열]
        result_p: (n_poi, n_params) 최종 파라미터
        result_zncc: (n_poi,) ZNCC 값
        result_iter: (n_poi,) 반복 횟수
        result_conv: (n_poi,) 수렴 여부
        result_fail: (n_poi,) 실패 코드

        [2D 작업 버퍼 — allocate_batch_buffers()로 생성]
    """
    n_poi = len(points_x)
    n_params = buf_p.shape[1]

    for idx in prange(n_poi):
        cx = points_x[idx]
        cy = points_y[idx]

        zncc, n_iter, conv, fail_code = process_poi_numba(
            ref_image, grad_x, grad_y,
            coeffs, order,
            cx, cy,
            initial_u[idx], initial_v[idx],
            xsi, eta,
            subset_size,
            max_iterations,
            convergence_threshold,
            shape_type,
            # 작업 버퍼 (idx 행)
            buf_f[idx], buf_dfdx[idx], buf_dfdy[idx],
            buf_J[idx], buf_H[idx], buf_H_inv[idx],
            buf_p[idx], buf_xsi_w[idx], buf_eta_w[idx],
            buf_x_def[idx], buf_y_def[idx],
            buf_g[idx], buf_b[idx], buf_dp[idx], buf_p_new[idx]
        )

        result_zncc[idx] = zncc
        result_iter[idx] = n_iter
        result_conv[idx] = conv
        result_fail[idx] = fail_code

        # 최종 파라미터 복사
        for k in range(n_params):
            result_p[idx, k] = buf_p[idx, k]


# =============================================================================
#  9. JIT 워밍업
# =============================================================================

def warmup_icgn_core():
    """
    Numba JIT 컴파일 워밍업

    실제 ICGN 실행 전에 호출하여 JIT 컴파일 오버헤드를 제거.
    작은 테스트 데이터로 모든 코드 경로를 실행.
    """
    np.random.seed(42)

    # 작은 테스트 이미지
    img_size = 50
    ref = np.random.rand(img_size, img_size).astype(np.float64) * 200 + 20
    deformed = ref.copy()

    # Gradient (단순 차분)
    grad_x = np.zeros_like(ref)
    grad_y = np.zeros_like(ref)
    grad_x[:, 1:-1] = (ref[:, 2:] - ref[:, :-2]) / 2.0
    grad_y[1:-1, :] = (ref[2:, :] - ref[:-2, :]) / 2.0

    subset_size = 11
    n_pixels = subset_size * subset_size

    from .shape_function_numba import generate_local_coordinates
    xsi, eta = generate_local_coordinates(subset_size)

    for shape_type in [AFFINE, QUADRATIC]:
        n_params = get_num_params(shape_type)

        # Prefilter
        for order in [3, 5]:
            coeffs = prefilter_image(deformed, order=order)

            bufs = allocate_poi_buffers(n_pixels, n_params)

            # 단일 POI
            process_poi_numba(
                ref, grad_x, grad_y,
                coeffs, order,
                img_size // 2, img_size // 2,
                0.5, 0.3,
                xsi, eta,
                subset_size,
                10, 0.001,
                shape_type,
                bufs['f'], bufs['dfdx'], bufs['dfdy'],
                bufs['J'], bufs['H'], bufs['H_inv'],
                bufs['p'], bufs['xsi_w'], bufs['eta_w'],
                bufs['x_def'], bufs['y_def'],
                bufs['g'], bufs['b'], bufs['dp'], bufs['p_new']
            )

    # 배치 처리 워밍업 (2 POI)
    n_poi = 2
    n_params = NUM_PARAMS_AFFINE
    coeffs = prefilter_image(deformed, order=5)
    batch_bufs = allocate_batch_buffers(n_poi, n_pixels, n_params)

    result_p = np.empty((n_poi, n_params), dtype=np.float64)
    result_zncc = np.empty(n_poi, dtype=np.float64)
    result_iter = np.empty(n_poi, dtype=np.int32)
    result_conv = np.empty(n_poi, dtype=np.bool_)
    result_fail = np.empty(n_poi, dtype=np.int32)

    pts_x = np.array([img_size // 2, img_size // 2 + 5], dtype=np.int64)
    pts_y = np.array([img_size // 2, img_size // 2 + 5], dtype=np.int64)
    init_u = np.array([0.5, 0.3], dtype=np.float64)
    init_v = np.array([0.3, 0.5], dtype=np.float64)

    process_all_pois_parallel(
        ref, grad_x, grad_y,
        coeffs, 5,
        pts_x, pts_y,
        init_u, init_v,
        xsi, eta,
        subset_size, 10, 0.001,
        AFFINE,
        result_p, result_zncc, result_iter, result_conv, result_fail,
        batch_bufs['f'], batch_bufs['dfdx'], batch_bufs['dfdy'],
        batch_bufs['J'], batch_bufs['H'], batch_bufs['H_inv'],
        batch_bufs['p'], batch_bufs['xsi_w'], batch_bufs['eta_w'],
        batch_bufs['x_def'], batch_bufs['y_def'],
        batch_bufs['g'], batch_bufs['b'], batch_bufs['dp'], batch_bufs['p_new']
    )
