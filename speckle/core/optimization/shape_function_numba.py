"""
Numba Shape Function 모듈

Affine (1차) 및 Quadratic (2차) 변형 함수의 Numba JIT 구현.
기존 shape_function.py와 수치적으로 동일한 결과를 생성하면서,
ICGN prange 병렬화에서 nopython 모드로 직접 호출 가능.

Numba 변환 설계 원칙:
    - str 분기 제거: nopython에서 str 비교 불가 → int 상수로 대체
    - 통합 인터페이스 유지: shape_type 인수 (AFFINE=0, QUADRATIC=1)
    - try/except 제거: nopython에서 예외 처리 불가 → det 체크로 대체
    - 내부 함수(compute_A_terms) 인라인화

References:
    - Pan, B., et al. Experimental Mechanics, 2013.
    - Jiang, Z., et al. Optics and Lasers in Engineering, 2014.
"""

import numpy as np
from numba import jit, float64, int64, boolean, types
from numba import prange


# =============================================================================
#  상수 정의
# =============================================================================

AFFINE = 0
QUADRATIC = 1
NUM_PARAMS_AFFINE = 6
NUM_PARAMS_QUADRATIC = 12


# =============================================================================
#  1. 로컬 좌표 생성 (Python — 1회 호출)
# =============================================================================
# Numba 변환 불필요: 초기화 시 1회만 호출

def generate_local_coordinates(subset_size):
    """
    로컬 좌표 (ξ, η) 생성

    Args:
        subset_size: 서브셋 크기 (홀수)

    Returns:
        (xsi, eta) — 1D 배열, 길이 = subset_size²
    """
    half = subset_size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    eta_2d, xsi_2d = np.meshgrid(coords, coords, indexing='ij')
    return xsi_2d.ravel().copy(), eta_2d.ravel().copy()


# =============================================================================
#  2. Warp 함수
# =============================================================================

@jit(nopython=True, cache=True)
def warp_affine(p, xsi, eta, xsi_w, eta_w):
    """
    Affine warp — 결과를 사전 할당 배열에 기록 (in-place)

    p = [u, ux, uy, v, vx, vy]

    xsi_w[i] = (1 + ux) * xsi[i] + uy * eta[i] + u
    eta_w[i] = vx * xsi[i] + (1 + vy) * eta[i] + v
    """
    u = p[0]; ux = p[1]; uy = p[2]
    v = p[3]; vx = p[4]; vy = p[5]

    n = len(xsi)
    for i in range(n):
        xsi_w[i] = (1.0 + ux) * xsi[i] + uy * eta[i] + u
        eta_w[i] = vx * xsi[i] + (1.0 + vy) * eta[i] + v


@jit(nopython=True, cache=True)
def warp_quadratic(p, xsi, eta, xsi_w, eta_w):
    """
    Quadratic warp — 결과를 사전 할당 배열에 기록 (in-place)

    p = [u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy]
    """
    u = p[0]; ux = p[1]; uy = p[2]
    uxx = p[3]; uxy = p[4]; uyy = p[5]
    v = p[6]; vx = p[7]; vy = p[8]
    vxx = p[9]; vxy = p[10]; vyy = p[11]

    n = len(xsi)
    for i in range(n):
        xi = xsi[i]
        et = eta[i]
        xi2 = xi * xi
        et2 = et * et
        xi_et = xi * et

        xsi_w[i] = (u + (1.0 + ux) * xi + uy * et
                     + 0.5 * uxx * xi2 + uxy * xi_et + 0.5 * uyy * et2)
        eta_w[i] = (v + vx * xi + (1.0 + vy) * et
                     + 0.5 * vxx * xi2 + vxy * xi_et + 0.5 * vyy * et2)


@jit(nopython=True, cache=True)
def warp(p, xsi, eta, xsi_w, eta_w, shape_type):
    """
    통합 warp 함수

    Args:
        p: 파라미터 배열
        xsi, eta: 로컬 좌표
        xsi_w, eta_w: 출력 배열 (사전 할당)
        shape_type: AFFINE(0) 또는 QUADRATIC(1)
    """
    if shape_type == AFFINE:
        warp_affine(p, xsi, eta, xsi_w, eta_w)
    else:
        warp_quadratic(p, xsi, eta, xsi_w, eta_w)


# =============================================================================
#  3. Steepest Descent Image
# =============================================================================

@jit(nopython=True, cache=True)
def compute_steepest_descent_affine(dfdx, dfdy, xsi, eta, J):
    """
    Steepest Descent Image (Affine) — in-place

    J[i, :] = [dfdx, dfdx*ξ, dfdx*η, dfdy, dfdy*ξ, dfdy*η]

    Args:
        dfdx, dfdy: gradient 배열 (n_pixels,)
        xsi, eta: 로컬 좌표 (n_pixels,)
        J: 출력 배열 (n_pixels, 6) — 사전 할당
    """
    n = len(dfdx)
    for i in range(n):
        dx = dfdx[i]
        dy = dfdy[i]
        xi = xsi[i]
        et = eta[i]
        J[i, 0] = dx
        J[i, 1] = dx * xi
        J[i, 2] = dx * et
        J[i, 3] = dy
        J[i, 4] = dy * xi
        J[i, 5] = dy * et


@jit(nopython=True, cache=True)
def compute_steepest_descent_quadratic(dfdx, dfdy, xsi, eta, J):
    """
    Steepest Descent Image (Quadratic) — in-place

    Args:
        J: 출력 배열 (n_pixels, 12) — 사전 할당
    """
    n = len(dfdx)
    for i in range(n):
        dx = dfdx[i]
        dy = dfdy[i]
        xi = xsi[i]
        et = eta[i]
        xi2 = xi * xi
        et2 = et * et
        xi_et = xi * et

        # u 관련 (0-5)
        J[i, 0] = dx
        J[i, 1] = dx * xi
        J[i, 2] = dx * et
        J[i, 3] = 0.5 * dx * xi2
        J[i, 4] = dx * xi_et
        J[i, 5] = 0.5 * dx * et2

        # v 관련 (6-11)
        J[i, 6] = dy
        J[i, 7] = dy * xi
        J[i, 8] = dy * et
        J[i, 9] = 0.5 * dy * xi2
        J[i, 10] = dy * xi_et
        J[i, 11] = 0.5 * dy * et2


@jit(nopython=True, cache=True)
def compute_steepest_descent(dfdx, dfdy, xsi, eta, J, shape_type):
    """통합 Steepest Descent"""
    if shape_type == AFFINE:
        compute_steepest_descent_affine(dfdx, dfdy, xsi, eta, J)
    else:
        compute_steepest_descent_quadratic(dfdx, dfdy, xsi, eta, J)


# =============================================================================
#  4. Hessian 계산
# =============================================================================

@jit(nopython=True, cache=True)
def compute_hessian(J, H):
    """
    Hessian matrix: H = J^T @ J — in-place

    Args:
        J: Steepest Descent Image (n_pixels, n_params)
        H: 출력 배열 (n_params, n_params) — 사전 할당
    """
    n_pixels = J.shape[0]
    n_params = J.shape[1]

    for i in range(n_params):
        for j in range(i, n_params):
            s = 0.0
            for k in range(n_pixels):
                s += J[k, i] * J[k, j]
            H[i, j] = s
            H[j, i] = s  # 대칭


# =============================================================================
#  5. 수렴 판정
# =============================================================================

@jit(nopython=True, cache=True)
def check_convergence_affine(dp, half, threshold):
    """
    Affine 수렴 판정 — Jiang et al. (2015) / OpenCorr 방식

    dp = [u, ux, uy, v, vx, vy]

    Returns:
        (converged, dp_norm)
    """
    R2 = float64(half * half)
    norm_sq = (dp[0] * dp[0]
               + dp[1] * dp[1] * R2
               + dp[2] * dp[2] * R2
               + dp[3] * dp[3]
               + dp[4] * dp[4] * R2
               + dp[5] * dp[5] * R2)
    norm_p = np.sqrt(norm_sq)
    return norm_p < threshold, norm_p


@jit(nopython=True, cache=True)
def check_convergence_quadratic(dp, half, threshold):
    """
    Quadratic 수렴 판정

    dp = [u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy]

    Returns:
        (converged, dp_norm)
    """
    R2 = float64(half * half)
    R4 = R2 * R2 * 0.25
    R2R2 = R2 * R2

    norm_sq = (dp[0] * dp[0]
               + dp[1] * dp[1] * R2
               + dp[2] * dp[2] * R2
               + dp[3] * dp[3] * R4
               + dp[4] * dp[4] * R2R2
               + dp[5] * dp[5] * R4
               + dp[6] * dp[6]
               + dp[7] * dp[7] * R2
               + dp[8] * dp[8] * R2
               + dp[9] * dp[9] * R4
               + dp[10] * dp[10] * R2R2
               + dp[11] * dp[11] * R4)
    norm_p = np.sqrt(norm_sq)
    return norm_p < threshold, norm_p


@jit(nopython=True, cache=True)
def check_convergence(dp, half, threshold, shape_type):
    """
    통합 수렴 판정

    Args:
        dp: 파라미터 업데이트 벡터
        half: subset_size // 2
        threshold: 수렴 임계값
        shape_type: AFFINE(0) 또는 QUADRATIC(1)

    Returns:
        (converged, dp_norm)
    """
    if shape_type == AFFINE:
        return check_convergence_affine(dp, half, threshold)
    else:
        return check_convergence_quadratic(dp, half, threshold)


# =============================================================================
#  6. Inverse Compositional Warp Update
# =============================================================================

@jit(nopython=True, cache=True)
def update_warp_affine(p, dp, p_new):
    """
    Affine Warp Update — W(p) ← W(p) · W(Δp)^(-1)

    Jiang et al. (2014) Eq. (12) 직접 계산.
    역행렬 회피로 6x6 inv 불필요.

    Args:
        p: 현재 파라미터 (6,)
        dp: delta 파라미터 (6,)
        p_new: 출력 배열 (6,) — 사전 할당

    Returns:
        0 if success, -1 if near-singular
    """
    u = p[0]; ux = p[1]; uy = p[2]
    v = p[3]; vx = p[4]; vy = p[5]

    du = dp[0]; dux = dp[1]; duy = dp[2]
    dv = dp[3]; dvx = dp[4]; dvy = dp[5]

    # Determinant: det = (1+dux)(1+dvy) - duy*dvx
    det = (1.0 + dux) * (1.0 + dvy) - duy * dvx

    if abs(det) < 1e-12:
        # near-singular: fallback to matrix method
        W_p = np.empty((3, 3), dtype=np.float64)
        W_p[0, 0] = 1.0 + ux;  W_p[0, 1] = uy;       W_p[0, 2] = u
        W_p[1, 0] = vx;        W_p[1, 1] = 1.0 + vy;  W_p[1, 2] = v
        W_p[2, 0] = 0.0;       W_p[2, 1] = 0.0;       W_p[2, 2] = 1.0

        W_dp = np.empty((3, 3), dtype=np.float64)
        W_dp[0, 0] = 1.0 + dux;  W_dp[0, 1] = duy;       W_dp[0, 2] = du
        W_dp[1, 0] = dvx;        W_dp[1, 1] = 1.0 + dvy;  W_dp[1, 2] = dv
        W_dp[2, 0] = 0.0;        W_dp[2, 1] = 0.0;        W_dp[2, 2] = 1.0

        W_dp_inv = np.linalg.inv(W_dp)
        W_new = W_p @ W_dp_inv

        p_new[0] = W_new[0, 2]
        p_new[1] = W_new[0, 0] - 1.0
        p_new[2] = W_new[0, 1]
        p_new[3] = W_new[1, 2]
        p_new[4] = W_new[1, 0]
        p_new[5] = W_new[1, 1] - 1.0
        return 0

    inv_det = 1.0 / det

    # W(dp)^(-1) 요소
    a = (1.0 + dvy) * inv_det
    b = -duy * inv_det
    c = (duy * dv - du * (1.0 + dvy)) * inv_det
    d = -dvx * inv_det
    e = (1.0 + dux) * inv_det
    f = (dvx * du - dv * (1.0 + dux)) * inv_det

    # W_new = W_p @ W_dp_inv
    new_00 = (1.0 + ux) * a + uy * d
    new_01 = (1.0 + ux) * b + uy * e
    new_02 = (1.0 + ux) * c + uy * f + u

    new_10 = vx * a + (1.0 + vy) * d
    new_11 = vx * b + (1.0 + vy) * e
    new_12 = vx * c + (1.0 + vy) * f + v

    p_new[0] = new_02
    p_new[1] = new_00 - 1.0
    p_new[2] = new_01
    p_new[3] = new_12
    p_new[4] = new_10
    p_new[5] = new_11 - 1.0
    return 0


@jit(nopython=True, cache=True)
def _compute_A_terms(u, ux, uy, uxx, uxy, uyy,
                     v, vx, vy, vxx, vxy, vyy, A):
    """
    Quadratic warp update용 W 행렬 중간 계산

    A[0..17]에 18개 항 기록
    """
    A[0] = 2.0*ux + ux*ux + u*uxx
    A[1] = 2.0*u*uxy + 2.0*(1.0+ux)*uy
    A[2] = uy*uy + u*uyy
    A[3] = 2.0*u*(1.0+ux)
    A[4] = 2.0*u*uy
    A[5] = u*u

    A[6] = 0.5*(v*uxx + 2.0*(1.0+ux)*vx + u*vxx)
    A[7] = uy*vx + ux*vy + v*uxy + u*vxy + vy + ux
    A[8] = 0.5*(v*uyy + 2.0*(1.0+vy)*uy + u*vyy)
    A[9] = v + v*ux + u*vx
    A[10] = u + v*uy + u*vy
    A[11] = u*v

    A[12] = vx*vx + v*vxx
    A[13] = 2.0*v*vxy + 2.0*vx*(1.0+vy)
    A[14] = 2.0*vy + vy*vy + v*vyy
    A[15] = 2.0*v*vx
    A[16] = 2.0*v*(1.0+vy)
    A[17] = v*v


@jit(nopython=True, cache=True)
def update_warp_quadratic(p, dp, p_new):
    """
    Quadratic Warp Update — W(p) ← W(p) · W(Δp)^(-1)

    6×6 행렬 역행렬 사용.

    Args:
        p: 현재 파라미터 (12,)
        dp: delta 파라미터 (12,)
        p_new: 출력 배열 (12,) — 사전 할당

    Returns:
        0 if success, -1 if singular
    """
    u = p[0]; ux = p[1]; uy = p[2]
    uxx = p[3]; uxy = p[4]; uyy = p[5]
    v = p[6]; vx = p[7]; vy = p[8]
    vxx = p[9]; vxy = p[10]; vyy = p[11]

    du = dp[0]; dux = dp[1]; duy = dp[2]
    duxx = dp[3]; duxy = dp[4]; duyy = dp[5]
    dv = dp[6]; dvx = dp[7]; dvy = dp[8]
    dvxx = dp[9]; dvxy = dp[10]; dvyy = dp[11]

    # A terms for W(p)
    A = np.empty(18, dtype=np.float64)
    _compute_A_terms(u, ux, uy, uxx, uxy, uyy,
                     v, vx, vy, vxx, vxy, vyy, A)

    # W(p) 구성
    W_p = np.empty((6, 6), dtype=np.float64)
    W_p[0, 0] = 1.0+A[0]; W_p[0, 1] = A[1];     W_p[0, 2] = A[2]
    W_p[0, 3] = A[3];     W_p[0, 4] = A[4];      W_p[0, 5] = A[5]
    W_p[1, 0] = A[6];     W_p[1, 1] = 1.0+A[7];  W_p[1, 2] = A[8]
    W_p[1, 3] = A[9];     W_p[1, 4] = A[10];     W_p[1, 5] = A[11]
    W_p[2, 0] = A[12];    W_p[2, 1] = A[13];     W_p[2, 2] = 1.0+A[14]
    W_p[2, 3] = A[15];    W_p[2, 4] = A[16];     W_p[2, 5] = A[17]
    W_p[3, 0] = 0.5*uxx;  W_p[3, 1] = uxy;       W_p[3, 2] = 0.5*uyy
    W_p[3, 3] = 1.0+ux;   W_p[3, 4] = uy;        W_p[3, 5] = u
    W_p[4, 0] = 0.5*vxx;  W_p[4, 1] = vxy;       W_p[4, 2] = 0.5*vyy
    W_p[4, 3] = vx;       W_p[4, 4] = 1.0+vy;    W_p[4, 5] = v
    W_p[5, 0] = 0.0;      W_p[5, 1] = 0.0;       W_p[5, 2] = 0.0
    W_p[5, 3] = 0.0;      W_p[5, 4] = 0.0;       W_p[5, 5] = 1.0

    # dA terms for W(dp)
    dA = np.empty(18, dtype=np.float64)
    _compute_A_terms(du, dux, duy, duxx, duxy, duyy,
                     dv, dvx, dvy, dvxx, dvxy, dvyy, dA)

    # W(dp) 구성
    W_dp = np.empty((6, 6), dtype=np.float64)
    W_dp[0, 0] = 1.0+dA[0]; W_dp[0, 1] = dA[1];     W_dp[0, 2] = dA[2]
    W_dp[0, 3] = dA[3];     W_dp[0, 4] = dA[4];      W_dp[0, 5] = dA[5]
    W_dp[1, 0] = dA[6];     W_dp[1, 1] = 1.0+dA[7];  W_dp[1, 2] = dA[8]
    W_dp[1, 3] = dA[9];     W_dp[1, 4] = dA[10];     W_dp[1, 5] = dA[11]
    W_dp[2, 0] = dA[12];    W_dp[2, 1] = dA[13];     W_dp[2, 2] = 1.0+dA[14]
    W_dp[2, 3] = dA[15];    W_dp[2, 4] = dA[16];     W_dp[2, 5] = dA[17]
    W_dp[3, 0] = 0.5*duxx;  W_dp[3, 1] = duxy;       W_dp[3, 2] = 0.5*duyy
    W_dp[3, 3] = 1.0+dux;   W_dp[3, 4] = duy;        W_dp[3, 5] = du
    W_dp[4, 0] = 0.5*dvxx;  W_dp[4, 1] = dvxy;       W_dp[4, 2] = 0.5*dvyy
    W_dp[4, 3] = dvx;       W_dp[4, 4] = 1.0+dvy;    W_dp[4, 5] = dv
    W_dp[5, 0] = 0.0;       W_dp[5, 1] = 0.0;        W_dp[5, 2] = 0.0
    W_dp[5, 3] = 0.0;       W_dp[5, 4] = 0.0;        W_dp[5, 5] = 1.0

    # W_new = W_p @ W_dp^{-1}
    W_dp_inv = np.linalg.inv(W_dp)
    W_new = W_p @ W_dp_inv

    # 파라미터 추출 (행 3, 4에서)
    p_new[0] = W_new[3, 5]           # u
    p_new[1] = W_new[3, 3] - 1.0     # ux
    p_new[2] = W_new[3, 4]           # uy
    p_new[3] = 2.0 * W_new[3, 0]     # uxx
    p_new[4] = W_new[3, 1]           # uxy
    p_new[5] = 2.0 * W_new[3, 2]     # uyy
    p_new[6] = W_new[4, 5]           # v
    p_new[7] = W_new[4, 3]           # vx
    p_new[8] = W_new[4, 4] - 1.0     # vy
    p_new[9] = 2.0 * W_new[4, 0]     # vxx
    p_new[10] = W_new[4, 1]          # vxy
    p_new[11] = 2.0 * W_new[4, 2]    # vyy
    return 0


@jit(nopython=True, cache=True)
def update_warp(p, dp, p_new, shape_type):
    """
    통합 Inverse Compositional Warp Update

    Args:
        p: 현재 파라미터
        dp: delta 파라미터
        p_new: 출력 배열 (사전 할당)
        shape_type: AFFINE(0) 또는 QUADRATIC(1)

    Returns:
        0 if success
    """
    if shape_type == AFFINE:
        return update_warp_affine(p, dp, p_new)
    else:
        return update_warp_quadratic(p, dp, p_new)


# =============================================================================
#  7. 유틸리티
# =============================================================================

@jit(nopython=True, cache=True)
def get_num_params(shape_type):
    """파라미터 개수 반환"""
    if shape_type == AFFINE:
        return NUM_PARAMS_AFFINE
    else:
        return NUM_PARAMS_QUADRATIC


# =============================================================================
#  8. JIT 워밍업
# =============================================================================

def warmup_numba_shape():
    """Numba JIT 컴파일 워밍업"""
    n = 441  # 21x21 subset
    xsi = np.random.rand(n).astype(np.float64)
    eta = np.random.rand(n).astype(np.float64)
    dfdx = np.random.rand(n).astype(np.float64)
    dfdy = np.random.rand(n).astype(np.float64)

    # Affine
    p6 = np.zeros(6, dtype=np.float64)
    dp6 = np.random.rand(6).astype(np.float64) * 0.01
    xsi_w = np.empty(n, dtype=np.float64)
    eta_w = np.empty(n, dtype=np.float64)
    J6 = np.empty((n, 6), dtype=np.float64)
    H6 = np.empty((6, 6), dtype=np.float64)
    p_new6 = np.empty(6, dtype=np.float64)

    warp(p6, xsi, eta, xsi_w, eta_w, AFFINE)
    compute_steepest_descent(dfdx, dfdy, xsi, eta, J6, AFFINE)
    compute_hessian(J6, H6)
    check_convergence(dp6, 10, 0.001, AFFINE)
    update_warp(p6, dp6, p_new6, AFFINE)

    # Quadratic
    p12 = np.zeros(12, dtype=np.float64)
    dp12 = np.random.rand(12).astype(np.float64) * 0.01
    J12 = np.empty((n, 12), dtype=np.float64)
    H12 = np.empty((12, 12), dtype=np.float64)
    p_new12 = np.empty(12, dtype=np.float64)

    warp(p12, xsi, eta, xsi_w, eta_w, QUADRATIC)
    compute_steepest_descent(dfdx, dfdy, xsi, eta, J12, QUADRATIC)
    compute_hessian(J12, H12)
    check_convergence(dp12, 10, 0.001, QUADRATIC)
    update_warp(p12, dp12, p_new12, QUADRATIC)

    get_num_params(AFFINE)
    get_num_params(QUADRATIC)
