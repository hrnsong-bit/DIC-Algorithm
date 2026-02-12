"""
Shape Function 모듈

Affine (1차) 및 Quadratic (2차) 변형 함수 지원

References:
    - Pan, B., et al. "Fast, robust and accurate DIC calculation without 
      redundant computations." Experimental Mechanics, 2013.
    - Jiang, Z., et al. "Path-independent digital image correlation with 
      high accuracy, speed and robustness." Optics and Lasers in Engineering, 2014.
"""

import numpy as np
from typing import Tuple


# ===== 로컬 좌표 생성 =====

def generate_local_coordinates(subset_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    로컬 좌표 (ξ, η) 생성
    """
    half = subset_size // 2
    coords = np.arange(-half, half + 1, dtype=np.float64)
    
    eta_2d, xsi_2d = np.meshgrid(coords, coords, indexing='ij')
    
    xsi = xsi_2d.ravel()
    eta = eta_2d.ravel()
    
    return xsi, eta


# ===== Affine (1차) Shape Function =====

def warp_affine(
    p: np.ndarray,
    xsi: np.ndarray,
    eta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Affine warp 적용
    
    p = [u, ux, uy, v, vx, vy]
    """
    u, ux, uy, v, vx, vy = p[:6]
    
    xsi_w = (1 + ux) * xsi + uy * eta + u
    eta_w = vx * xsi + (1 + vy) * eta + v
    
    return xsi_w, eta_w


def compute_steepest_descent_affine(
    dfdx: np.ndarray,
    dfdy: np.ndarray,
    xsi: np.ndarray,
    eta: np.ndarray
) -> np.ndarray:
    """
    Steepest Descent Image (Affine)
    
    J = [∂f/∂x, ∂f/∂x·ξ, ∂f/∂x·η, ∂f/∂y, ∂f/∂y·ξ, ∂f/∂y·η]
    """
    n_pixels = len(dfdx)
    
    J = np.zeros((n_pixels, 6), dtype=np.float64)
    
    J[:, 0] = dfdx
    J[:, 1] = dfdx * xsi
    J[:, 2] = dfdx * eta
    J[:, 3] = dfdy
    J[:, 4] = dfdy * xsi
    J[:, 5] = dfdy * eta
    
    return J


# ===== Quadratic (2차) Shape Function =====

def warp_quadratic(
    p: np.ndarray,
    xsi: np.ndarray,
    eta: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quadratic warp 적용
    
    p = [u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy]
         0   1   2   3    4    5   6   7   8   9    10   11
    """
    u, ux, uy, uxx, uxy, uyy = p[0:6]
    v, vx, vy, vxx, vxy, vyy = p[6:12]
    
    xsi2 = xsi * xsi
    eta2 = eta * eta
    xsi_eta = xsi * eta
    
    xsi_w = (u + (1 + ux) * xsi + uy * eta + 
             0.5 * uxx * xsi2 + uxy * xsi_eta + 0.5 * uyy * eta2)
    
    eta_w = (v + vx * xsi + (1 + vy) * eta + 
             0.5 * vxx * xsi2 + vxy * xsi_eta + 0.5 * vyy * eta2)
    
    return xsi_w, eta_w


def compute_steepest_descent_quadratic(
    dfdx: np.ndarray,
    dfdy: np.ndarray,
    xsi: np.ndarray,
    eta: np.ndarray
) -> np.ndarray:
    """
    Steepest Descent Image (Quadratic)
    """
    n_pixels = len(dfdx)
    
    xsi2 = xsi * xsi
    eta2 = eta * eta
    xsi_eta = xsi * eta
    
    J = np.zeros((n_pixels, 12), dtype=np.float64)
    
    # u 관련 (0-5)
    J[:, 0] = dfdx
    J[:, 1] = dfdx * xsi
    J[:, 2] = dfdx * eta
    J[:, 3] = 0.5 * dfdx * xsi2
    J[:, 4] = dfdx * xsi_eta
    J[:, 5] = 0.5 * dfdx * eta2
    
    # v 관련 (6-11)
    J[:, 6] = dfdy
    J[:, 7] = dfdy * xsi
    J[:, 8] = dfdy * eta
    J[:, 9] = 0.5 * dfdy * xsi2
    J[:, 10] = dfdy * xsi_eta
    J[:, 11] = 0.5 * dfdy * eta2
    
    return J


# ===== Inverse Compositional Warp Update =====
# Based on Jiang et al. (2014) Eq. (12) - 역행렬 계산 회피

def _update_affine_direct(p: np.ndarray, dp: np.ndarray) -> np.ndarray:
    """
    Affine Warp Update - 직접 계산 (역행렬 회피)
    
    p = [u, ux, uy, v, vx, vy]
    
    W(p) ← W(p) · W(Δp)^(-1)
    
    Based on Jiang et al. (2014) Eq. (12)
    """
    # 현재 파라미터
    u, ux, uy = p[0], p[1], p[2]
    v, vx, vy = p[3], p[4], p[5]
    
    # delta 파라미터
    du, dux, duy = dp[0], dp[1], dp[2]
    dv, dvx, dvy = dp[3], dp[4], dp[5]
    
    # Determinant of W(dp): det = (1+dux)(1+dvy) - duy*dvx
    det = (1.0 + dux) * (1.0 + dvy) - duy * dvx
    
    if abs(det) < 1e-12:
        # Fallback to matrix solve if near-singular
        return _update_affine_matrix_fallback(p, dp)
    
    inv_det = 1.0 / det
    
    # W(dp)^(-1) 요소 계산 (3x3 역행렬의 요소들)
    # W_dp = [[1+dux, duy, du], [dvx, 1+dvy, dv], [0, 0, 1]]
    # W_dp_inv = [[a, b, c], [d, e, f], [0, 0, 1]]
    a = (1.0 + dvy) * inv_det           # inv[0,0]
    b = -duy * inv_det                   # inv[0,1]
    c = (duy * dv - du * (1.0 + dvy)) * inv_det  # inv[0,2]
    d = -dvx * inv_det                   # inv[1,0]
    e = (1.0 + dux) * inv_det           # inv[1,1]
    f = (dvx * du - dv * (1.0 + dux)) * inv_det  # inv[1,2]
    
    # W_new = W_p @ W_dp_inv
    # W_p = [[1+ux, uy, u], [vx, 1+vy, v], [0, 0, 1]]
    
    # 새로운 파라미터 계산
    new_00 = (1.0 + ux) * a + uy * d      # 1 + ux_new
    new_01 = (1.0 + ux) * b + uy * e      # uy_new
    new_02 = (1.0 + ux) * c + uy * f + u  # u_new
    
    new_10 = vx * a + (1.0 + vy) * d      # vx_new
    new_11 = vx * b + (1.0 + vy) * e      # 1 + vy_new
    new_12 = vx * c + (1.0 + vy) * f + v  # v_new
    
    return np.array([
        new_02,         # u
        new_00 - 1.0,   # ux
        new_01,         # uy
        new_12,         # v
        new_10,         # vx
        new_11 - 1.0    # vy
    ], dtype=np.float64)


def _update_affine_matrix_fallback(p: np.ndarray, dp: np.ndarray) -> np.ndarray:
    """Fallback: 행렬 방식 (near-singular 케이스용)"""
    W_p = np.array([
        [1.0 + p[1], p[2], p[0]],
        [p[4], 1.0 + p[5], p[3]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    W_dp = np.array([
        [1.0 + dp[1], dp[2], dp[0]],
        [dp[4], 1.0 + dp[5], dp[3]],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    try:
        W_dp_inv = np.linalg.inv(W_dp)
        W_new = W_p @ W_dp_inv          # W(p) · W(Δp)^(-1)
    except np.linalg.LinAlgError:
        return p.copy()
    
    return np.array([
        W_new[0, 2],        # u
        W_new[0, 0] - 1.0,  # ux
        W_new[0, 1],        # uy
        W_new[1, 2],        # v
        W_new[1, 0],        # vx
        W_new[1, 1] - 1.0   # vy
    ], dtype=np.float64)


def _update_quadratic_direct(p: np.ndarray, dp: np.ndarray) -> np.ndarray:
    """
    Quadratic Warp Update - 직접 계산
    
    p = [u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy]
    
    Quadratic은 6x6 행렬이라 직접 계산이 복잡함.
    성능상 이점이 크지 않아 최적화된 행렬 방식 사용.
    """
    # Quadratic은 행렬이 6x6으로 커서 직접 계산의 이점이 적음
    # 대신 최적화된 행렬 계산 사용
    return _update_quadratic_matrix_optimized(p, dp)


def _update_quadratic_matrix_optimized(p: np.ndarray, dp: np.ndarray) -> np.ndarray:
    """
    Quadratic Warp Update - 최적화된 행렬 방식
    
    SunDIC 방식 기반, 중간 계산 최소화
    """
    # 파라미터 추출
    u, ux, uy, uxx, uxy, uyy = p[0:6]
    v, vx, vy, vxx, vxy, vyy = p[6:12]
    
    du, dux, duy, duxx, duxy, duyy = dp[0:6]
    dv, dvx, dvy, dvxx, dvxy, dvyy = dp[6:12]
    
    # W(p) 행렬 구성을 위한 중간 계산
    def compute_A_terms(u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy):
        A1 = 2*ux + ux**2 + u*uxx
        A2 = 2*u*uxy + 2*(1+ux)*uy
        A3 = uy**2 + u*uyy
        A4 = 2*u*(1+ux)
        A5 = 2*u*uy
        A6 = u**2
        
        A7 = 0.5*(v*uxx + 2*(1+ux)*vx + u*vxx)
        A8 = uy*vx + ux*vy + v*uxy + u*vxy + vy + ux
        A9 = 0.5*(v*uyy + 2*(1+vy)*uy + u*vyy)
        A10 = v + v*ux + u*vx
        A11 = u + v*uy + u*vy
        A12 = u*v
        
        A13 = vx**2 + v*vxx
        A14 = 2*v*vxy + 2*vx*(1+vy)
        A15 = 2*vy + vy**2 + v*vyy
        A16 = 2*v*vx
        A17 = 2*v*(1+vy)
        A18 = v**2
        
        return (A1, A2, A3, A4, A5, A6, A7, A8, A9, 
                A10, A11, A12, A13, A14, A15, A16, A17, A18)
    
    # W(p) 구성
    A = compute_A_terms(u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy)
    W_p = np.array([
        [1+A[0],    A[1],     A[2],    A[3],    A[4],   A[5]],
        [A[6],    1+A[7],     A[8],    A[9],   A[10],  A[11]],
        [A[12],    A[13],  1+A[14],   A[15],   A[16],  A[17]],
        [0.5*uxx,   uxy,  0.5*uyy,   1+ux,      uy,      u],
        [0.5*vxx,   vxy,  0.5*vyy,     vx,    1+vy,      v],
        [0,           0,        0,      0,       0,      1]
    ], dtype=np.float64)
    
    # W(dp) 구성
    dA = compute_A_terms(du, dux, duy, duxx, duxy, duyy, dv, dvx, dvy, dvxx, dvxy, dvyy)
    W_dp = np.array([
        [1+dA[0],    dA[1],     dA[2],    dA[3],    dA[4],   dA[5]],
        [dA[6],    1+dA[7],     dA[8],    dA[9],   dA[10],  dA[11]],
        [dA[12],    dA[13],  1+dA[14],   dA[15],   dA[16],  dA[17]],
        [0.5*duxx,   duxy,  0.5*duyy,   1+dux,      duy,      du],
        [0.5*dvxx,   dvxy,  0.5*dvyy,     dvx,    1+dvy,      dv],
        [0,            0,         0,       0,        0,       1]
    ], dtype=np.float64)

    # W_new = W_p @ W_dp^{-1}
    try:
        W_dp_inv = np.linalg.inv(W_dp)
        W_new = W_p @ W_dp_inv
    except np.linalg.LinAlgError:
        return p.copy()
    
    # 파라미터 추출 (행 3, 4에서)
    return np.array([
        W_new[3, 5],        # u
        W_new[3, 3] - 1.0,  # ux
        W_new[3, 4],        # uy
        2.0 * W_new[3, 0],  # uxx
        W_new[3, 1],        # uxy
        2.0 * W_new[3, 2],  # uyy
        W_new[4, 5],        # v
        W_new[4, 3],        # vx
        W_new[4, 4] - 1.0,  # vy
        2.0 * W_new[4, 0],  # vxx
        W_new[4, 1],        # vxy
        2.0 * W_new[4, 2]   # vyy
    ], dtype=np.float64)


# ===== 수렴 조건 =====
def check_convergence(
    dp: np.ndarray,
    subset_size: int,
    convergence_threshold: float,
    shape_function: str = 'affine'
) -> Tuple[bool, float]:
    """
    수렴 판정 — Jiang et al. (2015) / OpenCorr 방식
    
    서브셋 가장자리에서의 최대 변위 변화량을 기준으로 판정.
    dp_ux * R는 서브셋 끝에서 ux gradient로 인한 변위 기여분.
    """
    half = subset_size // 2
    R2 = half * half

    if shape_function == 'affine':
        # p = [u, ux, uy, v, vx, vy]
        norm_sq = (dp[0]**2 
                 + dp[1]**2 * R2     # ux² × R²
                 + dp[2]**2 * R2     # uy² × R²
                 + dp[3]**2 
                 + dp[4]**2 * R2     # vx² × R²
                 + dp[5]**2 * R2)    # vy² × R²
    else:
        # p = [u, ux, uy, uxx, uxy, uyy, v, vx, vy, vxx, vxy, vyy]
        R4 = R2 * R2 * 0.25
        R2R2 = R2 * R2
        norm_sq = (dp[0]**2 
                 + dp[1]**2 * R2
                 + dp[2]**2 * R2
                 + dp[3]**2 * R4       # uxx² × R⁴/4
                 + dp[4]**2 * R2R2     # uxy² × R²R²
                 + dp[5]**2 * R4       # uyy² × R⁴/4
                 + dp[6]**2 
                 + dp[7]**2 * R2
                 + dp[8]**2 * R2
                 + dp[9]**2 * R4
                 + dp[10]**2 * R2R2
                 + dp[11]**2 * R4)

    norm_p = np.sqrt(norm_sq)
    return norm_p < convergence_threshold, norm_p

# ===== 공통 함수 =====

def compute_hessian(J: np.ndarray) -> np.ndarray:
    """
    Hessian matrix 계산
    
    H = J^T @ J
    """
    return J.T @ J


# ===== 통합 인터페이스 =====

def warp(p: np.ndarray, xsi: np.ndarray, eta: np.ndarray, 
         shape_function: str = 'affine') -> Tuple[np.ndarray, np.ndarray]:
    """
    통합 warp 함수
    """
    if shape_function == 'affine':
        return warp_affine(p, xsi, eta)
    elif shape_function == 'quadratic':
        return warp_quadratic(p, xsi, eta)
    else:
        raise ValueError(f"Unknown shape function: {shape_function}")


def compute_steepest_descent(dfdx: np.ndarray, dfdy: np.ndarray,
                              xsi: np.ndarray, eta: np.ndarray,
                              shape_function: str = 'affine') -> np.ndarray:
    """
    통합 Steepest Descent 함수
    """
    if shape_function == 'affine':
        return compute_steepest_descent_affine(dfdx, dfdy, xsi, eta)
    elif shape_function == 'quadratic':
        return compute_steepest_descent_quadratic(dfdx, dfdy, xsi, eta)
    else:
        raise ValueError(f"Unknown shape function: {shape_function}")


def update_warp_inverse_compositional(
    p: np.ndarray, 
    dp: np.ndarray, 
    shape_function: str = 'affine'
) -> np.ndarray:
    """
    Inverse Compositional Warp Update
    
    W(p) ← W(p) · W(Δp)^(-1)
    
    Affine: Jiang et al. (2014) Eq. (12) 직접 계산으로 속도 향상
    Quadratic: 최적화된 행렬 방식
    """
    if shape_function == 'affine':
        return _update_affine_direct(p, dp)
    else:  # quadratic
        return _update_quadratic_direct(p, dp)


def get_initial_params(shape_function: str = 'affine') -> np.ndarray:
    """
    초기 파라미터 반환 (모두 0)
    """
    if shape_function == 'affine':
        return np.zeros(6, dtype=np.float64)
    elif shape_function == 'quadratic':
        return np.zeros(12, dtype=np.float64)
    else:
        raise ValueError(f"Unknown shape function: {shape_function}")


def get_num_params(shape_function: str = 'affine') -> int:
    """
    파라미터 개수 반환
    """
    if shape_function == 'affine':
        return 6
    elif shape_function == 'quadratic':
        return 12
    else:
        raise ValueError(f"Unknown shape function: {shape_function}")
