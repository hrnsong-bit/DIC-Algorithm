"""
Shape Function 모듈

Affine (1차) 및 Quadratic (2차) 변형 함수 지원
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


def compute_warp_matrix_affine(p: np.ndarray) -> np.ndarray:
    """
    Affine warp matrix 생성 (3x3)
    """
    u, ux, uy, v, vx, vy = p[:6]
    
    W = np.array([
        [1 + ux, uy,     u],
        [vx,     1 + vy, v],
        [0,      0,      1]
    ], dtype=np.float64)
    
    return W


def update_warp_inverse_compositional_affine(
    p: np.ndarray,
    dp: np.ndarray
) -> np.ndarray:
    """
    Inverse Compositional warp update (Affine)
    """
    W_p = compute_warp_matrix_affine(p)
    W_dp = compute_warp_matrix_affine(dp)
    
    W_dp_inv = np.linalg.inv(W_dp)
    W_new = W_p @ W_dp_inv
    
    p_new = np.array([
        W_new[0, 2],      # u
        W_new[0, 0] - 1,  # ux
        W_new[0, 1],      # uy
        W_new[1, 2],      # v
        W_new[1, 0],      # vx
        W_new[1, 1] - 1   # vy
    ], dtype=np.float64)
    
    return p_new


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
    
    ξ' = u + (1+ux)·ξ + uy·η + 0.5·uxx·ξ² + uxy·ξ·η + 0.5·uyy·η²
    η' = v + vx·ξ + (1+vy)·η + 0.5·vxx·ξ² + vxy·ξ·η + 0.5·vyy·η²
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


def compute_warp_matrix_quadratic(p: np.ndarray) -> np.ndarray:
    """
    Quadratic warp matrix 생성 (6x6)
    
    SUN-DIC 방식 참고
    """
    u, ux, uy, uxx, uxy, uyy = p[0:6]
    v, vx, vy, vxx, vxy, vyy = p[6:12]
    
    # 중간 계산
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
    
    W = np.array([
        [1+A1,      A2,      A3,      A4,      A5,   A6],
        [A7,      1+A8,      A9,     A10,     A11,  A12],
        [A13,      A14,   1+A15,     A16,     A17,  A18],
        [0.5*uxx,  uxy, 0.5*uyy,   1+ux,      uy,    u],
        [0.5*vxx,  vxy, 0.5*vyy,     vx,    1+vy,    v],
        [0,          0,       0,      0,       0,    1]
    ], dtype=np.float64)
    
    return W


def update_warp_inverse_compositional_quadratic(
    p: np.ndarray,
    dp: np.ndarray
) -> np.ndarray:
    """
    Inverse Compositional warp update (Quadratic)
    
    W(p) ← W(p) · W(Δp)^(-1)
    """
    W_p = compute_warp_matrix_quadratic(p)
    W_dp = compute_warp_matrix_quadratic(dp)
    
    W_dp_inv = np.linalg.inv(W_dp)
    W_new = W_p @ W_dp_inv
    
    # 행렬에서 파라미터 추출
    p_new = np.array([
        W_new[3, 5],        # u
        W_new[3, 3] - 1,    # ux
        W_new[3, 4],        # uy
        2 * W_new[3, 0],    # uxx
        W_new[3, 1],        # uxy
        2 * W_new[3, 2],    # uyy
        W_new[4, 5],        # v
        W_new[4, 3],        # vx
        W_new[4, 4] - 1,    # vy
        2 * W_new[4, 0],    # vxx
        W_new[4, 1],        # vxy
        2 * W_new[4, 2]     # vyy
    ], dtype=np.float64)
    
    return p_new


def compute_steepest_descent_quadratic(
    dfdx: np.ndarray,
    dfdy: np.ndarray,
    xsi: np.ndarray,
    eta: np.ndarray
) -> np.ndarray:
    """
    Steepest Descent Image (Quadratic)
    
    J = [∂f/∂x, ∂f/∂x·ξ, ∂f/∂x·η, 0.5·∂f/∂x·ξ², ∂f/∂x·ξη, 0.5·∂f/∂x·η²,
         ∂f/∂y, ∂f/∂y·ξ, ∂f/∂y·η, 0.5·∂f/∂y·ξ², ∂f/∂y·ξη, 0.5·∂f/∂y·η²]
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


def update_warp_inverse_compositional(p: np.ndarray, dp: np.ndarray,
                                       shape_function: str = 'affine') -> np.ndarray:
    """
    통합 Inverse Compositional Update 함수
    """
    if shape_function == 'affine':
        return update_warp_inverse_compositional_affine(p, dp)
    elif shape_function == 'quadratic':
        return update_warp_inverse_compositional_quadratic(p, dp)
    else:
        raise ValueError(f"Unknown shape function: {shape_function}")


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
