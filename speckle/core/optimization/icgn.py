"""
IC-GN (Inverse Compositional Gauss-Newton) 최적화 모듈

FFTCC 초기 추정값을 서브픽셀 정밀도로 최적화
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os

from ..initial_guess.results import FFTCCResult
from .results import ICGNResult
from .interpolation import create_interpolator, ImageInterpolator
from .shape_function import (
    generate_local_coordinates,
    warp_affine,
    update_warp_inverse_compositional_affine,
    compute_steepest_descent_affine,
    compute_hessian
)


# ===== 메인 함수 =====

def compute_icgn(
    ref_image: np.ndarray,
    def_image: np.ndarray,
    initial_guess: FFTCCResult,
    subset_size: int = 21,
    max_iterations: int = 50,
    convergence_threshold: float = 0.0001,
    zncc_threshold: float = 0.6,
    interpolation_order: int = 5,
    n_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> ICGNResult:
    """
    IC-GN 서브픽셀 최적화
    
    Args:
        ref_image: Reference 이미지
        def_image: Deformed 이미지
        initial_guess: FFTCC 결과 (정수 픽셀 변위)
        subset_size: 서브셋 크기 (홀수)
        max_iterations: 최대 반복 횟수
        convergence_threshold: 수렴 기준 (|Δp| norm)
        zncc_threshold: 유효 판정 ZNCC 임계값
        interpolation_order: 보간 차수 (3 or 5)
        n_workers: 병렬 워커 수
        progress_callback: 진행 콜백 (current, total)
    
    Returns:
        ICGNResult: 서브픽셀 정밀도 결과
    """
    start_time = time.time()
    
    # 전처리
    ref_gray = _to_gray(ref_image).astype(np.float64)
    def_gray = _to_gray(def_image).astype(np.float64)
    
    # Gradient 계산
    grad_x, grad_y = _compute_gradient(ref_gray)
    
    # Target 보간 함수 생성
    target_interp = create_interpolator(def_gray, order=interpolation_order)
    
    # 로컬 좌표 생성
    xsi, eta = generate_local_coordinates(subset_size)
    
    # POI 정보
    points_x = initial_guess.points_x
    points_y = initial_guess.points_y
    n_points = len(points_x)
    
    if n_points == 0:
        return _empty_result(subset_size, max_iterations, convergence_threshold)
    
    # 결과 배열 초기화
    disp_u = np.zeros(n_points, dtype=np.float64)
    disp_v = np.zeros(n_points, dtype=np.float64)
    disp_ux = np.zeros(n_points, dtype=np.float64)
    disp_uy = np.zeros(n_points, dtype=np.float64)
    disp_vx = np.zeros(n_points, dtype=np.float64)
    disp_vy = np.zeros(n_points, dtype=np.float64)
    zncc_values = np.zeros(n_points, dtype=np.float64)
    iterations = np.zeros(n_points, dtype=np.int32)
    converged = np.zeros(n_points, dtype=bool)
    valid_mask = np.zeros(n_points, dtype=bool)
    
    # 워커 수 설정
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)
    
    if progress_callback:
        progress_callback(0, n_points)
    
    # POI 처리 함수
    def process_poi(idx: int) -> Tuple[int, np.ndarray, float, int, bool]:
        px = points_x[idx]
        py = points_y[idx]
        
        # FFTCC 결과가 유효하지 않으면 스킵
        if not initial_guess.valid_mask[idx]:
            return idx, np.zeros(6), 0.0, 0, False
        
        # Reference subset 추출
        ref_data = _extract_reference_subset(
            ref_gray, grad_x, grad_y, px, py, subset_size
        )
        
        if ref_data is None:
            return idx, np.zeros(6), 0.0, 0, False
        
        f, dfdx, dfdy, f_mean, f_tilde = ref_data
        
        # Jacobian & Hessian (한 번만 계산)
        J = compute_steepest_descent_affine(dfdx, dfdy, xsi, eta)
        H = compute_hessian(J)
        
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return idx, np.zeros(6), 0.0, 0, False
        
        # 초기값 설정 (FFTCC 결과)
        p = np.array([
            float(initial_guess.disp_u[idx]),  # u
            0.0,                                # ux
            0.0,                                # uy
            float(initial_guess.disp_v[idx]),  # v
            0.0,                                # vx
            0.0                                 # vy
        ], dtype=np.float64)
        
        # IC-GN 반복
        p_final, zncc, n_iter, conv = _icgn_iterate(
            f, f_mean, f_tilde,
            J, H_inv,
            target_interp,
            px, py,
            xsi, eta,
            p,
            max_iterations,
            convergence_threshold
        )
        
        return idx, p_final, zncc, n_iter, conv
    
    # 병렬 처리
    completed = 0
    update_interval = max(1, n_points // 50)
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_poi, i): i for i in range(n_points)}
        
        for future in as_completed(futures):
            idx, p_final, zncc, n_iter, conv = future.result()
            
            disp_u[idx] = p_final[0]
            disp_ux[idx] = p_final[1]
            disp_uy[idx] = p_final[2]
            disp_v[idx] = p_final[3]
            disp_vx[idx] = p_final[4]
            disp_vy[idx] = p_final[5]
            zncc_values[idx] = zncc
            iterations[idx] = n_iter
            converged[idx] = conv
            valid_mask[idx] = conv and (zncc >= zncc_threshold)
            
            completed += 1
            if progress_callback and completed % update_interval == 0:
                progress_callback(completed, n_points)
    
    if progress_callback:
        progress_callback(n_points, n_points)
    
    processing_time = time.time() - start_time
    
    return ICGNResult(
        points_y=points_y,
        points_x=points_x,
        disp_u=disp_u,
        disp_v=disp_v,
        disp_ux=disp_ux,
        disp_uy=disp_uy,
        disp_vx=disp_vx,
        disp_vy=disp_vy,
        zncc_values=zncc_values,
        iterations=iterations,
        converged=converged,
        valid_mask=valid_mask,
        subset_size=subset_size,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        processing_time=processing_time
    )


# ===== IC-GN 반복 =====
def _icgn_iterate(
    f: np.ndarray,
    f_mean: float,
    f_tilde: float,
    J: np.ndarray,
    H_inv: np.ndarray,
    target_interp: ImageInterpolator,
    cx: int, cy: int,
    xsi: np.ndarray,
    eta: np.ndarray,
    p: np.ndarray,
    max_iterations: int,
    convergence_threshold: float,
    debug: bool = False
) -> Tuple[np.ndarray, float, int, bool]:
    """
    단일 POI에 대한 IC-GN 반복
    """
    n_iter = 0
    conv = False
    zncc = 0.0
    
    for iteration in range(max_iterations):
        n_iter = iteration + 1
        
        # Warp된 좌표 계산
        xsi_w, eta_w = warp_affine(p, xsi, eta)
        
        # 전역 좌표로 변환
        x_def = cx + xsi_w
        y_def = cy + eta_w
        
        # Target subset 보간
        g = target_interp(y_def, x_def)
        
        # g 통계
        g_mean = np.mean(g)
        g_tilde = np.linalg.norm(g - g_mean)
        
        if g_tilde < 1e-10:
            break
        
        # ZNSSD 계산
        znssd = _compute_znssd(f, f_mean, f_tilde, g, g_mean, g_tilde)
        zncc = 1.0 - 0.5 * znssd
        
        # Residual 계산 (SUN-DIC 방식!)
        residual = (f - f_mean) - (f_tilde / g_tilde) * (g - g_mean)
        
        # 파라미터 증분 계산 (SUN-DIC 방식!)
        b = -J.T @ residual
        dp = H_inv @ b
        
        # 디버그 출력
        if debug and iteration < 5:
            print(f"    iter {iteration}: p=[{p[0]:.4f}, {p[3]:.4f}], dp=[{dp[0]:.4f}, {dp[3]:.4f}], zncc={zncc:.4f}")
        
        # 수렴 체크
        dp_norm = np.sqrt(dp[0]**2 + dp[3]**2)
        
        if dp_norm < convergence_threshold:
            conv = True
            break
        
        # Inverse compositional update
        p = update_warp_inverse_compositional_affine(p, dp)
    
    return p, zncc, n_iter, conv

# ===== 전처리 함수 =====

def _to_gray(img: np.ndarray) -> np.ndarray:
    """그레이스케일 변환"""
    if img is None:
        raise ValueError("이미지가 None입니다")
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _compute_gradient(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    이미지 gradient 계산 (Sobel)
    
    Returns:
        (grad_x, grad_y)
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5) / 32.0
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5) / 32.0
    
    return grad_x, grad_y


def _extract_reference_subset(
    ref_image: np.ndarray,
    grad_x: np.ndarray,
    grad_y: np.ndarray,
    cx: int, cy: int,
    subset_size: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, float, float]]:
    """
    Reference subset 추출
    
    Returns:
        (f, dfdx, dfdy, f_mean, f_tilde) or None
    """
    half = subset_size // 2
    h, w = ref_image.shape
    
    # 경계 체크
    if (cy - half < 0 or cy + half >= h or
        cx - half < 0 or cx + half >= w):
        return None
    
    # Subset 추출
    f = ref_image[cy - half:cy + half + 1, cx - half:cx + half + 1].ravel()
    dfdx = grad_x[cy - half:cy + half + 1, cx - half:cx + half + 1].ravel()
    dfdy = grad_y[cy - half:cy + half + 1, cx - half:cx + half + 1].ravel()
    
    # 통계 계산
    f_mean = np.mean(f)
    f_tilde = np.linalg.norm(f - f_mean)
    
    if f_tilde < 1e-10:
        return None
    
    return f, dfdx, dfdy, f_mean, f_tilde


def _compute_znssd(
    f: np.ndarray, f_mean: float, f_tilde: float,
    g: np.ndarray, g_mean: float, g_tilde: float
) -> float:
    """
    ZNSSD (Zero-mean Normalized SSD) 계산
    
    C = Σ[(f-f̄)/f̃ - (g-ḡ)/g̃]²
    """
    diff = (f - f_mean) / f_tilde - (g - g_mean) / g_tilde
    return float(np.sum(diff ** 2))


def _empty_result(
    subset_size: int,
    max_iterations: int,
    convergence_threshold: float
) -> ICGNResult:
    """빈 결과 반환"""
    return ICGNResult(
        points_y=np.array([], dtype=np.int64),
        points_x=np.array([], dtype=np.int64),
        disp_u=np.array([], dtype=np.float64),
        disp_v=np.array([], dtype=np.float64),
        disp_ux=np.array([], dtype=np.float64),
        disp_uy=np.array([], dtype=np.float64),
        disp_vx=np.array([], dtype=np.float64),
        disp_vy=np.array([], dtype=np.float64),
        zncc_values=np.array([], dtype=np.float64),
        iterations=np.array([], dtype=np.int32),
        converged=np.array([], dtype=bool),
        valid_mask=np.array([], dtype=bool),
        subset_size=subset_size,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        processing_time=0.0
    )
