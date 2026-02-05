# speckle/core/optimization/icgn.py

"""
IC-GN (Inverse Compositional Gauss-Newton) 최적화 모듈

FFTCC 초기 추정값을 서브픽셀 정밀도로 최적화
Affine (1차) 및 Quadratic (2차) Shape Function 지원
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
    warp,
    update_warp_inverse_compositional,
    compute_steepest_descent,
    compute_hessian,
    get_initial_params,
    get_num_params
)


# ===== 메인 함수 =====

def compute_icgn(
    ref_image: np.ndarray,
    def_image: np.ndarray,
    initial_guess: FFTCCResult,
    subset_size: int = 21,
    max_iterations: int = 50,
    convergence_threshold: float = 0.001,
    zncc_threshold: float = 0.6,
    interpolation_order: int = 5,
    shape_function: str = 'affine',
    gaussian_blur: Optional[int] = None,  # 추가: Pan (2013) Gaussian pre-filtering
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
        shape_function: 'affine' (6 params) or 'quadratic' (12 params)
        gaussian_blur: Gaussian pre-filtering 커널 크기 (None이면 비활성화)
                       Pan et al. (2013) 권장: 5
        n_workers: 병렬 워커 수
        progress_callback: 진행 콜백 (current, total)
    
    Returns:
        ICGNResult: 서브픽셀 정밀도 결과
        
    References:
        Pan, B., et al. "Bias error reduction of digital image correlation 
        using Gaussian pre-filtering." Optics and Lasers in Engineering, 2013.
    """
    start_time = time.time()
    
    # Shape function 검증
    if shape_function not in ('affine', 'quadratic'):
        raise ValueError(f"shape_function must be 'affine' or 'quadratic', got '{shape_function}'")
    
    n_params = get_num_params(shape_function)
    
    # 전처리
    ref_gray = _to_gray(ref_image).astype(np.float64)
    def_gray = _to_gray(def_image).astype(np.float64)
    
    # Gaussian pre-filtering 적용 (Pan 2013)
    if gaussian_blur is not None and gaussian_blur > 0:
        # 커널 크기는 홀수여야 함
        if gaussian_blur % 2 == 0:
            gaussian_blur += 1
        print(f"[DEBUG] Gaussian Blur 적용: 커널 크기 = {gaussian_blur}")  
        ref_gray = cv2.GaussianBlur(ref_gray, (gaussian_blur, gaussian_blur), 0)
        def_gray = cv2.GaussianBlur(def_gray, (gaussian_blur, gaussian_blur), 0)
    else:
        print(f"[DEBUG] Gaussian Blur 미적용 (gaussian_blur = {gaussian_blur})")

    # Gradient 계산 (Blur된 이미지에서)
    grad_x, grad_y = _compute_gradient(ref_gray)
    
    # Target 보간 함수 생성 (Blur된 이미지로)
    target_interp = create_interpolator(def_gray, order=interpolation_order)
    
    # 로컬 좌표 생성
    xsi, eta = generate_local_coordinates(subset_size)
    
    # POI 정보
    points_x = initial_guess.points_x
    points_y = initial_guess.points_y
    n_points = len(points_x)
    
    if n_points == 0:
        return _empty_result(subset_size, max_iterations, convergence_threshold, shape_function)
    
    # 결과 배열 초기화
    disp_u = np.zeros(n_points, dtype=np.float64)
    disp_v = np.zeros(n_points, dtype=np.float64)
    disp_ux = np.zeros(n_points, dtype=np.float64)
    disp_uy = np.zeros(n_points, dtype=np.float64)
    disp_vx = np.zeros(n_points, dtype=np.float64)
    disp_vy = np.zeros(n_points, dtype=np.float64)
    
    # Quadratic 전용 필드
    if shape_function == 'quadratic':
        disp_uxx = np.zeros(n_points, dtype=np.float64)
        disp_uxy = np.zeros(n_points, dtype=np.float64)
        disp_uyy = np.zeros(n_points, dtype=np.float64)
        disp_vxx = np.zeros(n_points, dtype=np.float64)
        disp_vxy = np.zeros(n_points, dtype=np.float64)
        disp_vyy = np.zeros(n_points, dtype=np.float64)
    else:
        disp_uxx = disp_uxy = disp_uyy = None
        disp_vxx = disp_vxy = disp_vyy = None
    
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
        # 디버그 (처음 3개만)
        if idx < 3:
            print(f"[DEBUG] IC-GN POI[{idx}]: 위치=({px}, {py}), 초기변위=({initial_guess.disp_u[idx]}, {initial_guess.disp_v[idx]})")
        
        # FFTCC 결과가 유효하지 않으면 스킵
        if not initial_guess.valid_mask[idx]:
            return idx, np.zeros(n_params), 0.0, 0, False
        
        # Reference subset 추출
        ref_data = _extract_reference_subset(
            ref_gray, grad_x, grad_y, px, py, subset_size
        )
        
        if ref_data is None:
            return idx, np.zeros(n_params), 0.0, 0, False
        
        f, dfdx, dfdy, f_mean, f_tilde = ref_data
        
        # Jacobian & Hessian (한 번만 계산)
        J = compute_steepest_descent(dfdx, dfdy, xsi, eta, shape_function)
        H = compute_hessian(J)
        
        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return idx, np.zeros(n_params), 0.0, 0, False
        
        # 초기값 설정 (FFTCC 결과)
        p = get_initial_params(shape_function)
        if shape_function == 'affine':
            p[0] = float(initial_guess.disp_u[idx])  # u
            p[3] = float(initial_guess.disp_v[idx])  # v
        else:  # quadratic
            p[0] = float(initial_guess.disp_u[idx])  # u
            p[6] = float(initial_guess.disp_v[idx])  # v
        
        # IC-GN 반복
        p_final, zncc, n_iter, conv = _icgn_iterate(
            f, f_mean, f_tilde,
            J, H_inv,
            target_interp,
            px, py,
            xsi, eta,
            p,
            max_iterations,
            convergence_threshold,
            shape_function
        )
        
        return idx, p_final, zncc, n_iter, conv
    
    # 병렬 처리
    completed = 0
    update_interval = max(1, n_points // 50)
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_poi, i): i for i in range(n_points)}
        
        for future in as_completed(futures):
            idx, p_final, zncc, n_iter, conv = future.result()
            
            if shape_function == 'affine':
                disp_u[idx] = p_final[0]
                disp_ux[idx] = p_final[1]
                disp_uy[idx] = p_final[2]
                disp_v[idx] = p_final[3]
                disp_vx[idx] = p_final[4]
                disp_vy[idx] = p_final[5]
            else:  # quadratic
                disp_u[idx] = p_final[0]
                disp_ux[idx] = p_final[1]
                disp_uy[idx] = p_final[2]
                disp_uxx[idx] = p_final[3]
                disp_uxy[idx] = p_final[4]
                disp_uyy[idx] = p_final[5]
                disp_v[idx] = p_final[6]
                disp_vx[idx] = p_final[7]
                disp_vy[idx] = p_final[8]
                disp_vxx[idx] = p_final[9]
                disp_vxy[idx] = p_final[10]
                disp_vyy[idx] = p_final[11]
            
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
        disp_uxx=disp_uxx,
        disp_uxy=disp_uxy,
        disp_uyy=disp_uyy,
        disp_vxx=disp_vxx,
        disp_vxy=disp_vxy,
        disp_vyy=disp_vyy,
        zncc_values=zncc_values,
        iterations=iterations,
        converged=converged,
        valid_mask=valid_mask,
        subset_size=subset_size,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        processing_time=processing_time,
        shape_function=shape_function
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
    shape_function: str = 'affine',
    debug: bool = False
) -> Tuple[np.ndarray, float, int, bool]:
    """
    단일 POI에 대한 IC-GN 반복
    """
    n_iter = 0
    conv = False
    zncc = 0.0
    fail_reason = ""  # 추가: 실패 원인 추적
    
    # 수렴/발산 체크용 인덱스 (u, v 위치)
    if shape_function == 'affine':
        u_idx, v_idx = 0, 3
    else:  # quadratic
        u_idx, v_idx = 0, 6
    
    # 초기값 저장 (발산 판정용)
    p_initial_u = p[u_idx]
    p_initial_v = p[v_idx]
    
    max_displacement_change = 5.0
    
    for iteration in range(max_iterations):
        n_iter = iteration + 1
        
        # Warp된 좌표 계산
        xsi_w, eta_w = warp(p, xsi, eta, shape_function)
        
        # 전역 좌표로 변환
        x_def = cx + xsi_w
        y_def = cy + eta_w
        
        # Target subset 보간
        g = target_interp(y_def, x_def)
        
        # g 통계
        g_mean = np.mean(g)
        g_tilde = np.linalg.norm(g - g_mean)
        
        if g_tilde < 1e-10:
            fail_reason = "g_tilde too small"
            break
        
        # ZNSSD 계산
        znssd = _compute_znssd(f, f_mean, f_tilde, g, g_mean, g_tilde)
        zncc = 1.0 - 0.5 * znssd
        
        # 발산 체크 1: ZNCC가 너무 낮으면 발산
        if zncc < 0.5:
            fail_reason = f"ZNCC too low: {zncc:.4f}"
            conv = False
            break
        
        # Residual 계산
        residual = (f - f_mean) - (f_tilde / g_tilde) * (g - g_mean)
        
        # 파라미터 증분 계산
        b = -J.T @ residual
        dp = H_inv @ b
        
        # 발산 체크 2: 업데이트가 너무 크면 발산
        dp_norm = np.sqrt(dp[u_idx]**2 + dp[v_idx]**2)
        if dp_norm > 2.0:
            fail_reason = f"dp_norm too large: {dp_norm:.4f}"
            conv = False
            break
        
        # 수렴 체크
        if dp_norm < convergence_threshold:
            conv = True
            break
        
        # Inverse compositional update
        p = update_warp_inverse_compositional(p, dp, shape_function)
        
        # 발산 체크 3: 초기값에서 너무 벗어나면 발산
        displacement_change_u = abs(p[u_idx] - p_initial_u)
        displacement_change_v = abs(p[v_idx] - p_initial_v)
        
        if displacement_change_u > max_displacement_change or \
           displacement_change_v > max_displacement_change:
            fail_reason = f"displacement change too large: u={displacement_change_u:.2f}, v={displacement_change_v:.2f}"
            conv = False
            break
    
    # 최대 반복 도달 시
    if n_iter == max_iterations and not conv:
        fail_reason = f"max iterations reached, dp_norm={dp_norm:.4f}, zncc={zncc:.4f}"
    
    # ===== 디버그 출력: 실패한 경우만 =====
    if not conv and np.random.random() < 0.1:  # 10% 샘플링
        print(f"[ICGN FAIL] POI({cx},{cy}): {fail_reason}")
    
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
    """이미지 gradient 계산 (Sobel)"""
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
    """Reference subset 추출"""
    half = subset_size // 2
    h, w = ref_image.shape
    
    if (cy - half < 0 or cy + half >= h or
        cx - half < 0 or cx + half >= w):
        return None
    
    f = ref_image[cy - half:cy + half + 1, cx - half:cx + half + 1].ravel()
    dfdx = grad_x[cy - half:cy + half + 1, cx - half:cx + half + 1].ravel()
    dfdy = grad_y[cy - half:cy + half + 1, cx - half:cx + half + 1].ravel()
    
    f_mean = np.mean(f)
    f_tilde = np.linalg.norm(f - f_mean)
    
    if f_tilde < 1e-10:
        return None
    
    return f, dfdx, dfdy, f_mean, f_tilde


def _compute_znssd(
    f: np.ndarray, f_mean: float, f_tilde: float,
    g: np.ndarray, g_mean: float, g_tilde: float
) -> float:
    """ZNSSD (Zero-mean Normalized SSD) 계산"""
    diff = (f - f_mean) / f_tilde - (g - g_mean) / g_tilde
    return float(np.sum(diff ** 2))


def _empty_result(
    subset_size: int,
    max_iterations: int,
    convergence_threshold: float,
    shape_function: str = 'affine'
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
        disp_uxx=np.array([], dtype=np.float64) if shape_function == 'quadratic' else None,
        disp_uxy=np.array([], dtype=np.float64) if shape_function == 'quadratic' else None,
        disp_uyy=np.array([], dtype=np.float64) if shape_function == 'quadratic' else None,
        disp_vxx=np.array([], dtype=np.float64) if shape_function == 'quadratic' else None,
        disp_vxy=np.array([], dtype=np.float64) if shape_function == 'quadratic' else None,
        disp_vyy=np.array([], dtype=np.float64) if shape_function == 'quadratic' else None,
        zncc_values=np.array([], dtype=np.float64),
        iterations=np.array([], dtype=np.int32),
        converged=np.array([], dtype=bool),
        valid_mask=np.array([], dtype=bool),
        subset_size=subset_size,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        processing_time=0.0,
        shape_function=shape_function
    )
