"""
IC-GN (Inverse Compositional Gauss-Newton) 최적화 모듈

References:
    - Pan, B., et al. Experimental Mechanics, 2013.
    - Pan, B., et al. Optics and Lasers in Engineering, 2013.
    - Jiang, Z., et al. Optics and Lasers in Engineering, 2014.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
import os
import logging

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
    get_num_params,
    check_convergence,
)

_logger = logging.getLogger(__name__)


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
    gaussian_blur: Optional[int] = None,
    n_workers: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> ICGNResult:
    """IC-GN 서브픽셀 최적화"""
    start_time = time.time()

    if shape_function not in ('affine', 'quadratic'):
        raise ValueError(
            f"shape_function must be 'affine' or 'quadratic', "
            f"got '{shape_function}'")

    n_params = get_num_params(shape_function)

    # 전처리
    ref_gray = _to_gray(ref_image).astype(np.float64)
    def_gray = _to_gray(def_image).astype(np.float64)

    if gaussian_blur is not None and gaussian_blur > 0:
        if gaussian_blur % 2 == 0:
            gaussian_blur += 1
        ref_gray = cv2.GaussianBlur(
            ref_gray, (gaussian_blur, gaussian_blur), 0)
        def_gray = cv2.GaussianBlur(
            def_gray, (gaussian_blur, gaussian_blur), 0)

    grad_x, grad_y = _compute_gradient(ref_gray)
    target_interp = create_interpolator(def_gray, order=interpolation_order)
    xsi, eta = generate_local_coordinates(subset_size)

    points_x = initial_guess.points_x
    points_y = initial_guess.points_y
    n_points = len(points_x)

    if n_points == 0:
        return _empty_result(
            subset_size, max_iterations,
            convergence_threshold, shape_function)

    # 결과 배열
    disp_u = np.zeros(n_points, dtype=np.float64)
    disp_v = np.zeros(n_points, dtype=np.float64)
    disp_ux = np.zeros(n_points, dtype=np.float64)
    disp_uy = np.zeros(n_points, dtype=np.float64)
    disp_vx = np.zeros(n_points, dtype=np.float64)
    disp_vy = np.zeros(n_points, dtype=np.float64)

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

    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)

    if progress_callback:
        progress_callback(0, n_points)

    def process_poi(idx: int) -> Tuple[int, np.ndarray, float, int, bool]:
        px = points_x[idx]
        py = points_y[idx]

        ref_data = _extract_reference_subset(
            ref_gray, grad_x, grad_y, px, py, subset_size)

        if ref_data is None:
            return idx, np.zeros(n_params), 0.0, 0, False

        f, dfdx, dfdy, f_mean, f_tilde = ref_data

        J = compute_steepest_descent(dfdx, dfdy, xsi, eta, shape_function)
        H = compute_hessian(J)

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return idx, np.zeros(n_params), 0.0, 0, False

        p = get_initial_params(shape_function)
        if shape_function == 'affine':
            p[0] = float(initial_guess.disp_u[idx])
            p[3] = float(initial_guess.disp_v[idx])
        else:
            p[0] = float(initial_guess.disp_u[idx])
            p[6] = float(initial_guess.disp_v[idx])

        p_final, zncc, n_iter, conv = _icgn_iterate(
            f, f_mean, f_tilde,
            J, H_inv,
            target_interp,
            px, py,
            xsi, eta,
            p,
            subset_size,
            max_iterations,
            convergence_threshold,
            shape_function,
        )

        return idx, p_final, zncc, n_iter, conv

    # 병렬 처리
    completed = 0
    update_interval = max(1, n_points // 50)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_poi, i): i for i in range(n_points)}

        for future in as_completed(futures):
            idx, p_final, zncc, n_iter, conv = future.result()

            if shape_function == 'affine':
                disp_u[idx] = p_final[0]
                disp_ux[idx] = p_final[1]
                disp_uy[idx] = p_final[2]
                disp_v[idx] = p_final[3]
                disp_vx[idx] = p_final[4]
                disp_vy[idx] = p_final[5]
            else:
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
            valid_mask[idx] = zncc >= zncc_threshold

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
        shape_function=shape_function,
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
    subset_size: int,
    max_iterations: int,
    convergence_threshold: float,
    shape_function: str = 'affine',
) -> Tuple[np.ndarray, float, int, bool]:
    """단일 POI에 대한 IC-GN 반복"""
    n_iter = 0
    conv = False
    zncc = 0.0

    if shape_function == 'affine':
        u_idx, v_idx = 0, 3
    else:
        u_idx, v_idx = 0, 6

    p_initial_u = p[u_idx]
    p_initial_v = p[v_idx]

    # --- 발산 임계값: subset 크기에 비례 스케일링 ---
    # check_convergence의 norm은 gradient 성분에 half²을 곱하므로
    # subset이 커지면 norm도 비례해서 커짐
    # 기준: subset_size=21(half=10)에서 dp_norm > 1.0이 발산
    half = subset_size // 2
    reference_half = 10  # subset_size=21 기준
    divergence_threshold = 1.0 * (half / reference_half)

    # 변위 변화 허용량도 subset 크기에 비례
    # 큰 subset은 더 넓은 영역을 커버하므로 보정 범위도 커야 함
    max_displacement_change = 5.0 * (half / reference_half)

    for iteration in range(max_iterations):
        n_iter = iteration + 1

        # Warp
        xsi_w, eta_w = warp(p, xsi, eta, shape_function)
        x_def = cx + xsi_w
        y_def = cy + eta_w

        # 보간
        g = target_interp(y_def, x_def)

        # ZNCC
        g_mean = np.mean(g)
        g_tilde = np.linalg.norm(g - g_mean)

        if g_tilde < 1e-10:
            break

        znssd = _compute_znssd(f, f_mean, f_tilde, g, g_mean, g_tilde)
        zncc = 1.0 - 0.5 * znssd

        if zncc < 0.5:
            conv = False
            break

        # 파라미터 업데이트
        residual = (f - f_mean) - (f_tilde / g_tilde) * (g - g_mean)
        b = -J.T @ residual
        dp = H_inv @ b

        # 수렴 체크
        conv, dp_norm = check_convergence(
            dp, subset_size, convergence_threshold, shape_function)

        if conv:
            break

        # 발산 판단 — subset 크기 비례 임계값 사용
        if dp_norm > divergence_threshold:
            conv = False
            break

        # Warp 업데이트
        p = update_warp_inverse_compositional(p, dp, shape_function)

        displacement_change_u = abs(p[u_idx] - p_initial_u)
        displacement_change_v = abs(p[v_idx] - p_initial_v)

        if (displacement_change_u > max_displacement_change
                or displacement_change_v > max_displacement_change):
            conv = False
            break

    return p, zncc, n_iter, conv

# ===== 유틸 =====

def _to_gray(img: np.ndarray) -> np.ndarray:
    """그레이스케일 변환"""
    if img is None:
        raise ValueError("이미지가 None입니다")
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _compute_gradient(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    이미지 gradient 계산 (Sobel ksize=5, /32.0)

    SSSIG의 compute_gradient와 동일한 설정.
    """
    ksize = 5
    sobel_div = 32.0
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize) / sobel_div
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=ksize) / sobel_div
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

    if (cy - half < 0 or cy + half >= h
            or cx - half < 0 or cx + half >= w):
        return None

    f = ref_image[cy-half:cy+half+1, cx-half:cx+half+1].ravel()
    dfdx = grad_x[cy-half:cy+half+1, cx-half:cx+half+1].ravel()
    dfdy = grad_y[cy-half:cy+half+1, cx-half:cx+half+1].ravel()

    f_mean = np.mean(f)
    f_tilde = np.linalg.norm(f - f_mean)

    if f_tilde < 1e-10:
        return None

    return f, dfdx, dfdy, f_mean, f_tilde


def _compute_znssd(
    f: np.ndarray, f_mean: float, f_tilde: float,
    g: np.ndarray, g_mean: float, g_tilde: float
) -> float:
    """ZNSSD 계산"""
    diff = (f - f_mean) / f_tilde - (g - g_mean) / g_tilde
    return float(np.sum(diff ** 2))


def _empty_result(
    subset_size: int,
    max_iterations: int,
    convergence_threshold: float,
    shape_function: str = 'affine'
) -> ICGNResult:
    """빈 결과"""
    is_quad = shape_function == 'quadratic'
    empty_f = np.array([], dtype=np.float64)
    return ICGNResult(
        points_y=np.array([], dtype=np.int64),
        points_x=np.array([], dtype=np.int64),
        disp_u=empty_f, disp_v=empty_f,
        disp_ux=empty_f, disp_uy=empty_f,
        disp_vx=empty_f, disp_vy=empty_f,
        disp_uxx=empty_f if is_quad else None,
        disp_uxy=empty_f if is_quad else None,
        disp_uyy=empty_f if is_quad else None,
        disp_vxx=empty_f if is_quad else None,
        disp_vxy=empty_f if is_quad else None,
        disp_vyy=empty_f if is_quad else None,
        zncc_values=empty_f,
        iterations=np.array([], dtype=np.int32),
        converged=np.array([], dtype=bool),
        valid_mask=np.array([], dtype=bool),
        subset_size=subset_size,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        processing_time=0.0,
        shape_function=shape_function,
    )
