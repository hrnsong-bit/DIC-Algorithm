"""
IC-GN (Inverse Compositional Gauss-Newton) 최적화 모듈

두 가지 실행 경로를 지원:
    1. 기존 (use_numba=False): ThreadPoolExecutor 기반 병렬화
    2. Numba (use_numba=True):  JIT 컴파일 + prange 병렬화 (기본값)

Numba 경로는 기존 경로와 수치적으로 동일한 결과를 생성하면서
약 5~9배 빠른 성능을 제공한다.

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
from .results import (
    ICGNResult,
    ICGN_SUCCESS,
    ICGN_FAIL_LOW_ZNCC,
    ICGN_FAIL_DIVERGED,
    ICGN_FAIL_OUT_OF_BOUNDS,
    ICGN_FAIL_SINGULAR_HESSIAN,
    ICGN_FAIL_FLAT_SUBSET,
    ICGN_FAIL_MAX_DISPLACEMENT,
    ICGN_FAIL_FLAT_TARGET,
)
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

# Numba 모듈 가용성 체크 (import 실패 시 graceful fallback)
try:
    from .icgn_core_numba import (
        process_all_pois_parallel,
        allocate_batch_buffers,
        prefilter_image,
        warmup_icgn_core,
        AFFINE as _NUMBA_AFFINE,
        QUADRATIC as _NUMBA_QUADRATIC,
    )
    from .shape_function_numba import (
        generate_local_coordinates as _numba_gen_coords,
    )
    _NUMBA_AVAILABLE = True
except ImportError:
    _NUMBA_AVAILABLE = False
    _logger.info("Numba modules not available, using original implementation")


# ===== 참조 이미지 전처리 캐시 =====

def prepare_ref_cache(
    ref_image: np.ndarray,
    subset_size: int = 21,
    interpolation_order: int = 5,
    shape_function: str = 'affine',
    gaussian_blur: Optional[int] = None,
    use_numba: bool = True,
) -> dict:
    """
    참조 이미지 전처리 결과를 캐시용 딕셔너리로 반환.

    배치 분석 시 매 프레임마다 반복되는 전처리를 1회로 줄임:
        - 그레이스케일 변환
        - Gaussian blur (선택)
        - Sobel gradient
        - Numba: B-spline prefilter, 로컬 좌표
        - Original: ImageInterpolator, 로컬 좌표

    Args:
        ref_image: 참조 이미지
        subset_size: 서브셋 크기
        interpolation_order: B-spline 차수
        shape_function: 'affine' 또는 'quadratic'
        gaussian_blur: Gaussian 블러 커널 크기
        use_numba: Numba 경로 사용 여부

    Returns:
        dict: 캐시 데이터 (compute_icgn에 ref_cache로 전달)
    """
    ref_gray = _to_gray(ref_image).astype(np.float64)

    if gaussian_blur is not None and gaussian_blur > 0:
        if gaussian_blur % 2 == 0:
            gaussian_blur += 1
        ref_gray = cv2.GaussianBlur(
            ref_gray, (gaussian_blur, gaussian_blur), 0)

    grad_x, grad_y = _compute_gradient(ref_gray)

    cache = {
        'ref_gray': ref_gray,
        'grad_x': grad_x,
        'grad_y': grad_y,
        'subset_size': subset_size,
        'interpolation_order': interpolation_order,
        'shape_function': shape_function,
    }

    if use_numba and _NUMBA_AVAILABLE:
        cache['xsi'], cache['eta'] = _numba_gen_coords(subset_size)
    else:
        cache['xsi'], cache['eta'] = generate_local_coordinates(subset_size)

    return cache


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
    progress_callback: Optional[Callable[[int, int], None]] = None,
    use_numba: bool = True,
    ref_cache: Optional[dict] = None,
) -> ICGNResult:
    """
    IC-GN 서브픽셀 최적화

    Args:
        ref_image: 참조 이미지
        def_image: 변형 이미지
        initial_guess: FFT-CC 초기 추정 결과
        subset_size: 서브셋 크기 (홀수)
        max_iterations: 최대 반복 횟수
        convergence_threshold: 수렴 임계값
        zncc_threshold: 유효 POI 판정 ZNCC 임계값
        interpolation_order: B-spline 보간 차수 (3 또는 5)
        shape_function: 형상 함수 ('affine' 또는 'quadratic')
        gaussian_blur: 가우시안 블러 커널 크기 (None이면 미적용)
        n_workers: 워커 수 (use_numba=False일 때만 사용)
        progress_callback: 진행 콜백 함수 (current, total)
        use_numba: True이면 Numba JIT + prange 병렬화 사용 (기본값)
                   False이면 기존 ThreadPoolExecutor 사용
        ref_cache: prepare_ref_cache()로 생성한 캐시 (배치 분석용, None이면 내부 계산)

    Returns:
        ICGNResult 객체
    """
    # Numba 사용 가능 여부 확인 — 불가능하면 자동 fallback
    if use_numba and not _NUMBA_AVAILABLE:
        _logger.warning("Numba modules not available, falling back to original")
        use_numba = False

    n_points = len(initial_guess.points_x)
    engine = "Numba" if use_numba else "ThreadPool"
    cached = "cached" if ref_cache else "fresh"
    _logger.info(f"IC-GN 시작: {n_points} POIs, subset={subset_size}, "
                 f"order={interpolation_order}, shape={shape_function}, "
                 f"engine={engine}, ref={cached}")

    if use_numba:
        return _compute_icgn_numba(
            ref_image, def_image, initial_guess,
            subset_size, max_iterations, convergence_threshold,
            zncc_threshold, interpolation_order, shape_function,
            gaussian_blur, progress_callback, ref_cache)
    else:
        return _compute_icgn_original(
            ref_image, def_image, initial_guess,
            subset_size, max_iterations, convergence_threshold,
            zncc_threshold, interpolation_order, shape_function,
            gaussian_blur, n_workers, progress_callback, ref_cache)


# ===== Numba 경로 =====

def _compute_icgn_numba(
    ref_image, def_image, initial_guess,
    subset_size, max_iterations, convergence_threshold,
    zncc_threshold, interpolation_order, shape_function,
    gaussian_blur, progress_callback, ref_cache=None
) -> ICGNResult:
    """Numba JIT + prange 병렬화 경로"""
    start_time = time.time()

    if shape_function not in ('affine', 'quadratic'):
        raise ValueError(
            f"shape_function must be 'affine' or 'quadratic', "
            f"got '{shape_function}'")

    shape_type = _NUMBA_AFFINE if shape_function == 'affine' else _NUMBA_QUADRATIC
    n_params = get_num_params(shape_function)

    # 참조 전처리 (캐시 사용 가능 시 건너뜀)
    if ref_cache is not None:
        ref_gray = ref_cache['ref_gray']
        grad_x = ref_cache['grad_x']
        grad_y = ref_cache['grad_y']
        xsi = ref_cache['xsi']
        eta = ref_cache['eta']
    else:
        ref_gray = _to_gray(ref_image).astype(np.float64)
        if gaussian_blur is not None and gaussian_blur > 0:
            if gaussian_blur % 2 == 0:
                gaussian_blur += 1
            ref_gray = cv2.GaussianBlur(
                ref_gray, (gaussian_blur, gaussian_blur), 0)
        grad_x, grad_y = _compute_gradient(ref_gray)
        xsi, eta = _numba_gen_coords(subset_size)

    # 변형 이미지 전처리 (매 프레임)
    def_gray = _to_gray(def_image).astype(np.float64)
    if gaussian_blur is not None and gaussian_blur > 0:
        if gaussian_blur % 2 == 0:
            gaussian_blur += 1
        def_gray = cv2.GaussianBlur(
            def_gray, (gaussian_blur, gaussian_blur), 0)

    # Numba용 prefilter (B-spline 계수) — 변형 이미지만
    coeffs = prefilter_image(def_gray, order=interpolation_order)

    points_x = initial_guess.points_x
    points_y = initial_guess.points_y
    n_points = len(points_x)

    if n_points == 0:
        return _empty_result(
            subset_size, max_iterations,
            convergence_threshold, shape_function)

    n_pixels = subset_size * subset_size

    # 진행 콜백: prange 진입 전 알림
    if progress_callback:
        progress_callback(0, n_points)

    # 좌표를 int64로 변환 (Numba 호환)
    pts_x = np.asarray(points_x, dtype=np.int64)
    pts_y = np.asarray(points_y, dtype=np.int64)
    init_u = np.asarray(initial_guess.disp_u, dtype=np.float64)
    init_v = np.asarray(initial_guess.disp_v, dtype=np.float64)

    # 결과 배열
    result_p = np.empty((n_points, n_params), dtype=np.float64)
    result_zncc = np.empty(n_points, dtype=np.float64)
    result_iter = np.empty(n_points, dtype=np.int32)
    result_conv = np.empty(n_points, dtype=np.bool_)
    result_fail = np.empty(n_points, dtype=np.int32)

    # 작업 버퍼 할당
    batch_bufs = allocate_batch_buffers(n_points, n_pixels, n_params)

    # prange 병렬 실행
    process_all_pois_parallel(
        ref_gray, grad_x, grad_y,
        coeffs, interpolation_order,
        pts_x, pts_y,
        init_u, init_v,
        xsi, eta,
        subset_size, max_iterations, convergence_threshold,
        shape_type,
        result_p, result_zncc, result_iter, result_conv, result_fail,
        batch_bufs['f'], batch_bufs['dfdx'], batch_bufs['dfdy'],
        batch_bufs['J'], batch_bufs['H'], batch_bufs['H_inv'],
        batch_bufs['p'], batch_bufs['xsi_w'], batch_bufs['eta_w'],
        batch_bufs['x_def'], batch_bufs['y_def'],
        batch_bufs['g'], batch_bufs['b'], batch_bufs['dp'], batch_bufs['p_new']
    )

    # 결과 배열 분해
    if shape_function == 'affine':
        disp_u = result_p[:, 0]
        disp_ux = result_p[:, 1]
        disp_uy = result_p[:, 2]
        disp_v = result_p[:, 3]
        disp_vx = result_p[:, 4]
        disp_vy = result_p[:, 5]
        disp_uxx = disp_uxy = disp_uyy = None
        disp_vxx = disp_vxy = disp_vyy = None
    else:
        disp_u = result_p[:, 0]
        disp_ux = result_p[:, 1]
        disp_uy = result_p[:, 2]
        disp_uxx = result_p[:, 3]
        disp_uxy = result_p[:, 4]
        disp_uyy = result_p[:, 5]
        disp_v = result_p[:, 6]
        disp_vx = result_p[:, 7]
        disp_vy = result_p[:, 8]
        disp_vxx = result_p[:, 9]
        disp_vxy = result_p[:, 10]
        disp_vyy = result_p[:, 11]

    # valid_mask 및 failure_reason 보정 (기존 로직과 동일)
    valid_mask = (result_zncc >= zncc_threshold)
    failure_reason = result_fail.copy()

    # valid인데 fail_code가 0이 아닌 경우 보정
    bad_valid = valid_mask & (failure_reason != ICGN_SUCCESS)
    valid_mask[bad_valid] = False

    # valid가 아닌데 success인 경우 -> LOW_ZNCC
    low_zncc = (~valid_mask) & (failure_reason == ICGN_SUCCESS)
    failure_reason[low_zncc] = ICGN_FAIL_LOW_ZNCC

    # 진행 콜백: 완료 알림
    if progress_callback:
        progress_callback(n_points, n_points)

    processing_time = time.time() - start_time

    n_valid = int(np.sum(valid_mask))
    n_conv = int(np.sum(result_conv))
    mean_zncc = float(np.mean(result_zncc[valid_mask])) if n_valid > 0 else 0.0
    _logger.info(f"IC-GN(Numba) 완료: {n_conv}/{n_points} 수렴 "
                 f"({n_conv/n_points*100:.1f}%), "
                 f"mean_zncc={mean_zncc:.4f}, {processing_time:.3f}s "
                 f"({processing_time/n_points*1000:.2f}ms/POI)")

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
        zncc_values=result_zncc,
        iterations=result_iter,
        converged=result_conv,
        valid_mask=valid_mask,
        failure_reason=failure_reason,
        subset_size=subset_size,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        processing_time=processing_time,
        shape_function=shape_function,
    )


# ===== 기존 ThreadPoolExecutor 경로 =====

def _compute_icgn_original(
    ref_image, def_image, initial_guess,
    subset_size, max_iterations, convergence_threshold,
    zncc_threshold, interpolation_order, shape_function,
    gaussian_blur, n_workers, progress_callback, ref_cache=None
) -> ICGNResult:
    """기존 ThreadPoolExecutor 기반 구현 (fallback)"""
    start_time = time.time()

    if shape_function not in ('affine', 'quadratic'):
        raise ValueError(
            f"shape_function must be 'affine' or 'quadratic', "
            f"got '{shape_function}'")

    n_params = get_num_params(shape_function)

    # 참조 전처리 (캐시 사용 가능 시 건너뜀)
    if ref_cache is not None:
        ref_gray = ref_cache['ref_gray']
        grad_x = ref_cache['grad_x']
        grad_y = ref_cache['grad_y']
        xsi = ref_cache['xsi']
        eta = ref_cache['eta']
    else:
        ref_gray = _to_gray(ref_image).astype(np.float64)
        if gaussian_blur is not None and gaussian_blur > 0:
            if gaussian_blur % 2 == 0:
                gaussian_blur += 1
            ref_gray = cv2.GaussianBlur(
                ref_gray, (gaussian_blur, gaussian_blur), 0)
        grad_x, grad_y = _compute_gradient(ref_gray)
        xsi, eta = generate_local_coordinates(subset_size)

    # 변형 이미지 전처리 (매 프레임)
    def_gray = _to_gray(def_image).astype(np.float64)
    if gaussian_blur is not None and gaussian_blur > 0:
        if gaussian_blur % 2 == 0:
            gaussian_blur += 1
        def_gray = cv2.GaussianBlur(
            def_gray, (gaussian_blur, gaussian_blur), 0)

    target_interp = create_interpolator(def_gray, order=interpolation_order)

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
    failure_reason = np.zeros(n_points, dtype=np.int32)

    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)

    if progress_callback:
        progress_callback(0, n_points)

    def process_poi(idx: int) -> Tuple[int, np.ndarray, float, int, bool, int]:
        px = points_x[idx]
        py = points_y[idx]

        ref_data = _extract_reference_subset(
            ref_gray, grad_x, grad_y, px, py, subset_size)

        if ref_data is None:
            return idx, np.zeros(n_params), 0.0, 0, False, ICGN_FAIL_FLAT_SUBSET

        f, dfdx, dfdy, f_mean, f_tilde = ref_data

        J = compute_steepest_descent(dfdx, dfdy, xsi, eta, shape_function)
        H = compute_hessian(J)

        try:
            H_inv = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            return idx, np.zeros(n_params), 0.0, 0, False, ICGN_FAIL_SINGULAR_HESSIAN

        p = get_initial_params(shape_function)
        if shape_function == 'affine':
            p[0] = float(initial_guess.disp_u[idx])
            p[3] = float(initial_guess.disp_v[idx])
        else:
            p[0] = float(initial_guess.disp_u[idx])
            p[6] = float(initial_guess.disp_v[idx])

        p_final, zncc, n_iter, conv, fail_code = _icgn_iterate(
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

        return idx, p_final, zncc, n_iter, conv, fail_code

    # 병렬 처리
    completed = 0
    update_interval = max(1, n_points // 50)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {
            executor.submit(process_poi, i): i for i in range(n_points)}

        for future in as_completed(futures):
            idx, p_final, zncc, n_iter, conv, fail_code = future.result()

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
            failure_reason[idx] = fail_code

            if valid_mask[idx] and fail_code != ICGN_SUCCESS:
                valid_mask[idx] = False
            if not valid_mask[idx] and fail_code == ICGN_SUCCESS:
                failure_reason[idx] = ICGN_FAIL_LOW_ZNCC

            completed += 1
            if progress_callback and completed % update_interval == 0:
                progress_callback(completed, n_points)

    if progress_callback:
        progress_callback(n_points, n_points)

    processing_time = time.time() - start_time

    n_valid_orig = int(np.sum(valid_mask))
    n_conv_orig = int(np.sum(converged))
    mean_zncc_orig = float(np.mean(zncc_values[valid_mask])) if n_valid_orig > 0 else 0.0
    _logger.info(f"IC-GN(ThreadPool) 완료: {n_conv_orig}/{n_points} 수렴 "
                 f"({n_conv_orig/n_points*100:.1f}%), "
                 f"mean_zncc={mean_zncc_orig:.4f}, {processing_time:.3f}s "
                 f"({processing_time/n_points*1000:.2f}ms/POI)")

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
        failure_reason=failure_reason,
        subset_size=subset_size,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        processing_time=processing_time,
        shape_function=shape_function,
    )


# ===== IC-GN 반복 (기존 경로에서 사용) =====
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
) -> Tuple[np.ndarray, float, int, bool, int]:
    """
    단일 POI에 대한 IC-GN 반복

    Returns:
        (p_final, zncc, n_iter, converged, fail_code)
        fail_code: 0=success, 1=low_zncc, 2=diverged,
                   3=out_of_bounds, 6=max_displacement, 7=flat_target
    """
    n_iter = 0
    conv = False
    zncc = 0.0
    fail_code = ICGN_SUCCESS

    if shape_function == 'affine':
        u_idx, v_idx = 0, 3
    else:
        u_idx, v_idx = 0, 6

    p_initial_u = p[u_idx]
    p_initial_v = p[v_idx]

    half = subset_size // 2
    reference_half = 10
    divergence_threshold = 1.0 * (half / reference_half)
    max_displacement_change = 5.0 * (half / reference_half)

    for iteration in range(max_iterations):
        n_iter = iteration + 1

        # Warp
        xsi_w, eta_w = warp(p, xsi, eta, shape_function)
        x_def = cx + xsi_w
        y_def = cy + eta_w

        # 경계 체크
        if not np.all(target_interp.is_inside(y_def, x_def)):
            fail_code = ICGN_FAIL_OUT_OF_BOUNDS
            conv = False
            break

        # 보간
        g = target_interp(y_def, x_def)

        # target subset이 평탄한 경우
        g_mean = np.mean(g)
        g_tilde = np.linalg.norm(g - g_mean)

        if g_tilde < 1e-10:
            fail_code = ICGN_FAIL_FLAT_TARGET
            break

        znssd = _compute_znssd(f, f_mean, f_tilde, g, g_mean, g_tilde)
        zncc = 1.0 - 0.5 * znssd

        if zncc < 0.5:
            fail_code = ICGN_FAIL_LOW_ZNCC
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
            fail_code = ICGN_SUCCESS
            break

        # 발산 판단
        if dp_norm > divergence_threshold:
            fail_code = ICGN_FAIL_DIVERGED
            conv = False
            break

        # Warp 업데이트
        p = update_warp_inverse_compositional(p, dp, shape_function)

        displacement_change_u = abs(p[u_idx] - p_initial_u)
        displacement_change_v = abs(p[v_idx] - p_initial_v)

        if (displacement_change_u > max_displacement_change
                or displacement_change_v > max_displacement_change):
            fail_code = ICGN_FAIL_MAX_DISPLACEMENT
            conv = False
            break

    return p, zncc, n_iter, conv, fail_code

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
        failure_reason=np.array([], dtype=np.int32),
        subset_size=subset_size,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        processing_time=0.0,
        shape_function=shape_function,
    )
