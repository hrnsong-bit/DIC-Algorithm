"""
IC-GN (Inverse Compositional Gauss-Newton) 최적화 모듈

Numba JIT + prange 병렬화로 실행됨.

References:
    - Pan, B., et al. Experimental Mechanics, 2013.
    - Pan, B., et al. Optics and Lasers in Engineering, 2013.
    - Jiang, Z., et al. Optics and Lasers in Engineering, 2014.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Callable
import time
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
from .shape_function import (
    generate_local_coordinates,
    get_num_params,
)
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

_logger = logging.getLogger(__name__)


# ===== 참조 이미지 전처리 캐시 =====

def prepare_ref_cache(
    ref_image: np.ndarray,
    subset_size: int = 21,
    interpolation_order: int = 5,
    shape_function: str = 'affine',
    gaussian_blur: Optional[int] = None,
) -> dict:
    """
    참조 이미지 전처리 결과를 캐시용 딕셔너리로 반환.

    배치 분석 시 매 프레임마다 반복되는 전처리를 1회로 줄임:
        - 그레이스케일 변환
        - Gaussian blur (선택)
        - Sobel gradient
        - B-spline 로컬 좌표

    Args:
        ref_image: 참조 이미지
        subset_size: 서브셋 크기
        interpolation_order: B-spline 차수
        shape_function: 'affine' 또는 'quadratic'
        gaussian_blur: Gaussian 블러 커널 크기

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

    xsi, eta = _numba_gen_coords(subset_size)

    return {
        'ref_gray': ref_gray,
        'grad_x': grad_x,
        'grad_y': grad_y,
        'subset_size': subset_size,
        'interpolation_order': interpolation_order,
        'shape_function': shape_function,
        'xsi': xsi,
        'eta': eta,
    }


# ===== 메인 함수 =====

def compute_icgn(
    ref_image: np.ndarray,
    def_image: np.ndarray,
    initial_guess: FFTCCResult,
    subset_size: int = 21,
    max_iterations: int = 50,
    convergence_threshold: float = 0.001,
    zncc_threshold: float = 0.85,
    interpolation_order: int = 5,
    shape_function: str = 'affine',
    gaussian_blur: Optional[int] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
    ref_cache: Optional[dict] = None,
) -> ICGNResult:
    """
    IC-GN 서브픽셀 최적화 (Numba JIT + prange)

    Args:
        ref_image: 참조 이미지
        def_image: 변형 이미지
        initial_guess: FFT-CC 초기 추정 결과
        subset_size: 서브셋 크기 (홀수)
        max_iterations: 최대 반복 횟수
        convergence_threshold: 수렴 임계값
        zncc_threshold: IC-GN 최종 품질 판정 ZNCC 임계값 (0.85 권장)
        interpolation_order: B-spline 보간 차수 (3 또는 5)
        shape_function: 형상 함수 ('affine' 또는 'quadratic')
        gaussian_blur: 가우시안 블러 커널 크기 (None이면 미적용)
        progress_callback: 진행 콜백 함수 (current, total)
        ref_cache: prepare_ref_cache()로 생성한 캐시 (배치 분석용)

    Returns:
        ICGNResult 객체
    """
    n_points = len(initial_guess.points_x)
    cached = "cached" if ref_cache else "fresh"
    _logger.info(
        f"IC-GN 시작: {n_points} POIs, subset={subset_size}, "
        f"order={interpolation_order}, shape={shape_function}, "
        f"zncc_thr={zncc_threshold}, ref={cached}"
    )

    return _compute_icgn_numba(
        ref_image, def_image, initial_guess,
        subset_size, max_iterations, convergence_threshold,
        zncc_threshold, interpolation_order, shape_function,
        gaussian_blur, progress_callback, ref_cache,
    )


# ===== Numba 경로 =====

def _compute_icgn_numba(
    ref_image, def_image, initial_guess,
    subset_size, max_iterations, convergence_threshold,
    zncc_threshold, interpolation_order, shape_function,
    gaussian_blur, progress_callback, ref_cache=None,
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
        grad_x   = ref_cache['grad_x']
        grad_y   = ref_cache['grad_y']
        xsi      = ref_cache['xsi']
        eta      = ref_cache['eta']
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

    coeffs = prefilter_image(def_gray, order=interpolation_order)

    points_x = initial_guess.points_x
    points_y = initial_guess.points_y
    n_points = len(points_x)

    if n_points == 0:
        return _empty_result(
            subset_size, max_iterations,
            convergence_threshold, shape_function)

    n_pixels = subset_size * subset_size

    # ── FFT-CC valid_mask 적용 ──────────────────────────────────────
    if initial_guess.valid_mask is not None:
        fft_valid = np.asarray(initial_guess.valid_mask, dtype=np.bool_)
    else:
        fft_valid = np.ones(n_points, dtype=np.bool_)

    valid_idx   = np.where(fft_valid)[0]
    invalid_idx = np.where(~fft_valid)[0]
    n_valid     = len(valid_idx)

    _logger.info(
        f"FFT-CC valid: {n_valid}/{n_points} "
        f"({n_valid/n_points*100:.1f}%) — "
        f"{len(invalid_idx)}개 POI IC-GN 스킵 (텍스처 없음)"
    )

    # 전체 크기 결과 배열 (FFT 실패분 포함)
    result_p    = np.zeros((n_points, n_params), dtype=np.float64)
    result_zncc = np.zeros(n_points,             dtype=np.float64)
    result_iter = np.zeros(n_points,             dtype=np.int32)
    result_conv = np.zeros(n_points,             dtype=np.bool_)
    result_fail = np.full(n_points, ICGN_FAIL_LOW_ZNCC, dtype=np.int32)

    if n_valid > 0:
        pts_x  = np.asarray(points_x, dtype=np.int64)[valid_idx]
        pts_y  = np.asarray(points_y, dtype=np.int64)[valid_idx]
        init_u = np.asarray(initial_guess.disp_u, dtype=np.float64)[valid_idx]
        init_v = np.asarray(initial_guess.disp_v, dtype=np.float64)[valid_idx]

        # valid POI 전용 결과 배열
        vp    = np.empty((n_valid, n_params), dtype=np.float64)
        vzncc = np.empty(n_valid,             dtype=np.float64)
        viter = np.empty(n_valid,             dtype=np.int32)
        vconv = np.empty(n_valid,             dtype=np.bool_)
        vfail = np.empty(n_valid,             dtype=np.int32)

        batch_bufs = allocate_batch_buffers(n_valid, n_pixels, n_params)

        if progress_callback:
            progress_callback(0, n_valid)

        process_all_pois_parallel(
            ref_gray, grad_x, grad_y,
            coeffs, interpolation_order,
            pts_x, pts_y,
            init_u, init_v,
            xsi, eta,
            subset_size, max_iterations, convergence_threshold,
            shape_type,
            vp, vzncc, viter, vconv, vfail,
            batch_bufs['f'],     batch_bufs['dfdx'],  batch_bufs['dfdy'],
            batch_bufs['J'],     batch_bufs['H'],     batch_bufs['H_inv'],
            batch_bufs['p'],     batch_bufs['xsi_w'], batch_bufs['eta_w'],
            batch_bufs['x_def'], batch_bufs['y_def'],
            batch_bufs['g'],     batch_bufs['b'],     batch_bufs['dp'],
            batch_bufs['p_new'],
        )

        # valid 결과를 전체 배열에 매핑
        result_p[valid_idx]    = vp
        result_zncc[valid_idx] = vzncc
        result_iter[valid_idx] = viter
        result_conv[valid_idx] = vconv
        result_fail[valid_idx] = vfail

        if progress_callback:
            progress_callback(n_valid, n_valid)

    # ── valid_mask 및 failure_reason 보정 ──────────────────────────
    icgn_valid = np.zeros(n_points, dtype=np.bool_)
    icgn_valid[valid_idx] = (result_zncc[valid_idx] >= zncc_threshold)

    # valid인데 fail_code가 success가 아닌 경우 보정
    bad_valid = icgn_valid & (result_fail != ICGN_SUCCESS)
    icgn_valid[bad_valid] = False

    # valid가 아닌데 success인 경우 → LOW_ZNCC
    low_zncc = (~icgn_valid) & (result_fail == ICGN_SUCCESS)
    result_fail[low_zncc] = ICGN_FAIL_LOW_ZNCC

    # FFT 실패분 failure_reason 명시
    result_fail[invalid_idx] = ICGN_FAIL_LOW_ZNCC

    processing_time = time.time() - start_time

    n_final_valid = int(np.sum(icgn_valid))
    n_conv        = int(np.sum(result_conv))
    mean_zncc     = (float(np.mean(result_zncc[icgn_valid]))
                     if n_final_valid > 0 else 0.0)
    _logger.info(
        f"IC-GN(Numba) 완료: {n_conv}/{n_valid} 수렴 "
        f"({n_conv/max(n_valid,1)*100:.1f}% of FFT-valid), "
        f"최종 valid: {n_final_valid}/{n_points}, "
        f"mean_zncc={mean_zncc:.4f}, {processing_time:.3f}s "
        f"({processing_time/max(n_valid,1)*1000:.2f}ms/POI)"
    )

    # 결과 배열 분해
    if shape_function == 'affine':
        disp_u   = result_p[:, 0]
        disp_ux  = result_p[:, 1]
        disp_uy  = result_p[:, 2]
        disp_v   = result_p[:, 3]
        disp_vx  = result_p[:, 4]
        disp_vy  = result_p[:, 5]
        disp_uxx = disp_uxy = disp_uyy = None
        disp_vxx = disp_vxy = disp_vyy = None
    else:
        disp_u   = result_p[:, 0]
        disp_ux  = result_p[:, 1]
        disp_uy  = result_p[:, 2]
        disp_uxx = result_p[:, 3]
        disp_uxy = result_p[:, 4]
        disp_uyy = result_p[:, 5]
        disp_v   = result_p[:, 6]
        disp_vx  = result_p[:, 7]
        disp_vy  = result_p[:, 8]
        disp_vxx = result_p[:, 9]
        disp_vxy = result_p[:, 10]
        disp_vyy = result_p[:, 11]

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
        valid_mask=icgn_valid,
        failure_reason=result_fail,
        subset_size=subset_size,
        max_iterations=max_iterations,
        convergence_threshold=convergence_threshold,
        processing_time=processing_time,
        shape_function=shape_function,
    )


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
    sssig.py의 compute_gradient와 동일한 설정.
    """
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5) / 32.0
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5) / 32.0
    return grad_x, grad_y


def _empty_result(
    subset_size: int,
    max_iterations: int,
    convergence_threshold: float,
    shape_function: str = 'affine',
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
