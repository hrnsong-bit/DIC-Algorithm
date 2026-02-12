"""
FFT-CC (FFT-based Cross Correlation) 초기 추정 모듈

고속 정수 픽셀 변위 추정을 위한 FFT 기반 정규화 상호상관

OpenCorr 방식: 동일 크기 ref/tar 서브셋, search_range 없음
- ref_subset과 tar_subset을 동일 위치, 동일 크기로 추출
- 양쪽 모두 zero-mean 후 FFT 상호상관
- ZNCC = max(cc) / (ref_norm * tar_norm * N)
- 탐색 범위 = ±(subset_size // 2)

References:
    - Jiang, Z., et al. "Path-independent digital image correlation with
      high accuracy, speed and robustness." Optics and Lasers in Engineering, 2015.
    - Pan, B., et al. "Fast, robust and accurate DIC calculation without
      redundant computations." Experimental Mechanics, 2013.
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Callable, List, Dict
from pathlib import Path
import os
import time

try:
    from scipy.fft import fft2 as scipy_fft2, ifft2 as scipy_ifft2
    HAS_SCIPY_FFT = True
except ImportError:
    HAS_SCIPY_FFT = False

from .results import MatchResult, FFTCCResult


# ===== 배치 서브셋 추출 =====

def _extract_subsets_batch(image: np.ndarray,
                           points_y: np.ndarray,
                           points_x: np.ndarray,
                           half: int) -> np.ndarray:
    """
    모든 POI에서 서브셋을 일괄 추출 → (n_points, subset_size, subset_size)
    """
    subset_size = 2 * half + 1
    n_points = len(points_y)

    offsets = np.arange(-half, half + 1)
    row_idx = points_y[:, None] + offsets[None, :]
    col_idx = points_x[:, None] + offsets[None, :]

    rows = row_idx[:, :, None].repeat(subset_size, axis=2)
    cols = col_idx[:, None, :].repeat(subset_size, axis=1)

    subsets = image[rows, cols]
    return subsets.astype(np.float64)


# ===== 배치 FFT-ZNCC (OpenCorr 방식) =====

def _batch_fft_zncc(ref_subsets: np.ndarray,
                    tar_subsets: np.ndarray,
                    n_workers: int = -1,
                    chunk_size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    배치 FFT 기반 ZNCC (OpenCorr 방식, 동일 크기 서브셋)

    ref와 tar 모두 zero-mean → FFT → 상호상관 → ZNCC
    적분 이미지 불필요 (전체 서브셋 정규화)

    Args:
        ref_subsets: (n_points, ss, ss) — 참조 서브셋
        tar_subsets: (n_points, ss, ss) — 타겟 서브셋 (동일 크기)
        n_workers: scipy FFT worker 수 (-1 = 모든 코어)
        chunk_size: 한 번에 처리할 POI 수 (메모리 제어)

    Returns:
        disp_u, disp_v, zncc_scores: 각 (n_points,)
    """
    n_points, ss_h, ss_w = ref_subsets.shape
    half_y = ss_h // 2
    half_x = ss_w // 2
    subset_size = ss_h * ss_w

    disp_u_all = np.zeros(n_points, dtype=np.int32)
    disp_v_all = np.zeros(n_points, dtype=np.int32)
    zncc_all = np.zeros(n_points, dtype=np.float64)

    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        n_chunk = end - start

        r = ref_subsets[start:end]
        t = tar_subsets[start:end]

        # Zero-mean
        r_mean = r.mean(axis=(1, 2), keepdims=True)
        t_mean = t.mean(axis=(1, 2), keepdims=True)
        r_zm = r - r_mean
        t_zm = t - t_mean

        # Norm
        r_norm = np.sqrt(np.sum(r_zm ** 2, axis=(1, 2)))
        t_norm = np.sqrt(np.sum(t_zm ** 2, axis=(1, 2)))

        # 무효 처리 (균일 영역)
        zero_mask = (r_norm < 1e-10) | (t_norm < 1e-10)
        r_norm[zero_mask] = 1.0
        t_norm[zero_mask] = 1.0

        # 배치 FFT (3회)
        if HAS_SCIPY_FFT:
            F = scipy_fft2(r_zm, workers=n_workers)
            G = scipy_fft2(t_zm, workers=n_workers)
            cc = scipy_ifft2(np.conj(F) * G, workers=n_workers).real
        else:
            F = np.fft.fft2(r_zm)
            G = np.fft.fft2(t_zm)
            cc = np.fft.ifft2(np.conj(F) * G).real
        del F, G

        # ZNCC = cc / (r_norm * t_norm * N)
        denom = (r_norm * t_norm * subset_size)[:, None, None]
        zncc = cc / (denom + 1e-10)
        del cc

        # 피크 찾기
        zncc_flat = zncc.reshape(n_chunk, -1)
        peak_flat_idx = np.argmax(zncc_flat, axis=1)
        peak_scores = zncc_flat[np.arange(n_chunk), peak_flat_idx]
        peak_scores[zero_mask] = 0.0
        del zncc, zncc_flat

        # 피크 → 변위 (wrap-around 보정)
        peak_row = peak_flat_idx // ss_w
        peak_col = peak_flat_idx % ss_w

        # 순환 FFT 보정: 피크가 절반을 넘으면 음수 변위
        dv = np.where(peak_row > half_y, peak_row - ss_h, peak_row)
        du = np.where(peak_col > half_x, peak_col - ss_w, peak_col)

        disp_u_all[start:end] = du
        disp_v_all[start:end] = dv
        zncc_all[start:end] = peak_scores

    return disp_u_all, disp_v_all, zncc_all


# ===== 단일 POI ZNCC (하위 호환) =====

def _fft_zncc(template: np.ndarray, search_win: np.ndarray) -> Tuple[int, int, float]:
    """단일 POI용 FFT-ZNCC (OpenCorr 방식, 동일 크기)"""
    r = template[None, ...].astype(np.float64)
    t = search_win[None, ...].astype(np.float64)

    du, dv, zncc = _batch_fft_zncc(r, t, n_workers=1, chunk_size=1)
    return int(dv[0]), int(du[0]), float(zncc[0])


# ===== 메인 함수 =====

def compute_fft_cc(ref_image, def_image,
                   subset_size=21, spacing=10, search_range=50,
                   zncc_threshold=0.6, roi=None,
                   n_workers=None, progress_callback=None) -> FFTCCResult:
    """
    벡터화된 FFT-CC (OpenCorr 방식)

    search_range 파라미터는 호환성을 위해 유지되지만 내부에서 사용되지 않음.
    탐색 범위는 subset_size에 의해 결정됨 (±subset_size//2).
    """
    start_time = time.time()

    ref_gray = _to_gray(ref_image).astype(np.float32)
    def_gray = _to_gray(def_image).astype(np.float32)

    ref_h, ref_w = ref_gray.shape
    def_h, def_w = def_gray.shape

    points_y, points_x = _generate_poi_grid(
        ref_gray.shape, subset_size, spacing, roi=roi
    )

    n_points = len(points_y)
    if n_points == 0:
        return _empty_result(subset_size, search_range, spacing)

    if progress_callback:
        progress_callback(0, n_points)

    half = subset_size // 2

    # 경계 검사: ref와 tar 모두에서 서브셋 추출 가능한 POI만 유지
    valid = (
        (points_y - half >= 0) & (points_y + half < ref_h) &
        (points_x - half >= 0) & (points_x + half < ref_w) &
        (points_y - half >= 0) & (points_y + half < def_h) &
        (points_x - half >= 0) & (points_x + half < def_w)
    )

    disp_u = np.zeros(n_points, dtype=np.int32)
    disp_v = np.zeros(n_points, dtype=np.int32)
    zncc_values = np.zeros(n_points, dtype=np.float64)

    valid_idx = np.where(valid)[0]

    if len(valid_idx) > 0:
        v_py = points_y[valid_idx]
        v_px = points_x[valid_idx]

        # 배치 서브셋 추출 (동일 크기)
        ref_subsets = _extract_subsets_batch(ref_gray, v_py, v_px, half)
        tar_subsets = _extract_subsets_batch(def_gray, v_py, v_px, half)

        # 배치 FFT-ZNCC
        fft_workers = -1 if HAS_SCIPY_FFT else 1
        du, dv, scores = _batch_fft_zncc(
            ref_subsets, tar_subsets,
            n_workers=fft_workers
        )

        disp_u[valid_idx] = du
        disp_v[valid_idx] = dv
        zncc_values[valid_idx] = scores

    if progress_callback:
        progress_callback(n_points, n_points)

    valid_mask = zncc_values >= zncc_threshold
    invalid_points = _collect_invalid_points(
        points_y, points_x, disp_u, disp_v, zncc_values, valid_mask, zncc_threshold
    )

    processing_time = time.time() - start_time

    return FFTCCResult(
        points_y=points_y,
        points_x=points_x,
        disp_u=disp_u,
        disp_v=disp_v,
        zncc_values=zncc_values,
        valid_mask=valid_mask,
        invalid_points=invalid_points,
        subset_size=subset_size,
        search_range=search_range,
        spacing=spacing,
        processing_time=processing_time
    )


def compute_fft_cc_batch_cached(ref_image, def_file_paths, get_image_func,
                                 subset_size=21, spacing=10, search_range=50,
                                 zncc_threshold=0.6, roi=None,
                                 progress_callback=None, should_stop=None):
    """
    배치 처리 — 참조 서브셋 캐시, OpenCorr 방식
    """
    results = {}
    if not def_file_paths:
        return results

    ref_gray = _to_gray(ref_image).astype(np.float32)
    ref_h, ref_w = ref_gray.shape

    points_y, points_x = _generate_poi_grid(
        ref_gray.shape, subset_size, spacing, roi=roi
    )
    n_points = len(points_y)
    if n_points == 0:
        return results

    half = subset_size // 2

    # 참조 서브셋에 대한 경계 검사
    ref_valid = (
        (points_y - half >= 0) & (points_y + half < ref_h) &
        (points_x - half >= 0) & (points_x + half < ref_w)
    )

    # 참조 서브셋 사전 추출 (1회, 전체 배치에서 재사용)
    ref_valid_idx = np.where(ref_valid)[0]
    ref_subsets = None
    if len(ref_valid_idx) > 0:
        ref_subsets = _extract_subsets_batch(
            ref_gray, points_y[ref_valid_idx], points_x[ref_valid_idx], half
        )

    fft_workers = -1 if HAS_SCIPY_FFT else 1
    total_files = len(def_file_paths)

    for file_idx, def_path in enumerate(def_file_paths):
        if should_stop and should_stop():
            break

        filename = def_path.name
        start_time = time.time()

        if progress_callback:
            progress_callback(file_idx, total_files, filename)

        def_image = get_image_func(def_path)
        if def_image is None:
            continue

        def_gray = _to_gray(def_image).astype(np.float32)
        def_h, def_w = def_gray.shape

        disp_u = np.zeros(n_points, dtype=np.int32)
        disp_v = np.zeros(n_points, dtype=np.int32)
        zncc_values = np.zeros(n_points, dtype=np.float64)

        if ref_subsets is not None and len(ref_valid_idx) > 0:
            # 타겟 경계 검사
            tar_valid = (
                (points_y[ref_valid_idx] - half >= 0) &
                (points_y[ref_valid_idx] + half < def_h) &
                (points_x[ref_valid_idx] - half >= 0) &
                (points_x[ref_valid_idx] + half < def_w)
            )

            both_valid = np.where(tar_valid)[0]

            if len(both_valid) > 0:
                global_idx = ref_valid_idx[both_valid]
                v_py = points_y[global_idx]
                v_px = points_x[global_idx]

                tar_subsets = _extract_subsets_batch(def_gray, v_py, v_px, half)

                du, dv, scores = _batch_fft_zncc(
                    ref_subsets[both_valid], tar_subsets,
                    n_workers=fft_workers
                )

                disp_u[global_idx] = du
                disp_v[global_idx] = dv
                zncc_values[global_idx] = scores

        valid_mask = zncc_values >= zncc_threshold
        invalid_points = _collect_invalid_points(
            points_y, points_x, disp_u, disp_v,
            zncc_values, valid_mask, zncc_threshold
        )

        results[filename] = FFTCCResult(
            points_y=points_y.copy(),
            points_x=points_x.copy(),
            disp_u=disp_u.copy(),
            disp_v=disp_v.copy(),
            zncc_values=zncc_values.copy(),
            valid_mask=valid_mask.copy(),
            invalid_points=invalid_points,
            subset_size=subset_size,
            search_range=search_range,
            spacing=spacing,
            processing_time=time.time() - start_time
        )

    if progress_callback:
        progress_callback(total_files, total_files, "완료")

    return results


# ===== 유틸리티 함수 =====

def _to_gray(img: np.ndarray) -> np.ndarray:
    """그레이스케일 변환"""
    if img is None:
        raise ValueError("이미지가 None입니다")
    if len(img.shape) == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _generate_poi_grid(image_shape: Tuple[int, int],
                       subset_size: int,
                       spacing: int,
                       roi: Optional[Tuple[int, int, int, int]] = None
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """POI 그리드 생성 — 항상 글로벌 좌표"""
    h, w = image_shape
    half = subset_size // 2
    margin = half + 1

    if roi is not None:
        rx, ry, rw, rh = roi
        y_start = max(margin, ry)
        y_end   = min(h - margin, ry + rh)
        x_start = max(margin, rx)
        x_end   = min(w - margin, rx + rw)
    else:
        y_start, y_end = margin, h - margin
        x_start, x_end = margin, w - margin

    y_coords = np.arange(y_start, y_end, spacing)
    x_coords = np.arange(x_start, x_end, spacing)

    if len(y_coords) == 0 or len(x_coords) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    return yy.ravel().astype(np.int64), xx.ravel().astype(np.int64)


def _collect_invalid_points(points_y, points_x, disp_u, disp_v,
                            zncc_values, valid_mask, threshold):
    """불량 포인트 수집"""
    invalid_indices = np.where(~valid_mask)[0]
    invalid_points = []
    for idx in invalid_indices:
        flag = 'low_zncc' if zncc_values[idx] < threshold else 'unknown'
        invalid_points.append(MatchResult(
            ref_y=int(points_y[idx]),
            ref_x=int(points_x[idx]),
            disp_u=int(disp_u[idx]),
            disp_v=int(disp_v[idx]),
            zncc=float(zncc_values[idx]),
            valid=False,
            flag=flag
        ))
    return invalid_points


def _empty_result(subset_size, search_range, spacing):
    """빈 결과"""
    return FFTCCResult(
        points_y=np.array([], dtype=np.int64),
        points_x=np.array([], dtype=np.int64),
        disp_u=np.array([], dtype=np.int32),
        disp_v=np.array([], dtype=np.int32),
        zncc_values=np.array([], dtype=np.float64),
        valid_mask=np.array([], dtype=bool),
        invalid_points=[],
        subset_size=subset_size,
        search_range=search_range,
        spacing=spacing
    )


def warmup_fft_cc():
    """워밍업"""
    dummy = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    compute_fft_cc(dummy, dummy, subset_size=21, spacing=30, search_range=20)
