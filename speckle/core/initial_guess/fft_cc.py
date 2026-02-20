"""
FFT-CC (FFT-based Cross Correlation) 초기 추정 모듈

확장 탐색 범위 지원:
- ref_subset: subset_size × subset_size (원본)
- tar_subset: (subset_size + 2*search_range) × (subset_size + 2*search_range) (확장)
- ref를 tar 크기로 zero-pad 후 FFT 상호상관
- 탐색 범위 = ±search_range

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
import logging

_logger = logging.getLogger(__name__)

try:
    from scipy.fft import fft2 as scipy_fft2, ifft2 as scipy_ifft2
    HAS_SCIPY_FFT = True
except ImportError:
    HAS_SCIPY_FFT = False

from .results import MatchResult, FFTCCResult
from scipy.ndimage import distance_transform_edt
from speckle.core.masking import create_specimen_mask

_SEARCH_RANGE = 50

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

# 확장 타겟 서브셋 추출 함수
def _extract_subsets_batch_extended(image: np.ndarray,
                                    points_y: np.ndarray,
                                    points_x: np.ndarray,
                                    ext_half: int) -> np.ndarray:
    """
    확장된 타겟 서브셋 추출 → (n_points, ext_size, ext_size)
    ext_half = subset_half + search_range
    """
    ext_size = 2 * ext_half + 1
    n_points = len(points_y)

    offsets = np.arange(-ext_half, ext_half + 1)
    row_idx = points_y[:, None] + offsets[None, :]
    col_idx = points_x[:, None] + offsets[None, :]

    rows = row_idx[:, :, None].repeat(ext_size, axis=2)
    cols = col_idx[:, None, :].repeat(ext_size, axis=1)

    subsets = image[rows, cols]
    return subsets.astype(np.float64)

def _batch_fft_zncc(ref_subsets: np.ndarray,
                    tar_subsets: np.ndarray,
                    min_std: float,
                    n_workers: int = -1,
                    chunk_size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    배치 FFT 기반 ZNCC (확장 탐색 범위 지원)

    ref_subsets: (n, rs, rs) — 원본 크기
    tar_subsets: (n, ts, ts) — 확장 크기 (ts >= rs)

    ZNCC 정규화 증명:
        cc[dy,dx] = sum(r_zm * t_shifted)
        r_zm은 zero-mean이므로 sum(r_zm) = 0
        → sum(r_zm * t) = sum(r_zm * (t - t_mean)) = sum(r_zm * t_zm)
        이 성질은 zero-pad 후에도 유지됨 (0 추가는 합을 변경하지 않음)
        → ZNCC = sum(r_zm * t_zm) / (r_norm * t_local_norm) ∈ [-1, 1]

    수치 안정성:
        local_var = sum_sq/N - mean² 방식 (one-pass)은 catastrophic cancellation에
        취약할 수 있으나, 8/16-bit 이미지 + float64 연산(유효숫자 ~15자리)에서는
        실질적 문제 없음. np.maximum(local_var, 0.0)으로 음수 분산 방어 적용.

    텍스처 없음 처리:
        ref 또는 tar 서브셋의 픽셀 표준편차가 min_std 미만이면 ZNCC를 0.0으로
        강제 설정. min_std는 호출부(compute_fft_cc)에서 ref 이미지 전체 std의
        10%로 적응적으로 결정됨 (min_std = 0.1 * image_std).
        이를 통해 검은 영역에서 참조/변형 이미지가 동일한 카메라 노이즈 패턴을
        가져 ZNCC ≈ 1.0이 되는 오탐을 방지함.
    """
    n_points = ref_subsets.shape[0]
    ref_h, ref_w = ref_subsets.shape[1], ref_subsets.shape[2]
    tar_h, tar_w = tar_subsets.shape[1], tar_subsets.shape[2]
    ref_n_pixels = ref_h * ref_w

    fft_h, fft_w = tar_h, tar_w

    disp_u_all = np.zeros(n_points, dtype=np.int32)
    disp_v_all = np.zeros(n_points, dtype=np.int32)
    zncc_all   = np.zeros(n_points, dtype=np.float64)

    # 슬라이딩 윈도우 출력 크기
    out_h = tar_h - ref_h + 1
    out_w = tar_w - ref_w + 1

    # 벡터화된 박스 합을 위한 인덱스 사전 계산 (전 청크 공유)
    dy = np.arange(out_h)
    dx = np.arange(out_w)
    dy_grid, dx_grid = np.meshgrid(dy, dx, indexing='ij')
    r1 = dy_grid
    c1 = dx_grid
    r2 = dy_grid + ref_h
    c2 = dx_grid + ref_w

    for start in range(0, n_points, chunk_size):
        end     = min(start + chunk_size, n_points)
        n_chunk = end - start

        r = ref_subsets[start:end]
        t = tar_subsets[start:end]

        # === ref: zero-mean 및 norm ===
        r_mean = r.mean(axis=(1, 2), keepdims=True)
        r_zm   = r - r_mean
        r_norm = np.sqrt(np.sum(r_zm ** 2, axis=(1, 2)))  # (n_chunk,)

        # ref 텍스처 없음 판별
        # r_std: 서브셋 픽셀 표준편차 (0~255 스케일)
        # min_std = 0.1 * image_std로 이미지 통계 기반 적응적 결정
        r_std    = r_norm / np.sqrt(ref_n_pixels)          # (n_chunk,)
        zero_mask = r_std < min_std                        # ref 텍스처 없음 플래그

        r_norm_safe           = r_norm.copy()
        r_norm_safe[zero_mask] = 1.0                       # 0 나눗셈 방지

        # === tar: 벡터화된 적분 이미지 기반 local norm ===
        t_pad = np.zeros((n_chunk, tar_h + 1, tar_w + 1), dtype=np.float64)
        t_pad[:, 1:, 1:] = np.cumsum(np.cumsum(t, axis=1), axis=2)

        t_sq_pad = np.zeros((n_chunk, tar_h + 1, tar_w + 1), dtype=np.float64)
        t_sq_pad[:, 1:, 1:] = np.cumsum(np.cumsum(t ** 2, axis=1), axis=2)

        local_sum = (
            t_pad[:, r2, c2]
            - t_pad[:, r1, c2]
            - t_pad[:, r2, c1]
            + t_pad[:, r1, c1]
        )
        local_sum_sq = (
            t_sq_pad[:, r2, c2]
            - t_sq_pad[:, r1, c2]
            - t_sq_pad[:, r2, c1]
            + t_sq_pad[:, r1, c1]
        )

        local_mean = local_sum / ref_n_pixels
        local_var  = local_sum_sq / ref_n_pixels - local_mean ** 2
        local_var  = np.maximum(local_var, 0.0)
        t_local_norm = np.sqrt(local_var * ref_n_pixels)  # (n_chunk, out_h, out_w)

        # tar 텍스처 없음 판별 (슬라이딩 윈도우별 표준편차 기준)
        t_std     = t_local_norm / np.sqrt(ref_n_pixels)  # (n_chunk, out_h, out_w)
        low_t_mask = t_std < min_std                      # tar 텍스처 없음 플래그

        # ref 또는 tar 둘 중 하나라도 텍스처 없으면 ZNCC = 0
        texture_less = zero_mask[:, None, None] | low_t_mask  # (n_chunk, out_h, out_w)

        # === FFT cross-correlation ===
        r_padded = np.zeros((n_chunk, fft_h, fft_w), dtype=np.float64)
        r_padded[:, :ref_h, :ref_w] = r_zm

        if HAS_SCIPY_FFT:
            F      = scipy_fft2(r_padded, workers=n_workers)
            G      = scipy_fft2(t,        workers=n_workers)
            cc_raw = scipy_ifft2(np.conj(F) * G, workers=n_workers).real
        else:
            F      = np.fft.fft2(r_padded)
            G      = np.fft.fft2(t)
            cc_raw = np.fft.ifft2(np.conj(F) * G).real
        del F, G

        cc = cc_raw[:, :out_h, :out_w]
        del cc_raw

        # === ZNCC 정규화 ===
        denom = r_norm_safe[:, None, None] * t_local_norm
        denom = np.where(denom < 1e-10, 1.0, denom)

        zncc = cc / denom
        del cc

        # 텍스처 없음 영역 강제 0 처리
        zncc = np.where(texture_less, 0.0, zncc)

        # === 피크 찾기 ===
        zncc_flat      = zncc.reshape(n_chunk, -1)
        peak_flat_idx  = np.argmax(zncc_flat, axis=1)
        peak_scores    = zncc_flat[np.arange(n_chunk), peak_flat_idx]
        del zncc, zncc_flat

        # 피크 → 변위 변환
        peak_row = peak_flat_idx // out_w
        peak_col = peak_flat_idx % out_w

        search_y = (tar_h - ref_h) // 2
        search_x = (tar_w - ref_w) // 2

        dv = peak_row - search_y
        du = peak_col - search_x

        disp_u_all[start:end] = du
        disp_v_all[start:end] = dv
        zncc_all[start:end]   = peak_scores

    return disp_u_all, disp_v_all, zncc_all

# ===== 단일 POI ZNCC (하위 호환) =====

def _fft_zncc(template: np.ndarray, search_win: np.ndarray) -> Tuple[int, int, float]:
    """단일 POI용 FFT-ZNCC"""
    r = template[None, ...].astype(np.float64)
    t = search_win[None, ...].astype(np.float64)

    du, dv, zncc = _batch_fft_zncc(r, t, n_workers=1, chunk_size=1)
    return int(dv[0]), int(du[0]), float(zncc[0])

# ===== 메인 함수 =====

def compute_fft_cc(ref_image, def_image,
                   subset_size=21, spacing=10, search_range=None,
                   zncc_threshold=0.6, roi=None,
                   n_workers=None, progress_callback=None) -> FFTCCResult:
    """
    적응적 탐색 범위 FFT-CC (Adaptive Search Range FFT Cross-Correlation)

    시편 마스크의 distance transform을 이용하여 각 POI별 최적 search_range를
    자동 결정. 시편 경계 근처 POI는 축소된 범위로, 내부 POI는 전체 범위로 탐색.

    텍스처 없음 판별:
        ref 이미지 전체 표준편차(image_std)의 10%를 min_std로 사용.
        DIC 실험은 항상 적절한 조명 하에 수행되므로 저조도 예외는 고려하지 않음.
        → min_std = 0.1 * image_std
        → 서브셋 픽셀 표준편차 < min_std 이면 텍스처 없음으로 판별 → ZNCC = 0
    """
    start_time = time.time()

    if search_range is None:
        search_range = _SEARCH_RANGE

    _logger.info(f"FFT-CC 시작: subset={subset_size}, spacing={spacing}, "
                 f"search_range={search_range}, zncc_thr={zncc_threshold}")

    ref_gray = _to_gray(ref_image).astype(np.float32)
    def_gray = _to_gray(def_image).astype(np.float32)
    ref_h, ref_w = ref_gray.shape
    def_h, def_w = def_gray.shape

    # 적응적 텍스처 없음 임계값 (ref 이미지 전체 통계 기반, 1회 계산)
    image_std = float(np.std(ref_gray))
    min_std   = 0.1 * image_std
    _logger.debug(f"텍스처 임계값: image_std={image_std:.2f}, min_std={min_std:.2f}")

    points_y, points_x = _generate_poi_grid(
        ref_gray.shape, subset_size, spacing, roi=roi
    )

    n_points = len(points_y)
    if n_points == 0:
        return _empty_result(subset_size, search_range, spacing)

    if progress_callback:
        progress_callback(0, n_points)

    half = subset_size // 2

    # ref 경계 검사
    ref_valid = (
        (points_y - half >= 0) & (points_y + half < ref_h) &
        (points_x - half >= 0) & (points_x + half < ref_w)
    )

    # 이미지 경계 기반 적응적 search_range
    margin_top   = points_y
    margin_bot   = def_h - 1 - points_y
    margin_left  = points_x
    margin_right = def_w - 1 - points_x
    max_possible = np.minimum(
        np.minimum(margin_top, margin_bot),
        np.minimum(margin_left, margin_right)
    ) - half
    poi_search_range = np.clip(max_possible.astype(np.int32), 0, search_range)

    MIN_SR  = 3
    valid   = ref_valid & (poi_search_range >= MIN_SR)

    disp_u      = np.zeros(n_points, dtype=np.int32)
    disp_v      = np.zeros(n_points, dtype=np.int32)
    zncc_values = np.zeros(n_points, dtype=np.float64)

    valid_idx = np.where(valid)[0]

    if len(valid_idx) > 0:
        valid_sr  = poi_search_range[valid_idx]
        unique_sr = np.unique(valid_sr)

        fft_workers = -1 if HAS_SCIPY_FFT else 1

        for sr in unique_sr:
            sr         = int(sr)
            group_mask = valid_sr == sr
            group_idx  = valid_idx[group_mask]

            g_py      = points_y[group_idx]
            g_px      = points_x[group_idx]
            g_ext_half = half + sr

            def_valid = (
                (g_py - g_ext_half >= 0) & (g_py + g_ext_half < def_h) &
                (g_px - g_ext_half >= 0) & (g_px + g_ext_half < def_w)
            )

            def_valid_idx = np.where(def_valid)[0]
            if len(def_valid_idx) == 0:
                continue

            final_idx = group_idx[def_valid_idx]
            f_py      = points_y[final_idx]
            f_px      = points_x[final_idx]

            ref_subsets = _extract_subsets_batch(ref_gray, f_py, f_px, half)
            tar_subsets = _extract_subsets_batch_extended(def_gray, f_py, f_px, g_ext_half)

            du, dv, scores = _batch_fft_zncc(
                ref_subsets, tar_subsets,
                min_std=min_std,
                n_workers=fft_workers
            )

            disp_u[final_idx]      = du
            disp_v[final_idx]      = dv
            zncc_values[final_idx] = scores

    if progress_callback:
        progress_callback(n_points, n_points)

    valid_mask = zncc_values >= zncc_threshold

    # 홀 내부 POI 제거
    analyzable = poi_search_range >= MIN_SR
    if not np.all(analyzable):
        keep        = analyzable
        points_y    = points_y[keep]
        points_x    = points_x[keep]
        disp_u      = disp_u[keep]
        disp_v      = disp_v[keep]
        zncc_values = zncc_values[keep]
        valid_mask  = valid_mask[keep]

    invalid_points = _collect_invalid_points(
        points_y, points_x, disp_u, disp_v, zncc_values, valid_mask, zncc_threshold
    )

    processing_time = time.time() - start_time

    n_valid   = int(np.sum(valid_mask))
    n_total   = len(points_y)
    mean_zncc = float(np.mean(zncc_values[valid_mask])) if n_valid > 0 else 0.0
    _logger.info(f"FFT-CC 완료: {n_valid}/{n_total} POI 유효 "
                 f"({n_valid/n_total*100:.1f}%), "
                 f"mean_zncc={mean_zncc:.4f}, {processing_time:.3f}s")

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
                                 subset_size=21, spacing=10, search_range=None,
                                 zncc_threshold=0.6, roi=None,
                                 progress_callback=None, should_stop=None):
    """
    배치 FFT-CC — 참조 서브셋 캐시 + 적응적 탐색 범위

    텍스처 없음 판별:
        ref 이미지 기준 min_std를 1회 계산하여 모든 프레임 배치에 공유.
        배치 처리 특성상 ref가 고정되므로 min_std도 전 프레임 동일하게 적용.
        → min_std = 0.1 * image_std (ref 이미지 전체 std 기반)
    """
    results = {}
    if not def_file_paths:
        return results

    if search_range is None:
        search_range = _SEARCH_RANGE

    ref_gray = _to_gray(ref_image).astype(np.float32)
    ref_h, ref_w = ref_gray.shape

    # 적응적 텍스처 없음 임계값 (ref 이미지 기준, 전 프레임 공유)
    image_std = float(np.std(ref_gray))
    min_std   = 0.1 * image_std
    _logger.debug(f"텍스처 임계값(배치): image_std={image_std:.2f}, min_std={min_std:.2f}")

    points_y, points_x = _generate_poi_grid(
        ref_gray.shape, subset_size, spacing, roi=roi
    )
    n_points = len(points_y)
    if n_points == 0:
        return results

    half = subset_size // 2

    # ref 경계 검사
    ref_valid = (
        (points_y - half >= 0) & (points_y + half < ref_h) &
        (points_x - half >= 0) & (points_x + half < ref_w)
    )

    # 이미지 경계 기반 적응적 search_range
    margin_top   = points_y
    margin_bot   = ref_h - 1 - points_y
    margin_left  = points_x
    margin_right = ref_w - 1 - points_x
    max_possible = np.minimum(
        np.minimum(margin_top, margin_bot),
        np.minimum(margin_left, margin_right)
    ) - half
    poi_search_range = np.clip(max_possible.astype(np.int32), 0, search_range)

    MIN_SR    = 3
    valid     = ref_valid & (poi_search_range >= MIN_SR)
    valid_idx = np.where(valid)[0]

    analyzable = poi_search_range >= MIN_SR

    # search_range별 그룹 사전 구성 + ref 서브셋 캐시
    valid_sr  = poi_search_range[valid_idx]
    unique_sr = np.unique(valid_sr)

    ref_subsets_cache = {}
    for sr in unique_sr:
        sr         = int(sr)
        group_mask = valid_sr == sr
        group_idx  = valid_idx[group_mask]
        g_py       = points_y[group_idx]
        g_px       = points_x[group_idx]
        ref_subsets_cache[sr] = {
            'group_idx':   group_idx,
            'g_py':        g_py,
            'g_px':        g_px,
            'ref_subsets': _extract_subsets_batch(ref_gray, g_py, g_px, half),
            'ext_half':    half + sr
        }

    fft_workers = -1 if HAS_SCIPY_FFT else 1
    total_files = len(def_file_paths)

    for file_idx, def_path in enumerate(def_file_paths):
        if should_stop and should_stop():
            break

        filename        = def_path.name
        file_start_time = time.time()

        if progress_callback:
            progress_callback(file_idx, total_files, filename)

        def_image = get_image_func(def_path)
        if def_image is None:
            continue

        def_gray       = _to_gray(def_image).astype(np.float32)
        def_h, def_w   = def_gray.shape

        disp_u      = np.zeros(n_points, dtype=np.int32)
        disp_v      = np.zeros(n_points, dtype=np.int32)
        zncc_values = np.zeros(n_points, dtype=np.float64)

        for sr, cache in ref_subsets_cache.items():
            group_idx   = cache['group_idx']
            g_py        = cache['g_py']
            g_px        = cache['g_px']
            ref_subsets = cache['ref_subsets']
            g_ext_half  = cache['ext_half']

            def_valid = (
                (g_py - g_ext_half >= 0) & (g_py + g_ext_half < def_h) &
                (g_px - g_ext_half >= 0) & (g_px + g_ext_half < def_w)
            )

            def_valid_idx = np.where(def_valid)[0]
            if len(def_valid_idx) == 0:
                continue

            final_idx = group_idx[def_valid_idx]
            f_py      = points_y[final_idx]
            f_px      = points_x[final_idx]

            tar_subsets = _extract_subsets_batch_extended(def_gray, f_py, f_px, g_ext_half)

            du, dv, scores = _batch_fft_zncc(
                ref_subsets[def_valid_idx], tar_subsets,
                min_std=min_std,
                n_workers=fft_workers
            )

            disp_u[final_idx]      = du
            disp_v[final_idx]      = dv
            zncc_values[final_idx] = scores

        valid_mask = zncc_values >= zncc_threshold

        # 홀 내부 POI 제거
        if not np.all(analyzable):
            keep          = analyzable
            r_points_y    = points_y[keep]
            r_points_x    = points_x[keep]
            r_disp_u      = disp_u[keep]
            r_disp_v      = disp_v[keep]
            r_zncc_values = zncc_values[keep]
            r_valid_mask  = valid_mask[keep]
        else:
            r_points_y    = points_y
            r_points_x    = points_x
            r_disp_u      = disp_u
            r_disp_v      = disp_v
            r_zncc_values = zncc_values
            r_valid_mask  = valid_mask

        invalid_points = _collect_invalid_points(
            r_points_y, r_points_x, r_disp_u, r_disp_v,
            r_zncc_values, r_valid_mask, zncc_threshold
        )

        results[filename] = FFTCCResult(
            points_y=r_points_y.copy(),
            points_x=r_points_x.copy(),
            disp_u=r_disp_u.copy(),
            disp_v=r_disp_v.copy(),
            zncc_values=r_zncc_values.copy(),
            valid_mask=r_valid_mask.copy(),
            invalid_points=invalid_points,
            subset_size=subset_size,
            search_range=search_range,
            spacing=spacing,
            processing_time=time.time() - file_start_time
        )

    if progress_callback:
        progress_callback(total_files, total_files, "완료")

    return results

# ===== 유틸리티 함수 =====

def _to_gray(img: np.ndarray) -> np.ndarray:
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
    dummy = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    compute_fft_cc(dummy, dummy, subset_size=21, spacing=30, search_range=20)