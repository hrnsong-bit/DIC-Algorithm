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

try:
    from scipy.fft import fft2 as scipy_fft2, ifft2 as scipy_ifft2
    HAS_SCIPY_FFT = True
except ImportError:
    HAS_SCIPY_FFT = False

from .results import MatchResult, FFTCCResult


# ★ CHANGED: 내부 고정 탐색 범위 (UI에서 제어하지 않음)
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


# ★ CHANGED: 확장 타겟 서브셋 추출 함수 추가
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

# ★ 수정된 _batch_fft_zncc
def _batch_fft_zncc(ref_subsets: np.ndarray,
                    tar_subsets: np.ndarray,
                    n_workers: int = -1,
                    chunk_size: int = 256) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    배치 FFT 기반 ZNCC (확장 탐색 범위 지원)

    ref_subsets: (n, rs, rs) — 원본 크기
    tar_subsets: (n, ts, ts) — 확장 크기 (ts >= rs)
    """
    n_points = ref_subsets.shape[0]
    ref_h, ref_w = ref_subsets.shape[1], ref_subsets.shape[2]
    tar_h, tar_w = tar_subsets.shape[1], tar_subsets.shape[2]
    ref_n_pixels = ref_h * ref_w

    # FFT 크기 = tar 크기 (ref를 zero-pad)
    fft_h, fft_w = tar_h, tar_w

    disp_u_all = np.zeros(n_points, dtype=np.int32)
    disp_v_all = np.zeros(n_points, dtype=np.int32)
    zncc_all = np.zeros(n_points, dtype=np.float64)

    for start in range(0, n_points, chunk_size):
        end = min(start + chunk_size, n_points)
        n_chunk = end - start

        r = ref_subsets[start:end]  # (n_chunk, rs, rs)
        t = tar_subsets[start:end]  # (n_chunk, ts, ts)

        # === ref: zero-mean (ref 자체 픽셀 기준) ===
        r_mean = r.mean(axis=(1, 2), keepdims=True)
        r_zm = r - r_mean
        r_norm = np.sqrt(np.sum(r_zm ** 2, axis=(1, 2)))

        # === tar: local mean/norm 계산 (적분 이미지 사용) ===
        # ref 크기 윈도우가 tar 위를 슬라이딩할 때 각 위치의 local sum, sum_sq
        # 적분 이미지로 O(1) per position
        t_cumsum = np.cumsum(np.cumsum(t, axis=1), axis=2)
        t_sq_cumsum = np.cumsum(np.cumsum(t ** 2, axis=1), axis=2)

        # 슬라이딩 윈도우 합 계산
        # 출력 크기 = (tar_h - ref_h + 1, tar_w - ref_w + 1)
        out_h = tar_h - ref_h + 1
        out_w = tar_w - ref_w + 1

        def _box_sum(cumsum, r1, c1, r2, c2):
            """적분 이미지에서 박스 합 (0-indexed, inclusive r2,c2)"""
            s = cumsum[:, r2, c2]
            if r1 > 0:
                s = s - cumsum[:, r1 - 1, c2]
            if c1 > 0:
                s = s - cumsum[:, r2, c1 - 1]
            if r1 > 0 and c1 > 0:
                s = s + cumsum[:, r1 - 1, c1 - 1]
            return s

        # 각 슬라이딩 위치의 local sum, local sum_sq
        local_sum = np.zeros((n_chunk, out_h, out_w), dtype=np.float64)
        local_sum_sq = np.zeros((n_chunk, out_h, out_w), dtype=np.float64)

        for dy in range(out_h):
            for dx in range(out_w):
                r2_y = dy + ref_h - 1
                r2_x = dx + ref_w - 1
                local_sum[:, dy, dx] = _box_sum(t_cumsum, dy, dx, r2_y, r2_x)
                local_sum_sq[:, dy, dx] = _box_sum(t_sq_cumsum, dy, dx, r2_y, r2_x)

        local_mean = local_sum / ref_n_pixels
        local_var = local_sum_sq / ref_n_pixels - local_mean ** 2
        local_var = np.maximum(local_var, 0.0)
        t_local_norm = np.sqrt(local_var * ref_n_pixels)  # = sqrt(sum((t-t_mean)^2))

        # === FFT cross-correlation ===
        # ref를 tar 크기로 zero-pad
        r_padded = np.zeros((n_chunk, fft_h, fft_w), dtype=np.float64)
        r_padded[:, :ref_h, :ref_w] = r_zm

        if HAS_SCIPY_FFT:
            F = scipy_fft2(r_padded, workers=n_workers)
            G = scipy_fft2(t, workers=n_workers)  # t 원본 (mean 빼지 않음)
            cc_raw = scipy_ifft2(np.conj(F) * G, workers=n_workers).real
        else:
            F = np.fft.fft2(r_padded)
            G = np.fft.fft2(t)
            cc_raw = np.fft.ifft2(np.conj(F) * G).real
        del F, G

        # cc_raw에서 유효 영역만 추출 (out_h × out_w)
        cc = cc_raw[:, :out_h, :out_w]

        # ZNCC 보정: cc = sum(r_zm * t) 이므로
        # sum(r_zm * (t - t_mean)) = sum(r_zm * t) - t_mean * sum(r_zm)
        # sum(r_zm) = 0 이므로 cc = sum(r_zm * t_zm) 그대로!
        # → ZNCC = cc / (r_norm * t_local_norm)

        # 무효 처리
        zero_mask = (r_norm < 1e-10)
        r_norm[zero_mask] = 1.0

        denom = r_norm[:, None, None] * t_local_norm
        denom = np.where(denom < 1e-10, 1.0, denom)

        zncc = cc / denom
        del cc, cc_raw

        # 피크 찾기
        zncc_flat = zncc.reshape(n_chunk, -1)
        peak_flat_idx = np.argmax(zncc_flat, axis=1)
        peak_scores = zncc_flat[np.arange(n_chunk), peak_flat_idx]
        peak_scores[zero_mask] = 0.0
        del zncc, zncc_flat

        # 피크 → 변위
        # (0,0) = ref가 tar 좌상단과 일치 = 변위 -(search_range)
        # 중앙 = 변위 0
        peak_row = peak_flat_idx // out_w
        peak_col = peak_flat_idx % out_w

        search_y = (tar_h - ref_h) // 2  # = search_range
        search_x = (tar_w - ref_w) // 2

        dv = peak_row - search_y  # 변위 = 피크위치 - 중앙오프셋
        du = peak_col - search_x

        disp_u_all[start:end] = du
        disp_v_all[start:end] = dv
        zncc_all[start:end] = peak_scores

    return disp_u_all, disp_v_all, zncc_all

# ===== 단일 POI ZNCC (하위 호환) =====

def _fft_zncc(template: np.ndarray, search_win: np.ndarray) -> Tuple[int, int, float]:
    """단일 POI용 FFT-ZNCC"""
    r = template[None, ...].astype(np.float64)
    t = search_win[None, ...].astype(np.float64)

    du, dv, zncc = _batch_fft_zncc(r, t, n_workers=1, chunk_size=1)
    return int(dv[0]), int(du[0]), float(zncc[0])


# ===== 메인 함수 =====

# ★ CHANGED: search_range 기본값 _SEARCH_RANGE, 실제 사용
def compute_fft_cc(ref_image, def_image,
                   subset_size=21, spacing=10, search_range=None,
                   zncc_threshold=0.6, roi=None,
                   n_workers=None, progress_callback=None) -> FFTCCResult:
    """
    확장 탐색 범위 FFT-CC

    search_range: 탐색 범위 (±search_range 픽셀). None이면 내부 기본값 사용.
    """
    start_time = time.time()

    # ★ CHANGED: search_range 결정
    if search_range is None:
        search_range = _SEARCH_RANGE

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
    ext_half = half + search_range  # ★ CHANGED: 확장 반경

    # ★ CHANGED: 경계 검사 — ref는 half, tar(def)는 ext_half
    valid = (
        (points_y - half >= 0) & (points_y + half < ref_h) &
        (points_x - half >= 0) & (points_x + half < ref_w) &
        (points_y - ext_half >= 0) & (points_y + ext_half < def_h) &
        (points_x - ext_half >= 0) & (points_x + ext_half < def_w)
    )

    disp_u = np.zeros(n_points, dtype=np.int32)
    disp_v = np.zeros(n_points, dtype=np.int32)
    zncc_values = np.zeros(n_points, dtype=np.float64)

    valid_idx = np.where(valid)[0]

    if len(valid_idx) > 0:
        v_py = points_y[valid_idx]
        v_px = points_x[valid_idx]

        # ★ CHANGED: ref는 원본 크기, tar는 확장 크기
        ref_subsets = _extract_subsets_batch(ref_gray, v_py, v_px, half)
        tar_subsets = _extract_subsets_batch_extended(def_gray, v_py, v_px, ext_half)

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


# ★ CHANGED: batch_cached도 동일하게 확장
def compute_fft_cc_batch_cached(ref_image, def_file_paths, get_image_func,
                                 subset_size=21, spacing=10, search_range=None,
                                 zncc_threshold=0.6, roi=None,
                                 progress_callback=None, should_stop=None):
    """배치 처리 — 참조 서브셋 캐시, 확장 탐색 범위"""
    results = {}
    if not def_file_paths:
        return results

    if search_range is None:
        search_range = _SEARCH_RANGE

    ref_gray = _to_gray(ref_image).astype(np.float32)
    ref_h, ref_w = ref_gray.shape

    points_y, points_x = _generate_poi_grid(
        ref_gray.shape, subset_size, spacing, roi=roi
    )
    n_points = len(points_y)
    if n_points == 0:
        return results

    half = subset_size // 2
    ext_half = half + search_range  # ★ CHANGED

    # 참조 서브셋 경계 검사 (원본 크기)
    ref_valid = (
        (points_y - half >= 0) & (points_y + half < ref_h) &
        (points_x - half >= 0) & (points_x + half < ref_w)
    )

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
            # ★ CHANGED: 타겟 경계 검사 — ext_half 사용
            tar_valid = (
                (points_y[ref_valid_idx] - ext_half >= 0) &
                (points_y[ref_valid_idx] + ext_half < def_h) &
                (points_x[ref_valid_idx] - ext_half >= 0) &
                (points_x[ref_valid_idx] + ext_half < def_w)
            )

            both_valid = np.where(tar_valid)[0]

            if len(both_valid) > 0:
                global_idx = ref_valid_idx[both_valid]
                v_py = points_y[global_idx]
                v_px = points_x[global_idx]

                # ★ CHANGED: 확장 타겟 서브셋
                tar_subsets = _extract_subsets_batch_extended(def_gray, v_py, v_px, ext_half)

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
