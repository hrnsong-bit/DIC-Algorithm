"""
FFT-CC (FFT-based Cross Correlation) 초기 추정 모듈

고속 정수 픽셀 변위 추정을 위한 FFT 기반 정규화 상호상관
"""

import numpy as np
import cv2
from typing import Tuple, Optional, Callable, List, Dict
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import time

from .results import MatchResult, FFTCCResult


# ===== FFT 기반 ZNCC =====
def _fft_zncc(template: np.ndarray, search_win: np.ndarray) -> Tuple[int, int, float]:
    """
    FFT 기반 Zero-mean Normalized Cross-Correlation
    
    올바른 ZNCC: 각 위치에서 로컬 평균과 로컬 에너지를 계산하여 정규화
    """
    sh, sw = search_win.shape
    th, tw = template.shape
    
    # Template zero-mean & 정규화
    t = template.astype(np.float64)
    t_mean = np.mean(t)
    t_zm = t - t_mean
    t_norm = np.sqrt(np.sum(t_zm ** 2))
    if t_norm < 1e-10:
        return 0, 0, 0.0
    
    n_pixels = th * tw  # subset 내 픽셀 수
    s = search_win.astype(np.float64)
    
    # FFT cross-correlation: Σ(t_zm * s) 
    F = np.fft.fft2(t_zm, s=(sh, sw))
    G = np.fft.fft2(s)
    cc = np.fft.ifft2(np.conj(F) * G).real
    
    # 로컬 합 계산 (FFT 활용)
    ones = np.ones((th, tw), dtype=np.float64)
    Ones_fft = np.fft.fft2(ones, s=(sh, sw))
    
    # Σs (각 위치에서의 로컬 합)
    local_sum = np.fft.ifft2(np.fft.fft2(s) * np.conj(Ones_fft)).real
    
    # Σs² (각 위치에서의 로컬 제곱합)
    local_sum_sq = np.fft.ifft2(np.fft.fft2(s ** 2) * np.conj(Ones_fft)).real
    
    # 로컬 분산: Σ(s - s_mean)² = Σs² - (Σs)²/n
    local_var = local_sum_sq - (local_sum ** 2) / n_pixels
    local_var = np.maximum(local_var, 0.0)  # 수치 오차 보정
    local_norm = np.sqrt(local_var)
    
    # ZNCC: cc에서 t_zm과 s의 상관이므로
    # Σ(t_zm * s) = Σ(t_zm * (s - s_mean)) + Σ(t_zm * s_mean)
    # Σ(t_zm) = 0 이므로 Σ(t_zm * s_mean) = s_mean * Σ(t_zm) = 0
    # 따라서 cc = Σ(t_zm * (s - s_mean)) → 정확한 zero-mean cross-correlation
    
    zncc = cc / (t_norm * local_norm + 1e-10)
    
    # 유효 영역에서 피크 찾기
    valid_h = sh - th + 1
    valid_w = sw - tw + 1
    zncc_valid = zncc[:valid_h, :valid_w]
    
    peak_idx = np.argmax(zncc_valid)
    peak_y, peak_x = np.unravel_index(peak_idx, zncc_valid.shape)
    
    return int(peak_y), int(peak_x), float(zncc_valid[peak_y, peak_x])

def compute_fft_cc(ref_image: np.ndarray,
                   def_image: np.ndarray,
                   subset_size: int = 21,
                   spacing: int = 10,
                   search_range: int = 50,
                   zncc_threshold: float = 0.6,
                   roi: Optional[Tuple[int, int, int, int]] = None,
                   n_workers: Optional[int] = None,
                   progress_callback: Optional[Callable[[int, int], None]] = None) -> FFTCCResult:
    """
    FFT-CC 기반 전체 필드 초기 변위 추정 (단일 이미지 쌍)
    """
    start_time = time.time()
    
    # 전처리
    ref_gray = _to_gray(ref_image).astype(np.float32)
    def_gray = _to_gray(def_image).astype(np.float32)
    
    # ROI 처리
    ref_roi, def_extended, roi_offset = _apply_roi(ref_gray, def_gray, roi, search_range)
    
    # POI 그리드 생성
    points_y, points_x = _generate_poi_grid(ref_roi.shape, subset_size, spacing)
    
    n_points = len(points_y)
    if n_points == 0:
        return _empty_result(subset_size, search_range, spacing)
    
    if progress_callback:
        progress_callback(0, n_points)
    
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)
    
    # 결과 배열
    disp_u = np.zeros(n_points, dtype=np.int32)
    disp_v = np.zeros(n_points, dtype=np.int32)
    zncc_values = np.zeros(n_points, dtype=np.float64)
    
    half = subset_size // 2
    ox, oy = roi_offset
    def_h, def_w = def_extended.shape
    
    def process_poi(idx: int) -> Tuple[int, int, int, float]:
        py, px = points_y[idx], points_x[idx]
        
        template = ref_roi[py - half:py + half + 1, px - half:px + half + 1]
        
        sy1 = max(0, py + oy - search_range)
        sy2 = min(def_h, py + oy + search_range + subset_size)
        sx1 = max(0, px + ox - search_range)
        sx2 = min(def_w, px + ox + search_range + subset_size)
        
        search_win = def_extended[sy1:sy2, sx1:sx2]
        
        if search_win.shape[0] < subset_size or search_win.shape[1] < subset_size:
            return idx, 0, 0, 0.0
        
        # FFT-CC 호출
        best_y, best_x, zncc = _fft_zncc(template, search_win)
        
        du = (sx1 + best_x + half) - (px + ox)
        dv = (sy1 + best_y + half) - (py + oy)
        
        return idx, du, dv, zncc
    
    # 병렬 처리
    completed = 0
    update_interval = max(1, n_points // 50)
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(process_poi, i): i for i in range(n_points)}
        
        for future in as_completed(futures):
            idx, du, dv, zncc = future.result()
            disp_u[idx] = du
            disp_v[idx] = dv
            zncc_values[idx] = zncc
            
            completed += 1
            if progress_callback and completed % update_interval == 0:
                progress_callback(completed, n_points)
    
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


def compute_fft_cc_batch_cached(
    ref_image: np.ndarray,
    def_file_paths: List[Path],
    get_image_func: Callable[[Path], Optional[np.ndarray]],
    subset_size: int = 21,
    spacing: int = 10,
    search_range: int = 50,
    zncc_threshold: float = 0.6,
    roi: Optional[Tuple[int, int, int, int]] = None,
    progress_callback: Optional[Callable[[int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None
) -> Dict[str, FFTCCResult]:
    """
    캐시된 이미지를 사용한 배치 FFT-CC 분석
    """
    results = {}
    
    if not def_file_paths:
        return results
    
    # ===== 사전 처리 (한 번만) =====
    ref_gray = _to_gray(ref_image).astype(np.float32)
    
    # ROI 처리
    if roi is not None:
        rx, ry, rw, rh = roi
        ref_roi = ref_gray[ry:ry+rh, rx:rx+rw]
    else:
        ref_roi = ref_gray
        rx, ry = 0, 0
    
    # POI 그리드 생성
    h, w = ref_roi.shape
    half = subset_size // 2
    margin = half + search_range + 1
    
    y_coords = np.arange(margin, h - margin, spacing)
    x_coords = np.arange(margin, w - margin, spacing)
    
    if len(y_coords) == 0 or len(x_coords) == 0:
        return results
    
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    points_y = yy.ravel().astype(np.int64)
    points_x = xx.ravel().astype(np.int64)
    n_points = len(points_y)
    
    # 템플릿 사전 추출
    templates = []
    for idx in range(n_points):
        py, px = points_y[idx], points_x[idx]
        template = ref_roi[py - half:py + half + 1, px - half:px + half + 1].copy()
        templates.append(template)
    
    # ===== 배치 처리 =====
    n_workers = max(1, os.cpu_count() - 1)
    total_files = len(def_file_paths)
    
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        
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
            
            # ROI 영역 + search_range 확장
            if roi is not None:
                y1 = max(0, ry - search_range)
                y2 = min(def_gray.shape[0], ry + rh + search_range)
                x1 = max(0, rx - search_range)
                x2 = min(def_gray.shape[1], rx + rw + search_range)
                def_extended = def_gray[y1:y2, x1:x2]
                ox, oy = rx - x1, ry - y1
            else:
                def_extended = def_gray
                ox, oy = 0, 0
            
            def_h, def_w = def_extended.shape
            
            # 결과 배열
            disp_u = np.zeros(n_points, dtype=np.int32)
            disp_v = np.zeros(n_points, dtype=np.int32)
            zncc_values = np.zeros(n_points, dtype=np.float64)
            
            def make_process_poi(def_ext, offset_x, offset_y, d_h, d_w):
                def process_poi(idx: int) -> Tuple[int, int, int, float]:
                    py, px = points_y[idx], points_x[idx]
                    template = templates[idx]
                    
                    sy1 = max(0, py + offset_y - search_range)
                    sy2 = min(d_h, py + offset_y + search_range + subset_size)
                    sx1 = max(0, px + offset_x - search_range)
                    sx2 = min(d_w, px + offset_x + search_range + subset_size)
                    
                    search_win = def_ext[sy1:sy2, sx1:sx2]
                    
                    if search_win.shape[0] < subset_size or search_win.shape[1] < subset_size:
                        return idx, 0, 0, 0.0
                    
                    # FFT-CC 호출
                    best_y, best_x, zncc = _fft_zncc(template, search_win)
                    
                    du = (sx1 + best_x + half) - (px + offset_x)
                    dv = (sy1 + best_y + half) - (py + offset_y)
                    
                    return idx, du, dv, zncc
                return process_poi
            
            process_poi = make_process_poi(def_extended, ox, oy, def_h, def_w)
            
            futures = [executor.submit(process_poi, i) for i in range(n_points)]
            
            for future in as_completed(futures):
                idx, du, dv, zncc = future.result()
                disp_u[idx] = du
                disp_v[idx] = dv
                zncc_values[idx] = zncc
            
            valid_mask = zncc_values >= zncc_threshold
            
            invalid_points = []
            for idx in np.where(~valid_mask)[0]:
                invalid_points.append(MatchResult(
                    ref_y=int(points_y[idx]),
                    ref_x=int(points_x[idx]),
                    disp_u=int(disp_u[idx]),
                    disp_v=int(disp_v[idx]),
                    zncc=float(zncc_values[idx]),
                    valid=False,
                    flag='low_zncc'
                ))
            
            processing_time = time.time() - start_time
            
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
                processing_time=processing_time
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


def _apply_roi(ref: np.ndarray, 
               defm: np.ndarray,
               roi: Optional[Tuple[int, int, int, int]],
               search_range: int) -> Tuple[np.ndarray, np.ndarray, Tuple[int, int]]:
    """ROI 적용 및 확장 영역 계산"""
    if roi is None:
        return ref, defm, (0, 0)
    
    x, y, w, h = roi
    ref_roi = ref[y:y+h, x:x+w]
    
    y1 = max(0, y - search_range)
    y2 = min(defm.shape[0], y + h + search_range)
    x1 = max(0, x - search_range)
    x2 = min(defm.shape[1], x + w + search_range)
    def_extended = defm[y1:y2, x1:x2]
    
    roi_offset = (x - x1, y - y1)
    
    return ref_roi, def_extended, roi_offset


def _generate_poi_grid(shape: Tuple[int, int],
                       subset_size: int,
                       spacing: int) -> Tuple[np.ndarray, np.ndarray]:
    """POI 그리드 생성"""
    h, w = shape
    half = subset_size // 2
    margin = half + 1
    
    y_coords = np.arange(margin, h - margin, spacing)
    x_coords = np.arange(margin, w - margin, spacing)
    
    if len(y_coords) == 0 or len(x_coords) == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)
    
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    
    return yy.ravel().astype(np.int64), xx.ravel().astype(np.int64)


def _collect_invalid_points(points_y: np.ndarray,
                            points_x: np.ndarray,
                            disp_u: np.ndarray,
                            disp_v: np.ndarray,
                            zncc_values: np.ndarray,
                            valid_mask: np.ndarray,
                            threshold: float) -> list:
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


def _empty_result(subset_size: int, search_range: int, spacing: int) -> FFTCCResult:
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
    compute_fft_cc(dummy, dummy, subset_size=21, spacing=30, search_range=20, n_workers=2)
