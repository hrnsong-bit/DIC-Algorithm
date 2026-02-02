"""
SSSIG 계산 모듈
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from numba import jit, prange

from ..models.reports import BadPoint, SSSIGResult


@jit(nopython=True, parallel=True, cache=True)
def _compute_sssig_map_parallel(gx: np.ndarray, gy: np.ndarray,
                                 points_y: np.ndarray, points_x: np.ndarray,
                                 half: int) -> np.ndarray:
    """모든 POI에 대한 SSSIG 병렬 계산"""
    n_points = len(points_y)
    sssig_values = np.empty(n_points, dtype=np.float64)
    
    h, w = gx.shape
    
    for idx in prange(n_points):
        py, px = points_y[idx], points_x[idx]
        y1 = max(0, py - half)
        y2 = min(h, py + half + 1)
        x1 = max(0, px - half)
        x2 = min(w, px + half + 1)
        
        total = 0.0
        for y in range(y1, y2):
            for x in range(x1, x2):
                gx_val = gx[y, x]
                gy_val = gy[y, x]
                total += gx_val * gx_val + gy_val * gy_val
        
        sssig_values[idx] = total
    
    return sssig_values


def compute_sssig(image: np.ndarray, center: Tuple[int, int], 
                  subset_size: int, gx: np.ndarray = None, 
                  gy: np.ndarray = None) -> float:
    """단일 POI의 SSSIG 계산"""
    if gx is None or gy is None:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_float = image.astype(np.float64)
        gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    
    half = subset_size // 2
    cy, cx = center
    h, w = gx.shape
    
    y1, y2 = max(0, cy - half), min(h, cy + half + 1)
    x1, x2 = max(0, cx - half), min(w, cx + half + 1)
    
    gx_sub = gx[y1:y2, x1:x2]
    gy_sub = gy[y1:y2, x1:x2]
    
    return float(np.sum(gx_sub**2 + gy_sub**2))


def compute_sssig_map(image: np.ndarray,
                      subset_size: int = 21,
                      spacing: int = 10,
                      threshold: float = 1e5,
                      gx: np.ndarray = None,
                      gy: np.ndarray = None,
                      mask: np.ndarray = None) -> SSSIGResult:  # mask 파라미터 유지
    """
    전체 ROI에 대한 SSSIG 맵 계산
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    half = subset_size // 2
    
    # Gradient 계산
    if gx is None or gy is None:
        img_float = gray.astype(np.float64)
        gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    
    # POI 그리드 생성
    margin = half + 1
    y_coords = np.arange(margin, h - margin, spacing)
    x_coords = np.arange(margin, w - margin, spacing)
    
    if len(y_coords) == 0 or len(x_coords) == 0:
        return SSSIGResult(
            map=np.array([[]]),
            mean=0.0, min=float('inf'), max=float('-inf'),
            bad_points=[],
            points_y=np.array([]), points_x=np.array([]),
            subset_size=subset_size, spacing=spacing
        )
    
    # 모든 POI 좌표 생성 (벡터화)
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    all_points_y = yy.ravel()
    all_points_x = xx.ravel()
    
    # ===== 마스크 적용 (비활성화, 코드 보존) =====
    if mask is not None:
        # 마스크가 제공되면 적용
        mask_values = mask[all_points_y, all_points_x]
        valid_mask = mask_values > 0
        points_y = all_points_y[valid_mask].astype(np.int64)
        points_x = all_points_x[valid_mask].astype(np.int64)
    else:
        # 마스크 없으면 전체 사용
        points_y = all_points_y.astype(np.int64)
        points_x = all_points_x.astype(np.int64)
    
    if len(points_y) == 0:
        return SSSIGResult(
            map=np.array([[]]),
            mean=0.0, min=float('inf'), max=float('-inf'),
            bad_points=[],
            points_y=np.array([]), points_x=np.array([]),
            subset_size=subset_size, spacing=spacing
        )
    
    # SSSIG 계산 (Numba 가속)
    sssig_values = _compute_sssig_map_parallel(
        gx.astype(np.float64),
        gy.astype(np.float64),
        points_y, points_x, half
    )
    
    # 통계
    mean_sssig = float(np.mean(sssig_values))
    min_sssig = float(np.min(sssig_values))
    max_sssig = float(np.max(sssig_values))
    
    # 불량 포인트 (벡터화)
    bad_mask = sssig_values < threshold
    bad_indices = np.where(bad_mask)[0]
    
    bad_points = [
        BadPoint(
            y=int(points_y[idx]),
            x=int(points_x[idx]),
            sssig=float(sssig_values[idx])
        )
        for idx in bad_indices
    ]
    
    # 2D 맵
    if mask is None:
        sssig_map = sssig_values.reshape(len(y_coords), len(x_coords))
    else:
        sssig_map = np.full((len(y_coords), len(x_coords)), -1.0)
        y_indices = (points_y - margin) // spacing
        x_indices = (points_x - margin) // spacing
        valid_idx = (y_indices >= 0) & (y_indices < len(y_coords)) & \
                    (x_indices >= 0) & (x_indices < len(x_coords))
        sssig_map[y_indices[valid_idx], x_indices[valid_idx]] = sssig_values[valid_idx]
    
    return SSSIGResult(
        map=sssig_map,
        mean=mean_sssig,
        min=min_sssig,
        max=max_sssig,
        bad_points=bad_points,
        points_y=points_y,
        points_x=points_x,
        subset_size=subset_size,
        spacing=spacing
    )


def find_optimal_subset_size(image: np.ndarray,
                             spacing: int = 10,
                             threshold: float = 1e5,
                             min_size: int = 11,
                             max_size: int = 61,
                             step: int = 2,
                             mask: np.ndarray = None) -> Tuple[int, bool, Optional[SSSIGResult]]:
    """최적 subset size 탐색 (이진 탐색)"""
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    img_float = gray.astype(np.float64)
    gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    
    sizes = list(range(min_size, max_size + 1, step))
    
    left, right = 0, len(sizes) - 1
    best_size = max_size
    best_result = None
    found = False
    
    while left <= right:
        mid = (left + right) // 2
        size = sizes[mid]
        
        result = compute_sssig_map(
            gray, subset_size=size, spacing=spacing,
            threshold=threshold, gx=gx, gy=gy, mask=mask
        )
        
        if len(result.points_y) > 0 and result.min >= threshold:
            best_size = size
            best_result = result
            found = True
            right = mid - 1
        else:
            left = mid + 1
    
    if best_result is None:
        best_result = compute_sssig_map(
            gray, subset_size=max_size, spacing=spacing,
            threshold=threshold, gx=gx, gy=gy, mask=mask
        )
    
    return best_size, found, best_result


def warmup_numba():
    """Numba JIT 워밍업"""
    dummy = np.random.rand(100, 100).astype(np.float64)
    points_y = np.array([50], dtype=np.int64)
    points_x = np.array([50], dtype=np.int64)
    _compute_sssig_map_parallel(dummy, dummy, points_y, points_x, 10)
