"""
배경 마스킹 모듈 (최적화 버전)
"""

import numpy as np
import cv2
from typing import Tuple, Optional


def create_specimen_mask(image: np.ndarray,
                         method: str = 'otsu',
                         min_intensity: int = 30,
                         otsu_offset: int = -10,
                         morph_size: int = 5,
                         min_area_ratio: float = 0.01) -> np.ndarray:
    """
    시편 영역 마스크 생성 (최적화)
    """
    # 그레이스케일 변환
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    h, w = gray.shape
    
    if method == 'intensity':
        # 가장 빠름: 단순 밝기 기반
        mask = (gray >= min_intensity).astype(np.uint8) * 255
        
    elif method == 'otsu':
        # 빠름: Otsu만 사용
        otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adjusted_thresh = max(0, otsu_thresh + otsu_offset)
        mask = (gray > adjusted_thresh).astype(np.uint8) * 255
        
    else:  # 'combined'
        # Otsu + 밝기 (약간 느림)
        otsu_thresh, _ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        adjusted_thresh = max(0, min(otsu_thresh + otsu_offset, min_intensity))
        mask = (gray > adjusted_thresh).astype(np.uint8) * 255
    
    # 모폴로지 연산 (한 번만)
    if morph_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_size, morph_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    return mask


def apply_mask_to_roi(mask: np.ndarray, 
                      roi: Optional[Tuple[int, int, int, int]]) -> np.ndarray:
    """ROI 영역의 마스크 추출"""
    if roi is None:
        return mask
    
    x, y, w, h = roi
    return mask[y:y+h, x:x+w]


def get_mask_statistics(mask: np.ndarray) -> dict:
    """마스크 통계 정보"""
    total = mask.size
    specimen = np.count_nonzero(mask)
    
    return {
        'total_pixels': total,
        'specimen_pixels': specimen,
        'background_pixels': total - specimen,
        'coverage_ratio': specimen / total if total > 0 else 0
    }


def visualize_mask(image: np.ndarray, 
                   mask: np.ndarray,
                   alpha: float = 0.5,
                   mask_color: Tuple[int, int, int] = (0, 255, 0),
                   background_color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
    """마스크 시각화 (오버레이)"""
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    # 오버레이 생성
    overlay = vis.copy()
    overlay[mask > 0] = mask_color
    overlay[mask == 0] = background_color
    
    return cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)


def visualize_mask_boundary(image: np.ndarray,
                            mask: np.ndarray,
                            boundary_color: Tuple[int, int, int] = (0, 255, 255),
                            thickness: int = 2) -> np.ndarray:
    """마스크 경계선만 표시"""
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, contours, -1, boundary_color, thickness)
    
    return vis