"""
배경 마스킹 모듈 (최적화 버전)

시편/배경 분리 및 내부 결함(홀, 노치) 검출용.
DIC 파이프라인에서 distance_transform_edt와 연동하여
POI별 adaptive search_range를 결정하는 데 사용.
"""

import numpy as np
import cv2
from typing import Tuple, Optional

def create_specimen_mask(image: np.ndarray,
                         dark_threshold: int = 25,
                         min_hole_area: int = 100,
                         morph_size: int = 5) -> np.ndarray:
    """
    시편 내부 결함(홀, 노치 등) 검출용 마스크
    
    회색조 값이 dark_threshold 이하인 영역을 결함(0)으로 판정.
    시편 외곽 경계는 이미지 경계 기반 계산이 별도로 처리하므로
    이 마스크는 내부 결함만 담당.
    
    Parameters
    ----------
    dark_threshold : int
        이 값 이하 픽셀을 결함으로 판정 (기본 5)
    min_hole_area : int
        결함으로 인정할 최소 면적(px). 노이즈 제거용.
    morph_size : int
        모폴로지 커널 크기. 결함 경계 정리용.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if gray.dtype != np.uint8:
        if gray.max() <= 1.0:
            gray = (gray * 255).astype(np.uint8)
        else:
            gray = np.clip(gray, 0, 255).astype(np.uint8)
    
    # 전체를 시편(255)으로 시작
    mask = np.full_like(gray, 255)
    
    # 어두운 픽셀 = 결함(홀, 배경 등)
    dark_pixels = gray <= dark_threshold
    mask[dark_pixels] = 0
    
    # 모폴로지로 노이즈 제거 및 경계 정리
    if morph_size > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                           (morph_size, morph_size))
        # close: 결함 내부의 밝은 노이즈 메움
        mask_inv = cv2.bitwise_not(mask)
        mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_CLOSE, kernel)
        # open: 작은 노이즈 제거
        mask_inv = cv2.morphologyEx(mask_inv, cv2.MORPH_OPEN, kernel)
        mask = cv2.bitwise_not(mask_inv)
    
    # 면적 필터: 너무 작은 결함은 노이즈로 제거
    contours, _ = cv2.findContours(
        cv2.bitwise_not(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    mask = np.full_like(gray, 255)
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_hole_area:
            cv2.drawContours(mask, [cnt], -1, 0, cv2.FILLED)
    
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
    overlay = vis.copy()
    overlay[mask > 0] = mask_color
    overlay[mask == 0] = background_color
    return cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)


def visualize_mask_boundary(image: np.ndarray,
                            mask: np.ndarray,
                            boundary_color: Tuple[int, int, int] = (0, 255, 255),
                            hole_color: Tuple[int, int, int] = (0, 0, 255),
                            thickness: int = 2) -> np.ndarray:
    """마스크 경계선 표시 (외곽 + 내부 홀)"""
    if len(image.shape) == 2:
        vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        vis = image.copy()
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_CCOMP,
                                            cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is not None:
        for i, cnt in enumerate(contours):
            if hierarchy[0][i][3] < 0:
                cv2.drawContours(vis, [cnt], -1, boundary_color, thickness)
            else:
                cv2.drawContours(vis, [cnt], -1, hole_color, thickness)
    return vis
