"""
시각화 유틸리티

POI 오버레이, SSSIG 히트맵 등
"""

import numpy as np
import cv2
from typing import Optional, Tuple

from ..models.reports import QualityReport, SSSIGResult


def draw_poi_overlay(image: np.ndarray,
                     sssig_result: SSSIGResult,
                     threshold: float,
                     show_all: bool = True,
                     show_bad: bool = True) -> np.ndarray:
    """
    POI 오버레이 그리기
    
    Args:
        image: 원본 이미지
        sssig_result: SSSIG 결과
        threshold: SSSIG 임계값
        show_all: 모든 POI 표시 여부
        show_bad: 불량 POI 강조 여부
    
    Returns:
        오버레이가 그려진 이미지
    """
    # BGR 변환
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()
    
    sssig_values = sssig_result.map.ravel()
    points_y = sssig_result.points_y
    points_x = sssig_result.points_x
    
    # 모든 POI 표시 (작은 점)
    if show_all:
        for idx in range(len(points_y)):
            py, px = int(points_y[idx]), int(points_x[idx])
            sssig = sssig_values[idx]
            
            if sssig >= threshold:
                # 양호: 초록색
                cv2.circle(overlay, (px, py), 2, (0, 255, 0), -1)
            elif show_bad:
                # 불량: 빨간색 + 큰 마커
                cv2.circle(overlay, (px, py), 5, (0, 0, 255), -1)
                cv2.drawMarker(overlay, (px, py), (0, 0, 255),
                              cv2.MARKER_CROSS, 10, 2)
    
    # 불량 POI만 표시
    elif show_bad:
        for bp in sssig_result.bad_points:
            cv2.circle(overlay, (bp.x, bp.y), 5, (0, 0, 255), -1)
            cv2.drawMarker(overlay, (bp.x, bp.y), (0, 0, 255),
                          cv2.MARKER_CROSS, 10, 2)
    
    return overlay


def draw_sssig_heatmap(sssig_result: SSSIGResult,
                       image_shape: Tuple[int, int],
                       colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """
    SSSIG 히트맵 생성
    
    Args:
        sssig_result: SSSIG 결과
        image_shape: 원본 이미지 크기 (h, w)
        colormap: OpenCV 컬러맵
    
    Returns:
        히트맵 이미지
    """
    sssig_map = sssig_result.map
    
    # 정규화
    if sssig_map.size == 0:
        return np.zeros((*image_shape, 3), dtype=np.uint8)
    
    min_val, max_val = sssig_map.min(), sssig_map.max()
    if max_val - min_val > 0:
        normalized = (sssig_map - min_val) / (max_val - min_val) * 255
    else:
        normalized = np.zeros_like(sssig_map)
    
    normalized = normalized.astype(np.uint8)
    
    # 컬러맵 적용
    heatmap_small = cv2.applyColorMap(normalized, colormap)
    
    # 원본 크기로 리사이즈
    heatmap = cv2.resize(heatmap_small, (image_shape[1], image_shape[0]),
                         interpolation=cv2.INTER_NEAREST)
    
    return heatmap


def draw_roi(image: np.ndarray,
             roi: Tuple[int, int, int, int],
             color: Tuple[int, int, int] = (0, 255, 0),
             thickness: int = 2) -> np.ndarray:
    """
    ROI 사각형 그리기
    
    Args:
        image: 원본 이미지
        roi: (x, y, w, h)
        color: BGR 색상
        thickness: 선 두께
    
    Returns:
        ROI가 그려진 이미지
    """
    if len(image.shape) == 2:
        overlay = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        overlay = image.copy()
    
    x, y, w, h = roi
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, thickness)
    
    return overlay


def create_result_visualization(image: np.ndarray,
                                report: QualityReport,
                                show_all_poi: bool = True) -> np.ndarray:
    """
    평가 결과 종합 시각화
    
    Args:
        image: 원본 이미지
        report: 평가 결과
        show_all_poi: 모든 POI 표시 여부
    
    Returns:
        시각화 이미지
    """
    if report.sssig_result is None:
        # SSSIG 결과 없음 (MIG 실패)
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        return image.copy()
    
    return draw_poi_overlay(
        image,
        report.sssig_result,
        report.sssig_threshold,
        show_all=show_all_poi,
        show_bad=True
    )
