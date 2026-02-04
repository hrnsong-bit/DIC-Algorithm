"""
이미지 보간 모듈

Bicubic (3차) 및 Biquintic (5차) B-spline 보간 지원
"""

import numpy as np
from scipy.ndimage import map_coordinates
from typing import Callable


class ImageInterpolator:
    """이미지 보간 클래스"""
    
    def __init__(self, image: np.ndarray, order: int = 5):
        """
        Args:
            image: 2D 그레이스케일 이미지
            order: 보간 차수 (3=bicubic, 5=biquintic)
        """
        if order not in (3, 5):
            raise ValueError("order must be 3 (bicubic) or 5 (biquintic)")
        
        self.image = image.astype(np.float64)
        self.order = order
        self.height, self.width = image.shape
    
    def __call__(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """
        보간된 픽셀값 반환
        
        Args:
            y: y 좌표 배열
            x: x 좌표 배열
        
        Returns:
            보간된 intensity 값
        """
        coords = np.array([y.ravel(), x.ravel()])
        result = map_coordinates(
            self.image, 
            coords, 
            order=self.order, 
            mode='constant', 
            cval=0.0
        )
        return result.reshape(y.shape)
    
    def is_inside(self, y: np.ndarray, x: np.ndarray) -> np.ndarray:
        """좌표가 이미지 내부인지 확인"""
        margin = self.order // 2 + 1
        inside = (
            (y >= margin) & (y < self.height - margin) &
            (x >= margin) & (x < self.width - margin)
        )
        return inside


def create_interpolator(image: np.ndarray, order: int = 5) -> ImageInterpolator:
    """
    보간 함수 생성
    
    Args:
        image: 2D 그레이스케일 이미지
        order: 보간 차수 (3 or 5)
    
    Returns:
        ImageInterpolator 인스턴스
    """
    return ImageInterpolator(image, order)
