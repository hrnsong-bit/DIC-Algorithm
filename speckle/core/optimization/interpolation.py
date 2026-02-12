# speckle/core/optimization/interpolation.py

"""
이미지 보간 모듈

Bicubic (3차) 및 Biquintic (5차) B-spline 보간 지원
"""

import numpy as np
from scipy.ndimage import map_coordinates, spline_filter

class ImageInterpolator:
    """
    B-spline 보간기. 생성 시 spline 계수를 사전 계산(precompute)하여
    이후 보간 호출에서 중복 계산을 제거합니다.
    
    Pan et al. (2013) "Fast, robust and accurate digital image correlation
    calculation without redundant computations" 의 핵심 최적화 중 하나로,
    IC-GN에서 타겟 이미지 보간 계수를 한 번만 계산합니다.
    
    OpenCorr(oc_cubic_bspline.cpp)에서도 prepare() 단계에서
    전체 이미지에 대해 계수 테이블을 사전 계산하는 동일한 접근을 사용합니다.
    """
    
    def __init__(self, image, order=5):
        if order not in (3, 5):
            raise ValueError("order must be 3 or 5")
        self.order = order
        self.image = np.asarray(image, dtype=np.float64)
        self.height, self.width = self.image.shape
        
        # B-spline 계수를 사전 계산 (Pan 2013 / OpenCorr prepare() 방식)
        # 이후 map_coordinates 호출 시 prefilter=False로 중복 계산 제거
        self._coeffs = spline_filter(self.image, order=self.order,
                                     mode='constant')

    def __call__(self, y, x):
        coords = np.array([y, x])
        result = map_coordinates(self._coeffs, coords,
                                 order=self.order,
                                 mode='constant', cval=0.0,
                                 prefilter=False)
        return result.reshape(np.asarray(y).shape)

    def is_inside(self, y, x):
        margin = self.order // 2 + 1
        return ((y >= margin) & (y < self.height - margin) &
                (x >= margin) & (x < self.width - margin))


def create_interpolator(image, order=5):
    return ImageInterpolator(image, order=order)