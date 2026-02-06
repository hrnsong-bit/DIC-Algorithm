"""
변형률 계산 모듈

IC-GN 결과에서 변형률 필드 계산
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class StrainResult:
    """변형률 계산 결과 (POI 점 데이터)"""
    # 좌표
    points_x: np.ndarray
    points_y: np.ndarray
    
    # 변형률 성분
    exx: np.ndarray
    eyy: np.ndarray
    exy: np.ndarray
    
    # 주변형률
    e1: np.ndarray
    e2: np.ndarray
    angle: np.ndarray
    
    # 기타
    von_mises: np.ndarray
    valid_mask: np.ndarray
    
    @property
    def n_points(self) -> int:
        return len(self.points_x)
    
    @property
    def exx_mean(self) -> float:
        return float(np.mean(self.exx[self.valid_mask]))
    
    @property
    def eyy_mean(self) -> float:
        return float(np.mean(self.eyy[self.valid_mask]))
    
    @property
    def exy_mean(self) -> float:
        return float(np.mean(self.exy[self.valid_mask]))
    
    @property
    def e1_max(self) -> float:
        return float(np.max(self.e1[self.valid_mask]))
    
    @property
    def e2_min(self) -> float:
        return float(np.min(self.e2[self.valid_mask]))
    
    def to_grid(self, spacing: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """POI 데이터를 2D 그리드로 변환 (Savitzky-Golay 입력용)"""
        valid = self.valid_mask
        px, py = self.points_x[valid], self.points_y[valid]
        
        unique_x = np.unique(px)
        unique_y = np.unique(py)
        nx, ny = len(unique_x), len(unique_y)
        
        # 좌표 -> 인덱스 매핑
        x_to_idx = {x: i for i, x in enumerate(unique_x)}
        y_to_idx = {y: i for i, y in enumerate(unique_y)}
        
        # 변위 그리드 (u, v는 외부에서 제공 필요)
        # 여기서는 좌표 정보만 반환
        grid_x = unique_x
        grid_y = unique_y
        
        return grid_x, grid_y, nx, ny


def compute_strain_from_icgn(icgn_result, strain_type: str = 'engineering') -> StrainResult:
    """
    IC-GN 결과에서 변형률 계산
    
    Args:
        icgn_result: ICGNResult 객체
        strain_type: 'engineering' 또는 'green-lagrange'
    
    Returns:
        StrainResult
    """
    ux = icgn_result.disp_ux
    uy = icgn_result.disp_uy
    vx = icgn_result.disp_vx
    vy = icgn_result.disp_vy
    valid_mask = icgn_result.valid_mask
    
    if strain_type == 'engineering':
        exx = ux
        eyy = vy
        exy = 0.5 * (uy + vx)
    elif strain_type == 'green-lagrange':
        exx = ux + 0.5 * (ux**2 + vx**2)
        eyy = vy + 0.5 * (uy**2 + vy**2)
        exy = 0.5 * (uy + vx + ux*uy + vx*vy)
    else:
        raise ValueError(f"Unknown strain_type: {strain_type}")
    
    e1, e2, angle = _compute_principal_strains(exx, eyy, exy)
    von_mises = _compute_von_mises_strain(exx, eyy, exy)
    
    return StrainResult(
        points_x=icgn_result.points_x,
        points_y=icgn_result.points_y,
        exx=exx,
        eyy=eyy,
        exy=exy,
        e1=e1,
        e2=e2,
        angle=angle,
        von_mises=von_mises,
        valid_mask=valid_mask
    )


def _compute_principal_strains(
    exx: np.ndarray, 
    eyy: np.ndarray, 
    exy: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """주변형률 계산"""
    e_mean = 0.5 * (exx + eyy)
    R = np.sqrt(((exx - eyy) / 2)**2 + exy**2)
    
    e1 = e_mean + R
    e2 = e_mean - R
    angle = 0.5 * np.arctan2(2 * exy, exx - eyy)
    angle = np.degrees(angle)
    
    return e1, e2, angle


def _compute_von_mises_strain(
    exx: np.ndarray, 
    eyy: np.ndarray, 
    exy: np.ndarray
) -> np.ndarray:
    """von Mises 유효 변형률"""
    return np.sqrt(exx**2 + eyy**2 - exx*eyy + 3*exy**2)
