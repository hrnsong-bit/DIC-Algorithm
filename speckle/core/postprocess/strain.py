"""
변형률 계산 모듈

IC-GN 결과에서 변형률 필드 계산
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class StrainResult:
    """변형률 계산 결과"""
    # 좌표
    points_x: np.ndarray
    points_y: np.ndarray
    
    # 변형률 성분
    exx: np.ndarray  # ∂u/∂x (수평 방향 수직 변형률)
    eyy: np.ndarray  # ∂v/∂y (수직 방향 수직 변형률)
    exy: np.ndarray  # 0.5*(∂u/∂y + ∂v/∂x) (전단 변형률)
    
    # 주변형률
    e1: np.ndarray   # 최대 주변형률
    e2: np.ndarray   # 최소 주변형률
    angle: np.ndarray  # 주변형률 방향 (degree)
    
    # 기타
    von_mises: np.ndarray  # von Mises equivalent strain
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


def compute_strain_from_icgn(icgn_result, strain_type: str = 'engineering') -> StrainResult:
    """
    IC-GN 결과에서 변형률 계산
    
    IC-GN은 이미 변형률 성분(ux, uy, vx, vy)을 제공하므로
    직접 사용하거나 변환만 하면 됨
    
    Args:
        icgn_result: ICGNResult 객체
        strain_type: 'engineering' 또는 'green-lagrange'
    
    Returns:
        StrainResult: 변형률 결과
    """
    # IC-GN 결과에서 변형률 성분 추출
    ux = icgn_result.disp_ux  # ∂u/∂x
    uy = icgn_result.disp_uy  # ∂u/∂y
    vx = icgn_result.disp_vx  # ∂v/∂x
    vy = icgn_result.disp_vy  # ∂v/∂y
    
    valid_mask = icgn_result.valid_mask
    
    if strain_type == 'engineering':
        # Engineering strain (소변형 가정)
        # εxx = ∂u/∂x
        # εyy = ∂v/∂y
        # εxy = 0.5*(∂u/∂y + ∂v/∂x)
        exx = ux
        eyy = vy
        exy = 0.5 * (uy + vx)
        
    elif strain_type == 'green-lagrange':
        # Green-Lagrange strain (대변형)
        # Exx = ∂u/∂x + 0.5*((∂u/∂x)² + (∂v/∂x)²)
        # Eyy = ∂v/∂y + 0.5*((∂u/∂y)² + (∂v/∂y)²)
        # Exy = 0.5*(∂u/∂y + ∂v/∂x + ∂u/∂x*∂u/∂y + ∂v/∂x*∂v/∂y)
        exx = ux + 0.5 * (ux**2 + vx**2)
        eyy = vy + 0.5 * (uy**2 + vy**2)
        exy = 0.5 * (uy + vx + ux*uy + vx*vy)
    else:
        raise ValueError(f"Unknown strain_type: {strain_type}")
    
    # 주변형률 계산
    e1, e2, angle = _compute_principal_strains(exx, eyy, exy)
    
    # von Mises equivalent strain
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


def _compute_principal_strains(exx: np.ndarray, eyy: np.ndarray, exy: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    주변형률 계산
    
    2D 변형률 텐서의 고유값 문제:
    | exx  exy |
    | exy  eyy |
    
    e1, e2 = (exx + eyy)/2 ± sqrt(((exx - eyy)/2)² + exy²)
    """
    # 평균 변형률
    e_mean = 0.5 * (exx + eyy)
    
    # 반지름 (Mohr's circle)
    R = np.sqrt(((exx - eyy) / 2)**2 + exy**2)
    
    # 주변형률
    e1 = e_mean + R  # 최대
    e2 = e_mean - R  # 최소
    
    # 주변형률 방향 (degree)
    # tan(2θ) = 2*exy / (exx - eyy)
    angle = 0.5 * np.arctan2(2 * exy, exx - eyy)
    angle = np.degrees(angle)
    
    return e1, e2, angle


def _compute_von_mises_strain(exx: np.ndarray, eyy: np.ndarray, exy: np.ndarray) -> np.ndarray:
    """
    von Mises equivalent strain (2D plane strain 가정)
    
    ε_vm = sqrt(2/3 * (exx² + eyy² + 2*exy² - exx*eyy))
    
    또는 간단히 (평면 응력):
    ε_vm = sqrt(exx² + eyy² - exx*eyy + 3*exy²)
    """
    return np.sqrt(exx**2 + eyy**2 - exx*eyy + 3*exy**2)


def compute_strain_from_displacement_field(
    points_x: np.ndarray,
    points_y: np.ndarray,
    disp_u: np.ndarray,
    disp_v: np.ndarray,
    valid_mask: np.ndarray,
    method: str = 'least_squares',
    window_size: int = 3,
    strain_type: str = 'engineering'
) -> StrainResult:
    """
    변위장에서 변형률 계산 (수치 미분)
    
    FFT-CC 결과처럼 변형률 성분이 없는 경우 사용
    
    Args:
        points_x, points_y: POI 좌표
        disp_u, disp_v: 변위
        valid_mask: 유효 마스크
        method: 'least_squares' 또는 'finite_difference'
        window_size: 미분 계산에 사용할 이웃 크기
        strain_type: 'engineering' 또는 'green-lagrange'
    
    Returns:
        StrainResult: 변형률 결과
    """
    n_points = len(points_x)
    
    # 결과 배열 초기화
    ux = np.zeros(n_points)
    uy = np.zeros(n_points)
    vx = np.zeros(n_points)
    vy = np.zeros(n_points)
    strain_valid = np.zeros(n_points, dtype=bool)
    
    # POI를 그리드로 재구성 (spacing 추정)
    unique_x = np.unique(points_x)
    unique_y = np.unique(points_y)
    
    if len(unique_x) < 3 or len(unique_y) < 3:
        raise ValueError("변형률 계산에는 최소 3x3 POI 그리드가 필요합니다")
    
    spacing_x = np.median(np.diff(unique_x))
    spacing_y = np.median(np.diff(unique_y))
    
    # 좌표 -> 인덱스 매핑
    x_to_idx = {x: i for i, x in enumerate(unique_x)}
    y_to_idx = {y: i for i, y in enumerate(unique_y)}
    
    nx, ny = len(unique_x), len(unique_y)
    
    # 그리드 형태로 변환
    u_grid = np.full((ny, nx), np.nan)
    v_grid = np.full((ny, nx), np.nan)
    
    for i in range(n_points):
        if valid_mask[i]:
            xi = x_to_idx.get(points_x[i])
            yi = y_to_idx.get(points_y[i])
            if xi is not None and yi is not None:
                u_grid[yi, xi] = disp_u[i]
                v_grid[yi, xi] = disp_v[i]
    
    # 중심 차분으로 미분 계산
    ux_grid, uy_grid = _gradient_2d(u_grid, spacing_x, spacing_y)
    vx_grid, vy_grid = _gradient_2d(v_grid, spacing_x, spacing_y)
    
    # 결과를 1D 배열로 변환
    for i in range(n_points):
        xi = x_to_idx.get(points_x[i])
        yi = y_to_idx.get(points_y[i])
        if xi is not None and yi is not None:
            if not np.isnan(ux_grid[yi, xi]):
                ux[i] = ux_grid[yi, xi]
                uy[i] = uy_grid[yi, xi]
                vx[i] = vx_grid[yi, xi]
                vy[i] = vy_grid[yi, xi]
                strain_valid[i] = valid_mask[i]
    
    # 변형률 계산
    if strain_type == 'engineering':
        exx = ux
        eyy = vy
        exy = 0.5 * (uy + vx)
    else:  # green-lagrange
        exx = ux + 0.5 * (ux**2 + vx**2)
        eyy = vy + 0.5 * (uy**2 + vy**2)
        exy = 0.5 * (uy + vx + ux*uy + vx*vy)
    
    # 주변형률
    e1, e2, angle = _compute_principal_strains(exx, eyy, exy)
    von_mises = _compute_von_mises_strain(exx, eyy, exy)
    
    return StrainResult(
        points_x=points_x,
        points_y=points_y,
        exx=exx,
        eyy=eyy,
        exy=exy,
        e1=e1,
        e2=e2,
        angle=angle,
        von_mises=von_mises,
        valid_mask=strain_valid
    )


def _gradient_2d(field: np.ndarray, dx: float, dy: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D 필드의 gradient 계산 (중심 차분, NaN 처리)
    """
    ny, nx = field.shape
    grad_x = np.full_like(field, np.nan)
    grad_y = np.full_like(field, np.nan)
    
    # x 방향 미분
    for j in range(ny):
        for i in range(1, nx - 1):
            if not np.isnan(field[j, i-1]) and not np.isnan(field[j, i+1]):
                grad_x[j, i] = (field[j, i+1] - field[j, i-1]) / (2 * dx)
    
    # y 방향 미분
    for j in range(1, ny - 1):
        for i in range(nx):
            if not np.isnan(field[j-1, i]) and not np.isnan(field[j+1, i]):
                grad_y[j, i] = (field[j+1, i] - field[j-1, i]) / (2 * dy)
    
    return grad_x, grad_y
