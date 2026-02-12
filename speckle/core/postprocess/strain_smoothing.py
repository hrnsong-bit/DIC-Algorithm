"""
변형률 스무딩 모듈

Savitzky-Golay 필터 기반 변형률 계산
Reference: Pan et al. (2007) "Full-field strain measurement using a 
two-dimensional Savitzky-Golay digital differentiator in digital image correlation"
"""

import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import convolve
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class SmoothedStrainResult:
    """스무딩된 변형률 결과"""
    # 변형률 성분 (2D 배열)
    exx: np.ndarray
    eyy: np.ndarray
    exy: np.ndarray
    
    # 주변형률
    e1: np.ndarray
    e2: np.ndarray
    angle: np.ndarray  # degree
    
    # 유효 변형률
    von_mises: np.ndarray
    
    # 파라미터
    window_size: int
    poly_order: int
    grid_step: float
    
    # 그리드 좌표 (시각화용)
    grid_x: np.ndarray = None
    grid_y: np.ndarray = None
    
    @property
    def shape(self) -> Tuple[int, int]:
        return self.exx.shape


def compute_strain_savgol(
    disp_u: np.ndarray,
    disp_v: np.ndarray,
    window_size: int = 15,
    poly_order: int = 2,
    grid_step: float = 1.0,
    strain_type: str = 'engineering'
) -> SmoothedStrainResult:
    """
    Savitzky-Golay 필터 기반 변형률 계산
    
    Pan et al. (2007) 방법 구현
    - 로컬 다항식 피팅의 컨볼루션 형태
    - 노이즈 제거와 미분을 동시에 수행
    
    Args:
        disp_u: x방향 변위 필드 (2D array, shape: height × width)
        disp_v: y방향 변위 필드 (2D array)
        window_size: 필터 윈도우 크기 (홀수, 권장: 11~21)
        poly_order: 다항식 차수 (권장: 1~2)
        grid_step: POI 간격 (pixels)
        strain_type: 'engineering' 또는 'green-lagrange'
    
    Returns:
        SmoothedStrainResult
    """
    # 윈도우 크기 검증
    if window_size % 2 == 0:
        window_size += 1
    if window_size < 5:
        window_size = 5
    if poly_order >= window_size:
        poly_order = window_size - 1
    
    # NaN 처리 (경계)
    u = _fill_nan(disp_u)
    v = _fill_nan(disp_v)
    
    # Savitzky-Golay로 미분 계산
    # axis=1: x방향 (열), axis=0: y방향 (행)
    du_dx = savgol_filter(u, window_size, poly_order, deriv=1, axis=1, mode='nearest') / grid_step
    du_dy = savgol_filter(u, window_size, poly_order, deriv=1, axis=0, mode='nearest') / grid_step
    dv_dx = savgol_filter(v, window_size, poly_order, deriv=1, axis=1, mode='nearest') / grid_step
    dv_dy = savgol_filter(v, window_size, poly_order, deriv=1, axis=0, mode='nearest') / grid_step
    
    # 변형률 계산
    if strain_type == 'engineering':
        exx = du_dx
        eyy = dv_dy
        exy = 0.5 * (du_dy + dv_dx)
    elif strain_type == 'green-lagrange':
        exx = du_dx + 0.5 * (du_dx**2 + dv_dx**2)
        eyy = dv_dy + 0.5 * (du_dy**2 + dv_dy**2)
        exy = 0.5 * (du_dy + dv_dx + du_dx*du_dy + dv_dx*dv_dy)
    else:
        raise ValueError(f"Unknown strain_type: {strain_type}")
    
    # 원본 NaN 위치 복원
    nan_mask = np.isnan(disp_u) | np.isnan(disp_v)
    exx[nan_mask] = np.nan
    eyy[nan_mask] = np.nan
    exy[nan_mask] = np.nan
    
    # 주변형률
    e1, e2, angle = _compute_principal_strains(exx, eyy, exy)
    
    # von Mises
    von_mises = _compute_von_mises(exx, eyy, exy)
    
    return SmoothedStrainResult(
        exx=exx,
        eyy=eyy,
        exy=exy,
        e1=e1,
        e2=e2,
        angle=angle,
        von_mises=von_mises,
        window_size=window_size,
        poly_order=poly_order,
        grid_step=grid_step
    )


def compute_strain_savgol_2d(
    disp_u: np.ndarray,
    disp_v: np.ndarray,
    window_size: int = 15,
    grid_step: float = 1.0
) -> SmoothedStrainResult:
    """
    2D Savitzky-Golay 필터 (Pan et al. 2007 원본 방식)
    
    1차 다항식 피팅: u(x,y) = a₀ + a₁x + a₂y
    컨볼루션 커널로 직접 계산
    """
    if window_size % 2 == 0:
        window_size += 1
    
    half = window_size // 2
    
    # 2D SG 커널 생성 (1차 다항식)
    kernel_dx, kernel_dy = _create_2d_savgol_kernels(window_size)
    
    # NaN 처리
    u = _fill_nan(disp_u)
    v = _fill_nan(disp_v)
    
    # 컨볼루션으로 미분
    du_dx = convolve(u, kernel_dx, mode='nearest') / grid_step
    du_dy = convolve(u, kernel_dy, mode='nearest') / grid_step
    dv_dx = convolve(v, kernel_dx, mode='nearest') / grid_step
    dv_dy = convolve(v, kernel_dy, mode='nearest') / grid_step
    
    # 변형률
    exx = du_dx
    eyy = dv_dy
    exy = 0.5 * (du_dy + dv_dx)
    
    # NaN 복원
    nan_mask = np.isnan(disp_u) | np.isnan(disp_v)
    exx[nan_mask] = np.nan
    eyy[nan_mask] = np.nan
    exy[nan_mask] = np.nan
    
    # 주변형률, von Mises
    e1, e2, angle = _compute_principal_strains(exx, eyy, exy)
    von_mises = _compute_von_mises(exx, eyy, exy)
    
    return SmoothedStrainResult(
        exx=exx,
        eyy=eyy,
        exy=exy,
        e1=e1,
        e2=e2,
        angle=angle,
        von_mises=von_mises,
        window_size=window_size,
        poly_order=1,
        grid_step=grid_step
    )


def _create_2d_savgol_kernels(window_size: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    2D Savitzky-Golay 미분 커널 생성 (Pan et al. 2007, Eq. 10)
    
    1차 다항식 피팅: f(x,y) = a₀ + a₁x + a₂y
    커널 계수: 3 / [(2M+1)² × (M+1) × M]
    """
    M = window_size // 2
    n = window_size
    
    # 정규화 계수
    norm = 3.0 / (n * n * (M + 1) * M)
    
    # x 미분 커널
    kernel_dx = np.zeros((window_size, window_size))
    for j in range(window_size):
        for i in range(window_size):
            kernel_dx[j, i] = (i - M) * norm
    
    # y 미분 커널 (x 커널의 전치)
    kernel_dy = kernel_dx.T
    
    return kernel_dx, kernel_dy


def _fill_nan(data: np.ndarray) -> np.ndarray:
    """NaN을 주변값으로 채우기"""
    result = data.copy()
    nan_mask = np.isnan(result)
    
    if not np.any(nan_mask):
        return result
    
    # 간단한 보간 (nearest)
    from scipy.ndimage import distance_transform_edt
    
    indices = distance_transform_edt(
        nan_mask, return_distances=False, return_indices=True
    )
    result = result[tuple(indices)]
    
    return result


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
    angle = 0.5 * np.degrees(np.arctan2(2 * exy, exx - eyy))
    
    return e1, e2, angle


def _compute_von_mises(
    exx: np.ndarray, 
    eyy: np.ndarray, 
    exy: np.ndarray
) -> np.ndarray:
    """von Mises 유효 변형률"""
    return np.sqrt(exx**2 + eyy**2 - exx*eyy + 3*exy**2)


def get_vsg_size(window_size: int, step_size: int, subset_size: int) -> int:
    """
    Virtual Strain Gauge 크기 계산 (VIC-3D 방식)
    
    VSG = (filter_size - 1) × step_size + subset_size
    """
    return (window_size - 1) * step_size + subset_size
def poi_to_grid(
    points_x: np.ndarray,
    points_y: np.ndarray,
    values: np.ndarray,
    valid_mask: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    POI 점 데이터를 2D 그리드로 변환
    
    Returns:
        grid_data: 2D 배열
        grid_x: x 좌표 배열
        grid_y: y 좌표 배열
    """
    valid = valid_mask
    px, py = points_x[valid], points_y[valid]
    vals = values[valid]
    
    unique_x = np.unique(px)
    unique_y = np.unique(py)
    nx, ny = len(unique_x), len(unique_y)
    
    # 좌표 → 인덱스 매핑
    x_to_idx = {x: i for i, x in enumerate(unique_x)}
    y_to_idx = {y: i for i, y in enumerate(unique_y)}
    
    # 2D 그리드 생성
    grid = np.full((ny, nx), np.nan)
    for i in range(len(px)):
        xi = x_to_idx[px[i]]
        yi = y_to_idx[py[i]]
        grid[yi, xi] = vals[i]
    
    return grid, unique_x, unique_y

def compute_strain_from_icgn_smoothed(
    icgn_result,
    window_size: int = 15,
    poly_order: int = 2,
    strain_type: str = 'engineering'
) -> SmoothedStrainResult:
    """
    IC-GN 결과에서 직접 스무딩된 변형률 맵 계산
    
    POI 점 데이터 → 2D 그리드 → Savitzky-Golay 자동 처리
    """
    valid = icgn_result.valid_mask
    px, py = icgn_result.points_x, icgn_result.points_y
    
    # 유효한 점만 사용
    px_valid = px[valid]
    py_valid = py[valid]
    
    # 그리드 정보 추출
    unique_x = np.unique(px_valid)
    unique_y = np.unique(py_valid)
    nx, ny = len(unique_x), len(unique_y)
    
    if nx < 3 or ny < 3:
        raise ValueError(f"그리드가 너무 작습니다: {nx}x{ny}")
    
    # 그리드 간격 추정
    grid_step = float(np.median(np.diff(unique_x))) if len(unique_x) > 1 else 1.0
    
    # 좌표 → 인덱스 매핑
    x_to_idx = {x: i for i, x in enumerate(unique_x)}
    y_to_idx = {y: i for i, y in enumerate(unique_y)}
    
    # 변위를 2D 그리드로 변환
    disp_u_grid = np.full((ny, nx), np.nan)
    disp_v_grid = np.full((ny, nx), np.nan)
    
    for i in range(len(px)):
        if valid[i]:
            xi = x_to_idx.get(px[i])
            yi = y_to_idx.get(py[i])
            if xi is not None and yi is not None:
                disp_u_grid[yi, xi] = icgn_result.disp_u[i]
                disp_v_grid[yi, xi] = icgn_result.disp_v[i]
    
    # Savitzky-Golay 적용
    result = compute_strain_savgol(
        disp_u_grid, 
        disp_v_grid,
        window_size=window_size,
        poly_order=poly_order,
        grid_step=grid_step,
        strain_type=strain_type
    )
    
    # 그리드 좌표 정보 저장 (시각화용)
    result.grid_x = unique_x
    result.grid_y = unique_y
    
    return result 