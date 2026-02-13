"""
Pointwise Least Squares (PLS) 변형률 계산 모듈

각 POI에서 로컬 윈도우 내 변위 데이터에 다항식을 최소자승 피팅하고,
피팅 계수로부터 변형률(미분)을 직접 추출합니다.

Savitzky-Golay 대비 장점:
    - 불규칙 간격 POI 및 NaN(결측) 데이터 자연스럽게 처리
    - Gaussian 가중함수로 거리 기반 가중치 적용
    - POI별로 독립 계산 → 경계 처리 별도 불필요

Reference:
    Pan, B., et al. "Digital image correlation using iterative least squares
    and pointwise least squares for displacement field and strain field
    measurements." Optics and Lasers in Engineering, 47(7-8), 865-874, 2009.
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple, Optional

_logger = logging.getLogger(__name__)


@dataclass
class PLSStrainResult:
    """PLS 변형률 결과"""
    # 변형률 성분 (2D 배열)
    exx: np.ndarray
    eyy: np.ndarray
    exy: np.ndarray

    # 주변형률
    e1: np.ndarray
    e2: np.ndarray
    angle: np.ndarray  # degree

    # von Mises 유효 변형률
    von_mises: np.ndarray

    # 파라미터
    window_size: int
    poly_order: int
    grid_step: float

    # 그리드 좌표 (시각화용)
    grid_x: Optional[np.ndarray] = None
    grid_y: Optional[np.ndarray] = None

    @property
    def shape(self) -> Tuple[int, int]:
        return self.exx.shape


def compute_strain_pls(
    disp_u: np.ndarray,
    disp_v: np.ndarray,
    window_size: int = 15,
    poly_order: int = 2,
    grid_step: float = 1.0,
    strain_type: str = 'engineering'
) -> PLSStrainResult:
    """
    Pointwise Least Squares (PLS) 변형률 계산

    각 격자점에서 window_size × window_size 영역 내의 유효 변위 데이터에
    2D 다항식을 Gaussian 가중 최소자승 피팅하고, 피팅 계수에서 변형률을 추출.

    1차 다항식: u(x,y) = a0 + a1*x + a2*y
        → du/dx = a1, du/dy = a2

    2차 다항식: u(x,y) = a0 + a1*x + a2*y + a3*x² + a4*xy + a5*y²
        → du/dx = a1 + 2*a3*x + a4*y  (중심 x=0,y=0에서 a1)
        → du/dy = a2 + a4*x + 2*a5*y  (중심에서 a2)

    Args:
        disp_u: x방향 변위 필드 (2D, shape: ny × nx), NaN 허용
        disp_v: y방향 변위 필드 (2D), NaN 허용
        window_size: PLS 윈도우 크기 (홀수, 권장: 11~21)
        poly_order: 다항식 차수 (1 또는 2, 기본 2)
        grid_step: POI 간격 (pixels)
        strain_type: 'engineering' 또는 'green-lagrange'

    Returns:
        PLSStrainResult
    """
    # === 입력 검증 ===
    if disp_u.ndim != 2 or disp_v.ndim != 2:
        raise ValueError(f"변위 필드는 2D 배열이어야 합니다: u={disp_u.ndim}D, v={disp_v.ndim}D")
    if disp_u.shape != disp_v.shape:
        raise ValueError(f"u, v 크기 불일치: {disp_u.shape} vs {disp_v.shape}")

    if window_size % 2 == 0:
        window_size += 1
        _logger.warning(f"window_size를 홀수로 조정: {window_size}")
    if window_size < 5:
        window_size = 5
        _logger.warning("window_size 최소값 5로 조정")

    if poly_order not in (1, 2):
        _logger.warning(f"poly_order={poly_order} → 2로 조정 (1 또는 2만 지원)")
        poly_order = 2

    # 최소 데이터 수 확인: 1차=3개 계수, 2차=6개 계수
    min_points_needed = 3 if poly_order == 1 else 6

    ny, nx = disp_u.shape
    half = window_size // 2

    # === 출력 배열 초기화 ===
    du_dx = np.full((ny, nx), np.nan)
    du_dy = np.full((ny, nx), np.nan)
    dv_dx = np.full((ny, nx), np.nan)
    dv_dy = np.full((ny, nx), np.nan)

    # === Gaussian 가중함수 사전 계산 ===
    # σ = half / 2 → 윈도우 경계에서 가중치 ≈ 0.14 (충분히 감소)
    sigma = half / 2.0
    if sigma < 1.0:
        sigma = 1.0

    # 로컬 좌표 그리드 (윈도우 내, 물리 단위)
    local_y, local_x = np.mgrid[-half:half+1, -half:half+1]
    local_x_phys = local_x.ravel() * grid_step
    local_y_phys = local_y.ravel() * grid_step
    r_sq = (local_x.ravel()**2 + local_y.ravel()**2).astype(np.float64)
    gauss_weights_full = np.exp(-r_sq / (2.0 * sigma**2))

    # === 디자인 행렬 구성 함수 ===
    if poly_order == 1:
        # u(x,y) = a0 + a1*x + a2*y → 3 계수
        def build_design(x, y):
            return np.column_stack([np.ones_like(x), x, y])
    else:
        # u(x,y) = a0 + a1*x + a2*y + a3*x² + a4*xy + a5*y² → 6 계수
        def build_design(x, y):
            return np.column_stack([
                np.ones_like(x), x, y,
                x**2, x*y, y**2
            ])

    # === POI별 PLS 피팅 ===
    for iy in range(ny):
        for ix in range(nx):
            # 현재 POI의 변위가 NaN이면 건너뜀
            if np.isnan(disp_u[iy, ix]) or np.isnan(disp_v[iy, ix]):
                continue

            # 윈도우 범위 (배열 경계 클리핑)
            y_start = max(0, iy - half)
            y_end = min(ny, iy + half + 1)
            x_start = max(0, ix - half)
            x_end = min(nx, ix + half + 1)

            # 윈도우 내 변위 추출
            u_win = disp_u[y_start:y_end, x_start:x_end]
            v_win = disp_v[y_start:y_end, x_start:x_end]

            # 유효 (non-NaN) 마스크
            valid = ~(np.isnan(u_win) | np.isnan(v_win))
            n_valid = np.count_nonzero(valid)

            if n_valid < min_points_needed:
                continue

            # 로컬 좌표 (현재 POI 기준, 물리 단위)
            win_local_y, win_local_x = np.mgrid[
                y_start - iy : y_end - iy,
                x_start - ix : x_end - ix
            ]
            lx = win_local_x[valid].ravel() * grid_step
            ly = win_local_y[valid].ravel() * grid_step

            # Gaussian 가중치
            r2 = (win_local_x[valid].ravel()**2 +
                  win_local_y[valid].ravel()**2).astype(np.float64)
            w = np.exp(-r2 / (2.0 * sigma**2))

            # 가중 디자인 행렬: W^(1/2) * A
            sqrt_w = np.sqrt(w)
            A = build_design(lx, ly)
            Aw = A * sqrt_w[:, None]

            # 가중 관측값: W^(1/2) * b
            u_vals = u_win[valid].ravel()
            v_vals = v_win[valid].ravel()
            bu = u_vals * sqrt_w
            bv = v_vals * sqrt_w

            # 최소자승 풀기: (Aw^T Aw) c = Aw^T b
            # lstsq는 rank-deficient도 안전하게 처리
            try:
                cu, res_u, rank_u, sv_u = np.linalg.lstsq(Aw, bu, rcond=None)
                cv, res_v, rank_v, sv_v = np.linalg.lstsq(Aw, bv, rcond=None)
            except np.linalg.LinAlgError:
                continue

            # 계수에서 미분 추출 (중심점 x=0, y=0)
            # 1차: c = [a0, a1, a2]       → du/dx=a1, du/dy=a2
            # 2차: c = [a0, a1, a2, a3, a4, a5] → du/dx=a1, du/dy=a2
            du_dx[iy, ix] = cu[1]
            du_dy[iy, ix] = cu[2]
            dv_dx[iy, ix] = cv[1]
            dv_dy[iy, ix] = cv[2]

    # === 변형률 계산 ===
    if strain_type == 'engineering':
        exx = du_dx
        eyy = dv_dy
        exy = 0.5 * (du_dy + dv_dx)
    elif strain_type == 'green-lagrange':
        exx = du_dx + 0.5 * (du_dx**2 + dv_dx**2)
        eyy = dv_dy + 0.5 * (du_dy**2 + dv_dy**2)
        exy = 0.5 * (du_dy + dv_dx + du_dx * du_dy + dv_dx * dv_dy)
    else:
        raise ValueError(f"Unknown strain_type: {strain_type}")

    # 주변형률
    e1, e2, angle = _compute_principal_strains(exx, eyy, exy)

    # von Mises
    von_mises = _compute_von_mises(exx, eyy, exy)

    return PLSStrainResult(
        exx=exx, eyy=eyy, exy=exy,
        e1=e1, e2=e2, angle=angle,
        von_mises=von_mises,
        window_size=window_size,
        poly_order=poly_order,
        grid_step=grid_step
    )


def compute_strain_pls_from_icgn(
    icgn_result,
    window_size: int = 15,
    poly_order: int = 2,
    strain_type: str = 'engineering'
) -> PLSStrainResult:
    """
    IC-GN 결과에서 직접 PLS 변형률 계산

    POI 점 데이터 → 2D 그리드 변환 → PLS 자동 처리

    Args:
        icgn_result: ICGNResult 객체
        window_size: PLS 윈도우 크기 (홀수)
        poly_order: 다항식 차수 (1 또는 2)
        strain_type: 'engineering' 또는 'green-lagrange'

    Returns:
        PLSStrainResult (grid_x, grid_y 포함)
    """
    valid = icgn_result.valid_mask
    px = icgn_result.points_x
    py = icgn_result.points_y

    px_valid = px[valid]
    py_valid = py[valid]

    unique_x = np.unique(px_valid)
    unique_y = np.unique(py_valid)
    nx_grid, ny_grid = len(unique_x), len(unique_y)

    if nx_grid < 3 or ny_grid < 3:
        raise ValueError(f"그리드가 너무 작습니다: {nx_grid}×{ny_grid}")

    # 그리드 간격 추정
    grid_step = float(np.median(np.diff(unique_x))) if len(unique_x) > 1 else 1.0

    # 좌표 → 인덱스 매핑
    x_to_idx = {x: i for i, x in enumerate(unique_x)}
    y_to_idx = {y: i for i, y in enumerate(unique_y)}

    # 변위를 2D 그리드로 변환
    disp_u_grid = np.full((ny_grid, nx_grid), np.nan)
    disp_v_grid = np.full((ny_grid, nx_grid), np.nan)

    for i in range(len(px)):
        if valid[i]:
            xi = x_to_idx.get(px[i])
            yi = y_to_idx.get(py[i])
            if xi is not None and yi is not None:
                disp_u_grid[yi, xi] = icgn_result.disp_u[i]
                disp_v_grid[yi, xi] = icgn_result.disp_v[i]

    # PLS 적용
    result = compute_strain_pls(
        disp_u_grid, disp_v_grid,
        window_size=window_size,
        poly_order=poly_order,
        grid_step=grid_step,
        strain_type=strain_type
    )

    # 그리드 좌표 저장
    result.grid_x = unique_x
    result.grid_y = unique_y

    return result


def get_vsg_size(window_size: int, step_size: int, subset_size: int) -> int:
    """
    Virtual Strain Gauge 크기 계산

    VSG = (window_size - 1) × step_size + subset_size
    """
    return (window_size - 1) * step_size + subset_size


# ===== 내부 유틸리티 =====

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
    return np.sqrt(exx**2 + eyy**2 - exx * eyy + 3 * exy**2)
