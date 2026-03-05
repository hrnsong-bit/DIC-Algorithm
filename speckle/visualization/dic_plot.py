"""
DIC 결과 논문급 시각화 모듈

matplotlib 기반 고품질 변위/변형률 필드 시각화.
GUI 캔버스와 독립적으로 별도 창 또는 파일로 출력.

Usage:
    from speckle.visualization.dic_plot import DICPlotter
    
    plotter = DICPlotter(icgn_result, ref_image=ref_img)
    plotter.plot_displacement()          # 변위장
    plotter.plot_strain(method='pls')    # PLS 변형률
    plotter.plot_all()                   # 전체 요약
    plotter.save_all("./output")         # 일괄 저장
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging

_logger = logging.getLogger(__name__)


class DICPlotter:
    """DIC 결과 시각화 클래스"""

    DEFAULT_STYLE = {
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    }

    CMAPS = {
        'displacement': 'RdBu_r',   # u field용 (빨강=양, 파랑=음) — 기존과 동일
        'displacement_v': 'RdBu',   # v field용 (파랑=양, 빨강=음) — 새로 추가
        'strain': 'RdBu_r',
        'magnitude': 'jet',          # hot → jet 으로 변경
        'zncc': 'viridis',
        'von_mises': 'inferno',
    }

    def __init__(self, icgn_result, ref_image: Optional[np.ndarray] = None,
                 dpi: int = 150, style: Optional[Dict] = None,
                 upsample: int = 4, weight_threshold: float = 0.5):
        """
        Args:
            icgn_result: ICGNResult 객체
            ref_image: 참조 이미지 (배경용, optional)
            dpi: 출력 해상도
            style: matplotlib rcParams 오버라이드
            upsample: 보간 업샘플 배율 (1이면 보간 없음)
            weight_threshold: NaN 경계 판정 가중치 임계값 (0~1)
        """
        self.result = icgn_result
        self.ref_image = self._to_gray(ref_image) if ref_image is not None else None
        self.dpi = dpi
        self.upsample = max(1, upsample)
        self.weight_threshold = weight_threshold

        self._style = {**self.DEFAULT_STYLE}
        if style:
            self._style.update(style)

        self._grid_base = self._build_grid_base()

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return None
        if len(img.shape) == 3:
            import cv2
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    # ===== 2배 해상도 그리드 구성 =====

    def _build_grid_base(self) -> Dict:
        """전체 POI 좌표 기반으로 그리드 매핑 정보를 구성"""
        r = self.result

        all_unique_x = np.unique(r.points_x)
        all_unique_y = np.unique(r.points_y)
        nx_orig = len(all_unique_x)
        ny_orig = len(all_unique_y)

        spacing = float(all_unique_x[1] - all_unique_x[0]) if nx_orig > 1 else 1.0

        x_to_idx = {int(x): i for i, x in enumerate(all_unique_x)}
        y_to_idx = {int(y): i for i, y in enumerate(all_unique_y)}

        half = spacing / 4.0
        nx2 = nx_orig * 2
        ny2 = ny_orig * 2

        unique_x_2x = np.empty(nx2, dtype=np.float64)
        unique_y_2x = np.empty(ny2, dtype=np.float64)
        for i in range(nx_orig):
            unique_x_2x[2 * i]     = all_unique_x[i] - half
            unique_x_2x[2 * i + 1] = all_unique_x[i] + half
        for i in range(ny_orig):
            unique_y_2x[2 * i]     = all_unique_y[i] - half
            unique_y_2x[2 * i + 1] = all_unique_y[i] + half

        return {
            'all_unique_x': all_unique_x,
            'all_unique_y': all_unique_y,
            'nx_orig': nx_orig,
            'ny_orig': ny_orig,
            'spacing': spacing,
            'x_to_idx': x_to_idx,
            'y_to_idx': y_to_idx,
            'nx2': nx2,
            'ny2': ny2,
            'unique_x_2x': unique_x_2x,
            'unique_y_2x': unique_y_2x,
        }

    def _to_grid_2x(self, values: np.ndarray) -> np.ndarray:
        """POI 값을 2배 해상도 그리드로 변환"""
        b = self._grid_base
        r = self.result
        grid = np.full((b['ny2'], b['nx2']), np.nan)

        for idx in range(r.n_points):
            if not r.valid_mask[idx]:
                continue
            ix = b['x_to_idx'].get(int(r.points_x[idx]))
            iy = b['y_to_idx'].get(int(r.points_y[idx]))
            if ix is None or iy is None:
                continue
            val = values[idx]
            grid[2 * iy,     2 * ix]     = val
            grid[2 * iy,     2 * ix + 1] = val
            grid[2 * iy + 1, 2 * ix]     = val
            grid[2 * iy + 1, 2 * ix + 1] = val

        adss = getattr(r, 'adss_result', None)
        if adss is not None and adss.n_sub > 0:
            sub_values = self._get_adss_sub_values(values, adss)
            if sub_values is not None:
                # 사각형: quarter → 단일 서브셀
                qt_to_single = {5: [(0, 0)], 6: [(0, 1)], 7: [(1, 0)], 8: [(1, 1)]}
                # 삼각형: quarter → 2개 서브셀
                qt_to_double = {
                    1: [(0, 0), (0, 1)],   # Q1 상: 좌상 + 우상
                    2: [(1, 0), (1, 1)],   # Q2 하: 좌하 + 우하
                    3: [(0, 0), (1, 0)],   # Q3 좌: 좌상 + 좌하
                    4: [(0, 1), (1, 1)],   # Q4 우: 우상 + 우하
                }

                for s in range(adss.n_sub):
                    ix = b['x_to_idx'].get(int(adss.points_x[s]))
                    iy = b['y_to_idx'].get(int(adss.points_y[s]))
                    if ix is None or iy is None:
                        continue
                    qt = int(adss.quarter_types[s])

                    if qt in qt_to_double:
                        cells = qt_to_double[qt]
                    elif qt in qt_to_single:
                        cells = qt_to_single[qt]
                    else:
                        continue

                    for dy, dx in cells:
                        grid[2 * iy + dy, 2 * ix + dx] = sub_values[s]

        return grid

    def _get_adss_sub_values(self, values: np.ndarray, adss) -> Optional[np.ndarray]:
        """현재 필드에 대응하는 ADSS sub-POI 값 추출"""
        r = self.result
        n_params = adss.parameters.shape[1]
        is_affine = (n_params <= 6)

        if values is r.disp_u:
            return adss.parameters[:, 0]
        elif values is r.disp_v:
            return adss.parameters[:, 3] if is_affine else adss.parameters[:, 6]
        elif hasattr(r, 'disp_ux') and values is r.disp_ux:
            return adss.parameters[:, 1]
        elif hasattr(r, 'disp_uy') and values is r.disp_uy:
            return adss.parameters[:, 2]
        elif hasattr(r, 'disp_vx') and values is r.disp_vx:
            return adss.parameters[:, 4] if is_affine else adss.parameters[:, 7]
        elif hasattr(r, 'disp_vy') and values is r.disp_vy:
            return adss.parameters[:, 5] if is_affine else adss.parameters[:, 8]
        else:
            return None

    # ===== 가중치 기반 bilinear 업샘플링 =====

    def _upsample_field(self, grid: np.ndarray, unique_x: np.ndarray,
                        unique_y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        가중치 기반 bilinear 업샘플링.
        
        NaN을 0으로 채운 값과 유효 마스크(1/0)를 동시에 보간.
        보간 후 weight > threshold인 영역만 유효, 나머지는 NaN.
        불연속 경계에서 번짐 없이 유효 영역만 부드럽게 처리.
        
        Returns:
            (val_up, ux_up, uy_up): 업샘플된 값, x좌표, y좌표
        """
        if self.upsample <= 1 or grid.shape[0] < 3 or grid.shape[1] < 3:
            return grid.copy(), unique_x.copy(), unique_y.copy()

        nan_mask = np.isnan(grid)
        grid_filled = np.where(nan_mask, 0.0, grid)
        weight = (~nan_mask).astype(np.float64)

        yi_orig = np.arange(grid.shape[0], dtype=np.float64)
        xi_orig = np.arange(grid.shape[1], dtype=np.float64)

        interp_val = RegularGridInterpolator(
            (yi_orig, xi_orig), grid_filled, method='linear',
            bounds_error=False, fill_value=0.0
        )
        interp_w = RegularGridInterpolator(
            (yi_orig, xi_orig), weight, method='linear',
            bounds_error=False, fill_value=0.0
        )

        ny_up = (grid.shape[0] - 1) * self.upsample + 1
        nx_up = (grid.shape[1] - 1) * self.upsample + 1
        yi_new = np.linspace(0, grid.shape[0] - 1, ny_up)
        xi_new = np.linspace(0, grid.shape[1] - 1, nx_up)
        yy, xx = np.meshgrid(yi_new, xi_new, indexing='ij')
        pts = np.column_stack([yy.ravel(), xx.ravel()])

        val_up = interp_val(pts).reshape(ny_up, nx_up)
        w_up = interp_w(pts).reshape(ny_up, nx_up)

        valid_up = w_up > self.weight_threshold
        val_up[valid_up] /= w_up[valid_up]
        val_up[~valid_up] = np.nan

        ux_up = np.linspace(float(unique_x[0]), float(unique_x[-1]), nx_up)
        uy_up = np.linspace(float(unique_y[0]), float(unique_y[-1]), ny_up)

        return val_up, ux_up, uy_up

    def _make_edges(self, coords: np.ndarray) -> np.ndarray:
        """좌표 배열로부터 pcolormesh용 edge 배열 생성"""
        if len(coords) < 2:
            half = 0.5
        else:
            half = (coords[1] - coords[0]) / 2.0
        edges = np.concatenate([
            [coords[0] - half],
            (coords[:-1] + coords[1:]) / 2.0,
            [coords[-1] + half]
        ])
        return edges

    # ===== 렌더링 =====

    def _add_colorbar(self, fig, ax, im, label: str):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=10)
        cb.ax.tick_params(labelsize=9)
        return cb

    def _draw_field(self, ax, field_2d: np.ndarray, title: str, label: str,
                    cmap: str, symmetric: bool = True, fig=None,
                    vmin: float = None, vmax: float = None):
        """단일 필드: 2x 그리드 → bilinear 업샘플 → pcolormesh flat"""
        b = self._grid_base

        # 1) 가중치 기반 bilinear 업샘플링
        val_up, ux_up, uy_up = self._upsample_field(
            field_2d, b['unique_x_2x'], b['unique_y_2x']
        )

        valid_vals = val_up[~np.isnan(val_up)]
        if len(valid_vals) == 0:
            ax.set_title(title)
            return

        # 2) 값 범위
        if vmin is None or vmax is None:
            p1, p99 = np.percentile(valid_vals, [1, 99])
            if symmetric:
                v_abs = max(abs(p1), abs(p99))
                if v_abs < 1e-15:
                    v_abs = 1e-6
                vmin, vmax = -v_abs, v_abs
            else:
                vmin, vmax = p1, p99
                if abs(vmax - vmin) < 1e-15:
                    vmin, vmax = vmin - 1e-6, vmax + 1e-6

        if symmetric:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        # 3) 업샘플된 좌표에 맞는 edge 생성
        x_edges = self._make_edges(ux_up)
        y_edges = self._make_edges(uy_up)
        X_edges, Y_edges = np.meshgrid(x_edges, y_edges)

        # 4) pcolormesh flat
        im = ax.pcolormesh(X_edges, Y_edges, val_up,
                           cmap=cmap, norm=norm, shading='flat')
        ax.set_aspect('equal')
        ax.invert_yaxis()

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

        if fig is not None:
            self._add_colorbar(fig, ax, im, label)

    # ===== 공개 API =====

    def plot_displacement(self, show: bool = True, save_path: Optional[str] = None):
        with plt.rc_context(self._style):
            fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=self.dpi)
            fig.suptitle("Displacement Field", fontsize=15, fontweight='bold', y=1.02)

            r = self.result
            u_grid = self._to_grid_2x(r.disp_u)
            v_grid = self._to_grid_2x(r.disp_v)
            mag_grid = self._build_magnitude_grid(u_grid, v_grid)

            # --- u field: 최소값이 0이면 symmetric=False 사용 ---
            u_valid = u_grid[~np.isnan(u_grid)]
            u_min = np.min(u_valid) if len(u_valid) > 0 else 0
            u_symmetric = u_min < -1e-6  # 음수가 있으면 symmetric

            self._draw_field(axes[0], u_grid,
                            f"u field (max |u| = {np.nanmax(np.abs(u_grid)):.2f} px)",
                            "u (px)",
                            self.CMAPS['displacement'],
                            symmetric=u_symmetric, fig=fig)

            # --- v field: RdBu (reverse 없음) → 파랑=양, 빨강=음 ---
            self._draw_field(axes[1], v_grid,
                            f"v field (max |v| = {np.nanmax(np.abs(v_grid)):.2f} px)",
                            "v (px)",
                            self.CMAPS['displacement_v'],
                            symmetric=True, fig=fig)

            # --- magnitude: jet, non-symmetric ---
            self._draw_field(axes[2], mag_grid,
                            f"Magnitude (max = {np.nanmax(mag_grid):.2f} px)",
                            "|D| (px)",
                            self.CMAPS['magnitude'],
                            symmetric=False, fig=fig)

            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                _logger.info(f"변위 필드 저장: {save_path}")
            if show:
                plt.show()
            else:
                plt.close(fig)


    def _build_magnitude_grid(self, u_grid: np.ndarray, v_grid: np.ndarray) -> np.ndarray:
        mag = np.sqrt(np.where(np.isnan(u_grid), 0, u_grid)**2 +
                      np.where(np.isnan(v_grid), 0, v_grid)**2)
        mag[np.isnan(u_grid) & np.isnan(v_grid)] = np.nan
        return mag

    def plot_strain(self, method: str = 'pls', window_size: int = 15,
                    poly_order: int = 2, strain_type: str = 'engineering',
                    show: bool = True, save_path: Optional[str] = None):
        with plt.rc_context(self._style):
            fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=self.dpi)
            fig.suptitle(f"Strain Field ({method.upper()}, {strain_type})",
                         fontsize=15, fontweight='bold', y=1.02)

            if method == 'pls':
                strain = self._compute_pls_strain(window_size, poly_order, strain_type)
                exx, eyy, exy = strain.exx, strain.eyy, strain.exy
                e1, e2, von_mises = strain.e1, strain.e2, strain.von_mises
            elif method == 'direct':
                r = self.result
                exx = self._to_grid_2x(r.disp_ux)
                eyy = self._to_grid_2x(r.disp_vy)
                exy = 0.5 * (self._to_grid_2x(r.disp_uy) +
                              self._to_grid_2x(r.disp_vx))
                e_mean = 0.5 * (exx + eyy)
                R = np.sqrt(((exx - eyy) / 2)**2 + exy**2)
                e1, e2 = e_mean + R, e_mean - R
                von_mises = np.sqrt(exx**2 + eyy**2 - exx*eyy + 3*exy**2)
            else:
                raise ValueError(f"Unknown method: {method}")

            cmap = self.CMAPS['strain']
            self._draw_field(axes[0, 0], exx, "εxx", "εxx", cmap, True, fig)
            self._draw_field(axes[0, 1], eyy, "εyy", "εyy", cmap, True, fig)
            self._draw_field(axes[0, 2], exy, "εxy", "εxy", cmap, True, fig)
            self._draw_field(axes[1, 0], e1, "ε₁ (max principal)", "ε₁", cmap, True, fig)
            self._draw_field(axes[1, 1], e2, "ε₂ (min principal)", "ε₂", cmap, True, fig)
            self._draw_field(axes[1, 2], von_mises, "von Mises", "εᵥₘ",
                             self.CMAPS['von_mises'], False, fig)

            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                _logger.info(f"변형률 필드 저장: {save_path}")
            if show:
                plt.show()
            else:
                plt.close(fig)

    def plot_zncc(self, show: bool = True, save_path: Optional[str] = None):
        with plt.rc_context(self._style):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
            fig.suptitle("ZNCC Quality Map", fontsize=15, fontweight='bold', y=1.02)

            zncc_grid = self._to_grid_2x(self.result.zncc_values)

            self._draw_field(axes[0], zncc_grid, "ZNCC Map", "ZNCC",
                             self.CMAPS['zncc'], symmetric=False, fig=fig,
                             vmin=0.0, vmax=1.0)

            valid_zncc = self.result.zncc_values[self.result.valid_mask]
            axes[1].hist(valid_zncc, bins=50, color='steelblue', edgecolor='white',
                         alpha=0.8)
            axes[1].axvline(np.mean(valid_zncc), color='red', linestyle='--',
                            label=f'mean = {np.mean(valid_zncc):.4f}')
            axes[1].set_xlabel("ZNCC")
            axes[1].set_ylabel("Count")
            axes[1].set_title("ZNCC Distribution", fontweight='bold')
            axes[1].legend()

            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
            if show:
                plt.show()
            else:
                plt.close(fig)

    def plot_all(self, method: str = 'pls', window_size: int = 15,
                 poly_order: int = 2, show: bool = True,
                 save_dir: Optional[str] = None):
        save_dir = Path(save_dir) if save_dir else None
        if save_dir:
            save_dir.mkdir(parents=True, exist_ok=True)

        self.plot_displacement(
            show=show,
            save_path=str(save_dir / "displacement.png") if save_dir else None
        )
        self.plot_strain(
            method=method, window_size=window_size, poly_order=poly_order,
            show=show,
            save_path=str(save_dir / "strain.png") if save_dir else None
        )
        self.plot_zncc(
            show=show,
            save_path=str(save_dir / "zncc.png") if save_dir else None
        )

    def save_all(self, output_dir: str, method: str = 'pls',
                 window_size: int = 15, poly_order: int = 2,
                 fmt: str = 'png'):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        self.plot_displacement(show=False, save_path=str(out / f"displacement.{fmt}"))
        self.plot_strain(method=method, window_size=window_size,
                         poly_order=poly_order, show=False,
                         save_path=str(out / f"strain_{method}.{fmt}"))
        self.plot_zncc(show=False, save_path=str(out / f"zncc.{fmt}"))

        _logger.info(f"전체 결과 저장 완료: {out}")

    # ===== 내부 유틸리티 =====

    def _compute_pls_strain(self, window_size, poly_order, strain_type):
        from speckle.core.postprocess.strain_pls import compute_strain_pls

        b = self._grid_base
        u_grid = self._to_grid_2x(self.result.disp_u)
        v_grid = self._to_grid_2x(self.result.disp_v)

        grid_step_2x = b['spacing'] / 2.0

        return compute_strain_pls(
            u_grid, v_grid,
            window_size=window_size,
            poly_order=poly_order,
            grid_step=grid_step_2x,
            strain_type=strain_type
        )
