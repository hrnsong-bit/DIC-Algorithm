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
from pathlib import Path
from typing import Optional, Tuple, Dict
import logging

_logger = logging.getLogger(__name__)


class DICPlotter:
    """DIC 결과 시각화 클래스"""

    # 기본 스타일 설정
    DEFAULT_STYLE = {
        'figure.facecolor': 'white',
        'axes.facecolor': 'white',
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
    }

    # colormap 프리셋
    CMAPS = {
        'displacement': 'RdBu_r',      # 변위: 양/음 구분
        'strain': 'RdBu_r',            # 변형률: 양/음 구분
        'magnitude': 'hot',             # 크기: 단조증가
        'zncc': 'viridis',             # 상관계수: 0~1
        'von_mises': 'inferno',        # von Mises: 양수만
    }

    def __init__(self, icgn_result, ref_image: Optional[np.ndarray] = None,
                 dpi: int = 150, style: Optional[Dict] = None):
        """
        Args:
            icgn_result: ICGNResult 객체
            ref_image: 참조 이미지 (배경용, optional)
            dpi: 출력 해상도
            style: matplotlib rcParams 오버라이드
        """
        self.result = icgn_result
        self.ref_image = self._to_gray(ref_image) if ref_image is not None else None
        self.dpi = dpi

        # 스타일 적용
        self._style = {**self.DEFAULT_STYLE}
        if style:
            self._style.update(style)

        # 그리드 정보 사전 계산
        self._grid_info = self._build_grid()

    def _to_gray(self, img: np.ndarray) -> np.ndarray:
        if img is None:
            return None
        if len(img.shape) == 3:
            import cv2
            return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img

    def _build_grid(self) -> Dict:
        """POI 데이터를 2D 그리드로 변환"""
        r = self.result
        valid = r.valid_mask
        px, py = r.points_x[valid], r.points_y[valid]

        unique_x = np.unique(px)
        unique_y = np.unique(py)
        nx, ny = len(unique_x), len(unique_y)

        spacing = float(np.median(np.diff(unique_x))) if nx > 1 else 1.0

        x_to_idx = {x: i for i, x in enumerate(unique_x)}
        y_to_idx = {y: i for i, y in enumerate(unique_y)}

        def to_grid(values):
            grid = np.full((ny, nx), np.nan)
            v_valid = values[valid]
            for i, (x, y) in enumerate(zip(px, py)):
                xi, yi = x_to_idx.get(x), y_to_idx.get(y)
                if xi is not None and yi is not None:
                    grid[yi, xi] = v_valid[i]
            return grid

        return {
            'unique_x': unique_x, 'unique_y': unique_y,
            'nx': nx, 'ny': ny, 'spacing': spacing,
            'to_grid': to_grid,
            'extent': [unique_x[0], unique_x[-1], unique_y[-1], unique_y[0]]
        }

    def _add_colorbar(self, fig, ax, im, label: str):
        """축 옆에 정렬된 colorbar 추가"""
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.08)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=10)
        cb.ax.tick_params(labelsize=9)
        return cb

    def _draw_field(self, ax, field_2d: np.ndarray, title: str, label: str,
                    cmap: str, symmetric: bool = True, fig=None,
                    vmin: float = None, vmax: float = None):
        """단일 필드를 축에 그리기"""
        extent = self._grid_info['extent']

        # 배경 이미지
        if self.ref_image is not None:
            ax.imshow(self.ref_image, cmap='gray', alpha=0.3,
                      extent=[0, self.ref_image.shape[1],
                              self.ref_image.shape[0], 0])

        # 값 범위 결정
        valid_vals = field_2d[~np.isnan(field_2d)]
        if len(valid_vals) == 0:
            ax.set_title(title)
            return

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

        # 정규화
        if symmetric:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        im = ax.imshow(field_2d, cmap=cmap, norm=norm, extent=extent,
                       interpolation='bicubic', aspect='equal')

        ax.set_title(title, fontweight='bold')
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")

        if fig is not None:
            self._add_colorbar(fig, ax, im, label)

    # ===== 공개 API =====

    def plot_displacement(self, show: bool = True, save_path: Optional[str] = None):
        """변위 필드 시각화 (U, V, |D|)"""
        with plt.rc_context(self._style):
            fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=self.dpi)
            fig.suptitle("Displacement Field", fontsize=15, fontweight='bold', y=1.02)

            g = self._grid_info
            u_grid = g['to_grid'](self.result.disp_u)
            v_grid = g['to_grid'](self.result.disp_v)
            mag_grid = np.sqrt(np.where(np.isnan(u_grid), 0, u_grid)**2 +
                               np.where(np.isnan(v_grid), 0, v_grid)**2)
            mag_grid[np.isnan(u_grid) & np.isnan(v_grid)] = np.nan

            self._draw_field(axes[0], u_grid, "U (horizontal)", "U (px)",
                             self.CMAPS['displacement'], symmetric=True, fig=fig)
            self._draw_field(axes[1], v_grid, "V (vertical)", "V (px)",
                             self.CMAPS['displacement'], symmetric=True, fig=fig)
            self._draw_field(axes[2], mag_grid, "|D| (magnitude)", "|D| (px)",
                             self.CMAPS['magnitude'], symmetric=False, fig=fig)

            fig.tight_layout()
            if save_path:
                fig.savefig(save_path, dpi=self.dpi, bbox_inches='tight',
                            facecolor='white', edgecolor='none')
                _logger.info(f"변위 필드 저장: {save_path}")
            if show:
                plt.show()
            else:
                plt.close(fig)

    def plot_strain(self, method: str = 'pls', window_size: int = 15,
                    poly_order: int = 2, strain_type: str = 'engineering',
                    show: bool = True, save_path: Optional[str] = None):
        """
        변형률 필드 시각화

        Args:
            method: 'pls' (Pointwise Least Squares) 또는 'direct' (IC-GN gradient)
            window_size: PLS 윈도우 크기
            poly_order: PLS 다항식 차수
            strain_type: 'engineering' 또는 'green-lagrange'
        """
        with plt.rc_context(self._style):
            fig, axes = plt.subplots(2, 3, figsize=(16, 10), dpi=self.dpi)
            fig.suptitle(f"Strain Field ({method.upper()}, {strain_type})",
                         fontsize=15, fontweight='bold', y=1.02)

            if method == 'pls':
                strain = self._compute_pls_strain(window_size, poly_order, strain_type)
                exx, eyy, exy = strain.exx, strain.eyy, strain.exy
                e1, e2, von_mises = strain.e1, strain.e2, strain.von_mises
            elif method == 'direct':
                g = self._grid_info
                exx = g['to_grid'](self.result.disp_ux)
                eyy = g['to_grid'](self.result.disp_vy)
                exy_raw = 0.5 * (g['to_grid'](self.result.disp_uy) +
                                  g['to_grid'](self.result.disp_vx))
                exy = exy_raw
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
        """ZNCC 분포 시각화"""
        with plt.rc_context(self._style):
            fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=self.dpi)
            fig.suptitle("ZNCC Quality Map", fontsize=15, fontweight='bold', y=1.02)

            g = self._grid_info
            zncc_grid = g['to_grid'](self.result.zncc_values)

            # 맵
            self._draw_field(axes[0], zncc_grid, "ZNCC Map", "ZNCC",
                             self.CMAPS['zncc'], symmetric=False, fig=fig,
                             vmin=0.0, vmax=1.0)

            # 히스토그램
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
        """전체 결과 일괄 시각화"""
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
        """전체 결과 파일로 저장 (창 표시 없음)"""
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
        """PLS 변형률 계산"""
        from speckle.core.postprocess.strain_pls import compute_strain_pls

        g = self._grid_info
        u_grid = g['to_grid'](self.result.disp_u)
        v_grid = g['to_grid'](self.result.disp_v)

        return compute_strain_pls(
            u_grid, v_grid,
            window_size=window_size,
            poly_order=poly_order,
            grid_step=g['spacing'],
            strain_type=strain_type
        )
