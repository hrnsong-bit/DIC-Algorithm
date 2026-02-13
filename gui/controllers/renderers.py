"""
DIC 결과 시각화 렌더링 모듈

matplotlib Agg 백엔드로 고품질 필드 이미지를 렌더링하여
numpy 배열로 반환. GUI 캔버스에 직접 표시.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class FieldRenderer:
    """변위/변형률 필드 렌더링 담당"""

    def __init__(self, ctrl):
        self.ctrl = ctrl

    @property
    def view(self):
        return self.ctrl.view

    def create_overlay_image(self, base_img: np.ndarray, result) -> np.ndarray:
        """표시 모드에 따라 오버레이 이미지 생성"""
        if result is None:
            return base_img

        if len(base_img.shape) == 2:
            display_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            display_img = base_img.copy()

        mode = self.view.display_mode_var.get()

        render_map = {
            "vectors":   lambda img: self._draw_vectors(img, result),
            "u_field":   lambda img: self._draw_scalar_field(img, result, 'u'),
            "v_field":   lambda img: self._draw_scalar_field(img, result, 'v'),
            "magnitude": lambda img: self._draw_magnitude(img, result),
            "exx":       lambda img: self._draw_strain_field(img, result, 'exx'),
            "eyy":       lambda img: self._draw_strain_field(img, result, 'eyy'),
            "exy":       lambda img: self._draw_strain_field(img, result, 'exy'),
            "e1":        lambda img: self._draw_strain_field(img, result, 'e1'),
            "von_mises": lambda img: self._draw_strain_field(img, result, 'von_mises'),
            "zncc":      lambda img: self._draw_zncc_map(img, result),
            "invalid":   lambda img: self._draw_invalid_points(img, result),
        }

        renderer = render_map.get(mode)
        if renderer:
            display_img = renderer(display_img)

        return display_img

    # ===== matplotlib 기반 렌더링 =====

    def _render_field_matplotlib(self, img: np.ndarray, grid: np.ndarray,
                                  unique_x: np.ndarray, unique_y: np.ndarray,
                                  label: str, cmap: str, symmetric: bool,
                                  vmin: float = None, vmax: float = None) -> np.ndarray:
        """matplotlib Agg 백엔드로 필드를 렌더링하여 numpy 배열 반환"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm, Normalize
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        valid_vals = grid[~np.isnan(grid)]
        if len(valid_vals) == 0:
            return img

        # 값 범위
        color_range = self.view.get_color_range()
        if color_range is not None:
            vmin, vmax = color_range
        elif vmin is None or vmax is None:
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

        img_h, img_w = img.shape[:2]
        fig = plt.figure(figsize=(img_w / 100, img_h / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])

        # 배경
        if len(img.shape) == 3:
            gray_bg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_bg = img.copy()
        ax.imshow(gray_bg, cmap='gray', extent=[0, img_w, img_h, 0], alpha=0.4)

        # norm & colormap
        if symmetric:
            norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        else:
            norm = Normalize(vmin=vmin, vmax=vmax)

        extent = [float(unique_x[0]), float(unique_x[-1]),
                  float(unique_y[-1]), float(unique_y[0])]

        im = ax.imshow(grid, cmap=cmap, norm=norm, extent=extent,
                       interpolation='bicubic', alpha=0.75, aspect='auto')

        # colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="3%", pad=0.05)
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(label, fontsize=max(8, int(img_h / 80)))
        cb.ax.tick_params(labelsize=max(6, int(img_h / 100)))

        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.axis('off')

        # figure → numpy
        canvas_agg = FigureCanvasAgg(fig)
        canvas_agg.draw()
        rendered = np.asarray(canvas_agg.buffer_rgba())
        plt.close(fig)

        rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGBA2BGR)
        if rendered_bgr.shape[:2] != (img_h, img_w):
            rendered_bgr = cv2.resize(rendered_bgr, (img_w, img_h),
                                       interpolation=cv2.INTER_LINEAR)

        cmap_type = 'diverging' if symmetric else 'sequential'
        self.view.update_colorbar(vmin, vmax, label, cmap_type)

        return rendered_bgr

    # ===== 유틸리티: POI → 2D 그리드 =====

    def _to_grid(self, result, values):
        """POI 값을 2D 그리드로 변환"""
        valid = result.valid_mask
        px, py = result.points_x[valid], result.points_y[valid]

        unique_x = np.unique(px)
        unique_y = np.unique(py)
        nx, ny = len(unique_x), len(unique_y)

        x_to_idx = {x: i for i, x in enumerate(unique_x)}
        y_to_idx = {y: i for i, y in enumerate(unique_y)}

        grid = np.full((ny, nx), np.nan)
        v_valid = values[valid]
        for i, (x, y) in enumerate(zip(px, py)):
            xi = x_to_idx.get(x)
            yi = y_to_idx.get(y)
            if xi is not None and yi is not None:
                grid[yi, xi] = v_valid[i]

        return grid, unique_x, unique_y

    # ===== 개별 렌더러 =====

    def _draw_scalar_field(self, img, result, field_type):
        """변위 필드 시각화"""
        if result.n_points == 0:
            return img

        if field_type == 'u':
            values, label = result.disp_u, "U (px)"
        elif field_type == 'v':
            values, label = result.disp_v, "V (px)"
        else:
            values = np.sqrt(result.disp_u**2 + result.disp_v**2)
            label = "|D| (px)"

        grid, ux, uy = self._to_grid(result, values)
        if grid.size == 0 or len(ux) < 3 or len(uy) < 3:
            return img

        symmetric = field_type in ('u', 'v')
        cmap = 'RdBu_r' if symmetric else 'hot'
        return self._render_field_matplotlib(img, grid, ux, uy, label, cmap, symmetric)

    def _draw_strain_field(self, img, result, strain_type):
        """변형률 필드 시각화 (PLS)"""
        if result.n_points == 0:
            return img

        is_icgn = hasattr(result, 'disp_ux') and result.disp_ux is not None
        if not is_icgn:
            cv2.putText(img, "IC-GN 결과 필요", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return img

        valid = result.valid_mask
        px_valid = result.points_x[valid]
        unique_x = np.unique(px_valid)
        unique_y = np.unique(result.points_y[valid])

        if len(unique_x) < 5 or len(unique_y) < 5:
            return self._draw_strain_field_points(img, result, strain_type)

        grid_step = float(np.median(np.diff(unique_x))) if len(unique_x) > 1 else 1.0

        u_grid, _, _ = self._to_grid(result, result.disp_u)
        v_grid, _, _ = self._to_grid(result, result.disp_v)

        try:
            from speckle.core.postprocess.strain_pls import compute_strain_pls
            strain = compute_strain_pls(
                u_grid, v_grid,
                window_size=11, poly_order=2, grid_step=grid_step
            )
        except Exception as e:
            logger.warning(f"PLS 실패: {e}")
            return self._draw_strain_field_points(img, result, strain_type)

        field_map = {
            'exx': (strain.exx, "εxx"),
            'eyy': (strain.eyy, "εyy"),
            'exy': (strain.exy, "εxy"),
            'e1':  (strain.e1, "ε₁"),
            'von_mises': (strain.von_mises, "εᵥₘ")
        }

        if strain_type not in field_map:
            return img

        strain_2d, label = field_map[strain_type]
        symmetric = strain_type in ('exx', 'eyy', 'exy', 'e1')
        cmap = 'RdBu_r' if symmetric else 'inferno'

        return self._render_field_matplotlib(img, strain_2d, unique_x, unique_y,
                                              label, cmap, symmetric)

    def _draw_strain_field_points(self, img, result, strain_type):
        """변형률 필드 점 시각화 (PLS 실패 시 fallback)"""
        from speckle.core.postprocess import compute_strain_from_icgn
        strain = compute_strain_from_icgn(result)

        field_map = {
            'exx': (strain.exx, "εxx"),
            'eyy': (strain.eyy, "εyy"),
            'exy': (strain.exy, "εxy"),
            'e1':  (strain.e1, "ε₁"),
            'von_mises': (strain.von_mises, "εᵥₘ")
        }

        if strain_type not in field_map:
            return img

        values, label = field_map[strain_type]
        valid = strain.valid_mask
        valid_values = values[valid]

        if len(valid_values) == 0:
            return img

        vmin, vmax = np.min(valid_values), np.max(valid_values)
        v_abs = max(abs(vmin), abs(vmax))
        if v_abs < 1e-10:
            v_abs = 1e-6
        vmin, vmax = -v_abs, v_abs

        self.view.update_colorbar(vmin, vmax, label, 'diverging')

        for idx in range(result.n_points):
            x = int(result.points_x[idx])
            y = int(result.points_y[idx])
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            if strain.valid_mask[idx]:
                norm_val = np.clip((values[idx] - vmin) / (vmax - vmin + 1e-10), 0, 1)
                color = self._diverging_colormap(norm_val)
                cv2.circle(img, (x, y), 4, color, -1)
            else:
                cv2.circle(img, (x, y), 4, (128, 128, 128), -1)

        return img

    def _draw_vectors(self, img, result):
        """변위 벡터 시각화"""
        if result.n_points == 0:
            return img

        valid = result.valid_mask
        scale = max(1.0, np.max(np.abs(result.disp_u[valid])**2 +
                                 np.abs(result.disp_v[valid])**2)**0.5)
        arrow_scale = 20.0 / max(scale, 1e-10)

        for idx in range(result.n_points):
            if not valid[idx]:
                continue
            x = int(result.points_x[idx])
            y = int(result.points_y[idx])
            dx = int(result.disp_u[idx] * arrow_scale)
            dy = int(result.disp_v[idx] * arrow_scale)

            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue

            cv2.arrowedLine(img, (x, y), (x + dx, y + dy),
                            (0, 255, 0), 1, tipLength=0.3)

        return img

    def _draw_magnitude(self, img, result):
        """변위 크기 시각화"""
        if result.n_points == 0:
            return img

        mag = np.sqrt(result.disp_u**2 + result.disp_v**2)
        grid, ux, uy = self._to_grid(result, mag)
        if grid.size == 0 or len(ux) < 3 or len(uy) < 3:
            return img

        return self._render_field_matplotlib(img, grid, ux, uy,
                                              "|D| (px)", 'hot', False)

    def _draw_zncc_map(self, img, result):
        """ZNCC 맵 시각화"""
        if result.n_points == 0:
            return img

        grid, ux, uy = self._to_grid(result, result.zncc_values)
        if grid.size == 0 or len(ux) < 3 or len(uy) < 3:
            return img

        return self._render_field_matplotlib(img, grid, ux, uy,
                                              "ZNCC", 'viridis', False,
                                              vmin=0.0, vmax=1.0)

    def _draw_invalid_points(self, img, result):
        """불량 POI 시각화"""
        if result.n_points == 0:
            return img

        for idx in range(result.n_points):
            x = int(result.points_x[idx])
            y = int(result.points_y[idx])
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue

            if result.valid_mask[idx]:
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            else:
                cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                cv2.drawMarker(img, (x, y), (0, 0, 255),
                               cv2.MARKER_CROSS, 10, 2)

        return img

    # ===== colormap (fallback용) =====

    @staticmethod
    def _diverging_colormap(norm_value: float) -> tuple:
        """파랑-흰-빨강 (BGR)"""
        if norm_value < 0.5:
            t = norm_value * 2
            r, g, b = int(t * 255), int(t * 255), 255
        else:
            t = (norm_value - 0.5) * 2
            r, g, b = 255, int((1 - t) * 255), int((1 - t) * 255)
        return (b, g, r)

    @staticmethod
    def _sequential_colormap(norm_value: float) -> tuple:
        """검정-빨강-노랑 (BGR)"""
        if norm_value < 0.5:
            t = norm_value * 2
            r, g, b = int(t * 255), 0, 0
        else:
            t = (norm_value - 0.5) * 2
            r, g, b = 255, int(t * 255), 0
        return (b, g, r)
