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
        self._strain_cache = {}
        self._strain_cache_key = None

    @property
    def view(self):
        return self.ctrl.view

    def _resolve_display_source(self, mode: str) -> str:
        """Resolve the active display source for the current render mode."""
        source_var = getattr(self.view, 'display_source_var', None)
        source = source_var.get() if source_var is not None else 'auto'
        if source not in {'auto', 'raw', 'processed'}:
            source = 'auto'

        if source != 'auto':
            return source

        if mode in {'u_field', 'v_field', 'magnitude'}:
            return 'raw'
        if mode in {'exx', 'eyy', 'exy', 'e1', 'von_mises'}:
            return 'processed'
        return 'raw'

    def create_overlay_image(self, base_img: np.ndarray, result, mode: str = None) -> np.ndarray:
        """표시 모드에 따라 오버레이 이미지 생성"""
        if result is None:
            return base_img

        if len(base_img.shape) == 2:
            display_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            display_img = base_img.copy()

        if mode is None:
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

    @staticmethod
    def _get_adss_sub_cells(quarter_type: int):
        """Map one ADSS quarter type to the 2x2 sub-cell indices it occupies."""
        if quarter_type == 1:
            return ((0, 0), (0, 1))
        if quarter_type == 2:
            return ((1, 0), (1, 1))
        if quarter_type == 3:
            return ((0, 0), (1, 0))
        if quarter_type == 4:
            return ((0, 1), (1, 1))
        if quarter_type == 5:
            return ((0, 0),)
        if quarter_type == 6:
            return ((0, 1),)
        if quarter_type == 7:
            return ((1, 0),)
        if quarter_type == 8:
            return ((1, 1),)
        return ()

    @staticmethod
    def _make_dim_background(img: np.ndarray) -> np.ndarray:
        """Match the dim grayscale background used by field renderers."""
        if len(img.shape) == 3:
            gray_bg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_bg = img.copy()
        overlay = (gray_bg.astype(np.float32) * 0.3).astype(np.uint8)
        return cv2.cvtColor(overlay, cv2.COLOR_GRAY2BGR)

    @staticmethod
    def _estimate_spacing(result) -> int:
        unique_x = np.unique(result.points_x)
        diffs = np.diff(unique_x)
        diffs = diffs[diffs > 0]
        if len(diffs) > 0:
            return int(round(np.median(diffs)))
        return int(getattr(result, 'spacing', 10))

    @staticmethod
    def _axis_bounds(center: int, spacing: int, limit: int) -> tuple[int, int]:
        """Return half-open pixel bounds for one cell axis."""
        left = spacing // 2
        right = spacing - left
        start = max(0, center - left)
        end = min(limit, center + right)
        return start, end

    @classmethod
    def _cell_bounds(cls, x: int, y: int, spacing: int,
                     width: int, height: int) -> tuple[int, int, int, int]:
        """Return half-open pixel bounds for a POI parent cell."""
        x0, x1 = cls._axis_bounds(x, spacing, width)
        y0, y1 = cls._axis_bounds(y, spacing, height)
        return x0, x1, y0, y1

    @staticmethod
    def _triangle_mask(x0: int, x1: int, y0: int, y1: int,
                       vertices: np.ndarray) -> np.ndarray:
        """Rasterize a triangle using pixel-center sampling within half-open bounds."""
        xs = np.arange(x0, x1, dtype=np.float32) + 0.5
        ys = np.arange(y0, y1, dtype=np.float32) + 0.5
        xx, yy = np.meshgrid(xs, ys)

        ax, ay = vertices[0]
        bx, by = vertices[1]
        cx, cy = vertices[2]

        d1 = (xx - bx) * (ay - by) - (ax - bx) * (yy - by)
        d2 = (xx - cx) * (by - cy) - (bx - cx) * (yy - cy)
        d3 = (xx - ax) * (cy - ay) - (cx - ax) * (yy - ay)

        has_neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
        has_pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
        return ~(has_neg & has_pos)

    @classmethod
    def _fill_adss_quarter(cls, mask: np.ndarray, x: int, y: int,
                           quarter_type: int, spacing: int):
        """Fill one ADSS quarter into a binary mask using the same cell bounds as raw fields."""
        h, w = mask.shape[:2]
        x0, x1, y0, y1 = cls._cell_bounds(x, y, spacing, w, h)
        if x1 <= x0 or y1 <= y0:
            return

        x_mid = min(max(x, x0), x1)
        y_mid = min(max(y, y0), y1)

        if 5 <= quarter_type <= 8:
            rect_bounds = {
                5: (x0, x_mid, y0, y_mid),
                6: (x_mid, x1, y0, y_mid),
                7: (x0, x_mid, y_mid, y1),
                8: (x_mid, x1, y_mid, y1),
            }
            rx0, rx1, ry0, ry1 = rect_bounds[quarter_type]
            if rx1 > rx0 and ry1 > ry0:
                mask[ry0:ry1, rx0:rx1] = 255
            return

        if quarter_type == 1:
            vertices = np.array([[x0, y0], [x1, y0], [x, y]], dtype=np.float32)
        elif quarter_type == 2:
            vertices = np.array([[x0, y1], [x1, y1], [x, y]], dtype=np.float32)
        elif quarter_type == 3:
            vertices = np.array([[x0, y0], [x0, y1], [x, y]], dtype=np.float32)
        elif quarter_type == 4:
            vertices = np.array([[x1, y0], [x1, y1], [x, y]], dtype=np.float32)
        else:
            return

        tri_mask = cls._triangle_mask(x0, x1, y0, y1, vertices)
        mask[y0:y1, x0:x1][tri_mask] = 255

    @staticmethod
    def _outline_mask(mask: np.ndarray) -> np.ndarray:
        """Return a 1px outline for a filled binary mask."""
        if mask.size == 0:
            return mask

        kernel = np.ones((3, 3), dtype=np.uint8)
        inner = cv2.erode(mask, kernel, iterations=1)
        outline = mask.copy()
        outline[inner > 0] = 0
        return outline

    def _apply_adss_display_mask(self, base_img: np.ndarray,
                                 rendered_img: np.ndarray, result) -> np.ndarray:
        """Clip ADSS parent cells so only recovered quarter polygons remain visible."""
        adss = getattr(result, 'adss_result', None)
        if adss is None or adss.n_sub == 0:
            return rendered_img

        h, w = rendered_img.shape[:2]
        parent_mask = np.zeros((h, w), dtype=np.uint8)
        keep_mask = np.zeros((h, w), dtype=np.uint8)
        spacing = self._estimate_spacing(result)

        for parent_idx in adss.unique_parents.tolist():
            x = int(result.points_x[parent_idx])
            y = int(result.points_y[parent_idx])
            x0, x1, y0, y1 = self._cell_bounds(x, y, spacing, w, h)
            if x1 > x0 and y1 > y0:
                parent_mask[y0:y1, x0:x1] = 255

        for s in range(adss.n_sub):
            x = int(adss.points_x[s])
            y = int(adss.points_y[s])
            quarter_type = int(adss.quarter_types[s])
            self._fill_adss_quarter(keep_mask, x, y, quarter_type, spacing)

        erase_mask = (parent_mask > 0) & (keep_mask == 0)
        if not np.any(erase_mask):
            return rendered_img

        output = rendered_img.copy()
        dim_bg = self._make_dim_background(base_img)
        output[erase_mask] = dim_bg[erase_mask]
        return output

    # ===== 새 렌더러: 서브셋 영역 직접 채색 =====

    def _render_field_subset(self, img: np.ndarray, result,
                              values: np.ndarray, label: str,
                              symmetric: bool,
                              adss_values: np.ndarray = None) -> np.ndarray:
        """각 POI의 값을 서브셋 영역에 직접 칠하는 렌더러

        보간 없음. ADSS 복구 POI는 quarter-type에 따라 절반만 채색.
        invalid 영역은 투명(배경 투과).
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        valid = result.valid_mask
        valid_values = values[valid]
        adss = getattr(result, 'adss_result', None)
        if adss_values is None and adss is not None and adss.n_sub > 0:
            adss_values = self._get_adss_sub_values(result, values, adss)
        if adss_values is not None and len(adss_values) > 0:
            valid_values = np.concatenate([
                valid_values,
                np.asarray(adss_values, dtype=np.float64)
            ])
        if len(valid_values) == 0:
            return img

        # === 값 범위 결정 ===
        color_range = self.view.get_color_range()
        if color_range is not None:
            vmin, vmax = color_range
        else:
            p2, p98 = np.percentile(valid_values, [2, 98])
            if symmetric:
                v_abs = max(abs(p2), abs(p98))
                if v_abs < 1e-10:
                    v_abs = max(abs(np.min(valid_values)), abs(np.max(valid_values)))
                if v_abs < 1e-15:
                    v_abs = 1e-6
                vmin, vmax = -v_abs, v_abs
            else:
                vmin, vmax = p2, p98
                if abs(vmax - vmin) < 1e-15:
                    vmin, vmax = vmin - 1e-6, vmax + 1e-6

        # === spacing 추정 ===
        spacing = self._estimate_spacing(result)

        # === ADSS quarter-type 배열 ===
        adss_parent_set = set()
        if adss is not None and adss.n_sub > 0:
            adss_parent_set = set(adss.unique_parents.tolist())

        # === 배경 이미지 ===
        dim_bg = self._make_dim_background(img)
        overlay = dim_bg.astype(np.float32)

        img_h, img_w = img.shape[:2]
        cmap = plt.get_cmap('turbo')
        alpha = 0.9

        def blend_mask(mask: np.ndarray, color_bgr: np.ndarray):
            if np.any(mask):
                overlay[mask] = overlay[mask] * (1 - alpha) + color_bgr * alpha

        def color_for_value(val: float) -> np.ndarray:
            norm_val = np.clip((val - vmin) / (vmax - vmin + 1e-15), 0, 1)
            r, g, b, _ = cmap(norm_val)
            return np.array([b * 255, g * 255, r * 255], dtype=np.float32)

        # === 각 POI 서브셋 영역 채색 ===
        for idx in range(result.n_points):
            if not valid[idx] or idx in adss_parent_set:
                continue

            cx = int(result.points_x[idx])
            cy = int(result.points_y[idx])
            color_bgr = color_for_value(float(values[idx]))
            half_sp = spacing // 2

            # quarter-type에 따른 영역 결정
            x1 = max(0, cx - half_sp)
            x2 = min(img_w, cx + half_sp + 1)
            y1 = max(0, cy - half_sp)
            y2 = min(img_h, cy + half_sp + 1)
            if True:
                # 일반 POI: 전체 spacing × spacing
                x1 = max(0, cx - half_sp)
                x2 = min(img_w, cx + half_sp + 1)
                y1 = max(0, cy - half_sp)
                y2 = min(img_h, cy + half_sp + 1)
            elif qt == 1:  # Q1: Upper half (η: -M~0)
                x1 = max(0, cx - half_sp)
                x2 = min(img_w, cx + half_sp + 1)
                y1 = max(0, cy - half_sp)
                y2 = min(img_h, cy + 1)
            elif qt == 2:  # Q2: Lower half (η: 0~+M)
                x1 = max(0, cx - half_sp)
                x2 = min(img_w, cx + half_sp + 1)
                y1 = max(0, cy)
                y2 = min(img_h, cy + half_sp + 1)
            elif qt == 3:  # Q3: Left half (ξ: -M~0)
                x1 = max(0, cx - half_sp)
                x2 = min(img_w, cx + 1)
                y1 = max(0, cy - half_sp)
                y2 = min(img_h, cy + half_sp + 1)
            elif qt == 4:  # Q4: Right half (ξ: 0~+M)
                x1 = max(0, cx)
                x2 = min(img_w, cx + half_sp + 1)
                y1 = max(0, cy - half_sp)
                y2 = min(img_h, cy + half_sp + 1)
            elif qt == 5:  # Q5: Upper-left
                x1 = max(0, cx - half_sp)
                x2 = min(img_w, cx + 1)
                y1 = max(0, cy - half_sp)
                y2 = min(img_h, cy + 1)
            elif qt == 6:  # Q6: Upper-right
                x1 = max(0, cx)
                x2 = min(img_w, cx + half_sp + 1)
                y1 = max(0, cy - half_sp)
                y2 = min(img_h, cy + 1)
            elif qt == 7:  # Q7: Lower-left
                x1 = max(0, cx - half_sp)
                x2 = min(img_w, cx + 1)
                y1 = max(0, cy)
                y2 = min(img_h, cy + half_sp + 1)
            elif qt == 8:  # Q8: Lower-right
                x1 = max(0, cx)
                x2 = min(img_w, cx + half_sp + 1)
                y1 = max(0, cy)
                y2 = min(img_h, cy + half_sp + 1)
            else:
                x1 = max(0, cx - half_sp)
                x2 = min(img_w, cx + half_sp + 1)
                y1 = max(0, cy - half_sp)
                y2 = min(img_h, cy + half_sp + 1)

            x1, x2, y1, y2 = self._cell_bounds(cx, cy, spacing, img_w, img_h)
            if x2 <= x1 or y2 <= y1:
                continue

            overlay[y1:y2, x1:x2] = (
                overlay[y1:y2, x1:x2] * (1 - alpha) + color_bgr * alpha
            )

        if adss is not None and adss.n_sub > 0 and adss_values is not None:
            recovery_passes = adss.get_recovery_passes()
            pass2_outline_color = np.array([255, 0, 0], dtype=np.float32)
            for parent_idx in adss.unique_parents.tolist():
                cx = int(result.points_x[parent_idx])
                cy = int(result.points_y[parent_idx])
                x1 = max(0, cx - half_sp)
                x2 = min(img_w, cx + half_sp + 1)
                y1 = max(0, cy - half_sp)
                y2 = min(img_h, cy + half_sp + 1)
                x1, x2, y1, y2 = self._cell_bounds(cx, cy, spacing, img_w, img_h)
                if x2 > x1 and y2 > y1:
                    overlay[y1:y2, x1:x2] = dim_bg[y1:y2, x1:x2]

            for s in range(adss.n_sub):
                cx = int(adss.points_x[s])
                cy = int(adss.points_y[s])
                color_bgr = color_for_value(float(adss_values[s]))
                poly_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                self._fill_adss_quarter(poly_mask, cx, cy, int(adss.quarter_types[s]), spacing)
                blend_mask(poly_mask > 0, color_bgr)
                if int(recovery_passes[s]) == 2:
                    outline_mask = self._outline_mask(poly_mask) > 0
                    overlay[outline_mask] = pass2_outline_color

        result_img = np.clip(overlay, 0, 255).astype(np.uint8)
        self.view.update_colorbar(vmin, vmax, label, 'turbo')
        return result_img

    # ===== 기존 matplotlib 렌더러 (보간 포함 — 참고용 보존) =====

    def _render_field_matplotlib(self, img: np.ndarray, grid: np.ndarray,
                                unique_x: np.ndarray, unique_y: np.ndarray,
                                label: str, symmetric: bool,
                                vmin: float = None, vmax: float = None) -> np.ndarray:
        """matplotlib Agg 백엔드로 필드를 렌더링하여 numpy 배열 반환

        NaN 영역은 투명 처리하되, 유효 데이터는 부드럽게 업샘플링.
        방법: 가중치 기반 선형 보간 후 RGBA alpha 채널로 NaN 마스킹.
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.colors import Normalize
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from scipy.interpolate import RegularGridInterpolator

        valid_vals = grid[~np.isnan(grid)]
        if len(valid_vals) == 0:
            return img

        color_range = self.view.get_color_range()
        if color_range is not None:
            vmin, vmax = color_range
        elif vmin is None or vmax is None:
            p2, p98 = np.percentile(valid_vals, [2, 98])
            if symmetric:
                v_abs = max(abs(p2), abs(p98))
                if v_abs < 1e-10:
                    v_abs = max(abs(np.min(valid_vals)), abs(np.max(valid_vals)))
                if v_abs < 1e-15:
                    v_abs = 1e-6
                vmin, vmax = -v_abs, v_abs
            else:
                vmin, vmax = p2, p98
                if abs(vmax - vmin) < 1e-15:
                    vmin, vmax = vmin - 1e-6, vmax + 1e-6

        nan_mask = np.isnan(grid)
        grid_filled = np.where(nan_mask, 0.0, grid)
        weight = (~nan_mask).astype(np.float64)

        spacing = unique_x[1] - unique_x[0] if len(unique_x) > 1 else 1.0
        upsample = max(1, int(round(spacing)))

        if upsample > 1 and grid.shape[0] >= 3 and grid.shape[1] >= 3:
            yi_orig = np.arange(grid.shape[0]).astype(np.float64)
            xi_orig = np.arange(grid.shape[1]).astype(np.float64)

            interp_val = RegularGridInterpolator(
                (yi_orig, xi_orig), grid_filled, method='linear',
                bounds_error=False, fill_value=0.0
            )
            interp_w = RegularGridInterpolator(
                (yi_orig, xi_orig), weight, method='linear',
                bounds_error=False, fill_value=0.0
            )

            ny_up = (grid.shape[0] - 1) * upsample + 1
            nx_up = (grid.shape[1] - 1) * upsample + 1
            yi_new = np.linspace(0, grid.shape[0] - 1, ny_up)
            xi_new = np.linspace(0, grid.shape[1] - 1, nx_up)
            yy, xx = np.meshgrid(yi_new, xi_new, indexing='ij')
            pts = np.column_stack([yy.ravel(), xx.ravel()])

            val_up = interp_val(pts).reshape(ny_up, nx_up)
            w_up = interp_w(pts).reshape(ny_up, nx_up)

            valid_up = w_up > 0.3
            val_up[valid_up] /= w_up[valid_up]
            val_up[~valid_up] = np.nan

            ux_up = np.linspace(float(unique_x[0]), float(unique_x[-1]), nx_up)
            uy_up = np.linspace(float(unique_y[0]), float(unique_y[-1]), ny_up)
        else:
            val_up = grid.copy()
            ux_up = unique_x
            uy_up = unique_y

        cmap = plt.get_cmap('turbo')
        norm = Normalize(vmin=vmin, vmax=vmax)

        val_clipped = np.clip(val_up, vmin, vmax)
        val_safe = np.where(np.isnan(val_up), 0.0, val_up)
        val_normed = np.clip((val_safe - vmin) / (vmax - vmin + 1e-15), 0, 1)
        rgba = cmap(val_normed)

        nan_up = np.isnan(val_up)
        rgba[nan_up, 3] = 0.0
        rgba[~nan_up, 3] = 0.9

        img_h, img_w = img.shape[:2]
        fig = plt.figure(figsize=(img_w / 100, img_h / 100), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])

        if len(img.shape) == 3:
            gray_bg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_bg = img.copy()
        ax.imshow(gray_bg, cmap='gray', extent=[0, img_w, img_h, 0], alpha=0.3)

        extent = [float(ux_up[0]), float(ux_up[-1]),
                float(uy_up[-1]), float(uy_up[0])]

        ax.imshow(rgba, extent=extent, interpolation='bilinear', aspect='auto')

        ax.set_xlim(0, img_w)
        ax.set_ylim(img_h, 0)
        ax.axis('off')

        canvas_agg = FigureCanvasAgg(fig)
        canvas_agg.draw()
        rendered = np.asarray(canvas_agg.buffer_rgba())
        plt.close(fig)

        rendered_bgr = cv2.cvtColor(rendered, cv2.COLOR_RGBA2BGR)
        if rendered_bgr.shape[:2] != (img_h, img_w):
            rendered_bgr = cv2.resize(rendered_bgr, (img_w, img_h),
                                    interpolation=cv2.INTER_LINEAR)

        self.view.update_colorbar(vmin, vmax, label, 'turbo')

        return rendered_bgr

    # ===== 유틸리티: POI → 2D 그리드 =====

    def _to_grid(self, result, values):
        """POI 값을 2D 그리드로 변환.
        
        해상도를 2배로 올려서 각 POI 셀을 2×2 서브셀로 분할.
        - 정상 POI: 4개 서브셀 모두 같은 값
        - ADSS POI: 복원된 사분면에 해당하는 서브셀만 값 배치
        """
        all_unique_x = np.unique(result.points_x)
        all_unique_y = np.unique(result.points_y)
        nx_orig = len(all_unique_x)
        ny_orig = len(all_unique_y)

        if nx_orig < 2 or ny_orig < 2:
            return np.full((1, 1), np.nan), all_unique_x, all_unique_y

        spacing = float(all_unique_x[1] - all_unique_x[0]) if nx_orig > 1 else 1.0

        # 2배 해상도 격자
        nx2 = nx_orig * 2
        ny2 = ny_orig * 2
        grid = np.full((ny2, nx2), np.nan)

        # 2배 해상도 좌표 (서브셀 중심)
        half = spacing / 4.0
        unique_x_2x = np.empty(nx2, dtype=np.float64)
        unique_y_2x = np.empty(ny2, dtype=np.float64)
        for i in range(nx_orig):
            unique_x_2x[2 * i]     = all_unique_x[i] - half
            unique_x_2x[2 * i + 1] = all_unique_x[i] + half
        for i in range(ny_orig):
            unique_y_2x[2 * i]     = all_unique_y[i] - half
            unique_y_2x[2 * i + 1] = all_unique_y[i] + half

        # 원래 좌표 → 인덱스 매핑
        x_to_idx = {}
        for i, x in enumerate(all_unique_x):
            x_to_idx[int(x)] = i
        y_to_idx = {}
        for i, y in enumerate(all_unique_y):
            y_to_idx[int(y)] = i

        # ADSS로 복원된 sub-POI 정보
        adss = getattr(result, 'adss_result', None)

        # === 정상 POI: 4개 서브셀 모두 같은 값 ===
        valid = result.valid_mask
        for idx in range(result.n_points):
            if not valid[idx]:
                # ADSS 부모가 아닌 불량 POI → NaN 유지
                continue

            px_val = int(result.points_x[idx])
            py_val = int(result.points_y[idx])
            ix = x_to_idx.get(px_val)
            iy = y_to_idx.get(py_val)
            if ix is None or iy is None:
                continue

            val = values[idx]
            # 2×2 서브셀 모두 채움
            grid[2 * iy,     2 * ix]     = val  # 좌상
            grid[2 * iy,     2 * ix + 1] = val  # 우상
            grid[2 * iy + 1, 2 * ix]     = val  # 좌하
            grid[2 * iy + 1, 2 * ix + 1] = val  # 우하

        # === ADSS sub-POI: 해당 사분면 서브셀만 ===
        if adss is not None and adss.n_sub > 0:
            sub_values = self._get_adss_sub_values(result, values, adss)
            if sub_values is not None:
                # 사각형: quarter → 단일 서브셀
                qt_to_single = {5: [(0, 0)], 6: [(0, 1)], 7: [(1, 0)], 8: [(1, 1)]}
                # 삼각형: quarter → 2개 서브셀 (영역이 2×2의 절반을 차지)
                qt_to_double = {
                    1: [(0, 0), (0, 1)],   # Q1 상: 좌상 + 우상
                    2: [(1, 0), (1, 1)],   # Q2 하: 좌하 + 우하
                    3: [(0, 0), (1, 0)],   # Q3 좌: 좌상 + 좌하
                    4: [(0, 1), (1, 1)],   # Q4 우: 우상 + 우하
                }

                for s in range(adss.n_sub):
                    px_val = int(adss.points_x[s])
                    py_val = int(adss.points_y[s])
                    ix = x_to_idx.get(px_val)
                    iy = y_to_idx.get(py_val)
                    if ix is None or iy is None:
                        continue

                    qt = int(adss.quarter_types[s])
                    
                    # 삼각형 또는 사각형에 따라 서브셀 결정
                    cells = self._get_adss_sub_cells(qt)
                    if not cells:
                        continue

                    for dy, dx in cells:
                        grid[2 * iy + dy, 2 * ix + dx] = sub_values[s]


        return grid, unique_x_2x, unique_y_2x
    
    def _get_adss_sub_values(self, result, values, adss):
        """현재 표시 중인 필드에 대응하는 ADSS sub-POI 값 추출"""
        n_params = adss.parameters.shape[1]
        is_affine = (n_params <= 6)

        if values is result.disp_u:
            return adss.parameters[:, 0]
        elif values is result.disp_v:
            return adss.parameters[:, 3] if is_affine else adss.parameters[:, 6]
        elif hasattr(result, 'disp_ux') and values is result.disp_ux:
            return adss.parameters[:, 1]
        elif hasattr(result, 'disp_uy') and values is result.disp_uy:
            return adss.parameters[:, 2]
        elif hasattr(result, 'disp_vx') and values is result.disp_vx:
            return adss.parameters[:, 4] if is_affine else adss.parameters[:, 7]
        elif hasattr(result, 'disp_vy') and values is result.disp_vy:
            return adss.parameters[:, 5] if is_affine else adss.parameters[:, 8]
        else:
            # magnitude 등 파생 값
            return None

    # ===== 개별 렌더러 (변위 — 서브셋 직접 채색) =====
    def _draw_scalar_field(self, img, result, field_type):
        """변위 필드 시각화"""
        if result.n_points == 0:
            return img

        mode = {'u': 'u_field', 'v': 'v_field'}.get(field_type, 'magnitude')

        if field_type == 'u':
            values, label = result.disp_u, "U (px)"
        elif field_type == 'v':
            values, label = result.disp_v, "V (px)"
        else:
            values = np.sqrt(result.disp_u**2 + result.disp_v**2)
            label = "|D| (px)"

        symmetric = field_type in ('u', 'v')
        if self._resolve_display_source(mode) == 'raw':
            return self._render_field_subset(img, result, values, label, symmetric)

        grid, ux, uy = self._to_grid(result, values)
        if grid.size == 0 or len(ux) < 3 or len(uy) < 3:
            return img

        return self._render_field_matplotlib(img, grid, ux, uy, label, symmetric)

    def _get_strain_cache_key(self, result):
        """결과 객체의 고유 캐시 키 생성"""
        return (id(result), result.n_points, result.processing_time)

    def _get_cached_strain(self, result, grid_step):
        """캐시된 PLS 변형률 결과 반환"""
        # 2배 해상도이므로 grid_step은 원래의 절반
        grid_step_2x = grid_step / 2.0
        cache_key = (self._get_strain_cache_key(result), 11, 2, grid_step_2x)

        if cache_key == self._strain_cache_key and self._strain_cache:
            return self._strain_cache.get('strain')

        u_grid, _, _ = self._to_grid(result, result.disp_u)
        v_grid, _, _ = self._to_grid(result, result.disp_v)

        try:
            from speckle.core.postprocess.strain_pls import compute_strain_pls
            strain = compute_strain_pls(
                u_grid, v_grid,
                window_size=11, poly_order=2, grid_step=grid_step_2x
            )
            self._strain_cache = {'strain': strain}
            self._strain_cache_key = cache_key
            return strain
        except Exception as e:
            logger.warning(f"PLS 실패: {e}")
            return None

    def invalidate_strain_cache(self):
        """변형률 캐시 무효화"""
        self._strain_cache = {}
        self._strain_cache_key = None

    def _draw_strain_field(self, img, result, strain_type):
        """변형률 필드 시각화 (PLS, 캐시 적용)"""
        if result.n_points == 0:
            return img

        is_icgn = hasattr(result, 'disp_ux') and result.disp_ux is not None
        if not is_icgn:
            cv2.putText(img, "IC-GN 결과 필요", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return img

        if self._resolve_display_source(strain_type) == 'raw':
            return self._draw_strain_field_points(img, result, strain_type)

        valid = result.valid_mask
        px_valid = result.points_x[valid]
        unique_x = np.unique(px_valid)
        unique_y = np.unique(result.points_y[valid])

        if len(unique_x) < 5 or len(unique_y) < 5:
            return self._draw_strain_field_points(img, result, strain_type)

        grid_step = float(np.median(np.diff(unique_x))) if len(unique_x) > 1 else 1.0

        strain = self._get_cached_strain(result, grid_step)
        if strain is None:
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

        # strain 격자 좌표는 _to_grid의 2배 해상도 좌표
        # _get_cached_strain이 _to_grid를 호출하므로 2배 해상도 격자에서 PLS 계산됨
        all_unique_x = np.unique(result.points_x)
        all_unique_y = np.unique(result.points_y)
        spacing = float(all_unique_x[1] - all_unique_x[0]) if len(all_unique_x) > 1 else 1.0
        half = spacing / 4.0

        nx_orig = len(all_unique_x)
        ny_orig = len(all_unique_y)
        unique_x_2x = np.empty(nx_orig * 2, dtype=np.float64)
        unique_y_2x = np.empty(ny_orig * 2, dtype=np.float64)
        for i in range(nx_orig):
            unique_x_2x[2 * i]     = all_unique_x[i] - half
            unique_x_2x[2 * i + 1] = all_unique_x[i] + half
        for i in range(ny_orig):
            unique_y_2x[2 * i]     = all_unique_y[i] - half
            unique_y_2x[2 * i + 1] = all_unique_y[i] + half

        return self._render_field_matplotlib(
            img, strain_2d, unique_x_2x, unique_y_2x, label, symmetric
        )


    def _draw_strain_field_points(self, img, result, strain_type):
        """변형률 필드 점 시각화 (PLS 실패 시 fallback, turbo colormap)"""
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
        if abs(vmax - vmin) < 1e-15:
            vmin, vmax = vmin - 1e-6, vmax + 1e-6

        self.view.update_colorbar(vmin, vmax, label, 'turbo')

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('turbo')

        for idx in range(result.n_points):
            x = int(result.points_x[idx])
            y = int(result.points_y[idx])
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            if strain.valid_mask[idx]:
                norm_val = np.clip((values[idx] - vmin) / (vmax - vmin + 1e-10), 0, 1)
                r, g, b, _ = cmap(norm_val)
                color = (int(b * 255), int(g * 255), int(r * 255))
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
        adss = getattr(result, 'adss_result', None)
        sub_mag = None
        if adss is not None and adss.n_sub > 0:
            sub_u = adss.parameters[:, 0]
            n_params = adss.parameters.shape[1]
            sub_v = adss.parameters[:, 3] if n_params <= 6 else adss.parameters[:, 6]
            sub_mag = np.sqrt(sub_u**2 + sub_v**2)

        if self._resolve_display_source('magnitude') == 'raw':
            return self._render_field_subset(
                img, result, mag, "|D| (px)", False, adss_values=sub_mag
            )

        grid, ux, uy = self._to_grid(result, mag)

        # ADSS magnitude는 _to_grid에서 자동 처리 안 됨 (파생값)
        if adss is not None and adss.n_sub > 0:
            all_unique_x = np.unique(result.points_x)
            all_unique_y = np.unique(result.points_y)
            x_to_idx = {int(x): i for i, x in enumerate(all_unique_x)}
            y_to_idx = {int(y): i for i, y in enumerate(all_unique_y)}
            
            for s in range(adss.n_sub):
                ix = x_to_idx.get(int(adss.points_x[s]))
                iy = y_to_idx.get(int(adss.points_y[s]))
                if ix is None or iy is None:
                    continue
                qt = int(adss.quarter_types[s])

                cells = self._get_adss_sub_cells(qt)
                if not cells:
                    continue

                for dy, dx in cells:
                    grid[2 * iy + dy, 2 * ix + dx] = sub_mag[s]


        if grid.size == 0 or len(ux) < 3 or len(uy) < 3:
            return img

        return self._render_field_matplotlib(img, grid, ux, uy, "|D| (px)", False)


    def _draw_zncc_map(self, img, result):
        """ZNCC 맵 시각화 — POI 점별 표시 (보간 없음)"""
        if result.n_points == 0:
            return img

        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('turbo')

        valid = result.valid_mask
        zncc = result.zncc_values

        color_range = self.view.get_color_range()
        if color_range is not None:
            vmin, vmax = color_range
        else:
            vmin, vmax = 0.0, 1.0

        self.view.update_colorbar(vmin, vmax, "ZNCC", 'turbo')

        spacing = getattr(result, 'spacing', 10)
        radius = max(2, spacing // 4)

        for idx in range(result.n_points):
            x = int(result.points_x[idx])
            y = int(result.points_y[idx])
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue

            val = zncc[idx]
            norm_val = np.clip((val - vmin) / (vmax - vmin + 1e-10), 0, 1)
            r, g, b, _ = cmap(norm_val)
            color = (int(b * 255), int(g * 255), int(r * 255))

            if valid[idx]:
                cv2.circle(img, (x, y), radius, color, -1)
            else:
                cv2.circle(img, (x, y), radius, (128, 128, 128), -1)

        return img

    def _draw_invalid_points(self, img, result):
        """불량 POI 시각화
        
        - 초록점:    IC-GN 유효 (valid)
        - 파란 사분면: ADSS 복원 성공 (quarter별 영역 표시)
          - Q1~Q4: 삼각형 (대각선 분할) — 채워진 삼각형
          - Q5~Q8: 사각형 (축 분할) — 채워진 사각형
        - 빨간 ×:    복원 불가 (IC-GN 실패 + ADSS 실패 또는 미시도)
        - 표시 안 함:  FFT-CC 실패 (텍스처 없음)
        """
        if result.n_points == 0:
            return img

        has_fft_mask = (
            hasattr(result, 'fft_valid_mask') and
            result.fft_valid_mask is not None
        )

        # ADSS 정보 수집
        adss = getattr(result, 'adss_result', None)
        adss_parent_set = set()
        adss_parent_quarters = {}
        if adss is not None and adss.n_sub > 0:
            recovery_passes = adss.get_recovery_passes()
            adss_parent_set = set(adss.unique_parents.tolist())
            for s in range(adss.n_sub):
                pi = int(adss.parent_indices[s])
                qt = int(adss.quarter_types[s])
                if pi not in adss_parent_quarters:
                    adss_parent_quarters[pi] = []
                adss_parent_quarters[pi].append((qt, int(recovery_passes[s])))

        # spacing & subset 크기
        valid = result.valid_mask
        spacing = self._estimate_spacing(result)
        half_sp = spacing // 2

        # 사각형 quarter → 영역 오프셋
        qt_offsets = {
            5: (-half_sp, 0,         -half_sp, 0),
            6: (1,         half_sp+1, -half_sp, 0),
            7: (-half_sp, 0,          1,        half_sp+1),
            8: (1,         half_sp+1,  1,        half_sp+1),
        }

        # 삼각형 quarter 색상 (BGR)
        tri_color  = (255, 180, 0)   # 파란 계열 (사각형과 동일)
        rect_color = (255, 180, 0)
        pass2_outline_color = (255, 0, 0)

        img_h, img_w = img.shape[:2]

        for idx in range(result.n_points):
            x = int(result.points_x[idx])
            y = int(result.points_y[idx])
            if x < 0 or x >= img_w or y < 0 or y >= img_h:
                continue

            if idx in adss_parent_set:
                quarters = adss_parent_quarters.get(idx, [])
                for qt, recovery_pass in quarters:
                    poly_mask = np.zeros((img_h, img_w), dtype=np.uint8)
                    self._fill_adss_quarter(poly_mask, x, y, qt, spacing)
                    mask = poly_mask > 0
                    if np.any(mask):
                        img[mask] = rect_color if 5 <= qt <= 8 else tri_color
                        if recovery_pass == 2:
                            outline_mask = self._outline_mask(poly_mask) > 0
                            img[outline_mask] = pass2_outline_color
                    continue

                    if 5 <= qt <= 8:
                        # === 사각형 quarter ===
                        offsets = qt_offsets.get(qt)
                        if offsets is None:
                            continue
                        x1_off, x2_off, y1_off, y2_off = offsets
                        x1 = max(0, x + x1_off)
                        x2 = min(img_w, x + x2_off)
                        y1 = max(0, y + y1_off)
                        y2 = min(img_h, y + y2_off)
                        if x2 > x1 and y2 > y1:
                            cv2.rectangle(img, (x1, y1), (x2, y2),
                                          rect_color, -1)

                    elif 1 <= qt <= 4:
                        M = half_sp
                        if qt == 1:    # Q1: 상 삼각형 — 상단 변 + 두 대각선
                            pts = np.array([
                                [x - M, y - M],   # 좌상 모서리
                                [x + M, y - M],   # 우상 모서리
                                [x,     y]        # 중심
                            ], dtype=np.int32)
                        elif qt == 2:  # Q2: 하 삼각형
                            pts = np.array([
                                [x - M, y + M],   # 좌하 모서리
                                [x + M, y + M],   # 우하 모서리
                                [x,     y]        # 중심
                            ], dtype=np.int32)
                        elif qt == 3:  # Q3: 좌 삼각형
                            pts = np.array([
                                [x - M, y - M],   # 좌상 모서리
                                [x - M, y + M],   # 좌하 모서리
                                [x,     y]        # 중심
                            ], dtype=np.int32)
                        elif qt == 4:  # Q4: 우 삼각형
                            pts = np.array([
                                [x + M, y - M],   # 우상 모서리
                                [x + M, y + M],   # 우하 모서리
                                [x,     y]        # 중심
                            ], dtype=np.int32)

                        pts[:, 0] = np.clip(pts[:, 0], 0, img_w - 1)
                        pts[:, 1] = np.clip(pts[:, 1], 0, img_h - 1)
                        cv2.fillPoly(img, [pts], tri_color)

                # 중심 마커
                cv2.circle(img, (x, y), 3, (255, 100, 0), -1)

            elif result.valid_mask[idx]:
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

            else:
                if has_fft_mask and not result.fft_valid_mask[idx]:
                    pass
                else:
                    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                    cv2.drawMarker(img, (x, y), (0, 0, 255),
                                   cv2.MARKER_CROSS, 10, 2)

        return img
