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

    # ===== 새 렌더러: 서브셋 영역 직접 채색 =====

    def _render_field_subset(self, img: np.ndarray, result,
                              values: np.ndarray, label: str,
                              symmetric: bool) -> np.ndarray:
        """각 POI의 값을 서브셋 영역에 직접 칠하는 렌더러

        보간 없음. ADSS 복구 POI는 quarter-type에 따라 절반만 채색.
        invalid 영역은 투명(배경 투과).
        """
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        valid = result.valid_mask
        valid_values = values[valid]
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
        px_valid = result.points_x[valid]
        unique_x = np.unique(px_valid)
        spacing = int(round(np.median(np.diff(unique_x)))) if len(unique_x) > 1 else 20
        half_sp = spacing // 2

        # === ADSS quarter-type 배열 ===
        has_qt = (hasattr(result, 'adss_quarter_type')
                  and result.adss_quarter_type is not None)

        # === 배경 이미지 ===
        if len(img.shape) == 3:
            gray_bg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray_bg = img.copy()
        overlay = (gray_bg.astype(np.float32) * 0.3)
        overlay = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_GRAY2BGR).astype(np.float32)

        img_h, img_w = img.shape[:2]
        cmap = plt.get_cmap('turbo')
        alpha = 0.9

        # === 각 POI 서브셋 영역 채색 ===
        for idx in range(result.n_points):
            if not valid[idx]:
                continue

            cx = int(result.points_x[idx])
            cy = int(result.points_y[idx])
            val = values[idx]

            norm_val = np.clip((val - vmin) / (vmax - vmin + 1e-15), 0, 1)
            r, g, b, _ = cmap(norm_val)
            color_bgr = np.array([b * 255, g * 255, r * 255], dtype=np.float32)

            # quarter-type에 따른 영역 결정
            qt = 0
            if has_qt:
                qt = int(result.adss_quarter_type[idx])

            if qt == 0:
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

            if x2 <= x1 or y2 <= y1:
                continue

            overlay[y1:y2, x1:x2] = (
                overlay[y1:y2, x1:x2] * (1 - alpha) + color_bgr * alpha
            )

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
        """POI 값을 2D 그리드로 변환 (벡터화)"""
        valid = result.valid_mask
        px, py = result.points_x[valid], result.points_y[valid]

        unique_x = np.unique(px)
        unique_y = np.unique(py)
        nx, ny = len(unique_x), len(unique_y)

        xi = np.searchsorted(unique_x, px)
        yi = np.searchsorted(unique_y, py)

        grid = np.full((ny, nx), np.nan)
        v_valid = values[valid]

        in_bounds = (xi < nx) & (yi < ny)
        grid[yi[in_bounds], xi[in_bounds]] = v_valid[in_bounds]

        return grid, unique_x, unique_y

    # ===== 개별 렌더러 (변위 — 서브셋 직접 채색) =====

    def _draw_scalar_field(self, img, result, field_type):
        """변위 필드 시각화 — 서브셋 영역 직접 채색"""
        if result.n_points == 0:
            return img

        if field_type == 'u':
            values, label = result.disp_u, "U (px)"
        elif field_type == 'v':
            values, label = result.disp_v, "V (px)"
        else:
            values = np.sqrt(result.disp_u**2 + result.disp_v**2)
            label = "|D| (px)"

        symmetric = field_type in ('u', 'v')
        return self._render_field_subset(img, result, values, label, symmetric)

    def _get_strain_cache_key(self, result):
        """결과 객체의 고유 캐시 키 생성"""
        return (id(result), result.n_points, result.processing_time)

    def _get_cached_strain(self, result, grid_step):
        """캐시된 PLS 변형률 결과 반환 (없으면 계산 후 캐시)"""
        cache_key = (self._get_strain_cache_key(result), 11, 2, grid_step)

        if cache_key == self._strain_cache_key and self._strain_cache:
            return self._strain_cache.get('strain')

        u_grid, _, _ = self._to_grid(result, result.disp_u)
        v_grid, _, _ = self._to_grid(result, result.disp_v)

        try:
            from speckle.core.postprocess.strain_pls import compute_strain_pls
            strain = compute_strain_pls(
                u_grid, v_grid,
                window_size=11, poly_order=2, grid_step=grid_step
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
        """변형률 필드 시각화 — IC-GN gradient 직접 추출 + 서브셋 채색"""
        if result.n_points == 0:
            return img

        is_icgn = hasattr(result, 'disp_ux') and result.disp_ux is not None
        if not is_icgn:
            cv2.putText(img, "IC-GN 결과 필요", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return img

        # IC-GN shape function gradient에서 직접 strain 계산
        ux = result.disp_ux
        uy = result.disp_uy
        vx = result.disp_vx
        vy = result.disp_vy

        exx = ux
        eyy = vy
        exy = 0.5 * (uy + vx)

        e_mean = 0.5 * (exx + eyy)
        R = np.sqrt(((exx - eyy) / 2.0)**2 + exy**2)
        e1 = e_mean + R
        von_mises = np.sqrt(exx**2 + eyy**2 - exx * eyy + 3.0 * exy**2)

        field_map = {
            'exx': (exx, "εxx"),
            'eyy': (eyy, "εyy"),
            'exy': (exy, "εxy"),
            'e1':  (e1, "ε₁"),
            'von_mises': (von_mises, "εᵥₘ"),
        }

        if strain_type not in field_map:
            return img

        values, label = field_map[strain_type]
        symmetric = strain_type in ('exx', 'eyy', 'exy', 'e1')

        return self._render_field_subset(img, result, values, label, symmetric)

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
        """변위 크기 시각화 — 서브셋 영역 직접 채색"""
        if result.n_points == 0:
            return img

        mag = np.sqrt(result.disp_u**2 + result.disp_v**2)
        return self._render_field_subset(img, result, mag, "|D| (px)", False)

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
        """불량 POI 시각화"""
        if result.n_points == 0:
            return img

        has_fft_mask = (
            hasattr(result, 'fft_valid_mask') and
            result.fft_valid_mask is not None
        )

        for idx in range(result.n_points):
            x = int(result.points_x[idx])
            y = int(result.points_y[idx])
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue

            if result.valid_mask[idx]:
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            else:
                if has_fft_mask and not result.fft_valid_mask[idx]:
                    pass
                else:
                    cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
                    cv2.drawMarker(img, (x, y), (0, 0, 255),
                                cv2.MARKER_CROSS, 10, 2)

        return img
