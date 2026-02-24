"""이미지 캔버스 뷰"""

import tkinter as tk
from PIL import Image, ImageTk
import cv2
import numpy as np
from typing import Optional, Callable, Tuple

from speckle.models import QualityReport


class CanvasView:
    """이미지 표시 캔버스"""

    def __init__(self, parent: tk.Frame):
        self.canvas = tk.Canvas(parent, bg='gray20')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.photo: Optional[ImageTk.PhotoImage] = None
        self.image_offset: Tuple[int, int] = (0, 0)
        self.image_size: Tuple[int, int] = (0, 0)

        # 현재 상태 저장
        self.current_zoom: float = 1.0
        self.current_image_shape: Optional[Tuple[int, int]] = None

        # 드래그 상태
        self._pan_start: Optional[Tuple[int, int]] = None
        self._roi_start: Optional[Tuple[int, int]] = None

        # 콜백
        self.on_zoom: Optional[Callable[[float], None]] = None
        self.on_pan: Optional[Callable[[int, int], None]] = None
        self.on_roi_draw: Optional[Callable[[Tuple[int, int, int, int]], None]] = None

        # POI 조회 콜백: (img_x, img_y) → Optional[str]
        # 외부에서 설정하면, 마우스 이동 시 해당 좌표의 POI 정보를 툴팁으로 표시
        self.on_query_poi: Optional[Callable[[int, int], Optional[str]]] = None

        # 툴팁
        self._tooltip_id: Optional[int] = None
        self._tooltip_bg_id: Optional[int] = None

        self._bind_events()

    def _bind_events(self):
        """이벤트 바인딩"""
        # 마우스 휠 (줌)
        self.canvas.bind('<MouseWheel>', self._on_mouse_wheel)
        self.canvas.bind('<Button-4>', self._on_mouse_wheel)
        self.canvas.bind('<Button-5>', self._on_mouse_wheel)

        # 좌클릭 (팬)
        self.canvas.bind('<ButtonPress-1>', self._on_left_press)
        self.canvas.bind('<B1-Motion>', self._on_left_drag)
        self.canvas.bind('<ButtonRelease-1>', self._on_left_release)

        # 우클릭 (ROI)
        self.canvas.bind('<ButtonPress-3>', self._on_right_press)
        self.canvas.bind('<B3-Motion>', self._on_right_drag)
        self.canvas.bind('<ButtonRelease-3>', self._on_right_release)

        # 마우스 이동 (툴팁)
        self.canvas.bind('<Motion>', self._on_mouse_move)
        self.canvas.bind('<Leave>', self._hide_tooltip)

    def display(self, image: np.ndarray,
                zoom: float,
                pan_offset: Tuple[int, int],
                roi: Optional[Tuple[int, int, int, int]] = None,
                report: Optional[QualityReport] = None,
                show_all_poi: bool = True,
                show_bad_poi: bool = True):
        """이미지 표시"""
        self.current_zoom = zoom
        self.current_image_shape = image.shape[:2]

        display_img = self._prepare_image(image, roi, report, show_all_poi, show_bad_poi)

        h, w = display_img.shape[:2]
        new_w = int(w * zoom)
        new_h = int(h * zoom)

        if new_w > 0 and new_h > 0:
            display_img = cv2.resize(display_img, (new_w, new_h))

        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)

        pil_img = Image.fromarray(display_img)
        self.photo = ImageTk.PhotoImage(pil_img)

        self.canvas.delete("all")

        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()

        x = (canvas_w - new_w) // 2 + pan_offset[0]
        y = (canvas_h - new_h) // 2 + pan_offset[1]

        self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo)
        self.image_offset = (x, y)
        self.image_size = (new_w, new_h)

    def _prepare_image(self, image: np.ndarray,
                       roi: Optional[Tuple[int, int, int, int]],
                       report: Optional[QualityReport],
                       show_all_poi: bool,
                       show_bad_poi: bool) -> np.ndarray:
        """이미지 전처리 및 오버레이"""
        if len(image.shape) == 2:
            display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display_img = image.copy()

        if report and report.sssig_result and (show_all_poi or show_bad_poi):
            display_img = self._draw_poi(display_img, report, roi, show_all_poi, show_bad_poi)

        if roi:
            x, y, w, h = roi
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 255), 2)

        return display_img

    def _draw_poi(self, image: np.ndarray,
                  report: QualityReport,
                  roi: Optional[Tuple[int, int, int, int]],
                  show_all_poi: bool,
                  show_bad_poi: bool) -> np.ndarray:
        """POI 오버레이"""
        sr = report.sssig_result
        if sr is None:
            return image

        roi_x, roi_y = (roi[0], roi[1]) if roi else (0, 0)
        bad_coords = set((bp.y, bp.x) for bp in sr.bad_points)

        for idx in range(len(sr.points_y)):
            py_local = int(sr.points_y[idx])
            px_local = int(sr.points_x[idx])
            py = py_local + roi_y
            px = px_local + roi_x

            is_bad = (py_local, px_local) in bad_coords

            if is_bad and show_bad_poi:
                color = (0, 255, 255) if report.subset_size_found else (0, 0, 255)
                cv2.circle(image, (px, py), 5, color, -1)
                cv2.drawMarker(image, (px, py), color, cv2.MARKER_CROSS, 10, 2)
            elif not is_bad and show_all_poi:
                cv2.circle(image, (px, py), 2, (0, 255, 0), -1)

        return image

    def fit_to_canvas(self, image_shape: Tuple[int, int]) -> float:
        """캔버스에 맞는 줌 레벨 계산"""
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        img_h, img_w = image_shape

        if canvas_w <= 1 or canvas_h <= 1:
            return 1.0

        scale_w = canvas_w / img_w
        scale_h = canvas_h / img_h
        return min(scale_w, scale_h, 1.0)

    def _canvas_to_image(self, cx: int, cy: int) -> Tuple[Optional[int], Optional[int]]:
        """캔버스 좌표를 이미지 좌표로 변환"""
        if self.image_size[0] == 0 or self.current_image_shape is None:
            return None, None

        ox, oy = self.image_offset
        iw, ih = self.image_size

        ix = cx - ox
        iy = cy - oy

        if 0 <= ix < iw and 0 <= iy < ih:
            img_x = int(ix / self.current_zoom)
            img_y = int(iy / self.current_zoom)

            h, w = self.current_image_shape
            return max(0, min(w - 1, img_x)), max(0, min(h - 1, img_y))

        return None, None

    # ===== 툴팁 =====

    def _on_mouse_move(self, event):
        """마우스 이동 시 POI 툴팁 표시"""
        if self.on_query_poi is None:
            self._hide_tooltip()
            return

        img_x, img_y = self._canvas_to_image(event.x, event.y)
        if img_x is None:
            self._hide_tooltip()
            return

        text = self.on_query_poi(img_x, img_y)
        if text:
            self._show_tooltip(event.x, event.y, text)
        else:
            self._hide_tooltip()

    def _show_tooltip(self, cx: int, cy: int, text: str):
        """캔버스 위 툴팁 표시"""
        self._hide_tooltip()

        # 커서 약간 아래 오른쪽에 표시
        tx = cx + 15
        ty = cy + 10

        self._tooltip_bg_id = self.canvas.create_rectangle(
            0, 0, 0, 0, fill='#FFFFDD', outline='#333333', width=1)
        self._tooltip_id = self.canvas.create_text(
            tx, ty, anchor=tk.NW, text=text,
            font=('Consolas', 9), fill='#222222')

        # 텍스트 bbox로 배경 크기 조정
        bbox = self.canvas.bbox(self._tooltip_id)
        if bbox:
            pad = 4
            self.canvas.coords(
                self._tooltip_bg_id,
                bbox[0] - pad, bbox[1] - pad,
                bbox[2] + pad, bbox[3] + pad)
            # 배경을 텍스트 뒤로
            self.canvas.tag_lower(self._tooltip_bg_id, self._tooltip_id)

    def _hide_tooltip(self, event=None):
        """툴팁 제거"""
        if self._tooltip_id is not None:
            self.canvas.delete(self._tooltip_id)
            self._tooltip_id = None
        if self._tooltip_bg_id is not None:
            self.canvas.delete(self._tooltip_bg_id)
            self._tooltip_bg_id = None

    # ===== 마우스 이벤트 핸들러 =====

    def _on_mouse_wheel(self, event):
        """줌"""
        if self.on_zoom:
            factor = 1.1 if (event.delta > 0 or event.num == 4) else 0.9
            self.on_zoom(factor)

    def _on_left_press(self, event):
        """팬 시작"""
        self._pan_start = (event.x, event.y)

    def _on_left_drag(self, event):
        """팬 드래그"""
        if self._pan_start and self.on_pan:
            dx = event.x - self._pan_start[0]
            dy = event.y - self._pan_start[1]
            self._pan_start = (event.x, event.y)
            self.on_pan(dx, dy)

    def _on_left_release(self, event):
        """팬 종료"""
        self._pan_start = None

    def _on_right_press(self, event):
        """ROI 시작"""
        coords = self._canvas_to_image(event.x, event.y)
        if coords[0] is not None:
            self._roi_start = coords

    def _on_right_drag(self, event):
        """ROI 드래그"""
        if self._roi_start and self.on_roi_draw:
            current = self._canvas_to_image(event.x, event.y)
            if current[0] is not None:
                x1, y1 = self._roi_start
                x2, y2 = current
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                if w > 0 and h > 0:
                    self.on_roi_draw((x, y, w, h))

    def _on_right_release(self, event):
        """ROI 종료"""
        self._roi_start = None
