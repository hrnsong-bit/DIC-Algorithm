"""
DIC 분석 메인 컨트롤러

역할: 상태 관리, 콜백 설정, 모듈 간 조율
분석/렌더링/결과처리/내보내기는 각각 별도 모듈에 위임
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox

from speckle.core.initial_guess import (
    warmup_fft_cc, FFTCCResult, ValidationResult
)
from speckle.io import load_image, get_image_files
from ..models.app_state import AppState

from .analysis_runner import AnalysisRunner
from .result_handler import ResultHandler
from .renderers import FieldRenderer
from .export_handler import ExportHandler

logger = logging.getLogger(__name__)


@dataclass
class DICState:
    """DIC 분석 전용 상태"""
    ref_image: Optional[np.ndarray] = None
    ref_path: Optional[Path] = None
    def_image: Optional[np.ndarray] = None
    def_path: Optional[Path] = None

    sequence_folder: Optional[Path] = None
    sequence_files: List[Path] = field(default_factory=list)
    current_index: int = 0

    roi: Optional[tuple] = None

    zoom_level: float = 1.0
    pan_offset: tuple = (0, 0)

    fft_cc_result: Optional[FFTCCResult] = None
    icgn_result: Optional[Any] = None
    validation_result: Optional[ValidationResult] = None
    batch_results: Dict[str, Any] = field(default_factory=dict)

    is_running: bool = False
    should_stop: bool = False


class DICController:
    """DIC 분석 컨트롤러"""

    def __init__(self, view, app_state: AppState, main_controller=None):
        self.view = view
        self.app_state = app_state
        self.main_controller = main_controller
        self.state = DICState()

        # 위임 객체 생성
        self.runner = AnalysisRunner(self)
        self.result_handler = ResultHandler(self)
        self.renderer = FieldRenderer(self)
        self.exporter = ExportHandler(self)

        self._setup_callbacks()
        self._setup_canvas_callbacks()
        self._warmup()

    def _setup_callbacks(self):
        """뷰 콜백 설정"""
        self.view.set_callback('select_reference', self.select_reference)
        self.view.set_callback('select_deformed', self.select_deformed)
        self.view.set_callback('select_sequence', self.select_sequence)
        self.view.set_callback('sync_from_quality', self.sync_from_quality_tab)
        self.view.set_callback('run_analysis', self.runner.run_fft_cc)
        self.view.set_callback('run_batch_analysis', self.runner.run_batch_fft_cc)
        self.view.set_callback('stop_analysis', self.runner.stop_analysis)
        self.view.set_callback('export_csv', self.exporter.export_csv)
        self.view.set_callback('export_image', self.exporter.export_image)
        self.view.set_callback('update_display', self.update_display)
        self.view.set_callback('fit_to_canvas', self.fit_to_canvas)
        self.view.set_callback('select_image_index', self.select_image_index)
        self.view.set_callback('zoom_in', self.zoom_in)
        self.view.set_callback('zoom_out', self.zoom_out)
        self.view.set_callback('set_zoom_1to1', self.set_zoom_1to1)

    def _setup_canvas_callbacks(self):
        """캔버스 콜백 설정"""
        self.view.canvas_view.on_zoom = self._handle_zoom
        self.view.canvas_view.on_pan = self._handle_pan
        self.view.canvas_view.on_roi_draw = self._handle_roi_draw

    def _handle_zoom(self, factor: float):
        new_zoom = max(0.1, min(5.0, self.state.zoom_level * factor))
        self.state.zoom_level = new_zoom
        self._refresh_display()
        self.view.zoom_label.configure(text=f"{int(new_zoom * 100)}%")

    def _handle_pan(self, dx: int, dy: int):
        ox, oy = self.state.pan_offset
        self.state.pan_offset = (ox + dx, oy + dy)
        self._refresh_display()

    def _handle_roi_draw(self, roi: tuple):
        self.state.roi = roi
        self._refresh_display()

    def _warmup(self):
        try:
            warmup_fft_cc()
        except Exception as e:
            logger.debug(f"FFT-CC 워밍업 실패 (무시): {e}")

    # ===== 이미지 선택 =====

    def select_reference(self):
        path = filedialog.askopenfilename(
            title="Reference 이미지 선택",
            filetypes=[
                ("이미지 파일", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("모든 파일", "*.*")
            ]
        )
        if path:
            self._load_reference(Path(path))

    def _load_reference(self, path: Path):
        try:
            img = self.app_state.get_image(path.name)
            if img is None:
                img = load_image(path)
            if img is not None:
                self.state.ref_image = img
                self.state.ref_path = path
                self.view.update_reference_label(path.name)
                self._display_image(img)
                self.fit_to_canvas()
        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 실패: {e}")

    def select_deformed(self):
        path = filedialog.askopenfilename(
            title="Deformed 이미지 선택",
            filetypes=[
                ("이미지 파일", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"),
                ("모든 파일", "*.*")
            ]
        )
        if path:
            self._load_deformed(Path(path))

    def _load_deformed(self, path: Path):
        try:
            img = self.app_state.get_image(path.name)
            if img is None:
                img = load_image(path)
            if img is not None:
                self.state.def_image = img
                self.state.def_path = path
                self.view.update_deformed_label(path.name)
                self._display_image(img)
        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 실패: {e}")

    def select_sequence(self):
        folder = filedialog.askdirectory(title="이미지 시퀀스 폴더 선택")
        if folder:
            self._load_sequence(Path(folder))

    def _load_sequence(self, folder: Path):
        try:
            files = get_image_files(folder)
            if len(files) < 2:
                messagebox.showwarning("경고", "폴더에 2개 이상의 이미지가 필요합니다.")
                return
            self.state.sequence_folder = folder
            self.state.sequence_files = files
            self.state.current_index = 0
            self._load_reference(files[0])
            if len(files) > 1:
                self._load_deformed(files[1])
                self.state.current_index = 1
            self.view.update_sequence_label(f"{folder.name} ({len(files)} files)")
            self.view.update_file_list(files, self.state.current_index)
        except Exception as e:
            messagebox.showerror("오류", f"폴더 로드 실패: {e}")

    def select_image_index(self, index: int):
        if not self.state.sequence_files:
            return
        if 0 <= index < len(self.state.sequence_files):
            self.state.current_index = index
            self.view.current_index = index
            path = self.state.sequence_files[index]
            self._load_deformed(path)
            self.view.set_current_index(index)
            if path.name in self.state.batch_results:
                result = self.state.batch_results[path.name]
                self.state.fft_cc_result = result
                self._refresh_display()
                self.result_handler.update_result_ui(result)
            else:
                self.state.fft_cc_result = None
                self._refresh_display()

    # ===== 동기화 =====

    def sync_from_quality_tab(self):
        if not self.app_state.has_images():
            messagebox.showinfo("정보", "품질평가 탭에서 폴더를 먼저 열어주세요.")
            return

        if self.app_state.loading_state.is_loading:
            result = messagebox.askyesno(
                "로딩 중",
                f"이미지 로딩 중입니다.\n"
                f"({self.app_state.loading_state.loaded_files}/{self.app_state.loading_state.total_files})\n\n"
                f"로드된 이미지만 사용하시겠습니까?"
            )
            if not result:
                return

        try:
            self.state.sequence_files = self.app_state.file_paths
            self.state.sequence_folder = self.app_state.folder_path

            if self.app_state.file_list:
                ref_name = self.app_state.file_list[0]
                ref_img = self.app_state.get_image(ref_name)
                if ref_img is not None:
                    self.state.ref_image = ref_img
                    self.state.ref_path = self.app_state.file_paths[0] if self.app_state.file_paths else None
                    self.view.update_reference_label(ref_name)

            if len(self.app_state.file_list) > 1:
                def_name = self.app_state.file_list[1]
                def_img = self.app_state.get_image(def_name)
                if def_img is not None:
                    self.state.def_image = def_img
                    self.state.def_path = self.app_state.file_paths[1] if len(self.app_state.file_paths) > 1 else None
                    self.state.current_index = 1
                    self.view.update_deformed_label(def_name)

            self.state.roi = self.app_state.roi

            if self.main_controller:
                quality_params = self.main_controller.get_current_parameters()
                params = {
                    'subset_size': quality_params.get('subset_size', 21),
                    'spacing': quality_params.get('spacing', 16)
                }
                self.view.set_parameters(params)

            self.view.update_sequence_label(
                f"{self.app_state.folder_path.name if self.app_state.folder_path else ''} "
                f"({len(self.app_state.images)}/{len(self.app_state.file_list)} loaded)"
            )
            self.view.update_file_list(self.state.sequence_files, self.state.current_index)
            self._refresh_display()

            messagebox.showinfo(
                "동기화 완료",
                f"품질평가 탭에서 동기화 완료!\n\n"
                f"이미지: {len(self.app_state.images)}장 (캐시됨)\n"
                f"메모리: {self.app_state.memory_usage_mb:.1f}MB\n"
                f"ROI: {self.state.roi}"
            )
        except Exception as e:
            messagebox.showerror("오류", f"동기화 실패: {e}")

    # ===== 표시 =====

    def _display_image(self, img: np.ndarray):
        if img is None:
            return
        self._refresh_display()

    def update_display(self, mode: str = None):
        self._refresh_display(mode)

    def _refresh_display(self, mode: str = None):
        img = self.state.def_image if self.state.def_image is not None else self.state.ref_image
        if img is None:
            return

        if self.state.fft_cc_result is not None:
            display_img = self.renderer.create_overlay_image(
                img, self.state.fft_cc_result, mode
            )
        else:
            display_img = img

        self.view.canvas_view.display(
            image=display_img,
            zoom=self.state.zoom_level,
            pan_offset=self.state.pan_offset,
            roi=self.state.roi,
            report=None,
            show_all_poi=False,
            show_bad_poi=False
        )

    def fit_to_canvas(self):
        img = self.state.ref_image
        if img is None:
            return
        zoom = self.view.canvas_view.fit_to_canvas(img.shape[:2])
        self.state.zoom_level = zoom
        self.state.pan_offset = (0, 0)
        self.view.zoom_label.configure(text=f"{int(zoom * 100)}%")
        self._refresh_display()

    def zoom_in(self):
        self._handle_zoom(1.2)

    def zoom_out(self):
        self._handle_zoom(0.8)

    def set_zoom_1to1(self):
        self.state.zoom_level = 1.0
        self.state.pan_offset = (0, 0)
        self.view.zoom_label.configure(text="100%")
        self._refresh_display()
