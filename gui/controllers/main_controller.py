"""메인 컨트롤러"""

import threading
from queue import Queue
from pathlib import Path
from typing import Optional, Callable, Dict
from tkinter import filedialog, messagebox

from speckle import SpeckleQualityAssessor, load_image, load_folder, ResultExporter
from speckle.utils.logger import logger

from ..models.app_state import AppState
from ..views.canvas_view import CanvasView
from ..views.param_panel import ParamPanel


class MainController:
    """GUI 로직 컨트롤러"""
    
    def __init__(self, state: AppState):
        self.state = state
        self.assessor = SpeckleQualityAssessor()
        self._result_queue: Queue = Queue()
        
        # 뷰 참조 (나중에 설정)
        self.canvas_view: Optional[CanvasView] = None
        self.param_panel: Optional[ParamPanel] = None
        
        # 콜백
        self.on_state_changed: Optional[Callable[[], None]] = None
        self.on_progress: Optional[Callable[[int, int, str], None]] = None
        self.on_batch_complete: Optional[Callable[[], None]] = None
    
    def set_views(self, canvas_view: CanvasView, param_panel: ParamPanel):
        """뷰 연결"""
        self.canvas_view = canvas_view
        self.param_panel = param_panel
        
        # 캔버스 콜백 연결
        canvas_view.on_zoom = self._handle_zoom
        canvas_view.on_pan = self._handle_pan
        canvas_view.on_roi_draw = self._handle_roi_draw
    
    def open_image(self):
        """이미지 열기"""
        path = filedialog.askopenfilename(
            filetypes=[
                ("이미지", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("모든 파일", "*.*")
            ]
        )
        if path:
            image = load_image(path)
            if image is not None:
                self.state.clear()
                self.state.current_image = image
                self.state.current_file = Path(path).name
                self.state.images = {self.state.current_file: image}
                self.state.file_list = [self.state.current_file]
                self.state.current_index = 0
                
                if self.canvas_view:
                    zoom = self.canvas_view.fit_to_canvas(image.shape[:2])
                    self.state.zoom_level = zoom
                
                logger.info(f"이미지 로드: {path}")
                self._notify_state_changed()
            else:
                messagebox.showerror("오류", "이미지를 로드할 수 없습니다.")
    
    def open_folder(self):
        """폴더 열기"""
        path = filedialog.askdirectory()
        if path:
            self.state.clear()
            self.state.folder_path = Path(path)
            self.state.images = load_folder(path)
            
            if self.state.images:
                self.state.file_list = list(self.state.images.keys())
                self.state.current_index = 0
                self.state.current_file = self.state.file_list[0]
                self.state.current_image = self.state.images[self.state.current_file]
                
                if self.canvas_view:
                    zoom = self.canvas_view.fit_to_canvas(self.state.current_image.shape[:2])
                    self.state.zoom_level = zoom
                
                logger.info(f"폴더 로드: {path}, {len(self.state.images)}개 이미지")
                self._notify_state_changed()
            else:
                messagebox.showwarning("경고", "이미지가 없습니다.")
    
    def evaluate_current(self):
        """현재 이미지 평가"""
        if self.state.current_image is None:
            messagebox.showwarning("경고", "이미지를 먼저 로드하세요.")
            return
        
        self._update_assessor()
        report = self.assessor.evaluate(self.state.current_image, self.state.roi)
        self.state.set_report(self.state.current_file, report)
        
        logger.info(f"평가 완료: {self.state.current_file}, {report.quality_grade}")
        self._notify_state_changed()
    
    def evaluate_all(self):
        """전체 배치 평가"""
        if not self.state.has_images():
            messagebox.showwarning("경고", "이미지를 먼저 로드하세요.")
            return
        
        self._update_assessor()
        self.state.batch_running = True
        self.state.batch_stop_requested = False
        
        logger.info(f"배치 평가 시작: {self.state.total_images}개")
        
        thread = threading.Thread(target=self._run_batch, daemon=True)
        thread.start()
    
    def stop_batch(self):
        """배치 중지"""
        if self.state.batch_running:
            self.state.batch_stop_requested = True
            logger.info("배치 처리 중지 요청")
    
    def _run_batch(self):
        """배치 실행 (워커 스레드)"""
        total = self.state.total_images
        
        for idx, filename in enumerate(self.state.file_list):
            if self.state.batch_stop_requested:
                logger.info(f"배치 처리 중지됨: {idx}/{total}")
                break
            
            if self.on_progress:
                self.on_progress(idx + 1, total, filename)
            
            try:
                image = self.state.images[filename]
                report = self.assessor.evaluate(image, self.state.roi)
                self.state.set_report(filename, report)
            except Exception as e:
                logger.exception(f"평가 오류 ({filename})")
        
        self.state.batch_running = False
        
        if self.on_batch_complete:
            self.on_batch_complete()
    
    def navigate(self, direction: int):
        """이미지 탐색"""
        if self.state.navigate(direction):
            self._notify_state_changed()
    
    def navigate_to(self, index: int):
        """특정 인덱스로 이동"""
        if self.state.navigate_to(index):
            self._notify_state_changed()
    
    def reset_roi(self):
        """ROI 초기화"""
        self.state.reset_roi()
        self.state.clear_report(self.state.current_file)
        self._notify_state_changed()
    
    def fit_to_canvas(self):
        """캔버스에 맞춤"""
        if self.state.current_image is not None and self.canvas_view:
            zoom = self.canvas_view.fit_to_canvas(self.state.current_image.shape[:2])
            self.state.zoom_level = zoom
            self.state.pan_offset = (0, 0)
            self._notify_state_changed()
    
    def _handle_zoom(self, factor: float):
        """줌 처리"""
        new_zoom = max(0.1, min(5.0, self.state.zoom_level * factor))
        self.state.zoom_level = new_zoom
        self._notify_state_changed()
    
    def _handle_pan(self, dx: int, dy: int):
        """팬 처리"""
        ox, oy = self.state.pan_offset
        self.state.pan_offset = (ox + dx, oy + dy)
        self._notify_state_changed()
    
    def _handle_roi_draw(self, roi: tuple):
        """ROI 그리기"""
        self.state.roi = roi
        self.state.clear_report(self.state.current_file)
        self._notify_state_changed()
    
    def _update_assessor(self):
        """Assessor 파라미터 업데이트"""
        if self.param_panel:
            params = self.param_panel.get_parameters()
            if params:
                self.assessor = SpeckleQualityAssessor(
                    mig_threshold=params.mig_threshold,
                    sssig_threshold=params.sssig_threshold,
                    subset_size=params.subset_size,
                    poi_spacing=params.spacing
                )
                logger.debug(f"Assessor 업데이트: {params}")
            else:
                messagebox.showerror("오류", "잘못된 파라미터 값입니다.")
    
    def _notify_state_changed(self):
        """상태 변경 알림"""
        if self.on_state_changed:
            self.on_state_changed()
    
    def get_current_parameters(self) -> Dict:
        """현재 파라미터 반환 (DIC 탭 동기화용)"""
        if self.param_panel:
            params = self.param_panel.get_parameters()
            if params:
                return {
                    'mig_threshold': params.mig_threshold,
                    'sssig_threshold': params.sssig_threshold,
                    'subset_size': params.subset_size,
                    'spacing': params.spacing
                }
        return {'subset_size': 21, 'spacing': 10}
    
    def get_roi(self):
        """현재 ROI 반환"""
        return self.state.roi
    
    def get_batch_summary(self) -> dict:
        """배치 평가 요약 반환"""
        reports = self.state.get_all_reports()
        
        passed = 0
        warning = 0
        failed = 0
        max_subset = 21
        
        for report in reports.values():
            if not report.analyzable:
                failed += 1
            elif report.recommended_subset_size > report.current_subset_size:
                warning += 1
            else:
                passed += 1
            
            if report.recommended_subset_size > max_subset:
                max_subset = report.recommended_subset_size
        
        return {
            'total': len(reports),
            'passed': passed,
            'warning': warning,
            'failed': failed,
            'max_subset': max_subset
        }
    
    # ===== 내보내기 기능 =====
    
    def export_results(self, export_type: str = 'all', include_images: bool = False) -> Optional[Dict[str, Path]]:
        """결과 내보내기"""
        reports = self.state.get_all_reports()
        
        if not reports:
            messagebox.showwarning("경고", "내보낼 평가 결과가 없습니다.")
            return None
        
        output_dir = filedialog.askdirectory(title="결과 저장 폴더 선택")
        if not output_dir:
            return None
        
        try:
            exporter = ResultExporter(output_dir)
            parameters = self.get_current_parameters()
            roi = self.state.roi
            
            results = {}
            
            if export_type == 'csv':
                results['csv'] = exporter.export_csv(reports)
            elif export_type == 'json':
                results['json'] = exporter.export_json(reports, parameters, roi)
            elif export_type == 'txt':
                results['summary'] = exporter.export_summary_txt(reports, parameters, roi)
            elif export_type == 'image':
                if self.state.current_image is not None and self.state.current_file in reports:
                    results['image'] = exporter.export_image(
                        self.state.current_image,
                        reports[self.state.current_file],
                        roi
                    )
            elif export_type == 'all':
                results = exporter.export_all(
                    self.state.images,
                    reports,
                    parameters,
                    roi,
                    include_images=include_images
                )
            
            logger.info(f"결과 내보내기 완료: {list(results.keys())}")
            return results
            
        except Exception as e:
            logger.exception("내보내기 오류")
            messagebox.showerror("오류", f"내보내기 실패: {e}")
            return None
    
    def export_current_image(self) -> Optional[Path]:
        """현재 이미지만 내보내기"""
        if self.state.current_image is None:
            messagebox.showwarning("경고", "이미지가 없습니다.")
            return None
        
        report = self.state.get_report(self.state.current_file)
        if not report:
            messagebox.showwarning("경고", "평가 결과가 없습니다. 먼저 평가를 실행하세요.")
            return None
        
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("JPEG", "*.jpg"), ("TIFF", "*.tif")],
            initialfile=f"overlay_{Path(self.state.current_file).stem}.png"
        )
        
        if not path:
            return None
        
        try:
            exporter = ResultExporter(str(Path(path).parent))
            result = exporter.export_image(
                self.state.current_image,
                report,
                self.state.roi,
                filename=Path(path).name
            )
            logger.info(f"이미지 저장 완료: {result}")
            return result
        except Exception as e:
            logger.exception("이미지 저장 오류")
            messagebox.showerror("오류", f"저장 실패: {e}")
            return None
