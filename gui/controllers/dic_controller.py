"""DIC 분석 컨트롤러"""

import threading
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox

from speckle.core.initial_guess import (
    compute_fft_cc, warmup_fft_cc, 
    validate_displacement_field,
    FFTCCResult, ValidationResult
)
from speckle.io import load_image, get_image_files


@dataclass
class DICState:
    """DIC 분석 상태"""
    ref_image: Optional[np.ndarray] = None
    ref_path: Optional[Path] = None
    def_image: Optional[np.ndarray] = None
    def_path: Optional[Path] = None
    
    sequence_folder: Optional[Path] = None
    sequence_files: List[Path] = field(default_factory=list)
    current_index: int = 0
    
    roi: Optional[tuple] = None
    
    # 줌/팬 상태
    zoom_level: float = 1.0
    pan_offset: tuple = (0, 0)
    
    fft_cc_result: Optional[FFTCCResult] = None
    validation_result: Optional[ValidationResult] = None
    batch_results: Dict[str, FFTCCResult] = field(default_factory=dict)
    
    is_running: bool = False
    should_stop: bool = False


class DICController:
    """DIC 분석 컨트롤러"""
    
    def __init__(self, view, main_controller=None):
        self.view = view
        self.main_controller = main_controller
        self.state = DICState()
        
        self._setup_callbacks()
        self._setup_canvas_callbacks()
        self._warmup()
    
    def _setup_callbacks(self):
        """뷰 콜백 설정"""
        self.view.set_callback('select_reference', self.select_reference)
        self.view.set_callback('select_deformed', self.select_deformed)
        self.view.set_callback('select_sequence', self.select_sequence)
        self.view.set_callback('sync_from_quality', self.sync_from_quality_tab)
        self.view.set_callback('run_analysis', self.run_fft_cc)
        self.view.set_callback('run_batch_analysis', self.run_batch_fft_cc)
        self.view.set_callback('stop_analysis', self.stop_analysis)
        self.view.set_callback('export_csv', self.export_csv)
        self.view.set_callback('export_image', self.export_image)
        self.view.set_callback('update_display', self.update_display)
        self.view.set_callback('fit_to_canvas', self.fit_to_canvas)
        self.view.set_callback('select_image_index', self.select_image_index)
        self.view.set_callback('zoom_in', self.zoom_in)
        self.view.set_callback('zoom_out', self.zoom_out)
        self.view.set_callback('set_zoom_1to1', self.set_zoom_1to1)
    
    def _setup_canvas_callbacks(self):
        """캔버스 콜백 설정 (줌/팬)"""
        self.view.canvas_view.on_zoom = self._handle_zoom
        self.view.canvas_view.on_pan = self._handle_pan
        self.view.canvas_view.on_roi_draw = self._handle_roi_draw
    
    def _handle_zoom(self, factor: float):
        """줌 처리"""
        new_zoom = max(0.1, min(5.0, self.state.zoom_level * factor))
        self.state.zoom_level = new_zoom
        self._refresh_display()
        self.view.zoom_label.configure(text=f"{int(new_zoom * 100)}%")
    
    def _handle_pan(self, dx: int, dy: int):
        """팬 처리"""
        ox, oy = self.state.pan_offset
        self.state.pan_offset = (ox + dx, oy + dy)
        self._refresh_display()
    
    def _handle_roi_draw(self, roi: tuple):
        """ROI 그리기"""
        self.state.roi = roi
        self._refresh_display()
    
    def _warmup(self):
        """FFT 워밍업"""
        try:
            warmup_fft_cc()
        except Exception:
            pass
    
    # ===== 이미지 선택 =====
    
    def select_reference(self):
        """Reference 이미지 선택"""
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
        """Reference 이미지 로드"""
        try:
            img = load_image(path)
            self.state.ref_image = img
            self.state.ref_path = path
            self.view.update_reference_label(path.name)
            self._display_image(img)
            self.fit_to_canvas()
        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 실패: {e}")
    
    def select_deformed(self):
        """Deformed 이미지 선택"""
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
        """Deformed 이미지 로드"""
        try:
            img = load_image(path)
            self.state.def_image = img
            self.state.def_path = path
            self.view.update_deformed_label(path.name)
            self._display_image(img)
        except Exception as e:
            messagebox.showerror("오류", f"이미지 로드 실패: {e}")
    
    def select_sequence(self):
        """시퀀스 폴더 선택"""
        folder = filedialog.askdirectory(title="이미지 시퀀스 폴더 선택")
        if folder:
            self._load_sequence(Path(folder))
    
    def _load_sequence(self, folder: Path):
        """시퀀스 폴더 로드"""
        try:
            files = get_image_files(folder)
            if len(files) < 2:
                messagebox.showwarning("경고", "폴더에 2개 이상의 이미지가 필요합니다.")
                return
            
            self.state.sequence_folder = folder
            self.state.sequence_files = files
            self.state.current_index = 0
            
            # 첫 이미지를 Reference로 설정
            self._load_reference(files[0])
            
            # 두 번째 이미지를 Deformed로 설정
            if len(files) > 1:
                self._load_deformed(files[1])
                self.state.current_index = 1
            
            # UI 업데이트
            self.view.update_sequence_label(f"{folder.name} ({len(files)} files)")
            self.view.update_file_list(files, self.state.current_index)
            
        except Exception as e:
            messagebox.showerror("오류", f"폴더 로드 실패: {e}")
    
    def select_image_index(self, index: int):
        """파일 목록에서 이미지 선택"""
        if not self.state.sequence_files:
            return
        
        if 0 <= index < len(self.state.sequence_files):
            self.state.current_index = index
            path = self.state.sequence_files[index]
            
            self._load_deformed(path)
            self.view.set_current_index(index)
            
            # 해당 인덱스의 결과가 있으면 표시
            if path.name in self.state.batch_results:
                result = self.state.batch_results[path.name]
                self.state.fft_cc_result = result
                self._refresh_display()
                self._update_result_ui(result)
            else:
                # 결과 없으면 초기화
                self.state.fft_cc_result = None
                self._refresh_display()
    
    def _update_result_ui(self, result: FFTCCResult):
        """결과 UI 업데이트"""
        u_min = np.min(result.disp_u) if len(result.disp_u) > 0 else 0
        u_max = np.max(result.disp_u) if len(result.disp_u) > 0 else 0
        v_min = np.min(result.disp_v) if len(result.disp_v) > 0 else 0
        v_max = np.max(result.disp_v) if len(result.disp_v) > 0 else 0
        
        text = f"""총 POI: {result.n_points}
유효 포인트: {result.n_valid}
불량 포인트: {result.n_invalid}
유효율: {result.valid_ratio * 100:.1f}%
평균 ZNCC: {result.mean_zncc:.4f}

변위 범위:
  U: [{u_min}, {u_max}] px
  V: [{v_min}, {v_max}] px

처리 시간: {result.processing_time:.2f}초"""
        
        self.view.update_result_text(text)
    
    # ===== 동기화 =====
    
    def sync_from_quality_tab(self):
        """품질평가 탭에서 파라미터 동기화"""
        if self.main_controller is None:
            messagebox.showinfo("정보", "품질평가 탭과 연결되지 않았습니다.")
            return
        
        try:
            quality_params = self.main_controller.get_current_parameters()
            
            params = {
                'subset_size': quality_params.get('subset_size', 21),
                'spacing': quality_params.get('spacing', 10)
            }
            
            roi = self.main_controller.get_roi()
            if roi:
                self.state.roi = roi
                print(f"[DEBUG] ROI 동기화: {roi}")
            
            self.view.set_parameters(params)
            self._refresh_display()  # ROI 표시 업데이트
            messagebox.showinfo("동기화 완료", f"품질평가 탭의 파라미터를 가져왔습니다.\nROI: {roi}")
            
        except Exception as e:
            messagebox.showerror("오류", f"동기화 실패: {e}")
    
    # ===== 분석 =====
    
    def run_fft_cc(self, params: Dict[str, Any]):
        """FFT-CC 분석 실행"""
        if self.state.ref_image is None or self.state.def_image is None:
            messagebox.showwarning("경고", "Reference와 Deformed 이미지를 모두 선택해주세요.")
            return
        
        if self.state.is_running:
            return
        
        self.state.is_running = True
        self.state.should_stop = False
        self.view.set_analysis_state(True)
        
        # 현재 ROI 출력
        print(f"[DEBUG] 분석 시작 - ROI: {self.state.roi}")
        
        def worker():
            error_msg = None
            result = None
            validation = None
            
            try:
                def progress_callback(current, total):
                    if self.state.should_stop:
                        raise InterruptedError("사용자 중단")
                    progress = (current / total) * 100
                    self.view.after(0, lambda p=progress, c=current, t=total: 
                                    self.view.update_progress(p, f"처리 중: {c}/{t}"))
                
                result = compute_fft_cc(
                    self.state.ref_image,
                    self.state.def_image,
                    subset_size=params['subset_size'],
                    spacing=params['spacing'],
                    search_range=params['search_range'],
                    zncc_threshold=params['zncc_threshold'],
                    roi=self.state.roi,
                    progress_callback=progress_callback
                )
                
                print(f"[DEBUG] 분석 완료 - POI: {result.n_points}, Valid: {result.n_valid}")
                
                validation = validate_displacement_field(
                    result,
                    zncc_threshold=params['zncc_threshold']
                )
                
                self.state.fft_cc_result = result
                self.state.validation_result = validation
                
                if self.state.def_path:
                    self.state.batch_results[self.state.def_path.name] = result
                
            except InterruptedError:
                self.view.after(0, lambda: self.view.update_progress(0, "중단됨"))
            except Exception as ex:
                error_msg = str(ex)
                print(f"[DEBUG] 분석 오류: {error_msg}")
            finally:
                self.state.is_running = False
                
                if error_msg:
                    self.view.after(0, lambda msg=error_msg: self._show_error(msg))
                elif result and validation:
                    self.view.after(0, lambda r=result, v=validation: self._on_analysis_complete(r, v))
                
                self.view.after(0, lambda: self.view.set_analysis_state(False))
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    def _show_error(self, msg: str):
        """에러 메시지 표시"""
        messagebox.showerror("오류", f"분석 실패: {msg}")
    
    def run_batch_fft_cc(self, params: Dict[str, Any]):
        """전체 시퀀스 배치 분석"""
        if not self.state.sequence_files or len(self.state.sequence_files) < 2:
            messagebox.showwarning("경고", "시퀀스 폴더를 먼저 선택해주세요.")
            return
        
        if self.state.is_running:
            return
        
        self.state.is_running = True
        self.state.should_stop = False
        self.state.batch_results.clear()
        self.view.set_analysis_state(True)
        
        print(f"[DEBUG] 배치 분석 시작 - ROI: {self.state.roi}")
        
        def worker():
            error_msg = None
            last_result = None
            
            try:
                files = self.state.sequence_files
                total_files = len(files) - 1
                
                print(f"[DEBUG] 배치 시작: {total_files} 파일")
                
                for i, def_path in enumerate(files[1:], start=1):
                    if self.state.should_stop:
                        print(f"[DEBUG] 사용자 중단")
                        break
                    
                    print(f"[DEBUG] 처리 중: {i}/{total_files} - {def_path.name}")
                    
                    idx = i
                    self.view.after(0, lambda idx=idx: self.view.set_current_index(idx))
                    
                    def_img = load_image(def_path)
                    
                    if def_img is None:
                        print(f"[DEBUG] 이미지 로드 실패: {def_path}")
                        continue
                    
                    # 클로저 문제 해결을 위해 변수 캡처
                    current_i = i
                    current_name = def_path.name
                    
                    def progress_callback(current, total):
                        if self.state.should_stop:
                            raise InterruptedError("사용자 중단")
                        file_progress = (current_i - 1) / total_files
                        point_progress = (current / total) / total_files
                        overall = (file_progress + point_progress) * 100
                        self.view.after(0, lambda o=overall, fn=current_name, fi=current_i, tf=total_files: 
                                        self.view.update_progress(o, f"파일 {fi}/{tf}: {fn}"))
                    
                    result = compute_fft_cc(
                        self.state.ref_image,
                        def_img,
                        subset_size=params['subset_size'],
                        spacing=params['spacing'],
                        search_range=params['search_range'],
                        zncc_threshold=params['zncc_threshold'],
                        roi=self.state.roi,
                        progress_callback=progress_callback
                    )
                    
                    print(f"[DEBUG] 결과: {result.n_points} POI, {result.n_valid} valid")
                    
                    self.state.batch_results[def_path.name] = result
                    last_result = result
                
                print(f"[DEBUG] 배치 완료: {len(self.state.batch_results)} 결과 저장됨")
                
            except InterruptedError:
                self.view.after(0, lambda: self.view.update_progress(0, "중단됨"))
            except Exception as ex:
                error_msg = str(ex)
                print(f"[DEBUG] 배치 오류: {error_msg}")
                import traceback
                traceback.print_exc()
            finally:
                self.state.is_running = False
                
                if error_msg:
                    self.view.after(0, lambda msg=error_msg: self._show_error(msg))
                else:
                    # 마지막 결과 표시
                    if last_result:
                        self.state.fft_cc_result = last_result
                    self.view.after(0, self._on_batch_complete)
                
                self.view.after(0, lambda: self.view.set_analysis_state(False))
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
    
    def stop_analysis(self):
        """분석 중지"""
        self.state.should_stop = True
    
    def _on_analysis_complete(self, result: FFTCCResult, validation: ValidationResult):
        """분석 완료 처리"""
        self.view.update_progress(100, "완료")
        self._display_result(result)
        
        self._update_result_ui(result)
        
        # 검증 결과
        valid_text = f"""이상치 비율: {validation.outlier_ratio * 100:.1f}%
불연속 영역: {validation.discontinuity_ratio * 100:.1f}%
전체 유효: {'예' if validation.is_valid else '아니오'}
권장 조치: {validation.suggested_action}"""
        
        self.view.update_validation_text(valid_text)
    
    def _on_batch_complete(self):
        """배치 분석 완료"""
        self.view.update_progress(100, f"완료: {len(self.state.batch_results)} 파일 처리됨")
        self._refresh_display()
        messagebox.showinfo("완료", f"배치 분석 완료\n처리된 파일: {len(self.state.batch_results)}")
    
    # ===== 표시 =====
    
    def _display_image(self, img: np.ndarray):
        """이미지 표시"""
        if img is None:
            return
        self._refresh_display()
    
    def _refresh_display(self):
        """현재 상태로 디스플레이 갱신"""
        img = self.state.def_image if self.state.def_image is not None else self.state.ref_image
        if img is None:
            return
        
        # 결과가 있으면 오버레이 적용
        if self.state.fft_cc_result is not None:
            display_img = self._create_overlay_image(img)
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
    
    def _create_overlay_image(self, base_img: np.ndarray) -> np.ndarray:
        """결과 오버레이 이미지 생성"""
        result = self.state.fft_cc_result
        if result is None:
            return base_img
        
        # 컬러 이미지로 변환
        if len(base_img.shape) == 2:
            display_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            display_img = base_img.copy()
        
        mode = self.view.display_mode_var.get()
        scale = self.view.vector_scale_var.get()
        
        # ROI 오프셋 계산
        offset_x = 0
        offset_y = 0
        if self.state.roi is not None:
            offset_x = self.state.roi[0]
            offset_y = self.state.roi[1]
        
        if mode == "vectors":
            display_img = self._draw_vectors(display_img, result, scale, offset_x, offset_y)
        elif mode == "magnitude":
            display_img = self._draw_magnitude(display_img, result, offset_x, offset_y)
        elif mode == "zncc":
            display_img = self._draw_zncc_map(display_img, result, offset_x, offset_y)
        elif mode == "invalid":
            display_img = self._draw_invalid_points(display_img, result, offset_x, offset_y)
        
        return display_img
    
    def _draw_vectors(self, img: np.ndarray, result: FFTCCResult, scale: float,
                      offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
        """변위 벡터 그리기"""
        if result.n_points == 0:
            return img
        
        max_mag = max(np.max(np.abs(result.disp_u)), np.max(np.abs(result.disp_v)), 1)
        
        for idx in range(result.n_points):
            if not result.valid_mask[idx]:
                continue
            
            # ROI 오프셋 적용
            x = int(result.points_x[idx]) + offset_x
            y = int(result.points_y[idx]) + offset_y
            u = result.disp_u[idx]
            v = result.disp_v[idx]
            
            # 이미지 범위 확인
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            
            # 벡터 끝점 계산 (스케일 적용)
            end_x = int(x + u * scale * 5)
            end_y = int(y + v * scale * 5)
            
            # 변위 크기에 따른 색상
            magnitude = np.sqrt(u*u + v*v)
            ratio = min(magnitude / max_mag, 1.0) if max_mag > 0 else 0
            
            hue = int((1 - ratio) * 120)
            color = self._hsv_to_bgr(hue, 255, 255)
            
            # 화살표 그리기
            cv2.arrowedLine(img, (x, y), (end_x, end_y), color, 1, tipLength=0.3)
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        
        return img
    
    def _draw_magnitude(self, img: np.ndarray, result: FFTCCResult,
                        offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
        """변위 크기 컬러맵 그리기"""
        if result.n_points == 0:
            return img
        
        magnitudes = np.sqrt(result.disp_u.astype(float)**2 + result.disp_v.astype(float)**2)
        max_mag = np.max(magnitudes) if len(magnitudes) > 0 else 1
        
        for idx in range(result.n_points):
            x = int(result.points_x[idx]) + offset_x
            y = int(result.points_y[idx]) + offset_y
            
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            
            if result.valid_mask[idx]:
                norm_mag = magnitudes[idx] / max_mag if max_mag > 0 else 0
                color = self._magnitude_to_color(norm_mag)
                cv2.circle(img, (x, y), 4, color, -1)
            else:
                cv2.circle(img, (x, y), 4, (128, 128, 128), -1)
        
        img = self._draw_colorbar(img, 0, max_mag, "Displacement (px)")
        
        return img
    
    def _draw_zncc_map(self, img: np.ndarray, result: FFTCCResult,
                       offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
        """ZNCC 맵 그리기"""
        if result.n_points == 0:
            return img
        
        for idx in range(result.n_points):
            x = int(result.points_x[idx]) + offset_x
            y = int(result.points_y[idx]) + offset_y
            
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            
            zncc = result.zncc_values[idx]
            ratio = max(0, min(1, zncc))
            
            r = int((1 - ratio) * 255)
            g = int(ratio * 255)
            b = 0
            
            cv2.circle(img, (x, y), 4, (b, g, r), -1)
        
        img = self._draw_colorbar(img, 0, 1, "ZNCC")
        
        return img
    
    def _draw_invalid_points(self, img: np.ndarray, result: FFTCCResult,
                             offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
        """불량 포인트 강조 표시"""
        if result.n_points == 0:
            return img
        
        for idx in range(result.n_points):
            x = int(result.points_x[idx]) + offset_x
            y = int(result.points_y[idx]) + offset_y
            
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            
            if result.valid_mask[idx]:
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
            else:
                cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
                cv2.drawMarker(img, (x, y), (255, 255, 255), 
                              cv2.MARKER_CROSS, 10, 2)
        
        return img
    
    def _hsv_to_bgr(self, h: int, s: int, v: int) -> tuple:
        """HSV to BGR 변환"""
        hsv = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])
    
    def _magnitude_to_color(self, norm_value: float) -> tuple:
        """정규화된 값을 JET 컬러맵 색상으로 변환"""
        if norm_value < 0.25:
            r = 0
            g = int(255 * (norm_value / 0.25))
            b = 255
        elif norm_value < 0.5:
            r = 0
            g = 255
            b = int(255 * (1 - (norm_value - 0.25) / 0.25))
        elif norm_value < 0.75:
            r = int(255 * ((norm_value - 0.5) / 0.25))
            g = 255
            b = 0
        else:
            r = 255
            g = int(255 * (1 - (norm_value - 0.75) / 0.25))
            b = 0
        
        return (b, g, r)
    
    def _draw_colorbar(self, img: np.ndarray, min_val: float, max_val: float, label: str) -> np.ndarray:
        """컬러바 그리기"""
        h, w = img.shape[:2]
        
        bar_width = 20
        bar_height = 150
        margin = 10
        
        x_start = w - bar_width - margin
        y_start = margin
        
        cv2.rectangle(img, (x_start - 5, y_start - 5), 
                     (x_start + bar_width + 40, y_start + bar_height + 25),
                     (40, 40, 40), -1)
        
        for i in range(bar_height):
            norm_val = 1 - (i / bar_height)
            color = self._magnitude_to_color(norm_val)
            cv2.line(img, (x_start, y_start + i), 
                    (x_start + bar_width, y_start + i), color, 1)
        
        cv2.putText(img, f"{max_val:.1f}", (x_start + bar_width + 5, y_start + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(img, f"{min_val:.1f}", (x_start + bar_width + 5, y_start + bar_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        
        return img
    
    def _display_result(self, result: FFTCCResult):
        """결과 오버레이 표시"""
        self.state.fft_cc_result = result
        self._refresh_display()
    
    def update_display(self, mode: str, scale: float):
        """표시 모드 업데이트"""
        self._refresh_display()

    # ===== 줌/캔버스 =====
    
    def fit_to_canvas(self):
        """캔버스에 맞추기"""
        img = self.state.def_image if self.state.def_image is not None else self.state.ref_image
        if img is not None:
            zoom = self.view.canvas_view.fit_to_canvas(img.shape[:2])
            self.state.zoom_level = zoom
            self.state.pan_offset = (0, 0)
            self._refresh_display()
            self.view.zoom_label.configure(text=f"{int(zoom * 100)}%")
    
    def zoom_in(self):
        """줌 인"""
        self._handle_zoom(1.2)
    
    def zoom_out(self):
        """줌 아웃"""
        self._handle_zoom(0.8)
    
    def set_zoom_1to1(self):
        """1:1 줌"""
        self.state.zoom_level = 1.0
        self.state.pan_offset = (0, 0)
        self._refresh_display()
        self.view.zoom_label.configure(text="100%")

    # ===== 내보내기 =====
    
    def export_csv(self):
        """결과 CSV 내보내기"""
        if self.state.fft_cc_result is None:
            messagebox.showwarning("경고", "내보낼 결과가 없습니다.")
            return
        
        path = filedialog.asksaveasfilename(
            title="CSV 저장",
            defaultextension=".csv",
            filetypes=[("CSV 파일", "*.csv")]
        )
        if path:
            try:
                result = self.state.fft_cc_result
                import csv
                with open(path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(['x', 'y', 'u', 'v', 'zncc', 'valid'])
                    for pt in result.points:
                        writer.writerow([pt.x, pt.y, pt.u, pt.v, pt.zncc, pt.is_valid])
                messagebox.showinfo("완료", "CSV 저장 완료")
            except Exception as e:
                messagebox.showerror("오류", f"저장 실패: {e}")
    
    def export_image(self):
        """결과 이미지 내보내기"""
        messagebox.showinfo("정보", "이미지 내보내기 기능은 추후 구현 예정입니다.")
