"""DIC 분석 컨트롤러"""

import threading
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from tkinter import filedialog, messagebox
from speckle.core.initial_guess import (
    compute_fft_cc, compute_fft_cc_batch_cached, warmup_fft_cc, 
    validate_displacement_field,
    FFTCCResult, ValidationResult
)
from speckle.io import load_image, get_image_files
from ..models.app_state import AppState
from speckle.core.optimization import compute_icgn
from speckle.core.postprocess import compute_strain_from_icgn, StrainResult
from speckle.core.postprocess.strain_smoothing import compute_strain_savgol, SmoothedStrainResult

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
    
    # 줌/팬 상태
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
        self.app_state = app_state  # 공유 상태 (이미지 캐시)
        self.main_controller = main_controller
        self.state = DICState()  # DIC 전용 상태
        
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
            # 먼저 캐시에서 찾기
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
            # 먼저 캐시에서 찾기
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
            
            self._load_reference(files[0])
            
            if len(files) > 1:
                self._load_deformed(files[1])
                self.state.current_index = 1
            
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
            self.view.current_index = index  # 뷰 상태도 동기화
            path = self.state.sequence_files[index]
            
            self._load_deformed(path)
            self.view.set_current_index(index)
            
            # 기존 결과가 있으면 표시
            if path.name in self.state.batch_results:
                result = self.state.batch_results[path.name]
                self.state.fft_cc_result = result
                self._refresh_display()
                self._update_result_ui(result)
            else:
                self.state.fft_cc_result = None
                self._refresh_display()
    
    def _update_result_ui(self, result):
        """결과 UI 업데이트 (깔끔한 버전)"""
        if result is None:
            return
        
        u = result.disp_u
        v = result.disp_v
        valid = result.valid_mask if hasattr(result, 'valid_mask') else np.ones(len(u), bool)
        
        u_valid = u[valid]
        v_valid = v[valid]
        
        is_icgn = hasattr(result, 'converged')
        filename = self.state.def_path.name if self.state.def_path else "N/A"
        
        if is_icgn:
            lines = []
            lines.append(f"Shape: {result.shape_function}")
            lines.append("")
            lines.append(f"POI: {result.n_converged}/{result.n_points} ({result.convergence_rate*100:.1f}%)")
            lines.append(f"반복: {result.mean_iterations:.1f}회 | ZNCC: {result.mean_zncc:.4f}")
            lines.append("")
            lines.append("── 변위 ──")
            lines.append(f"U: {np.mean(u_valid):.4f} ± {np.std(u_valid):.4f}")
            lines.append(f"   [{np.min(u_valid):.4f} ~ {np.max(u_valid):.4f}]")
            lines.append(f"V: {np.mean(v_valid):.4f} ± {np.std(v_valid):.4f}")
            lines.append(f"   [{np.min(v_valid):.4f} ~ {np.max(v_valid):.4f}]")
            
            # 변형률
            if hasattr(result, 'disp_ux') and result.disp_ux is not None:
                ux = result.disp_ux[valid]
                uy = result.disp_uy[valid]
                vx = result.disp_vx[valid]
                vy = result.disp_vy[valid]
                
                lines.append("")
                lines.append("── 변형률 ──")
                lines.append(f"ux: {np.mean(ux): .5f} ± {np.std(ux):.5f}")
                lines.append(f"uy: {np.mean(uy): .5f} ± {np.std(uy):.5f}")
                lines.append(f"vx: {np.mean(vx): .5f} ± {np.std(vx):.5f}")
                lines.append(f"vy: {np.mean(vy): .5f} ± {np.std(vy):.5f}")
            
            lines.append("")
            lines.append(f"시간: {result.processing_time:.2f}초")
            
            text = "\n".join(lines)
        
        else:
            # FFTCCResult
            text = f"""파일: {filename}

    POI: {result.n_valid}/{result.n_points} ({result.valid_ratio*100:.1f}%)
    ZNCC: {result.mean_zncc:.4f}

    ── 변위 ──
    U: {np.mean(u_valid):.4f} ± {np.std(u_valid):.4f}
    [{np.min(u_valid):.4f} ~ {np.max(u_valid):.4f}]
    V: {np.mean(v_valid):.4f} ± {np.std(v_valid):.4f}
    [{np.min(v_valid):.4f} ~ {np.max(v_valid):.4f}]

    시간: {result.processing_time:.2f}초"""
        
        self.view.update_result_text(text)


    # ===== 동기화 (핵심 수정) =====
    
    def sync_from_quality_tab(self):
        """품질평가 탭에서 동기화 (캐시 사용)"""
        # 이미지 캐시 확인
        if not self.app_state.has_images():
            messagebox.showinfo("정보", "품질평가 탭에서 폴더를 먼저 열어주세요.")
            return
        
        # 로딩 중이면 경고
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
            # 파일 목록 동기화
            self.state.sequence_files = self.app_state.file_paths
            self.state.sequence_folder = self.app_state.folder_path
            
            # 첫 이미지를 Reference로
            if self.app_state.file_list:
                ref_name = self.app_state.file_list[0]
                ref_img = self.app_state.get_image(ref_name)
                if ref_img is not None:
                    self.state.ref_image = ref_img
                    self.state.ref_path = self.app_state.file_paths[0] if self.app_state.file_paths else None
                    self.view.update_reference_label(ref_name)
            
            # 두 번째 이미지를 Deformed로
            if len(self.app_state.file_list) > 1:
                def_name = self.app_state.file_list[1]
                def_img = self.app_state.get_image(def_name)
                if def_img is not None:
                    self.state.def_image = def_img
                    self.state.def_path = self.app_state.file_paths[1] if len(self.app_state.file_paths) > 1 else None
                    self.state.current_index = 1
                    self.view.update_deformed_label(def_name)
            
            # ROI 동기화
            self.state.roi = self.app_state.roi
                # ===== 여기에 디버그 추가 =====
            print(f"\n[DEBUG] ===== 동기화 확인 =====")
            print(f"[DEBUG] app_state.roi: {self.app_state.roi}")
            print(f"[DEBUG] self.state.roi: {self.state.roi}")
            print(f"[DEBUG] ================================\n")
            # ===== 디버그 끝 ====

            # 파라미터 동기화
            if self.main_controller:
                quality_params = self.main_controller.get_current_parameters()
                params = {
                    'subset_size': quality_params.get('subset_size', 21),
                    'spacing': quality_params.get('spacing', 16)
                }
                self.view.set_parameters(params)
            
            # UI 업데이트
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
    
    # ===== 분석 =====
    def run_fft_cc(self, params: Dict[str, Any]):
        """FFT-CC + IC-GN 분석 실행"""
        if self.state.ref_image is None or self.state.def_image is None:
            messagebox.showwarning("경고", "Reference와 Deformed 이미지를 모두 선택해주세요.")
            return
        
        if self.state.is_running:
            return
        
        self.state.is_running = True
        self.state.should_stop = False
        self.view.set_analysis_state(True)
        
        print(f"[DEBUG] 분석 시작 - ROI: {self.state.roi}")
        
        def worker():
            error_msg = None
            fftcc_result = None
            icgn_result = None
            
            try:
                # === 1단계: FFTCC ===
                def fftcc_progress(current, total):
                    if self.state.should_stop:
                        raise InterruptedError("사용자 중단")
                    progress = (current / total) * 50  # 0-50%
                    self.view.after(0, lambda p=progress, c=current, t=total: 
                                    self.view.update_progress(p, f"FFTCC: {c}/{t}"))
                
                fftcc_result = compute_fft_cc(
                    self.state.ref_image,
                    self.state.def_image,
                    subset_size=params['subset_size'],
                    spacing=params['spacing'],
                    search_range=params['search_range'],
                    zncc_threshold=params['zncc_threshold'],
                    roi=self.state.roi,
                    progress_callback=fftcc_progress
                )
                
                print(f"[DEBUG] FFTCC 완료 - POI: {fftcc_result.n_points}, Valid: {fftcc_result.n_valid}")
                print(f"[DEBUG] FFTCC U range: [{np.min(fftcc_result.disp_u):.4f}, {np.max(fftcc_result.disp_u):.4f}]")
                print(f"[DEBUG] FFTCC V range: [{np.min(fftcc_result.disp_v):.4f}, {np.max(fftcc_result.disp_v):.4f}]")
                print(f"[DEBUG] FFTCC Mean ZNCC: {fftcc_result.mean_zncc:.4f}")

                if self.state.should_stop:
                    raise InterruptedError("사용자 중단")
                
                # === 2단계: IC-GN ===
                self.view.after(0, lambda: self.view.update_progress(50, "IC-GN 시작..."))

                def icgn_progress(current, total):
                    if self.state.should_stop:
                        raise InterruptedError("사용자 중단")
                    progress = 50 + (current / total) * 50
                    self.view.after(0, lambda p=progress, c=current, t=total: 
                                    self.view.update_progress(p, f"IC-GN: {c}/{t}"))

                interp_order = 5 if params['interpolation'] == 'biquintic' else 3

                # POI 좌표를 전체 이미지 좌표로 변환
                if self.state.roi:
                    rx, ry, rw, rh = self.state.roi
                    fftcc_result.points_x = fftcc_result.points_x + rx
                    fftcc_result.points_y = fftcc_result.points_y + ry

                # 전체 이미지 사용
                icgn_result = compute_icgn(
                    self.state.ref_image,
                    self.state.def_image,
                    initial_guess=fftcc_result,
                    subset_size=params['subset_size'],
                    shape_function=params['shape_function'],
                    interpolation_order=interp_order,
                    convergence_threshold=params['conv_threshold'],
                    max_iterations=params['max_iter'],
                    gaussian_blur=params.get('gaussian_blur'),
                    progress_callback=icgn_progress
                )

                print(f"\n[DEBUG] ===== IC-GN 입력 확인 =====")
                print(f"[DEBUG] ROI: {self.state.roi}")
                print(f"[DEBUG] Ref image shape: {self.state.ref_image.shape}")
                print(f"[DEBUG] Def image shape: {self.state.def_image.shape}")
                print(f"[DEBUG] FFTCC points_x range: [{np.min(fftcc_result.points_x)}, {np.max(fftcc_result.points_x)}]")
                print(f"[DEBUG] FFTCC points_y range: [{np.min(fftcc_result.points_y)}, {np.max(fftcc_result.points_y)}]")
                print(f"[DEBUG] FFTCC disp_u (처음 5개): {fftcc_result.disp_u[:5]}")
                print(f"[DEBUG] FFTCC disp_v (처음 5개): {fftcc_result.disp_v[:5]}")
                # 첫 번째 POI 상세 확인
                idx = 0
                px, py = fftcc_result.points_x[idx], fftcc_result.points_y[idx]
                du, dv = fftcc_result.disp_u[idx], fftcc_result.disp_v[idx]
                print(f"[DEBUG] 첫 POI: 위치=({px}, {py}), FFTCC변위=({du}, {dv})")

                if self.state.roi:
                    rx, ry, rw, rh = self.state.roi
                    print(f"[DEBUG] 첫 POI 전체이미지 좌표: ({px + rx}, {py + ry})")
                print(f"[DEBUG] ================================\n")

                self.state.fft_cc_result = fftcc_result  # 로그용 보관
                self.state.icgn_result = icgn_result     # 최종 결과
                
                if self.state.def_path:
                    self.state.batch_results[self.state.def_path.name] = icgn_result
                
            except InterruptedError:
                self.view.after(0, lambda: self.view.update_progress(0, "중단됨"))
            except Exception as ex:
                error_msg = str(ex)
                print(f"[DEBUG] 분석 오류: {error_msg}")
                import traceback
                traceback.print_exc()
            finally:
                self.state.is_running = False
                
                if error_msg:
                    self.view.after(0, lambda msg=error_msg: self._show_error(msg))
                elif icgn_result:
                    self.view.after(0, lambda r=icgn_result: self._on_icgn_complete(r))
                
                self.view.after(0, lambda: self.view.set_analysis_state(False))
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def _on_icgn_complete(self, result):
        """IC-GN 분석 완료 처리"""
        self.view.update_progress(100, "완료")
        
        # 결과 표시용으로 저장 (시각화에 사용)
        self.state.fft_cc_result = result  # ICGNResult도 동일한 인터페이스 사용
        self._refresh_display()
        self._update_result_ui(result)
        
        # 검증 결과 (간단히)
        valid_text = f"""수렴율: {result.convergence_rate * 100:.1f}%
    유효 POI: {result.n_valid}/{result.n_points}
    평균 반복: {result.mean_iterations:.1f}회
    평균 ZNCC: {result.mean_zncc:.4f}"""
        
        self.view.update_validation_text(valid_text)

    def _show_error(self, msg: str):
        messagebox.showerror("오류", f"분석 실패: {msg}")
    
    def run_batch_fft_cc(self, params: Dict[str, Any]):
        """전체 시퀀스 배치 분석 (캐시 사용 최적화 버전)"""
        # 캐시 사용 가능 여부 확인
        use_cache = self.app_state.has_images() and len(self.app_state.images) > 1
        
        if use_cache:
            self._run_batch_cached(params)
        else:
            self._run_batch_streaming(params)
    
    def _run_batch_cached(self, params: Dict[str, Any]):
        """캐시된 이미지로 배치 분석 (FFTCC + IC-GN) - 시간 분석 포함"""
        if self.state.is_running:
            return
        
        if self.app_state.loading_state.is_loading:
            result = messagebox.askyesno(
                "로딩 중",
                f"이미지 로딩 중입니다.\n"
                f"({self.app_state.loading_state.loaded_files}/{self.app_state.loading_state.total_files})\n\n"
                f"로드된 이미지만 분석하시겠습니까?"
            )
            if not result:
                return
        
        self.state.is_running = True
        self.state.should_stop = False
        self.state.batch_results.clear()
        self.view.set_analysis_state(True)
        
        print(f"[DEBUG] 캐시 배치 분석 시작 (FFTCC + IC-GN)")
        
        def worker():
            error_msg = None
            
            # ===== 시간 측정용 변수 =====
            import time
            import traceback
            timings = {
                'image_load': [],
                'fftcc': [],
                'icgn': [],
                'icgn_internal': [],
                'result_save': [],
                'gui_update': [],
            }
            batch_start = time.time()
            
            try:
                all_files = self.state.sequence_files if self.state.sequence_files else self.app_state.file_paths
                cached_files = [f for f in all_files if f.name in self.app_state.images]

                if not cached_files:
                    raise ValueError("분석할 캐시된 이미지가 없습니다.")
                
                total_files = len(cached_files)
                print(f"[DEBUG] 분석 대상: {total_files}개 파일")
                
                for file_idx, def_path in enumerate(cached_files):
                    if self.state.should_stop:
                        break
                    
                    filename = def_path.name
                    iter_start = time.time()
                    
                    # 진행률 업데이트
                    t_gui_start = time.time()
                    progress = (file_idx / total_files) * 100
                    self.view.after(0, lambda p=progress, fn=filename, fi=file_idx, tf=total_files:
                                self.view.update_progress(p, f"분석 중 {fi+1}/{tf}: {fn}"))
                    self.view.after(0, lambda idx=file_idx+1: self.view.set_current_index(idx))
                    t_gui_end = time.time()
                    timings['gui_update'].append(t_gui_end - t_gui_start)
                    
                    # 캐시에서 이미지 가져오기
                    t_load_start = time.time()
                    def_image = self.app_state.get_image(filename)
                    t_load_end = time.time()
                    timings['image_load'].append(t_load_end - t_load_start)
                    
                    if def_image is None:
                        continue
                    
                    # === 1단계: FFTCC ===
                    t_fftcc_start = time.time()
                    fftcc_result = compute_fft_cc(
                        self.state.ref_image,
                        def_image,
                        subset_size=params['subset_size'],
                        spacing=params['spacing'],
                        search_range=params['search_range'],
                        zncc_threshold=params['zncc_threshold'],
                        roi=self.state.roi
                    )
                    t_fftcc_end = time.time()
                    timings['fftcc'].append(t_fftcc_end - t_fftcc_start)
                    
                    print(f"\n[DEBUG] ===== {filename} =====")
                    print(f"[DEBUG] FFTCC U: [{np.min(fftcc_result.disp_u)}, {np.max(fftcc_result.disp_u)}]")
                    print(f"[DEBUG] FFTCC V: [{np.min(fftcc_result.disp_v)}, {np.max(fftcc_result.disp_v)}]")
                    print(f"[DEBUG] 파라미터: gaussian_blur = {params.get('gaussian_blur')}")
                    
                   # === 2단계: IC-GN ===
                    t_icgn_start = time.time()
                    interp_order = 5 if params['interpolation'] == 'biquintic' else 3

                    # POI 좌표를 전체 이미지 좌표로 변환
                    if self.state.roi:
                        rx, ry, rw, rh = self.state.roi
                        fftcc_result.points_x = fftcc_result.points_x + rx
                        fftcc_result.points_y = fftcc_result.points_y + ry

                    # 전체 이미지 사용
                    icgn_result = compute_icgn(
                        self.state.ref_image,
                        def_image,
                        initial_guess=fftcc_result,
                        subset_size=params['subset_size'],
                        shape_function=params['shape_function'],
                        interpolation_order=interp_order,
                        convergence_threshold=params['conv_threshold'],
                        max_iterations=params['max_iter'],
                        gaussian_blur=params.get('gaussian_blur')
                    )
                    t_icgn_end = time.time()
                    timings['icgn'].append(t_icgn_end - t_icgn_start)
                    timings['icgn_internal'].append(icgn_result.processing_time)


                    # 결과 저장
                    t_save_start = time.time()
                    self.state.batch_results[filename] = icgn_result
                    t_save_end = time.time()
                    timings['result_save'].append(t_save_end - t_save_start)

                    iter_end = time.time()

                    print(f"[DEBUG] {file_idx+1}/{total_files}: {filename} - 수렴 {icgn_result.n_converged}/{icgn_result.n_points}")
                    print(f"[DEBUG] IC-GN U: [{np.min(icgn_result.disp_u):.4f}, {np.max(icgn_result.disp_u):.4f}]")
                    print(f"[DEBUG] IC-GN V: [{np.min(icgn_result.disp_v):.4f}, {np.max(icgn_result.disp_v):.4f}]")

                    # 개별 파일 시간 출력
                    print(f"[TIME] {filename}:")
                    print(f"       FFTCC: {timings['fftcc'][-1]:.3f}초")
                    print(f"       IC-GN 전체: {timings['icgn'][-1]:.3f}초")
                    print(f"       IC-GN 내부: {timings['icgn_internal'][-1]:.3f}초")
                    print(f"       반복 전체: {iter_end - iter_start:.3f}초")


                # 마지막 결과 저장
                if self.state.batch_results:
                    last_key = list(self.state.batch_results.keys())[-1]
                    self.state.icgn_result = self.state.batch_results[last_key]
                    self.state.fft_cc_result = self.state.batch_results[last_key]
                
                batch_end = time.time()
                
                # ===== 시간 분석 결과 출력 =====
                print(f"\n{'='*60}")
                print(f"[TIME] ========== 시간 분석 결과 ==========")
                print(f"{'='*60}")
                print(f"분석 파일 수: {len(timings['fftcc'])}개")
                print(f"")
                print(f"[평균 시간 per 파일]")
                print(f"  이미지 로드:    {np.mean(timings['image_load'])*1000:.1f} ms")
                print(f"  FFTCC:          {np.mean(timings['fftcc']):.3f} 초")
                print(f"  IC-GN 전체:     {np.mean(timings['icgn']):.3f} 초")
                print(f"  IC-GN 내부:     {np.mean(timings['icgn_internal']):.3f} 초  ← 표시되는 값")
                print(f"  IC-GN 오버헤드: {np.mean(timings['icgn']) - np.mean(timings['icgn_internal']):.3f} 초")
                print(f"  결과 저장:      {np.mean(timings['result_save'])*1000:.1f} ms")
                print(f"  GUI 업데이트:   {np.mean(timings['gui_update'])*1000:.1f} ms")
                print(f"")
                print(f"[총 시간]")
                print(f"  이미지 로드:    {np.sum(timings['image_load']):.3f} 초")
                print(f"  FFTCC:          {np.sum(timings['fftcc']):.3f} 초")
                print(f"  IC-GN 전체:     {np.sum(timings['icgn']):.3f} 초")
                print(f"  IC-GN 내부:     {np.sum(timings['icgn_internal']):.3f} 초")
                print(f"  결과 저장:      {np.sum(timings['result_save']):.3f} 초")
                print(f"  GUI 업데이트:   {np.sum(timings['gui_update']):.3f} 초")
                print(f"")
                print(f"  측정된 합계:    {sum(np.sum(v) for v in timings.values()):.3f} 초")
                print(f"  배치 전체:      {batch_end - batch_start:.3f} 초")
                print(f"  미측정 시간:    {(batch_end - batch_start) - sum(np.sum(v) for v in timings.values()):.3f} 초")
                print(f"")
                print(f"[비율 분석]")
                total_measured = sum(np.sum(v) for v in timings.values())
                if total_measured > 0:
                    print(f"  FFTCC:          {np.sum(timings['fftcc'])/total_measured*100:.1f}%")
                    print(f"  IC-GN:          {np.sum(timings['icgn'])/total_measured*100:.1f}%")
                    print(f"  기타:           {(np.sum(timings['image_load']) + np.sum(timings['result_save']) + np.sum(timings['gui_update']))/total_measured*100:.1f}%")
                print(f"{'='*60}\n")
                
                print(f"[DEBUG] 배치 완료: {len(self.state.batch_results)} 결과")
                
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
                    self.view.after(0, self._on_batch_complete)
                
                self.view.after(0, lambda: self.view.set_analysis_state(False))
        
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()


    def _run_batch_streaming(self, params: Dict[str, Any]):
            """스트리밍 배치 분석 (FFTCC + IC-GN)"""
            if not self.state.sequence_files or len(self.state.sequence_files) < 2:
                messagebox.showwarning("경고", "시퀀스 폴더를 먼저 선택해주세요.")
                return
            
            if self.state.is_running:
                return
            
            self.state.is_running = True
            self.state.should_stop = False
            self.state.batch_results.clear()
            self.view.set_analysis_state(True)
            
            print(f"[DEBUG] 배치 분석 시작 (FFTCC + IC-GN)")
            
            def worker():
                error_msg = None
                
                try:
                    files = self.state.sequence_files
                    total_files = len(files)
                    
                    for file_idx, def_path in enumerate(files):
                        if self.state.should_stop:
                            break
                        
                        filename = def_path.name
                        
                        # 진행률 업데이트
                        progress = (file_idx / total_files) * 100
                        self.view.after(0, lambda p=progress, fn=filename, fi=file_idx, tf=total_files:
                                    self.view.update_progress(p, f"분석 중 {fi+1}/{tf}: {fn}"))
                        
                        # 이미지 로드
                        def_image = self.app_state.get_image(filename)
                        if def_image is None:
                            def_image = load_image(def_path)
                        if def_image is None:
                            continue
                        
                        # === 1단계: FFTCC ===
                        fftcc_result = compute_fft_cc(
                            self.state.ref_image,
                            def_image,
                            subset_size=params['subset_size'],
                            spacing=params['spacing'],
                            search_range=params['search_range'],
                            zncc_threshold=params['zncc_threshold'],
                            roi=self.state.roi
                        )
                        print(f"\n[DEBUG] ===== {filename} =====")
                        print(f"[DEBUG] FFTCC U: [{np.min(fftcc_result.disp_u)}, {np.max(fftcc_result.disp_u)}]")
                        print(f"[DEBUG] FFTCC V: [{np.min(fftcc_result.disp_v)}, {np.max(fftcc_result.disp_v)}]")
                        
                        # === 2단계: IC-GN ===
                        interp_order = 5 if params['interpolation'] == 'biquintic' else 3

                        # POI 좌표를 전체 이미지 좌표로 변환
                        if self.state.roi:
                            rx, ry, rw, rh = self.state.roi
                            fftcc_result.points_x = fftcc_result.points_x + rx
                            fftcc_result.points_y = fftcc_result.points_y + ry

                        # 전체 이미지 사용
                        icgn_result = compute_icgn(
                            self.state.ref_image,
                            def_image,
                            initial_guess=fftcc_result,
                            subset_size=params['subset_size'],
                            shape_function=params['shape_function'],
                            interpolation_order=interp_order,
                            convergence_threshold=params['conv_threshold'],
                            max_iterations=params['max_iter'],
                            gaussian_blur=params.get('gaussian_blur')
                        )

                        
                        self.state.batch_results[filename] = icgn_result
                        
                        print(f"[DEBUG] {file_idx+1}/{total_files}: {filename} - 수렴 {icgn_result.n_converged}/{icgn_result.n_points}")
                    
                    # 마지막 결과 저장
                    if self.state.batch_results:
                        last_key = list(self.state.batch_results.keys())[-1]
                        self.state.icgn_result = self.state.batch_results[last_key]
                        self.state.fft_cc_result = self.state.batch_results[last_key]
                    
                except Exception as ex:
                    error_msg = str(ex)
                    import traceback
                    traceback.print_exc()
                finally:
                    self.state.is_running = False
                    
                    if error_msg:
                        self.view.after(0, lambda msg=error_msg: self._show_error(msg))
                    else:
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
        
        valid_text = f"""이상치 비율: {validation.outlier_ratio * 100:.1f}%
불연속 영역: {validation.discontinuity_ratio * 100:.1f}%
전체 유효: {'예' if validation.is_valid else '아니오'}
권장 조치: {validation.suggested_action}"""
        
        self.view.update_validation_text(valid_text)
    
    def _on_batch_complete(self):
        """배치 분석 완료"""
        n_results = len(self.state.batch_results)
        self.view.update_progress(100, f"완료: {n_results} 파일 처리됨")
        self._refresh_display()
        
        # 마지막 결과 표시
        if self.state.batch_results:
            last_key = list(self.state.batch_results.keys())[-1]
            last_result = self.state.batch_results[last_key]
            self._update_result_ui(last_result)
        
        # 총 처리 시간 계산
        total_time = sum(r.processing_time for r in self.state.batch_results.values())
        avg_time = total_time / n_results if n_results > 0 else 0
        
        # ICGNResult인지 확인 후 수렴율 계산
        first_result = list(self.state.batch_results.values())[0] if self.state.batch_results else None
        is_icgn = first_result and hasattr(first_result, 'convergence_rate')
        
        if is_icgn:
            avg_conv_rate = np.mean([r.convergence_rate for r in self.state.batch_results.values()]) * 100
            messagebox.showinfo(
                "배치 분석 완료", 
                f"처리된 파일: {n_results}개\n"
                f"총 소요 시간: {total_time:.1f}초\n"
                f"평균 처리 시간: {avg_time:.2f}초/파일\n"
                f"평균 수렴율: {avg_conv_rate:.1f}%"
            )
        else:
            messagebox.showinfo(
                "배치 분석 완료", 
                f"처리된 파일: {n_results}개\n"
                f"총 소요 시간: {total_time:.1f}초\n"
                f"평균 처리 시간: {avg_time:.2f}초/파일"
            )

    # ===== 표시 (기존과 동일) =====
    
    def _display_image(self, img: np.ndarray):
        if img is None:
            return
        self._refresh_display()
    
    def _refresh_display(self):
        img = self.state.def_image if self.state.def_image is not None else self.state.ref_image
        if img is None:
            return
        
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
        result = self.state.fft_cc_result
        if result is None:
            return base_img
        
        if len(base_img.shape) == 2:
            display_img = cv2.cvtColor(base_img, cv2.COLOR_GRAY2BGR)
        else:
            display_img = base_img.copy()
        
        mode = self.view.display_mode_var.get()
        
        offset_x = 0
        offset_y = 0
        
        if mode == "vectors":
            display_img = self._draw_vectors(display_img, result, offset_x, offset_y)
        elif mode == "u_field":
            display_img = self._draw_scalar_field(display_img, result, 'u', offset_x, offset_y)
        elif mode == "v_field":
            display_img = self._draw_scalar_field(display_img, result, 'v', offset_x, offset_y)
        elif mode == "magnitude":
            display_img = self._draw_magnitude(display_img, result, offset_x, offset_y)
        elif mode == "exx":
            display_img = self._draw_strain_field(display_img, result, 'exx', offset_x, offset_y)
        elif mode == "eyy":
            display_img = self._draw_strain_field(display_img, result, 'eyy', offset_x, offset_y)
        elif mode == "exy":
            display_img = self._draw_strain_field(display_img, result, 'exy', offset_x, offset_y)
        elif mode == "e1":
            display_img = self._draw_strain_field(display_img, result, 'e1', offset_x, offset_y)
        elif mode == "von_mises":
            display_img = self._draw_strain_field(display_img, result, 'von_mises', offset_x, offset_y)
        elif mode == "zncc":
            display_img = self._draw_zncc_map(display_img, result, offset_x, offset_y)
        elif mode == "invalid":
            display_img = self._draw_invalid_points(display_img, result, offset_x, offset_y)
        
        return display_img
    def _draw_scalar_field(self, img: np.ndarray, result, field_type: str,
                        offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
        """변위 필드 시각화 (U 또는 V)"""
        if result.n_points == 0:
            return img
        
        if field_type == 'u':
            values = result.disp_u
            label = "U (px)"
        elif field_type == 'v':
            values = result.disp_v
            label = "V (px)"
        else:  # magnitude
            values = np.sqrt(result.disp_u**2 + result.disp_v**2)
            label = "|D| (px)"
        
        valid = result.valid_mask
        valid_values = values[valid]
        
        if len(valid_values) == 0:
            return img
        
        # 범위 결정
        color_range = self.view.get_color_range()
        if color_range is not None:
            vmin, vmax = color_range
        else:
            vmin, vmax = np.min(valid_values), np.max(valid_values)
            # 대칭 범위
            if field_type in ['u', 'v']:
                v_abs_max = max(abs(vmin), abs(vmax))
                if v_abs_max < 1e-10:
                    v_abs_max = 1.0
                vmin, vmax = -v_abs_max, v_abs_max
        
        # 컬러바 업데이트 (이미지 밖)
        colormap = 'diverging' if field_type in ['u', 'v'] else 'sequential'
        self.view.update_colorbar(vmin, vmax, label, colormap)
        
        # POI 그리기
        for idx in range(result.n_points):
            x = int(result.points_x[idx]) + offset_x
            y = int(result.points_y[idx]) + offset_y
            
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            
            if result.valid_mask[idx]:
                norm_val = (values[idx] - vmin) / (vmax - vmin + 1e-10)
                norm_val = np.clip(norm_val, 0, 1)
                
                if colormap == 'diverging':
                    color = self._diverging_colormap(norm_val)
                else:
                    color = self._sequential_colormap(norm_val)
                
                cv2.circle(img, (x, y), 4, color, -1)
            else:
                cv2.circle(img, (x, y), 4, (128, 128, 128), -1)
        
        return img

    def _draw_strain_field(self, img: np.ndarray, result, strain_type: str,
                        offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
        """변형률 필드 시각화 (Savitzky-Golay + 부드러운 보간)"""
        if result.n_points == 0:
            return img
        
        is_icgn = hasattr(result, 'disp_ux') and result.disp_ux is not None
        if not is_icgn:
            cv2.putText(img, "IC-GN 결과 필요", (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return img
        
        # === POI를 2D 그리드로 변환 ===
        valid = result.valid_mask
        px, py = result.points_x, result.points_y
        
        px_valid = px[valid]
        py_valid = py[valid]
        
        unique_x = np.unique(px_valid)
        unique_y = np.unique(py_valid)
        nx, ny = len(unique_x), len(unique_y)
        
        if nx < 5 or ny < 5:
            return self._draw_strain_field_points(img, result, strain_type, offset_x, offset_y)
        
        grid_step = float(np.median(np.diff(unique_x))) if len(unique_x) > 1 else 1.0
        
        # 좌표 → 인덱스 매핑
        x_to_idx = {x: i for i, x in enumerate(unique_x)}
        y_to_idx = {y: i for i, y in enumerate(unique_y)}
        
        # 변위를 2D 그리드로 변환
        disp_u_grid = np.full((ny, nx), np.nan)
        disp_v_grid = np.full((ny, nx), np.nan)
        
        for i in range(len(px)):
            if valid[i]:
                xi = x_to_idx.get(px[i])
                yi = y_to_idx.get(py[i])
                if xi is not None and yi is not None:
                    disp_u_grid[yi, xi] = result.disp_u[i]
                    disp_v_grid[yi, xi] = result.disp_v[i]
        
        # === Savitzky-Golay 스무딩 ===
        try:
            from speckle.core.postprocess.strain_smoothing import compute_strain_savgol
            strain_smooth = compute_strain_savgol(
                disp_u_grid, disp_v_grid,
                window_size=11,  # 작게 조정
                poly_order=2,
                grid_step=grid_step
            )
        except Exception as e:
            print(f"[WARN] Savitzky-Golay 실패: {e}")
            return self._draw_strain_field_points(img, result, strain_type, offset_x, offset_y)
        
        # 필드 선택
        field_map = {
            'exx': (strain_smooth.exx, "εxx"),
            'eyy': (strain_smooth.eyy, "εyy"),
            'exy': (strain_smooth.exy, "εxy"),
            'e1': (strain_smooth.e1, "ε1"),
            'von_mises': (strain_smooth.von_mises, "ε_vm")
        }
        
        if strain_type not in field_map:
            return img
        
        strain_2d, label = field_map[strain_type]
        
        # NaN이 아닌 값들
        valid_strain = ~np.isnan(strain_2d)
        valid_values = strain_2d[valid_strain]
        if len(valid_values) == 0:
            return img
        
        # 범위 결정
        color_range = self.view.get_color_range()
        if color_range is not None:
            vmin, vmax = color_range
        else:
            # 이상치 제거 (1%, 99% 퍼센타일)
            vmin = np.percentile(valid_values, 1)
            vmax = np.percentile(valid_values, 99)
            v_abs_max = max(abs(vmin), abs(vmax))
            if v_abs_max < 1e-10:
                v_abs_max = 1e-6
            vmin, vmax = -v_abs_max, v_abs_max
        
        # 컬러바 업데이트
        self.view.update_colorbar(vmin, vmax, label, 'diverging')
        
        # === scipy로 부드럽게 보간 ===
        from scipy.interpolate import RegularGridInterpolator
        
        # 유효한 영역만 보간
        strain_filled = strain_2d.copy()
        strain_filled[np.isnan(strain_filled)] = 0  # NaN을 0으로
        
        # 원본 그리드 좌표
        grid_y_coords = unique_y
        grid_x_coords = unique_x
        
        # 보간기 생성
        interp = RegularGridInterpolator(
            (grid_y_coords, grid_x_coords), 
            strain_filled,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        
        # 목표 좌표 (원본 이미지 픽셀)
        x_min, x_max = int(unique_x[0]), int(unique_x[-1])
        y_min, y_max = int(unique_y[0]), int(unique_y[-1])
        
        # 픽셀 단위 그리드 생성
        target_y = np.arange(y_min, y_max + 1)
        target_x = np.arange(x_min, x_max + 1)
        target_yy, target_xx = np.meshgrid(target_y, target_x, indexing='ij')
        target_points = np.stack([target_yy.ravel(), target_xx.ravel()], axis=-1)
        
        # 보간 수행
        strain_interp = interp(target_points).reshape(len(target_y), len(target_x))
        
        # === 컬러맵 적용 ===
        strain_norm = (strain_interp - vmin) / (vmax - vmin + 1e-10)
        strain_norm = np.clip(strain_norm, 0, 1)
        
        # 컬러 이미지 생성
        h, w = strain_norm.shape
        strain_color = np.zeros((h, w, 3), dtype=np.uint8)
        
        for j in range(h):
            for i in range(w):
                if not np.isnan(strain_interp[j, i]):
                    strain_color[j, i] = self._diverging_colormap(strain_norm[j, i])
        
        # === 오버레이 ===
        alpha = 0.6
        x_start = x_min + offset_x
        y_start = y_min + offset_y
        
        # 경계 체크
        x_end = min(x_start + w, img.shape[1])
        y_end = min(y_start + h, img.shape[0])
        x_start = max(0, x_start)
        y_start = max(0, y_start)
        
        crop_x = x_start - (x_min + offset_x)
        crop_y = y_start - (y_min + offset_y)
        crop_w = x_end - x_start
        crop_h = y_end - y_start
        
        if crop_w > 0 and crop_h > 0:
            strain_crop = strain_color[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            interp_crop = strain_interp[crop_y:crop_y+crop_h, crop_x:crop_x+crop_w]
            
            roi = img[y_start:y_end, x_start:x_end].copy()
            
            # NaN이 아닌 영역만 블렌딩
            valid_mask = ~np.isnan(interp_crop)
            for c in range(3):
                roi[:, :, c] = np.where(
                    valid_mask,
                    (alpha * strain_crop[:, :, c] + (1 - alpha) * roi[:, :, c]).astype(np.uint8),
                    roi[:, :, c]
                )
            
            img[y_start:y_end, x_start:x_end] = roi
        
        return img

    def _draw_strain_field_points(self, img: np.ndarray, result, strain_type: str,
                                offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
        """변형률 필드 점 시각화 (폴백용)"""
        from speckle.core.postprocess import compute_strain_from_icgn
        strain = compute_strain_from_icgn(result)
        
        field_map = {
            'exx': (strain.exx, "εxx"),
            'eyy': (strain.eyy, "εyy"),
            'exy': (strain.exy, "εxy"),
            'e1': (strain.e1, "ε1"),
            'von_mises': (strain.von_mises, "ε_vm")
        }
        
        if strain_type not in field_map:
            return img
        
        values, label = field_map[strain_type]
        valid = strain.valid_mask
        valid_values = values[valid]
        
        if len(valid_values) == 0:
            return img
        
        color_range = self.view.get_color_range()
        if color_range is not None:
            vmin, vmax = color_range
        else:
            vmin, vmax = np.min(valid_values), np.max(valid_values)
            v_abs_max = max(abs(vmin), abs(vmax))
            if v_abs_max < 1e-10:
                v_abs_max = 1e-6
            vmin, vmax = -v_abs_max, v_abs_max
        
        self.view.update_colorbar(vmin, vmax, label, 'diverging')
        
        for idx in range(result.n_points):
            x = int(result.points_x[idx]) + offset_x
            y = int(result.points_y[idx]) + offset_y
            
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            
            if strain.valid_mask[idx]:
                norm_val = (values[idx] - vmin) / (vmax - vmin + 1e-10)
                norm_val = np.clip(norm_val, 0, 1)
                color = self._diverging_colormap(norm_val)
                cv2.circle(img, (x, y), 4, color, -1)
            else:
                cv2.circle(img, (x, y), 4, (128, 128, 128), -1)
        
        return img

    def _diverging_colormap(self, norm_value: float) -> tuple:
        """
        Diverging colormap (파랑 - 흰색 - 빨강)
        0.0 = 파랑 (음수)
        0.5 = 흰색 (0)
        1.0 = 빨강 (양수)
        """
        if norm_value < 0.5:
            # 파랑 → 흰색
            t = norm_value * 2
            r = int(t * 255)
            g = int(t * 255)
            b = 255
        else:
            # 흰색 → 빨강
            t = (norm_value - 0.5) * 2
            r = 255
            g = int((1 - t) * 255)
            b = int((1 - t) * 255)
        
        return (b, g, r)  # BGR
    
    def _draw_colorbar_diverging(self, img: np.ndarray, min_val: float, max_val: float, label: str) -> np.ndarray:
        """Diverging colorbar"""
        h, w = img.shape[:2]
        bar_width, bar_height, margin = 20, 150, 10
        x_start, y_start = w - bar_width - margin - 50, margin
        
        # 배경
        cv2.rectangle(img, (x_start - 5, y_start - 5), 
                    (x_start + bar_width + 60, y_start + bar_height + 35), (40, 40, 40), -1)
        
        # 컬러바
        for i in range(bar_height):
            norm_val = 1 - (i / bar_height)
            color = self._diverging_colormap(norm_val)
            cv2.line(img, (x_start, y_start + i), (x_start + bar_width, y_start + i), color, 1)
        
        # 라벨
        cv2.putText(img, f"{max_val:.2e}", (x_start + bar_width + 3, y_start + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(img, "0", (x_start + bar_width + 3, y_start + bar_height // 2 + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(img, f"{min_val:.2e}", (x_start + bar_width + 3, y_start + bar_height),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
        cv2.putText(img, label, (x_start, y_start + bar_height + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return img
    
    def _draw_vectors(self, img: np.ndarray, result: FFTCCResult,
                    offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
        """변위 벡터 시각화 (자동 스케일)"""
        if result.n_points == 0:
            return img
        
        # 자동 스케일: 최대 화살표 길이가 30px이 되도록
        max_disp = max(np.max(np.abs(result.disp_u)), np.max(np.abs(result.disp_v)), 1)
        auto_scale = 30.0 / max_disp
        
        for idx in range(result.n_points):
            if not result.valid_mask[idx]:
                continue
            
            x = int(result.points_x[idx]) + offset_x
            y = int(result.points_y[idx]) + offset_y
            u = result.disp_u[idx]
            v = result.disp_v[idx]
            
            if x < 0 or x >= img.shape[1] or y < 0 or y >= img.shape[0]:
                continue
            
            end_x = int(x + u * auto_scale)
            end_y = int(y + v * auto_scale)
            
            # 색상: 변위 크기에 따라 (초록 → 빨강)
            magnitude = np.sqrt(u*u + v*v)
            ratio = min(magnitude / max_disp, 1.0)
            hue = int((1 - ratio) * 120)  # 120(초록) → 0(빨강)
            color = self._hsv_to_bgr(hue, 255, 255)
            
            cv2.arrowedLine(img, (x, y), (end_x, end_y), color, 1, tipLength=0.3)
        
        return img
    
    def _draw_magnitude(self, img: np.ndarray, result,
                        offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
        """변위 크기 시각화"""
        return self._draw_scalar_field(img, result, 'magnitude', offset_x, offset_y)
    
    def _sequential_colormap(self, norm_value: float) -> tuple:
        """
        Sequential colormap (파랑 → 초록 → 노랑 → 빨강)
        """
        if norm_value < 0.25:
            r = 0
            g = int(norm_value * 4 * 255)
            b = 255
        elif norm_value < 0.5:
            r = 0
            g = 255
            b = int((1 - (norm_value - 0.25) * 4) * 255)
        elif norm_value < 0.75:
            r = int((norm_value - 0.5) * 4 * 255)
            g = 255
            b = 0
        else:
            r = 255
            g = int((1 - (norm_value - 0.75) * 4) * 255)
            b = 0
        
        return (b, g, r)  # BGR

    def _draw_zncc_map(self, img: np.ndarray, result: FFTCCResult,
                       offset_x: int = 0, offset_y: int = 0) -> np.ndarray:
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
                cv2.drawMarker(img, (x, y), (255, 255, 255), cv2.MARKER_CROSS, 10, 2)
        
        return img
    
    def _hsv_to_bgr(self, h: int, s: int, v: int) -> tuple:
        hsv = np.uint8([[[h, s, v]]])
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return int(bgr[0, 0, 0]), int(bgr[0, 0, 1]), int(bgr[0, 0, 2])
    
    def _magnitude_to_color(self, norm_value: float) -> tuple:
        if norm_value < 0.25:
            r, g, b = 0, int(255 * (norm_value / 0.25)), 255
        elif norm_value < 0.5:
            r, g, b = 0, 255, int(255 * (1 - (norm_value - 0.25) / 0.25))
        elif norm_value < 0.75:
            r, g, b = int(255 * ((norm_value - 0.5) / 0.25)), 255, 0
        else:
            r, g, b = 255, int(255 * (1 - (norm_value - 0.75) / 0.25)), 0
        return (b, g, r)
    
    def _draw_colorbar(self, img: np.ndarray, min_val: float, max_val: float, label: str) -> np.ndarray:
        h, w = img.shape[:2]
        bar_width, bar_height, margin = 20, 150, 10
        x_start, y_start = w - bar_width - margin, margin
        
        cv2.rectangle(img, (x_start - 5, y_start - 5), 
                     (x_start + bar_width + 40, y_start + bar_height + 25), (40, 40, 40), -1)
        
        for i in range(bar_height):
            norm_val = 1 - (i / bar_height)
            color = self._magnitude_to_color(norm_val)
            cv2.line(img, (x_start, y_start + i), (x_start + bar_width, y_start + i), color, 1)
        
        cv2.putText(img, f"{max_val:.1f}", (x_start + bar_width + 5, y_start + 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        cv2.putText(img, f"{min_val:.1f}", (x_start + bar_width + 5, y_start + bar_height),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
        return img
    
    def _display_result(self, result: FFTCCResult):
        self.state.fft_cc_result = result
        self._refresh_display()
    
    def update_display(self, mode: str):
        """표시 모드 변경"""
        self._refresh_display()

    # ===== 줌/캔버스 =====
    
    def fit_to_canvas(self):
        img = self.state.def_image if self.state.def_image is not None else self.state.ref_image
        if img is not None:
            zoom = self.view.canvas_view.fit_to_canvas(img.shape[:2])
            self.state.zoom_level = zoom
            self.state.pan_offset = (0, 0)
            self._refresh_display()
            self.view.zoom_label.configure(text=f"{int(zoom * 100)}%")
    
    def zoom_in(self):
        self._handle_zoom(1.2)
    
    def zoom_out(self):
        self._handle_zoom(0.8)
    
    def set_zoom_1to1(self):
        self.state.zoom_level = 1.0
        self.state.pan_offset = (0, 0)
        self._refresh_display()
        self.view.zoom_label.configure(text="100%")
    
    # ===== 내보내기 =====
    
    def export_csv(self):
        """IC-GN 결과 CSV 저장"""
        result = self.state.icgn_result if self.state.icgn_result else self.state.fft_cc_result
        
        if result is None:
            messagebox.showwarning("경고", "내보낼 결과가 없습니다.")
            return
        
        # 기본 파일명 설정
        if self.state.def_path:
            default_name = f"dic_result_{self.state.def_path.stem}.csv"
        else:
            default_name = "dic_result.csv"
        
        path = filedialog.asksaveasfilename(
            title="CSV 저장",
            defaultextension=".csv",
            initialfile=default_name,
            filetypes=[("CSV 파일", "*.csv")]
        )
        
        if not path:
            return
        
        try:
            is_icgn = hasattr(result, 'converged')
            
            with open(path, 'w', newline='', encoding='utf-8') as f:
                # 메타 정보 헤더
                f.write(f"# DIC Analysis Result\n")
                f.write(f"# Reference: {self.state.ref_path.name if self.state.ref_path else 'N/A'}\n")
                f.write(f"# Deformed: {self.state.def_path.name if self.state.def_path else 'N/A'}\n")
                f.write(f"# ROI: {self.state.roi}\n")
                f.write(f"# Subset Size: {result.subset_size}\n")
                
                if is_icgn:
                    f.write(f"# Shape Function: {result.shape_function}\n")
                    f.write(f"# Convergence Threshold: {result.convergence_threshold}\n")
                    f.write(f"# Max Iterations: {result.max_iterations}\n")
                    f.write(f"# Total POI: {result.n_points}\n")
                    f.write(f"# Converged: {result.n_converged} ({result.convergence_rate*100:.1f}%)\n")
                    f.write(f"# Valid: {result.n_valid}\n")
                    f.write(f"# Mean Iterations: {result.mean_iterations:.1f}\n")
                else:
                    f.write(f"# Total POI: {result.n_points}\n")
                    f.write(f"# Valid: {result.n_valid}\n")
                
                f.write(f"# Mean ZNCC: {result.mean_zncc:.6f}\n")
                f.write(f"# Processing Time: {result.processing_time:.2f}s\n")
                f.write(f"#\n")
                
                # CSV 데이터
                import csv
                writer = csv.writer(f)
                
                if is_icgn:
                    # IC-GN 결과 (변형률 정보 포함)
                    if result.shape_function == 'affine':
                        writer.writerow(['x', 'y', 'u', 'v', 'ux', 'uy', 'vx', 'vy', 'zncc', 'converged', 'iterations'])
                        for i in range(result.n_points):
                            writer.writerow([
                                result.points_x[i],
                                result.points_y[i],
                                f"{result.disp_u[i]:.6f}",
                                f"{result.disp_v[i]:.6f}",
                                f"{result.disp_ux[i]:.6f}",
                                f"{result.disp_uy[i]:.6f}",
                                f"{result.disp_vx[i]:.6f}",
                                f"{result.disp_vy[i]:.6f}",
                                f"{result.zncc_values[i]:.6f}",
                                result.converged[i],
                                result.iterations[i]
                            ])
                    else:  # quadratic
                        writer.writerow(['x', 'y', 'u', 'v', 'ux', 'uy', 'uxx', 'uxy', 'uyy', 
                                        'vx', 'vy', 'vxx', 'vxy', 'vyy', 'zncc', 'converged', 'iterations'])
                        for i in range(result.n_points):
                            writer.writerow([
                                result.points_x[i],
                                result.points_y[i],
                                f"{result.disp_u[i]:.6f}",
                                f"{result.disp_v[i]:.6f}",
                                f"{result.disp_ux[i]:.6f}",
                                f"{result.disp_uy[i]:.6f}",
                                f"{result.disp_uxx[i]:.6f}",
                                f"{result.disp_uxy[i]:.6f}",
                                f"{result.disp_uyy[i]:.6f}",
                                f"{result.disp_vx[i]:.6f}",
                                f"{result.disp_vy[i]:.6f}",
                                f"{result.disp_vxx[i]:.6f}",
                                f"{result.disp_vxy[i]:.6f}",
                                f"{result.disp_vyy[i]:.6f}",
                                f"{result.zncc_values[i]:.6f}",
                                result.converged[i],
                                result.iterations[i]
                            ])
                else:
                    # FFTCC 결과
                    writer.writerow(['x', 'y', 'u', 'v', 'zncc', 'valid'])
                    for i in range(result.n_points):
                        writer.writerow([
                            result.points_x[i],
                            result.points_y[i],
                            f"{result.disp_u[i]:.6f}",
                            f"{result.disp_v[i]:.6f}",
                            f"{result.zncc_values[i]:.6f}",
                            result.valid_mask[i]
                        ])
            
            messagebox.showinfo("완료", f"CSV 저장 완료\n{path}")
            
        except Exception as e:
            messagebox.showerror("오류", f"저장 실패: {e}")

    def export_image(self):
        messagebox.showinfo("정보", "이미지 내보내기 기능은 추후 구현 예정입니다.")
