"""
DIC 분석 실행 모듈

FFT-CC + IC-GN 단일/배치 분석 실행 및 스레드 관리
"""

import logging
import threading
import time
import numpy as np
from typing import Dict, Any
from tkinter import messagebox

from speckle.core.initial_guess import compute_fft_cc
from speckle.core.optimization import compute_icgn
from speckle.io import load_image

logger = logging.getLogger(__name__)


class AnalysisRunner:
    """분석 실행 담당"""

    def __init__(self, ctrl):
        """
        Args:
            ctrl: DICController 인스턴스
        """
        self.ctrl = ctrl

    @property
    def state(self):
        return self.ctrl.state

    @property
    def view(self):
        return self.ctrl.view

    @property
    def app_state(self):
        return self.ctrl.app_state

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

        def worker():
            error_msg = None
            fftcc_result = None
            icgn_result = None

            try:
                def fftcc_progress(current, total):
                    if self.state.should_stop:
                        raise InterruptedError("사용자 중단")
                    progress = (current / total) * 50
                    self.view.after(0, lambda p=progress, c=current, t=total:
                                    self.view.update_progress(p, f"FFTCC: {c}/{t}"))

                fftcc_result = compute_fft_cc(
                    self.state.ref_image,
                    self.state.def_image,
                    subset_size=params['subset_size'],
                    spacing=params['spacing'],
                    zncc_threshold=params['zncc_threshold'],
                    roi=self.state.roi,
                )

                if self.state.should_stop:
                    raise InterruptedError("사용자 중단")

                self.view.after(0, lambda: self.view.update_progress(50, "IC-GN 시작..."))

                def icgn_progress(current, total):
                    if self.state.should_stop:
                        raise InterruptedError("사용자 중단")
                    progress = 50 + (current / total) * 50
                    self.view.after(0, lambda p=progress, c=current, t=total:
                                    self.view.update_progress(p, f"IC-GN: {c}/{t}"))

                interp_order = 5 if params['interpolation'] == 'biquintic' else 3

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
                    progress_callback=icgn_progress,
                    use_numba=True,
                )

                self.state.fft_cc_result = fftcc_result
                self.state.icgn_result = icgn_result

                if self.state.def_path:
                    self.state.batch_results[self.state.def_path.name] = icgn_result

            except InterruptedError:
                self.view.after(0, lambda: self.view.update_progress(0, "중단됨"))
            except Exception as ex:
                error_msg = str(ex)
                logger.error(f"분석 오류: {error_msg}", exc_info=True)
            finally:
                self.state.is_running = False
                if error_msg:
                    self.view.after(0, lambda msg=error_msg:
                                    self.ctrl.result_handler.show_error(msg))
                elif icgn_result:
                    self.view.after(0, lambda r=icgn_result:
                                    self.ctrl.result_handler.on_icgn_complete(r))
                self.view.after(0, lambda: self.view.set_analysis_state(False))

        threading.Thread(target=worker, daemon=True).start()

    def run_batch_fft_cc(self, params: Dict[str, Any]):
        """전체 시퀀스 배치 분석"""
        use_cache = self.app_state.has_images() and len(self.app_state.images) > 1
        if use_cache:
            self._run_batch_cached(params)
        else:
            self._run_batch_streaming(params)

    def _run_batch_cached(self, params: Dict[str, Any]):
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

        def worker():
            error_msg = None
            batch_start = time.time()

            try:
                all_files = self.state.sequence_files if self.state.sequence_files else self.app_state.file_paths
                cached_files = [f for f in all_files if f.name in self.app_state.images]

                if not cached_files:
                    raise ValueError("분석할 캐시된 이미지가 없습니다.")

                total_files = len(cached_files)

                for file_idx, def_path in enumerate(cached_files):
                    if self.state.should_stop:
                        break

                    filename = def_path.name
                    progress = (file_idx / total_files) * 100
                    self.view.after(0, lambda p=progress, fn=filename, fi=file_idx, tf=total_files:
                                self.view.update_progress(p, f"분석 중 {fi+1}/{tf}: {fn}"))
                    self.view.after(0, lambda idx=file_idx+1: self.view.set_current_index(idx))

                    def_image = self.app_state.get_image(filename)
                    if def_image is None:
                        continue

                    fftcc_result = compute_fft_cc(
                        self.state.ref_image, def_image,
                        subset_size=params['subset_size'],
                        spacing=params['spacing'],
                        zncc_threshold=params['zncc_threshold'],
                        roi=self.state.roi,
                    )

                    interp_order = 5 if params['interpolation'] == 'biquintic' else 3
                    icgn_result = compute_icgn(
                        self.state.ref_image, def_image,
                        initial_guess=fftcc_result,
                        subset_size=params['subset_size'],
                        shape_function=params['shape_function'],
                        interpolation_order=interp_order,
                        convergence_threshold=params['conv_threshold'],
                        max_iterations=params['max_iter'],
                        gaussian_blur=params.get('gaussian_blur'),
                        use_numba=True,
                    )

                    self.state.batch_results[filename] = icgn_result

                if self.state.batch_results:
                    last_key = list(self.state.batch_results.keys())[-1]
                    self.state.icgn_result = self.state.batch_results[last_key]
                    self.state.fft_cc_result = self.state.batch_results[last_key]

                batch_elapsed = time.time() - batch_start
                logger.info(f"배치 완료: {len(self.state.batch_results)}개 파일, {batch_elapsed:.1f}초")

            except InterruptedError:
                self.view.after(0, lambda: self.view.update_progress(0, "중단됨"))
            except Exception as ex:
                error_msg = str(ex)
                logger.error(f"배치 오류: {error_msg}", exc_info=True)
            finally:
                self.state.is_running = False
                if error_msg:
                    self.view.after(0, lambda msg=error_msg:
                                    self.ctrl.result_handler.show_error(msg))
                else:
                    self.view.after(0, self.ctrl.result_handler.on_batch_complete)
                self.view.after(0, lambda: self.view.set_analysis_state(False))

        threading.Thread(target=worker, daemon=True).start()

    def _run_batch_streaming(self, params: Dict[str, Any]):
        if not self.state.sequence_files or len(self.state.sequence_files) < 2:
            messagebox.showwarning("경고", "시퀀스 폴더를 먼저 선택해주세요.")
            return

        if self.state.is_running:
            return

        self.state.is_running = True
        self.state.should_stop = False
        self.state.batch_results.clear()
        self.view.set_analysis_state(True)

        def worker():
            error_msg = None

            try:
                files = self.state.sequence_files
                total_files = len(files)

                for file_idx, def_path in enumerate(files):
                    if self.state.should_stop:
                        break

                    filename = def_path.name
                    progress = (file_idx / total_files) * 100
                    self.view.after(0, lambda p=progress, fn=filename, fi=file_idx, tf=total_files:
                                self.view.update_progress(p, f"분석 중 {fi+1}/{tf}: {fn}"))

                    def_image = self.app_state.get_image(filename)
                    if def_image is None:
                        def_image = load_image(def_path)
                    if def_image is None:
                        continue

                    fftcc_result = compute_fft_cc(
                        self.state.ref_image, def_image,
                        subset_size=params['subset_size'],
                        spacing=params['spacing'],
                        zncc_threshold=params['zncc_threshold'],
                        roi=self.state.roi
                    )

                    interp_order = 5 if params['interpolation'] == 'biquintic' else 3
                    icgn_result = compute_icgn(
                        self.state.ref_image, def_image,
                        initial_guess=fftcc_result,
                        subset_size=params['subset_size'],
                        shape_function=params['shape_function'],
                        interpolation_order=interp_order,
                        convergence_threshold=params['conv_threshold'],
                        max_iterations=params['max_iter'],
                        gaussian_blur=params.get('gaussian_blur'),
                        use_numba=True,
                    )

                    self.state.batch_results[filename] = icgn_result

                if self.state.batch_results:
                    last_key = list(self.state.batch_results.keys())[-1]
                    self.state.icgn_result = self.state.batch_results[last_key]
                    self.state.fft_cc_result = self.state.batch_results[last_key]

            except Exception as ex:
                error_msg = str(ex)
                logger.error(f"배치 오류: {error_msg}", exc_info=True)
            finally:
                self.state.is_running = False
                if error_msg:
                    self.view.after(0, lambda msg=error_msg:
                                    self.ctrl.result_handler.show_error(msg))
                else:
                    self.view.after(0, self.ctrl.result_handler.on_batch_complete)
                self.view.after(0, lambda: self.view.set_analysis_state(False))

        threading.Thread(target=worker, daemon=True).start()

    def stop_analysis(self):
        self.state.should_stop = True
