"""스페클 품질 분석 GUI - 메인 앱"""

import logging
import threading
import tkinter as tk
from tkinter import ttk, messagebox

from .models.app_state import AppState
from .models.settings import SettingsManager
from .controllers.main_controller import MainController
from .controllers.dic_controller import DICController
from .views.canvas_view import CanvasView

_logger = logging.getLogger(__name__)
from .views.param_panel import ParamPanel
from .views.dic_tab import DICTab


class SpeckleQualityGUI:
    """스페클 품질 분석 메인 윈도우"""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Speckle Analysis Suite v3.3")
        self.root.geometry("1500x950")
        self._last_navigate_time = 0

        # 설정 관리자 초기화
        self.settings = SettingsManager()

        # 상태 초기화
        self.state = AppState()

        # UI 먼저 생성
        self._create_notebook()
        self._create_menu()

        # 컨트롤러 초기화 (UI 생성 후!)
        self.controller = MainController(self.state)
        self.controller.set_views(self.canvas_view, self.param_panel)

        # DIC 컨트롤러 (DICTab 생성 후!)
        self.dic_controller = DICController(
            self.dic_tab, self.state, self.controller)

        self._connect_controller()
        self._start_periodic_check()

        # 저장된 설정 적용
        self._apply_saved_settings()

        # 초기 탭에 맞게 키 바인딩 설정
        self._on_tab_changed(None)

        # 종료 시 설정 저장
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        # UI 표시 후 200ms 뒤 백그라운드 워밍업 시작
        # (사용자가 버튼을 누르기 전에 JIT 컴파일 완료 목적)
        self.root.after(200, self._warmup_background)

    def _warmup_background(self):
        """앱 시작 후 백그라운드에서 Numba 전체 워밍업"""

        def _run():
            try:
                # SSSIG Numba 워밍업
                from speckle.core.sssig import warmup_numba
                warmup_numba()
                _logger.debug("백그라운드 워밍업: SSSIG 완료")

                # IC-GN Numba 워밍업 (가장 오래 걸리는 부분)
                from speckle.core.optimization.icgn_core_numba import (
                    warmup_icgn_core)
                warmup_icgn_core()
                _logger.debug("백그라운드 워밍업: IC-GN 완료")

                # FFT-CC 워밍업
                from speckle.core.initial_guess.fft_cc import warmup_fft_cc
                warmup_fft_cc()
                _logger.debug("백그라운드 워밍업: FFT-CC 완료")

                _logger.info("백그라운드 Numba 워밍업 전체 완료")

            except Exception as e:
                _logger.warning(f"백그라운드 워밍업 일부 실패 (무시): {e}")

        thread = threading.Thread(
            target=_run, daemon=True, name="numba-warmup")
        thread.start()

    def _apply_saved_settings(self):
        """저장된 설정 적용"""
        try:
            dic_params = self.settings.get_dic_params()
            self.dic_tab.set_parameters(dic_params)

            from .views.param_panel import Parameters
            quality_params = Parameters(
                mig_threshold=self.settings.get('mig_threshold', 30.0),
                subset_size=self.settings.get('quality_subset_size', 21),
                spacing=self.settings.get('quality_spacing', 16)
            )
            self.param_panel.set_parameters(quality_params)
        except Exception as e:
            _logger.debug(f"설정 복원 실패 (초기 실행 시 정상): {e}")

    def _on_closing(self):
        """앱 종료 시 설정 저장"""
        try:
            dic_params = self.dic_tab.get_parameters()
            self.settings.update(dic_params)

            quality_params = self.param_panel.get_parameters()
            if quality_params:
                self.settings.set('mig_threshold', quality_params.mig_threshold)
                self.settings.set(
                    'quality_subset_size', quality_params.subset_size)
                self.settings.set('quality_spacing', quality_params.spacing)

            if self.state.folder_path:
                self.settings.set('last_folder', str(self.state.folder_path))

            self.settings.save()
        except Exception as e:
            _logger.warning(f"종료 시 설정 저장 실패: {e}")

        self.root.destroy()

    def _create_menu(self):
        """메뉴바 생성"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # 파일 메뉴
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="파일", menu=file_menu)
        file_menu.add_command(label="이미지 열기",
                              command=lambda: self.controller.open_image(),
                              accelerator="Ctrl+O")
        file_menu.add_command(label="폴더 열기",
                              command=lambda: self.controller.open_folder(),
                              accelerator="Ctrl+Shift+O")
        file_menu.add_separator()
        file_menu.add_command(label="종료", command=self.root.quit)

        # 내보내기 메뉴
        export_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="내보내기", menu=export_menu)
        export_menu.add_command(
            label="CSV로 내보내기",
            command=lambda: self._export('csv'))
        export_menu.add_command(
            label="JSON으로 내보내기",
            command=lambda: self._export('json'))
        export_menu.add_command(
            label="요약 보고서 (TXT)",
            command=lambda: self._export('txt'))

        # 단축키 바인딩
        self.root.bind('<Control-o>', lambda e: self.controller.open_image())
        self.root.bind('<Control-O>', lambda e: self.controller.open_folder())

    def _export(self, export_type: str, include_images: bool = False):
        """내보내기 실행"""
        results = self.controller.export_results(export_type, include_images)
        if results:
            paths = '\n'.join(str(p) for p in results.values())
            messagebox.showinfo("내보내기 완료", f"저장 완료:\n{paths}")

    def _create_notebook(self):
        """탭 노트북 생성"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # 탭 1: 품질 평가
        self.quality_frame = ttk.Frame(self.notebook)
        self.notebook.add(self.quality_frame, text="  품질 평가  ")
        self._create_quality_tab()

        # 탭 2: 변위 분석 (DICTab은 자체적으로 Frame)
        self.dic_tab = DICTab(self.notebook)
        self.notebook.add(self.dic_tab, text="  변위 분석  ")

        # 탭 전환 이벤트 바인딩
        self.notebook.bind('<<NotebookTabChanged>>', self._on_tab_changed)

    def _on_tab_changed(self, event):
        """탭 전환 시 키 바인딩 변경"""
        current_tab = self.notebook.index(self.notebook.select())

        # 기존 바인딩 해제
        self.root.unbind('<Left>')
        self.root.unbind('<Right>')
        self.root.unbind('<Up>')
        self.root.unbind('<Down>')

        if current_tab == 0:  # 품질 평가 탭
            self.root.bind('<Left>', lambda e: self.controller.navigate(-1))
            self.root.bind('<Right>', lambda e: self.controller.navigate(1))
        elif current_tab == 1:  # 변위 분석 탭
            self.root.bind('<Left>', lambda e: self._dic_navigate(-1))
            self.root.bind('<Right>', lambda e: self._dic_navigate(1))
            self.root.bind('<Up>', lambda e: self._dic_navigate(-10))
            self.root.bind('<Down>', lambda e: self._dic_navigate(10))

    def _dic_navigate(self, delta: int):
        """변위 분석 탭 이미지 네비게이션 (디바운싱 적용)"""
        import time
        current_time = time.time()

        # 100ms 이내 반복 입력 무시
        if current_time - self._last_navigate_time < 0.1:
            return
        self._last_navigate_time = current_time

        if self.dic_tab.sequence_files:
            new_index = self.dic_tab.current_index + delta
            new_index = max(
                0, min(len(self.dic_tab.sequence_files) - 1, new_index))
            self.dic_controller.select_image_index(new_index)

    def _create_quality_tab(self):
        """품질 평가 탭 (기존 UI)"""
        main_frame = self.quality_frame

        # 좌측 패널
        left_panel = ttk.Frame(main_frame, width=300)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 5))
        left_panel.pack_propagate(False)

        self._create_file_buttons(left_panel)
        self.param_panel = ParamPanel(left_panel)
        self._create_roi_panel(left_panel)
        self._create_eval_buttons(left_panel)
        self._create_viz_options(left_panel)
        self._create_status_panel(left_panel)
        self._create_result_panel(left_panel)

        # 중앙 캔버스
        center_frame = ttk.Frame(main_frame)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.canvas_view = CanvasView(center_frame)

        # 우측 파일 목록
        self._create_file_list(main_frame)

    def _create_file_buttons(self, parent):
        """파일 버튼"""
        frame = ttk.LabelFrame(parent, text="파일")
        frame.pack(fill=tk.X, pady=5)

        btn_row = ttk.Frame(frame)
        btn_row.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_row, text="이미지 열기",
                   command=lambda: self.controller.open_image()
                   ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        ttk.Button(btn_row, text="폴더 열기",
                   command=lambda: self.controller.open_folder()
                   ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))

    def _create_roi_panel(self, parent):
        """ROI 패널"""
        frame = ttk.LabelFrame(parent, text="ROI (우클릭 드래그)")
        frame.pack(fill=tk.X, pady=5)

        self.roi_label = ttk.Label(frame, text="ROI: 전체 이미지")
        self.roi_label.pack(padx=5, pady=2)

        ttk.Button(frame, text="ROI 초기화",
                   command=lambda: self.controller.reset_roi()
                   ).pack(fill=tk.X, padx=5, pady=2)

    def _create_eval_buttons(self, parent):
        """평가 버튼"""
        frame = ttk.LabelFrame(parent, text="평가")
        frame.pack(fill=tk.X, pady=5)

        ttk.Button(frame, text="현재 이미지 평가",
                   command=lambda: self.controller.evaluate_current()
                   ).pack(fill=tk.X, padx=5, pady=2)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, pady=2, padx=5)

        self.btn_evaluate_all = ttk.Button(
            btn_frame, text="전체 이미지 평가",
            command=lambda: self.controller.evaluate_all()
        )
        self.btn_evaluate_all.pack(side=tk.LEFT, fill=tk.X, expand=True)

        self.btn_stop = ttk.Button(
            btn_frame, text="정지",
            command=lambda: self.controller.stop_batch(),
            state='disabled'
        )
        self.btn_stop.pack(side=tk.LEFT, padx=(5, 0))

        # 진행률
        self.progress_var = tk.StringVar(value="")
        ttk.Label(frame, textvariable=self.progress_var).pack(pady=2)
        self.progress_bar = ttk.Progressbar(frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, padx=5, pady=2)

    def _create_export_buttons(self, parent):
        """내보내기 버튼"""
        frame = ttk.LabelFrame(parent, text="결과 저장")
        frame.pack(fill=tk.X, pady=5)

        btn_frame = ttk.Frame(frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=2)

        ttk.Button(btn_frame, text="CSV", width=6,
                   command=lambda: self._export('csv')
                   ).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="JSON", width=6,
                   command=lambda: self._export('json')
                   ).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="TXT", width=6,
                   command=lambda: self._export('txt')
                   ).pack(side=tk.LEFT, padx=1)
        ttk.Button(btn_frame, text="이미지", width=6,
                   command=lambda: self.controller.export_current_image()
                   ).pack(side=tk.LEFT, padx=1)

        ttk.Button(frame, text="전체 내보내기",
                   command=lambda: self._export('all', include_images=True)
                   ).pack(fill=tk.X, padx=5, pady=2)

    def _create_viz_options(self, parent):
        """시각화 옵션"""
        frame = ttk.LabelFrame(parent, text="시각화")
        frame.pack(fill=tk.X, pady=5)

        self.show_all_poi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame, text="모든 POI 표시",
            variable=self.show_all_poi_var,
            command=self._update_display
        ).pack(anchor=tk.W, padx=5)

        self.show_bad_poi_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            frame, text="불량 POI 강조",
            variable=self.show_bad_poi_var,
            command=self._update_display
        ).pack(anchor=tk.W, padx=5)

        # 범례
        ttk.Label(frame, text="● 양호  ● 증가필요  ● 실패",
                  font=('Arial', 8), foreground='gray'
                  ).pack(anchor=tk.W, padx=5, pady=2)

    def _create_status_panel(self, parent):
        """상태 패널"""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.X, pady=5)

        self.zoom_label = ttk.Label(frame, text="줌: 100%")
        self.zoom_label.pack(side=tk.LEFT, padx=5)

        ttk.Button(frame, text="맞춤", width=6,
                   command=lambda: self.controller.fit_to_canvas()
                   ).pack(side=tk.RIGHT, padx=5)

    def _create_result_panel(self, parent):
        """결과 패널"""
        frame = ttk.LabelFrame(parent, text="평가 결과")
        frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.result_text = tk.Text(
            frame, width=35, height=12, font=('Consolas', 9))
        scroll = ttk.Scrollbar(
            frame, orient=tk.VERTICAL, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scroll.set)
        self.result_text.pack(
            side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(5, 0), pady=5)
        scroll.pack(side=tk.RIGHT, fill=tk.Y, pady=5)

    def _create_file_list(self, parent):
        """파일 목록"""
        right_panel = ttk.Frame(parent, width=200)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))
        right_panel.pack_propagate(False)

        # 파일 리스트
        frame = ttk.LabelFrame(right_panel, text="파일 목록")
        frame.pack(fill=tk.BOTH, expand=True, pady=5)

        container = ttk.Frame(frame)
        container.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.file_listbox = tk.Listbox(
            container, height=20, font=('Consolas', 9))
        scroll = ttk.Scrollbar(
            container, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=scroll.set)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.bind('<<ListboxSelect>>', self._on_file_select)

        # 네비게이션
        nav_frame = ttk.LabelFrame(right_panel, text="이미지 탐색")
        nav_frame.pack(fill=tk.X, pady=5)

        self.nav_label = ttk.Label(
            nav_frame, text="0 / 0", font=('Arial', 10, 'bold'))
        self.nav_label.pack(pady=5)

        btn_frame = ttk.Frame(nav_frame)
        btn_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(btn_frame, text="◀ 이전",
                   command=lambda: self.controller.navigate(-1)
                   ).pack(side=tk.LEFT)
        ttk.Button(btn_frame, text="다음 ▶",
                   command=lambda: self.controller.navigate(1)
                   ).pack(side=tk.RIGHT)

        ttk.Label(nav_frame, text="← → 키로 이동",
                  font=('Arial', 8), foreground='gray').pack(pady=2)

    def _connect_controller(self):
        """컨트롤러 콜백 연결"""
        self.controller.on_state_changed = self._on_state_changed
        self.controller.on_progress = self._on_progress
        self.controller.on_batch_complete = self._on_batch_complete

        # 로딩 콜백 추가
        self.controller.on_loading_progress = self._on_loading_progress
        self.controller.on_loading_complete = self._on_loading_complete

    def _on_loading_progress(self, current: int, total: int, filename: str):
        """이미지 로딩 진행률"""
        self.root.after(
            0, lambda: self._update_loading_ui(current, total, filename))

    def _update_loading_ui(self, current: int, total: int, filename: str):
        """로딩 UI 업데이트"""
        self.progress_var.set(f"로딩: {current}/{total}")
        self.progress_bar['value'] = (current / total) * 100
        self._update_file_list()

    def _on_loading_complete(self, total_loaded: int):
        """로딩 완료"""
        self.root.after(0, lambda: self._show_loading_complete(total_loaded))

    def _show_loading_complete(self, total_loaded: int):
        """로딩 완료 메시지"""
        self.progress_var.set(f"로드 완료: {total_loaded}개")
        self._update_all_ui()

        # 메모리 사용량 표시
        memory_mb = self.state.memory_usage_mb
        messagebox.showinfo(
            "로딩 완료",
            f"이미지 로드 완료!\n\n"
            f"로드된 파일: {total_loaded}개\n"
            f"메모리 사용: {memory_mb:.1f}MB"
        )

    def _start_periodic_check(self):
        """주기적 상태 확인 시작"""
        self._check_batch_state()

    def _check_batch_state(self):
        """
        배치 상태 확인 - 유휴 시 폴링 주기를 늘려 CPU 절약

        배치 실행 중: 100ms (빠른 반응)
        유휴 상태   : 500ms (CPU 부하 최소화)
        """
        if self.state.batch_running:
            self.btn_evaluate_all.config(state='disabled')
            self.btn_stop.config(state='normal')
            interval = 100
        else:
            self.btn_evaluate_all.config(state='normal')
            self.btn_stop.config(state='disabled')
            interval = 500

        self.root.after(interval, self._check_batch_state)

    def _on_state_changed(self):
        """상태 변경 시 UI 업데이트"""
        self.root.after(0, self._update_all_ui)

    def _update_all_ui(self):
        """모든 UI 업데이트"""
        self._update_file_list()
        self._update_nav_label()
        self._update_roi_label()
        self._update_zoom_label()
        self._update_display()
        self._update_result()

    def _on_progress(self, current: int, total: int, filename: str):
        """진행 상황 업데이트"""
        self.root.after(0, lambda: self._update_progress_ui(current, total))

    def _update_progress_ui(self, current: int, total: int):
        """진행률 UI 업데이트"""
        self.progress_var.set(f"{current}/{total}")
        self.progress_bar['value'] = (current / total) * 100

    def _on_batch_complete(self):
        """배치 완료"""
        self.root.after(0, self._show_batch_summary)

    def _show_batch_summary(self):
        """배치 요약 표시"""
        self._update_all_ui()
        self.progress_var.set("완료!")

        summary = self.controller.get_batch_summary()

        messagebox.showinfo(
            "배치 평가 완료",
            f"총 {summary['total']}개 이미지\n\n"
            f"✓ 통과: {summary['passed']}\n"
            f"△ subset 증가 필요: {summary['warning']}\n"
            f"✗ 실패: {summary['failed']}\n\n"
            f"권장 Subset: {summary['max_subset']}px"
        )

    def _update_display(self):
        """캔버스 업데이트"""
        if self.state.current_image is None:
            return

        report = self.state.get_report(self.state.current_file)

        self.canvas_view.display(
            image=self.state.current_image,
            zoom=self.state.zoom_level,
            pan_offset=self.state.pan_offset,
            roi=self.state.roi,
            report=report,
            show_all_poi=self.show_all_poi_var.get(),
            show_bad_poi=self.show_bad_poi_var.get()
        )

    def _update_file_list(self):
        """파일 목록 업데이트"""
        self.file_listbox.delete(0, tk.END)

        for filename in self.state.file_list:
            report = self.state.get_report(filename)
            if report:
                if not report.analyzable:
                    grade = "✗"
                elif report.recommended_subset_size > report.current_subset_size:
                    grade = "△"
                else:
                    grade = "✓"
                self.file_listbox.insert(tk.END, f"{grade} {filename}")
            else:
                self.file_listbox.insert(tk.END, f"  {filename}")

        # 현재 선택 유지
        if self.state.file_list:
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(self.state.current_index)
            self.file_listbox.see(self.state.current_index)

    def _update_nav_label(self):
        """네비게이션 레이블 업데이트"""
        self.nav_label.config(
            text=f"{self.state.current_position} / {self.state.total_images}"
        )

    def _update_roi_label(self):
        """ROI 레이블 업데이트"""
        if self.state.roi:
            x, y, w, h = self.state.roi
            self.roi_label.config(text=f"ROI: ({x},{y}) {w}x{h}")
        else:
            self.roi_label.config(text="ROI: 전체 이미지")

    def _update_zoom_label(self):
        """줌 레이블 업데이트"""
        self.zoom_label.config(
            text=f"줌: {int(self.state.zoom_level * 100)}%")

    def _update_result(self):
        """결과 텍스트 업데이트"""
        self.result_text.delete(1.0, tk.END)

        if not self.state.current_file:
            return

        report = self.state.get_report(self.state.current_file)

        if not report:
            self.result_text.insert(tk.END, "평가되지 않음")
            return

        text = f"===== {self.state.current_file} =====\n\n"

        # MIG
        mig_status = "✓" if report.mig_pass else "✗"
        text += f"[MIG] {report.mig:.2f} (≥{report.mig_threshold}) {mig_status}\n\n"

        # SSSIG
        if report.sssig_result:
            sr = report.sssig_result
            sssig_status = "✓" if report.sssig_pass else "✗"
            text += f"[SSSIG] size={sr.subset_size}, spacing={sr.spacing}\n"
            text += f"  POI 수: {len(sr.points_y)}개\n"
            text += f"  Mean: {sr.mean:.2e}\n"
            text += f"  Min:  {sr.min:.2e}\n"
            text += f"  Bad:  {sr.n_bad_points}개 {sssig_status}\n\n"

        # Subset 권장
        text += f"[현재 Subset] {report.current_subset_size}px\n"
        text += f"[권장 Subset] {report.recommended_subset_size}px\n"

        if report.recommended_subset_size > report.current_subset_size:
            text += "  ⚠ Subset 크기 증가 필요!\n"

        # 예상 정확도
        if report.predicted_accuracy:
            text += f"[예상 정확도] {report.predicted_accuracy:.4f} px\n"

        # 결과
        text += f"\n[결과] {report.quality_grade}\n"
        text += f"[처리시간] {report.processing_time:.3f}s\n"

        # 경고
        if report.warnings:
            text += "\n[경고]\n"
            for w in report.warnings:
                text += f"  ⚠ {w}\n"

        self.result_text.insert(tk.END, text)

    def _on_file_select(self, event):
        """파일 선택 이벤트"""
        selection = self.file_listbox.curselection()
        if selection:
            self.controller.navigate_to(selection[0])


def main():
    """메인 함수"""
    root = tk.Tk()
    app = SpeckleQualityGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
