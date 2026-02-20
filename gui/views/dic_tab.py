"""DIC 변위 분석 탭"""

import logging
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import numpy as np

from .canvas_view import CanvasView
from ..models.settings import SettingsManager

_logger = logging.getLogger(__name__)

class DICTab(ttk.Frame):
    """DIC 변위 분석을 위한 탭"""
    
    def _zoom_in(self):
        """줌 인"""
        self._call('zoom_in')

    def _zoom_out(self):
        """줌 아웃"""
        self._call('zoom_out')

    def _set_zoom_1to1(self):
        """1:1 줌"""
        self._call('set_zoom_1to1')

    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.callbacks: Dict[str, Callable] = {}
        self.current_index = 0
        self.sequence_files: List[Path] = []
        
        self._setup_ui()
        self._setup_key_bindings()

    def _setup_key_bindings(self):
        """키보드 바인딩 설정"""
        self.bind('<Left>', self._on_key_left)
        self.bind('<Right>', self._on_key_right)
        self.bind('<Up>', self._on_key_up)
        self.bind('<Down>', self._on_key_down)
        
        self.canvas_view.canvas.bind('<Left>', self._on_key_left)
        self.canvas_view.canvas.bind('<Right>', self._on_key_right)
        self.canvas_view.canvas.bind('<Up>', self._on_key_up)
        self.canvas_view.canvas.bind('<Down>', self._on_key_down)

    def _on_key_left(self, event):
        if self.sequence_files and self.current_index > 0:
            self._call('select_image_index', self.current_index - 1)
        return "break"

    def _on_key_right(self, event):
        if self.sequence_files and self.current_index < len(self.sequence_files) - 1:
            self._call('select_image_index', self.current_index + 1)
        return "break"

    def _on_key_up(self, event):
        if self.sequence_files:
            new_index = max(0, self.current_index - 10)
            self._call('select_image_index', new_index)
        return "break"

    def _on_key_down(self, event):
        if self.sequence_files:
            new_index = min(len(self.sequence_files) - 1, self.current_index + 10)
            self._call('select_image_index', new_index)
        return "break"

    def _setup_ui(self):
        """UI 구성"""
        self.main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self._setup_left_panel()
        self._setup_center_panel()
        self._setup_right_panel()
    
    def _setup_left_panel(self):
        """왼쪽 패널: 이미지 선택 및 파라미터"""
        left_frame = ttk.Frame(self.main_paned, width=280)
        self.main_paned.add(left_frame, weight=0)
        
        # 스크롤 가능한 영역
        canvas = tk.Canvas(left_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left_frame, orient="vertical", command=canvas.yview)
        self.left_content = ttk.Frame(canvas)
        
        self.left_content.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=self.left_content, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # === 이미지 선택 섹션 ===
        img_frame = ttk.LabelFrame(self.left_content, text="이미지", padding=5)
        img_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Reference
        ref_frame = ttk.Frame(img_frame)
        ref_frame.pack(fill=tk.X, pady=2)
        ttk.Label(ref_frame, text="Ref:", width=5).pack(side=tk.LEFT)
        self.ref_label = ttk.Label(ref_frame, text="선택 안됨", foreground="gray", width=20)
        self.ref_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.ref_btn = ttk.Button(ref_frame, text="선택", width=5, command=self._select_reference)
        self.ref_btn.pack(side=tk.RIGHT)
        
        # Deformed
        def_frame = ttk.Frame(img_frame)
        def_frame.pack(fill=tk.X, pady=2)
        ttk.Label(def_frame, text="Def:", width=5).pack(side=tk.LEFT)
        self.def_label = ttk.Label(def_frame, text="선택 안됨", foreground="gray", width=20)
        self.def_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.def_btn = ttk.Button(def_frame, text="선택", width=5, command=self._select_deformed)
        self.def_btn.pack(side=tk.RIGHT)
        
        # 시퀀스 폴더
        seq_frame = ttk.Frame(img_frame)
        seq_frame.pack(fill=tk.X, pady=2)
        ttk.Label(seq_frame, text="폴더:", width=5).pack(side=tk.LEFT)
        self.seq_label = ttk.Label(seq_frame, text="선택 안됨", foreground="gray", width=20)
        self.seq_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.seq_btn = ttk.Button(seq_frame, text="선택", width=5, command=self._select_sequence)
        self.seq_btn.pack(side=tk.RIGHT)
        
        # 품질평가 동기화 버튼
        self.sync_btn = ttk.Button(img_frame, text="품질평가 탭에서 가져오기",
                                    command=self._sync_from_quality_tab)
        self.sync_btn.pack(fill=tk.X, pady=(5, 0))
        
        # === 파라미터 섹션 (통합) ===
        param_frame = ttk.LabelFrame(self.left_content, text="파라미터", padding=5)
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 그리드 레이아웃
        grid = ttk.Frame(param_frame)
        grid.pack(fill=tk.X)
        
        # Subset Size
        ttk.Label(grid, text="Subset:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.subset_var = tk.IntVar(value=21)
        self.subset_spin = ttk.Spinbox(grid, textvariable=self.subset_var,
                                        values=[11,13,15,17,19,21,23,25,27,29,31,35,41,51,61],
                                        width=8)
        self.subset_spin.grid(row=0, column=1, pady=2)
        
        # Spacing
        ttk.Label(grid, text="Spacing:").grid(row=0, column=2, sticky=tk.W, padx=(10,0), pady=2)
        self.spacing_var = tk.IntVar(value=16)
        self.spacing_spin = ttk.Spinbox(grid, from_=5, to=50, textvariable=self.spacing_var, width=8)
        self.spacing_spin.grid(row=0, column=3, pady=2)
        
        # Search Range
        ttk.Label(grid, text="Search:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.search_var = tk.IntVar(value=50)
        self.search_spin = ttk.Spinbox(grid, from_=10, to=200, textvariable=self.search_var, width=8)
        self.search_spin.grid(row=1, column=1, pady=2)
        
        # FFT-CC ZNCC Threshold (낮게, 텍스처 없음 검출용)
        ttk.Label(grid, text="FFT ZNCC:").grid(row=1, column=2, sticky=tk.W, padx=(10,0), pady=2)
        self.zncc_var = tk.DoubleVar(value=0.3)
        self.zncc_spin = ttk.Spinbox(grid, from_=0.1, to=0.99, increment=0.05,
                                    textvariable=self.zncc_var, width=8)
        self.zncc_spin.grid(row=1, column=3, pady=2)
        
        # Shape Function
        ttk.Label(grid, text="Shape:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.shape_func_var = tk.StringVar(value='affine')
        shape_combo = ttk.Combobox(grid, textvariable=self.shape_func_var,
                                    values=['affine', 'quadratic'], width=10, state='readonly')
        shape_combo.grid(row=2, column=1, pady=2)
        
        # Interpolation
        ttk.Label(grid, text="Interp:").grid(row=2, column=2, sticky=tk.W, padx=(10,0), pady=2)
        self.interp_var = tk.StringVar(value='bicubic')
        interp_combo = ttk.Combobox(grid, textvariable=self.interp_var,
                                    values=['bicubic', 'biquintic'], width=10, state='readonly')
        interp_combo.grid(row=2, column=3, pady=2)

        # IC-GN ZNCC Threshold (높게, 최종 품질 판단용) — 별도 행
        icgn_zncc_frame = ttk.Frame(param_frame)
        icgn_zncc_frame.pack(fill=tk.X, pady=(4, 0))
        ttk.Label(icgn_zncc_frame, text="IC-GN ZNCC:").pack(side=tk.LEFT)
        self.icgn_zncc_var = tk.DoubleVar(value=0.85)
        ttk.Spinbox(icgn_zncc_frame, from_=0.1, to=0.99, increment=0.05,
                    textvariable=self.icgn_zncc_var, width=8).pack(side=tk.LEFT, padx=(5, 0))
        ttk.Label(icgn_zncc_frame, text="(최종 품질)", foreground="gray",
                font=("", 8)).pack(side=tk.LEFT, padx=(5, 0))

        # Gaussian Blur
        blur_frame = ttk.Frame(param_frame)
        blur_frame.pack(fill=tk.X, pady=(5, 0))
        
        self.gaussian_blur_var = tk.BooleanVar(value=False)
        self.gaussian_blur_check = ttk.Checkbutton(blur_frame, text="Gaussian Blur:",
                                                    variable=self.gaussian_blur_var,
                                                    command=self._toggle_gaussian_blur)
        self.gaussian_blur_check.pack(side=tk.LEFT)
        
        self.blur_kernel_var = tk.IntVar(value=5)
        self.blur_kernel_combo = ttk.Combobox(blur_frame, textvariable=self.blur_kernel_var,
                                            values=[3, 5, 7, 9], width=5, state='disabled')
        self.blur_kernel_combo.pack(side=tk.LEFT, padx=(5, 0))
        
        # 고급 옵션
        self.advanced_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(param_frame, text="고급 옵션",
                        variable=self.advanced_var,
                        command=self._toggle_advanced_options).pack(anchor=tk.W, pady=(5, 0))
        
        self.advanced_frame = ttk.Frame(param_frame)
        adv_grid = ttk.Frame(self.advanced_frame)
        adv_grid.pack(fill=tk.X)
        
        ttk.Label(adv_grid, text="수렴 기준:").grid(row=0, column=0, sticky=tk.W)
        self.conv_threshold_var = tk.DoubleVar(value=0.001)
        ttk.Entry(adv_grid, textvariable=self.conv_threshold_var, width=10).grid(row=0, column=1)
        
        ttk.Label(adv_grid, text="최대 반복:").grid(row=1, column=0, sticky=tk.W)
        self.max_iter_var = tk.IntVar(value=20)
        ttk.Spinbox(adv_grid, from_=10, to=200, textvariable=self.max_iter_var, width=8).grid(row=1, column=1)
        
        # === 분석 섹션 ===
        analysis_frame = ttk.LabelFrame(self.left_content, text="분석", padding=5)
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        btn_frame = ttk.Frame(analysis_frame)
        btn_frame.pack(fill=tk.X)
        
        self.analyze_btn = ttk.Button(btn_frame, text="현재 분석", command=self._run_analysis)
        self.analyze_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 2))
        
        self.batch_btn = ttk.Button(btn_frame, text="전체 분석", command=self._run_batch_analysis)
        self.batch_btn.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(2, 0))
        
        # 진행 상황
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(analysis_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        self.progress_label = ttk.Label(analysis_frame, text="대기 중", font=("", 8))
        self.progress_label.pack(anchor=tk.W)
        
        self.stop_btn = ttk.Button(analysis_frame, text="중지", state=tk.DISABLED,
                                    command=self._stop_analysis)
        self.stop_btn.pack(fill=tk.X, pady=(5, 0))

    def _setup_center_panel(self):
        """중앙 패널: 캔버스"""
        center_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(center_frame, weight=1)
        
        canvas_container = ttk.Frame(center_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_view = CanvasView(canvas_container)
        
        # 줌 컨트롤
        zoom_frame = ttk.Frame(center_frame)
        zoom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(zoom_frame, text="−", width=3, command=self._zoom_out).pack(side=tk.LEFT)
        self.zoom_label = ttk.Label(zoom_frame, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=10)
        ttk.Button(zoom_frame, text="+", width=3, command=self._zoom_in).pack(side=tk.LEFT)
        ttk.Button(zoom_frame, text="Fit", width=5, command=self._fit_to_canvas).pack(side=tk.LEFT, padx=10)
        ttk.Button(zoom_frame, text="1:1", width=5, command=self._set_zoom_1to1).pack(side=tk.LEFT)

    def _setup_right_panel(self):
        """오른쪽 패널: 표시 옵션 + 결과 + 파일 목록"""
        right_frame = ttk.Frame(self.main_paned, width=280)
        self.main_paned.add(right_frame, weight=0)
        
        # === 표시 옵션 ===
        display_frame = ttk.LabelFrame(right_frame, text="표시", padding=5)
        display_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.display_mode_var = tk.StringVar(value="invalid")
        
        # 변위 섹션
        ttk.Label(display_frame, text="변위:", font=("", 8, "bold")).pack(anchor=tk.W)
        disp_frame = ttk.Frame(display_frame)
        disp_frame.pack(fill=tk.X)
        
        for text, value in [("벡터", "vectors"), ("U", "u_field"), ("V", "v_field"), ("크기", "magnitude")]:
            ttk.Radiobutton(disp_frame, text=text, value=value,
                        variable=self.display_mode_var,
                        command=self._update_display).pack(side=tk.LEFT)
        
        # 변형률 섹션
        ttk.Label(display_frame, text="변형률:", font=("", 8, "bold")).pack(anchor=tk.W, pady=(5, 0))
        strain_frame1 = ttk.Frame(display_frame)
        strain_frame1.pack(fill=tk.X)
        
        for text, value in [("εxx", "exx"), ("εyy", "eyy"), ("εxy", "exy")]:
            ttk.Radiobutton(strain_frame1, text=text, value=value,
                        variable=self.display_mode_var,
                        command=self._update_display).pack(side=tk.LEFT)
        
        strain_frame2 = ttk.Frame(display_frame)
        strain_frame2.pack(fill=tk.X)
        
        for text, value in [("ε1", "e1"), ("von Mises", "von_mises")]:
            ttk.Radiobutton(strain_frame2, text=text, value=value,
                        variable=self.display_mode_var,
                        command=self._update_display).pack(side=tk.LEFT)
        
        # 기타 섹션
        ttk.Label(display_frame, text="기타:", font=("", 8, "bold")).pack(anchor=tk.W, pady=(5, 0))
        etc_frame = ttk.Frame(display_frame)
        etc_frame.pack(fill=tk.X)
        
        for text, value in [("ZNCC", "zncc"), ("불량", "invalid")]:
            ttk.Radiobutton(etc_frame, text=text, value=value,
                        variable=self.display_mode_var,
                        command=self._update_display).pack(side=tk.LEFT)
        
        # === 컬러바 + 범위 조절 ===
        colorbar_frame = ttk.LabelFrame(right_frame, text="컬러 범위", padding=5)
        colorbar_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 컬러바 캔버스
        self.colorbar_canvas = tk.Canvas(colorbar_frame, width=200, height=25, bg='gray20', highlightthickness=0)
        self.colorbar_canvas.pack(fill=tk.X, pady=(0, 5))
        
        # 자동/수동 범위
        range_mode_frame = ttk.Frame(colorbar_frame)
        range_mode_frame.pack(fill=tk.X)
        
        self.range_auto_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(range_mode_frame, text="자동 범위", 
                    variable=self.range_auto_var,
                    command=self._on_range_mode_changed).pack(side=tk.LEFT)
        
        # 수동 범위 입력
        self.range_frame = ttk.Frame(colorbar_frame)
        self.range_frame.pack(fill=tk.X, pady=(5, 0))
        
        ttk.Label(self.range_frame, text="Min:").pack(side=tk.LEFT)
        self.range_min_var = tk.StringVar(value="-1.0")
        self.range_min_entry = ttk.Entry(self.range_frame, textvariable=self.range_min_var, width=8)
        self.range_min_entry.pack(side=tk.LEFT, padx=(2, 10))
        
        ttk.Label(self.range_frame, text="Max:").pack(side=tk.LEFT)
        self.range_max_var = tk.StringVar(value="1.0")
        self.range_max_entry = ttk.Entry(self.range_frame, textvariable=self.range_max_var, width=8)
        self.range_max_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Button(self.range_frame, text="적용", width=4,
                command=self._update_display).pack(side=tk.LEFT, padx=(5, 0))
        
        # 초기에는 수동 범위 비활성화
        self._on_range_mode_changed()
        
        # 현재 범위 표시 라벨
        self.range_label = ttk.Label(colorbar_frame, text="", font=("Consolas", 8))
        self.range_label.pack(anchor=tk.W)
        
        # === 결과 정보 ===
        result_frame = ttk.LabelFrame(right_frame, text="결과", padding=5)
        result_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.result_text = tk.Text(result_frame, height=8, width=28, 
                                    state=tk.DISABLED, font=("Consolas", 8))
        result_scroll = ttk.Scrollbar(result_frame, orient=tk.VERTICAL,
                                    command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scroll.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # === 검증 결과 ===
        valid_frame = ttk.LabelFrame(right_frame, text="검증", padding=5)
        valid_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.valid_text = tk.Text(valid_frame, height=4, width=28,
                                state=tk.DISABLED, font=("Consolas", 8))
        self.valid_text.pack(fill=tk.BOTH, expand=True)
        
        # === 파일 목록 ===
        file_frame = ttk.LabelFrame(right_frame, text="시퀀스", padding=5)
        file_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.nav_label = ttk.Label(file_frame, text="0 / 0", font=("", 9, "bold"))
        self.nav_label.pack(anchor=tk.W)
        
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.file_listbox = tk.Listbox(list_frame, height=6, font=("", 8), selectmode=tk.SINGLE)
        file_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=file_scroll.set)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.bind('<<ListboxSelect>>', self._on_file_select)
    
    def _on_range_mode_changed(self):
        """자동/수동 범위 모드 변경"""
        if self.range_auto_var.get():
            # 자동 모드: 입력 비활성화
            self.range_min_entry.configure(state='disabled')
            self.range_max_entry.configure(state='disabled')
        else:
            # 수동 모드: 입력 활성화
            self.range_min_entry.configure(state='normal')
            self.range_max_entry.configure(state='normal')
        
        self._update_display()

    def update_colorbar(self, vmin: float, vmax: float, label: str = "", colormap: str = 'turbo'):
        """컬러바 업데이트"""
        self.colorbar_canvas.delete("all")

        width = self.colorbar_canvas.winfo_width()
        if width < 10:
            width = 200
        height = 25

        # matplotlib turbo colormap 사용
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            cmap = plt.get_cmap('turbo')

            for i in range(width):
                norm_val = i / width
                r, g, b, _ = cmap(norm_val)
                r, g, b = int(r * 255), int(g * 255), int(b * 255)
                color = f'#{r:02x}{g:02x}{b:02x}'
                self.colorbar_canvas.create_line(i, 0, i, height, fill=color)
        except Exception:
            # fallback: 수동 그라데이션
            for i in range(width):
                norm_val = i / width
                r = int(norm_val * 255)
                color = f'#{r:02x}00{255-r:02x}'
                self.colorbar_canvas.create_line(i, 0, i, height, fill=color)

        self.range_label.configure(text=f"{label}: [{vmin:.4g}, {vmax:.4g}]")

        if self.range_auto_var.get():
            self.range_min_var.set(f"{vmin:.4g}")
            self.range_max_var.set(f"{vmax:.4g}")

    def get_color_range(self):
        """현재 컬러 범위 반환 (자동이면 None, 수동이면 (min, max))"""
        if self.range_auto_var.get():
            return None
        else:
            try:
                vmin = float(self.range_min_var.get())
                vmax = float(self.range_max_var.get())
                return (vmin, vmax)
            except ValueError:
                return None
    
    # === 콜백 메서드 ===
    
    def set_callback(self, name: str, callback: Callable):
        self.callbacks[name] = callback
    
    def _call(self, name: str, *args, **kwargs):
        if name in self.callbacks:
            return self.callbacks[name](*args, **kwargs)
    
    def _select_reference(self):
        self._call('select_reference')
    
    def _select_deformed(self):
        self._call('select_deformed')
    
    def _select_sequence(self):
        self._call('select_sequence')
    
    def _sync_from_quality_tab(self):
        self._call('sync_from_quality')
    
    def _run_analysis(self):
        self._call('run_analysis', self.get_parameters())
    
    def _run_batch_analysis(self):
        self._call('run_batch_analysis', self.get_parameters())
    
    def _stop_analysis(self):
        self._call('stop_analysis')
    
    def _update_display(self):
        self._call('update_display', self.display_mode_var.get())

    def _toggle_gaussian_blur(self):
        if self.gaussian_blur_var.get():
            self.blur_kernel_combo.configure(state='readonly')
        else:
            self.blur_kernel_combo.configure(state='disabled')

    def _toggle_advanced_options(self):
        if self.advanced_var.get():
            self.advanced_frame.pack(fill=tk.X, pady=(5, 0))
        else:
            self.advanced_frame.pack_forget()
    
    def _fit_to_canvas(self):
        self._call('fit_to_canvas')
    
    def _on_file_select(self, event):
        selection = self.file_listbox.curselection()
        if selection:
            self._call('select_image_index', selection[0])
    
    # === 공개 메서드 ===
    def get_parameters(self) -> Dict[str, Any]:
        """현재 파라미터 반환"""
        gaussian_blur = None
        if self.gaussian_blur_var.get():
            gaussian_blur = self.blur_kernel_var.get()
        
        return {
            'subset_size': self.subset_var.get(),
            'spacing': self.spacing_var.get(),
            'search_range': self.search_var.get(),
            'zncc_threshold': self.zncc_var.get(),          # FFT-CC용 (0.3 기본)
            'icgn_zncc_threshold': self.icgn_zncc_var.get(), # IC-GN용 (0.85 기본)
            'shape_function': self.shape_func_var.get(),
            'interpolation': self.interp_var.get(),
            'gaussian_blur': gaussian_blur,
            'gaussian_blur_enabled': self.gaussian_blur_var.get(),
            'conv_threshold': self.conv_threshold_var.get(),
            'max_iter': self.max_iter_var.get(),
        }

    def set_parameters(self, params: Dict[str, Any]):
        """파라미터 설정"""
        try:
            if params.get('subset_size') is not None:
                self.subset_var.set(params['subset_size'])
            if params.get('spacing') is not None:
                self.spacing_var.set(params['spacing'])
            if params.get('search_range') is not None:
                self.search_var.set(params['search_range'])
            if params.get('zncc_threshold') is not None:
                self.zncc_var.set(params['zncc_threshold'])
            if params.get('icgn_zncc_threshold') is not None:
                self.icgn_zncc_var.set(params['icgn_zncc_threshold'])
            if params.get('shape_function') is not None:
                self.shape_func_var.set(params['shape_function'])
            if params.get('interpolation') is not None:
                self.interp_var.set(params['interpolation'])
            if params.get('conv_threshold') is not None:
                self.conv_threshold_var.set(params['conv_threshold'])
            if params.get('max_iter') is not None:
                self.max_iter_var.set(params['max_iter'])
            if params.get('gaussian_blur_enabled'):
                self.gaussian_blur_var.set(True)
                self._toggle_gaussian_blur()
                if params.get('gaussian_blur'):
                    self.blur_kernel_var.set(params['gaussian_blur'])
            else:
                self.gaussian_blur_var.set(False)
                self._toggle_gaussian_blur()
        except Exception as e:
            _logger.warning(f"파라미터 설정 중 오류: {e}")
    
    def update_reference_label(self, text: str):
        self.ref_label.configure(text=text[:20], foreground="black")
    
    def update_deformed_label(self, text: str):
        self.def_label.configure(text=text[:20], foreground="black")
    
    def update_sequence_label(self, text: str):
        self.seq_label.configure(text=text[:20], foreground="black")
    
    def update_file_list(self, files: List[Path], current_index: int = 0):
        self.sequence_files = files
        self.current_index = current_index
        
        self.file_listbox.delete(0, tk.END)
        for f in files:
            self.file_listbox.insert(tk.END, f.name)
        
        if files:
            self.file_listbox.selection_set(current_index)
            self.file_listbox.see(current_index)
        
        self._update_nav_label()
    
    def set_current_index(self, index: int):
        if 0 <= index < len(self.sequence_files):
            self.current_index = index
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(index)
            self.file_listbox.see(index)
            self._update_nav_label()
    
    def _update_nav_label(self):
        total = len(self.sequence_files)
        current = self.current_index + 1 if total > 0 else 0
        self.nav_label.configure(text=f"{current} / {total}")
    
    def update_progress(self, value: float, text: str = ""):
        self.progress_var.set(value)
        if text:
            self.progress_label.configure(text=text)
    
    def update_result_text(self, text: str):
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state=tk.DISABLED)
    
    def update_validation_text(self, text: str):
        self.valid_text.configure(state=tk.NORMAL)
        self.valid_text.delete(1.0, tk.END)
        self.valid_text.insert(tk.END, text)
        self.valid_text.configure(state=tk.DISABLED)
    
    def set_analysis_state(self, running: bool):
        state = tk.DISABLED if running else tk.NORMAL
        self.analyze_btn.configure(state=state)
        self.batch_btn.configure(state=state)
        self.stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.ref_btn.configure(state=state)
        self.def_btn.configure(state=state)
        self.seq_btn.configure(state=state)
