"""DIC 변위 분석 탭"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from typing import Optional, Callable, Dict, Any, List
import numpy as np

from .canvas_view import CanvasView


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
        
        # ZNCC Threshold
        ttk.Label(grid, text="ZNCC:").grid(row=1, column=2, sticky=tk.W, padx=(10,0), pady=2)
        self.zncc_var = tk.DoubleVar(value=0.7)
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
        """오른쪽 패널: 결과 정보 + 파일 목록"""
        right_frame = ttk.Frame(self.main_paned, width=250)
        self.main_paned.add(right_frame, weight=0)
        
        # === 표시 옵션 ===
        display_frame = ttk.LabelFrame(right_frame, text="표시", padding=5)
        display_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.display_mode_var = tk.StringVar(value="vectors")
        modes = [("벡터", "vectors"), ("크기", "magnitude"), ("ZNCC", "zncc"), ("불량", "invalid")]
        mode_frame = ttk.Frame(display_frame)
        mode_frame.pack(fill=tk.X)
        for text, value in modes:
            ttk.Radiobutton(mode_frame, text=text, value=value,
                           variable=self.display_mode_var,
                           command=self._update_display).pack(side=tk.LEFT)
        
        # === 결과 정보 ===
        result_frame = ttk.LabelFrame(right_frame, text="결과", padding=5)
        result_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.result_text = tk.Text(result_frame, height=10, width=28, 
                                    state=tk.DISABLED, font=("Consolas", 8))
        result_scroll = ttk.Scrollbar(result_frame, orient=tk.VERTICAL,
                                       command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scroll.set)
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # === 검증 결과 ===
        valid_frame = ttk.LabelFrame(right_frame, text="검증", padding=5)
        valid_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.valid_text = tk.Text(valid_frame, height=5, width=28,
                                   state=tk.DISABLED, font=("Consolas", 8))
        self.valid_text.pack(fill=tk.BOTH, expand=True)
        
        # === 파일 목록 ===
        file_frame = ttk.LabelFrame(right_frame, text="시퀀스", padding=5)
        file_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        self.nav_label = ttk.Label(file_frame, text="0 / 0", font=("", 9, "bold"))
        self.nav_label.pack(anchor=tk.W)
        
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.file_listbox = tk.Listbox(list_frame, height=8, font=("", 8), selectmode=tk.SINGLE)
        file_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=file_scroll.set)
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.file_listbox.bind('<<ListboxSelect>>', self._on_file_select)
    
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
        gaussian_blur = None
        if self.gaussian_blur_var.get():
            gaussian_blur = self.blur_kernel_var.get()
        
        return {
            'subset_size': self.subset_var.get(),
            'spacing': self.spacing_var.get(),
            'search_range': self.search_var.get(),
            'zncc_threshold': self.zncc_var.get(),
            'shape_function': self.shape_func_var.get(),
            'interpolation': self.interp_var.get(),
            'conv_threshold': self.conv_threshold_var.get(),
            'max_iter': self.max_iter_var.get(),
            'gaussian_blur': gaussian_blur,
        }
    
    def set_parameters(self, params: Dict[str, Any]):
        if 'subset_size' in params:
            self.subset_var.set(params['subset_size'])
        if 'spacing' in params:
            self.spacing_var.set(params['spacing'])
    
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
