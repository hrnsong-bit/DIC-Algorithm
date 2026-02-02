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
        
        # 콜백 저장
        self.callbacks: Dict[str, Callable] = {}
        
        # 현재 상태
        self.current_index = 0
        self.sequence_files: List[Path] = []
        
        self._setup_ui()
    
    def _setup_ui(self):
        """UI 구성"""
        # 메인 PanedWindow (좌-중앙-우 분할)
        self.main_paned = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 왼쪽 패널 (파라미터)
        self._setup_left_panel()
        
        # 중앙 패널 (캔버스)
        self._setup_center_panel()
        
        # 오른쪽 패널 (결과 + 파일 목록)
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
        img_frame = ttk.LabelFrame(self.left_content, text="이미지 선택", padding=10)
        img_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Reference 이미지
        ttk.Label(img_frame, text="Reference 이미지:").pack(anchor=tk.W)
        ref_frame = ttk.Frame(img_frame)
        ref_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.ref_label = ttk.Label(ref_frame, text="선택 안됨", foreground="gray")
        self.ref_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.ref_btn = ttk.Button(ref_frame, text="선택", width=6,
                                   command=self._select_reference)
        self.ref_btn.pack(side=tk.RIGHT)
        
        # Deformed 이미지
        ttk.Label(img_frame, text="Deformed 이미지:").pack(anchor=tk.W)
        def_frame = ttk.Frame(img_frame)
        def_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.def_label = ttk.Label(def_frame, text="선택 안됨", foreground="gray")
        self.def_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.def_btn = ttk.Button(def_frame, text="선택", width=6,
                                   command=self._select_deformed)
        self.def_btn.pack(side=tk.RIGHT)
        
        ttk.Separator(img_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        
        # 시퀀스 폴더
        ttk.Label(img_frame, text="또는 시퀀스 폴더:").pack(anchor=tk.W)
        seq_frame = ttk.Frame(img_frame)
        seq_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.seq_label = ttk.Label(seq_frame, text="선택 안됨", foreground="gray")
        self.seq_label.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.seq_btn = ttk.Button(seq_frame, text="선택", width=6,
                                   command=self._select_sequence)
        self.seq_btn.pack(side=tk.RIGHT)
        
        # === DIC 파라미터 섹션 ===
        param_frame = ttk.LabelFrame(self.left_content, text="DIC 파라미터", padding=10)
        param_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # Subset Size
        ttk.Label(param_frame, text="Subset Size (px):").pack(anchor=tk.W)
        self.subset_var = tk.IntVar(value=21)
        self.subset_spin = ttk.Spinbox(param_frame, from_=11, to=101, increment=2,
                                        textvariable=self.subset_var, width=10)
        self.subset_spin.pack(anchor=tk.W, pady=(0, 5))
        
        # Spacing
        ttk.Label(param_frame, text="Spacing (px):").pack(anchor=tk.W)
        self.spacing_var = tk.IntVar(value=10)
        self.spacing_spin = ttk.Spinbox(param_frame, from_=1, to=50,
                                         textvariable=self.spacing_var, width=10)
        self.spacing_spin.pack(anchor=tk.W, pady=(0, 5))
        
        # Search Range
        ttk.Label(param_frame, text="Search Range (px):").pack(anchor=tk.W)
        search_tip = ttk.Label(param_frame, text="예상 최대 변위보다 크게", 
                               foreground="gray", font=("", 8))
        search_tip.pack(anchor=tk.W)
        self.search_var = tk.IntVar(value=50)
        self.search_spin = ttk.Spinbox(param_frame, from_=10, to=200,
                                        textvariable=self.search_var, width=10)
        self.search_spin.pack(anchor=tk.W, pady=(0, 5))
        
        # ZNCC Threshold
        ttk.Label(param_frame, text="ZNCC Threshold:").pack(anchor=tk.W)
        zncc_tip = ttk.Label(param_frame, text="낮을수록 관대 (불연속 검출용)", 
                             foreground="gray", font=("", 8))
        zncc_tip.pack(anchor=tk.W)
        self.zncc_var = tk.DoubleVar(value=0.7)
        self.zncc_spin = ttk.Spinbox(param_frame, from_=0.1, to=0.99, increment=0.05,
                                      textvariable=self.zncc_var, width=10)
        self.zncc_spin.pack(anchor=tk.W, pady=(0, 5))
        
        # 품질평가 파라미터 동기화 버튼
        self.sync_btn = ttk.Button(param_frame, text="품질평가 탭에서 가져오기",
                                    command=self._sync_from_quality_tab)
        self.sync_btn.pack(fill=tk.X, pady=(10, 0))
        
        # === 분석 섹션 ===
        analysis_frame = ttk.LabelFrame(self.left_content, text="분석", padding=10)
        analysis_frame.pack(fill=tk.X, padx=5, pady=5)
        
        # 단일 분석 버튼
        self.analyze_btn = ttk.Button(analysis_frame, text="FFT-CC 분석",
                                       command=self._run_analysis)
        self.analyze_btn.pack(fill=tk.X, pady=(0, 5))
        
        # 배치 분석 버튼
        self.batch_btn = ttk.Button(analysis_frame, text="전체 시퀀스 분석",
                                     command=self._run_batch_analysis)
        self.batch_btn.pack(fill=tk.X, pady=(0, 5))
        
        # 진행 상황
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ttk.Progressbar(analysis_frame, variable=self.progress_var,
                                             maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        
        self.progress_label = ttk.Label(analysis_frame, text="대기 중")
        self.progress_label.pack(anchor=tk.W)
        
        # 중지 버튼
        self.stop_btn = ttk.Button(analysis_frame, text="중지", state=tk.DISABLED,
                                    command=self._stop_analysis)
        self.stop_btn.pack(fill=tk.X, pady=(5, 0))
        
        # === 내보내기 섹션 ===
        export_frame = ttk.LabelFrame(self.left_content, text="내보내기", padding=10)
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.export_csv_btn = ttk.Button(export_frame, text="결과 CSV 저장",
                                          command=self._export_csv)
        self.export_csv_btn.pack(fill=tk.X, pady=(0, 5))
        
        self.export_img_btn = ttk.Button(export_frame, text="변위 이미지 저장",
                                          command=self._export_image)
        self.export_img_btn.pack(fill=tk.X)
    
    def _setup_center_panel(self):
        """중앙 패널: 캔버스"""
        center_frame = ttk.Frame(self.main_paned)
        self.main_paned.add(center_frame, weight=1)
        
        # 캔버스 영역 (CanvasView가 내부에서 pack 처리함)
        canvas_container = ttk.Frame(center_frame)
        canvas_container.pack(fill=tk.BOTH, expand=True)
        
        self.canvas_view = CanvasView(canvas_container)
        # pack 호출하지 않음!
        
        # 줌 컨트롤 (하단)
        zoom_frame = ttk.Frame(center_frame)
        zoom_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(zoom_frame, text="−", width=3,
                command=self._zoom_out).pack(side=tk.LEFT)
        
        self.zoom_label = ttk.Label(zoom_frame, text="100%")
        self.zoom_label.pack(side=tk.LEFT, padx=10)
        
        ttk.Button(zoom_frame, text="+", width=3,
                command=self._zoom_in).pack(side=tk.LEFT)
        
        ttk.Button(zoom_frame, text="Fit", width=5,
                command=self._fit_to_canvas).pack(side=tk.LEFT, padx=10)
        
        ttk.Button(zoom_frame, text="1:1", width=5,
                command=self._set_zoom_1to1).pack(side=tk.LEFT)

    def _setup_right_panel(self):
        """오른쪽 패널: 결과 정보 + 파일 목록"""
        right_frame = ttk.Frame(self.main_paned, width=280)
        self.main_paned.add(right_frame, weight=0)
        
        # === 표시 옵션 ===
        display_frame = ttk.LabelFrame(right_frame, text="표시 옵션", padding=10)
        display_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(display_frame, text="표시 모드:").pack(anchor=tk.W)
        self.display_mode_var = tk.StringVar(value="vectors")
        modes = [
            ("변위 벡터", "vectors"),
            ("변위 크기 (컬러맵)", "magnitude"),
            ("ZNCC 맵", "zncc"),
            ("불량 포인트", "invalid")
        ]
        for text, value in modes:
            ttk.Radiobutton(display_frame, text=text, value=value,
                           variable=self.display_mode_var,
                           command=self._update_display).pack(anchor=tk.W)
        
        # 벡터 스케일
        ttk.Label(display_frame, text="벡터 스케일:").pack(anchor=tk.W, pady=(10, 0))
        self.vector_scale_var = tk.DoubleVar(value=1.0)
        scale_frame = ttk.Frame(display_frame)
        scale_frame.pack(fill=tk.X)
        
        self.vector_scale = ttk.Scale(scale_frame, from_=0.1, to=5.0,
                                       variable=self.vector_scale_var,
                                       command=lambda v: self._update_display())
        self.vector_scale.pack(side=tk.LEFT, fill=tk.X, expand=True)
        
        self.scale_label = ttk.Label(scale_frame, text="1.0x", width=5)
        self.scale_label.pack(side=tk.RIGHT)
        
        # === 결과 정보 ===
        result_frame = ttk.LabelFrame(right_frame, text="분석 결과", padding=10)
        result_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.result_text = tk.Text(result_frame, height=12, width=30, 
                                    state=tk.DISABLED, font=("Consolas", 9))
        result_scroll = ttk.Scrollbar(result_frame, orient=tk.VERTICAL,
                                       command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=result_scroll.set)
        
        self.result_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        result_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # === 검증 결과 ===
        valid_frame = ttk.LabelFrame(right_frame, text="검증 결과", padding=10)
        valid_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.valid_text = tk.Text(valid_frame, height=6, width=30,
                                   state=tk.DISABLED, font=("Consolas", 9))
        self.valid_text.pack(fill=tk.BOTH, expand=True)
        
        # === 파일 목록 (오른쪽 하단) ===
        file_frame = ttk.LabelFrame(right_frame, text="이미지 시퀀스", padding=5)
        file_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # 현재 위치 표시
        nav_frame = ttk.Frame(file_frame)
        nav_frame.pack(fill=tk.X, pady=(0, 5))
        
        self.nav_label = ttk.Label(nav_frame, text="0 / 0", font=("", 10, "bold"))
        self.nav_label.pack(side=tk.LEFT)
        
        # 파일 목록 (작은 리스트박스)
        list_frame = ttk.Frame(file_frame)
        list_frame.pack(fill=tk.BOTH, expand=True)
        
        self.file_listbox = tk.Listbox(list_frame, height=6, font=("", 8),
                                        selectmode=tk.SINGLE)
        file_scroll = ttk.Scrollbar(list_frame, orient=tk.VERTICAL,
                                     command=self.file_listbox.yview)
        self.file_listbox.configure(yscrollcommand=file_scroll.set)
        
        self.file_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        file_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 리스트 선택 이벤트
        self.file_listbox.bind('<<ListboxSelect>>', self._on_file_select)
    
    # === 콜백 메서드 ===
    
    def set_callback(self, name: str, callback: Callable):
        """콜백 등록"""
        self.callbacks[name] = callback
    
    def _call(self, name: str, *args, **kwargs):
        """콜백 호출"""
        if name in self.callbacks:
            return self.callbacks[name](*args, **kwargs)
    
    def _select_reference(self):
        """Reference 이미지 선택"""
        self._call('select_reference')
    
    def _select_deformed(self):
        """Deformed 이미지 선택"""
        self._call('select_deformed')
    
    def _select_sequence(self):
        """시퀀스 폴더 선택"""
        self._call('select_sequence')
    
    def _sync_from_quality_tab(self):
        """품질평가 탭에서 파라미터 가져오기"""
        self._call('sync_from_quality')
    
    def _run_analysis(self):
        """단일 FFT-CC 분석 실행"""
        self._call('run_analysis', self.get_parameters())
    
    def _run_batch_analysis(self):
        """배치 분석 실행"""
        self._call('run_batch_analysis', self.get_parameters())
    
    def _stop_analysis(self):
        """분석 중지"""
        self._call('stop_analysis')
    
    def _export_csv(self):
        """CSV 내보내기"""
        self._call('export_csv')
    
    def _export_image(self):
        """이미지 내보내기"""
        self._call('export_image')
    
    def _update_display(self):
        """표시 업데이트"""
        scale = self.vector_scale_var.get()
        self.scale_label.configure(text=f"{scale:.1f}x")
        self._call('update_display', self.display_mode_var.get(), scale)
    
    def _fit_to_canvas(self):
        """캔버스에 맞추기"""
        self._call('fit_to_canvas')
    
    def _on_zoom_changed(self, zoom_level: float):
        """줌 변경 시"""
        self.zoom_label.configure(text=f"{zoom_level * 100:.0f}%")
    
    def _on_file_select(self, event):
        """파일 목록에서 선택 시"""
        selection = self.file_listbox.curselection()
        if selection:
            index = selection[0]
            self._call('select_image_index', index)
    
    # === 공개 메서드 ===
    
    def get_parameters(self) -> Dict[str, Any]:
        """현재 파라미터 반환"""
        return {
            'subset_size': self.subset_var.get(),
            'spacing': self.spacing_var.get(),
            'search_range': self.search_var.get(),
            'zncc_threshold': self.zncc_var.get()
        }
    
    def set_parameters(self, params: Dict[str, Any]):
        """파라미터 설정 (품질평가에서 동기화 시 사용)"""
        if 'subset_size' in params:
            self.subset_var.set(params['subset_size'])
        if 'spacing' in params:
            self.spacing_var.set(params['spacing'])
        if 'search_range' in params:
            self.search_var.set(params.get('search_range', 50))
        if 'zncc_threshold' in params:
            self.zncc_var.set(params.get('zncc_threshold', 0.7))
    
    def update_reference_label(self, text: str):
        """Reference 라벨 업데이트"""
        self.ref_label.configure(text=text, foreground="black")
    
    def update_deformed_label(self, text: str):
        """Deformed 라벨 업데이트"""
        self.def_label.configure(text=text, foreground="black")
    
    def update_sequence_label(self, text: str):
        """시퀀스 라벨 업데이트"""
        self.seq_label.configure(text=text, foreground="black")
    
    def update_file_list(self, files: List[Path], current_index: int = 0):
        """파일 목록 업데이트"""
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
        """현재 인덱스 설정"""
        if 0 <= index < len(self.sequence_files):
            self.current_index = index
            self.file_listbox.selection_clear(0, tk.END)
            self.file_listbox.selection_set(index)
            self.file_listbox.see(index)
            self._update_nav_label()
    
    def _update_nav_label(self):
        """네비게이션 라벨 업데이트"""
        total = len(self.sequence_files)
        current = self.current_index + 1 if total > 0 else 0
        self.nav_label.configure(text=f"{current} / {total}")
    
    def update_progress(self, value: float, text: str = ""):
        """진행 상황 업데이트"""
        self.progress_var.set(value)
        if text:
            self.progress_label.configure(text=text)
    
    def update_result_text(self, text: str):
        """결과 텍스트 업데이트"""
        self.result_text.configure(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.configure(state=tk.DISABLED)
    
    def update_validation_text(self, text: str):
        """검증 결과 텍스트 업데이트"""
        self.valid_text.configure(state=tk.NORMAL)
        self.valid_text.delete(1.0, tk.END)
        self.valid_text.insert(tk.END, text)
        self.valid_text.configure(state=tk.DISABLED)
    
    def set_analysis_state(self, running: bool):
        """분석 상태에 따른 UI 업데이트"""
        state = tk.DISABLED if running else tk.NORMAL
        self.analyze_btn.configure(state=state)
        self.batch_btn.configure(state=state)
        self.stop_btn.configure(state=tk.NORMAL if running else tk.DISABLED)
        self.ref_btn.configure(state=state)
        self.def_btn.configure(state=state)
        self.seq_btn.configure(state=state)
