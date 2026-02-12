"""파라미터 패널 (노이즈 추정 섹션 포함)"""

import tkinter as tk
from tkinter import ttk, filedialog
from typing import Optional, Tuple, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Parameters:
    """품질평가 파라미터"""
    mig_threshold: float
    subset_size: int
    spacing: int


@dataclass
class NoiseConfig:
    """노이즈 추정 설정"""
    method: str = 'auto'          # 'auto', 'pair', 'manual', 'single'
    variance: Optional[float] = None
    std: Optional[float] = None
    pair_paths: Optional[Tuple[str, str]] = None
    estimation_done: bool = False


class ParamPanel:
    """파라미터 설정 패널"""

    def __init__(self, parent: tk.Frame):
        self.frame = ttk.LabelFrame(parent, text="파라미터")
        self.frame.pack(fill=tk.X, pady=5)

        # 콜백
        self.on_noise_measured: Optional[Callable[[float, str], None]] = None

        self._noise_config = NoiseConfig()
        self._create_widgets()
        self._create_noise_section()

    # ================================================================
    #  파라미터 섹션
    # ================================================================

    def _create_widgets(self):
        grid = ttk.Frame(self.frame)
        grid.pack(fill=tk.X, padx=5, pady=5)

        # MIG Threshold
        ttk.Label(grid, text="MIG 임계값:").grid(
            row=0, column=0, sticky=tk.W, pady=2)
        self.mig_var = tk.DoubleVar(value=30.0)
        self.mig_spin = ttk.Spinbox(
            grid, textvariable=self.mig_var,
            from_=10.0, to=100.0, increment=5.0, width=10)
        self.mig_spin.grid(row=0, column=1, pady=2)

        # Subset Size
        ttk.Label(grid, text="Subset 크기:").grid(
            row=1, column=0, sticky=tk.W, pady=2)
        self.subset_var = tk.IntVar(value=21)
        self.subset_spin = ttk.Spinbox(
            grid, textvariable=self.subset_var,
            values=[11, 13, 15, 17, 19, 21, 23, 25, 27, 29,
                    31, 33, 35, 37, 39, 41, 51, 61],
            width=10)
        self.subset_spin.grid(row=1, column=1, pady=2)

        # Spacing
        ttk.Label(grid, text="POI 간격:").grid(
            row=2, column=0, sticky=tk.W, pady=2)
        self.spacing_var = tk.IntVar(value=16)
        self.spacing_spin = ttk.Spinbox(
            grid, textvariable=self.spacing_var,
            from_=5, to=50, increment=1, width=10)
        self.spacing_spin.grid(row=2, column=1, pady=2)

    # ================================================================
    #  노이즈 추정 섹션
    # ================================================================

    def _create_noise_section(self):
        nf = ttk.LabelFrame(self.frame, text="노이즈 추정")
        nf.pack(fill=tk.X, padx=5, pady=(5, 10))

        # ── 방법 선택 ──
        method_frame = ttk.Frame(nf)
        method_frame.pack(fill=tk.X, padx=5, pady=(5, 2))

        ttk.Label(method_frame, text="방법:").pack(side=tk.LEFT)
        self.noise_method_var = tk.StringVar(value='auto')
        self.noise_method_combo = ttk.Combobox(
            method_frame,
            textvariable=self.noise_method_var,
            values=['auto', 'pair', 'manual', 'single'],
            state='readonly', width=12)
        self.noise_method_combo.pack(side=tk.LEFT, padx=(5, 0))
        self.noise_method_combo.bind(
            '<<ComboboxSelected>>', self._on_noise_method_changed)

        # 툴팁 스타일 설명
        self.noise_desc_var = tk.StringVar(
            value="폴더 로드 시 첫 2장으로 자동 측정")
        ttk.Label(nf, textvariable=self.noise_desc_var,
                  font=('Arial', 8), foreground='gray').pack(
            anchor=tk.W, padx=10)

        # ── pair 선택 프레임 ──
        self.pair_frame = ttk.Frame(nf)
        self.pair_frame.pack(fill=tk.X, padx=5, pady=2)

        self.btn_select_pair = ttk.Button(
            self.pair_frame, text="정지 이미지 2장 선택",
            command=self._select_pair_images)
        self.btn_select_pair.pack(fill=tk.X, pady=2)

        self.pair_label = ttk.Label(
            self.pair_frame, text="선택 안 됨",
            font=('Arial', 8), foreground='gray')
        self.pair_label.pack(anchor=tk.W)

        # ── manual 입력 프레임 ──
        self.manual_frame = ttk.Frame(nf)
        self.manual_frame.pack(fill=tk.X, padx=5, pady=2)

        manual_grid = ttk.Frame(self.manual_frame)
        manual_grid.pack(fill=tk.X)
        ttk.Label(manual_grid, text="D(η) =").pack(side=tk.LEFT)
        self.noise_manual_var = tk.DoubleVar(value=4.0)
        self.noise_manual_spin = ttk.Spinbox(
            manual_grid, textvariable=self.noise_manual_var,
            from_=0.1, to=1000.0, increment=0.5, width=8,
            format="%.1f")
        self.noise_manual_spin.pack(side=tk.LEFT, padx=5)
        ttk.Label(manual_grid, text="GL²").pack(side=tk.LEFT)

        ttk.Button(self.manual_frame, text="적용",
                   command=self._apply_manual_noise).pack(
            fill=tk.X, pady=2)

        # ── 측정 버튼 ──
        self.btn_measure = ttk.Button(
            nf, text="노이즈 측정 실행",
            command=self._request_noise_measurement)
        self.btn_measure.pack(fill=tk.X, padx=5, pady=2)

        # ── 결과 표시 ──
        self.noise_result_frame = ttk.Frame(nf)
        self.noise_result_frame.pack(fill=tk.X, padx=5, pady=(2, 5))

        self.noise_result_var = tk.StringVar(value="측정 전")
        self.noise_result_label = ttk.Label(
            self.noise_result_frame,
            textvariable=self.noise_result_var,
            font=('Consolas', 9))
        self.noise_result_label.pack(anchor=tk.W)

        self.noise_status_var = tk.StringVar(value="")
        self.noise_status_label = ttk.Label(
            self.noise_result_frame,
            textvariable=self.noise_status_var,
            font=('Arial', 8))
        self.noise_status_label.pack(anchor=tk.W)

        # 초기 UI 상태
        self._update_noise_ui_visibility()

    # ── UI 상태 전환 ──

    def _on_noise_method_changed(self, event=None):
        method = self.noise_method_var.get()
        descriptions = {
            'auto': "폴더 로드 시 첫 2장으로 자동 측정",
            'pair': "정지 이미지 2장을 직접 선택하여 측정",
            'manual': "카메라 스펙 기반으로 D(η)를 직접 입력",
            'single': "단일 이미지 로컬분산 추정 (정확도 낮음)",
        }
        self.noise_desc_var.set(descriptions.get(method, ""))
        self._update_noise_ui_visibility()

    def _update_noise_ui_visibility(self):
        method = self.noise_method_var.get()

        # pair 프레임
        if method == 'pair':
            self.pair_frame.pack(fill=tk.X, padx=5, pady=2)
            self.btn_select_pair.config(state='normal')
        else:
            self.pair_frame.pack_forget()

        # manual 프레임
        if method == 'manual':
            self.manual_frame.pack(fill=tk.X, padx=5, pady=2)
        else:
            self.manual_frame.pack_forget()

        # 측정 버튼 (manual은 "적용" 버튼이 별도)
        if method in ('auto', 'pair', 'single'):
            self.btn_measure.pack(fill=tk.X, padx=5, pady=2)
        else:
            self.btn_measure.pack_forget()

        # 결과 프레임은 항상 표시
        self.noise_result_frame.pack(fill=tk.X, padx=5, pady=(2, 5))

    # ── pair 이미지 선택 ──

    def _select_pair_images(self):
        paths = filedialog.askopenfilenames(
            title="정지 이미지 2장 선택 (동일 조건)",
            filetypes=[
                ("이미지", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("모든 파일", "*.*")
            ])
        if len(paths) >= 2:
            p1, p2 = paths[0], paths[1]
            self._noise_config.pair_paths = (p1, p2)
            name1 = Path(p1).name
            name2 = Path(p2).name
            self.pair_label.config(
                text=f"{name1}\n{name2}",
                foreground='black')
        elif len(paths) == 1:
            self.pair_label.config(
                text="2장을 선택해야 합니다",
                foreground='red')

    # ── manual 적용 ──

    def _apply_manual_noise(self):
        try:
            variance = float(self.noise_manual_var.get())
            if variance <= 0:
                raise ValueError
        except (ValueError, tk.TclError):
            self.noise_status_var.set("⚠ 양수를 입력하세요")
            self.noise_status_label.config(foreground='red')
            return

        self._noise_config.method = 'manual'
        self._noise_config.variance = variance
        self._noise_config.std = np.sqrt(variance)
        self._noise_config.estimation_done = True

        self._display_noise_result(variance, 'manual')

        if self.on_noise_measured:
            self.on_noise_measured(variance, 'manual')

    # ── 측정 요청 (컨트롤러가 처리) ──

    def _request_noise_measurement(self):
        """측정 버튼 클릭 → 컨트롤러에 위임"""
        if self.on_noise_measured:
            method = self.noise_method_var.get()
            self.on_noise_measured(None, method)

    # ── 결과 표시 ──

    def _display_noise_result(self, variance: float, method: str):
        std = np.sqrt(variance)
        self.noise_result_var.set(
            f"D(η) = {variance:.2f}    σ = {std:.2f} GL")

        method_names = {
            'pair': 'pair 차분법 ✓',
            'manual': '사용자 입력',
            'single_local_std': '단일이미지 (local_std)',
            'auto': '자동 (pair)',
        }
        display_method = method_names.get(method, method)
        self.noise_status_var.set(f"방법: {display_method}")

        if method == 'pair':
            self.noise_status_label.config(foreground='green')
        elif method == 'manual':
            self.noise_status_label.config(foreground='blue')
        else:
            self.noise_status_label.config(foreground='orange')

    def set_noise_result(self, variance: float, method: str):
        """외부(컨트롤러)에서 측정 결과를 표시할 때 호출"""
        self._noise_config.variance = variance
        self._noise_config.std = np.sqrt(variance)
        self._noise_config.method = method
        self._noise_config.estimation_done = True
        self._display_noise_result(variance, method)

    # ================================================================
    #  공개 인터페이스
    # ================================================================

    def get_parameters(self) -> Optional[Parameters]:
        try:
            return Parameters(
                mig_threshold=float(self.mig_var.get()),
                subset_size=int(self.subset_var.get()),
                spacing=int(self.spacing_var.get()))
        except ValueError:
            return None

    def set_parameters(self, params: Parameters):
        self.mig_var.set(params.mig_threshold)
        self.subset_var.set(params.subset_size)
        self.spacing_var.set(params.spacing)

    def reset_to_default(self):
        self.mig_var.set(30.0)
        self.subset_var.set(21)
        self.spacing_var.set(16)

    def get_noise_config(self) -> NoiseConfig:
        """현재 노이즈 설정 반환"""
        self._noise_config.method = self.noise_method_var.get()
        return self._noise_config

    def get_noise_variance(self) -> Optional[float]:
        """측정된 노이즈 분산 반환 (없으면 None)"""
        if self._noise_config.estimation_done:
            return self._noise_config.variance
        return None
