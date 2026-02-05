"""파라미터 패널"""

import tkinter as tk
from tkinter import ttk
from typing import Optional
from dataclasses import dataclass


@dataclass
class Parameters:
    """파라미터 값"""
    mig_threshold: float
    sssig_threshold: float
    subset_size: int
    spacing: int


class ParamPanel:
    """파라미터 설정 패널"""
    
    def __init__(self, parent: tk.Frame):
        self.frame = ttk.LabelFrame(parent, text="파라미터")
        self.frame.pack(fill=tk.X, pady=5)
        
        self._create_widgets()
    
    def _create_widgets(self):
        """위젯 생성"""
        grid = ttk.Frame(self.frame)
        grid.pack(fill=tk.X, padx=5, pady=5)
        
        # MIG Threshold
        ttk.Label(grid, text="MIG 임계값:").grid(row=0, column=0, sticky=tk.W, pady=2)
        self.mig_var = tk.DoubleVar(value=30.0)
        self.mig_spin = ttk.Spinbox(
            grid, textvariable=self.mig_var, 
            from_=10.0, to=100.0, increment=5.0,
            width=10
        )
        self.mig_spin.grid(row=0, column=1, pady=2)
        
        # Subset Size (홀수만)
        ttk.Label(grid, text="Subset 크기:").grid(row=1, column=0, sticky=tk.W, pady=2)
        self.subset_var = tk.IntVar(value=21)
        self.subset_spin = ttk.Spinbox(
            grid, textvariable=self.subset_var,
            values=[11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 51, 61],
            width=10
        )
        self.subset_spin.grid(row=1, column=1, pady=2)
        
        # Spacing
        ttk.Label(grid, text="POI 간격:").grid(row=2, column=0, sticky=tk.W, pady=2)
        self.spacing_var = tk.IntVar(value=16)
        self.spacing_spin = ttk.Spinbox(
            grid, textvariable=self.spacing_var,
            from_=5, to=50, increment=1,
            width=10
        )
        self.spacing_spin.grid(row=2, column=1, pady=2)
        
        # SSSIG Threshold (고급 설정으로 숨김 가능)
        ttk.Label(grid, text="SSSIG 임계값:").grid(row=3, column=0, sticky=tk.W, pady=2)
        self.sssig_var = tk.StringVar(value="1e5")
        self.sssig_entry = ttk.Entry(grid, textvariable=self.sssig_var, width=12)
        self.sssig_entry.grid(row=3, column=1, pady=2)
        ttk.Label(grid, text="(자동)", font=('Arial', 8), foreground='gray').grid(row=3, column=2, sticky=tk.W, padx=5)
        
        # 기본값 버튼
        ttk.Button(grid, text="기본값", command=self.reset_to_default, width=8).grid(
            row=4, column=0, columnspan=2, pady=(10, 5)
        )
    
    def get_parameters(self) -> Optional[Parameters]:
        """파라미터 값 반환, 실패시 None"""
        try:
            return Parameters(
                mig_threshold=float(self.mig_var.get()),
                sssig_threshold=float(self.sssig_var.get()),
                subset_size=int(self.subset_var.get()),
                spacing=int(self.spacing_var.get())
            )
        except ValueError:
            return None
    
    def set_parameters(self, params: Parameters):
        """파라미터 값 설정"""
        self.mig_var.set(params.mig_threshold)
        self.sssig_var.set(str(params.sssig_threshold))
        self.subset_var.set(params.subset_size)
        self.spacing_var.set(params.spacing)
    
    def reset_to_default(self):
        """기본값으로 초기화"""
        self.mig_var.set(30.0)
        self.sssig_var.set("1e5")
        self.subset_var.set(21)
        self.spacing_var.set(16)
