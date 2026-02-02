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
        
        # MIG
        ttk.Label(grid, text="MIG:").grid(row=0, column=0, sticky=tk.W)
        self.mig_var = tk.StringVar(value="30.0")
        ttk.Entry(grid, textvariable=self.mig_var, width=10).grid(row=0, column=1)
        
        # SSSIG
        ttk.Label(grid, text="SSSIG:").grid(row=0, column=2, sticky=tk.W, padx=(10, 0))
        self.sssig_var = tk.StringVar(value="1e5")
        ttk.Entry(grid, textvariable=self.sssig_var, width=10).grid(row=0, column=3)
        
        # Subset
        ttk.Label(grid, text="Subset:").grid(row=1, column=0, sticky=tk.W)
        self.subset_var = tk.StringVar(value="21")
        ttk.Entry(grid, textvariable=self.subset_var, width=10).grid(row=1, column=1)
        
        # Spacing
        ttk.Label(grid, text="Spacing:").grid(row=1, column=2, sticky=tk.W, padx=(10, 0))
        self.spacing_var = tk.StringVar(value="10")
        ttk.Entry(grid, textvariable=self.spacing_var, width=10).grid(row=1, column=3)
    
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
        self.mig_var.set(str(params.mig_threshold))
        self.sssig_var.set(str(params.sssig_threshold))
        self.subset_var.set(str(params.subset_size))
        self.spacing_var.set(str(params.spacing))
    
    def reset_to_default(self):
        """기본값으로 초기화"""
        self.mig_var.set("30.0")
        self.sssig_var.set("1e5")
        self.subset_var.set("21")
        self.spacing_var.set("10")
