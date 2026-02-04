"""
IC-GN 최적화 결과 데이터 클래스
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ICGNResult:
    """IC-GN 최적화 결과"""
    
    # POI 좌표
    points_y: np.ndarray
    points_x: np.ndarray
    
    # 서브픽셀 변위
    disp_u: np.ndarray   # float64
    disp_v: np.ndarray   # float64
    
    # 변형 gradient (1차 shape function)
    disp_ux: np.ndarray  # ∂u/∂x
    disp_uy: np.ndarray  # ∂u/∂y
    disp_vx: np.ndarray  # ∂v/∂x
    disp_vy: np.ndarray  # ∂v/∂y
    
    # 품질 지표
    zncc_values: np.ndarray
    iterations: np.ndarray
    converged: np.ndarray
    valid_mask: np.ndarray
    
    # 메타데이터
    subset_size: int
    max_iterations: int
    convergence_threshold: float
    processing_time: float = 0.0
    
    @property
    def n_points(self) -> int:
        return len(self.points_y)
    
    @property
    def n_converged(self) -> int:
        return int(np.sum(self.converged))
    
    @property
    def n_valid(self) -> int:
        return int(np.sum(self.valid_mask))
    
    @property
    def convergence_rate(self) -> float:
        if self.n_points == 0:
            return 0.0
        return self.n_converged / self.n_points
    
    @property
    def mean_iterations(self) -> float:
        if self.n_converged == 0:
            return 0.0
        return float(np.mean(self.iterations[self.converged]))
    
    @property
    def mean_zncc(self) -> float:
        if self.n_valid == 0:
            return 0.0
        return float(np.mean(self.zncc_values[self.valid_mask]))
