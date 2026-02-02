"""
초기 추정 결과 데이터 모델
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Any
import numpy as np


@dataclass
class MatchResult:
    """단일 POI 매칭 결과"""
    ref_y: int
    ref_x: int
    disp_u: int
    disp_v: int
    zncc: float
    valid: bool = True
    flag: str = ""
    
    @property
    def x(self) -> int:
        return self.ref_x
    
    @property
    def y(self) -> int:
        return self.ref_y
    
    @property
    def u(self) -> int:
        return self.disp_u
    
    @property
    def v(self) -> int:
        return self.disp_v
    
    @property
    def is_valid(self) -> bool:
        return self.valid
    
    @property
    def displacement(self) -> Tuple[int, int]:
        return (self.disp_u, self.disp_v)


@dataclass  
class FFTCCResult:
    """FFT-CC 전체 결과"""
    points_y: np.ndarray
    points_x: np.ndarray
    disp_u: np.ndarray
    disp_v: np.ndarray
    zncc_values: np.ndarray
    valid_mask: np.ndarray
    invalid_points: List[MatchResult] = field(default_factory=list)
    
    subset_size: int = 21
    search_range: int = 50
    spacing: int = 10
    processing_time: float = 0.0
    
    @property
    def n_points(self) -> int:
        return len(self.points_y)
    
    @property
    def n_valid(self) -> int:
        return int(np.sum(self.valid_mask))
    
    @property
    def n_invalid(self) -> int:
        return self.n_points - self.n_valid
    
    @property
    def valid_ratio(self) -> float:
        if self.n_points == 0:
            return 0.0
        return self.n_valid / self.n_points
    
    @property
    def mean_zncc(self) -> float:
        if self.n_valid == 0:
            return 0.0
        return float(np.mean(self.zncc_values[self.valid_mask]))
    
    @property
    def min_zncc(self) -> float:
        if self.n_valid == 0:
            return 0.0
        return float(np.min(self.zncc_values[self.valid_mask]))
    
    @property
    def points(self) -> List[MatchResult]:
        """모든 포인트를 MatchResult 리스트로 반환"""
        result = []
        for idx in range(self.n_points):
            result.append(MatchResult(
                ref_y=int(self.points_y[idx]),
                ref_x=int(self.points_x[idx]),
                disp_u=int(self.disp_u[idx]),
                disp_v=int(self.disp_v[idx]),
                zncc=float(self.zncc_values[idx]),
                valid=bool(self.valid_mask[idx])
            ))
        return result
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """메타데이터 반환"""
        return {
            'subset_size': self.subset_size,
            'search_range': self.search_range,
            'spacing': self.spacing,
            'processing_time': self.processing_time,
            'n_points': self.n_points,
            'n_valid': self.n_valid,
            'n_invalid': self.n_invalid
        }
    
    def get_displacement_field_2d(self, shape: Tuple[int, int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """2D 변위 필드 반환 (시각화용)"""
        if shape is None:
            y_max = int(np.max(self.points_y)) + 1 if len(self.points_y) > 0 else 0
            x_max = int(np.max(self.points_x)) + 1 if len(self.points_x) > 0 else 0
            shape = (y_max, x_max)
        
        u_field = np.full(shape, np.nan, dtype=np.float64)
        v_field = np.full(shape, np.nan, dtype=np.float64)
        
        for idx in range(self.n_points):
            if self.valid_mask[idx]:
                y, x = int(self.points_y[idx]), int(self.points_x[idx])
                if 0 <= y < shape[0] and 0 <= x < shape[1]:
                    u_field[y, x] = self.disp_u[idx]
                    v_field[y, x] = self.disp_v[idx]
        
        return u_field, v_field
