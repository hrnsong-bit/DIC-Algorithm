"""
IC-GN 최적화 결과 데이터 클래스
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict


# 실패 원인 코드 상수
ICGN_SUCCESS = 0
ICGN_FAIL_LOW_ZNCC = 1
ICGN_FAIL_DIVERGED = 2
ICGN_FAIL_OUT_OF_BOUNDS = 3
ICGN_FAIL_SINGULAR_HESSIAN = 4
ICGN_FAIL_FLAT_SUBSET = 5
ICGN_FAIL_MAX_DISPLACEMENT = 6
ICGN_FAIL_FLAT_TARGET = 7

FAILURE_REASON_NAMES = {
    ICGN_SUCCESS: 'success',
    ICGN_FAIL_LOW_ZNCC: 'low_zncc',
    ICGN_FAIL_DIVERGED: 'diverged',
    ICGN_FAIL_OUT_OF_BOUNDS: 'out_of_bounds',
    ICGN_FAIL_SINGULAR_HESSIAN: 'singular_hessian',
    ICGN_FAIL_FLAT_SUBSET: 'flat_subset',
    ICGN_FAIL_MAX_DISPLACEMENT: 'max_displacement',
    ICGN_FAIL_FLAT_TARGET: 'flat_target',
}


@dataclass
class ICGNResult:
    """IC-GN 최적화 결과"""

    # POI 좌표
    points_y: np.ndarray
    points_x: np.ndarray

    # 서브픽셀 변위
    disp_u: np.ndarray
    disp_v: np.ndarray

    # 변형 gradient (1차)
    disp_ux: np.ndarray
    disp_uy: np.ndarray
    disp_vx: np.ndarray
    disp_vy: np.ndarray

    # 2차 변형 gradient (Quadratic)
    disp_uxx: Optional[np.ndarray] = None
    disp_uxy: Optional[np.ndarray] = None
    disp_uyy: Optional[np.ndarray] = None
    disp_vxx: Optional[np.ndarray] = None
    disp_vxy: Optional[np.ndarray] = None
    disp_vyy: Optional[np.ndarray] = None

    # 품질 지표
    zncc_values: np.ndarray = None
    iterations: np.ndarray = None
    converged: np.ndarray = None
    valid_mask: np.ndarray = None

    # FFT-CC 통과 여부 (텍스처 없음 구분용)
    # True  = FFT-CC 통과 → IC-GN 실행됨
    # False = FFT-CC 실패 → 텍스처 없음 (홀/배경), IC-GN 스킵됨
    fft_valid_mask: Optional[np.ndarray] = None

    # 실패 원인 (POI별)
    failure_reason: np.ndarray = None

    # 메타데이터
    subset_size: int = 21
    max_iterations: int = 50
    convergence_threshold: float = 0.001
    processing_time: float = 0.0
    shape_function: str = 'affine'

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
    def n_texture_less(self) -> int:
        """FFT-CC 실패 POI 수 (텍스처 없음)"""
        if self.fft_valid_mask is None:
            return 0
        return int(np.sum(~self.fft_valid_mask))

    @property
    def n_ic_fail(self) -> int:
        """IC-GN 실패 POI 수 (불연속 후보)"""
        if self.fft_valid_mask is None:
            return int(np.sum(~self.valid_mask))
        return int(np.sum(self.fft_valid_mask & ~self.valid_mask))

    @property
    def ic_fail_mask(self) -> np.ndarray:
        """IC-GN 실패 POI 마스크 (불연속 후보)"""
        if self.fft_valid_mask is None:
            return ~self.valid_mask
        return self.fft_valid_mask & ~self.valid_mask

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

    @property
    def is_quadratic(self) -> bool:
        return self.shape_function == 'quadratic'

    @property
    def failure_summary(self) -> Dict[str, int]:
        """실패 원인별 POI 수 요약"""
        if self.failure_reason is None:
            return {}
        summary = {}
        for code, name in FAILURE_REASON_NAMES.items():
            count = int(np.sum(self.failure_reason == code))
            if count > 0:
                summary[name] = count
        return summary
