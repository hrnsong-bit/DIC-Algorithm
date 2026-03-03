"""
IC-GN 최적화 결과 데이터 클래스
"""

import numpy as np
from dataclasses import dataclass, field
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
    ICGN_SUCCESS:               'success',
    ICGN_FAIL_LOW_ZNCC:         'low_zncc',
    ICGN_FAIL_DIVERGED:         'diverged',
    ICGN_FAIL_OUT_OF_BOUNDS:    'out_of_bounds',
    ICGN_FAIL_SINGULAR_HESSIAN: 'singular_hessian',
    ICGN_FAIL_FLAT_SUBSET:      'flat_subset',
    ICGN_FAIL_MAX_DISPLACEMENT: 'max_displacement',
    ICGN_FAIL_FLAT_TARGET:      'flat_target',
}

# ADSS quarter-subset 상수
ADSS_Q5 = 5  # Upper-left
ADSS_Q6 = 6  # Upper-right
ADSS_Q7 = 7  # Lower-left
ADSS_Q8 = 8  # Lower-right


@dataclass
class ADSSResult:
    """
    ADSS-DIC 사분면 복수 채택 결과.

    하나의 불량 POI에서 threshold를 넘는 사분면이 k개(1~4)면
    k개의 sub-POI가 생성된다. 모든 sub-POI를 flat 배열로 저장.

    v1에서는 대표값(최대 ZNCC)을 ICGNResult에 반영하여
    기존 파이프라인과 100% 호환을 유지한다.
    v2에서 균열 양측 분리 strain 계산에 이 데이터를 직접 사용.
    """

    # --- sub-POI 배열 (n_sub 길이) ---
    parent_indices: np.ndarray      # (n_sub,) int64  — 원래 불량 POI의 flat index
    quarter_types: np.ndarray       # (n_sub,) int32  — Q5=5, Q6=6, Q7=7, Q8=8
    points_x: np.ndarray            # (n_sub,) int64  — 중심 좌표 (부모와 동일)
    points_y: np.ndarray            # (n_sub,) int64
    parameters: np.ndarray          # (n_sub, n_params) float64
    zncc_values: np.ndarray         # (n_sub,) float64
    iterations: np.ndarray          # (n_sub,) int32

    # --- 사분면 영역 정보 ---
    xsi_mins: np.ndarray            # (n_sub,) int32
    xsi_maxs: np.ndarray            # (n_sub,) int32
    eta_mins: np.ndarray            # (n_sub,) int32
    eta_maxs: np.ndarray            # (n_sub,) int32

    # --- 1-warp 평가 ZNCC (디버그/분석용) ---
    # (n_bad_original, 4) — 각 불량 POI의 Q5~Q8 1-warp ZNCC
    candidate_zncc: Optional[np.ndarray] = None

    # --- 통계 ---
    n_bad_original: int = 0
    n_sub_total: int = 0
    n_parent_recovered: int = 0     # 1개 이상 사분면 복원된 부모 POI 수
    n_unrecoverable: int = 0        # 사분면이 하나도 안 된 부모 POI 수
    elapsed_time: float = 0.0

    @property
    def n_sub(self) -> int:
        return len(self.parent_indices)

    def get_sub_pois_for_parent(self, parent_idx: int) -> np.ndarray:
        """특정 부모 POI의 모든 sub-POI 인덱스 반환"""
        return np.where(self.parent_indices == parent_idx)[0]

    def get_representative(self, parent_idx: int) -> Optional[int]:
        """특정 부모 POI의 대표 sub-POI (최대 ZNCC) 인덱스 반환"""
        sub_indices = self.get_sub_pois_for_parent(parent_idx)
        if len(sub_indices) == 0:
            return None
        best = sub_indices[np.argmax(self.zncc_values[sub_indices])]
        return int(best)

    def get_disp_u(self) -> np.ndarray:
        """sub-POI별 u 변위"""
        return self.parameters[:, 0]

    def get_disp_v(self) -> np.ndarray:
        """sub-POI별 v 변위 (affine: idx=3, quad: idx=6)"""
        if self.parameters.shape[1] <= 6:
            return self.parameters[:, 3]
        else:
            return self.parameters[:, 6]

    @property
    def unique_parents(self) -> np.ndarray:
        """복원된 고유 부모 POI 인덱스"""
        return np.unique(self.parent_indices)

    @property
    def summary(self) -> Dict[str, int]:
        """통계 요약"""
        return {
            'n_bad_original': self.n_bad_original,
            'n_sub_total': self.n_sub_total,
            'n_parent_recovered': self.n_parent_recovered,
            'n_unrecoverable': self.n_unrecoverable,
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
    fft_valid_mask: Optional[np.ndarray] = None

    # 실패 원인 (POI별)
    failure_reason: np.ndarray = None

    # ADSS-DIC 결과 (None이면 ADSS 미사용 또는 불량 POI 없음)
    adss_result: Optional[ADSSResult] = None

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
        if self.fft_valid_mask is None:
            return 0
        return int(np.sum(~self.fft_valid_mask))

    @property
    def n_ic_fail(self) -> int:
        if self.fft_valid_mask is None:
            return int(np.sum(~self.valid_mask))
        return int(np.sum(self.fft_valid_mask & ~self.valid_mask))

    @property
    def ic_fail_mask(self) -> np.ndarray:
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
        if self.failure_reason is None:
            return {}
        summary = {}
        for code, name in FAILURE_REASON_NAMES.items():
            count = int(np.sum(self.failure_reason == code))
            if count > 0:
                summary[name] = count
        return summary
