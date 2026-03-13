"""
Result models for the IC-GN and ADSS optimization pipeline.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


# IC-GN failure codes
ICGN_SUCCESS = 0
ICGN_FAIL_LOW_ZNCC = 1
ICGN_FAIL_DIVERGED = 2
ICGN_FAIL_OUT_OF_BOUNDS = 3
ICGN_FAIL_SINGULAR_HESSIAN = 4
ICGN_FAIL_FLAT_SUBSET = 5
ICGN_FAIL_MAX_DISPLACEMENT = 6
ICGN_FAIL_FLAT_TARGET = 7

FAILURE_REASON_NAMES = {
    ICGN_SUCCESS: "success",
    ICGN_FAIL_LOW_ZNCC: "low_zncc",
    ICGN_FAIL_DIVERGED: "diverged",
    ICGN_FAIL_OUT_OF_BOUNDS: "out_of_bounds",
    ICGN_FAIL_SINGULAR_HESSIAN: "singular_hessian",
    ICGN_FAIL_FLAT_SUBSET: "flat_subset",
    ICGN_FAIL_MAX_DISPLACEMENT: "max_displacement",
    ICGN_FAIL_FLAT_TARGET: "flat_target",
}

# ADSS quarter identifiers
ADSS_Q1 = 1  # Upper triangle
ADSS_Q2 = 2  # Lower triangle
ADSS_Q3 = 3  # Left triangle
ADSS_Q4 = 4  # Right triangle
ADSS_Q5 = 5  # Upper-left rectangle
ADSS_Q6 = 6  # Upper-right rectangle
ADSS_Q7 = 7  # Lower-left rectangle
ADSS_Q8 = 8  # Lower-right rectangle


@dataclass
class ADSSResult:
    """
    Quarter-resolved ADSS recovery results.

    Each recovered sub-POI is stored as one flat row. The parent POI can have
    multiple recovered quarters, and `recovery_passes` keeps whether that
    quarter was solved during pass 1 or rescued during pass 2.
    """

    parent_indices: np.ndarray
    quarter_types: np.ndarray
    points_x: np.ndarray
    points_y: np.ndarray
    parameters: np.ndarray
    zncc_values: np.ndarray
    iterations: np.ndarray
    recovery_passes: Optional[np.ndarray] = None

    xsi_mins: Optional[np.ndarray] = None
    xsi_maxs: Optional[np.ndarray] = None
    eta_mins: Optional[np.ndarray] = None
    eta_maxs: Optional[np.ndarray] = None

    candidate_zncc: Optional[np.ndarray] = None

    n_bad_original: int = 0
    n_sub_total: int = 0
    n_parent_recovered: int = 0
    n_unrecoverable: int = 0
    elapsed_time: float = 0.0

    @property
    def n_sub(self) -> int:
        return len(self.parent_indices)

    def get_sub_pois_for_parent(self, parent_idx: int) -> np.ndarray:
        return np.where(self.parent_indices == parent_idx)[0]

    def get_representative(self, parent_idx: int) -> Optional[int]:
        sub_indices = self.get_sub_pois_for_parent(parent_idx)
        if len(sub_indices) == 0:
            return None
        best = sub_indices[np.argmax(self.zncc_values[sub_indices])]
        return int(best)

    def get_disp_u(self) -> np.ndarray:
        return self.parameters[:, 0]

    def get_disp_v(self) -> np.ndarray:
        if self.parameters.shape[1] <= 6:
            return self.parameters[:, 3]
        return self.parameters[:, 6]

    def get_recovery_passes(self) -> np.ndarray:
        if self.recovery_passes is None or len(self.recovery_passes) != self.n_sub:
            return np.ones(self.n_sub, dtype=np.int32)
        return np.asarray(self.recovery_passes, dtype=np.int32)

    @property
    def unique_parents(self) -> np.ndarray:
        return np.unique(self.parent_indices)

    @property
    def summary(self) -> Dict[str, int]:
        return {
            "n_bad_original": self.n_bad_original,
            "n_sub_total": self.n_sub_total,
            "n_parent_recovered": self.n_parent_recovered,
            "n_unrecoverable": self.n_unrecoverable,
        }

    @property
    def is_triangle_quarter(self) -> np.ndarray:
        return self.quarter_types <= 4


@dataclass
class ICGNResult:
    """IC-GN optimization result on the parent POI grid."""

    points_y: np.ndarray
    points_x: np.ndarray

    disp_u: np.ndarray
    disp_v: np.ndarray

    disp_ux: np.ndarray
    disp_uy: np.ndarray
    disp_vx: np.ndarray
    disp_vy: np.ndarray

    disp_uxx: Optional[np.ndarray] = None
    disp_uxy: Optional[np.ndarray] = None
    disp_uyy: Optional[np.ndarray] = None
    disp_vxx: Optional[np.ndarray] = None
    disp_vxy: Optional[np.ndarray] = None
    disp_vyy: Optional[np.ndarray] = None

    zncc_values: np.ndarray = None
    iterations: np.ndarray = None
    converged: np.ndarray = None
    valid_mask: np.ndarray = None

    fft_valid_mask: Optional[np.ndarray] = None
    failure_reason: np.ndarray = None
    adss_result: Optional[ADSSResult] = None

    subset_size: int = 21
    max_iterations: int = 50
    convergence_threshold: float = 0.001
    processing_time: float = 0.0
    shape_function: str = "affine"

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
        return self.shape_function == "quadratic"

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
