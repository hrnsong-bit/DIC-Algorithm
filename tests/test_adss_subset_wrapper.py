"""
Tests for compute_adss_recalc wrapper behavior.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter, sobel, spline_filter

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speckle.core.optimization.adss_subset import compute_adss_recalc
from speckle.core.optimization.shape_function_numba import AFFINE, get_num_params


def _make_speckle_image(h=256, w=256, seed=42):
    rng = np.random.RandomState(seed)
    img = rng.rand(h, w).astype(np.float64)
    img = gaussian_filter(img, sigma=2.0)
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return img


def _compute_gradient(img):
    grad_x = sobel(img, axis=1).astype(np.float64) / 32.0
    grad_y = sobel(img, axis=0).astype(np.float64) / 32.0
    return grad_x, grad_y


def _make_shifted_image(ref, shift_x, shift_y):
    sx, sy = int(round(shift_x)), int(round(shift_y))
    deformed = np.zeros_like(ref)
    h, w = ref.shape
    src_r0 = max(0, -sy)
    src_r1 = min(h, h - sy)
    src_c0 = max(0, -sx)
    src_c1 = min(w, w - sx)
    dst_r0 = src_r0 + sy
    dst_r1 = src_r1 + sy
    dst_c0 = src_c0 + sx
    dst_c1 = src_c1 + sx
    deformed[dst_r0:dst_r1, dst_c0:dst_c1] = ref[src_r0:src_r1, src_c0:src_c1]
    return deformed


def _create_test_environment(n_bad_pois=4):
    """
    Build one synthetic ADSS test setup.
    """
    shift_x, shift_y = 3.0, 2.0
    subset_size = 21
    spacing = 5
    n_params = get_num_params(AFFINE)

    ref = _make_speckle_image(256, 256, seed=42)
    deformed = _make_shifted_image(ref, shift_x, shift_y)
    grad_x, grad_y = _compute_gradient(ref)
    coeffs = spline_filter(deformed, order=5).astype(np.float64)

    margin = subset_size + 10
    xs = np.arange(margin, 256 - margin, spacing, dtype=np.int64)
    ys = np.arange(margin, 256 - margin, spacing, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points_x = grid_x.ravel().copy()
    points_y = grid_y.ravel().copy()
    n_total = len(points_x)
    ny, nx = len(ys), len(xs)

    valid_mask = np.ones(n_total, dtype=np.bool_)
    zncc_values = np.full(n_total, 0.99, dtype=np.float64)
    parameters = np.zeros((n_total, n_params), dtype=np.float64)
    parameters[:, 0] = shift_x
    parameters[:, 3] = shift_y
    convergence_flags = np.ones(n_total, dtype=np.bool_)
    iteration_counts = np.full(n_total, 5, dtype=np.int32)
    failure_reasons = np.zeros(n_total, dtype=np.int32)

    center_row = ny // 2
    center_col = nx // 2
    bad_indices = []
    for dc in range(n_bad_pois):
        idx = center_row * nx + center_col + dc
        if idx < n_total:
            bad_indices.append(idx)

    for idx in bad_indices:
        valid_mask[idx] = False
        zncc_values[idx] = 0.3
        parameters[idx, :] = 0.0
        convergence_flags[idx] = False
        failure_reasons[idx] = 1

    return {
        "ref": ref,
        "grad_x": grad_x,
        "grad_y": grad_y,
        "coeffs": coeffs,
        "points_x": points_x,
        "points_y": points_y,
        "valid_mask": valid_mask,
        "zncc_values": zncc_values,
        "parameters": parameters,
        "convergence_flags": convergence_flags,
        "iteration_counts": iteration_counts,
        "failure_reasons": failure_reasons,
        "subset_size": subset_size,
        "shift_x": shift_x,
        "shift_y": shift_y,
        "bad_indices": np.array(bad_indices, dtype=np.int64),
    }


class TestComputeAdssRecalc:
    def _run(self, n_bad_pois):
        env = _create_test_environment(n_bad_pois=n_bad_pois)
        result = compute_adss_recalc(
            env["ref"],
            env["grad_x"],
            env["grad_y"],
            env["coeffs"],
            5,
            env["points_x"],
            env["points_y"],
            env["valid_mask"],
            env["zncc_values"],
            env["parameters"],
            env["convergence_flags"],
            env["iteration_counts"],
            env["failure_reasons"],
            env["subset_size"],
        )
        return env, result

    def test_no_bad_pois_skip(self):
        _, result = self._run(n_bad_pois=0)
        assert result.n_bad_original == 0
        assert result.n_parent_recovered == 0
        assert result.n_unrecoverable == 0
        assert result.n_sub_total == 0
        assert result.parent_indices.size == 0

    def test_basic_recovery(self):
        _, result = self._run(n_bad_pois=4)
        assert result.n_bad_original == 4
        assert result.n_parent_recovered > 0
        assert result.n_parent_recovered + result.n_unrecoverable == result.n_bad_original
        assert result.n_sub_total == len(result.parent_indices)

    def test_recovered_parents_are_bad_indices(self):
        env, result = self._run(n_bad_pois=4)
        recovered_parents = set(np.unique(result.parent_indices).tolist())
        expected_bad = set(env["bad_indices"].tolist())
        assert recovered_parents.issubset(expected_bad)
        assert len(recovered_parents) == result.n_parent_recovered

    def test_displacement_accuracy(self):
        env, result = self._run(n_bad_pois=4)
        for parent_idx in result.unique_parents:
            rep = result.get_representative(int(parent_idx))
            assert rep is not None
            assert result.parameters[rep, 0] == pytest.approx(env["shift_x"], abs=0.1)
            assert result.parameters[rep, 3] == pytest.approx(env["shift_y"], abs=0.1)

    def test_return_structure(self):
        _, result = self._run(n_bad_pois=2)
        assert isinstance(result.parent_indices, np.ndarray)
        assert isinstance(result.quarter_types, np.ndarray)
        assert isinstance(result.parameters, np.ndarray)
        assert isinstance(result.zncc_values, np.ndarray)
        assert isinstance(result.iterations, np.ndarray)
        assert isinstance(result.elapsed_time, float)
        assert result.candidate_zncc is not None
        assert result.candidate_zncc.shape == (result.n_bad_original, 8)
        assert result.parameters.shape[0] == result.n_sub_total

    def test_quarter_types_valid(self):
        _, result = self._run(n_bad_pois=4)
        assert np.all((result.quarter_types >= 1) & (result.quarter_types <= 8))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
