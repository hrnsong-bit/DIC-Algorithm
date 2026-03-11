"""
Algorithm integrity tests for ICGN/ADSS pipeline.

Scope:
- Excludes variable-subset algorithm by design.
- Focuses on numerical consistency and invariant properties.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from speckle.core.initial_guess.results import FFTCCResult
from speckle.core.optimization.icgn import compute_icgn, prepare_ref_cache
from speckle.core.optimization.results import ICGN_FAIL_LOW_ZNCC


def _make_speckle_image(height=220, width=220, seed=0):
    rng = np.random.RandomState(seed)
    img = rng.rand(height, width).astype(np.float64)
    img = cv2.GaussianBlur(img, (11, 11), 2.0)
    return img * 255.0


def _make_grid_points(height, width, subset_size=21, spacing=16):
    half = subset_size // 2
    margin = half + 5
    xs = np.arange(margin, width - margin, spacing, dtype=np.int64)
    ys = np.arange(margin, height - margin, spacing, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_x.ravel(), grid_y.ravel()


def _build_initial_guess(points_x, points_y, disp_u, disp_v, valid_mask=None):
    n = len(points_x)
    if np.isscalar(disp_u):
        disp_u = np.full(n, float(disp_u), dtype=np.float64)
    if np.isscalar(disp_v):
        disp_v = np.full(n, float(disp_v), dtype=np.float64)
    if valid_mask is None:
        valid_mask = np.ones(n, dtype=np.bool_)

    return FFTCCResult(
        points_y=np.asarray(points_y, dtype=np.int64),
        points_x=np.asarray(points_x, dtype=np.int64),
        disp_u=np.asarray(disp_u, dtype=np.float64),
        disp_v=np.asarray(disp_v, dtype=np.float64),
        zncc_values=np.ones(n, dtype=np.float64),
        valid_mask=np.asarray(valid_mask, dtype=np.bool_),
    )


class TestICGNIntegrity:
    def test_translation_recovery_and_zero_strain(self):
        ref = _make_speckle_image(220, 220, seed=1)
        shift_x, shift_y = 0.37, -0.28
        transform = np.float64([[1, 0, shift_x], [0, 1, shift_y]])
        deformed = cv2.warpAffine(
            ref,
            transform,
            (ref.shape[1], ref.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )

        points_x, points_y = _make_grid_points(
            ref.shape[0], ref.shape[1], subset_size=21, spacing=16
        )
        guess = _build_initial_guess(points_x, points_y, shift_x, shift_y)

        result = compute_icgn(
            ref,
            deformed,
            guess,
            subset_size=21,
            zncc_threshold=0.95,
            enable_adss_subset=False,
            enable_variable_subset=False,
        )

        assert result.n_valid == result.n_points
        assert result.mean_zncc > 0.999
        assert np.mean(np.abs(result.disp_u - shift_x)) < 0.06
        assert np.mean(np.abs(result.disp_v - shift_y)) < 0.06
        assert np.max(np.abs(result.disp_ux)) < 0.01
        assert np.max(np.abs(result.disp_uy)) < 0.01
        assert np.max(np.abs(result.disp_vx)) < 0.01
        assert np.max(np.abs(result.disp_vy)) < 0.01

    def test_photometric_invariance_scale_and_offset(self):
        ref = _make_speckle_image(220, 220, seed=2)
        deformed = ref * 1.35 + 25.0

        points_x, points_y = _make_grid_points(
            ref.shape[0], ref.shape[1], subset_size=21, spacing=16
        )
        guess = _build_initial_guess(points_x, points_y, 0.0, 0.0)

        result = compute_icgn(
            ref,
            deformed,
            guess,
            subset_size=21,
            zncc_threshold=0.99,
            enable_adss_subset=False,
            enable_variable_subset=False,
        )

        assert result.n_valid == result.n_points
        assert result.mean_zncc > 0.999999
        assert np.max(np.abs(result.disp_u)) < 1e-12
        assert np.max(np.abs(result.disp_v)) < 1e-12

    def test_fft_valid_mask_propagation(self):
        ref = _make_speckle_image(220, 220, seed=3)
        shift_x, shift_y = 0.37, -0.28
        transform = np.float64([[1, 0, shift_x], [0, 1, shift_y]])
        deformed = cv2.warpAffine(
            ref,
            transform,
            (ref.shape[1], ref.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )

        points_x, points_y = _make_grid_points(
            ref.shape[0], ref.shape[1], subset_size=21, spacing=16
        )
        valid_mask = np.ones(len(points_x), dtype=np.bool_)
        valid_mask[::7] = False
        guess = _build_initial_guess(
            points_x, points_y, shift_x, shift_y, valid_mask=valid_mask
        )

        result = compute_icgn(
            ref,
            deformed,
            guess,
            subset_size=21,
            zncc_threshold=0.95,
            enable_adss_subset=False,
            enable_variable_subset=False,
        )

        assert np.array_equal(result.fft_valid_mask, valid_mask)
        assert np.sum(result.valid_mask[~valid_mask]) == 0
        assert np.all(result.failure_reason[~valid_mask] == ICGN_FAIL_LOW_ZNCC)
        assert result.n_valid == int(np.sum(valid_mask))

    def test_ref_cache_path_matches_fresh_path(self):
        ref = _make_speckle_image(220, 220, seed=4)
        shift_x, shift_y = 0.37, -0.28
        transform = np.float64([[1, 0, shift_x], [0, 1, shift_y]])
        deformed = cv2.warpAffine(
            ref,
            transform,
            (ref.shape[1], ref.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )
        points_x, points_y = _make_grid_points(
            ref.shape[0], ref.shape[1], subset_size=21, spacing=16
        )
        guess = _build_initial_guess(points_x, points_y, shift_x, shift_y)

        fresh = compute_icgn(
            ref,
            deformed,
            guess,
            subset_size=21,
            zncc_threshold=0.95,
            enable_adss_subset=False,
            enable_variable_subset=False,
        )

        cache = prepare_ref_cache(
            ref, subset_size=21, interpolation_order=5, shape_function="affine"
        )
        cached = compute_icgn(
            ref,
            deformed,
            guess,
            subset_size=21,
            zncc_threshold=0.95,
            ref_cache=cache,
            enable_adss_subset=False,
            enable_variable_subset=False,
        )

        assert np.array_equal(fresh.valid_mask, cached.valid_mask)
        assert np.array_equal(fresh.failure_reason, cached.failure_reason)
        np.testing.assert_allclose(fresh.disp_u, cached.disp_u, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(fresh.disp_v, cached.disp_v, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            fresh.zncc_values, cached.zncc_values, atol=0.0, rtol=0.0
        )

    def test_ref_cache_mismatch_fallback_matches_fresh(self):
        ref = _make_speckle_image(220, 220, seed=44)
        shift_x, shift_y = 0.37, -0.28
        transform = np.float64([[1, 0, shift_x], [0, 1, shift_y]])
        deformed = cv2.warpAffine(
            ref,
            transform,
            (ref.shape[1], ref.shape[0]),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REFLECT,
        )

        points_x, points_y = _make_grid_points(
            ref.shape[0], ref.shape[1], subset_size=21, spacing=16
        )
        guess = _build_initial_guess(points_x, points_y, shift_x, shift_y)

        fresh = compute_icgn(
            ref,
            deformed,
            guess,
            subset_size=21,
            zncc_threshold=0.95,
            enable_adss_subset=False,
            enable_variable_subset=False,
        )

        # subset_size가 다른 캐시를 고의로 전달 -> 내부에서 fresh fallback 되어야 함
        wrong_cache = prepare_ref_cache(
            ref, subset_size=23, interpolation_order=5, shape_function="affine"
        )
        fallback = compute_icgn(
            ref,
            deformed,
            guess,
            subset_size=21,
            zncc_threshold=0.95,
            ref_cache=wrong_cache,
            enable_adss_subset=False,
            enable_variable_subset=False,
        )

        assert np.array_equal(fresh.valid_mask, fallback.valid_mask)
        assert np.array_equal(fresh.failure_reason, fallback.failure_reason)
        np.testing.assert_allclose(fresh.disp_u, fallback.disp_u, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(fresh.disp_v, fallback.disp_v, atol=0.0, rtol=0.0)
        np.testing.assert_allclose(
            fresh.zncc_values, fallback.zncc_values, atol=0.0, rtol=0.0
        )


class TestADSSIntegrity:
    def test_adss_recovery_metadata_on_crack_like_case(self):
        ref = _make_speckle_image(280, 280, seed=5)
        crack_y = 140
        shift = 3

        deformed = ref.copy()
        deformed[crack_y + shift :, :] = ref[crack_y:-shift, :]
        deformed[crack_y : crack_y + shift, :] = 0.0

        points_x, points_y = _make_grid_points(
            ref.shape[0], ref.shape[1], subset_size=21, spacing=14
        )
        init_u = np.zeros(len(points_x), dtype=np.float64)
        init_v = np.where(points_y > crack_y, float(shift), 0.0)
        guess = _build_initial_guess(points_x, points_y, init_u, init_v)

        result = compute_icgn(
            ref,
            deformed,
            guess,
            subset_size=21,
            zncc_threshold=0.9,
            enable_adss_subset=True,
            enable_variable_subset=False,
        )

        adss = result.adss_result
        assert adss is not None
        assert adss.n_bad_original > 0
        assert adss.n_parent_recovered > 0
        assert adss.n_parent_recovered <= adss.n_bad_original
        assert adss.n_parent_recovered + adss.n_unrecoverable == adss.n_bad_original
        assert adss.n_sub_total == len(adss.parent_indices)
        assert adss.candidate_zncc is not None
        assert adss.candidate_zncc.shape == (adss.n_bad_original, 8)

        if adss.n_sub_total > 0:
            assert np.all((adss.quarter_types >= 1) & (adss.quarter_types <= 8))
            assert np.all(adss.zncc_values >= 0.9)
            assert np.mean(np.abs(adss.get_disp_v() - shift)) < 0.2
            recovered_parents = adss.unique_parents.astype(np.int64)
            assert np.all(result.valid_mask[recovered_parents])
            assert np.all(result.failure_reason[recovered_parents] == 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
