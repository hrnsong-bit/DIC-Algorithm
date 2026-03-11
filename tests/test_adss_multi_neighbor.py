"""
Tests for ADSS multi-neighbor data construction and usage.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from speckle.core.optimization.adss_subset import _build_neighbor_info_adss_v2
from speckle.core.optimization.adss_subset_numba import (
    Q5,
    Q6,
    Q7,
    Q8,
    predict_initial_params,
    process_poi_adss_multi,
)
from speckle.core.optimization.shape_function_numba import AFFINE


def _make_grid(ny=5, nx=5, spacing=10, offset=50):
    points_x = []
    points_y = []
    for iy in range(ny):
        for ix in range(nx):
            points_x.append(ix * spacing + offset)
            points_y.append(iy * spacing + offset)
    return np.array(points_x, dtype=np.int64), np.array(points_y, dtype=np.int64), ny, nx


class TestNeighborInfoShape:
    def test_output_shape_is_8x3(self):
        points_x, points_y, ny, nx = _make_grid()
        n_poi = len(points_x)
        n_params = 6

        valid_mask = np.ones(n_poi, dtype=np.bool_)
        valid_mask[12] = False
        zncc_values = np.full(n_poi, 0.95)
        zncc_values[12] = 0.5
        parameters = np.random.rand(n_poi, n_params)
        bad_indices = np.array([12], dtype=np.int64)

        nv, np_, nx_, ny_ = _build_neighbor_info_adss_v2(
            bad_indices,
            points_x,
            points_y,
            valid_mask,
            zncc_values,
            parameters,
            0.9,
            ny,
            nx,
            n_params,
        )

        assert nv.shape == (1, 8, 3)
        assert np_.shape == (1, 8, 3, n_params)
        assert nx_.shape == (1, 8, 3)
        assert ny_.shape == (1, 8, 3)

    def test_center_poi_has_all_24_neighbors(self):
        points_x, points_y, ny, nx = _make_grid()
        n_poi = len(points_x)
        n_params = 6

        valid_mask = np.ones(n_poi, dtype=np.bool_)
        valid_mask[12] = False
        zncc_values = np.full(n_poi, 0.95)
        zncc_values[12] = 0.5
        parameters = np.random.rand(n_poi, n_params)
        bad_indices = np.array([12], dtype=np.int64)

        nv, _, _, _ = _build_neighbor_info_adss_v2(
            bad_indices,
            points_x,
            points_y,
            valid_mask,
            zncc_values,
            parameters,
            0.9,
            ny,
            nx,
            n_params,
        )

        assert int(nv[0].sum()) == 24

    def test_corner_poi_neighbor_counts(self):
        points_x, points_y, ny, nx = _make_grid()
        n_poi = len(points_x)
        n_params = 6

        valid_mask = np.ones(n_poi, dtype=np.bool_)
        valid_mask[0] = False
        zncc_values = np.full(n_poi, 0.95)
        zncc_values[0] = 0.5
        parameters = np.random.rand(n_poi, n_params)
        bad_indices = np.array([0], dtype=np.int64)

        nv, _, _, _ = _build_neighbor_info_adss_v2(
            bad_indices,
            points_x,
            points_y,
            valid_mask,
            zncc_values,
            parameters,
            0.9,
            ny,
            nx,
            n_params,
        )

        expected_counts = np.array([0, 2, 0, 2, 0, 1, 1, 3], dtype=np.int64)
        actual_counts = nv[0].sum(axis=1).astype(np.int64)
        assert np.array_equal(actual_counts, expected_counts), (
            f"Expected {expected_counts.tolist()}, got {actual_counts.tolist()}"
        )


class TestDiagonalNeighborFallback:
    def test_diagonal_bad_still_has_candidates(self):
        points_x, points_y, ny, nx = _make_grid()
        n_poi = len(points_x)
        n_params = 6

        valid_mask = np.ones(n_poi, dtype=np.bool_)
        zncc_values = np.full(n_poi, 0.95)
        parameters = np.random.rand(n_poi, n_params)

        center = 12
        valid_mask[center] = False
        zncc_values[center] = 0.5

        for iy, ix in [(1, 1), (1, 3), (3, 1), (3, 3)]:
            idx = iy * nx + ix
            valid_mask[idx] = False
            zncc_values[idx] = 0.3

        bad_indices = np.array([center], dtype=np.int64)
        nv, _, _, _ = _build_neighbor_info_adss_v2(
            bad_indices,
            points_x,
            points_y,
            valid_mask,
            zncc_values,
            parameters,
            0.9,
            ny,
            nx,
            n_params,
        )

        for q_idx in range(Q5 - 1, Q8):
            assert not bool(nv[0, q_idx, 0])
            assert bool(nv[0, q_idx, 1]) or bool(nv[0, q_idx, 2])

        assert np.any(nv[0].reshape(-1))

    def test_old_1to1_would_have_no_candidates(self):
        _, _, ny, nx = _make_grid()
        valid_mask = np.ones(ny * nx, dtype=np.bool_)
        zncc_values = np.full(ny * nx, 0.95)

        center = 12
        valid_mask[center] = False
        zncc_values[center] = 0.5

        for iy, ix in [(1, 1), (1, 3), (3, 1), (3, 3)]:
            idx = iy * nx + ix
            valid_mask[idx] = False
            zncc_values[idx] = 0.3

        old_neighbor_dirs = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        iy_center, ix_center = 2, 2
        old_has_candidate = False
        for dy, dx in old_neighbor_dirs:
            nf = (iy_center + dy) * nx + (ix_center + dx)
            if valid_mask[nf] and zncc_values[nf] >= 0.9:
                old_has_candidate = True
                break

        assert not old_has_candidate


class TestBestNeighborSelection:
    def test_predict_uses_different_params_per_neighbor(self):
        n_params = 6
        target_x, target_y = 100.0, 100.0
        neighbors = [
            (90.0, 90.0, np.array([2.0, 0.01, 0.02, 1.0, 0.01, 0.01])),
            (100.0, 90.0, np.array([3.0, 0.03, 0.01, 2.0, 0.02, 0.01])),
            (90.0, 100.0, np.array([1.5, 0.02, 0.03, 0.5, 0.01, 0.02])),
        ]

        results = []
        for nx_i, ny_i, params_i in neighbors:
            out = np.zeros(n_params, dtype=np.float64)
            predict_initial_params(nx_i, ny_i, params_i, target_x, target_y, AFFINE, out)
            results.append(out.copy())

        assert not np.allclose(results[0], results[1])
        assert not np.allclose(results[0], results[2])
        assert not np.allclose(results[1], results[2])

        for i, (nx_i, ny_i, params_i) in enumerate(neighbors):
            dx = target_x - nx_i
            dy = target_y - ny_i
            expected_u = params_i[0] + params_i[1] * dx + params_i[2] * dy
            assert abs(results[i][0] - expected_u) < 1e-10


class TestProcessPoiInputShape:
    def test_neighbor_arrays_accept_8x3_shape(self):
        np.random.seed(42)
        img_size = 64
        ref_image = np.random.rand(img_size, img_size).astype(np.float64) * 200
        grad_x = np.gradient(ref_image, axis=1)
        grad_y = np.gradient(ref_image, axis=0)

        from speckle.core.optimization.icgn_core_numba import prefilter_image

        coeffs = prefilter_image(ref_image, order=3)

        subset_size = 15
        m = subset_size // 2
        n_params = 6
        n_pixels_max = (m + 1) * (m + 1)
        cx, cy = 32, 32

        neighbor_valid = np.zeros((8, 3), dtype=np.bool_)
        neighbor_params = np.zeros((8, 3, n_params), dtype=np.float64)
        neighbor_x = np.zeros((8, 3), dtype=np.float64)
        neighbor_y = np.zeros((8, 3), dtype=np.float64)

        q5 = Q5 - 1
        neighbor_valid[q5, 0] = True
        neighbor_valid[q5, 1] = True

        neighbor_x[q5, 0] = cx - 11
        neighbor_y[q5, 0] = cy - 11
        neighbor_params[q5, 0] = [0.5, 0.001, 0.001, 0.3, 0.001, 0.001]

        neighbor_x[q5, 1] = cx
        neighbor_y[q5, 1] = cy - 11
        neighbor_params[q5, 1] = [0.6, 0.002, 0.001, 0.4, 0.001, 0.002]

        out_p = np.zeros((4, n_params), dtype=np.float64)
        out_zncc = np.zeros(4, dtype=np.float64)
        out_iter = np.zeros(4, dtype=np.int32)
        out_qt = np.zeros(4, dtype=np.int32)
        out_cand_zncc = np.full(8, -1.0, dtype=np.float64)
        out_fail_info = np.full((8, 3), -1.0, dtype=np.float64)

        f = np.zeros(n_pixels_max, dtype=np.float64)
        dfdx = np.zeros(n_pixels_max, dtype=np.float64)
        dfdy = np.zeros(n_pixels_max, dtype=np.float64)
        j = np.zeros((n_pixels_max, n_params), dtype=np.float64)
        h = np.zeros((n_params, n_params), dtype=np.float64)
        h_inv = np.zeros((n_params, n_params), dtype=np.float64)
        p = np.zeros(n_params, dtype=np.float64)
        xsi_w = np.zeros(n_pixels_max, dtype=np.float64)
        eta_w = np.zeros(n_pixels_max, dtype=np.float64)
        x_def = np.zeros(n_pixels_max, dtype=np.float64)
        y_def = np.zeros(n_pixels_max, dtype=np.float64)
        g = np.zeros(n_pixels_max, dtype=np.float64)
        b = np.zeros(n_params, dtype=np.float64)
        dp = np.zeros(n_params, dtype=np.float64)
        p_new = np.zeros(n_params, dtype=np.float64)
        xsi_local = np.zeros(n_pixels_max, dtype=np.float64)
        eta_local = np.zeros(n_pixels_max, dtype=np.float64)
        init_params = np.zeros(n_params, dtype=np.float64)

        n_rec = process_poi_adss_multi(
            ref_image,
            grad_x,
            grad_y,
            coeffs,
            3,
            cx,
            cy,
            subset_size,
            50,
            0.001,
            AFFINE,
            neighbor_valid,
            neighbor_params,
            neighbor_x,
            neighbor_y,
            0.9,
            out_p,
            out_zncc,
            out_iter,
            out_qt,
            out_cand_zncc,
            out_fail_info,
            f,
            dfdx,
            dfdy,
            j,
            h,
            h_inv,
            p,
            xsi_w,
            eta_w,
            x_def,
            y_def,
            g,
            b,
            dp,
            p_new,
            xsi_local,
            eta_local,
            init_params,
        )

        assert isinstance(n_rec, (int, np.integer))
        assert 0 <= int(n_rec) <= 4


class TestNeighborDirectionMapping:
    def test_q5_neighbors_are_upper_left_region(self):
        points_x, points_y, ny, nx = _make_grid(5, 5, 10)
        n_poi = len(points_x)
        n_params = 6

        valid_mask = np.ones(n_poi, dtype=np.bool_)
        valid_mask[12] = False
        zncc_values = np.full(n_poi, 0.95)
        zncc_values[12] = 0.5
        parameters = np.random.rand(n_poi, n_params)
        bad_indices = np.array([12], dtype=np.int64)

        _, _, nx_, ny_ = _build_neighbor_info_adss_v2(
            bad_indices,
            points_x,
            points_y,
            valid_mask,
            zncc_values,
            parameters,
            0.9,
            ny,
            nx,
            n_params,
        )

        center_x = points_x[12]
        center_y = points_y[12]
        q5 = Q5 - 1

        assert nx_[0, q5, 0] == center_x - 10
        assert ny_[0, q5, 0] == center_y - 10
        assert nx_[0, q5, 1] == center_x
        assert ny_[0, q5, 1] == center_y - 10
        assert nx_[0, q5, 2] == center_x - 10
        assert ny_[0, q5, 2] == center_y

    def test_q8_neighbors_are_lower_right_region(self):
        points_x, points_y, ny, nx = _make_grid(5, 5, 10)
        n_poi = len(points_x)
        n_params = 6

        valid_mask = np.ones(n_poi, dtype=np.bool_)
        valid_mask[12] = False
        zncc_values = np.full(n_poi, 0.95)
        zncc_values[12] = 0.5
        parameters = np.random.rand(n_poi, n_params)
        bad_indices = np.array([12], dtype=np.int64)

        _, _, nx_, ny_ = _build_neighbor_info_adss_v2(
            bad_indices,
            points_x,
            points_y,
            valid_mask,
            zncc_values,
            parameters,
            0.9,
            ny,
            nx,
            n_params,
        )

        center_x = points_x[12]
        center_y = points_y[12]
        q8 = Q8 - 1

        assert nx_[0, q8, 0] == center_x + 10
        assert ny_[0, q8, 0] == center_y + 10
        assert nx_[0, q8, 1] == center_x
        assert ny_[0, q8, 1] == center_y + 10
        assert nx_[0, q8, 2] == center_x + 10
        assert ny_[0, q8, 2] == center_y


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
