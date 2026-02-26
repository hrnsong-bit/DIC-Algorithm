"""
process_bad_pois_adss_parallel 및 allocate_adss_batch_buffers 검증 테스트.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from scipy.ndimage import spline_filter, sobel, gaussian_filter

from speckle.core.optimization.adss_subset_numba import (
    process_bad_pois_adss_parallel,
    allocate_adss_batch_buffers,
)
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
    src_r0 = max(0, -sy); src_r1 = min(h, h - sy)
    src_c0 = max(0, -sx); src_c1 = min(w, w - sx)
    dst_r0 = src_r0 + sy; dst_r1 = src_r1 + sy
    dst_c0 = src_c0 + sx; dst_c1 = src_c1 + sx
    deformed[dst_r0:dst_r1, dst_c0:dst_c1] = ref[src_r0:src_r1, src_c0:src_c1]
    return deformed


class TestAdsBatchParallel:
    """process_bad_pois_adss_parallel 검증."""

    @pytest.fixture
    def setup(self):
        """
        테스트 환경: 균일 변위 (3, 2), POI 격자 생성,
        일부를 불량으로 지정하여 ADSS 재계산.
        """
        shift_x, shift_y = 3.0, 2.0
        subset_size = 21
        M = subset_size // 2
        n_params = get_num_params(AFFINE)
        n_pixels_max = (2 * M + 1) * (M + 1)  # 십자형 최대 크기
        spacing = 5

        ref = _make_speckle_image(256, 256, seed=42)
        deformed = _make_shifted_image(ref, shift_x, shift_y)
        grad_x, grad_y = _compute_gradient(ref)
        coeffs = spline_filter(deformed, order=5).astype(np.float64)

        # POI 격자 생성 (중앙 영역)
        margin = subset_size + 10
        xs = np.arange(margin, 256 - margin, spacing, dtype=np.int64)
        ys = np.arange(margin, 256 - margin, spacing, dtype=np.int64)
        grid_x, grid_y = np.meshgrid(xs, ys)
        points_x = grid_x.ravel().copy()
        points_y = grid_y.ravel().copy()
        n_total = len(points_x)

        # 모든 POI에 대해 "잘 매칭된" 파라미터 생성 (균일 변위)
        all_params = np.zeros((n_total, n_params), dtype=np.float64)
        all_params[:, 0] = shift_x  # u
        all_params[:, 3] = shift_y  # v
        valid_mask = np.ones(n_total, dtype=np.bool_)

        # 불량 POI 지정 (중앙 근처 4개)
        ny = len(ys)
        nx = len(xs)
        center_row = ny // 2
        center_col = nx // 2
        bad_grid_indices = [
            center_row * nx + center_col,
            center_row * nx + center_col + 1,
            (center_row + 1) * nx + center_col,
            (center_row + 1) * nx + center_col + 1,
        ]
        bad_indices = np.array(bad_grid_indices, dtype=np.int64)
        n_bad = len(bad_indices)

        # 이웃 정보 구성
        neighbor_dirs = [
            (-1,  0),  # Q1
            ( 1,  0),  # Q2
            ( 0, -1),  # Q3
            ( 0,  1),  # Q4
            (-1, -1),  # Q5
            (-1,  1),  # Q6
            ( 1, -1),  # Q7
            ( 1,  1),  # Q8
        ]

        all_neighbor_valid = np.zeros((n_bad, 8), dtype=np.bool_)
        all_neighbor_params = np.zeros((n_bad, 8, n_params), dtype=np.float64)
        all_neighbor_x = np.zeros((n_bad, 8), dtype=np.float64)
        all_neighbor_y = np.zeros((n_bad, 8), dtype=np.float64)

        for k, bi in enumerate(bad_indices):
            row_idx = bi // nx
            col_idx = bi % nx
            for d, (dy, dx) in enumerate(neighbor_dirs):
                nr = row_idx + dy
                nc = col_idx + dx
                if 0 <= nr < ny and 0 <= nc < nx:
                    ni = nr * nx + nc
                    # 불량 POI가 아닌 이웃만 valid
                    if ni not in bad_grid_indices:
                        all_neighbor_valid[k, d] = True
                        all_neighbor_params[k, d, :] = all_params[ni]
                        all_neighbor_x[k, d] = points_x[ni]
                        all_neighbor_y[k, d] = points_y[ni]

        return {
            'ref': ref, 'grad_x': grad_x, 'grad_y': grad_y,
            'coeffs': coeffs,
            'points_x': points_x, 'points_y': points_y,
            'subset_size': subset_size, 'n_params': n_params,
            'n_pixels_max': n_pixels_max,
            'bad_indices': bad_indices, 'n_bad': n_bad,
            'all_neighbor_valid': all_neighbor_valid,
            'all_neighbor_params': all_neighbor_params,
            'all_neighbor_x': all_neighbor_x,
            'all_neighbor_y': all_neighbor_y,
            'shift_x': shift_x, 'shift_y': shift_y,
        }

    # ── allocate_adss_batch_buffers 검증 ────────────────

    def test_buffer_shapes(self, setup):
        """버퍼 shape이 올바른지 확인."""
        s = setup
        bufs = allocate_adss_batch_buffers(s['n_bad'], s['n_pixels_max'], s['n_params'])

        assert bufs['f'].shape == (s['n_bad'], s['n_pixels_max'])
        assert bufs['J'].shape == (s['n_bad'], s['n_pixels_max'], s['n_params'])
        assert bufs['H'].shape == (s['n_bad'], s['n_params'], s['n_params'])
        assert bufs['p'].shape == (s['n_bad'], s['n_params'])
        assert bufs['init_params'].shape == (s['n_bad'], s['n_params'])

    # ── 병렬 처리 기본 수렴 ─────────────────────────────

    def test_all_bad_pois_converge(self, setup):
        """모든 불량 POI가 수렴."""
        s = setup
        bufs = allocate_adss_batch_buffers(s['n_bad'], s['n_pixels_max'], s['n_params'])

        result_p = np.zeros((s['n_bad'], s['n_params']), dtype=np.float64)
        result_zncc = np.zeros(s['n_bad'], dtype=np.float64)
        result_iter = np.zeros(s['n_bad'], dtype=np.int32)
        result_conv = np.zeros(s['n_bad'], dtype=np.bool_)
        result_fail = np.zeros(s['n_bad'], dtype=np.int32)
        result_qt = np.zeros(s['n_bad'], dtype=np.int32)

        process_bad_pois_adss_parallel(
            s['ref'], s['grad_x'], s['grad_y'],
            s['coeffs'], 5,
            s['points_x'], s['points_y'],
            s['subset_size'], 50, 0.001, AFFINE,
            s['bad_indices'],
            s['all_neighbor_valid'],
            s['all_neighbor_params'],
            s['all_neighbor_x'], s['all_neighbor_y'],
            0.85,
            result_p, result_zncc, result_iter, result_conv,
            result_fail, result_qt,
            bufs['f'], bufs['dfdx'], bufs['dfdy'],
            bufs['J'], bufs['H'], bufs['H_inv'],
            bufs['p'], bufs['xsi_w'], bufs['eta_w'],
            bufs['x_def'], bufs['y_def'],
            bufs['g'], bufs['b'], bufs['dp'], bufs['p_new'],
            bufs['xsi_local'], bufs['eta_local'],
            bufs['init_params'],
        )

        for k in range(s['n_bad']):
            assert result_conv[k], \
                f"POI {k} not converged: fail={result_fail[k]}, zncc={result_zncc[k]:.4f}"

    # ── 변위 정확도 ─────────────────────────────────────

    def test_displacement_accuracy(self, setup):
        """수렴 후 변위가 정확."""
        s = setup
        bufs = allocate_adss_batch_buffers(s['n_bad'], s['n_pixels_max'], s['n_params'])

        result_p = np.zeros((s['n_bad'], s['n_params']), dtype=np.float64)
        result_zncc = np.zeros(s['n_bad'], dtype=np.float64)
        result_iter = np.zeros(s['n_bad'], dtype=np.int32)
        result_conv = np.zeros(s['n_bad'], dtype=np.bool_)
        result_fail = np.zeros(s['n_bad'], dtype=np.int32)
        result_qt = np.zeros(s['n_bad'], dtype=np.int32)

        process_bad_pois_adss_parallel(
            s['ref'], s['grad_x'], s['grad_y'],
            s['coeffs'], 5,
            s['points_x'], s['points_y'],
            s['subset_size'], 50, 0.001, AFFINE,
            s['bad_indices'],
            s['all_neighbor_valid'],
            s['all_neighbor_params'],
            s['all_neighbor_x'], s['all_neighbor_y'],
            0.85,
            result_p, result_zncc, result_iter, result_conv,
            result_fail, result_qt,
            bufs['f'], bufs['dfdx'], bufs['dfdy'],
            bufs['J'], bufs['H'], bufs['H_inv'],
            bufs['p'], bufs['xsi_w'], bufs['eta_w'],
            bufs['x_def'], bufs['y_def'],
            bufs['g'], bufs['b'], bufs['dp'], bufs['p_new'],
            bufs['xsi_local'], bufs['eta_local'],
            bufs['init_params'],
        )

        for k in range(s['n_bad']):
            assert result_p[k, 0] == pytest.approx(s['shift_x'], abs=0.05), \
                f"POI {k}: u={result_p[k, 0]:.4f}, expected {s['shift_x']}"
            assert result_p[k, 3] == pytest.approx(s['shift_y'], abs=0.05), \
                f"POI {k}: v={result_p[k, 3]:.4f}, expected {s['shift_y']}"

    # ── ZNCC 값 검증 ───────────────────────────────────

    def test_zncc_high(self, setup):
        """수렴 후 ZNCC > 0.99."""
        s = setup
        bufs = allocate_adss_batch_buffers(s['n_bad'], s['n_pixels_max'], s['n_params'])

        result_p = np.zeros((s['n_bad'], s['n_params']), dtype=np.float64)
        result_zncc = np.zeros(s['n_bad'], dtype=np.float64)
        result_iter = np.zeros(s['n_bad'], dtype=np.int32)
        result_conv = np.zeros(s['n_bad'], dtype=np.bool_)
        result_fail = np.zeros(s['n_bad'], dtype=np.int32)
        result_qt = np.zeros(s['n_bad'], dtype=np.int32)

        process_bad_pois_adss_parallel(
            s['ref'], s['grad_x'], s['grad_y'],
            s['coeffs'], 5,
            s['points_x'], s['points_y'],
            s['subset_size'], 50, 0.001, AFFINE,
            s['bad_indices'],
            s['all_neighbor_valid'],
            s['all_neighbor_params'],
            s['all_neighbor_x'], s['all_neighbor_y'],
            0.85,
            result_p, result_zncc, result_iter, result_conv,
            result_fail, result_qt,
            bufs['f'], bufs['dfdx'], bufs['dfdy'],
            bufs['J'], bufs['H'], bufs['H_inv'],
            bufs['p'], bufs['xsi_w'], bufs['eta_w'],
            bufs['x_def'], bufs['y_def'],
            bufs['g'], bufs['b'], bufs['dp'], bufs['p_new'],
            bufs['xsi_local'], bufs['eta_local'],
            bufs['init_params'],
        )

        for k in range(s['n_bad']):
            assert result_zncc[k] > 0.99, f"POI {k}: ZNCC={result_zncc[k]:.4f}"

    # ── quarter type 유효 범위 ──────────────────────────

    def test_quarter_type_valid(self, setup):
        """선택된 quarter type이 1~8 범위."""
        s = setup
        bufs = allocate_adss_batch_buffers(s['n_bad'], s['n_pixels_max'], s['n_params'])

        result_p = np.zeros((s['n_bad'], s['n_params']), dtype=np.float64)
        result_zncc = np.zeros(s['n_bad'], dtype=np.float64)
        result_iter = np.zeros(s['n_bad'], dtype=np.int32)
        result_conv = np.zeros(s['n_bad'], dtype=np.bool_)
        result_fail = np.zeros(s['n_bad'], dtype=np.int32)
        result_qt = np.zeros(s['n_bad'], dtype=np.int32)

        process_bad_pois_adss_parallel(
            s['ref'], s['grad_x'], s['grad_y'],
            s['coeffs'], 5,
            s['points_x'], s['points_y'],
            s['subset_size'], 50, 0.001, AFFINE,
            s['bad_indices'],
            s['all_neighbor_valid'],
            s['all_neighbor_params'],
            s['all_neighbor_x'], s['all_neighbor_y'],
            0.85,
            result_p, result_zncc, result_iter, result_conv,
            result_fail, result_qt,
            bufs['f'], bufs['dfdx'], bufs['dfdy'],
            bufs['J'], bufs['H'], bufs['H_inv'],
            bufs['p'], bufs['xsi_w'], bufs['eta_w'],
            bufs['x_def'], bufs['y_def'],
            bufs['g'], bufs['b'], bufs['dp'], bufs['p_new'],
            bufs['xsi_local'], bufs['eta_local'],
            bufs['init_params'],
        )

        for k in range(s['n_bad']):
            assert 1 <= result_qt[k] <= 8, f"POI {k}: quarter_type={result_qt[k]}"

    # ── 이웃 전부 invalid → 전부 실패 ──────────────────

    def test_no_neighbors_all_fail(self, setup):
        """모든 이웃이 invalid → 모든 불량 POI 실패."""
        s = setup
        s['all_neighbor_valid'][:] = False
        bufs = allocate_adss_batch_buffers(s['n_bad'], s['n_pixels_max'], s['n_params'])

        result_p = np.zeros((s['n_bad'], s['n_params']), dtype=np.float64)
        result_zncc = np.zeros(s['n_bad'], dtype=np.float64)
        result_iter = np.zeros(s['n_bad'], dtype=np.int32)
        result_conv = np.zeros(s['n_bad'], dtype=np.bool_)
        result_fail = np.zeros(s['n_bad'], dtype=np.int32)
        result_qt = np.zeros(s['n_bad'], dtype=np.int32)

        process_bad_pois_adss_parallel(
            s['ref'], s['grad_x'], s['grad_y'],
            s['coeffs'], 5,
            s['points_x'], s['points_y'],
            s['subset_size'], 50, 0.001, AFFINE,
            s['bad_indices'],
            s['all_neighbor_valid'],
            s['all_neighbor_params'],
            s['all_neighbor_x'], s['all_neighbor_y'],
            0.85,
            result_p, result_zncc, result_iter, result_conv,
            result_fail, result_qt,
            bufs['f'], bufs['dfdx'], bufs['dfdy'],
            bufs['J'], bufs['H'], bufs['H_inv'],
            bufs['p'], bufs['xsi_w'], bufs['eta_w'],
            bufs['x_def'], bufs['y_def'],
            bufs['g'], bufs['b'], bufs['dp'], bufs['p_new'],
            bufs['xsi_local'], bufs['eta_local'],
            bufs['init_params'],
        )

        for k in range(s['n_bad']):
            assert not result_conv[k]
            assert result_qt[k] == -1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
