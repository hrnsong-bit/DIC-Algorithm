"""
compute_adss_recalc 래퍼 함수 검증 테스트.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from scipy.ndimage import spline_filter, sobel, gaussian_filter

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
    src_r0 = max(0, -sy); src_r1 = min(h, h - sy)
    src_c0 = max(0, -sx); src_c1 = min(w, w - sx)
    dst_r0 = src_r0 + sy; dst_r1 = src_r1 + sy
    dst_c0 = src_c0 + sx; dst_c1 = src_c1 + sx
    deformed[dst_r0:dst_r1, dst_c0:dst_c1] = ref[src_r0:src_r1, src_c0:src_c1]
    return deformed


def _create_test_environment(n_bad_pois=4):
    """
    테스트 환경 생성: 균일 변위, POI 격자, 일부 불량 POI.

    Returns: dict with all arrays needed for compute_adss_recalc.
    """
    shift_x, shift_y = 3.0, 2.0
    subset_size = 21
    spacing = 5
    n_params = get_num_params(AFFINE)

    ref = _make_speckle_image(256, 256, seed=42)
    deformed = _make_shifted_image(ref, shift_x, shift_y)
    grad_x, grad_y = _compute_gradient(ref)
    coeffs = spline_filter(deformed, order=5).astype(np.float64)

    # POI 격자
    margin = subset_size + 10
    xs = np.arange(margin, 256 - margin, spacing, dtype=np.int64)
    ys = np.arange(margin, 256 - margin, spacing, dtype=np.int64)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points_x = grid_x.ravel().copy()
    points_y = grid_y.ravel().copy()
    n_total = len(points_x)
    ny, nx = len(ys), len(xs)

    # 모든 POI well-matched로 초기화
    valid_mask = np.ones(n_total, dtype=np.bool_)
    zncc_values = np.ones(n_total, dtype=np.float64) * 0.99
    parameters = np.zeros((n_total, n_params), dtype=np.float64)
    parameters[:, 0] = shift_x
    parameters[:, 3] = shift_y
    convergence_flags = np.ones(n_total, dtype=np.bool_)
    iteration_counts = np.full(n_total, 5, dtype=np.int32)
    failure_reasons = np.zeros(n_total, dtype=np.int32)

    # 중앙 근처를 불량으로 지정
    center_row = ny // 2
    center_col = nx // 2
    bad_grid = []
    for dr in range(n_bad_pois):
        idx = center_row * nx + center_col + dr
        if idx < n_total:
            bad_grid.append(idx)

    for idx in bad_grid:
        valid_mask[idx] = False
        zncc_values[idx] = 0.3
        parameters[idx, :] = 0.0
        convergence_flags[idx] = False
        failure_reasons[idx] = 1  # ICGN_FAIL_LOW_ZNCC

    return {
        'ref': ref, 'grad_x': grad_x, 'grad_y': grad_y,
        'coeffs': coeffs,
        'points_x': points_x, 'points_y': points_y,
        'valid_mask': valid_mask, 'zncc_values': zncc_values,
        'parameters': parameters,
        'convergence_flags': convergence_flags,
        'iteration_counts': iteration_counts,
        'failure_reasons': failure_reasons,
        'subset_size': subset_size,
        'shift_x': shift_x, 'shift_y': shift_y,
        'n_total': n_total, 'n_params': n_params,
        'bad_grid': bad_grid,
    }


class TestComputeAdssRecalc:
    """compute_adss_recalc 래퍼 함수 검증."""

    # ── 불량 POI 없음 → 스킵 ───────────────────────────

    def test_no_bad_pois_skip(self):
        """불량 POI가 없으면 스킵."""
        env = _create_test_environment(n_bad_pois=0)
        # 모두 valid로 복원
        env['valid_mask'][:] = True
        env['zncc_values'][:] = 0.99

        result = compute_adss_recalc(
            env['ref'], env['grad_x'], env['grad_y'],
            env['coeffs'], 5,
            env['points_x'], env['points_y'],
            env['valid_mask'], env['zncc_values'], env['parameters'],
            env['convergence_flags'], env['iteration_counts'], env['failure_reasons'],
            env['subset_size'],
        )

        assert result['n_bad'] == 0
        assert result['n_recovered'] == 0
        assert result['n_failed'] == 0

    # ── 기본 복원 성공 ──────────────────────────────────

    def test_basic_recovery(self):
        """불량 POI가 복원됨."""
        env = _create_test_environment(n_bad_pois=4)

        result = compute_adss_recalc(
            env['ref'], env['grad_x'], env['grad_y'],
            env['coeffs'], 5,
            env['points_x'], env['points_y'],
            env['valid_mask'], env['zncc_values'], env['parameters'],
            env['convergence_flags'], env['iteration_counts'], env['failure_reasons'],
            env['subset_size'],
        )

        assert result['n_bad'] == 4
        assert result['n_recovered'] > 0
        assert result['n_recovered'] + result['n_failed'] == result['n_bad']

    # ── in-place 업데이트 검증 ──────────────────────────

    def test_inplace_update(self):
        """복원된 POI의 valid_mask, zncc, parameters가 업데이트됨."""
        env = _create_test_environment(n_bad_pois=4)

        result = compute_adss_recalc(
            env['ref'], env['grad_x'], env['grad_y'],
            env['coeffs'], 5,
            env['points_x'], env['points_y'],
            env['valid_mask'], env['zncc_values'], env['parameters'],
            env['convergence_flags'], env['iteration_counts'], env['failure_reasons'],
            env['subset_size'],
        )

        for fidx in result['recovered_indices']:
            assert env['valid_mask'][fidx] == True
            assert env['zncc_values'][fidx] >= 0.9
            assert env['convergence_flags'][fidx] == True
            assert env['failure_reasons'][fidx] == 0

    # ── 변위 정확도 ─────────────────────────────────────

    def test_displacement_accuracy(self):
        """복원된 POI의 변위가 정확."""
        env = _create_test_environment(n_bad_pois=4)

        result = compute_adss_recalc(
            env['ref'], env['grad_x'], env['grad_y'],
            env['coeffs'], 5,
            env['points_x'], env['points_y'],
            env['valid_mask'], env['zncc_values'], env['parameters'],
            env['convergence_flags'], env['iteration_counts'], env['failure_reasons'],
            env['subset_size'],
        )

        for fidx in result['recovered_indices']:
            assert env['parameters'][fidx, 0] == pytest.approx(env['shift_x'], abs=0.1)
            assert env['parameters'][fidx, 3] == pytest.approx(env['shift_y'], abs=0.1)

    # ── 반환값 구조 검증 ────────────────────────────────

    def test_return_structure(self):
        """반환 dict의 키와 타입 검증."""
        env = _create_test_environment(n_bad_pois=2)

        result = compute_adss_recalc(
            env['ref'], env['grad_x'], env['grad_y'],
            env['coeffs'], 5,
            env['points_x'], env['points_y'],
            env['valid_mask'], env['zncc_values'], env['parameters'],
            env['convergence_flags'], env['iteration_counts'], env['failure_reasons'],
            env['subset_size'],
        )

        assert 'n_bad' in result
        assert 'n_recovered' in result
        assert 'n_failed' in result
        assert 'recovered_indices' in result
        assert 'failed_indices' in result
        assert 'quarter_types' in result
        assert 'elapsed_time' in result
        assert isinstance(result['recovered_indices'], np.ndarray)
        assert isinstance(result['elapsed_time'], float)

    # ── quarter_types 유효성 ────────────────────────────

    def test_quarter_types_valid(self):
        """복원된 POI의 quarter_type이 1~8 범위."""
        env = _create_test_environment(n_bad_pois=4)

        result = compute_adss_recalc(
            env['ref'], env['grad_x'], env['grad_y'],
            env['coeffs'], 5,
            env['points_x'], env['points_y'],
            env['valid_mask'], env['zncc_values'], env['parameters'],
            env['convergence_flags'], env['iteration_counts'], env['failure_reasons'],
            env['subset_size'],
        )

        for i, fidx in enumerate(result['recovered_indices']):
            orig_k = np.searchsorted(np.where(~env['valid_mask'] | (env['zncc_values'] < 0.9))[0], fidx)
            qt = result['quarter_types'][orig_k]
            assert 1 <= qt <= 8, f"POI {fidx}: quarter_type={qt}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
