"""
process_poi_adss 함수 검증 테스트.

합성 이미지(균일 변위)로 quarter-subset IC-GN이
올바르게 수렴하는지 검증한다.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from scipy.ndimage import spline_filter, sobel, gaussian_filter

from speckle.core.optimization.adss_subset_numba import (
    process_poi_adss,
    Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8,
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


class TestProcessPoiAdss:
    """process_poi_adss 함수 검증."""

    @pytest.fixture
    def setup(self):
        """테스트 환경 구성: 균일 변위 (3, 2)."""
        shift_x, shift_y = 3.0, 2.0
        subset_size = 21
        M = subset_size // 2
        n_params = get_num_params(AFFINE)
        n_pixels_max = subset_size * subset_size

        ref = _make_speckle_image(256, 256, seed=42)
        deformed = _make_shifted_image(ref, shift_x, shift_y)
        grad_x, grad_y = _compute_gradient(ref)
        coeffs = spline_filter(deformed, order=5).astype(np.float64)

        # 버퍼 할당
        bufs = {
            'f': np.zeros(n_pixels_max, dtype=np.float64),
            'dfdx': np.zeros(n_pixels_max, dtype=np.float64),
            'dfdy': np.zeros(n_pixels_max, dtype=np.float64),
            'J': np.zeros((n_pixels_max, n_params), dtype=np.float64),
            'H': np.zeros((n_params, n_params), dtype=np.float64),
            'H_inv': np.zeros((n_params, n_params), dtype=np.float64),
            'p': np.zeros(n_params, dtype=np.float64),
            'xsi_w': np.zeros(n_pixels_max, dtype=np.float64),
            'eta_w': np.zeros(n_pixels_max, dtype=np.float64),
            'x_def': np.zeros(n_pixels_max, dtype=np.float64),
            'y_def': np.zeros(n_pixels_max, dtype=np.float64),
            'g': np.zeros(n_pixels_max, dtype=np.float64),
            'b': np.zeros(n_params, dtype=np.float64),
            'dp': np.zeros(n_params, dtype=np.float64),
            'p_new': np.zeros(n_params, dtype=np.float64),
            'xsi_local': np.zeros(n_pixels_max, dtype=np.float64),
            'eta_local': np.zeros(n_pixels_max, dtype=np.float64),
            'init_params': np.zeros(n_params, dtype=np.float64),
        }

        # 이웃 정보: 모든 이웃이 well-matched이고 정확한 변위를 가짐
        spacing = 5
        cx, cy = 128, 128
        # Q1~Q8 대응 이웃 방향 (dy, dx)
        neighbor_dirs = [
            (-1,  0),  # Q1 → 위
            ( 1,  0),  # Q2 → 아래
            ( 0, -1),  # Q3 → 왼쪽
            ( 0,  1),  # Q4 → 오른쪽
            (-1, -1),  # Q5 → 좌상
            (-1,  1),  # Q6 → 우상
            ( 1, -1),  # Q7 → 좌하
            ( 1,  1),  # Q8 → 우하
        ]

        neighbor_valid = np.ones(8, dtype=np.bool_)
        neighbor_params = np.zeros((8, n_params), dtype=np.float64)
        neighbor_x = np.zeros(8, dtype=np.float64)
        neighbor_y = np.zeros(8, dtype=np.float64)

        for i, (dy, dx) in enumerate(neighbor_dirs):
            neighbor_x[i] = cx + dx * spacing
            neighbor_y[i] = cy + dy * spacing
            # 균일 변위이므로 이웃도 동일한 파라미터
            neighbor_params[i, 0] = shift_x  # u
            neighbor_params[i, 3] = shift_y  # v

        return {
            'ref': ref, 'grad_x': grad_x, 'grad_y': grad_y,
            'coeffs': coeffs, 'shift_x': shift_x, 'shift_y': shift_y,
            'subset_size': subset_size, 'cx': cx, 'cy': cy,
            'neighbor_valid': neighbor_valid,
            'neighbor_params': neighbor_params,
            'neighbor_x': neighbor_x, 'neighbor_y': neighbor_y,
            'bufs': bufs, 'n_params': n_params,
        }

    def _run_adss(self, s):
        """process_poi_adss 실행 헬퍼."""
        b = s['bufs']
        # 버퍼 초기화
        b['p'][:] = 0.0
        return process_poi_adss(
            s['ref'], s['grad_x'], s['grad_y'],
            s['coeffs'], 5,
            s['cx'], s['cy'],
            s['subset_size'],
            50,    # max_iterations
            0.001, # convergence_threshold
            AFFINE,
            s['neighbor_valid'],
            s['neighbor_params'],
            s['neighbor_x'], s['neighbor_y'],
            0.85,  # zncc_threshold
            b['f'], b['dfdx'], b['dfdy'],
            b['J'], b['H'], b['H_inv'],
            b['p'], b['xsi_w'], b['eta_w'],
            b['x_def'], b['y_def'],
            b['g'], b['b'], b['dp'], b['p_new'],
            b['xsi_local'], b['eta_local'],
            b['init_params'],
        )

    # ── 기본 수렴 테스트 ────────────────────────────────

    def test_converges_with_all_neighbors(self, setup):
        """모든 이웃이 valid → 수렴 및 높은 ZNCC."""
        zncc, n_iter, conv, fail_code, best_q = self._run_adss(setup)

        assert conv is True, f"Not converged: fail_code={fail_code}"
        assert zncc > 0.99, f"ZNCC={zncc:.4f}"
        assert best_q >= 1 and best_q <= 8
        assert n_iter > 0

    def test_displacement_accuracy(self, setup):
        """수렴 후 변위가 정확한지 확인."""
        self._run_adss(setup)
        p = setup['bufs']['p']

        assert p[0] == pytest.approx(setup['shift_x'], abs=0.05)
        assert p[3] == pytest.approx(setup['shift_y'], abs=0.05)

    # ── 세트 선택 테스트 ────────────────────────────────

    def test_only_cross_neighbors_valid(self, setup):
        """십자 이웃(Q1~Q4)만 valid → 십자 세트에서 선택."""
        setup['neighbor_valid'][4:] = False  # Q5~Q8 비활성화
        zncc, n_iter, conv, fail_code, best_q = self._run_adss(setup)

        assert conv is True
        assert best_q >= 1 and best_q <= 4, f"Expected cross set, got Q{best_q}"

    def test_only_diagonal_neighbors_valid(self, setup):
        """대각선 이웃(Q5~Q8)만 valid → 대각선 세트에서 선택."""
        setup['neighbor_valid'][:4] = False  # Q1~Q4 비활성화
        zncc, n_iter, conv, fail_code, best_q = self._run_adss(setup)

        assert conv is True
        assert best_q >= 5 and best_q <= 8, f"Expected diag set, got Q{best_q}"

    # ── 이웃 없음 테스트 ────────────────────────────────

    def test_no_valid_neighbors(self, setup):
        """모든 이웃 invalid → 실패 반환."""
        setup['neighbor_valid'][:] = False
        zncc, n_iter, conv, fail_code, best_q = self._run_adss(setup)

        assert conv is False
        assert best_q == -1

    # ── 단일 이웃만 valid ───────────────────────────────

    def test_single_neighbor_valid(self, setup):
        """이웃 하나만 valid해도 수렴 가능."""
        setup['neighbor_valid'][:] = False
        setup['neighbor_valid'][0] = True  # Q1만 valid
        zncc, n_iter, conv, fail_code, best_q = self._run_adss(setup)

        assert conv is True
        assert best_q == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
