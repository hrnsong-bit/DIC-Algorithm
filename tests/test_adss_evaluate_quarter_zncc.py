"""
evaluate_quarter_zncc 함수 검증 테스트.

합성 이미지(균일 변위)를 생성하여 quarter-subset ZNCC가
올바르게 계산되는지 검증한다.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from scipy.ndimage import spline_filter

from speckle.core.optimization.adss_subset_numba import (
    generate_quarter_local_coordinates,
    evaluate_quarter_zncc,
    Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8,
)
from speckle.core.optimization.shape_function_numba import AFFINE


def _make_speckle_image(h=256, w=256, seed=42):
    """재현 가능한 합성 스페클 이미지 생성."""
    rng = np.random.RandomState(seed)
    img = rng.rand(h, w).astype(np.float64)
    # 가우시안 블러로 스페클 패턴 생성
    from scipy.ndimage import gaussian_filter
    img = gaussian_filter(img, sigma=2.0)
    # 0~255 범위로 정규화
    img = (img - img.min()) / (img.max() - img.min()) * 255.0
    return img


def _compute_gradient(img):
    """Sobel gradient 계산 (icgn.py와 동일)."""
    from scipy.ndimage import sobel
    grad_x = sobel(img, axis=1) / 32.0
    grad_y = sobel(img, axis=0) / 32.0
    return grad_x, grad_y


def _make_shifted_image(ref, shift_x=0.0, shift_y=0.0):
    """정수 픽셀 이동으로 deformed 이미지 생성."""
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


class TestEvaluateQuarterZncc:
    """evaluate_quarter_zncc 함수 검증."""

    @pytest.fixture
    def setup_images(self):
        """테스트용 이미지 세트 (균일 변위 3px, 2px)."""
        ref = _make_speckle_image(256, 256, seed=42)
        shift_x, shift_y = 3.0, 2.0
        deformed = _make_shifted_image(ref, shift_x, shift_y)
        grad_x, grad_y = _compute_gradient(ref)
        coeffs = spline_filter(deformed, order=5).astype(np.float64)
        return ref, grad_x, grad_y, coeffs, shift_x, shift_y

    @pytest.fixture
    def setup_buffers(self):
        """작업 버퍼 할당 (최대 크기)."""
        subset_size = 21
        n_max = subset_size * subset_size  # 441 (Q0 최대)
        return {
            'f': np.zeros(n_max, dtype=np.float64),
            'dfdx': np.zeros(n_max, dtype=np.float64),
            'dfdy': np.zeros(n_max, dtype=np.float64),
            'xsi_w': np.zeros(n_max, dtype=np.float64),
            'eta_w': np.zeros(n_max, dtype=np.float64),
            'x_def': np.zeros(n_max, dtype=np.float64),
            'y_def': np.zeros(n_max, dtype=np.float64),
            'g': np.zeros(n_max, dtype=np.float64),
        }

    def _get_quarter_ranges(self, subset_size, q):
        """quarter-subset의 xsi/eta min/max를 반환."""
        M = subset_size // 2
        ranges = {
            Q0: (-M, M, -M, M),
            Q1: (-M, M, -M, 0),
            Q2: (-M, M, 0, M),
            Q3: (-M, 0, -M, M),
            Q4: (0, M, -M, M),
            Q5: (-M, 0, -M, 0),
            Q6: (0, M, -M, 0),
            Q7: (-M, 0, 0, M),
            Q8: (0, M, 0, M),
        }
        return ranges[q]

    # ── 정확한 초기값 → 높은 ZNCC ──────────────────────

    @pytest.mark.parametrize("q", [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8])
    def test_correct_init_high_zncc(self, setup_images, setup_buffers, q):
        """정확한 변위를 초기값으로 주면 ZNCC ≈ 1.0."""
        ref, grad_x, grad_y, coeffs, sx, sy = setup_images
        buf = setup_buffers
        subset_size = 21
        cx, cy = 128, 128

        lx, ly = generate_quarter_local_coordinates(subset_size, q)
        n_pixels = len(lx)
        xsi_min, xsi_max, eta_min, eta_max = self._get_quarter_ranges(subset_size, q)

        init_params = np.array([sx, 0.0, 0.0, sy, 0.0, 0.0], dtype=np.float64)

        zncc = evaluate_quarter_zncc(
            ref, grad_x, grad_y, coeffs, 5,
            cx, cy, init_params,
            xsi_min, xsi_max, eta_min, eta_max,
            lx, ly, AFFINE, n_pixels,
            buf['f'], buf['dfdx'], buf['dfdy'],
            buf['xsi_w'], buf['eta_w'],
            buf['x_def'], buf['y_def'],
            buf['g']
        )

        assert zncc > 0.99, f"Q{q}: ZNCC={zncc:.4f}, expected > 0.99"

    # ── 잘못된 초기값 → 낮은 ZNCC ──────────────────────

    @pytest.mark.parametrize("q", [Q1, Q5, Q8])
    def test_wrong_init_low_zncc(self, setup_images, setup_buffers, q):
        """잘못된 변위(0, 0)를 초기값으로 주면 ZNCC가 낮아짐."""
        ref, grad_x, grad_y, coeffs, sx, sy = setup_images
        buf = setup_buffers
        subset_size = 21
        cx, cy = 128, 128

        lx, ly = generate_quarter_local_coordinates(subset_size, q)
        n_pixels = len(lx)
        xsi_min, xsi_max, eta_min, eta_max = self._get_quarter_ranges(subset_size, q)

        init_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)

        zncc = evaluate_quarter_zncc(
            ref, grad_x, grad_y, coeffs, 5,
            cx, cy, init_params,
            xsi_min, xsi_max, eta_min, eta_max,
            lx, ly, AFFINE, n_pixels,
            buf['f'], buf['dfdx'], buf['dfdy'],
            buf['xsi_w'], buf['eta_w'],
            buf['x_def'], buf['y_def'],
            buf['g']
        )

        assert zncc < 0.95, f"Q{q}: ZNCC={zncc:.4f}, expected < 0.95"

    # ── 경계 밖 POI → -1.0 반환 ────────────────────────

    def test_boundary_returns_negative(self, setup_images, setup_buffers):
        """POI가 이미지 경계 근처면 -1.0 반환."""
        ref, grad_x, grad_y, coeffs, sx, sy = setup_images
        buf = setup_buffers
        subset_size = 21
        cx, cy = 5, 5  # 경계에 너무 가까움

        lx, ly = generate_quarter_local_coordinates(subset_size, Q0)
        n_pixels = len(lx)
        xsi_min, xsi_max, eta_min, eta_max = self._get_quarter_ranges(subset_size, Q0)

        init_params = np.array([sx, 0.0, 0.0, sy, 0.0, 0.0], dtype=np.float64)

        zncc = evaluate_quarter_zncc(
            ref, grad_x, grad_y, coeffs, 5,
            cx, cy, init_params,
            xsi_min, xsi_max, eta_min, eta_max,
            lx, ly, AFFINE, n_pixels,
            buf['f'], buf['dfdx'], buf['dfdy'],
            buf['xsi_w'], buf['eta_w'],
            buf['x_def'], buf['y_def'],
            buf['g']
        )

        assert zncc == -1.0

    # ── Q0 vs Quarter: 정확한 초기값일 때 ZNCC 비슷 ───

    def test_quarter_zncc_similar_to_full(self, setup_images, setup_buffers):
        """정확한 초기값이면 Q1~Q8 ZNCC가 Q0과 유사."""
        ref, grad_x, grad_y, coeffs, sx, sy = setup_images
        buf = setup_buffers
        subset_size = 21
        cx, cy = 128, 128

        init_params = np.array([sx, 0.0, 0.0, sy, 0.0, 0.0], dtype=np.float64)

        # Q0 ZNCC
        lx0, ly0 = generate_quarter_local_coordinates(subset_size, Q0)
        xsi_min0, xsi_max0, eta_min0, eta_max0 = self._get_quarter_ranges(subset_size, Q0)
        zncc_full = evaluate_quarter_zncc(
            ref, grad_x, grad_y, coeffs, 5,
            cx, cy, init_params,
            xsi_min0, xsi_max0, eta_min0, eta_max0,
            lx0, ly0, AFFINE, len(lx0),
            buf['f'], buf['dfdx'], buf['dfdy'],
            buf['xsi_w'], buf['eta_w'],
            buf['x_def'], buf['y_def'],
            buf['g']
        )

        # Q1~Q8 ZNCC
        for q in [Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8]:
            lx, ly = generate_quarter_local_coordinates(subset_size, q)
            xsi_min, xsi_max, eta_min, eta_max = self._get_quarter_ranges(subset_size, q)
            zncc_q = evaluate_quarter_zncc(
                ref, grad_x, grad_y, coeffs, 5,
                cx, cy, init_params,
                xsi_min, xsi_max, eta_min, eta_max,
                lx, ly, AFFINE, len(lx),
                buf['f'], buf['dfdx'], buf['dfdy'],
                buf['xsi_w'], buf['eta_w'],
                buf['x_def'], buf['y_def'],
                buf['g']
            )
            assert abs(zncc_q - zncc_full) < 0.02, \
                f"Q{q}: ZNCC={zncc_q:.4f} vs full={zncc_full:.4f}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
