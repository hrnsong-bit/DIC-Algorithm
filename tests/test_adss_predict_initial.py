"""
predict_initial_params 함수 검증 테스트.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from speckle.core.optimization.adss_subset_numba import predict_initial_params

AFFINE = 0
QUADRATIC = 1


class TestPredictInitialParams:
    """predict_initial_params 함수 검증."""

    # ── Affine 기본 동작 ────────────────────────────────

    def test_affine_same_location(self):
        """이웃과 대상이 같은 위치 → 파라미터 그대로 복사."""
        params_k = np.array([2.5, 0.01, -0.02, 1.3, 0.03, -0.01], dtype=np.float64)
        out = np.zeros(6, dtype=np.float64)

        predict_initial_params(100.0, 200.0, params_k, 100.0, 200.0, AFFINE, out)

        np.testing.assert_allclose(out, params_k, atol=1e-15)

    def test_affine_x_offset(self):
        """x 방향으로 5픽셀 떨어진 경우 (spacing=5)."""
        # 이웃: (100, 200), 대상: (105, 200) → dx = -5, dy = 0
        params_k = np.array([2.0, 0.1, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        out = np.zeros(6, dtype=np.float64)

        predict_initial_params(100.0, 200.0, params_k, 105.0, 200.0, AFFINE, out)

        # u_init = 2.0 + 0.1 * (-5) + 0.0 * 0 = 1.5
        # v_init = 1.0 + 0.0 * (-5) + 0.0 * 0 = 1.0
        assert out[0] == pytest.approx(1.5)
        assert out[1] == pytest.approx(0.1)   # ux 복사
        assert out[2] == pytest.approx(0.0)   # uy 복사
        assert out[3] == pytest.approx(1.0)
        assert out[4] == pytest.approx(0.0)   # vx 복사
        assert out[5] == pytest.approx(0.0)   # vy 복사

    def test_affine_y_offset(self):
        """y 방향으로 5픽셀 떨어진 경우."""
        # 이웃: (100, 200), 대상: (100, 205) → dx = 0, dy = -5
        params_k = np.array([2.0, 0.0, 0.2, 1.0, 0.0, -0.1], dtype=np.float64)
        out = np.zeros(6, dtype=np.float64)

        predict_initial_params(100.0, 200.0, params_k, 100.0, 205.0, AFFINE, out)

        # u_init = 2.0 + 0.0 * 0 + 0.2 * (-5) = 1.0
        # v_init = 1.0 + 0.0 * 0 + (-0.1) * (-5) = 1.5
        assert out[0] == pytest.approx(1.0)
        assert out[3] == pytest.approx(1.5)

    def test_affine_diagonal_offset(self):
        """대각선 방향 (dx=-5, dy=-5)."""
        # 이웃: (100, 200), 대상: (105, 205)
        params_k = np.array([3.0, 0.1, 0.2, 2.0, -0.1, 0.05], dtype=np.float64)
        out = np.zeros(6, dtype=np.float64)

        predict_initial_params(100.0, 200.0, params_k, 105.0, 205.0, AFFINE, out)

        dx, dy = -5.0, -5.0
        expected_u = 3.0 + 0.1 * dx + 0.2 * dy   # 3.0 - 0.5 - 1.0 = 1.5
        expected_v = 2.0 + (-0.1) * dx + 0.05 * dy  # 2.0 + 0.5 - 0.25 = 2.25

        assert out[0] == pytest.approx(expected_u)
        assert out[3] == pytest.approx(expected_v)
        assert out[1] == pytest.approx(0.1)
        assert out[2] == pytest.approx(0.2)
        assert out[4] == pytest.approx(-0.1)
        assert out[5] == pytest.approx(0.05)

    # ── Affine 변위 0 (무변형) ──────────────────────────

    def test_affine_zero_displacement(self):
        """변위·변형률 모두 0이면 출력도 0."""
        params_k = np.zeros(6, dtype=np.float64)
        out = np.zeros(6, dtype=np.float64)

        predict_initial_params(50.0, 50.0, params_k, 55.0, 55.0, AFFINE, out)

        np.testing.assert_allclose(out, 0.0, atol=1e-15)

    # ── Affine 균일 변위 (gradient 0) ───────────────────

    def test_affine_uniform_displacement(self):
        """균일 변위 (gradient 0)이면 위치에 무관하게 동일."""
        params_k = np.array([5.0, 0.0, 0.0, -3.0, 0.0, 0.0], dtype=np.float64)
        out = np.zeros(6, dtype=np.float64)

        predict_initial_params(10.0, 20.0, params_k, 100.0, 200.0, AFFINE, out)

        assert out[0] == pytest.approx(5.0)
        assert out[3] == pytest.approx(-3.0)

    # ── Quadratic 기본 동작 ─────────────────────────────

    def test_quadratic_same_location(self):
        """Quadratic: 같은 위치 → 파라미터 그대로 복사."""
        params_k = np.array([
            1.0, 0.01, -0.02, 0.001, 0.002, -0.001,  # u 계열
            2.0, 0.03, -0.01, 0.003, -0.002, 0.001    # v 계열
        ], dtype=np.float64)
        out = np.zeros(12, dtype=np.float64)

        predict_initial_params(50.0, 50.0, params_k, 50.0, 50.0, QUADRATIC, out)

        np.testing.assert_allclose(out, params_k, atol=1e-15)

    def test_quadratic_x_offset(self):
        """Quadratic: x 방향 오프셋."""
        params_k = np.array([
            2.0, 0.1, 0.0, 0.0, 0.0, 0.0,  # u, ux, uy, uxx, uxy, uyy
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0   # v, vx, vy, vxx, vxy, vyy
        ], dtype=np.float64)
        out = np.zeros(12, dtype=np.float64)

        predict_initial_params(100.0, 200.0, params_k, 105.0, 200.0, QUADRATIC, out)

        # u_init = 2.0 + 0.1 * (-5) = 1.5
        assert out[0] == pytest.approx(1.5)
        assert out[1] == pytest.approx(0.1)   # ux 복사
        assert out[6] == pytest.approx(1.0)   # v_init 변화 없음

    def test_quadratic_higher_order_copied(self):
        """Quadratic: 2차 항 (uxx, uxy, uyy, vxx, vxy, vyy)은 그대로 복사."""
        params_k = np.array([
            0.0, 0.0, 0.0, 0.11, 0.22, 0.33,
            0.0, 0.0, 0.0, 0.44, 0.55, 0.66
        ], dtype=np.float64)
        out = np.zeros(12, dtype=np.float64)

        predict_initial_params(50.0, 50.0, params_k, 55.0, 55.0, QUADRATIC, out)

        assert out[3] == pytest.approx(0.11)
        assert out[4] == pytest.approx(0.22)
        assert out[5] == pytest.approx(0.33)
        assert out[9] == pytest.approx(0.44)
        assert out[10] == pytest.approx(0.55)
        assert out[11] == pytest.approx(0.66)

    # ── 출력 배열 덮어쓰기 검증 ─────────────────────────

    def test_output_overwrites_existing(self):
        """out_params에 기존 값이 있어도 완전히 덮어씀."""
        params_k = np.array([1.0, 0.0, 0.0, 2.0, 0.0, 0.0], dtype=np.float64)
        out = np.full(6, 999.0, dtype=np.float64)

        predict_initial_params(50.0, 50.0, params_k, 50.0, 50.0, AFFINE, out)

        assert out[0] == pytest.approx(1.0)
        assert out[3] == pytest.approx(2.0)

    # ── 부호 방향 검증 (Eq.3 핵심) ──────────────────────

    def test_sign_convention(self):
        """
        dx = x⁽ᵏ⁾ - x 부호 검증.
        이웃이 왼쪽(x=95), 대상이 오른쪽(x=100) → dx = -5.
        ux > 0이면 u_init < u⁽ᵏ⁾.
        """
        params_k = np.array([10.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        out = np.zeros(6, dtype=np.float64)

        predict_initial_params(95.0, 100.0, params_k, 100.0, 100.0, AFFINE, out)

        # u_init = 10.0 + 0.2 * (95-100) = 10.0 - 1.0 = 9.0
        assert out[0] == pytest.approx(9.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
