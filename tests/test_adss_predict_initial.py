"""
predict_initial_params 함수 검증 테스트.

Taylor 전개 부호 convention:
    dx = target_x - neighbor_x = x - x⁽ᵏ⁾
    dy = target_y - neighbor_y = y - y⁽ᵏ⁾

    u_init = u⁽ᵏ⁾ + ux⁽ᵏ⁾·dx + uy⁽ᵏ⁾·dy
    v_init = v⁽ᵏ⁾ + vx⁽ᵏ⁾·dx + vy⁽ᵏ⁾·dy

Reference:
    Pan, B., Li, K. & Tong, W., Experimental Mechanics 53, 1277–1289, 2013, Eq.(8).
    Zhao, J. & Pan, B., Experimental Mechanics 66(2), 417–432, 2025, Eq.(3).
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
        """
        x 방향으로 5픽셀 떨어진 경우 (spacing=5).
        이웃: (100, 200), 대상: (105, 200)
        dx = 105 - 100 = +5, dy = 0

        u_init = 2.0 + 0.1 * 5 = 2.5
        v_init = 1.0 + 0.0 * 5 = 1.0
        """
        params_k = np.array([2.0, 0.1, 0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        out = np.zeros(6, dtype=np.float64)

        predict_initial_params(100.0, 200.0, params_k, 105.0, 200.0, AFFINE, out)

        assert out[0] == pytest.approx(2.5)
        assert out[1] == pytest.approx(0.1)   # ux 복사
        assert out[2] == pytest.approx(0.0)   # uy 복사
        assert out[3] == pytest.approx(1.0)
        assert out[4] == pytest.approx(0.0)   # vx 복사
        assert out[5] == pytest.approx(0.0)   # vy 복사

    def test_affine_y_offset(self):
        """
        y 방향으로 5픽셀 떨어진 경우.
        이웃: (100, 200), 대상: (100, 205)
        dx = 0, dy = 205 - 200 = +5

        u_init = 2.0 + 0.2 * 5 = 3.0
        v_init = 1.0 + (-0.1) * 5 = 0.5
        """
        params_k = np.array([2.0, 0.0, 0.2, 1.0, 0.0, -0.1], dtype=np.float64)
        out = np.zeros(6, dtype=np.float64)

        predict_initial_params(100.0, 200.0, params_k, 100.0, 205.0, AFFINE, out)

        assert out[0] == pytest.approx(3.0)
        assert out[3] == pytest.approx(0.5)

    def test_affine_diagonal_offset(self):
        """
        대각선 방향.
        이웃: (100, 200), 대상: (105, 205)
        dx = +5, dy = +5

        u_init = 3.0 + 0.1 * 5 + 0.2 * 5 = 3.0 + 0.5 + 1.0 = 4.5
        v_init = 2.0 + (-0.1) * 5 + 0.05 * 5 = 2.0 - 0.5 + 0.25 = 1.75
        """
        params_k = np.array([3.0, 0.1, 0.2, 2.0, -0.1, 0.05], dtype=np.float64)
        out = np.zeros(6, dtype=np.float64)

        predict_initial_params(100.0, 200.0, params_k, 105.0, 205.0, AFFINE, out)

        dx, dy = 5.0, 5.0
        expected_u = 3.0 + 0.1 * dx + 0.2 * dy     # 4.5
        expected_v = 2.0 + (-0.1) * dx + 0.05 * dy  # 1.75

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
        """
        Quadratic: x 방향 오프셋.
        이웃: (100, 200), 대상: (105, 200)
        dx = +5, dy = 0

        u_init = 2.0 + 0.1 * 5 = 2.5
        v_init = 1.0 (변화 없음)
        """
        params_k = np.array([
            2.0, 0.1, 0.0, 0.0, 0.0, 0.0,  # u, ux, uy, uxx, uxy, uyy
            1.0, 0.0, 0.0, 0.0, 0.0, 0.0   # v, vx, vy, vxx, vxy, vyy
        ], dtype=np.float64)
        out = np.zeros(12, dtype=np.float64)

        predict_initial_params(100.0, 200.0, params_k, 105.0, 200.0, QUADRATIC, out)

        assert out[0] == pytest.approx(2.5)
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
        부호 검증: 표준 Taylor 전개 dx = x_target - x_neighbor.

        이웃이 왼쪽(x=95), 대상이 오른쪽(x=100)
        → dx = 100 - 95 = +5.
        ux = ∂u/∂x > 0이면 x가 증가할수록 u도 증가하므로,
        대상이 이웃보다 오른쪽에 있으면 u_init > u⁽ᵏ⁾.

        u_init = 10.0 + 0.2 * 5 = 11.0
        """
        params_k = np.array([10.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        out = np.zeros(6, dtype=np.float64)

        predict_initial_params(95.0, 100.0, params_k, 100.0, 100.0, AFFINE, out)

        assert out[0] == pytest.approx(11.0)

    # ── 추가: 물리적 일관성 검증 ────────────────────────

    def test_physical_consistency_tension(self):
        """
        물리적 검증: 균일 인장 (ux = 0.01).
        x가 커질수록 u가 선형 증가해야 함.

        이웃: (100, 100), u = 1.0, ux = 0.01
        대상: (110, 100), dx = +10
        u_init = 1.0 + 0.01 * 10 = 1.1 > u⁽ᵏ⁾
        """
        params_k = np.array([1.0, 0.01, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        out = np.zeros(6, dtype=np.float64)

        predict_initial_params(100.0, 100.0, params_k, 110.0, 100.0, AFFINE, out)

        assert out[0] == pytest.approx(1.1)
        assert out[0] > params_k[0]  # 인장: 대상이 오른쪽이면 변위 더 큼

    def test_physical_consistency_shear(self):
        """
        물리적 검증: 단순 전단 (uy = 0.02).
        y가 커질수록 u가 선형 증가해야 함.

        이웃: (50, 50), u = 0.5, uy = 0.02
        대상: (50, 60), dy = +10
        u_init = 0.5 + 0.02 * 10 = 0.7
        """
        params_k = np.array([0.5, 0.0, 0.02, 0.0, 0.0, 0.0], dtype=np.float64)
        out = np.zeros(6, dtype=np.float64)

        predict_initial_params(50.0, 50.0, params_k, 50.0, 60.0, AFFINE, out)

        assert out[0] == pytest.approx(0.7)

    def test_physical_consistency_symmetric(self):
        """
        대칭성 검증: 대상↔이웃을 바꾸면 외삽 방향이 반대.

        이웃 A→대상 B: dx = +5  → u_init = 1.0 + 0.1 * 5 = 1.5
        이웃 B→대상 A: dx = -5  → u_init = 1.5 + 0.1 * (-5) = 1.0

        즉 A에서 B를 예측한 뒤, B에서 A를 역예측하면 원래 값으로 복원.
        """
        params_a = np.array([1.0, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        out_b = np.zeros(6, dtype=np.float64)

        # A(100) → B(105) 예측
        predict_initial_params(100.0, 200.0, params_a, 105.0, 200.0, AFFINE, out_b)
        assert out_b[0] == pytest.approx(1.5)

        # B(105) → A(100) 역예측 (gradient 동일 가정)
        out_a = np.zeros(6, dtype=np.float64)
        predict_initial_params(105.0, 200.0, out_b, 100.0, 200.0, AFFINE, out_a)
        assert out_a[0] == pytest.approx(1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
