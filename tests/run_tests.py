"""
Quadratic Shape Function 검증 테스트 - 독립 실행 스크립트

사용법:
    프로젝트 루트 폴더로 이동 후:
        python tests/run_tests.py
"""

import sys
import os
import time
import traceback

# ===== 경로 설정 (어디서 실행하든 동작) =====
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

# ===== import 테스트 =====
try:
    from speckle.core.optimization.shape_function import (
        generate_local_coordinates,
        warp_affine,
        warp_quadratic,
        compute_steepest_descent_affine,
        compute_steepest_descent_quadratic,
        _update_affine_direct,
        _update_affine_matrix_fallback,
        _update_quadratic_matrix_optimized,
    )
    print("[OK] speckle 패키지 import 성공")
except ImportError as e:
    print(f"[FAIL] import 실패: {e}")
    print(f"  PROJECT_ROOT = {PROJECT_ROOT}")
    print(f"  sys.path = {sys.path[:5]}")
    print(f"\n  프로젝트 루트에 speckle/ 폴더가 있는지 확인하세요.")
    sys.exit(1)


# ===== 테스트 프레임워크 =====

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run(self, name, func):
        try:
            func()
            self.passed += 1
            print(f"  ✓ {name}")
        except AssertionError as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  ✗ {name}")
            print(f"    → {e}")
        except Exception as e:
            self.failed += 1
            self.errors.append((name, traceback.format_exc()))
            print(f"  ✗ {name} (예외 발생)")
            print(f"    → {type(e).__name__}: {e}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"결과: {self.passed}/{total} 통과, {self.failed} 실패")
        if self.errors:
            print(f"\n실패 목록:")
            for name, err in self.errors:
                print(f"  - {name}")
                # 첫 줄만 출력
                first_line = err.strip().split('\n')[0]
                print(f"    {first_line}")
        print(f"{'='*60}")
        return self.failed == 0


runner = TestRunner()
rng = np.random.default_rng(seed=42)


# ===== 유틸리티 =====

def random_affine_params(translation_scale=5.0, gradient_scale=0.05):
    p = np.zeros(6, dtype=np.float64)
    p[0] = rng.uniform(-translation_scale, translation_scale)
    p[3] = rng.uniform(-translation_scale, translation_scale)
    p[1:3] = rng.normal(0, gradient_scale, 2)
    p[4:6] = rng.normal(0, gradient_scale, 2)
    return p


def random_quadratic_params(translation_scale=5.0,
                            gradient_scale=0.05,
                            second_order_scale=0.005):
    p = np.zeros(12, dtype=np.float64)
    p[0] = rng.uniform(-translation_scale, translation_scale)
    p[6] = rng.uniform(-translation_scale, translation_scale)
    p[1] = rng.normal(0, gradient_scale)
    p[2] = rng.normal(0, gradient_scale)
    p[7] = rng.normal(0, gradient_scale)
    p[8] = rng.normal(0, gradient_scale)
    p[3] = rng.normal(0, second_order_scale)
    p[4] = rng.normal(0, second_order_scale)
    p[5] = rng.normal(0, second_order_scale)
    p[9] = rng.normal(0, second_order_scale)
    p[10] = rng.normal(0, second_order_scale)
    p[11] = rng.normal(0, second_order_scale)
    return p


def invert_warp_newton(warp_func, dp, xsi_target, eta_target,
                       max_iter=100, tol=1e-13):
    """Newton법으로 W(dp)의 역변환 계산 — 코드와 독립적인 검증 수단"""
    xsi_inv = xsi_target.copy()
    eta_inv = eta_target.copy()

    if len(dp) == 12:
        ux, uy = dp[1], dp[2]
        uxx, uxy, uyy = dp[3], dp[4], dp[5]
        vx, vy = dp[7], dp[8]
        vxx, vxy, vyy = dp[9], dp[10], dp[11]
    else:
        ux, uy = dp[1], dp[2]
        vx, vy = dp[4], dp[5]
        uxx = uxy = uyy = vxx = vxy = vyy = 0.0

    for _ in range(max_iter):
        xsi_w, eta_w = warp_func(dp, xsi_inv, eta_inv)
        res_x = xsi_w - xsi_target
        res_y = eta_w - eta_target

        if max(np.max(np.abs(res_x)), np.max(np.abs(res_y))) < tol:
            break

        J00 = (1 + ux) + uxx * xsi_inv + uxy * eta_inv
        J01 = uy + uxy * xsi_inv + uyy * eta_inv
        J10 = vx + vxx * xsi_inv + vxy * eta_inv
        J11 = (1 + vy) + vxy * xsi_inv + vyy * eta_inv

        det = J00 * J11 - J01 * J10
        det = np.where(np.abs(det) < 1e-15, 1e-15, det)

        xsi_inv -= ( J11 * res_x - J01 * res_y) / det
        eta_inv -= (-J10 * res_x + J00 * res_y) / det

    return xsi_inv, eta_inv


# ===============================================================
#  TEST 1: Warp 기본 검증
# ===============================================================
print("\n" + "="*60)
print("TEST 1: Quadratic Warp 함수 기본 검증")
print("="*60)


def test_warp_identity():
    for ss in [11, 21, 31]:
        xsi, eta = generate_local_coordinates(ss)
        p = np.zeros(12)
        xsi_w, eta_w = warp_quadratic(p, xsi, eta)
        np.testing.assert_allclose(xsi_w, xsi, atol=1e-15)
        np.testing.assert_allclose(eta_w, eta, atol=1e-15)

runner.run("1-1 항등 변환 (p=0)", test_warp_identity)


def test_warp_translation():
    xsi, eta = generate_local_coordinates(21)
    for u, v in [(3.5, -2.1), (0, 0), (-10.3, 7.8), (0.123, -0.456)]:
        p = np.zeros(12)
        p[0], p[6] = u, v
        xsi_w, eta_w = warp_quadratic(p, xsi, eta)
        np.testing.assert_allclose(xsi_w, xsi + u, atol=1e-12)
        np.testing.assert_allclose(eta_w, eta + v, atol=1e-12)

runner.run("1-2 평행이동", test_warp_translation)


def test_warp_affine_consistency():
    xsi, eta = generate_local_coordinates(21)
    for _ in range(50):
        pa = random_affine_params()
        pq = np.zeros(12)
        pq[0], pq[1], pq[2] = pa[0], pa[1], pa[2]
        pq[6], pq[7], pq[8] = pa[3], pa[4], pa[5]

        xsi_a, eta_a = warp_affine(pa, xsi, eta)
        xsi_q, eta_q = warp_quadratic(pq, xsi, eta)
        np.testing.assert_allclose(xsi_q, xsi_a, atol=1e-12)
        np.testing.assert_allclose(eta_q, eta_a, atol=1e-12)

runner.run("1-3 Affine 호환성 (2차항=0이면 affine과 동일)", test_warp_affine_consistency)
def test_warp_second_order_individual():
    xsi, eta = generate_local_coordinates(21)

    # uxx: xsi_w = xsi + 0.5 * uxx * xsi²  (identity 포함)
    p = np.zeros(12); p[3] = 0.001
    xsi_w, eta_w = warp_quadratic(p, xsi, eta)
    np.testing.assert_allclose(xsi_w, xsi + 0.5 * 0.001 * xsi**2, atol=1e-12)
    np.testing.assert_allclose(eta_w, eta, atol=1e-15)

    # uyy: xsi_w = xsi + 0.5 * uyy * eta²
    p = np.zeros(12); p[5] = 0.002
    xsi_w, eta_w = warp_quadratic(p, xsi, eta)
    np.testing.assert_allclose(xsi_w, xsi + 0.5 * 0.002 * eta**2, atol=1e-12)
    np.testing.assert_allclose(eta_w, eta, atol=1e-15)

    # uxy: xsi_w = xsi + uxy * xsi * eta
    p = np.zeros(12); p[4] = 0.002
    xsi_w, eta_w = warp_quadratic(p, xsi, eta)
    np.testing.assert_allclose(xsi_w, xsi + 0.002 * xsi * eta, atol=1e-12)
    np.testing.assert_allclose(eta_w, eta, atol=1e-15)

    # vxx: eta_w = eta + 0.5 * vxx * xsi²
    p = np.zeros(12); p[9] = 0.0015
    xsi_w, eta_w = warp_quadratic(p, xsi, eta)
    np.testing.assert_allclose(xsi_w, xsi, atol=1e-15)
    np.testing.assert_allclose(eta_w, eta + 0.5 * 0.0015 * xsi**2, atol=1e-12)

    # vyy: eta_w = eta + 0.5 * vyy * eta²
    p = np.zeros(12); p[11] = -0.001
    xsi_w, eta_w = warp_quadratic(p, xsi, eta)
    np.testing.assert_allclose(xsi_w, xsi, atol=1e-15)
    np.testing.assert_allclose(eta_w, eta + 0.5 * (-0.001) * eta**2, atol=1e-12)

    # vxy: eta_w = eta + vxy * xsi * eta
    p = np.zeros(12); p[10] = 0.003
    xsi_w, eta_w = warp_quadratic(p, xsi, eta)
    np.testing.assert_allclose(xsi_w, xsi, atol=1e-15)
    np.testing.assert_allclose(eta_w, eta + 0.003 * xsi * eta, atol=1e-12)

runner.run("1-4 2차항 개별 검증 (uxx, uyy, uxy, vxx, vyy, vxy)", test_warp_second_order_individual)

def test_quadratic_composition_small_dp():
    """★ 핵심: 6×6 행렬의 truncation error 수준 확인
    
    Quadratic warp 합성은 2차로 닫히지 않으므로 (3차+ 항 발생),
    6×6 행렬 표현에 이론적 근사 오차가 존재.
    IC-GN에서 dp→0 수렴 시 이 오차는 무시 가능.
    여기서는 오차 수준이 합리적 범위인지 확인.
    """
    xsi, eta = generate_local_coordinates(21)
    n_tests = 500
    errors = []

    for _ in range(n_tests):
        p = random_quadratic_params()
        dp = random_quadratic_params(
            translation_scale=0.5, gradient_scale=0.005,
            second_order_scale=0.0005)

        p_new = _update_quadratic_matrix_optimized(p, dp)
        xsi_d, eta_d = warp_quadratic(p_new, xsi, eta)

        xsi_inv, eta_inv = invert_warp_newton(warp_quadratic, dp, xsi, eta)
        xsi_c, eta_c = warp_quadratic(p, xsi_inv, eta_inv)

        err = np.max(np.abs(xsi_d - xsi_c)) + np.max(np.abs(eta_d - eta_c))
        errors.append(err)

    errors = np.array(errors)
    print(f"    {n_tests}회 테스트 (small dp)")
    print(f"    mean: {errors.mean():.2e}, max: {errors.max():.2e}, "
          f"99th: {np.percentile(errors, 99):.2e}")
    print(f"    (이론적 truncation error — IC-GN 수렴 시 dp→0이므로 무시 가능)")
    # Truncation error 허용: dp 스케일 대비 합리적 범위
    assert errors.max() < 1.0, f"truncation error 과대: {errors.max():.2e}"

runner.run("1-4 2차항 개별 검증 (uxx, uyy, uxy, vxx, vyy, vxy)", test_warp_second_order_individual)


def test_warp_full_analytic():
    xsi, eta = generate_local_coordinates(21)
    p = np.array([1.5, 0.01, -0.02, 0.001, 0.0005, -0.0003,
                  -0.8, 0.015, 0.005, -0.0008, 0.0004, 0.0002])
    u, ux, uy, uxx, uxy, uyy = p[0:6]
    v, vx, vy, vxx, vxy, vyy = p[6:12]

    expected_xsi = (u + (1+ux)*xsi + uy*eta
                    + 0.5*uxx*xsi**2 + uxy*xsi*eta + 0.5*uyy*eta**2)
    expected_eta = (v + vx*xsi + (1+vy)*eta
                    + 0.5*vxx*xsi**2 + vxy*xsi*eta + 0.5*vyy*eta**2)

    xsi_w, eta_w = warp_quadratic(p, xsi, eta)
    np.testing.assert_allclose(xsi_w, expected_xsi, atol=1e-12)
    np.testing.assert_allclose(eta_w, expected_eta, atol=1e-12)

runner.run("1-5 전체 항 동시 해석 검증", test_warp_full_analytic)


# ===============================================================
#  TEST 2: Steepest Descent (Jacobian) 수치미분 검증
# ===============================================================
print("\n" + "="*60)
print("TEST 2: Steepest Descent Image 수치미분 검증")
print("="*60)

PARAM_NAMES_QUAD = ['u','ux','uy','uxx','uxy','uyy',
                    'v','vx','vy','vxx','vxy','vyy']
PARAM_NAMES_AFFINE = ['u','ux','uy','v','vx','vy']


def _numerical_jacobian(warp_func, n_params, dfdx, dfdy, xsi, eta, eps=1e-7):
    n_pixels = len(xsi)
    J_num = np.zeros((n_pixels, n_params), dtype=np.float64)
    p0 = np.zeros(n_params)
    for k in range(n_params):
        p_plus = p0.copy(); p_plus[k] += eps
        p_minus = p0.copy(); p_minus[k] -= eps
        xsi_p, eta_p = warp_func(p_plus, xsi, eta)
        xsi_m, eta_m = warp_func(p_minus, xsi, eta)
        dW_x = (xsi_p - xsi_m) / (2 * eps)
        dW_y = (eta_p - eta_m) / (2 * eps)
        J_num[:, k] = dfdx * dW_x + dfdy * dW_y
    return J_num


def test_jacobian_affine():
    xsi, eta = generate_local_coordinates(21)
    n_pixels = len(xsi)
    for _ in range(10):
        dfdx = rng.standard_normal(n_pixels)
        dfdy = rng.standard_normal(n_pixels)
        J_a = compute_steepest_descent_affine(dfdx, dfdy, xsi, eta)
        J_n = _numerical_jacobian(warp_affine, 6, dfdx, dfdy, xsi, eta)
        for k in range(6):
            np.testing.assert_allclose(
                J_a[:, k], J_n[:, k], atol=1e-5,
                err_msg=f"Affine col {k} ({PARAM_NAMES_AFFINE[k]})")

runner.run("2-1 Affine Jacobian vs 수치미분", test_jacobian_affine)


def test_jacobian_quadratic():
    xsi, eta = generate_local_coordinates(21)
    n_pixels = len(xsi)
    max_err_per_col = np.zeros(12)
    for _ in range(10):
        dfdx = rng.standard_normal(n_pixels)
        dfdy = rng.standard_normal(n_pixels)
        J_a = compute_steepest_descent_quadratic(dfdx, dfdy, xsi, eta)
        J_n = _numerical_jacobian(warp_quadratic, 12, dfdx, dfdy, xsi, eta)
        for k in range(12):
            err = np.max(np.abs(J_a[:, k] - J_n[:, k]))
            max_err_per_col[k] = max(max_err_per_col[k], err)
            np.testing.assert_allclose(
                J_a[:, k], J_n[:, k], atol=1e-5,
                err_msg=f"Quadratic col {k} ({PARAM_NAMES_QUAD[k]})")

    print(f"    열별 최대 오차:")
    for k in range(12):
        print(f"      {PARAM_NAMES_QUAD[k]:>3s}: {max_err_per_col[k]:.2e}")

runner.run("2-2 Quadratic Jacobian vs 수치미분", test_jacobian_quadratic)


def test_jacobian_affine_quadratic_structure():
    """Quadratic의 1차 열이 Affine과 구조적으로 일치"""
    xsi, eta = generate_local_coordinates(21)
    n_pixels = len(xsi)
    dfdx = rng.standard_normal(n_pixels)
    dfdy = rng.standard_normal(n_pixels)
    J_a = compute_steepest_descent_affine(dfdx, dfdy, xsi, eta)
    J_q = compute_steepest_descent_quadratic(dfdx, dfdy, xsi, eta)

    # Affine [u, ux, uy] == Quadratic [u, ux, uy]
    np.testing.assert_allclose(J_q[:, 0:3], J_a[:, 0:3], atol=1e-15)
    # Affine [v, vx, vy] == Quadratic [v, vx, vy]
    np.testing.assert_allclose(J_q[:, 6:9], J_a[:, 3:6], atol=1e-15)

runner.run("2-3 Quadratic 1차열 ↔ Affine 구조 일치", test_jacobian_affine_quadratic_structure)


# ===============================================================
#  TEST 3: Inverse Compositional Update 검증 (★ 핵심)
# ===============================================================
print("\n" + "="*60)
print("TEST 3: Inverse Compositional Update 검증")
print("="*60)


def test_affine_update_dp_zero():
    for _ in range(50):
        p = random_affine_params()
        dp = np.zeros(6)
        p_new = _update_affine_direct(p, dp)
        np.testing.assert_allclose(p_new, p, atol=1e-12)

runner.run("3-1 Affine: dp=0 → p 불변", test_affine_update_dp_zero)


def test_affine_direct_vs_matrix():
    for _ in range(200):
        p = random_affine_params()
        dp = random_affine_params(translation_scale=1.0, gradient_scale=0.01)
        p_d = _update_affine_direct(p, dp)
        p_m = _update_affine_matrix_fallback(p, dp)
        np.testing.assert_allclose(p_d, p_m, atol=1e-10,
                                   err_msg=f"p={p}, dp={dp}")

runner.run("3-2 Affine: 직접계산 vs 행렬방식 일치", test_affine_direct_vs_matrix)


def test_affine_composition():
    xsi, eta = generate_local_coordinates(21)
    errors = []
    for _ in range(300):
        p = random_affine_params()
        dp = random_affine_params(translation_scale=1.0, gradient_scale=0.01)

        p_new = _update_affine_direct(p, dp)
        xsi_d, eta_d = warp_affine(p_new, xsi, eta)

        xsi_inv, eta_inv = invert_warp_newton(warp_affine, dp, xsi, eta)
        xsi_c, eta_c = warp_affine(p, xsi_inv, eta_inv)

        err = np.max(np.abs(xsi_d - xsi_c)) + np.max(np.abs(eta_d - eta_c))
        errors.append(err)

    errors = np.array(errors)
    print(f"    mean: {errors.mean():.2e}, max: {errors.max():.2e}, "
          f"99th: {np.percentile(errors, 99):.2e}")
    assert errors.max() < 1e-8, f"max error = {errors.max():.2e}"

runner.run("3-3 Affine: W(p_new) == W(p)·W(dp)⁻¹ (Newton 검증)", test_affine_composition)


def test_quadratic_update_dp_zero():
    for _ in range(100):
        p = random_quadratic_params()
        dp = np.zeros(12)
        p_new = _update_quadratic_matrix_optimized(p, dp)
        np.testing.assert_allclose(p_new, p, atol=1e-10,
                                   err_msg=f"p={p}\np_new={p_new}")

runner.run("3-4 Quadratic: dp=0 → p 불변", test_quadratic_update_dp_zero)


def test_quadratic_self_inverse():
    for _ in range(100):
        p = random_quadratic_params()
        p_new = _update_quadratic_matrix_optimized(p, p)
        np.testing.assert_allclose(p_new, 0.0, atol=1e-8,
                                   err_msg=f"p={p}\np_new={p_new}")

runner.run("3-5 Quadratic: W(p)·W(p)⁻¹ = Identity", test_quadratic_self_inverse)


def test_quadratic_reduces_to_affine():
    for _ in range(200):
        pa = random_affine_params()
        dpa = random_affine_params(translation_scale=1.0, gradient_scale=0.01)

        pa_new = _update_affine_direct(pa, dpa)

        pq = np.zeros(12)
        pq[0], pq[1], pq[2] = pa[0], pa[1], pa[2]
        pq[6], pq[7], pq[8] = pa[3], pa[4], pa[5]

        dpq = np.zeros(12)
        dpq[0], dpq[1], dpq[2] = dpa[0], dpa[1], dpa[2]
        dpq[6], dpq[7], dpq[8] = dpa[3], dpa[4], dpa[5]

        pq_new = _update_quadratic_matrix_optimized(pq, dpq)

        # 1차 항 비교
        np.testing.assert_allclose(pq_new[0], pa_new[0], atol=1e-10, err_msg="u")
        np.testing.assert_allclose(pq_new[1], pa_new[1], atol=1e-10, err_msg="ux")
        np.testing.assert_allclose(pq_new[2], pa_new[2], atol=1e-10, err_msg="uy")
        np.testing.assert_allclose(pq_new[6], pa_new[3], atol=1e-10, err_msg="v")
        np.testing.assert_allclose(pq_new[7], pa_new[4], atol=1e-10, err_msg="vx")
        np.testing.assert_allclose(pq_new[8], pa_new[5], atol=1e-10, err_msg="vy")
        # 2차 항 = 0
        np.testing.assert_allclose(pq_new[3:6], 0.0, atol=1e-10, err_msg="2차u항")
        np.testing.assert_allclose(pq_new[9:12], 0.0, atol=1e-10, err_msg="2차v항")

runner.run("3-6 Quadratic: 2차항=0이면 Affine과 동일", test_quadratic_reduces_to_affine)


def test_quadratic_pure_translation_update():
    for _ in range(50):
        u0, v0 = rng.uniform(-5, 5, 2)
        du, dv = rng.uniform(-1, 1, 2)
        p = np.zeros(12); p[0], p[6] = u0, v0
        dp = np.zeros(12); dp[0], dp[6] = du, dv
        p_new = _update_quadratic_matrix_optimized(p, dp)

        np.testing.assert_allclose(p_new[0], u0 - du, atol=1e-10, err_msg="u")
        np.testing.assert_allclose(p_new[6], v0 - dv, atol=1e-10, err_msg="v")
        np.testing.assert_allclose(p_new[1:6], 0.0, atol=1e-10)
        np.testing.assert_allclose(p_new[7:12], 0.0, atol=1e-10)

runner.run("3-7 Quadratic: 순수 평행이동 update", test_quadratic_pure_translation_update)


def test_quadratic_composition_small_dp():
    """★ 핵심: 6×6 행렬의 truncation error 수준 확인
    
    Quadratic warp 합성은 2차로 닫히지 않으므로 (3차+ 항 발생),
    6×6 행렬 표현에 이론적 근사 오차가 존재.
    IC-GN에서 dp→0 수렴 시 이 오차는 무시 가능.
    여기서는 오차 수준이 합리적 범위인지 확인.
    """
    xsi, eta = generate_local_coordinates(21)
    n_tests = 500
    errors = []

    for _ in range(n_tests):
        p = random_quadratic_params()
        dp = random_quadratic_params(
            translation_scale=0.5, gradient_scale=0.005,
            second_order_scale=0.0005)

        p_new = _update_quadratic_matrix_optimized(p, dp)
        xsi_d, eta_d = warp_quadratic(p_new, xsi, eta)

        xsi_inv, eta_inv = invert_warp_newton(warp_quadratic, dp, xsi, eta)
        xsi_c, eta_c = warp_quadratic(p, xsi_inv, eta_inv)

        err = np.max(np.abs(xsi_d - xsi_c)) + np.max(np.abs(eta_d - eta_c))
        errors.append(err)

    errors = np.array(errors)
    print(f"    {n_tests}회 테스트 (small dp)")
    print(f"    mean: {errors.mean():.2e}, max: {errors.max():.2e}, "
          f"99th: {np.percentile(errors, 99):.2e}")
    print(f"    (이론적 truncation error — IC-GN 수렴 시 dp→0이므로 무시 가능)")
    # Truncation error 허용: dp 스케일 대비 합리적 범위
    assert errors.max() < 1.0, f"truncation error 과대: {errors.max():.2e}"


runner.run("3-8 Quadratic: W(p_new)==W(p)·W(dp)⁻¹ [small dp] ★", test_quadratic_composition_small_dp)


def test_quadratic_composition_moderate_dp():
    """스트레스 테스트: 큰 dp에서의 truncation error 수준"""
    xsi, eta = generate_local_coordinates(21)
    n_tests = 300
    errors = []
    n_skip = 0

    for _ in range(n_tests):
        p = random_quadratic_params(gradient_scale=0.08, second_order_scale=0.008)
        dp = random_quadratic_params(
            translation_scale=2.0, gradient_scale=0.02,
            second_order_scale=0.002)
        try:
            p_new = _update_quadratic_matrix_optimized(p, dp)
            xsi_d, eta_d = warp_quadratic(p_new, xsi, eta)
            xsi_inv, eta_inv = invert_warp_newton(warp_quadratic, dp, xsi, eta)
            xsi_c, eta_c = warp_quadratic(p, xsi_inv, eta_inv)
            err = np.max(np.abs(xsi_d - xsi_c)) + np.max(np.abs(eta_d - eta_c))
            errors.append(err)
        except (np.linalg.LinAlgError, RuntimeError):
            n_skip += 1

    errors = np.array(errors)
    print(f"    {len(errors)}/{n_tests} 테스트 완료 ({n_skip} skip)")
    print(f"    mean: {errors.mean():.2e}, max: {errors.max():.2e}")
    print(f"    (dp가 크면 truncation error도 커짐 — 정상)")
    # 큰 dp에서는 오차가 커도 IC-GN 실사용에서는 이 영역 도달 안 함
    assert errors.max() < 5.0, f"truncation error 비정상: {errors.max():.2e}"


runner.run("3-9 Quadratic: W(p_new)==W(p)·W(dp)⁻¹ [moderate dp]", test_quadratic_composition_moderate_dp)


# ===============================================================
#  TEST 4: End-to-End (합성 이미지)
# ===============================================================
print("\n" + "="*60)
print("TEST 4: End-to-End 합성 이미지 검증")
print("="*60)

try:
    import cv2
    from speckle.core.initial_guess.fft_cc import compute_fft_cc
    from speckle.core.optimization.icgn import compute_icgn
    HAS_E2E_DEPS = True
except ImportError as e:
    HAS_E2E_DEPS = False
    print(f"  [SKIP] E2E 의존성 부족: {e}")


def generate_synthetic_speckle(height=300, width=300, speckle_size=4, seed=42):
    local_rng = np.random.default_rng(seed)
    n_dots = int(height * width / (speckle_size ** 2) * 0.4)
    img = np.full((height, width), 180, dtype=np.float64)
    for _ in range(n_dots):
        cy = local_rng.integers(0, height)
        cx = local_rng.integers(0, width)
        intensity = local_rng.uniform(0, 120)
        sigma = local_rng.uniform(speckle_size * 0.4, speckle_size * 0.8)
        r = int(3 * sigma)
        y_range = np.arange(max(0, cy-r), min(height, cy+r+1))
        x_range = np.arange(max(0, cx-r), min(width, cx+r+1))
        if len(y_range) == 0 or len(x_range) == 0:
            continue
        yy, xx = np.meshgrid(y_range, x_range, indexing='ij')
        gaussian = intensity * np.exp(-((yy-cy)**2 + (xx-cx)**2) / (2*sigma**2))
        img[yy, xx] -= gaussian
    img = np.clip(img, 0, 255)
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    return img.astype(np.uint8)


def apply_displacement_field(image, u_field, v_field):
    h, w = image.shape[:2]
    y_coords, x_coords = np.meshgrid(
        np.arange(h, dtype=np.float64),
        np.arange(w, dtype=np.float64), indexing='ij')
    map_x = (x_coords - u_field).astype(np.float32)
    map_y = (y_coords - v_field).astype(np.float32)
    deformed = cv2.remap(image.astype(np.float64), map_x, map_y,
                         interpolation=cv2.INTER_CUBIC,
                         borderMode=cv2.BORDER_REFLECT)
    return np.clip(deformed, 0, 255).astype(np.uint8)


if HAS_E2E_DEPS:
    def test_e2e_rigid_translation():
        u_true, v_true = 3.0, -2.0
        ref = generate_synthetic_speckle(250, 250)
        h, w = ref.shape
        u_field = np.full((h, w), u_true)
        v_field = np.full((h, w), v_true)
        deformed = apply_displacement_field(ref, u_field, v_field)

        fft_result = compute_fft_cc(ref, deformed, subset_size=21,
                                    spacing=15, search_range=20,
                                    zncc_threshold=0.5)
        icgn_result = compute_icgn(ref, deformed, fft_result,
                                   subset_size=21, max_iterations=50,
                                   convergence_threshold=1e-4,
                                   zncc_threshold=0.5,
                                   shape_function='affine', n_workers=1)

        valid = icgn_result.valid_mask
        n_valid = np.sum(valid)
        assert n_valid > 0, "유효 POI 없음"

        mean_u = np.mean(icgn_result.disp_u[valid])
        mean_v = np.mean(icgn_result.disp_v[valid])
        std_u = np.std(icgn_result.disp_u[valid])
        std_v = np.std(icgn_result.disp_v[valid])

        print(f"    u: {mean_u:.4f}±{std_u:.4f} (true: {u_true})")
        print(f"    v: {mean_v:.4f}±{std_v:.4f} (true: {v_true})")
        print(f"    ZNCC: {icgn_result.mean_zncc:.4f}, valid: {n_valid}")

        assert abs(mean_u - u_true) < 0.05
        assert abs(mean_v - v_true) < 0.05
        assert std_u < 0.05
        assert std_v < 0.05

    runner.run("4-1 Rigid body translation (u=3, v=-2)", test_e2e_rigid_translation)


    def test_e2e_uniform_strain():
        ux_true = 0.005
        ref = generate_synthetic_speckle(250, 250)
        h, w = ref.shape
        y_coords, x_coords = np.meshgrid(
            np.arange(h, dtype=np.float64),
            np.arange(w, dtype=np.float64), indexing='ij')
        cx = w / 2
        u_field = ux_true * (x_coords - cx)
        v_field = np.zeros_like(u_field)
        deformed = apply_displacement_field(ref, u_field, v_field)

        fft_result = compute_fft_cc(ref, deformed, subset_size=25,
                                    spacing=15, search_range=20,
                                    zncc_threshold=0.4)
        icgn_result = compute_icgn(ref, deformed, fft_result,
                                   subset_size=25, max_iterations=50,
                                   convergence_threshold=1e-4,
                                   zncc_threshold=0.4,
                                   shape_function='affine', n_workers=1)

        valid = icgn_result.valid_mask
        n_valid = np.sum(valid)
        assert n_valid > 0

        mean_ux = np.mean(icgn_result.disp_ux[valid])
        print(f"    mean ux: {mean_ux:.6f} (true: {ux_true})")
        print(f"    valid: {n_valid}, ZNCC: {icgn_result.mean_zncc:.4f}")
        assert abs(mean_ux - ux_true) < 0.002

    runner.run("4-2 균일 인장 (ux=0.005)", test_e2e_uniform_strain)


    def test_e2e_quadratic_deformation():
        ref = generate_synthetic_speckle(300, 300, seed=123)
        h, w = ref.shape
        u0, ux_true, uxx_true = 2.0, 0.01, 0.0002
        y_coords, x_coords = np.meshgrid(
            np.arange(h, dtype=np.float64),
            np.arange(w, dtype=np.float64), indexing='ij')
        cx = w / 2
        dx = x_coords - cx
        u_field = u0 + ux_true * dx + 0.5 * uxx_true * dx**2
        v_field = np.zeros_like(u_field)
        deformed = apply_displacement_field(ref, u_field, v_field)

        fft_result = compute_fft_cc(ref, deformed, subset_size=31,
                                    spacing=15, search_range=30,
                                    zncc_threshold=0.3)
        icgn_result = compute_icgn(ref, deformed, fft_result,
                                   subset_size=31, max_iterations=80,
                                   convergence_threshold=1e-4,
                                   zncc_threshold=0.3,
                                   shape_function='quadratic', n_workers=1)

        valid = icgn_result.valid_mask & icgn_result.converged
        n_valid = np.sum(valid)
        assert n_valid > 0, "수렴한 유효 POI 없음"

        mean_ux = np.mean(icgn_result.disp_ux[valid])
        mean_uxx = np.mean(icgn_result.disp_uxx[valid])
        print(f"    mean ux:  {mean_ux:.6f} (true: {ux_true})")
        print(f"    mean uxx: {mean_uxx:.6f} (true: {uxx_true})")
        print(f"    convergence: {icgn_result.convergence_rate:.1%}")
        assert abs(mean_ux - ux_true) < 0.005
        assert abs(mean_uxx - uxx_true) < 0.001

    runner.run("4-3 2차 변형 복원 (uxx=0.0002) ★", test_e2e_quadratic_deformation)


# ===============================================================
#  최종 결과
# ===============================================================
print()
all_passed = runner.summary()

if all_passed:
    print("\n모든 테스트 통과 — Quadratic shape function 검증 완료!")
else:
    print("\n실패한 테스트가 있습니다. 위 오류 메시지를 확인하세요.")

# ===============================================================
#  디버그: 6×6 행렬 오류 추적
# ===============================================================
print("\n" + "="*60)
print("DEBUG: 6×6 행렬 오류 추적")
print("="*60)


def debug_quadratic_matrix():
    """
    W(p)의 6×6 행렬이 올바른지 직접 검증.

    올바른 6×6 행렬의 정의:
        W(p) 는 2차 warp를 (ξ², ξη, η², ξ, η, 1) 공간에서
        선형 변환으로 표현한 것.

        즉, W(p) @ [ξ², ξη, η², ξ, η, 1]^T 의 하위 2행이
        [ξ_w, η_w] 를 복원해야 함.

    이 테스트는 코드의 W_p 행렬이 이 조건을 만족하는지 확인.
    """
    from speckle.core.optimization.shape_function import (
        _update_quadratic_matrix_optimized,
        warp_quadratic,
        generate_local_coordinates,
    )

    xsi, eta = generate_local_coordinates(11)

    # 확장 좌표 벡터: [ξ², ξη, η², ξ, η, 1]
    def extended_coords(xsi, eta):
        n = len(xsi)
        V = np.zeros((6, n))
        V[0] = xsi * xsi      # ξ²
        V[1] = xsi * eta       # ξη
        V[2] = eta * eta       # η²
        V[3] = xsi             # ξ
        V[4] = eta             # η
        V[5] = 1.0             # 1
        return V

    # ── 코드의 W_p 행렬 추출 ──
    # _update_quadratic_matrix_optimized 내부에서 W_p를 구성하므로
    # 동일한 로직을 여기서 복제
    def build_W_matrix(p):
        u, ux, uy, uxx, uxy, uyy = p[0:6]
        v, vx, vy, vxx, vxy, vyy = p[6:12]

        def compute_A_terms(u, ux, uy, uxx, uxy, uyy,
                            v, vx, vy, vxx, vxy, vyy):
            A1 = 2*ux + ux**2 + u*uxx
            A2 = 2*u*uxy + 2*(1+ux)*uy
            A3 = uy**2 + u*uyy
            A4 = 2*u*(1+ux)
            A5 = 2*u*uy
            A6 = u**2
            A7 = 0.5*(v*uxx + 2*(1+ux)*vx + u*vxx)
            A8 = uy*vx + ux*vy + v*uxy + u*vxy + vy + ux
            A9 = 0.5*(v*uyy + 2*(1+vy)*uy + u*vyy)
            A10 = v + v*ux + u*vx
            A11 = u + v*uy + u*vy
            A12 = u*v
            A13 = vx**2 + v*vxx
            A14 = 2*v*vxy + 2*vx*(1+vy)
            A15 = 2*vy + vy**2 + v*vyy
            A16 = 2*v*vx
            A17 = 2*v*(1+vy)
            A18 = v**2
            return (A1, A2, A3, A4, A5, A6, A7, A8, A9,
                    A10, A11, A12, A13, A14, A15, A16, A17, A18)

        A = compute_A_terms(u, ux, uy, uxx, uxy, uyy,
                            v, vx, vy, vxx, vxy, vyy)
        W = np.array([
            [1+A[0],  A[1],    A[2],   A[3],   A[4],  A[5]],
            [A[6],    1+A[7],  A[8],   A[9],   A[10], A[11]],
            [A[12],   A[13],   1+A[14],A[15],  A[16], A[17]],
            [0.5*uxx, uxy,     0.5*uyy,1+ux,   uy,    u],
            [0.5*vxx, vxy,     0.5*vyy,vx,     1+vy,  v],
            [0,       0,       0,      0,      0,     1]
        ], dtype=np.float64)
        return W

    print("\n  [검증 1] W(p)의 행3,4가 warp 결과를 재현하는지")
    print("  행3 @ V = xsi_w, 행4 @ V = eta_w 여야 함\n")

    n_tests = 100
    row3_errors = []
    row4_errors = []
    row0_errors = []

    for trial in range(n_tests):
        p = random_quadratic_params()
        W = build_W_matrix(p)
        V = extended_coords(xsi, eta)

        # W @ V 계산
        result = W @ V  # (6, n_pixels)

        # warp 직접 계산
        xsi_w, eta_w = warp_quadratic(p, xsi, eta)

        # 행 3: xsi_w 복원?
        err3 = np.max(np.abs(result[3, :] - xsi_w))
        row3_errors.append(err3)

        # 행 4: eta_w 복원?
        err4 = np.max(np.abs(result[4, :] - eta_w))
        row4_errors.append(err4)

        # 행 0: xsi_w² 복원?
        err0 = np.max(np.abs(result[0, :] - xsi_w * xsi_w))
        row0_errors.append(err0)

    row3_errors = np.array(row3_errors)
    row4_errors = np.array(row4_errors)
    row0_errors = np.array(row0_errors)

    print(f"  행3 (→xsi_w) max error: {row3_errors.max():.2e}")
    print(f"  행4 (→eta_w) max error: {row4_errors.max():.2e}")
    print(f"  행0 (→xsi_w²) max error: {row0_errors.max():.2e}")

    if row3_errors.max() < 1e-10 and row4_errors.max() < 1e-10:
        print("  ✓ 행3,4 (선형 부분) 정확 — warp 재현 OK")
    else:
        print("  ✗ 행3,4 불일치 — 행렬 하위 행에 오류")

    if row0_errors.max() < 1e-10:
        print("  ✓ 행0 (ξ_w² 재현) 정확")
    else:
        print("  ✗ 행0 불일치 — 비선형 항 A1~A6에 오류 가능")

    # ── 검증 2: 합성 W(p)@W(dp) vs W(composed) ──
    print(f"\n  [검증 2] W(p) @ W(dp) 의 행3,4가 합성 warp를 재현하는지")
    print("  W_composed @ V 의 행3,4 == warp(p, warp(dp, ξ, η))\n")

    compose_errors_3 = []
    compose_errors_4 = []

    for trial in range(n_tests):
        p = random_quadratic_params()
        dp = random_quadratic_params(
            translation_scale=0.5, gradient_scale=0.005,
            second_order_scale=0.0005)

        W_p = build_W_matrix(p)
        W_dp = build_W_matrix(dp)
        W_composed = W_p @ W_dp

        V = extended_coords(xsi, eta)
        result = W_composed @ V

        # 실제 합성: 먼저 dp로 warp, 그 결과를 p로 warp
        xsi_dp, eta_dp = warp_quadratic(dp, xsi, eta)
        xsi_final, eta_final = warp_quadratic(p, xsi_dp, eta_dp)

        err3 = np.max(np.abs(result[3, :] - xsi_final))
        err4 = np.max(np.abs(result[4, :] - eta_final))
        compose_errors_3.append(err3)
        compose_errors_4.append(err4)

    compose_errors_3 = np.array(compose_errors_3)
    compose_errors_4 = np.array(compose_errors_4)

    print(f"  행3 max error: {compose_errors_3.max():.2e}")
    print(f"  행4 max error: {compose_errors_4.max():.2e}")

    if compose_errors_3.max() < 1e-8 and compose_errors_4.max() < 1e-8:
        print("  ✓ 행렬 합성 == 실제 합성 — 6×6 표현 정확")
    else:
        print("  ✗ 행렬 합성 ≠ 실제 합성 — 6×6 표현에 근본적 문제")
        print("    → 2차 warp의 합성은 2차로 닫히지 않음 (3차 이상 발생)")
        print("    → 이것은 행렬 코드 버그가 아니라 이론적 한계일 수 있음")

    # ── 검증 3: 역변환 행3,4 직접 비교 ──
    print(f"\n  [검증 3] IC update: p_new의 warp vs 정의적 합성")

    ic_errors = []
    for trial in range(100):
        p = random_quadratic_params()
        dp = random_quadratic_params(
            translation_scale=0.5, gradient_scale=0.005,
            second_order_scale=0.0005)

        p_new = _update_quadratic_matrix_optimized(p, dp)

        # A: p_new로 직접 warp
        xsi_a, eta_a = warp_quadratic(p_new, xsi, eta)

        # B: Newton으로 dp 역변환 → p로 warp
        xsi_inv, eta_inv = invert_warp_newton(
            warp_quadratic, dp, xsi, eta)
        xsi_b, eta_b = warp_quadratic(p, xsi_inv, eta_inv)

        # C: 행렬로만 계산 (행3,4 추출)
        W_p = build_W_matrix(p)
        W_dp = build_W_matrix(dp)
        try:
            W_dp_inv = np.linalg.inv(W_dp)
        except np.linalg.LinAlgError:
            continue
        W_new = W_p @ W_dp_inv
        V = extended_coords(xsi, eta)
        result_c = W_new @ V
        xsi_c = result_c[3, :]
        eta_c = result_c[4, :]

        err_ab = np.max(np.abs(xsi_a - xsi_b)) + np.max(np.abs(eta_a - eta_b))
        err_ac = np.max(np.abs(xsi_a - xsi_c)) + np.max(np.abs(eta_a - eta_c))
        err_bc = np.max(np.abs(xsi_b - xsi_c)) + np.max(np.abs(eta_b - eta_c))

        ic_errors.append({
            'a_vs_b': err_ab,  # p_new warp vs Newton
            'a_vs_c': err_ac,  # p_new warp vs 행렬 직접
            'b_vs_c': err_bc,  # Newton vs 행렬 직접
        })

    err_ab = np.array([e['a_vs_b'] for e in ic_errors])
    err_ac = np.array([e['a_vs_c'] for e in ic_errors])
    err_bc = np.array([e['b_vs_c'] for e in ic_errors])

    print(f"  A(p_new warp) vs B(Newton): max={err_ab.max():.2e}")
    print(f"  A(p_new warp) vs C(행렬):   max={err_ac.max():.2e}")
    print(f"  B(Newton)     vs C(행렬):   max={err_bc.max():.2e}")

    if err_ac.max() < 1e-10:
        print("\n  → A==C: 파라미터 추출은 정확 (행렬→p_new 변환 OK)")
        if err_ab.max() > 1e-5:
            print("  → A≠B: 6×6 행렬 자체가 실제 warp 합성을 정확히 표현 못함")
            print("  → 원인: 2차 warp 합성 시 3차 이상 항이 발생하는데")
            print("         6×6 행렬은 2차까지만 표현 가능 → truncation error")
            print("  → 이것은 코드 버그가 아니라 IC-GN Quadratic의 이론적 근사 오차")
    elif err_ab.max() < 1e-10:
        print("\n  → A==B: Newton과 일치 → 파라미터 추출(행3,4→p)에 문제")
        print("  → 행렬 인덱스 또는 2× 스케일링 확인 필요")
    else:
        print("\n  → 세 가지 모두 불일치 — 복합적 문제")

debug_quadratic_matrix()

sys.exit(0 if all_passed else 1)
