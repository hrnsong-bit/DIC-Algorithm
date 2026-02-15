"""
Numba Shape Function 검증 테스트

검증 전략:
    기존 shape_function.py의 결과를 reference로 사용하여
    shape_function_numba.py의 모든 함수가 수치적으로 동일한
    결과를 생성하는지 확인한다.

검증 항목:
    1. Warp (affine, quadratic)
    2. Steepest Descent Image (affine, quadratic)
    3. Hessian
    4. 수렴 판정
    5. Inverse Compositional Warp Update (affine, quadratic)
    6. 다양한 입력 조건 강건성
    7. ICGN 통합 시나리오
    8. 성능 비교

사용법:
    cd /home/user/webapp
    python tests/test_shape_function_numba.py
"""

import sys
import os
import time
import traceback

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(THIS_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

# 기존 구현 (reference)
from speckle.core.optimization.shape_function import (
    generate_local_coordinates as gen_coords_ref,
    warp_affine as warp_affine_ref,
    warp_quadratic as warp_quadratic_ref,
    compute_steepest_descent_affine as sd_affine_ref,
    compute_steepest_descent_quadratic as sd_quadratic_ref,
    compute_hessian as hessian_ref,
    check_convergence as check_conv_ref,
    update_warp_inverse_compositional as update_warp_ref,
    warp as warp_ref,
    compute_steepest_descent as sd_ref,
)

# Numba 구현 (검증 대상)
from speckle.core.optimization.shape_function_numba import (
    AFFINE, QUADRATIC,
    generate_local_coordinates as gen_coords_numba,
    warp_affine as warp_affine_numba,
    warp_quadratic as warp_quadratic_numba,
    warp as warp_numba,
    compute_steepest_descent_affine as sd_affine_numba,
    compute_steepest_descent_quadratic as sd_quadratic_numba,
    compute_steepest_descent as sd_numba,
    compute_hessian as hessian_numba,
    check_convergence_affine as check_conv_affine_numba,
    check_convergence_quadratic as check_conv_quadratic_numba,
    check_convergence as check_conv_numba,
    update_warp_affine as update_warp_affine_numba,
    update_warp_quadratic as update_warp_quadratic_numba,
    update_warp as update_warp_numba,
    get_num_params as get_num_params_numba,
    warmup_numba_shape,
)


# =============================================================================
#  테스트 프레임워크
# =============================================================================

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def run(self, name, func):
        try:
            func()
            self.passed += 1
            print(f"  \u2713 {name}")
        except AssertionError as e:
            self.failed += 1
            self.errors.append((name, str(e)))
            print(f"  \u2717 {name}")
            print(f"    \u2192 {e}")
        except Exception as e:
            self.failed += 1
            self.errors.append((name, traceback.format_exc()))
            print(f"  \u2717 {name} (exception)")
            print(f"    \u2192 {type(e).__name__}: {e}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*70}")
        print(f"Result: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print(f"\nFailed:")
            for name, err in self.errors:
                print(f"  - {name}")
                first_line = err.strip().split('\n')[0]
                print(f"    {first_line}")
        print(f"{'='*70}")
        return self.failed == 0


runner = TestRunner()
rng = np.random.default_rng(seed=42)


# =============================================================================
#  JIT 워밍업
# =============================================================================
print("JIT warmup...")
t0 = time.perf_counter()
warmup_numba_shape()
print(f"JIT warmup: {time.perf_counter() - t0:.2f}s\n")


# =============================================================================
#  공통 테스트 데이터
# =============================================================================
SUBSET_SIZE = 21
N_PIXELS = SUBSET_SIZE * SUBSET_SIZE  # 441
HALF = SUBSET_SIZE // 2

xsi_ref, eta_ref = gen_coords_ref(SUBSET_SIZE)
xsi_nb, eta_nb = gen_coords_numba(SUBSET_SIZE)


# =============================================================================
#  TEST 1: 로컬 좌표 생성
# =============================================================================
print("=" * 70)
print("TEST 1: Local Coordinate Generation")
print("=" * 70)


def test_local_coordinates():
    """로컬 좌표가 기존 구현과 동일"""
    assert np.array_equal(xsi_ref, xsi_nb), "xsi mismatch"
    assert np.array_equal(eta_ref, eta_nb), "eta mismatch"
    print(f"    shape: ({len(xsi_ref)},), range: [{xsi_ref.min()}, {xsi_ref.max()}]")

runner.run("1-1 Local coordinates match", test_local_coordinates)


# =============================================================================
#  TEST 2: Warp 함수
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: Warp Functions")
print("=" * 70)


def test_warp_affine_basic():
    """Affine warp: 기본 파라미터"""
    p = np.array([0.5, 0.01, -0.02, -0.3, 0.005, -0.01], dtype=np.float64)
    xw_ref, ew_ref = warp_affine_ref(p, xsi_ref, eta_ref)

    xw_nb = np.empty(N_PIXELS, dtype=np.float64)
    ew_nb = np.empty(N_PIXELS, dtype=np.float64)
    warp_affine_numba(p, xsi_nb, eta_nb, xw_nb, ew_nb)

    assert np.allclose(xw_ref, xw_nb, atol=1e-14), \
        f"xsi_w max diff: {np.abs(xw_ref - xw_nb).max()}"
    assert np.allclose(ew_ref, ew_nb, atol=1e-14), \
        f"eta_w max diff: {np.abs(ew_ref - ew_nb).max()}"

runner.run("2-1 Affine warp basic", test_warp_affine_basic)


def test_warp_affine_zero():
    """Affine warp: 제로 파라미터 (항등 변환)"""
    p = np.zeros(6, dtype=np.float64)
    xw_ref, ew_ref = warp_affine_ref(p, xsi_ref, eta_ref)

    xw_nb = np.empty(N_PIXELS, dtype=np.float64)
    ew_nb = np.empty(N_PIXELS, dtype=np.float64)
    warp_affine_numba(p, xsi_nb, eta_nb, xw_nb, ew_nb)

    # 제로 파라미터 → xsi_w = xsi, eta_w = eta
    assert np.allclose(xw_nb, xsi_nb, atol=1e-15)
    assert np.allclose(ew_nb, eta_nb, atol=1e-15)
    assert np.allclose(xw_ref, xw_nb, atol=1e-15)

runner.run("2-2 Affine warp zero (identity)", test_warp_affine_zero)


def test_warp_affine_random():
    """Affine warp: 다수 랜덤 파라미터"""
    max_err = 0.0
    for _ in range(100):
        p = rng.uniform(-0.5, 0.5, 6).astype(np.float64)
        p[1:3] *= 0.1  # ux, uy 작게
        p[4:6] *= 0.1  # vx, vy 작게
        xw_ref, ew_ref = warp_affine_ref(p, xsi_ref, eta_ref)
        xw_nb = np.empty(N_PIXELS, dtype=np.float64)
        ew_nb = np.empty(N_PIXELS, dtype=np.float64)
        warp_affine_numba(p, xsi_nb, eta_nb, xw_nb, ew_nb)
        max_err = max(max_err, np.abs(xw_ref - xw_nb).max(),
                      np.abs(ew_ref - ew_nb).max())
    print(f"    100 random params: max_err={max_err:.2e}")
    assert max_err < 1e-13

runner.run("2-3 Affine warp 100 random params", test_warp_affine_random)


def test_warp_quadratic_basic():
    """Quadratic warp: 기본 파라미터"""
    p = np.array([0.5, 0.01, -0.02, 0.001, -0.0005, 0.0003,
                  -0.3, 0.005, -0.01, -0.001, 0.0002, -0.0001],
                 dtype=np.float64)
    xw_ref, ew_ref = warp_quadratic_ref(p, xsi_ref, eta_ref)

    xw_nb = np.empty(N_PIXELS, dtype=np.float64)
    ew_nb = np.empty(N_PIXELS, dtype=np.float64)
    warp_quadratic_numba(p, xsi_nb, eta_nb, xw_nb, ew_nb)

    assert np.allclose(xw_ref, xw_nb, atol=1e-13), \
        f"xsi_w max diff: {np.abs(xw_ref - xw_nb).max()}"
    assert np.allclose(ew_ref, ew_nb, atol=1e-13), \
        f"eta_w max diff: {np.abs(ew_ref - ew_nb).max()}"

runner.run("2-4 Quadratic warp basic", test_warp_quadratic_basic)


def test_warp_quadratic_random():
    """Quadratic warp: 다수 랜덤 파라미터"""
    max_err = 0.0
    for _ in range(100):
        p = rng.uniform(-0.5, 0.5, 12).astype(np.float64)
        p[1:6] *= 0.01  # 1차/2차 항 작게
        p[7:12] *= 0.01
        xw_ref, ew_ref = warp_quadratic_ref(p, xsi_ref, eta_ref)
        xw_nb = np.empty(N_PIXELS, dtype=np.float64)
        ew_nb = np.empty(N_PIXELS, dtype=np.float64)
        warp_quadratic_numba(p, xsi_nb, eta_nb, xw_nb, ew_nb)
        max_err = max(max_err, np.abs(xw_ref - xw_nb).max(),
                      np.abs(ew_ref - ew_nb).max())
    print(f"    100 random params: max_err={max_err:.2e}")
    assert max_err < 1e-12

runner.run("2-5 Quadratic warp 100 random params", test_warp_quadratic_random)


def test_warp_unified():
    """통합 warp 인터페이스"""
    p6 = rng.uniform(-0.1, 0.1, 6).astype(np.float64)
    xw_ref, ew_ref = warp_ref(p6, xsi_ref, eta_ref, 'affine')
    xw_nb = np.empty(N_PIXELS, dtype=np.float64)
    ew_nb = np.empty(N_PIXELS, dtype=np.float64)
    warp_numba(p6, xsi_nb, eta_nb, xw_nb, ew_nb, AFFINE)
    assert np.allclose(xw_ref, xw_nb, atol=1e-14)

    p12 = np.zeros(12, dtype=np.float64)
    p12[0] = 0.3; p12[6] = -0.2
    xw_ref2, ew_ref2 = warp_ref(p12, xsi_ref, eta_ref, 'quadratic')
    warp_numba(p12, xsi_nb, eta_nb, xw_nb, ew_nb, QUADRATIC)
    assert np.allclose(xw_ref2, xw_nb, atol=1e-14)

runner.run("2-6 Unified warp interface", test_warp_unified)


# =============================================================================
#  TEST 3: Steepest Descent Image
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: Steepest Descent Image")
print("=" * 70)


def _make_gradients(seed=42):
    local_rng = np.random.default_rng(seed)
    dfdx = local_rng.uniform(-10, 10, N_PIXELS).astype(np.float64)
    dfdy = local_rng.uniform(-10, 10, N_PIXELS).astype(np.float64)
    return dfdx, dfdy


def test_sd_affine():
    """Affine Steepest Descent"""
    dfdx, dfdy = _make_gradients()
    J_ref = sd_affine_ref(dfdx, dfdy, xsi_ref, eta_ref)

    J_nb = np.empty((N_PIXELS, 6), dtype=np.float64)
    sd_affine_numba(dfdx, dfdy, xsi_nb, eta_nb, J_nb)

    diff = np.abs(J_ref - J_nb)
    print(f"    max diff: {diff.max():.2e}")
    assert diff.max() < 1e-14

runner.run("3-1 Affine steepest descent", test_sd_affine)


def test_sd_quadratic():
    """Quadratic Steepest Descent"""
    dfdx, dfdy = _make_gradients()
    J_ref = sd_quadratic_ref(dfdx, dfdy, xsi_ref, eta_ref)

    J_nb = np.empty((N_PIXELS, 12), dtype=np.float64)
    sd_quadratic_numba(dfdx, dfdy, xsi_nb, eta_nb, J_nb)

    diff = np.abs(J_ref - J_nb)
    print(f"    max diff: {diff.max():.2e}")
    assert diff.max() < 1e-13

runner.run("3-2 Quadratic steepest descent", test_sd_quadratic)


def test_sd_unified():
    """통합 인터페이스"""
    dfdx, dfdy = _make_gradients(seed=99)
    J_ref6 = sd_ref(dfdx, dfdy, xsi_ref, eta_ref, 'affine')
    J_nb6 = np.empty((N_PIXELS, 6), dtype=np.float64)
    sd_numba(dfdx, dfdy, xsi_nb, eta_nb, J_nb6, AFFINE)
    assert np.abs(J_ref6 - J_nb6).max() < 1e-14

    J_ref12 = sd_ref(dfdx, dfdy, xsi_ref, eta_ref, 'quadratic')
    J_nb12 = np.empty((N_PIXELS, 12), dtype=np.float64)
    sd_numba(dfdx, dfdy, xsi_nb, eta_nb, J_nb12, QUADRATIC)
    assert np.abs(J_ref12 - J_nb12).max() < 1e-13

runner.run("3-3 Unified steepest descent", test_sd_unified)


def test_sd_random_gradients():
    """다양한 gradient에서 steepest descent 검증"""
    max_err_a = 0.0
    max_err_q = 0.0
    for seed in range(50):
        dfdx, dfdy = _make_gradients(seed=seed)

        J_ref_a = sd_affine_ref(dfdx, dfdy, xsi_ref, eta_ref)
        J_nb_a = np.empty((N_PIXELS, 6), dtype=np.float64)
        sd_affine_numba(dfdx, dfdy, xsi_nb, eta_nb, J_nb_a)
        max_err_a = max(max_err_a, np.abs(J_ref_a - J_nb_a).max())

        J_ref_q = sd_quadratic_ref(dfdx, dfdy, xsi_ref, eta_ref)
        J_nb_q = np.empty((N_PIXELS, 12), dtype=np.float64)
        sd_quadratic_numba(dfdx, dfdy, xsi_nb, eta_nb, J_nb_q)
        max_err_q = max(max_err_q, np.abs(J_ref_q - J_nb_q).max())

    print(f"    50 gradients: affine max_err={max_err_a:.2e}, "
          f"quadratic max_err={max_err_q:.2e}")
    assert max_err_a < 1e-13
    assert max_err_q < 1e-12

runner.run("3-4 SD with 50 random gradients", test_sd_random_gradients)


# =============================================================================
#  TEST 4: Hessian
# =============================================================================
print("\n" + "=" * 70)
print("TEST 4: Hessian Matrix")
print("=" * 70)


def test_hessian_affine():
    """Affine Hessian: H = J^T @ J"""
    dfdx, dfdy = _make_gradients()
    J_ref = sd_affine_ref(dfdx, dfdy, xsi_ref, eta_ref)
    H_ref = hessian_ref(J_ref)

    J_nb = np.empty((N_PIXELS, 6), dtype=np.float64)
    sd_affine_numba(dfdx, dfdy, xsi_nb, eta_nb, J_nb)
    H_nb = np.empty((6, 6), dtype=np.float64)
    hessian_numba(J_nb, H_nb)

    diff = np.abs(H_ref - H_nb)
    print(f"    6x6 Hessian max diff: {diff.max():.2e}")
    # 루프 합산 vs J.T@J (BLAS) 부동소수점 순서 차이 허용
    assert diff.max() < 1e-8, f"Affine Hessian diff: {diff.max():.2e}"

runner.run("4-1 Affine Hessian", test_hessian_affine)


def test_hessian_quadratic():
    """Quadratic Hessian: H = J^T @ J"""
    dfdx, dfdy = _make_gradients()
    J_ref = sd_quadratic_ref(dfdx, dfdy, xsi_ref, eta_ref)
    H_ref = hessian_ref(J_ref)

    J_nb = np.empty((N_PIXELS, 12), dtype=np.float64)
    sd_quadratic_numba(dfdx, dfdy, xsi_nb, eta_nb, J_nb)
    H_nb = np.empty((12, 12), dtype=np.float64)
    hessian_numba(J_nb, H_nb)

    diff = np.abs(H_ref - H_nb)
    print(f"    12x12 Hessian max diff: {diff.max():.2e}")
    # 루프 합산 vs J.T@J (BLAS) 부동소수점 순서 차이 허용
    assert diff.max() < 1e-7, f"Quadratic Hessian diff: {diff.max():.2e}"

runner.run("4-2 Quadratic Hessian", test_hessian_quadratic)


def test_hessian_symmetry():
    """Hessian 대칭 확인"""
    dfdx, dfdy = _make_gradients()

    J_nb = np.empty((N_PIXELS, 6), dtype=np.float64)
    sd_affine_numba(dfdx, dfdy, xsi_nb, eta_nb, J_nb)
    H = np.empty((6, 6), dtype=np.float64)
    hessian_numba(J_nb, H)
    assert np.allclose(H, H.T, atol=1e-15), "Affine Hessian not symmetric"

    J_nb12 = np.empty((N_PIXELS, 12), dtype=np.float64)
    sd_quadratic_numba(dfdx, dfdy, xsi_nb, eta_nb, J_nb12)
    H12 = np.empty((12, 12), dtype=np.float64)
    hessian_numba(J_nb12, H12)
    assert np.allclose(H12, H12.T, atol=1e-15), "Quadratic Hessian not symmetric"

runner.run("4-3 Hessian symmetry", test_hessian_symmetry)


# =============================================================================
#  TEST 5: 수렴 판정
# =============================================================================
print("\n" + "=" * 70)
print("TEST 5: Convergence Check")
print("=" * 70)


def test_convergence_affine():
    """Affine 수렴 판정: 기존 구현과 일치"""
    for _ in range(100):
        dp = rng.uniform(-0.01, 0.01, 6).astype(np.float64)
        threshold = rng.uniform(0.0001, 0.01)
        conv_ref, norm_ref = check_conv_ref(dp, SUBSET_SIZE, threshold, 'affine')
        conv_nb, norm_nb = check_conv_numba(dp, HALF, threshold, AFFINE)

        assert conv_ref == conv_nb, \
            f"convergence mismatch: ref={conv_ref}, numba={conv_nb}, norm={norm_ref:.6e}"
        assert abs(norm_ref - norm_nb) < 1e-14, \
            f"norm mismatch: ref={norm_ref}, numba={norm_nb}"

runner.run("5-1 Affine convergence check", test_convergence_affine)


def test_convergence_quadratic():
    """Quadratic 수렴 판정: 기존 구현과 일치"""
    for _ in range(100):
        dp = rng.uniform(-0.01, 0.01, 12).astype(np.float64)
        threshold = rng.uniform(0.0001, 0.01)
        conv_ref, norm_ref = check_conv_ref(dp, SUBSET_SIZE, threshold, 'quadratic')
        conv_nb, norm_nb = check_conv_numba(dp, HALF, threshold, QUADRATIC)

        assert conv_ref == conv_nb, \
            f"convergence mismatch: ref={conv_ref}, numba={conv_nb}"
        assert abs(norm_ref - norm_nb) < 1e-13, \
            f"norm mismatch: ref={norm_ref:.6e}, numba={norm_nb:.6e}"

runner.run("5-2 Quadratic convergence check", test_convergence_quadratic)


def test_convergence_boundary():
    """수렴 경계 부근에서 정확성 확인"""
    threshold = 0.001
    # 정확히 경계 위아래의 dp 생성
    dp = np.array([0.0005, 0.0001, 0.0001, 0.0005, 0.0001, 0.0001],
                  dtype=np.float64)
    conv_ref, norm_ref = check_conv_ref(dp, SUBSET_SIZE, threshold, 'affine')
    conv_nb, norm_nb = check_conv_numba(dp, HALF, threshold, AFFINE)
    assert conv_ref == conv_nb
    assert abs(norm_ref - norm_nb) < 1e-14

runner.run("5-3 Convergence at threshold boundary", test_convergence_boundary)


# =============================================================================
#  TEST 6: Warp Update (핵심)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 6: Inverse Compositional Warp Update")
print("=" * 70)


def test_update_affine_basic():
    """Affine warp update: 기본 케이스"""
    p = np.array([0.5, 0.01, -0.02, -0.3, 0.005, -0.01], dtype=np.float64)
    dp = np.array([0.001, 0.0005, -0.0003, -0.002, 0.0001, 0.0002],
                  dtype=np.float64)

    p_ref = update_warp_ref(p, dp, 'affine')
    p_new = np.empty(6, dtype=np.float64)
    update_warp_affine_numba(p, dp, p_new)

    diff = np.abs(p_ref - p_new)
    print(f"    max diff: {diff.max():.2e}")
    assert diff.max() < 1e-14, f"affine update: max diff={diff.max():.2e}"

runner.run("6-1 Affine warp update basic", test_update_affine_basic)


def test_update_affine_random():
    """Affine warp update: 100개 랜덤 케이스"""
    max_err = 0.0
    for _ in range(100):
        p = rng.uniform(-1.0, 1.0, 6).astype(np.float64)
        p[1:3] *= 0.05; p[4:6] *= 0.05
        dp = rng.uniform(-0.01, 0.01, 6).astype(np.float64)

        p_ref = update_warp_ref(p, dp, 'affine')
        p_new = np.empty(6, dtype=np.float64)
        update_warp_affine_numba(p, dp, p_new)
        max_err = max(max_err, np.abs(p_ref - p_new).max())

    print(f"    100 random: max_err={max_err:.2e}")
    assert max_err < 1e-13

runner.run("6-2 Affine warp update 100 random", test_update_affine_random)


def test_update_affine_zero_dp():
    """Affine warp update: dp=0이면 p 불변"""
    p = np.array([1.0, 0.05, -0.03, -0.8, 0.02, 0.01], dtype=np.float64)
    dp = np.zeros(6, dtype=np.float64)

    p_new = np.empty(6, dtype=np.float64)
    update_warp_affine_numba(p, dp, p_new)
    assert np.allclose(p, p_new, atol=1e-15), \
        f"dp=0 should keep p unchanged: diff={np.abs(p - p_new).max():.2e}"

runner.run("6-3 Affine update: dp=0 → p unchanged", test_update_affine_zero_dp)


def test_update_quadratic_basic():
    """Quadratic warp update: 기본 케이스"""
    p = np.zeros(12, dtype=np.float64)
    p[0] = 0.5; p[1] = 0.01; p[2] = -0.02
    p[6] = -0.3; p[7] = 0.005; p[8] = -0.01
    dp = rng.uniform(-0.005, 0.005, 12).astype(np.float64)

    p_ref = update_warp_ref(p, dp, 'quadratic')
    p_new = np.empty(12, dtype=np.float64)
    update_warp_quadratic_numba(p, dp, p_new)

    diff = np.abs(p_ref - p_new)
    print(f"    max diff: {diff.max():.2e}")
    assert diff.max() < 1e-12, f"quadratic update: max diff={diff.max():.2e}"

runner.run("6-4 Quadratic warp update basic", test_update_quadratic_basic)


def test_update_quadratic_random():
    """Quadratic warp update: 100개 랜덤 케이스"""
    max_err = 0.0
    for _ in range(100):
        p = rng.uniform(-0.5, 0.5, 12).astype(np.float64)
        p[1:6] *= 0.02; p[7:12] *= 0.02  # 변형 항 작게
        dp = rng.uniform(-0.005, 0.005, 12).astype(np.float64)

        p_ref = update_warp_ref(p, dp, 'quadratic')
        p_new = np.empty(12, dtype=np.float64)
        update_warp_quadratic_numba(p, dp, p_new)
        max_err = max(max_err, np.abs(p_ref - p_new).max())

    print(f"    100 random: max_err={max_err:.2e}")
    assert max_err < 1e-11

runner.run("6-5 Quadratic warp update 100 random", test_update_quadratic_random)


def test_update_quadratic_zero_dp():
    """Quadratic warp update: dp=0이면 p 불변"""
    p = rng.uniform(-0.5, 0.5, 12).astype(np.float64)
    p[1:6] *= 0.02; p[7:12] *= 0.02
    dp = np.zeros(12, dtype=np.float64)

    p_new = np.empty(12, dtype=np.float64)
    update_warp_quadratic_numba(p, dp, p_new)
    diff = np.abs(p - p_new).max()
    assert diff < 1e-13, f"dp=0 should keep p unchanged: diff={diff:.2e}"

runner.run("6-6 Quadratic update: dp=0 → p unchanged", test_update_quadratic_zero_dp)


def test_update_unified():
    """통합 update_warp 인터페이스"""
    p6 = rng.uniform(-0.5, 0.5, 6).astype(np.float64)
    p6[1:3] *= 0.05; p6[4:6] *= 0.05
    dp6 = rng.uniform(-0.01, 0.01, 6).astype(np.float64)

    p_ref6 = update_warp_ref(p6, dp6, 'affine')
    p_new6 = np.empty(6, dtype=np.float64)
    update_warp_numba(p6, dp6, p_new6, AFFINE)
    assert np.abs(p_ref6 - p_new6).max() < 1e-13

    p12 = rng.uniform(-0.5, 0.5, 12).astype(np.float64)
    p12[1:6] *= 0.02; p12[7:12] *= 0.02
    dp12 = rng.uniform(-0.005, 0.005, 12).astype(np.float64)

    p_ref12 = update_warp_ref(p12, dp12, 'quadratic')
    p_new12 = np.empty(12, dtype=np.float64)
    update_warp_numba(p12, dp12, p_new12, QUADRATIC)
    assert np.abs(p_ref12 - p_new12).max() < 1e-11

runner.run("6-7 Unified update_warp", test_update_unified)


# =============================================================================
#  TEST 7: ICGN 통합 시나리오
# =============================================================================
print("\n" + "=" * 70)
print("TEST 7: ICGN Integration Scenario")
print("=" * 70)


def test_icgn_affine_mini_loop():
    """Affine ICGN 미니 반복: 전체 파이프라인 일관성"""
    dfdx, dfdy = _make_gradients(seed=77)

    # Reference: steepest descent → hessian → inv → iterate
    J_ref = sd_affine_ref(dfdx, dfdy, xsi_ref, eta_ref)
    H_ref = hessian_ref(J_ref)
    H_inv_ref = np.linalg.inv(H_ref)

    # Numba: 동일 파이프라인
    J_nb = np.empty((N_PIXELS, 6), dtype=np.float64)
    sd_affine_numba(dfdx, dfdy, xsi_nb, eta_nb, J_nb)
    H_nb = np.empty((6, 6), dtype=np.float64)
    hessian_numba(J_nb, H_nb)
    H_inv_nb = np.linalg.inv(H_nb)

    # H_inv 비교
    diff_hinv = np.abs(H_inv_ref - H_inv_nb).max()
    assert diff_hinv < 1e-10, f"H_inv diff: {diff_hinv:.2e}"

    # 5회 반복 시뮬레이션
    p_ref = np.array([0.3, 0.0, 0.0, -0.5, 0.0, 0.0], dtype=np.float64)
    p_nb = p_ref.copy()
    p_new_nb = np.empty(6, dtype=np.float64)

    for _ in range(5):
        # 가상 residual
        residual = rng.standard_normal(N_PIXELS) * 0.1
        b_ref = -J_ref.T @ residual
        dp_ref = H_inv_ref @ b_ref

        b_nb = np.zeros(6, dtype=np.float64)
        for k in range(N_PIXELS):
            for j in range(6):
                b_nb[j] -= J_nb[k, j] * residual[k]
        dp_nb = H_inv_nb @ b_nb

        # dp 비교
        assert np.abs(dp_ref - dp_nb).max() < 1e-10

        # warp update
        p_ref = update_warp_ref(p_ref, dp_ref, 'affine')
        update_warp_affine_numba(p_nb, dp_nb, p_new_nb)
        p_nb = p_new_nb.copy()

    diff_final = np.abs(p_ref - p_nb).max()
    print(f"    5-iteration final p diff: {diff_final:.2e}")
    assert diff_final < 1e-10

runner.run("7-1 Affine ICGN mini-loop (5 iters)", test_icgn_affine_mini_loop)


def test_icgn_quadratic_mini_loop():
    """Quadratic ICGN 미니 반복"""
    dfdx, dfdy = _make_gradients(seed=88)

    J_ref = sd_quadratic_ref(dfdx, dfdy, xsi_ref, eta_ref)
    H_ref = hessian_ref(J_ref)
    H_inv_ref = np.linalg.inv(H_ref)

    J_nb = np.empty((N_PIXELS, 12), dtype=np.float64)
    sd_quadratic_numba(dfdx, dfdy, xsi_nb, eta_nb, J_nb)
    H_nb = np.empty((12, 12), dtype=np.float64)
    hessian_numba(J_nb, H_nb)
    H_inv_nb = np.linalg.inv(H_nb)

    p_ref = np.zeros(12, dtype=np.float64)
    p_ref[0] = 0.3; p_ref[6] = -0.5
    p_nb = p_ref.copy()
    p_new_nb = np.empty(12, dtype=np.float64)

    for _ in range(5):
        residual = rng.standard_normal(N_PIXELS) * 0.1
        dp_ref = H_inv_ref @ (-J_ref.T @ residual)

        b_nb = np.zeros(12, dtype=np.float64)
        for k in range(N_PIXELS):
            for j in range(12):
                b_nb[j] -= J_nb[k, j] * residual[k]
        dp_nb = H_inv_nb @ b_nb

        p_ref = update_warp_ref(p_ref, dp_ref, 'quadratic')
        update_warp_quadratic_numba(p_nb, dp_nb, p_new_nb)
        p_nb = p_new_nb.copy()

    diff_final = np.abs(p_ref - p_nb).max()
    print(f"    5-iteration final p diff: {diff_final:.2e}")
    assert diff_final < 1e-9

runner.run("7-2 Quadratic ICGN mini-loop (5 iters)", test_icgn_quadratic_mini_loop)


def test_warp_update_composition_identity():
    """W(p) · W(p)^{-1} = I 확인 (affine)"""
    p = np.array([0.5, 0.03, -0.02, -0.3, 0.01, -0.015], dtype=np.float64)
    # dp = p이면 W(p)·W(p)^{-1} = I → 결과는 zero params
    p_new = np.empty(6, dtype=np.float64)
    update_warp_affine_numba(p, p, p_new)

    # 결과가 거의 0이어야 함 (항등 변환)
    diff = np.abs(p_new).max()
    print(f"    W(p)·W(p)^{{-1}} residual: {diff:.2e}")
    assert diff < 1e-14, f"not identity: max={diff:.2e}"

runner.run("7-3 Warp composition identity (affine)", test_warp_update_composition_identity)


# =============================================================================
#  TEST 8: get_num_params
# =============================================================================
print("\n" + "=" * 70)
print("TEST 8: Utilities")
print("=" * 70)


def test_get_num_params():
    """파라미터 개수"""
    assert get_num_params_numba(AFFINE) == 6
    assert get_num_params_numba(QUADRATIC) == 12

runner.run("8-1 get_num_params", test_get_num_params)


# =============================================================================
#  TEST 9: 성능 벤치마크
# =============================================================================
print("\n" + "=" * 70)
print("TEST 9: Performance Benchmark")
print("=" * 70)


def test_performance():
    """Numba vs 기존 구현 성능 비교"""
    dfdx, dfdy = _make_gradients()
    n_iter = 2000

    # --- Warp (Affine) ---
    p6 = np.array([0.5, 0.01, -0.02, -0.3, 0.005, -0.01], dtype=np.float64)
    dp6 = np.array([0.001, 0.0005, -0.0003, -0.002, 0.0001, 0.0002],
                   dtype=np.float64)

    t0 = time.perf_counter()
    for _ in range(n_iter):
        warp_affine_ref(p6, xsi_ref, eta_ref)
    t_ref_warp = time.perf_counter() - t0

    xw_nb = np.empty(N_PIXELS, dtype=np.float64)
    ew_nb = np.empty(N_PIXELS, dtype=np.float64)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        warp_affine_numba(p6, xsi_nb, eta_nb, xw_nb, ew_nb)
    t_nb_warp = time.perf_counter() - t0

    # --- Warp Update (Affine) ---
    t0 = time.perf_counter()
    for _ in range(n_iter):
        update_warp_ref(p6, dp6, 'affine')
    t_ref_update = time.perf_counter() - t0

    p_new6 = np.empty(6, dtype=np.float64)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        update_warp_affine_numba(p6, dp6, p_new6)
    t_nb_update = time.perf_counter() - t0

    # --- Steepest Descent (Affine) ---
    t0 = time.perf_counter()
    for _ in range(n_iter):
        sd_affine_ref(dfdx, dfdy, xsi_ref, eta_ref)
    t_ref_sd = time.perf_counter() - t0

    J_nb = np.empty((N_PIXELS, 6), dtype=np.float64)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        sd_affine_numba(dfdx, dfdy, xsi_nb, eta_nb, J_nb)
    t_nb_sd = time.perf_counter() - t0

    # --- Hessian ---
    J_ref = sd_affine_ref(dfdx, dfdy, xsi_ref, eta_ref)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        hessian_ref(J_ref)
    t_ref_h = time.perf_counter() - t0

    H_nb = np.empty((6, 6), dtype=np.float64)
    t0 = time.perf_counter()
    for _ in range(n_iter):
        hessian_numba(J_nb, H_nb)
    t_nb_h = time.perf_counter() - t0

    print(f"\n    {n_iter} iterations, subset={SUBSET_SIZE}x{SUBSET_SIZE}:")
    print(f"      Warp (affine):   ref={t_ref_warp*1000:.1f}ms  "
          f"numba={t_nb_warp*1000:.1f}ms  "
          f"({t_ref_warp/t_nb_warp:.1f}x)")
    print(f"      Warp Update:     ref={t_ref_update*1000:.1f}ms  "
          f"numba={t_nb_update*1000:.1f}ms  "
          f"({t_ref_update/t_nb_update:.1f}x)")
    print(f"      Steepest Desc:   ref={t_ref_sd*1000:.1f}ms  "
          f"numba={t_nb_sd*1000:.1f}ms  "
          f"({t_ref_sd/t_nb_sd:.1f}x)")
    print(f"      Hessian:         ref={t_ref_h*1000:.1f}ms  "
          f"numba={t_nb_h*1000:.1f}ms  "
          f"({t_ref_h/t_nb_h:.1f}x)")

runner.run("9-1 Performance comparison", test_performance)


# =============================================================================
#  최종 결과
# =============================================================================
print()
all_passed = runner.summary()

if all_passed:
    print("\nAll tests passed — Numba shape functions verified!")
else:
    print("\nSome tests failed. Check errors above.")

sys.exit(0 if all_passed else 1)
