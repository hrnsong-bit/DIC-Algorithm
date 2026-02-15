"""
Numba B-spline 보간 검증 테스트

검증 항목:
    1. scipy 대비 정밀도 (비트 정확 수준)
    2. 정수 좌표 정확 복원 (interpolation property)
    3. 경계 조건 (mode='constant', cval=0)
    4. 다양한 이미지/좌표 조건
    5. 성능 비교 (scipy vs numba)

사용법:
    cd /home/user/webapp
    python -m pytest tests/test_interpolation_numba.py -v
    # 또는 직접 실행:
    python tests/test_interpolation_numba.py
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
from scipy.ndimage import map_coordinates, spline_filter

from speckle.core.optimization.interpolation_numba import (
    _beta3,
    _beta5,
    _mirror_index,
    prefilter_image,
    _interp2d_cubic,
    _interp2d_quintic,
    interp2d_batch_cubic,
    interp2d_batch_quintic,
    interp2d,
    is_inside_batch,
    warmup_numba_interp,
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
#  유틸리티
# =============================================================================

def generate_smooth_image(h=200, w=200, seed=42):
    """다양한 주파수 성분을 가진 부드러운 테스트 이미지"""
    local_rng = np.random.default_rng(seed)
    img = np.zeros((h, w), dtype=np.float64)
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    # 저주파 + 중주파 성분
    img += 128.0
    img += 40.0 * np.sin(2 * np.pi * x / w * 3) * np.cos(2 * np.pi * y / h * 2)
    img += 20.0 * np.sin(2 * np.pi * x / w * 7 + 0.5)
    img += 15.0 * np.cos(2 * np.pi * y / h * 5 + 1.2)
    # 약간의 노이즈
    img += local_rng.normal(0, 2, (h, w))
    return np.clip(img, 0, 255)


def scipy_interp(coeffs, y_coords, x_coords, order):
    """scipy 기준 보간"""
    return map_coordinates(coeffs, [y_coords, x_coords],
                           order=order, mode='constant',
                           cval=0.0, prefilter=False)


# =============================================================================
#  JIT 워밍업
# =============================================================================

print("JIT warmup...")
t0 = time.perf_counter()
warmup_numba_interp()
print(f"JIT warmup: {time.perf_counter() - t0:.2f}s\n")


# =============================================================================
#  TEST 1: Basis Function 검증
# =============================================================================
print("=" * 70)
print("TEST 1: B-spline Basis Functions")
print("=" * 70)


def test_beta3_partition_of_unity():
    """Cubic basis: sum over integer shifts = 1 (partition of unity)"""
    for x in np.linspace(-0.5, 0.5, 100):
        total = sum(_beta3(x - k) for k in range(-3, 4))
        assert abs(total - 1.0) < 1e-14, \
            f"beta3 partition of unity failed at x={x}: sum={total}"

runner.run("1-1 beta3 partition of unity", test_beta3_partition_of_unity)


def test_beta5_partition_of_unity():
    """Quintic basis: sum over integer shifts = 1"""
    for x in np.linspace(-0.5, 0.5, 100):
        total = sum(_beta5(x - k) for k in range(-5, 6))
        assert abs(total - 1.0) < 1e-14, \
            f"beta5 partition of unity failed at x={x}: sum={total}"

runner.run("1-2 beta5 partition of unity", test_beta5_partition_of_unity)


def test_beta3_symmetry():
    """beta3(-x) == beta3(x)"""
    for x in np.linspace(0, 2.5, 50):
        assert abs(_beta3(x) - _beta3(-x)) < 1e-15, \
            f"beta3 not symmetric at x={x}"

runner.run("1-3 beta3 symmetry", test_beta3_symmetry)


def test_beta5_symmetry():
    """beta5(-x) == beta5(x)"""
    for x in np.linspace(0, 3.5, 50):
        assert abs(_beta5(x) - _beta5(-x)) < 1e-15, \
            f"beta5 not symmetric at x={x}"

runner.run("1-4 beta5 symmetry", test_beta5_symmetry)


def test_beta3_known_values():
    """Cubic basis at key points"""
    assert abs(_beta3(0.0) - 2.0 / 3.0) < 1e-15
    assert abs(_beta3(1.0) - 1.0 / 6.0) < 1e-15
    assert abs(_beta3(2.0) - 0.0) < 1e-15
    assert abs(_beta3(0.5) - (2.0/3.0 - 0.25 + 0.0625)) < 1e-15

runner.run("1-5 beta3 known values", test_beta3_known_values)


def test_beta5_known_values():
    """Quintic basis at key points"""
    assert abs(_beta5(0.0) - 11.0 / 20.0) < 1e-15
    # beta5(1): 17/40 + 5/8 - 7/4 + 5/4 - 3/8 + 1/24 = 11/120
    val_at_1 = 17.0/40.0 + 5.0/8.0 - 7.0/4.0 + 5.0/4.0 - 3.0/8.0 + 1.0/24.0
    assert abs(_beta5(1.0) - val_at_1) < 1e-14, \
        f"beta5(1)={_beta5(1.0)}, expected={val_at_1}"
    assert abs(_beta5(3.0) - 0.0) < 1e-15

runner.run("1-6 beta5 known values", test_beta5_known_values)


def test_beta3_continuity():
    """beta3 경계에서 C2 연속"""
    eps = 1e-8
    # x=1 경계
    left = _beta3(1.0 - eps)
    right = _beta3(1.0 + eps)
    center = _beta3(1.0)
    assert abs(left - center) < 1e-6, f"beta3 discontinuous at 1: left={left}, center={center}"
    assert abs(right - center) < 1e-6, f"beta3 discontinuous at 1: right={right}, center={center}"

runner.run("1-7 beta3 continuity at boundaries", test_beta3_continuity)


def test_beta5_continuity():
    """beta5 경계에서 연속"""
    eps = 1e-8
    for boundary in [1.0, 2.0]:
        left = _beta5(boundary - eps)
        right = _beta5(boundary + eps)
        center = _beta5(boundary)
        assert abs(left - center) < 1e-6, \
            f"beta5 discontinuous at {boundary}: left={left}, center={center}"
        assert abs(right - center) < 1e-6, \
            f"beta5 discontinuous at {boundary}: right={right}, center={center}"

runner.run("1-8 beta5 continuity at boundaries", test_beta5_continuity)


# =============================================================================
#  TEST 2: 정수 좌표 정확 복원 (Interpolation Property)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 2: Integer Coordinate Exact Reproduction")
print("=" * 70)


def test_integer_coords_cubic():
    """정수 좌표에서 원본 값 정확 복원 — cubic"""
    img = generate_smooth_image(50, 50)
    coeffs = prefilter_image(img, order=3)
    margin = 2
    max_err = 0.0
    for y in range(margin, 50 - margin):
        for x in range(margin, 50 - margin):
            val = _interp2d_cubic(coeffs, float(y), float(x))
            err = abs(val - img[y, x])
            max_err = max(max_err, err)
    print(f"    max error at integer coords: {max_err:.2e}")
    assert max_err < 1e-10, f"cubic integer reproduction failed: {max_err:.2e}"

runner.run("2-1 Cubic: integer coords reproduce original", test_integer_coords_cubic)


def test_integer_coords_quintic():
    """정수 좌표에서 원본 값 정확 복원 — quintic"""
    img = generate_smooth_image(50, 50)
    coeffs = prefilter_image(img, order=5)
    margin = 3
    max_err = 0.0
    for y in range(margin, 50 - margin):
        for x in range(margin, 50 - margin):
            val = _interp2d_quintic(coeffs, float(y), float(x))
            err = abs(val - img[y, x])
            max_err = max(max_err, err)
    print(f"    max error at integer coords: {max_err:.2e}")
    assert max_err < 1e-10, f"quintic integer reproduction failed: {max_err:.2e}"

runner.run("2-2 Quintic: integer coords reproduce original", test_integer_coords_quintic)


# =============================================================================
#  TEST 3: scipy 대비 정밀도 (핵심 검증)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 3: Accuracy vs scipy (core validation)")
print("=" * 70)


def test_vs_scipy_cubic_random():
    """Cubic: 랜덤 좌표에서 scipy vs numba 비교"""
    img = generate_smooth_image(200, 200)
    coeffs = prefilter_image(img, order=3)
    n_test = 10000
    margin = 2
    y = rng.uniform(margin, 200 - margin - 1, n_test)
    x = rng.uniform(margin, 200 - margin - 1, n_test)

    ref = scipy_interp(coeffs, y, x, order=3)
    numba_result = interp2d_batch_cubic(coeffs, y, x)

    diff = np.abs(ref - numba_result)
    print(f"    {n_test} random points: max={diff.max():.2e}, "
          f"mean={diff.mean():.2e}, rms={np.sqrt(np.mean(diff**2)):.2e}")
    assert diff.max() < 1e-10, \
        f"cubic vs scipy max error: {diff.max():.2e}"

runner.run("3-1 Cubic vs scipy: 10k random coords", test_vs_scipy_cubic_random)


def test_vs_scipy_quintic_random():
    """Quintic: 랜덤 좌표에서 scipy vs numba 비교"""
    img = generate_smooth_image(200, 200)
    coeffs = prefilter_image(img, order=5)
    n_test = 10000
    margin = 3
    y = rng.uniform(margin, 200 - margin - 1, n_test)
    x = rng.uniform(margin, 200 - margin - 1, n_test)

    ref = scipy_interp(coeffs, y, x, order=5)
    numba_result = interp2d_batch_quintic(coeffs, y, x)

    diff = np.abs(ref - numba_result)
    print(f"    {n_test} random points: max={diff.max():.2e}, "
          f"mean={diff.mean():.2e}, rms={np.sqrt(np.mean(diff**2)):.2e}")
    assert diff.max() < 1e-10, \
        f"quintic vs scipy max error: {diff.max():.2e}"

runner.run("3-2 Quintic vs scipy: 10k random coords", test_vs_scipy_quintic_random)


def test_vs_scipy_cubic_subpixel_grid():
    """Cubic: 0.1 간격 서브픽셀 격자에서 정밀 비교"""
    img = generate_smooth_image(50, 50)
    coeffs = prefilter_image(img, order=3)
    margin = 2

    y_vals = np.arange(margin + 0.05, 50 - margin - 1, 0.1)
    x_vals = np.arange(margin + 0.05, 50 - margin - 1, 0.1)
    yy, xx = np.meshgrid(y_vals, x_vals, indexing='ij')
    y_flat = yy.ravel()
    x_flat = xx.ravel()

    ref = scipy_interp(coeffs, y_flat, x_flat, order=3)
    numba_result = interp2d_batch_cubic(coeffs, y_flat, x_flat)

    diff = np.abs(ref - numba_result)
    print(f"    {len(y_flat)} grid points: max={diff.max():.2e}")
    assert diff.max() < 1e-10

runner.run("3-3 Cubic vs scipy: subpixel grid", test_vs_scipy_cubic_subpixel_grid)


def test_vs_scipy_quintic_subpixel_grid():
    """Quintic: 0.1 간격 서브픽셀 격자에서 정밀 비교"""
    img = generate_smooth_image(50, 50)
    coeffs = prefilter_image(img, order=5)
    margin = 3

    y_vals = np.arange(margin + 0.05, 50 - margin - 1, 0.1)
    x_vals = np.arange(margin + 0.05, 50 - margin - 1, 0.1)
    yy, xx = np.meshgrid(y_vals, x_vals, indexing='ij')
    y_flat = yy.ravel()
    x_flat = xx.ravel()

    ref = scipy_interp(coeffs, y_flat, x_flat, order=5)
    numba_result = interp2d_batch_quintic(coeffs, y_flat, x_flat)

    diff = np.abs(ref - numba_result)
    print(f"    {len(y_flat)} grid points: max={diff.max():.2e}")
    assert diff.max() < 1e-10

runner.run("3-4 Quintic vs scipy: subpixel grid", test_vs_scipy_quintic_subpixel_grid)


def test_vs_scipy_cubic_various_images():
    """Cubic: 다양한 이미지 유형에서 검증"""
    images = {
        'uniform': np.full((100, 100), 128.0),
        'gradient_x': np.tile(np.arange(100, dtype=np.float64), (100, 1)),
        'gradient_y': np.tile(np.arange(100, dtype=np.float64), (100, 1)).T,
        'quadratic': np.fromfunction(
            lambda y, x: 0.01 * (x - 50)**2 + 0.01 * (y - 50)**2,
            (100, 100)),
        'random_smooth': generate_smooth_image(100, 100, seed=123),
    }
    margin = 2
    for name, img in images.items():
        coeffs = prefilter_image(img, order=3)
        n = 2000
        y = rng.uniform(margin, 100 - margin - 1, n)
        x = rng.uniform(margin, 100 - margin - 1, n)
        ref = scipy_interp(coeffs, y, x, order=3)
        result = interp2d_batch_cubic(coeffs, y, x)
        diff = np.abs(ref - result)
        assert diff.max() < 1e-10, \
            f"cubic {name}: max error = {diff.max():.2e}"
    print(f"    5 image types verified")

runner.run("3-5 Cubic vs scipy: various image types", test_vs_scipy_cubic_various_images)


def test_vs_scipy_quintic_various_images():
    """Quintic: 다양한 이미지 유형에서 검증"""
    images = {
        'uniform': np.full((100, 100), 128.0),
        'gradient_x': np.tile(np.arange(100, dtype=np.float64), (100, 1)),
        'gradient_y': np.tile(np.arange(100, dtype=np.float64), (100, 1)).T,
        'quadratic': np.fromfunction(
            lambda y, x: 0.01 * (x - 50)**2 + 0.01 * (y - 50)**2,
            (100, 100)),
        'random_smooth': generate_smooth_image(100, 100, seed=123),
    }
    margin = 3
    for name, img in images.items():
        coeffs = prefilter_image(img, order=5)
        n = 2000
        y = rng.uniform(margin, 100 - margin - 1, n)
        x = rng.uniform(margin, 100 - margin - 1, n)
        ref = scipy_interp(coeffs, y, x, order=5)
        result = interp2d_batch_quintic(coeffs, y, x)
        diff = np.abs(ref - result)
        assert diff.max() < 1e-10, \
            f"quintic {name}: max error = {diff.max():.2e}"
    print(f"    5 image types verified")

runner.run("3-6 Quintic vs scipy: various image types", test_vs_scipy_quintic_various_images)


def test_vs_scipy_unified_interface():
    """interp2d 통합 인터페이스 검증"""
    img = generate_smooth_image(100, 100)
    for order in [3, 5]:
        coeffs = prefilter_image(img, order=order)
        margin = order // 2 + 1
        n = 5000
        y = rng.uniform(margin, 100 - margin - 1, n)
        x = rng.uniform(margin, 100 - margin - 1, n)
        ref = scipy_interp(coeffs, y, x, order=order)
        result = interp2d(coeffs, y, x, order)
        diff = np.abs(ref - result)
        assert diff.max() < 1e-10, \
            f"interp2d order={order}: max error = {diff.max():.2e}"
    print(f"    order=3 and order=5 verified")

runner.run("3-7 Unified interp2d interface", test_vs_scipy_unified_interface)


# =============================================================================
#  TEST 4: 경계 조건
# =============================================================================
print("\n" + "=" * 70)
print("TEST 4: Boundary Conditions")
print("=" * 70)


def test_boundary_cubic():
    """Cubic: 경계 근처 좌표에서 scipy와 일치"""
    img = generate_smooth_image(30, 30)
    coeffs = prefilter_image(img, order=3)

    # 경계 바로 안쪽 좌표들
    boundary_coords = [
        (1.5, 1.5), (1.0, 15.0), (15.0, 1.0),
        (28.0, 15.0), (15.0, 28.0), (27.5, 27.5),
        (0.5, 15.0), (15.0, 0.5),
    ]
    for (y, x) in boundary_coords:
        ya, xa = np.array([y]), np.array([x])
        ref = scipy_interp(coeffs, ya, xa, order=3)[0]
        val = _interp2d_cubic(coeffs, y, x)
        err = abs(ref - val)
        assert err < 1e-10, \
            f"cubic boundary ({y},{x}): err={err:.2e}, scipy={ref}, numba={val}"
    print(f"    {len(boundary_coords)} boundary coords verified")

runner.run("4-1 Cubic: boundary coords match scipy", test_boundary_cubic)


def test_boundary_quintic():
    """Quintic: 경계 근처 좌표에서 scipy와 일치"""
    img = generate_smooth_image(30, 30)
    coeffs = prefilter_image(img, order=5)

    boundary_coords = [
        (2.5, 2.5), (2.0, 15.0), (15.0, 2.0),
        (27.0, 15.0), (15.0, 27.0), (26.5, 26.5),
        (0.5, 15.0), (15.0, 0.5),
    ]
    for (y, x) in boundary_coords:
        ya, xa = np.array([y]), np.array([x])
        ref = scipy_interp(coeffs, ya, xa, order=5)[0]
        val = _interp2d_quintic(coeffs, y, x)
        err = abs(ref - val)
        assert err < 1e-10, \
            f"quintic boundary ({y},{x}): err={err:.2e}, scipy={ref}, numba={val}"
    print(f"    {len(boundary_coords)} boundary coords verified")

runner.run("4-2 Quintic: boundary coords match scipy", test_boundary_quintic)


def test_is_inside_batch_check():
    """is_inside_batch 경계 체크 정확성"""
    h, w = 100, 100
    # 안전한 좌표
    y_safe = np.array([10.0, 50.0, 90.0], dtype=np.float64)
    x_safe = np.array([10.0, 50.0, 90.0], dtype=np.float64)
    assert is_inside_batch(y_safe, x_safe, h, w, 3) == True
    assert is_inside_batch(y_safe, x_safe, h, w, 5) == True

    # quintic 경계 밖
    y_edge = np.array([2.5], dtype=np.float64)
    x_edge = np.array([50.0], dtype=np.float64)
    assert is_inside_batch(y_edge, x_edge, h, w, 5) == False
    # cubic은 OK
    assert is_inside_batch(y_edge, x_edge, h, w, 3) == True

runner.run("4-3 is_inside_batch correctness", test_is_inside_batch_check)


# =============================================================================
#  TEST 5: DIC 실전 조건 시뮬레이션
# =============================================================================
print("\n" + "=" * 70)
print("TEST 5: DIC Realistic Conditions")
print("=" * 70)


def test_dic_subset_interpolation():
    """DIC subset 보간 시뮬레이션: 21x21 subset, 서브픽셀 이동"""
    import cv2
    np.random.seed(42)
    # 스페클 패턴 유사 이미지
    img = np.random.rand(300, 300).astype(np.float64) * 200
    img = cv2.GaussianBlur(img, (5, 5), 2.0)

    subset_size = 21
    half = subset_size // 2
    cx, cy = 150, 150

    for order in [3, 5]:
        coeffs = prefilter_image(img, order=order)
        margin = order // 2 + 1

        # 서브픽셀 변위
        for du, dv in [(0.3, -0.7), (0.001, 0.999), (-0.5, 0.5)]:
            coords_1d = np.arange(-half, half + 1, dtype=np.float64)
            eta_2d, xsi_2d = np.meshgrid(coords_1d, coords_1d, indexing='ij')
            y_coords = (cy + eta_2d.ravel() + dv)
            x_coords = (cx + xsi_2d.ravel() + du)

            ref = scipy_interp(coeffs, y_coords, x_coords, order=order)
            result = interp2d(coeffs, y_coords, x_coords, order)

            diff = np.abs(ref - result)
            assert diff.max() < 1e-10, \
                f"DIC subset order={order}, disp=({du},{dv}): " \
                f"max error = {diff.max():.2e}"

    print(f"    3 displacements x 2 orders: all < 1e-10")

runner.run("5-1 DIC subset interpolation accuracy", test_dic_subset_interpolation)


def test_dic_large_scale():
    """대규모 POI 정밀도 (9000+ POI 시뮬레이션)"""
    import cv2
    np.random.seed(42)
    img = np.random.rand(500, 500).astype(np.float64) * 200
    img = cv2.GaussianBlur(img, (5, 5), 2.0)

    subset_size = 21
    n_pixels = subset_size * subset_size  # 441
    spacing = 5
    half = subset_size // 2

    for order in [3, 5]:
        coeffs = prefilter_image(img, order=order)
        margin = order // 2 + 1 + half

        # POI 그리드
        y_pois = np.arange(margin, 500 - margin, spacing)
        x_pois = np.arange(margin, 500 - margin, spacing)
        yy, xx = np.meshgrid(y_pois, x_pois, indexing='ij')
        poi_y = yy.ravel()
        poi_x = xx.ravel()
        n_poi = len(poi_y)

        # 랜덤 서브픽셀 이동
        du = rng.uniform(-1, 1, n_poi)
        dv = rng.uniform(-1, 1, n_poi)

        # 첫 100 POI만 정밀도 검증 (속도)
        n_check = min(100, n_poi)
        max_err = 0.0
        for i in range(n_check):
            coords_1d = np.arange(-half, half + 1, dtype=np.float64)
            eta_2d, xsi_2d = np.meshgrid(coords_1d, coords_1d, indexing='ij')
            y_c = (poi_y[i] + eta_2d.ravel() + dv[i])
            x_c = (poi_x[i] + xsi_2d.ravel() + du[i])

            ref = scipy_interp(coeffs, y_c, x_c, order=order)
            result = interp2d(coeffs, y_c, x_c, order)
            max_err = max(max_err, np.abs(ref - result).max())

        assert max_err < 1e-10, \
            f"order={order}: max error = {max_err:.2e}"
        print(f"    order={order}: {n_poi} POIs, checked {n_check}, max_err={max_err:.2e}")

runner.run("5-2 DIC large-scale accuracy", test_dic_large_scale)


# =============================================================================
#  TEST 6: 성능 벤치마크
# =============================================================================
print("\n" + "=" * 70)
print("TEST 6: Performance Benchmark")
print("=" * 70)


def test_performance_comparison():
    """Numba vs scipy 성능 비교"""
    import cv2
    np.random.seed(42)
    img = np.random.rand(500, 500).astype(np.float64) * 200
    img = cv2.GaussianBlur(img, (5, 5), 2.0)

    subset_size = 21
    n_pixels = subset_size * subset_size  # 441
    n_poi = 2000

    for order in [3, 5]:
        coeffs = prefilter_image(img, order=order)
        margin = order // 2 + 1

        # 좌표 생성 (2000 POI x 441 pixels = 882,000 calls)
        all_y = rng.uniform(margin + 10, 490 - 10, n_poi * n_pixels)
        all_x = rng.uniform(margin + 10, 490 - 10, n_poi * n_pixels)

        # --- scipy ---
        t0 = time.perf_counter()
        ref = scipy_interp(coeffs, all_y, all_x, order=order)
        t_scipy = time.perf_counter() - t0

        # --- numba batch ---
        t0 = time.perf_counter()
        result = interp2d(coeffs, all_y, all_x, order)
        t_numba = time.perf_counter() - t0

        # --- numba per-POI (ICGN 실사용 패턴) ---
        t0 = time.perf_counter()
        for i in range(n_poi):
            start = i * n_pixels
            end = start + n_pixels
            interp2d(coeffs, all_y[start:end], all_x[start:end], order)
        t_numba_perpoi = time.perf_counter() - t0

        ratio_batch = t_scipy / t_numba if t_numba > 0 else float('inf')
        ratio_perpoi = t_scipy / t_numba_perpoi if t_numba_perpoi > 0 else float('inf')

        print(f"\n    order={order} ({n_poi} POI x {n_pixels} px = "
              f"{n_poi * n_pixels:,} interps):")
        print(f"      scipy:             {t_scipy*1000:.1f} ms")
        print(f"      numba batch:       {t_numba*1000:.1f} ms  "
              f"({ratio_batch:.2f}x vs scipy)")
        print(f"      numba per-POI:     {t_numba_perpoi*1000:.1f} ms  "
              f"({ratio_perpoi:.2f}x vs scipy)")
        print(f"      per-POI overhead:  "
              f"{(t_numba_perpoi / t_numba - 1) * 100:.1f}%")

    # 정밀도 확인도 포함
    diff = np.abs(ref - result)
    assert diff.max() < 1e-10

runner.run("6-1 Performance: numba vs scipy", test_performance_comparison)


# =============================================================================
#  TEST 7: 경계 미러링 심층 검증 (mirror boundary edge cases)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 7: Mirror Boundary Deep Validation")
print("=" * 70)


def test_mirror_index_values():
    """_mirror_index 함수 직접 검증"""
    from speckle.core.optimization.interpolation_numba import _mirror_index
    from numba import int64 as nb_int64

    # 기본 규칙: idx < 0 → -idx,  idx >= N → 2*(N-1) - idx
    N = 30
    # 내부 인덱스: 변환 없음
    assert _mirror_index(nb_int64(0), nb_int64(N)) == 0
    assert _mirror_index(nb_int64(15), nb_int64(N)) == 15
    assert _mirror_index(nb_int64(29), nb_int64(N)) == 29
    # 음수 미러
    assert _mirror_index(nb_int64(-1), nb_int64(N)) == 1
    assert _mirror_index(nb_int64(-2), nb_int64(N)) == 2
    assert _mirror_index(nb_int64(-3), nb_int64(N)) == 3
    # 초과 미러
    assert _mirror_index(nb_int64(30), nb_int64(N)) == 28
    assert _mirror_index(nb_int64(31), nb_int64(N)) == 27
    assert _mirror_index(nb_int64(32), nb_int64(N)) == 26

runner.run("7-1 _mirror_index direct verification", test_mirror_index_values)


def test_boundary_extreme_coords_cubic():
    """Cubic: 극단적 경계 좌표 (0.0, 0.01, N-1.0, N-1.01)에서 scipy 일치"""
    img = generate_smooth_image(30, 30)
    coeffs = prefilter_image(img, order=3)

    extreme_coords = [
        (0.0, 15.0), (0.01, 15.0), (0.1, 0.1),
        (29.0, 15.0), (28.99, 15.0), (28.9, 28.9),
        (15.0, 0.0), (15.0, 0.01), (15.0, 29.0), (15.0, 28.99),
        (0.0, 0.0), (29.0, 29.0), (0.0, 29.0), (29.0, 0.0),
    ]
    max_err = 0.0
    for (y, x) in extreme_coords:
        ya, xa = np.array([y]), np.array([x])
        ref = scipy_interp(coeffs, ya, xa, order=3)[0]
        val = _interp2d_cubic(coeffs, y, x)
        err = abs(ref - val)
        max_err = max(max_err, err)
        assert err < 1e-10, \
            f"cubic extreme ({y},{x}): err={err:.2e}, scipy={ref}, numba={val}"
    print(f"    {len(extreme_coords)} extreme boundary coords: max_err={max_err:.2e}")

runner.run("7-2 Cubic: extreme boundary coords", test_boundary_extreme_coords_cubic)


def test_boundary_extreme_coords_quintic():
    """Quintic: 극단적 경계 좌표에서 scipy 일치"""
    img = generate_smooth_image(30, 30)
    coeffs = prefilter_image(img, order=5)

    extreme_coords = [
        (0.0, 15.0), (0.01, 15.0), (0.1, 0.1),
        (29.0, 15.0), (28.99, 15.0), (28.9, 28.9),
        (15.0, 0.0), (15.0, 0.01), (15.0, 29.0), (15.0, 28.99),
        (0.0, 0.0), (29.0, 29.0), (0.0, 29.0), (29.0, 0.0),
        (1.0, 15.0), (15.0, 1.0), (28.0, 28.0),
    ]
    max_err = 0.0
    for (y, x) in extreme_coords:
        ya, xa = np.array([y]), np.array([x])
        ref = scipy_interp(coeffs, ya, xa, order=5)[0]
        val = _interp2d_quintic(coeffs, y, x)
        err = abs(ref - val)
        max_err = max(max_err, err)
        assert err < 1e-10, \
            f"quintic extreme ({y},{x}): err={err:.2e}, scipy={ref}, numba={val}"
    print(f"    {len(extreme_coords)} extreme boundary coords: max_err={max_err:.2e}")

runner.run("7-3 Quintic: extreme boundary coords", test_boundary_extreme_coords_quintic)


def test_boundary_sweep_cubic():
    """Cubic: y축 경계 스윕 (y=0.0~2.0, 0.05 간격)으로 전 구간 scipy 일치"""
    img = generate_smooth_image(50, 50)
    coeffs = prefilter_image(img, order=3)
    max_err = 0.0
    n_checked = 0
    for y in np.arange(0.0, 2.01, 0.05):
        for x in [15.0, 25.0]:
            ya, xa = np.array([y]), np.array([x])
            ref = scipy_interp(coeffs, ya, xa, order=3)[0]
            val = _interp2d_cubic(coeffs, y, x)
            err = abs(ref - val)
            max_err = max(max_err, err)
            n_checked += 1
    assert max_err < 1e-10, f"cubic boundary sweep: max_err={max_err:.2e}"
    print(f"    {n_checked} coords swept near y-boundary: max_err={max_err:.2e}")

runner.run("7-4 Cubic: boundary sweep near edges", test_boundary_sweep_cubic)


def test_boundary_sweep_quintic():
    """Quintic: y축 경계 스윕 (y=0.0~3.0, 0.05 간격)으로 전 구간 scipy 일치"""
    img = generate_smooth_image(50, 50)
    coeffs = prefilter_image(img, order=5)
    max_err = 0.0
    n_checked = 0
    for y in np.arange(0.0, 3.01, 0.05):
        for x in [15.0, 25.0]:
            ya, xa = np.array([y]), np.array([x])
            ref = scipy_interp(coeffs, ya, xa, order=5)[0]
            val = _interp2d_quintic(coeffs, y, x)
            err = abs(ref - val)
            max_err = max(max_err, err)
            n_checked += 1
    assert max_err < 1e-10, f"quintic boundary sweep: max_err={max_err:.2e}"
    print(f"    {n_checked} coords swept near y-boundary: max_err={max_err:.2e}")

runner.run("7-5 Quintic: boundary sweep near edges", test_boundary_sweep_quintic)


# =============================================================================
#  TEST 8: 보간 수학적 성질 검증
# =============================================================================
print("\n" + "=" * 70)
print("TEST 8: Mathematical Properties of Interpolation")
print("=" * 70)


def test_linear_function_exactness_cubic():
    """Cubic B-spline은 2차 다항식까지 정확히 재현 (order 3 → degree 3-1=2)
    
    Note: prefilter의 mode='constant' 경계 효과를 피하기 위해
    충분히 큰 이미지에서 내부 영역만 테스트한다.
    """
    h, w = 200, 200
    y_arr, x_arr = np.mgrid[0:h, 0:w].astype(np.float64)
    # 2차 다항식: f(y,x) = 0.5*y + 1.3*x + 0.01*y*x
    img = 0.5 * y_arr + 1.3 * x_arr + 0.01 * y_arr * x_arr
    coeffs = prefilter_image(img, order=3)
    # 경계 prefilter 영향을 피하기 위해 충분한 margin 사용
    margin = 20

    n_test = 1000
    y_test = rng.uniform(margin, h - margin - 1, n_test)
    x_test = rng.uniform(margin, w - margin - 1, n_test)

    expected = 0.5 * y_test + 1.3 * x_test + 0.01 * y_test * x_test
    result = interp2d_batch_cubic(coeffs, y_test, x_test)

    diff = np.abs(expected - result)
    print(f"    quadratic function max error: {diff.max():.2e}")
    assert diff.max() < 1e-8, \
        f"cubic should reproduce quadratic: {diff.max():.2e}"

runner.run("8-1 Cubic: quadratic polynomial exactness", test_linear_function_exactness_cubic)


def test_polynomial_exactness_quintic():
    """Quintic B-spline은 4차 다항식까지 정확히 재현 (order 5 → degree 5-1=4)

    Note: prefilter의 mode='constant' 경계 효과를 피하기 위해
    충분히 큰 이미지에서 내부 영역만 테스트한다.
    """
    h, w = 200, 200
    y_arr, x_arr = np.mgrid[0:h, 0:w].astype(np.float64)
    # 4차 다항식 (작은 계수로 이미지 범위 제한)
    img = (1.0 + 0.1 * x_arr + 0.05 * y_arr
           + 0.001 * x_arr**2 + 0.002 * y_arr**2
           + 0.0005 * x_arr * y_arr
           + 1e-5 * x_arr**3 + 1e-5 * y_arr**3
           + 1e-7 * x_arr**4 + 1e-7 * y_arr**4)
    coeffs = prefilter_image(img, order=5)
    # 경계 prefilter 영향을 피하기 위해 충분한 margin 사용
    margin = 30

    n_test = 1000
    y_test = rng.uniform(margin, h - margin - 1, n_test)
    x_test = rng.uniform(margin, w - margin - 1, n_test)

    expected = (1.0 + 0.1 * x_test + 0.05 * y_test
                + 0.001 * x_test**2 + 0.002 * y_test**2
                + 0.0005 * x_test * y_test
                + 1e-5 * x_test**3 + 1e-5 * y_test**3
                + 1e-7 * x_test**4 + 1e-7 * y_test**4)
    result = interp2d_batch_quintic(coeffs, y_test, x_test)

    diff = np.abs(expected - result)
    print(f"    4th-degree polynomial max error: {diff.max():.2e}")
    assert diff.max() < 1e-6, \
        f"quintic should reproduce 4th-degree poly well: {diff.max():.2e}"

runner.run("8-2 Quintic: 4th-degree polynomial exactness", test_polynomial_exactness_quintic)


def test_interpolation_gradient_preservation():
    """보간된 gradient가 해석적 gradient에 근사하는지 확인 (DIC 정밀도 핵심)"""
    h, w = 200, 200
    y_arr, x_arr = np.mgrid[0:h, 0:w].astype(np.float64)
    # 부드러운 함수: f(y,x) = 100*sin(2πx/w) * cos(2πy/h)
    img = 100.0 * np.sin(2 * np.pi * x_arr / w) * np.cos(2 * np.pi * y_arr / h)
    coeffs5 = prefilter_image(img, order=5)

    # 중심 근처에서 유한 차분으로 gradient 추정
    eps = 1e-5
    test_y, test_x = 100.3, 100.7
    ya = np.array([test_y])
    xa = np.array([test_x])

    val_c = interp2d_batch_quintic(coeffs5, ya, xa)[0]
    val_dx = interp2d_batch_quintic(coeffs5, ya, np.array([test_x + eps]))[0]
    val_dy = interp2d_batch_quintic(coeffs5, np.array([test_y + eps]), xa)[0]

    num_dfdx = (val_dx - val_c) / eps
    num_dfdy = (val_dy - val_c) / eps

    # 해석적 gradient
    ana_dfdx = 100.0 * (2 * np.pi / w) * np.cos(2 * np.pi * test_x / w) * \
               np.cos(2 * np.pi * test_y / h)
    ana_dfdy = 100.0 * np.sin(2 * np.pi * test_x / w) * \
               (-2 * np.pi / h) * np.sin(2 * np.pi * test_y / h)

    err_dx = abs(num_dfdx - ana_dfdx)
    err_dy = abs(num_dfdy - ana_dfdy)
    print(f"    dfdx error: {err_dx:.2e} (numerical={num_dfdx:.6f}, "
          f"analytical={ana_dfdx:.6f})")
    print(f"    dfdy error: {err_dy:.2e} (numerical={num_dfdy:.6f}, "
          f"analytical={ana_dfdy:.6f})")
    # 5차 보간은 gradient를 매우 정확하게 보존해야 함
    assert err_dx < 1e-3, f"gradient dfdx error too large: {err_dx}"
    assert err_dy < 1e-3, f"gradient dfdy error too large: {err_dy}"

runner.run("8-3 Gradient preservation (quintic)", test_interpolation_gradient_preservation)


def test_interpolation_symmetry():
    """대칭 이미지에서 대칭 좌표의 보간값이 일치하는지 확인"""
    h, w = 100, 100
    y_arr, x_arr = np.mgrid[0:h, 0:w].astype(np.float64)
    cy, cx = h / 2.0, w / 2.0
    # 방사 대칭 이미지: f = exp(-(r²)/σ²)
    r2 = (y_arr - cy)**2 + (x_arr - cx)**2
    img = 200.0 * np.exp(-r2 / (2 * 20.0**2))

    for order in [3, 5]:
        coeffs = prefilter_image(img, order=order)
        # (cy+δ, cx) vs (cy-δ, cx) 비교 (y-대칭)
        delta = 7.3
        y1, y2 = np.array([cy + delta]), np.array([cy - delta])
        x_c = np.array([cx])
        v1 = interp2d(coeffs, y1, x_c, order)[0]
        v2 = interp2d(coeffs, y2, x_c, order)[0]
        err = abs(v1 - v2)
        assert err < 1e-10, \
            f"order={order} y-symmetry: v1={v1}, v2={v2}, err={err:.2e}"
        # (cy, cx+δ) vs (cy, cx-δ) 비교 (x-대칭)
        x1, x2 = np.array([cx + delta]), np.array([cx - delta])
        v3 = interp2d(coeffs, np.array([cy]), x1, order)[0]
        v4 = interp2d(coeffs, np.array([cy]), x2, order)[0]
        err2 = abs(v3 - v4)
        assert err2 < 1e-10, \
            f"order={order} x-symmetry: v3={v3}, v4={v4}, err={err2:.2e}"
    print(f"    y-symmetry and x-symmetry verified for order=3,5")

runner.run("8-4 Radial symmetry preservation", test_interpolation_symmetry)


# =============================================================================
#  TEST 9: 다양한 이미지 크기 강건성
# =============================================================================
print("\n" + "=" * 70)
print("TEST 9: Robustness Across Image Sizes")
print("=" * 70)


def test_various_image_sizes():
    """다양한 이미지 크기에서 scipy와 일치 확인 (소형~대형)"""
    sizes = [(10, 10), (16, 32), (50, 50), (100, 200), (300, 300)]
    for (h, w) in sizes:
        img = generate_smooth_image(h, w, seed=h * w)
        for order in [3, 5]:
            coeffs = prefilter_image(img, order=order)
            margin = order // 2 + 1
            n_test = min(500, (h - 2 * margin) * (w - 2 * margin))
            if n_test < 10:
                continue
            y = rng.uniform(margin, h - margin - 1, n_test)
            x = rng.uniform(margin, w - margin - 1, n_test)
            ref = scipy_interp(coeffs, y, x, order=order)
            result = interp2d(coeffs, y, x, order)
            diff = np.abs(ref - result)
            assert diff.max() < 1e-10, \
                f"size=({h},{w}) order={order}: max_err={diff.max():.2e}"
    print(f"    {len(sizes)} image sizes x 2 orders verified")

runner.run("9-1 Various image sizes", test_various_image_sizes)


def test_non_square_image():
    """비정방형 이미지 (가로/세로 크기 다름) 처리 확인"""
    for (h, w) in [(20, 80), (80, 20), (30, 100), (100, 30)]:
        img = generate_smooth_image(h, w, seed=h + w)
        for order in [3, 5]:
            coeffs = prefilter_image(img, order=order)
            margin = order // 2 + 1
            n_test = 200
            y = rng.uniform(margin, h - margin - 1, n_test)
            x = rng.uniform(margin, w - margin - 1, n_test)
            ref = scipy_interp(coeffs, y, x, order=order)
            result = interp2d(coeffs, y, x, order)
            diff = np.abs(ref - result)
            assert diff.max() < 1e-10, \
                f"({h}x{w}) order={order}: max_err={diff.max():.2e}"
    print(f"    4 non-square sizes x 2 orders verified")

runner.run("9-2 Non-square images", test_non_square_image)


# =============================================================================
#  TEST 10: ICGN 호환성 (실제 사용 시나리오 검증)
# =============================================================================
print("\n" + "=" * 70)
print("TEST 10: ICGN Compatibility Scenarios")
print("=" * 70)


def test_icgn_warp_pattern():
    """ICGN의 실제 warp 패턴 시뮬레이션: affine 변형 좌표 보간"""
    import cv2
    np.random.seed(42)
    img = np.random.rand(300, 300).astype(np.float64) * 200
    img = cv2.GaussianBlur(img, (5, 5), 2.0)

    subset_size = 21
    half = subset_size // 2
    cx, cy = 150, 150

    # Affine 변형 파라미터 (u, ux, uy, v, vx, vy)
    test_params = [
        (0.3, 0.001, -0.002, -0.5, 0.003, -0.001),  # 약한 변형
        (1.5, 0.01, 0.005, -2.0, -0.005, 0.01),       # 강한 변형
        (0.0, 0.0, 0.0, 0.0, 0.0, 0.0),               # 무변형 (정수)
    ]

    for order in [3, 5]:
        coeffs = prefilter_image(img, order=order)
        for u, ux, uy, v, vx, vy in test_params:
            coords_1d = np.arange(-half, half + 1, dtype=np.float64)
            eta_2d, xsi_2d = np.meshgrid(coords_1d, coords_1d, indexing='ij')
            xsi = xsi_2d.ravel()
            eta = eta_2d.ravel()

            # Warp 좌표 (affine)
            x_def = cx + xsi + u + ux * xsi + uy * eta
            y_def = cy + eta + v + vx * xsi + vy * eta

            ref = scipy_interp(coeffs, y_def, x_def, order=order)
            result = interp2d(coeffs, y_def, x_def, order)
            diff = np.abs(ref - result)
            assert diff.max() < 1e-10, \
                f"order={order}, params=({u},{v}): max_err={diff.max():.2e}"
    print(f"    3 warp patterns x 2 orders verified")

runner.run("10-1 ICGN affine warp pattern", test_icgn_warp_pattern)


def test_icgn_iterative_consistency():
    """반복 보간 호출의 일관성: 동일 좌표에서 반복 호출 시 동일 결과"""
    import cv2
    np.random.seed(42)
    img = np.random.rand(200, 200).astype(np.float64) * 200
    img = cv2.GaussianBlur(img, (5, 5), 2.0)

    for order in [3, 5]:
        coeffs = prefilter_image(img, order=order)
        y = rng.uniform(10, 190, 441)
        x = rng.uniform(10, 190, 441)

        # 50번 반복 호출 (ICGN max_iterations 시뮬레이션)
        results = []
        for _ in range(50):
            results.append(interp2d(coeffs, y, x, order).copy())

        # 모든 결과가 비트 동일해야 함
        for i in range(1, len(results)):
            assert np.array_equal(results[0], results[i]), \
                f"order={order}: iteration {i} differs from iteration 0"
    print(f"    50 iterations x 2 orders: bit-exact consistency verified")

runner.run("10-2 Iterative call consistency", test_icgn_iterative_consistency)


# =============================================================================
#  최종 결과
# =============================================================================
print()
all_passed = runner.summary()

if all_passed:
    print("\nAll tests passed — Numba B-spline interpolation verified!")
else:
    print("\nSome tests failed. Check errors above.")

sys.exit(0 if all_passed else 1)
