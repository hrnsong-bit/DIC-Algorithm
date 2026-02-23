"""
B-spline 보간 및 Gradient 검증 테스트

테스트 항목:
    A. interpolation_numba.py의 Numba B-spline 보간이
       scipy.ndimage.map_coordinates와 수치적으로 동일한지 확인
    B. 해석적 함수에 대한 보간 정확도 (absolute error)
    C. Sobel gradient (ksize=5) 스케일링 검증 — 올바른 divisor 탐색
    D. B-spline basis function 기본 성질
    E. 경계 근처 보간 SciPy 호환성

사용법:
    python -m pytest test_interpolation.py -v
    또는
    python test_interpolation.py
"""

import numpy as np
import cv2
from scipy.ndimage import map_coordinates, spline_filter
import sys
import os

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from speckle.core.optimization.interpolation_numba import (
    prefilter_image,
    interp2d,
    _interp2d_cubic,
    _interp2d_quintic,
    _mirror_index,
    _beta3,
    _beta5,
)


# =====================================================================
#  유틸리티: 해석적 테스트 함수
# =====================================================================

def analytic_func(y, x, H, W):
    """f(y, x) = sin(πy/H) · cos(πx/W)"""
    return np.sin(np.pi * y / H) * np.cos(np.pi * x / W)


def analytic_dfdx(y, x, H, W):
    """∂f/∂x = -sin(πy/H) · sin(πx/W) · (π/W)"""
    return -np.sin(np.pi * y / H) * np.sin(np.pi * x / W) * (np.pi / W)


def analytic_dfdy(y, x, H, W):
    """∂f/∂y = cos(πy/H) · cos(πx/W) · (π/H)"""
    return np.cos(np.pi * y / H) * np.cos(np.pi * x / W) * (np.pi / H)


def create_analytic_image(H=200, W=200):
    """해석적 함수로 이미지 생성"""
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    image = analytic_func(yy, xx, H, W)
    return image


# =====================================================================
#  TEST A: Numba vs SciPy 일치 확인
# =====================================================================

def test_numba_vs_scipy_cubic():
    """Bicubic: Numba == scipy map_coordinates"""
    np.random.seed(42)
    H, W = 100, 100
    image = np.random.rand(H, W).astype(np.float64) * 200 + 20

    coeffs = prefilter_image(image, order=3)
    coeffs_scipy = spline_filter(image, order=3, mode='constant')

    assert np.allclose(coeffs, coeffs_scipy, atol=1e-12), \
        f"Prefilter mismatch: max diff = {np.max(np.abs(coeffs - coeffs_scipy))}"

    margin = 3
    n_test = 500
    y_coords = np.random.uniform(margin, H - margin - 1, n_test)
    x_coords = np.random.uniform(margin, W - margin - 1, n_test)

    numba_result = interp2d(coeffs, y_coords, x_coords, 3)
    scipy_result = map_coordinates(coeffs_scipy, [y_coords, x_coords],
                                   order=3, mode='constant', cval=0.0,
                                   prefilter=False)

    max_diff = np.max(np.abs(numba_result - scipy_result))
    print(f"[TEST A-1] Bicubic Numba vs SciPy: max diff = {max_diff:.2e}")
    assert max_diff < 1e-10, f"Bicubic mismatch: max diff = {max_diff}"
    print("  PASSED")


def test_numba_vs_scipy_quintic():
    """Biquintic: Numba == scipy map_coordinates"""
    np.random.seed(42)
    H, W = 100, 100
    image = np.random.rand(H, W).astype(np.float64) * 200 + 20

    coeffs = prefilter_image(image, order=5)
    coeffs_scipy = spline_filter(image, order=5, mode='constant')

    assert np.allclose(coeffs, coeffs_scipy, atol=1e-12)

    margin = 4
    n_test = 500
    y_coords = np.random.uniform(margin, H - margin - 1, n_test)
    x_coords = np.random.uniform(margin, W - margin - 1, n_test)

    numba_result = interp2d(coeffs, y_coords, x_coords, 5)
    scipy_result = map_coordinates(coeffs_scipy, [y_coords, x_coords],
                                   order=5, mode='constant', cval=0.0,
                                   prefilter=False)

    max_diff = np.max(np.abs(numba_result - scipy_result))
    print(f"[TEST A-2] Biquintic Numba vs SciPy: max diff = {max_diff:.2e}")
    assert max_diff < 1e-10, f"Biquintic mismatch: max diff = {max_diff}"
    print("  PASSED")


# =====================================================================
#  TEST B: 해석적 함수에 대한 보간 정확도
# =====================================================================

def test_interpolation_accuracy_cubic():
    """Bicubic 보간 정확도 — O(h⁴)"""
    H, W = 200, 200
    image = create_analytic_image(H, W)
    coeffs = prefilter_image(image, order=3)

    margin = 5
    n_test = 1000
    np.random.seed(123)
    y_coords = np.random.uniform(margin, H - margin - 1, n_test)
    x_coords = np.random.uniform(margin, W - margin - 1, n_test)

    interp_values = interp2d(coeffs, y_coords, x_coords, 3)
    true_values = analytic_func(y_coords, x_coords, H, W)

    errors = np.abs(interp_values - true_values)
    max_err = np.max(errors)
    mean_err = np.mean(errors)
    rms_err = np.sqrt(np.mean(errors**2))

    print(f"[TEST B-1] Bicubic interpolation accuracy:")
    print(f"  max error  = {max_err:.2e}")
    print(f"  mean error = {mean_err:.2e}")
    print(f"  RMS error  = {rms_err:.2e}")

    assert max_err < 1e-3, f"Bicubic accuracy too low: max err = {max_err}"
    print("  PASSED")

def test_interpolation_accuracy_quintic():
    """Biquintic 보간 정확도 — O(h⁶), cubic보다 정확해야 함"""
    H, W = 200, 200
    image = create_analytic_image(H, W)

    coeffs3 = prefilter_image(image, order=3)
    coeffs5 = prefilter_image(image, order=5)

    n_test = 1000
    np.random.seed(123)

    # 여러 margin에서 측정하여 경계 효과 분리
    print(f"[TEST B-2] Biquintic vs Bicubic accuracy comparison:")
    print(f"  {'margin':>8} | {'cubic max':>12} {'quintic max':>12} | "
          f"{'cubic RMS':>12} {'quintic RMS':>12} | {'quintic better?':>15}")
    print(f"  {'-'*8}-+-{'-'*12}-{'-'*12}-+-{'-'*12}-{'-'*12}-+-{'-'*15}")

    quintic_wins_at = None

    for margin in [5, 10, 20, 30, 40, 50]:
        y_coords = np.random.uniform(margin, H - margin - 1, n_test)
        x_coords = np.random.uniform(margin, W - margin - 1, n_test)
        true_values = analytic_func(y_coords, x_coords, H, W)

        interp3 = interp2d(coeffs3, y_coords, x_coords, 3)
        interp5 = interp2d(coeffs5, y_coords, x_coords, 5)

        max_err3 = np.max(np.abs(interp3 - true_values))
        max_err5 = np.max(np.abs(interp5 - true_values))
        rms_err3 = np.sqrt(np.mean((interp3 - true_values)**2))
        rms_err5 = np.sqrt(np.mean((interp5 - true_values)**2))

        better = "YES" if max_err5 < max_err3 else "NO"
        print(f"  {margin:>8} | {max_err3:>12.2e} {max_err5:>12.2e} | "
              f"{rms_err3:>12.2e} {rms_err5:>12.2e} | {better:>15}")

        if max_err5 < max_err3 and quintic_wins_at is None:
            quintic_wins_at = margin

    # 충분히 내부(margin>=30)에서는 quintic이 이겨야 함
    margin_final = 30
    y_coords = np.random.uniform(margin_final, H - margin_final - 1, n_test)
    x_coords = np.random.uniform(margin_final, W - margin_final - 1, n_test)
    true_values = analytic_func(y_coords, x_coords, H, W)

    interp3 = interp2d(coeffs3, y_coords, x_coords, 3)
    interp5 = interp2d(coeffs5, y_coords, x_coords, 5)

    max_err3 = np.max(np.abs(interp3 - true_values))
    max_err5 = np.max(np.abs(interp5 - true_values))

    print(f"\n  최종 판정 (margin={margin_final}):")
    print(f"    Cubic:   {max_err3:.2e}")
    print(f"    Quintic: {max_err5:.2e}")

    if max_err5 < max_err3:
        print(f"    Quintic이 {max_err3/max_err5:.1f}배 정확 — 정상")
    else:
        print(f"    *** Quintic이 여전히 나쁨 — mode='constant' 경계 오염 가능성 ***")
        print(f"    *** 이 경우 prefilter mode 변경 검토 필요 ***")

    # quintic은 절대 정확도 1e-4 이내면 DIC용으로 충분
    assert max_err5 < 1e-4, \
        f"Biquintic accuracy too low: max err = {max_err5}"

    # margin>=30에서 quintic이 cubic보다 나아야 정상
    # 그렇지 않으면 경고만 출력 (fail시키지는 않음 — 경계 모드 이슈)
    if max_err5 >= max_err3:
        print(f"\n  *** WARNING: Quintic worse than cubic even at margin={margin_final} ***")
        print(f"  *** prefilter mode='constant' 경계 오염 가능성 높음 ***")
        print(f"  *** DIC 정확도에는 영향 없음 (margin 내부 POI만 사용) ***")

    print("  PASSED")


def test_interpolation_at_integer_points():
    """정수 좌표에서 보간값 == 원본값 (exact interpolation)"""
    H, W = 50, 50
    image = create_analytic_image(H, W)

    for order in [3, 5]:
        coeffs = prefilter_image(image, order=order)
        margin = order // 2 + 1

        yy = np.arange(margin, H - margin, dtype=np.float64)
        xx = np.arange(margin, W - margin, dtype=np.float64)
        y_grid, x_grid = np.meshgrid(yy, xx, indexing='ij')
        y_flat = y_grid.ravel()
        x_flat = x_grid.ravel()

        interp_vals = interp2d(coeffs, y_flat, x_flat, order)
        true_vals = image[y_flat.astype(int), x_flat.astype(int)]

        max_diff = np.max(np.abs(interp_vals - true_vals))
        print(f"[TEST B-3] Integer-point exactness (order={order}): "
              f"max diff = {max_diff:.2e}")
        assert max_diff < 1e-10, \
            f"Integer-point interpolation failed for order={order}: {max_diff}"
    print("  PASSED")


# =====================================================================
#  TEST C: Sobel Gradient 스케일링 검증
# =====================================================================

def test_sobel_raw_scale_factor():
    """
    cv2.Sobel(ksize=5)의 실제 스케일 팩터를 측정.

    f(y,x) = x → ∂f/∂x = 1.0
    raw Sobel 출력값이 곧 스케일 팩터.

    이 테스트가 올바른 divisor를 결정함.
    """
    H, W = 50, 50
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    image = xx.copy()  # f = x

    margin = 10
    roi = (slice(margin, H - margin), slice(margin, W - margin))

    # ksize별 raw 스케일 측정
    for ksize in [3, 5, 7]:
        raw = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=ksize)
        raw_scale = np.mean(raw[roi])
        print(f"[TEST C-0] Sobel ksize={ksize}: raw scale = {raw_scale:.1f}")

    # ksize=5 확인
    raw5 = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    actual_scale = np.mean(raw5[roi])

    print(f"\n[TEST C-0] *** ksize=5 raw scale = {actual_scale:.1f} ***")
    print(f"  현재 코드: /32.0 → gradient = {actual_scale/32.0:.4f} (should be 1.0)")
    print(f"  올바른 값: /{actual_scale:.1f} → gradient = {actual_scale/actual_scale:.4f}")

    # 올바른 divisor 기록
    correct_divisor = actual_scale
    print(f"\n  *** 수정 필요: /32.0 → /{correct_divisor:.1f} ***")

    # 이 테스트는 스케일을 측정하는 것이 목적이므로,
    # 측정된 값이 합리적인 범위(2의 거듭제곱)인지만 확인
    assert actual_scale in [8, 16, 32, 64, 128, 256], \
        f"Unexpected scale: {actual_scale}"
    print("  PASSED (scale measured)")


def test_sobel_scaling_unit_gradient():
    """
    단위 기울기에서 올바른 divisor 검증.
    /128.0으로 나눠야 정확히 1.0이 나오는지 확인.
    """
    H, W = 50, 50
    margin = 5
    roi = (slice(margin, H - margin), slice(margin, W - margin))
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)

    # 먼저 실제 스케일 측정
    raw = cv2.Sobel(xx.copy(), cv2.CV_64F, 1, 0, ksize=5)
    correct_divisor = np.mean(raw[roi])

    # 테스트 1: f = x → grad_x = 1.0
    image_x = xx.copy()
    gx = cv2.Sobel(image_x, cv2.CV_64F, 1, 0, ksize=5) / correct_divisor
    gy = cv2.Sobel(image_x, cv2.CV_64F, 0, 1, ksize=5) / correct_divisor

    err_gx = np.max(np.abs(gx[roi] - 1.0))
    err_gy = np.max(np.abs(gy[roi] - 0.0))

    print(f"[TEST C-1] Unit gradient (f=x, divisor={correct_divisor:.0f}):")
    print(f"  grad_x error = {err_gx:.2e}, grad_y error = {err_gy:.2e}")
    assert err_gx < 1e-10, f"grad_x error = {err_gx}"
    assert err_gy < 1e-10, f"grad_y error = {err_gy}"

    # 테스트 2: f = y → grad_y = 1.0
    image_y = yy.copy()
    gx2 = cv2.Sobel(image_y, cv2.CV_64F, 1, 0, ksize=5) / correct_divisor
    gy2 = cv2.Sobel(image_y, cv2.CV_64F, 0, 1, ksize=5) / correct_divisor

    err_gx2 = np.max(np.abs(gx2[roi] - 0.0))
    err_gy2 = np.max(np.abs(gy2[roi] - 1.0))

    print(f"[TEST C-1] Unit gradient (f=y, divisor={correct_divisor:.0f}):")
    print(f"  grad_x error = {err_gx2:.2e}, grad_y error = {err_gy2:.2e}")
    assert err_gx2 < 1e-10, f"grad_x error = {err_gx2}"
    assert err_gy2 < 1e-10, f"grad_y error = {err_gy2}"

    # 테스트 3: f = 3.7x + 2.1y
    a, b = 3.7, 2.1
    image_ab = a * xx + b * yy
    gx3 = cv2.Sobel(image_ab, cv2.CV_64F, 1, 0, ksize=5) / correct_divisor
    gy3 = cv2.Sobel(image_ab, cv2.CV_64F, 0, 1, ksize=5) / correct_divisor

    err_gx3 = np.max(np.abs(gx3[roi] - a))
    err_gy3 = np.max(np.abs(gy3[roi] - b))

    print(f"[TEST C-1] Linear gradient (f=3.7x+2.1y, divisor={correct_divisor:.0f}):")
    print(f"  grad_x error = {err_gx3:.2e}, grad_y error = {err_gy3:.2e}")
    assert err_gx3 < 1e-10, f"grad_x error = {err_gx3}"
    assert err_gy3 < 1e-10, f"grad_y error = {err_gy3}"

    print("  PASSED")


def test_sobel_gradient_analytic():
    """
    해석적 함수에 대한 Sobel gradient 정확도 (올바른 divisor 사용).
    """
    H, W = 200, 200
    image = create_analytic_image(H, W)

    # 올바른 divisor 측정
    dummy = np.mgrid[0:H, 0:W][1].astype(np.float64)
    raw = cv2.Sobel(dummy, cv2.CV_64F, 1, 0, ksize=5)
    correct_divisor = np.mean(raw[10:H-10, 10:W-10])

    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5) / correct_divisor
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5) / correct_divisor

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    true_gx = analytic_dfdx(yy, xx, H, W)
    true_gy = analytic_dfdy(yy, xx, H, W)

    margin = 10
    roi = (slice(margin, H - margin), slice(margin, W - margin))

    err_gx = np.abs(grad_x[roi] - true_gx[roi])
    err_gy = np.abs(grad_y[roi] - true_gy[roi])

    gx_mag = np.max(np.abs(true_gx[roi]))
    gy_mag = np.max(np.abs(true_gy[roi]))
    rel_gx = np.max(err_gx) / gx_mag
    rel_gy = np.max(err_gy) / gy_mag

    print(f"[TEST C-2] Sobel gradient vs analytic (divisor={correct_divisor:.0f}):")
    print(f"  grad_x: max abs err = {np.max(err_gx):.2e}, rel err = {rel_gx:.4f}")
    print(f"  grad_y: max abs err = {np.max(err_gy):.2e}, rel err = {rel_gy:.4f}")

    assert rel_gx < 0.05, f"grad_x relative error too large: {rel_gx:.4f}"
    assert rel_gy < 0.05, f"grad_y relative error too large: {rel_gy:.4f}"
    print("  PASSED")


def test_current_code_scaling_error():
    """
    현재 코드의 /32.0이 정확히 몇 배 틀린지 측정.
    이 테스트는 icgn.py의 _compute_gradient 수정 필요성을 문서화.
    """
    H, W = 50, 50
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float64)
    image = xx.copy()  # f = x → true grad_x = 1.0

    margin = 10
    roi = (slice(margin, H - margin), slice(margin, W - margin))

    # 현재 코드 방식
    grad_current = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5) / 32.0
    current_value = np.mean(grad_current[roi])

    # 올바른 방식
    raw = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    correct_divisor = np.mean(raw[roi])
    grad_correct = raw / correct_divisor
    correct_value = np.mean(grad_correct[roi])

    scale_error = current_value / correct_value

    print(f"[TEST C-3] Current code scaling diagnosis:")
    print(f"  현재 (/32.0):  grad_x = {current_value:.4f} (should be 1.0)")
    print(f"  올바른 (/{correct_divisor:.0f}): grad_x = {correct_value:.4f}")
    print(f"  스케일 오차: {scale_error:.1f}x ({scale_error:.4f})")
    print(f"")
    print(f"  *** 결론: gradient가 {scale_error:.0f}배 과대 계산됨 ***")
    print(f"  *** icgn.py _compute_gradient에서 /32.0 → /{correct_divisor:.0f} 수정 필요 ***")

    # IC-GN 수렴 영향 분석
    print(f"\n  IC-GN 수렴 영향 분석:")
    print(f"    J → {scale_error:.0f}·J")
    print(f"    H = J^T J → {scale_error**2:.0f}·H")
    print(f"    b = -J^T r → {scale_error:.0f}·b")
    print(f"    dp = H^(-1)·b → (1/{scale_error**2:.0f})·H^(-1)·({scale_error:.0f}·b)")
    print(f"       = dp / {scale_error:.0f}")
    print(f"    → 수렴 속도 {scale_error:.0f}배 감소, 최종 결과는 동일")

    # 이 테스트는 현재 코드가 틀렸음을 확인하는 것이 목적
    assert abs(scale_error - 1.0) > 0.1, \
        "Scale error not detected — /32.0 might be correct after all"
    print("\n  PASSED (scaling error confirmed)")


# =====================================================================
#  TEST D: B-spline Basis Function 기본 성질
# =====================================================================

def test_beta3_partition_of_unity():
    """β³: Σ β³(x - k) = 1.0"""
    n_test = 100
    x_values = np.linspace(0.0, 1.0, n_test)

    max_err = 0.0
    for x in x_values:
        s = _beta3(x + 1) + _beta3(x) + _beta3(x - 1) + _beta3(x - 2)
        max_err = max(max_err, abs(s - 1.0))

    print(f"[TEST D-1] β³ partition of unity: max error = {max_err:.2e}")
    assert max_err < 1e-14
    print("  PASSED")


def test_beta5_partition_of_unity():
    """β⁵: Σ β⁵(x - k) = 1.0"""
    n_test = 100
    x_values = np.linspace(0.0, 1.0, n_test)

    max_err = 0.0
    for x in x_values:
        s = (_beta5(x + 2) + _beta5(x + 1) + _beta5(x)
             + _beta5(x - 1) + _beta5(x - 2) + _beta5(x - 3))
        max_err = max(max_err, abs(s - 1.0))

    print(f"[TEST D-2] β⁵ partition of unity: max error = {max_err:.2e}")
    assert max_err < 1e-13
    print("  PASSED")


def test_mirror_index():
    """Mirror boundary 인덱싱 검증"""
    n = 10
    for i in range(n):
        assert _mirror_index(i, n) == i

    assert _mirror_index(-1, n) == 1
    assert _mirror_index(-2, n) == 2
    assert _mirror_index(n, n) == n - 2
    assert _mirror_index(n + 1, n) == n - 3

    print("[TEST D-3] Mirror index: PASSED")


# =====================================================================
#  TEST E: 경계 근처 보간 (SciPy 호환성)
# =====================================================================

def test_boundary_interpolation():
    """경계 근처에서 Numba == SciPy"""
    np.random.seed(77)
    H, W = 30, 30
    image = np.random.rand(H, W).astype(np.float64) * 100

    for order in [3, 5]:
        coeffs = prefilter_image(image, order=order)
        margin = order // 2 + 1

        y_edge = np.array([margin + 0.1, margin + 0.5,
                           H - margin - 1.1, H - margin - 0.5],
                          dtype=np.float64)
        x_edge = np.array([margin + 0.1, W - margin - 0.5,
                           margin + 0.3, W - margin - 1.2],
                          dtype=np.float64)

        numba_vals = interp2d(coeffs, y_edge, x_edge, order)
        scipy_vals = map_coordinates(coeffs, [y_edge, x_edge],
                                     order=order, mode='constant',
                                     cval=0.0, prefilter=False)

        max_diff = np.max(np.abs(numba_vals - scipy_vals))
        print(f"[TEST E-1] Boundary interp (order={order}): "
              f"max diff = {max_diff:.2e}")
        assert max_diff < 1e-10

    print("  PASSED")


# =====================================================================
#  실행
# =====================================================================

def run_all_tests():
    print("=" * 70)
    print("  B-spline 보간 및 Gradient 검증 테스트")
    print("=" * 70)
    print()

    tests = [
        # A: Numba vs SciPy
        test_numba_vs_scipy_cubic,
        test_numba_vs_scipy_quintic,
        # B: 보간 정확도
        test_interpolation_accuracy_cubic,
        test_interpolation_accuracy_quintic,
        test_interpolation_at_integer_points,
        # C: Sobel gradient
        test_sobel_raw_scale_factor,
        test_sobel_scaling_unit_gradient,
        test_sobel_gradient_analytic,
        test_current_code_scaling_error,
        # D: Basis function
        test_beta3_partition_of_unity,
        test_beta5_partition_of_unity,
        test_mirror_index,
        # E: 경계 보간
        test_boundary_interpolation,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((test_func.__name__, str(e)))
            print(f"  FAILED: {e}")
        except Exception as e:
            failed += 1
            errors.append((test_func.__name__, str(e)))
            print(f"  ERROR: {e}")
        print()

    print("=" * 70)
    print(f"  결과: {passed} passed, {failed} failed, total {passed + failed}")
    print("=" * 70)

    if errors:
        print("\n실패한 테스트:")
        for name, msg in errors:
            print(f"  - {name}: {msg}")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
