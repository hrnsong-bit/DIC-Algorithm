#!/usr/bin/env python3
"""
IC-GN Core Numba 모듈 테스트

기존 icgn.py 구현과 icgn_core_numba.py의 수치적 동치를 검증.
성능 벤치마크 포함.

테스트 카테고리:
    1. extract_reference_subset 검증
    2. compute_znssd 검증
    3. compute_b_vector 검증
    4. icgn_iterate — 기존 _icgn_iterate와 수치 비교
    5. process_poi_numba — 기존 process_poi와 수치 비교
    6. 다양한 변위 시나리오 (정수, 서브픽셀, 큰 변위)
    7. 실패 케이스 (flat subset, out-of-bounds, divergence)
    8. Quadratic shape function
    9. process_all_pois_parallel 검증
   10. 성능 벤치마크 (기존 vs Numba, 단일 vs 병렬)
"""

import sys
import time
import numpy as np
import cv2

# === Path setup ===
sys.path.insert(0, '.')

# === Imports: Original ===
from speckle.core.optimization.icgn import (
    _icgn_iterate as orig_icgn_iterate,
    _extract_reference_subset as orig_extract_ref,
    _compute_znssd as orig_znssd,
    _compute_gradient,
)
from speckle.core.optimization.interpolation import create_interpolator
from speckle.core.optimization.shape_function import (
    generate_local_coordinates as orig_gen_coords,
    compute_steepest_descent as orig_steepest_descent,
    compute_hessian as orig_hessian,
    get_initial_params as orig_get_init_params,
    get_num_params as orig_get_num_params,
    warp as orig_warp,
    update_warp_inverse_compositional as orig_update_warp,
    check_convergence as orig_check_conv,
)

# === Imports: Numba ===
from speckle.core.optimization.icgn_core_numba import (
    extract_reference_subset,
    compute_znssd,
    compute_b_vector,
    icgn_iterate,
    process_poi_numba,
    process_all_pois_parallel,
    allocate_poi_buffers,
    allocate_batch_buffers,
    warmup_icgn_core,
    ICGN_SUCCESS, ICGN_FAIL_FLAT_SUBSET, ICGN_FAIL_OUT_OF_BOUNDS,
    ICGN_FAIL_DIVERGED, ICGN_FAIL_FLAT_TARGET, ICGN_FAIL_LOW_ZNCC,
    ICGN_FAIL_SINGULAR_HESSIAN, ICGN_FAIL_MAX_DISPLACEMENT,
)
from speckle.core.optimization.interpolation_numba import prefilter_image
from speckle.core.optimization.shape_function_numba import (
    AFFINE, QUADRATIC,
    generate_local_coordinates as numba_gen_coords,
    compute_steepest_descent as numba_steepest_descent,
    compute_hessian as numba_hessian,
    get_num_params as numba_get_num_params,
)


# =============================================================================
#  테스트 인프라
# =============================================================================

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            print(f"  PASS: {name}")
        else:
            self.failed += 1
            self.errors.append(name)
            print(f"  FAIL: {name}  {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("Failed tests:")
            for e in self.errors:
                print(f"  - {e}")
        print(f"{'='*60}")
        return self.failed == 0


runner = TestRunner()


# =============================================================================
#  테스트 데이터 생성
# =============================================================================

def make_test_image(size=200, seed=42):
    """부드러운 테스트 이미지 생성 (B-spline 보간에 적합)"""
    np.random.seed(seed)
    img = np.random.rand(size, size).astype(np.float64) * 200 + 20
    img = cv2.GaussianBlur(img, (15, 15), 3.0)
    return img


def make_shifted_image(ref, shift_x, shift_y):
    """서브픽셀 이동된 이미지 생성"""
    h, w = ref.shape
    M = np.float64([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(ref, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT)


def make_affine_deformed_image(ref, u, v, ux, uy, vx, vy, cx, cy):
    """Affine 변형된 이미지 생성"""
    h, w = ref.shape
    # Inverse mapping: src = dst에서의 변형 역
    # 간단하게: warpAffine으로 전체 이미지에 적용
    # 변형 행렬: [1+ux, uy, u; vx, 1+vy, v]
    # OpenCV warpAffine은 dst→src 매핑을 사용하므로 역변환 필요
    M_fwd = np.array([
        [1.0 + ux, uy, u + cx - (1.0 + ux) * cx - uy * cy],
        [vx, 1.0 + vy, v + cy - vx * cx - (1.0 + vy) * cy]
    ])
    return cv2.warpAffine(ref, M_fwd, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REFLECT)


# =============================================================================
#  JIT 워밍업
# =============================================================================

print("=" * 60)
print("IC-GN Core Numba Test Suite")
print("=" * 60)

print("\nWarming up JIT...")
t0 = time.time()
warmup_icgn_core()
print(f"Warmup: {time.time()-t0:.2f}s\n")


# =============================================================================
#  Test 1: extract_reference_subset
# =============================================================================

print("--- Test 1: extract_reference_subset ---")

ref = make_test_image(200)
grad_x, grad_y = _compute_gradient(ref)

subset_size = 21
n_pixels = subset_size * subset_size
cx, cy = 100, 100

# Original
orig_result = orig_extract_ref(ref, grad_x, grad_y, cx, cy, subset_size)
assert orig_result is not None
f_orig, dfdx_orig, dfdy_orig, f_mean_orig, f_tilde_orig = orig_result

# Numba
f_nb = np.empty(n_pixels, dtype=np.float64)
dfdx_nb = np.empty(n_pixels, dtype=np.float64)
dfdy_nb = np.empty(n_pixels, dtype=np.float64)
f_mean_nb, f_tilde_nb, valid = extract_reference_subset(
    ref, grad_x, grad_y, cx, cy, subset_size, f_nb, dfdx_nb, dfdy_nb)

runner.check("1-1 valid flag", valid)
runner.check("1-2 f values match",
             np.allclose(f_orig, f_nb, atol=1e-14),
             f"max_diff={np.max(np.abs(f_orig - f_nb)):.2e}")
runner.check("1-3 dfdx match",
             np.allclose(dfdx_orig, dfdx_nb, atol=1e-14),
             f"max_diff={np.max(np.abs(dfdx_orig - dfdx_nb)):.2e}")
runner.check("1-4 dfdy match",
             np.allclose(dfdy_orig, dfdy_nb, atol=1e-14),
             f"max_diff={np.max(np.abs(dfdy_orig - dfdy_nb)):.2e}")
runner.check("1-5 f_mean match",
             abs(f_mean_orig - f_mean_nb) < 1e-12,
             f"diff={abs(f_mean_orig - f_mean_nb):.2e}")
runner.check("1-6 f_tilde match",
             abs(f_tilde_orig - f_tilde_nb) < 1e-10,
             f"diff={abs(f_tilde_orig - f_tilde_nb):.2e}")

# 경계 밖 케이스
_, _, valid_oob = extract_reference_subset(
    ref, grad_x, grad_y, 5, 5, subset_size, f_nb, dfdx_nb, dfdy_nb)
runner.check("1-7 boundary check (too close to edge)", not valid_oob)

# Flat subset
flat_img = np.ones((200, 200), dtype=np.float64) * 128.0
flat_gx = np.zeros_like(flat_img)
flat_gy = np.zeros_like(flat_img)
_, _, valid_flat = extract_reference_subset(
    flat_img, flat_gx, flat_gy, 100, 100, subset_size, f_nb, dfdx_nb, dfdy_nb)
runner.check("1-8 flat subset detection", not valid_flat)


# =============================================================================
#  Test 2: compute_znssd
# =============================================================================

print("\n--- Test 2: compute_znssd ---")

np.random.seed(123)
f_test = np.random.rand(n_pixels).astype(np.float64) * 200
g_test = f_test + np.random.rand(n_pixels) * 5  # 약간 다른 패턴

fm = np.mean(f_test)
ft = np.linalg.norm(f_test - fm)
gm = np.mean(g_test)
gt = np.linalg.norm(g_test - gm)

znssd_orig = orig_znssd(f_test, fm, ft, g_test, gm, gt)
znssd_nb = compute_znssd(f_test, fm, ft, g_test, gm, gt, n_pixels)

runner.check("2-1 ZNSSD match",
             abs(znssd_orig - znssd_nb) < 1e-10,
             f"orig={znssd_orig:.10f} numba={znssd_nb:.10f}")

# 완벽 매칭
znssd_perfect = compute_znssd(f_test, fm, ft, f_test, fm, ft, n_pixels)
runner.check("2-2 ZNSSD perfect match ≈ 0",
             abs(znssd_perfect) < 1e-14,
             f"val={znssd_perfect:.2e}")


# =============================================================================
#  Test 3: compute_b_vector
# =============================================================================

print("\n--- Test 3: compute_b_vector ---")

xsi, eta = orig_gen_coords(subset_size)
J_orig = orig_steepest_descent(dfdx_orig, dfdy_orig, xsi, eta, 'affine')
n_params = 6

# Original: b = -J.T @ residual
residual = (f_orig - f_mean_orig) - (f_tilde_orig / gt) * (g_test[:n_pixels] - gm)
b_orig = -J_orig.T @ residual

# Numba
xsi_nb, eta_nb = numba_gen_coords(subset_size)
J_nb = np.empty((n_pixels, n_params), dtype=np.float64)
numba_steepest_descent(dfdx_orig, dfdy_orig, xsi_nb, eta_nb, J_nb, AFFINE)

b_nb = np.empty(n_params, dtype=np.float64)
compute_b_vector(f_orig, f_mean_orig, f_tilde_orig,
                 g_test[:n_pixels], gm, gt,
                 J_nb, b_nb, n_pixels, n_params)

runner.check("3-1 b_vector (affine) match",
             np.allclose(b_orig, b_nb, atol=1e-8),
             f"max_diff={np.max(np.abs(b_orig - b_nb)):.2e}")

# Quadratic
n_params_q = 12
J_q_orig = orig_steepest_descent(dfdx_orig, dfdy_orig, xsi, eta, 'quadratic')
residual_q = (f_orig - f_mean_orig) - (f_tilde_orig / gt) * (g_test[:n_pixels] - gm)
b_q_orig = -J_q_orig.T @ residual_q

J_q_nb = np.empty((n_pixels, n_params_q), dtype=np.float64)
numba_steepest_descent(dfdx_orig, dfdy_orig, xsi_nb, eta_nb, J_q_nb, QUADRATIC)
b_q_nb = np.empty(n_params_q, dtype=np.float64)
compute_b_vector(f_orig, f_mean_orig, f_tilde_orig,
                 g_test[:n_pixels], gm, gt,
                 J_q_nb, b_q_nb, n_pixels, n_params_q)

runner.check("3-2 b_vector (quadratic) match",
             np.allclose(b_q_orig, b_q_nb, atol=1e-8),
             f"max_diff={np.max(np.abs(b_q_orig - b_q_nb)):.2e}")


# =============================================================================
#  Test 4: icgn_iterate — 기존 _icgn_iterate와 비교
# =============================================================================

print("\n--- Test 4: icgn_iterate vs original ---")

# 테스트 이미지 (서브픽셀 이동)
ref_img = make_test_image(300, seed=10)
shift_x, shift_y = 0.35, -0.22
def_img = make_shifted_image(ref_img, shift_x, shift_y)

grad_x_t, grad_y_t = _compute_gradient(ref_img)

subset_size = 21
n_pixels = subset_size * subset_size
cx, cy = 150, 150
max_iter = 50
conv_thresh = 0.001
interp_order = 5

# === Original ===
target_interp = create_interpolator(def_img, order=interp_order)
xsi, eta = orig_gen_coords(subset_size)

orig_ref = orig_extract_ref(ref_img, grad_x_t, grad_y_t, cx, cy, subset_size)
f, dfdx, dfdy, fm, ft = orig_ref

J_o = orig_steepest_descent(dfdx, dfdy, xsi, eta, 'affine')
H_o = orig_hessian(J_o)
H_inv_o = np.linalg.inv(H_o)

p_orig = orig_get_init_params('affine')
p_orig[0] = shift_x  # initial guess for u
p_orig[3] = shift_y  # initial guess for v

p_result_o, zncc_o, niter_o, conv_o, fail_o = orig_icgn_iterate(
    f, fm, ft, J_o, H_inv_o, target_interp,
    cx, cy, xsi, eta, p_orig, subset_size,
    max_iter, conv_thresh, 'affine')

# === Numba ===
coeffs = prefilter_image(def_img, order=interp_order)
xsi_nb, eta_nb = numba_gen_coords(subset_size)

bufs = allocate_poi_buffers(n_pixels, 6)

f_mean_nb, f_tilde_nb, valid = extract_reference_subset(
    ref_img, grad_x_t, grad_y_t, cx, cy, subset_size,
    bufs['f'], bufs['dfdx'], bufs['dfdy'])

J_nb = bufs['J']
H_nb = bufs['H']
numba_steepest_descent(bufs['dfdx'], bufs['dfdy'], xsi_nb, eta_nb, J_nb, AFFINE)
numba_hessian(J_nb, H_nb)
H_inv_nb = np.linalg.inv(H_nb)
bufs['H_inv'][:] = H_inv_nb

p_nb = bufs['p']
p_nb[:] = 0.0
p_nb[0] = shift_x
p_nb[3] = shift_y

zncc_nb, niter_nb, conv_nb, fail_nb = icgn_iterate(
    bufs['f'], f_mean_nb, f_tilde_nb,
    J_nb, bufs['H_inv'],
    coeffs, interp_order,
    ref_img.shape[0], ref_img.shape[1],
    cx, cy,
    xsi_nb, eta_nb,
    p_nb,
    subset_size, max_iter, conv_thresh,
    AFFINE, n_pixels, 6,
    bufs['xsi_w'], bufs['eta_w'],
    bufs['x_def'], bufs['y_def'],
    bufs['g'], bufs['b'], bufs['dp'], bufs['p_new'])

runner.check("4-1 convergence flag match", conv_o == conv_nb,
             f"orig={conv_o}, numba={conv_nb}")
runner.check("4-2 iteration count match", niter_o == niter_nb,
             f"orig={niter_o}, numba={niter_nb}")
runner.check("4-3 ZNCC match",
             abs(zncc_o - zncc_nb) < 1e-10,
             f"orig={zncc_o:.10f}, numba={zncc_nb:.10f}")
runner.check("4-4 fail code match", fail_o == fail_nb,
             f"orig={fail_o}, numba={fail_nb}")
runner.check("4-5 parameters match",
             np.allclose(p_result_o, p_nb, atol=1e-10),
             f"max_diff={np.max(np.abs(p_result_o - p_nb)):.2e}")
runner.check("4-6 displacement u accuracy",
             abs(p_nb[0] - shift_x) < 0.05,
             f"expected={shift_x}, got={p_nb[0]:.6f} (warpAffine interp error)")
runner.check("4-7 displacement v accuracy",
             abs(p_nb[3] - shift_y) < 0.05,
             f"expected={shift_y}, got={p_nb[3]:.6f} (warpAffine interp error)")


# =============================================================================
#  Test 5: process_poi_numba — 전체 POI 처리
# =============================================================================

print("\n--- Test 5: process_poi_numba (full POI) ---")

bufs2 = allocate_poi_buffers(n_pixels, 6)

zncc_poi, niter_poi, conv_poi, fail_poi = process_poi_numba(
    ref_img, grad_x_t, grad_y_t,
    coeffs, interp_order,
    cx, cy,
    shift_x, shift_y,
    xsi_nb, eta_nb,
    subset_size, max_iter, conv_thresh,
    AFFINE,
    bufs2['f'], bufs2['dfdx'], bufs2['dfdy'],
    bufs2['J'], bufs2['H'], bufs2['H_inv'],
    bufs2['p'], bufs2['xsi_w'], bufs2['eta_w'],
    bufs2['x_def'], bufs2['y_def'],
    bufs2['g'], bufs2['b'], bufs2['dp'], bufs2['p_new'])

runner.check("5-1 process_poi converged", conv_poi)
runner.check("5-2 process_poi ZNCC high", zncc_poi > 0.99,
             f"zncc={zncc_poi:.6f}")
runner.check("5-3 process_poi u match",
             abs(bufs2['p'][0] - shift_x) < 0.05,
             f"expected={shift_x}, p[0]={bufs2['p'][0]:.6f} (warpAffine interp error)")
runner.check("5-4 process_poi v match",
             abs(bufs2['p'][3] - shift_y) < 0.05,
             f"expected={shift_y}, p[3]={bufs2['p'][3]:.6f} (warpAffine interp error)")
runner.check("5-5 process_poi matches iterate",
             abs(zncc_poi - zncc_nb) < 1e-10,
             f"poi={zncc_poi:.10f}, iterate={zncc_nb:.10f}")


# =============================================================================
#  Test 6: 다양한 변위 시나리오
# =============================================================================

print("\n--- Test 6: Various displacement scenarios ---")

displacements = [
    (0.0, 0.0, "zero displacement"),
    (1.0, 0.0, "integer x"),
    (0.0, 1.0, "integer y"),
    (0.5, 0.5, "half pixel"),
    (0.123, -0.456, "arbitrary sub-pixel"),
    (2.7, -1.3, "larger displacement"),
]

for dx, dy, desc in displacements:
    def_img_t = make_shifted_image(ref_img, dx, dy)
    coeffs_t = prefilter_image(def_img_t, order=interp_order)

    bufs_t = allocate_poi_buffers(n_pixels, 6)
    zncc_t, niter_t, conv_t, fail_t = process_poi_numba(
        ref_img, grad_x_t, grad_y_t,
        coeffs_t, interp_order,
        cx, cy, dx, dy,
        xsi_nb, eta_nb,
        subset_size, max_iter, conv_thresh,
        AFFINE,
        bufs_t['f'], bufs_t['dfdx'], bufs_t['dfdy'],
        bufs_t['J'], bufs_t['H'], bufs_t['H_inv'],
        bufs_t['p'], bufs_t['xsi_w'], bufs_t['eta_w'],
        bufs_t['x_def'], bufs_t['y_def'],
        bufs_t['g'], bufs_t['b'], bufs_t['dp'], bufs_t['p_new'])

    runner.check(f"6-{desc} converged", conv_t,
                 f"zncc={zncc_t:.6f}, niter={niter_t}, fail={fail_t}")


# =============================================================================
#  Test 7: 실패 케이스
# =============================================================================

print("\n--- Test 7: Failure cases ---")

# 7-1: Flat subset
flat_ref = np.ones((200, 200), dtype=np.float64) * 128.0
flat_gx = np.zeros_like(flat_ref)
flat_gy = np.zeros_like(flat_ref)
flat_coeffs = prefilter_image(flat_ref, order=5)

bufs_fail = allocate_poi_buffers(n_pixels, 6)
zncc_f, _, _, fail_f = process_poi_numba(
    flat_ref, flat_gx, flat_gy,
    flat_coeffs, 5, 100, 100, 0.0, 0.0,
    xsi_nb, eta_nb, subset_size, max_iter, conv_thresh, AFFINE,
    bufs_fail['f'], bufs_fail['dfdx'], bufs_fail['dfdy'],
    bufs_fail['J'], bufs_fail['H'], bufs_fail['H_inv'],
    bufs_fail['p'], bufs_fail['xsi_w'], bufs_fail['eta_w'],
    bufs_fail['x_def'], bufs_fail['y_def'],
    bufs_fail['g'], bufs_fail['b'], bufs_fail['dp'], bufs_fail['p_new'])

runner.check("7-1 flat subset → FAIL_FLAT_SUBSET",
             fail_f == ICGN_FAIL_FLAT_SUBSET,
             f"got fail_code={fail_f}")

# 7-2: Out of bounds (edge POI)
bufs_oob = allocate_poi_buffers(n_pixels, 6)
_, _, _, fail_oob = process_poi_numba(
    ref_img, grad_x_t, grad_y_t,
    coeffs, interp_order,
    5, 5, 0.0, 0.0,  # too close to edge
    xsi_nb, eta_nb, subset_size, max_iter, conv_thresh, AFFINE,
    bufs_oob['f'], bufs_oob['dfdx'], bufs_oob['dfdy'],
    bufs_oob['J'], bufs_oob['H'], bufs_oob['H_inv'],
    bufs_oob['p'], bufs_oob['xsi_w'], bufs_oob['eta_w'],
    bufs_oob['x_def'], bufs_oob['y_def'],
    bufs_oob['g'], bufs_oob['b'], bufs_oob['dp'], bufs_oob['p_new'])

runner.check("7-2 edge POI → FAIL_FLAT_SUBSET (boundary)",
             fail_oob == ICGN_FAIL_FLAT_SUBSET,
             f"got fail_code={fail_oob}")

# 7-3: Large initial guess → may diverge or go OOB
bufs_large = allocate_poi_buffers(n_pixels, 6)
_, _, _, fail_large = process_poi_numba(
    ref_img, grad_x_t, grad_y_t,
    coeffs, interp_order,
    cx, cy, 50.0, 50.0,  # way off
    xsi_nb, eta_nb, subset_size, max_iter, conv_thresh, AFFINE,
    bufs_large['f'], bufs_large['dfdx'], bufs_large['dfdy'],
    bufs_large['J'], bufs_large['H'], bufs_large['H_inv'],
    bufs_large['p'], bufs_large['xsi_w'], bufs_large['eta_w'],
    bufs_large['x_def'], bufs_large['y_def'],
    bufs_large['g'], bufs_large['b'], bufs_large['dp'], bufs_large['p_new'])

runner.check("7-3 large init → failure (not SUCCESS)",
             fail_large != ICGN_SUCCESS,
             f"got fail_code={fail_large}")


# =============================================================================
#  Test 8: Quadratic shape function
# =============================================================================

print("\n--- Test 8: Quadratic shape function ---")

n_params_q = 12
bufs_q = allocate_poi_buffers(n_pixels, n_params_q)

zncc_q, niter_q, conv_q, fail_q = process_poi_numba(
    ref_img, grad_x_t, grad_y_t,
    coeffs, interp_order,
    cx, cy, shift_x, shift_y,
    xsi_nb, eta_nb,
    subset_size, max_iter, conv_thresh,
    QUADRATIC,
    bufs_q['f'], bufs_q['dfdx'], bufs_q['dfdy'],
    bufs_q['J'], bufs_q['H'], bufs_q['H_inv'],
    bufs_q['p'], bufs_q['xsi_w'], bufs_q['eta_w'],
    bufs_q['x_def'], bufs_q['y_def'],
    bufs_q['g'], bufs_q['b'], bufs_q['dp'], bufs_q['p_new'])

runner.check("8-1 quadratic converged", conv_q,
             f"zncc={zncc_q:.6f}, niter={niter_q}, fail={fail_q}")
runner.check("8-2 quadratic u accuracy",
             abs(bufs_q['p'][0] - shift_x) < 0.05,
             f"expected={shift_x}, got={bufs_q['p'][0]:.6f} (warpAffine interp error)")
runner.check("8-3 quadratic v accuracy",
             abs(bufs_q['p'][6] - shift_y) < 0.05,
             f"expected={shift_y}, got={bufs_q['p'][6]:.6f} (warpAffine interp error)")
runner.check("8-4 quadratic ZNCC high", zncc_q > 0.99,
             f"zncc={zncc_q:.6f}")

# Quadratic with original for comparison
target_interp_cmp = create_interpolator(def_img, order=interp_order)
orig_ref_q = orig_extract_ref(ref_img, grad_x_t, grad_y_t, cx, cy, subset_size)
f_oq, dfdx_oq, dfdy_oq, fm_oq, ft_oq = orig_ref_q
J_oq = orig_steepest_descent(dfdx_oq, dfdy_oq, xsi, eta, 'quadratic')
H_oq = orig_hessian(J_oq)
H_inv_oq = np.linalg.inv(H_oq)
p_oq = orig_get_init_params('quadratic')
p_oq[0] = shift_x
p_oq[6] = shift_y

p_result_oq, zncc_oq, niter_oq, conv_oq, fail_oq = orig_icgn_iterate(
    f_oq, fm_oq, ft_oq, J_oq, H_inv_oq, target_interp_cmp,
    cx, cy, xsi, eta, p_oq, subset_size,
    max_iter, conv_thresh, 'quadratic')

runner.check("8-5 quadratic ZNCC match (orig vs numba)",
             abs(zncc_oq - zncc_q) < 1e-8,
             f"orig={zncc_oq:.10f}, numba={zncc_q:.10f}")
runner.check("8-6 quadratic params match",
             np.allclose(p_result_oq, bufs_q['p'], atol=1e-8),
             f"max_diff={np.max(np.abs(p_result_oq - bufs_q['p'])):.2e}")


# =============================================================================
#  Test 9: process_all_pois_parallel
# =============================================================================

print("\n--- Test 9: process_all_pois_parallel ---")

# 여러 POI 생성
n_poi = 20
np.random.seed(55)
half_ss = subset_size // 2
margin = half_ss + 5  # 이미지 경계에서 충분히 떨어진 위치

pts_x = np.random.randint(margin, ref_img.shape[1] - margin, n_poi).astype(np.int64)
pts_y = np.random.randint(margin, ref_img.shape[0] - margin, n_poi).astype(np.int64)
init_u = np.full(n_poi, shift_x, dtype=np.float64)
init_v = np.full(n_poi, shift_y, dtype=np.float64)

n_params_aff = 6
batch_bufs = allocate_batch_buffers(n_poi, n_pixels, n_params_aff)

result_p = np.empty((n_poi, n_params_aff), dtype=np.float64)
result_zncc = np.empty(n_poi, dtype=np.float64)
result_iter = np.empty(n_poi, dtype=np.int32)
result_conv = np.empty(n_poi, dtype=np.bool_)
result_fail = np.empty(n_poi, dtype=np.int32)

process_all_pois_parallel(
    ref_img, grad_x_t, grad_y_t,
    coeffs, interp_order,
    pts_x, pts_y,
    init_u, init_v,
    xsi_nb, eta_nb,
    subset_size, max_iter, conv_thresh,
    AFFINE,
    result_p, result_zncc, result_iter, result_conv, result_fail,
    batch_bufs['f'], batch_bufs['dfdx'], batch_bufs['dfdy'],
    batch_bufs['J'], batch_bufs['H'], batch_bufs['H_inv'],
    batch_bufs['p'], batch_bufs['xsi_w'], batch_bufs['eta_w'],
    batch_bufs['x_def'], batch_bufs['y_def'],
    batch_bufs['g'], batch_bufs['b'], batch_bufs['dp'], batch_bufs['p_new']
)

n_converged = np.sum(result_conv)
runner.check("9-1 parallel: most POIs converged",
             n_converged >= n_poi * 0.9,
             f"{n_converged}/{n_poi}")

# 개별 process_poi와 비교
mismatches = 0
for i in range(n_poi):
    bufs_single = allocate_poi_buffers(n_pixels, n_params_aff)
    zncc_s, niter_s, conv_s, fail_s = process_poi_numba(
        ref_img, grad_x_t, grad_y_t,
        coeffs, interp_order,
        pts_x[i], pts_y[i],
        init_u[i], init_v[i],
        xsi_nb, eta_nb,
        subset_size, max_iter, conv_thresh,
        AFFINE,
        bufs_single['f'], bufs_single['dfdx'], bufs_single['dfdy'],
        bufs_single['J'], bufs_single['H'], bufs_single['H_inv'],
        bufs_single['p'], bufs_single['xsi_w'], bufs_single['eta_w'],
        bufs_single['x_def'], bufs_single['y_def'],
        bufs_single['g'], bufs_single['b'], bufs_single['dp'], bufs_single['p_new'])

    if abs(zncc_s - result_zncc[i]) > 1e-10:
        mismatches += 1
    if not np.allclose(bufs_single['p'], result_p[i], atol=1e-10):
        mismatches += 1

runner.check("9-2 parallel == sequential (all POIs)",
             mismatches == 0,
             f"mismatches={mismatches}")

# 수렴된 POI의 평균 displacement 정확도
conv_mask = result_conv
if np.sum(conv_mask) > 0:
    u_error = np.mean(np.abs(result_p[conv_mask, 0] - shift_x))
    v_error = np.mean(np.abs(result_p[conv_mask, 3] - shift_y))
    runner.check("9-3 parallel: mean u error < 0.05",
                 u_error < 0.05,
                 f"mean_u_error={u_error:.6f} (warpAffine interp error)")
    runner.check("9-4 parallel: mean v error < 0.05",
                 v_error < 0.05,
                 f"mean_v_error={v_error:.6f} (warpAffine interp error)")


# =============================================================================
#  Test 10: 보간 차수 3 (cubic)
# =============================================================================

print("\n--- Test 10: Interpolation order 3 (cubic) ---")

coeffs3 = prefilter_image(def_img, order=3)

bufs3 = allocate_poi_buffers(n_pixels, 6)
zncc_3, niter_3, conv_3, fail_3 = process_poi_numba(
    ref_img, grad_x_t, grad_y_t,
    coeffs3, 3,
    cx, cy, shift_x, shift_y,
    xsi_nb, eta_nb,
    subset_size, max_iter, conv_thresh,
    AFFINE,
    bufs3['f'], bufs3['dfdx'], bufs3['dfdy'],
    bufs3['J'], bufs3['H'], bufs3['H_inv'],
    bufs3['p'], bufs3['xsi_w'], bufs3['eta_w'],
    bufs3['x_def'], bufs3['y_def'],
    bufs3['g'], bufs3['b'], bufs3['dp'], bufs3['p_new'])

runner.check("10-1 cubic converged", conv_3,
             f"zncc={zncc_3:.6f}, fail={fail_3}")
runner.check("10-2 cubic u accuracy",
             abs(bufs3['p'][0] - shift_x) < 0.02,
             f"expected={shift_x}, got={bufs3['p'][0]:.6f}")

# Compare with original (cubic)
target_interp_3 = create_interpolator(def_img, order=3)
orig_ref_3 = orig_extract_ref(ref_img, grad_x_t, grad_y_t, cx, cy, subset_size)
f_3, dfdx_3, dfdy_3, fm_3, ft_3 = orig_ref_3
J_3 = orig_steepest_descent(dfdx_3, dfdy_3, xsi, eta, 'affine')
H_3 = orig_hessian(J_3)
H_inv_3 = np.linalg.inv(H_3)
p_3 = orig_get_init_params('affine')
p_3[0] = shift_x
p_3[3] = shift_y

p_r3, zncc_r3, niter_r3, conv_r3, fail_r3 = orig_icgn_iterate(
    f_3, fm_3, ft_3, J_3, H_inv_3, target_interp_3,
    cx, cy, xsi, eta, p_3, subset_size,
    max_iter, conv_thresh, 'affine')

runner.check("10-3 cubic: numba vs orig ZNCC match",
             abs(zncc_3 - zncc_r3) < 1e-10,
             f"numba={zncc_3:.10f}, orig={zncc_r3:.10f}")
runner.check("10-4 cubic: numba vs orig params match",
             np.allclose(bufs3['p'], p_r3, atol=1e-10),
             f"max_diff={np.max(np.abs(bufs3['p'] - p_r3)):.2e}")


# =============================================================================
#  Test 11: 성능 벤치마크
# =============================================================================

print("\n--- Test 11: Performance Benchmark ---")

# 큰 이미지, 많은 POI
bench_size = 500
ref_bench = make_test_image(bench_size, seed=99)
def_bench = make_shifted_image(ref_bench, 0.42, -0.31)
grad_x_b, grad_y_b = _compute_gradient(ref_bench)
coeffs_b = prefilter_image(def_bench, order=5)
target_interp_b = create_interpolator(def_bench, order=5)

subset_size_b = 21
n_pixels_b = subset_size_b * subset_size_b
xsi_b, eta_b = orig_gen_coords(subset_size_b)
xsi_b_nb, eta_b_nb = numba_gen_coords(subset_size_b)

# POI 그리드 생성
n_poi_bench = 500
np.random.seed(77)
margin_b = subset_size_b // 2 + 5
pts_x_b = np.random.randint(margin_b, bench_size - margin_b, n_poi_bench)
pts_y_b = np.random.randint(margin_b, bench_size - margin_b, n_poi_bench)

# === Original (sequential ThreadPoolExecutor, 1 worker) ===
t0 = time.time()
for i in range(n_poi_bench):
    px, py = int(pts_x_b[i]), int(pts_y_b[i])
    ref_data = orig_extract_ref(ref_bench, grad_x_b, grad_y_b, px, py, subset_size_b)
    if ref_data is None:
        continue
    f_b, dfdx_b, dfdy_b, fm_b, ft_b = ref_data
    J_b = orig_steepest_descent(dfdx_b, dfdy_b, xsi_b, eta_b, 'affine')
    H_b = orig_hessian(J_b)
    H_inv_b = np.linalg.inv(H_b)
    p_b = orig_get_init_params('affine')
    p_b[0] = 0.42
    p_b[3] = -0.31
    orig_icgn_iterate(
        f_b, fm_b, ft_b, J_b, H_inv_b, target_interp_b,
        px, py, xsi_b, eta_b, p_b, subset_size_b,
        max_iter, conv_thresh, 'affine')
t_orig = time.time() - t0

# === Numba sequential ===
t0 = time.time()
for i in range(n_poi_bench):
    bufs_bench = allocate_poi_buffers(n_pixels_b, 6)
    process_poi_numba(
        ref_bench, grad_x_b, grad_y_b,
        coeffs_b, 5,
        int(pts_x_b[i]), int(pts_y_b[i]),
        0.42, -0.31,
        xsi_b_nb, eta_b_nb,
        subset_size_b, max_iter, conv_thresh,
        AFFINE,
        bufs_bench['f'], bufs_bench['dfdx'], bufs_bench['dfdy'],
        bufs_bench['J'], bufs_bench['H'], bufs_bench['H_inv'],
        bufs_bench['p'], bufs_bench['xsi_w'], bufs_bench['eta_w'],
        bufs_bench['x_def'], bufs_bench['y_def'],
        bufs_bench['g'], bufs_bench['b'], bufs_bench['dp'], bufs_bench['p_new'])
t_numba_seq = time.time() - t0

# === Numba parallel ===
pts_x_b_i64 = pts_x_b.astype(np.int64)
pts_y_b_i64 = pts_y_b.astype(np.int64)
init_u_b = np.full(n_poi_bench, 0.42, dtype=np.float64)
init_v_b = np.full(n_poi_bench, -0.31, dtype=np.float64)

batch_bufs_b = allocate_batch_buffers(n_poi_bench, n_pixels_b, 6)
result_p_b = np.empty((n_poi_bench, 6), dtype=np.float64)
result_zncc_b = np.empty(n_poi_bench, dtype=np.float64)
result_iter_b = np.empty(n_poi_bench, dtype=np.int32)
result_conv_b = np.empty(n_poi_bench, dtype=np.bool_)
result_fail_b = np.empty(n_poi_bench, dtype=np.int32)

# Warmup parallel
process_all_pois_parallel(
    ref_bench, grad_x_b, grad_y_b,
    coeffs_b, 5,
    pts_x_b_i64, pts_y_b_i64,
    init_u_b, init_v_b,
    xsi_b_nb, eta_b_nb,
    subset_size_b, max_iter, conv_thresh,
    AFFINE,
    result_p_b, result_zncc_b, result_iter_b, result_conv_b, result_fail_b,
    batch_bufs_b['f'], batch_bufs_b['dfdx'], batch_bufs_b['dfdy'],
    batch_bufs_b['J'], batch_bufs_b['H'], batch_bufs_b['H_inv'],
    batch_bufs_b['p'], batch_bufs_b['xsi_w'], batch_bufs_b['eta_w'],
    batch_bufs_b['x_def'], batch_bufs_b['y_def'],
    batch_bufs_b['g'], batch_bufs_b['b'], batch_bufs_b['dp'], batch_bufs_b['p_new']
)

t0 = time.time()
process_all_pois_parallel(
    ref_bench, grad_x_b, grad_y_b,
    coeffs_b, 5,
    pts_x_b_i64, pts_y_b_i64,
    init_u_b, init_v_b,
    xsi_b_nb, eta_b_nb,
    subset_size_b, max_iter, conv_thresh,
    AFFINE,
    result_p_b, result_zncc_b, result_iter_b, result_conv_b, result_fail_b,
    batch_bufs_b['f'], batch_bufs_b['dfdx'], batch_bufs_b['dfdy'],
    batch_bufs_b['J'], batch_bufs_b['H'], batch_bufs_b['H_inv'],
    batch_bufs_b['p'], batch_bufs_b['xsi_w'], batch_bufs_b['eta_w'],
    batch_bufs_b['x_def'], batch_bufs_b['y_def'],
    batch_bufs_b['g'], batch_bufs_b['b'], batch_bufs_b['dp'], batch_bufs_b['p_new']
)
t_numba_par = time.time() - t0

print(f"\n  Performance ({n_poi_bench} POIs, {subset_size_b}x{subset_size_b} subset, order=5):")
print(f"    Original (sequential):     {t_orig*1000:.1f} ms  ({t_orig/n_poi_bench*1000:.3f} ms/POI)")
print(f"    Numba (sequential):        {t_numba_seq*1000:.1f} ms  ({t_numba_seq/n_poi_bench*1000:.3f} ms/POI)")
print(f"    Numba (parallel prange):   {t_numba_par*1000:.1f} ms  ({t_numba_par/n_poi_bench*1000:.3f} ms/POI)")
print(f"    Speedup (seq): {t_orig/t_numba_seq:.2f}x")
print(f"    Speedup (par): {t_orig/t_numba_par:.2f}x")

runner.check("11-1 Numba sequential faster than original",
             t_numba_seq < t_orig,
             f"orig={t_orig*1000:.1f}ms, numba={t_numba_seq*1000:.1f}ms")

n_conv_bench = np.sum(result_conv_b)
runner.check("11-2 benchmark: high convergence rate",
             n_conv_bench >= n_poi_bench * 0.9,
             f"{n_conv_bench}/{n_poi_bench}")


# =============================================================================
#  요약
# =============================================================================

print()
success = runner.summary()
sys.exit(0 if success else 1)
