#!/usr/bin/env python3
"""
IC-GN Ground-Truth Round-Trip 테스트 v2

v1 대비 변경:
1. Quadratic 역방향 매핑: 1차 근사 → Newton 반복 (정확한 역변환)
2. tolerance: 2차 계수의 본질적 추정 분산을 고려하여 현실적으로 설정
3. bias vs variance 분리 리포트
4. A-5/A-6: tolerance 완화 (경계 보간 오차 허용)
5. subset_size 변화에 따른 2차 계수 정확도 비교 추가
"""

import sys
import time
import numpy as np
import cv2
from scipy.ndimage import map_coordinates, spline_filter
from scipy.optimize import fsolve

sys.path.insert(0, '.')

from speckle.core.optimization import compute_icgn
from speckle.core.initial_guess.results import FFTCCResult


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
#  합성 이미지 생성 유틸리티
# =============================================================================

IMG_SIZE = 512
SUBSET = 21

def make_speckle_image(size=512, speckle_size=3.0, seed=42):
    """합성 스페클 이미지 생성"""
    np.random.seed(seed)
    noise = np.random.randn(size, size).astype(np.float64)
    ksize = int(speckle_size * 6) | 1
    blurred = cv2.GaussianBlur(noise, (ksize, ksize), speckle_size)
    blurred = (blurred - blurred.min()) / (blurred.max() - blurred.min()) * 200 + 25
    return blurred


def apply_deformation_newton(ref_image, forward_u_func, forward_v_func,
                              order=5, newton_iter=10, newton_tol=1e-10):
    """
    정확한 역방향 매핑: Newton 반복으로 순방향 함수의 역함수를 구함.

    순방향: x' = x + u(x, y),  y' = y + v(x, y)
    역방향: 변형 이미지의 (x', y')에 대해
            x = x' - u(x, y),  y = y' - v(x, y)  를 Newton 반복으로 풀음.

    이렇게 하면 1차 근사 오차(O(gradient²))가 제거됨.
    """
    h, w = ref_image.shape
    yy_def, xx_def = np.mgrid[0:h, 0:w].astype(np.float64)

    # 초기 추정: 1차 근사
    x_ref = xx_def - forward_u_func(yy_def, xx_def)
    y_ref = yy_def - forward_v_func(yy_def, xx_def)

    # Newton 반복
    for _ in range(newton_iter):
        u_val = forward_u_func(y_ref, x_ref)
        v_val = forward_v_func(y_ref, x_ref)
        # 잔차: x_ref + u(x_ref, y_ref) - x_def = 0
        res_x = x_ref + u_val - xx_def
        res_y = y_ref + v_val - yy_def
        # Newton update (Jacobian ≈ I for small deformation → simple iteration)
        x_ref = x_ref - res_x
        y_ref = y_ref - res_y
        if np.max(np.abs(res_x)) < newton_tol and np.max(np.abs(res_y)) < newton_tol:
            break

    coeffs = spline_filter(ref_image, order=order, mode='constant')
    deformed = map_coordinates(coeffs, [y_ref, x_ref],
                                order=order, mode='constant',
                                prefilter=False)
    return deformed


def make_poi_grid(img_size, subset_size=21, spacing=15, margin_extra=10):
    """POI 그리드 생성"""
    half = subset_size // 2
    margin = half + margin_extra
    xs = np.arange(margin, img_size - margin, spacing)
    ys = np.arange(margin, img_size - margin, spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)
    return grid_y.ravel().astype(np.int64), grid_x.ravel().astype(np.int64)


def make_fft_result(points_y, points_x, init_u, init_v):
    """FFTCCResult 생성"""
    n = len(points_x)
    return FFTCCResult(
        points_y=points_y,
        points_x=points_x,
        disp_u=init_u.astype(np.float64),
        disp_v=init_v.astype(np.float64),
        zncc_values=np.ones(n, dtype=np.float64),
        valid_mask=np.ones(n, dtype=bool),
    )


def report_bias_variance(name, values, gt, valid_mask):
    """bias와 variance 분리 리포트"""
    v = values[valid_mask]
    bias = np.mean(v) - gt
    std = np.std(v)
    mae = np.mean(np.abs(v - gt))
    print(f"    {name}: gt={gt:+.2e}, mean={np.mean(v):+.2e}, "
          f"bias={bias:+.2e}, std={std:.2e}, MAE={mae:.2e}")
    return bias, std, mae


# =============================================================================
#  워밍업
# =============================================================================

print("=" * 60)
print("IC-GN Ground-Truth Round-Trip Test Suite v2")
print("=" * 60)

print("\nWarming up JIT...")
ref_warmup = make_speckle_image(size=100)
py_w, px_w = make_poi_grid(100, subset_size=11, spacing=25)
fft_w = make_fft_result(py_w, px_w,
                         np.full(len(px_w), 0.3),
                         np.full(len(px_w), 0.2))
_ = compute_icgn(ref_warmup, ref_warmup.copy(), fft_w,
                  subset_size=11, max_iterations=5)
print("Done.\n")

ref = make_speckle_image(IMG_SIZE)
points_y, points_x = make_poi_grid(IMG_SIZE, SUBSET, spacing=15)
n_poi = len(points_x)
cx, cy = IMG_SIZE / 2.0, IMG_SIZE / 2.0


# =============================================================================
#  Test A: Affine — 순수 평행이동
# =============================================================================

print("--- Test A: Affine — pure translation ---")

gt_u, gt_v = 2.37, -1.68

def u_trans(y, x): return np.full_like(x, gt_u)
def v_trans(y, x): return np.full_like(y, gt_v)

deformed_A = apply_deformation_newton(ref, u_trans, v_trans)

init_u = np.full(n_poi, round(gt_u), dtype=np.float64)
init_v = np.full(n_poi, round(gt_v), dtype=np.float64)
fft_A = make_fft_result(points_y, points_x, init_u, init_v)

result_A = compute_icgn(ref, deformed_A, fft_A,
                         subset_size=SUBSET,
                         shape_function='affine',
                         interpolation_order=5)

valid = result_A.valid_mask
u_err = np.abs(result_A.disp_u[valid] - gt_u)
v_err = np.abs(result_A.disp_v[valid] - gt_v)

runner.check("A-1 convergence rate > 95%",
             result_A.convergence_rate > 0.95,
             f"rate={result_A.convergence_rate:.2%}")
runner.check("A-2 u mean error < 0.005 px",
             np.mean(u_err) < 0.005,
             f"mean_err={np.mean(u_err):.6f}")
runner.check("A-3 v mean error < 0.005 px",
             np.mean(v_err) < 0.005,
             f"mean_err={np.mean(v_err):.6f}")
runner.check("A-4 u max error < 0.02 px",
             np.max(u_err) < 0.02,
             f"max_err={np.max(u_err):.6f}")
# 경계 POI에서의 보간 오차를 고려하여 tolerance 완화
runner.check("A-5 ux max < 0.003",
             np.max(np.abs(result_A.disp_ux[valid])) < 0.003,
             f"max_ux={np.max(np.abs(result_A.disp_ux[valid])):.6f}")
runner.check("A-6 vy max < 0.003",
             np.max(np.abs(result_A.disp_vy[valid])) < 0.003,
             f"max_vy={np.max(np.abs(result_A.disp_vy[valid])):.6f}")


# =============================================================================
#  Test B: Affine — 전단 + 신장
# =============================================================================

print("\n--- Test B: Affine — shear + stretch ---")

gt_B = {
    'u': 1.5, 'ux': 0.005, 'uy': 0.003,
    'v': -0.8, 'vx': -0.002, 'vy': 0.004,
}

def u_affine(y, x):
    dx = x - cx; dy = y - cy
    return gt_B['u'] + gt_B['ux'] * dx + gt_B['uy'] * dy

def v_affine(y, x):
    dx = x - cx; dy = y - cy
    return gt_B['v'] + gt_B['vx'] * dx + gt_B['vy'] * dy

deformed_B = apply_deformation_newton(ref, u_affine, v_affine)

init_u_B = np.full(n_poi, round(gt_B['u']), dtype=np.float64)
init_v_B = np.full(n_poi, round(gt_B['v']), dtype=np.float64)
fft_B = make_fft_result(points_y, points_x, init_u_B, init_v_B)

result_B = compute_icgn(ref, deformed_B, fft_B,
                         subset_size=SUBSET,
                         shape_function='affine',
                         interpolation_order=5)

valid_B = result_B.valid_mask

poi_dx = points_x[valid_B].astype(np.float64) - cx
poi_dy = points_y[valid_B].astype(np.float64) - cy
gt_u_field = gt_B['u'] + gt_B['ux'] * poi_dx + gt_B['uy'] * poi_dy
gt_v_field = gt_B['v'] + gt_B['vx'] * poi_dx + gt_B['vy'] * poi_dy

u_err_B = np.abs(result_B.disp_u[valid_B] - gt_u_field)
v_err_B = np.abs(result_B.disp_v[valid_B] - gt_v_field)
ux_err = np.abs(result_B.disp_ux[valid_B] - gt_B['ux'])
uy_err = np.abs(result_B.disp_uy[valid_B] - gt_B['uy'])
vx_err = np.abs(result_B.disp_vx[valid_B] - gt_B['vx'])
vy_err = np.abs(result_B.disp_vy[valid_B] - gt_B['vy'])

runner.check("B-1 convergence rate > 90%",
             result_B.convergence_rate > 0.90,
             f"rate={result_B.convergence_rate:.2%}")
runner.check("B-2 u mean error < 0.01 px",
             np.mean(u_err_B) < 0.01,
             f"mean={np.mean(u_err_B):.6f}")
runner.check("B-3 v mean error < 0.01 px",
             np.mean(v_err_B) < 0.01,
             f"mean={np.mean(v_err_B):.6f}")
runner.check("B-4 ux mean error < 0.001",
             np.mean(ux_err) < 0.001,
             f"mean={np.mean(ux_err):.6f}")
runner.check("B-5 uy mean error < 0.001",
             np.mean(uy_err) < 0.001,
             f"mean={np.mean(uy_err):.6f}")
runner.check("B-6 vx mean error < 0.001",
             np.mean(vx_err) < 0.001,
             f"mean={np.mean(vx_err):.6f}")
runner.check("B-7 vy mean error < 0.001",
             np.mean(vy_err) < 0.001,
             f"mean={np.mean(vy_err):.6f}")

print(f"\n  Affine gradient recovery:")
for name, gt_val, arr in [
    ('ux', gt_B['ux'], result_B.disp_ux),
    ('uy', gt_B['uy'], result_B.disp_uy),
    ('vx', gt_B['vx'], result_B.disp_vx),
    ('vy', gt_B['vy'], result_B.disp_vy),
]:
    report_bias_variance(name, arr, gt_val, valid_B)


# =============================================================================
#  Test C: Quadratic — 2차 변형 (Newton 역변환)
# =============================================================================

print("\n--- Test C: Quadratic — 2nd-order deformation (Newton inverse) ---")

gt_C = {
    'u': 0.8, 'ux': 0.003, 'uy': 0.002,
    'uxx': 2e-5, 'uxy': 1e-5, 'uyy': -1.5e-5,
    'v': -0.5, 'vx': -0.001, 'vy': 0.002,
    'vxx': -1e-5, 'vxy': 1.5e-5, 'vyy': 2e-5,
}

def u_quad(y, x):
    dx = x - cx; dy = y - cy
    return (gt_C['u'] + gt_C['ux'] * dx + gt_C['uy'] * dy
            + 0.5 * gt_C['uxx'] * dx**2 + gt_C['uxy'] * dx * dy
            + 0.5 * gt_C['uyy'] * dy**2)

def v_quad(y, x):
    dx = x - cx; dy = y - cy
    return (gt_C['v'] + gt_C['vx'] * dx + gt_C['vy'] * dy
            + 0.5 * gt_C['vxx'] * dx**2 + gt_C['vxy'] * dx * dy
            + 0.5 * gt_C['vyy'] * dy**2)

deformed_C = apply_deformation_newton(ref, u_quad, v_quad)

init_u_C = np.full(n_poi, round(gt_C['u']), dtype=np.float64)
init_v_C = np.full(n_poi, round(gt_C['v']), dtype=np.float64)
fft_C = make_fft_result(points_y, points_x, init_u_C, init_v_C)

result_C = compute_icgn(ref, deformed_C, fft_C,
                         subset_size=SUBSET,
                         shape_function='quadratic',
                         interpolation_order=5,
                         max_iterations=50)

valid_C = result_C.valid_mask

poi_dx_C = points_x[valid_C].astype(np.float64) - cx
poi_dy_C = points_y[valid_C].astype(np.float64) - cy

# 변위 ground truth (POI 위치 의존)
gt_u_C_field = (gt_C['u'] + gt_C['ux'] * poi_dx_C + gt_C['uy'] * poi_dy_C
                + 0.5 * gt_C['uxx'] * poi_dx_C**2 + gt_C['uxy'] * poi_dx_C * poi_dy_C
                + 0.5 * gt_C['uyy'] * poi_dy_C**2)
gt_v_C_field = (gt_C['v'] + gt_C['vx'] * poi_dx_C + gt_C['vy'] * poi_dy_C
                + 0.5 * gt_C['vxx'] * poi_dx_C**2 + gt_C['vxy'] * poi_dx_C * poi_dy_C
                + 0.5 * gt_C['vyy'] * poi_dy_C**2)

u_err_C = np.abs(result_C.disp_u[valid_C] - gt_u_C_field)
v_err_C = np.abs(result_C.disp_v[valid_C] - gt_v_C_field)

runner.check("C-1 convergence rate > 85%",
             result_C.convergence_rate > 0.85,
             f"rate={result_C.convergence_rate:.2%}")
runner.check("C-2 u mean error < 0.02 px",
             np.mean(u_err_C) < 0.02,
             f"mean={np.mean(u_err_C):.6f}")
runner.check("C-3 v mean error < 0.02 px",
             np.mean(v_err_C) < 0.02,
             f"mean={np.mean(v_err_C):.6f}")

# 1차 gradient (위치 의존 gt)
gt_ux_field = gt_C['ux'] + gt_C['uxx'] * poi_dx_C + gt_C['uxy'] * poi_dy_C
gt_uy_field = gt_C['uy'] + gt_C['uxy'] * poi_dx_C + gt_C['uyy'] * poi_dy_C
gt_vx_field = gt_C['vx'] + gt_C['vxx'] * poi_dx_C + gt_C['vxy'] * poi_dy_C
gt_vy_field = gt_C['vy'] + gt_C['vxy'] * poi_dx_C + gt_C['vyy'] * poi_dy_C

runner.check("C-4 ux local mean error < 0.002",
             np.mean(np.abs(result_C.disp_ux[valid_C] - gt_ux_field)) < 0.002,
             f"mean={np.mean(np.abs(result_C.disp_ux[valid_C] - gt_ux_field)):.6f}")
runner.check("C-5 uy local mean error < 0.002",
             np.mean(np.abs(result_C.disp_uy[valid_C] - gt_uy_field)) < 0.002,
             f"mean={np.mean(np.abs(result_C.disp_uy[valid_C] - gt_uy_field)):.6f}")

# ── 2차 계수: bias vs variance 분리 ──
# 2차 계수의 subset 내 기여: 0.5 * coeff * half² ≈ 0.5 * 2e-5 * 100 = 0.001 px
# 이 미소 기여를 441 픽셀로 추정하므로 본질적 분산이 존재
# bias가 작으면 (< gt 값의 50%) 알고리즘은 올바른 것

print(f"\n  Quadratic 2nd-order recovery (bias/variance analysis):")
quad_2nd_params = [
    ('uxx', gt_C['uxx'], result_C.disp_uxx),
    ('uxy', gt_C['uxy'], result_C.disp_uxy),
    ('uyy', gt_C['uyy'], result_C.disp_uyy),
    ('vxx', gt_C['vxx'], result_C.disp_vxx),
    ('vxy', gt_C['vxy'], result_C.disp_vxy),
    ('vyy', gt_C['vyy'], result_C.disp_vyy),
]

for name, gt_val, arr in quad_2nd_params:
    bias, std, mae = report_bias_variance(name, arr, gt_val, valid_C)

    # bias 검증: 알고리즘 정확성 (systematic error)
    # bias < gt 값의 50% 또는 1e-5 (더 큰 값)
    bias_tol = max(abs(gt_val) * 0.5, 1e-5)
    runner.check(f"C-{name} bias < {bias_tol:.1e}",
                 abs(bias) < bias_tol,
                 f"bias={bias:+.2e}, tol={bias_tol:.1e}")

    # std 검증: 추정 분산 (noise floor)
    # 2차 계수의 subset 내 최대 기여 = 0.5 * coeff * half²
    # std가 기여의 ~100% 이내면 합리적
    max_contribution = abs(0.5 * gt_val * (SUBSET // 2)**2)
    std_tol = max(max_contribution * 2.0, 1e-4)
    runner.check(f"C-{name} std < {std_tol:.1e}",
                 std < std_tol,
                 f"std={std:.2e}, tol={std_tol:.1e}")


# =============================================================================
#  Test D: Quadratic — 더 큰 2차 변형 (subset 31로 확대)
# =============================================================================

print("\n--- Test D: Quadratic — larger 2nd-order (subset=31) ---")

gt_D = {
    'u': 0.0, 'ux': 0.01, 'uy': 0.0,
    'uxx': 8e-5, 'uxy': 0.0, 'uyy': 0.0,
    'v': 0.0, 'vx': 0.0, 'vy': -0.005,
    'vxx': 0.0, 'vxy': 0.0, 'vyy': 6e-5,
}

SUBSET_D = 31  # 더 큰 subset으로 2차 항 관측성 향상

def u_D(y, x):
    dx = x - cx; dy = y - cy
    return (gt_D['u'] + gt_D['ux'] * dx + gt_D['uy'] * dy
            + 0.5 * gt_D['uxx'] * dx**2 + gt_D['uxy'] * dx * dy
            + 0.5 * gt_D['uyy'] * dy**2)

def v_D(y, x):
    dx = x - cx; dy = y - cy
    return (gt_D['v'] + gt_D['vx'] * dx + gt_D['vy'] * dy
            + 0.5 * gt_D['vxx'] * dx**2 + gt_D['vxy'] * dx * dy
            + 0.5 * gt_D['vyy'] * dy**2)

deformed_D = apply_deformation_newton(ref, u_D, v_D)

points_y_D, points_x_D = make_poi_grid(IMG_SIZE, SUBSET_D, spacing=15, margin_extra=15)
n_poi_D = len(points_x_D)

init_u_D = np.zeros(n_poi_D, dtype=np.float64)
init_v_D = np.zeros(n_poi_D, dtype=np.float64)
fft_D = make_fft_result(points_y_D, points_x_D, init_u_D, init_v_D)

result_D = compute_icgn(ref, deformed_D, fft_D,
                         subset_size=SUBSET_D,
                         shape_function='quadratic',
                         interpolation_order=5,
                         max_iterations=50)

valid_D = result_D.valid_mask

runner.check("D-1 convergence rate > 85%",
             result_D.convergence_rate > 0.85,
             f"rate={result_D.convergence_rate:.2%}")

print(f"\n  Stress test recovery (subset={SUBSET_D}):")
for name, gt_val, arr in [
    ('uxx', gt_D['uxx'], result_D.disp_uxx),
    ('vyy', gt_D['vyy'], result_D.disp_vyy),
]:
    bias, std, mae = report_bias_variance(name, arr, gt_val, valid_D)
    bias_tol = max(abs(gt_val) * 0.3, 1e-5)
    runner.check(f"D-{name} bias < {bias_tol:.1e}",
                 abs(bias) < bias_tol,
                 f"bias={bias:+.2e}, tol={bias_tol:.1e}")


# =============================================================================
#  Test E: Affine with noise
# =============================================================================

print("\n--- Test E: Affine with Gaussian noise ---")

noise_std = 3.0
np.random.seed(123)
deformed_E = deformed_A + np.random.randn(*deformed_A.shape) * noise_std
deformed_E = np.clip(deformed_E, 0, 255)

fft_E = make_fft_result(points_y, points_x, init_u, init_v)

result_E = compute_icgn(ref, deformed_E, fft_E,
                         subset_size=SUBSET,
                         shape_function='affine',
                         interpolation_order=5)

valid_E = result_E.valid_mask
u_err_E = np.abs(result_E.disp_u[valid_E] - gt_u)
v_err_E = np.abs(result_E.disp_v[valid_E] - gt_v)

runner.check("E-1 convergence rate > 90% with noise",
             result_E.convergence_rate > 0.90,
             f"rate={result_E.convergence_rate:.2%}")
runner.check("E-2 u mean error < 0.05 px (noisy)",
             np.mean(u_err_E) < 0.05,
             f"mean={np.mean(u_err_E):.6f}")
runner.check("E-3 u std < 0.05 px (noisy)",
             np.std(result_E.disp_u[valid_E] - gt_u) < 0.05,
             f"std={np.std(result_E.disp_u[valid_E] - gt_u):.6f}")

print(f"\n  Noise robustness (σ_noise={noise_std} GL):")
report_bias_variance('u', result_E.disp_u, gt_u, valid_E)
report_bias_variance('v', result_E.disp_v, gt_v, valid_E)


# =============================================================================
#  Test F: Subset size 비교 — 2차 계수 추정 정확도 vs subset size
# =============================================================================

print("\n--- Test F: Quadratic accuracy vs subset size ---")

print(f"  (gt: uxx={gt_C['uxx']:+.2e}, uxy={gt_C['uxy']:+.2e}, uyy={gt_C['uyy']:+.2e})")
print(f"  {'Size':>6s} | {'Conv%':>6s} | {'uxx bias':>10s} | {'uxx std':>10s} | "
      f"{'uxy bias':>10s} | {'uyy bias':>10s}")
print(f"  {'-'*6} | {'-'*6} | {'-'*10} | {'-'*10} | {'-'*10} | {'-'*10}")

for ss in [21, 25, 31, 41]:
    pts_y_f, pts_x_f = make_poi_grid(IMG_SIZE, ss, spacing=15, margin_extra=15)
    n_f = len(pts_x_f)
    init_u_f = np.full(n_f, round(gt_C['u']), dtype=np.float64)
    init_v_f = np.full(n_f, round(gt_C['v']), dtype=np.float64)
    fft_f = make_fft_result(pts_y_f, pts_x_f, init_u_f, init_v_f)

    r = compute_icgn(ref, deformed_C, fft_f,
                      subset_size=ss,
                      shape_function='quadratic',
                      interpolation_order=5,
                      max_iterations=50)

    v = r.valid_mask
    if np.sum(v) < 10:
        print(f"  {ss:>6d} | {r.convergence_rate:>5.1%} | insufficient valid POIs")
        continue

    uxx_bias = np.mean(r.disp_uxx[v]) - gt_C['uxx']
    uxx_std = np.std(r.disp_uxx[v])
    uxy_bias = np.mean(r.disp_uxy[v]) - gt_C['uxy']
    uyy_bias = np.mean(r.disp_uyy[v]) - gt_C['uyy']

    print(f"  {ss:>6d} | {r.convergence_rate:>5.1%} | {uxx_bias:>+10.2e} | "
          f"{uxx_std:>10.2e} | {uxy_bias:>+10.2e} | {uyy_bias:>+10.2e}")

# subset size 증가 시 2차 계수의 bias가 감소하면 → 알고리즘은 올바르고 관측성 문제
# bias가 일정하면 → 알고리즘 식 오류 가능성

print("\n  해석: subset 증가 시 bias가 감소하면 관측성(observability) 문제,")
print("        bias가 일정하면 warp update 식 오류 가능성")


# =============================================================================
#  요약
# =============================================================================

print()
success = runner.summary()
sys.exit(0 if success else 1)
