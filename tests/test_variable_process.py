"""
검증: process_poi_variable
1) 변형 없는 이미지에서 valid 이웃이 있을 때 ZNCC ≈ 1.0
2) 모든 이웃이 invalid일 때 fail 반환
3) 기존 process_poi_numba(S₀)와 비교 — 동일 조건에서 유사한 ZNCC
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from speckle.core.optimization.interpolation_numba import prefilter_image
from speckle.core.optimization.shape_function_numba import (
    AFFINE, generate_local_coordinates, get_num_params,
)
from speckle.core.optimization.icgn_core_numba import (
    process_poi_numba, allocate_poi_buffers,
)
from speckle.core.optimization.variable_subset_numba import (
    process_poi_variable,
)

# === 테스트 이미지 ===
np.random.seed(42)
img_size = 200
ref_image = np.random.rand(img_size, img_size).astype(np.float64) * 200 + 20
grad_x = np.zeros_like(ref_image)
grad_y = np.zeros_like(ref_image)
grad_x[:, 1:-1] = (ref_image[:, 2:] - ref_image[:, :-2]) / 2.0
grad_y[1:-1, :] = (ref_image[2:, :] - ref_image[:-2, :]) / 2.0

def_image = ref_image.copy()
order = 5
coeffs = prefilter_image(def_image, order=order)

subset_size = 19
n_pixels = subset_size * subset_size
shape_type = AFFINE
n_params = get_num_params(shape_type)
max_iter = 50
conv_thresh = 0.001
zncc_threshold = 0.9

cx, cy = 100, 100
initial_u, initial_v = 0.0, 0.0

# === 버퍼 할당 ===
bufs = allocate_poi_buffers(n_pixels, n_params)
# 추가 버퍼
xsi_local = np.empty(n_pixels, dtype=np.float64)
eta_local = np.empty(n_pixels, dtype=np.float64)

# ============================================================
#  테스트 1: 모든 이웃 valid → 최적 후보 선택 후 IC-GN 성공
# ============================================================
print("=== 테스트 1: 모든 이웃 valid, 변형 없음 → ZNCC ≈ 1.0 ===")
neighbor_valid = np.array([True, True, True, True, True, True, True, True])

zncc, n_iter, conv, fail_code, best_st = process_poi_variable(
    ref_image, grad_x, grad_y,
    coeffs, order,
    cx, cy,
    initial_u, initial_v,
    subset_size, max_iter, conv_thresh,
    shape_type,
    neighbor_valid, zncc_threshold,
    bufs['f'], bufs['dfdx'], bufs['dfdy'],
    bufs['J'], bufs['H'], bufs['H_inv'],
    bufs['p'], bufs['xsi_w'], bufs['eta_w'],
    bufs['x_def'], bufs['y_def'],
    bufs['g'], bufs['b'], bufs['dp'], bufs['p_new'],
    xsi_local, eta_local
)
t1_ok = conv and zncc > 0.99
print(f"  ZNCC={zncc:.6f}, iter={n_iter}, conv={conv}, fail={fail_code}, subset=S{best_st}")
print(f"  결과: {'OK' if t1_ok else 'FAIL'}")

# ============================================================
#  테스트 2: 모든 이웃 invalid → fail
# ============================================================
print("\n=== 테스트 2: 모든 이웃 invalid → fail 반환 ===")
neighbor_invalid = np.array([False, False, False, False, False, False, False, False])

zncc2, n_iter2, conv2, fail_code2, best_st2 = process_poi_variable(
    ref_image, grad_x, grad_y,
    coeffs, order,
    cx, cy,
    initial_u, initial_v,
    subset_size, max_iter, conv_thresh,
    shape_type,
    neighbor_invalid, zncc_threshold,
    bufs['f'], bufs['dfdx'], bufs['dfdy'],
    bufs['J'], bufs['H'], bufs['H_inv'],
    bufs['p'], bufs['xsi_w'], bufs['eta_w'],
    bufs['x_def'], bufs['y_def'],
    bufs['g'], bufs['b'], bufs['dp'], bufs['p_new'],
    xsi_local, eta_local
)
t2_ok = (not conv2) and (best_st2 == -1)
print(f"  ZNCC={zncc2:.6f}, conv={conv2}, fail={fail_code2}, subset={best_st2}")
print(f"  결과: {'OK' if t2_ok else 'FAIL'}")

# ============================================================
#  테스트 3: 단일 방향만 valid (S7=N 확장)
# ============================================================
print("\n=== 테스트 3: S 이웃만 valid → S7(N 확장) 선택 ===")
neighbor_s_only = np.array([False, False, False, False, False, False, True, False])

zncc3, n_iter3, conv3, fail_code3, best_st3 = process_poi_variable(
    ref_image, grad_x, grad_y,
    coeffs, order,
    cx, cy,
    initial_u, initial_v,
    subset_size, max_iter, conv_thresh,
    shape_type,
    neighbor_s_only, zncc_threshold,
    bufs['f'], bufs['dfdx'], bufs['dfdy'],
    bufs['J'], bufs['H'], bufs['H_inv'],
    bufs['p'], bufs['xsi_w'], bufs['eta_w'],
    bufs['x_def'], bufs['y_def'],
    bufs['g'], bufs['b'], bufs['dp'], bufs['p_new'],
    xsi_local, eta_local
)
t3_ok = conv3 and zncc3 > 0.99 and best_st3 == 7
print(f"  ZNCC={zncc3:.6f}, iter={n_iter3}, conv={conv3}, subset=S{best_st3}")
print(f"  결과: {'OK' if t3_ok else 'FAIL'}")

# ============================================================
#  테스트 4: 기존 process_poi_numba(S₀)와 변위 비교
# ============================================================
print("\n=== 테스트 4: 알려진 변위(u=2.5, v=1.3) — S₀ 기존 vs Variable 비교 ===")
shift_u, shift_v = 2.5, 1.3

# shifted deformed (서브픽셀 시프트 — 간단한 정수 근사)
def_shifted = np.zeros_like(ref_image)
su, sv = int(round(shift_u)), int(round(shift_v))
def_shifted[sv:, su:] = ref_image[:img_size-sv, :img_size-su]
coeffs_shifted = prefilter_image(def_shifted, order=order)

# 기존 S₀
xsi_s0, eta_s0 = generate_local_coordinates(subset_size)
bufs_s0 = allocate_poi_buffers(n_pixels, n_params)

zncc_s0, iter_s0, conv_s0, fail_s0 = process_poi_numba(
    ref_image, grad_x, grad_y,
    coeffs_shifted, order,
    cx, cy,
    float(su), float(sv),
    xsi_s0, eta_s0,
    subset_size, max_iter, conv_thresh, shape_type,
    bufs_s0['f'], bufs_s0['dfdx'], bufs_s0['dfdy'],
    bufs_s0['J'], bufs_s0['H'], bufs_s0['H_inv'],
    bufs_s0['p'], bufs_s0['xsi_w'], bufs_s0['eta_w'],
    bufs_s0['x_def'], bufs_s0['y_def'],
    bufs_s0['g'], bufs_s0['b'], bufs_s0['dp'], bufs_s0['p_new']
)
print(f"  기존 S₀: ZNCC={zncc_s0:.6f}, iter={iter_s0}, conv={conv_s0}")
print(f"    u={bufs_s0['p'][0]:.4f}, v={bufs_s0['p'][3]:.4f}")

# Variable (모든 이웃 valid)
neighbor_all = np.array([True, True, True, True, True, True, True, True])
bufs_var = allocate_poi_buffers(n_pixels, n_params)
xsi_l2 = np.empty(n_pixels, dtype=np.float64)
eta_l2 = np.empty(n_pixels, dtype=np.float64)

zncc_var, iter_var, conv_var, fail_var, best_var = process_poi_variable(
    ref_image, grad_x, grad_y,
    coeffs_shifted, order,
    cx, cy,
    float(su), float(sv),
    subset_size, max_iter, conv_thresh,
    shape_type,
    neighbor_all, zncc_threshold,
    bufs_var['f'], bufs_var['dfdx'], bufs_var['dfdy'],
    bufs_var['J'], bufs_var['H'], bufs_var['H_inv'],
    bufs_var['p'], bufs_var['xsi_w'], bufs_var['eta_w'],
    bufs_var['x_def'], bufs_var['y_def'],
    bufs_var['g'], bufs_var['b'], bufs_var['dp'], bufs_var['p_new'],
    xsi_l2, eta_l2
)
print(f"  Variable: ZNCC={zncc_var:.6f}, iter={iter_var}, conv={conv_var}, subset=S{best_var}")
print(f"    u={bufs_var['p'][0]:.4f}, v={bufs_var['p'][3]:.4f}")

t4_ok = conv_s0 and conv_var and abs(zncc_s0 - zncc_var) < 0.05
print(f"  ZNCC 차이: {abs(zncc_s0 - zncc_var):.6f}")
print(f"  결과: {'OK' if t4_ok else 'FAIL'}")

# === 최종 ===
all_pass = t1_ok and t2_ok and t3_ok and t4_ok
print(f"\n전체 결과: {'ALL PASSED' if all_pass else 'FAILED'}")
