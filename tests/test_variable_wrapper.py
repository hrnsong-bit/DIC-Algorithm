"""
검증: compute_variable_subset_recalc
1) 불량 POI 없으면 스킵
2) 불량 POI가 있으면 복원 시도 후 결과 병합
3) 격자 구조 감지 확인
4) valid_mask, zncc_values가 in-place 업데이트되는지 확인
"""
import numpy as np
import sys
import logging
sys.path.insert(0, '.')

logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

from speckle.core.optimization.interpolation_numba import prefilter_image
from speckle.core.optimization.shape_function_numba import (
    AFFINE, get_num_params, generate_local_coordinates,
)
from speckle.core.optimization.icgn_core_numba import (
    process_all_pois_parallel, allocate_batch_buffers, ICGN_SUCCESS,
)
from speckle.core.optimization.variable_subset import (
    compute_variable_subset_recalc, _detect_grid_structure,
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

# === POI 격자 (6x6) ===
spacing = 15
margin = 30
xs = np.arange(margin, margin + 6 * spacing, spacing, dtype=np.int64)
ys = np.arange(margin, margin + 6 * spacing, spacing, dtype=np.int64)
grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
points_x = grid_x.ravel().copy()
points_y = grid_y.ravel().copy()
n_poi = len(points_x)

initial_u = np.zeros(n_poi, dtype=np.float64)
initial_v = np.zeros(n_poi, dtype=np.float64)

# ============================================================
#  테스트 0: 격자 구조 감지
# ============================================================
print("=== 테스트 0: 격자 구조 감지 ===")
grid_info = _detect_grid_structure(points_x, points_y)
t0_ok = grid_info is not None and grid_info == (6, 6, 15)
print(f"  결과: {grid_info} {'OK' if t0_ok else 'FAIL'}")

# ============================================================
#  1단계 IC-GN 실행 (정상 결과 생성)
# ============================================================
print("\n=== 1단계 IC-GN 실행 ===")
xsi, eta = generate_local_coordinates(subset_size)
batch_bufs = allocate_batch_buffers(n_poi, n_pixels, n_params)

result_p = np.empty((n_poi, n_params), dtype=np.float64)
result_zncc = np.empty(n_poi, dtype=np.float64)
result_iter = np.empty(n_poi, dtype=np.int32)
result_conv = np.empty(n_poi, dtype=np.bool_)
result_fail = np.empty(n_poi, dtype=np.int32)

process_all_pois_parallel(
    ref_image, grad_x, grad_y,
    coeffs, order,
    points_x, points_y,
    initial_u, initial_v,
    xsi, eta,
    subset_size, 50, 0.001, shape_type,
    result_p, result_zncc, result_iter, result_conv, result_fail,
    batch_bufs['f'], batch_bufs['dfdx'], batch_bufs['dfdy'],
    batch_bufs['J'], batch_bufs['H'], batch_bufs['H_inv'],
    batch_bufs['p'], batch_bufs['xsi_w'], batch_bufs['eta_w'],
    batch_bufs['x_def'], batch_bufs['y_def'],
    batch_bufs['g'], batch_bufs['b'], batch_bufs['dp'], batch_bufs['p_new'],
)
print(f"  전체 POI: {n_poi}, 수렴: {np.sum(result_conv)}")

# ============================================================
#  테스트 1: 불량 POI 없음 → 스킵
# ============================================================
print("\n=== 테스트 1: 불량 POI 없음 → 스킵 ===")
valid_mask_1 = result_conv.copy()
zncc_values_1 = result_zncc.copy()
params_1 = result_p.copy()
conv_flags_1 = result_conv.copy()
iter_counts_1 = result_iter.copy()
fail_reasons_1 = result_fail.copy()

report_1 = compute_variable_subset_recalc(
    ref_image, grad_x, grad_y, coeffs, order,
    points_x, points_y, initial_u, initial_v,
    valid_mask_1, zncc_values_1, params_1,
    conv_flags_1, iter_counts_1, fail_reasons_1,
    subset_size,
)
t1_ok = report_1['n_bad'] == 0 and report_1['n_recovered'] == 0
print(f"  n_bad={report_1['n_bad']}, n_recovered={report_1['n_recovered']}")
print(f"  결과: {'OK' if t1_ok else 'FAIL'}")

# ============================================================
#  테스트 2: 인위적 불량 POI → 복원 확인
# ============================================================
print("\n=== 테스트 2: 인위적 불량 POI 5개 → 복원 ===")
valid_mask_2 = result_conv.copy()
zncc_values_2 = result_zncc.copy()
params_2 = result_p.copy()
conv_flags_2 = result_conv.copy()
iter_counts_2 = result_iter.copy()
fail_reasons_2 = result_fail.copy()

# POI 8, 14, 15, 20, 26을 불량으로 설정 (내부 POI, 이웃 있음)
fake_bad = [8, 14, 15, 20, 26]
for idx in fake_bad:
    valid_mask_2[idx] = False
    zncc_values_2[idx] = 0.5
    conv_flags_2[idx] = False

print(f"  불량으로 설정: {fake_bad}")
print(f"  valid 수 (전): {np.sum(valid_mask_2)}")

report_2 = compute_variable_subset_recalc(
    ref_image, grad_x, grad_y, coeffs, order,
    points_x, points_y, initial_u, initial_v,
    valid_mask_2, zncc_values_2, params_2,
    conv_flags_2, iter_counts_2, fail_reasons_2,
    subset_size,
)
print(f"  n_bad={report_2['n_bad']}, n_recovered={report_2['n_recovered']}, n_failed={report_2['n_failed']}")
print(f"  valid 수 (후): {np.sum(valid_mask_2)}")
print(f"  subset_types: {report_2['subset_types']}")

t2_ok = report_2['n_bad'] == 5 and report_2['n_recovered'] == 5
print(f"  결과: {'OK' if t2_ok else 'FAIL'}")

# 복원된 POI의 ZNCC 확인
print("\n  복원된 POI 상세:")
for idx in fake_bad:
    print(f"    POI#{idx}: valid={valid_mask_2[idx]}, ZNCC={zncc_values_2[idx]:.4f}, conv={conv_flags_2[idx]}")

# === 최종 ===
all_pass = t0_ok and t1_ok and t2_ok
print(f"\n전체 결과: {'ALL PASSED' if all_pass else 'FAILED'}")
