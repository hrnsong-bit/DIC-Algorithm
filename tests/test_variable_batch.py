"""
검증: process_bad_pois_parallel
1) 변형 없는 이미지에서 다수 불량 POI 일괄 처리
2) process_poi_variable 개별 결과와 배치 결과 일치 확인
3) 이웃 invalid POI는 실패 반환 확인
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from speckle.core.optimization.interpolation_numba import prefilter_image
from speckle.core.optimization.shape_function_numba import (
    AFFINE, get_num_params,
)
from speckle.core.optimization.icgn_core_numba import allocate_poi_buffers
from speckle.core.optimization.variable_subset_numba import (
    process_poi_variable,
    process_bad_pois_parallel,
    allocate_variable_batch_buffers,
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

# === POI 그리드 (5x5 = 25개) ===
spacing = 15
xs = np.arange(50, 50 + 5 * spacing, spacing, dtype=np.int64)
ys = np.arange(50, 50 + 5 * spacing, spacing, dtype=np.int64)
grid_y, grid_x = np.meshgrid(ys, xs, indexing='ij')
points_x = grid_x.ravel().copy()
points_y = grid_y.ravel().copy()
n_total = len(points_x)

initial_u = np.zeros(n_total, dtype=np.float64)
initial_v = np.zeros(n_total, dtype=np.float64)

# === 불량 POI 설정 (인덱스 3, 7, 12, 18) ===
bad_indices = np.array([3, 7, 12, 18], dtype=np.int64)
n_bad = len(bad_indices)

# 이웃 valid 정보: 모두 valid (테스트 단순화)
all_neighbor_valid = np.ones((n_bad, 8), dtype=np.bool_)
# 인덱스 18번은 모든 이웃 invalid로 설정 (실패 테스트)
all_neighbor_valid[3, :] = False  # bad_indices[3] = 18

# ============================================================
#  테스트 1: 배치 처리 실행
# ============================================================
print("=== 테스트 1: 배치 처리 (4개 불량 POI) ===")
batch_bufs = allocate_variable_batch_buffers(n_bad, n_pixels, n_params)

result_p = np.empty((n_bad, n_params), dtype=np.float64)
result_zncc = np.empty(n_bad, dtype=np.float64)
result_iter = np.empty(n_bad, dtype=np.int32)
result_conv = np.empty(n_bad, dtype=np.bool_)
result_fail = np.empty(n_bad, dtype=np.int32)
result_subset = np.empty(n_bad, dtype=np.int32)

process_bad_pois_parallel(
    ref_image, grad_x, grad_y,
    coeffs, order,
    points_x, points_y,
    initial_u, initial_v,
    subset_size, max_iter, conv_thresh,
    shape_type,
    bad_indices,
    all_neighbor_valid,
    zncc_threshold,
    result_p, result_zncc, result_iter, result_conv, result_fail, result_subset,
    batch_bufs['f'], batch_bufs['dfdx'], batch_bufs['dfdy'],
    batch_bufs['J'], batch_bufs['H'], batch_bufs['H_inv'],
    batch_bufs['p'], batch_bufs['xsi_w'], batch_bufs['eta_w'],
    batch_bufs['x_def'], batch_bufs['y_def'],
    batch_bufs['g'], batch_bufs['b'], batch_bufs['dp'], batch_bufs['p_new'],
    batch_bufs['xsi_local'], batch_bufs['eta_local']
)

for k in range(n_bad):
    idx = bad_indices[k]
    print(f"  POI#{idx} ({points_x[idx]},{points_y[idx]}): "
          f"ZNCC={result_zncc[k]:.4f}, conv={result_conv[k]}, "
          f"fail={result_fail[k]}, subset=S{result_subset[k]}")

# ============================================================
#  테스트 2: 개별 vs 배치 결과 비교 (bad_indices[0] = 3)
# ============================================================
print("\n=== 테스트 2: 개별 처리와 배치 결과 비교 (POI#3) ===")
single_bufs = allocate_poi_buffers(n_pixels, n_params)
xsi_l = np.empty(n_pixels, dtype=np.float64)
eta_l = np.empty(n_pixels, dtype=np.float64)

idx0 = bad_indices[0]
zncc_s, iter_s, conv_s, fail_s, st_s = process_poi_variable(
    ref_image, grad_x, grad_y,
    coeffs, order,
    points_x[idx0], points_y[idx0],
    initial_u[idx0], initial_v[idx0],
    subset_size, max_iter, conv_thresh,
    shape_type,
    all_neighbor_valid[0],
    zncc_threshold,
    single_bufs['f'], single_bufs['dfdx'], single_bufs['dfdy'],
    single_bufs['J'], single_bufs['H'], single_bufs['H_inv'],
    single_bufs['p'], single_bufs['xsi_w'], single_bufs['eta_w'],
    single_bufs['x_def'], single_bufs['y_def'],
    single_bufs['g'], single_bufs['b'], single_bufs['dp'], single_bufs['p_new'],
    xsi_l, eta_l
)

zncc_match = np.isclose(zncc_s, result_zncc[0])
conv_match = (conv_s == result_conv[0])
subset_match = (st_s == result_subset[0])
print(f"  개별: ZNCC={zncc_s:.6f}, conv={conv_s}, subset=S{st_s}")
print(f"  배치: ZNCC={result_zncc[0]:.6f}, conv={result_conv[0]}, subset=S{result_subset[0]}")
print(f"  일치: ZNCC={zncc_match}, conv={conv_match}, subset={subset_match}")

# ============================================================
#  테스트 3: invalid 이웃 POI 실패 확인
# ============================================================
print("\n=== 테스트 3: 모든 이웃 invalid (POI#18) → 실패 ===")
t3_ok = (not result_conv[3]) and (result_subset[3] == -1)
print(f"  conv={result_conv[3]}, subset={result_subset[3]}")
print(f"  결과: {'OK' if t3_ok else 'FAIL'}")

# === 최종 ===
t1_ok = all(result_conv[k] for k in range(3))  # 0,1,2는 성공
t2_ok = zncc_match and conv_match and subset_match
all_pass = t1_ok and t2_ok and t3_ok
print(f"\n전체 결과: {'ALL PASSED' if all_pass else 'FAILED'}")
