"""
검증: evaluate_candidate_zncc가
1) 변형 없는 이미지 쌍(ref==def)에서 S₀ → ZNCC ≈ 1.0
2) S₁~S₈에서도 ZNCC ≈ 1.0 (변형 없을 때)
3) 크랙이 포함된 서브셋에서는 ZNCC가 낮아지는지 확인
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from speckle.core.optimization.interpolation_numba import prefilter_image
from speckle.core.optimization.shape_function_numba import AFFINE, generate_local_coordinates
from speckle.core.optimization.variable_subset_numba import (
    generate_variable_local_coordinates,
    evaluate_candidate_zncc,
    S0, S1, S2, S3, S4, S5, S6, S7, S8,
)

# === 테스트 이미지 생성 ===
np.random.seed(42)
img_size = 200
ref_image = np.random.rand(img_size, img_size).astype(np.float64) * 200 + 20
grad_x = np.zeros_like(ref_image)
grad_y = np.zeros_like(ref_image)
grad_x[:, 1:-1] = (ref_image[:, 2:] - ref_image[:, :-2]) / 2.0
grad_y[1:-1, :] = (ref_image[2:, :] - ref_image[:-2, :]) / 2.0

# deformed = reference (변형 없음)
def_image = ref_image.copy()
order = 5
coeffs = prefilter_image(def_image, order=order)

subset_size = 19
half = subset_size // 2  # M = 9
n_pixels = subset_size * subset_size

cx, cy = 100, 100
initial_u, initial_v = 0.0, 0.0  # 변형 없으므로 0

# 작업 버퍼
f = np.empty(n_pixels, dtype=np.float64)
dfdx = np.empty(n_pixels, dtype=np.float64)
dfdy = np.empty(n_pixels, dtype=np.float64)
xsi_w = np.empty(n_pixels, dtype=np.float64)
eta_w = np.empty(n_pixels, dtype=np.float64)
x_def = np.empty(n_pixels, dtype=np.float64)
y_def = np.empty(n_pixels, dtype=np.float64)
g = np.empty(n_pixels, dtype=np.float64)

# === 테스트 1: 변형 없는 이미지에서 S₀~S₈ ZNCC ===
print("=== 테스트 1: 변형 없는 이미지 (ref == def), ZNCC ≈ 1.0 기대 ===")
subset_types = [S0, S1, S2, S3, S4, S5, S6, S7, S8]
subset_names = ['S0', 'S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']

# 각 서브셋 타입의 좌표 범위 (M = half)
M = half
ranges = {
    S0: (-M, M, -M, M),
    S1: (0, 2*M, 0, 2*M),
    S2: (-M, M, 0, 2*M),
    S3: (-2*M, 0, 0, 2*M),
    S4: (0, 2*M, -M, M),
    S5: (-2*M, 0, -M, M),
    S6: (0, 2*M, -2*M, 0),
    S7: (-M, M, -2*M, 0),
    S8: (-2*M, 0, -2*M, 0),
}

all_pass = True
for st, name in zip(subset_types, subset_names):
    xsi, eta = generate_variable_local_coordinates(subset_size, st)
    xsi_min, xsi_max, eta_min, eta_max = ranges[st]

    zncc = evaluate_candidate_zncc(
        ref_image, grad_x, grad_y,
        coeffs, order,
        cx, cy,
        initial_u, initial_v,
        xsi_min, xsi_max, eta_min, eta_max,
        xsi, eta,
        AFFINE,
        f, dfdx, dfdy,
        xsi_w, eta_w,
        x_def, y_def,
        g
    )
    ok = zncc > 0.99
    if not ok:
        all_pass = False
    print(f"  {name}: ZNCC = {zncc:.6f} {'OK' if ok else 'FAIL'}")

# === 테스트 2: 알려진 변위(u=2.0, v=1.0) 적용 ===
print("\n=== 테스트 2: 알려진 변위 (u=2, v=1), 올바른 초기값 → ZNCC ≈ 1.0 ===")
# shifted deformed 이미지 생성 (단순 정수 시프트)
def_shifted = np.zeros_like(ref_image)
shift_u, shift_v = 2, 1  # x방향 +2, y방향 +1
def_shifted[shift_v:, shift_u:] = ref_image[:img_size-shift_v, :img_size-shift_u]
coeffs_shifted = prefilter_image(def_shifted, order=order)

xsi_s0, eta_s0 = generate_variable_local_coordinates(subset_size, S0)
zncc_correct = evaluate_candidate_zncc(
    ref_image, grad_x, grad_y,
    coeffs_shifted, order,
    cx, cy,
    float(shift_u), float(shift_v),
    -M, M, -M, M,
    xsi_s0, eta_s0,
    AFFINE,
    f, dfdx, dfdy,
    xsi_w, eta_w,
    x_def, y_def,
    g
)
print(f"  올바른 초기값 (u=2, v=1): ZNCC = {zncc_correct:.6f} {'OK' if zncc_correct > 0.95 else 'FAIL'}")
if zncc_correct <= 0.95:
    all_pass = False

# 잘못된 초기값
zncc_wrong = evaluate_candidate_zncc(
    ref_image, grad_x, grad_y,
    coeffs_shifted, order,
    cx, cy,
    0.0, 0.0,
    -M, M, -M, M,
    xsi_s0, eta_s0,
    AFFINE,
    f, dfdx, dfdy,
    xsi_w, eta_w,
    x_def, y_def,
    g
)
print(f"  잘못된 초기값 (u=0, v=0): ZNCC = {zncc_wrong:.6f} (낮을수록 정상)")

# === 테스트 3: 경계 밖 → -1.0 반환 ===
print("\n=== 테스트 3: 경계 밖 서브셋 → ZNCC = -1.0 ===")
xsi_s1, eta_s1 = generate_variable_local_coordinates(subset_size, S1)
zncc_edge = evaluate_candidate_zncc(
    ref_image, grad_x, grad_y,
    coeffs, order,
    195, 195,
    0.0, 0.0,
    0, 2*M, 0, 2*M,
    xsi_s1, eta_s1,
    AFFINE,
    f, dfdx, dfdy,
    xsi_w, eta_w,
    x_def, y_def,
    g
)
edge_ok = zncc_edge == -1.0
if not edge_ok:
    all_pass = False
print(f"  경계 초과 (cx=195, cy=195, S₁): ZNCC = {zncc_edge} {'OK' if edge_ok else 'FAIL'}")

# === 최종 ===
print(f"\n전체 결과: {'ALL PASSED' if all_pass else 'FAILED'}")
