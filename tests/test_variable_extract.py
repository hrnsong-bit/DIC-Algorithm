"""
검증: extract_reference_subset_variable이 S₀ 입력 시
기존 extract_reference_subset과 동일한 결과를 생성하는지 확인
"""
import numpy as np
import sys
sys.path.insert(0, '.')

from speckle.core.optimization.icgn_core_numba import extract_reference_subset
from speckle.core.optimization.variable_subset_numba import extract_reference_subset_variable

# 테스트 이미지 생성
np.random.seed(42)
img_size = 100
ref_image = np.random.rand(img_size, img_size).astype(np.float64) * 200 + 20
grad_x = np.zeros_like(ref_image)
grad_y = np.zeros_like(ref_image)
grad_x[:, 1:-1] = (ref_image[:, 2:] - ref_image[:, :-2]) / 2.0
grad_y[1:-1, :] = (ref_image[2:, :] - ref_image[:-2, :]) / 2.0

subset_size = 19
half = subset_size // 2  # M = 9
n_pixels = subset_size * subset_size

cx, cy = 50, 50

# === 기존 함수 ===
f_old = np.empty(n_pixels, dtype=np.float64)
dfdx_old = np.empty(n_pixels, dtype=np.float64)
dfdy_old = np.empty(n_pixels, dtype=np.float64)
f_mean_old, f_tilde_old, valid_old = extract_reference_subset(
    ref_image, grad_x, grad_y, cx, cy, subset_size,
    f_old, dfdx_old, dfdy_old
)

# === 새 함수 (S₀ 좌표: xsi_min=-M, xsi_max=+M, eta_min=-M, eta_max=+M) ===
f_new = np.empty(n_pixels, dtype=np.float64)
dfdx_new = np.empty(n_pixels, dtype=np.float64)
dfdy_new = np.empty(n_pixels, dtype=np.float64)
f_mean_new, f_tilde_new, valid_new = extract_reference_subset_variable(
    ref_image, grad_x, grad_y, cx, cy,
    -half, half, -half, half,
    f_new, dfdx_new, dfdy_new
)

# === 비교 ===
print("=== S₀ 동일성 검증 ===")
print(f"valid: 기존={valid_old}, 새={valid_new}, 동일={valid_old == valid_new}")
print(f"f_mean: 기존={f_mean_old:.10f}, 새={f_mean_new:.10f}, 동일={np.isclose(f_mean_old, f_mean_new)}")
print(f"f_tilde: 기존={f_tilde_old:.10f}, 새={f_tilde_new:.10f}, 동일={np.isclose(f_tilde_old, f_tilde_new)}")
print(f"f 배열 동일: {np.allclose(f_old, f_new)}")
print(f"dfdx 배열 동일: {np.allclose(dfdx_old, dfdx_new)}")
print(f"dfdy 배열 동일: {np.allclose(dfdy_old, dfdy_new)}")

# === 비대칭 서브셋 테스트 (S₁: SE 확장) ===
print("\n=== S₁ (SE 확장) 기본 검증 ===")
M = half
f_s1 = np.empty(n_pixels, dtype=np.float64)
dfdx_s1 = np.empty(n_pixels, dtype=np.float64)
dfdy_s1 = np.empty(n_pixels, dtype=np.float64)
f_mean_s1, f_tilde_s1, valid_s1 = extract_reference_subset_variable(
    ref_image, grad_x, grad_y, cx, cy,
    0, 2 * M, 0, 2 * M,
    f_s1, dfdx_s1, dfdy_s1
)
print(f"valid: {valid_s1}")
print(f"f_mean: {f_mean_s1:.6f}")
print(f"f_tilde: {f_tilde_s1:.6f}")
print(f"추출 범위: col=[{cx+0}, {cx+2*M}], row=[{cy+0}, {cy+2*M}]")
print(f"수동 확인 - 첫 픽셀: ref_image[{cy},{cx}]={ref_image[cy,cx]:.6f}, f_s1[0]={f_s1[0]:.6f}, 동일={np.isclose(ref_image[cy,cx], f_s1[0])}")
print(f"수동 확인 - 마지막 픽셀: ref_image[{cy+2*M},{cx+2*M}]={ref_image[cy+2*M,cx+2*M]:.6f}, f_s1[-1]={f_s1[-1]:.6f}, 동일={np.isclose(ref_image[cy+2*M,cx+2*M], f_s1[-1])}")

# === 경계 체크 테스트 ===
print("\n=== 경계 체크 검증 ===")
_, _, valid_edge = extract_reference_subset_variable(
    ref_image, grad_x, grad_y, 95, 95,
    0, 2 * M, 0, 2 * M,
    f_s1, dfdx_s1, dfdy_s1
)
print(f"경계 초과 (cx=95, cy=95, S₁): valid={valid_edge} (예상=False)")

# === 최종 판정 ===
all_pass = (
    valid_old == valid_new
    and np.isclose(f_mean_old, f_mean_new)
    and np.isclose(f_tilde_old, f_tilde_new)
    and np.allclose(f_old, f_new)
    and np.allclose(dfdx_old, dfdx_new)
    and np.allclose(dfdy_old, dfdy_new)
    and valid_s1
    and np.isclose(ref_image[cy, cx], f_s1[0])
    and np.isclose(ref_image[cy + 2*M, cx + 2*M], f_s1[-1])
    and not valid_edge
)
print(f"\n전체 결과: {'ALL PASSED' if all_pass else 'FAILED'}")