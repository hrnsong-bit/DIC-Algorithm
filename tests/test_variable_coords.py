"""generate_variable_local_coordinates 검증 스크립트"""

import numpy as np
from speckle.core.optimization.shape_function_numba import generate_local_coordinates
from speckle.core.optimization.variable_subset_numba import generate_variable_local_coordinates

subset_size = 19
M = subset_size // 2

# === 1. S₀ 동일성 검증 ===
xsi_old, eta_old = generate_local_coordinates(subset_size)
xsi_new, eta_new = generate_variable_local_coordinates(subset_size, subset_type=0)

print("=== S₀ 동일성 검증 ===")
print(f"xsi 동일: {np.allclose(xsi_old, xsi_new)}")
print(f"eta 동일: {np.allclose(eta_old, eta_new)}")
print(f"길이: 기존={len(xsi_old)}, 새={len(xsi_new)}")

# === 2. S₁~S₈ 기본 속성 검증 ===
print("\n=== S₁~S₈ 기본 속성 검증 ===")
for st in range(1, 9):
    xsi_v, eta_v = generate_variable_local_coordinates(subset_size, subset_type=st)
    n_pixels = len(xsi_v)
    expected = subset_size * subset_size
    
    print(f"S{st}: 픽셀수={n_pixels} (예상={expected}, {'OK' if n_pixels == expected else 'FAIL'}), "
          f"ξ=[{xsi_v.min():.0f}, {xsi_v.max():.0f}], "
          f"η=[{eta_v.min():.0f}, {eta_v.max():.0f}]")

# === 3. 좌표 범위 상세 검증 ===
print(f"\n=== 좌표 범위 상세 검증 (M={M}) ===")
expected_ranges = {
    0: ((-M, M), (-M, M)),
    1: ((0, 2*M), (0, 2*M)),
    2: ((-M, M), (0, 2*M)),
    3: ((-2*M, 0), (0, 2*M)),
    4: ((0, 2*M), (-M, M)),
    5: ((-2*M, 0), (-M, M)),
    6: ((0, 2*M), (-2*M, 0)),
    7: ((-M, M), (-2*M, 0)),
    8: ((-2*M, 0), (-2*M, 0)),
}

all_pass = True
for st, ((xsi_min, xsi_max), (eta_min, eta_max)) in expected_ranges.items():
    xsi_v, eta_v = generate_variable_local_coordinates(subset_size, subset_type=st)
    
    xsi_ok = (xsi_v.min() == xsi_min) and (xsi_v.max() == xsi_max)
    eta_ok = (eta_v.min() == eta_min) and (eta_v.max() == eta_max)
    status = "OK" if (xsi_ok and eta_ok) else "FAIL"
    
    if status == "FAIL":
        all_pass = False
        print(f"S{st}: {status} — "
              f"ξ 예상=[{xsi_min}, {xsi_max}] 실제=[{xsi_v.min():.0f}, {xsi_v.max():.0f}], "
              f"η 예상=[{eta_min}, {eta_max}] 실제=[{eta_v.min():.0f}, {eta_v.max():.0f}]")
    else:
        print(f"S{st}: {status}")

print(f"\n전체 결과: {'ALL PASSED' if all_pass else 'SOME FAILED'}")
