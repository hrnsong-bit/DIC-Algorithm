# test_debug_variable.py — 프로젝트 루트에 저장 후 실행
import sys; sys.path.insert(0, '.')
import numpy as np
import cv2
from speckle.core.initial_guess.fft_cc import compute_fft_cc
from speckle.core.optimization.icgn import compute_icgn

ref = cv2.imread('synthetic_crack_data/reference.tiff', cv2.IMREAD_GRAYSCALE).astype(np.float64)
defm = cv2.imread('synthetic_crack_data/deformed.tiff', cv2.IMREAD_GRAYSCALE).astype(np.float64)

fft = compute_fft_cc(ref, defm, subset_size=13, spacing=5)

# 1단계: Variable Subset OFF
r_off = compute_icgn(ref, defm, fft, subset_size=13, zncc_threshold=0.90,
                     enable_variable_subset=False)

# 2단계: Variable Subset ON
r_on = compute_icgn(ref, defm, fft, subset_size=13, zncc_threshold=0.90,
                    enable_variable_subset=True)

# 불량 POI 비교
bad_off = np.where(~r_off.valid_mask)[0]
bad_on = np.where(~r_on.valid_mask)[0]

print(f"\n=== 비교 결과 ===")
print(f"OFF invalid: {len(bad_off)}")
print(f"ON  invalid: {len(bad_on)}")
print(f"복원된 POI: {len(bad_off) - len(bad_on)}개")

# OFF에서 불량이었던 POI들의 ZNCC 변화 추적
print(f"\n=== OFF 불량 POI {len(bad_off)}개의 ZNCC 변화 ===")
print(f"{'IDX':>5} {'X':>4} {'Y':>4} {'ZNCC_OFF':>10} {'ZNCC_ON':>10} {'변화':>10} {'ON유효':>6}")
unchanged = 0
for idx in bad_off:
    x, y = r_off.points_x[idx], r_off.points_y[idx]
    z_off = r_off.zncc_values[idx]
    z_on = r_on.zncc_values[idx]
    delta = z_on - z_off
    valid_on = r_on.valid_mask[idx]
    flag = "" if abs(delta) > 1e-6 else " ◀ 변화없음"
    if abs(delta) < 1e-6:
        unchanged += 1
    print(f"{idx:5d} {x:4.0f} {y:4.0f} {z_off:10.6f} {z_on:10.6f} {delta:+10.6f} {str(valid_on):>6}{flag}")

print(f"\n총 {len(bad_off)}개 불량 중 ZNCC 변화 없음: {unchanged}개")
print(f"ZNCC가 변한 POI: {len(bad_off) - unchanged}개")

# 변화없는 POI의 패턴 분석
if unchanged > 0:
    print(f"\n=== 변화없는 POI 상세 분석 ===")
    crack_mask = np.load('synthetic_crack_data/crack_mask.npy')
    half = 13 // 2  # subset half
    for idx in bad_off:
        z_off = r_off.zncc_values[idx]
        z_on = r_on.zncc_values[idx]
        if abs(z_on - z_off) < 1e-6:
            x, y = int(r_off.points_x[idx]), int(r_off.points_y[idx])
            # 크랙 픽셀 수 (S0 기준)
            y_lo, y_hi = max(0, y-half), min(500, y+half+1)
            x_lo, x_hi = max(0, x-half), min(500, x+half+1)
            crack_s0 = int(np.sum(crack_mask[y_lo:y_hi, x_lo:x_hi]))
            # 8방향 이웃 valid 여부
            dist = abs(y - 250)
            fail_off = r_off.failure_reason[idx] if hasattr(r_off, 'failure_reason') else "?"
            print(f"  POI#{idx} ({x},{y}): dist_crack={dist}, crack_px_S0={crack_s0}, "
                  f"zncc={z_off:.4f}, fail={fail_off}")
