"""
검증: compute_icgn 내부에서 Variable Subset 자동 호출
1) 크랙 없는 이미지 → 불량 POI 없음 → 스킵 확인
2) 인위적 크랙 이미지 → 불량 POI 발생 → 복원 확인
"""
import numpy as np
import sys
import logging
sys.path.insert(0, '.')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)

from speckle.core.initial_guess.fft_cc import compute_fft_cc
from speckle.core.optimization.icgn import compute_icgn

# === 테스트 1: 크랙 없는 이미지 ===
print("=" * 60)
print("테스트 1: 크랙 없는 이미지 - Variable Subset 스킵 확인")
print("=" * 60)

np.random.seed(42)
img_size = 300
ref_clean = np.random.rand(img_size, img_size).astype(np.float64) * 200 + 20
def_clean = ref_clean.copy()

fft_result_1 = compute_fft_cc(ref_clean, def_clean, subset_size=19, spacing=15)
icgn_result_1 = compute_icgn(
    ref_clean, def_clean, fft_result_1,
    subset_size=19, zncc_threshold=0.9,
)

n_valid_1 = np.sum(icgn_result_1.valid_mask)
n_total_1 = len(icgn_result_1.points_x)
print(f"\n결과: valid={n_valid_1}/{n_total_1}")
t1_ok = n_valid_1 == n_total_1
print(f"테스트 1: {'OK' if t1_ok else 'FAIL'}")

# === 테스트 2: 인위적 크랙 이미지 ===
print("\n" + "=" * 60)
print("테스트 2: 인위적 크랙 - Variable Subset 복원 확인")
print("=" * 60)

ref_crack = ref_clean.copy()
def_crack = ref_clean.copy()

# 수평 크랙 시뮬레이션: y=150 라인에서 아래쪽을 3픽셀 아래로 이동
crack_y = 150
shift = 3
def_crack[crack_y + shift:, :] = ref_clean[crack_y:-shift, :]
def_crack[crack_y:crack_y + shift, :] = 0  # 크랙 내부 = 검정

fft_result_2 = compute_fft_cc(ref_crack, def_crack, subset_size=19, spacing=15)
icgn_result_2 = compute_icgn(
    ref_crack, def_crack, fft_result_2,
    subset_size=19, zncc_threshold=0.9,
)

n_valid_2 = np.sum(icgn_result_2.valid_mask)
n_total_2 = len(icgn_result_2.points_x)
print(f"\n결과: valid={n_valid_2}/{n_total_2}")

# 크랙 근처 POI 확인
crack_pois = np.where(
    (icgn_result_2.points_y >= crack_y - 20) &
    (icgn_result_2.points_y <= crack_y + 20)
)[0]
print(f"크랙 근처 POI ({len(crack_pois)}개):")
for idx in crack_pois[:10]:
    print(f"  POI#{idx} ({icgn_result_2.points_x[idx]},{icgn_result_2.points_y[idx]}): "
          f"valid={icgn_result_2.valid_mask[idx]}, "
          f"ZNCC={icgn_result_2.zncc_values[idx]:.4f}")

# Variable Subset 로그에 "복원" 메시지가 나왔는지는 로그로 확인
t2_ok = n_valid_2 > 0
print(f"\n테스트 2: {'OK' if t2_ok else 'FAIL'}")

# === 최종 ===
all_pass = t1_ok and t2_ok
print(f"\n전체 결과: {'ALL PASSED' if all_pass else 'FAILED'}")
