"""
검증: synthetic_crack_data에서 Variable Subset 실측 검증

비교 항목:
1) 1단계만 (기존 DIC) vs 2단계 포함 (Variable Subset) — valid POI 수
2) 복원된 POI의 ground truth 대비 변위 오차
3) 크랙 근처 POI 복원율
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

from pathlib import Path
import cv2

from speckle.core.initial_guess.fft_cc import compute_fft_cc
from speckle.core.optimization.icgn import compute_icgn

# === 데이터 로드 ===
data_dir = Path('synthetic_crack_data')
ref_image = cv2.imread(str(data_dir / 'reference.tiff'), cv2.IMREAD_GRAYSCALE)
def_image = cv2.imread(str(data_dir / 'deformed.tiff'), cv2.IMREAD_GRAYSCALE)
gt_u = np.load(str(data_dir / 'ground_truth_u.npy'))
gt_v = np.load(str(data_dir / 'ground_truth_v.npy'))
crack_mask = np.load(str(data_dir / 'crack_mask.npy'))

print(f"이미지 크기: {ref_image.shape}")
print(f"크랙 픽셀: {np.sum(crack_mask)}")
print(f"GT u 범위: [{gt_u.min():.3f}, {gt_u.max():.3f}]")
print(f"GT v 범위: [{gt_v.min():.3f}, {gt_v.max():.3f}]")

# === DIC 파라미터 ===
subset_size = 21
spacing = 10
zncc_threshold = 0.9

# === FFT-CC ===
print("\n" + "=" * 60)
print("FFT-CC 실행")
print("=" * 60)
fft_result = compute_fft_cc(
    ref_image.astype(np.float64),
    def_image.astype(np.float64),
    subset_size=subset_size,
    spacing=spacing,
)

# === 1단계 IC-GN (Variable Subset 비활성) ===
# Variable Subset 호출을 일시적으로 우회하기 위해
# 먼저 기존 결과를 기록해둠
print("\n" + "=" * 60)
print("IC-GN + Variable Subset 실행")
print("=" * 60)
icgn_result = compute_icgn(
    ref_image.astype(np.float64),
    def_image.astype(np.float64),
    fft_result,
    subset_size=subset_size,
    zncc_threshold=zncc_threshold,
)

# === 결과 분석 ===
points_x = icgn_result.points_x
points_y = icgn_result.points_y
n_total = len(points_x)
n_valid = np.sum(icgn_result.valid_mask)

print(f"\n{'=' * 60}")
print(f"결과 요약")
print(f"{'=' * 60}")
print(f"전체 POI: {n_total}")
print(f"최종 valid: {n_valid} ({n_valid/n_total*100:.1f}%)")

# === Ground Truth 비교 ===
valid_idx = np.where(icgn_result.valid_mask)[0]

# POI 위치에서 GT 값 추출
gt_u_at_poi = np.array([gt_u[points_y[i], points_x[i]] for i in range(n_total)])
gt_v_at_poi = np.array([gt_v[points_y[i], points_x[i]] for i in range(n_total)])

# valid POI에서의 오차
if len(valid_idx) > 0:
    err_u = icgn_result.disp_u[valid_idx] - gt_u_at_poi[valid_idx]
    err_v = icgn_result.disp_v[valid_idx] - gt_v_at_poi[valid_idx]
    err_mag = np.sqrt(err_u**2 + err_v**2)

    print(f"\n--- Valid POI 변위 오차 (Ground Truth 대비) ---")
    print(f"  u 오차: mean={np.mean(np.abs(err_u)):.4f}, "
          f"std={np.std(err_u):.4f}, max={np.max(np.abs(err_u)):.4f}")
    print(f"  v 오차: mean={np.mean(np.abs(err_v)):.4f}, "
          f"std={np.std(err_v):.4f}, max={np.max(np.abs(err_v)):.4f}")
    print(f"  총 오차: mean={np.mean(err_mag):.4f}, "
          f"std={np.std(err_mag):.4f}, max={np.max(err_mag):.4f}")

    # 오차 구간별 분포
    bins = [0, 0.01, 0.05, 0.1, 0.5, 1.0, float('inf')]
    hist, _ = np.histogram(err_mag, bins=bins)
    print(f"\n  오차 분포:")
    for i in range(len(hist)):
        pct = hist[i] / len(valid_idx) * 100
        if bins[i+1] == float('inf'):
            print(f"    > {bins[i]:.2f} px: {hist[i]}개 ({pct:.1f}%)")
        else:
            print(f"    {bins[i]:.2f} ~ {bins[i+1]:.2f} px: {hist[i]}개 ({pct:.1f}%)")

# === 크랙 근처 POI 분석 ===
crack_tip_y, crack_tip_x = 250, 250
crack_line_y = 250

print(f"\n--- 크랙 근처 POI 분석 (|y - {crack_line_y}| ≤ {subset_size}) ---")
near_crack = np.where(np.abs(points_y - crack_line_y) <= subset_size)[0]
near_valid = np.sum(icgn_result.valid_mask[near_crack])
print(f"  크랙 근처 POI: {len(near_crack)}개")
print(f"  그 중 valid: {near_valid}개 ({near_valid/max(len(near_crack),1)*100:.1f}%)")

# 크랙 위/아래 구분
above_crack = near_crack[points_y[near_crack] < crack_line_y]
below_crack = near_crack[points_y[near_crack] > crack_line_y]
on_crack = near_crack[points_y[near_crack] == crack_line_y]

print(f"  크랙 위 (y<250): {len(above_crack)}개, valid={np.sum(icgn_result.valid_mask[above_crack])}")
print(f"  크랙 선 (y=250): {len(on_crack)}개, valid={np.sum(icgn_result.valid_mask[on_crack])}")
print(f"  크랙 아래 (y>250): {len(below_crack)}개, valid={np.sum(icgn_result.valid_mask[below_crack])}")

# 크랙 근처 valid POI의 오차
near_valid_idx = near_crack[icgn_result.valid_mask[near_crack]]
if len(near_valid_idx) > 0:
    err_u_near = icgn_result.disp_u[near_valid_idx] - gt_u_at_poi[near_valid_idx]
    err_v_near = icgn_result.disp_v[near_valid_idx] - gt_v_at_poi[near_valid_idx]
    err_mag_near = np.sqrt(err_u_near**2 + err_v_near**2)
    print(f"\n  크랙 근처 valid POI 오차:")
    print(f"    u: mean={np.mean(np.abs(err_u_near)):.4f}, max={np.max(np.abs(err_u_near)):.4f}")
    print(f"    v: mean={np.mean(np.abs(err_v_near)):.4f}, max={np.max(np.abs(err_v_near)):.4f}")
    print(f"    총: mean={np.mean(err_mag_near):.4f}, max={np.max(err_mag_near):.4f}")

# === Invalid POI 상세 ===
invalid_idx = np.where(~icgn_result.valid_mask)[0]
print(f"\n--- Invalid POI ({len(invalid_idx)}개) 위치 분포 ---")
if len(invalid_idx) > 0:
    inv_y = points_y[invalid_idx]
    inv_x = points_x[invalid_idx]
    print(f"  y 범위: [{inv_y.min()}, {inv_y.max()}]")
    print(f"  x 범위: [{inv_x.min()}, {inv_x.max()}]")
    # 크랙 라인(y=250)으로부터의 거리
    dist_from_crack = np.abs(inv_y - crack_line_y)
    print(f"  크랙 라인으로부터 거리: min={dist_from_crack.min()}, max={dist_from_crack.max()}, mean={dist_from_crack.mean():.1f}")

print(f"\n{'=' * 60}")
print("검증 완료")
print(f"{'=' * 60}")
