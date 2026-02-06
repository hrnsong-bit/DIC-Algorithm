"""
SSSIG Threshold 디버깅 스크립트
- DIC Challenge 이미지에서 subset size별 불량 POI 원인 분석
- 노이즈 과대추정 여부 확인
- x/y 분리 판정의 영향 확인
"""

import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import cv2
from speckle.core.sssig import (
    estimate_noise_variance,
    calculate_sssig_threshold,
    compute_sssig_map,
    _compute_sssig_map_parallel,
    predict_displacement_accuracy
)


def load_test_image(path: str) -> np.ndarray:
    """테스트 이미지 로드 (한글 경로 지원)"""
    img = cv2.imdecode(
        np.fromfile(path, dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE
    )
    if img is None:
        raise FileNotFoundError(f"이미지를 찾을 수 없습니다: {path}")
    print(f"이미지 로드: {path}")
    print(f"  크기: {img.shape}, dtype: {img.dtype}")
    print(f"  밝기 범위: [{img.min()}, {img.max()}], 평균: {img.mean():.1f}")
    return img

# ============================================================
# 테스트 1: 노이즈 분산 추정 비교
# ============================================================
def test_noise_estimation(img: np.ndarray):
    """여러 방법으로 노이즈 분산을 추정하고 비교"""
    print("\n" + "=" * 60)
    print("TEST 1: 노이즈 분산 추정 비교")
    print("=" * 60)

    # 방법 1: 현재 코드 (Laplacian)
    noise_laplacian = estimate_noise_variance(img, method='laplacian')

    # 방법 2: 로컬 분산 하위 퍼센타일
    img_float = img.astype(np.float64)
    kernel = 5
    local_mean = cv2.blur(img_float, (kernel, kernel))
    local_sq_mean = cv2.blur(img_float ** 2, (kernel, kernel))
    local_var = local_sq_mean - local_mean ** 2
    valid_var = local_var[local_var > 0]
    noise_local_p5 = float(np.percentile(valid_var, 5)) if len(valid_var) > 0 else 0.0
    noise_local_p10 = float(np.percentile(valid_var, 10)) if len(valid_var) > 0 else 0.0

    # 방법 3: 고정값 (일반적인 8-bit 카메라)
    noise_fixed = 4.0

    print(f"\n  Laplacian (현재 코드):    분산 = {noise_laplacian:.1f},  σ = {np.sqrt(noise_laplacian):.1f} GL")
    print(f"  로컬분산 하위 5%:         분산 = {noise_local_p5:.1f},  σ = {np.sqrt(noise_local_p5):.1f} GL")
    print(f"  로컬분산 하위 10%:        분산 = {noise_local_p10:.1f}, σ = {np.sqrt(noise_local_p10):.1f} GL")
    print(f"  고정값 (카메라 일반):     분산 = {noise_fixed:.1f},  σ = {np.sqrt(noise_fixed):.1f} GL")

    # 과대추정 경고
    if noise_laplacian > 100:
        print(f"\n  ⚠ Laplacian 분산이 {noise_laplacian:.0f}로 매우 높음!")
        print(f"    → 스페클 텍스처를 노이즈로 오인했을 가능성 높음")
        print(f"    → threshold가 비정상적으로 높아져 작은 subset에서 불량 다발")

    # 각 노이즈 추정에 따른 threshold 비교
    print(f"\n  [desired_accuracy=0.02 기준 threshold 비교]")
    for name, nv in [("Laplacian", noise_laplacian),
                      ("로컬 5%", noise_local_p5),
                      ("로컬 10%", noise_local_p10),
                      ("고정값", noise_fixed)]:
        th = calculate_sssig_threshold(nv, 0.02)
        print(f"    {name:20s}: threshold = {th:>12,.0f}")

    return {
        'laplacian': noise_laplacian,
        'local_p5': noise_local_p5,
        'local_p10': noise_local_p10,
        'fixed': noise_fixed
    }


# ============================================================
# 테스트 2: subset size별 SSSIG 분포
# ============================================================
def test_sssig_vs_subset_size(img: np.ndarray, noise_estimates: dict):
    """다양한 subset size에서 SSSIG 분포와 불량 POI 확인"""
    print("\n" + "=" * 60)
    print("TEST 2: Subset Size별 SSSIG 분포")
    print("=" * 60)

    noise_var = noise_estimates['laplacian']  # 현재 코드가 쓰는 값
    sizes = [11, 15, 21, 25, 31, 41]
    spacing = 16

    print(f"\n  노이즈 분산 (Laplacian): {noise_var:.1f}")
    print(f"  desired_accuracy: 0.02")
    print(f"  spacing: {spacing}")

    print(f"\n  {'size':>6} {'threshold':>12} {'min':>12} {'mean':>12} {'max':>12} "
          f"{'bad/total':>12} {'min/th':>8} {'픽셀수':>8}")
    print(f"  {'-'*82}")

    for size in sizes:
        result = compute_sssig_map(
            img, subset_size=size, spacing=spacing,
            noise_variance=noise_var, desired_accuracy=0.02
        )
        ratio = result.min / result.threshold if result.threshold > 0 else 0
        n_pixels = size * size
        print(f"  {size:>6} {result.threshold:>12,.0f} {result.min:>12,.0f} "
              f"{result.mean:>12,.0f} {result.max:>12,.0f} "
              f"{result.n_bad_points:>5}/{result.n_points:<5} "
              f"{ratio:>8.2f} {n_pixels:>8}")

    # SSSIG가 면적에 비례하는지 확인
    print(f"\n  [면적 비례 확인] (size=31 대비)")
    ref_result = compute_sssig_map(
        img, subset_size=31, spacing=spacing,
        noise_variance=noise_var, desired_accuracy=0.02
    )
    ref_mean = ref_result.mean
    ref_pixels = 31 * 31

    for size in sizes:
        result = compute_sssig_map(
            img, subset_size=size, spacing=spacing,
            noise_variance=noise_var, desired_accuracy=0.02
        )
        expected_ratio = (size * size) / ref_pixels
        actual_ratio = result.mean / ref_mean if ref_mean > 0 else 0
        print(f"    size={size:>3}: 면적비={expected_ratio:.3f}, "
              f"실제 SSSIG비={actual_ratio:.3f}, "
              f"차이={abs(actual_ratio - expected_ratio):.3f}")


# ============================================================
# 테스트 3: 노이즈 분산을 바꿔서 영향 확인
# ============================================================
def test_noise_sensitivity(img: np.ndarray, noise_estimates: dict):
    """노이즈 추정값을 바꾸면 결과가 어떻게 달라지는지"""
    print("\n" + "=" * 60)
    print("TEST 3: 노이즈 분산 민감도 분석")
    print("=" * 60)

    sizes = [11, 21, 31]
    spacing = 16
    test_noises = [
        ("Laplacian (현재)", noise_estimates['laplacian']),
        ("로컬 5%", noise_estimates['local_p5']),
        ("고정 σ=2 (분산=4)", 4.0),
        ("고정 σ=5 (분산=25)", 25.0),
        ("고정 σ=10 (분산=100)", 100.0),
    ]

    for noise_name, noise_var in test_noises:
        print(f"\n  --- {noise_name}: 분산={noise_var:.1f}, σ={np.sqrt(noise_var):.1f} ---")
        threshold = calculate_sssig_threshold(noise_var, 0.02)
        print(f"  threshold = {threshold:,.0f}")

        for size in sizes:
            result = compute_sssig_map(
                img, subset_size=size, spacing=spacing,
                noise_variance=noise_var, desired_accuracy=0.02
            )
            print(f"    subset={size:>3}: bad={result.n_bad_points:>4}/{result.n_points:<4} "
                  f"min={result.min:>12,.0f}  "
                  f"min/th={result.min / threshold:.2f}")


# ============================================================
# 테스트 4: x/y 분리 판정 vs 총합 판정
# ============================================================
def test_xy_split_vs_total(img: np.ndarray, noise_estimates: dict):
    """x/y 분리 판정이 얼마나 더 엄격한지 확인"""
    print("\n" + "=" * 60)
    print("TEST 4: x/y 분리 판정 vs 총합 판정")
    print("=" * 60)

    noise_var = noise_estimates['laplacian']
    sizes = [11, 15, 21, 31]
    spacing = 16

    img_float = img.astype(np.float64)
    gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)

    print(f"\n  노이즈 분산: {noise_var:.1f}")

    for size in sizes:
        threshold = calculate_sssig_threshold(noise_var, 0.02)
        half = size // 2
        h, w = img.shape
        margin = half + 1

        y_coords = np.arange(margin, h - margin, spacing)
        x_coords = np.arange(margin, w - margin, spacing)
        yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
        points_y = yy.ravel().astype(np.int64)
        points_x = xx.ravel().astype(np.int64)

        sssig_x, sssig_y = _compute_sssig_map_parallel(
            gx.astype(np.float64), gy.astype(np.float64),
            points_y, points_x, half
        )

        # 현재 코드: x/y 분리 (OR 조건)
        bad_split = np.sum((sssig_x < threshold / 2) | (sssig_y < threshold / 2))

        # 대안 1: 총합 판정
        bad_total = np.sum((sssig_x + sssig_y) < threshold)

        # 대안 2: x/y 분리 AND 조건
        bad_and = np.sum((sssig_x < threshold / 2) & (sssig_y < threshold / 2))

        # x/y 비등방성 확인
        ratio = sssig_x / (sssig_y + 1e-10)
        anisotropy = np.std(ratio) / np.mean(ratio) if np.mean(ratio) > 0 else 0

        print(f"\n  subset={size}: (총 {len(points_y)} POI, threshold={threshold:,.0f})")
        print(f"    현재 (x|y 분리, OR):  불량 {bad_split:>5}")
        print(f"    대안1 (총합):          불량 {bad_total:>5}")
        print(f"    대안2 (x&y 분리, AND): 불량 {bad_and:>5}")
        print(f"    차이 (현재 - 총합):    {bad_split - bad_total:>+5}개 더 엄격")
        print(f"    x/y 비등방성 (CV):     {anisotropy:.3f}")


# ============================================================
# 테스트 5: 최종 진단
# ============================================================
def diagnose(img: np.ndarray, noise_estimates: dict):
    """종합 진단"""
    print("\n" + "=" * 60)
    print("종합 진단")
    print("=" * 60)

    noise_lap = noise_estimates['laplacian']
    noise_local = noise_estimates['local_p5']

    # 진단 1: 노이즈 과대추정
    overestimate_ratio = noise_lap / max(noise_local, 1.0)
    print(f"\n  1. 노이즈 과대추정 비율: {overestimate_ratio:.1f}x")
    if overestimate_ratio > 5:
        print(f"     → ⚠ Laplacian이 실제 노이즈의 {overestimate_ratio:.0f}배를 추정")
        print(f"     → threshold가 {overestimate_ratio:.0f}배 높아짐")
        print(f"     → 해결: estimate_noise_variance()를 로컬분산 방식으로 교체")
    elif overestimate_ratio > 2:
        print(f"     → △ 약간의 과대추정. 주의 필요")
    else:
        print(f"     → ✓ 노이즈 추정 양호")

    # 진단 2: 작은 subset에서의 불량 원인
    result_small = compute_sssig_map(img, subset_size=11, spacing=16,
                                      noise_variance=noise_lap, desired_accuracy=0.02)
    result_large = compute_sssig_map(img, subset_size=31, spacing=16,
                                      noise_variance=noise_lap, desired_accuracy=0.02)

    if result_small.n_bad_points > 0 and result_large.n_bad_points == 0:
        # 노이즈를 줄여서 해결되는지
        result_fix = compute_sssig_map(img, subset_size=11, spacing=16,
                                        noise_variance=noise_local, desired_accuracy=0.02)
        if result_fix.n_bad_points == 0:
            print(f"\n  2. 원인: 노이즈 과대추정이 주 원인")
            print(f"     → subset=11에서 Laplacian 노이즈: 불량 {result_small.n_bad_points}개")
            print(f"     → subset=11에서 로컬 노이즈:      불량 {result_fix.n_bad_points}개 ✓")
            print(f"     → 노이즈 추정만 고치면 해결됨")
        else:
            print(f"\n  2. 원인: 노이즈 추정 + 스페클 품질 복합")
            print(f"     → 노이즈 수정 후에도 불량 {result_fix.n_bad_points}개 남음")
            print(f"     → x/y 분리 판정도 확인 필요")
    else:
        print(f"\n  2. subset=11 불량: {result_small.n_bad_points}, subset=31 불량: {result_large.n_bad_points}")


# ============================================================
# 메인
# ============================================================
if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser(description="SSSIG Threshold 디버깅")
    parser.add_argument("path", help="테스트 이미지 파일 또는 폴더 경로")
    args = parser.parse_args()

    target = Path(args.path)

    # 폴더면 첫 번째 이미지 자동 선택
    if target.is_dir():
        extensions = ['*.tif', '*.tiff', '*.png', '*.bmp', '*.jpg']
        files = []
        for ext in extensions:
            files.extend(sorted(target.glob(ext)))
        
        if not files:
            print(f"오류: {target} 에 이미지 파일이 없습니다.")
            sys.exit(1)
        
        print(f"폴더 내 이미지 {len(files)}개 발견:")
        for f in files[:10]:  # 최대 10개만 표시
            print(f"  {f.name}")
        if len(files) > 10:
            print(f"  ... 외 {len(files) - 10}개")
        
        # 첫 번째 파일 (보통 reference)
        image_path = str(files[0])
        print(f"\n첫 번째 이미지로 분석: {files[0].name}")
    else:
        image_path = str(target)

    img = load_test_image(image_path)

    noise_estimates = test_noise_estimation(img)
    test_sssig_vs_subset_size(img, noise_estimates)
    test_noise_sensitivity(img, noise_estimates)
    test_xy_split_vs_total(img, noise_estimates)
    diagnose(img, noise_estimates)

    print("\n" + "=" * 60)
    print("완료. 위 결과를 바탕으로 sssig.py 수정 방향을 결정하세요.")
    print("=" * 60)