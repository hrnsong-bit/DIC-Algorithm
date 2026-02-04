"""
IC-GN 테스트 (Affine vs Quadratic)
"""

import numpy as np
import cv2
import sys
from pathlib import Path

# 프로젝트 루트를 path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from speckle.core.initial_guess import compute_fft_cc
from speckle.core.optimization import compute_icgn


def create_speckle_pattern(size=300, speckle_size=3):
    """
    실제 DIC에 가까운 스페클 패턴 생성
    """
    np.random.seed(42)
    
    img = np.random.randn(size, size)
    img = cv2.GaussianBlur(img, (0, 0), speckle_size)
    
    img = img - img.min()
    img = (img / img.max() * 200 + 30).astype(np.uint8)
    
    return img


def apply_translation(image, dx, dy):
    """순수 평행이동 (Affine에 적합)"""
    M = np.float32([
        [1, 0, dx],
        [0, 1, dy]
    ])
    return cv2.warpAffine(
        image, M, (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT
    )


def apply_quadratic_deformation(image, u0=2.0, v0=1.5, uxx=0.0001, vyy=0.0001):
    """
    2차 변형 적용 (Quadratic에 적합)
    
    변위 정의:
      u(x,y) = u0 + 0.5*uxx*(x-cx)²
      v(x,y) = v0 + 0.5*vyy*(y-cy)²
    """
    h, w = image.shape[:2]
    cy, cx = h // 2, w // 2
    
    y_out, x_out = np.mgrid[0:h, 0:w].astype(np.float32)
    
    xr = x_out - cx
    yr = y_out - cy
    
    u = u0 + 0.5 * uxx * xr**2
    v = v0 + 0.5 * vyy * yr**2
    
    map_x = x_out - u
    map_y = y_out - v
    
    return cv2.remap(
        image, map_x, map_y,
        interpolation=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT
    )


# ===== 테스트 1: Affine (순수 평행이동) =====

def test_affine_translation():
    """Affine Shape Function 테스트 - 순수 평행이동"""
    print("=" * 60)
    print("TEST 1: Affine Shape Function (순수 평행이동)")
    print("=" * 60)
    
    true_u, true_v = 2.5, 1.3
    
    ref = create_speckle_pattern(size=300, speckle_size=3)
    deformed = apply_translation(ref, true_u, true_v)
    
    print(f"\n[설정]")
    print(f"  실제 변위: u={true_u}, v={true_v}")
    print(f"  이미지 크기: {ref.shape}")
    
    print(f"\n[1] FFTCC...")
    fftcc = compute_fft_cc(
        ref, deformed,
        subset_size=21,
        spacing=27,
        search_range=10,
        zncc_threshold=0.5
    )
    print(f"  POI: {fftcc.n_valid}/{fftcc.n_points}")
    
    print(f"\n[2] IC-GN (Affine)...")
    result = compute_icgn(
        ref, deformed,
        initial_guess=fftcc,
        subset_size=21,
        max_iterations=50,
        convergence_threshold=0.001,
        interpolation_order=5,
        shape_function='affine'
    )
    
    valid = result.valid_mask
    if np.any(valid):
        mean_u = np.mean(result.disp_u[valid])
        mean_v = np.mean(result.disp_v[valid])
        std_u = np.std(result.disp_u[valid])
        std_v = np.std(result.disp_v[valid])
        
        err_u = abs(mean_u - true_u)
        err_v = abs(mean_v - true_v)
        
        print(f"\n[결과]")
        print(f"  수렴: {result.n_converged}/{result.n_points} ({result.convergence_rate*100:.1f}%)")
        print(f"  평균 반복: {result.mean_iterations:.1f}")
        print(f"  평균 ZNCC: {result.mean_zncc:.4f}")
        print(f"\n  측정 변위: u={mean_u:.4f} (±{std_u:.4f}), v={mean_v:.4f} (±{std_v:.4f})")
        print(f"  오차: Δu={err_u:.4f}, Δv={err_v:.4f}")
        
        passed = err_u < 0.05 and err_v < 0.05
        print(f"\n  {'✅ PASS' if passed else '❌ FAIL'} (기준: 오차 < 0.05 px)")
        return passed
    else:
        print("\n  ❌ 유효한 결과 없음")
        return False

# ===== 테스트 2: Quadratic (2차 변형) =====

def test_quadratic_deformation():
    """Quadratic Shape Function 테스트 - 2차 변형"""
    print("\n" + "=" * 60)
    print("TEST 2: Quadratic Shape Function (2차 변형)")
    print("=" * 60)
    
    true_u0 = 2.0
    true_v0 = 1.5
    true_uxx = 0.001  # ← 0.0002 → 0.001
    true_vyy = 0.001  # ← 0.0002 → 0.001
    
    subset_size = 31  # ← 21 → 31
    image_size = 400  # ← 300 → 400
    
    ref = create_speckle_pattern(size=image_size, speckle_size=3)
    deformed = apply_quadratic_deformation(ref, true_u0, true_v0, true_uxx, true_vyy)
    
    half = subset_size // 2
    edge_disp = 0.5 * true_uxx * half**2
    
    print(f"\n[설정]")
    print(f"  u0={true_u0}, v0={true_v0}")
    print(f"  uxx={true_uxx}, vyy={true_vyy}")
    print(f"  subset_size={subset_size}, edge_disp={edge_disp:.4f} px")
    
    print(f"\n[1] FFTCC...")
    fftcc = compute_fft_cc(
        ref, deformed,
        subset_size=subset_size,
        spacing=50,
        search_range=15,
        zncc_threshold=0.5
    )
    print(f"  POI: {fftcc.n_valid}/{fftcc.n_points}")
    
    print(f"\n[2] IC-GN (Affine)...")
    result_affine = compute_icgn(
        ref, deformed,
        initial_guess=fftcc,
        subset_size=subset_size,
        max_iterations=50,
        convergence_threshold=0.001,
        shape_function='affine'
    )
    
    print(f"\n[3] IC-GN (Quadratic)...")
    result_quad = compute_icgn(
        ref, deformed,
        initial_guess=fftcc,
        subset_size=subset_size,
        max_iterations=50,
        convergence_threshold=0.001,
        shape_function='quadratic'
    )
    
    print(f"\n[비교 결과]")
    print(f"{'':20} {'Affine':>12} {'Quadratic':>12}")
    print(f"{'-'*46}")
    print(f"  수렴율:      {result_affine.convergence_rate*100:>10.1f}% {result_quad.convergence_rate*100:>10.1f}%")
    
    valid = result_quad.valid_mask
    if np.any(valid):
        mean_uxx = np.mean(result_quad.disp_uxx[valid])
        mean_vyy = np.mean(result_quad.disp_vyy[valid])
        
        uxx_err = abs(mean_uxx - true_uxx) / true_uxx * 100
        vyy_err = abs(mean_vyy - true_vyy) / true_vyy * 100
        
        print(f"\n[Quadratic 상세]")
        print(f"  mean_u: {np.mean(result_quad.disp_u[valid]):.4f}")
        print(f"  mean_v: {np.mean(result_quad.disp_v[valid]):.4f}")
        print(f"  mean_zncc: {result_quad.mean_zncc:.4f}")
        print(f"  mean_uxx: {mean_uxx:.6f} (실제: {true_uxx}, 오차: {uxx_err:.1f}%)")
        print(f"  mean_vyy: {mean_vyy:.6f} (실제: {true_vyy}, 오차: {vyy_err:.1f}%)")
        
        print(f"\n  ZNCC 비교: Affine={result_affine.mean_zncc:.4f}, Quadratic={result_quad.mean_zncc:.4f}")
        
        passed = uxx_err < 15 and vyy_err < 15
        print(f"\n  {'✅ PASS' if passed else '❌ FAIL'} (기준: 2차항 오차 < 15%)")
        return passed
    
    return False


# ===== 테스트 3: Quadratic 위치별 검증 (수정) =====

def test_quadratic_detailed():
    """Quadratic 상세 검증 - 위치별 이론값 비교"""
    print("\n" + "=" * 60)
    print("TEST 3: Quadratic 위치별 검증")
    print("=" * 60)
    
    true_u0 = 2.0
    true_v0 = 1.5
    true_uxx = 0.0005  # ← 0.001 → 0.0005 (변형 줄임)
    true_vyy = 0.0005
    
    subset_size = 31  # ← 31 → 41 (더 큰 subset)
    image_size = 400
    
    ref = create_speckle_pattern(size=image_size, speckle_size=3)
    deformed = apply_quadratic_deformation(ref, true_u0, true_v0, true_uxx, true_vyy)
    
    h, w = ref.shape
    cx, cy = w // 2, h // 2
    
    half = subset_size // 2
    edge_disp = 0.5 * true_uxx * half**2
    
    print(f"\n[설정]")
    print(f"  이미지 중심: ({cx}, {cy})")
    print(f"  u(x,y) = {true_u0} + 0.5*{true_uxx}*(x-{cx})²")
    print(f"  v(x,y) = {true_v0} + 0.5*{true_vyy}*(y-{cy})²")
    print(f"  subset_size={subset_size}, edge_disp={edge_disp:.4f} px")
    
    fftcc = compute_fft_cc(
        ref, deformed,
        subset_size=subset_size,
        spacing=60,
        search_range=20,  # ← 15 → 20
        zncc_threshold=0.5
    )
    
    result = compute_icgn(
        ref, deformed,
        initial_guess=fftcc,
        subset_size=subset_size,
        max_iterations=50,
        convergence_threshold=0.001,
        shape_function='quadratic'
    )
    
    print(f"\n[POI별 비교] (중심 근처 5개)")
    print(f"{'POI':>4} {'위치':>12} {'이론 u':>10} {'측정 u':>10} {'이론 v':>10} {'측정 v':>10}")
    print("-" * 70)
    
    valid = result.valid_mask
    count = 0
    errors_u = []
    errors_v = []
    
    # 중심 근처 POI만 분석 (경계 효과 제거)
    for i in range(result.n_points):
        if not valid[i]:
            continue
        
        px, py = result.points_x[i], result.points_y[i]
        
        # 중심에서 100px 이내만
        dist_from_center = np.sqrt((px - cx)**2 + (py - cy)**2)
        if dist_from_center > 100:
            continue
        
        xr = px - cx
        yr = py - cy
        theory_u = true_u0 + 0.5 * true_uxx * xr**2
        theory_v = true_v0 + 0.5 * true_vyy * yr**2
        
        meas_u = result.disp_u[i]
        meas_v = result.disp_v[i]
        
        errors_u.append(abs(meas_u - theory_u))
        errors_v.append(abs(meas_v - theory_v))
        
        if count < 5:
            print(f"{i:>4} ({px:>3},{py:>3}) {theory_u:>10.4f} {meas_u:>10.4f} "
                  f"{theory_v:>10.4f} {meas_v:>10.4f}")
        count += 1
    
    if len(errors_u) == 0:
        print("\n  ❌ 유효한 중심 POI 없음")
        return False
    
    print(f"\n[오차 통계] (중심 100px 이내, {len(errors_u)}개 POI)")
    print(f"  u 오차: mean={np.mean(errors_u):.4f}, max={np.max(errors_u):.4f}")
    print(f"  v 오차: mean={np.mean(errors_v):.4f}, max={np.max(errors_v):.4f}")
    
    passed = np.mean(errors_u) < 0.1 and np.mean(errors_v) < 0.1
    print(f"\n  {'✅ PASS' if passed else '❌ FAIL'} (기준: 평균 오차 < 0.1 px)")
    
    return passed

# ===== 메인 =====

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("IC-GN Shape Function 테스트")
    print("=" * 60)
    
    results = []
    
    results.append(("Affine Translation", test_affine_translation()))
    results.append(("Quadratic Deformation", test_quadratic_deformation()))
    results.append(("Quadratic Detailed", test_quadratic_detailed()))
    
    # 요약
    print("\n" + "=" * 60)
    print("테스트 요약")
    print("=" * 60)
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {name}: {status}")
