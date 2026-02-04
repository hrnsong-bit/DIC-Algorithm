"""
IC-GN 테스트
"""

import numpy as np
import cv2
import sys
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from speckle.core.initial_guess import compute_fft_cc
from speckle.core.optimization import compute_icgn


def create_test_images(size=200, shift_x=2.5, shift_y=1.3):
    """테스트 이미지 생성 (서브픽셀 이동)"""
    np.random.seed(42)
    ref = np.random.randint(50, 200, (size, size), dtype=np.uint8)
    ref = cv2.GaussianBlur(ref, (5, 5), 1.5)
    
    M = np.float32([
        [1, 0, shift_x],
        [0, 1, shift_y]
    ])
    
    deformed = cv2.warpAffine(
        ref, M, (size, size),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REFLECT
    )
    
    return ref, deformed, shift_x, shift_y


def test_icgn():
    """IC-GN 기본 테스트"""
    print("=" * 50)
    print("IC-GN 테스트")
    print("=" * 50)
    
    true_u, true_v = 2.5, 1.3
    ref, deformed, _, _ = create_test_images(shift_x=true_u, shift_y=true_v)
    
    print(f"\n실제 변위: u={true_u}, v={true_v}")
    
    print("\n[1] FFTCC 실행 중...")
    fftcc_result = compute_fft_cc(
        ref, deformed,
        subset_size=21,
        spacing=20,
        search_range=10,
        zncc_threshold=0.5
    )
    
    print(f"  - POI 수: {fftcc_result.n_points}")
    print(f"  - 유효 POI: {fftcc_result.n_valid}")
    print(f"  - FFTCC 변위 (정수): u={np.mean(fftcc_result.disp_u):.1f}, v={np.mean(fftcc_result.disp_v):.1f}")
    
    print("\n[2] IC-GN 실행 중...")
    icgn_result = compute_icgn(
        ref, deformed,
        initial_guess=fftcc_result,
        subset_size=21,
        max_iterations=50,
        convergence_threshold=0.0001,
        interpolation_order=5
    )
    
    print(f"  - 수렴 POI: {icgn_result.n_converged}/{icgn_result.n_points}")
    print(f"  - 평균 반복: {icgn_result.mean_iterations:.1f}")
    print(f"  - 처리 시간: {icgn_result.processing_time:.2f}초")
    
    valid = icgn_result.valid_mask
    if np.any(valid):
        mean_u = np.mean(icgn_result.disp_u[valid])
        mean_v = np.mean(icgn_result.disp_v[valid])
        mean_zncc = np.mean(icgn_result.zncc_values[valid])
        
        error_u = abs(mean_u - true_u)
        error_v = abs(mean_v - true_v)
        
        print(f"\n[결과]")
        print(f"  - IC-GN 변위: u={mean_u:.4f}, v={mean_v:.4f}")
        print(f"  - 오차: Δu={error_u:.4f}, Δv={error_v:.4f}")
        print(f"  - 평균 ZNCC: {mean_zncc:.4f}")
        
        if error_u < 0.1 and error_v < 0.1:
            print("\n✅ 테스트 통과! (오차 < 0.1 px)")
            return True
        else:
            print("\n⚠️ 오차가 큽니다.")
            return False
    else:
        print("\n❌ 유효한 결과 없음")
        return False


def test_icgn_debug():
    """IC-GN 디버그 테스트 - 단일 POI 상세 분석"""
    print("=" * 50)
    print("IC-GN 디버그 테스트")
    print("=" * 50)
    
    true_u, true_v = 2.5, 1.3
    ref, deformed, _, _ = create_test_images(shift_x=true_u, shift_y=true_v)
    
    print(f"\n실제 변위: u={true_u}, v={true_v}")
    
    # FFTCC
    fftcc_result = compute_fft_cc(
        ref, deformed,
        subset_size=21,
        spacing=50,  # POI 줄이기
        search_range=10,
        zncc_threshold=0.5
    )
    
    print(f"\n[FFTCC 결과]")
    for i in range(min(3, fftcc_result.n_points)):
        print(f"  POI {i}: u={fftcc_result.disp_u[i]}, v={fftcc_result.disp_v[i]}, zncc={fftcc_result.zncc_values[i]:.4f}")
    
    # IC-GN 직접 호출 (디버그)
    from speckle.core.optimization.icgn import (
        _to_gray, _compute_gradient, _extract_reference_subset,
        _icgn_iterate
    )
    from speckle.core.optimization.interpolation import create_interpolator
    from speckle.core.optimization.shape_function import (
        generate_local_coordinates, compute_steepest_descent_affine, compute_hessian
    )
    
    ref_gray = _to_gray(ref).astype(np.float64)
    def_gray = _to_gray(deformed).astype(np.float64)
    grad_x, grad_y = _compute_gradient(ref_gray)
    target_interp = create_interpolator(def_gray, order=5)
    xsi, eta = generate_local_coordinates(21)
    
    # 첫 번째 유효 POI
    idx = 0
    px, py = int(fftcc_result.points_x[idx]), int(fftcc_result.points_y[idx])
    
    print(f"\n[POI {idx} 분석] 위치: ({px}, {py})")
    
    ref_data = _extract_reference_subset(ref_gray, grad_x, grad_y, px, py, 21)
    if ref_data is None:
        print("  Reference subset 추출 실패!")
        return
    
    f, dfdx, dfdy, f_mean, f_tilde = ref_data
    print(f"  f_mean={f_mean:.2f}, f_tilde={f_tilde:.2f}")
    
    J = compute_steepest_descent_affine(dfdx, dfdy, xsi, eta)
    H = compute_hessian(J)
    H_inv = np.linalg.inv(H)
    
    print(f"  Hessian 조건수: {np.linalg.cond(H):.2f}")
    
    p_init = np.array([
        float(fftcc_result.disp_u[idx]),
        0.0, 0.0,
        float(fftcc_result.disp_v[idx]),
        0.0, 0.0
    ])
    
    print(f"\n  초기값: u={p_init[0]:.1f}, v={p_init[3]:.1f}")
    print(f"  목표값: u={true_u}, v={true_v}")
    print(f"\n  [IC-GN 반복]")
    
    p_final, zncc, n_iter, conv = _icgn_iterate(
        f, f_mean, f_tilde,
        J, H_inv,
        target_interp,
        px, py, xsi, eta,
        p_init.copy(),
        max_iterations=20,
        convergence_threshold=0.001,
        debug=True  # 디버그 활성화
    )
    
    print(f"\n  [최종 결과]")
    print(f"  변위: u={p_final[0]:.4f}, v={p_final[3]:.4f}")
    print(f"  오차: Δu={abs(p_final[0]-true_u):.4f}, Δv={abs(p_final[3]-true_v):.4f}")
    print(f"  수렴: {conv}, 반복: {n_iter}, ZNCC: {zncc:.4f}")
    print(f"  grad_x 범위: [{np.min(grad_x):.2f}, {np.max(grad_x):.2f}]")
    print(f"  grad_y 범위: [{np.min(grad_y):.2f}, {np.max(grad_y):.2f}]")
    print(f"  J 범위: [{np.min(J):.4f}, {np.max(J):.4f}]")
    print(f"  H_inv 범위: [{np.min(H_inv):.6f}, {np.max(H_inv):.6f}]")


if __name__ == "__main__":
    test_icgn_debug()  # 디버그 테스트 실행
