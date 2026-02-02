"""FFT-CC 테스트"""

import numpy as np
import cv2
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from speckle.core.initial_guess import (
    compute_fft_cc, 
    validate_displacement_field,
    warmup_fft_cc
)


def generate_test_pair(size: tuple = (512, 512),
                       disp_u: int = 10,
                       disp_v: int = 5) -> tuple:
    """테스트용 이미지 쌍 생성 (균일 변위)"""
    # 스페클 패턴 생성
    ref = np.ones(size, dtype=np.uint8) * 128
    n_speckles = int(size[0] * size[1] * 0.3 / 25)
    
    for _ in range(n_speckles):
        x = np.random.randint(0, size[1])
        y = np.random.randint(0, size[0])
        r = np.random.randint(2, 5)
        c = np.random.choice([0, 255])
        cv2.circle(ref, (x, y), r, int(c), -1)
    
    # 노이즈 추가
    noise = np.random.normal(0, 3, size)
    ref = np.clip(ref.astype(np.float64) + noise, 0, 255).astype(np.uint8)
    
    # 변형 이미지 (평행 이동)
    M = np.float32([[1, 0, disp_u], [0, 1, disp_v]])
    defm = cv2.warpAffine(ref, M, (size[1], size[0]), borderValue=128)
    
    return ref, defm, disp_u, disp_v


def test_uniform_displacement():
    """균일 변위 테스트"""
    print("=" * 50)
    print("테스트: 균일 변위")
    print("=" * 50)
    
    # 워밍업
    warmup_fft_cc()
    
    # 테스트 이미지 생성
    true_u, true_v = 15, -8
    ref, defm, _, _ = generate_test_pair(disp_u=true_u, disp_v=true_v)
    
    print(f"실제 변위: u={true_u}, v={true_v}")
    
    # FFT-CC 실행
    result = compute_fft_cc(
        ref, defm,
        subset_size=21,
        spacing=20,
        search_range=30,
        zncc_threshold=0.6
    )
    
    print(f"\n결과:")
    print(f"  POI 수: {result.n_points}")
    print(f"  유효: {result.n_valid} ({result.valid_ratio*100:.1f}%)")
    print(f"  평균 ZNCC: {result.mean_zncc:.4f}")
    print(f"  처리 시간: {result.processing_time:.3f}s")
    
    # 변위 통계
    valid_u = result.disp_u[result.valid_mask]
    valid_v = result.disp_v[result.valid_mask]
    
    print(f"\n변위 통계:")
    print(f"  u: mean={np.mean(valid_u):.2f}, std={np.std(valid_u):.2f}")
    print(f"  v: mean={np.mean(valid_v):.2f}, std={np.std(valid_v):.2f}")
    
    # 오차 확인
    u_error = abs(np.mean(valid_u) - true_u)
    v_error = abs(np.mean(valid_v) - true_v)
    print(f"\n오차: u={u_error:.2f}px, v={v_error:.2f}px")
    
    assert u_error < 1.0, f"u 오차 과대: {u_error}"
    assert v_error < 1.0, f"v 오차 과대: {v_error}"
    print("\n✓ 테스트 통과!")
    
    # 검증
    validation = validate_displacement_field(result)
    print(f"\n검증 결과:")
    print(f"  유효: {validation.is_valid}")
    print(f"  권장 조치: {validation.suggested_action}")
    print(f"  불연속 비율: {validation.discontinuity_ratio*100:.1f}%")


def test_with_roi():
    """ROI 테스트"""
    print("\n" + "=" * 50)
    print("테스트: ROI 적용")
    print("=" * 50)
    
    ref, defm, true_u, true_v = generate_test_pair(
        size=(800, 800), disp_u=20, disp_v=10
    )
    
    roi = (100, 100, 400, 400)
    print(f"ROI: {roi}")
    print(f"실제 변위: u={true_u}, v={true_v}")
    
    result = compute_fft_cc(
        ref, defm,
        subset_size=21,
        spacing=15,
        search_range=40,
        roi=roi
    )
    
    print(f"\n결과:")
    print(f"  POI 수: {result.n_points}")
    print(f"  유효 비율: {result.valid_ratio*100:.1f}%")
    print(f"  처리 시간: {result.processing_time:.3f}s")
    
    valid_u = result.disp_u[result.valid_mask]
    valid_v = result.disp_v[result.valid_mask]
    
    print(f"\n변위: u={np.mean(valid_u):.2f}, v={np.mean(valid_v):.2f}")
    print("✓ ROI 테스트 완료!")


def test_large_image():
    """대용량 이미지 테스트 (메모리 최적화 확인)"""
    print("\n" + "=" * 50)
    print("테스트: 대용량 이미지 (2048x2048)")
    print("=" * 50)
    
    ref, defm, true_u, true_v = generate_test_pair(
        size=(2048, 2048), disp_u=25, disp_v=15
    )
    
    print(f"이미지 크기: {ref.shape}")
    print(f"실제 변위: u={true_u}, v={true_v}")
    
    def progress(current, total):
        if current % 1000 == 0 or current == total:
            print(f"  진행: {current}/{total} ({100*current/total:.1f}%)")
    
    result = compute_fft_cc(
        ref, defm,
        subset_size=21,
        spacing=10,
        search_range=50,
        chunk_size=500,
        progress_callback=progress
    )
    
    print(f"\n결과:")
    print(f"  POI 수: {result.n_points}")
    print(f"  유효 비율: {result.valid_ratio*100:.1f}%")
    print(f"  처리 시간: {result.processing_time:.2f}s")
    
    valid_u = result.disp_u[result.valid_mask]
    print(f"  u 평균: {np.mean(valid_u):.2f} (실제: {true_u})")
    
    print("✓ 대용량 테스트 완료!")


if __name__ == "__main__":
    test_uniform_displacement()
    test_with_roi()
    test_large_image()
    
    print("\n" + "=" * 50)
    print("모든 테스트 완료!")
    print("=" * 50)
