#!/usr/bin/env python3
"""
compute_icgn 통합 테스트

use_numba=True와 use_numba=False의 결과가 동일한지 E2E 검증.
기존 API 호환성 및 GUI 호출 경로 검증 포함.
"""

import sys
import time
import numpy as np
import cv2

sys.path.insert(0, '.')

from speckle.core.optimization import compute_icgn
from speckle.core.optimization.results import ICGN_SUCCESS
from speckle.core.initial_guess.results import FFTCCResult


# =============================================================================
#  테스트 인프라
# =============================================================================

class TestRunner:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def check(self, name, condition, detail=""):
        if condition:
            self.passed += 1
            print(f"  PASS: {name}")
        else:
            self.failed += 1
            self.errors.append(name)
            print(f"  FAIL: {name}  {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("Failed tests:")
            for e in self.errors:
                print(f"  - {e}")
        print(f"{'='*60}")
        return self.failed == 0

runner = TestRunner()


# =============================================================================
#  테스트 데이터 생성
# =============================================================================

def make_test_data(img_size=300, shift_x=0.35, shift_y=-0.22,
                   subset_size=21, spacing=20, seed=42):
    """테스트용 이미지 쌍 + FFTCCResult 생성"""
    np.random.seed(seed)
    ref = np.random.rand(img_size, img_size).astype(np.float64) * 200 + 20
    ref = cv2.GaussianBlur(ref, (15, 15), 3.0)

    M = np.float64([[1, 0, shift_x], [0, 1, shift_y]])
    deformed = cv2.warpAffine(ref, M, (img_size, img_size),
                               flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REFLECT)

    # POI 그리드 생성 (FFTCCResult 모사)
    half = subset_size // 2
    margin = half + 5
    xs = np.arange(margin, img_size - margin, spacing)
    ys = np.arange(margin, img_size - margin, spacing)
    grid_x, grid_y = np.meshgrid(xs, ys)
    points_x = grid_x.ravel().astype(np.int64)
    points_y = grid_y.ravel().astype(np.int64)
    n_points = len(points_x)

    # FFTCCResult 생성 (초기 변위 추정 = 실제 shift)
    fft_result = FFTCCResult(
        points_y=points_y,
        points_x=points_x,
        disp_u=np.full(n_points, shift_x, dtype=np.float64),
        disp_v=np.full(n_points, shift_y, dtype=np.float64),
        zncc_values=np.ones(n_points, dtype=np.float64),
        valid_mask=np.ones(n_points, dtype=bool),
    )

    return ref, deformed, fft_result


# =============================================================================
#  JIT 워밍업
# =============================================================================

print("=" * 60)
print("compute_icgn Integration Test Suite")
print("=" * 60)

print("\nWarming up JIT (first call may be slow)...")
ref_small, def_small, fft_small = make_test_data(img_size=100, spacing=30)
t0 = time.time()
_ = compute_icgn(ref_small, def_small, fft_small,
                  subset_size=11, max_iterations=5, use_numba=True)
print(f"Warmup: {time.time()-t0:.2f}s\n")


# =============================================================================
#  Test 1: use_numba=True vs use_numba=False 결과 비교
# =============================================================================

print("--- Test 1: Numba vs Original result equivalence ---")

ref, deformed, fft_result = make_test_data()
n_points = len(fft_result.points_x)

result_numba = compute_icgn(ref, deformed, fft_result,
                             subset_size=21, max_iterations=50,
                             interpolation_order=5,
                             shape_function='affine',
                             use_numba=True)

result_orig = compute_icgn(ref, deformed, fft_result,
                            subset_size=21, max_iterations=50,
                            interpolation_order=5,
                            shape_function='affine',
                            use_numba=False)

runner.check("1-1 n_points match",
             result_numba.n_points == result_orig.n_points,
             f"numba={result_numba.n_points}, orig={result_orig.n_points}")

runner.check("1-2 disp_u match",
             np.allclose(result_numba.disp_u, result_orig.disp_u, atol=1e-10),
             f"max_diff={np.max(np.abs(result_numba.disp_u - result_orig.disp_u)):.2e}")

runner.check("1-3 disp_v match",
             np.allclose(result_numba.disp_v, result_orig.disp_v, atol=1e-10),
             f"max_diff={np.max(np.abs(result_numba.disp_v - result_orig.disp_v)):.2e}")

runner.check("1-4 disp_ux match",
             np.allclose(result_numba.disp_ux, result_orig.disp_ux, atol=1e-10))

runner.check("1-5 disp_vy match",
             np.allclose(result_numba.disp_vy, result_orig.disp_vy, atol=1e-10))

runner.check("1-6 zncc_values match",
             np.allclose(result_numba.zncc_values, result_orig.zncc_values, atol=1e-10),
             f"max_diff={np.max(np.abs(result_numba.zncc_values - result_orig.zncc_values)):.2e}")

runner.check("1-7 iterations match",
             np.array_equal(result_numba.iterations, result_orig.iterations))

runner.check("1-8 converged match",
             np.array_equal(result_numba.converged, result_orig.converged))

runner.check("1-9 valid_mask match",
             np.array_equal(result_numba.valid_mask, result_orig.valid_mask))

runner.check("1-10 failure_reason match",
             np.array_equal(result_numba.failure_reason, result_orig.failure_reason))


# =============================================================================
#  Test 2: Quadratic shape function 비교
# =============================================================================

print("\n--- Test 2: Quadratic shape function ---")

result_q_numba = compute_icgn(ref, deformed, fft_result,
                                subset_size=21, shape_function='quadratic',
                                use_numba=True)

result_q_orig = compute_icgn(ref, deformed, fft_result,
                               subset_size=21, shape_function='quadratic',
                               use_numba=False)

runner.check("2-1 quadratic disp_u match",
             np.allclose(result_q_numba.disp_u, result_q_orig.disp_u, atol=1e-8),
             f"max_diff={np.max(np.abs(result_q_numba.disp_u - result_q_orig.disp_u)):.2e}")

runner.check("2-2 quadratic disp_v match",
             np.allclose(result_q_numba.disp_v, result_q_orig.disp_v, atol=1e-8))

runner.check("2-3 quadratic zncc match",
             np.allclose(result_q_numba.zncc_values, result_q_orig.zncc_values, atol=1e-8))

runner.check("2-4 quadratic converged match",
             np.array_equal(result_q_numba.converged, result_q_orig.converged))

runner.check("2-5 quadratic has 2nd order derivatives",
             result_q_numba.disp_uxx is not None and result_q_numba.disp_vyy is not None)


# =============================================================================
#  Test 3: Cubic interpolation (order=3)
# =============================================================================

print("\n--- Test 3: Cubic interpolation (order=3) ---")

result_c3_numba = compute_icgn(ref, deformed, fft_result,
                                 interpolation_order=3, use_numba=True)
result_c3_orig = compute_icgn(ref, deformed, fft_result,
                                interpolation_order=3, use_numba=False)

runner.check("3-1 cubic disp_u match",
             np.allclose(result_c3_numba.disp_u, result_c3_orig.disp_u, atol=1e-10))

runner.check("3-2 cubic zncc match",
             np.allclose(result_c3_numba.zncc_values, result_c3_orig.zncc_values, atol=1e-10))


# =============================================================================
#  Test 4: progress_callback 동작 확인
# =============================================================================

print("\n--- Test 4: progress_callback ---")

progress_calls = []
def cb(current, total):
    progress_calls.append((current, total))

result_cb = compute_icgn(ref, deformed, fft_result, use_numba=True,
                          progress_callback=cb)

runner.check("4-1 callback called at least twice (start + end)",
             len(progress_calls) >= 2,
             f"calls={len(progress_calls)}")

runner.check("4-2 first callback is (0, n)",
             progress_calls[0] == (0, n_points),
             f"first={progress_calls[0]}")

runner.check("4-3 last callback is (n, n)",
             progress_calls[-1] == (n_points, n_points),
             f"last={progress_calls[-1]}")


# =============================================================================
#  Test 5: Gaussian blur 옵션
# =============================================================================

print("\n--- Test 5: Gaussian blur ---")

result_blur_numba = compute_icgn(ref, deformed, fft_result,
                                   gaussian_blur=5, use_numba=True)
result_blur_orig = compute_icgn(ref, deformed, fft_result,
                                  gaussian_blur=5, use_numba=False)

runner.check("5-1 blur disp_u match",
             np.allclose(result_blur_numba.disp_u, result_blur_orig.disp_u, atol=1e-10))

runner.check("5-2 blur converged match",
             np.array_equal(result_blur_numba.converged, result_blur_orig.converged))


# =============================================================================
#  Test 6: 빈 POI 리스트
# =============================================================================

print("\n--- Test 6: Empty POI list ---")

empty_fft = FFTCCResult(
    points_y=np.array([], dtype=np.int64),
    points_x=np.array([], dtype=np.int64),
    disp_u=np.array([], dtype=np.float64),
    disp_v=np.array([], dtype=np.float64),
    zncc_values=np.array([], dtype=np.float64),
    valid_mask=np.array([], dtype=bool),
)

result_empty = compute_icgn(ref, deformed, empty_fft, use_numba=True)
runner.check("6-1 empty result n_points==0",
             result_empty.n_points == 0)


# =============================================================================
#  Test 7: API 하위 호환성 (use_numba 미지정)
# =============================================================================

print("\n--- Test 7: API backward compatibility ---")

# use_numba 미지정 시 기본값 True
result_default = compute_icgn(ref, deformed, fft_result, subset_size=21)
runner.check("7-1 default call works",
             result_default.n_points == n_points)

# 기존 파라미터만 사용
result_legacy = compute_icgn(ref, deformed, fft_result,
                              subset_size=21, max_iterations=50,
                              convergence_threshold=0.001,
                              zncc_threshold=0.6,
                              interpolation_order=5,
                              shape_function='affine',
                              gaussian_blur=None,
                              n_workers=None,
                              progress_callback=None)
runner.check("7-2 legacy call signature works",
             result_legacy.n_points == n_points)


# =============================================================================
#  Test 8: 성능 벤치마크
# =============================================================================

print("\n--- Test 8: Performance benchmark ---")

ref_perf, def_perf, fft_perf = make_test_data(img_size=500, spacing=10, seed=99)
n_perf = len(fft_perf.points_x)

# Original
t0 = time.time()
r_orig_perf = compute_icgn(ref_perf, def_perf, fft_perf, use_numba=False)
t_orig = time.time() - t0

# Numba
t0 = time.time()
r_numba_perf = compute_icgn(ref_perf, def_perf, fft_perf, use_numba=True)
t_numba = time.time() - t0

speedup = t_orig / t_numba if t_numba > 0 else float('inf')

print(f"\n  Performance ({n_perf} POIs):")
print(f"    Original:  {t_orig*1000:.0f} ms ({t_orig/n_perf*1000:.3f} ms/POI)")
print(f"    Numba:     {t_numba*1000:.0f} ms ({t_numba/n_perf*1000:.3f} ms/POI)")
print(f"    Speedup:   {speedup:.1f}x")

runner.check("8-1 Numba faster than original",
             t_numba < t_orig,
             f"numba={t_numba*1000:.0f}ms, orig={t_orig*1000:.0f}ms")

runner.check("8-2 results match in perf test",
             np.allclose(r_numba_perf.disp_u, r_orig_perf.disp_u, atol=1e-10))


# =============================================================================
#  Test 9: ICGNResult metadata 확인
# =============================================================================

print("\n--- Test 9: ICGNResult metadata ---")

runner.check("9-1 shape_function preserved",
             result_numba.shape_function == 'affine')
runner.check("9-2 subset_size preserved",
             result_numba.subset_size == 21)
runner.check("9-3 processing_time > 0",
             result_numba.processing_time > 0,
             f"time={result_numba.processing_time:.3f}s")
runner.check("9-4 convergence_rate sensible",
             0.0 <= result_numba.convergence_rate <= 1.0,
             f"rate={result_numba.convergence_rate:.2f}")
runner.check("9-5 mean_zncc sensible",
             result_numba.mean_zncc > 0.5,
             f"mean_zncc={result_numba.mean_zncc:.4f}")


# =============================================================================
#  요약
# =============================================================================

print()
success = runner.summary()
sys.exit(0 if success else 1)
