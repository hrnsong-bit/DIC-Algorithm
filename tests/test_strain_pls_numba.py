#!/usr/bin/env python3
"""
PLS Numba 병렬화 테스트

기존 strain_pls.py vs strain_pls_numba.py 수치 동일성 및 성능 벤치마크
"""

import sys
import time
import numpy as np

sys.path.insert(0, '.')

from speckle.core.postprocess.strain_pls import compute_strain_pls
from speckle.core.postprocess.strain_pls_numba import (
    compute_strain_pls_numba, warmup_pls_numba
)


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

def make_test_data(ny=30, nx=40, seed=42):
    """선형 변위 + 노이즈 테스트 데이터"""
    np.random.seed(seed)
    y, x = np.mgrid[0:ny, 0:nx].astype(np.float64)

    # 선형 변위: u = 0.001*x + 0.0005*y, v = -0.0003*x + 0.002*y
    disp_u = 0.001 * x + 0.0005 * y + np.random.randn(ny, nx) * 0.0001
    disp_v = -0.0003 * x + 0.002 * y + np.random.randn(ny, nx) * 0.0001

    return disp_u, disp_v


def make_test_data_with_nan(ny=30, nx=40, nan_fraction=0.1, seed=42):
    """NaN 포함 테스트 데이터"""
    disp_u, disp_v = make_test_data(ny, nx, seed)
    np.random.seed(seed + 1)
    mask = np.random.rand(ny, nx) < nan_fraction
    disp_u[mask] = np.nan
    disp_v[mask] = np.nan
    return disp_u, disp_v


# =============================================================================
#  JIT 워밍업
# =============================================================================

print("=" * 60)
print("PLS Numba Parallelization Test Suite")
print("=" * 60)

print("\nWarming up JIT...")
t0 = time.time()
warmup_pls_numba()
print(f"Warmup: {time.time()-t0:.2f}s\n")


# =============================================================================
#  Test 1: poly_order=2 수치 동일성
# =============================================================================

print("--- Test 1: poly_order=2 numerical equivalence ---")

disp_u, disp_v = make_test_data()

# 기존
orig = compute_strain_pls(disp_u, disp_v, window_size=11, poly_order=2, grid_step=1.0)

# Numba
numba_result = compute_strain_pls_numba(disp_u, disp_v, window_size=11, poly_order=2, grid_step=1.0)

# 유효 영역만 비교 (NaN이 아닌 곳)
valid = ~np.isnan(orig.exx) & ~np.isnan(numba_result['exx'])
n_valid = np.sum(valid)

runner.check("1-1 valid count match",
             abs(n_valid - np.sum(~np.isnan(orig.exx))) <= 0,
             f"orig_valid={np.sum(~np.isnan(orig.exx))}, numba_valid={np.sum(~np.isnan(numba_result['exx']))}")

if n_valid > 0:
    max_diff_exx = np.max(np.abs(orig.exx[valid] - numba_result['exx'][valid]))
    max_diff_eyy = np.max(np.abs(orig.eyy[valid] - numba_result['eyy'][valid]))
    max_diff_exy = np.max(np.abs(orig.exy[valid] - numba_result['exy'][valid]))

    runner.check("1-2 exx match",
                 max_diff_exx < 1e-10,
                 f"max_diff={max_diff_exx:.2e}")

    runner.check("1-3 eyy match",
                 max_diff_eyy < 1e-10,
                 f"max_diff={max_diff_eyy:.2e}")

    runner.check("1-4 exy match",
                 max_diff_exy < 1e-10,
                 f"max_diff={max_diff_exy:.2e}")

    # 주변형률, von Mises
    valid_e1 = valid & ~np.isnan(orig.e1) & ~np.isnan(numba_result['e1'])
    if np.sum(valid_e1) > 0:
        max_diff_e1 = np.max(np.abs(orig.e1[valid_e1] - numba_result['e1'][valid_e1]))
        runner.check("1-5 e1 match", max_diff_e1 < 1e-10, f"max_diff={max_diff_e1:.2e}")

    valid_vm = valid & ~np.isnan(orig.von_mises) & ~np.isnan(numba_result['von_mises'])
    if np.sum(valid_vm) > 0:
        max_diff_vm = np.max(np.abs(orig.von_mises[valid_vm] - numba_result['von_mises'][valid_vm]))
        runner.check("1-6 von_mises match", max_diff_vm < 1e-10, f"max_diff={max_diff_vm:.2e}")
else:
    runner.check("1-2 no valid points", False, "n_valid=0")


# =============================================================================
#  Test 2: poly_order=1 수치 동일성
# =============================================================================

print("\n--- Test 2: poly_order=1 numerical equivalence ---")

orig1 = compute_strain_pls(disp_u, disp_v, window_size=11, poly_order=1, grid_step=1.0)
numba1 = compute_strain_pls_numba(disp_u, disp_v, window_size=11, poly_order=1, grid_step=1.0)

valid1 = ~np.isnan(orig1.exx) & ~np.isnan(numba1['exx'])
n_valid1 = np.sum(valid1)

runner.check("2-1 valid count match",
             abs(n_valid1 - np.sum(~np.isnan(orig1.exx))) <= 0)

if n_valid1 > 0:
    max_diff = np.max(np.abs(orig1.exx[valid1] - numba1['exx'][valid1]))
    runner.check("2-2 exx match (order=1)", max_diff < 1e-10, f"max_diff={max_diff:.2e}")

    max_diff_eyy = np.max(np.abs(orig1.eyy[valid1] - numba1['eyy'][valid1]))
    runner.check("2-3 eyy match (order=1)", max_diff_eyy < 1e-10, f"max_diff={max_diff_eyy:.2e}")


# =============================================================================
#  Test 3: NaN 처리
# =============================================================================

print("\n--- Test 3: NaN handling ---")

disp_u_nan, disp_v_nan = make_test_data_with_nan(nan_fraction=0.15)

orig_nan = compute_strain_pls(disp_u_nan, disp_v_nan, window_size=11, poly_order=2)
numba_nan = compute_strain_pls_numba(disp_u_nan, disp_v_nan, window_size=11, poly_order=2)

valid_nan = ~np.isnan(orig_nan.exx) & ~np.isnan(numba_nan['exx'])
n_valid_nan = np.sum(valid_nan)

runner.check("3-1 NaN data: valid count reasonable",
             n_valid_nan > 0 and n_valid_nan <= disp_u_nan.size)

if n_valid_nan > 0:
    max_diff = np.max(np.abs(orig_nan.exx[valid_nan] - numba_nan['exx'][valid_nan]))
    runner.check("3-2 NaN data: exx match", max_diff < 1e-10, f"max_diff={max_diff:.2e}")


# =============================================================================
#  Test 4: Green-Lagrange strain
# =============================================================================

print("\n--- Test 4: Green-Lagrange strain ---")

orig_gl = compute_strain_pls(disp_u, disp_v, window_size=11, poly_order=2,
                              strain_type='green-lagrange')
numba_gl = compute_strain_pls_numba(disp_u, disp_v, window_size=11, poly_order=2,
                                     strain_type='green-lagrange')

valid_gl = ~np.isnan(orig_gl.exx) & ~np.isnan(numba_gl['exx'])
if np.sum(valid_gl) > 0:
    max_diff = np.max(np.abs(orig_gl.exx[valid_gl] - numba_gl['exx'][valid_gl]))
    runner.check("4-1 Green-Lagrange exx match", max_diff < 1e-10, f"max_diff={max_diff:.2e}")
    max_diff_exy = np.max(np.abs(orig_gl.exy[valid_gl] - numba_gl['exy'][valid_gl]))
    runner.check("4-2 Green-Lagrange exy match", max_diff_exy < 1e-10, f"max_diff={max_diff_exy:.2e}")


# =============================================================================
#  Test 5: 다양한 window_size
# =============================================================================

print("\n--- Test 5: Various window sizes ---")

for ws in [5, 7, 15, 21]:
    orig_ws = compute_strain_pls(disp_u, disp_v, window_size=ws, poly_order=2)
    numba_ws = compute_strain_pls_numba(disp_u, disp_v, window_size=ws, poly_order=2)
    v = ~np.isnan(orig_ws.exx) & ~np.isnan(numba_ws['exx'])
    if np.sum(v) > 0:
        md = np.max(np.abs(orig_ws.exx[v] - numba_ws['exx'][v]))
        runner.check(f"5-{ws} window_size={ws} exx match", md < 1e-10, f"max_diff={md:.2e}")


# =============================================================================
#  Test 6: grid_step 파라미터
# =============================================================================

print("\n--- Test 6: grid_step parameter ---")

for gs in [1.0, 5.0, 10.0]:
    orig_gs = compute_strain_pls(disp_u, disp_v, window_size=11, grid_step=gs)
    numba_gs = compute_strain_pls_numba(disp_u, disp_v, window_size=11, grid_step=gs)
    v = ~np.isnan(orig_gs.exx) & ~np.isnan(numba_gs['exx'])
    if np.sum(v) > 0:
        md = np.max(np.abs(orig_gs.exx[v] - numba_gs['exx'][v]))
        runner.check(f"6-{gs} grid_step={gs} match", md < 1e-10, f"max_diff={md:.2e}")


# =============================================================================
#  Test 7: 큰 데이터 성능 벤치마크
# =============================================================================

print("\n--- Test 7: Performance benchmark ---")

disp_u_big, disp_v_big = make_test_data(ny=100, nx=100, seed=99)

# Numba 한번 더 워밍업 (이 크기에서)
_ = compute_strain_pls_numba(disp_u_big[:10, :10], disp_v_big[:10, :10],
                              window_size=11, poly_order=2)

# Original
t0 = time.time()
orig_big = compute_strain_pls(disp_u_big, disp_v_big, window_size=11, poly_order=2)
t_orig = time.time() - t0

# Numba
t0 = time.time()
numba_big = compute_strain_pls_numba(disp_u_big, disp_v_big, window_size=11, poly_order=2)
t_numba = time.time() - t0

speedup = t_orig / t_numba if t_numba > 0 else float('inf')

print(f"\n  Performance ({disp_u_big.shape[0]}×{disp_u_big.shape[1]} = {disp_u_big.size} POIs, window=11, order=2):")
print(f"    Original:  {t_orig*1000:.1f} ms")
print(f"    Numba:     {t_numba*1000:.1f} ms")
print(f"    Speedup:   {speedup:.1f}x")

runner.check("7-1 Numba faster than original",
             t_numba < t_orig,
             f"numba={t_numba*1000:.0f}ms, orig={t_orig*1000:.0f}ms")

# 결과 동일성 확인
v_big = ~np.isnan(orig_big.exx) & ~np.isnan(numba_big['exx'])
if np.sum(v_big) > 0:
    md = np.max(np.abs(orig_big.exx[v_big] - numba_big['exx'][v_big]))
    runner.check("7-2 benchmark results match", md < 1e-10, f"max_diff={md:.2e}")


# =============================================================================
#  요약
# =============================================================================

print()
success = runner.summary()
sys.exit(0 if success else 1)
