"""Warp Update 일관성 검증"""

import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from speckle.core.optimization.shape_function import (
    _update_affine_direct,
    _update_affine_matrix_fallback
)


def test_warp_update_consistency():
    """직접 계산과 fallback이 같은 결과를 내는지 검증"""
    
    # 케이스 1: 일반적인 DIC 파라미터
    p = np.array([2.5, 0.01, -0.005, 1.3, 0.003, 0.02])
    dp = np.array([0.1, 0.001, -0.0003, 0.05, 0.0002, 0.001])
    
    result_direct = _update_affine_direct(p, dp)
    result_fallback = _update_affine_matrix_fallback(p, dp)
    
    print("=" * 50)
    print("케이스 1: 일반적인 파라미터")
    print("=" * 50)
    print(f"p  = {p}")
    print(f"dp = {dp}")
    print(f"\nDirect:   {result_direct}")
    print(f"Fallback: {result_fallback}")
    print(f"Diff:     {np.abs(result_direct - result_fallback)}")
    
    match1 = np.allclose(result_direct, result_fallback, atol=1e-10)
    print(f"\n{'✅ 일치' if match1 else '❌ 불일치!'}")
    
    # 케이스 2: dp가 작은 경우 (수렴 직전)
    p2 = np.array([5.0, 0.02, -0.01, 3.0, -0.005, 0.03])
    dp2 = np.array([0.001, 0.0001, -0.00005, 0.0005, 0.00002, 0.0001])
    
    result_direct2 = _update_affine_direct(p2, dp2)
    result_fallback2 = _update_affine_matrix_fallback(p2, dp2)
    
    print("\n" + "=" * 50)
    print("케이스 2: 작은 dp (수렴 직전)")
    print("=" * 50)
    print(f"Direct:   {result_direct2}")
    print(f"Fallback: {result_fallback2}")
    print(f"Diff:     {np.abs(result_direct2 - result_fallback2)}")
    
    match2 = np.allclose(result_direct2, result_fallback2, atol=1e-10)
    print(f"\n{'✅ 일치' if match2 else '❌ 불일치!'}")
    
    # 케이스 3: dp가 큰 경우 (첫 반복)
    p3 = np.array([10.0, 0.0, 0.0, 5.0, 0.0, 0.0])
    dp3 = np.array([0.5, 0.01, -0.008, 0.3, 0.005, 0.01])
    
    result_direct3 = _update_affine_direct(p3, dp3)
    result_fallback3 = _update_affine_matrix_fallback(p3, dp3)
    
    print("\n" + "=" * 50)
    print("케이스 3: 큰 dp (첫 반복)")
    print("=" * 50)
    print(f"Direct:   {result_direct3}")
    print(f"Fallback: {result_fallback3}")
    print(f"Diff:     {np.abs(result_direct3 - result_fallback3)}")
    
    match3 = np.allclose(result_direct3, result_fallback3, atol=1e-10)
    print(f"\n{'✅ 일치' if match3 else '❌ 불일치!'}")
    
    # 종합
    print("\n" + "=" * 50)
    print("종합 결과")
    print("=" * 50)
    all_pass = match1 and match2 and match3
    print(f"{'✅ 모두 일치 - fallback 정상' if all_pass else '❌ 불일치 발견 - fallback 수정 필요'}")
    
    return all_pass


if __name__ == "__main__":
    test_warp_update_consistency()
