"""
generate_quarter_local_coordinates 함수 검증 테스트.

실행:
    cd "프로젝트 루트"
    python -m pytest tests/test_adss_quarter_coords.py -v
"""

import sys
from pathlib import Path

# 프로젝트 루트를 sys.path에 추가
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest

from speckle.core.optimization.adss_subset_numba import (
    generate_quarter_local_coordinates,
    Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8,
)


class TestQuarterLocalCoordinates:
    """generate_quarter_local_coordinates 함수 검증."""

    # ── 픽셀 수 검증 ──────────────────────────────────

    @pytest.mark.parametrize("subset_size", [13, 21, 25])
    def test_q0_full_subset_pixel_count(self, subset_size):
        """Q0(전체 서브셋)의 픽셀 수 = (2M+1)²."""
        lx, ly = generate_quarter_local_coordinates(subset_size, Q0)
        expected = subset_size ** 2
        assert len(lx) == expected
        assert len(ly) == expected

    @pytest.mark.parametrize("subset_size", [13, 21, 25])
    def test_cross_quarter_pixel_count(self, subset_size):
        """Q1~Q4 (십자 반분할)의 픽셀 수 = (2M+1)×(M+1)."""
        M = subset_size // 2
        expected = (2 * M + 1) * (M + 1)
        for q in [Q1, Q2, Q3, Q4]:
            lx, ly = generate_quarter_local_coordinates(subset_size, q)
            assert len(lx) == expected, f"Q{q}: got {len(lx)}, expected {expected}"

    @pytest.mark.parametrize("subset_size", [13, 21, 25])
    def test_diagonal_quarter_pixel_count(self, subset_size):
        """Q5~Q8 (대각 사분할)의 픽셀 수 = (M+1)²."""
        M = subset_size // 2
        expected = (M + 1) ** 2
        for q in [Q5, Q6, Q7, Q8]:
            lx, ly = generate_quarter_local_coordinates(subset_size, q)
            assert len(lx) == expected, f"Q{q}: got {len(lx)}, expected {expected}"

    # ── 좌표 범위 검증 ──────────────────────────────────

    @pytest.mark.parametrize("subset_size", [13, 21])
    def test_q0_range(self, subset_size):
        M = subset_size // 2
        lx, ly = generate_quarter_local_coordinates(subset_size, Q0)
        assert lx.min() == -M and lx.max() == M
        assert ly.min() == -M and ly.max() == M

    @pytest.mark.parametrize("subset_size", [13, 21])
    def test_q1_upper_half_range(self, subset_size):
        M = subset_size // 2
        lx, ly = generate_quarter_local_coordinates(subset_size, Q1)
        assert lx.min() == -M and lx.max() == M
        assert ly.min() == -M and ly.max() == 0

    @pytest.mark.parametrize("subset_size", [13, 21])
    def test_q2_lower_half_range(self, subset_size):
        M = subset_size // 2
        lx, ly = generate_quarter_local_coordinates(subset_size, Q2)
        assert lx.min() == -M and lx.max() == M
        assert ly.min() == 0 and ly.max() == M

    @pytest.mark.parametrize("subset_size", [13, 21])
    def test_q3_left_half_range(self, subset_size):
        M = subset_size // 2
        lx, ly = generate_quarter_local_coordinates(subset_size, Q3)
        assert lx.min() == -M and lx.max() == 0
        assert ly.min() == -M and ly.max() == M

    @pytest.mark.parametrize("subset_size", [13, 21])
    def test_q4_right_half_range(self, subset_size):
        M = subset_size // 2
        lx, ly = generate_quarter_local_coordinates(subset_size, Q4)
        assert lx.min() == 0 and lx.max() == M
        assert ly.min() == -M and ly.max() == M

    @pytest.mark.parametrize("subset_size", [13, 21])
    def test_q5_upper_left_range(self, subset_size):
        M = subset_size // 2
        lx, ly = generate_quarter_local_coordinates(subset_size, Q5)
        assert lx.min() == -M and lx.max() == 0
        assert ly.min() == -M and ly.max() == 0

    @pytest.mark.parametrize("subset_size", [13, 21])
    def test_q6_upper_right_range(self, subset_size):
        M = subset_size // 2
        lx, ly = generate_quarter_local_coordinates(subset_size, Q6)
        assert lx.min() == 0 and lx.max() == M
        assert ly.min() == -M and ly.max() == 0

    @pytest.mark.parametrize("subset_size", [13, 21])
    def test_q7_lower_left_range(self, subset_size):
        M = subset_size // 2
        lx, ly = generate_quarter_local_coordinates(subset_size, Q7)
        assert lx.min() == -M and lx.max() == 0
        assert ly.min() == 0 and ly.max() == M

    @pytest.mark.parametrize("subset_size", [13, 21])
    def test_q8_lower_right_range(self, subset_size):
        M = subset_size // 2
        lx, ly = generate_quarter_local_coordinates(subset_size, Q8)
        assert lx.min() == 0 and lx.max() == M
        assert ly.min() == 0 and ly.max() == M

    # ── 포함 관계 검증 ──────────────────────────────────

    @pytest.mark.parametrize("subset_size", [13, 21])
    def test_upper_lower_cover_full(self, subset_size):
        """Q1 ∪ Q2 = Q0 (η=0 행 중복 포함)."""
        full_set = self._to_set(subset_size, Q0)
        union = self._to_set(subset_size, Q1) | self._to_set(subset_size, Q2)
        assert full_set == union

    @pytest.mark.parametrize("subset_size", [13, 21])
    def test_left_right_cover_full(self, subset_size):
        """Q3 ∪ Q4 = Q0 (ξ=0 열 중복 포함)."""
        full_set = self._to_set(subset_size, Q0)
        union = self._to_set(subset_size, Q3) | self._to_set(subset_size, Q4)
        assert full_set == union

    @pytest.mark.parametrize("subset_size", [13, 21])
    def test_four_diagonal_cover_full(self, subset_size):
        """Q5 ∪ Q6 ∪ Q7 ∪ Q8 = Q0."""
        full_set = self._to_set(subset_size, Q0)
        union = set()
        for q in [Q5, Q6, Q7, Q8]:
            union |= self._to_set(subset_size, q)
        assert full_set == union

    # ── POI 중심(0,0) 포함 검증 ────────────────────────

    @pytest.mark.parametrize("q", [Q0, Q1, Q2, Q3, Q4, Q5, Q6, Q7, Q8])
    def test_origin_included(self, q):
        """모든 quarter-subset은 원점(0,0)을 포함."""
        coords = self._to_set(21, q)
        assert (0.0, 0.0) in coords

    # ── 데이터 타입 검증 ────────────────────────────────

    def test_output_dtype(self):
        lx, ly = generate_quarter_local_coordinates(21, Q1)
        assert lx.dtype == np.float64
        assert ly.dtype == np.float64

    # ── 예외 처리 검증 ──────────────────────────────────

    def test_invalid_subset_type_raises(self):
        with pytest.raises(ValueError):
            generate_quarter_local_coordinates(21, 9)
        with pytest.raises(ValueError):
            generate_quarter_local_coordinates(21, -1)

    # ── 헬퍼 ───────────────────────────────────────────

    @staticmethod
    def _to_set(subset_size, q):
        lx, ly = generate_quarter_local_coordinates(subset_size, q)
        return set(zip(lx.tolist(), ly.tolist()))
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
