"""
변위장 검증 및 불연속 검출 모듈

연속적 변형 가정 위반 시 SIFT 등 대체 알고리즘으로 전환하기 위한 검출기
"""

import numpy as np
from typing import Tuple, List, Optional
from numba import jit, prange
from dataclasses import dataclass

from .results import FFTCCResult


@dataclass
class ValidationResult:
    """검증 결과"""
    is_valid: bool
    discontinuous_points: List[Tuple[int, int]]  # (y, x) 좌표
    outlier_points: List[Tuple[int, int]]
    low_zncc_points: List[Tuple[int, int]]
    
    discontinuity_ratio: float  # 불연속 비율
    outlier_ratio: float
    
    suggested_action: str  # 'proceed', 'retry_sift', 'increase_subset', 'fail'


def validate_displacement_field(result: FFTCCResult,
                                 zncc_threshold: float = 0.6,
                                 gradient_threshold: float = 2.0,
                                 outlier_std_factor: float = 3.0,
                                 discontinuity_tolerance: float = 0.1) -> ValidationResult:
    """
    변위장 검증
    
    Args:
        result: FFT-CC 결과
        zncc_threshold: ZNCC 임계값
        gradient_threshold: 변위 그래디언트 임계값 (pixel/spacing)
        outlier_std_factor: 이상치 판정 표준편차 배수
        discontinuity_tolerance: 허용 불연속 비율
    
    Returns:
        ValidationResult
    """
    n_points = result.n_points
    
    if n_points == 0:
        return ValidationResult(
            is_valid=False,
            discontinuous_points=[],
            outlier_points=[],
            low_zncc_points=[],
            discontinuity_ratio=0.0,
            outlier_ratio=0.0,
            suggested_action='fail'
        )
    
    # 1. Low ZNCC 검출
    low_zncc_mask = result.zncc_values < zncc_threshold
    low_zncc_points = [
        (int(result.points_y[i]), int(result.points_x[i]))
        for i in range(n_points) if low_zncc_mask[i]
    ]
    
    # 2. 이상치 검출 (MAD 기반)
    outlier_mask = _detect_outliers_mad(
        result.disp_u, result.disp_v, outlier_std_factor
    )
    outlier_points = [
        (int(result.points_y[i]), int(result.points_x[i]))
        for i in range(n_points) if outlier_mask[i]
    ]
    
    # 3. 불연속 검출 (변위 그래디언트 기반)
    discontinuous_mask = _detect_discontinuity(
        result.points_y, result.points_x,
        result.disp_u, result.disp_v,
        result.spacing, gradient_threshold
    )
    discontinuous_points = [
        (int(result.points_y[i]), int(result.points_x[i]))
        for i in range(n_points) if discontinuous_mask[i]
    ]
    
    # 비율 계산
    discontinuity_ratio = len(discontinuous_points) / n_points
    outlier_ratio = len(outlier_points) / n_points
    low_zncc_ratio = len(low_zncc_points) / n_points
    
    # 종합 판정
    total_bad_ratio = low_zncc_ratio + discontinuity_ratio
    
    if total_bad_ratio < 0.05:
        is_valid = True
        suggested_action = 'proceed'
    elif total_bad_ratio < discontinuity_tolerance:
        is_valid = True
        suggested_action = 'proceed'  # 일부 불량은 허용
    elif discontinuity_ratio > 0.2:
        is_valid = False
        suggested_action = 'retry_sift'  # 큰 불연속 -> SIFT 권장
    elif low_zncc_ratio > 0.3:
        is_valid = False
        suggested_action = 'increase_subset'  # 스페클 품질 문제
    else:
        is_valid = False
        suggested_action = 'retry_sift'
    
    return ValidationResult(
        is_valid=is_valid,
        discontinuous_points=discontinuous_points,
        outlier_points=outlier_points,
        low_zncc_points=low_zncc_points,
        discontinuity_ratio=discontinuity_ratio,
        outlier_ratio=outlier_ratio,
        suggested_action=suggested_action
    )


def _detect_outliers_mad(disp_u: np.ndarray,
                          disp_v: np.ndarray,
                          factor: float) -> np.ndarray:
    """MAD(Median Absolute Deviation) 기반 이상치 검출"""
    # MAD for u
    med_u = np.median(disp_u)
    mad_u = np.median(np.abs(disp_u - med_u))
    mad_u = mad_u if mad_u > 0 else 1.0
    
    # MAD for v
    med_v = np.median(disp_v)
    mad_v = np.median(np.abs(disp_v - med_v))
    mad_v = mad_v if mad_v > 0 else 1.0
    
    # 이상치 판정 (MAD * 1.4826 ≈ std for normal dist)
    threshold_u = factor * 1.4826 * mad_u
    threshold_v = factor * 1.4826 * mad_v
    
    outlier_u = np.abs(disp_u - med_u) > threshold_u
    outlier_v = np.abs(disp_v - med_v) > threshold_v
    
    return outlier_u | outlier_v


@jit(nopython=True, parallel=True, cache=True)
def _detect_discontinuity(points_y: np.ndarray,
                           points_x: np.ndarray,
                           disp_u: np.ndarray,
                           disp_v: np.ndarray,
                           spacing: int,
                           threshold: float) -> np.ndarray:
    """
    변위 그래디언트 기반 불연속 검출 (Numba 병렬화)
    
    인접 POI와의 변위 차이가 threshold * spacing 초과 시 불연속으로 판정
    """
    n_points = len(points_y)
    discontinuous = np.zeros(n_points, dtype=np.bool_)
    
    max_diff = threshold * spacing
    neighbor_range = int(spacing * 1.5)
    
    for idx in prange(n_points):
        py, px = points_y[idx], points_x[idx]
        u, v = disp_u[idx], disp_v[idx]
        
        max_gradient = 0.0
        has_neighbor = False
        
        for j in range(n_points):
            if idx == j:
                continue
            
            dy = abs(points_y[j] - py)
            dx = abs(points_x[j] - px)
            
            # 인접 POI 확인 (상하좌우 + 대각선)
            if dy <= neighbor_range and dx <= neighbor_range:
                if dy == 0 and dx == 0:
                    continue
                
                has_neighbor = True
                
                du = abs(disp_u[j] - u)
                dv = abs(disp_v[j] - v)
                gradient = np.sqrt(du * du + dv * dv)
                
                if gradient > max_gradient:
                    max_gradient = gradient
        
        if has_neighbor and max_gradient > max_diff:
            discontinuous[idx] = True
    
    return discontinuous


def detect_crack_region(result: FFTCCResult,
                         validation: ValidationResult,
                         expansion: int = 2) -> List[Tuple[int, int, int, int]]:
    """
    크랙/불연속 영역 검출 및 바운딩 박스 반환
    
    Args:
        result: FFT-CC 결과
        validation: 검증 결과
        expansion: 영역 확장 계수 (spacing 배수)
    
    Returns:
        불연속 영역 바운딩 박스 리스트 [(x, y, w, h), ...]
    """
    if not validation.discontinuous_points:
        return []
    
    # 불연속 포인트들을 클러스터링 (간단한 연결 성분 분석)
    points = np.array(validation.discontinuous_points)
    spacing = result.spacing
    
    # DBSCAN 스타일 클러스터링 (간단 버전)
    clusters = []
    visited = set()
    
    for i, (y, x) in enumerate(points):
        if i in visited:
            continue
        
        cluster = [(y, x)]
        visited.add(i)
        
        # 인접 포인트 찾기
        for j, (y2, x2) in enumerate(points):
            if j in visited:
                continue
            if abs(y2 - y) <= spacing * 2 and abs(x2 - x) <= spacing * 2:
                cluster.append((y2, x2))
                visited.add(j)
        
        clusters.append(cluster)
    
    # 각 클러스터의 바운딩 박스
    bboxes = []
    exp_pixels = expansion * spacing
    
    for cluster in clusters:
        if len(cluster) < 2:
            continue
        
        ys = [p[0] for p in cluster]
        xs = [p[1] for p in cluster]
        
        y_min = max(0, min(ys) - exp_pixels)
        y_max = max(ys) + exp_pixels
        x_min = max(0, min(xs) - exp_pixels)
        x_max = max(xs) + exp_pixels
        
        bboxes.append((x_min, y_min, x_max - x_min, y_max - y_min))
    
    return bboxes
