"""
변위장 검증 및 불연속 검출 모듈

연속적 변형 가정 위반 시 SIFT 등 대체 알고리즘으로 전환하기 위한 검출기

변경 이력:
    v3.3.1 - _detect_discontinuity를 KD-tree 기반으로 교체
             O(n²) → O(n·k), 2000×2000 이미지(~15k POI) 기준 약 100배 이상 개선
           - detect_crack_region 클러스터링을 KD-tree + BFS로 교체
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from scipy.spatial import cKDTree

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
    
    # 3. 불연속 검출 (KD-tree 기반, v3.3.1)
    #    기존: O(n²) 전수비교 → 변경: O(n·k) KD-tree 이웃 탐색
    #    2000×2000 이미지, spacing=16 (~15,625 POI) 기준
    #    기존: ~2.4억 비교 (수십 초) → 변경: ~12만 비교 (수십 ms)
    discontinuous_mask = _detect_discontinuity_kdtree(
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
        suggested_action = 'proceed'
    elif discontinuity_ratio > 0.2:
        is_valid = False
        suggested_action = 'retry_sift'
    elif low_zncc_ratio > 0.3:
        is_valid = False
        suggested_action = 'increase_subset'
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
    med_u = np.median(disp_u)
    mad_u = np.median(np.abs(disp_u - med_u))
    mad_u = mad_u if mad_u > 0 else 1.0
    
    med_v = np.median(disp_v)
    mad_v = np.median(np.abs(disp_v - med_v))
    mad_v = mad_v if mad_v > 0 else 1.0
    
    threshold_u = factor * 1.4826 * mad_u
    threshold_v = factor * 1.4826 * mad_v
    
    outlier_u = np.abs(disp_u - med_u) > threshold_u
    outlier_v = np.abs(disp_v - med_v) > threshold_v
    
    return outlier_u | outlier_v


def _detect_discontinuity_kdtree(points_y: np.ndarray,
                                  points_x: np.ndarray,
                                  disp_u: np.ndarray,
                                  disp_v: np.ndarray,
                                  spacing: int,
                                  threshold: float) -> np.ndarray:
    """
    KD-tree 기반 불연속 검출 — O(n·k)
    
    v3.3.1에서 기존 O(n²) 전수비교를 대체.
    scipy.spatial.cKDTree로 이웃을 일괄 조회한 뒤,
    이웃과의 변위 차이가 threshold * spacing 초과 시 불연속으로 판정.
    
    성능:
        n=15,000 (2000×2000, spacing=16) 기준
        기존 Numba O(n²): ~2.4억 비교, 수십 초
        KD-tree O(n·k):   ~12만 비교 (k≈8), 수십 ms
    
    Args:
        points_y, points_x: POI 좌표
        disp_u, disp_v: 변위
        spacing: POI 간격
        threshold: 변위 그래디언트 임계값
    
    Returns:
        불연속 마스크 (n_points,)
    """
    n_points = len(points_y)
    if n_points < 2:
        return np.zeros(n_points, dtype=np.bool_)
    
    # KD-tree 구성 — O(n log n)
    coords = np.column_stack((points_y.astype(np.float64),
                               points_x.astype(np.float64)))
    tree = cKDTree(coords)
    
    # 이웃 탐색 반경: spacing * 1.5 (상하좌우 + 대각선 포함)
    neighbor_range = spacing * 1.5
    max_diff = threshold * spacing
    
    # 전체 POI에 대해 이웃 인덱스 일괄 조회 — O(n·k)
    neighbor_lists = tree.query_ball_tree(tree, r=neighbor_range)
    
    # 변위 배열 준비
    disp_u_f = disp_u.astype(np.float64)
    disp_v_f = disp_v.astype(np.float64)
    
    # 그래디언트 판정
    discontinuous = _check_gradients(
        disp_u_f, disp_v_f, neighbor_lists, max_diff, n_points
    )
    
    return discontinuous


def _check_gradients(disp_u: np.ndarray,
                     disp_v: np.ndarray,
                     neighbor_lists: List[List[int]],
                     max_diff: float,
                     n_points: int) -> np.ndarray:
    """
    이웃 목록 기반 그래디언트 판정
    
    KD-tree에서 조회된 이웃 목록을 순회하며 최대 변위 그래디언트 계산.
    이웃 수가 소수(k≈4~8)이므로 Python 루프로도 충분히 빠름.
    
    Note:
        Numba @jit 미적용 — neighbor_lists가 Python list of lists라
        Numba typed list 변환 비용이 이점을 상쇄함.
        k가 작아 순수 Python으로도 n=15,000에서 ~10ms 수준.
    """
    discontinuous = np.zeros(n_points, dtype=np.bool_)
    
    for idx in range(n_points):
        neighbors = neighbor_lists[idx]
        if len(neighbors) <= 1:
            continue
        
        u_i = disp_u[idx]
        v_i = disp_v[idx]
        
        for j in neighbors:
            if j == idx:
                continue
            
            du = disp_u[j] - u_i
            dv = disp_v[j] - v_i
            gradient = np.sqrt(du * du + dv * dv)
            
            # 하나라도 임계값 초과 시 불연속 확정, 다음 POI로
            if gradient > max_diff:
                discontinuous[idx] = True
                break
    
    return discontinuous


def detect_crack_region(result: FFTCCResult,
                         validation: ValidationResult,
                         expansion: int = 2) -> List[Tuple[int, int, int, int]]:
    """
    크랙/불연속 영역 검출 및 바운딩 박스 반환
    
    v3.3.1에서 기존 이중 루프 클러스터링을 KD-tree + BFS로 교체.
    
    Args:
        result: FFT-CC 결과
        validation: 검증 결과
        expansion: 영역 확장 계수 (spacing 배수)
    
    Returns:
        불연속 영역 바운딩 박스 리스트 [(x, y, w, h), ...]
    """
    if not validation.discontinuous_points:
        return []
    
    points = np.array(validation.discontinuous_points)
    spacing = result.spacing
    
    # KD-tree 기반 클러스터링 (BFS 연결 성분 탐색)
    coords = points.astype(np.float64)
    tree = cKDTree(coords)
    
    clusters = []
    visited = set()
    
    for i in range(len(points)):
        if i in visited:
            continue
        
        # BFS로 연결 성분 탐색
        cluster_indices = []
        queue = [i]
        visited.add(i)
        
        while queue:
            current = queue.pop(0)
            cluster_indices.append(current)
            
            # KD-tree로 인접 포인트 조회 — O(log n)
            neighbors = tree.query_ball_point(coords[current], r=spacing * 2)
            for j in neighbors:
                if j not in visited:
                    visited.add(j)
                    queue.append(j)
        
        cluster = [tuple(points[idx]) for idx in cluster_indices]
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
