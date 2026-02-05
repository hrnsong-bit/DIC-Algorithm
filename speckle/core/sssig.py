"""
SSSIG 계산 모듈

References:
- Pan et al. (2008) "Study on subset size selection in digital image 
  correlation for speckle patterns" Optics Express
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from numba import jit, prange

from ..models.reports import BadPoint, SSSIGResult


@jit(nopython=True, parallel=True, cache=True)
def _compute_sssig_map_parallel(gx: np.ndarray, gy: np.ndarray,
                                 points_y: np.ndarray, points_x: np.ndarray,
                                 half: int) -> Tuple[np.ndarray, np.ndarray]:
    """모든 POI에 대한 SSSIG 병렬 계산 (x, y 방향 분리)"""
    n_points = len(points_y)
    sssig_x = np.empty(n_points, dtype=np.float64)
    sssig_y = np.empty(n_points, dtype=np.float64)
    
    h, w = gx.shape
    
    for idx in prange(n_points):
        py, px = points_y[idx], points_x[idx]
        y1 = max(0, py - half)
        y2 = min(h, py + half + 1)
        x1 = max(0, px - half)
        x2 = min(w, px + half + 1)
        
        total_x = 0.0
        total_y = 0.0
        for y in range(y1, y2):
            for x in range(x1, x2):
                gx_val = gx[y, x]
                gy_val = gy[y, x]
                total_x += gx_val * gx_val
                total_y += gy_val * gy_val
        
        sssig_x[idx] = total_x
        sssig_y[idx] = total_y
    
    return sssig_x, sssig_y


def estimate_noise_variance(image: np.ndarray, method: str = 'laplacian') -> float:
    """
    이미지 노이즈 분산 추정 (논문 Section 5.1 참조)
    
    Args:
        image: 그레이스케일 이미지
        method: 'laplacian' (단일 이미지) 또는 'difference' (두 이미지 필요)
    
    Returns:
        노이즈 분산 D(η)
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img_float = image.astype(np.float64)
    
    if method == 'laplacian':
        # Robust Median Estimator (Donoho & Johnstone, 1994)
        # σ = median(|Laplacian|) / 0.6745
        laplacian = cv2.Laplacian(img_float, cv2.CV_64F)
        sigma = np.median(np.abs(laplacian)) / 0.6745
        return sigma ** 2
    
    elif method == 'local_std':
        # 로컬 영역의 표준편차 기반 추정
        kernel_size = 7
        local_mean = cv2.blur(img_float, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(img_float ** 2, (kernel_size, kernel_size))
        local_var = local_sq_mean - local_mean ** 2
        # 하위 10% (평탄한 영역)의 분산을 노이즈로 추정
        return float(np.percentile(local_var[local_var > 0], 10))
    
    else:
        # 기본값: 일반적인 8-bit 카메라 노이즈
        return 4.0


def estimate_noise_from_pair(image1: np.ndarray, image2: np.ndarray) -> float:
    """
    두 장의 동일 조건 이미지에서 노이즈 분산 추정
    (논문에서 권장하는 방법)
    
    Args:
        image1, image2: 동일 조건에서 촬영한 두 이미지
    
    Returns:
        노이즈 분산 D(η)
    """
    if len(image1.shape) == 3:
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    if len(image2.shape) == 3:
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    
    diff = image1.astype(np.float64) - image2.astype(np.float64)
    # Var(diff) = Var(η1) + Var(η2) = 2 * D(η)
    return float(np.var(diff) / 2)


def calculate_sssig_threshold(noise_variance: float, 
                               desired_accuracy: float = 0.01) -> float:
    """
    논문 Eq. 18, 19 기반 SSSIG threshold 계산
    
    σ(Δu) ≈ √[D(η) / SSSIG]
    → SSSIG ≥ D(η) / σ²
    
    Args:
        noise_variance: 이미지 노이즈 분산 D(η)
        desired_accuracy: 원하는 변위 측정 정확도 (pixels)
    
    Returns:
        SSSIG threshold
    """
    if desired_accuracy <= 0:
        desired_accuracy = 0.01
    
    threshold = noise_variance / (desired_accuracy ** 2)
    
    # 최소/최대 제한
    threshold = max(threshold, 1e4)   # 최소 1e4
    threshold = min(threshold, 1e7)   # 최대 1e7
    
    return threshold


def predict_displacement_accuracy(sssig: float, noise_variance: float) -> float:
    """
    논문 Eq. 18, 19: SSSIG와 노이즈 분산으로 예상 정확도 계산
    
    Args:
        sssig: SSSIG 값
        noise_variance: 노이즈 분산
    
    Returns:
        예상 변위 측정 정확도 (pixels)
    """
    if sssig <= 0:
        return float('inf')
    
    return np.sqrt(noise_variance / sssig)


def compute_sssig(image: np.ndarray, center: Tuple[int, int], 
                  subset_size: int, gx: np.ndarray = None, 
                  gy: np.ndarray = None) -> Tuple[float, float]:
    """
    단일 POI의 SSSIG 계산 (x, y 방향 분리)
    
    Returns:
        (sssig_x, sssig_y) 튜플
    """
    if gx is None or gy is None:
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_float = image.astype(np.float64)
        gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    
    half = subset_size // 2
    cy, cx = center
    h, w = gx.shape
    
    y1, y2 = max(0, cy - half), min(h, cy + half + 1)
    x1, x2 = max(0, cx - half), min(w, cx + half + 1)
    
    gx_sub = gx[y1:y2, x1:x2]
    gy_sub = gy[y1:y2, x1:x2]
    
    sssig_x = float(np.sum(gx_sub ** 2))
    sssig_y = float(np.sum(gy_sub ** 2))
    
    return sssig_x, sssig_y


def compute_sssig_map(image: np.ndarray,
                      subset_size: int = 21,
                      spacing: int = 10,
                      threshold: float = None,
                      noise_variance: float = None,
                      desired_accuracy: float = 0.01,
                      gx: np.ndarray = None,
                      gy: np.ndarray = None,
                      mask: np.ndarray = None) -> SSSIGResult:
    """
    전체 ROI에 대한 SSSIG 맵 계산 (논문 기반 개선 버전)
    
    Args:
        image: 입력 이미지
        subset_size: subset 크기 (pixels)
        spacing: POI 간격 (pixels)
        threshold: SSSIG threshold (None이면 자동 계산)
        noise_variance: 노이즈 분산 (None이면 자동 추정)
        desired_accuracy: 원하는 정확도 (pixels)
        gx, gy: 미리 계산된 그라디언트 (선택)
        mask: ROI 마스크 (선택)
    
    Returns:
        SSSIGResult
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape
    half = subset_size // 2
    
    # 노이즈 분산 추정 (없으면)
    if noise_variance is None:
        noise_variance = estimate_noise_variance(gray)
    
    # Threshold 계산 (없으면)
    if threshold is None:
        threshold = calculate_sssig_threshold(noise_variance, desired_accuracy)
    
    # Gradient 계산
    if gx is None or gy is None:
        img_float = gray.astype(np.float64)
        gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    
    # POI 그리드 생성
    margin = half + 1
    y_coords = np.arange(margin, h - margin, spacing)
    x_coords = np.arange(margin, w - margin, spacing)
    
    if len(y_coords) == 0 or len(x_coords) == 0:
        return SSSIGResult(
            map=np.array([[]]),
            mean=0.0, min=float('inf'), max=float('-inf'),
            bad_points=[],
            points_y=np.array([]), points_x=np.array([]),
            subset_size=subset_size, spacing=spacing,
            noise_variance=noise_variance,
            threshold=threshold,
            predicted_accuracy=float('inf')
        )
    
    # 모든 POI 좌표 생성
    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    all_points_y = yy.ravel()
    all_points_x = xx.ravel()
    
    # 마스크 적용
    if mask is not None:
        mask_values = mask[all_points_y, all_points_x]
        valid_mask = mask_values > 0
        points_y = all_points_y[valid_mask].astype(np.int64)
        points_x = all_points_x[valid_mask].astype(np.int64)
    else:
        points_y = all_points_y.astype(np.int64)
        points_x = all_points_x.astype(np.int64)
    
    if len(points_y) == 0:
        return SSSIGResult(
            map=np.array([[]]),
            mean=0.0, min=float('inf'), max=float('-inf'),
            bad_points=[],
            points_y=np.array([]), points_x=np.array([]),
            subset_size=subset_size, spacing=spacing,
            noise_variance=noise_variance,
            threshold=threshold,
            predicted_accuracy=float('inf')
        )
    
    # SSSIG 계산 (x, y 분리)
    sssig_x, sssig_y = _compute_sssig_map_parallel(
        gx.astype(np.float64),
        gy.astype(np.float64),
        points_y, points_x, half
    )
    
    # 총 SSSIG (x + y)
    sssig_values = sssig_x + sssig_y
    
    # 통계
    mean_sssig = float(np.mean(sssig_values))
    min_sssig = float(np.min(sssig_values))
    max_sssig = float(np.max(sssig_values))
    
    # 예상 정확도 계산 (논문 Eq. 18, 19)
    # 최소 SSSIG 기준으로 worst-case 정확도
    predicted_accuracy_x = predict_displacement_accuracy(np.min(sssig_x), noise_variance)
    predicted_accuracy_y = predict_displacement_accuracy(np.min(sssig_y), noise_variance)
    predicted_accuracy = max(predicted_accuracy_x, predicted_accuracy_y)
    
    # 불량 포인트 (x 또는 y 방향 중 하나라도 threshold 미만)
    bad_mask = (sssig_x < threshold / 2) | (sssig_y < threshold / 2)
    bad_indices = np.where(bad_mask)[0]
    
    bad_points = [
        BadPoint(
            y=int(points_y[idx]),
            x=int(points_x[idx]),
            sssig=float(sssig_values[idx])
        )
        for idx in bad_indices
    ]
    
    # 2D 맵
    if mask is None:
        sssig_map = sssig_values.reshape(len(y_coords), len(x_coords))
    else:
        sssig_map = np.full((len(y_coords), len(x_coords)), -1.0)
        y_indices = (points_y - margin) // spacing
        x_indices = (points_x - margin) // spacing
        valid_idx = (y_indices >= 0) & (y_indices < len(y_coords)) & \
                    (x_indices >= 0) & (x_indices < len(x_coords))
        sssig_map[y_indices[valid_idx], x_indices[valid_idx]] = sssig_values[valid_idx]
    
    return SSSIGResult(
        map=sssig_map,
        mean=mean_sssig,
        min=min_sssig,
        max=max_sssig,
        bad_points=bad_points,
        points_y=points_y,
        points_x=points_x,
        subset_size=subset_size,
        spacing=spacing,
        noise_variance=noise_variance,
        threshold=threshold,
        predicted_accuracy=predicted_accuracy
    )


def find_optimal_subset_size(image: np.ndarray,
                             spacing: int = 10,
                             desired_accuracy: float = 0.01,
                             noise_variance: float = None,
                             min_size: int = 11,
                             max_size: int = 61,
                             step: int = 2,
                             mask: np.ndarray = None) -> Tuple[int, bool, Optional[SSSIGResult], dict]:
    """
    최적 subset size 탐색 (논문 Fig. 3 알고리즘)
    
    Args:
        image: 입력 이미지
        spacing: POI 간격
        desired_accuracy: 원하는 변위 측정 정확도 (pixels)
        noise_variance: 노이즈 분산 (None이면 자동 추정)
        min_size: 최소 subset size
        max_size: 최대 subset size
        step: 증가 단위
        mask: ROI 마스크
    
    Returns:
        (optimal_size, found, result, info) 튜플
        - optimal_size: 최적 subset size
        - found: 찾았는지 여부
        - result: SSSIGResult
        - info: 추가 정보 (노이즈 분산, threshold 등)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 노이즈 분산 추정
    if noise_variance is None:
        noise_variance = estimate_noise_variance(gray)
    
    # Threshold 계산 (논문 공식)
    threshold = calculate_sssig_threshold(noise_variance, desired_accuracy)
    
    # 그라디언트 미리 계산 (재사용)
    img_float = gray.astype(np.float64)
    gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    
    info = {
        'noise_variance': noise_variance,
        'threshold': threshold,
        'desired_accuracy': desired_accuracy,
        'search_history': []
    }
    
    # 이진 탐색
    sizes = list(range(min_size, max_size + 1, step))
    left, right = 0, len(sizes) - 1
    best_size = max_size
    best_result = None
    found = False
    
    while left <= right:
        mid = (left + right) // 2
        size = sizes[mid]
        
        result = compute_sssig_map(
            gray, subset_size=size, spacing=spacing,
            threshold=threshold, noise_variance=noise_variance,
            gx=gx, gy=gy, mask=mask
        )
        
        # 기록
        info['search_history'].append({
            'size': size,
            'min_sssig': result.min,
            'predicted_accuracy': result.predicted_accuracy,
            'pass': len(result.bad_points) == 0 and result.min >= threshold
        })
        
        if len(result.points_y) > 0 and result.min >= threshold:
            best_size = size
            best_result = result
            found = True
            right = mid - 1
        else:
            left = mid + 1
    
    # 최종 결과 (찾지 못했으면 max_size로)
    if best_result is None:
        best_result = compute_sssig_map(
            gray, subset_size=max_size, spacing=spacing,
            threshold=threshold, noise_variance=noise_variance,
            gx=gx, gy=gy, mask=mask
        )
        best_size = max_size
    
    return best_size, found, best_result, info


def analyze_image_quality_for_dic(image: np.ndarray,
                                   desired_accuracy: float = 0.01) -> dict:
    """
    DIC 분석을 위한 이미지 품질 종합 분석
    
    Args:
        image: 입력 이미지
        desired_accuracy: 원하는 정확도 (pixels)
    
    Returns:
        종합 분석 결과 딕셔너리
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # 노이즈 분산 추정
    noise_variance = estimate_noise_variance(gray)
    
    # Threshold 계산
    threshold = calculate_sssig_threshold(noise_variance, desired_accuracy)
    
    # 최적 subset size 찾기
    optimal_size, found, result, search_info = find_optimal_subset_size(
        gray,
        desired_accuracy=desired_accuracy,
        noise_variance=noise_variance
    )
    
    # 결과 정리
    analysis = {
        'noise_variance': noise_variance,
        'noise_std': np.sqrt(noise_variance),
        'desired_accuracy': desired_accuracy,
        'calculated_threshold': threshold,
        'optimal_subset_size': optimal_size,
        'found_optimal': found,
        'predicted_accuracy': result.predicted_accuracy if result else None,
        'min_sssig': result.min if result else None,
        'mean_sssig': result.mean if result else None,
        'bad_points_count': len(result.bad_points) if result else 0,
        'search_history': search_info['search_history'],
        'recommendation': _generate_recommendation(
            found, optimal_size, result, noise_variance, desired_accuracy
        )
    }
    
    return analysis


def _generate_recommendation(found: bool, optimal_size: int, 
                              result: SSSIGResult, noise_variance: float,
                              desired_accuracy: float) -> str:
    """분석 결과 기반 권장 사항 생성"""
    if found and result.predicted_accuracy <= desired_accuracy:
        return (f"권장 subset size: {optimal_size}px\n"
                f"예상 정확도: {result.predicted_accuracy:.4f} px (목표: {desired_accuracy} px)\n"
                f"상태: 양호")
    elif found:
        return (f"권장 subset size: {optimal_size}px\n"
                f"예상 정확도: {result.predicted_accuracy:.4f} px (목표 미달)\n"
                f"제안: subset size 증가 또는 스페클 패턴 개선 필요")
    else:
        return (f"최적 subset size를 찾지 못함\n"
                f"현재 최대 size: {optimal_size}px\n"
                f"제안: 스페클 패턴 contrast 개선 또는 노이즈 저감 필요")


def warmup_numba():
    """Numba JIT 워밍업"""
    dummy = np.random.rand(100, 100).astype(np.float64)
    points_y = np.array([50], dtype=np.int64)
    points_x = np.array([50], dtype=np.int64)
    _compute_sssig_map_parallel(dummy, dummy, points_y, points_x, 10)
