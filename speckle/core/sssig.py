"""
SSSIG 계산 모듈 (v3.4.0)

변경 사항 (v3.4.0):
- estimate_noise_variance (단일 이미지 local_std 기반) 삭제
  → 단일 이미지에서 텍스처와 노이즈를 분리하는 것은 원리적으로 불가능.
    percentile 선택(0.5%)이 임의적이며 논문 리뷰에서 방어 불가.
- max(noise_var, 1.0) 클램프 삭제 → 물리적 근거 없음
- estimate_noise_from_pair: max 하한을 0.1 → 0.01로 완화 (고품질 카메라 대응)
- 노이즈가 필요하지만 제공되지 않은 경우 default_noise_variance=4.0 사용 + 경고
- analyze_image_quality_for_dic: noise_variance 필수 인자로 변경

이전 버전 (v3.3.0):
- compute_gradient: IC-GN과 동일한 Sobel ksize=5, /32.0
- calculate_sssig_threshold: 하드코딩 클램프 제거, 경고 로깅으로 전환
- compute_sssig_map: bad-point를 x/y 각각 threshold 기준으로 판정
- find_optimal_subset_size: 마스크 안전 선형 탐색

References:
- Pan et al. (2008) "Study on subset size selection in digital image
  correlation for speckle patterns" Optics Express
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from numba import jit, prange

from ..models.reports import BadPoint, SSSIGResult
from ..utils.logger import logger


# ===== 디폴트 노이즈 분산 =====
# 일반적인 8-bit 산업용 카메라 센서 노이즈 수준 (σ ≈ 2 GL)
DEFAULT_NOISE_VARIANCE = 4.0


# ===== Numba 병렬 SSSIG =====

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


# ===== 유틸: 그레이스케일 변환 =====

def _ensure_gray(image: np.ndarray) -> np.ndarray:
    """BGR → Gray 변환 (이미 gray면 그대로)"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def _ensure_gray_float(image: np.ndarray) -> np.ndarray:
    """Gray + float64 변환"""
    return _ensure_gray(image).astype(np.float64)


# ===== Gradient 계산 (IC-GN 일치) =====

def compute_gradient(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    이미지 gradient 계산 — IC-GN과 동일한 Sobel ksize=5, /32.0

    SSSIG와 IC-GN이 동일한 gradient를 사용해야
    σ(Δu) ≈ √[D(η) / SSSIG] 관계가 성립함 (Pan et al., 2008, Eq. 18-19).

    MIG용 gradient(mig.py, ksize=3, 정규화 없음)와는 의도적으로 다름.
    MIG는 상대적 품질 비교 지표이므로 gradient 스케일에 무관.
    """
    img_float = _ensure_gray_float(image)
    ksize = 5
    sobel_div = 32.0
    gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=ksize) / sobel_div
    gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=ksize) / sobel_div
    return gx, gy


# ===== 노이즈 추정 =====

def estimate_noise_from_pair(image1: np.ndarray, image2: np.ndarray,
                              roi: Optional[Tuple[int, int, int, int]] = None) -> float:
    """
    두 정지 이미지 차분으로 노이즈 분산 추정 (DIC Challenge 방식)

    D(η) = Var(diff) / 2

    동일한 시편을 동일 조건에서 연속 촬영한 두 이미지를 사용.
    두 이미지의 노이즈가 i.i.d.라는 가정 하에 정확한 추정이 가능.

    Args:
        image1, image2: 동일 조건에서 촬영한 두 이미지
        roi: 분석 영역 (x, y, w, h)

    Returns:
        노이즈 분산 D(η) [GL² 단위]
    """
    gray1 = _ensure_gray_float(image1)
    gray2 = _ensure_gray_float(image2)

    if roi is not None:
        x, y, w, h = roi
        gray1 = gray1[y:y+h, x:x+w]
        gray2 = gray2[y:y+h, x:x+w]

    diff = gray1 - gray2
    diff = diff - np.mean(diff)  # DC 성분 제거

    noise_variance = float(np.var(diff) / 2.0)

    return max(noise_variance, 0.01)  # 수치 안전 하한 (고품질 카메라 대응)


def _resolve_default_noise(noise_variance: Optional[float]) -> float:
    """
    노이즈 분산이 None이면 디폴트 값 반환 + 경고.

    내부 유틸리티. compute_sssig_map, find_optimal_subset_size 등에서
    noise_variance=None으로 호출될 때 일관된 처리를 위해 사용.
    """
    if noise_variance is not None:
        return noise_variance

    logger.warning(
        f"노이즈 분산 미지정 → 디폴트 {DEFAULT_NOISE_VARIANCE} GL² 사용. "
        f"정확한 분석을 위해 pair 이미지 또는 카메라 스펙 기반 값을 지정하세요.")
    return DEFAULT_NOISE_VARIANCE


# ===== Threshold 계산 =====

def calculate_sssig_threshold(noise_variance: float,
                               desired_accuracy: float = 0.01) -> float:
    """
    논문 Eq. 18, 19 기반 SSSIG threshold 계산

    σ(Δu) ≈ √[D(η) / SSSIG]
    → SSSIG ≥ D(η) / σ_target²

    Args:
        noise_variance: 이미지 노이즈 분산 D(η) [GL²]
        desired_accuracy: 원하는 변위 측정 정확도 [pixels]

    Returns:
        SSSIG threshold
    """
    if desired_accuracy <= 0:
        desired_accuracy = 0.01

    threshold = noise_variance / (desired_accuracy ** 2)

    if threshold < 1e3:
        logger.warning(
            f"SSSIG threshold가 매우 낮음: {threshold:.2e} "
            f"(noise_var={noise_variance:.2f}, accuracy={desired_accuracy}). "
            f"노이즈 추정을 확인하세요.")
    elif threshold > 1e8:
        logger.warning(
            f"SSSIG threshold가 매우 높음: {threshold:.2e} "
            f"(noise_var={noise_variance:.2f}, accuracy={desired_accuracy}). "
            f"노이즈가 과대추정되었을 수 있습니다.")

    return threshold


# ===== 예측 정확도 =====

def predict_displacement_accuracy(sssig: float, noise_variance: float) -> float:
    """
    논문 Eq. 18, 19: SSSIG와 노이즈 분산으로 예상 정확도 계산

    Returns:
        예상 변위 측정 정확도 [pixels]
    """
    if sssig <= 0:
        return float('inf')
    return float(np.sqrt(noise_variance / sssig))


# ===== 단일 POI SSSIG =====

def compute_sssig(image: np.ndarray, center: Tuple[int, int],
                  subset_size: int, gx: np.ndarray = None,
                  gy: np.ndarray = None) -> Tuple[float, float]:
    """
    단일 POI의 SSSIG 계산 (x, y 방향 분리)

    Returns:
        (sssig_x, sssig_y) 튜플
    """
    if gx is None or gy is None:
        gx, gy = compute_gradient(image)

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


# ===== 전체 SSSIG 맵 =====

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
    전체 ROI에 대한 SSSIG 맵 계산

    bad-point 판정 (Pan et al., 2008, Eq. 18-19):
        threshold = D(η) / σ_target² 는 각 방향 독립 기준.
        sssig_x < threshold OR sssig_y < threshold 이면 불량.
    """
    gray = _ensure_gray(image).copy()
    h, w = gray.shape
    half = subset_size // 2

    # 노이즈 분산 결정
    noise_variance = _resolve_default_noise(noise_variance)

    # Threshold 계산
    if threshold is None:
        threshold = calculate_sssig_threshold(noise_variance, desired_accuracy)

    # Gradient 계산
    if gx is None or gy is None:
        gx, gy = compute_gradient(gray)

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

    sssig_values = sssig_x + sssig_y

    mean_sssig = float(np.mean(sssig_values))
    min_sssig = float(np.min(sssig_values))
    max_sssig = float(np.max(sssig_values))

    # 예상 정확도 (방향별 worst-case)
    min_sssig_x = float(np.min(sssig_x)) if len(sssig_x) > 0 else 0.0
    min_sssig_y = float(np.min(sssig_y)) if len(sssig_y) > 0 else 0.0
    accuracy_x = predict_displacement_accuracy(min_sssig_x, noise_variance)
    accuracy_y = predict_displacement_accuracy(min_sssig_y, noise_variance)
    predicted_accuracy = max(accuracy_x, accuracy_y)

    # bad-point 판정: 방향별 독립 기준
    bad_mask = (sssig_x < threshold) | (sssig_y < threshold)
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
        valid_idx = ((y_indices >= 0) & (y_indices < len(y_coords)) &
                     (x_indices >= 0) & (x_indices < len(x_coords)))
        sssig_map[y_indices[valid_idx], x_indices[valid_idx]] = \
            sssig_values[valid_idx]

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


# ===== 최적 Subset Size 탐색 =====

def find_optimal_subset_size(image: np.ndarray,
                             spacing: int = 10,
                             desired_accuracy: float = 0.01,
                             noise_variance: float = None,
                             min_size: int = 11,
                             max_size: int = 61,
                             step: int = 2,
                             mask: np.ndarray = None
                             ) -> Tuple[int, bool, Optional[SSSIGResult], dict]:
    """
    최적 subset size 탐색 (선형 탐색)

    Returns:
        (optimal_size, found, result, info)
    """
    gray = _ensure_gray(image).copy()

    noise_variance = _resolve_default_noise(noise_variance)
    threshold = calculate_sssig_threshold(noise_variance, desired_accuracy)

    gx, gy = compute_gradient(gray)

    info = {
        'noise_variance': noise_variance,
        'threshold': threshold,
        'desired_accuracy': desired_accuracy,
        'search_history': []
    }

    sizes = list(range(min_size, max_size + 1, step))
    best_size = max_size
    best_result = None
    found = False

    for size in sizes:
        result = compute_sssig_map(
            gray, subset_size=size, spacing=spacing,
            threshold=threshold, noise_variance=noise_variance,
            gx=gx, gy=gy, mask=mask
        )

        passed = (len(result.points_y) > 0
                  and len(result.bad_points) == 0)

        info['search_history'].append({
            'size': size,
            'min_sssig': result.min,
            'n_bad': len(result.bad_points),
            'predicted_accuracy': result.predicted_accuracy,
            'pass': passed,
        })

        if passed and not found:
            best_size = size
            best_result = result
            found = True
            break

    if best_result is None:
        best_result = compute_sssig_map(
            gray, subset_size=max_size, spacing=spacing,
            threshold=threshold, noise_variance=noise_variance,
            gx=gx, gy=gy, mask=mask
        )
        best_size = max_size

    return best_size, found, best_result, info


# ===== 종합 분석 =====

def analyze_image_quality_for_dic(image: np.ndarray,
                                   noise_variance: float,
                                   desired_accuracy: float = 0.01) -> dict:
    """
    DIC 분석을 위한 이미지 품질 종합 분석

    Args:
        image: 분석 대상 이미지
        noise_variance: 노이즈 분산 D(η) [GL²].
                        pair 이미지 또는 카메라 스펙에서 결정하여 전달.
        desired_accuracy: 원하는 변위 정확도 [pixels]
    """
    gray = _ensure_gray(image).copy()

    threshold = calculate_sssig_threshold(noise_variance, desired_accuracy)

    optimal_size, found, result, search_info = find_optimal_subset_size(
        gray,
        desired_accuracy=desired_accuracy,
        noise_variance=noise_variance
    )

    return {
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


def _generate_recommendation(found: bool, optimal_size: int,
                              result: SSSIGResult, noise_variance: float,
                              desired_accuracy: float) -> str:
    """분석 결과 기반 권장 사항 생성"""
    if found and result.predicted_accuracy <= desired_accuracy:
        return (f"권장 subset size: {optimal_size}px\n"
                f"예상 정확도: {result.predicted_accuracy:.4f} px "
                f"(목표: {desired_accuracy} px)\n"
                f"상태: 양호")
    elif found:
        return (f"권장 subset size: {optimal_size}px\n"
                f"예상 정확도: {result.predicted_accuracy:.4f} px (목표 미달)\n"
                f"제안: subset size 증가 또는 스페클 패턴 개선 필요")
    else:
        return (f"최적 subset size를 찾지 못함\n"
                f"현재 최대 size: {optimal_size}px\n"
                f"제안: 스페클 패턴 contrast 개선 또는 노이즈 저감 필요")


# ===== Numba 워밍업 =====

def warmup_numba():
    """Numba JIT 워밍업"""
    dummy = np.random.rand(100, 100).astype(np.float64)
    points_y = np.array([50], dtype=np.int64)
    points_x = np.array([50], dtype=np.int64)
    _compute_sssig_map_parallel(dummy, dummy, points_y, points_x, 10)