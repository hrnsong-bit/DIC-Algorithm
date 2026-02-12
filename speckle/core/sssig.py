"""
SSSIG 계산 모듈

References:
- Pan et al. (2008) "Study on subset size selection in digital image
  correlation for speckle patterns" Optics Express
- Pan et al. (2009) "Noise estimation and its effect on accuracy..."
  Strain, 45(6)
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from numba import jit, prange

from ..models.reports import BadPoint, SSSIGResult


# =====================================================================
#  Numba 커널
# =====================================================================

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


# =====================================================================
#  Gradient 유틸 — SSSIG와 IC-GN 공통 사양
# =====================================================================

# IC-GN(icgn.py)과 동일한 Sobel 사양을 사용해야
# σ(Δu) ≈ √[D(η)/SSSIG] 공식이 정확히 성립합니다.
# 변경 시 icgn.py `_compute_gradient`도 반드시 동기화하세요.
GRADIENT_KSIZE = 5
GRADIENT_DIV = 32.0          # 2^(2*3 - 1) for ksize=5 실무 정규화


def compute_gradient(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    SSSIG 및 IC-GN 공용 gradient 계산

    Returns:
        (gx, gy): Sobel ksize=5, /32.0 정규화 적용
    """
    img_float = image.astype(np.float64)
    gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=GRADIENT_KSIZE) / GRADIENT_DIV
    gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=GRADIENT_KSIZE) / GRADIENT_DIV
    return gx, gy


# =====================================================================
#  노이즈 분산 추정
# =====================================================================

def estimate_noise_from_pair(image1: np.ndarray, image2: np.ndarray,
                              roi: Optional[Tuple[int, int, int, int]] = None
                              ) -> float:
    """
    두 정지 이미지 차분으로 노이즈 **분산** D(η) 추정 (권장, 1순위)

    동일 조건·동일 위치에서 촬영한 두 장의 이미지를 빼면
    스페클 텍스처는 상쇄되고 순수 카메라 노이즈만 남습니다.
    SEM DIC Challenge, DIC Challenge 2.0 에서 사용하는 표준 방법입니다.

    σ_noise = std(image1 - image2) / √2
    D(η)    = σ_noise²

    Args:
        image1, image2: 동일 조건에서 촬영한 두 이미지
        roi: 분석 영역 (x, y, w, h). None이면 전체 이미지

    Returns:
        노이즈 분산 D(η)  (GL² 단위)
    """
    gray1 = _ensure_gray_float(image1)
    gray2 = _ensure_gray_float(image2)

    if roi is not None:
        x, y, w, h = roi
        gray1 = gray1[y:y+h, x:x+w]
        gray2 = gray2[y:y+h, x:x+w]

    diff = gray1 - gray2
    diff = diff - np.mean(diff)            # DC 성분 제거

    noise_std = np.std(diff) / np.sqrt(2)
    noise_var = noise_std ** 2

    # 안전 하한: 완전 무노이즈는 비현실적
    return max(noise_var, 0.1)


def estimate_noise_variance(image: np.ndarray,
                             method: str = 'local_std') -> float:
    """
    단일 이미지에서 노이즈 분산 D(η) 추정 (2순위 — pair가 없을 때 사용)

    ⚠ 단일 이미지 추정은 본질적으로 스페클 텍스처와 노이즈를
       완벽히 분리할 수 없습니다.  가능하면 estimate_noise_from_pair를
       사용하세요.

    Args:
        image: 그레이스케일 이미지
        method:
            'local_std'  — 5×5 로컬분산의 하위 5‰(permil).  (기본)
            'laplacian'  — Robust Median Estimator. 스페클에서 과대추정 경향.
            'fixed'      — 8-bit 카메라 일반 기본값 4.0 반환.

    Returns:
        노이즈 분산 D(η)  (GL² 단위)
    """
    gray = _ensure_gray_float(image)

    if method == 'laplacian':
        # Donoho & Johnstone (1994) — 스페클 텍스처를 노이즈로 오인 가능
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sigma = np.median(np.abs(laplacian)) / 0.6745
        return max(sigma ** 2, 0.1)

    elif method == 'local_std':
        kernel_size = 5
        local_mean = cv2.blur(gray, (kernel_size, kernel_size))
        local_sq_mean = cv2.blur(gray ** 2, (kernel_size, kernel_size))
        local_var = local_sq_mean - local_mean ** 2

        valid_var = local_var[local_var > 0]
        if len(valid_var) == 0:
            return 4.0

        # 하위 0.5% (5‰) — 기존 5%보다 보수적으로 평탄영역만 선택
        noise_var = float(np.percentile(valid_var, 0.5))
        return max(noise_var, 0.1)

    else:  # 'fixed'
        return 4.0


# =====================================================================
#  SSSIG Threshold 계산
# =====================================================================

def calculate_sssig_threshold(noise_variance: float,
                               desired_accuracy: float = 0.01) -> float:
    """
    Pan (2008) Eq. 18, 19 기반 SSSIG threshold 계산

        σ(Δu) ≈ √[ D(η) / SSSIG ]
        → SSSIG_min  ≥  D(η) / σ²_target

    Args:
        noise_variance: 이미지 노이즈 분산 D(η) (GL²)
        desired_accuracy: 원하는 변위 정확도 (pixels)

    Returns:
        SSSIG threshold (하한 없음 — 물리량 그대로 반환)
    """
    if desired_accuracy <= 0:
        desired_accuracy = 0.01

    threshold = noise_variance / (desired_accuracy ** 2)

    # ── 클램핑 제거 ──
    # 기존 max(1e4) / min(1e7)은 이론 근거가 불분명하여 제거.
    # 대신 threshold가 비정상적으로 낮거나 높으면 호출부에서
    # warning을 출력하도록 assessor.py에서 처리합니다.
    return threshold


def predict_displacement_accuracy(sssig: float,
                                   noise_variance: float) -> float:
    """
    Pan (2008) Eq. 18, 19: SSSIG와 D(η)로 예상 변위 정확도 계산

    Args:
        sssig: SSSIG 값 (한 방향, 또는 총합)
        noise_variance: 노이즈 분산 D(η)

    Returns:
        예상 변위 표준편차 σ(Δu) (pixels)
    """
    if sssig <= 0:
        return float('inf')
    return np.sqrt(noise_variance / sssig)


# =====================================================================
#  단일 POI SSSIG
# =====================================================================

def compute_sssig(image: np.ndarray, center: Tuple[int, int],
                  subset_size: int,
                  gx: np.ndarray = None,
                  gy: np.ndarray = None) -> Tuple[float, float]:
    """
    단일 POI의 SSSIG (x, y 방향 분리)

    Args:
        center: (cy, cx) — row, col 순서

    Returns:
        (sssig_x, sssig_y)
    """
    if gx is None or gy is None:
        gray = _ensure_gray_float(image)
        gx, gy = compute_gradient(gray)

    half = subset_size // 2
    cy, cx = center
    h, w = gx.shape

    y1, y2 = max(0, cy - half), min(h, cy + half + 1)
    x1, x2 = max(0, cx - half), min(w, cx + half + 1)

    gx_sub = gx[y1:y2, x1:x2]
    gy_sub = gy[y1:y2, x1:x2]

    return float(np.sum(gx_sub ** 2)), float(np.sum(gy_sub ** 2))


# =====================================================================
#  SSSIG Map (전체 ROI)
# =====================================================================

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

    Args:
        image: 입력 이미지 (grayscale or BGR)
        subset_size: subset 크기 (pixels, 홀수)
        spacing: POI 간격 (pixels)
        threshold: SSSIG threshold (None → 자동 계산)
        noise_variance: D(η) (None → 단일이미지 추정)
        desired_accuracy: 목표 변위 정확도 (pixels)
        gx, gy: 사전 계산 gradient (없으면 내부에서 계산)
        mask: ROI 마스크 (0=제외, >0=포함)

    Returns:
        SSSIGResult
    """
    gray = _ensure_gray(image)
    h, w = gray.shape
    half = subset_size // 2

    # ── 노이즈 분산 ──
    if noise_variance is None:
        noise_variance = estimate_noise_variance(gray)

    # ── Threshold ──
    if threshold is None:
        threshold = calculate_sssig_threshold(noise_variance, desired_accuracy)

    # ── Gradient (IC-GN과 동일 사양) ──
    if gx is None or gy is None:
        gx, gy = compute_gradient(gray)

    # ── POI 그리드 ──
    margin = half + 1
    y_coords = np.arange(margin, h - margin, spacing)
    x_coords = np.arange(margin, w - margin, spacing)

    if len(y_coords) == 0 or len(x_coords) == 0:
        return _empty_sssig_result(subset_size, spacing,
                                   noise_variance, threshold)

    yy, xx = np.meshgrid(y_coords, x_coords, indexing='ij')
    all_points_y = yy.ravel()
    all_points_x = xx.ravel()

    # ── 마스크 필터 ──
    if mask is not None:
        valid = mask[all_points_y, all_points_x] > 0
        points_y = all_points_y[valid].astype(np.int64)
        points_x = all_points_x[valid].astype(np.int64)
    else:
        points_y = all_points_y.astype(np.int64)
        points_x = all_points_x.astype(np.int64)

    if len(points_y) == 0:
        return _empty_sssig_result(subset_size, spacing,
                                   noise_variance, threshold)

    # ── SSSIG 계산 ──
    sssig_x, sssig_y = _compute_sssig_map_parallel(
        gx.astype(np.float64), gy.astype(np.float64),
        points_y, points_x, half
    )
    sssig_total = sssig_x + sssig_y

    # ── 통계 ──
    mean_sssig = float(np.mean(sssig_total))
    min_sssig = float(np.min(sssig_total))
    max_sssig = float(np.max(sssig_total))

    # ── 예측 정확도 (x, y 분리 → worst-case) ──
    pred_acc_x = predict_displacement_accuracy(np.min(sssig_x), noise_variance)
    pred_acc_y = predict_displacement_accuracy(np.min(sssig_y), noise_variance)
    predicted_accuracy = max(pred_acc_x, pred_acc_y)

    # ── Bad Points (x/y 분리 OR 판정) ──
    # Pan (2008) 원논문: σ(Δu) ∝ 1/√Σ(∂f/∂x)², 방향별 독립.
    # x 또는 y 어느 한쪽이라도 부족하면 해당 방향 변위가 부정확.
    threshold_per_dir = threshold / 2.0
    bad_mask = (sssig_x < threshold_per_dir) | (sssig_y < threshold_per_dir)

    bad_indices = np.where(bad_mask)[0]
    bad_points = [
        BadPoint(y=int(points_y[i]), x=int(points_x[i]),
                 sssig=float(sssig_total[i]))
        for i in bad_indices
    ]

    # ── 2D 맵 ──
    if mask is None:
        sssig_map = sssig_total.reshape(len(y_coords), len(x_coords))
    else:
        sssig_map = np.full((len(y_coords), len(x_coords)), -1.0)
        y_idx = (points_y - margin) // spacing
        x_idx = (points_x - margin) // spacing
        ok = ((y_idx >= 0) & (y_idx < len(y_coords)) &
              (x_idx >= 0) & (x_idx < len(x_coords)))
        sssig_map[y_idx[ok], x_idx[ok]] = sssig_total[ok]

    return SSSIGResult(
        map=sssig_map,
        mean=mean_sssig, min=min_sssig, max=max_sssig,
        bad_points=bad_points,
        points_y=points_y, points_x=points_x,
        subset_size=subset_size, spacing=spacing,
        noise_variance=noise_variance,
        threshold=threshold,
        predicted_accuracy=predicted_accuracy
    )


# =====================================================================
#  최적 Subset Size 탐색
# =====================================================================

def find_optimal_subset_size(
        image: np.ndarray,
        spacing: int = 10,
        desired_accuracy: float = 0.01,
        noise_variance: float = None,
        min_size: int = 11,
        max_size: int = 61,
        step: int = 2,
        mask: np.ndarray = None
) -> Tuple[int, bool, Optional[SSSIGResult], dict]:
    """
    최적 subset size 탐색 — 선형 스캔 (작은 size부터)

    [11, 13, 15, ..., 61] 범위는 26개뿐이므로 이진 탐색 대비
    성능 차이가 무의미하며, 마스크 사용 시 단조성 미보장
    문제를 원천 회피합니다.

    Returns:
        (optimal_size, found, sssig_result, info)
    """
    gray = _ensure_gray(image)

    if noise_variance is None:
        noise_variance = estimate_noise_variance(gray)

    threshold = calculate_sssig_threshold(noise_variance, desired_accuracy)

    # Gradient 1회 계산 (모든 size에서 재사용)
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

        passes = (len(result.points_y) > 0
                  and result.n_bad_points == 0
                  and result.min >= threshold)

        info['search_history'].append({
            'size': size,
            'min_sssig': result.min,
            'predicted_accuracy': result.predicted_accuracy,
            'pass': passes
        })

        if passes and not found:
            best_size = size
            best_result = result
            found = True
            break                       # 최소 통과 size = 최적

    # 못 찾으면 max_size 결과 반환
    if best_result is None:
        best_result = compute_sssig_map(
            gray, subset_size=max_size, spacing=spacing,
            threshold=threshold, noise_variance=noise_variance,
            gx=gx, gy=gy, mask=mask
        )
        best_size = max_size

    return best_size, found, best_result, info


# =====================================================================
#  종합 분석 (편의 함수)
# =====================================================================

def analyze_image_quality_for_dic(image: np.ndarray,
                                   desired_accuracy: float = 0.01,
                                   noise_variance: float = None) -> dict:
    """
    DIC 분석을 위한 이미지 품질 종합 분석

    Args:
        image: 입력 이미지
        desired_accuracy: 목표 변위 정확도 (pixels)
        noise_variance: 외부에서 구한 D(η). None이면 단일이미지 추정.

    Returns:
        종합 분석 dict
    """
    gray = _ensure_gray(image)

    if noise_variance is None:
        noise_variance = estimate_noise_variance(gray)

    threshold = calculate_sssig_threshold(noise_variance, desired_accuracy)

    optimal_size, found, result, search_info = find_optimal_subset_size(
        gray, desired_accuracy=desired_accuracy,
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


# =====================================================================
#  내부 유틸
# =====================================================================

def _ensure_gray(image: np.ndarray) -> np.ndarray:
    """BGR → Grayscale (uint8 유지)"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.copy()


def _ensure_gray_float(image: np.ndarray) -> np.ndarray:
    """BGR → Grayscale float64"""
    if len(image.shape) == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)
    return image.astype(np.float64)


def _empty_sssig_result(subset_size, spacing, noise_variance, threshold):
    """POI 0개일 때 안전 반환"""
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


def _generate_recommendation(found, optimal_size, result,
                              noise_variance, desired_accuracy):
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


def warmup_numba():
    """Numba JIT 워밍업"""
    dummy = np.random.rand(100, 100).astype(np.float64)
    pts_y = np.array([50], dtype=np.int64)
    pts_x = np.array([50], dtype=np.int64)
    _compute_sssig_map_parallel(dummy, dummy, pts_y, pts_x, 10)
