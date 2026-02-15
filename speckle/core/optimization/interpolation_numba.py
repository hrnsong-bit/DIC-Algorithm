"""
Numba B-spline 보간 모듈

Bicubic (3차) 및 Biquintic (5차) B-spline 보간을 Numba JIT로 구현.
scipy.ndimage.map_coordinates와 동일한 결과를 생성하면서,
ICGN prange 병렬화에 직접 사용 가능.

구현 구조:
    1. Prefilter (spline_filter): scipy 재사용 — 이미지당 1회, 충분히 빠름
    2. 보간 평가: Numba nopython — POI당 수백 회 호출되는 핵심 경로

B-spline 보간 과정:
    (a) 원본 이미지 → prefilter → B-spline 계수 (scipy, 1회)
    (b) 임의 좌표 (y, x) → 계수에 대해 basis function 평가 → 보간값

References:
    - Unser, M. (1993). "B-spline signal processing: Part I/II"
      IEEE Trans. Signal Processing. (basis functions, prefilter poles)
    - Thévenaz, P., Blu, T., Unser, M. (2000). "Interpolation revisited"
      IEEE Trans. Medical Imaging. (B-spline interpolation theory)
"""

import numpy as np
from numba import jit, prange, float64, int64, types
from scipy.ndimage import spline_filter


# =============================================================================
#  1. B-spline Basis Functions (1D)
# =============================================================================

@jit(float64(float64), nopython=True, cache=True)
def _beta3(x):
    """
    Cubic B-spline basis β³(x)

    Support: [-2, 2], 4-tap

        β³(x) = 2/3 - |x|² + |x|³/2,     0 ≤ |x| < 1
               = (2 - |x|)³ / 6,           1 ≤ |x| < 2
               = 0,                          |x| ≥ 2
    """
    ax = abs(x)
    if ax < 1.0:
        return 2.0 / 3.0 - ax * ax + ax * ax * ax / 2.0
    elif ax < 2.0:
        t = 2.0 - ax
        return t * t * t / 6.0
    else:
        return 0.0


@jit(float64(float64), nopython=True, cache=True)
def _beta5(x):
    """
    Quintic B-spline basis β⁵(x)

    Support: [-3, 3], 6-tap

    Unser (1993), Table I:
        β⁵(x) = 11/20 - |x|²/2 + |x|⁴/4 - |x|⁵/12,         0 ≤ |x| < 1
               = 17/40 + 5|x|/8 - 7|x|²/4 + 5|x|³/4
                 - 3|x|⁴/8 + |x|⁵/24,                         1 ≤ |x| < 2
               = (3 - |x|)⁵ / 120,                             2 ≤ |x| < 3
               = 0,                                             |x| ≥ 3
    """
    ax = abs(x)
    if ax < 1.0:
        ax2 = ax * ax
        ax4 = ax2 * ax2
        ax5 = ax4 * ax
        return 11.0 / 20.0 - ax2 / 2.0 + ax4 / 4.0 - ax5 / 12.0
    elif ax < 2.0:
        ax2 = ax * ax
        ax3 = ax2 * ax
        ax4 = ax3 * ax
        ax5 = ax4 * ax
        return (17.0 / 40.0 + 5.0 * ax / 8.0 - 7.0 * ax2 / 4.0
                + 5.0 * ax3 / 4.0 - 3.0 * ax4 / 8.0 + ax5 / 24.0)
    elif ax < 3.0:
        t = 3.0 - ax
        return t * t * t * t * t / 120.0
    else:
        return 0.0


# =============================================================================
#  2. Prefilter (scipy 위임)
# =============================================================================
# prefilter는 이미지당 1회만 수행되므로 scipy를 그대로 사용.
# Numba로 재구현하더라도 성능 이점이 없고, 정확성 위험만 증가.

def prefilter_image(image, order=5):
    """
    B-spline prefilter 적용 (scipy 위임)

    이미지 → B-spline 계수 변환. 이미지당 1회만 호출.

    Args:
        image: 2D float64 배열 (그레이스케일)
        order: 3 (cubic) 또는 5 (quintic)

    Returns:
        B-spline 계수 배열 (입력과 동일 shape)
    """
    if order not in (3, 5):
        raise ValueError("order must be 3 or 5")
    return spline_filter(np.asarray(image, dtype=np.float64),
                         order=order, mode='constant')


# =============================================================================
#  3. 경계 인덱스 미러링 (SciPy 호환)
# =============================================================================
# SciPy ndimage의 map_coordinates는 mode='constant'에서도 B-spline 계수
# 접근 시 mirror reflection을 사용한다.
# 규칙: index < 0 → -index,  index ≥ N → 2*(N-1) - index
# 이를 정확히 재현해야 경계 근처에서 SciPy와 동일한 결과를 얻는다.

@jit(int64(int64, int64), nopython=True, cache=True)
def _mirror_index(idx, n):
    """
    Mirror boundary index (SciPy mode='constant' 호환)

    SciPy의 map_coordinates C 구현은 B-spline 계수 테이블 접근 시
    범위 밖 인덱스를 mirror 반사로 처리한다.

    Args:
        idx: 원래 인덱스 (음수 또는 >= n 가능)
        n: 배열 길이

    Returns:
        [0, n-1] 범위의 미러링된 인덱스
    """
    if idx < 0:
        idx = -idx
    if idx >= n:
        idx = 2 * (n - 1) - idx
    return idx


# =============================================================================
#  4. 2D B-spline 보간 (Numba)
# =============================================================================
# 핵심 성능 경로: ICGN 반복 루프 내에서 POI당 매 반복 호출

@jit(float64(float64[:, :], float64, float64), nopython=True, cache=True)
def _interp2d_cubic(coeffs, y, x):
    """
    단일 좌표 Bicubic B-spline 보간

    계수 배열에서 (y, x)에서의 보간값 계산.
    4×4 이웃 계수에 대해 separable 1D basis 적용.

    경계 외부: mirror reflection (SciPy map_coordinates 호환)
    """
    h = coeffs.shape[0]
    w = coeffs.shape[1]

    iy = int(np.floor(y))
    ix = int(np.floor(x))
    fy = y - iy
    fx = x - ix

    # 1D 가중치 (4-tap: offset -1 ~ +2)
    wy0 = _beta3(fy + 1.0)    # iy-1
    wy1 = _beta3(fy)           # iy
    wy2 = _beta3(fy - 1.0)    # iy+1
    wy3 = _beta3(fy - 2.0)    # iy+2

    wx0 = _beta3(fx + 1.0)    # ix-1
    wx1 = _beta3(fx)           # ix
    wx2 = _beta3(fx - 1.0)    # ix+1
    wx3 = _beta3(fx - 2.0)    # ix+2

    result = 0.0

    for ky in range(4):
        cy = iy - 1 + ky
        if ky == 0:
            wy = wy0
        elif ky == 1:
            wy = wy1
        elif ky == 2:
            wy = wy2
        else:
            wy = wy3

        if wy == 0.0:
            continue

        # 경계 미러링 (SciPy 호환)
        my = _mirror_index(int64(cy), int64(h))

        row_sum = 0.0
        for kx in range(4):
            cx = ix - 1 + kx
            if kx == 0:
                wx = wx0
            elif kx == 1:
                wx = wx1
            elif kx == 2:
                wx = wx2
            else:
                wx = wx3

            if wx == 0.0:
                continue

            # 경계 미러링 (SciPy 호환)
            mx = _mirror_index(int64(cx), int64(w))
            row_sum += coeffs[my, mx] * wx

        result += row_sum * wy

    return result


@jit(float64(float64[:, :], float64, float64), nopython=True, cache=True)
def _interp2d_quintic(coeffs, y, x):
    """
    단일 좌표 Biquintic B-spline 보간

    계수 배열에서 (y, x)에서의 보간값 계산.
    6×6 이웃 계수에 대해 separable 1D basis 적용.

    경계 외부: mirror reflection (SciPy map_coordinates 호환)
    """
    h = coeffs.shape[0]
    w = coeffs.shape[1]

    iy = int(np.floor(y))
    ix = int(np.floor(x))
    fy = y - iy
    fx = x - ix

    # 1D 가중치 (6-tap: offset -2 ~ +3)
    wy0 = _beta5(fy + 2.0)    # iy-2
    wy1 = _beta5(fy + 1.0)    # iy-1
    wy2 = _beta5(fy)           # iy
    wy3 = _beta5(fy - 1.0)    # iy+1
    wy4 = _beta5(fy - 2.0)    # iy+2
    wy5 = _beta5(fy - 3.0)    # iy+3

    wx0 = _beta5(fx + 2.0)    # ix-2
    wx1 = _beta5(fx + 1.0)    # ix-1
    wx2 = _beta5(fx)           # ix
    wx3 = _beta5(fx - 1.0)    # ix+1
    wx4 = _beta5(fx - 2.0)    # ix+2
    wx5 = _beta5(fx - 3.0)    # ix+3

    result = 0.0

    for ky in range(6):
        cy = iy - 2 + ky
        if ky == 0:
            wy = wy0
        elif ky == 1:
            wy = wy1
        elif ky == 2:
            wy = wy2
        elif ky == 3:
            wy = wy3
        elif ky == 4:
            wy = wy4
        else:
            wy = wy5

        if wy == 0.0:
            continue

        # 경계 미러링 (SciPy 호환)
        my = _mirror_index(int64(cy), int64(h))

        row_sum = 0.0
        for kx in range(6):
            cx = ix - 2 + kx
            if kx == 0:
                wx = wx0
            elif kx == 1:
                wx = wx1
            elif kx == 2:
                wx = wx2
            elif kx == 3:
                wx = wx3
            elif kx == 4:
                wx = wx4
            else:
                wx = wx5

            if wx == 0.0:
                continue

            # 경계 미러링 (SciPy 호환)
            mx = _mirror_index(int64(cx), int64(w))
            row_sum += coeffs[my, mx] * wx

        result += row_sum * wy

    return result


# =============================================================================
#  5. 배치 보간 (다수 좌표)
# =============================================================================
# ICGN의 process_poi에서 사용: 한 POI의 subset_size² 좌표를 한번에 보간

@jit(nopython=True, cache=True)
def interp2d_batch_cubic(coeffs, y_coords, x_coords):
    """
    다수 좌표에 대한 Bicubic B-spline 보간 (순차)

    Args:
        coeffs: prefilter된 B-spline 계수 (H, W)
        y_coords: y좌표 배열 (N,)
        x_coords: x좌표 배열 (N,)

    Returns:
        보간값 배열 (N,)
    """
    n = len(y_coords)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = _interp2d_cubic(coeffs, y_coords[i], x_coords[i])
    return result


@jit(nopython=True, cache=True)
def interp2d_batch_quintic(coeffs, y_coords, x_coords):
    """
    다수 좌표에 대한 Biquintic B-spline 보간 (순차)

    Args:
        coeffs: prefilter된 B-spline 계수 (H, W)
        y_coords: y좌표 배열 (N,)
        x_coords: x좌표 배열 (N,)

    Returns:
        보간값 배열 (N,)
    """
    n = len(y_coords)
    result = np.empty(n, dtype=np.float64)
    for i in range(n):
        result[i] = _interp2d_quintic(coeffs, y_coords[i], x_coords[i])
    return result


# =============================================================================
#  6. 경계 체크 함수
# =============================================================================

@jit(nopython=True, cache=True)
def is_inside_cubic(y, x, height, width):
    """Bicubic 보간에서 모든 이웃이 이미지 내부인지 확인"""
    margin = 2  # order//2 + 1 = 3//2 + 1 = 2
    return (y >= margin and y < height - margin
            and x >= margin and x < width - margin)


@jit(nopython=True, cache=True)
def is_inside_quintic(y, x, height, width):
    """Biquintic 보간에서 모든 이웃이 이미지 내부인지 확인"""
    margin = 3  # order//2 + 1 = 5//2 + 1 = 3
    return (y >= margin and y < height - margin
            and x >= margin and x < width - margin)


@jit(nopython=True, cache=True)
def is_inside_batch(y_coords, x_coords, height, width, order):
    """
    배치 경계 체크: 모든 좌표가 보간 가능 영역 내부인지 확인

    Args:
        y_coords, x_coords: 좌표 배열 (N,)
        height, width: 이미지 크기
        order: 3 또는 5

    Returns:
        True if 모든 좌표가 내부
    """
    if order == 3:
        margin = 2
    else:
        margin = 3

    for i in range(len(y_coords)):
        y = y_coords[i]
        x = x_coords[i]
        if y < margin or y >= height - margin:
            return False
        if x < margin or x >= width - margin:
            return False
    return True


# =============================================================================
#  7. 통합 인터페이스 (ICGN에서 사용)
# =============================================================================

@jit(nopython=True, cache=True)
def interp2d(coeffs, y_coords, x_coords, order):
    """
    통합 B-spline 보간 함수

    Args:
        coeffs: prefilter된 B-spline 계수 (H, W)
        y_coords: y좌표 배열 (N,)
        x_coords: x좌표 배열 (N,)
        order: 3 (cubic) 또는 5 (quintic)

    Returns:
        보간값 배열 (N,)
    """
    if order == 3:
        return interp2d_batch_cubic(coeffs, y_coords, x_coords)
    else:
        return interp2d_batch_quintic(coeffs, y_coords, x_coords)


# =============================================================================
#  8. Numba JIT 워밍업
# =============================================================================

def warmup_numba_interp():
    """Numba JIT 컴파일 워밍업"""
    dummy = np.random.rand(20, 20).astype(np.float64)
    coeffs3 = prefilter_image(dummy, order=3)
    coeffs5 = prefilter_image(dummy, order=5)
    y = np.array([10.5, 10.3], dtype=np.float64)
    x = np.array([10.5, 10.7], dtype=np.float64)
    interp2d(coeffs3, y, x, 3)
    interp2d(coeffs5, y, x, 5)
    is_inside_batch(y, x, 20, 20, 3)
    is_inside_batch(y, x, 20, 20, 5)
