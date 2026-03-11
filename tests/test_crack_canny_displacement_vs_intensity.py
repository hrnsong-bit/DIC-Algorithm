"""
tests/test_crack_boundary_precision.py
크랙 경계 정밀도 비교: 변위장 Canny vs Mean+kσ contour
- 평가 기준: IoU가 아닌 경계선 거리 오차 (평균, Hausdorff)
- 변위장 gradient의 크랙 팁 도달 거리 비교
"""
import sys
import time
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage
from scipy.spatial import distance
from scipy.interpolate import griddata
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / 'synthetic_crack_data'
TEST_OUTPUT_ROOT = PROJECT_ROOT / 'tests' / '_outputs'
OUTPUT_DIR = TEST_OUTPUT_ROOT / 'output_boundary_precision'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════
# 유틸리티
# ══════════════════════════════════════════════════

def _imread_unicode(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지 로딩 실패: {path}")
    return img


def displacement_to_grid(poi_x, poi_y, values, img_shape):
    grid = np.full(img_shape, np.nan, dtype=np.float64)
    for i in range(len(poi_x)):
        x, y = int(round(poi_x[i])), int(round(poi_y[i]))
        if 0 <= y < img_shape[0] and 0 <= x < img_shape[1]:
            grid[y, x] = values[i]
    return grid


def interpolate_grid(grid):
    h, w = grid.shape
    valid = ~np.isnan(grid)
    if valid.sum() < 4:
        return np.zeros_like(grid)
    ys, xs = np.where(valid)
    vals = grid[valid]
    yy, xx = np.mgrid[0:h, 0:w]
    interp = griddata((xs, ys), vals, (xx, yy), method='linear', fill_value=np.nan)
    still_nan = np.isnan(interp)
    if still_nan.any():
        nearest = griddata((xs, ys), vals, (xx, yy), method='nearest')
        interp[still_nan] = nearest[still_nan]
    return interp


# ══════════════════════════════════════════════════
# GT 경계선 추출
# ══════════════════════════════════════════════════

def extract_gt_boundary(gt_mask):
    """GT 마스크에서 경계선 좌표 추출"""
    gt_uint8 = (gt_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(gt_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_pts = []
    for cnt in contours:
        pts = cnt.squeeze()
        if pts.ndim == 2:
            all_pts.append(pts)
    if all_pts:
        return np.vstack(all_pts).astype(np.float64)
    return np.array([]).reshape(0, 2)


def extract_gt_tip(gt_mask):
    """GT 크랙 팁 위치 (가장 오른쪽 크랙 픽셀의 x좌표)"""
    ys, xs = np.where(gt_mask)
    if len(xs) == 0:
        return None
    tip_idx = np.argmax(xs)
    return (float(xs[tip_idx]), float(ys[tip_idx]))


# ══════════════════════════════════════════════════
# 경계 검출 방법들
# ══════════════════════════════════════════════════

def method_mean_k_sigma_contour(roi, k=1.5, sigma_blur=0):
    """Mean+kσ → contour 경계선"""
    if sigma_blur > 0:
        ksize = max(3, int(np.ceil(sigma_blur * 3)) * 2 + 1)
        blurred = cv2.GaussianBlur(roi, (ksize, ksize), sigma_blur)
    else:
        blurred = roi.copy()
    mu = float(blurred.mean())
    std = float(blurred.std())
    thr = mu - k * std
    binary = (blurred < thr).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    all_pts = []
    for cnt in contours:
        pts = cnt.squeeze()
        if pts.ndim == 2:
            all_pts.append(pts)
    if all_pts:
        return np.vstack(all_pts).astype(np.float64)
    return np.array([]).reshape(0, 2)


def method_disp_gradient_threshold(disp_field, gradient_threshold, sigma_blur=1.0):
    """
    변위장 gradient 크기가 임계값 이상인 픽셀 → 크랙 경계선
    """
    # NaN 처리
    field = disp_field.copy()
    nan_mask = np.isnan(field)
    if nan_mask.any():
        field[nan_mask] = np.nanmedian(field)

    # 블러
    if sigma_blur > 0:
        ksize = max(3, int(np.ceil(sigma_blur * 3)) * 2 + 1)
        field_f = cv2.GaussianBlur(field.astype(np.float32), (ksize, ksize), sigma_blur)
    else:
        field_f = field.astype(np.float32)

    # Sobel gradient
    grad_x = cv2.Sobel(field_f, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(field_f, cv2.CV_64F, 0, 1, ksize=3)
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)

    # 임계값 적용
    edge_mask = (grad_mag > gradient_threshold).astype(np.uint8) * 255

    # 세선화 (thinning)
    edge_mask = cv2.ximgproc.thinning(edge_mask) if hasattr(cv2, 'ximgproc') else edge_mask

    # NMS (Non-Maximum Suppression) 대안: Canny 대신 직접 NMS
    # grad 방향에 따라 최대값만 남김
    angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
    angle[angle < 0] += 180
    h, w = grad_mag.shape
    nms = np.zeros_like(grad_mag)
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if grad_mag[i, j] < gradient_threshold:
                continue
            a = angle[i, j]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = grad_mag[i, j - 1], grad_mag[i, j + 1]
            elif 22.5 <= a < 67.5:
                n1, n2 = grad_mag[i - 1, j + 1], grad_mag[i + 1, j - 1]
            elif 67.5 <= a < 112.5:
                n1, n2 = grad_mag[i - 1, j], grad_mag[i + 1, j]
            else:
                n1, n2 = grad_mag[i - 1, j - 1], grad_mag[i + 1, j + 1]
            if grad_mag[i, j] >= n1 and grad_mag[i, j] >= n2:
                nms[i, j] = grad_mag[i, j]

    boundary = (nms > gradient_threshold).astype(np.uint8) * 255
    ys, xs = np.where(boundary > 0)
    if len(xs) > 0:
        return np.column_stack([xs, ys]).astype(np.float64), grad_mag, nms
    return np.array([]).reshape(0, 2), grad_mag, nms


def method_disp_canny(disp_field, sigma_blur=1.0, low_ratio=0.3, high_ratio=0.7):
    """변위장에 Canny 적용 → 경계선 좌표"""
    field = disp_field.copy()
    nan_mask = np.isnan(field)
    if nan_mask.any():
        field[nan_mask] = np.nanmedian(field)
    fmin, fmax = field.min(), field.max()
    if fmax - fmin < 1e-10:
        return np.array([]).reshape(0, 2)
    normalized = ((field - fmin) / (fmax - fmin) * 255).astype(np.uint8)
    ksize = max(3, int(np.ceil(sigma_blur * 3)) * 2 + 1)
    blurred = cv2.GaussianBlur(normalized, (ksize, ksize), sigma_blur)
    v = np.median(blurred)
    low = int(max(1, low_ratio * v))
    high = int(min(255, high_ratio * v))
    if low >= high:
        high = low + 1
    edges = cv2.Canny(blurred, low, high)

    # 길이 필터 (짧은 성분 제거)
    labeled, n_comp = ndimage.label(edges)
    for i in range(1, n_comp + 1):
        if np.sum(labeled == i) < 5:
            edges[labeled == i] = 0

    ys, xs = np.where(edges > 0)
    if len(xs) > 0:
        return np.column_stack([xs, ys]).astype(np.float64)
    return np.array([]).reshape(0, 2)


# ══════════════════════════════════════════════════
# 경계 정밀도 평가
# ══════════════════════════════════════════════════

def evaluate_boundary(detected_pts, gt_pts):
    """
    경계선 정밀도 평가

    Returns:
        det_to_gt_mean: 검출→GT 평균 거리 (정밀도)
        gt_to_det_mean: GT→검출 평균 거리 (완전성)
        hausdorff: 최악 오차
        det_to_gt_all: 모든 검출점의 GT 거리
        gt_to_det_all: 모든 GT점의 검출 거리
    """
    if len(detected_pts) == 0 or len(gt_pts) == 0:
        return {
            'det_to_gt_mean': float('inf'),
            'gt_to_det_mean': float('inf'),
            'hausdorff': float('inf'),
            'det_to_gt_median': float('inf'),
            'gt_to_det_median': float('inf'),
            'det_to_gt_all': np.array([]),
            'gt_to_det_all': np.array([]),
            'n_detected': len(detected_pts),
            'n_gt': len(gt_pts),
        }

    d2g = distance.cdist(detected_pts, gt_pts).min(axis=1)
    g2d = distance.cdist(gt_pts, detected_pts).min(axis=1)

    return {
        'det_to_gt_mean': float(np.mean(d2g)),
        'gt_to_det_mean': float(np.mean(g2d)),
        'hausdorff': max(float(np.max(d2g)), float(np.max(g2d))),
        'det_to_gt_median': float(np.median(d2g)),
        'gt_to_det_median': float(np.median(g2d)),
        'det_to_gt_all': d2g,
        'gt_to_det_all': g2d,
        'n_detected': len(detected_pts),
        'n_gt': len(gt_pts),
    }


def compute_tip_reach(detected_pts, gt_tip, crack_direction='horizontal'):
    """
    검출된 경계선이 크랙 팁에 얼마나 가까이 도달했는지 계산
    """
    if len(detected_pts) == 0 or gt_tip is None:
        return {'tip_error': float('inf'), 'max_x': 0}

    if crack_direction == 'horizontal':
        max_x = float(np.max(detected_pts[:, 0]))
        tip_error = abs(gt_tip[0] - max_x)
    else:
        max_y = float(np.max(detected_pts[:, 1]))
        tip_error = abs(gt_tip[1] - max_y)

    return {
        'tip_error': tip_error,
        'max_x': max_x if crack_direction == 'horizontal' else float(np.max(detected_pts[:, 1])),
        'gt_tip_x': gt_tip[0],
    }


# ══════════════════════════════════════════════════
# 메인 실험
# ══════════════════════════════════════════════════

def run_experiment():
    print("=" * 70)
    print("크랙 경계 정밀도 비교: 변위장 vs Mean+kσ")
    print("평가 기준: 경계선 거리 오차 + 크랙 팁 도달 거리")
    print("=" * 70)

    # ── 1. 데이터 로드 ──
    print("\n[1] 데이터 로드...")
    def_img = _imread_unicode(DATA_DIR / 'deformed.tiff')
    ref_img = _imread_unicode(DATA_DIR / 'reference.tiff')
    gt_mask = np.load(str(DATA_DIR / 'crack_mask.npy')).astype(bool)
    H, W = def_img.shape
    print(f"  이미지: {W}×{H}px, GT 크랙: {gt_mask.sum()}px")

    # ── 2. DIC 실행 ──
    print("\n[2] DIC 실행...")
    from speckle.core.initial_guess import compute_fft_cc
    from speckle.core.optimization import compute_icgn

    subset_size = 25
    M = subset_size // 2

    # spacing=5
    print("  spacing=5...")
    t0 = time.time()
    fft5 = compute_fft_cc(ref_img.astype(np.float64), def_img.astype(np.float64),
                           subset_size=subset_size, spacing=11, zncc_threshold=0.6)
    icgn5 = compute_icgn(ref_img.astype(np.float64), def_img.astype(np.float64),
                          subset_size=subset_size, initial_guess=fft5,
                          max_iterations=50, convergence_threshold=1e-3,
                          zncc_threshold=0.9, shape_function='affine',
                          enable_variable_subset=False, enable_adss_subset=False)
    t5 = time.time() - t0
    print(f"    {len(icgn5.points_x)} POI, 유효 {icgn5.valid_mask.sum()}, "
          f"불량 {(~icgn5.valid_mask).sum()}, {t5:.1f}s")

    # spacing=11
    print("  spacing=11...")
    fft11 = compute_fft_cc(ref_img.astype(np.float64), def_img.astype(np.float64),
                            subset_size=subset_size, spacing=11, zncc_threshold=0.6)
    icgn11 = compute_icgn(ref_img.astype(np.float64), def_img.astype(np.float64),
                           subset_size=subset_size, initial_guess=fft11,
                           max_iterations=50, convergence_threshold=1e-3,
                           zncc_threshold=0.9, shape_function='affine',
                           enable_variable_subset=False, enable_adss_subset=False)
    print(f"    {len(icgn11.points_x)} POI, 유효 {icgn11.valid_mask.sum()}, "
          f"불량 {(~icgn11.valid_mask).sum()}")

    # ── 3. ROI 구성 ──
    print("\n[3] ROI 구성 (spacing=5 불량 POI 기반)...")
    bad5 = np.where(~icgn5.valid_mask)[0]
    bad_coords = set()
    for idx in bad5:
        px, py = int(icgn5.points_x[idx]), int(icgn5.points_y[idx])
        for dx in [-5, 0, 5]:
            for dy in [-5, 0, 5]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < W and 0 <= ny < H:
                    bad_coords.add((nx, ny))

    all_x = [c[0] for c in bad_coords]
    all_y = [c[1] for c in bad_coords]
    roi_x1 = max(0, min(all_x) - M - 5)
    roi_x2 = min(W, max(all_x) + M + 6)
    roi_y1 = max(0, min(all_y) - M - 5)
    roi_y2 = min(H, max(all_y) + M + 6)

    roi_def = def_img[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_gt = gt_mask[roi_y1:roi_y2, roi_x1:roi_x2]
    rH, rW = roi_def.shape
    print(f"  ROI: ({roi_x1},{roi_y1})~({roi_x2},{roi_y2}), {rW}×{rH}px")

    # GT 경계선 + 팁
    gt_bnd = extract_gt_boundary(roi_gt)
    gt_tip = extract_gt_tip(roi_gt)
    print(f"  GT 경계점: {len(gt_bnd)}, GT 팁: {gt_tip}")

    # ── 4. 변위장 생성 ──
    print("\n[4] 변위장 생성...")

    def build_disp_fields(icgn_res, spacing_val):
        u_grid = displacement_to_grid(icgn_res.points_x, icgn_res.points_y,
                                       icgn_res.disp_u, (H, W))
        v_grid = displacement_to_grid(icgn_res.points_x, icgn_res.points_y,
                                       icgn_res.disp_v, (H, W))
        bad = np.where(~icgn_res.valid_mask)[0]
        for idx in bad:
            x, y = int(round(icgn_res.points_x[idx])), int(round(icgn_res.points_y[idx]))
            if 0 <= y < H and 0 <= x < W:
                u_grid[y, x] = np.nan
                v_grid[y, x] = np.nan
        u_roi = interpolate_grid(u_grid[roi_y1:roi_y2, roi_x1:roi_x2])
        v_roi = interpolate_grid(v_grid[roi_y1:roi_y2, roi_x1:roi_x2])
        mag_roi = np.sqrt(u_roi**2 + v_roi**2)
        return u_roi, v_roi, mag_roi

    u5, v5, mag5 = build_disp_fields(icgn5, 5)
    u11, v11, mag11 = build_disp_fields(icgn11, 11)
    print(f"  sp5 v 범위: {v5.min():.3f}~{v5.max():.3f}")
    print(f"  sp11 v 범위: {v11.min():.3f}~{v11.max():.3f}")

    # ── 5. 각 방법으로 경계 추출 ──
    print("\n[5] 경계 추출...")

    results = {}

    # (A) Mean+kσ contour (k 스윕)
    print("\n  [A] Mean+kσ contour...")
    for k in [0.8, 1.0, 1.2, 1.5, 2.0]:
        pts = method_mean_k_sigma_contour(roi_def, k=k, sigma_blur=0)
        ev = evaluate_boundary(pts, gt_bnd)
        tip = compute_tip_reach(pts, gt_tip)
        name = f'Mean+kσ (k={k})'
        results[name] = {'pts': pts, 'eval': ev, 'tip': tip, 'method': 'mean'}
        print(f"    k={k}: {ev['n_detected']}pts, "
              f"det→GT={ev['det_to_gt_mean']:.2f}, GT→det={ev['gt_to_det_mean']:.2f}, "
              f"Haus={ev['hausdorff']:.2f}, tip_err={tip['tip_error']:.1f}")

    # (B) 변위장 gradient threshold (v변위, sp5)
    print("\n  [B] 변위 gradient threshold (v, sp5)...")
    # gradient 통계 먼저 파악
    _, grad_mag_v5, _ = method_disp_gradient_threshold(v5, gradient_threshold=1e10)
    grad_vals = grad_mag_v5[grad_mag_v5 > 0]
    print(f"    gradient 범위: {grad_vals.min():.2f}~{grad_vals.max():.2f}, "
          f"median={np.median(grad_vals):.2f}, p90={np.percentile(grad_vals, 90):.2f}")

    for thr_pct in [50, 60, 70, 80, 90, 95]:
        thr_val = np.percentile(grad_vals, thr_pct)
        for sigma in [0.5, 1.0]:
            pts, gm, nms = method_disp_gradient_threshold(v5, thr_val, sigma_blur=sigma)
            ev = evaluate_boundary(pts, gt_bnd)
            tip = compute_tip_reach(pts, gt_tip)
            name = f'v_grad p{thr_pct} σ={sigma} (sp5)'
            results[name] = {'pts': pts, 'eval': ev, 'tip': tip,
                             'method': 'disp_grad', 'grad_mag': gm, 'nms': nms}
            print(f"    p{thr_pct}(thr={thr_val:.1f}) σ={sigma}: {ev['n_detected']}pts, "
                  f"det→GT={ev['det_to_gt_mean']:.2f}, Haus={ev['hausdorff']:.2f}, "
                  f"tip_err={tip['tip_error']:.1f}")

    # (C) 변위장 Canny (v변위, sp5)
    print("\n  [C] 변위 Canny (v, sp5)...")
    for sigma in [0.5, 1.0, 1.5]:
        for lr, hr in [(0.2, 0.5), (0.3, 0.7), (0.4, 0.8)]:
            pts = method_disp_canny(v5, sigma_blur=sigma, low_ratio=lr, high_ratio=hr)
            ev = evaluate_boundary(pts, gt_bnd)
            tip = compute_tip_reach(pts, gt_tip)
            name = f'v_canny σ={sigma} lr={lr} (sp5)'
            results[name] = {'pts': pts, 'eval': ev, 'tip': tip, 'method': 'disp_canny'}
            print(f"    σ={sigma} lr={lr} hr={hr}: {ev['n_detected']}pts, "
                  f"det→GT={ev['det_to_gt_mean']:.2f}, Haus={ev['hausdorff']:.2f}, "
                  f"tip_err={tip['tip_error']:.1f}")

    # (D) 변위장 gradient (|d|, sp5)
    print("\n  [D] |d| gradient threshold (sp5)...")
    _, grad_mag_d5, _ = method_disp_gradient_threshold(mag5, gradient_threshold=1e10)
    grad_vals_d = grad_mag_d5[grad_mag_d5 > 0]
    for thr_pct in [70, 80, 90]:
        thr_val = np.percentile(grad_vals_d, thr_pct)
        pts, gm, nms = method_disp_gradient_threshold(mag5, thr_val, sigma_blur=1.0)
        ev = evaluate_boundary(pts, gt_bnd)
        tip = compute_tip_reach(pts, gt_tip)
        name = f'|d|_grad p{thr_pct} (sp5)'
        results[name] = {'pts': pts, 'eval': ev, 'tip': tip, 'method': 'disp_grad'}
        print(f"    p{thr_pct}: {ev['n_detected']}pts, "
              f"det→GT={ev['det_to_gt_mean']:.2f}, Haus={ev['hausdorff']:.2f}, "
              f"tip_err={tip['tip_error']:.1f}")

    # (E) spacing=11 비교
    print("\n  [E] v gradient (sp11)...")
    _, grad_mag_v11, _ = method_disp_gradient_threshold(v11, gradient_threshold=1e10)
    grad_vals_11 = grad_mag_v11[grad_mag_v11 > 0]
    for thr_pct in [70, 80, 90]:
        thr_val = np.percentile(grad_vals_11, thr_pct)
        pts, gm, nms = method_disp_gradient_threshold(v11, thr_val, sigma_blur=1.0)
        ev = evaluate_boundary(pts, gt_bnd)
        tip = compute_tip_reach(pts, gt_tip)
        name = f'v_grad p{thr_pct} (sp11)'
        results[name] = {'pts': pts, 'eval': ev, 'tip': tip, 'method': 'disp_grad'}
        print(f"    p{thr_pct}: {ev['n_detected']}pts, "
              f"det→GT={ev['det_to_gt_mean']:.2f}, Haus={ev['hausdorff']:.2f}, "
              f"tip_err={tip['tip_error']:.1f}")

    # ── 6. 최적 결과 선별 ──
    print("\n[6] 최적 결과 선별...")

    # 각 방법 카테고리별 최적 (det_to_gt_mean 기준)
    categories = {
        'Mean+kσ': [k for k in results if results[k]['method'] == 'mean'],
        'v_grad (sp5)': [k for k in results if 'v_grad' in k and 'sp5' in k],
        'v_canny (sp5)': [k for k in results if 'v_canny' in k],
        '|d|_grad (sp5)': [k for k in results if '|d|_grad' in k],
        'v_grad (sp11)': [k for k in results if 'v_grad' in k and 'sp11' in k],
    }

    best_per_cat = {}
    for cat, keys in categories.items():
        if not keys:
            continue
        valid_keys = [k for k in keys if results[k]['eval']['det_to_gt_mean'] < float('inf')]
        if not valid_keys:
            continue
        best_key = min(valid_keys, key=lambda k: results[k]['eval']['det_to_gt_mean'])
        best_per_cat[cat] = best_key

    print(f"\n{'카테고리':<20} {'최적 설정':<35} {'det→GT':>7} {'GT→det':>7} "
          f"{'Haus':>7} {'팁오차':>7} {'경계점':>7}")
    print("-" * 100)
    for cat, key in best_per_cat.items():
        ev = results[key]['eval']
        tip = results[key]['tip']
        print(f"{cat:<20} {key:<35} {ev['det_to_gt_mean']:>7.2f} "
              f"{ev['gt_to_det_mean']:>7.2f} {ev['hausdorff']:>7.2f} "
              f"{tip['tip_error']:>7.1f} {ev['n_detected']:>7}")

    # ══════════════════════════════════════════════════
    # 시각화
    # ══════════════════════════════════════════════════

    # ── Figure 1: 경계선 오버레이 비교 ──
    print("\n[7] Figure 1 생성...")
    n_cats = len(best_per_cat)
    fig1, axes1 = plt.subplots(2, max(n_cats, 1), figsize=(5 * n_cats, 10))
    fig1.suptitle('Figure 1: 경계선 정밀도 비교 (초록=GT, 빨강=검출)',
                  fontsize=16, fontweight='bold')

    if n_cats == 1:
        axes1 = axes1.reshape(2, 1)

    for col, (cat, key) in enumerate(best_per_cat.items()):
        pts = results[key]['pts']
        ev = results[key]['eval']
        tip = results[key]['tip']

        # Row 0: 경계선 오버레이
        axes1[0, col].imshow(roi_def, cmap='gray', alpha=0.6)
        # GT 경계
        if len(gt_bnd) > 0:
            axes1[0, col].plot(gt_bnd[:, 0], gt_bnd[:, 1], 'g.', markersize=0.5, alpha=0.6)
        # 검출 경계
        if len(pts) > 0:
            axes1[0, col].plot(pts[:, 0], pts[:, 1], 'r.', markersize=0.5, alpha=0.8)
        # GT 팁
        if gt_tip:
            axes1[0, col].plot(gt_tip[0], gt_tip[1], 'g*', markersize=12)
        # 검출 최대 x
        if len(pts) > 0:
            max_x_idx = np.argmax(pts[:, 0])
            axes1[0, col].plot(pts[max_x_idx, 0], pts[max_x_idx, 1], 'r*', markersize=12)

        axes1[0, col].set_title(f'{cat}\n{key}\n경계점: {len(pts)}',
                                fontsize=8)
        axes1[0, col].axis('off')

        # Row 1: 검출→GT 거리 히트맵
        if len(pts) > 0 and len(ev['det_to_gt_all']) > 0:
            sc = axes1[1, col].scatter(pts[:, 0], pts[:, 1],
                                        c=ev['det_to_gt_all'],
                                        cmap='hot_r', s=1,
                                        vmin=0, vmax=max(3, np.percentile(ev['det_to_gt_all'], 95)))
            plt.colorbar(sc, ax=axes1[1, col], fraction=0.046, label='GT거리(px)')
            # GT 경계 참조선
            if len(gt_bnd) > 0:
                axes1[1, col].plot(gt_bnd[:, 0], gt_bnd[:, 1], 'g-', linewidth=0.5, alpha=0.3)
        axes1[1, col].set_title(
            f'det→GT: {ev["det_to_gt_mean"]:.2f}px\n'
            f'GT→det: {ev["gt_to_det_mean"]:.2f}px\n'
            f'Haus: {ev["hausdorff"]:.2f}px | 팁오차: {tip["tip_error"]:.1f}px',
            fontsize=8)
        axes1[1, col].set_xlim(0, rW)
        axes1[1, col].set_ylim(rH, 0)
        axes1[1, col].axis('off')

    fig1.tight_layout()
    path1 = OUTPUT_DIR / 'fig1_boundary_overlay.png'
    fig1.savefig(str(path1), dpi=150, bbox_inches='tight')
    print(f"  → {path1}")

    # ── Figure 2: 팁 근처 줌 ──
    print("[8] Figure 2 생성: 팁 근처 줌...")
    if gt_tip:
        tip_x = int(gt_tip[0])
        zoom_x1 = max(0, tip_x - 50)
        zoom_x2 = min(rW, tip_x + 30)
        zoom_y1 = max(0, int(gt_tip[1]) - 20)
        zoom_y2 = min(rH, int(gt_tip[1]) + 20)

        fig2, axes2 = plt.subplots(2, max(n_cats, 1), figsize=(5 * n_cats, 8))
        fig2.suptitle('Figure 2: 크랙 팁 근처 줌 (초록=GT, 빨강=검출, ★=팁)',
                      fontsize=14, fontweight='bold')
        if n_cats == 1:
            axes2 = axes2.reshape(2, 1)

        for col, (cat, key) in enumerate(best_per_cat.items()):
            pts = results[key]['pts']
            tip = results[key]['tip']

            # Row 0: 이미지 + 경계
            axes2[0, col].imshow(roi_def[zoom_y1:zoom_y2, zoom_x1:zoom_x2],
                                  cmap='gray', extent=[zoom_x1, zoom_x2, zoom_y2, zoom_y1])
            if len(gt_bnd) > 0:
                mask = ((gt_bnd[:, 0] >= zoom_x1) & (gt_bnd[:, 0] < zoom_x2) &
                        (gt_bnd[:, 1] >= zoom_y1) & (gt_bnd[:, 1] < zoom_y2))
                if mask.any():
                    axes2[0, col].plot(gt_bnd[mask, 0], gt_bnd[mask, 1],
                                       'g.', markersize=3, alpha=0.8)
            if len(pts) > 0:
                mask = ((pts[:, 0] >= zoom_x1) & (pts[:, 0] < zoom_x2) &
                        (pts[:, 1] >= zoom_y1) & (pts[:, 1] < zoom_y2))
                if mask.any():
                    axes2[0, col].plot(pts[mask, 0], pts[mask, 1],
                                       'r.', markersize=3, alpha=0.8)
            axes2[0, col].plot(gt_tip[0], gt_tip[1], 'g*', markersize=15)
            if len(pts) > 0:
                max_x_idx = np.argmax(pts[:, 0])
                axes2[0, col].plot(pts[max_x_idx, 0], pts[max_x_idx, 1],
                                    'r*', markersize=15)
            axes2[0, col].set_title(f'{cat}\n팁 오차: {tip["tip_error"]:.1f}px',
                                     fontsize=9)

            # Row 1: 변위 gradient (해당 방법이 변위 기반인 경우)
            if results[key]['method'] == 'disp_grad' and 'grad_mag' in results[key]:
                gm = results[key]['grad_mag']
                axes2[1, col].imshow(gm[zoom_y1:zoom_y2, zoom_x1:zoom_x2],
                                      cmap='hot',
                                      extent=[zoom_x1, zoom_x2, zoom_y2, zoom_y1])
                axes2[1, col].plot(gt_tip[0], gt_tip[1], 'g*', markersize=15)
                axes2[1, col].set_title('변위 gradient 크기', fontsize=9)
            else:
                axes2[1, col].imshow(roi_def[zoom_y1:zoom_y2, zoom_x1:zoom_x2],
                                      cmap='gray',
                                      extent=[zoom_x1, zoom_x2, zoom_y2, zoom_y1])
                # Mean+kσ mask overlay
                mask_img = method_mean_k_sigma_contour(roi_def, k=1.5)
                axes2[1, col].set_title('이미지 (참조)', fontsize=9)

        fig2.tight_layout()
        path2 = OUTPUT_DIR / 'fig2_tip_zoom.png'
        fig2.savefig(str(path2), dpi=150, bbox_inches='tight')
        print(f"  → {path2}")

    # ── Figure 3: 정량 비교 바 차트 ──
    print("[9] Figure 3 생성: 정량 비교...")
    fig3, axes3 = plt.subplots(1, 4, figsize=(20, 5))
    fig3.suptitle('Figure 3: 경계 정밀도 정량 비교', fontsize=16, fontweight='bold')

    cat_names = list(best_per_cat.keys())
    x = np.arange(len(cat_names))

    # (a) det→GT 평균 거리
    vals = [results[best_per_cat[c]]['eval']['det_to_gt_mean'] for c in cat_names]
    bars = axes3[0].bar(x, vals, color='steelblue', edgecolor='black')
    for i, v in enumerate(vals):
        axes3[0].text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=9)
    axes3[0].set_xticks(x)
    axes3[0].set_xticklabels(cat_names, rotation=25, ha='right', fontsize=8)
    axes3[0].set_ylabel('거리 (px)')
    axes3[0].set_title('(a) 검출→GT 평균 거리\n(낮을수록 정밀)')
    axes3[0].grid(True, alpha=0.3, axis='y')

    # (b) GT→det 평균 거리
    vals = [results[best_per_cat[c]]['eval']['gt_to_det_mean'] for c in cat_names]
    axes3[1].bar(x, vals, color='coral', edgecolor='black')
    for i, v in enumerate(vals):
        axes3[1].text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=9)
    axes3[1].set_xticks(x)
    axes3[1].set_xticklabels(cat_names, rotation=25, ha='right', fontsize=8)
    axes3[1].set_ylabel('거리 (px)')
    axes3[1].set_title('(b) GT→검출 평균 거리\n(낮을수록 완전)')
    axes3[1].grid(True, alpha=0.3, axis='y')

    # (c) Hausdorff
    vals = [results[best_per_cat[c]]['eval']['hausdorff'] for c in cat_names]
    axes3[2].bar(x, vals, color='goldenrod', edgecolor='black')
    for i, v in enumerate(vals):
        axes3[2].text(i, v + 0.2, f'{v:.1f}', ha='center', fontsize=9)
    axes3[2].set_xticks(x)
    axes3[2].set_xticklabels(cat_names, rotation=25, ha='right', fontsize=8)
    axes3[2].set_ylabel('거리 (px)')
    axes3[2].set_title('(c) Hausdorff 거리\n(최악 오차)')
    axes3[2].grid(True, alpha=0.3, axis='y')

    # (d) 팁 오차
    vals = [results[best_per_cat[c]]['tip']['tip_error'] for c in cat_names]
    colors = ['mediumseagreen' if v < 20 else 'coral' for v in vals]
    axes3[3].bar(x, vals, color=colors, edgecolor='black')
    for i, v in enumerate(vals):
        axes3[3].text(i, v + 0.3, f'{v:.1f}', ha='center', fontsize=9)
    axes3[3].set_xticks(x)
    axes3[3].set_xticklabels(cat_names, rotation=25, ha='right', fontsize=8)
    axes3[3].set_ylabel('거리 (px)')
    axes3[3].set_title('(d) 크랙 팁 오차\n(낮을수록 팁에 가까이 도달)')
    axes3[3].grid(True, alpha=0.3, axis='y')

    fig3.tight_layout()
    path3 = OUTPUT_DIR / 'fig3_quantitative.png'
    fig3.savefig(str(path3), dpi=150, bbox_inches='tight')
    print(f"  → {path3}")

    # ── Figure 4: gradient 분포 분석 ──
    print("[10] Figure 4 생성: gradient 분포...")
    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
    fig4.suptitle('Figure 4: 변위 Gradient 분포 (크랙경계 vs 비크랙)',
                  fontsize=16, fontweight='bold')

    # GT 경계 마스크
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    gt_dilated = cv2.dilate(roi_gt.astype(np.uint8), kernel)
    boundary_mask = (gt_dilated - roi_gt.astype(np.uint8)).astype(bool)
    non_crack_mask = ~(roi_gt | boundary_mask)

    # (a) v gradient (sp5)
    _, gm_v5, _ = method_disp_gradient_threshold(v5, 1e10)
    if boundary_mask.any() and non_crack_mask.any():
        axes4[0].hist(gm_v5[boundary_mask], bins=50, alpha=0.6, color='red',
                      label='크랙 경계', density=True)
        axes4[0].hist(gm_v5[non_crack_mask], bins=50, alpha=0.6, color='blue',
                      label='비크랙', density=True)
        # 최적 임계값 표시
        if 'v_grad (sp5)' in best_per_cat:
            best_key_vg = best_per_cat['v_grad (sp5)']
            # 사용된 thr 추정
            axes4[0].axvline(np.percentile(gm_v5[gm_v5 > 0], 80),
                             color='green', linestyle='--', linewidth=2, label='p80 임계값')
        axes4[0].set_xlabel('Gradient 크기')
        axes4[0].set_title('(a) v gradient 분포 (sp5)')
        axes4[0].legend(fontsize=8)
        axes4[0].grid(True, alpha=0.3)

    # (b) |d| gradient (sp5)
    _, gm_d5, _ = method_disp_gradient_threshold(mag5, 1e10)
    if boundary_mask.any() and non_crack_mask.any():
        axes4[1].hist(gm_d5[boundary_mask], bins=50, alpha=0.6, color='red',
                      label='크랙 경계', density=True)
        axes4[1].hist(gm_d5[non_crack_mask], bins=50, alpha=0.6, color='blue',
                      label='비크랙', density=True)
        axes4[1].set_xlabel('Gradient 크기')
        axes4[1].set_title('(b) |d| gradient 분포 (sp5)')
        axes4[1].legend(fontsize=8)
        axes4[1].grid(True, alpha=0.3)

    # (c) 이미지 밝기 gradient (비교용)
    img_gx = cv2.Sobel(roi_def, cv2.CV_64F, 1, 0, ksize=3)
    img_gy = cv2.Sobel(roi_def, cv2.CV_64F, 0, 1, ksize=3)
    img_gm = np.sqrt(img_gx**2 + img_gy**2)
    if boundary_mask.any() and non_crack_mask.any():
        axes4[2].hist(img_gm[boundary_mask], bins=50, alpha=0.6, color='red',
                      label='크랙 경계', density=True)
        axes4[2].hist(img_gm[non_crack_mask], bins=50, alpha=0.6, color='blue',
                      label='비크랙', density=True)
        axes4[2].set_xlabel('Gradient 크기')
        axes4[2].set_title('(c) 밝기 gradient 분포 (비교)')
        axes4[2].legend(fontsize=8)
        axes4[2].grid(True, alpha=0.3)

    fig4.tight_layout()
    path4 = OUTPUT_DIR / 'fig4_gradient_distribution.png'
    fig4.savefig(str(path4), dpi=150, bbox_inches='tight')
    print(f"  → {path4}")

    # ── 최종 요약 ──
    print("\n" + "=" * 70)
    print("최종 요약: 크랙 경계 정밀도 비교")
    print("=" * 70)
    print(f"\n{'카테고리':<20} {'det→GT(px)':>10} {'GT→det(px)':>10} "
          f"{'Hausdorff':>10} {'팁오차(px)':>10}")
    print("-" * 65)
    for cat, key in best_per_cat.items():
        ev = results[key]['eval']
        tip = results[key]['tip']
        print(f"{cat:<20} {ev['det_to_gt_mean']:>10.2f} {ev['gt_to_det_mean']:>10.2f} "
              f"{ev['hausdorff']:>10.2f} {tip['tip_error']:>10.1f}")
    print("=" * 70)
    print(f"\n결과 저장: {OUTPUT_DIR}")

    plt.show()
    print("완료!")


if __name__ == '__main__':
    run_experiment()

