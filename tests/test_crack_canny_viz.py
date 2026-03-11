# tests/test_crack_boundary_canny.py
"""
크랙 경계 추출: Mean+kσ vs Canny vs |Ref-Def|+Canny 비교

파이프라인:
1. DIC → 불량 POI + 8방위 이웃 → ROI 구성
2. ROI에서 3가지 방법으로 contour 경계선 추출
3. GT contour와 경계 오차(평균, Hausdorff) + 팁 도달 거리 비교
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial import distance
from scipy import ndimage
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import platform
import time

# 한글 폰트
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
OUTPUT_DIR = TEST_OUTPUT_ROOT / 'output_boundary_canny'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 유틸리티
# ============================================================

def _imread_unicode(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지 로딩 실패: {path}")
    return img


def contours_to_points(contours):
    if not contours:
        return np.array([]).reshape(0, 2)
    all_pts = []
    for cnt in contours:
        pts = cnt.squeeze()
        if pts.ndim == 2:
            all_pts.append(pts)
        elif pts.ndim == 1 and len(pts) == 2:
            all_pts.append(pts.reshape(1, 2))
    return np.vstack(all_pts).astype(np.float64) if all_pts else np.array([]).reshape(0, 2)


def compute_metrics(gt, pred):
    gt_b, pred_b = gt.astype(bool), pred.astype(bool)
    tp = np.sum(gt_b & pred_b)
    fp = np.sum(~gt_b & pred_b)
    fn = np.sum(gt_b & ~pred_b)
    iou = tp / max(tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-10)
    return {'iou': iou, 'precision': prec, 'recall': rec, 'f1': f1,
            'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


def evaluate_boundary(det_pts, gt_pts):
    if len(det_pts) == 0 or len(gt_pts) == 0:
        return {'det2gt_mean': float('inf'), 'gt2det_mean': float('inf'),
                'hausdorff': float('inf'), 'det2gt_median': float('inf'),
                'det2gt_all': np.array([]), 'gt2det_all': np.array([]),
                'n_det': len(det_pts), 'n_gt': len(gt_pts)}
    d2g = distance.cdist(det_pts, gt_pts).min(axis=1)
    g2d = distance.cdist(gt_pts, det_pts).min(axis=1)
    return {'det2gt_mean': float(np.mean(d2g)), 'gt2det_mean': float(np.mean(g2d)),
            'hausdorff': max(float(np.max(d2g)), float(np.max(g2d))),
            'det2gt_median': float(np.median(d2g)),
            'det2gt_all': d2g, 'gt2det_all': g2d,
            'n_det': len(det_pts), 'n_gt': len(gt_pts)}


def compute_tip_reach(det_pts, gt_tip):
    if len(det_pts) == 0 or gt_tip is None:
        return {'tip_error': float('inf'), 'det_max_x': 0, 'gt_tip_x': 0}
    max_x = float(np.max(det_pts[:, 0]))
    return {'tip_error': abs(gt_tip[0] - max_x),
            'det_max_x': max_x, 'gt_tip_x': gt_tip[0]}


def extract_gt_boundary(gt_mask):
    gt_u8 = (gt_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(gt_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return contours, contours_to_points(contours)


def extract_gt_tip(gt_mask):
    ys, xs = np.where(gt_mask)
    if len(xs) == 0:
        return None
    idx = np.argmax(xs)
    return (float(xs[idx]), float(ys[idx]))


# ============================================================
# ROI 구성
# ============================================================

def build_roi(def_img, icgn_result, spacing, subset_size):
    H, W = def_img.shape
    M = subset_size // 2
    bad_indices = np.where(~icgn_result.valid_mask)[0]
    poi_x, poi_y = icgn_result.points_x, icgn_result.points_y

    bad_only = set()
    all_coords = set()
    for idx in bad_indices:
        px, py = int(poi_x[idx]), int(poi_y[idx])
        bad_only.add((px, py))
        for dx in [-spacing, 0, spacing]:
            for dy in [-spacing, 0, spacing]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < W and 0 <= ny < H:
                    all_coords.add((nx, ny))

    all_x = [c[0] for c in all_coords]
    all_y = [c[1] for c in all_coords]
    roi_x1 = max(0, min(all_x) - M - 5)
    roi_x2 = min(W, max(all_x) + M + 6)
    roi_y1 = max(0, min(all_y) - M - 5)
    roi_y2 = min(H, max(all_y) + M + 6)

    return {
        'roi': def_img[roi_y1:roi_y2, roi_x1:roi_x2],
        'bounds': (roi_x1, roi_y1, roi_x2, roi_y2),
        'bad_indices': bad_indices, 'bad_coords': bad_only,
    }


# ============================================================
# 경계 추출 방법들
# ============================================================

def detect_mean_k_sigma(roi, k=1.0, sigma_blur=1.0, min_area=10):
    """Mean+kσ → contour"""
    ksize = max(3, int(np.ceil(sigma_blur * 3)) * 2 + 1)
    blurred = cv2.GaussianBlur(roi, (ksize, ksize), sigma_blur)
    mu, std = float(blurred.mean()), float(blurred.std())
    thr = mu - k * std
    binary = (blurred < thr).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    n_lab, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, n_lab):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary[labels == i] = 0

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return {'contours': contours, 'mask': binary, 'pts': contours_to_points(contours),
            'params': f'k={k}, σ={sigma_blur}, thr={thr:.1f}'}


def detect_canny_image(roi, sigma_blur=1.0, low=None, high=None, dilate_iter=0, min_length=5):
    """이미지에 직접 Canny → contour"""
    ksize = max(3, int(np.ceil(sigma_blur * 3)) * 2 + 1)
    blurred = cv2.GaussianBlur(roi, (ksize, ksize), sigma_blur)

    if low is None or high is None:
        v = np.median(blurred)
        low = int(max(0, 0.5 * v))
        high = int(min(255, 1.0 * v))

    edges = cv2.Canny(blurred, low, high)

    # 짧은 성분 제거
    labeled, n_comp = ndimage.label(edges)
    for i in range(1, n_comp + 1):
        if np.sum(labeled == i) < min_length:
            edges[labeled == i] = 0

    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=dilate_iter)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return {'contours': contours, 'mask': edges, 'pts': contours_to_points(contours),
            'params': f'low={low}, high={high}, σ={sigma_blur}, dilate={dilate_iter}'}


def detect_ref_diff_canny(ref_roi, def_roi, sigma_blur=1.0, low=None, high=None,
                           dilate_iter=0, min_length=5):
    """|Ref-Def| 차분 영상에 Canny → contour"""
    diff = np.abs(ref_roi.astype(np.float64) - def_roi.astype(np.float64))
    diff_u8 = np.clip(diff, 0, 255).astype(np.uint8)

    ksize = max(3, int(np.ceil(sigma_blur * 3)) * 2 + 1)
    blurred = cv2.GaussianBlur(diff_u8, (ksize, ksize), sigma_blur)

    if low is None or high is None:
        v = np.median(blurred[blurred > 0]) if np.any(blurred > 0) else 1
        low = int(max(1, 0.3 * v))
        high = int(min(255, 0.8 * v))

    edges = cv2.Canny(blurred, low, high)

    labeled, n_comp = ndimage.label(edges)
    for i in range(1, n_comp + 1):
        if np.sum(labeled == i) < min_length:
            edges[labeled == i] = 0

    if dilate_iter > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        edges = cv2.dilate(edges, kernel, iterations=dilate_iter)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return {'contours': contours, 'mask': edges, 'pts': contours_to_points(contours),
            'params': f'low={low}, high={high}, σ={sigma_blur}, dilate={dilate_iter}',
            'diff_img': diff_u8}


# ============================================================
# 메인 실험
# ============================================================

def run_test():
    t_total = time.time()
    print("=" * 70)
    print("크랙 경계 추출: Mean+kσ vs Canny vs |Ref-Def|+Canny")
    print("=" * 70)

    # ── 데이터 로드 ──
    def_img = _imread_unicode(DATA_DIR / 'deformed.tiff')
    ref_img = _imread_unicode(DATA_DIR / 'reference.tiff')
    gt_mask = np.load(str(DATA_DIR / 'crack_mask.npy')).astype(bool)
    H, W = def_img.shape
    print(f"이미지: {W}×{H}, GT 크랙: {gt_mask.sum()}px")

    # ── DIC 실행 ──
    print("\n[DIC] FFT-CC + IC-GN...")
    from speckle.core.initial_guess import compute_fft_cc
    from speckle.core.optimization import compute_icgn

    subset_size = 25
    spacing = 11

    fft_result = compute_fft_cc(
        ref_img.astype(np.float64), def_img.astype(np.float64),
        subset_size=subset_size, spacing=spacing, zncc_threshold=0.6)
    icgn_result = compute_icgn(
        ref_img.astype(np.float64), def_img.astype(np.float64),
        subset_size=subset_size, initial_guess=fft_result,
        max_iterations=50, convergence_threshold=1e-3,
        zncc_threshold=0.9, shape_function='affine',
        enable_variable_subset=False, enable_adss_subset=False)

    n_bad = int((~icgn_result.valid_mask).sum())
    print(f"  유효: {icgn_result.valid_mask.sum()}, 불량: {n_bad}")

    # ── ROI 구성 ──
    print("\n[ROI] 불량 POI + 8방위 이웃...")
    roi_data = build_roi(def_img, icgn_result, spacing, subset_size)
    roi_def = roi_data['roi']
    rx1, ry1, rx2, ry2 = roi_data['bounds']
    rH, rW = roi_def.shape
    roi_ref = ref_img[ry1:ry2, rx1:rx2]
    roi_gt = gt_mask[ry1:ry2, rx1:rx2]
    print(f"  ROI: ({rx1},{ry1})~({rx2},{ry2}), {rW}×{rH}px")

    gt_contours, gt_pts = extract_gt_boundary(roi_gt)
    gt_tip = extract_gt_tip(roi_gt)
    print(f"  GT 경계점: {len(gt_pts)}, GT 팁: {gt_tip}")

    # ════════════════════════════════════════════════
    # 파라미터 스윕: 3가지 방법
    # ════════════════════════════════════════════════
    all_results = {}

    # (A) Mean+kσ
    print("\n[A] Mean+kσ 스윕...")
    print(f"  {'k':>5} | {'경계점':>7} | {'det→GT':>7} | {'GT→det':>7} | {'Haus':>7} | {'팁오차':>7} | {'IoU':>6}")
    print("  " + "-" * 60)
    for k in [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]:
        res = detect_mean_k_sigma(roi_def, k=k, sigma_blur=1.0)
        ev = evaluate_boundary(res['pts'], gt_pts)
        tip = compute_tip_reach(res['pts'], gt_tip)
        met = compute_metrics(roi_gt, res['mask'] > 0)
        name = f'Mean+kσ k={k}'
        all_results[name] = {'res': res, 'ev': ev, 'tip': tip, 'met': met, 'cat': 'mean'}
        print(f"  {k:5.1f} | {ev['n_det']:>7} | {ev['det2gt_mean']:>7.2f} | "
              f"{ev['gt2det_mean']:>7.2f} | {ev['hausdorff']:>7.2f} | "
              f"{tip['tip_error']:>7.1f} | {met['iou']:>6.4f}")

    # (B) Canny (이미지 직접)
    print("\n[B] Canny (이미지) 스윕...")
    print(f"  {'설정':>25} | {'경계점':>7} | {'det→GT':>7} | {'GT→det':>7} | {'Haus':>7} | {'팁오차':>7}")
    print("  " + "-" * 75)
    for sigma in [0.5, 1.0, 1.5]:
        for low, high in [(30, 80), (50, 120), (70, 150), (40, 100)]:
            for dil in [0, 1]:
                res = detect_canny_image(roi_def, sigma_blur=sigma,
                                          low=low, high=high, dilate_iter=dil)
                ev = evaluate_boundary(res['pts'], gt_pts)
                tip = compute_tip_reach(res['pts'], gt_tip)
                met = compute_metrics(roi_gt, res['mask'] > 0)
                name = f'Canny σ={sigma} L={low} H={high} d={dil}'
                all_results[name] = {'res': res, 'ev': ev, 'tip': tip, 'met': met, 'cat': 'canny'}
                if ev['det2gt_mean'] < 10:  # 유의미한 결과만 출력
                    print(f"  {name:>25} | {ev['n_det']:>7} | {ev['det2gt_mean']:>7.2f} | "
                          f"{ev['gt2det_mean']:>7.2f} | {ev['hausdorff']:>7.2f} | "
                          f"{tip['tip_error']:>7.1f}")

    # (C) |Ref-Def| + Canny
    print("\n[C] |Ref-Def| + Canny 스윕...")
    print(f"  {'설정':>25} | {'경계점':>7} | {'det→GT':>7} | {'GT→det':>7} | {'Haus':>7} | {'팁오차':>7}")
    print("  " + "-" * 75)
    for sigma in [0.5, 1.0, 1.5]:
        for low, high in [(5, 20), (10, 30), (15, 50), (20, 60), (30, 80)]:
            for dil in [0, 1]:
                res = detect_ref_diff_canny(roi_ref, roi_def, sigma_blur=sigma,
                                             low=low, high=high, dilate_iter=dil)
                ev = evaluate_boundary(res['pts'], gt_pts)
                tip = compute_tip_reach(res['pts'], gt_tip)
                met = compute_metrics(roi_gt, res['mask'] > 0)
                name = f'Diff+Canny σ={sigma} L={low} H={high} d={dil}'
                all_results[name] = {'res': res, 'ev': ev, 'tip': tip, 'met': met, 'cat': 'diff_canny'}
                if ev['det2gt_mean'] < 10:
                    print(f"  {name:>25} | {ev['n_det']:>7} | {ev['det2gt_mean']:>7.2f} | "
                          f"{ev['gt2det_mean']:>7.2f} | {ev['hausdorff']:>7.2f} | "
                          f"{tip['tip_error']:>7.1f}")

    # ════════════════════════════════════════════════
    # 카테고리별 최적 선별
    # ════════════════════════════════════════════════
    print("\n[최적 선별]")
    categories = {
        'Mean+kσ': 'mean',
        'Canny(이미지)': 'canny',
        '|Ref-Def|+Canny': 'diff_canny',
    }

    best = {}
    for cat_name, cat_key in categories.items():
        candidates = {k: v for k, v in all_results.items()
                      if v['cat'] == cat_key and v['ev']['det2gt_mean'] < float('inf')
                      and v['ev']['n_det'] > 0}
        if not candidates:
            print(f"  {cat_name}: 유효한 결과 없음")
            continue
        best_key = min(candidates, key=lambda k: candidates[k]['ev']['det2gt_mean'])
        best[cat_name] = all_results[best_key]
        best[cat_name]['key'] = best_key
        ev = all_results[best_key]['ev']
        tip = all_results[best_key]['tip']
        print(f"  {cat_name}: {best_key}")
        print(f"    det→GT={ev['det2gt_mean']:.2f}, GT→det={ev['gt2det_mean']:.2f}, "
              f"Haus={ev['hausdorff']:.2f}, 팁오차={tip['tip_error']:.1f}, "
              f"경계점={ev['n_det']}")

    # ════════════════════════════════════════════════
    # Figure 1: 경계선 오버레이 비교
    # ════════════════════════════════════════════════
    print("\n[Figure 1] 경계선 오버레이...")
    n_best = len(best)
    if n_best == 0:
        print("  유효 결과 없음 — 종료")
        return

    colors = {'Mean+kσ': 'red', 'Canny(이미지)': 'orange', '|Ref-Def|+Canny': 'cyan'}

    fig1 = plt.figure(figsize=(6 * n_best, 12))
    fig1.suptitle('Figure 1: 경계선 정밀도 비교 (초록=GT)', fontsize=16, fontweight='bold')
    gs1 = gridspec.GridSpec(2, n_best, hspace=0.3)

    for col, (cat_name, data) in enumerate(best.items()):
        pts = data['res']['pts']
        ev = data['ev']
        tip = data['tip']
        color = colors.get(cat_name, 'red')

        # Row 0: 경계선 오버레이
        ax = fig1.add_subplot(gs1[0, col])
        ax.imshow(roi_def, cmap='gray', alpha=0.6)
        for cnt in gt_contours:
            p = cnt.squeeze()
            if p.ndim == 2 and len(p) > 1:
                ax.plot(p[:, 0], p[:, 1], 'g-', linewidth=2, alpha=0.8)
        for cnt in data['res']['contours']:
            p = cnt.squeeze()
            if p.ndim == 2 and len(p) > 1:
                ax.plot(p[:, 0], p[:, 1], color=color, linewidth=1.5, alpha=0.9)
        if gt_tip:
            ax.plot(gt_tip[0], gt_tip[1], 'g*', markersize=14)
        if len(pts) > 0:
            idx_max = np.argmax(pts[:, 0])
            ax.plot(pts[idx_max, 0], pts[idx_max, 1], '*', color=color, markersize=14)
        ax.set_title(f'{cat_name}\n{data["key"]}\n경계점: {ev["n_det"]}', fontsize=9)
        ax.axis('off')

        # Row 1: 경계점 오차 히트맵
        ax = fig1.add_subplot(gs1[1, col])
        ax.imshow(roi_def, cmap='gray', alpha=0.4)
        if len(pts) > 0 and len(ev['det2gt_all']) > 0:
            vmax = max(3, np.percentile(ev['det2gt_all'], 95))
            sc = ax.scatter(pts[:, 0], pts[:, 1], c=ev['det2gt_all'],
                           cmap='hot_r', s=2, vmin=0, vmax=vmax)
            plt.colorbar(sc, ax=ax, fraction=0.046, label='GT거리(px)')
        for cnt in gt_contours:
            p = cnt.squeeze()
            if p.ndim == 2 and len(p) > 1:
                ax.plot(p[:, 0], p[:, 1], 'g-', linewidth=0.8, alpha=0.5)
        ax.set_title(f'det→GT: {ev["det2gt_mean"]:.2f}px | '
                     f'Haus: {ev["hausdorff"]:.2f}px\n'
                     f'팁오차: {tip["tip_error"]:.1f}px', fontsize=9)
        ax.axis('off')

    fig1.tight_layout(rect=[0, 0, 1, 0.95])
    path1 = OUTPUT_DIR / 'fig1_boundary_compare.png'
    fig1.savefig(str(path1), dpi=150, bbox_inches='tight')
    print(f"  → {path1}")

    # ════════════════════════════════════════════════
    # Figure 2: 팁 근처 줌
    # ════════════════════════════════════════════════
    if gt_tip:
        print("[Figure 2] 팁 근처 줌...")
        tip_x = int(gt_tip[0])
        zx1 = max(0, tip_x - 50)
        zx2 = min(rW, tip_x + 30)
        zy1 = max(0, int(gt_tip[1]) - 20)
        zy2 = min(rH, int(gt_tip[1]) + 20)

        fig2, axes2 = plt.subplots(1, n_best, figsize=(6 * n_best, 5))
        fig2.suptitle('Figure 2: 크랙 팁 근처 줌 (★=팁)', fontsize=14, fontweight='bold')
        if n_best == 1:
            axes2 = [axes2]

        for col, (cat_name, data) in enumerate(best.items()):
            pts = data['res']['pts']
            tip = data['tip']
            color = colors.get(cat_name, 'red')

            axes2[col].imshow(roi_def[zy1:zy2, zx1:zx2], cmap='gray',
                              extent=[zx1, zx2, zy2, zy1])
            # GT
            if len(gt_pts) > 0:
                m = ((gt_pts[:, 0] >= zx1) & (gt_pts[:, 0] < zx2) &
                     (gt_pts[:, 1] >= zy1) & (gt_pts[:, 1] < zy2))
                if m.any():
                    axes2[col].plot(gt_pts[m, 0], gt_pts[m, 1], 'g.', markersize=4)
            # 검출
            if len(pts) > 0:
                m = ((pts[:, 0] >= zx1) & (pts[:, 0] < zx2) &
                     (pts[:, 1] >= zy1) & (pts[:, 1] < zy2))
                if m.any():
                    axes2[col].plot(pts[m, 0], pts[m, 1], '.', color=color, markersize=4)
            axes2[col].plot(gt_tip[0], gt_tip[1], 'g*', markersize=16)
            if len(pts) > 0:
                idx_max = np.argmax(pts[:, 0])
                axes2[col].plot(pts[idx_max, 0], pts[idx_max, 1], '*',
                               color=color, markersize=16)
            axes2[col].set_title(f'{cat_name}\n팁 오차: {tip["tip_error"]:.1f}px', fontsize=10)

        fig2.tight_layout()
        path2 = OUTPUT_DIR / 'fig2_tip_zoom.png'
        fig2.savefig(str(path2), dpi=150, bbox_inches='tight')
        print(f"  → {path2}")

    # ════════════════════════════════════════════════
    # Figure 3: 경계 오차 분포 + 정량 비교
    # ════════════════════════════════════════════════
    print("[Figure 3] 정량 비교...")
    fig3, axes3 = plt.subplots(2, max(n_best, 2), figsize=(6 * max(n_best, 2), 10))
    fig3.suptitle('Figure 3: 경계 오차 분포 + 정량 비교', fontsize=16, fontweight='bold')

    # Row 0: 각 방법 경계 오차 히스토그램
    for col, (cat_name, data) in enumerate(best.items()):
        ev = data['ev']
        color = colors.get(cat_name, 'steelblue')
        ax = axes3[0, col]
        if len(ev['det2gt_all']) > 0:
            ax.hist(ev['det2gt_all'], bins=30, color=color, edgecolor='black', alpha=0.7)
            ax.axvline(ev['det2gt_mean'], color='navy', linewidth=2, linestyle='--',
                       label=f'평균={ev["det2gt_mean"]:.2f}')
            ax.axvline(ev['hausdorff'], color='red', linewidth=1.5, linestyle=':',
                       label=f'Haus={ev["hausdorff"]:.2f}')
            ax.legend(fontsize=8)
        ax.set_xlabel('GT까지 거리 (px)')
        ax.set_ylabel('빈도')
        ax.set_title(f'{cat_name} 경계 오차 분포', fontsize=10)
        ax.grid(True, alpha=0.3)

    # 남은 열 비움
    for col in range(n_best, axes3.shape[1]):
        axes3[0, col].axis('off')

    # Row 1: 종합 바 차트
    ax = axes3[1, 0]
    cat_names = list(best.keys())
    x = np.arange(len(cat_names))
    w = 0.2

    d2g = [best[c]['ev']['det2gt_mean'] for c in cat_names]
    g2d = [best[c]['ev']['gt2det_mean'] for c in cat_names]
    haus = [best[c]['ev']['hausdorff'] for c in cat_names]
    tips = [best[c]['tip']['tip_error'] for c in cat_names]

    bars1 = ax.bar(x - 1.5 * w, d2g, w, color='steelblue', edgecolor='black', label='det→GT')
    bars2 = ax.bar(x - 0.5 * w, g2d, w, color='coral', edgecolor='black', label='GT→det')
    bars3 = ax.bar(x + 0.5 * w, haus, w, color='goldenrod', edgecolor='black', label='Hausdorff')
    bars4 = ax.bar(x + 1.5 * w, tips, w, color='mediumseagreen', edgecolor='black', label='팁오차')

    for bars in [bars1, bars2, bars3, bars4]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                    f'{h:.1f}', ha='center', fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, fontsize=9)
    ax.set_ylabel('거리 (px)')
    ax.set_title('경계 오차 종합 비교', fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3, axis='y')

    # IoU/F1 비교
    ax = axes3[1, 1]
    ious = [best[c]['met']['iou'] for c in cat_names]
    f1s = [best[c]['met']['f1'] for c in cat_names]
    precs = [best[c]['met']['precision'] for c in cat_names]
    recs = [best[c]['met']['recall'] for c in cat_names]

    ax.bar(x - 1.5 * w, ious, w, color='steelblue', edgecolor='black', label='IoU')
    ax.bar(x - 0.5 * w, f1s, w, color='goldenrod', edgecolor='black', label='F1')
    ax.bar(x + 0.5 * w, precs, w, color='mediumseagreen', edgecolor='black', label='Precision')
    ax.bar(x + 1.5 * w, recs, w, color='coral', edgecolor='black', label='Recall')
    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('마스크 성능 비교', fontsize=10)
    ax.legend(fontsize=7)
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    for col in range(2, axes3.shape[1]):
        axes3[1, col].axis('off')

    fig3.tight_layout()
    path3 = OUTPUT_DIR / 'fig3_quantitative.png'
    fig3.savefig(str(path3), dpi=150, bbox_inches='tight')
    print(f"  → {path3}")

    # ════════════════════════════════════════════════
    # Figure 4: |Ref-Def| 차분 영상 + Canny 과정
    # ════════════════════════════════════════════════
    if '|Ref-Def|+Canny' in best:
        print("[Figure 4] |Ref-Def| + Canny 과정 시각화...")
        diff_data = best['|Ref-Def|+Canny']['res']

        fig4, axes4 = plt.subplots(1, 4, figsize=(20, 5))
        fig4.suptitle('Figure 4: |Ref-Def| + Canny 검출 과정', fontsize=14, fontweight='bold')

        axes4[0].imshow(roi_ref, cmap='gray')
        axes4[0].set_title('Reference')
        axes4[0].axis('off')

        axes4[1].imshow(roi_def, cmap='gray')
        axes4[1].set_title('Deformed')
        axes4[1].axis('off')

        if 'diff_img' in diff_data:
            axes4[2].imshow(diff_data['diff_img'], cmap='hot')
            axes4[2].set_title('|Ref - Def|')
        axes4[2].axis('off')

        axes4[3].imshow(roi_def, cmap='gray', alpha=0.5)
        for cnt in gt_contours:
            p = cnt.squeeze()
            if p.ndim == 2 and len(p) > 1:
                axes4[3].plot(p[:, 0], p[:, 1], 'g-', linewidth=2, alpha=0.7)
        for cnt in diff_data['contours']:
            p = cnt.squeeze()
            if p.ndim == 2 and len(p) > 1:
                axes4[3].plot(p[:, 0], p[:, 1], 'c-', linewidth=1.5, alpha=0.9)
        axes4[3].set_title(f'Canny 경계선\n{diff_data["params"]}')
        axes4[3].axis('off')

        fig4.tight_layout()
        path4 = OUTPUT_DIR / 'fig4_diff_canny_process.png'
        fig4.savefig(str(path4), dpi=150, bbox_inches='tight')
        print(f"  → {path4}")

    # ════════════════════════════════════════════════
    # 콘솔 최종 요약
    # ════════════════════════════════════════════════
    elapsed = time.time() - t_total
    print(f"\n{'='*70}")
    print("최종 결과 요약")
    print(f"{'='*70}")
    print(f"\n{'방법':<20} | {'설정':<30} | {'det→GT':>7} | {'GT→det':>7} | "
          f"{'Haus':>7} | {'팁오차':>7} | {'IoU':>6}")
    print("-" * 105)
    for cat_name, data in best.items():
        ev, tip, met = data['ev'], data['tip'], data['met']
        print(f"{cat_name:<20} | {data['key']:<30} | {ev['det2gt_mean']:>7.2f} | "
              f"{ev['gt2det_mean']:>7.2f} | {ev['hausdorff']:>7.2f} | "
              f"{tip['tip_error']:>7.1f} | {met['iou']:>6.4f}")

    print(f"\n총 소요: {elapsed:.1f}s")
    print(f"결과 저장: {OUTPUT_DIR}")
    print(f"{'='*70}")

    plt.show()
    print("완료!")


if __name__ == '__main__':
    run_test()

