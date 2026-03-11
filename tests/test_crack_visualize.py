"""
크랙 경계선 추출 비교: Mean+kσ vs Canny vs |Ref-Def|
— 3가지 방법으로 contour 경계선을 추출하고, GT contour와 경계 오차를 비교

위치: tests/test_crack_detection.py
"""
import sys
import numpy as np
import cv2
from pathlib import Path
from scipy.spatial import distance
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import platform
import time

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
OUTPUT_DIR = TEST_OUTPUT_ROOT / 'output_crack_detection'
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


def compute_metrics(gt, pred):
    """마스크 기반 IoU/Precision/Recall"""
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    tp = np.sum(gt_b & pred_b)
    fp = np.sum(~gt_b & pred_b)
    fn = np.sum(gt_b & ~pred_b)
    iou = tp / max(tp + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-10)
    return {'iou': iou, 'precision': precision, 'recall': recall,
            'f1': f1, 'tp': int(tp), 'fp': int(fp), 'fn': int(fn)}


def contours_to_points(contours):
    """contour 리스트 → (N,2) 배열 [x,y]"""
    if not contours:
        return np.array([]).reshape(0, 2)
    all_pts = []
    for cnt in contours:
        pts = cnt.squeeze()
        if pts.ndim == 2:
            all_pts.append(pts)
        elif pts.ndim == 1 and len(pts) == 2:
            all_pts.append(pts.reshape(1, 2))
    if all_pts:
        return np.vstack(all_pts).astype(np.float64)
    return np.array([]).reshape(0, 2)


def compute_boundary_errors(det_pts, gt_pts):
    """검출 경계점 ↔ GT 경계점 사이 오차 통계"""
    if len(det_pts) == 0 or len(gt_pts) == 0:
        return {'mean': float('inf'), 'max': float('inf'),
                'hausdorff': float('inf'), 'dists_det_to_gt': np.array([])}
    d2g = distance.cdist(det_pts, gt_pts).min(axis=1)
    g2d = distance.cdist(gt_pts, det_pts).min(axis=1)
    return {
        'mean': float(np.mean(d2g)),
        'median': float(np.median(d2g)),
        'max': float(np.max(d2g)),
        'hausdorff': max(float(np.max(d2g)), float(np.max(g2d))),
        'dists_det_to_gt': d2g,
        'dists_gt_to_det': g2d,
    }


# ============================================================
# 3가지 검출 방법 → 이진 마스크 + contour 추출 통합
# ============================================================

def detect_and_extract(img, method, ref_img=None,
                       sigma_blur=1.0, k=1.0,
                       canny_low=None, canny_high=None,
                       dilate_iter=1, min_area=10):
    """
    이미지에서 크랙 이진 마스크 생성 → contour 경계선 추출.

    Parameters
    ----------
    method : 'mean_k_sigma' | 'canny' | 'ref_diff'
    """
    ksize = max(3, int(np.ceil(sigma_blur * 3)) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    if method == 'mean_k_sigma':
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma_blur)
        mu, std = float(blurred.mean()), float(blurred.std())
        thr = mu - k * std
        binary = (blurred < thr).astype(np.uint8) * 255
        params_str = f"μ={mu:.1f}, σ={std:.1f}, thr={thr:.1f}, k={k}"

    elif method == 'canny':
        blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma_blur)
        if canny_low is None or canny_high is None:
            v = np.median(blurred)
            canny_low = int(max(0, 0.5 * v))
            canny_high = int(min(255, 1.0 * v))
        edges = cv2.Canny(blurred, canny_low, canny_high)
        if dilate_iter > 0:
            edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
        binary = edges
        params_str = f"low={canny_low}, high={canny_high}, dilate={dilate_iter}"

    elif method == 'ref_diff':
        if ref_img is None:
            raise ValueError("ref_diff에는 ref_img 필요")
        diff = np.abs(ref_img.astype(np.float64) - img.astype(np.float64))
        blurred = cv2.GaussianBlur(diff, (ksize, ksize), sigma_blur)
        mu, std = float(blurred.mean()), float(blurred.std())
        thr = mu + k * std
        binary = (blurred > thr).astype(np.uint8) * 255
        params_str = f"μ={mu:.1f}, σ={std:.1f}, thr={thr:.1f}, k={k}"

    else:
        raise ValueError(f"Unknown method: {method}")

    # 후처리
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # 작은 성분 제거
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            binary[labels == i] = 0

    # contour 추출
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_NONE)
    bnd_pts = contours_to_points(contours)

    return {
        'mask': binary,
        'contours': contours,
        'boundary_pts': bnd_pts,
        'params_str': params_str,
        'method': method,
    }


# ============================================================
# GT contour 추출
# ============================================================

def extract_gt_contours(gt_mask):
    """GT 마스크에서 contour 추출"""
    gt_binary = (gt_mask.astype(np.uint8)) * 255
    contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_NONE)
    bnd_pts = contours_to_points(contours)
    return contours, bnd_pts


# ============================================================
# 메인 시각화
# ============================================================

def visualize_boundary_comparison():
    t_total = time.time()

    # 데이터 로드
    def_img = _imread_unicode(DATA_DIR / 'deformed.tiff')
    ref_img = _imread_unicode(DATA_DIR / 'reference.tiff')
    gt_mask = np.load(str(DATA_DIR / 'crack_mask.npy'))
    H, W = def_img.shape

    gt_clean = gt_mask.copy()
    gt_clean[:220, :] = False
    gt_clean[280:, :] = False

    print(f"이미지: {W}×{H}, GT 크랙 픽셀: {gt_clean.sum()}")

    # ROI 설정
    roi_y1, roi_y2 = 230, 270
    roi_x1, roi_x2 = 0, 280
    roi_def = def_img[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_ref = ref_img[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_gt = gt_clean[roi_y1:roi_y2, roi_x1:roi_x2]
    rH, rW = roi_def.shape

    print(f"ROI: ({roi_x1},{roi_y1})~({roi_x2},{roi_y2}), {rW}×{rH}px")

    # GT contour
    gt_contours, gt_bnd_pts = extract_gt_contours(roi_gt)
    print(f"GT contour: {len(gt_contours)}개, 경계점: {len(gt_bnd_pts)}개")

    # ════════════════════════════════════════════════
    # 3가지 방법 경계선 추출
    # ════════════════════════════════════════════════
    sigma_blur = 1.0
    k_value = 1.0

    methods = [
        ('Mean+kσ', 'mean_k_sigma', {'k': k_value}),
        ('Canny', 'canny', {'dilate_iter': 1}),
        ('|Ref-Def|', 'ref_diff', {'k': k_value}),
    ]

    results = {}
    print(f"\n{'방법':<12} | {'Contours':>8} | {'경계점':>7} | "
          f"{'평균오차':>8} | {'Hausdorff':>9} | {'IoU':>6} | {'P':>6} | {'R':>6}")
    print("-" * 85)

    for name, method, kwargs in methods:
        t0 = time.time()
        res = detect_and_extract(
            roi_def, method, ref_img=roi_ref,
            sigma_blur=sigma_blur, **kwargs
        )
        elapsed = time.time() - t0

        # 경계 오차
        bnd_err = compute_boundary_errors(res['boundary_pts'], gt_bnd_pts)

        # 마스크 메트릭
        mask_metrics = compute_metrics(roi_gt, res['mask'] > 0)

        res['boundary_errors'] = bnd_err
        res['mask_metrics'] = mask_metrics
        res['elapsed'] = elapsed
        results[name] = res

        print(f"{name:<12} | {len(res['contours']):>8} | {len(res['boundary_pts']):>7} | "
              f"{bnd_err['mean']:>7.2f}px | {bnd_err['hausdorff']:>8.2f}px | "
              f"{mask_metrics['iou']:>6.4f} | {mask_metrics['precision']:>6.4f} | "
              f"{mask_metrics['recall']:>6.4f}")

    # ════════════════════════════════════════════════
    # Figure 1: 3가지 방법 경계선 비교
    # ════════════════════════════════════════════════
    fig1 = plt.figure(figsize=(22, 16))
    fig1.suptitle('Figure 1: 크랙 경계선 추출 비교 — Mean+kσ vs Canny vs |Ref-Def|',
                  fontsize=16, fontweight='bold')
    gs1 = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.3)

    colors = {'Mean+kσ': 'red', 'Canny': 'orange', '|Ref-Def|': 'cyan'}

    for row, (name, _, _) in enumerate(methods):
        res = results[name]
        bnd_err = res['boundary_errors']
        metrics = res['mask_metrics']

        # Col 0: 원본 + 경계선 오버레이 (검출=빨강, GT=초록)
        ax = fig1.add_subplot(gs1[row, 0])
        ax.imshow(roi_def, cmap='gray')
        for cnt in gt_contours:
            pts = cnt.squeeze()
            if pts.ndim == 2 and len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=2, alpha=0.8)
        for cnt in res['contours']:
            pts = cnt.squeeze()
            if pts.ndim == 2 and len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], color=colors[name],
                        linewidth=2, alpha=0.9)
        ax.set_title(f'{name} 경계선\n{res["params_str"]}', fontsize=9)
        ax.set_ylabel(name, fontsize=12, fontweight='bold')
        ax.axis('off')

        # Col 1: TP/FP/FN 마스크
        ax = fig1.add_subplot(gs1[row, 1])
        vis = np.zeros((*roi_gt.shape, 3), dtype=np.uint8)
        det_mask = res['mask'] > 0
        vis[det_mask & roi_gt] = [0, 255, 0]       # TP
        vis[det_mask & ~roi_gt] = [255, 0, 0]      # FP
        vis[~det_mask & roi_gt] = [0, 0, 255]      # FN
        ax.imshow(vis)
        ax.set_title(f'TP/FP/FN\n'
                     f'IoU={metrics["iou"]:.3f} P={metrics["precision"]:.3f} '
                     f'R={metrics["recall"]:.3f}', fontsize=9)
        ax.axis('off')

        # Col 2: 경계점별 GT까지 거리 히트맵
        ax = fig1.add_subplot(gs1[row, 2])
        ax.imshow(roi_def, cmap='gray', alpha=0.4)
        if len(res['boundary_pts']) > 0 and len(bnd_err['dists_det_to_gt']) > 0:
            sc = ax.scatter(res['boundary_pts'][:, 0], res['boundary_pts'][:, 1],
                           c=bnd_err['dists_det_to_gt'], cmap='hot_r', s=3,
                           vmin=0, vmax=max(3, np.percentile(bnd_err['dists_det_to_gt'], 95)))
            plt.colorbar(sc, ax=ax, label='GT까지 거리(px)', shrink=0.8)
        for cnt in gt_contours:
            pts = cnt.squeeze()
            if pts.ndim == 2 and len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=1.5, alpha=0.6)
        ax.set_title(f'경계점 오차 히트맵\n평균={bnd_err["mean"]:.2f}px', fontsize=9)
        ax.axis('off')

        # Col 3: 경계 오차 분포
        ax = fig1.add_subplot(gs1[row, 3])
        if len(bnd_err['dists_det_to_gt']) > 0:
            ax.hist(bnd_err['dists_det_to_gt'], bins=30,
                    color=colors[name], edgecolor='black', alpha=0.7)
            ax.axvline(bnd_err['mean'], color='navy', linewidth=2, linestyle='--',
                       label=f'평균={bnd_err["mean"]:.2f}px')
            ax.axvline(bnd_err['hausdorff'], color='red', linewidth=1.5, linestyle=':',
                       label=f'Hausdorff={bnd_err["hausdorff"]:.2f}px')
            ax.legend(fontsize=7)
        ax.set_xlabel('GT까지 거리 (px)')
        ax.set_ylabel('빈도')
        ax.set_title(f'경계 오차 분포', fontsize=9)
        ax.grid(True, alpha=0.3)

    legend_patches = [
        mpatches.Patch(color='green', alpha=0.8, label='GT 경계선'),
        mpatches.Patch(color='red', alpha=0.8, label='Mean+kσ'),
        mpatches.Patch(color='orange', alpha=0.8, label='Canny'),
        mpatches.Patch(color='cyan', alpha=0.8, label='|Ref-Def|'),
    ]
    fig1.legend(handles=legend_patches, loc='lower center', ncol=4,
                fontsize=11, frameon=True)
    fig1.tight_layout(rect=[0, 0.04, 1, 0.95])
    path1 = OUTPUT_DIR / 'fig1_boundary_3method_compare.png'
    fig1.savefig(str(path1), dpi=150, bbox_inches='tight')
    print(f"\n저장: {path1}")

    # ════════════════════════════════════════════════
    # Figure 2: 파라미터 민감도 — 경계 오차 기준
    # ════════════════════════════════════════════════
    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle('Figure 2: 파라미터 민감도 — 경계 오차 기준', fontsize=16, fontweight='bold')

    # (a) Mean+kσ: k값 변화
    k_values = np.arange(0.4, 2.6, 0.2)
    mean_errs_k = []
    haus_errs_k = []
    ious_k = []
    for k in k_values:
        res = detect_and_extract(roi_def, 'mean_k_sigma', sigma_blur=sigma_blur, k=k)
        err = compute_boundary_errors(res['boundary_pts'], gt_bnd_pts)
        met = compute_metrics(roi_gt, res['mask'] > 0)
        mean_errs_k.append(err['mean'])
        haus_errs_k.append(err['hausdorff'])
        ious_k.append(met['iou'])

    ax = axes2[0, 0]
    ax.plot(k_values, mean_errs_k, 'bo-', markersize=5, label='평균 경계 오차')
    ax.plot(k_values, haus_errs_k, 'rs--', markersize=5, label='Hausdorff')
    best_k_idx = np.argmin(mean_errs_k)
    ax.axvline(k_values[best_k_idx], color='green', linestyle=':', linewidth=2,
               label=f'최적 k={k_values[best_k_idx]:.1f} (평균={mean_errs_k[best_k_idx]:.2f}px)')
    ax.set_xlabel('k 값')
    ax.set_ylabel('경계 오차 (px)')
    ax.set_title('(a) Mean+kσ: k → 경계 오차')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(k_values, ious_k, 'g^--', markersize=4, alpha=0.6)
    ax2.set_ylabel('IoU', color='green')

    # (b) Canny: high 임계값 변화
    canny_highs = np.arange(30, 180, 10)
    mean_errs_c = []
    haus_errs_c = []
    ious_c = []
    for ch in canny_highs:
        cl = int(ch * 0.5)
        res = detect_and_extract(roi_def, 'canny', sigma_blur=sigma_blur,
                                  canny_low=cl, canny_high=int(ch))
        err = compute_boundary_errors(res['boundary_pts'], gt_bnd_pts)
        met = compute_metrics(roi_gt, res['mask'] > 0)
        mean_errs_c.append(err['mean'])
        haus_errs_c.append(err['hausdorff'])
        ious_c.append(met['iou'])

    ax = axes2[0, 1]
    ax.plot(canny_highs, mean_errs_c, 'bo-', markersize=5, label='평균 경계 오차')
    ax.plot(canny_highs, haus_errs_c, 'rs--', markersize=5, label='Hausdorff')
    best_c_idx = np.argmin(mean_errs_c)
    ax.axvline(canny_highs[best_c_idx], color='green', linestyle=':', linewidth=2,
               label=f'최적 high={canny_highs[best_c_idx]} '
                     f'(평균={mean_errs_c[best_c_idx]:.2f}px)')
    ax.set_xlabel('Canny high threshold')
    ax.set_ylabel('경계 오차 (px)')
    ax.set_title('(b) Canny: threshold → 경계 오차')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(canny_highs, ious_c, 'g^--', markersize=4, alpha=0.6)
    ax2.set_ylabel('IoU', color='green')

    # (c) |Ref-Def|: k값 변화
    mean_errs_d = []
    haus_errs_d = []
    ious_d = []
    for k in k_values:
        res = detect_and_extract(roi_def, 'ref_diff', ref_img=roi_ref,
                                  sigma_blur=sigma_blur, k=k)
        err = compute_boundary_errors(res['boundary_pts'], gt_bnd_pts)
        met = compute_metrics(roi_gt, res['mask'] > 0)
        mean_errs_d.append(err['mean'])
        haus_errs_d.append(err['hausdorff'])
        ious_d.append(met['iou'])

    ax = axes2[0, 2]
    ax.plot(k_values, mean_errs_d, 'bo-', markersize=5, label='평균 경계 오차')
    ax.plot(k_values, haus_errs_d, 'rs--', markersize=5, label='Hausdorff')
    best_d_idx = np.argmin(mean_errs_d)
    ax.axvline(k_values[best_d_idx], color='green', linestyle=':', linewidth=2,
               label=f'최적 k={k_values[best_d_idx]:.1f} '
                     f'(평균={mean_errs_d[best_d_idx]:.2f}px)')
    ax.set_xlabel('k 값')
    ax.set_ylabel('경계 오차 (px)')
    ax.set_title('(c) |Ref-Def|: k → 경계 오차')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    ax2 = ax.twinx()
    ax2.plot(k_values, ious_d, 'g^--', markersize=4, alpha=0.6)
    ax2.set_ylabel('IoU', color='green')

    # ── Row 2: 종합 비교 (최적 파라미터) ──

    # (d) 최적 파라미터 경계선 오버레이
    best_k_mean = k_values[best_k_idx]
    best_ch_canny = canny_highs[best_c_idx]
    best_k_diff = k_values[best_d_idx]

    res_best_m = detect_and_extract(roi_def, 'mean_k_sigma', sigma_blur=sigma_blur, k=best_k_mean)
    res_best_c = detect_and_extract(roi_def, 'canny', sigma_blur=sigma_blur,
                                     canny_low=int(best_ch_canny * 0.5),
                                     canny_high=int(best_ch_canny))
    res_best_d = detect_and_extract(roi_def, 'ref_diff', ref_img=roi_ref,
                                     sigma_blur=sigma_blur, k=best_k_diff)

    ax = axes2[1, 0]
    ax.imshow(roi_def, cmap='gray')
    for cnt in gt_contours:
        pts = cnt.squeeze()
        if pts.ndim == 2 and len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=3, alpha=0.8, label='GT')
    for cnt in res_best_m['contours']:
        pts = cnt.squeeze()
        if pts.ndim == 2 and len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=1.5, alpha=0.9)
    for cnt in res_best_c['contours']:
        pts = cnt.squeeze()
        if pts.ndim == 2 and len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], color='orange', linewidth=1.5, alpha=0.9)
    for cnt in res_best_d['contours']:
        pts = cnt.squeeze()
        if pts.ndim == 2 and len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], 'c-', linewidth=1.5, alpha=0.9)
    ax.set_title('(d) 최적 경계선 오버레이\n(초록=GT)')
    ax.axis('off')

    # (e) 종합 바 차트
    err_best_m = compute_boundary_errors(res_best_m['boundary_pts'], gt_bnd_pts)
    err_best_c = compute_boundary_errors(res_best_c['boundary_pts'], gt_bnd_pts)
    err_best_d = compute_boundary_errors(res_best_d['boundary_pts'], gt_bnd_pts)

    met_best_m = compute_metrics(roi_gt, res_best_m['mask'] > 0)
    met_best_c = compute_metrics(roi_gt, res_best_c['mask'] > 0)
    met_best_d = compute_metrics(roi_gt, res_best_d['mask'] > 0)

    ax = axes2[1, 1]
    method_names = [f'Mean+kσ\nk={best_k_mean:.1f}',
                    f'Canny\nhigh={best_ch_canny}',
                    f'|Ref-Def|\nk={best_k_diff:.1f}']
    mean_vals = [err_best_m['mean'], err_best_c['mean'], err_best_d['mean']]
    haus_vals = [err_best_m['hausdorff'], err_best_c['hausdorff'], err_best_d['hausdorff']]

    x_pos = np.arange(3)
    bars1 = ax.bar(x_pos - 0.2, mean_vals, 0.35, color='steelblue',
                   edgecolor='black', label='평균 오차')
    bars2 = ax.bar(x_pos + 0.2, haus_vals, 0.35, color='coral',
                   edgecolor='black', label='Hausdorff')
    for bar, val in zip(bars1, mean_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontsize=8, fontweight='bold')
    for bar, val in zip(bars2, haus_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f'{val:.2f}', ha='center', fontsize=8, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, fontsize=9)
    ax.set_ylabel('경계 오차 (px)')
    ax.set_title('(e) 최적 파라미터 경계 오차 비교')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')

    # (f) IoU 비교
    ax = axes2[1, 2]
    iou_vals = [met_best_m['iou'], met_best_c['iou'], met_best_d['iou']]
    f1_vals = [met_best_m['f1'], met_best_c['f1'], met_best_d['f1']]
    prec_vals = [met_best_m['precision'], met_best_c['precision'], met_best_d['precision']]
    rec_vals = [met_best_m['recall'], met_best_c['recall'], met_best_d['recall']]

    w = 0.18
    ax.bar(x_pos - 1.5 * w, iou_vals, w, color='steelblue', edgecolor='black', label='IoU')
    ax.bar(x_pos - 0.5 * w, f1_vals, w, color='goldenrod', edgecolor='black', label='F1')
    ax.bar(x_pos + 0.5 * w, prec_vals, w, color='mediumseagreen', edgecolor='black', label='Precision')
    ax.bar(x_pos + 1.5 * w, rec_vals, w, color='coral', edgecolor='black', label='Recall')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_names, fontsize=9)
    ax.set_ylabel('Score')
    ax.set_title('(f) 최적 파라미터 마스크 성능 비교')
    ax.legend(fontsize=7, loc='upper right')
    ax.set_ylim(0, 1.15)
    ax.grid(True, alpha=0.3, axis='y')

    fig2.tight_layout()
    path2 = OUTPUT_DIR / 'fig2_sensitivity_boundary_error.png'
    fig2.savefig(str(path2), dpi=150, bbox_inches='tight')
    print(f"저장: {path2}")

    # ════════════════════════════════════════════════
    # Figure 3: Tip 근처 확대 — 경계선 정밀도
    # ════════════════════════════════════════════════
    subset_size = 21
    M = subset_size // 2
    tip_pois = [
        (181, 251, "opening~4px"),
        (221, 251, "opening~2px"),
        (241, 251, "마지막 불량"),
        (251, 251, "첫 정상"),
    ]

    fig3, axes3 = plt.subplots(len(tip_pois), 4, figsize=(18, 4 * len(tip_pois)))
    fig3.suptitle('Figure 3: Tip 근처 확대 — 경계선 정밀도 비교', fontsize=16, fontweight='bold')

    for row, (px, py, desc) in enumerate(tip_pois):
        y1, y2 = py - M, py + M + 1
        x1, x2 = px - M, px + M + 1
        patch_d = def_img[y1:y2, x1:x2]
        patch_r = ref_img[y1:y2, x1:x2]
        gt_p = gt_clean[y1:y2, x1:x2]
        gt_cnt_p, gt_pts_p = extract_gt_contours(gt_p)

        res_m = detect_and_extract(patch_d, 'mean_k_sigma', sigma_blur=sigma_blur, k=best_k_mean)
        res_c = detect_and_extract(patch_d, 'canny', sigma_blur=sigma_blur,
                                    canny_low=int(best_ch_canny * 0.5),
                                    canny_high=int(best_ch_canny))
        res_d = detect_and_extract(patch_d, 'ref_diff', ref_img=patch_r,
                                    sigma_blur=sigma_blur, k=best_k_diff)

        patch_results = [('Mean+kσ', res_m, 'red'),
                         ('Canny', res_c, 'orange'),
                         ('|Ref-Def|', res_d, 'cyan')]

        # Col 0: 원본 + GT 경계
        axes3[row, 0].imshow(patch_d, cmap='gray', vmin=0, vmax=255)
        for cnt in gt_cnt_p:
            pts = cnt.squeeze()
            if pts.ndim == 2 and len(pts) > 1:
                axes3[row, 0].plot(pts[:, 0], pts[:, 1], 'g-', linewidth=2)
        axes3[row, 0].set_title(f'({px},{py}) {desc}\nGT경계(초록)', fontsize=9)
        axes3[row, 0].axis('off')

        # Col 1~3: 각 방법
        for col, (name, res, color) in enumerate(patch_results):
            ax = axes3[row, col + 1]
            ax.imshow(patch_d, cmap='gray', vmin=0, vmax=255)
            for cnt in gt_cnt_p:
                pts = cnt.squeeze()
                if pts.ndim == 2 and len(pts) > 1:
                    ax.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=2, alpha=0.6)
            for cnt in res['contours']:
                pts = cnt.squeeze()
                if pts.ndim == 2 and len(pts) > 1:
                    ax.plot(pts[:, 0], pts[:, 1], color=color, linewidth=2, alpha=0.9)

            err = compute_boundary_errors(res['boundary_pts'], gt_pts_p)
            ax.set_title(f'{name}\n평균오차={err["mean"]:.2f}px '
                         f'Haus={err["hausdorff"]:.2f}px', fontsize=9)
            ax.axis('off')

    fig3.tight_layout()
    path3 = OUTPUT_DIR / 'fig3_tip_boundary_compare.png'
    fig3.savefig(str(path3), dpi=150, bbox_inches='tight')
    print(f"저장: {path3}")

    # ════════════════════════════════════════════════
    # 콘솔 최종 요약
    # ════════════════════════════════════════════════
    elapsed_total = time.time() - t_total
    print(f"\n{'='*70}")
    print("크랙 경계선 추출 비교 최종 결과")
    print(f"{'='*70}")
    print(f"\n{'방법':<12} | {'최적파라미터':>14} | {'평균오차':>8} | "
          f"{'Hausdorff':>9} | {'IoU':>6} | {'F1':>6}")
    print("-" * 75)
    print(f"{'Mean+kσ':<12} | {'k='+str(best_k_mean):>14} | "
          f"{err_best_m['mean']:>7.2f}px | {err_best_m['hausdorff']:>8.2f}px | "
          f"{met_best_m['iou']:>6.4f} | {met_best_m['f1']:>6.4f}")
    print(f"{'Canny':<12} | {'high='+str(best_ch_canny):>14} | "
          f"{err_best_c['mean']:>7.2f}px | {err_best_c['hausdorff']:>8.2f}px | "
          f"{met_best_c['iou']:>6.4f} | {met_best_c['f1']:>6.4f}")
    print(f"{'|Ref-Def|':<12} | {'k='+str(best_k_diff):>14} | "
          f"{err_best_d['mean']:>7.2f}px | {err_best_d['hausdorff']:>8.2f}px | "
          f"{met_best_d['iou']:>6.4f} | {met_best_d['f1']:>6.4f}")
    print(f"\n총 소요시간: {elapsed_total:.2f}s")
    print(f"결과 저장: {OUTPUT_DIR}")
    print(f"{'='*70}")

    plt.show()
    print("\n완료!")


if __name__ == '__main__':
    visualize_boundary_comparison()

