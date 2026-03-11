# tests/test_crack_detection.py
"""
크랙 경계 검출 성능 검증 + 시각화

synthetic_crack_data의 GT 마스크와 비교하여
Otsu / Mean+k·σ 의 크랙 분리 성능을 정량 평가한다.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from speckle.core.crack_detection import (
    detect_crack_otsu,
    detect_crack_mean_k_sigma,
    evaluate_detection,
)

# ===== 경로 설정 =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'synthetic_crack_data'
TEST_OUTPUT_ROOT = PROJECT_ROOT / 'tests' / '_outputs'
OUTPUT_DIR = TEST_OUTPUT_ROOT / 'output_crack_detection'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def load_test_data():
    """합성 크랙 데이터 로드."""
    def_image = cv2.imread(
        str(DATA_DIR / 'deformed.tiff'), cv2.IMREAD_GRAYSCALE
    )
    gt_mask = np.load(str(DATA_DIR / 'crack_mask.npy'))

    if def_image is None:
        raise FileNotFoundError(
            f"{DATA_DIR / 'deformed.tiff'} 를 찾을 수 없습니다. "
            f"python tests/generate_crack_images.py 를 먼저 실행하세요."
        )

    print(f"이미지 크기: {def_image.shape}")
    print(f"GT 크랙 픽셀: {np.sum(gt_mask)} "
          f"({np.sum(gt_mask) / gt_mask.size * 100:.2f}%)")
    print(f"이미지 밝기: min={def_image.min()}, max={def_image.max()}, "
          f"mean={def_image.mean():.1f}")

    return def_image, gt_mask


# ===== Test 1: Otsu σ sweep (전체 이미지) + 시각화 =====

def test_otsu_sigma_sweep():
    print("\n" + "=" * 70)
    print("TEST 1: Otsu — Gaussian σ sweep (전체 이미지)")
    print("=" * 70)

    image, gt_mask = load_test_data()
    sigmas = [0, 0.5, 1.0, 1.5, 2.0, 3.0]

    results_list = []
    metrics_list = []

    print(f"\n{'σ':>5s} | {'Thr':>4s} | {'IoU':>6s} | {'Prec':>6s} | "
          f"{'Recall':>6s} | {'F1':>6s} | {'BndErr':>7s} | {'CrackPx':>8s}")
    print("-" * 70)

    best_iou = -1
    best_sigma = 0
    best_idx = 0

    for i, sigma in enumerate(sigmas):
        result = detect_crack_otsu(
            image, gaussian_sigma=sigma, morph_size=3, min_crack_area=10,
        )
        metrics = evaluate_detection(result['crack_mask'], gt_mask)

        results_list.append(result)
        metrics_list.append(metrics)

        crack_px = int(np.sum(result['crack_mask']))
        bnd = metrics['boundary_error_mean']
        bnd_str = f"{bnd:.2f}" if bnd != float('inf') else "  inf"

        print(f"{sigma:5.1f} | {result['threshold']:4d} | "
              f"{metrics['iou']:6.4f} | {metrics['precision']:6.4f} | "
              f"{metrics['recall']:6.4f} | {metrics['f1']:6.4f} | "
              f"{bnd_str:>7s} | {crack_px:8d}")

        if metrics['iou'] > best_iou:
            best_iou = metrics['iou']
            best_sigma = sigma
            best_idx = i

    print(f"\n→ 최적: σ={best_sigma}, IoU={best_iou:.4f}")

    # --- 시각화: Figure 1 ---
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Test 1: Otsu — Gaussian σ Sweep (Full Image)', fontsize=16, fontweight='bold')

    gs = gridspec.GridSpec(3, 6, hspace=0.35, wspace=0.3)

    # Row 1: 각 σ별 검출 마스크
    for i, sigma in enumerate(sigmas):
        ax = fig.add_subplot(gs[0, i])
        mask = results_list[i]['crack_mask']
        # TP/FP/FN 컬러맵
        vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
        tp = mask & gt_mask
        fp = mask & ~gt_mask
        fn = ~mask & gt_mask
        vis[tp] = [0, 255, 0]    # TP: 초록
        vis[fp] = [255, 0, 0]    # FP: 빨강
        vis[fn] = [0, 0, 255]    # FN: 파랑
        ax.imshow(vis)
        iou_val = metrics_list[i]['iou']
        ax.set_title(f'σ={sigma}\nIoU={iou_val:.3f}\nThr={results_list[i]["threshold"]}',
                      fontsize=9)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('Detection\n(G=TP R=FP B=FN)', fontsize=9)

    # Row 2: 성능 그래프
    ious = [m['iou'] for m in metrics_list]
    precs = [m['precision'] for m in metrics_list]
    recs = [m['recall'] for m in metrics_list]
    f1s = [m['f1'] for m in metrics_list]
    bnds = [m['boundary_error_mean'] if m['boundary_error_mean'] != float('inf') else np.nan
            for m in metrics_list]
    thrs = [r['threshold'] for r in results_list]

    # 2a: IoU, Precision, Recall, F1
    ax1 = fig.add_subplot(gs[1, 0:3])
    ax1.plot(sigmas, ious, 'ko-', linewidth=2, markersize=8, label='IoU')
    ax1.plot(sigmas, precs, 's--', color='tab:orange', label='Precision')
    ax1.plot(sigmas, recs, '^--', color='tab:blue', label='Recall')
    ax1.plot(sigmas, f1s, 'd--', color='tab:green', label='F1')
    ax1.axvline(best_sigma, color='red', linestyle=':', alpha=0.7, label=f'Best σ={best_sigma}')
    ax1.set_xlabel('Gaussian σ')
    ax1.set_ylabel('Score')
    ax1.set_title('Detection Metrics vs Gaussian σ')
    ax1.legend(fontsize=8)
    ax1.set_ylim(0, 1.05)
    ax1.grid(True, alpha=0.3)

    # 2b: Threshold & Boundary Error
    ax2 = fig.add_subplot(gs[1, 3:5])
    color1 = 'tab:red'
    ax2.plot(sigmas, thrs, 'o-', color=color1, linewidth=2, label='Threshold')
    ax2.set_xlabel('Gaussian σ')
    ax2.set_ylabel('Otsu Threshold', color=color1)
    ax2.tick_params(axis='y', labelcolor=color1)
    ax2.set_title('Threshold & Boundary Error')

    ax2b = ax2.twinx()
    color2 = 'tab:purple'
    ax2b.plot(sigmas, bnds, 's--', color=color2, linewidth=2, label='BndErr (px)')
    ax2b.set_ylabel('Boundary Error (px)', color=color2)
    ax2b.tick_params(axis='y', labelcolor=color2)
    ax2.grid(True, alpha=0.3)

    # 2c: 검출 픽셀 수 vs GT
    ax3 = fig.add_subplot(gs[1, 5])
    crack_pxs = [int(np.sum(r['crack_mask'])) for r in results_list]
    gt_px = int(np.sum(gt_mask))
    ax3.barh(range(len(sigmas)), crack_pxs, color='tab:orange', alpha=0.7, label='Detected')
    ax3.axvline(gt_px, color='green', linewidth=2, linestyle='--', label=f'GT={gt_px}')
    ax3.set_yticks(range(len(sigmas)))
    ax3.set_yticklabels([f'σ={s}' for s in sigmas], fontsize=8)
    ax3.set_xlabel('Crack Pixels')
    ax3.set_title('Detected vs GT')
    ax3.legend(fontsize=7)

    # Row 3: 최적 σ의 상세 분석
    best_result = results_list[best_idx]
    best_mask = best_result['crack_mask']

    # 3a: 원본 이미지 + GT 경계
    ax4 = fig.add_subplot(gs[2, 0:2])
    ax4.imshow(image, cmap='gray')
    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in gt_contours:
        pts = c.squeeze()
        if pts.ndim == 2:
            ax4.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=1, alpha=0.8)
    det_contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in det_contours:
        pts = c.squeeze()
        if pts.ndim == 2:
            ax4.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=1, alpha=0.8)
    ax4.set_title(f'Best σ={best_sigma}: Contours (Green=GT, Red=Det)')
    ax4.axis('off')

    # 3b: 히스토그램 + 임계값
    ax5 = fig.add_subplot(gs[2, 2:4])
    if best_sigma > 0:
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(image.astype(np.float64), sigma=best_sigma)
    else:
        blurred = image.astype(np.float64)
    ax5.hist(blurred.ravel(), bins=256, range=(0, 255), color='gray', alpha=0.7, density=True)
    ax5.axvline(best_result['threshold'], color='red', linewidth=2, linestyle='--',
                label=f"Otsu Thr={best_result['threshold']}")
    # GT 크랙 영역 밝기 분포
    crack_pixels = blurred[gt_mask]
    non_crack_pixels = blurred[~gt_mask]
    ax5.hist(crack_pixels, bins=128, range=(0, 255), color='blue', alpha=0.4,
             density=True, label='GT Crack pixels')
    ax5.hist(non_crack_pixels, bins=128, range=(0, 255), color='green', alpha=0.3,
             density=True, label='GT Non-crack pixels')
    ax5.set_xlabel('Intensity')
    ax5.set_ylabel('Density')
    ax5.set_title(f'Intensity Distribution (σ={best_sigma})')
    ax5.legend(fontsize=8)

    # 3c: 크랙 중심 행 프로파일
    ax6 = fig.add_subplot(gs[2, 4:6])
    center_y = 250  # 합성 데이터 크랙 중심
    profile = blurred[center_y, :]
    gt_profile = gt_mask[center_y, :].astype(float)
    det_profile = best_mask[center_y, :].astype(float)
    ax6.plot(profile, 'gray', alpha=0.7, label='Intensity')
    ax6.axhline(best_result['threshold'], color='red', linestyle='--', alpha=0.7, label='Threshold')
    ax6_b = ax6.twinx()
    ax6_b.fill_between(range(len(gt_profile)), gt_profile * 0.5, alpha=0.3, color='green', label='GT')
    ax6_b.fill_between(range(len(det_profile)), det_profile * 0.5 + 0.5, alpha=0.3, color='red', label='Det')
    ax6.set_xlabel('X pixel')
    ax6.set_ylabel('Intensity')
    ax6_b.set_ylabel('Mask')
    ax6.set_title(f'Profile at y={center_y}')
    ax6.legend(loc='upper left', fontsize=7)
    ax6_b.legend(loc='upper right', fontsize=7)

    plt.savefig(str(OUTPUT_DIR / 'fig1_otsu_sigma_sweep.png'), dpi=150, bbox_inches='tight')
    print(f"  → 저장: {OUTPUT_DIR / 'fig1_otsu_sigma_sweep.png'}")

    return best_sigma, best_iou


# ===== Test 2: Mean+k·σ sweep + 시각화 =====

def test_mean_k_sigma_sweep():
    print("\n" + "=" * 70)
    print("TEST 2: Mean + k·σ — k & Gaussian σ sweep (전체 이미지)")
    print("=" * 70)

    image, gt_mask = load_test_data()
    k_values = [1.5, 2.0, 2.5, 3.0, 3.5]
    sigmas = [0, 1.0, 2.0]

    # 결과 저장 (sigma → k → metrics)
    all_results = {}
    best_iou = -1
    best_k = 0
    best_sigma = 0
    best_result = None

    for sigma in sigmas:
        all_results[sigma] = {}
        print(f"\n--- Gaussian σ = {sigma} ---")
        print(f"{'k':>5s} | {'Thr':>6s} | {'IoU':>6s} | {'Prec':>6s} | "
              f"{'Recall':>6s} | {'F1':>6s} | {'BndErr':>7s} | {'CrackPx':>8s}")
        print("-" * 70)

        for k in k_values:
            result = detect_crack_mean_k_sigma(
                image, k=k, gaussian_sigma=sigma,
                morph_size=3, min_crack_area=10,
            )
            metrics = evaluate_detection(result['crack_mask'], gt_mask)
            all_results[sigma][k] = {'result': result, 'metrics': metrics}

            crack_px = int(np.sum(result['crack_mask']))
            bnd = metrics['boundary_error_mean']
            bnd_str = f"{bnd:.2f}" if bnd != float('inf') else "  inf"

            print(f"{k:5.1f} | {result['threshold']:6.1f} | "
                  f"{metrics['iou']:6.4f} | {metrics['precision']:6.4f} | "
                  f"{metrics['recall']:6.4f} | {metrics['f1']:6.4f} | "
                  f"{bnd_str:>7s} | {crack_px:8d}")

            if metrics['iou'] > best_iou:
                best_iou = metrics['iou']
                best_k = k
                best_sigma = sigma
                best_result = result

    print(f"\n→ 최적: k={best_k}, σ={best_sigma}, IoU={best_iou:.4f}")

    # --- 시각화: Figure 2 ---
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle('Test 2: Mean + k·σ Sweep (Full Image)', fontsize=16, fontweight='bold')

    gs = gridspec.GridSpec(3, 5, hspace=0.35, wspace=0.35)

    # Row 1: 각 k별 검출 마스크 (σ=best_sigma)
    for i, k in enumerate(k_values):
        ax = fig.add_subplot(gs[0, i])
        entry = all_results[best_sigma][k]
        mask = entry['result']['crack_mask']
        vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
        vis[mask & gt_mask] = [0, 255, 0]
        vis[mask & ~gt_mask] = [255, 0, 0]
        vis[~mask & gt_mask] = [0, 0, 255]
        ax.imshow(vis)
        iou_val = entry['metrics']['iou']
        thr_val = entry['result']['threshold']
        ax.set_title(f'k={k}\nIoU={iou_val:.3f}\nThr={thr_val:.1f}', fontsize=9)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel(f'σ={best_sigma}\n(G=TP R=FP B=FN)', fontsize=9)

    # Row 2: 2D 히트맵 (σ × k → IoU)
    ax_heat = fig.add_subplot(gs[1, 0:2])
    iou_matrix = np.zeros((len(sigmas), len(k_values)))
    for si, sigma in enumerate(sigmas):
        for ki, k in enumerate(k_values):
            iou_matrix[si, ki] = all_results[sigma][k]['metrics']['iou']
    im = ax_heat.imshow(iou_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax_heat.set_xticks(range(len(k_values)))
    ax_heat.set_xticklabels([f'{k}' for k in k_values])
    ax_heat.set_yticks(range(len(sigmas)))
    ax_heat.set_yticklabels([f'{s}' for s in sigmas])
    ax_heat.set_xlabel('k')
    ax_heat.set_ylabel('Gaussian σ')
    ax_heat.set_title('IoU Heatmap (σ × k)')
    for si in range(len(sigmas)):
        for ki in range(len(k_values)):
            ax_heat.text(ki, si, f'{iou_matrix[si, ki]:.3f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax_heat, shrink=0.8)

    # Row 2: Precision-Recall 곡선
    ax_pr = fig.add_subplot(gs[1, 2:4])
    colors_sigma = ['tab:blue', 'tab:orange', 'tab:green']
    for si, sigma in enumerate(sigmas):
        precs = [all_results[sigma][k]['metrics']['precision'] for k in k_values]
        recs = [all_results[sigma][k]['metrics']['recall'] for k in k_values]
        ax_pr.plot(recs, precs, 'o-', color=colors_sigma[si], label=f'σ={sigma}', markersize=8)
        for ki, k in enumerate(k_values):
            ax_pr.annotate(f'k={k}', (recs[ki], precs[ki]), fontsize=7,
                           textcoords='offset points', xytext=(5, 5))
    ax_pr.set_xlabel('Recall')
    ax_pr.set_ylabel('Precision')
    ax_pr.set_title('Precision-Recall Curve')
    ax_pr.legend(fontsize=8)
    ax_pr.set_xlim(0, 1.05)
    ax_pr.set_ylim(0, 1.05)
    ax_pr.grid(True, alpha=0.3)

    # Row 2: 최적 파라미터 요약 텍스트
    ax_txt = fig.add_subplot(gs[1, 4])
    ax_txt.axis('off')
    best_m = all_results[best_sigma][best_k]['metrics']
    summary_text = (
        f"Best Parameters\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"k = {best_k}\n"
        f"Gaussian σ = {best_sigma}\n"
        f"Threshold = {best_result['threshold']:.1f}\n\n"
        f"Performance\n"
        f"━━━━━━━━━━━━━━━━━━\n"
        f"IoU      = {best_m['iou']:.4f}\n"
        f"Precision= {best_m['precision']:.4f}\n"
        f"Recall   = {best_m['recall']:.4f}\n"
        f"F1       = {best_m['f1']:.4f}\n"
        f"BndErr   = {best_m['boundary_error_mean']:.2f} px"
    )
    ax_txt.text(0.1, 0.95, summary_text, transform=ax_txt.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    # Row 3: 히스토그램 + 최적 Mean+kσ 분석
    if best_sigma > 0:
        from scipy.ndimage import gaussian_filter
        blurred = gaussian_filter(image.astype(np.float64), sigma=best_sigma)
    else:
        blurred = image.astype(np.float64)

    mean_val = blurred.mean()
    std_val = blurred.std()

    # 3a: 히스토그램
    ax_hist = fig.add_subplot(gs[2, 0:2])
    ax_hist.hist(blurred.ravel(), bins=256, range=(0, 255), color='gray', alpha=0.7, density=True)
    crack_pixels = blurred[gt_mask]
    non_crack_pixels = blurred[~gt_mask]
    ax_hist.hist(crack_pixels, bins=128, range=(0, 255), color='blue', alpha=0.4,
                 density=True, label='GT Crack')
    ax_hist.hist(non_crack_pixels, bins=128, range=(0, 255), color='green', alpha=0.3,
                 density=True, label='GT Non-crack')
    for k in k_values:
        thr = mean_val - k * std_val
        ax_hist.axvline(thr, linestyle='--', alpha=0.5, label=f'k={k} (thr={thr:.1f})')
    ax_hist.set_xlabel('Intensity')
    ax_hist.set_ylabel('Density')
    ax_hist.set_title(f'Intensity Distribution (σ={best_sigma})\nμ={mean_val:.1f}, std={std_val:.1f}')
    ax_hist.legend(fontsize=7, loc='upper left')

    # 3b: 경계 오버레이
    ax_ov = fig.add_subplot(gs[2, 2:4])
    ax_ov.imshow(image, cmap='gray')
    best_mask = best_result['crack_mask']
    gt_contours, _ = cv2.findContours(gt_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in gt_contours:
        pts = c.squeeze()
        if pts.ndim == 2:
            ax_ov.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=1.5, alpha=0.8)
    det_contours, _ = cv2.findContours(best_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in det_contours:
        pts = c.squeeze()
        if pts.ndim == 2:
            ax_ov.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=1.5, alpha=0.8)
    ax_ov.set_title(f'Best (k={best_k}, σ={best_sigma}): Green=GT, Red=Det')
    ax_ov.axis('off')

    # 3c: k별 IoU 바 차트 (모든 σ)
    ax_bar = fig.add_subplot(gs[2, 4])
    x = np.arange(len(k_values))
    width = 0.25
    for si, sigma in enumerate(sigmas):
        ious = [all_results[sigma][k]['metrics']['iou'] for k in k_values]
        ax_bar.bar(x + si * width, ious, width, label=f'σ={sigma}', alpha=0.8)
    ax_bar.set_xticks(x + width)
    ax_bar.set_xticklabels([f'k={k}' for k in k_values], fontsize=8)
    ax_bar.set_ylabel('IoU')
    ax_bar.set_title('IoU by k & σ')
    ax_bar.legend(fontsize=7)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.grid(True, alpha=0.3, axis='y')

    plt.savefig(str(OUTPUT_DIR / 'fig2_mean_k_sigma_sweep.png'), dpi=150, bbox_inches='tight')
    print(f"  → 저장: {OUTPUT_DIR / 'fig2_mean_k_sigma_sweep.png'}")

    return best_k, best_sigma, best_iou


# ===== Test 3: ROI 제한 — Otsu vs Mean+k·σ + 시각화 =====

def test_roi_restricted():
    print("\n" + "=" * 70)
    print("TEST 3: ROI 제한 — Otsu vs Mean+k·σ")
    print("=" * 70)

    image, gt_mask = load_test_data()
    H, W = image.shape

    crack_center_y = 250
    crack_x_min = 0
    crack_x_max = 260
    subset_size = 21
    spacing = 10

    gt_clean = gt_mask.copy()
    gt_clean[:crack_center_y - 30, :] = False
    gt_clean[crack_center_y + 30:, :] = False

    gt_crack_px = np.sum(gt_clean)
    print(f"정제된 GT 크랙 픽셀: {gt_crack_px} "
          f"({gt_crack_px / gt_clean.size * 100:.2f}%)")

    expand_values = [1, 2, 3, 5, 10, 15, 20]
    k_values = [1.5, 2.0, 2.5, 3.0, 3.5]

    # --- 3-1: Otsu ---
    otsu_results = {}
    best_otsu_iou = -1
    best_otsu_expand = 0

    print(f"\n{'='*70}")
    print("3-1: Otsu — ROI 높이별 성능")
    print(f"{'='*70}")
    print(f"{'expand':>7s} | {'ROI_H':>5s} | {'Crack%':>6s} | {'Thr':>4s} | "
          f"{'IoU':>6s} | {'Prec':>6s} | {'Recall':>6s} | {'F1':>6s} | "
          f"{'BndErr':>7s}")
    print("-" * 78)

    for n_exp in expand_values:
        pad = n_exp * spacing + subset_size // 2
        roi_y_min = max(0, crack_center_y - pad)
        roi_y_max = min(H, crack_center_y + pad + 1)
        roi_x_min = max(0, crack_x_min)
        roi_x_max = min(W, crack_x_max + subset_size // 2 + 1)

        roi_image = image[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        roi_gt = gt_clean[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        roi_h = roi_y_max - roi_y_min
        crack_ratio = np.sum(roi_gt) / max(roi_gt.size, 1) * 100

        result = detect_crack_otsu(
            roi_image, gaussian_sigma=1.0, morph_size=3, min_crack_area=5
        )
        metrics = evaluate_detection(result['crack_mask'], roi_gt)

        otsu_results[n_exp] = {
            'result': result, 'metrics': metrics,
            'roi_h': roi_h, 'crack_ratio': crack_ratio,
            'roi_bounds': (roi_y_min, roi_y_max, roi_x_min, roi_x_max),
            'roi_image': roi_image, 'roi_gt': roi_gt,
        }

        bnd = metrics['boundary_error_mean']
        bnd_str = f"{bnd:.2f}" if bnd != float('inf') else "  inf"

        print(f"{n_exp:>7d} | {roi_h:>5d} | {crack_ratio:6.2f} | "
              f"{result['threshold']:>4d} | {metrics['iou']:6.4f} | "
              f"{metrics['precision']:6.4f} | {metrics['recall']:6.4f} | "
              f"{metrics['f1']:6.4f} | {bnd_str:>7s}")

        if metrics['iou'] > best_otsu_iou:
            best_otsu_iou = metrics['iou']
            best_otsu_expand = n_exp

    print(f"\n→ Otsu 최적: expand={best_otsu_expand}, IoU={best_otsu_iou:.4f}")

    # --- 3-2: Mean+k·σ ---
    mk_results = {}
    best_mk_iou = -1
    best_mk_k = 0
    best_mk_expand = 0

    print(f"\n{'='*70}")
    print("3-2: Mean+k·σ — ROI 높이별 × k sweep")
    print(f"{'='*70}")

    for n_exp in expand_values:
        mk_results[n_exp] = {}
        pad = n_exp * spacing + subset_size // 2
        roi_y_min = max(0, crack_center_y - pad)
        roi_y_max = min(H, crack_center_y + pad + 1)
        roi_x_min = max(0, crack_x_min)
        roi_x_max = min(W, crack_x_max + subset_size // 2 + 1)

        roi_image = image[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        roi_gt = gt_clean[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
        roi_h = roi_y_max - roi_y_min
        crack_ratio = np.sum(roi_gt) / max(roi_gt.size, 1) * 100

        print(f"\n--- expand={n_exp}, ROI_H={roi_h}, crack={crack_ratio:.2f}% ---")
        print(f"{'k':>5s} | {'Thr':>6s} | {'IoU':>6s} | {'Prec':>6s} | "
              f"{'Recall':>6s} | {'F1':>6s} | {'BndErr':>7s}")
        print("-" * 55)

        for k in k_values:
            result = detect_crack_mean_k_sigma(
                roi_image, k=k, gaussian_sigma=1.0,
                morph_size=3, min_crack_area=5
            )
            metrics = evaluate_detection(result['crack_mask'], roi_gt)
            mk_results[n_exp][k] = {
                'result': result, 'metrics': metrics,
                'roi_image': roi_image, 'roi_gt': roi_gt,
            }

            bnd = metrics['boundary_error_mean']
            bnd_str = f"{bnd:.2f}" if bnd != float('inf') else "  inf"

            print(f"{k:5.1f} | {result['threshold']:6.1f} | "
                  f"{metrics['iou']:6.4f} | {metrics['precision']:6.4f} | "
                  f"{metrics['recall']:6.4f} | {metrics['f1']:6.4f} | "
                  f"{bnd_str:>7s}")

            if metrics['iou'] > best_mk_iou:
                best_mk_iou = metrics['iou']
                best_mk_k = k
                best_mk_expand = n_exp

    print(f"\n→ Mean+k·σ 최적: expand={best_mk_expand}, k={best_mk_k}, "
          f"IoU={best_mk_iou:.4f}")

    # --- 3-3: 최종 비교 ---
    print(f"\n{'='*70}")
    print("3-3: ROI 제한 최종 비교")
    print(f"{'='*70}")
    print(f"  Otsu:       expand={best_otsu_expand}, IoU={best_otsu_iou:.4f}")
    print(f"  Mean+k·σ:   expand={best_mk_expand}, k={best_mk_k}, IoU={best_mk_iou:.4f}")

    if best_otsu_iou > best_mk_iou:
        print(f"  → 승자: Otsu (차이: {best_otsu_iou - best_mk_iou:.4f})")
        winner = 'Otsu'
    else:
        print(f"  → 승자: Mean+k·σ (차이: {best_mk_iou - best_otsu_iou:.4f})")
        winner = 'Mean+k·σ'

    # --- 시각화: Figure 3 (ROI 비교) ---
    fig = plt.figure(figsize=(22, 16))
    fig.suptitle('Test 3: ROI-Restricted — Otsu vs Mean+k·σ', fontsize=16, fontweight='bold')

    gs = gridspec.GridSpec(4, len(expand_values), hspace=0.4, wspace=0.3)

    # Row 1: Otsu 검출 결과 (각 expand별)
    for i, n_exp in enumerate(expand_values):
        ax = fig.add_subplot(gs[0, i])
        entry = otsu_results[n_exp]
        mask = entry['result']['crack_mask']
        roi_gt = entry['roi_gt']
        vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
        vis[mask & roi_gt] = [0, 255, 0]
        vis[mask & ~roi_gt] = [255, 0, 0]
        vis[~mask & roi_gt] = [0, 0, 255]
        ax.imshow(vis, aspect='auto')
        iou_val = entry['metrics']['iou']
        ax.set_title(f'exp={n_exp}\nH={entry["roi_h"]}\nIoU={iou_val:.3f}', fontsize=8)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel('Otsu', fontsize=10, fontweight='bold')

    # Row 2: Mean+k·σ 검출 결과 (각 expand별, best_mk_k)
    for i, n_exp in enumerate(expand_values):
        ax = fig.add_subplot(gs[1, i])
        entry = mk_results[n_exp][best_mk_k]
        mask = entry['result']['crack_mask']
        roi_gt = entry['roi_gt']
        vis = np.zeros((*mask.shape, 3), dtype=np.uint8)
        vis[mask & roi_gt] = [0, 255, 0]
        vis[mask & ~roi_gt] = [255, 0, 0]
        vis[~mask & roi_gt] = [0, 0, 255]
        ax.imshow(vis, aspect='auto')
        iou_val = entry['metrics']['iou']
        ax.set_title(f'exp={n_exp}\nk={best_mk_k}\nIoU={iou_val:.3f}', fontsize=8)
        ax.axis('off')
        if i == 0:
            ax.set_ylabel(f'Mean+k·σ\n(k={best_mk_k})', fontsize=10, fontweight='bold')

    # Row 3: IoU 비교 그래프
    ax_cmp = fig.add_subplot(gs[2, 0:3])
    otsu_ious = [otsu_results[n]['metrics']['iou'] for n in expand_values]
    ax_cmp.plot(expand_values, otsu_ious, 'o-', color='tab:blue', linewidth=2,
                markersize=8, label='Otsu (σ=1.0)')
    for k in k_values:
        mk_ious = [mk_results[n][k]['metrics']['iou'] for n in expand_values]
        ax_cmp.plot(expand_values, mk_ious, 's--', label=f'Mean k={k}', alpha=0.7)
    ax_cmp.set_xlabel('ROI expand (n × spacing)')
    ax_cmp.set_ylabel('IoU')
    ax_cmp.set_title('IoU vs ROI Size')
    ax_cmp.legend(fontsize=8, ncol=2)
    ax_cmp.grid(True, alpha=0.3)
    ax_cmp.set_ylim(0, 1.05)

    # Row 3: 크랙 비율 vs IoU
    ax_ratio = fig.add_subplot(gs[2, 3:5])
    crack_ratios = [otsu_results[n]['crack_ratio'] for n in expand_values]
    ax_ratio.plot(crack_ratios, otsu_ious, 'o-', color='tab:blue', linewidth=2,
                  markersize=8, label='Otsu')
    mk_best_ious = [mk_results[n][best_mk_k]['metrics']['iou'] for n in expand_values]
    ax_ratio.plot(crack_ratios, mk_best_ious, 's-', color='tab:orange', linewidth=2,
                  markersize=8, label=f'Mean k={best_mk_k}')
    ax_ratio.set_xlabel('Crack Pixel Ratio in ROI (%)')
    ax_ratio.set_ylabel('IoU')
    ax_ratio.set_title('Crack Ratio vs IoU')
    ax_ratio.legend(fontsize=9)
    ax_ratio.grid(True, alpha=0.3)

    # Row 3: Otsu threshold vs ROI expand
    ax_thr = fig.add_subplot(gs[2, 5:7] if len(expand_values) >= 7 else gs[2, 5:])
    otsu_thrs = [otsu_results[n]['result']['threshold'] for n in expand_values]
    ax_thr.plot(expand_values, otsu_thrs, 'o-', color='tab:red', linewidth=2, label='Otsu Thr')
    mk_thrs = [mk_results[n][best_mk_k]['result']['threshold'] for n in expand_values]
    ax_thr.plot(expand_values, mk_thrs, 's-', color='tab:purple', linewidth=2, label=f'Mean k={best_mk_k} Thr')
    ax_thr.set_xlabel('ROI expand')
    ax_thr.set_ylabel('Threshold')
    ax_thr.set_title('Threshold vs ROI Size')
    ax_thr.legend(fontsize=9)
    ax_thr.grid(True, alpha=0.3)

    # Row 4: 최적 조건 상세 비교
    # 4a: Otsu best
    ax_ob = fig.add_subplot(gs[3, 0:2])
    ob_entry = otsu_results[best_otsu_expand]
    ax_ob.imshow(ob_entry['roi_image'], cmap='gray', aspect='auto')
    ob_mask = ob_entry['result']['crack_mask']
    ob_gt = ob_entry['roi_gt']
    gt_c, _ = cv2.findContours(ob_gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in gt_c:
        pts = c.squeeze()
        if pts.ndim == 2:
            ax_ob.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=1.5)
    det_c, _ = cv2.findContours(ob_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in det_c:
        pts = c.squeeze()
        if pts.ndim == 2:
            ax_ob.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=1.5)
    ax_ob.set_title(f'Otsu Best: exp={best_otsu_expand}, IoU={best_otsu_iou:.4f}\n(Green=GT, Red=Det)')
    ax_ob.axis('off')

    # 4b: Mean+k·σ best
    ax_mb = fig.add_subplot(gs[3, 2:4])
    mb_entry = mk_results[best_mk_expand][best_mk_k]
    ax_mb.imshow(mb_entry['roi_image'], cmap='gray', aspect='auto')
    mb_mask = mb_entry['result']['crack_mask']
    mb_gt = mb_entry['roi_gt']
    gt_c2, _ = cv2.findContours(mb_gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in gt_c2:
        pts = c.squeeze()
        if pts.ndim == 2:
            ax_mb.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=1.5)
    det_c2, _ = cv2.findContours(mb_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in det_c2:
        pts = c.squeeze()
        if pts.ndim == 2:
            ax_mb.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=1.5)
    ax_mb.set_title(f'Mean+k·σ Best: exp={best_mk_expand}, k={best_mk_k}\nIoU={best_mk_iou:.4f}')
    ax_mb.axis('off')

    # 4c: 최종 요약 텍스트
    ax_sum = fig.add_subplot(gs[3, 4:])
    ax_sum.axis('off')
    ob_m = otsu_results[best_otsu_expand]['metrics']
    mb_m = mk_results[best_mk_expand][best_mk_k]['metrics']
    ob_bnd = f"{ob_m['boundary_error_mean']:.2f}" if ob_m['boundary_error_mean'] != float('inf') else "inf"
    mb_bnd = f"{mb_m['boundary_error_mean']:.2f}" if mb_m['boundary_error_mean'] != float('inf') else "inf"

    summary = (
        f"{'='*35}\n"
        f"  FINAL COMPARISON\n"
        f"{'='*35}\n\n"
        f"  Otsu (exp={best_otsu_expand})\n"
        f"    IoU      = {ob_m['iou']:.4f}\n"
        f"    Precision= {ob_m['precision']:.4f}\n"
        f"    Recall   = {ob_m['recall']:.4f}\n"
        f"    F1       = {ob_m['f1']:.4f}\n"
        f"    BndErr   = {ob_bnd} px\n\n"
        f"  Mean+k·σ (exp={best_mk_expand}, k={best_mk_k})\n"
        f"    IoU      = {mb_m['iou']:.4f}\n"
        f"    Precision= {mb_m['precision']:.4f}\n"
        f"    Recall   = {mb_m['recall']:.4f}\n"
        f"    F1       = {mb_m['f1']:.4f}\n"
        f"    BndErr   = {mb_bnd} px\n\n"
        f"  Winner: {winner}\n"
        f"{'='*35}"
    )
    ax_sum.text(0.05, 0.95, summary, transform=ax_sum.transAxes,
                fontsize=10, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    plt.savefig(str(OUTPUT_DIR / 'fig3_roi_comparison.png'), dpi=150, bbox_inches='tight')
    print(f"  → 저장: {OUTPUT_DIR / 'fig3_roi_comparison.png'}")


# ===== 실행 =====

if __name__ == '__main__':
    print("크랙 경계 검출 성능 검증 + 시각화")
    print(f"데이터: {DATA_DIR.resolve()}")
    print(f"출력:   {OUTPUT_DIR.resolve()}\n")

    test_otsu_sigma_sweep()
    test_mean_k_sigma_sweep()
    test_roi_restricted()

    print("\n" + "=" * 70)
    print("전체 검증 완료")
    print(f"결과 저장 위치: {OUTPUT_DIR.resolve()}")
    print("=" * 70)

    plt.show()

