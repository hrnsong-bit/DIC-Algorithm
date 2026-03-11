# tests/test_crack_multiscale_canny.py
"""
멀티스케일 Canny 크랙 검출 테스트

핵심 아이디어: Gaussian blur를 강하게 주면 스펙클 에지는 사라지지만
크랙 에지는 남는다. σ=0(원본) Canny와 σ=high Canny의 교집합을 취하면
"블러에도 살아남는 에지 = 크랙"만 추출할 수 있다.

파이프라인:
1. Canny(σ=0) → 모든 에지 (크랙 + 스펙클)
2. Canny(σ=high) → 강한 에지만 (크랙 위주)
3. 교집합 또는 블러 Canny 근방의 원본 에지만 유지
4. 연결 성분 필터링 (길이, 방향 일관성)
5. 크랙 영역 채우기 (두 경계선 사이)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.ndimage import gaussian_filter

from speckle.core.crack_detection import evaluate_detection

# ===== 경로 설정 =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / 'synthetic_crack_data'
TEST_OUTPUT_ROOT = PROJECT_ROOT / 'tests' / '_outputs'
OUTPUT_DIR = TEST_OUTPUT_ROOT / 'output_multiscale_canny'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _imread_unicode(path):
    """한글 경로 대응 이미지 로드."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
    return img


def load_test_data():
    """합성 크랙 데이터 로드."""
    image = _imread_unicode(DATA_DIR / 'deformed.tiff')
    if image is None:
        raise FileNotFoundError(f"{DATA_DIR / 'deformed.tiff'} 를 찾을 수 없습니다.")
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gt_mask = np.load(str(DATA_DIR / 'crack_mask.npy')).astype(bool)
    print(f"이미지: {image.shape}, GT 크랙 픽셀: {np.sum(gt_mask)}")
    return image, gt_mask


# ===== 멀티스케일 Canny 함수들 =====

def canny_at_sigma(image, sigma, low_threshold, high_threshold):
    """Gaussian blur 후 Canny 적용."""
    if sigma > 0:
        blurred = gaussian_filter(image.astype(np.float64), sigma=sigma)
        blurred = np.clip(blurred, 0, 255).astype(np.uint8)
    else:
        blurred = image.copy()
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    return edges, blurred


def multiscale_canny_intersection(image, sigma_low=0, sigma_high=2.0,
                                   low_thr=50, high_thr=150,
                                   dilate_radius=3):
    """
    멀티스케일 Canny 교집합.
    
    1. Canny(σ_low) → edges_fine (크랙 + 스펙클)
    2. Canny(σ_high) → edges_coarse (크랙 위주)
    3. edges_coarse를 dilate → 크랙 근방 마스크
    4. edges_fine AND 크랙 근방 마스크 → 크랙 에지만 유지
    """
    edges_fine, _ = canny_at_sigma(image, sigma_low, low_thr, high_thr)
    edges_coarse, blurred_high = canny_at_sigma(image, sigma_high, low_thr, high_thr)

    # coarse 에지를 팽창시켜 근방 마스크 생성
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (2 * dilate_radius + 1, 2 * dilate_radius + 1))
    coarse_dilated = cv2.dilate(edges_coarse, kernel, iterations=1)

    # 교집합: fine 에지 중 coarse 근방에 있는 것만
    edges_filtered = edges_fine & coarse_dilated

    return {
        'edges_fine': edges_fine,
        'edges_coarse': edges_coarse,
        'coarse_dilated': coarse_dilated,
        'edges_filtered': edges_filtered,
        'blurred_high': blurred_high,
    }


def fill_between_edges(edge_mask, method='columnwise'):
    """
    두 크랙 경계선 사이를 채워 크랙 영역 마스크 생성.
    
    각 열(column)에서 에지 픽셀의 최소~최대 y 범위를 채움.
    """
    filled = np.zeros_like(edge_mask, dtype=bool)
    H, W = edge_mask.shape

    for x in range(W):
        col = edge_mask[:, x]
        ys = np.where(col > 0)[0]
        if len(ys) >= 2:
            filled[ys[0]:ys[-1] + 1, x] = True
        elif len(ys) == 1:
            filled[ys[0], x] = True

    return filled


def connected_component_filter(edge_mask, min_length=10):
    """연결 성분 중 min_length 이상만 유지."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        edge_mask.astype(np.uint8), connectivity=8
    )
    filtered = np.zeros_like(edge_mask, dtype=np.uint8)
    component_sizes = []

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        component_sizes.append(area)
        if area >= min_length:
            filtered[labels == i] = 255

    return filtered, component_sizes


def detect_crack_mean_k_sigma_simple(image, k=1.5, gaussian_sigma=0):
    """Mean+kσ 크랙 검출 (간단 버전)."""
    if gaussian_sigma > 0:
        blurred = gaussian_filter(image.astype(np.float64), sigma=gaussian_sigma)
    else:
        blurred = image.astype(np.float64)
    mean_val = blurred.mean()
    std_val = blurred.std()
    threshold = mean_val - k * std_val
    mask = blurred < threshold
    # morphological cleaning
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_u8 = mask.astype(np.uint8) * 255
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    return (mask_u8 > 0), threshold


# ===== 테스트 실행 =====

def run_tests():
    image, gt_mask = load_test_data()

    # ─── 파라미터 설정 ───
    sigma_low = 0
    sigma_highs = [1.0, 1.5, 2.0, 2.5, 3.0]
    dilate_radii = [1, 2, 3, 5, 7]
    canny_low = 50
    canny_high = 150
    min_lengths = [5, 10, 15, 20, 30]

    # ═══════════════════════════════════════════════════
    # TEST 1: σ_high sweep (dilate=3, min_length=10 고정)
    # ═══════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TEST 1: σ_high sweep — 블러 강도별 멀티스케일 Canny")
    print("=" * 70)

    test1_results = {}

    print(f"\n{'σ_high':>6s} | {'Edge_F':>6s} | {'Edge_C':>6s} | {'Filt':>5s} | "
          f"{'Fill_IoU':>8s} | {'Prec':>6s} | {'Recall':>6s} | {'F1':>6s}")
    print("-" * 70)

    for sigma_h in sigma_highs:
        ms = multiscale_canny_intersection(image, sigma_low, sigma_h,
                                            canny_low, canny_high, dilate_radius=3)
        # 연결 성분 필터링
        filtered, comp_sizes = connected_component_filter(ms['edges_filtered'], min_length=10)
        # 채우기
        filled = fill_between_edges(filtered)
        # 평가
        metrics = evaluate_detection(filled, gt_mask)

        test1_results[sigma_h] = {
            'ms': ms, 'filtered': filtered, 'filled': filled,
            'metrics': metrics, 'comp_sizes': comp_sizes,
        }

        n_fine = int(np.sum(ms['edges_fine'] > 0))
        n_coarse = int(np.sum(ms['edges_coarse'] > 0))
        n_filt = int(np.sum(filtered > 0))

        print(f"{sigma_h:6.1f} | {n_fine:6d} | {n_coarse:6d} | {n_filt:5d} | "
              f"{metrics['iou']:8.4f} | {metrics['precision']:6.4f} | "
              f"{metrics['recall']:6.4f} | {metrics['f1']:6.4f}")

    # ═══════════════════════════════════════════════════
    # TEST 2: dilate_radius sweep (best σ_high, min_length=10 고정)
    # ═══════════════════════════════════════════════════
    best_sigma_h = max(test1_results, key=lambda s: test1_results[s]['metrics']['iou'])

    print(f"\n{'='*70}")
    print(f"TEST 2: dilate_radius sweep (σ_high={best_sigma_h})")
    print(f"{'='*70}")

    test2_results = {}

    print(f"\n{'dilate':>6s} | {'Filt':>5s} | {'Fill_IoU':>8s} | {'Prec':>6s} | "
          f"{'Recall':>6s} | {'F1':>6s} | {'BndErr':>7s}")
    print("-" * 60)

    for dr in dilate_radii:
        ms = multiscale_canny_intersection(image, sigma_low, best_sigma_h,
                                            canny_low, canny_high, dilate_radius=dr)
        filtered, _ = connected_component_filter(ms['edges_filtered'], min_length=10)
        filled = fill_between_edges(filtered)
        metrics = evaluate_detection(filled, gt_mask)

        test2_results[dr] = {
            'ms': ms, 'filtered': filtered, 'filled': filled, 'metrics': metrics,
        }

        n_filt = int(np.sum(filtered > 0))
        bnd = metrics['boundary_error_mean']
        bnd_str = f"{bnd:.2f}" if bnd != float('inf') else "  inf"

        print(f"{dr:6d} | {n_filt:5d} | {metrics['iou']:8.4f} | "
              f"{metrics['precision']:6.4f} | {metrics['recall']:6.4f} | "
              f"{metrics['f1']:6.4f} | {bnd_str:>7s}")

    best_dr = max(test2_results, key=lambda d: test2_results[d]['metrics']['iou'])

    # ═══════════════════════════════════════════════════
    # TEST 3: min_length sweep (best σ_high, best dilate)
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"TEST 3: min_length sweep (σ_high={best_sigma_h}, dilate={best_dr})")
    print(f"{'='*70}")

    test3_results = {}

    print(f"\n{'min_len':>7s} | {'Comps':>5s} | {'Filt':>5s} | {'Fill_IoU':>8s} | "
          f"{'Prec':>6s} | {'Recall':>6s} | {'F1':>6s}")
    print("-" * 60)

    ms_best = multiscale_canny_intersection(image, sigma_low, best_sigma_h,
                                             canny_low, canny_high, dilate_radius=best_dr)

    for ml in min_lengths:
        filtered, comp_sizes = connected_component_filter(ms_best['edges_filtered'], min_length=ml)
        filled = fill_between_edges(filtered)
        metrics = evaluate_detection(filled, gt_mask)

        n_comps = len([s for s in comp_sizes if s >= ml])
        n_filt = int(np.sum(filtered > 0))

        test3_results[ml] = {
            'filtered': filtered, 'filled': filled,
            'metrics': metrics, 'n_comps': n_comps,
        }

        print(f"{ml:7d} | {n_comps:5d} | {n_filt:5d} | {metrics['iou']:8.4f} | "
              f"{metrics['precision']:6.4f} | {metrics['recall']:6.4f} | "
              f"{metrics['f1']:6.4f}")

    best_ml = max(test3_results, key=lambda m: test3_results[m]['metrics']['iou'])

    # ═══════════════════════════════════════════════════
    # TEST 4: Canny threshold sweep
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"TEST 4: Canny threshold sweep (σ_high={best_sigma_h}, dilate={best_dr}, min_len={best_ml})")
    print(f"{'='*70}")

    canny_thresholds = [(30, 90), (50, 150), (70, 200), (30, 150), (50, 200)]
    test4_results = {}

    print(f"\n{'Low':>4s} {'High':>4s} | {'Filt':>5s} | {'Fill_IoU':>8s} | "
          f"{'Prec':>6s} | {'Recall':>6s} | {'F1':>6s}")
    print("-" * 55)

    for (cl, ch) in canny_thresholds:
        ms = multiscale_canny_intersection(image, sigma_low, best_sigma_h,
                                            cl, ch, dilate_radius=best_dr)
        filtered, _ = connected_component_filter(ms['edges_filtered'], min_length=best_ml)
        filled = fill_between_edges(filtered)
        metrics = evaluate_detection(filled, gt_mask)

        test4_results[(cl, ch)] = {
            'ms': ms, 'filtered': filtered, 'filled': filled, 'metrics': metrics,
        }

        n_filt = int(np.sum(filtered > 0))
        print(f"{cl:4d} {ch:4d} | {n_filt:5d} | {metrics['iou']:8.4f} | "
              f"{metrics['precision']:6.4f} | {metrics['recall']:6.4f} | "
              f"{metrics['f1']:6.4f}")

    best_ct = max(test4_results, key=lambda t: test4_results[t]['metrics']['iou'])

    # ═══════════════════════════════════════════════════
    # Mean+kσ 비교 기준
    # ═══════════════════════════════════════════════════
    mean_mask, mean_thr = detect_crack_mean_k_sigma_simple(image, k=1.5, gaussian_sigma=0)
    mean_metrics = evaluate_detection(mean_mask, gt_mask)
    print(f"\n--- 비교: Mean+kσ (k=1.5, σ=0) ---")
    print(f"  IoU={mean_metrics['iou']:.4f}, Prec={mean_metrics['precision']:.4f}, "
          f"Recall={mean_metrics['recall']:.4f}, F1={mean_metrics['f1']:.4f}")

    # ═══════════════════════════════════════════════════
    # 최적 멀티스케일 Canny 결과
    # ═══════════════════════════════════════════════════
    ms_final = multiscale_canny_intersection(image, sigma_low, best_sigma_h,
                                              best_ct[0], best_ct[1],
                                              dilate_radius=best_dr)
    filt_final, _ = connected_component_filter(ms_final['edges_filtered'], min_length=best_ml)
    filled_final = fill_between_edges(filt_final)
    final_metrics = evaluate_detection(filled_final, gt_mask)

    print(f"\n{'='*70}")
    print(f"최적 멀티스케일 Canny")
    print(f"  σ_high={best_sigma_h}, dilate={best_dr}, min_len={best_ml}, "
          f"Canny=({best_ct[0]},{best_ct[1]})")
    print(f"  IoU={final_metrics['iou']:.4f}, Prec={final_metrics['precision']:.4f}, "
          f"Recall={final_metrics['recall']:.4f}, F1={final_metrics['f1']:.4f}")
    print(f"{'='*70}")

    return {
        'image': image, 'gt_mask': gt_mask,
        'test1': test1_results, 'test2': test2_results,
        'test3': test3_results, 'test4': test4_results,
        'best_params': {
            'sigma_high': best_sigma_h, 'dilate_radius': best_dr,
            'min_length': best_ml, 'canny_thresholds': best_ct,
        },
        'final': {
            'ms': ms_final, 'filtered': filt_final,
            'filled': filled_final, 'metrics': final_metrics,
        },
        'mean_baseline': {
            'mask': mean_mask, 'metrics': mean_metrics,
        },
    }


# ===== 시각화 =====

def visualize_results(data):
    image = data['image']
    gt_mask = data['gt_mask']
    test1 = data['test1']
    test2 = data['test2']
    test3 = data['test3']
    test4 = data['test4']
    best = data['best_params']
    final = data['final']
    mean_bl = data['mean_baseline']

    # ═══════════════════════════════════════════════════
    # Figure 1: 멀티스케일 Canny 파이프라인 단계별 시각화
    # ═══════════════════════════════════════════════════
    fig1 = plt.figure(figsize=(24, 16))
    fig1.suptitle('Figure 1: Multiscale Canny Pipeline Stages', fontsize=16, fontweight='bold')

    # 최적 σ_high 결과 사용
    best_sh = best['sigma_high']
    ms = test1[best_sh]['ms']

    gs1 = gridspec.GridSpec(3, 4, hspace=0.35, wspace=0.3)

    # Row 1: 파이프라인 단계
    # 1a: 원본 이미지
    ax = fig1.add_subplot(gs1[0, 0])
    ax.imshow(image, cmap='gray')
    ax.set_title('(a) Original Image')
    ax.axis('off')

    # 1b: Canny(σ=0) — 모든 에지
    ax = fig1.add_subplot(gs1[0, 1])
    ax.imshow(ms['edges_fine'], cmap='gray')
    n_fine = int(np.sum(ms['edges_fine'] > 0))
    ax.set_title(f'(b) Canny(σ=0)\n{n_fine} edge px')
    ax.axis('off')

    # 1c: Canny(σ=high) — 크랙 위주
    ax = fig1.add_subplot(gs1[0, 2])
    ax.imshow(ms['edges_coarse'], cmap='gray')
    n_coarse = int(np.sum(ms['edges_coarse'] > 0))
    ax.set_title(f'(c) Canny(σ={best_sh})\n{n_coarse} edge px')
    ax.axis('off')

    # 1d: Coarse dilated (근방 마스크)
    ax = fig1.add_subplot(gs1[0, 3])
    ax.imshow(ms['coarse_dilated'], cmap='gray')
    ax.set_title(f'(d) Coarse dilated\n(radius={best["dilate_radius"]})')
    ax.axis('off')

    # Row 2: 필터링 결과
    # 2a: 교집합 (fine ∩ dilated coarse)
    ax = fig1.add_subplot(gs1[1, 0])
    ax.imshow(ms['edges_filtered'], cmap='gray')
    n_inter = int(np.sum(ms['edges_filtered'] > 0))
    ax.set_title(f'(e) Intersection\n{n_inter} edge px')
    ax.axis('off')

    # 2b: 연결 성분 필터링 후
    ax = fig1.add_subplot(gs1[1, 1])
    filt = test1[best_sh]['filtered']
    ax.imshow(filt, cmap='gray')
    n_filt = int(np.sum(filt > 0))
    ax.set_title(f'(f) Length filtered\n(≥{best["min_length"]}px) {n_filt} px')
    ax.axis('off')

    # 2c: 채우기 결과
    ax = fig1.add_subplot(gs1[1, 2])
    filled = test1[best_sh]['filled']
    vis_fill = np.zeros((*filled.shape, 3), dtype=np.uint8)
    vis_fill[filled & gt_mask] = [0, 255, 0]
    vis_fill[filled & ~gt_mask] = [255, 0, 0]
    vis_fill[~filled & gt_mask] = [0, 0, 255]
    ax.imshow(vis_fill)
    m = test1[best_sh]['metrics']
    ax.set_title(f'(g) Filled — IoU={m["iou"]:.3f}\n(G=TP R=FP B=FN)')
    ax.axis('off')

    # 2d: Mean+kσ 비교
    ax = fig1.add_subplot(gs1[1, 3])
    vis_mean = np.zeros((*mean_bl['mask'].shape, 3), dtype=np.uint8)
    vis_mean[mean_bl['mask'] & gt_mask] = [0, 255, 0]
    vis_mean[mean_bl['mask'] & ~gt_mask] = [255, 0, 0]
    vis_mean[~mean_bl['mask'] & gt_mask] = [0, 0, 255]
    ax.imshow(vis_mean)
    mm = mean_bl['metrics']
    ax.set_title(f'(h) Mean+kσ — IoU={mm["iou"]:.3f}\n(baseline)')
    ax.axis('off')

    # Row 3: σ별 비교
    sigma_highs = sorted(test1.keys())
    n_sigma = len(sigma_highs)
    for i, sh in enumerate(sigma_highs):
        if i < 4:
            ax = fig1.add_subplot(gs1[2, i])
            entry = test1[sh]
            vis = np.zeros((*entry['filled'].shape, 3), dtype=np.uint8)
            vis[entry['filled'] & gt_mask] = [0, 255, 0]
            vis[entry['filled'] & ~gt_mask] = [255, 0, 0]
            vis[~entry['filled'] & gt_mask] = [0, 0, 255]
            ax.imshow(vis)
            m = entry['metrics']
            marker = ' ★' if sh == best_sh else ''
            ax.set_title(f'σ_high={sh}{marker}\nIoU={m["iou"]:.3f}', fontsize=9)
            ax.axis('off')

    fig1.savefig(str(OUTPUT_DIR / 'fig1_pipeline_stages.png'), dpi=150, bbox_inches='tight')
    print(f"  → 저장: {OUTPUT_DIR / 'fig1_pipeline_stages.png'}")

    # ═══════════════════════════════════════════════════
    # Figure 2: 파라미터 민감도 분석
    # ═══════════════════════════════════════════════════
    fig2 = plt.figure(figsize=(22, 12))
    fig2.suptitle('Figure 2: Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
    gs2 = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)

    # 2a: σ_high vs IoU
    ax = fig2.add_subplot(gs2[0, 0])
    shs = sorted(test1.keys())
    ious = [test1[s]['metrics']['iou'] for s in shs]
    precs = [test1[s]['metrics']['precision'] for s in shs]
    recs = [test1[s]['metrics']['recall'] for s in shs]
    f1s = [test1[s]['metrics']['f1'] for s in shs]
    ax.plot(shs, ious, 'ko-', linewidth=2, markersize=8, label='IoU')
    ax.plot(shs, precs, 's--', color='tab:orange', label='Precision')
    ax.plot(shs, recs, '^--', color='tab:blue', label='Recall')
    ax.plot(shs, f1s, 'd--', color='tab:green', label='F1')
    ax.axhline(mean_bl['metrics']['iou'], color='red', linestyle=':', alpha=0.7,
               label=f'Mean+kσ IoU={mean_bl["metrics"]["iou"]:.3f}')
    ax.set_xlabel('σ_high')
    ax.set_ylabel('Score')
    ax.set_title('(a) σ_high Sweep')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 2b: dilate_radius vs IoU
    ax = fig2.add_subplot(gs2[0, 1])
    drs = sorted(test2.keys())
    ious2 = [test2[d]['metrics']['iou'] for d in drs]
    precs2 = [test2[d]['metrics']['precision'] for d in drs]
    recs2 = [test2[d]['metrics']['recall'] for d in drs]
    ax.plot(drs, ious2, 'ko-', linewidth=2, markersize=8, label='IoU')
    ax.plot(drs, precs2, 's--', color='tab:orange', label='Precision')
    ax.plot(drs, recs2, '^--', color='tab:blue', label='Recall')
    ax.axhline(mean_bl['metrics']['iou'], color='red', linestyle=':', alpha=0.7,
               label=f'Mean+kσ IoU')
    ax.set_xlabel('Dilate Radius')
    ax.set_ylabel('Score')
    ax.set_title(f'(b) Dilate Radius Sweep (σ_h={best_sh})')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 2c: min_length vs IoU
    ax = fig2.add_subplot(gs2[0, 2])
    mls = sorted(test3.keys())
    ious3 = [test3[m]['metrics']['iou'] for m in mls]
    n_comps = [test3[m]['n_comps'] for m in mls]
    ax.plot(mls, ious3, 'ko-', linewidth=2, markersize=8, label='IoU')
    ax.axhline(mean_bl['metrics']['iou'], color='red', linestyle=':', alpha=0.7,
               label=f'Mean+kσ IoU')
    ax.set_xlabel('Min Length (px)')
    ax.set_ylabel('IoU')
    ax.set_title(f'(c) Min Length Sweep (σ_h={best_sh}, dr={best["dilate_radius"]})')
    ax.legend(fontsize=8, loc='lower left')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    ax_b = ax.twinx()
    ax_b.bar(mls, n_comps, alpha=0.3, color='tab:cyan', width=2, label='# Components')
    ax_b.set_ylabel('# Components')
    ax_b.legend(fontsize=8, loc='upper right')

    # 2d: Canny threshold 비교
    ax = fig2.add_subplot(gs2[1, 0])
    ct_labels = [f'({cl},{ch})' for cl, ch in test4.keys()]
    ct_ious = [test4[k]['metrics']['iou'] for k in test4.keys()]
    ct_precs = [test4[k]['metrics']['precision'] for k in test4.keys()]
    ct_recs = [test4[k]['metrics']['recall'] for k in test4.keys()]
    x_pos = range(len(ct_labels))
    width = 0.25
    ax.bar([p - width for p in x_pos], ct_ious, width, label='IoU', color='tab:blue')
    ax.bar(list(x_pos), ct_precs, width, label='Precision', color='tab:orange')
    ax.bar([p + width for p in x_pos], ct_recs, width, label='Recall', color='tab:green')
    ax.axhline(mean_bl['metrics']['iou'], color='red', linestyle=':', alpha=0.7)
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(ct_labels, fontsize=8, rotation=30)
    ax.set_ylabel('Score')
    ax.set_title('(d) Canny Threshold Sweep')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')

    # 2e: 에지 수 변화 (파이프라인 단계별)
    ax = fig2.add_subplot(gs2[1, 1])
    best_entry = test1[best_sh]
    stages = ['Fine\n(σ=0)', f'Coarse\n(σ={best_sh})', 'Intersection', f'Length\n(≥{best["min_length"]})', 'Filled']
    counts = [
        int(np.sum(best_entry['ms']['edges_fine'] > 0)),
        int(np.sum(best_entry['ms']['edges_coarse'] > 0)),
        int(np.sum(best_entry['ms']['edges_filtered'] > 0)),
        int(np.sum(best_entry['filtered'] > 0)),
        int(np.sum(best_entry['filled'])),
    ]
    gt_count = int(np.sum(gt_mask))
    colors = ['tab:gray', 'tab:cyan', 'tab:blue', 'tab:purple', 'tab:green']
    bars = ax.bar(stages, counts, color=colors, alpha=0.8)
    ax.axhline(gt_count, color='red', linewidth=2, linestyle='--', label=f'GT={gt_count}')
    ax.set_ylabel('Pixel Count')
    ax.set_title('(e) Pipeline Pixel Count')
    ax.legend(fontsize=9)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                str(count), ha='center', fontsize=8)

    # 2f: 최종 비교 요약
    ax = fig2.add_subplot(gs2[1, 2])
    ax.axis('off')
    fm = final['metrics']
    mm = mean_bl['metrics']
    fm_bnd = f"{fm['boundary_error_mean']:.2f}" if fm['boundary_error_mean'] != float('inf') else "inf"
    mm_bnd = f"{mm['boundary_error_mean']:.2f}" if mm['boundary_error_mean'] != float('inf') else "inf"

    summary = (
        f"{'='*40}\n"
        f"  FINAL COMPARISON\n"
        f"{'='*40}\n\n"
        f"  Multiscale Canny (Optimal)\n"
        f"    σ_high  = {best['sigma_high']}\n"
        f"    dilate  = {best['dilate_radius']}\n"
        f"    min_len = {best['min_length']}\n"
        f"    Canny   = {best['canny_thresholds']}\n"
        f"    ─────────────────────────\n"
        f"    IoU      = {fm['iou']:.4f}\n"
        f"    Precision= {fm['precision']:.4f}\n"
        f"    Recall   = {fm['recall']:.4f}\n"
        f"    F1       = {fm['f1']:.4f}\n"
        f"    BndErr   = {fm_bnd} px\n\n"
        f"  Mean+kσ (k=1.5, σ=0)\n"
        f"    ─────────────────────────\n"
        f"    IoU      = {mm['iou']:.4f}\n"
        f"    Precision= {mm['precision']:.4f}\n"
        f"    Recall   = {mm['recall']:.4f}\n"
        f"    F1       = {mm['f1']:.4f}\n"
        f"    BndErr   = {mm_bnd} px\n\n"
    )

    if fm['iou'] > mm['iou']:
        summary += f"  ★ Winner: Multiscale Canny\n"
        summary += f"    (Δ IoU = +{fm['iou'] - mm['iou']:.4f})"
    else:
        summary += f"  ★ Winner: Mean+kσ\n"
        summary += f"    (Δ IoU = +{mm['iou'] - fm['iou']:.4f})"

    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

    fig2.savefig(str(OUTPUT_DIR / 'fig2_parameter_sensitivity.png'), dpi=150, bbox_inches='tight')
    print(f"  → 저장: {OUTPUT_DIR / 'fig2_parameter_sensitivity.png'}")

    # ═══════════════════════════════════════════════════
    # Figure 3: 크랙 영역 상세 비교 (줌)
    # ═══════════════════════════════════════════════════
    fig3 = plt.figure(figsize=(22, 10))
    fig3.suptitle('Figure 3: Crack Region Detail Comparison', fontsize=16, fontweight='bold')
    gs3 = gridspec.GridSpec(2, 4, hspace=0.3, wspace=0.3)

    # 크랙 영역 줌 (y=230~270, x=0~300)
    zy1, zy2, zx1, zx2 = 230, 270, 0, 300

    # 3a: 원본 줌
    ax = fig3.add_subplot(gs3[0, 0])
    ax.imshow(image[zy1:zy2, zx1:zx2], cmap='gray', aspect='auto')
    ax.set_title('(a) Original (zoom)')
    ax.axis('off')

    # 3b: Fine edges 줌
    ax = fig3.add_subplot(gs3[0, 1])
    ax.imshow(ms['edges_fine'][zy1:zy2, zx1:zx2], cmap='gray', aspect='auto')
    ax.set_title('(b) Fine edges (σ=0)')
    ax.axis('off')

    # 3c: Coarse edges 줌
    ax = fig3.add_subplot(gs3[0, 2])
    ax.imshow(ms['edges_coarse'][zy1:zy2, zx1:zx2], cmap='gray', aspect='auto')
    ax.set_title(f'(c) Coarse edges (σ={best_sh})')
    ax.axis('off')

    # 3d: Filtered edges 줌
    ax = fig3.add_subplot(gs3[0, 3])
    ax.imshow(final['filtered'][zy1:zy2, zx1:zx2], cmap='gray', aspect='auto')
    ax.set_title('(d) Filtered edges')
    ax.axis('off')

    # 3e: Filled (멀티스케일 Canny) — TP/FP/FN
    ax = fig3.add_subplot(gs3[1, 0])
    vis_ms = np.zeros((zy2-zy1, zx2-zx1, 3), dtype=np.uint8)
    f_roi = final['filled'][zy1:zy2, zx1:zx2]
    g_roi = gt_mask[zy1:zy2, zx1:zx2]
    vis_ms[f_roi & g_roi] = [0, 255, 0]
    vis_ms[f_roi & ~g_roi] = [255, 0, 0]
    vis_ms[~f_roi & g_roi] = [0, 0, 255]
    ax.imshow(vis_ms, aspect='auto')
    ax.set_title(f'(e) MS-Canny IoU={final["metrics"]["iou"]:.3f}')
    ax.axis('off')

    # 3f: Mean+kσ — TP/FP/FN
    ax = fig3.add_subplot(gs3[1, 1])
    vis_mk = np.zeros((zy2-zy1, zx2-zx1, 3), dtype=np.uint8)
    m_roi = mean_bl['mask'][zy1:zy2, zx1:zx2]
    vis_mk[m_roi & g_roi] = [0, 255, 0]
    vis_mk[m_roi & ~g_roi] = [255, 0, 0]
    vis_mk[~m_roi & g_roi] = [0, 0, 255]
    ax.imshow(vis_mk, aspect='auto')
    ax.set_title(f'(f) Mean+kσ IoU={mean_bl["metrics"]["iou"]:.3f}')
    ax.axis('off')

    # 3g: GT
    ax = fig3.add_subplot(gs3[1, 2])
    ax.imshow(g_roi, cmap='Greens', aspect='auto')
    ax.set_title('(g) Ground Truth')
    ax.axis('off')

    # 3h: 오버레이 (원본 + 경계)
    ax = fig3.add_subplot(gs3[1, 3])
    overlay = cv2.cvtColor(image[zy1:zy2, zx1:zx2], cv2.COLOR_GRAY2RGB)
    # GT 경계: 초록
    gt_c, _ = cv2.findContours(g_roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, gt_c, -1, (0, 255, 0), 1)
    # MS-Canny 경계: 빨강
    ms_c, _ = cv2.findContours(f_roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, ms_c, -1, (255, 0, 0), 1)
    # Mean+kσ 경계: 파랑
    mk_c, _ = cv2.findContours(m_roi.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, mk_c, -1, (0, 100, 255), 1)
    ax.imshow(overlay, aspect='auto')
    ax.set_title('(h) Overlay\n(G=GT, R=MS-Canny, B=Mean)')
    ax.axis('off')

    fig3.savefig(str(OUTPUT_DIR / 'fig3_crack_detail.png'), dpi=150, bbox_inches='tight')
    print(f"  → 저장: {OUTPUT_DIR / 'fig3_crack_detail.png'}")

    # ═══════════════════════════════════════════════════
    # Figure 4: 크랙 팁 영역 상세
    # ═══════════════════════════════════════════════════
    fig4 = plt.figure(figsize=(20, 8))
    fig4.suptitle('Figure 4: Crack Tip Region Detail', fontsize=16, fontweight='bold')
    gs4 = gridspec.GridSpec(1, 5, wspace=0.3)

    # 팁 영역 (y=235~265, x=200~280)
    ty1, ty2, tx1, tx2 = 235, 265, 200, 280

    titles = ['(a) Original', '(b) Fine edges', f'(c) Coarse (σ={best_sh})',
              '(d) MS-Canny filled', '(e) Mean+kσ']
    images_tip = [
        image[ty1:ty2, tx1:tx2],
        ms['edges_fine'][ty1:ty2, tx1:tx2],
        ms['edges_coarse'][ty1:ty2, tx1:tx2],
        None,  # TP/FP/FN
        None,  # TP/FP/FN
    ]

    for i in range(5):
        ax = fig4.add_subplot(gs4[0, i])
        if i < 3:
            ax.imshow(images_tip[i], cmap='gray', aspect='auto')
        elif i == 3:
            vis_t = np.zeros((ty2-ty1, tx2-tx1, 3), dtype=np.uint8)
            ft = final['filled'][ty1:ty2, tx1:tx2]
            gt = gt_mask[ty1:ty2, tx1:tx2]
            vis_t[ft & gt] = [0, 255, 0]
            vis_t[ft & ~gt] = [255, 0, 0]
            vis_t[~ft & gt] = [0, 0, 255]
            ax.imshow(vis_t, aspect='auto')
        elif i == 4:
            vis_t = np.zeros((ty2-ty1, tx2-tx1, 3), dtype=np.uint8)
            mt = mean_bl['mask'][ty1:ty2, tx1:tx2]
            gt = gt_mask[ty1:ty2, tx1:tx2]
            vis_t[mt & gt] = [0, 255, 0]
            vis_t[mt & ~gt] = [255, 0, 0]
            vis_t[~mt & gt] = [0, 0, 255]
            ax.imshow(vis_t, aspect='auto')
        ax.set_title(titles[i], fontsize=10)
        ax.axis('off')

    fig4.savefig(str(OUTPUT_DIR / 'fig4_tip_detail.png'), dpi=150, bbox_inches='tight')
    print(f"  → 저장: {OUTPUT_DIR / 'fig4_tip_detail.png'}")


# ===== 실행 =====

if __name__ == '__main__':
    print("멀티스케일 Canny 크랙 검출 테스트")
    print(f"데이터: {DATA_DIR.resolve()}")
    print(f"출력:   {OUTPUT_DIR.resolve()}\n")

    data = run_tests()
    visualize_results(data)

    print("\n" + "=" * 70)
    print("전체 테스트 완료")
    print(f"결과 저장 위치: {OUTPUT_DIR.resolve()}")
    print("=" * 70)

    plt.show()

