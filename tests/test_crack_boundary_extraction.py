# tests/test_crack_boundary_extraction.py
"""
크랙 경계 추출 테스트

파이프라인:
1. ADSS 불량 POI + 8방위 이웃 → ROI 구성
2. ROI에서 Mean+kσ → contour로 크랙 경계선 좌표 추출
3. 각 복원 POI에서 크랙 경계까지 거리/방향 계산
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.spatial import distance
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
OUTPUT_DIR = TEST_OUTPUT_ROOT / 'output_boundary_extraction'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _imread_unicode(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지 로딩 실패: {path}")
    return img


def compute_metrics(gt, pred):
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


# ══════════════════════════════════════════════════
# Step 1: ROI 구성
# ══════════════════════════════════════════════════

def build_roi(def_img, icgn_result, spacing, subset_size):
    """불량 POI + 8방위 이웃 → ROI."""
    H, W = def_img.shape
    M = subset_size // 2
    bad_indices = np.where(~icgn_result.valid_mask)[0]
    poi_x = icgn_result.points_x
    poi_y = icgn_result.points_y

    bad_coords = set()
    bad_only = set()
    for idx in bad_indices:
        px, py = int(poi_x[idx]), int(poi_y[idx])
        bad_only.add((px, py))
        for dx in [-spacing, 0, spacing]:
            for dy in [-spacing, 0, spacing]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < W and 0 <= ny < H:
                    bad_coords.add((nx, ny))

    all_x = [c[0] for c in bad_coords]
    all_y = [c[1] for c in bad_coords]
    roi_x1 = max(0, min(all_x) - M - 5)
    roi_x2 = min(W, max(all_x) + M + 6)
    roi_y1 = max(0, min(all_y) - M - 5)
    roi_y2 = min(H, max(all_y) + M + 6)

    roi = def_img[roi_y1:roi_y2, roi_x1:roi_x2]

    return {
        'roi': roi,
        'bounds': (roi_x1, roi_y1, roi_x2, roi_y2),
        'bad_indices': bad_indices,
        'bad_coords': bad_only,
        'all_coords': bad_coords,
    }


# ══════════════════════════════════════════════════
# Step 2: Mean+kσ → contour 경계선 추출
# ══════════════════════════════════════════════════

def extract_crack_boundary(roi, k=1.0, sigma_blur=1.0, min_area=10):
    """
    ROI에서 Mean+kσ 임계값 이하 픽셀의 외곽선(contour)을 추출.

    Returns
    -------
    contours : list of np.ndarray — 크랙 경계선 좌표 (ROI 로컬)
    hierarchy : contour 계층 정보
    threshold : 사용된 임계값
    mu, std : ROI 통계
    mask : 이진 마스크 (디버그/시각화용)
    """
    # Gaussian blur
    ksize = max(3, int(np.ceil(sigma_blur * 3)) * 2 + 1)
    blurred = cv2.GaussianBlur(roi, (ksize, ksize), sigma_blur)

    # Mean+kσ 임계값
    mu = float(blurred.mean())
    std = float(blurred.std())
    threshold = mu - k * std

    # 이진화
    binary = (blurred < threshold).astype(np.uint8) * 255

    # 후처리: 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
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

    return {
        'contours': contours,
        'hierarchy': hierarchy,
        'threshold': threshold,
        'mu': mu,
        'std': std,
        'mask': binary,
    }


def contours_to_boundary_points(contours):
    """contour 리스트를 (N, 2) 배열로 변환. 열: [x, y]."""
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
        return np.vstack(all_pts)  # (N, 2), columns = [x, y]
    return np.array([]).reshape(0, 2)


# ══════════════════════════════════════════════════
# Step 3: POI → 크랙 경계 거리/방향 계산
# ══════════════════════════════════════════════════

def compute_poi_to_boundary(icgn_result, boundary_points_global, roi_bounds):
    """
    각 유효 POI에서 가장 가까운 크랙 경계 픽셀까지의 거리와 방향을 계산.

    Returns
    -------
    poi_boundary_info : list of dict
        각 POI에 대해:
        - poi_idx: POI 인덱스
        - poi_x, poi_y: POI 좌표 (global)
        - is_bad: 불량 POI 여부
        - nearest_boundary: 가장 가까운 경계점 (x, y)
        - distance: 최단 거리 (px)
        - direction: 경계까지의 방향 벡터 (dx, dy) 정규화
        - angle_deg: 경계 방향 각도 (degrees)
    """
    poi_x = icgn_result.points_x
    poi_y = icgn_result.points_y
    valid_mask = icgn_result.valid_mask
    n_poi = len(poi_x)

    if len(boundary_points_global) == 0:
        return []

    results = []

    for i in range(n_poi):
        px, py = float(poi_x[i]), float(poi_y[i])

        # 모든 경계점까지 거리
        dists = np.sqrt((boundary_points_global[:, 0] - px)**2 +
                        (boundary_points_global[:, 1] - py)**2)
        nearest_idx = np.argmin(dists)
        nearest_dist = dists[nearest_idx]
        nearest_pt = boundary_points_global[nearest_idx]

        # 방향 벡터 (POI → 경계)
        dx = nearest_pt[0] - px
        dy = nearest_pt[1] - py
        norm = max(np.sqrt(dx**2 + dy**2), 1e-10)
        dir_x = dx / norm
        dir_y = dy / norm
        angle = np.degrees(np.arctan2(dy, dx))

        results.append({
            'poi_idx': i,
            'poi_x': px,
            'poi_y': py,
            'is_valid': bool(valid_mask[i]),
            'is_bad': not bool(valid_mask[i]),
            'nearest_boundary': (float(nearest_pt[0]), float(nearest_pt[1])),
            'distance': float(nearest_dist),
            'direction': (float(dir_x), float(dir_y)),
            'angle_deg': float(angle),
        })

    return results


# ══════════════════════════════════════════════════
# 메인 테스트
# ══════════════════════════════════════════════════

def run_test():
    print("데이터 로드...")
    def_img = _imread_unicode(DATA_DIR / 'deformed.tiff')
    ref_img = _imread_unicode(DATA_DIR / 'reference.tiff')
    gt_mask = np.load(str(DATA_DIR / 'crack_mask.npy')).astype(bool)
    H, W = def_img.shape
    print(f"이미지: {W}×{H}, GT 크랙 픽셀: {gt_mask.sum()}")

    # ── DIC 실행 ──
    from speckle.core.initial_guess import compute_fft_cc
    from speckle.core.optimization import compute_icgn

    subset_size = 25
    spacing = 21

    print("\n[DIC] FFT-CC + IC-GN...")
    fft_result = compute_fft_cc(
        ref_image=ref_img.astype(np.float64),
        def_image=def_img.astype(np.float64),
        subset_size=subset_size, spacing=spacing, zncc_threshold=0.6
    )
    icgn_result = compute_icgn(
        ref_image=ref_img.astype(np.float64),
        def_image=def_img.astype(np.float64),
        subset_size=subset_size, initial_guess=fft_result,
        max_iterations=50, convergence_threshold=1e-3,
        zncc_threshold=0.9, shape_function='affine',
        enable_variable_subset=False, enable_adss_subset=False,
    )

    n_valid = int(icgn_result.valid_mask.sum())
    n_bad = int((~icgn_result.valid_mask).sum())
    print(f"  유효 POI: {n_valid}, 불량 POI: {n_bad}")

    # ═══════════════════════════════════════════════════
    # Step 1: ROI 구성
    # ═══════════════════════════════════════════════════
    print("\n[Step 1] ROI 구성...")
    t0 = time.time()
    roi_data = build_roi(def_img, icgn_result, spacing, subset_size)
    roi = roi_data['roi']
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_data['bounds']
    rH, rW = roi.shape
    roi_gt = gt_mask[roi_y1:roi_y2, roi_x1:roi_x2]
    print(f"  ROI: ({roi_x1},{roi_y1})~({roi_x2},{roi_y2}), {rW}×{rH}px")
    print(f"  불량 POI: {len(roi_data['bad_indices'])}개")
    print(f"  소요 시간: {time.time()-t0:.3f}s")

    # ═══════════════════════════════════════════════════
    # Step 2: Mean+kσ → contour 경계선 추출
    # ═══════════════════════════════════════════════════
    print("\n[Step 2] 크랙 경계선 추출...")

    # k값 스윕
    k_values = [0.6, 0.8, 1.0, 1.2, 1.5, 2.0]
    step2_results = {}

    print(f"\n{'k':>5s} | {'Thr':>6s} | {'Contours':>8s} | {'경계점':>7s} | "
          f"{'IoU':>6s} | {'Prec':>6s} | {'Recall':>6s} | {'시간':>6s}")
    print("-" * 65)

    for k in k_values:
        t0 = time.time()
        boundary = extract_crack_boundary(roi, k=k, sigma_blur=1.0, min_area=10)
        elapsed = time.time() - t0

        # 경계점 배열
        bnd_pts = contours_to_boundary_points(boundary['contours'])

        # 평가: mask 기반 IoU
        metrics = compute_metrics(roi_gt, boundary['mask'] > 0)

        step2_results[k] = {
            'boundary': boundary,
            'bnd_pts': bnd_pts,
            'metrics': metrics,
            'elapsed': elapsed,
        }

        print(f"{k:5.1f} | {boundary['threshold']:6.1f} | "
              f"{len(boundary['contours']):8d} | {len(bnd_pts):7d} | "
              f"{metrics['iou']:6.4f} | {metrics['precision']:6.4f} | "
              f"{metrics['recall']:6.4f} | {elapsed:5.3f}s")

    best_k = max(step2_results, key=lambda k: step2_results[k]['metrics']['iou'])
    best_step2 = step2_results[best_k]
    best_boundary = best_step2['boundary']
    best_bnd_pts = best_step2['bnd_pts']
    best_metrics = best_step2['metrics']

    print(f"\n  최적 k={best_k}, IoU={best_metrics['iou']:.4f}")
    print(f"  contour 수: {len(best_boundary['contours'])}")
    print(f"  경계점 수: {len(best_bnd_pts)}")

    # 경계점을 global 좌표로 변환
    if len(best_bnd_pts) > 0:
        bnd_pts_global = best_bnd_pts.copy().astype(np.float64)
        bnd_pts_global[:, 0] += roi_x1
        bnd_pts_global[:, 1] += roi_y1
    else:
        bnd_pts_global = np.array([]).reshape(0, 2)

    # GT 경계선
    gt_binary = (roi_gt.astype(np.uint8)) * 255
    gt_contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    gt_bnd_pts = contours_to_boundary_points(gt_contours)

    # 경계선 오차: 검출 경계점에서 GT 경계까지의 평균 거리
    if len(best_bnd_pts) > 0 and len(gt_bnd_pts) > 0:
        dists_det_to_gt = distance.cdist(best_bnd_pts, gt_bnd_pts).min(axis=1)
        dists_gt_to_det = distance.cdist(gt_bnd_pts, best_bnd_pts).min(axis=1)
        boundary_err_mean = float(np.mean(dists_det_to_gt))
        boundary_err_max = float(np.max(dists_det_to_gt))
        hausdorff = max(float(np.max(dists_det_to_gt)), float(np.max(dists_gt_to_det)))
        print(f"  경계 오차: 평균={boundary_err_mean:.2f}px, "
              f"최대={boundary_err_max:.2f}px, Hausdorff={hausdorff:.2f}px")
    else:
        boundary_err_mean = float('inf')
        boundary_err_max = float('inf')
        hausdorff = float('inf')

    # ═══════════════════════════════════════════════════
    # Step 3: POI → 경계 거리/방향 계산
    # ═══════════════════════════════════════════════════
    print("\n[Step 3] POI → 경계 거리/방향 계산...")
    t0 = time.time()
    poi_info = compute_poi_to_boundary(icgn_result, bnd_pts_global, roi_data['bounds'])
    elapsed3 = time.time() - t0

    # 통계
    valid_info = [p for p in poi_info if p['is_valid']]
    bad_info = [p for p in poi_info if p['is_bad']]
    near_crack_info = [p for p in valid_info if p['distance'] < spacing]

    if valid_info:
        valid_dists = [p['distance'] for p in valid_info]
        print(f"  유효 POI → 경계 거리: min={min(valid_dists):.1f}, "
              f"max={max(valid_dists):.1f}, mean={np.mean(valid_dists):.1f}px")
    if bad_info:
        bad_dists = [p['distance'] for p in bad_info]
        print(f"  불량 POI → 경계 거리: min={min(bad_dists):.1f}, "
              f"max={max(bad_dists):.1f}, mean={np.mean(bad_dists):.1f}px")
    print(f"  크랙 근처 유효 POI (거리<{spacing}px): {len(near_crack_info)}개")
    print(f"  소요 시간: {elapsed3:.3f}s")

    # ═══════════════════════════════════════════════════
    # 시각화
    # ═══════════════════════════════════════════════════

    # ── Figure 1: 파이프라인 단계별 ──
    fig1 = plt.figure(figsize=(24, 14))
    fig1.suptitle('Figure 1: 크랙 경계 추출 파이프라인', fontsize=16, fontweight='bold')
    gs1 = gridspec.GridSpec(3, 4, hspace=0.4, wspace=0.3)

    # 1a: ROI + GT + 불량 POI
    ax = fig1.add_subplot(gs1[0, 0])
    ax.imshow(roi, cmap='gray')
    gt_ov = np.zeros((*roi_gt.shape, 4))
    gt_ov[roi_gt] = [1, 0, 0, 0.35]
    ax.imshow(gt_ov)
    for px, py in roi_data['bad_coords']:
        lx, ly = px - roi_x1, py - roi_y1
        if 0 <= lx < rW and 0 <= ly < rH:
            ax.plot(lx, ly, 'c+', markersize=4, markeredgewidth=0.8)
    ax.set_title(f'(a) ROI + GT(빨강) + 불량POI(+)\n{rW}×{rH}px')
    ax.axis('off')

    # 1b: Mean+kσ → contour
    ax = fig1.add_subplot(gs1[0, 1])
    ax.imshow(roi, cmap='gray')
    # contour 그리기
    contour_img = np.zeros((*roi.shape, 4))
    cv2.drawContours(contour_img[:, :, :3].astype(np.uint8),
                     best_boundary['contours'], -1, (255, 0, 0), 1)
    # 수동으로 contour 라인
    for cnt in best_boundary['contours']:
        pts = cnt.squeeze()
        if pts.ndim == 2 and len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=1.5, alpha=0.9)
    # GT contour
    for cnt in gt_contours:
        pts = cnt.squeeze()
        if pts.ndim == 2 and len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=1.5, alpha=0.7)
    ax.set_title(f'(b) 크랙 경계선 (k={best_k})\n'
                 f'빨강=검출, 초록=GT\n경계점 {len(best_bnd_pts)}개')
    ax.axis('off')

    # 1c: TP/FP/FN
    ax = fig1.add_subplot(gs1[0, 2])
    vis = np.zeros((*roi_gt.shape, 3), dtype=np.uint8)
    det_mask = best_boundary['mask'] > 0
    vis[det_mask & roi_gt] = [0, 255, 0]
    vis[det_mask & ~roi_gt] = [255, 0, 0]
    vis[~det_mask & roi_gt] = [0, 0, 255]
    ax.imshow(vis)
    ax.set_title(f'(c) TP/FP/FN\nIoU={best_metrics["iou"]:.3f} '
                 f'P={best_metrics["precision"]:.3f} R={best_metrics["recall"]:.3f}')
    ax.axis('off')

    # 1d: 요약 텍스트
    ax = fig1.add_subplot(gs1[0, 3])
    ax.axis('off')
    summary = (
        f"크랙 경계 추출 결과\n"
        f"{'='*30}\n\n"
        f"ROI: {rW}×{rH}px\n"
        f"불량 POI: {len(roi_data['bad_indices'])}개\n\n"
        f"Mean+kσ (k={best_k}):\n"
        f"  μ={best_boundary['mu']:.1f}\n"
        f"  σ={best_boundary['std']:.1f}\n"
        f"  Thr={best_boundary['threshold']:.1f}\n\n"
        f"경계선:\n"
        f"  contour 수: {len(best_boundary['contours'])}\n"
        f"  경계점 수: {len(best_bnd_pts)}\n\n"
        f"경계 정밀도:\n"
        f"  평균 오차: {boundary_err_mean:.2f}px\n"
        f"  Hausdorff: {hausdorff:.2f}px\n\n"
        f"마스크 IoU: {best_metrics['iou']:.4f}"
    )
    ax.text(0.05, 0.95, summary, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # Row 2: POI → 경계 거리 시각화
    # 2a: 전체 이미지에 POI 거리 표시
    ax = fig1.add_subplot(gs1[1, 0:2])
    ax.imshow(def_img, cmap='gray', alpha=0.6)

    # 크랙 경계선
    if len(bnd_pts_global) > 0:
        ax.plot(bnd_pts_global[:, 0], bnd_pts_global[:, 1], 'r.', markersize=0.5, alpha=0.5)

    # POI 거리별 색상
    if poi_info:
        all_dists = [p['distance'] for p in poi_info]
        max_dist = max(all_dists) if all_dists else 1

        for p in poi_info:
            if p['is_bad']:
                ax.plot(p['poi_x'], p['poi_y'], 'rx', markersize=4, markeredgewidth=0.8)
            else:
                # 거리에 따라 색상 (가까우면 노랑, 멀면 파랑)
                norm_d = min(p['distance'] / max(spacing * 2, 1), 1.0)
                color = plt.cm.coolwarm(1.0 - norm_d)
                ax.plot(p['poi_x'], p['poi_y'], 'o', color=color,
                        markersize=3, markeredgewidth=0)

        # 크랙 근처 POI에서 경계까지 화살표
        for p in near_crack_info[:20]:  # 최대 20개
            ax.annotate('', xy=p['nearest_boundary'],
                        xytext=(p['poi_x'], p['poi_y']),
                        arrowprops=dict(arrowstyle='->', color='yellow',
                                        lw=0.8, alpha=0.7))

    # ROI 경계
    rect = plt.Rectangle((roi_x1, roi_y1), roi_x2 - roi_x1, roi_y2 - roi_y1,
                          linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    ax.set_xlim(max(0, roi_x1 - 30), min(W, roi_x2 + 30))
    ax.set_ylim(min(H, roi_y2 + 30), max(0, roi_y1 - 30))
    ax.set_title('(d) POI → 경계 거리\n(노랑=가까움, 파랑=멀음, 빨강×=불량)')

    # 2b: 거리 히스토그램
    ax = fig1.add_subplot(gs1[1, 2])
    if valid_info:
        valid_dists = [p['distance'] for p in valid_info]
        bad_dists = [p['distance'] for p in bad_info]
        bins = np.arange(0, max(max(valid_dists, default=0), max(bad_dists, default=0)) + 5, 2)
        ax.hist(valid_dists, bins=bins, alpha=0.6, color='steelblue', label='유효 POI')
        if bad_dists:
            ax.hist(bad_dists, bins=bins, alpha=0.6, color='red', label='불량 POI')
        ax.axvline(spacing, color='orange', linestyle='--', linewidth=2,
                   label=f'spacing={spacing}px')
        ax.set_xlabel('경계까지 거리 (px)')
        ax.set_ylabel('POI 수')
        ax.set_title('(e) POI→경계 거리 분포')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # 2c: 방향 분포 (크랙 근처 POI만)
    ax = fig1.add_subplot(gs1[1, 3])
    if near_crack_info:
        angles = [p['angle_deg'] for p in near_crack_info]
        ax.hist(angles, bins=36, range=(-180, 180), color='steelblue',
                edgecolor='black', alpha=0.7)
        ax.set_xlabel('경계 방향 (°)')
        ax.set_ylabel('POI 수')
        ax.set_title(f'(f) 크랙 근처 POI의 경계 방향\n({len(near_crack_info)}개 POI)')
        ax.grid(True, alpha=0.3)

    # Row 3: k값 민감도 + 경계 줌
    # 3a: k값 vs IoU
    ax = fig1.add_subplot(gs1[2, 0])
    ks = sorted(step2_results.keys())
    ious = [step2_results[k]['metrics']['iou'] for k in ks]
    precs = [step2_results[k]['metrics']['precision'] for k in ks]
    recs = [step2_results[k]['metrics']['recall'] for k in ks]
    n_pts = [len(step2_results[k]['bnd_pts']) for k in ks]

    ax.plot(ks, ious, 'ko-', linewidth=2, markersize=8, label='IoU')
    ax.plot(ks, precs, 's--', color='tab:orange', label='Precision')
    ax.plot(ks, recs, '^--', color='tab:blue', label='Recall')
    ax.axvline(best_k, color='green', linestyle=':', linewidth=2,
               label=f'최적 k={best_k}')
    ax.set_xlabel('k 값')
    ax.set_ylabel('Score')
    ax.set_title('(g) k값 → 검출 성능')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    # 3b: 크랙 경계 줌 (중앙부)
    ax = fig1.add_subplot(gs1[2, 1:3])
    cy = rH // 2
    zy1 = max(0, cy - 25)
    zy2 = min(rH, cy + 25)
    ax.imshow(roi[zy1:zy2, :], cmap='gray', aspect='auto')

    # GT contour
    for cnt in gt_contours:
        pts = cnt.squeeze()
        if pts.ndim == 2 and len(pts) > 1:
            mask_y = (pts[:, 1] >= zy1) & (pts[:, 1] < zy2)
            if np.any(mask_y):
                ax.plot(pts[mask_y, 0], pts[mask_y, 1] - zy1, 'g-',
                        linewidth=2, alpha=0.8)

    # 검출 contour
    for cnt in best_boundary['contours']:
        pts = cnt.squeeze()
        if pts.ndim == 2 and len(pts) > 1:
            mask_y = (pts[:, 1] >= zy1) & (pts[:, 1] < zy2)
            if np.any(mask_y):
                ax.plot(pts[mask_y, 0], pts[mask_y, 1] - zy1, 'r-',
                        linewidth=2, alpha=0.8)

    ax.set_title('(h) 경계선 줌 (초록=GT, 빨강=검출)')
    ax.axis('off')

    # 3c: 경계 오차 분포
    ax = fig1.add_subplot(gs1[2, 3])
    if len(best_bnd_pts) > 0 and len(gt_bnd_pts) > 0:
        ax.hist(dists_det_to_gt, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(boundary_err_mean, color='red', linewidth=2, linestyle='--',
                   label=f'평균={boundary_err_mean:.2f}px')
        ax.set_xlabel('경계 오차 (px)')
        ax.set_ylabel('빈도')
        ax.set_title(f'(i) 경계 오차 분포\n'
                     f'평균={boundary_err_mean:.2f}, Hausdorff={hausdorff:.2f}px')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig1.savefig(str(OUTPUT_DIR / 'fig1_boundary_extraction.png'), dpi=150, bbox_inches='tight')
    print(f"\n  → {OUTPUT_DIR / 'fig1_boundary_extraction.png'}")

    # ── Figure 2: POI 상세 (크랙 근처) ──
    fig2 = plt.figure(figsize=(20, 10))
    fig2.suptitle('Figure 2: 크랙 근처 POI 상세', fontsize=16, fontweight='bold')
    gs2 = gridspec.GridSpec(2, 4, hspace=0.35, wspace=0.3)

    # 크랙 근처 POI 상세 테이블
    ax = fig2.add_subplot(gs2[0, 0:2])
    if near_crack_info:
        sorted_near = sorted(near_crack_info, key=lambda p: p['distance'])[:15]
        table_data = []
        for p in sorted_near:
            table_data.append([
                f"{int(p['poi_x'])},{int(p['poi_y'])}",
                f"{p['distance']:.1f}",
                f"{p['angle_deg']:.0f}°",
                f"({p['nearest_boundary'][0]:.0f},{p['nearest_boundary'][1]:.0f})",
            ])

        table = ax.table(cellText=table_data,
                         colLabels=['POI (x,y)', '거리(px)', '방향', '최근접 경계'],
                         loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)
    ax.axis('off')
    ax.set_title(f'크랙 근처 유효 POI (거리 < {spacing}px)')

    # POI → 경계 화살표 줌
    ax = fig2.add_subplot(gs2[0, 2:4])
    ax.imshow(roi, cmap='gray', alpha=0.6)

    # 크랙 경계선
    for cnt in best_boundary['contours']:
        pts = cnt.squeeze()
        if pts.ndim == 2 and len(pts) > 1:
            ax.plot(pts[:, 0], pts[:, 1], 'r-', linewidth=2, alpha=0.8)

    # 유효 POI + 화살표
    for p in poi_info:
        lx = p['poi_x'] - roi_x1
        ly = p['poi_y'] - roi_y1
        if not (0 <= lx < rW and 0 <= ly < rH):
            continue

        if p['is_bad']:
            ax.plot(lx, ly, 'rx', markersize=5, markeredgewidth=1)
        elif p['distance'] < spacing:
            ax.plot(lx, ly, 'yo', markersize=4, markeredgewidth=0)
            bx = p['nearest_boundary'][0] - roi_x1
            by = p['nearest_boundary'][1] - roi_y1
            ax.annotate('', xy=(bx, by), xytext=(lx, ly),
                        arrowprops=dict(arrowstyle='->', color='yellow',
                                        lw=1, alpha=0.8))
        else:
            ax.plot(lx, ly, 'b.', markersize=2, alpha=0.3)

    ax.set_title('POI → 경계 화살표\n(노랑=근접 유효, 빨강×=불량, 파랑=원거리)')
    ax.axis('off')

    # Row 2: 경계 오차 상세
    # 2a: k값별 경계 오차
    ax = fig2.add_subplot(gs2[1, 0:2])
    bnd_errs_mean = []
    bnd_errs_hausdorff = []
    for k in ks:
        bpts = step2_results[k]['bnd_pts']
        if len(bpts) > 0 and len(gt_bnd_pts) > 0:
            d2g = distance.cdist(bpts, gt_bnd_pts).min(axis=1)
            g2d = distance.cdist(gt_bnd_pts, bpts).min(axis=1)
            bnd_errs_mean.append(float(np.mean(d2g)))
            bnd_errs_hausdorff.append(max(float(np.max(d2g)), float(np.max(g2d))))
        else:
            bnd_errs_mean.append(float('inf'))
            bnd_errs_hausdorff.append(float('inf'))

    ax.plot(ks, bnd_errs_mean, 'bo-', linewidth=2, markersize=8, label='평균 경계 오차')
    ax.plot(ks, bnd_errs_hausdorff, 'rs--', linewidth=2, markersize=8, label='Hausdorff')
    ax.axvline(best_k, color='green', linestyle=':', linewidth=2)
    ax.set_xlabel('k 값')
    ax.set_ylabel('경계 오차 (px)')
    ax.set_title('k값 → 경계 오차')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # 2b: 경계점별 오차 시각화 (위치별)
    ax = fig2.add_subplot(gs2[1, 2:4])
    if len(best_bnd_pts) > 0 and len(gt_bnd_pts) > 0:
        ax.imshow(roi, cmap='gray', alpha=0.4)
        sc = ax.scatter(best_bnd_pts[:, 0], best_bnd_pts[:, 1],
                        c=dists_det_to_gt, cmap='hot_r', s=2,
                        vmin=0, vmax=max(3, np.percentile(dists_det_to_gt, 95)))
        plt.colorbar(sc, ax=ax, label='GT까지 거리 (px)', shrink=0.8)
        for cnt in gt_contours:
            pts = cnt.squeeze()
            if pts.ndim == 2 and len(pts) > 1:
                ax.plot(pts[:, 0], pts[:, 1], 'g-', linewidth=1.5, alpha=0.6)
        ax.set_title('경계점별 GT까지 거리\n(빨강=큰 오차, 파랑=작은 오차)')
    ax.axis('off')

    fig2.savefig(str(OUTPUT_DIR / 'fig2_poi_detail.png'), dpi=150, bbox_inches='tight')
    print(f"  → {OUTPUT_DIR / 'fig2_poi_detail.png'}")

    # ── 콘솔 최종 요약 ──
    print(f"\n{'='*70}")
    print("크랙 경계 추출 최종 결과")
    print(f"{'='*70}")
    print(f"ROI:              {rW}×{rH}px")
    print(f"불량 POI:          {len(roi_data['bad_indices'])}개")
    print(f"최적 k:            {best_k}")
    print(f"마스크 IoU:        {best_metrics['iou']:.4f}")
    print(f"contour 수:        {len(best_boundary['contours'])}")
    print(f"경계점 수:         {len(best_bnd_pts)}")
    print(f"경계 평균 오차:    {boundary_err_mean:.2f}px")
    print(f"Hausdorff 거리:    {hausdorff:.2f}px")
    print(f"크랙 근처 유효 POI: {len(near_crack_info)}개")
    print(f"\n결과 저장: {OUTPUT_DIR}")
    print(f"{'='*70}")

    plt.show()
    print("완료!")


if __name__ == '__main__':
    print("크랙 경계 추출 테스트")
    print(f"데이터: {DATA_DIR}")
    print(f"출력:   {OUTPUT_DIR}\n")
    run_test()

