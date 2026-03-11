"""
tests/test_crack_shape_extraction.py
크랙 형상 + Tip 추출 파이프라인
1. 불량 POI → ROI 구성
2. ROI에서 Mean+kσ 크랙 마스크 생성
3. 수평 커널 closing으로 끊긴 부분 연결
4. 골격화(skeletonize)로 크랙 중심선 추출
5. 중심선에서 tip 위치, 크랙 길이, 방향, opening 프로파일 계산
"""
import sys
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage
from skimage.morphology import skeletonize, thin
from skimage.measure import label as sk_label, regionprops
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
OUTPUT_DIR = TEST_OUTPUT_ROOT / 'output_crack_shape'
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


def make_tp_fp_fn_overlay(gt, pred):
    comp = np.zeros((*gt.shape, 3), dtype=np.uint8)
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    comp[gt_b & pred_b] = [0, 255, 0]
    comp[~gt_b & pred_b] = [255, 0, 0]
    comp[gt_b & ~pred_b] = [0, 0, 255]
    return comp


# ══════════════════════════════════════════════════
# 크랙 형상 추출 파이프라인
# ══════════════════════════════════════════════════

def step1_build_roi(def_img, icgn_result, spacing, subset_size):
    """1단계: 불량 POI + 8방위 이웃으로 ROI 구성"""
    H, W = def_img.shape
    M = subset_size // 2
    bad_indices = np.where(~icgn_result.valid_mask)[0]
    poi_x = icgn_result.points_x
    poi_y = icgn_result.points_y

    bad_coords = set()
    bad_only_coords = set()
    for idx in bad_indices:
        px, py = int(poi_x[idx]), int(poi_y[idx])
        bad_only_coords.add((px, py))
        bad_coords.add((px, py))
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

    return roi, (roi_x1, roi_y1, roi_x2, roi_y2), bad_indices, bad_only_coords


def step2_detect_crack_mask(roi, sigma_blur=1.0, k=1.0):
    """2단계: Mean+kσ 크랙 마스크 생성"""
    ksize = max(3, int(np.ceil(sigma_blur * 3)) * 2 + 1)
    blurred = cv2.GaussianBlur(roi, (ksize, ksize), sigma_blur)
    mu = float(blurred.mean())
    std = float(blurred.std())
    thr = mu - k * std
    detected = (blurred < thr).astype(np.uint8)

    # 기본 후처리
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    detected = cv2.morphologyEx(detected, cv2.MORPH_CLOSE, kernel_small)
    detected = cv2.morphologyEx(detected, cv2.MORPH_OPEN, kernel_small)

    return detected, thr, mu, std


def step3_connect_gaps(mask, h_kernel_width=11, v_kernel_height=3):
    """3단계: 수평 커널 closing으로 끊긴 부분 연결"""
    # 수평 방향 closing (좌우 끊김 연결)
    kernel_h = cv2.getStructuringElement(cv2.MORPH_RECT, (h_kernel_width, 1))
    connected = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_h)

    # 수직 방향 약한 closing (위아래 미세 갭)
    kernel_v = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_kernel_height))
    connected = cv2.morphologyEx(connected, cv2.MORPH_CLOSE, kernel_v)

    # 노이즈 제거: 작은 성분 제거
    labeled, n_comp = ndimage.label(connected)
    sizes = ndimage.sum(connected, labeled, range(1, n_comp + 1))
    if len(sizes) > 0:
        max_size = max(sizes)
        for i in range(1, n_comp + 1):
            if sizes[i - 1] < max_size * 0.05:  # 최대 성분의 5% 미만 제거
                connected[labeled == i] = 0

    return connected


def step4_skeletonize_and_extract(crack_mask):
    """4단계: 골격화 + 중심선 추출"""
    skeleton = skeletonize(crack_mask > 0).astype(np.uint8)

    # 골격 위의 점 좌표 추출
    skel_ys, skel_xs = np.where(skeleton > 0)
    if len(skel_xs) == 0:
        return skeleton, np.array([]), np.array([]), [], []

    # x 기준 정렬
    sort_idx = np.argsort(skel_xs)
    skel_xs = skel_xs[sort_idx]
    skel_ys = skel_ys[sort_idx]

    # endpoint 검출 (이웃이 1개인 골격 픽셀)
    endpoints = []
    for i in range(len(skel_xs)):
        x, y = int(skel_xs[i]), int(skel_ys[i])
        # 8방위 이웃 카운트
        neighbors = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                    if skeleton[ny, nx] > 0:
                        neighbors += 1
        if neighbors == 1:
            endpoints.append((x, y))

    # branchpoint 검출 (이웃이 3개 이상)
    branchpoints = []
    for i in range(len(skel_xs)):
        x, y = int(skel_xs[i]), int(skel_ys[i])
        neighbors = 0
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1]:
                    if skeleton[ny, nx] > 0:
                        neighbors += 1
        if neighbors >= 3:
            branchpoints.append((x, y))

    return skeleton, skel_xs, skel_ys, endpoints, branchpoints


def step5_measure_crack(crack_mask, skeleton, skel_xs, skel_ys, endpoints,
                         roi_bounds, pixel_size=1.0):
    """
    5단계: 크랙 측정
    - tip 위치 (가장 안쪽 endpoint)
    - 크랙 길이 (골격 픽셀 수 × pixel_size)
    - 방향 (주 방향 각도)
    - opening 프로파일 (각 x에서의 크랙 두께)
    """
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
    H, W = crack_mask.shape

    results = {}

    # ── Tip 위치 ──
    if len(endpoints) >= 2:
        # x가 가장 큰 endpoint = 크랙 tip (크랙이 좌→우 진행 가정)
        endpoints_sorted = sorted(endpoints, key=lambda p: p[0])
        tip_local = endpoints_sorted[-1]  # 가장 오른쪽
        mouth_local = endpoints_sorted[0]  # 가장 왼쪽

        results['tip_local'] = tip_local
        results['tip_global'] = (tip_local[0] + roi_x1, tip_local[1] + roi_y1)
        results['mouth_local'] = mouth_local
        results['mouth_global'] = (mouth_local[0] + roi_x1, mouth_local[1] + roi_y1)
    elif len(endpoints) == 1:
        results['tip_local'] = endpoints[0]
        results['tip_global'] = (endpoints[0][0] + roi_x1, endpoints[0][1] + roi_y1)
        results['mouth_local'] = (0, int(skel_ys[0]) if len(skel_ys) > 0 else 0)
        results['mouth_global'] = (roi_x1, results['mouth_local'][1] + roi_y1)
    else:
        results['tip_local'] = (0, 0)
        results['tip_global'] = (roi_x1, roi_y1)
        results['mouth_local'] = (0, 0)
        results['mouth_global'] = (roi_x1, roi_y1)

    # ── 크랙 길이 ──
    # 골격 픽셀 간 누적 거리
    if len(skel_xs) > 1:
        dx = np.diff(skel_xs.astype(np.float64))
        dy = np.diff(skel_ys.astype(np.float64))
        segment_lengths = np.sqrt(dx**2 + dy**2)
        crack_length = float(np.sum(segment_lengths)) * pixel_size
    else:
        crack_length = 0.0
    results['crack_length_px'] = crack_length
    results['n_skeleton_pixels'] = len(skel_xs)

    # ── 주 방향 ──
    if len(skel_xs) > 5:
        # 선형 회귀로 주 방향
        coeffs = np.polyfit(skel_xs.astype(float), skel_ys.astype(float), 1)
        slope = coeffs[0]
        angle_deg = float(np.degrees(np.arctan(slope)))
        results['direction_deg'] = angle_deg
        results['slope'] = float(slope)
        results['polyfit'] = coeffs
    else:
        results['direction_deg'] = 0.0
        results['slope'] = 0.0
        results['polyfit'] = [0, 0]

    # ── Opening 프로파일 ──
    opening_profile_x = []
    opening_profile_val = []
    centerline_y = []

    for x in range(W):
        col = crack_mask[:, x]
        crack_ys = np.where(col > 0)[0]
        if len(crack_ys) >= 1:
            opening = crack_ys.max() - crack_ys.min() + 1
            center = (crack_ys.max() + crack_ys.min()) / 2.0
        else:
            opening = 0
            center = np.nan
        opening_profile_x.append(x + roi_x1)
        opening_profile_val.append(opening)
        centerline_y.append(center + roi_y1 if not np.isnan(center) else np.nan)

    results['opening_x'] = np.array(opening_profile_x)
    results['opening_val'] = np.array(opening_profile_val)
    results['centerline_y'] = np.array(centerline_y)
    results['max_opening'] = max(opening_profile_val) if opening_profile_val else 0
    results['mean_opening'] = float(np.mean([v for v in opening_profile_val if v > 0])) \
        if any(v > 0 for v in opening_profile_val) else 0.0

    # ── Endpoints / Branchpoints ──
    results['endpoints'] = endpoints
    results['n_endpoints'] = len(endpoints)

    return results


# ══════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════

def run_pipeline():
    # ── 데이터 로드 ──
    def_img = _imread_unicode(DATA_DIR / 'deformed.tiff')
    ref_img = _imread_unicode(DATA_DIR / 'reference.tiff')
    gt_mask = np.load(str(DATA_DIR / 'crack_mask.npy'))
    H, W = def_img.shape

    gt_clean = gt_mask.copy()
    gt_clean[:220, :] = False
    gt_clean[280:, :] = False

    subset_size = 25
    spacing = 21
    zncc_threshold = 0.9
    crack_y_gt = 250
    tip_x_gt = 250

    # ── DIC 실행 ──
    from speckle.core.initial_guess import compute_fft_cc
    from speckle.core.optimization import compute_icgn

    print("=" * 60)
    print("크랙 형상 + Tip 추출 파이프라인")
    print("=" * 60)

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
        zncc_threshold=zncc_threshold, shape_function='affine',
        enable_variable_subset=False, enable_adss_subset=False,
    )

    # ── 파이프라인 실행 ──
    print("\n[Step 1] ROI 구성...")
    roi, roi_bounds, bad_indices, bad_coords = step1_build_roi(
        def_img, icgn_result, spacing, subset_size)
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds
    rH, rW = roi.shape
    roi_gt = gt_clean[roi_y1:roi_y2, roi_x1:roi_x2]
    print(f"  ROI: ({roi_x1},{roi_y1})~({roi_x2},{roi_y2}), {rW}×{rH}px")
    print(f"  불량 POI: {len(bad_indices)}개")

    print("\n[Step 2] Mean+kσ 크랙 마스크...")
    raw_mask, thr, mu, std = step2_detect_crack_mask(roi, sigma_blur=1.0, k=1.0)
    m_raw = compute_metrics(roi_gt, raw_mask)
    print(f"  임계값: {thr:.1f} (μ={mu:.1f}, σ={std:.1f})")
    print(f"  검출 픽셀: {raw_mask.sum()}, IoU={m_raw['iou']:.4f}")

    print("\n[Step 3] 끊김 연결 (수평 closing)...")
    connected_mask = step3_connect_gaps(raw_mask, h_kernel_width=11, v_kernel_height=3)
    m_conn = compute_metrics(roi_gt, connected_mask)
    print(f"  검출 픽셀: {connected_mask.sum()}, IoU={m_conn['iou']:.4f}")

    print("\n[Step 4] 골격화 + 중심선 추출...")
    skeleton, skel_xs, skel_ys, endpoints, branchpoints = \
        step4_skeletonize_and_extract(connected_mask)
    print(f"  골격 픽셀: {len(skel_xs)}")
    print(f"  Endpoints: {len(endpoints)}")
    print(f"  Branchpoints: {len(branchpoints)}")

    print("\n[Step 5] 크랙 측정...")
    results = step5_measure_crack(
        connected_mask, skeleton, skel_xs, skel_ys, endpoints, roi_bounds)

    print(f"  Tip (global):  {results['tip_global']}")
    print(f"  Mouth (global): {results['mouth_global']}")
    print(f"  크랙 길이:     {results['crack_length_px']:.1f}px")
    print(f"  주 방향:       {results['direction_deg']:.2f}°")
    print(f"  최대 opening:  {results['max_opening']}px")
    print(f"  평균 opening:  {results['mean_opening']:.1f}px")
    print(f"  Tip 오차:      {abs(results['tip_global'][0] - tip_x_gt)}px (GT={tip_x_gt})")

    # ══════════════════════════════════════════════════
    # Figure 1: 파이프라인 단계별 시각화
    # ══════════════════════════════════════════════════
    print("\nFigure 1: 단계별 시각화...")

    fig1, axes = plt.subplots(2, 4, figsize=(24, 10))
    fig1.suptitle('Figure 1: 크랙 형상 추출 파이프라인 단계별', fontsize=16, fontweight='bold')

    # (a) 원본 + GT + 불량 POI
    axes[0, 0].imshow(roi, cmap='gray')
    gt_ov = np.zeros((*roi_gt.shape, 4))
    gt_ov[roi_gt.astype(bool)] = [1, 0, 0, 0.35]
    axes[0, 0].imshow(gt_ov)
    for px, py in bad_coords:
        lx, ly = px - roi_x1, py - roi_y1
        if 0 <= lx < rW and 0 <= ly < rH:
            axes[0, 0].plot(lx, ly, 'c+', markersize=3, markeredgewidth=0.5)
    axes[0, 0].set_title(f'(a) Step1: ROI + GT(빨강)\n{rW}×{rH}px, 불량 POI {len(bad_indices)}개')

    # (b) Mean+kσ 원본 마스크
    axes[0, 1].imshow(roi, cmap='gray', alpha=0.5)
    mask_ov = np.zeros((*raw_mask.shape, 4))
    mask_ov[raw_mask > 0] = [0, 1, 0, 0.6]
    axes[0, 1].imshow(mask_ov)
    axes[0, 1].set_title(f'(b) Step2: Mean+kσ 마스크\n'
                          f'IoU={m_raw["iou"]:.3f}, {raw_mask.sum()}px')

    # (c) 끊김 연결 후
    axes[0, 2].imshow(roi, cmap='gray', alpha=0.5)
    conn_ov = np.zeros((*connected_mask.shape, 4))
    conn_ov[connected_mask > 0] = [0, 1, 0, 0.6]
    axes[0, 2].imshow(conn_ov)
    # 연결로 추가된 픽셀 표시
    added = (connected_mask > 0) & (raw_mask == 0)
    add_ov = np.zeros((*added.shape, 4))
    add_ov[added] = [1, 1, 0, 0.8]
    axes[0, 2].imshow(add_ov)
    axes[0, 2].set_title(f'(c) Step3: 끊김 연결\n'
                          f'IoU={m_conn["iou"]:.3f}, 추가 {added.sum()}px (노랑)')

    # (d) 골격화
    axes[0, 3].imshow(roi, cmap='gray', alpha=0.4)
    # 마스크 반투명
    mask_bg = np.zeros((*connected_mask.shape, 4))
    mask_bg[connected_mask > 0] = [0.5, 0.8, 0.5, 0.3]
    axes[0, 3].imshow(mask_bg)
    # 골격: 빨간 선
    skel_ov = np.zeros((*skeleton.shape, 4))
    skel_ov[skeleton > 0] = [1, 0, 0, 1.0]
    axes[0, 3].imshow(skel_ov)
    # endpoints: 노란 점
    for ep in endpoints:
        axes[0, 3].plot(ep[0], ep[1], 'yo', markersize=8, markeredgecolor='black',
                        markeredgewidth=1.5)
    # branchpoints: 파란 점
    for bp in branchpoints:
        axes[0, 3].plot(bp[0], bp[1], 'bs', markersize=6, markeredgecolor='white',
                        markeredgewidth=1)
    axes[0, 3].set_title(f'(d) Step4: 골격화\n'
                          f'골격 {len(skel_xs)}px, '
                          f'EP {len(endpoints)}개(노랑), '
                          f'BP {len(branchpoints)}개(파랑)')

    # (e) 크랙 형상 종합 (전체 이미지 위에)
    full_rgb = np.stack([def_img / 255.0] * 3, axis=-1)
    # GT 반투명
    for ch in range(3):
        full_rgb[:, :, ch] = np.where(gt_clean,
                                       full_rgb[:, :, ch] * 0.5 + [1, 0, 0][ch] * 0.5,
                                       full_rgb[:, :, ch])
    # 검출 마스크
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[roi_y1:roi_y2, roi_x1:roi_x2] = connected_mask
    for ch in range(3):
        full_rgb[:, :, ch] = np.where(full_mask > 0,
                                       full_rgb[:, :, ch] * 0.4 + [0, 1, 0][ch] * 0.6,
                                       full_rgb[:, :, ch])

    # ROI 경계
    axes[1, 0].imshow(full_rgb)
    rect = plt.Rectangle((roi_x1, roi_y1), roi_x2 - roi_x1, roi_y2 - roi_y1,
                          linewidth=2, edgecolor='cyan', facecolor='none', linestyle='--')
    axes[1, 0].add_patch(rect)
    # Tip, Mouth 표시
    tip_g = results['tip_global']
    mouth_g = results['mouth_global']
    axes[1, 0].plot(tip_g[0], tip_g[1], 'y*', markersize=15, markeredgecolor='black',
                    markeredgewidth=1.5, label=f'Tip ({tip_g[0]},{tip_g[1]})')
    axes[1, 0].plot(mouth_g[0], mouth_g[1], 'c*', markersize=15, markeredgecolor='black',
                    markeredgewidth=1.5, label=f'Mouth ({mouth_g[0]},{mouth_g[1]})')
    axes[1, 0].plot(tip_x_gt, crack_y_gt, 'r+', markersize=15, markeredgewidth=3,
                    label=f'GT Tip ({tip_x_gt},{crack_y_gt})')
    axes[1, 0].set_xlim(0, W)
    axes[1, 0].set_ylim(H, 0)
    axes[1, 0].legend(fontsize=8, loc='upper right')
    axes[1, 0].set_title('(e) 전체 이미지: GT(빨강) + 검출(초록) + Tip')

    # (f) Opening 프로파일
    ox = results['opening_x']
    ov = results['opening_val']
    # GT opening
    gt_opening_x = []
    gt_opening_val = []
    for x in range(roi_x1, roi_x2):
        col_gt = gt_clean[:, x]
        ys_gt = np.where(col_gt)[0]
        gt_opening_x.append(x)
        gt_opening_val.append((ys_gt.max() - ys_gt.min() + 1) if len(ys_gt) >= 2 else 0)

    axes[1, 1].fill_between(gt_opening_x, gt_opening_val, alpha=0.2, color='red')
    axes[1, 1].plot(gt_opening_x, gt_opening_val, 'r-', linewidth=1.5, label='GT opening')
    axes[1, 1].fill_between(ox, ov, alpha=0.2, color='blue')
    axes[1, 1].plot(ox, ov, 'b-', linewidth=1.5, label='검출 opening')
    axes[1, 1].axvline(tip_x_gt, color='red', linestyle='--', alpha=0.5, label=f'GT tip x={tip_x_gt}')
    axes[1, 1].axvline(tip_g[0], color='blue', linestyle='--', alpha=0.5,
                        label=f'검출 tip x={tip_g[0]}')
    axes[1, 1].set_xlabel('x (global)')
    axes[1, 1].set_ylabel('Opening (px)')
    axes[1, 1].set_title('(f) Opening 프로파일: GT vs 검출')
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    # (g) 중심선 경로 + 방향
    axes[1, 2].imshow(roi, cmap='gray', alpha=0.4)
    mask_bg2 = np.zeros((*connected_mask.shape, 4))
    mask_bg2[connected_mask > 0] = [0.5, 0.8, 0.5, 0.3]
    axes[1, 2].imshow(mask_bg2)

    if len(skel_xs) > 5:
        # 골격 점
        axes[1, 2].scatter(skel_xs, skel_ys, c='red', s=1, zorder=3)
        # 선형 피팅 라인
        poly = results['polyfit']
        fit_x = np.linspace(skel_xs.min(), skel_xs.max(), 100)
        fit_y = np.polyval(poly, fit_x)
        axes[1, 2].plot(fit_x, fit_y, 'yellow', linewidth=2, linestyle='--',
                        label=f'주 방향: {results["direction_deg"]:.1f}°')

    # 중심선 (opening 중심)
    cl_y_local = results['centerline_y'] - roi_y1
    cl_x_local = results['opening_x'] - roi_x1
    valid_cl = ~np.isnan(cl_y_local)
    if np.any(valid_cl):
        axes[1, 2].plot(cl_x_local[valid_cl], cl_y_local[valid_cl],
                        'c-', linewidth=1.5, alpha=0.8, label='Opening 중심선')

    for ep in endpoints:
        axes[1, 2].plot(ep[0], ep[1], 'yo', markersize=10, markeredgecolor='black',
                        markeredgewidth=2)
    axes[1, 2].legend(fontsize=8)
    axes[1, 2].set_title(f'(g) 중심선 + 방향\n'
                          f'길이={results["crack_length_px"]:.0f}px, '
                          f'방향={results["direction_deg"]:.1f}°')

    # (h) 결과 요약 텍스트
    axes[1, 3].axis('off')
    tip_err = abs(results['tip_global'][0] - tip_x_gt)
    summary = (
        f'크랙 형상 추출 결과\n'
        f'{"=" * 35}\n\n'
        f'ROI 크기:      {rW}×{rH}px\n'
        f'불량 POI:       {len(bad_indices)}개\n\n'
        f'크랙 마스크:\n'
        f'  Mean+kσ IoU:  {m_raw["iou"]:.4f}\n'
        f'  연결후 IoU:    {m_conn["iou"]:.4f}\n\n'
        f'크랙 형상:\n'
        f'  길이:         {results["crack_length_px"]:.1f}px\n'
        f'  주 방향:      {results["direction_deg"]:.2f}°\n'
        f'  최대 opening: {results["max_opening"]}px\n'
        f'  평균 opening: {results["mean_opening"]:.1f}px\n\n'
        f'Tip 위치:\n'
        f'  GT:           ({tip_x_gt}, {crack_y_gt})\n'
        f'  검출:         {results["tip_global"]}\n'
        f'  오차:         {tip_err}px\n\n'
        f'골격 분석:\n'
        f'  골격 픽셀:    {results["n_skeleton_pixels"]}\n'
        f'  Endpoints:    {results["n_endpoints"]}\n'
    )
    axes[1, 3].text(0.05, 0.95, summary, transform=axes[1, 3].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    for ax in axes.flatten()[:7]:
        ax.set_xticks([])
        ax.set_yticks([])

    fig1.tight_layout()
    path1 = OUTPUT_DIR / 'fig1_pipeline_stages.png'
    fig1.savefig(str(path1), dpi=150, bbox_inches='tight')
    print(f"\n  → {path1}")

    # ══════════════════════════════════════════════════
    # Figure 2: Tip 근처 확대
    # ══════════════════════════════════════════════════
    print("Figure 2: Tip 확대...")

    tip_l = results['tip_local']
    tip_margin = 40
    tx1 = max(0, tip_l[0] - tip_margin)
    tx2 = min(rW, tip_l[0] + tip_margin)

    fig2, axes2 = plt.subplots(2, 3, figsize=(18, 10))
    fig2.suptitle(f'Figure 2: Tip 근처 확대 (global x={tx1+roi_x1}~{tx2+roi_x1})',
                  fontsize=16, fontweight='bold')

    tip_roi = roi[:, tx1:tx2]
    tip_gt = roi_gt[:, tx1:tx2]
    tip_raw = raw_mask[:, tx1:tx2]
    tip_conn = connected_mask[:, tx1:tx2]
    tip_skel = skeleton[:, tx1:tx2]
    tip_lx = tip_l[0] - tx1
    gt_tip_lx = tip_x_gt - roi_x1 - tx1

    # (a) 원본 + GT
    axes2[0, 0].imshow(tip_roi, cmap='gray')
    gt_tip_ov = np.zeros((*tip_gt.shape, 4))
    gt_tip_ov[tip_gt.astype(bool)] = [1, 0, 0, 0.4]
    axes2[0, 0].imshow(gt_tip_ov)
    if 0 <= gt_tip_lx < tip_roi.shape[1]:
        axes2[0, 0].axvline(gt_tip_lx, color='red', linewidth=2, linestyle='--',
                            label=f'GT tip')
    if 0 <= tip_lx < tip_roi.shape[1]:
        axes2[0, 0].axvline(tip_lx, color='yellow', linewidth=2, linestyle='--',
                            label=f'검출 tip')
    axes2[0, 0].legend(fontsize=9)
    axes2[0, 0].set_title('(a) 원본 + GT(빨강)')

    # (b) Mean+kσ 원본
    axes2[0, 1].imshow(tip_roi, cmap='gray', alpha=0.5)
    raw_tip_ov = np.zeros((*tip_raw.shape, 4))
    raw_tip_ov[tip_raw > 0] = [0, 1, 0, 0.6]
    axes2[0, 1].imshow(raw_tip_ov)
    if 0 <= gt_tip_lx < tip_roi.shape[1]:
        axes2[0, 1].axvline(gt_tip_lx, color='red', linewidth=2, linestyle='--')
    axes2[0, 1].set_title('(b) Mean+kσ 원본')

    # (c) 연결 후
    axes2[0, 2].imshow(tip_roi, cmap='gray', alpha=0.5)
    conn_tip_ov = np.zeros((*tip_conn.shape, 4))
    conn_tip_ov[tip_conn > 0] = [0, 1, 0, 0.6]
    axes2[0, 2].imshow(conn_tip_ov)
    added_tip = (tip_conn > 0) & (tip_raw == 0)
    add_tip_ov = np.zeros((*added_tip.shape, 4))
    add_tip_ov[added_tip] = [1, 1, 0, 0.8]
    axes2[0, 2].imshow(add_tip_ov)
    if 0 <= gt_tip_lx < tip_roi.shape[1]:
        axes2[0, 2].axvline(gt_tip_lx, color='red', linewidth=2, linestyle='--')
    axes2[0, 2].set_title(f'(c) 연결 후 (추가={added_tip.sum()}px 노랑)')

    # (d) 골격 + endpoint
    axes2[1, 0].imshow(tip_roi, cmap='gray', alpha=0.4)
    conn_bg = np.zeros((*tip_conn.shape, 4))
    conn_bg[tip_conn > 0] = [0.5, 0.8, 0.5, 0.3]
    axes2[1, 0].imshow(conn_bg)
    skel_tip_ov = np.zeros((*tip_skel.shape, 4))
    skel_tip_ov[tip_skel > 0] = [1, 0, 0, 1.0]
    axes2[1, 0].imshow(skel_tip_ov)
    for ep in endpoints:
        elx = ep[0] - tx1
        if 0 <= elx < tip_roi.shape[1]:
            axes2[1, 0].plot(elx, ep[1], 'yo', markersize=12,
                             markeredgecolor='black', markeredgewidth=2)
    if 0 <= gt_tip_lx < tip_roi.shape[1]:
        axes2[1, 0].axvline(gt_tip_lx, color='red', linewidth=2, linestyle='--')
    axes2[1, 0].set_title('(d) 골격(빨강) + Endpoint(노랑)')

    # (e) TP/FP/FN
    m_tip = compute_metrics(tip_gt, tip_conn)
    comp_tip = make_tp_fp_fn_overlay(tip_gt, tip_conn)
    axes2[1, 1].imshow(tip_roi, cmap='gray', alpha=0.3)
    axes2[1, 1].imshow(comp_tip, alpha=0.7)
    if 0 <= gt_tip_lx < tip_roi.shape[1]:
        axes2[1, 1].axvline(gt_tip_lx, color='yellow', linewidth=2, linestyle='--')
    axes2[1, 1].set_title(f'(e) TP/FP/FN\nIoU={m_tip["iou"]:.3f} '
                           f'P={m_tip["precision"]:.3f} R={m_tip["recall"]:.3f}')

    # (f) Opening 프로파일 (tip 근처)
    ox_local = np.arange(tx1, tx2)
    ov_local = results['opening_val'][tx1:tx2] if tx2 <= len(results['opening_val']) else []
    gt_ov_local = []
    for x in range(tx1 + roi_x1, min(tx2 + roi_x1, W)):
        col_gt = gt_clean[:, x]
        ys_gt = np.where(col_gt)[0]
        gt_ov_local.append((ys_gt.max() - ys_gt.min() + 1) if len(ys_gt) >= 2 else 0)

    if len(ov_local) > 0 and len(gt_ov_local) > 0:
        x_axis = np.arange(len(gt_ov_local)) + tx1 + roi_x1
        axes2[1, 2].plot(x_axis, gt_ov_local, 'r-', linewidth=2, label='GT')
        axes2[1, 2].plot(x_axis[:len(ov_local)], ov_local, 'b-', linewidth=2, label='검출')
        axes2[1, 2].axvline(tip_x_gt, color='red', linestyle='--', alpha=0.7)
        axes2[1, 2].axvline(results['tip_global'][0], color='blue', linestyle='--', alpha=0.7)
        axes2[1, 2].set_xlabel('x (global)')
        axes2[1, 2].set_ylabel('Opening (px)')
        axes2[1, 2].legend(fontsize=10)
        axes2[1, 2].grid(True, alpha=0.3)
    axes2[1, 2].set_title('(f) Tip 근처 Opening 프로파일')

    for ax in axes2.flatten()[:5]:
        ax.set_xticks([])
        ax.set_yticks([])

    fig2.tight_layout()
    path2 = OUTPUT_DIR / 'fig2_tip_detail.png'
    fig2.savefig(str(path2), dpi=150, bbox_inches='tight')
    print(f"  → {path2}")

    # ══════════════════════════════════════════════════
    # Figure 3: 수평 커널 크기 민감도
    # ══════════════════════════════════════════════════
    print("Figure 3: 커널 크기 민감도...")

    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Figure 3: 끊김 연결 파라미터 민감도', fontsize=16, fontweight='bold')

    # (a) 수평 커널 크기 변화
    h_kernels = list(range(1, 25, 2))
    ious_h = []
    tip_errs_h = []
    for hk in h_kernels:
        conn = step3_connect_gaps(raw_mask, h_kernel_width=hk, v_kernel_height=3)
        m = compute_metrics(roi_gt, conn)
        ious_h.append(m['iou'])
        # Tip 오차
        skel_h, sx, sy, eps, _ = step4_skeletonize_and_extract(conn)
        if len(eps) >= 1:
            tip_x_h = max(ep[0] for ep in eps) + roi_x1
        else:
            tip_x_h = 0
        tip_errs_h.append(abs(tip_x_h - tip_x_gt))

    ax_iou = axes3[0, 0]
    ax_tip = ax_iou.twinx()
    ax_iou.plot(h_kernels, ious_h, 'b-o', markersize=4, label='IoU')
    ax_tip.plot(h_kernels, tip_errs_h, 'r--s', markersize=4, label='Tip 오차 (px)')
    ax_iou.axvline(11, color='green', linestyle=':', linewidth=2, label='현재 h_kernel=11')
    ax_iou.set_xlabel('수평 커널 폭')
    ax_iou.set_ylabel('IoU', color='blue')
    ax_tip.set_ylabel('Tip 오차 (px)', color='red')
    ax_iou.set_title('(a) 수평 커널 폭 → IoU & Tip 오차')
    ax_iou.legend(loc='upper left', fontsize=8)
    ax_tip.legend(loc='upper right', fontsize=8)
    ax_iou.grid(True, alpha=0.3)

    # (b) k값 변화
    k_vals = np.arange(0.4, 2.5, 0.2)
    ious_k = []
    tip_errs_k = []
    for kv in k_vals:
        rm, _, _, _ = step2_detect_crack_mask(roi, k=kv)
        conn = step3_connect_gaps(rm, h_kernel_width=11, v_kernel_height=3)
        m = compute_metrics(roi_gt, conn)
        ious_k.append(m['iou'])
        skel_k, sx, sy, eps, _ = step4_skeletonize_and_extract(conn)
        if len(eps) >= 1:
            tip_x_k = max(ep[0] for ep in eps) + roi_x1
        else:
            tip_x_k = 0
        tip_errs_k.append(abs(tip_x_k - tip_x_gt))

    ax_iou2 = axes3[0, 1]
    ax_tip2 = ax_iou2.twinx()
    ax_iou2.plot(k_vals, ious_k, 'b-o', markersize=4, label='IoU')
    ax_tip2.plot(k_vals, tip_errs_k, 'r--s', markersize=4, label='Tip 오차 (px)')
    ax_iou2.axvline(1.0, color='green', linestyle=':', linewidth=2, label='현재 k=1.0')
    ax_iou2.set_xlabel('k 값')
    ax_iou2.set_ylabel('IoU', color='blue')
    ax_tip2.set_ylabel('Tip 오차 (px)', color='red')
    ax_iou2.set_title('(b) k값 → IoU & Tip 오차')
    ax_iou2.legend(loc='upper left', fontsize=8)
    ax_tip2.legend(loc='upper right', fontsize=8)
    ax_iou2.grid(True, alpha=0.3)

    # (c) 검출 마스크 크기별 비교 (step별)
    stages = ['Mean+kσ\n원본', '끊김\n연결']
    stage_ious = [m_raw['iou'], m_conn['iou']]
    stage_precs = [m_raw['precision'], m_conn['precision']]
    stage_recs = [m_raw['recall'], m_conn['recall']]
    stage_f1s = [m_raw['f1'], m_conn['f1']]

    x = np.arange(len(stages))
    w = 0.2
    axes3[1, 0].bar(x - 1.5 * w, stage_ious, w, color='steelblue', label='IoU')
    axes3[1, 0].bar(x - 0.5 * w, stage_f1s, w, color='goldenrod', label='F1')
    axes3[1, 0].bar(x + 0.5 * w, stage_precs, w, color='mediumseagreen', label='Precision')
    axes3[1, 0].bar(x + 1.5 * w, stage_recs, w, color='coral', label='Recall')
    axes3[1, 0].set_xticks(x)
    axes3[1, 0].set_xticklabels(stages)
    axes3[1, 0].set_ylabel('점수')
    axes3[1, 0].set_title('(c) 단계별 성능 비교')
    axes3[1, 0].legend(fontsize=8)
    axes3[1, 0].set_ylim(0, 1.1)
    axes3[1, 0].grid(True, alpha=0.3, axis='y')

    for i, (iou_v, f1_v) in enumerate(zip(stage_ious, stage_f1s)):
        axes3[1, 0].text(i - 1.5 * w, iou_v + 0.02, f'{iou_v:.3f}',
                         ha='center', fontsize=8)

    # (d) Opening 오차 히스토그램
    ov_pred = results['opening_val']
    gt_ov_arr = np.array(gt_opening_val)
    pred_ov_arr = ov_pred[:len(gt_ov_arr)]
    # 크랙이 있는 열만
    crack_cols = gt_ov_arr > 0
    if np.any(crack_cols):
        opening_error = pred_ov_arr[crack_cols] - gt_ov_arr[crack_cols]
        axes3[1, 1].hist(opening_error, bins=30, color='steelblue', edgecolor='black',
                         alpha=0.7)
        axes3[1, 1].axvline(0, color='red', linestyle='--', linewidth=2)
        axes3[1, 1].axvline(np.mean(opening_error), color='orange', linestyle='--',
                            linewidth=2, label=f'평균 오차={np.mean(opening_error):.2f}px')
        axes3[1, 1].set_xlabel('Opening 오차 (검출 - GT) [px]')
        axes3[1, 1].set_ylabel('빈도')
        axes3[1, 1].set_title(f'(d) 열별 Opening 오차 분포\n'
                               f'MAE={np.mean(np.abs(opening_error)):.2f}px, '
                               f'RMSE={np.sqrt(np.mean(opening_error**2)):.2f}px')
        axes3[1, 1].legend(fontsize=9)
    axes3[1, 1].grid(True, alpha=0.3)

    fig3.tight_layout()
    path3 = OUTPUT_DIR / 'fig3_parameter_sensitivity.png'
    fig3.savefig(str(path3), dpi=150, bbox_inches='tight')
    print(f"  → {path3}")

    # ══════════════════════════════════════════════════
    # 콘솔 최종 요약
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("크랙 형상 추출 최종 결과")
    print("=" * 60)
    print(f"크랙 길이:       {results['crack_length_px']:.1f}px")
    print(f"주 방향:         {results['direction_deg']:.2f}°")
    print(f"최대 opening:    {results['max_opening']}px")
    print(f"평균 opening:    {results['mean_opening']:.1f}px")
    print(f"Tip (검출):      {results['tip_global']}")
    print(f"Tip (GT):        ({tip_x_gt}, {crack_y_gt})")
    print(f"Tip 오차:        {abs(results['tip_global'][0] - tip_x_gt)}px")
    print(f"Mouth (검출):    {results['mouth_global']}")
    print(f"마스크 IoU:      {m_conn['iou']:.4f}")
    print(f"골격 픽셀:       {results['n_skeleton_pixels']}")
    print(f"Endpoints:       {results['n_endpoints']}")
    if np.any(crack_cols):
        print(f"Opening MAE:     {np.mean(np.abs(opening_error)):.2f}px")
        print(f"Opening RMSE:    {np.sqrt(np.mean(opening_error**2)):.2f}px")
    print(f"\n결과 저장: {OUTPUT_DIR}")
    print("=" * 60)

    plt.show()
    print("완료!")


if __name__ == '__main__':
    run_pipeline()

