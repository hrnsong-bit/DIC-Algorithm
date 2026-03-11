"""
tests/test_crack_edge_detection.py
ADSS 불량 POI 기반 ROI에서 Canny 엣지 필터링 실험
- 불량 POI + 8방위 이웃으로 ROI 구성
- Canny → 연결 성분 분석 → 길이/방향 필터링
- Mean+kσ와 비교
"""
import sys
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import ndimage
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
OUTPUT_DIR = TEST_OUTPUT_ROOT / 'output_edge_detection'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _imread_unicode(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지 로딩 실패: {path}")
    return img


def compute_iou(gt, pred):
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    tp = np.sum(gt_b & pred_b)
    fp = np.sum(~gt_b & pred_b)
    fn = np.sum(gt_b & ~pred_b)
    return tp / max(tp + fp + fn, 1), int(tp), int(fp), int(fn)


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


def make_comparison_overlay(gt, pred):
    comp = np.zeros((*gt.shape, 3), dtype=np.uint8)
    gt_b = gt.astype(bool)
    pred_b = pred.astype(bool)
    comp[gt_b & pred_b] = [0, 255, 0]
    comp[~gt_b & pred_b] = [255, 0, 0]
    comp[gt_b & ~pred_b] = [0, 0, 255]
    return comp


# ══════════════════════════════════════════════════
# 크랙 엣지 검출 알고리즘들
# ══════════════════════════════════════════════════

def detect_mean_k_sigma(img, sigma_blur=1.0, k=1.0):
    """Mean+kσ 기준 검출"""
    ksize = max(3, int(np.ceil(sigma_blur * 3)) * 2 + 1)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma_blur)
    mu = float(blurred.mean())
    std = float(blurred.std())
    thr = mu - k * std
    detected = (blurred < thr).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    detected = cv2.morphologyEx(detected, cv2.MORPH_CLOSE, kernel)
    detected = cv2.morphologyEx(detected, cv2.MORPH_OPEN, kernel)
    return detected


def detect_canny_raw(img, sigma_blur=1.0):
    """Canny 엣지 검출 (필터링 없음)"""
    ksize = max(3, int(np.ceil(sigma_blur * 3)) * 2 + 1)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma_blur)
    v = np.median(blurred)
    low = int(max(0, 0.5 * v))
    high = int(min(255, 1.0 * v))
    edges = cv2.Canny(blurred, low, high)
    return edges, low, high


def detect_canny_length_filter(img, sigma_blur=1.0, min_length=10):
    """
    Canny + 연결 성분 길이 필터링
    짧은 엣지(스펙클) 제거, 긴 엣지(크랙)만 남김
    """
    edges, low, high = detect_canny_raw(img, sigma_blur)

    # 연결 성분 분석
    labeled, n_components = ndimage.label(edges)
    filtered = np.zeros_like(edges)

    for i in range(1, n_components + 1):
        component = (labeled == i)
        n_pixels = np.sum(component)
        if n_pixels >= min_length:
            filtered[component] = 255

    return filtered, n_components


def detect_canny_length_direction_filter(img, sigma_blur=1.0, min_length=10,
                                          direction_consistency=0.6):
    """
    Canny + 길이 + 방향 일관성 필터링
    - 길이가 min_length 이상
    - 방향 일관성: 주 방향 비율이 threshold 이상
    """
    edges, low, high = detect_canny_raw(img, sigma_blur)

    # Sobel로 그라디언트 방향 계산
    ksize_s = max(3, int(np.ceil(sigma_blur * 3)) * 2 + 1)
    blurred = cv2.GaussianBlur(img, (ksize_s, ksize_s), sigma_blur)
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    angles = np.arctan2(grad_y, grad_x) * 180 / np.pi  # -180 ~ 180

    labeled, n_components = ndimage.label(edges)
    filtered = np.zeros_like(edges)
    component_info = []

    for i in range(1, n_components + 1):
        component = (labeled == i)
        n_pixels = np.sum(component)

        if n_pixels < min_length:
            component_info.append({
                'id': i, 'size': n_pixels, 'kept': False, 'reason': 'too_short'
            })
            continue

        # 방향 일관성 평가
        comp_angles = angles[component]
        # 4방향으로 양자화 (0°, 45°, 90°, 135°)
        quantized = ((comp_angles + 180) / 45).astype(int) % 4
        hist = np.bincount(quantized, minlength=4)
        dominant_ratio = hist.max() / max(n_pixels, 1)

        if dominant_ratio >= direction_consistency:
            filtered[component] = 255
            component_info.append({
                'id': i, 'size': n_pixels, 'kept': True,
                'dominant_ratio': dominant_ratio,
                'dominant_dir': int(np.argmax(hist)) * 45
            })
        else:
            component_info.append({
                'id': i, 'size': n_pixels, 'kept': False,
                'reason': f'direction={dominant_ratio:.2f}<{direction_consistency}',
                'dominant_ratio': dominant_ratio
            })

    return filtered, component_info, n_components


def detect_canny_hough_filter(img, sigma_blur=1.0, min_line_length=15,
                               max_line_gap=5, dilate_width=3):
    """
    Canny + HoughLinesP 기반 필터링
    직선 성분만 추출하여 크랙 엣지로 판단
    """
    edges, low, high = detect_canny_raw(img, sigma_blur)

    lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180,
                             threshold=10,
                             minLineLength=min_line_length,
                             maxLineGap=max_line_gap)

    line_mask = np.zeros_like(edges)
    n_lines = 0
    line_angles = []

    if lines is not None:
        n_lines = len(lines)
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_mask, (x1, y1), (x2, y2), 255, dilate_width)
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            line_angles.append(angle)

    return line_mask, n_lines, line_angles


def detect_canny_fill_mean(img, sigma_blur=1.0, min_length=10, k=1.0,
                            fill_dilate=3):
    """
    Canny 엣지 필터링 → 엣지 사이 영역을 Mean+kσ로 채우기
    1) Canny + 길이 필터로 크랙 엣지 추출
    2) 엣지를 팽창하여 크랙 근처 마스크 생성
    3) 마스크 내에서 Mean+kσ 적용하여 크랙 영역 채우기
    """
    # 1) 크랙 엣지 추출
    filtered_edges, _ = detect_canny_length_filter(img, sigma_blur, min_length)

    if filtered_edges.sum() == 0:
        return np.zeros_like(img, dtype=np.uint8), filtered_edges

    # 2) 엣지 주변 확장 (크랙 근처 ROI)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                       (fill_dilate * 2 + 1, fill_dilate * 2 + 1))
    edge_roi = cv2.dilate(filtered_edges, kernel)

    # 3) ROI 내에서 Mean+kσ
    ksize = max(3, int(np.ceil(sigma_blur * 3)) * 2 + 1)
    blurred = cv2.GaussianBlur(img, (ksize, ksize), sigma_blur)

    # ROI 내 픽셀만으로 통계 계산
    roi_pixels = blurred[edge_roi > 0]
    if len(roi_pixels) == 0:
        return np.zeros_like(img, dtype=np.uint8), filtered_edges

    mu = float(roi_pixels.mean())
    std = float(roi_pixels.std())
    thr = mu - k * std

    # ROI 내에서 임계값 이하인 픽셀 = 크랙
    detected = np.zeros_like(img, dtype=np.uint8)
    detected[(blurred < thr) & (edge_roi > 0)] = 1

    # 후처리
    kernel_m = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    detected = cv2.morphologyEx(detected, cv2.MORPH_CLOSE, kernel_m)
    detected = cv2.morphologyEx(detected, cv2.MORPH_OPEN, kernel_m)

    return detected, filtered_edges


# ══════════════════════════════════════════════════
# 메인 시각화
# ══════════════════════════════════════════════════

def run_visualization():
    # ── 데이터 로드 ──
    def_img = _imread_unicode(DATA_DIR / 'deformed.tiff')
    ref_img = _imread_unicode(DATA_DIR / 'reference.tiff')
    gt_mask = np.load(str(DATA_DIR / 'crack_mask.npy'))
    H, W = def_img.shape

    gt_clean = gt_mask.copy()

    subset_size = 25
    M = subset_size // 2
    spacing = 11
    crack_y = 250
    zncc_threshold = 0.9

    # ── DIC 실행하여 불량 POI 식별 ──
    from speckle.core.initial_guess import compute_fft_cc
    from speckle.core.optimization import compute_icgn

    print("=" * 60)
    print("1단계: FFT-CC + IC-GN")
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

    bad_mask = ~icgn_result.valid_mask
    bad_indices = np.where(bad_mask)[0]
    n_bad = len(bad_indices)
    print(f"불량 POI: {n_bad}개")

    poi_x = icgn_result.points_x
    poi_y = icgn_result.points_y

    # ── 불량 POI + 8방위 이웃으로 ROI 구성 ──
    bad_coords = set()
    for idx in bad_indices:
        px, py = int(poi_x[idx]), int(poi_y[idx])
        bad_coords.add((px, py))
        # 8방위 이웃 추가
        for dx in [-spacing, 0, spacing]:
            for dy in [-spacing, 0, spacing]:
                nx, ny = px + dx, py + dy
                if 0 <= nx < W and 0 <= ny < H:
                    bad_coords.add((nx, ny))

    # ROI 바운딩 박스 계산
    all_x = [c[0] for c in bad_coords]
    all_y = [c[1] for c in bad_coords]
    roi_x1 = max(0, min(all_x) - M - 5)
    roi_x2 = min(W, max(all_x) + M + 6)
    roi_y1 = max(0, min(all_y) - M - 5)
    roi_y2 = min(H, max(all_y) + M + 6)

    roi_def = def_img[roi_y1:roi_y2, roi_x1:roi_x2]
    roi_gt = gt_clean[roi_y1:roi_y2, roi_x1:roi_x2]

    print(f"ROI: ({roi_x1},{roi_y1}) ~ ({roi_x2},{roi_y2}), "
          f"크기: {roi_x2 - roi_x1}×{roi_y2 - roi_y1}")
    print(f"불량+이웃 POI 수: {len(bad_coords)}")

    # ══════════════════════════════════════════════════
    # Figure 1: Canny 원본 vs 필터링 단계별 시각화
    # ══════════════════════════════════════════════════
    print("\nFigure 1 생성: Canny 필터링 단계별...")
    fig1, axes1 = plt.subplots(2, 4, figsize=(20, 10))
    fig1.suptitle('Figure 1: Canny 엣지 필터링 단계별 결과 (불량 POI ROI)',
                  fontsize=16, fontweight='bold')

    # (a) 원본 ROI + GT
    axes1[0, 0].imshow(roi_def, cmap='gray')
    gt_ov = np.zeros((*roi_gt.shape, 4))
    gt_ov[roi_gt.astype(bool)] = [1, 0, 0, 0.4]
    axes1[0, 0].imshow(gt_ov)
    # 불량 POI 위치 표시
    for px, py in bad_coords:
        lx, ly = px - roi_x1, py - roi_y1
        if 0 <= lx < roi_def.shape[1] and 0 <= ly < roi_def.shape[0]:
            axes1[0, 0].plot(lx, ly, 'c+', markersize=3, markeredgewidth=0.5)
    axes1[0, 0].set_title(f'(a) ROI + GT(빨강) + POI(+)\n'
                           f'{roi_x2-roi_x1}×{roi_y2-roi_y1}px, '
                           f'GT={roi_gt.sum()}px')

    # (b) Canny 원본 (필터링 없음)
    canny_raw, canny_low, canny_high = detect_canny_raw(roi_def)
    axes1[0, 1].imshow(canny_raw, cmap='gray')
    n_edge_pixels = np.sum(canny_raw > 0)
    axes1[0, 1].set_title(f'(b) Canny 원본\nlow={canny_low}, high={canny_high}\n'
                           f'엣지 픽셀: {n_edge_pixels}')

    # (c) 길이 필터링 (min_length 변화)
    for min_len, ax, label in [(5, axes1[0, 2], 'c'), (15, axes1[0, 3], 'd')]:
        filtered, n_comp = detect_canny_length_filter(roi_def, min_length=min_len)
        # 필터링 후 연결 성분 재분석
        labeled_f, n_kept = ndimage.label(filtered)
        ax.imshow(filtered, cmap='gray')
        ax.set_title(f'({label}) 길이 필터 ≥{min_len}px\n'
                     f'원본 {n_comp}개 → {n_kept}개 성분\n'
                     f'엣지 픽셀: {np.sum(filtered > 0)}')

    # (e) 길이+방향 필터링
    filtered_dir, comp_info, n_total = detect_canny_length_direction_filter(
        roi_def, min_length=10, direction_consistency=0.5)
    n_kept_dir = sum(1 for c in comp_info if c['kept'])
    axes1[1, 0].imshow(filtered_dir, cmap='gray')
    axes1[1, 0].set_title(f'(e) 길이≥10 + 방향일관성≥0.5\n'
                           f'{n_total}개 → {n_kept_dir}개 성분')

    # (f) Hough 직선 필터링
    hough_mask, n_lines, line_angles = detect_canny_hough_filter(
        roi_def, min_line_length=15, max_line_gap=5)
    axes1[1, 1].imshow(roi_def, cmap='gray', alpha=0.5)
    hough_ov = np.zeros((*hough_mask.shape, 4))
    hough_ov[hough_mask > 0] = [0, 1, 0, 0.7]
    axes1[1, 1].imshow(hough_ov)
    axes1[1, 1].set_title(f'(f) HoughLinesP\n'
                           f'{n_lines}개 직선 검출')

    # (g) Canny+Mean 조합
    canny_mean_det, canny_edges_used = detect_canny_fill_mean(
        roi_def, min_length=10, k=1.0, fill_dilate=5)
    axes1[1, 2].imshow(roi_def, cmap='gray', alpha=0.5)
    det_ov = np.zeros((*canny_mean_det.shape, 4))
    det_ov[canny_mean_det > 0] = [0, 1, 0, 0.6]
    edge_ov = np.zeros((*canny_edges_used.shape, 4))
    edge_ov[canny_edges_used > 0] = [1, 1, 0, 0.8]
    axes1[1, 2].imshow(det_ov)
    axes1[1, 2].imshow(edge_ov)
    m_cm = compute_metrics(roi_gt, canny_mean_det)
    axes1[1, 2].set_title(f'(g) Canny→길이필터→Mean+kσ 채움\n'
                           f'IoU={m_cm["iou"]:.3f}')

    # (h) Mean+kσ 단독 (비교 기준)
    mean_det = detect_mean_k_sigma(roi_def, k=1.0)
    axes1[1, 3].imshow(roi_def, cmap='gray', alpha=0.5)
    mean_ov = np.zeros((*mean_det.shape, 4))
    mean_ov[mean_det > 0] = [0, 1, 0, 0.6]
    axes1[1, 3].imshow(mean_ov)
    m_mean = compute_metrics(roi_gt, mean_det)
    axes1[1, 3].set_title(f'(h) Mean+kσ 단독 (비교 기준)\n'
                           f'IoU={m_mean["iou"]:.3f}')

    for ax in axes1.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    fig1.tight_layout()
    path1 = OUTPUT_DIR / 'fig1_canny_filtering_stages.png'
    fig1.savefig(str(path1), dpi=150, bbox_inches='tight')
    print(f"  → {path1}")

    # ══════════════════════════════════════════════════
    # Figure 2: 각 방법별 TP/FP/FN 비교
    # ══════════════════════════════════════════════════
    print("Figure 2 생성: 방법별 정량 비교...")

    methods = {
        'Canny 원본': (canny_raw > 0).astype(np.uint8),
        'Canny 길이≥10': (detect_canny_length_filter(roi_def, min_length=10)[0] > 0).astype(np.uint8),
        'Canny 길이+방향': (filtered_dir > 0).astype(np.uint8),
        'Hough 직선': (hough_mask > 0).astype(np.uint8),
        'Canny→Mean 채움': canny_mean_det,
        'Mean+kσ 단독': mean_det,
    }

    fig2, axes2 = plt.subplots(2, len(methods), figsize=(4 * len(methods), 8))
    fig2.suptitle('Figure 2: 크랙 검출 방법별 TP/FP/FN 비교', fontsize=16, fontweight='bold')

    all_metrics = {}
    for col, (name, det) in enumerate(methods.items()):
        m = compute_metrics(roi_gt, det)
        all_metrics[name] = m

        # Row 0: 검출 결과 오버레이
        axes2[0, col].imshow(roi_def, cmap='gray', alpha=0.5)
        det_ov = np.zeros((*det.shape, 4))
        det_ov[det.astype(bool)] = [0, 1, 0, 0.6]
        axes2[0, col].imshow(det_ov)
        axes2[0, col].set_title(f'{name}\n검출: {det.sum()}px', fontsize=9)

        # Row 1: TP/FP/FN
        comp = make_comparison_overlay(roi_gt, det)
        axes2[1, col].imshow(roi_def, cmap='gray', alpha=0.3)
        axes2[1, col].imshow(comp, alpha=0.7)
        axes2[1, col].set_title(
            f'IoU={m["iou"]:.3f}\n'
            f'P={m["precision"]:.3f} R={m["recall"]:.3f} F1={m["f1"]:.3f}',
            fontsize=9)

    for ax in axes2.flatten():
        ax.set_xticks([])
        ax.set_yticks([])

    legend_patches = [
        mpatches.Patch(color='green', alpha=0.7, label='TP'),
        mpatches.Patch(color='red', alpha=0.7, label='FP'),
        mpatches.Patch(color='blue', alpha=0.7, label='FN'),
    ]
    fig2.legend(handles=legend_patches, loc='lower center', ncol=3, fontsize=11)
    fig2.tight_layout(rect=[0, 0.04, 1, 0.95])
    path2 = OUTPUT_DIR / 'fig2_method_comparison.png'
    fig2.savefig(str(path2), dpi=150, bbox_inches='tight')
    print(f"  → {path2}")

    # ══════════════════════════════════════════════════
    # Figure 3: 연결 성분 분석 상세
    # ══════════════════════════════════════════════════
    print("Figure 3 생성: 연결 성분 분석...")

    canny_raw_edges, _, _ = detect_canny_raw(roi_def)
    labeled_raw, n_raw = ndimage.label(canny_raw_edges)

    # 각 성분의 크기 계산
    component_sizes = []
    for i in range(1, n_raw + 1):
        size = np.sum(labeled_raw == i)
        component_sizes.append(size)
    component_sizes = np.array(component_sizes)

    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Figure 3: Canny 연결 성분 분석', fontsize=16, fontweight='bold')

    # (a) 성분 크기 히스토그램
    axes3[0, 0].hist(component_sizes, bins=50, color='steelblue', edgecolor='black',
                     alpha=0.7)
    axes3[0, 0].axvline(10, color='red', linestyle='--', linewidth=2,
                        label='min_length=10')
    axes3[0, 0].axvline(15, color='orange', linestyle='--', linewidth=2,
                        label='min_length=15')
    axes3[0, 0].set_xlabel('성분 크기 (픽셀)')
    axes3[0, 0].set_ylabel('빈도')
    axes3[0, 0].set_title(f'(a) 연결 성분 크기 분포 (총 {n_raw}개)\n'
                           f'중앙값={np.median(component_sizes):.0f}, '
                           f'최대={component_sizes.max()}')
    axes3[0, 0].legend()
    axes3[0, 0].set_yscale('log')

    # (b) min_length에 따른 성능 변화
    min_lengths = list(range(1, 40, 2))
    ious_len = []
    n_comps_len = []
    for ml in min_lengths:
        filt, nc = detect_canny_length_filter(roi_def, min_length=ml)
        # 팽창하여 영역화
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        filt_dilated = cv2.dilate(filt, kernel, iterations=2)
        iou, _, _, _ = compute_iou(roi_gt, filt_dilated)
        ious_len.append(iou)
        labeled_f, nk = ndimage.label(filt)
        n_comps_len.append(nk)

    ax_iou = axes3[0, 1]
    ax_comp = ax_iou.twinx()
    ax_iou.plot(min_lengths, ious_len, 'b-o', markersize=3, label='IoU (dilate=2)')
    ax_comp.plot(min_lengths, n_comps_len, 'r--s', markersize=3, label='남은 성분 수')
    ax_iou.set_xlabel('min_length')
    ax_iou.set_ylabel('IoU', color='blue')
    ax_comp.set_ylabel('남은 성분 수', color='red')
    ax_iou.set_title('(b) min_length에 따른 IoU & 성분 수')
    ax_iou.legend(loc='upper left', fontsize=8)
    ax_comp.legend(loc='upper right', fontsize=8)
    ax_iou.grid(True, alpha=0.3)

    # (c) 큰 성분 vs 작은 성분 시각화
    large_mask = np.zeros_like(canny_raw_edges)
    small_mask = np.zeros_like(canny_raw_edges)
    threshold_size = 10
    for i in range(1, n_raw + 1):
        comp = (labeled_raw == i)
        if component_sizes[i - 1] >= threshold_size:
            large_mask[comp] = 255
        else:
            small_mask[comp] = 255

    rgb_comp = np.stack([roi_def / 255.0] * 3, axis=-1)
    # 큰 성분: 초록, 작은 성분: 빨강
    for c in range(3):
        rgb_comp[:, :, c] = np.where(large_mask > 0,
                                      rgb_comp[:, :, c] * 0.3 + [0, 1, 0][c] * 0.7,
                                      rgb_comp[:, :, c])
        rgb_comp[:, :, c] = np.where(small_mask > 0,
                                      rgb_comp[:, :, c] * 0.3 + [1, 0, 0][c] * 0.7,
                                      rgb_comp[:, :, c])

    axes3[1, 0].imshow(rgb_comp)
    n_large = np.sum(component_sizes >= threshold_size)
    n_small = np.sum(component_sizes < threshold_size)
    axes3[1, 0].set_title(f'(c) 성분 분류 (기준={threshold_size}px)\n'
                           f'큰 성분(초록): {n_large}개, '
                           f'작은 성분(빨강): {n_small}개')
    axes3[1, 0].set_xticks([])
    axes3[1, 0].set_yticks([])

    # (d) 종합 성능 비교 바 차트
    method_names = list(all_metrics.keys())
    ious_bar = [all_metrics[n]['iou'] for n in method_names]
    f1s_bar = [all_metrics[n]['f1'] for n in method_names]
    precs_bar = [all_metrics[n]['precision'] for n in method_names]
    recs_bar = [all_metrics[n]['recall'] for n in method_names]

    x = np.arange(len(method_names))
    w = 0.2
    axes3[1, 1].bar(x - 1.5 * w, ious_bar, w, color='steelblue', label='IoU')
    axes3[1, 1].bar(x - 0.5 * w, f1s_bar, w, color='goldenrod', label='F1')
    axes3[1, 1].bar(x + 0.5 * w, precs_bar, w, color='mediumseagreen', label='Precision')
    axes3[1, 1].bar(x + 1.5 * w, recs_bar, w, color='coral', label='Recall')

    axes3[1, 1].set_xticks(x)
    axes3[1, 1].set_xticklabels(method_names, rotation=25, ha='right', fontsize=8)
    axes3[1, 1].set_ylabel('점수')
    axes3[1, 1].set_title('(d) 전체 방법 종합 비교')
    axes3[1, 1].legend(fontsize=8, ncol=2)
    axes3[1, 1].set_ylim(0, 1.15)
    axes3[1, 1].grid(True, alpha=0.3, axis='y')

    for i, (iou_v, f1_v) in enumerate(zip(ious_bar, f1s_bar)):
        axes3[1, 1].text(i - 1.5 * w, iou_v + 0.02, f'{iou_v:.2f}',
                         ha='center', fontsize=6, rotation=90)

    fig3.tight_layout()
    path3 = OUTPUT_DIR / 'fig3_component_analysis.png'
    fig3.savefig(str(path3), dpi=150, bbox_inches='tight')
    print(f"  → {path3}")

    # ══════════════════════════════════════════════════
    # Figure 4: Canny→Mean 조합 파라미터 민감도
    # ══════════════════════════════════════════════════
    print("Figure 4 생성: Canny→Mean 조합 파라미터 민감도...")

    fig4, axes4 = plt.subplots(1, 3, figsize=(18, 5))
    fig4.suptitle('Figure 4: Canny→Mean 조합 파라미터 민감도', fontsize=16, fontweight='bold')

    # (a) min_length 변화
    mls = list(range(3, 30, 2))
    ious_ml = []
    for ml in mls:
        det, _ = detect_canny_fill_mean(roi_def, min_length=ml, k=1.0, fill_dilate=5)
        iou, _, _, _ = compute_iou(roi_gt, det)
        ious_ml.append(iou)
    axes4[0].plot(mls, ious_ml, 'b-o', markersize=4)
    axes4[0].axhline(m_mean['iou'], color='red', linestyle='--',
                     label=f'Mean+kσ 단독={m_mean["iou"]:.3f}')
    best_ml = mls[np.argmax(ious_ml)]
    axes4[0].axvline(best_ml, color='orange', linestyle='--',
                     label=f'최적 min_len={best_ml} (IoU={max(ious_ml):.3f})')
    axes4[0].set_xlabel('min_length')
    axes4[0].set_ylabel('IoU')
    axes4[0].set_title('(a) Canny 길이 필터 min_length')
    axes4[0].legend(fontsize=8)
    axes4[0].grid(True, alpha=0.3)

    # (b) fill_dilate 변화
    dilates = list(range(1, 15))
    ious_fd = []
    for fd in dilates:
        det, _ = detect_canny_fill_mean(roi_def, min_length=10, k=1.0, fill_dilate=fd)
        iou, _, _, _ = compute_iou(roi_gt, det)
        ious_fd.append(iou)
    axes4[1].plot(dilates, ious_fd, 'b-o', markersize=4)
    axes4[1].axhline(m_mean['iou'], color='red', linestyle='--',
                     label=f'Mean+kσ 단독={m_mean["iou"]:.3f}')
    best_fd = dilates[np.argmax(ious_fd)]
    axes4[1].axvline(best_fd, color='orange', linestyle='--',
                     label=f'최적 fill_dilate={best_fd} (IoU={max(ious_fd):.3f})')
    axes4[1].set_xlabel('fill_dilate (px)')
    axes4[1].set_ylabel('IoU')
    axes4[1].set_title('(b) 엣지 주변 확장 범위')
    axes4[1].legend(fontsize=8)
    axes4[1].grid(True, alpha=0.3)

    # (c) k값 변화 (Canny→Mean vs Mean 단독)
    k_vals = np.arange(0.2, 3.1, 0.2)
    ious_cm_k = []
    ious_m_k = []
    for kv in k_vals:
        det_cm, _ = detect_canny_fill_mean(roi_def, min_length=10, k=kv, fill_dilate=5)
        iou_cm, _, _, _ = compute_iou(roi_gt, det_cm)
        ious_cm_k.append(iou_cm)

        det_m = detect_mean_k_sigma(roi_def, k=kv)
        iou_m, _, _, _ = compute_iou(roi_gt, det_m)
        ious_m_k.append(iou_m)

    axes4[2].plot(k_vals, ious_cm_k, 'b-o', markersize=4, label='Canny→Mean')
    axes4[2].plot(k_vals, ious_m_k, 'r--s', markersize=4, label='Mean+kσ 단독')
    axes4[2].set_xlabel('k 값')
    axes4[2].set_ylabel('IoU')
    axes4[2].set_title('(c) k값 변화: Canny→Mean vs Mean 단독')
    axes4[2].legend(fontsize=9)
    axes4[2].grid(True, alpha=0.3)

    fig4.tight_layout()
    path4 = OUTPUT_DIR / 'fig4_canny_mean_sensitivity.png'
    fig4.savefig(str(path4), dpi=150, bbox_inches='tight')
    print(f"  → {path4}")

    # ══════════════════════════════════════════════════
    # 콘솔 요약
    # ══════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("크랙 검출 성능 비교 (불량 POI ROI)")
    print("=" * 60)
    print(f"{'방법':<20} {'IoU':>7} {'Prec':>7} {'Recall':>7} {'F1':>7} {'검출px':>8}")
    print("-" * 60)
    for name, m in all_metrics.items():
        det = methods[name]
        print(f"{name:<20} {m['iou']:>7.4f} {m['precision']:>7.4f} "
              f"{m['recall']:>7.4f} {m['f1']:>7.4f} {det.sum():>8d}")
    print("=" * 60)
    print(f"\n결과 저장: {OUTPUT_DIR}")

    plt.show()
    print("분석 완료!")


if __name__ == '__main__':
    run_visualization()

