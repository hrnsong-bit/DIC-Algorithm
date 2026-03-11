"""
크랙 검출 시각화: k값 변화에 따른 Mean+k·σ 검출 품질 비교
"""
import sys
import numpy as np
import cv2
from pathlib import Path
import matplotlib.pyplot as plt
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
OUTPUT_DIR = TEST_OUTPUT_ROOT / 'output_crack_k_sweep_viz'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _imread_unicode(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지 로딩 실패: {path}")
    return img


def visualize_k_sweep():
    def_img = _imread_unicode(DATA_DIR / 'deformed.tiff')
    gt_mask = np.load(str(DATA_DIR / 'crack_mask.npy'))
    H, W = def_img.shape

    gt_clean = gt_mask.copy()
    gt_clean[:220, :] = False
    gt_clean[280:, :] = False

    subset_size = 21
    M = subset_size // 2
    sigma_blur = 1.0
    ksize = max(3, int(np.ceil(sigma_blur * 3)) * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    tip_x_gt = 250
    crack_y = 250

    k_values = [0.5, 0.8, 1.0, 1.2, 1.5, 2.0]

    # ============================
    # Figure 1: 전체 ROI에서 k값별 비교
    # ============================
    roi_y_min, roi_y_max = 230, 270
    roi_x_min, roi_x_max = 0, 280
    roi_def = def_img[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
    roi_gt = gt_clean[roi_y_min:roi_y_max, roi_x_min:roi_x_max]

    fig1, axes1 = plt.subplots(len(k_values), 3, figsize=(18, 3.5 * len(k_values)))
    fig1.suptitle('전체 ROI: k값별 Mean+k·σ 검출 비교 (σ=1.0)', fontsize=14, y=1.0)

    for row, k in enumerate(k_values):
        blurred = cv2.GaussianBlur(roi_def, (ksize, ksize), sigma_blur)
        mu = float(blurred.mean())
        std = float(blurred.std())
        thr = mu - k * std
        detected = (blurred < thr).astype(np.uint8)
        detected = cv2.morphologyEx(detected, cv2.MORPH_CLOSE, kernel)
        detected = cv2.morphologyEx(detected, cv2.MORPH_OPEN, kernel)

        tp = roi_gt & (detected > 0)
        fp = ~roi_gt & (detected > 0)
        fn = roi_gt & (detected == 0)
        iou = tp.sum() / max(tp.sum() + fp.sum() + fn.sum(), 1)
        prec = tp.sum() / max(tp.sum() + fp.sum(), 1)
        rec = tp.sum() / max(tp.sum() + fn.sum(), 1)

        # Col 0: 검출 결과 오버레이
        axes1[row, 0].imshow(roi_def, cmap='gray', aspect='equal')
        det_ov = np.zeros((*detected.shape, 4))
        det_ov[detected > 0, :] = [0, 1, 0, 0.5]
        axes1[row, 0].imshow(det_ov, aspect='equal')
        axes1[row, 0].set_title(f'k={k} (thr={thr:.0f}, 검출={detected.sum()}px)', fontsize=10)
        axes1[row, 0].axvline(x=tip_x_gt - roi_x_min, color='r', linewidth=0.5, linestyle='--')

        # Col 1: TP/FP/FN
        comp = np.zeros((*roi_gt.shape, 3), dtype=np.uint8)
        comp[tp] = [0, 255, 0]
        comp[fp] = [255, 0, 0]
        comp[fn] = [0, 0, 255]
        axes1[row, 1].imshow(roi_def, cmap='gray', aspect='equal', alpha=0.3)
        axes1[row, 1].imshow(comp, aspect='equal', alpha=0.7)
        axes1[row, 1].set_title(f'TP={tp.sum()} FP={fp.sum()} FN={fn.sum()}', fontsize=10)
        axes1[row, 1].axvline(x=tip_x_gt - roi_x_min, color='yellow', linewidth=0.5, linestyle='--')

        # Col 2: 메트릭 텍스트
        axes1[row, 2].axis('off')
        text = (f'k = {k}\n'
                f'Threshold = {thr:.1f}\n'
                f'IoU = {iou:.3f}\n'
                f'Precision = {prec:.3f}\n'
                f'Recall = {rec:.3f}\n'
                f'TP = {tp.sum()}\n'
                f'FP = {fp.sum()}\n'
                f'FN = {fn.sum()}\n'
                f'검출 픽셀 = {detected.sum()}')
        axes1[row, 2].text(0.1, 0.5, text, fontsize=12, verticalalignment='center',
                           fontfamily='monospace',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    save_path1 = OUTPUT_DIR / 'crack_detection_k_sweep_roi.png'
    fig1.savefig(str(save_path1), dpi=150, bbox_inches='tight')
    print(f"저장: {save_path1}")

    # ============================
    # Figure 2: 대표 POI subset에서 k값별 비교
    # ============================
    selected_pois = [
        (41, 251, "opening=10px"),
        (181, 251, "opening=4px"),
        (241, 251, "opening=2px (마지막 불량)"),
    ]

    fig2, axes2 = plt.subplots(len(selected_pois), len(k_values) + 1,
                                figsize=(3 * (len(k_values) + 1), 3.5 * len(selected_pois)))
    fig2.suptitle('POI subset: k값별 검출 비교 (σ=1.0)', fontsize=14, y=1.0)

    for row, (px, py, desc) in enumerate(selected_pois):
        y_min = py - M
        y_max = py + M + 1
        x_min = px - M
        x_max = px + M + 1

        patch = def_img[y_min:y_max, x_min:x_max]
        gt_patch = gt_clean[y_min:y_max, x_min:x_max]

        tip_local_x = tip_x_gt - x_min
        tip_local_y = crack_y - y_min

        # Col 0: 원본 + GT
        axes2[row, 0].imshow(patch, cmap='gray', vmin=0, vmax=255)
        gt_ov = np.zeros((*gt_patch.shape, 4))
        gt_ov[gt_patch, :] = [1, 0, 0, 0.5]
        axes2[row, 0].imshow(gt_ov)
        axes2[row, 0].set_title(f'{desc}\nGT={gt_patch.sum()}px', fontsize=9)
        if 0 <= tip_local_x < subset_size:
            axes2[row, 0].plot(tip_local_x, tip_local_y, 'r+', markersize=8, markeredgewidth=2)

        for col, k in enumerate(k_values):
            blurred_p = cv2.GaussianBlur(patch, (ksize, ksize), sigma_blur)
            mu_p = float(blurred_p.mean())
            std_p = float(blurred_p.std())
            thr_p = mu_p - k * std_p
            det_p = (blurred_p < thr_p).astype(np.uint8)
            det_p = cv2.morphologyEx(det_p, cv2.MORPH_CLOSE, kernel)
            det_p = cv2.morphologyEx(det_p, cv2.MORPH_OPEN, kernel)

            tp_p = np.logical_and(gt_patch, det_p > 0).sum()
            fp_p = np.logical_and(~gt_patch, det_p > 0).sum()
            fn_p = np.logical_and(gt_patch, det_p == 0).sum()
            iou_p = tp_p / max(tp_p + fp_p + fn_p, 1)

            comp_p = np.zeros((*gt_patch.shape, 3), dtype=np.uint8)
            comp_p[gt_patch & (det_p > 0)] = [0, 255, 0]
            comp_p[~gt_patch & (det_p > 0)] = [255, 0, 0]
            comp_p[gt_patch & (det_p == 0)] = [0, 0, 255]

            axes2[row, col + 1].imshow(patch, cmap='gray', vmin=0, vmax=255, alpha=0.3)
            axes2[row, col + 1].imshow(comp_p, alpha=0.7)
            axes2[row, col + 1].set_title(f'k={k}\nIoU={iou_p:.3f} FP={fp_p}', fontsize=9)
            if 0 <= tip_local_x < subset_size:
                axes2[row, col + 1].plot(tip_local_x, tip_local_y, 'r+',
                                          markersize=8, markeredgewidth=2)

    for ax_row in axes2:
        for ax in ax_row:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    save_path2 = OUTPUT_DIR / 'crack_detection_k_sweep_subsets.png'
    fig2.savefig(str(save_path2), dpi=150, bbox_inches='tight')
    print(f"저장: {save_path2}")

    # ============================
    # Figure 3: k값별 IoU/Precision/Recall 그래프
    # ============================
    k_range = np.arange(0.3, 3.1, 0.1)

    # 전체 ROI 메트릭
    roi_ious, roi_precs, roi_recs = [], [], []
    for k in k_range:
        blurred = cv2.GaussianBlur(roi_def, (ksize, ksize), sigma_blur)
        mu = float(blurred.mean())
        std = float(blurred.std())
        thr = mu - k * std
        detected = (blurred < thr).astype(np.uint8)
        detected = cv2.morphologyEx(detected, cv2.MORPH_CLOSE, kernel)
        detected = cv2.morphologyEx(detected, cv2.MORPH_OPEN, kernel)

        tp = (roi_gt & (detected > 0)).sum()
        fp = (~roi_gt & (detected > 0)).sum()
        fn = (roi_gt & (detected == 0)).sum()
        roi_ious.append(tp / max(tp + fp + fn, 1))
        roi_precs.append(tp / max(tp + fp, 1))
        roi_recs.append(tp / max(tp + fn, 1))

    # POI별 메트릭
    poi_metrics = {}
    for px, py, desc in selected_pois:
        y_min = py - M
        y_max = py + M + 1
        x_min = px - M
        x_max = px + M + 1
        patch = def_img[y_min:y_max, x_min:x_max]
        gt_patch = gt_clean[y_min:y_max, x_min:x_max]

        ious, precs, recs = [], [], []
        for k in k_range:
            blurred_p = cv2.GaussianBlur(patch, (ksize, ksize), sigma_blur)
            mu_p = float(blurred_p.mean())
            std_p = float(blurred_p.std())
            thr_p = mu_p - k * std_p
            det_p = (blurred_p < thr_p).astype(np.uint8)
            det_p = cv2.morphologyEx(det_p, cv2.MORPH_CLOSE, kernel)
            det_p = cv2.morphologyEx(det_p, cv2.MORPH_OPEN, kernel)

            tp = np.logical_and(gt_patch, det_p > 0).sum()
            fp = np.logical_and(~gt_patch, det_p > 0).sum()
            fn = np.logical_and(gt_patch, det_p == 0).sum()
            ious.append(tp / max(tp + fp + fn, 1))
            precs.append(tp / max(tp + fp, 1))
            recs.append(tp / max(tp + fn, 1))
        poi_metrics[desc] = (ious, precs, recs)

    fig3, axes3 = plt.subplots(1, 3, figsize=(18, 5))
    fig3.suptitle('k값에 따른 검출 성능 변화 (σ=1.0)', fontsize=14)

    # IoU
    axes3[0].plot(k_range, roi_ious, 'k-', linewidth=2, label='전체 ROI')
    for desc, (ious, _, _) in poi_metrics.items():
        axes3[0].plot(k_range, ious, '--', linewidth=1.5, label=desc)
    axes3[0].set_xlabel('k')
    axes3[0].set_ylabel('IoU')
    axes3[0].set_title('IoU vs k')
    axes3[0].legend(fontsize=8)
    axes3[0].grid(True, alpha=0.3)
    axes3[0].set_xlim(0.3, 3.0)

    # Precision
    axes3[1].plot(k_range, roi_precs, 'k-', linewidth=2, label='전체 ROI')
    for desc, (_, precs, _) in poi_metrics.items():
        axes3[1].plot(k_range, precs, '--', linewidth=1.5, label=desc)
    axes3[1].set_xlabel('k')
    axes3[1].set_ylabel('Precision')
    axes3[1].set_title('Precision vs k')
    axes3[1].legend(fontsize=8)
    axes3[1].grid(True, alpha=0.3)
    axes3[1].set_xlim(0.3, 3.0)

    # Recall
    axes3[2].plot(k_range, roi_recs, 'k-', linewidth=2, label='전체 ROI')
    for desc, (_, _, recs) in poi_metrics.items():
        axes3[2].plot(k_range, recs, '--', linewidth=1.5, label=desc)
    axes3[2].set_xlabel('k')
    axes3[2].set_ylabel('Recall')
    axes3[2].set_title('Recall vs k')
    axes3[2].legend(fontsize=8)
    axes3[2].grid(True, alpha=0.3)
    axes3[2].set_xlim(0.3, 3.0)

    plt.tight_layout()
    save_path3 = OUTPUT_DIR / 'crack_detection_k_sweep_metrics.png'
    fig3.savefig(str(save_path3), dpi=150, bbox_inches='tight')
    print(f"저장: {save_path3}")

    print("\n시각화 완료!")
    plt.show()


if __name__ == '__main__':
    visualize_k_sweep()

