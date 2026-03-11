# ============================================================
# TEST 5: 실제 파이프라인 조건 – 불량 POI별 로컬 패치 검출 성능
# ============================================================
import numpy as np
import cv2
from pathlib import Path

def _imread_unicode(path):
    """한글 경로 대응 이미지 로딩"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지 로딩 실패: {path}")
    return img


def test_local_patch_detection():
    """
    불량 POI + 8방위 이웃 기반 ROI에서의 크랙 검출 성능을 평가.
    패치 크기를 변화시키며 최적 ROI 범위를 탐색한다.
    """
    DATA_DIR = Path(r'C:\Users\HP\OneDrive\바탕 화면\DIC Algorithm\synthetic_crack_data')
    
    image = _imread_unicode(DATA_DIR / 'deformed.tiff')
    gt_mask = np.load(str(DATA_DIR / 'crack_mask.npy'))
    H, W = image.shape
    
    # --- 파라미터 ---
    subset_size = 21
    M = subset_size // 2  # 10
    spacing = 10
    crack_y = 250  # 크랙 중심 y좌표
    
    # GT 클리닝: y=250 ± 30 범위만 유효
    gt_clean = gt_mask.copy()
    gt_clean[:220, :] = False
    gt_clean[280:, :] = False
    
    # 크랙 위의 불량 POI 위치 시뮬레이션
    # y=250 라인의 POI들 (y=250, x=15,25,...,245)
    bad_poi_xs = np.arange(M, 250, spacing)  # crack tip(250) 이전까지
    bad_poi_y = crack_y
    
    print("=" * 70)
    print("TEST 5: 불량 POI별 로컬 패치 크랙 검출 성능")
    print("=" * 70)
    print(f"Subset size: {subset_size}, Spacing: {spacing}, M: {M}")
    print(f"불량 POI 수: {len(bad_poi_xs)}, 크랙 y={crack_y}")
    print()
    
    # --- ROI 크기 후보 ---
    # expand_pixels: POI 중심에서 상하로 확장하는 픽셀 수
    # 1×spacing+M = 20px (8방위 이웃 1층 + subset 반경)
    # 2×spacing+M = 30px (이웃 2층 + subset 반경)
    # 3×spacing+M = 40px (이웃 3층 + subset 반경)
    roi_configs = [
        ("1이웃 (subset만)", M),                    # ±10 → 21px
        ("1이웃 + subset",   spacing + M),           # ±20 → 41px
        ("2이웃 + subset",   2 * spacing + M),       # ±30 → 61px
        ("3이웃 + subset",   3 * spacing + M),       # ±40 → 81px
        ("4이웃 + subset",   4 * spacing + M),       # ±50 → 101px
        ("5이웃 + subset",   5 * spacing + M),       # ±60 → 121px
    ]
    
    k_values = [1.0, 1.5, 2.0, 2.5, 3.0]
    sigma_blur = 1.0
    
    print("=" * 70)
    print("Part A: ROI 크기별 × k값별 평균 성능 (Mean + k·σ)")
    print("=" * 70)
    
    best_overall_iou = 0
    best_overall_config = ""
    
    for config_name, half_h in roi_configs:
        roi_h = 2 * half_h + 1
        print(f"\n--- {config_name}: 반경={half_h}px, ROI 높이={roi_h}px ---")
        print(f"  {'k':>5} | {'AvgIoU':>7} | {'AvgPrec':>7} | {'AvgRec':>7} | "
              f"{'AvgF1':>7} | {'Crack%':>7} | {'ValidPatch':>10}")
        print("  " + "-" * 68)
        
        for k in k_values:
            ious, precs, recs, f1s, crack_ratios = [], [], [], [], []
            valid_count = 0
            
            for px in bad_poi_xs:
                # ROI 경계 계산
                y_min = max(0, bad_poi_y - half_h)
                y_max = min(H, bad_poi_y + half_h + 1)
                x_min = max(0, px - half_h)
                x_max = min(W, px + half_h + 1)
                
                patch = image[y_min:y_max, x_min:x_max]
                gt_patch = gt_clean[y_min:y_max, x_min:x_max]
                
                # GT에 크랙 픽셀이 없으면 스킵
                if gt_patch.sum() == 0:
                    continue
                
                # Gaussian blur
                if sigma_blur > 0:
                    ksize = int(np.ceil(sigma_blur * 3)) * 2 + 1
                    blurred = cv2.GaussianBlur(patch, (ksize, ksize), sigma_blur)
                else:
                    blurred = patch.copy()
                
                # Mean + k·σ threshold
                mu = float(blurred.mean())
                std = float(blurred.std())
                thr = mu - k * std
                detected = (blurred < thr).astype(np.uint8)
                
                # 모폴로지 정리
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                detected = cv2.morphologyEx(detected, cv2.MORPH_CLOSE, kernel)
                detected = cv2.morphologyEx(detected, cv2.MORPH_OPEN, kernel)
                
                detected_bool = detected.astype(bool)
                gt_bool = gt_patch.astype(bool)
                
                # 메트릭 계산
                tp = np.logical_and(detected_bool, gt_bool).sum()
                fp = np.logical_and(detected_bool, ~gt_bool).sum()
                fn = np.logical_and(~detected_bool, gt_bool).sum()
                
                iou = tp / max(tp + fp + fn, 1)
                prec = tp / max(tp + fp, 1)
                rec = tp / max(tp + fn, 1)
                f1 = 2 * prec * rec / max(prec + rec, 1e-10)
                crack_ratio = gt_bool.sum() / gt_bool.size * 100
                
                ious.append(iou)
                precs.append(prec)
                recs.append(rec)
                f1s.append(f1)
                crack_ratios.append(crack_ratio)
                valid_count += 1
            
            if valid_count > 0:
                avg_iou = np.mean(ious)
                avg_prec = np.mean(precs)
                avg_rec = np.mean(recs)
                avg_f1 = np.mean(f1s)
                avg_crack = np.mean(crack_ratios)
                
                print(f"  {k:5.1f} | {avg_iou:7.4f} | {avg_prec:7.4f} | {avg_rec:7.4f} | "
                      f"{avg_f1:7.4f} | {avg_crack:6.2f}% | {valid_count:>10}")
                
                if avg_iou > best_overall_iou:
                    best_overall_iou = avg_iou
                    best_overall_config = f"{config_name}, k={k}"
    
    print(f"\n{'='*70}")
    print(f"Best: {best_overall_config} → Avg IoU = {best_overall_iou:.4f}")
    print(f"{'='*70}")
    
    # --- Part B: Otsu 비교 ---
    print(f"\n{'='*70}")
    print("Part B: ROI 크기별 Otsu 평균 성능 (비교용)")
    print(f"{'='*70}")
    
    best_otsu_iou = 0
    best_otsu_config = ""
    
    for config_name, half_h in roi_configs:
        roi_h = 2 * half_h + 1
        ious = []
        valid_count = 0
        
        for px in bad_poi_xs:
            y_min = max(0, bad_poi_y - half_h)
            y_max = min(H, bad_poi_y + half_h + 1)
            x_min = max(0, px - half_h)
            x_max = min(W, px + half_h + 1)
            
            patch = image[y_min:y_max, x_min:x_max]
            gt_patch = gt_clean[y_min:y_max, x_min:x_max]
            
            if gt_patch.sum() == 0:
                continue
            
            if sigma_blur > 0:
                ksize = int(np.ceil(sigma_blur * 3)) * 2 + 1
                blurred = cv2.GaussianBlur(patch, (ksize, ksize), sigma_blur)
            else:
                blurred = patch.copy()
            
            thr, binary = cv2.threshold(blurred, 0, 255, 
                                         cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            detected = (binary > 0)
            
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            detected_u8 = detected.astype(np.uint8)
            detected_u8 = cv2.morphologyEx(detected_u8, cv2.MORPH_CLOSE, kernel)
            detected_u8 = cv2.morphologyEx(detected_u8, cv2.MORPH_OPEN, kernel)
            detected_bool = detected_u8.astype(bool)
            gt_bool = gt_patch.astype(bool)
            
            tp = np.logical_and(detected_bool, gt_bool).sum()
            fp = np.logical_and(detected_bool, ~gt_bool).sum()
            fn = np.logical_and(~detected_bool, gt_bool).sum()
            iou = tp / max(tp + fp + fn, 1)
            ious.append(iou)
            valid_count += 1
        
        if valid_count > 0:
            avg_iou = np.mean(ious)
            print(f"  {config_name:20s} | ROI={roi_h:3d}px | Avg IoU = {avg_iou:.4f} | "
                  f"Patches = {valid_count}")
            if avg_iou > best_otsu_iou:
                best_otsu_iou = avg_iou
                best_otsu_config = config_name
    
    print(f"\nOtsu Best: {best_otsu_config} → Avg IoU = {best_otsu_iou:.4f}")
    
    # --- Part C: 크랙 위치별 성능 분포 ---
    print(f"\n{'='*70}")
    print("Part C: 크랙 x위치별 검출 성능 분포 (best config)")
    print(f"{'='*70}")
    print("  크랙 opening이 큰 좌측(x≈15) vs 작은 우측(x≈245, tip 근처)")
    print(f"  {'x':>5} | {'IoU':>7} | {'Prec':>7} | {'Recall':>7} | "
          f"{'Opening':>8} | {'CrackPx':>8}")
    print("  " + "-" * 58)
    
    # best config에서 k와 half_h 추출 (간단히 1이웃+subset, k=1.5 사용)
    best_half_h = spacing + M  # 20
    best_k = 1.5
    
    for px in bad_poi_xs:
        y_min = max(0, bad_poi_y - best_half_h)
        y_max = min(H, bad_poi_y + best_half_h + 1)
        x_min = max(0, px - best_half_h)
        x_max = min(W, px + best_half_h + 1)
        
        patch = image[y_min:y_max, x_min:x_max]
        gt_patch = gt_clean[y_min:y_max, x_min:x_max]
        
        if gt_patch.sum() == 0:
            continue
        
        ksize = int(np.ceil(sigma_blur * 3)) * 2 + 1
        blurred = cv2.GaussianBlur(patch, (ksize, ksize), sigma_blur)
        mu = float(blurred.mean())
        std = float(blurred.std())
        thr = mu - best_k * std
        detected = (blurred < thr).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        detected = cv2.morphologyEx(detected, cv2.MORPH_CLOSE, kernel)
        detected = cv2.morphologyEx(detected, cv2.MORPH_OPEN, kernel)
        
        detected_bool = detected.astype(bool)
        gt_bool = gt_patch.astype(bool)
        
        tp = np.logical_and(detected_bool, gt_bool).sum()
        fp = np.logical_and(detected_bool, ~gt_bool).sum()
        fn = np.logical_and(~detected_bool, gt_bool).sum()
        
        iou = tp / max(tp + fp + fn, 1)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        crack_px = gt_bool.sum()
        
        # 크랙 opening 추정 (해당 x에서 GT y방향 폭)
        gt_col = gt_clean[y_min:y_max, px]
        opening = gt_col.sum()
        
        print(f"  {px:5d} | {iou:7.4f} | {prec:7.4f} | {rec:7.4f} | "
              f"{opening:6d}px | {crack_px:8d}")


if __name__ == '__main__':
    test_local_patch_detection()
