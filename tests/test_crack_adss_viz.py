"""
tests/test_crack_adss_viz.py
ADSS-DIC v3 quarter-subset 결과를 활용한 크랙 위치 예측 시각화
- 삼각형(Q1~Q4) + 사각형(Q5~Q8) 적응 선택 방식 대응
"""
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon as MplPolygon
from pathlib import Path
import logging
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / 'synthetic_crack_data'
TEST_OUTPUT_ROOT = PROJECT_ROOT / 'tests' / '_outputs'
OUTPUT_DIR = TEST_OUTPUT_ROOT / 'output_crack_adss_viz'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _imread_unicode(path):
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지 로딩 실패: {path}")
    return img


# ── v3: Q1~Q8 이름 및 색상 ──
QT_NAMES = {
    -1: "실패",
    1: "Q1 상삼각", 2: "Q2 하삼각", 3: "Q3 좌삼각", 4: "Q4 우삼각",
    5: "Q5 UL", 6: "Q6 UR", 7: "Q7 LL", 8: "Q8 LR",
}

# 복원된 quarter의 반대편 = 크랙이 있는 영역
QT_CRACK_REGION = {
    1: "하",   # Q1 상삼각 복원 → 크랙은 하단
    2: "상",   # Q2 하삼각 복원 → 크랙은 상단
    3: "우",   # Q3 좌삼각 복원 → 크랙은 우측
    4: "좌",   # Q4 우삼각 복원 → 크랙은 좌측
    5: "우하", 6: "좌하", 7: "우상", 8: "좌상",
}

qt_colors = {
    1: [0.0, 0.6, 0.0],   # Q1 상삼각: 녹색
    2: [0.9, 0.9, 0.0],   # Q2 하삼각: 노랑
    3: [0.6, 0.3, 0.0],   # Q3 좌삼각: 갈색
    4: [1.0, 0.4, 0.6],   # Q4 우삼각: 분홍
    5: [0.2, 0.5, 1.0],   # Q5 UL: 파랑
    6: [0.0, 0.8, 0.8],   # Q6 UR: 청록
    7: [1.0, 0.6, 0.0],   # Q7 LL: 주황
    8: [0.8, 0.2, 0.8],   # Q8 LR: 보라
}


def _is_triangle_set(quarters):
    """quarters 리스트에 삼각형(Q1~Q4)이 있으면 True"""
    return any(1 <= q <= 4 for q in quarters)


def _is_square_set(quarters):
    """quarters 리스트에 사각형(Q5~Q8)이 있으면 True"""
    return any(5 <= q <= 8 for q in quarters)


def get_crack_mask_from_quarters(subset_size, quarter_types_list):
    """복수의 quarter로부터 복원 영역과 크랙 예측 영역 생성.
    
    삼각형(Q1~Q4): 대각선 분할 영역
    사각형(Q5~Q8): 축 분할 영역
    """
    M = subset_size // 2
    s = subset_size
    recovered = np.zeros((s, s), dtype=np.uint8)
    
    # 로컬 좌표 생성 (중심 = (M, M))
    for qt in quarter_types_list:
        if qt == 1:    # Q1 상삼각: η ≤ -|ξ|  →  row ≤ M - |col - M|
            for r in range(s):
                for c in range(s):
                    eta = r - M   # η
                    xi = c - M    # ξ
                    if eta <= -abs(xi):
                        recovered[r, c] = 1
        elif qt == 2:  # Q2 하삼각: η ≥ |ξ|  →  row ≥ M + |col - M|
            for r in range(s):
                for c in range(s):
                    eta = r - M
                    xi = c - M
                    if eta >= abs(xi):
                        recovered[r, c] = 1
        elif qt == 3:  # Q3 좌삼각: ξ ≤ -|η|  →  col ≤ M - |row - M|
            for r in range(s):
                for c in range(s):
                    eta = r - M
                    xi = c - M
                    if xi <= -abs(eta):
                        recovered[r, c] = 1
        elif qt == 4:  # Q4 우삼각: ξ ≥ |η|  →  col ≥ M + |row - M|
            for r in range(s):
                for c in range(s):
                    eta = r - M
                    xi = c - M
                    if xi >= abs(eta):
                        recovered[r, c] = 1
        elif qt == 5:  # UL
            recovered[:M+1, :M+1] = 1
        elif qt == 6:  # UR
            recovered[:M+1, M:] = 1
        elif qt == 7:  # LL
            recovered[M:, :M+1] = 1
        elif qt == 8:  # LR
            recovered[M:, M:] = 1
    
    crack = 1 - recovered
    return recovered, crack


def _draw_quarter_overlay(ax, patch_h, patch_w, M, quarters, is_triangle):
    """Row 0에 quarter 경계선과 라벨을 그림"""
    if is_triangle:
        # 대각선 경계
        ax.plot([0, patch_w], [0, patch_h], color='yellow', linewidth=1.5, linestyle='--', alpha=0.8)
        ax.plot([0, patch_w], [patch_h, 0], color='yellow', linewidth=1.5, linestyle='--', alpha=0.8)
        # 라벨
        cx, cy = patch_w / 2, patch_h / 2
        ax.text(cx, cy * 0.3, 'Q1', ha='center', va='center',
                color='white', fontsize=9, fontweight='bold')
        ax.text(cx, cy * 1.7, 'Q2', ha='center', va='center',
                color='white', fontsize=9, fontweight='bold')
        ax.text(cx * 0.3, cy, 'Q3', ha='center', va='center',
                color='white', fontsize=9, fontweight='bold')
        ax.text(cx * 1.7, cy, 'Q4', ha='center', va='center',
                color='white', fontsize=9, fontweight='bold')
    else:
        # 축 분할 경계
        ax.axhline(patch_h / 2, color='yellow', linewidth=1.5, linestyle='--', alpha=0.8)
        ax.axvline(patch_w / 2, color='yellow', linewidth=1.5, linestyle='--', alpha=0.8)
        ax.text(patch_w * 0.25, patch_h * 0.25, 'Q5', ha='center', va='center',
                color='white', fontsize=9, fontweight='bold')
        ax.text(patch_w * 0.75, patch_h * 0.25, 'Q6', ha='center', va='center',
                color='white', fontsize=9, fontweight='bold')
        ax.text(patch_w * 0.25, patch_h * 0.75, 'Q7', ha='center', va='center',
                color='white', fontsize=9, fontweight='bold')
        ax.text(patch_w * 0.75, patch_h * 0.75, 'Q8', ha='center', va='center',
                color='white', fontsize=9, fontweight='bold')


def _get_quarter_pixel_mask(qt, subset_size, M):
    """특정 quarter의 픽셀 마스크 반환"""
    s = subset_size
    mask = np.zeros((s, s), dtype=np.uint8)
    if qt == 1:
        for r in range(s):
            for c in range(s):
                if (r - M) <= -abs(c - M):
                    mask[r, c] = 1
    elif qt == 2:
        for r in range(s):
            for c in range(s):
                if (r - M) >= abs(c - M):
                    mask[r, c] = 1
    elif qt == 3:
        for r in range(s):
            for c in range(s):
                if (c - M) <= -abs(r - M):
                    mask[r, c] = 1
    elif qt == 4:
        for r in range(s):
            for c in range(s):
                if (c - M) >= abs(r - M):
                    mask[r, c] = 1
    elif qt == 5:
        mask[:M+1, :M+1] = 1
    elif qt == 6:
        mask[:M+1, M:] = 1
    elif qt == 7:
        mask[M:, :M+1] = 1
    elif qt == 8:
        mask[M:, M:] = 1
    return mask


def visualize_adss_crack_prediction():
    # ── 데이터 로드 ──
    ref = _imread_unicode(DATA_DIR / 'reference.tiff').astype(np.float64)
    deformed = _imread_unicode(DATA_DIR / 'deformed.tiff').astype(np.float64)
    gt_mask = np.load(str(DATA_DIR / 'crack_mask.npy'))
    gt_mask[:220, :] = 0
    gt_mask[280:, :] = 0
    gt_u = np.load(str(DATA_DIR / 'ground_truth_u.npy'))
    gt_v = np.load(str(DATA_DIR / 'ground_truth_v.npy'))

    H, W = deformed.shape
    subset_size = 25
    M = subset_size // 2
    spacing =11
    crack_y = 250
    zncc_threshold = 0.9

    print(f"이미지: {H}x{W}, Subset={subset_size}, Spacing={spacing}")

    # ── DIC 실행 ──
    from speckle.core.initial_guess import compute_fft_cc
    from speckle.core.optimization import compute_icgn
    from speckle.core.optimization.icgn import _to_gray, _compute_gradient
    from speckle.core.optimization.icgn_core_numba import prefilter_image
    from speckle.core.optimization.adss_subset import compute_adss_recalc

    print("\n[1] FFT-CC...")
    fft_result = compute_fft_cc(
        ref_image=ref, def_image=deformed,
        subset_size=subset_size, spacing=spacing, zncc_threshold=0.6
    )
    poi_y = fft_result.points_y
    poi_x = fft_result.points_x
    n_poi = len(poi_y)

    print("[2] IC-GN (ADSS OFF)...")
    icgn_result = compute_icgn(
        ref_image=ref, def_image=deformed,
        subset_size=subset_size, initial_guess=fft_result,
        max_iterations=50, convergence_threshold=1e-3,
        zncc_threshold=zncc_threshold, shape_function='affine',
        enable_variable_subset=False, enable_adss_subset=False,
    )
    valid_before = icgn_result.valid_mask.copy()
    zncc_before = icgn_result.zncc_values.copy()

    print("[3] ADSS v3 재계산 (삼각형+사각형 적응 선택)...")
    ref_gray = _to_gray(ref).astype(np.float64)
    grad_x, grad_y = _compute_gradient(ref_gray)
    def_gray = _to_gray(deformed).astype(np.float64)
    coeffs = prefilter_image(def_gray, order=5)

    n_params = 6
    parameters = np.zeros((n_poi, n_params), dtype=np.float64)
    parameters[:, 0] = icgn_result.disp_u
    parameters[:, 1] = icgn_result.disp_ux
    parameters[:, 2] = icgn_result.disp_uy
    parameters[:, 3] = icgn_result.disp_v
    parameters[:, 4] = icgn_result.disp_vx
    parameters[:, 5] = icgn_result.disp_vy

    valid_mask_adss = valid_before.copy()
    zncc_adss = zncc_before.copy()

    adss_result = compute_adss_recalc(
        ref_gray, grad_x, grad_y, coeffs, 5,
        np.asarray(poi_x, dtype=np.int64), np.asarray(poi_y, dtype=np.int64),
        valid_mask_adss, zncc_adss, parameters,
        icgn_result.converged.copy(), icgn_result.iterations.copy(),
        icgn_result.failure_reason.copy(), subset_size,
        max_iterations=50, convergence_threshold=1e-3,
        shape_function='affine', zncc_threshold=zncc_threshold,
    )

    # ADSSResult 객체에서 정보 추출
    bad_indices = np.where(~valid_before)[0]
    n_bad = len(bad_indices)

    print(f"    불량: {n_bad}")
    print(f"    부모 복원: {adss_result.n_parent_recovered}")
    print(f"    sub-POI 총: {adss_result.n_sub_total}")
    print(f"    복원불가: {adss_result.n_unrecoverable}")

    if n_bad == 0:
        print("불량 POI 없음")
        return

    # ── 부모 POI별 분석 데이터 구축 ──
    poi_analysis = []
    for k in range(n_bad):
        idx = bad_indices[k]
        py, px = int(poi_y[idx]), int(poi_x[idx])
        
        sub_indices = adss_result.get_sub_pois_for_parent(idx)
        quarters = [int(adss_result.quarter_types[s]) for s in sub_indices]
        n_recovered_qt = len(quarters)
        
        best_sub = adss_result.get_representative(idx)
        best_zncc = float(adss_result.zncc_values[best_sub]) if best_sub is not None else 0.0
        best_qt = int(adss_result.quarter_types[best_sub]) if best_sub is not None else -1
        
        sub_u = [float(adss_result.parameters[s, 0]) for s in sub_indices]
        sub_v = [float(adss_result.parameters[s, 3]) for s in sub_indices]
        
        y1, y2 = max(0, py - M), min(H, py + M + 1)
        x1, x2 = max(0, px - M), min(W, px + M + 1)
        patch_gt = gt_mask[y1:y2, x1:x2]
        opening = int(np.max(np.sum(patch_gt, axis=0))) if patch_gt.sum() > 0 else 0

        # 세트 판별
        use_triangle = _is_triangle_set(quarters)

        # 크랙 위치 예측
        pred_crack_ys = []
        pred_crack_xs = []
        for qt in quarters:
            if qt == 1:        # 상삼각 복원 → 크랙은 하단
                pred_crack_ys.append(py + M // 2)
                pred_crack_xs.append(px)
            elif qt == 2:      # 하삼각 복원 → 크랙은 상단
                pred_crack_ys.append(py - M // 2)
                pred_crack_xs.append(px)
            elif qt == 3:      # 좌삼각 복원 → 크랙은 우측
                pred_crack_ys.append(py)
                pred_crack_xs.append(px + M // 2)
            elif qt == 4:      # 우삼각 복원 → 크랙은 좌측
                pred_crack_ys.append(py)
                pred_crack_xs.append(px - M // 2)
            elif qt in [5, 6]: # 상단 복원 → 크랙은 하단
                pred_crack_ys.append(py + M // 2)
                pred_crack_xs.append(px)
            elif qt in [7, 8]: # 하단 복원 → 크랙은 상단
                pred_crack_ys.append(py - M // 2)
                pred_crack_xs.append(px)
        pred_crack_y = int(np.mean(pred_crack_ys)) if pred_crack_ys else py
        pred_crack_x = int(np.mean(pred_crack_xs)) if pred_crack_xs else px


        disp_jump_u = max(sub_u) - min(sub_u) if len(sub_u) >= 2 else 0.0
        disp_jump_v = max(sub_v) - min(sub_v) if len(sub_v) >= 2 else 0.0

        poi_analysis.append({
            'idx': idx, 'py': py, 'px': px,
            'z_before': zncc_before[idx],
            'z_after_best': best_zncc,
            'best_qt': best_qt,
            'quarters': quarters,
            'n_recovered_qt': n_recovered_qt,
            'sub_u': sub_u, 'sub_v': sub_v,
            'disp_jump_u': disp_jump_u,
            'disp_jump_v': disp_jump_v,
            'gt_crack_px': int(patch_gt.sum()),
            'opening': opening,
            'pred_crack_y': pred_crack_y,
            'err_y': abs(pred_crack_y - crack_y),
            'recovered': n_recovered_qt > 0,
            'use_triangle': use_triangle,
            'pred_crack_y': pred_crack_y,
            'pred_crack_x': pred_crack_x,
            'err_y': abs(pred_crack_y - crack_y),
        })

    # ══════════════════════════════════════════════════════════════
    # Figure 1: 대표 POI 6개 — quarter 복수 채택 상세 분석
    # ══════════════════════════════════════════════════════════════
    sorted_by_x = sorted(poi_analysis, key=lambda p: p['px'])
    representatives = []
    target_n_qt = [4, 3, 2, 2, 1, 0]
    used = set()
    for target in target_n_qt:
        for pa in sorted_by_x:
            if pa['idx'] not in used and pa['n_recovered_qt'] == target:
                representatives.append(pa)
                used.add(pa['idx'])
                break
    for pa in sorted_by_x:
        if len(representatives) >= 6:
            break
        if pa['idx'] not in used:
            representatives.append(pa)
            used.add(pa['idx'])

    n_rep = min(len(representatives), 6)
    representatives = representatives[:n_rep]

    fig1, axes1 = plt.subplots(2, n_rep, figsize=(5 * n_rep, 10))
    fig1.suptitle('Figure 1: ADSS v3 적응 선택 — 대표 POI 분석', fontsize=16, fontweight='bold')

    if n_rep == 1:
        axes1 = axes1.reshape(-1, 1)

    for col, pa in enumerate(representatives):
        py, px = pa['py'], pa['px']
        quarters = pa['quarters']
        use_tri = pa['use_triangle']
        y1, y2 = max(0, py - M), min(H, py + M + 1)
        x1, x2 = max(0, px - M), min(W, px + M + 1)
        patch = deformed[y1:y2, x1:x2]
        patch_gt = gt_mask[y1:y2, x1:x2]
        ph, pw = patch.shape

        # ── Row 0: 원본 + GT + quarter 경계선 ──
        rgb0 = np.stack([patch / 255.0] * 3, axis=-1)
        for c in range(3):
            rgb0[:, :, c] = np.where(patch_gt == 1,
                                      rgb0[:, :, c] * 0.3 + [0, 0.9, 0][c] * 0.7,
                                      rgb0[:, :, c])
        axes1[0, col].imshow(rgb0)
        _draw_quarter_overlay(axes1[0, col], ph, pw, M, quarters, use_tri)
        
        set_name = "삼각형" if use_tri else "사각형"
        qt_str = ', '.join([QT_NAMES.get(q, f'Q{q}') for q in quarters]) if quarters else '없음'
        axes1[0, col].set_title(
            f'POI ({py},{px})\nopening={pa["opening"]}px | {set_name} 세트\n복원: {qt_str}',
            fontsize=9, fontweight='bold')
        axes1[0, col].axis('off')

        # ── Row 1: quarter별 색상 채색 + 크랙 예측 ──
        rgb1 = np.stack([patch / 255.0] * 3, axis=-1)

        if quarters:
            rec_mask, crack_pred = get_crack_mask_from_quarters(subset_size, quarters)
            rec_mask = rec_mask[:ph, :pw]
            crack_pred = crack_pred[:ph, :pw]

            for qt in quarters:
                qt_mask = _get_quarter_pixel_mask(qt, subset_size, M)
                qt_mask = qt_mask[:ph, :pw]
                color = qt_colors.get(qt, [0.5, 0.5, 0.5])
                for c in range(3):
                    rgb1[:, :, c] = np.where(qt_mask == 1,
                                              rgb1[:, :, c] * 0.3 + color[c] * 0.7,
                                              rgb1[:, :, c])

            # 크랙 예측 영역: 반투명 빨강
            for c in range(3):
                rgb1[:, :, c] = np.where((crack_pred == 1) & (rec_mask == 0),
                                          rgb1[:, :, c] * 0.4 + [1.0, 0.2, 0.1][c] * 0.6,
                                          rgb1[:, :, c])

            # GT 크랙: 초록 윤곽선
            gt_contour = cv2.dilate(patch_gt.astype(np.uint8),
                                     np.ones((3, 3), np.uint8)) - patch_gt.astype(np.uint8)
            rgb1[gt_contour == 1] = [0, 1, 0]

            tp = np.sum((crack_pred == 1) & (patch_gt == 1))
            fp = np.sum((crack_pred == 1) & (patch_gt == 0))
            fn = np.sum((crack_pred == 0) & (patch_gt == 1))
            iou = tp / max(tp + fp + fn, 1)
        else:
            iou = 0.0

        axes1[1, col].imshow(rgb1)
        if use_tri:
            axes1[1, col].plot([0, pw], [0, ph], color='yellow', linewidth=1, linestyle='--', alpha=0.5)
            axes1[1, col].plot([0, pw], [ph, 0], color='yellow', linewidth=1, linestyle='--', alpha=0.5)
        else:
            axes1[1, col].axhline(ph / 2, color='yellow', linewidth=1, linestyle='--', alpha=0.5)
            axes1[1, col].axvline(pw / 2, color='yellow', linewidth=1, linestyle='--', alpha=0.5)

        jump_str = ''
        if pa['n_recovered_qt'] >= 2:
            jump_str = f'\nΔu={pa["disp_jump_u"]:.3f}, Δv={pa["disp_jump_v"]:.3f}'

        axes1[1, col].set_title(
            f'{pa["n_recovered_qt"]}개 quarter 복원\n'
            f'ZNCC: {pa["z_before"]:.2f} → {pa["z_after_best"]:.2f} | IoU={iou:.2f}'
            f'{jump_str}',
            fontsize=9)
        axes1[1, col].axis('off')

    # 범례 (Q1~Q8 전체)
    legend_patches = [
        mpatches.Patch(color=qt_colors[1], label='Q1 상삼각'),
        mpatches.Patch(color=qt_colors[2], label='Q2 하삼각'),
        mpatches.Patch(color=qt_colors[3], label='Q3 좌삼각'),
        mpatches.Patch(color=qt_colors[4], label='Q4 우삼각'),
        mpatches.Patch(color=qt_colors[5], label='Q5 UL'),
        mpatches.Patch(color=qt_colors[6], label='Q6 UR'),
        mpatches.Patch(color=qt_colors[7], label='Q7 LL'),
        mpatches.Patch(color=qt_colors[8], label='Q8 LR'),
        mpatches.Patch(color=[1.0, 0.2, 0.1], label='크랙 예측 영역'),
        mpatches.Patch(facecolor='none', edgecolor='lime', linewidth=2, label='GT 크랙 윤곽'),
    ]
    fig1.legend(handles=legend_patches, loc='lower center', ncol=5, fontsize=9,
                frameon=True, fancybox=True)
    fig1.tight_layout(rect=[0, 0.08, 1, 0.93])
    path1 = OUTPUT_DIR / 'crack_adss_v3_quarter_detail.png'
    fig1.savefig(str(path1), dpi=150, bbox_inches='tight')
    print(f"\nFigure 1 저장: {path1}")

    # ══════════════════════════════════════════════════════════════
    # Figure 2: 전체 이미지 — ADSS 전/후 + quarter 채색
    # ══════════════════════════════════════════════════════════════
    # 불량 POI 위치 기반 가변 ROI
    bad_ys = [pa['py'] for pa in poi_analysis]
    bad_xs = [pa['px'] for pa in poi_analysis]
    margin = M + spacing  # subset 반크기 + spacing 여유
    roi_y1 = max(0, min(bad_ys) - margin)
    roi_y2 = min(H, max(bad_ys) + margin)
    roi_x1 = max(0, min(bad_xs) - margin)
    roi_x2 = min(W, max(bad_xs) + margin)
    roi_img = deformed[roi_y1:roi_y2, roi_x1:roi_x2]
    gt_roi = gt_mask[roi_y1:roi_y2, roi_x1:roi_x2]

    fig2, axes2 = plt.subplots(1, 3, figsize=(24, 8))
    fig2.suptitle('Figure 2: 전체 크랙 영역 — ADSS v3 적응 선택', fontsize=16, fontweight='bold')

    # ── (a) ADSS 전: 불량 POI ──
    rgb_a = np.stack([roi_img / 255.0] * 3, axis=-1)
    for c in range(3):
        rgb_a[:, :, c] = np.where(gt_roi == 1,
                                    rgb_a[:, :, c] * 0.5 + [0, 0.8, 0][c] * 0.5,
                                    rgb_a[:, :, c])
    axes2[0].imshow(rgb_a, extent=[roi_x1, roi_x2, roi_y2, roi_y1])
    for pa in poi_analysis:
        py, px = pa['py'], pa['px']
        if roi_y1 <= py <= roi_y2 and roi_x1 <= px <= roi_x2:
            rect = plt.Rectangle((px - M, py - M), subset_size, subset_size,
                                  linewidth=1.2, edgecolor='red', facecolor='red', alpha=0.2)
            axes2[0].add_patch(rect)
            axes2[0].plot(px, py, 'rx', markersize=4)
    axes2[0].axhline(crack_y, color='cyan', linewidth=1.5, linestyle='--', alpha=0.8)
    axes2[0].set_title(f'(a) ADSS 전: 불량 POI {n_bad}개 (빨간 사각형) + GT 크랙 (초록)', fontsize=12)
    axes2[0].set_xlim(roi_x1, roi_x2)
    axes2[0].set_ylim(roi_y2, roi_y1)

    # ── (b) ADSS 후: quarter별 색상 표시 ──
    rgb_b = np.stack([roi_img / 255.0] * 3, axis=-1)
    for c in range(3):
        rgb_b[:, :, c] = np.where(gt_roi == 1,
                                    rgb_b[:, :, c] * 0.5 + [0, 0.8, 0][c] * 0.5,
                                    rgb_b[:, :, c])
    axes2[1].imshow(rgb_b, extent=[roi_x1, roi_x2, roi_y2, roi_y1])

    half_sp = spacing // 2
    for pa in poi_analysis:
        py, px = pa['py'], pa['px']
        quarters = pa['quarters']
        if not (roi_y1 <= py <= roi_y2 and roi_x1 <= px <= roi_x2):
            continue
        if not quarters:
            axes2[1].plot(px, py, 'rx', markersize=5)
            continue

        use_tri = pa['use_triangle']

        for qt in quarters:
            color = qt_colors.get(qt, [0.5, 0.5, 0.5])
            if 1 <= qt <= 4:
                # 삼각형: subset 변 + 중심으로 이등변삼각형
                if qt == 1:   # Q1 상삼각: 상변 + 중심
                    verts = [(px - half_sp, py - half_sp),
                             (px + half_sp, py - half_sp),
                             (px, py)]
                elif qt == 2: # Q2 하삼각: 하변 + 중심
                    verts = [(px - half_sp, py + half_sp),
                             (px + half_sp, py + half_sp),
                             (px, py)]
                elif qt == 3: # Q3 좌삼각: 좌변 + 중심
                    verts = [(px - half_sp, py - half_sp),
                             (px - half_sp, py + half_sp),
                             (px, py)]
                elif qt == 4: # Q4 우삼각: 우변 + 중심
                    verts = [(px + half_sp, py - half_sp),
                             (px + half_sp, py + half_sp),
                             (px, py)]

                tri_patch = MplPolygon(verts, closed=True,
                                       linewidth=0.8, edgecolor=color,
                                       facecolor=color, alpha=0.45)
                axes2[1].add_patch(tri_patch)
            else:
                # 사각형: 기존 방식
                qt_rects = {
                    5: (px - half_sp, py - half_sp, half_sp, half_sp),
                    6: (px,           py - half_sp, half_sp, half_sp),
                    7: (px - half_sp, py,           half_sp, half_sp),
                    8: (px,           py,           half_sp, half_sp),
                }
                if qt in qt_rects:
                    rx, ry, rw, rh = qt_rects[qt]
                    rect = plt.Rectangle((rx, ry), rw, rh,
                                          linewidth=0.8, edgecolor=color,
                                          facecolor=color, alpha=0.45)
                    axes2[1].add_patch(rect)
        axes2[1].plot(px, py, '.', color='white', markersize=2)

    axes2[1].axhline(crack_y, color='cyan', linewidth=1.5, linestyle='--', alpha=0.8)

    # 세트 선택 통계
    n_tri_pois = sum(1 for pa in poi_analysis if pa['use_triangle'] and pa['recovered'])
    n_sq_pois = sum(1 for pa in poi_analysis if not pa['use_triangle'] and pa['recovered'])
    axes2[1].set_title(
        f'(b) ADSS v3 후: 복원된 quarter (색상별)\n'
        f'    삼각형 세트={n_tri_pois}개 POI, 사각형 세트={n_sq_pois}개 POI | '
        f'sub-POI: {adss_result.n_sub_total}개', fontsize=12)
    axes2[1].set_xlim(roi_x1, roi_x2)
    axes2[1].set_ylim(roi_y2, roi_y1)

    # ── (c) 크랙 예측 vs GT ──
    # 투표 기반: 각 픽셀에서 "크랙" 투표 수 vs "복원" 투표 수
    crack_vote_map = np.zeros((H, W), dtype=np.float64)
    total_vote_map = np.zeros((H, W), dtype=np.float64)

    for pa in poi_analysis:
        py, px = pa['py'], pa['px']
        quarters = pa['quarters']
        if not quarters:
            continue
        y1 = max(0, py - M)
        y2 = min(H, py + M + 1)
        x1 = max(0, px - M)
        x2 = min(W, px + M + 1)
        recovered, crack_pred = get_crack_mask_from_quarters(subset_size, quarters)
        recovered = recovered[:y2 - y1, :x2 - x1]
        crack_pred = crack_pred[:y2 - y1, :x2 - x1]
        # 복원 영역 + 크랙 영역 모두 투표
        total_vote_map[y1:y2, x1:x2] += (recovered + crack_pred).astype(np.float64)
        crack_vote_map[y1:y2, x1:x2] += crack_pred.astype(np.float64)

    # 크랙 비율 > 50%인 픽셀만 크랙으로 판정
    # 또한 투표가 있는 영역만 (total > 0)
    crack_ratio = np.zeros((H, W), dtype=np.float64)
    valid_votes = total_vote_map > 0
    crack_ratio[valid_votes] = crack_vote_map[valid_votes] / total_vote_map[valid_votes]
    crack_binary_full = ((crack_ratio > 0.5) & valid_votes).astype(np.uint8)

    crack_roi = crack_binary_full[roi_y1:roi_y2, roi_x1:roi_x2]
    crack_binary = crack_roi


    rgb_c = np.stack([roi_img / 255.0] * 3, axis=-1)
    for c in range(3):
        rgb_c[:, :, c] = np.where(crack_binary == 1,
                                    rgb_c[:, :, c] * 0.4 + [1.0, 0.3, 0.2][c] * 0.6,
                                    rgb_c[:, :, c])
    for c in range(3):
        rgb_c[:, :, c] = np.where(gt_roi == 1,
                                    rgb_c[:, :, c] * 0.4 + [0, 1.0, 0][c] * 0.6,
                                    rgb_c[:, :, c])
    overlap = (crack_binary == 1) & (gt_roi == 1)
    rgb_c[overlap] = [1, 1, 0]

    axes2[2].imshow(rgb_c, extent=[roi_x1, roi_x2, roi_y2, roi_y1])
    axes2[2].axhline(crack_y, color='cyan', linewidth=1.5, linestyle='--', alpha=0.8)

    tp_total = np.sum((crack_binary == 1) & (gt_roi == 1))
    fp_total = np.sum((crack_binary == 1) & (gt_roi == 0))
    fn_total = np.sum((crack_binary == 0) & (gt_roi == 1))
    iou_total = tp_total / max(tp_total + fp_total + fn_total, 1)

    axes2[2].set_title(
        f'(c) 크랙 예측(빨강) vs GT(초록), 겹침(노랑)\n'
        f'    IoU={iou_total:.3f} | TP={tp_total} FP={fp_total} FN={fn_total}',
        fontsize=12)
    axes2[2].set_xlim(roi_x1, roi_x2)
    axes2[2].set_ylim(roi_y2, roi_y1)

    for ax in axes2:
        ax.set_xlabel('x (px)', fontsize=10)
        ax.set_ylabel('y (px)', fontsize=10)

    fig2.tight_layout()
    path2 = OUTPUT_DIR / 'crack_adss_v3_prediction_map.png'
    fig2.savefig(str(path2), dpi=150, bbox_inches='tight')
    print(f"Figure 2 저장: {path2}")

    # ══════════════════════════════════════════════════════════════
    # Figure 3: 요약 통계
    # ══════════════════════════════════════════════════════════════
    fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
    fig3.suptitle('Figure 3: ADSS v3 결과 통계', fontsize=16, fontweight='bold')

    # (a) quarter별 복원 빈도 (Q1~Q8)
    qt_freq = {q: 0 for q in range(1, 9)}
    for pa in poi_analysis:
        for qt in pa['quarters']:
            if qt in qt_freq:
                qt_freq[qt] += 1
    names_a = [QT_NAMES[q] for q in range(1, 9)]
    counts_a = [qt_freq[q] for q in range(1, 9)]
    colors_a = [qt_colors[q] for q in range(1, 9)]
    bars = axes3[0, 0].bar(names_a, counts_a, color=colors_a, edgecolor='navy', alpha=0.8)
    for bar, cnt in zip(bars, counts_a):
        axes3[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                          str(cnt), ha='center', fontsize=10, fontweight='bold')
    axes3[0, 0].set_ylabel('복원 횟수')
    axes3[0, 0].set_title(f'(a) quarter별 복원 빈도 (총 {adss_result.n_sub_total}개 sub-POI)')
    axes3[0, 0].tick_params(axis='x', rotation=45)

    # (b) POI당 복원 quarter 수 분포
    n_qt_per_poi = [pa['n_recovered_qt'] for pa in poi_analysis]
    qt_dist = {}
    for n in n_qt_per_poi:
        qt_dist[n] = qt_dist.get(n, 0) + 1
    labels_b = sorted(qt_dist.keys())
    counts_b = [qt_dist[k] for k in labels_b]
    axes3[0, 1].bar([str(k) for k in labels_b], counts_b,
                     color='steelblue', edgecolor='navy', alpha=0.8)
    for i, (k, cnt) in enumerate(zip(labels_b, counts_b)):
        axes3[0, 1].text(i, cnt + 0.3, str(cnt), ha='center', fontsize=11, fontweight='bold')
    axes3[0, 1].set_xlabel('복원 quarter 수')
    axes3[0, 1].set_ylabel('POI 수')
    axes3[0, 1].set_title(f'(b) POI당 복원 quarter 수 분포')

    # (c) ZNCC 전/후 비교
    z_befores = [pa['z_before'] for pa in poi_analysis]
    z_afters = [pa['z_after_best'] for pa in poi_analysis]
    recovered_flags = [pa['recovered'] for pa in poi_analysis]

    for zb, za, rec in zip(z_befores, z_afters, recovered_flags):
        color = 'limegreen' if rec else 'tomato'
        axes3[1, 0].annotate('', xy=(za, 1), xytext=(zb, 0),
                              arrowprops=dict(arrowstyle='->', color=color, alpha=0.4, lw=1.2))
    axes3[1, 0].scatter(z_befores, [0] * len(z_befores), c='tomato', s=30, zorder=5, label='ADSS 전')
    axes3[1, 0].scatter(z_afters, [1] * len(z_afters), c='limegreen', s=30, zorder=5, label='ADSS 후')
    axes3[1, 0].axvline(zncc_threshold, color='black', linestyle='--', linewidth=1.5,
                         label=f'임계값={zncc_threshold}')
    axes3[1, 0].set_xlim(-0.05, 1.05)
    axes3[1, 0].set_ylim(-0.3, 1.3)
    axes3[1, 0].set_yticks([0, 1])
    axes3[1, 0].set_yticklabels(['전', '후'], fontsize=12)
    axes3[1, 0].set_xlabel('ZNCC')
    n_rec = sum(1 for r in recovered_flags if r)
    axes3[1, 0].set_title(f'(c) ZNCC 변화: {n_rec}/{n_bad} 부모 POI 복원')
    axes3[1, 0].legend(fontsize=9, loc='upper left')

    # (d) ADSS 복원 변위 vs Ground Truth 오차 분석
    gt_u_full = np.load(str(DATA_DIR / 'ground_truth_u.npy'))
    gt_v_full = np.load(str(DATA_DIR / 'ground_truth_v.npy'))

    adss_errors_u = []
    adss_errors_v = []
    adss_qt_labels = []
    regular_errors_u = []
    regular_errors_v = []

    for pa in poi_analysis:
        idx = pa['idx']
        py, px = pa['py'], pa['px']
        sub_indices = adss_result.get_sub_pois_for_parent(idx)

        if not pa['recovered']:
            continue

        gt_u_val = gt_u_full[py, px]
        gt_v_val = gt_v_full[py, px]

        reg_u = icgn_result.disp_u[idx]
        reg_v = icgn_result.disp_v[idx]
        regular_errors_u.append(abs(reg_u - gt_u_val))
        regular_errors_v.append(abs(reg_v - gt_v_val))

        for s in sub_indices:
            qt = int(adss_result.quarter_types[s])
            adss_u = float(adss_result.parameters[s, 0])
            adss_v = float(adss_result.parameters[s, 3])
            adss_errors_u.append(abs(adss_u - gt_u_val))
            adss_errors_v.append(abs(adss_v - gt_v_val))
            adss_qt_labels.append(qt)

    adss_errors_u = np.array(adss_errors_u)
    adss_errors_v = np.array(adss_errors_v)
    adss_qt_labels = np.array(adss_qt_labels)
    regular_errors_u = np.array(regular_errors_u)
    regular_errors_v = np.array(regular_errors_v)

    # quarter별 분류 (Q1~Q8)
    all_qt_list = list(range(1, 9))
    qt_err_v = {q: adss_errors_v[adss_qt_labels == q] for q in all_qt_list}

    box_data = [regular_errors_v]
    box_labels = [f'일반DIC\n(n={len(regular_errors_v)})']
    box_colors = ['tomato']

    for q in all_qt_list:
        if len(qt_err_v[q]) > 0:
            box_data.append(qt_err_v[q])
            box_labels.append(f'{QT_NAMES[q]}\n(n={len(qt_err_v[q])})')
            box_colors.append(qt_colors[q])

    bp = axes3[1, 1].boxplot(box_data, labels=box_labels, patch_artist=True,
                              widths=0.6, showfliers=True,
                              flierprops=dict(marker='.', markersize=3, alpha=0.5))

    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    reg_mae_v = np.mean(regular_errors_v) if len(regular_errors_v) > 0 else 0
    adss_mae_v = np.mean(adss_errors_v) if len(adss_errors_v) > 0 else 0

    axes3[1, 1].axhline(0, color='gray', linewidth=0.5, linestyle='-')
    axes3[1, 1].set_ylabel('|v 오차| (px)')
    axes3[1, 1].set_title(
        f'(d) v변위 오차: 일반DIC vs ADSS (GT 기준)\n'
        f'MAE: 일반DIC={reg_mae_v:.4f}px → ADSS={adss_mae_v:.4f}px',
        fontsize=10)
    axes3[1, 1].grid(axis='y', alpha=0.3)
    axes3[1, 1].tick_params(axis='x', rotation=45)

    fig3.tight_layout()
    path3 = OUTPUT_DIR / 'crack_adss_v3_statistics.png'
    fig3.savefig(str(path3), dpi=150, bbox_inches='tight')
    print(f"Figure 3 저장: {path3}")

    # ── 요약 출력 ──
    print("\n" + "=" * 60)
    print("ADSS v3 결과 요약")
    print("=" * 60)
    print(f"불량 POI:          {n_bad}")
    print(f"부모 복원:         {adss_result.n_parent_recovered}")
    print(f"sub-POI 총:        {adss_result.n_sub_total}")
    print(f"복원불가:          {adss_result.n_unrecoverable}")
    print(f"평균 quarter/POI:  {adss_result.n_sub_total / max(n_bad, 1):.1f}")
    print(f"삼각형 세트 POI:   {n_tri_pois}")
    print(f"사각형 세트 POI:   {n_sq_pois}")
    print(f"전체 IoU:          {iou_total:.3f}")
    print(f"소요시간:          {adss_result.elapsed_time:.3f}s")

    plt.show()
    print("\n분석 완료")


if __name__ == '__main__':
    visualize_adss_crack_prediction()

