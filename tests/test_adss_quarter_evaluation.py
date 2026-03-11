"""
tests/test_adss_quarter_evaluation.py
ADSS-DIC v3 사분면 평가 데이터 시각화
- 불량 POI별 사전 ZNCC (pre-filter) 값
- IC-GN 후 최종 ZNCC 값
- 사분면별 통과/탈락 현황
- 임계값 분석
"""
import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
import logging
import platform

# ── 한글 폰트 설정 ──
if platform.system() == 'Windows':
    plt.rcParams['font.family'] = 'Malgun Gothic'
elif platform.system() == 'Darwin':
    plt.rcParams['font.family'] = 'AppleGothic'
else:
    plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

logging.basicConfig(level=logging.INFO)

# ── 경로 설정 ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = PROJECT_ROOT / 'synthetic_crack_data'
TEST_OUTPUT_ROOT = PROJECT_ROOT / 'tests' / '_outputs'
OUTPUT_DIR = TEST_OUTPUT_ROOT / 'output_adss_evaluation'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 설정 ──
SUBSET_SIZE = 25
STEP = 21
ZNCC_THRESHOLD = 0.9
ZNCC_PRE_THRESHOLD = 0.5

QUARTER_NAMES = {5: "Q5:UL", 6: "Q6:UR", 7: "Q7:LL", 8: "Q8:LR"}
QUARTER_COLORS = {5: "#2196F3", 6: "#FF9800", 7: "#4CAF50", 8: "#E91E63"}


def _imread_unicode(path):
    """한글 경로 대응 이미지 로딩"""
    buf = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"이미지 로딩 실패: {path}")
    return img


def run_analysis():
    """FFT-CC → IC-GN → ADSS 실행"""
    from speckle.core.initial_guess import compute_fft_cc
    from speckle.core.optimization import compute_icgn
    from speckle.core.optimization.icgn import _to_gray, _compute_gradient
    from speckle.core.optimization.icgn_core_numba import prefilter_image
    from speckle.core.optimization.adss_subset import compute_adss_recalc

    ref = _imread_unicode(DATA_DIR / 'reference.tiff').astype(np.float64)
    deformed = _imread_unicode(DATA_DIR / 'deformed.tiff').astype(np.float64)

    print("=" * 60)
    print("1단계: FFT-CC")
    fft_result = compute_fft_cc(
        ref_image=ref, def_image=deformed,
        subset_size=SUBSET_SIZE, spacing=11, zncc_threshold=0.6
    )
    n_poi = len(fft_result.points_y)
    print(f"   POI 수: {n_poi}")

    print("2단계: IC-GN (ADSS OFF)")
    icgn_result = compute_icgn(
        ref_image=ref, def_image=deformed,
        subset_size=SUBSET_SIZE, initial_guess=fft_result,
        max_iterations=50, convergence_threshold=1e-3,
        zncc_threshold=ZNCC_THRESHOLD, shape_function='affine',
        enable_variable_subset=False, enable_adss_subset=False,
    )
    valid_before = icgn_result.valid_mask.copy()
    zncc_before = icgn_result.zncc_values.copy()

    bad_indices = np.where(~valid_before)[0]
    n_bad = len(bad_indices)
    print(f"   Valid: {np.sum(valid_before)}/{n_poi}")
    print(f"   불량 POI: {n_bad}개")

    print("3단계: ADSS-DIC v2")
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
        np.asarray(fft_result.points_x, dtype=np.int64),
        np.asarray(fft_result.points_y, dtype=np.int64),
        valid_mask_adss, zncc_adss, parameters,
        icgn_result.converged.copy(), icgn_result.iterations.copy(),
        icgn_result.failure_reason.copy(), SUBSET_SIZE,
        max_iterations=50, convergence_threshold=1e-3,
        shape_function='affine', zncc_threshold=ZNCC_THRESHOLD,
    )

    print(f"   부모 복원: {adss_result.n_parent_recovered}")
    print(f"   sub-POI: {adss_result.n_sub_total}")
    print(f"   복원불가: {adss_result.n_unrecoverable}")
    print(f"   소요시간: {adss_result.elapsed_time:.3f}s")
    print("=" * 60)

    return ref, deformed, icgn_result, fft_result, adss_result, bad_indices, zncc_before


def visualize_pre_zncc_heatmap(adss_result, bad_indices, ax):
    """
    Figure 1-A: 불량 POI × 사분면 사전 ZNCC 히트맵
    candidate_zncc: shape (n_bad, 4) → Q5, Q6, Q7, Q8
    """
    matrix = adss_result.candidate_zncc
    if matrix is None or len(matrix) == 0:
        ax.text(0.5, 0.5, "No candidate_zncc data", transform=ax.transAxes,
                ha='center', va='center', fontsize=14)
        return

    n_bad = matrix.shape[0]

    im = ax.imshow(matrix.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Pre-filter ZNCC', shrink=0.8)

    # 임계값 표시
    for j in range(4):
        for i in range(n_bad):
            val = matrix[i, j]
            if val > 0:
                color = 'white' if val < 0.5 else 'black'
                ax.text(i, j, f'{val:.2f}', ha='center', va='center',
                        fontsize=4 if n_bad > 30 else 6, color=color)

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Q5:UL', 'Q6:UR', 'Q7:LL', 'Q8:LR'])
    ax.set_xlabel('Bad POI index')
    ax.set_title(f'A) 사전 ZNCC 평가 (Pre-filter)\n'
                 f'threshold={ZNCC_PRE_THRESHOLD}, 0=이웃없음/미평가')

    if n_bad <= 50:
        ax.set_xticks(range(n_bad))
        ax.set_xticklabels([str(idx) for idx in bad_indices],
                           rotation=90, fontsize=5)
    else:
        ax.set_xlabel(f'Bad POI index (총 {n_bad}개)')


def visualize_final_zncc_heatmap(adss_result, bad_indices, ax):
    """
    Figure 1-B: IC-GN 후 최종 ZNCC 히트맵
    """
    n_bad = len(bad_indices)
    final_matrix = np.full((n_bad, 4), np.nan)

    parent_to_bad_idx = {int(bad_indices[i]): i for i in range(n_bad)}

    if adss_result.n_sub_total > 0:
        for s in range(adss_result.n_sub_total):
            parent = int(adss_result.parent_indices[s])
            qt = int(adss_result.quarter_types[s])
            zncc_val = float(adss_result.zncc_values[s])

            if parent in parent_to_bad_idx:
                bad_idx = parent_to_bad_idx[parent]
                q_col = qt - 5
                if 0 <= q_col < 4:
                    final_matrix[bad_idx, q_col] = zncc_val

    im = ax.imshow(final_matrix.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1,
                   interpolation='nearest')
    plt.colorbar(im, ax=ax, label='Final ZNCC (post IC-GN)', shrink=0.8)

    for j in range(4):
        for i in range(n_bad):
            val = final_matrix[i, j]
            if not np.isnan(val):
                color = 'white' if val < 0.5 else 'black'
                ax.text(i, j, f'{val:.2f}', ha='center', va='center',
                        fontsize=4 if n_bad > 30 else 6, color=color)

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Q5:UL', 'Q6:UR', 'Q7:LL', 'Q8:LR'])
    ax.set_xlabel('Bad POI index')
    ax.set_title(f'B) IC-GN 후 최종 ZNCC\n'
                 f'threshold={ZNCC_THRESHOLD}, NaN=사전평가 탈락')

    if n_bad <= 50:
        ax.set_xticks(range(n_bad))
        ax.set_xticklabels([str(idx) for idx in bad_indices],
                           rotation=90, fontsize=5)


def visualize_pass_fail_matrix(adss_result, bad_indices, ax):
    """
    Figure 1-C: 사분면 통과/탈락 상태 매트릭스
    0=이웃없음, 1=사전탈락, 2=IC-GN실패, 3=복원성공
    """
    n_bad = len(bad_indices)
    status_matrix = np.zeros((n_bad, 4), dtype=int)

    pre_matrix = adss_result.candidate_zncc
    parent_to_bad_idx = {int(bad_indices[i]): i for i in range(n_bad)}

    # 성공한 sub-POI 수집
    success_set = set()
    if adss_result.n_sub_total > 0:
        for s in range(adss_result.n_sub_total):
            parent = int(adss_result.parent_indices[s])
            qt = int(adss_result.quarter_types[s])
            if parent in parent_to_bad_idx:
                bad_idx = parent_to_bad_idx[parent]
                q_col = qt - 5
                success_set.add((bad_idx, q_col))

    if pre_matrix is not None:
        for i in range(n_bad):
            for j in range(4):
                pre_val = pre_matrix[i, j]
                if pre_val == 0:
                    status_matrix[i, j] = 0  # 이웃 없음
                elif pre_val < ZNCC_PRE_THRESHOLD:
                    status_matrix[i, j] = 1  # 사전 탈락
                elif (i, j) in success_set:
                    status_matrix[i, j] = 3  # 복원 성공
                else:
                    status_matrix[i, j] = 2  # IC-GN 실패

    cmap = plt.cm.colors.ListedColormap(['#9E9E9E', '#F44336', '#FF9800', '#4CAF50'])
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(status_matrix.T, aspect='auto', cmap=cmap, norm=norm,
                   interpolation='nearest')

    labels = ['이웃없음', f'사전탈락\n(pre<{ZNCC_PRE_THRESHOLD})',
              f'IC-GN실패\n(final<{ZNCC_THRESHOLD})', '복원성공']
    colors = ['#9E9E9E', '#F44336', '#FF9800', '#4CAF50']
    patches = [mpatches.Patch(color=c, label=l) for c, l in zip(colors, labels)]
    ax.legend(handles=patches, loc='upper right', fontsize=7, ncol=2)

    ax.set_yticks([0, 1, 2, 3])
    ax.set_yticklabels(['Q5:UL', 'Q6:UR', 'Q7:LL', 'Q8:LR'])
    ax.set_xlabel('Bad POI index')
    ax.set_title('C) 사분면 통과/탈락 상태 매트릭스')

    if n_bad <= 50:
        ax.set_xticks(range(n_bad))
        ax.set_xticklabels([str(idx) for idx in bad_indices],
                           rotation=90, fontsize=5)

    counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for v in status_matrix.flatten():
        counts[v] += 1
    summary = (f"이웃없음:{counts[0]}  사전탈락:{counts[1]}  "
               f"IC-GN실패:{counts[2]}  성공:{counts[3]}")
    ax.text(0.5, -0.15, summary, transform=ax.transAxes, ha='center',
            fontsize=8, style='italic')


def visualize_zncc_distribution(adss_result, bad_indices, ax):
    """
    Figure 1-D: 사전 ZNCC vs 최종 ZNCC 분포 비교
    """
    pre_matrix = adss_result.candidate_zncc
    parent_to_bad_idx = {int(bad_indices[i]): i for i in range(len(bad_indices))}

    pre_values = []
    final_values = []
    labels_scatter = []

    if pre_matrix is not None and adss_result.n_sub_total > 0:
        for s in range(adss_result.n_sub_total):
            parent = int(adss_result.parent_indices[s])
            qt = int(adss_result.quarter_types[s])
            zncc_final = float(adss_result.zncc_values[s])

            if parent in parent_to_bad_idx:
                bad_idx = parent_to_bad_idx[parent]
                q_col = qt - 5
                if 0 <= q_col < 4:
                    pre_val = float(pre_matrix[bad_idx, q_col])
                    pre_values.append(pre_val)
                    final_values.append(zncc_final)
                    labels_scatter.append(QUARTER_NAMES.get(qt, f"Q{qt}"))

    if len(pre_values) > 0:
        pre_arr = np.array(pre_values)
        final_arr = np.array(final_values)

        for qt_id, qt_name in QUARTER_NAMES.items():
            mask = np.array([l == qt_name for l in labels_scatter])
            if np.any(mask):
                ax.scatter(pre_arr[mask], final_arr[mask],
                           c=QUARTER_COLORS[qt_id], label=qt_name,
                           s=30, alpha=0.7, edgecolors='black', linewidths=0.5)

        ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='y=x')
        ax.axvline(x=ZNCC_PRE_THRESHOLD, color='red', linestyle=':', alpha=0.5,
                   label=f'pre_threshold={ZNCC_PRE_THRESHOLD}')
        ax.axhline(y=ZNCC_THRESHOLD, color='blue', linestyle=':', alpha=0.5,
                   label=f'final_threshold={ZNCC_THRESHOLD}')

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel('사전 ZNCC (Pre-filter, 1-warp)')
        ax.set_ylabel('최종 ZNCC (Post IC-GN)')
        ax.legend(fontsize=7, loc='lower right')

        improvement = final_arr - pre_arr
        ax.text(0.02, 0.98,
                f'n={len(pre_arr)}\n'
                f'평균 개선: +{np.mean(improvement):.3f}\n'
                f'최소 pre: {np.min(pre_arr):.3f}\n'
                f'최소 final: {np.min(final_arr):.3f}',
                transform=ax.transAxes, va='top', fontsize=7,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax.text(0.5, 0.5, "No sub-POI data", transform=ax.transAxes,
                ha='center', va='center')

    ax.set_title('D) 사전 ZNCC → 최종 ZNCC 변화 (복원 성공 사분면)')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)


def visualize_spatial_map(ref_img, icgn_result, adss_result, bad_indices, ax):
    """
    Figure 2: 공간 맵 - 불량 POI 위치에 사분면 복원 상태 표시
    """
    if len(ref_img.shape) == 3:
        display_img = ref_img.copy()
    else:
        display_img = ref_img.copy()
        if display_img.dtype != np.uint8:
            display_img = (display_img / display_img.max() * 255).astype(np.uint8)

    ax.imshow(display_img, cmap='gray', alpha=0.5)

    parent_to_bad_idx = {int(bad_indices[i]): i for i in range(len(bad_indices))}
    pre_matrix = adss_result.candidate_zncc
    M = SUBSET_SIZE // 2

    # 성공한 sub-POI 수집
    recovered_parents = set()
    parent_quarters = {}
    if adss_result.n_sub_total > 0:
        for s in range(adss_result.n_sub_total):
            parent = int(adss_result.parent_indices[s])
            qt = int(adss_result.quarter_types[s])
            recovered_parents.add(parent)
            if parent not in parent_quarters:
                parent_quarters[parent] = []
            parent_quarters[parent].append(qt)

    qt_offsets = {
        5: (-M // 2, -M // 2),  # UL
        6: (+M // 2, -M // 2),  # UR
        7: (-M // 2, +M // 2),  # LL
        8: (+M // 2, +M // 2),  # LR
    }

    for i, parent_idx in enumerate(bad_indices):
        px = icgn_result.points_x[parent_idx]
        py = icgn_result.points_y[parent_idx]

        for q_col, qt_id in enumerate([5, 6, 7, 8]):
            dx, dy = qt_offsets[qt_id]
            qx, qy = px + dx, py + dy

            if pre_matrix is not None and pre_matrix[i, q_col] == 0:
                rect = plt.Rectangle((qx - M // 4, qy - M // 4), M // 2, M // 2,
                                     fill=True, facecolor='gray', alpha=0.3,
                                     edgecolor='gray', linewidth=0.5)
                ax.add_patch(rect)
            elif pre_matrix is not None and pre_matrix[i, q_col] < ZNCC_PRE_THRESHOLD:
                rect = plt.Rectangle((qx - M // 4, qy - M // 4), M // 2, M // 2,
                                     fill=True, facecolor='red', alpha=0.4,
                                     edgecolor='red', linewidth=0.5)
                ax.add_patch(rect)
            elif parent_idx in parent_quarters and qt_id in parent_quarters[parent_idx]:
                rect = plt.Rectangle((qx - M // 4, qy - M // 4), M // 2, M // 2,
                                     fill=True, facecolor=QUARTER_COLORS[qt_id],
                                     alpha=0.6, edgecolor='black', linewidth=0.5)
                ax.add_patch(rect)
            else:
                rect = plt.Rectangle((qx - M // 4, qy - M // 4), M // 2, M // 2,
                                     fill=True, facecolor='orange', alpha=0.4,
                                     edgecolor='orange', linewidth=0.5)
                ax.add_patch(rect)

        ax.plot(px, py, 'k+', markersize=3, markeredgewidth=0.5)

    legend_elements = [
        mpatches.Patch(facecolor='gray', alpha=0.3, label='이웃없음'),
        mpatches.Patch(facecolor='red', alpha=0.4, label=f'사전탈락 (pre<{ZNCC_PRE_THRESHOLD})'),
        mpatches.Patch(facecolor='orange', alpha=0.4, label=f'IC-GN실패 (final<{ZNCC_THRESHOLD})'),
        mpatches.Patch(facecolor=QUARTER_COLORS[5], alpha=0.6, label='Q5:UL 성공'),
        mpatches.Patch(facecolor=QUARTER_COLORS[6], alpha=0.6, label='Q6:UR 성공'),
        mpatches.Patch(facecolor=QUARTER_COLORS[7], alpha=0.6, label='Q7:LL 성공'),
        mpatches.Patch(facecolor=QUARTER_COLORS[8], alpha=0.6, label='Q8:LR 성공'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=6, ncol=2)
    ax.set_title('공간 맵: 불량 POI 사분면 복원 상태')


def visualize_per_poi_detail(adss_result, bad_indices, icgn_result, zncc_before):
    """
    Figure 3: 대표 POI 6개의 상세 사분면 평가 데이터
    """
    n_bad = len(bad_indices)
    n_show = min(6, n_bad)

    if n_bad <= 6:
        show_indices = list(range(n_bad))
    else:
        show_indices = np.linspace(0, n_bad - 1, n_show, dtype=int).tolist()

    pre_matrix = adss_result.candidate_zncc
    parent_to_bad_idx = {int(bad_indices[i]): i for i in range(n_bad)}

    # sub-POI 매핑
    parent_sub_data = {}
    if adss_result.n_sub_total > 0:
        for s in range(adss_result.n_sub_total):
            parent = int(adss_result.parent_indices[s])
            qt = int(adss_result.quarter_types[s])
            zncc = float(adss_result.zncc_values[s])
            iters = int(adss_result.iterations[s])
            u = float(adss_result.parameters[s, 0]) if adss_result.parameters.shape[1] > 0 else 0
            v = float(adss_result.parameters[s, 3]) if adss_result.parameters.shape[1] > 3 else 0

            if parent not in parent_sub_data:
                parent_sub_data[parent] = []
            parent_sub_data[parent].append({
                'qt': qt, 'zncc': zncc, 'iters': iters, 'u': u, 'v': v
            })

    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    if n_show == 1:
        axes = axes.reshape(2, 1)

    for col, bad_idx in enumerate(show_indices):
        parent_idx = int(bad_indices[bad_idx])
        px = icgn_result.points_x[parent_idx]
        py = icgn_result.points_y[parent_idx]

        # ── 상단: 사전 ZNCC 바 차트 ──
        ax_top = axes[0, col]
        pre_vals = [0, 0, 0, 0]
        if pre_matrix is not None and bad_idx < pre_matrix.shape[0]:
            pre_vals = [float(pre_matrix[bad_idx, j]) for j in range(4)]

        bars_x = [0, 1, 2, 3]
        bar_colors = []
        for j, val in enumerate(pre_vals):
            if val == 0:
                bar_colors.append('#9E9E9E')
            elif val < ZNCC_PRE_THRESHOLD:
                bar_colors.append('#F44336')
            else:
                bar_colors.append(QUARTER_COLORS[j + 5])

        ax_top.bar(bars_x, pre_vals, color=bar_colors, edgecolor='black', linewidth=0.5)
        ax_top.axhline(y=ZNCC_PRE_THRESHOLD, color='red', linestyle='--', alpha=0.7,
                       label=f'pre_th={ZNCC_PRE_THRESHOLD}')
        ax_top.set_xticks(bars_x)
        ax_top.set_xticklabels(['Q5:UL', 'Q6:UR', 'Q7:LL', 'Q8:LR'], fontsize=7)
        ax_top.set_ylim(0, 1.05)
        ax_top.set_ylabel('Pre-filter ZNCC')
        ax_top.set_title(f'POI[{parent_idx}] ({px:.0f},{py:.0f})\n사전 ZNCC',
                         fontsize=9)
        ax_top.legend(fontsize=6)

        for j, val in enumerate(pre_vals):
            if val > 0:
                ax_top.text(j, val + 0.02, f'{val:.3f}', ha='center', fontsize=7)

        # ── 하단: 최종 ZNCC 바 차트 + 변위 정보 ──
        ax_bot = axes[1, col]
        final_vals = [0, 0, 0, 0]
        iter_vals = [0, 0, 0, 0]
        u_vals = [np.nan, np.nan, np.nan, np.nan]
        v_vals = [np.nan, np.nan, np.nan, np.nan]

        if parent_idx in parent_sub_data:
            for sub in parent_sub_data[parent_idx]:
                q_col = sub['qt'] - 5
                if 0 <= q_col < 4:
                    final_vals[q_col] = sub['zncc']
                    iter_vals[q_col] = sub['iters']
                    u_vals[q_col] = sub['u']
                    v_vals[q_col] = sub['v']

        bar_colors_f = []
        for j, val in enumerate(final_vals):
            if val == 0:
                if pre_vals[j] == 0:
                    bar_colors_f.append('#9E9E9E')
                else:
                    bar_colors_f.append('#FFCDD2')
            elif val < ZNCC_THRESHOLD:
                bar_colors_f.append('#FF9800')
            else:
                bar_colors_f.append(QUARTER_COLORS[j + 5])

        ax_bot.bar(bars_x, final_vals, color=bar_colors_f, edgecolor='black',
                   linewidth=0.5)
        ax_bot.axhline(y=ZNCC_THRESHOLD, color='blue', linestyle='--', alpha=0.7,
                       label=f'final_th={ZNCC_THRESHOLD}')
        ax_bot.set_xticks(bars_x)
        ax_bot.set_xticklabels(['Q5:UL', 'Q6:UR', 'Q7:LL', 'Q8:LR'], fontsize=7)
        ax_bot.set_ylim(0, 1.05)
        ax_bot.set_ylabel('Final ZNCC')
        ax_bot.set_title('최종 ZNCC + 변위', fontsize=9)
        ax_bot.legend(fontsize=6)

        for j, val in enumerate(final_vals):
            if val > 0:
                txt = f'{val:.3f}\nit={iter_vals[j]}'
                if not np.isnan(u_vals[j]):
                    txt += f'\nu={u_vals[j]:.2f}\nv={v_vals[j]:.2f}'
                ax_bot.text(j, val + 0.02, txt, ha='center', fontsize=5,
                            va='bottom')

    fig.suptitle('Figure 3: 대표 POI별 사분면 상세 평가', fontsize=14, fontweight='bold')
    fig.tight_layout()
    path = OUTPUT_DIR / 'fig3_poi_detail.png'
    fig.savefig(str(path), dpi=200, bbox_inches='tight')
    print(f"  → {path} 저장")
    return fig


def visualize_threshold_analysis(adss_result, bad_indices, ax):
    """
    Figure 4: 임계값 민감도 분석
    """
    pre_matrix = adss_result.candidate_zncc
    if pre_matrix is None:
        return

    all_pre = pre_matrix.flatten()
    valid_pre = all_pre[all_pre > 0]

    if len(valid_pre) == 0:
        return

    ax.hist(valid_pre, bins=30, color='steelblue', alpha=0.7, edgecolor='black',
            linewidth=0.5, density=True, label='사전 ZNCC 분포')

    thresholds = np.arange(0.1, 1.0, 0.05)
    pass_rates = [(valid_pre >= th).mean() for th in thresholds]

    ax2 = ax.twinx()
    ax2.plot(thresholds, pass_rates, 'r-o', markersize=3, linewidth=1.5,
             label='통과율')
    ax2.set_ylabel('통과율 (사전 ZNCC ≥ threshold)', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(0, 1.05)

    current_rate = (valid_pre >= ZNCC_PRE_THRESHOLD).mean()
    ax.axvline(x=ZNCC_PRE_THRESHOLD, color='green', linestyle='--', linewidth=2,
               label=f'현재 threshold={ZNCC_PRE_THRESHOLD}\n통과율={current_rate:.1%}')

    ax.set_xlabel('사전 ZNCC 값')
    ax.set_ylabel('밀도')
    ax.set_title(f'E) 사전 ZNCC 분포 & 임계값 민감도 (n={len(valid_pre)})')
    ax.legend(loc='upper left', fontsize=7)
    ax2.legend(loc='upper right', fontsize=7)
    ax.grid(True, alpha=0.3)


# ── 메인 실행 ──
if __name__ == "__main__":
    (ref_img, def_img, icgn_result, fft_result,
     adss_result, bad_indices, zncc_before) = run_analysis()
    n_bad = len(bad_indices)

    if n_bad == 0:
        print("불량 POI 없음 — 시각화 건너뜀")
        sys.exit(0)

    # ════════════════════════════════════════
    # Figure 1: 4-panel 사분면 평가 개요
    # ════════════════════════════════════════
    print("\nFigure 1 생성 중...")
    fig1, axes1 = plt.subplots(2, 2, figsize=(18, 12))

    visualize_pre_zncc_heatmap(adss_result, bad_indices, axes1[0, 0])
    visualize_final_zncc_heatmap(adss_result, bad_indices, axes1[0, 1])
    visualize_pass_fail_matrix(adss_result, bad_indices, axes1[1, 0])
    visualize_zncc_distribution(adss_result, bad_indices, axes1[1, 1])

    fig1.suptitle('ADSS-DIC v2 사분면 평가 데이터 종합', fontsize=16, fontweight='bold')
    fig1.tight_layout()
    path1 = OUTPUT_DIR / 'fig1_quarter_evaluation.png'
    fig1.savefig(str(path1), dpi=200, bbox_inches='tight')
    print(f"  → {path1} 저장")

    # ════════════════════════════════════════
    # Figure 2: 공간 맵
    # ════════════════════════════════════════
    print("\nFigure 2 생성 중...")
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 10))
    visualize_spatial_map(ref_img, icgn_result, adss_result, bad_indices, ax2)
    fig2.tight_layout()
    path2 = OUTPUT_DIR / 'fig2_spatial_map.png'
    fig2.savefig(str(path2), dpi=200, bbox_inches='tight')
    print(f"  → {path2} 저장")

    # ════════════════════════════════════════
    # Figure 3: POI별 상세
    # ════════════════════════════════════════
    print("\nFigure 3 생성 중...")
    fig3 = visualize_per_poi_detail(adss_result, bad_indices, icgn_result, zncc_before)

    # ════════════════════════════════════════
    # Figure 4: 임계값 분석
    # ════════════════════════════════════════
    print("\nFigure 4 생성 중...")
    fig4, ax4 = plt.subplots(1, 1, figsize=(10, 6))
    visualize_threshold_analysis(adss_result, bad_indices, ax4)
    fig4.tight_layout()
    path4 = OUTPUT_DIR / 'fig4_threshold_analysis.png'
    fig4.savefig(str(path4), dpi=200, bbox_inches='tight')
    print(f"  → {path4} 저장")

    # ════════════════════════════════════════
    # 콘솔 요약
    # ════════════════════════════════════════
    print("\n" + "=" * 60)
    print("사분면 평가 데이터 요약")
    print("=" * 60)

    pre_matrix = adss_result.candidate_zncc
    if pre_matrix is not None:
        valid_pre = pre_matrix[pre_matrix > 0]
        passed_pre = valid_pre[valid_pre >= ZNCC_PRE_THRESHOLD]
        print(f"평가 대상 사분면: {len(valid_pre)}/{n_bad * 4} "
              f"(이웃있음: {len(valid_pre)}, 이웃없음: {n_bad * 4 - len(valid_pre)})")
        print(f"사전 평가 통과: {len(passed_pre)}/{len(valid_pre)} "
              f"({len(passed_pre) / max(len(valid_pre), 1):.1%})")
        print(f"사전 ZNCC 통계: min={valid_pre.min():.4f}, "
              f"max={valid_pre.max():.4f}, mean={valid_pre.mean():.4f}")

    print(f"IC-GN 복원 성공: {adss_result.n_sub_total}개 sub-POI")
    print(f"부모 POI 복원: {adss_result.n_parent_recovered}/{n_bad}")
    print(f"복원불가: {adss_result.n_unrecoverable}")

    if adss_result.n_sub_total > 0:
        print(f"최종 ZNCC 통계: min={adss_result.zncc_values.min():.4f}, "
              f"max={adss_result.zncc_values.max():.4f}, "
              f"mean={adss_result.zncc_values.mean():.4f}")
        print(f"평균 사분면/POI: "
              f"{adss_result.n_sub_total / max(adss_result.n_parent_recovered, 1):.2f}")

    print(f"\n결과 저장 폴더: {OUTPUT_DIR}")
    print("=" * 60)

    plt.show()

