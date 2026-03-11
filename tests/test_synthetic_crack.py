"""
ADSS-DIC vs Ground Truth 오차 측정 시각화
==========================================
합성 이미지의 Ground Truth 변위장과 ADSS-DIC 결과를 비교 분석합니다.

생성되는 Figure:
    fig1_displacement_field.png  — GT vs DIC 변위 필드 비교 (4패널)
    fig2_error_map.png           — u/v 오차 공간 맵 (2패널)
    fig3_displacement_profile.png — 변위 프로파일 라인플롯 (2패널)
    fig4_zncc.png                — ZNCC 공간 맵 + 산점도 (2패널)
    fig5_statistics.png          — 정량 통계 막대그래프 + 히스토그램

Usage:
    python tests/visualize_adss_vs_gt.py

Output:
    tests/_outputs/results_visualization/fig1~fig5
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import distance_transform_edt
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from speckle.core.initial_guess import compute_fft_cc
from speckle.core.optimization import compute_icgn


# ======================================================================
#  데이터 로드 / DIC 실행
# ======================================================================

def load_synthetic_data(data_dir='synthetic_crack_data'):
    import cv2
    d = Path(data_dir)
    ref = cv2.imread(str(d / 'reference.tiff'), cv2.IMREAD_GRAYSCALE)
    deformed = cv2.imread(str(d / 'deformed.tiff'), cv2.IMREAD_GRAYSCALE)
    gt_u = np.load(str(d / 'ground_truth_u.npy'))
    gt_v = np.load(str(d / 'ground_truth_v.npy'))
    crack_mask = np.load(str(d / 'crack_mask.npy'))
    return ref, deformed, gt_u, gt_v, crack_mask


def run_pipeline(ref, deformed, subset_size=31, spacing=10,
                 zncc_threshold=0.9):
    fft_result = compute_fft_cc(ref, deformed,
                                subset_size=subset_size,
                                spacing=spacing)
    icgn_result = compute_icgn(
        ref, deformed,
        initial_guess=fft_result,
        subset_size=subset_size,
        max_iterations=50,
        convergence_threshold=0.001,
        shape_function='affine',
        zncc_threshold=zncc_threshold,
        enable_variable_subset=False,
        enable_adss_subset=True,
    )
    return fft_result, icgn_result


# ======================================================================
#  ADSS 대표값 반영 유틸리티
# ======================================================================

def build_merged_arrays(icgn_result):
    """
    ICGNResult에서 ADSS 대표 sub-POI 변위를 부모 POI에 덮어쓴
    최종 변위/ZNCC/valid 배열을 반환.
    """
    res = icgn_result
    disp_u = res.disp_u.copy()
    disp_v = res.disp_v.copy()
    zncc = res.zncc_values.copy()
    valid = res.valid_mask.copy()

    adss = res.adss_result
    adss_parents = set()
    if adss is not None and adss.n_sub > 0:
        for pi in adss.unique_parents:
            rep = adss.get_representative(int(pi))
            if rep is not None:
                disp_u[pi] = adss.parameters[rep, 0]
                disp_v[pi] = adss.parameters[rep, 3]
                zncc[pi] = adss.zncc_values[rep]
                valid[pi] = True
                adss_parents.add(int(pi))

    return disp_u, disp_v, zncc, valid, adss_parents


# ======================================================================
#  POI → 2D 격자 매핑
# ======================================================================

def detect_grid(icgn_result):
    """POI 좌표에서 격자 구조(unique_x, unique_y)를 추출."""
    px = icgn_result.points_x
    py = icgn_result.points_y
    unique_x = np.sort(np.unique(px))
    unique_y = np.sort(np.unique(py))
    return unique_x, unique_y


def to_grid(values, px, py, valid, unique_x, unique_y,
            fill_value=np.nan):
    """1D POI 배열을 2D 격자로 매핑."""
    nx = len(unique_x)
    ny = len(unique_y)
    x2i = {int(x): i for i, x in enumerate(unique_x)}
    y2i = {int(y): i for i, y in enumerate(unique_y)}

    grid = np.full((ny, nx), fill_value, dtype=np.float64)
    for k in range(len(values)):
        if not valid[k]:
            continue
        xi = x2i.get(int(px[k]))
        yi = y2i.get(int(py[k]))
        if xi is not None and yi is not None:
            grid[yi, xi] = values[k]
    return grid


def make_extent(unique_x, unique_y):
    """imshow extent (origin='upper')."""
    return [unique_x[0], unique_x[-1], unique_y[-1], unique_y[0]]


def gt_on_grid(gt_field, unique_x, unique_y):
    """GT 필드를 POI 격자 해상도로 샘플링."""
    nx = len(unique_x)
    ny = len(unique_y)
    grid = np.full((ny, nx), np.nan, dtype=np.float64)
    h, w = gt_field.shape
    for j, y in enumerate(unique_y):
        for i, x in enumerate(unique_x):
            iy, ix = int(y), int(x)
            if 0 <= iy < h and 0 <= ix < w:
                grid[j, i] = gt_field[iy, ix]
    return grid


# ======================================================================
#  Fig 1: GT vs DIC 변위 필드
# ======================================================================

def plot_fig1(icgn_result, gt_u, gt_v, crack_mask, save_dir):
    res = icgn_result
    px, py = res.points_x, res.points_y
    disp_u, disp_v, zncc, valid, _ = build_merged_arrays(res)
    ux, uy = detect_grid(res)
    ext = make_extent(ux, uy)

    gt_u_g = gt_on_grid(gt_u, ux, uy)
    gt_v_g = gt_on_grid(gt_v, ux, uy)
    dic_u_g = to_grid(disp_u, px, py, valid, ux, uy)
    dic_v_g = to_grid(disp_v, px, py, valid, ux, uy)

    crack_rows = np.where(np.any(crack_mask, axis=1))[0]
    cy_min = crack_rows[0] if len(crack_rows) > 0 else gt_u.shape[0] // 2
    cy_max = crack_rows[-1] if len(crack_rows) > 0 else gt_u.shape[0] // 2

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 공통 컬러 범위
    u_vmin = np.nanmin([np.nanmin(gt_u_g), np.nanmin(dic_u_g)])
    u_vmax = np.nanmax([np.nanmax(gt_u_g), np.nanmax(dic_u_g)])
    v_vmin = np.nanmin([np.nanmin(gt_v_g), np.nanmin(dic_v_g)])
    v_vmax = np.nanmax([np.nanmax(gt_v_g), np.nanmax(dic_v_g)])

    def crack_lines(ax):
        ax.axhline(y=cy_min, color='k', ls='--', lw=0.8, alpha=0.7)
        ax.axhline(y=cy_max, color='k', ls='--', lw=0.8, alpha=0.7)

    # (a) GT u
    im = axes[0, 0].imshow(gt_u_g, cmap='jet', extent=ext, aspect='equal',
                            vmin=u_vmin, vmax=u_vmax, interpolation='nearest')
    crack_lines(axes[0, 0])
    axes[0, 0].set_title('(a) Ground Truth  u (pixel)', fontsize=12,
                          fontweight='bold')
    axes[0, 0].set_ylabel('Y (pixel)')
    plt.colorbar(im, ax=axes[0, 0], shrink=0.8)

    # (b) DIC u
    im = axes[0, 1].imshow(dic_u_g, cmap='jet', extent=ext, aspect='equal',
                            vmin=u_vmin, vmax=u_vmax, interpolation='nearest')
    crack_lines(axes[0, 1])
    axes[0, 1].set_title('(b) ADSS-DIC  u (pixel)', fontsize=12,
                          fontweight='bold')
    plt.colorbar(im, ax=axes[0, 1], shrink=0.8)

    # (c) GT v
    im = axes[1, 0].imshow(gt_v_g, cmap='jet', extent=ext, aspect='equal',
                            vmin=v_vmin, vmax=v_vmax, interpolation='nearest')
    crack_lines(axes[1, 0])
    axes[1, 0].set_title('(c) Ground Truth  v (pixel)', fontsize=12,
                          fontweight='bold')
    axes[1, 0].set_ylabel('Y (pixel)')
    axes[1, 0].set_xlabel('X (pixel)')
    plt.colorbar(im, ax=axes[1, 0], shrink=0.8)

    # (d) DIC v
    im = axes[1, 1].imshow(dic_v_g, cmap='jet', extent=ext, aspect='equal',
                            vmin=v_vmin, vmax=v_vmax, interpolation='nearest')
    crack_lines(axes[1, 1])
    axes[1, 1].set_title('(d) ADSS-DIC  v (pixel)', fontsize=12,
                          fontweight='bold')
    axes[1, 1].set_xlabel('X (pixel)')
    plt.colorbar(im, ax=axes[1, 1], shrink=0.8)

    plt.tight_layout()
    plt.savefig(str(Path(save_dir) / 'fig1_displacement_field.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  [저장] fig1_displacement_field.png")


# ======================================================================
#  Fig 2: 오차 공간 맵
# ======================================================================

def plot_fig2(icgn_result, gt_u, gt_v, crack_mask, save_dir):
    res = icgn_result
    px, py = res.points_x.astype(int), res.points_y.astype(int)
    disp_u, disp_v, _, valid, _ = build_merged_arrays(res)
    ux, uy = detect_grid(res)
    ext = make_extent(ux, uy)

    # 오차 계산
    err_u = np.where(valid, disp_u - gt_u[py, px], np.nan)
    err_v = np.where(valid, disp_v - gt_v[py, px], np.nan)

    err_u_g = to_grid(err_u, px, py, valid, ux, uy)
    err_v_g = to_grid(err_v, px, py, valid, ux, uy)

    crack_rows = np.where(np.any(crack_mask, axis=1))[0]
    cy_min = crack_rows[0] if len(crack_rows) > 0 else gt_u.shape[0] // 2
    cy_max = crack_rows[-1] if len(crack_rows) > 0 else gt_u.shape[0] // 2

    # 컬러 범위 (대칭, 95 percentile)
    finite_eu = np.abs(err_u_g[np.isfinite(err_u_g)])
    finite_ev = np.abs(err_v_g[np.isfinite(err_v_g)])
    clim_u = np.percentile(finite_eu, 95) if len(finite_eu) > 0 else 0.1
    clim_v = np.percentile(finite_ev, 95) if len(finite_ev) > 0 else 0.1

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def crack_lines(ax):
        ax.axhline(y=cy_min, color='k', ls='--', lw=0.8, alpha=0.7)
        ax.axhline(y=cy_max, color='k', ls='--', lw=0.8, alpha=0.7)

    im_u = axes[0].imshow(err_u_g, cmap='RdBu_r', extent=ext,
                           aspect='equal', vmin=-clim_u, vmax=clim_u,
                           interpolation='nearest')
    crack_lines(axes[0])
    axes[0].set_title('(a) Error in u  (DIC − GT)  [pixel]', fontsize=12,
                       fontweight='bold')
    axes[0].set_ylabel('Y (pixel)')
    axes[0].set_xlabel('X (pixel)')
    plt.colorbar(im_u, ax=axes[0], shrink=0.8)

    im_v = axes[1].imshow(err_v_g, cmap='RdBu_r', extent=ext,
                           aspect='equal', vmin=-clim_v, vmax=clim_v,
                           interpolation='nearest')
    crack_lines(axes[1])
    axes[1].set_title('(b) Error in v  (DIC − GT)  [pixel]', fontsize=12,
                       fontweight='bold')
    axes[1].set_xlabel('X (pixel)')
    plt.colorbar(im_v, ax=axes[1], shrink=0.8)

    plt.tight_layout()
    plt.savefig(str(Path(save_dir) / 'fig2_error_map.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  [저장] fig2_error_map.png")


# ======================================================================
#  Fig 3: 변위 프로파일 라인플롯
# ======================================================================

def plot_fig3(icgn_result, gt_u, gt_v, crack_mask, save_dir, profile_x=150):
    res = icgn_result
    px, py = res.points_x, res.points_y
    disp_u, disp_v, _, valid, adss_parents = build_merged_arrays(res)
    adss = res.adss_result
    h, w = gt_u.shape
    spacing = 10

    # 해당 x열 근처 POI
    near = np.abs(px - profile_x) <= spacing // 2

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax_idx, (comp, gt_field, dic_disp, p_idx) in enumerate([
            ('u', gt_u, disp_u, 0),
            ('v', gt_v, disp_v, 3)]):
        ax = axes[ax_idx]

        # GT 실선 — 모든 y 픽셀
        y_range = np.arange(h)
        gt_line = gt_field[:, profile_x].copy()
        gt_line[crack_mask[:, profile_x]] = np.nan
        ax.plot(y_range, gt_line, 'r-', lw=1.8, label='Ground truth',
                zorder=1)

        # 균열 영역 음영
        crack_rows = np.where(crack_mask[:, profile_x])[0]
        if len(crack_rows) > 0:
            ax.axvspan(crack_rows[0], crack_rows[-1], alpha=0.12,
                       color='gray', label='Crack region')

        # --- IC-GN 1단계 성공 POI (ADSS 부모 제외) ---
        for i in range(len(px)):
            if not near[i] or not valid[i]:
                continue
            if int(i) in adss_parents:
                continue
            ax.scatter(py[i], dic_disp[i], marker='o', s=25,
                       facecolors='tab:blue', edgecolors='tab:blue',
                       alpha=0.7, zorder=2)

        # --- ADSS 복원: 모든 sub-POI 개별 표시 ---
        if adss is not None and adss.n_sub > 0:
            for i in range(len(px)):
                if not near[i]:
                    continue
                if int(i) not in adss_parents:
                    continue

                sub_indices = adss.get_sub_pois_for_parent(int(i))
                for si in sub_indices:
                    sub_disp = adss.parameters[si, p_idx]
                    qt = int(adss.quarter_types[si])

                    if qt <= 4:
                        ax.scatter(py[i], sub_disp, marker='^', s=50,
                                   facecolors='tab:orange',
                                   edgecolors='tab:orange',
                                   alpha=0.8, zorder=3)
                    else:
                        ax.scatter(py[i], sub_disp, marker='s', s=45,
                                   facecolors='tab:green',
                                   edgecolors='tab:green',
                                   alpha=0.8, zorder=3)

        # 범례용 더미
        ax.scatter([], [], marker='o', s=25, c='tab:blue',
                   label='IC-GN success')
        ax.scatter([], [], marker='^', s=50, c='tab:orange',
                   label='ADSS triangle (Q1-Q4)')
        ax.scatter([], [], marker='s', s=45, c='tab:green',
                   label='ADSS rectangle (Q5-Q8)')

        ax.set_xlabel('Y (pixel)', fontsize=11)
        ax.set_ylabel(f'{comp} (pixel)', fontsize=11)
        ax.set_title(f'({"ab"[ax_idx]}) {comp}-displacement profile '
                     f'at X={profile_x}',
                     fontsize=12, fontweight='bold')
        ax.legend(fontsize=9, loc='best')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(str(Path(save_dir) / 'fig3_displacement_profile.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  [저장] fig3_displacement_profile.png")


# ======================================================================
#  Fig 4: ZNCC 맵 + 산점도
# ======================================================================

def plot_fig4(icgn_result, crack_mask, save_dir, profile_y=250):
    res = icgn_result
    px, py = res.points_x, res.points_y
    _, _, zncc, valid, _ = build_merged_arrays(res)
    ux, uy = detect_grid(res)
    ext = make_extent(ux, uy)
    spacing = 10

    zncc_g = to_grid(zncc, px, py, valid, ux, uy)

    crack_rows = np.where(np.any(crack_mask, axis=1))[0]
    cy_min = crack_rows[0] if len(crack_rows) > 0 else crack_mask.shape[0]//2
    cy_max = crack_rows[-1] if len(crack_rows) > 0 else crack_mask.shape[0]//2

    fig = plt.figure(figsize=(16, 6))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.3], wspace=0.3)

    # (a) ZNCC 공간 맵
    ax_map = fig.add_subplot(gs[0, 0])
    im = ax_map.imshow(zncc_g, cmap='jet', extent=ext, aspect='equal',
                       vmin=0.8, vmax=1.0, interpolation='nearest')
    ax_map.axhline(y=cy_min, color='w', ls='--', lw=0.8, alpha=0.8)
    ax_map.axhline(y=cy_max, color='w', ls='--', lw=0.8, alpha=0.8)
    ax_map.set_title('(a) ZNCC map', fontsize=12, fontweight='bold')
    ax_map.set_ylabel('Y (pixel)')
    ax_map.set_xlabel('X (pixel)')
    plt.colorbar(im, ax=ax_map, shrink=0.8)

    # (b) ZNCC 산점도 (y=profile_y 라인)
    ax_sc = fig.add_subplot(gs[0, 1])
    near = np.abs(py - profile_y) <= spacing // 2
    sc_px = px[near]
    sc_zncc = zncc[near]
    sc_valid = valid[near]

    ax_sc.scatter(sc_px[sc_valid], sc_zncc[sc_valid],
                  marker='D', s=35, facecolors='tab:blue',
                  edgecolors='tab:blue', alpha=0.7, label='ADSS-DIC')
    ax_sc.scatter(sc_px[~sc_valid], sc_zncc[~sc_valid],
                  marker='x', s=30, c='gray', alpha=0.5, label='Failed')
    ax_sc.axhline(y=0.9, color='red', ls='--', lw=1, alpha=0.6,
                  label='Threshold 0.9')
    ax_sc.set_xlabel('X (pixel)', fontsize=11)
    ax_sc.set_ylabel('ZNCC', fontsize=11)
    ax_sc.set_ylim(0.5, 1.02)
    ax_sc.set_title(f'(b) ZNCC at Y={profile_y}', fontsize=12,
                    fontweight='bold')
    ax_sc.legend(fontsize=9, loc='lower left')
    ax_sc.grid(True, alpha=0.2)

    # 인셋 히스토그램 — fig.add_axes로 부모 축 단위 충돌 방지
    n_total = int(np.sum(near))
    n_low = int(np.sum(sc_zncc[sc_valid] < 0.9)) if np.any(sc_valid) else 0
    n_high = int(np.sum(sc_zncc[sc_valid] >= 0.9)) if np.any(sc_valid) else 0
    n_valid_line = n_low + n_high
    pct_low = n_low / max(n_valid_line, 1) * 100
    pct_high = n_high / max(n_valid_line, 1) * 100

    # 레이아웃 확정 후 위치 계산
    fig.canvas.draw()
    sc_pos = ax_sc.get_position()
    ax_ins = fig.add_axes([
        sc_pos.x0 + sc_pos.width * 0.58,
        sc_pos.y0 + sc_pos.height * 0.05,
        sc_pos.width * 0.38,
        sc_pos.height * 0.32,
    ])
    x_bar = [0, 1]
    bars = ax_ins.bar(x_bar, [pct_low, pct_high],
                      color=['tab:red', 'tab:blue'], alpha=0.7)
    for bar, val in zip(bars, [pct_low, pct_high]):
        ax_ins.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.1f}%', ha='center', va='bottom', fontsize=8,
                    fontweight='bold')
    ax_ins.set_xticks(x_bar)
    ax_ins.set_xticklabels(['ZNCC<0.9', 'ZNCC≥0.9'], fontsize=7)
    ax_ins.set_ylabel('Percentage (%)', fontsize=7)
    ax_ins.set_ylim(0, 115)
    ax_ins.tick_params(labelsize=7)

    plt.savefig(str(Path(save_dir) / 'fig4_zncc.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  [저장] fig4_zncc.png")


# ======================================================================
#  Fig 5: 정량 통계
# ======================================================================

def plot_fig5(icgn_result, gt_u, gt_v, crack_mask, save_dir):
    res = icgn_result
    px = res.points_x.astype(int)
    py = res.points_y.astype(int)
    disp_u, disp_v, _, valid, _ = build_merged_arrays(res)
    subset_size = res.subset_size

    dist_map = distance_transform_edt(~crack_mask)

    # 영역 분류
    poi_dist = dist_map[py, px]
    near_mask = valid & (poi_dist <= subset_size)
    far_mask = valid & (poi_dist > subset_size)
    all_valid = valid.copy()

    def stats(mask):
        if np.sum(mask) == 0:
            return {'n': 0, 'mae_u': 0, 'mae_v': 0,
                    'rmse_u': 0, 'rmse_v': 0,
                    'max_u': 0, 'max_v': 0,
                    'err_u': np.array([]), 'err_v': np.array([])}
        eu = disp_u[mask] - gt_u[py[mask], px[mask]]
        ev = disp_v[mask] - gt_v[py[mask], px[mask]]
        return {
            'n': int(np.sum(mask)),
            'mae_u': np.mean(np.abs(eu)),
            'mae_v': np.mean(np.abs(ev)),
            'rmse_u': np.sqrt(np.mean(eu**2)),
            'rmse_v': np.sqrt(np.mean(ev**2)),
            'max_u': np.max(np.abs(eu)),
            'max_v': np.max(np.abs(ev)),
            'err_u': eu, 'err_v': ev,
        }

    s_all = stats(all_valid)
    s_near = stats(near_mask)
    s_far = stats(far_mask)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # (a) 영역별 MAE / RMSE 막대그래프
    ax = axes[0]
    labels = ['MAE(u)', 'MAE(v)', 'RMSE(u)', 'RMSE(v)']
    x_pos = np.arange(len(labels))
    w_bar = 0.25

    vals_all = [s_all['mae_u'], s_all['mae_v'],
                s_all['rmse_u'], s_all['rmse_v']]
    vals_near = [s_near['mae_u'], s_near['mae_v'],
                 s_near['rmse_u'], s_near['rmse_v']]
    vals_far = [s_far['mae_u'], s_far['mae_v'],
                s_far['rmse_u'], s_far['rmse_v']]

    b1 = ax.bar(x_pos - w_bar, vals_all, w_bar, color='tab:blue',
                alpha=0.7, label=f'All valid (n={s_all["n"]})')
    b2 = ax.bar(x_pos, vals_near, w_bar, color='tab:orange',
                alpha=0.7, label=f'Near crack (n={s_near["n"]})')
    b3 = ax.bar(x_pos + w_bar, vals_far, w_bar, color='tab:green',
                alpha=0.7, label=f'Far field (n={s_far["n"]})')

    for bars, vals in [(b1, vals_all), (b2, vals_near), (b3, vals_far)]:
        for bar, val in zip(bars, vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.001,
                        f'{val:.4f}', ha='center', va='bottom', fontsize=7,
                        rotation=45)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Error (pixel)', fontsize=11)
    ax.set_title('(a) Error by Region', fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='y', alpha=0.3)

    # (b) 오차 히스토그램
    ax = axes[1]
    if len(s_all['err_u']) > 0:
        ax.hist(s_all['err_u'], bins=60, alpha=0.5, color='tab:blue',
                edgecolor='black', linewidth=0.3, density=True,
                label=f'u error (MAE={s_all["mae_u"]:.4f})')
    if len(s_all['err_v']) > 0:
        ax.hist(s_all['err_v'], bins=60, alpha=0.5, color='tab:red',
                edgecolor='black', linewidth=0.3, density=True,
                label=f'v error (MAE={s_all["mae_v"]:.4f})')
    ax.axvline(x=0, color='k', ls='-', lw=0.8, alpha=0.5)
    ax.set_xlabel('Error (pixel)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('(b) Error Distribution (all valid POIs)', fontsize=12,
                 fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(str(Path(save_dir) / 'fig5_statistics.png'),
                dpi=200, bbox_inches='tight')
    plt.close()
    print("  [저장] fig5_statistics.png")

    # 콘솔 정량 요약
    print("\n  ┌─────────────────────────────────────────────────────────┐")
    print("  │            ADSS-DIC vs Ground Truth 정량 요약            │")
    print("  ├────────────────┬──────────┬──────────┬──────────────────┤")
    print("  │                │   전체   │ 균열근처 │     원거리       │")
    print(f"  │  유효 POI      │ {s_all['n']:>7d}  │ {s_near['n']:>7d}  │ {s_far['n']:>7d}            │")
    print(f"  │  MAE(u) [px]   │ {s_all['mae_u']:>8.5f} │ {s_near['mae_u']:>8.5f} │ {s_far['mae_u']:>8.5f}          │")
    print(f"  │  MAE(v) [px]   │ {s_all['mae_v']:>8.5f} │ {s_near['mae_v']:>8.5f} │ {s_far['mae_v']:>8.5f}          │")
    print(f"  │  RMSE(u) [px]  │ {s_all['rmse_u']:>8.5f} │ {s_near['rmse_u']:>8.5f} │ {s_far['rmse_u']:>8.5f}          │")
    print(f"  │  RMSE(v) [px]  │ {s_all['rmse_v']:>8.5f} │ {s_near['rmse_v']:>8.5f} │ {s_far['rmse_v']:>8.5f}          │")
    print(f"  │  Max|u| [px]   │ {s_all['max_u']:>8.5f} │ {s_near['max_u']:>8.5f} │ {s_far['max_u']:>8.5f}          │")
    print(f"  │  Max|v| [px]   │ {s_all['max_v']:>8.5f} │ {s_near['max_v']:>8.5f} │ {s_far['max_v']:>8.5f}          │")
    print("  └────────────────┴──────────┴──────────┴──────────────────┘")


# ======================================================================
#  메인
# ======================================================================

def main():
    print("=" * 60)
    print("  ADSS-DIC vs Ground Truth 오차 측정")
    print("=" * 60)

    save_dir = Path('tests') / '_outputs' / 'results_visualization'
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. 데이터 로드
    print("\n[1/3] 합성 데이터 로드...")
    ref, deformed, gt_u, gt_v, crack_mask = load_synthetic_data()
    print(f"  이미지: {ref.shape}")
    print(f"  GT u: [{gt_u.min():.3f}, {gt_u.max():.3f}]")
    print(f"  GT v: [{gt_v.min():.3f}, {gt_v.max():.3f}]")
    print(f"  크랙 픽셀: {np.sum(crack_mask)}")

    # 2. DIC 파이프라인 실행
    print("\n[2/3] FFT-CC → IC-GN → ADSS 실행...")
    fft_result, icgn_result = run_pipeline(ref, deformed)

    res = icgn_result
    _, _, _, _, adss_parents = build_merged_arrays(res)
    print(f"  총 POI:      {res.n_points}")
    print(f"  IC-GN 유효:  {res.n_valid}")
    if res.adss_result is not None:
        adss = res.adss_result
        print(f"  ADSS 불량:   {adss.n_bad_original}")
        print(f"  ADSS 복원:   {adss.n_parent_recovered} 부모, "
              f"{adss.n_sub} sub-POI")
        print(f"  ADSS 실패:   {adss.n_unrecoverable}")

    # 3. 시각화
    print(f"\n[3/3] 시각화 생성 → {save_dir.resolve()}")

    crack_rows = np.where(np.any(crack_mask, axis=1))[0]
    profile_x = 150

    plot_fig1(icgn_result, gt_u, gt_v, crack_mask, save_dir)
    plot_fig2(icgn_result, gt_u, gt_v, crack_mask, save_dir)
    plot_fig3(icgn_result, gt_u, gt_v, crack_mask, save_dir, profile_x)
    plot_fig4(icgn_result, crack_mask, save_dir, profile_x)
    plot_fig5(icgn_result, gt_u, gt_v, crack_mask, save_dir)

    print("\n" + "=" * 60)
    print("  완료!")
    print("=" * 60)


if __name__ == '__main__':
    main()

