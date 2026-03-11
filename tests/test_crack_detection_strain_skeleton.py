"""
테스트: Principal Strain + Morphological Skeleton 기반 크랙 검출

합성 데이터(synthetic_crack_data)를 사용하여 검증.
    - Part 1: Ground Truth 변위장에서 직접 검증 (DIC 오차 없는 이상적 조건)
    - Part 2: DIC 파이프라인(FFT-CC → IC-GN) 통과 후 검증
    - Part 3: Threshold 민감도 분석

Ground Truth:
    - 크랙 경로: y=250, x ∈ [0, 250] (수평, 왼쪽 가장자리 → 이미지 중심)
    - 크랙 팁: (250, 250)
    - crack_mask.npy: forward splatting에서 빈 영역 마스크

사용법:
    python tests/test_crack_detection_strain_skeleton.py

Requirements:
    - scikit-image (skimage)
    - matplotlib (시각화용)
"""

import numpy as np
import sys
import logging
import time
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

sys.path.insert(0, '.')

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(name)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# ─── 외부 의존성 확인 ───
try:
    from skimage.morphology import skeletonize, remove_small_objects, binary_dilation, disk
    from skimage.measure import label, regionprops
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False
    logger.error("scikit-image 미설치. pip install scikit-image")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    _MPL_OK = True
except ImportError:
    _MPL_OK = False
    logger.warning("matplotlib 미설치. 시각화 스킵.")

import cv2


# =============================================================================
#  크랙 검출 알고리즘 (테스트용 인라인 구현)
# =============================================================================

@dataclass
class CrackDetectionResult:
    """크랙 검출 결과"""
    # skeleton 맵 (2D bool, PLS 그리드 크기)
    skeleton: np.ndarray
    # 이진화 맵 (threshold 적용 후)
    binary_map: np.ndarray
    # e1 맵 (주인장변형률)
    e1_map: np.ndarray
    # 검출된 크랙 팁 좌표 리스트 [(grid_row, grid_col), ...]
    tips_grid: List[Tuple[int, int]]
    # 크랙 팁의 물리 좌표 [(phys_x, phys_y), ...]
    tips_physical: List[Tuple[float, float]]
    # 개별 크랙 가지 라벨 맵
    label_map: np.ndarray
    # 각 라벨의 skeleton 픽셀 수
    branch_lengths: dict
    # 사용된 threshold
    threshold: float
    # 그리드 좌표
    grid_x: Optional[np.ndarray] = None
    grid_y: Optional[np.ndarray] = None


def compute_principal_strain_e1(disp_u_2d, disp_v_2d, window_size=15,
                                 poly_order=2, grid_step=1.0):
    """
    2D 변위 그리드에서 주인장변형률 ε₁ 계산.
    기존 PLS 모듈 사용.
    """
    from speckle.core.postprocess.strain_pls import compute_strain_pls

    result = compute_strain_pls(
        disp_u_2d, disp_v_2d,
        window_size=window_size,
        poly_order=poly_order,
        grid_step=grid_step,
        strain_type='engineering',
    )
    return result.e1, result


def estimate_threshold_auto(e1_map, k=5.0):
    """
    ε₁ threshold 자동 추정: median + k × MAD

    크랙이 없는 영역(대부분)의 변형률은 낮고 크랙 영역만 돌출되므로,
    robust 통계량(median, MAD)으로 이상치 경계를 설정한다.

    Args:
        e1_map: 주인장변형률 맵 (2D, NaN 허용)
        k: MAD 배수 (기본 5.0, 높을수록 보수적)

    Returns:
        threshold: float
    """
    valid = e1_map[~np.isnan(e1_map)]
    if len(valid) == 0:
        return 0.0

    # 음수 값 제거 (주인장변형률은 양수가 의미 있음)
    valid_pos = valid[valid > 0]
    if len(valid_pos) == 0:
        return 0.0

    median = np.median(valid_pos)
    mad = np.median(np.abs(valid_pos - median))

    # MAD가 0이면 (거의 동일 값) std 기반 fallback
    if mad < 1e-15:
        std = np.std(valid_pos)
        threshold = median + k * std
    else:
        threshold = median + k * 1.4826 * mad  # 1.4826: MAD → σ 변환 계수

    return threshold


def detect_cracks_from_e1(e1_map, threshold=None, k=5.0,
                           min_size=5, grid_x=None, grid_y=None):
    """
    ε₁ 맵에서 크랙 검출 (threshold → binary → skeleton → tips)

    Args:
        e1_map: 주인장변형률 맵 (2D)
        threshold: 사용자 지정 threshold (None이면 자동)
        k: 자동 threshold의 MAD 배수
        min_size: morphological cleaning 최소 크기 (pixel)
        grid_x, grid_y: POI 그리드의 물리 좌표

    Returns:
        CrackDetectionResult
    """
    # ── 1. Threshold 결정 ──
    if threshold is None:
        threshold = estimate_threshold_auto(e1_map, k=k)
    logger.info(f"  Threshold: {threshold:.6e}")

    # ── 2. 이진화 ──
    e1_clean = np.nan_to_num(e1_map, nan=0.0)
    binary = e1_clean > threshold

    # ── 3. Morphological cleaning ──
    if min_size > 0:
        binary = remove_small_objects(binary, min_size=min_size)

    # ── 4. Skeletonization ──
    skeleton = skeletonize(binary)

    # ── 5. 라벨링 ──
    label_map = label(skeleton.astype(np.uint8), connectivity=2)
    n_labels = label_map.max()

    branch_lengths = {}
    for lbl in range(1, n_labels + 1):
        branch_lengths[lbl] = int(np.sum(label_map == lbl))

    # ── 6. Endpoint (크랙 팁) 검출 ──
    # skeleton 위의 각 픽셀에서 3×3 이웃 중 skeleton 픽셀 수를 카운트.
    # 이웃이 1개인 점 = endpoint (크랙 팁 또는 크랙 입구)
    tips_grid = []
    skel_ys, skel_xs = np.where(skeleton)

    if len(skel_ys) > 0:
        ny, nx = skeleton.shape
        for sy, sx in zip(skel_ys, skel_xs):
            count = 0
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    if dy == 0 and dx == 0:
                        continue
                    ny_ = sy + dy
                    nx_ = sx + dx
                    if 0 <= ny_ < ny and 0 <= nx_ < nx:
                        if skeleton[ny_, nx_]:
                            count += 1
            if count == 1:  # endpoint
                tips_grid.append((sy, sx))

    # ── 7. 물리 좌표 변환 ──
    tips_physical = []
    if grid_x is not None and grid_y is not None:
        for (gy, gx) in tips_grid:
            if gx < len(grid_x) and gy < len(grid_y):
                tips_physical.append((float(grid_x[gx]), float(grid_y[gy])))
            else:
                tips_physical.append((float(gx), float(gy)))
    else:
        tips_physical = [(float(gx), float(gy)) for (gy, gx) in tips_grid]

    return CrackDetectionResult(
        skeleton=skeleton,
        binary_map=binary,
        e1_map=e1_clean,
        tips_grid=tips_grid,
        tips_physical=tips_physical,
        label_map=label_map,
        branch_lengths=branch_lengths,
        threshold=threshold,
        grid_x=grid_x,
        grid_y=grid_y,
    )


# =============================================================================
#  검증 지표 계산
# =============================================================================

def evaluate_crack_tip(detected_tips, true_tip=(250.0, 250.0)):
    """
    크랙 팁 검출 정확도 평가

    Returns:
        best_tip: 가장 가까운 검출 팁
        min_distance: 최소 Euclidean distance (pixels)
    """
    if len(detected_tips) == 0:
        return None, float('inf')

    distances = []
    for (tx, ty) in detected_tips:
        d = np.sqrt((tx - true_tip[0])**2 + (ty - true_tip[1])**2)
        distances.append(d)

    best_idx = np.argmin(distances)
    return detected_tips[best_idx], distances[best_idx]


def evaluate_crack_path(skeleton, grid_x, grid_y,
                         true_y=250.0, true_x_range=(0.0, 250.0)):
    """
    크랙 경로 정확도 평가

    검출된 skeleton 점들과 실제 크랙 라인(y=250, x∈[0,250]) 사이의
    직교 거리(y 방향 오차)를 계산.

    Returns:
        mean_path_error: 평균 경로 오차 (pixels)
        max_path_error: 최대 경로 오차 (pixels)
        n_skeleton_points: skeleton 점 수
    """
    skel_ys, skel_xs = np.where(skeleton)
    if len(skel_ys) == 0:
        return float('inf'), float('inf'), 0

    # skeleton 그리드 좌표 → 물리 좌표
    phys_x = grid_x[skel_xs] if grid_x is not None else skel_xs.astype(float)
    phys_y = grid_y[skel_ys] if grid_y is not None else skel_ys.astype(float)

    # 크랙 경로 범위 내의 점만 필터
    in_range = (phys_x >= true_x_range[0]) & (phys_x <= true_x_range[1])
    if np.sum(in_range) == 0:
        return float('inf'), float('inf'), 0

    path_errors = np.abs(phys_y[in_range] - true_y)

    return float(np.mean(path_errors)), float(np.max(path_errors)), int(np.sum(in_range))


def evaluate_crack_length(skeleton, grid_x, grid_y, true_length=250.0):
    """
    크랙 길이 정확도 평가

    skeleton의 가장 큰 연결 성분의 물리적 길이를 추정.

    Returns:
        detected_length: 검출 길이 (pixels)
        length_error: |detected - true| (pixels)
        relative_error: 상대 오차 (%)
    """
    label_map = label(skeleton.astype(np.uint8), connectivity=2)
    if label_map.max() == 0:
        return 0.0, true_length, 100.0

    # 가장 큰 연결 성분
    biggest_label = 0
    biggest_count = 0
    for lbl in range(1, label_map.max() + 1):
        count = np.sum(label_map == lbl)
        if count > biggest_count:
            biggest_count = count
            biggest_label = lbl

    # 해당 성분의 물리 좌표 범위로 길이 추정
    ys, xs = np.where(label_map == biggest_label)
    if grid_x is not None:
        phys_x = grid_x[xs]
    else:
        phys_x = xs.astype(float)

    detected_length = float(np.max(phys_x) - np.min(phys_x))
    length_error = abs(detected_length - true_length)
    relative_error = length_error / true_length * 100.0

    return detected_length, length_error, relative_error


# =============================================================================
#  시각화
# =============================================================================

def visualize_detection(e1_map, binary_map, skeleton, tips_physical,
                        grid_x, grid_y, true_tip=(250, 250),
                        title="Crack Detection", save_path=None):
    """검출 결과 4-패널 시각화"""
    if not _MPL_OK:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # (a) ε₁ 맵
    e1_display = e1_map.copy()
    e1_display[e1_display <= 0] = np.nan
    im0 = axes[0, 0].imshow(e1_display, cmap='hot', aspect='equal')
    axes[0, 0].set_title('(a) Principal strain ε₁', fontsize=12)
    plt.colorbar(im0, ax=axes[0, 0], shrink=0.8)

    # (b) 이진화
    axes[0, 1].imshow(binary_map, cmap='gray', aspect='equal')
    axes[0, 1].set_title('(b) Binary (thresholded)', fontsize=12)

    # (c) Skeleton
    axes[1, 0].imshow(skeleton, cmap='gray', aspect='equal')
    axes[1, 0].set_title(f'(c) Skeleton ({np.sum(skeleton)} px)', fontsize=12)

    # (d) 결과 오버레이
    # e1 위에 skeleton + tips 표시
    axes[1, 1].imshow(e1_display, cmap='hot', aspect='equal', alpha=0.7)

    skel_ys, skel_xs = np.where(skeleton)
    axes[1, 1].scatter(skel_xs, skel_ys, c='cyan', s=1, label='Skeleton')

    # 검출된 팁
    for i, (tx, ty) in enumerate(tips_physical):
        # 물리 좌표 → 그리드 인덱스
        if grid_x is not None and grid_y is not None:
            gx_idx = np.argmin(np.abs(grid_x - tx))
            gy_idx = np.argmin(np.abs(grid_y - ty))
        else:
            gx_idx, gy_idx = int(tx), int(ty)
        axes[1, 1].plot(gx_idx, gy_idx, 'g^', markersize=12,
                        label=f'Detected tip {i}' if i < 3 else None)

    # 실제 팁
    if grid_x is not None and grid_y is not None:
        true_gx = np.argmin(np.abs(grid_x - true_tip[0]))
        true_gy = np.argmin(np.abs(grid_y - true_tip[1]))
    else:
        true_gx, true_gy = true_tip
    axes[1, 1].plot(true_gx, true_gy, 'r*', markersize=15, label='True tip')
    axes[1, 1].legend(loc='upper right', fontsize=9)
    axes[1, 1].set_title('(d) Overlay: skeleton + tips', fontsize=12)

    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"  시각화 저장: {save_path}")
    plt.close()


# =============================================================================
#  Part 1: Ground Truth 변위장에서 직접 검증
# =============================================================================

def run_part1(data_dir, output_dir):
    """
    DIC 오차 없는 이상적 조건에서 알고리즘 자체의 정확도 확인.

    ground_truth_u/v.npy를 POI 그리드로 다운샘플링 후
    PLS → ε₁ → 크랙 검출.
    """
    print("\n" + "=" * 70)
    print("  Part 1: Ground Truth 변위장에서 직접 검증")
    print("=" * 70)

    gt_u = np.load(str(data_dir / 'ground_truth_u.npy'))
    gt_v = np.load(str(data_dir / 'ground_truth_v.npy'))
    params = dict(np.load(str(data_dir / 'params.npz'), allow_pickle=True))
    crack_tip = params['crack_tip']  # [250, 250]

    print(f"  이미지 크기: {gt_u.shape}")
    print(f"  크랙 팁 (GT): ({crack_tip[0]}, {crack_tip[1]})")

    # POI 그리드로 다운샘플링 (DIC와 동일 조건: spacing=10, subset=21)
    spacing = 10
    subset_size = 21
    half_sub = subset_size // 2

    # DIC에서 POI 그리드가 생성되는 범위와 동일하게
    poi_y = np.arange(half_sub, gt_u.shape[0] - half_sub, spacing)
    poi_x = np.arange(half_sub, gt_u.shape[1] - half_sub, spacing)
    ny_grid = len(poi_y)
    nx_grid = len(poi_x)

    print(f"  POI 그리드: {ny_grid}×{nx_grid}, spacing={spacing}")

    # 그리드 변위 추출
    disp_u_grid = np.zeros((ny_grid, nx_grid), dtype=np.float64)
    disp_v_grid = np.zeros((ny_grid, nx_grid), dtype=np.float64)
    for iy, py in enumerate(poi_y):
        for ix, px in enumerate(poi_x):
            disp_u_grid[iy, ix] = gt_u[py, px]
            disp_v_grid[iy, ix] = gt_v[py, px]

    # PLS → ε₁
    print("  PLS 변형률 계산 중...")
    t0 = time.time()
    e1, pls_result = compute_principal_strain_e1(
        disp_u_grid, disp_v_grid,
        window_size=15, poly_order=2, grid_step=float(spacing)
    )
    print(f"  PLS 완료 ({time.time()-t0:.2f}s)")

    e1_valid = e1[~np.isnan(e1)]
    print(f"  ε₁ 범위: [{np.min(e1_valid):.6e}, {np.max(e1_valid):.6e}]")
    print(f"  ε₁ median: {np.median(e1_valid):.6e}")

    # 크랙 검출
    print("  크랙 검출 중...")
    result = detect_cracks_from_e1(
        e1, threshold=None, k=5.0, min_size=3,
        grid_x=poi_x.astype(float), grid_y=poi_y.astype(float)
    )

    # 평가
    true_tip = (float(crack_tip[1]), float(crack_tip[0]))  # (x, y)
    best_tip, tip_dist = evaluate_crack_tip(result.tips_physical, true_tip)
    mean_path, max_path, n_skel = evaluate_crack_path(
        result.skeleton, poi_x.astype(float), poi_y.astype(float),
        true_y=float(crack_tip[0]), true_x_range=(0.0, float(crack_tip[1]))
    )
    det_len, len_err, len_rel = evaluate_crack_length(
        result.skeleton, poi_x.astype(float), poi_y.astype(float),
        true_length=float(crack_tip[1])  # 크랙 길이 = 250
    )

    print(f"\n  --- Part 1 결과 ---")
    print(f"  Skeleton 픽셀 수: {np.sum(result.skeleton)}")
    print(f"  검출된 가지 수: {result.label_map.max()}")
    print(f"  검출된 endpoint 수: {len(result.tips_grid)}")
    print(f"  Threshold (auto): {result.threshold:.6e}")
    print(f"")
    print(f"  [크랙 팁]")
    print(f"    검출된 팁: {best_tip}")
    print(f"    실제 팁: {true_tip}")
    print(f"    오차: {tip_dist:.2f} px")
    print(f"")
    print(f"  [크랙 경로]")
    print(f"    경로 내 skeleton 점: {n_skel}")
    print(f"    평균 경로 오차: {mean_path:.2f} px")
    print(f"    최대 경로 오차: {max_path:.2f} px")
    print(f"")
    print(f"  [크랙 길이]")
    print(f"    검출 길이: {det_len:.1f} px (실제: 250.0 px)")
    print(f"    절대 오차: {len_err:.1f} px")
    print(f"    상대 오차: {len_rel:.1f}%")

    # 시각화
    if _MPL_OK:
        visualize_detection(
            result.e1_map, result.binary_map, result.skeleton,
            result.tips_physical,
            poi_x.astype(float), poi_y.astype(float),
            true_tip=true_tip,
            title="Part 1: Ground Truth Displacement → Crack Detection",
            save_path=str(output_dir / 'part1_gt_detection.png')
        )

    return result


# =============================================================================
#  Part 2: DIC 파이프라인 통과 후 검증
# =============================================================================

def run_part2(data_dir, output_dir):
    """
    FFT-CC → IC-GN → PLS → ε₁ → 크랙 검출.
    실제 DIC 파이프라인에서의 노이즈, 불량 POI를 포함한 현실 조건.
    """
    print("\n" + "=" * 70)
    print("  Part 2: DIC 파이프라인 통과 후 검증")
    print("=" * 70)

    from speckle.core.initial_guess.fft_cc import compute_fft_cc
    from speckle.core.optimization.icgn import compute_icgn
    from speckle.core.postprocess.strain_pls import compute_strain_pls_from_icgn

    ref_image = cv2.imread(str(data_dir / 'reference.tiff'), cv2.IMREAD_GRAYSCALE)
    def_image = cv2.imread(str(data_dir / 'deformed.tiff'), cv2.IMREAD_GRAYSCALE)
    params = dict(np.load(str(data_dir / 'params.npz'), allow_pickle=True))
    crack_tip = params['crack_tip']

    subset_size = 21
    spacing = 10

    # FFT-CC
    print("  FFT-CC 실행 중...")
    t0 = time.time()
    fft_result = compute_fft_cc(
        ref_image.astype(np.float64),
        def_image.astype(np.float64),
        subset_size=subset_size,
        spacing=spacing,
    )
    print(f"  FFT-CC 완료 ({time.time()-t0:.2f}s)")

    # IC-GN
    print("  IC-GN 실행 중...")
    t0 = time.time()
    icgn_result = compute_icgn(
        ref_image.astype(np.float64),
        def_image.astype(np.float64),
        fft_result,
        subset_size=subset_size,
        zncc_threshold=0.9,
    )
    print(f"  IC-GN 완료 ({time.time()-t0:.2f}s)")
    print(f"    Valid POI: {icgn_result.n_valid}/{icgn_result.n_points}")

    # PLS
    print("  PLS 변형률 계산 중...")
    t0 = time.time()
    pls_result = compute_strain_pls_from_icgn(
        icgn_result, window_size=15, poly_order=2
    )
    print(f"  PLS 완료 ({time.time()-t0:.2f}s)")

    e1 = pls_result.e1
    e1_valid = e1[~np.isnan(e1)]
    print(f"  ε₁ 범위: [{np.min(e1_valid):.6e}, {np.max(e1_valid):.6e}]")

    # 크랙 검출
    print("  크랙 검출 중...")
    result = detect_cracks_from_e1(
        e1, threshold=None, k=5.0, min_size=3,
        grid_x=pls_result.grid_x, grid_y=pls_result.grid_y,
    )

    # 평가
    true_tip = (float(crack_tip[1]), float(crack_tip[0]))
    best_tip, tip_dist = evaluate_crack_tip(result.tips_physical, true_tip)

    gx = pls_result.grid_x if pls_result.grid_x is not None else None
    gy = pls_result.grid_y if pls_result.grid_y is not None else None
    mean_path, max_path, n_skel = evaluate_crack_path(
        result.skeleton, gx, gy,
        true_y=float(crack_tip[0]), true_x_range=(0.0, float(crack_tip[1]))
    )
    det_len, len_err, len_rel = evaluate_crack_length(
        result.skeleton, gx, gy,
        true_length=float(crack_tip[1])
    )

    print(f"\n  --- Part 2 결과 ---")
    print(f"  Skeleton 픽셀 수: {np.sum(result.skeleton)}")
    print(f"  검출된 가지 수: {result.label_map.max()}")
    print(f"  검출된 endpoint 수: {len(result.tips_grid)}")
    print(f"  Threshold (auto): {result.threshold:.6e}")
    print(f"")
    print(f"  [크랙 팁]")
    print(f"    검출된 팁: {best_tip}")
    print(f"    실제 팁: {true_tip}")
    print(f"    오차: {tip_dist:.2f} px")
    print(f"")
    print(f"  [크랙 경로]")
    print(f"    경로 내 skeleton 점: {n_skel}")
    print(f"    평균 경로 오차: {mean_path:.2f} px")
    print(f"    최대 경로 오차: {max_path:.2f} px")
    print(f"")
    print(f"  [크랙 길이]")
    print(f"    검출 길이: {det_len:.1f} px (실제: 250.0 px)")
    print(f"    절대 오차: {len_err:.1f} px")
    print(f"    상대 오차: {len_rel:.1f}%")

    # ADSS 정보 활용 가능성 표시
    if icgn_result.adss_result is not None:
        adss = icgn_result.adss_result
        print(f"\n  [ADSS 정보]")
        print(f"    불량 POI: {adss.n_bad_original}")
        print(f"    복원됨: {adss.n_parent_recovered}")
        print(f"    복원 불가: {adss.n_unrecoverable}")

    # 시각화
    if _MPL_OK:
        visualize_detection(
            result.e1_map, result.binary_map, result.skeleton,
            result.tips_physical,
            gx, gy,
            true_tip=true_tip,
            title="Part 2: DIC Pipeline → Crack Detection",
            save_path=str(output_dir / 'part2_dic_detection.png')
        )

    return result, icgn_result


# =============================================================================
#  Part 3: Threshold 민감도 분석
# =============================================================================

def run_part3(data_dir, output_dir):
    """
    Ground Truth 변위장에서 threshold를 체계적으로 변화시키며
    크랙 팁 오차, 경로 오차, 길이 오차를 기록.
    """
    print("\n" + "=" * 70)
    print("  Part 3: Threshold 민감도 분석")
    print("=" * 70)

    gt_u = np.load(str(data_dir / 'ground_truth_u.npy'))
    gt_v = np.load(str(data_dir / 'ground_truth_v.npy'))
    params = dict(np.load(str(data_dir / 'params.npz'), allow_pickle=True))
    crack_tip = params['crack_tip']
    true_tip = (float(crack_tip[1]), float(crack_tip[0]))

    # POI 그리드
    spacing = 10
    subset_size = 21
    half_sub = subset_size // 2

    poi_y = np.arange(half_sub, gt_u.shape[0] - half_sub, spacing)
    poi_x = np.arange(half_sub, gt_u.shape[1] - half_sub, spacing)
    ny_grid, nx_grid = len(poi_y), len(poi_x)

    disp_u_grid = np.zeros((ny_grid, nx_grid), dtype=np.float64)
    disp_v_grid = np.zeros((ny_grid, nx_grid), dtype=np.float64)
    for iy, py in enumerate(poi_y):
        for ix, px in enumerate(poi_x):
            disp_u_grid[iy, ix] = gt_u[py, px]
            disp_v_grid[iy, ix] = gt_v[py, px]

    # PLS
    print("  PLS 계산 중...")
    e1, _ = compute_principal_strain_e1(
        disp_u_grid, disp_v_grid,
        window_size=15, poly_order=2, grid_step=float(spacing)
    )

    # 자동 threshold 기준값 확인
    auto_threshold = estimate_threshold_auto(e1, k=5.0)
    print(f"  자동 threshold (k=5): {auto_threshold:.6e}")

    # threshold 범위: 자동값의 0.1배 ~ 5배를 로그 스케일로
    e1_valid = e1[~np.isnan(e1)]
    e1_max = np.max(e1_valid)
    e1_p50 = np.median(e1_valid[e1_valid > 0])

    # MAD 배수 k를 변화: 1, 2, 3, 4, 5, 7, 10, 15, 20
    k_values = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0, 20.0]

    results_table = []

    for k in k_values:
        threshold = estimate_threshold_auto(e1, k=k)

        if threshold >= e1_max:
            # threshold가 최대값 이상이면 아무것도 검출 안 됨
            results_table.append({
                'k': k, 'threshold': threshold,
                'tip_dist': float('inf'), 'mean_path': float('inf'),
                'max_path': float('inf'), 'det_len': 0.0,
                'len_err': 250.0, 'len_rel': 100.0,
                'n_skeleton': 0, 'n_tips': 0, 'n_branches': 0,
            })
            continue

        res = detect_cracks_from_e1(
            e1, threshold=threshold, min_size=3,
            grid_x=poi_x.astype(float), grid_y=poi_y.astype(float)
        )

        _, tip_dist = evaluate_crack_tip(res.tips_physical, true_tip)
        mean_path, max_path, n_skel = evaluate_crack_path(
            res.skeleton, poi_x.astype(float), poi_y.astype(float),
            true_y=float(crack_tip[0]), true_x_range=(0.0, float(crack_tip[1]))
        )
        det_len, len_err, len_rel = evaluate_crack_length(
            res.skeleton, poi_x.astype(float), poi_y.astype(float),
            true_length=float(crack_tip[1])
        )

        results_table.append({
            'k': k, 'threshold': threshold,
            'tip_dist': tip_dist, 'mean_path': mean_path,
            'max_path': max_path, 'det_len': det_len,
            'len_err': len_err, 'len_rel': len_rel,
            'n_skeleton': int(np.sum(res.skeleton)),
            'n_tips': len(res.tips_grid),
            'n_branches': int(res.label_map.max()),
        })

    # 결과 출력
    print(f"\n  {'k':>5} | {'Threshold':>12} | {'Tip Err':>8} | {'Path Mean':>9} | "
          f"{'Path Max':>8} | {'Length':>8} | {'Len Err%':>8} | "
          f"{'Skel px':>7} | {'Tips':>4} | {'Br':>3}")
    print("  " + "-" * 105)

    for r in results_table:
        print(f"  {r['k']:5.1f} | {r['threshold']:12.4e} | "
              f"{r['tip_dist']:8.2f} | {r['mean_path']:9.2f} | "
              f"{r['max_path']:8.2f} | {r['det_len']:8.1f} | "
              f"{r['len_rel']:7.1f}% | "
              f"{r['n_skeleton']:7d} | {r['n_tips']:4d} | {r['n_branches']:3d}")

    # 시각화: threshold vs 지표
    if _MPL_OK:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        ks = [r['k'] for r in results_table]
        tip_dists = [r['tip_dist'] if r['tip_dist'] != float('inf') else np.nan
                     for r in results_table]
        mean_paths = [r['mean_path'] if r['mean_path'] != float('inf') else np.nan
                      for r in results_table]
        det_lens = [r['det_len'] for r in results_table]
        n_skels = [r['n_skeleton'] for r in results_table]

        axes[0, 0].plot(ks, tip_dists, 'bo-', linewidth=2, markersize=8)
        axes[0, 0].set_xlabel('k (MAD multiplier)')
        axes[0, 0].set_ylabel('Crack tip error (px)')
        axes[0, 0].set_title('(a) Tip detection error vs threshold')
        axes[0, 0].grid(True, alpha=0.3)

        axes[0, 1].plot(ks, mean_paths, 'rs-', linewidth=2, markersize=8)
        axes[0, 1].set_xlabel('k (MAD multiplier)')
        axes[0, 1].set_ylabel('Mean path error (px)')
        axes[0, 1].set_title('(b) Path error vs threshold')
        axes[0, 1].grid(True, alpha=0.3)

        axes[1, 0].plot(ks, det_lens, 'g^-', linewidth=2, markersize=8)
        axes[1, 0].axhline(y=250, color='k', linestyle='--', label='True length')
        axes[1, 0].set_xlabel('k (MAD multiplier)')
        axes[1, 0].set_ylabel('Detected length (px)')
        axes[1, 0].set_title('(c) Crack length vs threshold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        axes[1, 1].plot(ks, n_skels, 'mD-', linewidth=2, markersize=8)
        axes[1, 1].set_xlabel('k (MAD multiplier)')
        axes[1, 1].set_ylabel('Skeleton pixels')
        axes[1, 1].set_title('(d) Skeleton size vs threshold')
        axes[1, 1].grid(True, alpha=0.3)

        plt.suptitle('Part 3: Threshold Sensitivity Analysis',
                      fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(str(output_dir / 'part3_threshold_sensitivity.png'),
                    dpi=150, bbox_inches='tight')
        logger.info(f"  시각화 저장: {output_dir / 'part3_threshold_sensitivity.png'}")
        plt.close()

    return results_table


# =============================================================================
#  메인
# =============================================================================

def main():
    if not _SKIMAGE_OK:
        print("ERROR: scikit-image가 필요합니다. pip install scikit-image")
        return

    data_dir = Path('synthetic_crack_data')
    if not data_dir.exists():
        print(f"ERROR: {data_dir} 디렉토리가 없습니다.")
        print("       먼저 python tests/generate_crack_images.py 를 실행하세요.")
        return

    output_dir = Path('tests') / '_outputs' / 'crack_detection_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  크랙 검출 테스트: Principal Strain + Morphological Skeleton")
    print("  합성 데이터: Mode I 수평 크랙, 팁 (250,250)")
    print("=" * 70)

    # Part 1
    result1 = run_part1(data_dir, output_dir)

    # Part 2
    try:
        result2, icgn_result = run_part2(data_dir, output_dir)
    except Exception as e:
        logger.error(f"Part 2 실패: {e}")
        import traceback
        traceback.print_exc()
        result2, icgn_result = None, None

    # Part 3
    results3 = run_part3(data_dir, output_dir)

    # 종합 요약
    print("\n" + "=" * 70)
    print("  종합 요약")
    print("=" * 70)
    print(f"  출력 디렉토리: {output_dir.resolve()}")
    print(f"  생성된 파일:")
    for f in sorted(output_dir.glob('*.png')):
        print(f"    - {f.name}")
    print("=" * 70)


if __name__ == '__main__':
    main()

