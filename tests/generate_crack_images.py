"""
Mode I 크랙 합성 스페클 이미지 생성기

ADSS-DIC (Zhao & Pan, 2025, Experimental Mechanics) 논문의
수치 실험과 동일한 조건으로 합성 이미지를 생성합니다.

순방향 매핑(forward splatting)을 사용하여 크랙 틈이 실제 검정으로 나타남.

생성물:
    - reference.tiff      : 스페클 참조 이미지 (500×500)
    - deformed.tiff        : Mode I 크랙 변형 이미지 (크랙 틈 = 검정)
    - ground_truth_u.npy   : x방향 변위 ground truth
    - ground_truth_v.npy   : y방향 변위 ground truth
    - crack_mask.npy       : 크랙 틈 마스크 (True = 크랙 내부)
    - visualization.png    : 시각화
    - params.npz           : 생성 파라미터

Parameters (ADSS-DIC 논문 조건):
    - 이미지 크기: 500×500 px
    - KI = 17 GPa·px^0.5
    - E = 70 GPa
    - ν = 0.3 (평면 변형률)
    - 크랙 팁: 이미지 중심 (250, 250)
    - 크랙 방향: 수평 (왼쪽 가장자리 → 중심)
    - 노이즈: σ = 5 Gray Levels (참조 + 변형 모두)
    - 최대 크랙 개구량: ~11.12 px

Usage:
    python generate_crack_images.py

References:
    Zhao, J. & Pan, B. (2025). Adaptive Subset-Subdivision for Automatic
    Digital Image Correlation Calculation on Discontinuous Shape and
    Deformation. Experimental Mechanics. doi:10.1007/s11340-025-01243-5
"""

import numpy as np
from scipy.ndimage import map_coordinates
import cv2
from pathlib import Path
import time

try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


# =============================================================================
#  1. 스페클 이미지 생성
# =============================================================================

def generate_speckle_image(height=500, width=500,
                           speckle_radius=1.5,
                           density=0.8,
                           seed=42):
    """
    합성 스페클 이미지 생성 (회색 배경 + 촘촘한 검은 스페클)

    Args:
        speckle_radius: 스페클 반점 반경 (px). 작을수록 미세한 점.
        density: 스페클 밀도 (0~1). 높을수록 촘촘함.
    """
    rng = np.random.RandomState(seed)

    area_per_speckle = np.pi * (speckle_radius * 2.0) ** 2
    n_speckles = int(height * width * density / area_per_speckle)

    img = np.ones((height, width), dtype=np.float64) * 180.0

    yy, xx = np.mgrid[0:height, 0:width]

    centers_y = rng.uniform(0, height, n_speckles)
    centers_x = rng.uniform(0, width, n_speckles)
    amplitudes = rng.uniform(80, 160, n_speckles)
    radii = rng.uniform(speckle_radius * 0.6,
                        speckle_radius * 1.4, n_speckles)

    for i in range(n_speckles):
        r_max = int(radii[i] * 3.0)
        y_lo = max(0, int(centers_y[i]) - r_max)
        y_hi = min(height, int(centers_y[i]) + r_max + 1)
        x_lo = max(0, int(centers_x[i]) - r_max)
        x_hi = min(width, int(centers_x[i]) + r_max + 1)

        local_y = yy[y_lo:y_hi, x_lo:x_hi]
        local_x = xx[y_lo:y_hi, x_lo:x_hi]

        dist_sq = ((local_y - centers_y[i])**2 +
                   (local_x - centers_x[i])**2)
        gauss = amplitudes[i] * np.exp(-dist_sq / (2 * radii[i]**2))

        img[y_lo:y_hi, x_lo:x_hi] -= gauss

    return np.clip(img, 0, 255)


# =============================================================================
#  2. Mode I 크랙 변위장
# =============================================================================

def mode1_crack_displacement(height, width, KI=17.0, E=70.0, nu=0.3,
                              crack_tip_x=None, crack_tip_y=None):
    """
    Mode I 크랙 변위장 계산 (ADSS-DIC Eq.7)

    크랙 경로: y == crack_tip_y, x <= crack_tip_x (수평, 왼쪽→팁)

    Args:
        height, width: 이미지 크기
        KI: 응력확대계수 (GPa·px^0.5)
        E: 영률 (GPa)
        nu: 포아송비
        crack_tip_x, crack_tip_y: 크랙 팁 좌표

    Returns:
        u: x방향 변위장 (height, width)
        v: y방향 변위장 (height, width)
    """
    if crack_tip_x is None:
        crack_tip_x = width / 2
    if crack_tip_y is None:
        crack_tip_y = height / 2

    kappa = 3 - 4 * nu  # 평면 변형률

    y, x = np.mgrid[0:height, 0:width]
    dx = x.astype(np.float64) - crack_tip_x
    dy = y.astype(np.float64) - crack_tip_y

    r = np.sqrt(dx**2 + dy**2)
    r = np.maximum(r, 1e-6)
    theta = np.arctan2(dy, dx)

    sqrt_r_2pi = np.sqrt(r / (2 * np.pi))
    coeff = KI / (2 * E)

    cos_half = np.cos(theta / 2)
    sin_half = np.sin(theta / 2)
    cos_3half = np.cos(3 * theta / 2)
    sin_3half = np.sin(3 * theta / 2)

    u = coeff * sqrt_r_2pi * (1 + nu) * (
        (2 * kappa - 1) * cos_half - cos_3half
    )
    v = coeff * sqrt_r_2pi * (1 + nu) * (
        (2 * kappa + 1) * sin_half - sin_3half
    )

    return u, v


# =============================================================================
#  3. 순방향 워프 (Forward Splatting)
# =============================================================================

def _forward_warp_python(ref, u, v, deformed, weight):
    """순방향 워프 — 순수 Python (Numba 없을 때 fallback)"""
    h, w = ref.shape
    for iy in range(h):
        for ix in range(w):
            dx = ix + u[iy, ix]
            dy = iy + v[iy, ix]

            x0 = int(np.floor(dx))
            y0 = int(np.floor(dy))
            fx = dx - x0
            fy = dy - y0

            val = ref[iy, ix]

            pairs = [
                (y0,     x0,     (1.0 - fx) * (1.0 - fy)),
                (y0,     x0 + 1, fx * (1.0 - fy)),
                (y0 + 1, x0,     (1.0 - fx) * fy),
                (y0 + 1, x0 + 1, fx * fy),
            ]
            for (yy, xx, ww) in pairs:
                if 0 <= yy < h and 0 <= xx < w:
                    deformed[yy, xx] += val * ww
                    weight[yy, xx] += ww


if HAS_NUMBA:
    @jit(nopython=True, cache=True)
    def _forward_warp_numba(ref, u, v, deformed, weight):
        """Numba 가속 순방향 워프 (bilinear splatting)"""
        h, w = ref.shape
        for iy in range(h):
            for ix in range(w):
                dx = ix + u[iy, ix]
                dy = iy + v[iy, ix]

                x0 = int(np.floor(dx))
                y0 = int(np.floor(dy))
                fx = dx - x0
                fy = dy - y0

                val = ref[iy, ix]

                w00 = (1.0 - fx) * (1.0 - fy)
                w01 = fx * (1.0 - fy)
                w10 = (1.0 - fx) * fy
                w11 = fx * fy

                if 0 <= y0 < h and 0 <= x0 < w:
                    deformed[y0, x0] += val * w00
                    weight[y0, x0] += w00
                if 0 <= y0 < h and 0 <= x0 + 1 < w:
                    deformed[y0, x0 + 1] += val * w01
                    weight[y0, x0 + 1] += w01
                if 0 <= y0 + 1 < h and 0 <= x0 < w:
                    deformed[y0 + 1, x0] += val * w10
                    weight[y0 + 1, x0] += w10
                if 0 <= y0 + 1 < h and 0 <= x0 + 1 < w:
                    deformed[y0 + 1, x0 + 1] += val * w11
                    weight[y0 + 1, x0 + 1] += w11


def warp_image_forward(ref_image, u, v, noise_std=0.0, seed=None):
    """
    순방향 매핑으로 변형 이미지 생성

    각 참조 픽셀 (x, y)를 (x+u, y+v)로 이동.
    크랙 양쪽이 벌어지면서 빈 틈은 검정(0)으로 남음.

    Args:
        ref_image: 참조 이미지 (H, W) float64
        u, v: 변위장 (H, W)
        noise_std: 가우시안 노이즈 σ
        seed: 노이즈 시드

    Returns:
        deformed: 변형 이미지 (H, W) float64
        crack_mask: 크랙 틈 마스크 (True = 빈 영역)
    """
    h, w = ref_image.shape
    ref = ref_image.astype(np.float64)

    deformed = np.zeros((h, w), dtype=np.float64)
    weight = np.zeros((h, w), dtype=np.float64)

    if HAS_NUMBA:
        _forward_warp_numba(ref, u, v, deformed, weight)
    else:
        print("  [경고] Numba 미설치. Python 루프 사용 (느림)...")
        _forward_warp_python(ref, u, v, deformed, weight)

    # 가중 평균
    valid = weight > 0
    deformed[valid] /= weight[valid]
    # weight == 0 → 검정 유지 (크랙 틈)

    crack_mask = ~valid

    if noise_std > 0:
        rng = np.random.RandomState(seed)
        noise = rng.normal(0, noise_std, deformed.shape)
        # 크랙 내부에는 카메라 노이즈만 약하게
        noise[crack_mask] *= 0.1
        deformed += noise

    return np.clip(deformed, 0, 255), crack_mask


# =============================================================================
#  4. 시각화
# =============================================================================

def visualize_results(ref, deformed, u, v, crack_mask, save_path=None):
    """결과 시각화 (6 패널)"""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    axes[0, 0].imshow(ref, cmap='gray', vmin=0, vmax=255)
    axes[0, 0].set_title('(a) Reference Image', fontsize=13)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(deformed, cmap='gray', vmin=0, vmax=255)
    axes[0, 1].set_title('(b) Deformed Image (Mode I Crack)', fontsize=13)
    axes[0, 1].axis('off')

    # 크랙 마스크 오버레이
    overlay = np.stack([deformed/255]*3, axis=-1)
    overlay[crack_mask] = [1, 0, 0]  # 크랙 틈 = 빨강
    axes[0, 2].imshow(overlay)
    axes[0, 2].set_title(f'(c) Crack mask ({np.sum(crack_mask)} px)', fontsize=13)
    axes[0, 2].axis('off')

    im_u = axes[1, 0].imshow(u, cmap='RdBu_r')
    axes[1, 0].set_title(f'u field (max |u| = {np.max(np.abs(u)):.2f} px)',
                          fontsize=13)
    plt.colorbar(im_u, ax=axes[1, 0], shrink=0.8)
    axes[1, 0].axis('off')

    im_v = axes[1, 1].imshow(v, cmap='RdBu_r')
    axes[1, 1].set_title(f'v field (max |v| = {np.max(np.abs(v)):.2f} px)',
                          fontsize=13)
    plt.colorbar(im_v, ax=axes[1, 1], shrink=0.8)
    axes[1, 1].axis('off')

    mag = np.sqrt(u**2 + v**2)
    im_m = axes[1, 2].imshow(mag, cmap='jet')
    axes[1, 2].set_title(f'Magnitude (max = {np.max(mag):.2f} px)', fontsize=13)
    plt.colorbar(im_m, ax=axes[1, 2], shrink=0.8)
    axes[1, 2].axis('off')

    plt.suptitle('Mode I Crack Synthetic Images for DIC\n'
                 r'$K_I$=17, E=70, $\nu$=0.3, noise $\sigma$=5 GL',
                 fontsize=15, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  시각화 저장: {save_path}")
    plt.close()


# =============================================================================
#  5. 메인 생성 함수
# =============================================================================

def generate_crack_dataset(output_dir='synthetic_crack_data',
                           image_size=500,
                           KI=17.0, E=70.0, nu=0.3,
                           noise_std=5.0,
                           speckle_radius=1.0,
                           speckle_density=0.9,
                           speckle_seed=42,
                           noise_seed=123):
    """
    Mode I 크랙 합성 데이터셋 생성

    Args:
        output_dir: 출력 디렉토리
        image_size: 이미지 크기 (정사각형)
        KI, E, nu: 파괴역학 파라미터
        noise_std: 노이즈 σ (Gray Levels)
        speckle_radius: 스페클 반경
        speckle_density: 스페클 밀도
        speckle_seed: 스페클 시드
        noise_seed: 노이즈 시드

    Returns:
        dict with ref, deformed, u, v, crack_mask
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Mode I 크랙 합성 데이터 생성")
    print("=" * 60)
    print(f"  이미지 크기 : {image_size} x {image_size}")
    print(f"  KI={KI}, E={E}, nu={nu}")
    print(f"  노이즈 σ   : {noise_std} GL")
    print(f"  Numba 가속 : {'사용' if HAS_NUMBA else '미사용 (느림)'}")
    print(f"  출력 경로   : {out.resolve()}")
    print("-" * 60)

    # --- 1. 스페클 ---
    t0 = time.time()
    print("[1/5] 스페클 참조 이미지 생성...")
    ref = generate_speckle_image(image_size, image_size,
                                 speckle_radius=speckle_radius,
                                 density=speckle_density,
                                 seed=speckle_seed)

    # 참조 이미지 노이즈
    rng_ref = np.random.RandomState(noise_seed)
    ref_noisy = np.clip(ref + rng_ref.normal(0, noise_std, ref.shape), 0, 255)
    print(f"       완료 ({time.time()-t0:.2f}s)")

    # --- 2. 변위장 ---
    t0 = time.time()
    print("[2/5] Mode I 변위장 계산...")
    u, v = mode1_crack_displacement(image_size, image_size,
                                     KI=KI, E=E, nu=nu)
    v_max = np.max(np.abs(v))
    u_max = np.max(np.abs(u))
    print(f"       max|u|={u_max:.2f} px, max|v|={v_max:.2f} px")
    print(f"       크랙 최대 개구량 ≈ {2*v_max:.2f} px")
    print(f"       완료 ({time.time()-t0:.2f}s)")

    # --- 3. 순방향 워프 ---
    t0 = time.time()
    print("[3/5] 순방향 워프 (forward splatting)...")
    deformed, crack_mask = warp_image_forward(
        ref, u, v,
        noise_std=noise_std,
        seed=noise_seed + 1
    )
    n_crack_px = np.sum(crack_mask)
    print(f"       크랙 틈 픽셀: {n_crack_px} "
          f"({n_crack_px/(image_size**2)*100:.2f}%)")
    print(f"       완료 ({time.time()-t0:.2f}s)")

    # --- 4. 저장 ---
    t0 = time.time()
    print("[4/5] 파일 저장...")

    ref_8bit = ref_noisy.astype(np.uint8)
    def_8bit = deformed.astype(np.uint8)

    cv2.imwrite(str(out / 'reference.tiff'), ref_8bit)
    cv2.imwrite(str(out / 'deformed.tiff'), def_8bit)
    np.save(str(out / 'ground_truth_u.npy'), u)
    np.save(str(out / 'ground_truth_v.npy'), v)
    np.save(str(out / 'crack_mask.npy'), crack_mask)

    params = {
        'image_size': image_size,
        'KI': KI, 'E': E, 'nu': nu,
        'noise_std': noise_std,
        'speckle_radius': speckle_radius,
        'speckle_density': speckle_density,
        'speckle_seed': speckle_seed,
        'noise_seed': noise_seed,
        'u_max_px': float(u_max),
        'v_max_px': float(v_max),
        'crack_opening_max_px': float(2 * v_max),
        'crack_tip': [image_size // 2, image_size // 2],
        'n_crack_pixels': int(n_crack_px),
    }
    np.savez(str(out / 'params.npz'), **params)
    print(f"       완료 ({time.time()-t0:.2f}s)")

    # --- 5. 시각화 ---
    t0 = time.time()
    print("[5/5] 시각화 생성...")
    visualize_results(ref_noisy, deformed, u, v, crack_mask,
                      save_path=str(out / 'visualization.png'))
    print(f"       완료 ({time.time()-t0:.2f}s)")

    print("-" * 60)
    print("생성 완료!")
    print(f"  reference.tiff       : 참조 이미지")
    print(f"  deformed.tiff        : 크랙 변형 이미지")
    print(f"  ground_truth_u/v.npy : 변위장 정답")
    print(f"  crack_mask.npy       : 크랙 틈 마스크")
    print(f"  params.npz           : 생성 파라미터")
    print(f"  visualization.png    : 시각화")
    print("=" * 60)

    return {
        'ref': ref_noisy,
        'deformed': deformed,
        'u': u, 'v': v,
        'crack_mask': crack_mask,
        'output_dir': str(out),
    }


# =============================================================================
#  실행
# =============================================================================

if __name__ == '__main__':
    result = generate_crack_dataset(
        output_dir='synthetic_crack_data',
        image_size=500,
        KI=17.0, E=70.0, nu=0.3,
        noise_std=5.0,
    )

