"""
MIG (Mean Intensity Gradient) 계산 모듈

References:
- Pan et al. (2010) "Mean intensity gradient: An effective global parameter 
  for quality assessment of the speckle patterns used in digital image correlation"
"""

import numpy as np
import cv2
from typing import Tuple


def compute_mig(image: np.ndarray) -> float:
    """
    Mean Intensity Gradient 계산
    
    Sobel ksize=3을 정규화 없이 사용.
    
    MIG는 스페클 패턴 간 상대적 품질 비교 지표이므로 (Pan et al., 2010),
    gradient의 절대 스케일이 아닌 상대 크기만 의미를 가짐.
    따라서 SSSIG/IC-GN의 gradient (ksize=5, /32.0)와 의도적으로 다름:
    - ksize=3: 스페클 미세 구조(고주파)에 더 민감 → 품질 변별력 향상
    - 정규화 없음: 상대 비교 지표이므로 불필요
    
    SSSIG/IC-GN은 σ(Δu) ≈ √[D(η)/SSSIG] 관계를 위해
    동일한 gradient 설정(ksize=5, /32.0)을 사용해야 하며,
    이는 sssig.py와 icgn.py에서 보장됨.
    
    References:
        Pan et al. (2010) "Mean intensity gradient: An effective global parameter
        for quality assessment of the speckle patterns used in DIC"
        Optics and Lasers in Engineering, 48(4), 469-477.
    """
    if image is None or image.size == 0:
        return 0.0
    
    # 그레이스케일 변환
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # float 변환
    img_float = image.astype(np.float64)
    
    # Sobel gradient 계산
    gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    
    # Gradient magnitude
    magnitude = np.sqrt(gx**2 + gy**2)
    
    # Mean Intensity Gradient
    mig = np.mean(magnitude)
    
    return float(mig)


def compute_gradient(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    이미지의 x, y 방향 gradient 계산 (캐싱용)
    
    Args:
        image: 그레이스케일 이미지
    
    Returns:
        (gx, gy) gradient 배열 튜플
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    img_float = image.astype(np.float64)
    gx = cv2.Sobel(img_float, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img_float, cv2.CV_64F, 0, 1, ksize=3)
    
    return gx, gy


def compute_local_mig(image: np.ndarray, 
                      grid_size: Tuple[int, int] = (8, 8)) -> Tuple[float, np.ndarray]:
    """
    로컬 MIG 기반 균일성 계산
    
    Args:
        image: 그레이스케일 이미지
        grid_size: 그리드 분할 크기 (rows, cols)
    
    Returns:
        (uniformity_score, mig_map) 튜플
    """
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h, w = image.shape
    rows, cols = grid_size
    cell_h, cell_w = h // rows, w // cols
    
    mig_map = np.zeros((rows, cols), dtype=np.float64)
    
    for i in range(rows):
        for j in range(cols):
            y1, y2 = i * cell_h, (i + 1) * cell_h
            x1, x2 = j * cell_w, (j + 1) * cell_w
            cell = image[y1:y2, x1:x2]
            mig_map[i, j] = compute_mig(cell)
    
    mean_mig = mig_map.mean()
    if mean_mig == 0:
        return 0.0, mig_map
    
    cv = mig_map.std() / mean_mig
    uniformity = 1 / (1 + cv)
    
    return float(uniformity), mig_map
