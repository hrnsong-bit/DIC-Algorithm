"""
합성 이미지 테스트

다양한 결함 패턴의 합성 이미지 생성 및 검증
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from speckle import SpeckleQualityAssessor, load_image, save_image


class SyntheticSpeckleGenerator:
    """합성 스페클 이미지 생성기"""
    
    def __init__(self, size: Tuple[int, int] = (512, 512)):
        self.size = size
        self.h, self.w = size
    
    def _add_noise(self, image: np.ndarray, std: float = 5.0) -> np.ndarray:
        """가우시안 노이즈 추가"""
        noise = np.random.normal(0, std, image.shape)
        noisy = image.astype(np.float64) + noise
        return np.clip(noisy, 0, 255).astype(np.uint8)
    
    def good_speckle(self, speckle_size: int = 5, density: float = 0.3) -> np.ndarray:
        """양호한 스페클 패턴"""
        image = np.ones(self.size, dtype=np.uint8) * 128
        
        n_speckles = int(self.h * self.w * density / (speckle_size ** 2))
        
        for _ in range(n_speckles):
            x = np.random.randint(0, self.w)
            y = np.random.randint(0, self.h)
            radius = np.random.randint(speckle_size // 2, speckle_size)
            color = np.random.choice([0, 255])
            cv2.circle(image, (x, y), radius, int(color), -1)
        
        return self._add_noise(image)
    
    def low_contrast(self, contrast: float = 0.3) -> np.ndarray:
        """낮은 대비"""
        base = self.good_speckle()
        mean = 128
        low_contrast = ((base.astype(np.float64) - mean) * contrast + mean)
        return np.clip(low_contrast, 0, 255).astype(np.uint8)
    
    def biased_coverage(self, side: str = 'left') -> np.ndarray:
        """편향된 커버리지"""
        image = np.ones(self.size, dtype=np.uint8) * 128
        
        # 스페클이 있는 영역
        if side == 'left':
            region = image[:, :self.w // 2]
        elif side == 'right':
            region = image[:, self.w // 2:]
        elif side == 'top':
            region = image[:self.h // 2, :]
        else:
            region = image[self.h // 2:, :]
        
        # 해당 영역에만 스페클 추가
        n_speckles = int(region.shape[0] * region.shape[1] * 0.3 / 25)
        for _ in range(n_speckles):
            x = np.random.randint(0, region.shape[1])
            y = np.random.randint(0, region.shape[0])
            radius = np.random.randint(2, 5)
            color = np.random.choice([0, 255])
            cv2.circle(region, (x, y), radius, int(color), -1)
        
        return self._add_noise(image)
    
    def center_empty(self) -> np.ndarray:
        """중앙 빈 영역"""
        image = self.good_speckle()
        
        # 중앙 영역 지우기
        margin = self.w // 4
        image[margin:-margin, margin:-margin] = 128
        
        return image
    
    def sparse_dots(self, n_dots: int = 50) -> np.ndarray:
        """희소한 점"""
        image = np.ones(self.size, dtype=np.uint8) * 128
        
        for _ in range(n_dots):
            x = np.random.randint(0, self.w)
            y = np.random.randint(0, self.h)
            color = np.random.choice([0, 255])
            cv2.circle(image, (x, y), 2, int(color), -1)
        
        return self._add_noise(image)


def generate_test_images(output_dir: str = "test_images") -> Dict[str, str]:
    """테스트 이미지 생성"""
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    generator = SyntheticSpeckleGenerator((512, 512))
    
    test_cases = {
        "01_good_standard": ("good", generator.good_speckle()),
        "10_low_contrast": ("mig_fail", generator.low_contrast(0.2)),
        "20_biased_left": ("coverage_fail", generator.biased_coverage('left')),
        "21_biased_right": ("coverage_fail", generator.biased_coverage('right')),
        "22_center_empty": ("coverage_fail", generator.center_empty()),
        "30_sparse_dots": ("sssig_fail", generator.sparse_dots(30)),
    }
    
    metadata = {}
    
    for filename, (expected, image) in test_cases.items():
        path = output_path / f"{filename}.png"
        save_image(image, path)
        metadata[filename] = expected
        print(f"생성됨: {path}")
    
    # 메타데이터 저장
    with open(output_path / "test_cases.txt", 'w') as f:
        for name, expected in metadata.items():
            f.write(f"{name}: {expected}\n")
    
    return metadata


def run_validation(image_dir: str = "test_images"):
    """검증 실행"""
    print("\n===== 검증 시작 =====\n")
    
    assessor = SpeckleQualityAssessor()
    image_path = Path(image_dir)
    
    results = []
    
    for img_file in sorted(image_path.glob("*.png")):
        image = load_image(img_file)
        if image is None:
            continue
        
        report = assessor.evaluate(image)
        
        status = "PASS" if report.analyzable else "FAIL"
        print(f"{img_file.name}: {status}")
        print(f"  MIG: {report.mig:.2f}, SSSIG min: ", end="")
        if report.sssig_result:
            print(f"{report.sssig_result.min:.2e}")
        else:
            print("N/A")
        print()
        
        results.append((img_file.name, report))
    
    return results


if __name__ == "__main__":
    generate_test_images()
    run_validation()
