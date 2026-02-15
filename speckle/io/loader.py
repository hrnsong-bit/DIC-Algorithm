"""
이미지 입출력 모듈

유니코드 경로 지원
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Union

_logger = logging.getLogger(__name__)


def load_image(path: Union[str, Path]) -> Optional[np.ndarray]:
    """
    이미지 로드 (유니코드 경로 지원)
    
    Args:
        path: 이미지 파일 경로
    
    Returns:
        이미지 배열 또는 None (실패 시)
    """
    path = Path(path)
    
    if not path.exists():
        return None
    
    try:
        # 유니코드 경로 지원
        with open(path, 'rb') as f:
            data = np.frombuffer(f.read(), dtype=np.uint8)
        
        image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        return image
    
    except Exception as e:
        _logger.warning(f"이미지 로드 실패 '{path}': {e}")
        return None


def load_image_grayscale(path: Union[str, Path]) -> Optional[np.ndarray]:
    """그레이스케일로 이미지 로드"""
    image = load_image(path)
    if image is not None and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image


def load_folder(folder_path: Union[str, Path],
                extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                ) -> Dict[str, np.ndarray]:
    """
    폴더 내 모든 이미지 로드
    
    Args:
        folder_path: 폴더 경로
        extensions: 지원 확장자 튜플
    
    Returns:
        {파일명: 이미지} 딕셔너리
    """
    folder = Path(folder_path)
    images = {}
    
    if not folder.exists() or not folder.is_dir():
        return images
    
    # 정렬된 파일 목록
    files = sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ])
    
    for file_path in files:
        image = load_image(file_path)
        if image is not None:
            images[file_path.name] = image
    
    return images


def save_image(image: np.ndarray, path: Union[str, Path]) -> bool:
    """
    이미지 저장 (유니코드 경로 지원)
    
    Args:
        image: 이미지 배열
        path: 저장 경로
    
    Returns:
        성공 여부
    """
    path = Path(path)
    
    try:
        # 유니코드 경로 지원
        ext = path.suffix.lower()
        success, encoded = cv2.imencode(ext, image)
        
        if success:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'wb') as f:
                f.write(encoded.tobytes())
            return True
        return False
    
    except Exception as e:
        _logger.error(f"이미지 저장 실패 '{path}': {e}")
        return False


def get_image_files(folder_path: Union[str, Path],
                    extensions: Tuple[str, ...] = ('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')
                    ) -> List[Path]:
    """폴더 내 이미지 파일 경로 목록 반환"""
    folder = Path(folder_path)
    
    if not folder.exists():
        return []
    
    return sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    ])
