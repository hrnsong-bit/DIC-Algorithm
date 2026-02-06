"""설정 저장/불러오기 관리"""

import json
from pathlib import Path
from typing import Dict, Any


class SettingsManager:
    """애플리케이션 설정 관리"""
    
    DEFAULT_SETTINGS = {
        # DIC 파라미터
        'subset_size': 21,
        'spacing': 16,
        'search_range': 50,
        'zncc_threshold': 0.7,
        'shape_function': 'affine',
        'interpolation': 'bicubic',
        'gaussian_blur': None,
        'gaussian_blur_enabled': False,
        'conv_threshold': 0.002,
        'max_iter': 20,
        
        # 품질평가 파라미터
        'quality_subset_size': 21,
        'quality_spacing': 10,
        'mig_threshold': 0.2,
        'sssig_threshold': 1e5,
        
        # 마지막 경로
        'last_folder': '',
    }
    
    def __init__(self):
        # 사용자 홈 디렉토리에 설정 저장
        config_dir = Path.home() / '.speckle_analysis'
        config_dir.mkdir(exist_ok=True)
        self.config_path = config_dir / 'settings.json'
        
        self.settings = self.DEFAULT_SETTINGS.copy()
        self.load()
    
    def load(self):
        """설정 파일 로드"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    saved = json.load(f)
                    self.settings.update(saved)
                print(f"[INFO] 설정 로드: {self.config_path}")
        except Exception as e:
            print(f"[WARN] 설정 로드 실패: {e}")
    
    def save(self):
        """설정 파일 저장"""
        try:
            with open(self.config_path, 'w', encoding='utf-8') as f:
                json.dump(self.settings, f, indent=2, ensure_ascii=False)
            print(f"[INFO] 설정 저장: {self.config_path}")
        except Exception as e:
            print(f"[WARN] 설정 저장 실패: {e}")
    
    def get(self, key: str, default=None):
        """설정값 가져오기"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value):
        """설정값 설정"""
        self.settings[key] = value
    
    def update(self, params: Dict[str, Any]):
        """여러 설정값 업데이트"""
        self.settings.update(params)
    
    def get_dic_params(self) -> Dict[str, Any]:
        """DIC 파라미터만 반환"""
        keys = ['subset_size', 'spacing', 'search_range', 'zncc_threshold',
                'shape_function', 'interpolation', 'gaussian_blur', 
                'gaussian_blur_enabled', 'conv_threshold', 'max_iter']
        return {k: self.settings.get(k) for k in keys}
    
    def get_quality_params(self) -> Dict[str, Any]:
        """품질평가 파라미터만 반환"""
        keys = ['quality_subset_size', 'quality_spacing', 
                'mig_threshold', 'sssig_threshold']
        return {k: self.settings.get(k) for k in keys}
