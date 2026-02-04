"""애플리케이션 상태 관리"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List, Callable
from pathlib import Path
import numpy as np
import threading

from speckle.models import QualityReport


@dataclass
class ImageLoadingState:
    """이미지 로딩 상태"""
    is_loading: bool = False
    total_files: int = 0
    loaded_files: int = 0
    current_file: str = ""
    
    @property
    def progress(self) -> float:
        """로딩 진행률 (0~100)"""
        if self.total_files == 0:
            return 0.0
        return (self.loaded_files / self.total_files) * 100
    
    @property
    def is_complete(self) -> bool:
        """로딩 완료 여부"""
        return not self.is_loading and self.loaded_files == self.total_files


@dataclass
class AppState:
    """GUI 상태를 중앙 관리"""
    
    # 이미지 상태
    current_image: Optional[np.ndarray] = None
    current_file: str = ""
    folder_path: Optional[Path] = None
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    file_list: List[str] = field(default_factory=list)
    file_paths: List[Path] = field(default_factory=list)  # 추가: 전체 경로 목록
    current_index: int = 0
    
    # 이미지 로딩 상태 (추가)
    loading_state: ImageLoadingState = field(default_factory=ImageLoadingState)
    _loading_lock: threading.Lock = field(default_factory=threading.Lock)
    _stop_loading: bool = False
    
    # 평가 결과
    file_reports: Dict[str, QualityReport] = field(default_factory=dict)
    _reports_lock: threading.Lock = field(default_factory=threading.Lock)
    
    # ROI
    roi: Optional[Tuple[int, int, int, int]] = None
    
    # 뷰 상태
    zoom_level: float = 1.0
    pan_offset: Tuple[int, int] = (0, 0)
    
    # 배치 처리 상태
    batch_running: bool = False
    batch_stop_requested: bool = False
    
    # ===== 이미지 캐시 메서드 (추가) =====
    
    def get_image(self, filename: str) -> Optional[np.ndarray]:
        """파일명으로 이미지 가져오기 (스레드 안전)"""
        with self._loading_lock:
            return self.images.get(filename)
    
    def get_image_by_index(self, index: int) -> Optional[np.ndarray]:
        """인덱스로 이미지 가져오기"""
        if 0 <= index < len(self.file_list):
            return self.get_image(self.file_list[index])
        return None
    
    def get_image_by_path(self, path: Path) -> Optional[np.ndarray]:
        """경로로 이미지 가져오기"""
        return self.get_image(path.name)
    
    def set_image(self, filename: str, image: np.ndarray):
        """이미지 캐시에 저장 (스레드 안전)"""
        with self._loading_lock:
            self.images[filename] = image
    
    def is_all_loaded(self) -> bool:
        """모든 이미지가 로드되었는지 확인"""
        return (len(self.images) == len(self.file_list) and 
                len(self.file_list) > 0 and
                not self.loading_state.is_loading)
    
    @property
    def memory_usage_mb(self) -> float:
        """현재 캐시 메모리 사용량 (MB)"""
        with self._loading_lock:
            total = sum(img.nbytes for img in self.images.values())
            return total / (1024 * 1024)
    
    def stop_loading(self):
        """로딩 중지 요청"""
        self._stop_loading = True
    
    def should_stop_loading(self) -> bool:
        """로딩 중지 여부 확인"""
        return self._stop_loading
    
    # ===== 기존 메서드들 =====
    
    def set_report(self, filename: str, report: QualityReport):
        """스레드 안전한 리포트 저장"""
        with self._reports_lock:
            self.file_reports[filename] = report
    
    def get_report(self, filename: str) -> Optional[QualityReport]:
        """스레드 안전한 리포트 조회"""
        with self._reports_lock:
            return self.file_reports.get(filename)
    
    def get_all_reports(self) -> Dict[str, QualityReport]:
        """모든 리포트 복사본 반환"""
        with self._reports_lock:
            return dict(self.file_reports)
    
    def clear_report(self, filename: str):
        """특정 파일의 리포트 삭제"""
        with self._reports_lock:
            if filename in self.file_reports:
                del self.file_reports[filename]
    
    def clear_all_reports(self):
        """모든 리포트 삭제"""
        with self._reports_lock:
            self.file_reports.clear()
    
    def navigate(self, direction: int) -> bool:
        """이미지 탐색, 성공 여부 반환"""
        new_index = self.current_index + direction
        if 0 <= new_index < len(self.file_list):
            self.current_index = new_index
            self.current_file = self.file_list[new_index]
            self.current_image = self.images.get(self.current_file)
            return True
        return False
    
    def navigate_to(self, index: int) -> bool:
        """특정 인덱스로 이동"""
        if 0 <= index < len(self.file_list):
            self.current_index = index
            self.current_file = self.file_list[index]
            self.current_image = self.images.get(self.current_file)
            return True
        return False
    
    def reset_roi(self):
        """ROI 초기화"""
        self.roi = None
    
    def reset_view(self):
        """뷰 상태 초기화"""
        self.zoom_level = 1.0
        self.pan_offset = (0, 0)
    
    def clear(self):
        """상태 전체 초기화"""
        self._stop_loading = True  # 로딩 중지
        
        self.current_image = None
        self.current_file = ""
        self.folder_path = None
        self.images.clear()
        self.file_list.clear()
        self.file_paths.clear()
        self.current_index = 0
        self.clear_all_reports()
        self.roi = None
        self.reset_view()
        
        # 로딩 상태 초기화
        self.loading_state = ImageLoadingState()
        self._stop_loading = False
    
    def has_images(self) -> bool:
        """이미지 로드 여부"""
        return len(self.images) > 0
    
    @property
    def total_images(self) -> int:
        """총 이미지 수"""
        return len(self.file_list)
    
    @property
    def current_position(self) -> int:
        """현재 위치 (1-based)"""
        return self.current_index + 1 if self.file_list else 0
