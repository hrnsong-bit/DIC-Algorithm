"""애플리케이션 상태 관리"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, List
from pathlib import Path
import numpy as np
import threading

from speckle.models import QualityReport


@dataclass
class AppState:
    """GUI 상태를 중앙 관리"""
    
    # 이미지 상태
    current_image: Optional[np.ndarray] = None
    current_file: str = ""
    folder_path: Optional[Path] = None
    images: Dict[str, np.ndarray] = field(default_factory=dict)
    file_list: List[str] = field(default_factory=list)
    current_index: int = 0
    
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
            self.current_image = self.images[self.current_file]
            return True
        return False
    
    def navigate_to(self, index: int) -> bool:
        """특정 인덱스로 이동"""
        if 0 <= index < len(self.file_list):
            self.current_index = index
            self.current_file = self.file_list[index]
            self.current_image = self.images[self.current_file]
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
        self.current_image = None
        self.current_file = ""
        self.folder_path = None
        self.images.clear()
        self.file_list.clear()
        self.current_index = 0
        self.clear_all_reports()
        self.roi = None
        self.reset_view()
    
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
