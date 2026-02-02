"""
스페클 품질 평가 데이터 모델
- BadPoint: 불량 POI 정보
- SSSIGResult: SSSIG 맵 계산 결과
- QualityReport: 단일 이미지 평가 결과
- BatchReport: 배치 평가 결과
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import numpy as np


@dataclass
class BadPoint:
    """SSSIG 임계값 미달 포인트"""
    y: int
    x: int
    sssig: float
    
    def __str__(self) -> str:
        return f"({self.y}, {self.x}) SSSIG: {self.sssig:.2e}"


@dataclass
class SSSIGResult:
    """SSSIG 맵 계산 결과"""
    map: np.ndarray                    # 2D SSSIG 맵
    mean: float                        # 평균 SSSIG
    min: float                         # 최소 SSSIG
    max: float                         # 최대 SSSIG
    bad_points: List[BadPoint]         # 임계값 미달 포인트 목록
    points_y: np.ndarray               # POI y 좌표 배열
    points_x: np.ndarray               # POI x 좌표 배열
    subset_size: int = 21              # 사용된 subset 크기
    spacing: int = 10                  # 사용된 spacing
    
    @property
    def n_points(self) -> int:
        return len(self.points_y)
    
    @property
    def n_bad_points(self) -> int:
        return len(self.bad_points)
    
    @property
    def pass_rate(self) -> float:
        if self.n_points == 0:
            return 0.0
        return (self.n_points - self.n_bad_points) / self.n_points * 100


@dataclass
class QualityReport:
    """단일 이미지 품질 평가 결과"""
    # MIG 결과
    mig: float
    mig_pass: bool
    mig_threshold: float
    
    # SSSIG 결과
    sssig_result: Optional[SSSIGResult]
    sssig_pass: bool
    sssig_threshold: float
    
    # Subset 크기 추천
    recommended_subset_size: int
    subset_size_found: bool
    current_subset_size: int
    
    # 종합 판정
    analyzable: bool
    quality_grade: str = "Unknown"
    warnings: List[str] = field(default_factory=list)
    
    # 메타데이터
    image_shape: tuple = (0, 0)
    processing_time: float = 0.0
    
    def __post_init__(self):
        self._update_grade()
    
    def _update_grade(self):
        if not self.mig_pass:
            self.quality_grade = "FAIL - Low Contrast"
        elif not self.sssig_pass:
            self.quality_grade = "FAIL - Local Quality"
        elif not self.subset_size_found:
            self.quality_grade = "WARNING - Large Subset Needed"
        else:
            self.quality_grade = "PASS"
        
        self.analyzable = self.mig_pass and self.sssig_pass


@dataclass
class BatchReport:
    """배치 평가 결과"""
    total_images: int
    passed_images: int
    failed_images: int
    
    # 글로벌 최적 subset size
    global_recommended_size: int
    global_size_found: bool
    worst_case_file: str
    
    # 개별 결과
    individual_reports: Dict[str, QualityReport] = field(default_factory=dict)
    
    # 통계
    mig_stats: Dict[str, float] = field(default_factory=dict)
    sssig_stats: Dict[str, float] = field(default_factory=dict)
    
    # 메타데이터
    total_processing_time: float = 0.0
    
    @property
    def pass_rate(self) -> float:
        if self.total_images == 0:
            return 0.0
        return self.passed_images / self.total_images * 100
    
    @property
    def summary(self) -> str:
        return (
            f"총 {self.total_images}개 이미지 중 "
            f"{self.passed_images}개 통과 ({self.pass_rate:.1f}%)\n"
            f"글로벌 권장 Subset Size: {self.global_recommended_size}px\n"
            f"Worst Case: {self.worst_case_file}"
        )
