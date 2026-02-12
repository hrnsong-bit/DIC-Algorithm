"""
스페클 품질 평가 데이터 모델

변경 이력
---------
v3.3  - QualityReport에 noise_method 필드 추가
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
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
    map: np.ndarray
    mean: float
    min: float
    max: float
    bad_points: List[BadPoint]
    points_y: np.ndarray
    points_x: np.ndarray
    subset_size: int = 21
    spacing: int = 10
    noise_variance: float = 4.0
    threshold: float = 1e5
    predicted_accuracy: float = 0.01

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
    # MIG
    mig: float
    mig_pass: bool
    mig_threshold: float

    # SSSIG
    sssig_result: Optional[SSSIGResult]
    sssig_pass: bool
    sssig_threshold: float

    # Subset 추천
    recommended_subset_size: int
    subset_size_found: bool
    current_subset_size: int

    # 종합
    analyzable: bool
    quality_grade: str = "Unknown"
    warnings: List[str] = field(default_factory=list)

    # 메타
    image_shape: tuple = (0, 0)
    processing_time: float = 0.0

    # 노이즈
    noise_variance: Optional[float] = None
    predicted_accuracy: Optional[float] = None
    noise_method: str = 'unknown'       # ← 추가: 'user' | 'pair' | 'single_local_std'

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

    global_recommended_size: int
    global_size_found: bool
    worst_case_file: str

    individual_reports: Dict[str, QualityReport] = field(default_factory=dict)

    mig_stats: Dict[str, float] = field(default_factory=dict)
    sssig_stats: Dict[str, float] = field(default_factory=dict)
    noise_stats: Dict[str, float] = field(default_factory=dict)
    accuracy_stats: Dict[str, float] = field(default_factory=dict)

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
