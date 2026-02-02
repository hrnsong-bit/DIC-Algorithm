"""스페클 품질 평가 메인 클래스"""

import numpy as np
import cv2
import time
from typing import Optional, Tuple, Callable, Dict

from ..models.reports import BadPoint, SSSIGResult, QualityReport, BatchReport
from ..utils.logger import logger
from .mig import compute_mig
from .sssig import compute_sssig_map, find_optimal_subset_size, warmup_numba


class SpeckleQualityAssessor:
    """
    스페클 패턴 품질 평가기
    
    평가 흐름:
    1. MIG로 ROI 내 전역 대비 확인
    2. SSSIG 맵으로 ROI 내 로컬 품질 확인
    3. 최적 subset size 추천
    
    Note: ROI를 시편 내부(게이지부)에 정확히 설정해야 함
    """
    
    def __init__(self,
                 mig_threshold: float = 30.0,
                 sssig_threshold: float = 1e5,
                 subset_size: int = 21,
                 poi_spacing: int = 10,
                 auto_find_size: bool = True):
        """
        Args:
            mig_threshold: MIG 임계값 (기본 30.0)
            sssig_threshold: SSSIG 임계값 (기본 1e5)
            subset_size: 초기 subset 크기 (기본 21)
            poi_spacing: POI 간격 (기본 10)
            auto_find_size: 자동 최적 크기 탐색 여부
        """
        self.mig_threshold = mig_threshold
        self.sssig_threshold = sssig_threshold
        self.subset_size = subset_size
        self.poi_spacing = poi_spacing
        self.auto_find_size = auto_find_size
        
        warmup_numba()
        logger.debug(f"SpeckleQualityAssessor 초기화: MIG={mig_threshold}, SSSIG={sssig_threshold}")
    
    def evaluate(self, image: np.ndarray, 
                 roi: Optional[Tuple[int, int, int, int]] = None) -> QualityReport:
        """
        단일 이미지 품질 평가
        
        Args:
            image: 입력 이미지
            roi: ROI 영역 (x, y, w, h) - 시편 내부에 설정 권장
        
        Returns:
            QualityReport 객체
        """
        start_time = time.time()
        warnings = []
        
        # 그레이스케일 변환
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # ROI 적용
        if roi is not None:
            x, y, w, h = roi
            roi_gray = gray[y:y+h, x:x+w]
        else:
            roi_gray = gray
        
        image_shape = roi_gray.shape
        
        # ROI 밝기 체크 (배경 포함 경고)
        mean_intensity = np.mean(roi_gray)
        if mean_intensity < 30:
            warnings.append(f"ROI 평균 밝기가 낮음 ({mean_intensity:.1f}) - 배경 포함 가능성")
        
        # ====== Stage 1: MIG ======
        mig = compute_mig(roi_gray)
        mig_pass = mig >= self.mig_threshold
        
        if not mig_pass:
            logger.warning(f"MIG 실패: {mig:.2f} < {self.mig_threshold}")
            return QualityReport(
                mig=mig,
                mig_pass=False,
                mig_threshold=self.mig_threshold,
                sssig_result=None,
                sssig_pass=False,
                sssig_threshold=self.sssig_threshold,
                recommended_subset_size=self.subset_size,
                subset_size_found=False,
                current_subset_size=self.subset_size,
                analyzable=False,
                warnings=warnings,
                image_shape=image_shape,
                processing_time=time.time() - start_time
            )
        
        # ====== Stage 2: SSSIG ======
        if self.auto_find_size:
            optimal_size, size_found, sssig_result = find_optimal_subset_size(
                roi_gray,
                spacing=self.poi_spacing,
                threshold=self.sssig_threshold,
                min_size=11,
                max_size=61
            )
            recommended_size = optimal_size
        else:
            sssig_result = compute_sssig_map(
                roi_gray,
                subset_size=self.subset_size,
                spacing=self.poi_spacing,
                threshold=self.sssig_threshold
            )
            recommended_size = self.subset_size
            size_found = len(sssig_result.bad_points) == 0
        
        if sssig_result and len(sssig_result.points_y) == 0:
            warnings.append("유효한 POI가 없음 - ROI 확인 필요")
        
        sssig_pass = sssig_result.n_bad_points == 0 if sssig_result else False
        
        if not sssig_pass and sssig_result and sssig_result.n_bad_points > 0:
            warnings.append(
                f"불량 POI {sssig_result.n_bad_points}개 "
                f"(min: {sssig_result.min:.2e})"
            )
        
        if not size_found:
            warnings.append("최대 크기(61px)로도 모든 POI 통과 불가")
        
        processing_time = time.time() - start_time
        logger.info(f"평가 완료: MIG={mig:.2f}, 처리시간={processing_time:.3f}s")
        
        return QualityReport(
            mig=mig,
            mig_pass=mig_pass,
            mig_threshold=self.mig_threshold,
            sssig_result=sssig_result,
            sssig_pass=sssig_pass,
            sssig_threshold=self.sssig_threshold,
            recommended_subset_size=recommended_size,
            subset_size_found=size_found,
            current_subset_size=self.subset_size,
            analyzable=mig_pass and sssig_pass,
            warnings=warnings,
            image_shape=image_shape,
            processing_time=processing_time
        )
    
    def evaluate_batch(self, 
                       images: Dict[str, np.ndarray],
                       roi: Optional[Tuple[int, int, int, int]] = None,
                       progress_callback: Optional[Callable[[int, int, str], None]] = None
                       ) -> BatchReport:
        """배치 평가"""
        start_time = time.time()
        
        total = len(images)
        individual_reports = {}
        max_recommended_size = 11
        worst_file = ""
        passed = 0
        failed = 0
        
        mig_values = []
        sssig_min_values = []
        
        logger.info(f"배치 평가 시작: {total}개 이미지")
        
        for idx, (filename, image) in enumerate(images.items()):
            if progress_callback:
                progress_callback(idx + 1, total, filename)
            
            try:
                report = self.evaluate(image, roi)
                individual_reports[filename] = report
                
                mig_values.append(report.mig)
                
                if report.sssig_result and len(report.sssig_result.points_y) > 0:
                    sssig_min_values.append(report.sssig_result.min)
                
                if report.analyzable:
                    passed += 1
                else:
                    failed += 1
                
                if report.recommended_subset_size > max_recommended_size:
                    max_recommended_size = report.recommended_subset_size
                    worst_file = filename
                    
            except cv2.error as e:
                logger.error(f"OpenCV 오류 ({filename}): {e}")
                failed += 1
            except ValueError as e:
                logger.error(f"값 오류 ({filename}): {e}")
                failed += 1
            except Exception as e:
                logger.exception(f"예상치 못한 오류 ({filename})")
                failed += 1
        
        global_size_found = all(
            r.subset_size_found for r in individual_reports.values()
        )
        
        total_time = time.time() - start_time
        logger.info(f"배치 평가 완료: {passed}/{total} 통과, 처리시간={total_time:.2f}s")
        
        return BatchReport(
            total_images=total,
            passed_images=passed,
            failed_images=failed,
            global_recommended_size=max_recommended_size,
            global_size_found=global_size_found,
            worst_case_file=worst_file,
            individual_reports=individual_reports,
            mig_stats={
                'mean': float(np.mean(mig_values)) if mig_values else 0.0,
                'min': float(np.min(mig_values)) if mig_values else 0.0,
                'max': float(np.max(mig_values)) if mig_values else 0.0
            },
            sssig_stats={
                'mean': float(np.mean(sssig_min_values)) if sssig_min_values else 0.0,
                'min': float(np.min(sssig_min_values)) if sssig_min_values else 0.0,
                'max': float(np.max(sssig_min_values)) if sssig_min_values else 0.0
            },
            total_processing_time=total_time
        )
