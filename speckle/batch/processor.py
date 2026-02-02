"""
배치 처리 모듈

여러 이미지 일괄 평가 및 글로벌 최적 파라미터 계산
"""

import numpy as np
from pathlib import Path
from typing import Optional, Callable, Dict, List, Tuple, Union
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

from ..core.assessor import SpeckleQualityAssessor
from ..models.reports import QualityReport, BatchReport
from ..io.loader import load_image, get_image_files


def evaluate_single_file(args: Tuple) -> Tuple[str, Optional[QualityReport]]:
    """
    단일 파일 평가 (멀티프로세싱용)
    
    Args:
        args: (file_path, roi, assessor_params)
    
    Returns:
        (filename, report) 튜플
    """
    file_path, roi, assessor_params = args
    
    try:
        image = load_image(file_path)
        if image is None:
            return (Path(file_path).name, None)
        
        assessor = SpeckleQualityAssessor(**assessor_params)
        report = assessor.evaluate(image, roi)
        
        return (Path(file_path).name, report)
    
    except Exception as e:
        print(f"평가 실패: {file_path}, 오류: {e}")
        return (Path(file_path).name, None)


class BatchProcessor:
    """
    배치 이미지 처리기
    
    Usage:
        processor = BatchProcessor()
        report = processor.process_folder("./images", roi=(0, 0, 512, 512))
    """
    
    def __init__(self,
                 mig_threshold: float = 30.0,
                 sssig_threshold: float = 1e5,
                 subset_size: int = 21,
                 poi_spacing: int = 10,
                 n_workers: Optional[int] = None):
        """
        Args:
            mig_threshold: MIG 임계값
            sssig_threshold: SSSIG 임계값
            subset_size: 초기 subset 크기
            poi_spacing: POI 간격
            n_workers: 병렬 처리 워커 수 (None = CPU 코어 수 - 1)
        """
        self.assessor_params = {
            'mig_threshold': mig_threshold,
            'sssig_threshold': sssig_threshold,
            'subset_size': subset_size,
            'poi_spacing': poi_spacing
        }
        
        if n_workers is None:
            self.n_workers = max(1, multiprocessing.cpu_count() - 1)
        else:
            self.n_workers = n_workers
    
    def process_folder(self,
                       folder_path: Union[str, Path],
                       roi: Optional[Tuple[int, int, int, int]] = None,
                       progress_callback: Optional[Callable[[int, int, str], None]] = None
                       ) -> BatchReport:
        """
        폴더 내 모든 이미지 배치 처리
        
        Args:
            folder_path: 이미지 폴더 경로
            roi: 공통 ROI
            progress_callback: 진행 콜백 (current, total, filename)
        
        Returns:
            BatchReport 객체
        """
        import time
        start_time = time.time()
        
        files = get_image_files(folder_path)
        total = len(files)
        
        if total == 0:
            return BatchReport(
                total_images=0,
                passed_images=0,
                failed_images=0,
                global_recommended_size=21,
                global_size_found=False,
                worst_case_file=""
            )
        
        individual_reports = {}
        max_recommended_size = 11
        worst_file = ""
        passed = 0
        failed = 0
        
        # 순차 처리 (GUI 콜백 지원)
        assessor = SpeckleQualityAssessor(**self.assessor_params)
        
        for idx, file_path in enumerate(files):
            filename = file_path.name
            
            if progress_callback:
                progress_callback(idx + 1, total, filename)
            
            image = load_image(file_path)
            if image is None:
                failed += 1
                continue
            
            report = assessor.evaluate(image, roi)
            individual_reports[filename] = report
            
            if report.analyzable:
                passed += 1
            else:
                failed += 1
            
            if report.recommended_subset_size > max_recommended_size:
                max_recommended_size = report.recommended_subset_size
                worst_file = filename
        
        global_size_found = all(
            r.subset_size_found for r in individual_reports.values()
            if r is not None
        )
        
        # MIG/SSSIG 통계
        mig_values = [r.mig for r in individual_reports.values()]
        sssig_min_values = [
            r.sssig_result.min for r in individual_reports.values()
            if r.sssig_result is not None
        ]
        
        mig_stats = {
            'mean': float(np.mean(mig_values)) if mig_values else 0.0,
            'min': float(np.min(mig_values)) if mig_values else 0.0,
            'max': float(np.max(mig_values)) if mig_values else 0.0
        }
        
        sssig_stats = {
            'mean': float(np.mean(sssig_min_values)) if sssig_min_values else 0.0,
            'min': float(np.min(sssig_min_values)) if sssig_min_values else 0.0,
            'max': float(np.max(sssig_min_values)) if sssig_min_values else 0.0
        }
        
        return BatchReport(
            total_images=total,
            passed_images=passed,
            failed_images=failed,
            global_recommended_size=max_recommended_size,
            global_size_found=global_size_found,
            worst_case_file=worst_file,
            individual_reports=individual_reports,
            mig_stats=mig_stats,
            sssig_stats=sssig_stats,
            total_processing_time=time.time() - start_time
        )
    
    def process_images(self,
                       images: Dict[str, np.ndarray],
                       roi: Optional[Tuple[int, int, int, int]] = None,
                       progress_callback: Optional[Callable[[int, int, str], None]] = None
                       ) -> BatchReport:
        """
        이미지 딕셔너리 배치 처리
        
        Args:
            images: {파일명: 이미지} 딕셔너리
            roi: 공통 ROI
            progress_callback: 진행 콜백
        
        Returns:
            BatchReport 객체
        """
        assessor = SpeckleQualityAssessor(**self.assessor_params)
        return assessor.evaluate_batch(images, roi, progress_callback)
