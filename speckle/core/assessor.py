"""
스페클 품질 평가 메인 클래스 (v3.3.0 수정)

수정 사항:
- 3-tier 노이즈 추정: user override > pair > single(local_std)
- set_noise_variance / set_noise_pair 사후 설정 메서드
- noise_method 기록 → QualityReport에 전달
- __init__에 noise_variance 파라미터 추가
"""

import numpy as np
import cv2
import time
from typing import Optional, Tuple, Callable, Dict

from ..models.reports import BadPoint, SSSIGResult, QualityReport, BatchReport
from ..utils.logger import logger
from .mig import compute_mig
from .sssig import (
    compute_sssig_map,
    find_optimal_subset_size,
    warmup_numba,
    estimate_noise_variance,
    estimate_noise_from_pair,
    calculate_sssig_threshold,
)


class SpeckleQualityAssessor:
    """
    스페클 패턴 품질 평가기

    평가 흐름:
    1. 노이즈 분산 결정 (3-tier fallback)
    2. MIG로 ROI 내 전역 대비 확인
    3. SSSIG 맵으로 ROI 내 로컬 품질 확인
    4. 최적 subset size 추천
    """

    def __init__(self,
                 mig_threshold: float = 30.0,
                 sssig_threshold: float = 1e5,
                 subset_size: int = 21,
                 poi_spacing: int = 10,
                 auto_find_size: bool = True,
                 desired_accuracy: float = 0.02,
                 noise_variance: Optional[float] = None):
        """
        Args:
            mig_threshold: MIG 임계값
            sssig_threshold: SSSIG 임계값 (자동 계산 시 무시)
            subset_size: 초기 subset 크기
            poi_spacing: POI 간격
            auto_find_size: 자동 최적 크기 탐색 여부
            desired_accuracy: 원하는 변위 측정 정확도 (pixels)
            noise_variance: 노이즈 분산 D(η) 사전 지정 (None이면 자동)
        """
        self.mig_threshold = mig_threshold
        self.sssig_threshold = sssig_threshold
        self.subset_size = subset_size
        self.poi_spacing = poi_spacing
        self.auto_find_size = auto_find_size
        self.desired_accuracy = desired_accuracy

        # 노이즈 관련
        self._noise_variance_override: Optional[float] = noise_variance
        self._noise_pair: Optional[Tuple[np.ndarray, np.ndarray]] = None

        warmup_numba()
        logger.debug(
            f"SpeckleQualityAssessor 초기화: MIG={mig_threshold}, "
            f"desired_accuracy={desired_accuracy}, "
            f"noise_override={noise_variance}")

    # ── 노이즈 설정 API ──

    def set_noise_variance(self, variance: float):
        """사용자/외부에서 노이즈 분산을 직접 지정"""
        self._noise_variance_override = variance
        logger.info(f"노이즈 분산 설정: D(η)={variance:.2f}")

    def set_noise_pair(self, image1: np.ndarray, image2: np.ndarray):
        """pair 이미지를 설정 (evaluate 시 자동 사용)"""
        self._noise_pair = (image1, image2)
        logger.info("노이즈 pair 이미지 설정 완료")

    def clear_noise_override(self):
        """노이즈 override 해제 → 자동 추정으로 복귀"""
        self._noise_variance_override = None
        self._noise_pair = None
        logger.info("노이즈 override 해제")

    def _resolve_noise_variance(self, roi_gray: np.ndarray,
                                 roi: Optional[Tuple[int, int, int, int]] = None
                                 ) -> Tuple[float, str]:
        """
        3-tier 노이즈 분산 결정

        Tier 1: 사용자 직접 지정 (set_noise_variance)
        Tier 2: pair 이미지 차분 (set_noise_pair)
        Tier 3: 단일 이미지 local_std (fallback)

        Returns:
            (noise_variance, method_name)
        """
        # Tier 1: 사용자 override
        if self._noise_variance_override is not None:
            return self._noise_variance_override, 'user'

        # Tier 2: pair 이미지
        if self._noise_pair is not None:
            try:
                nv = estimate_noise_from_pair(
                    self._noise_pair[0], self._noise_pair[1], roi)
                return nv, 'pair'
            except Exception as e:
                logger.warning(f"Pair 노이즈 추정 실패, fallback: {e}")

        # Tier 3: 단일 이미지
        nv = estimate_noise_variance(roi_gray, method='local_std')
        return nv, 'single_local_std'

    # ── 평가 ──

    def evaluate(self, image: np.ndarray,
                 roi: Optional[Tuple[int, int, int, int]] = None
                 ) -> QualityReport:
        """단일 이미지 품질 평가"""
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

        # ROI 밝기 체크
        mean_intensity = np.mean(roi_gray)
        if mean_intensity < 30:
            warnings.append(
                f"ROI 평균 밝기가 낮음 ({mean_intensity:.1f}) "
                f"- 배경 포함 가능성")

        # ====== 노이즈 분산 결정 (3-tier) ======
        noise_variance, noise_method = self._resolve_noise_variance(
            roi_gray, roi)
        calculated_threshold = calculate_sssig_threshold(
            noise_variance, self.desired_accuracy)

        # 단일 이미지 추정 시 정보 경고 (1회성, 로그로만)
        if noise_method == 'single_local_std':
            logger.info(
                "노이즈: 단일이미지 추정(local_std) 사용 중. "
                "pair 추정 권장.")

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
                sssig_threshold=calculated_threshold,
                recommended_subset_size=self.subset_size,
                subset_size_found=False,
                current_subset_size=self.subset_size,
                analyzable=False,
                warnings=warnings,
                image_shape=image_shape,
                processing_time=time.time() - start_time,
                noise_variance=noise_variance,
                predicted_accuracy=None,
                noise_method=noise_method,
            )

        # ====== Stage 2: SSSIG ======
        if self.auto_find_size:
            optimal_size, size_found, _, info = find_optimal_subset_size(
                roi_gray,
                spacing=self.poi_spacing,
                desired_accuracy=self.desired_accuracy,
                noise_variance=noise_variance,
                min_size=11,
                max_size=61,
            )
            recommended_size = optimal_size

            sssig_result = compute_sssig_map(
                roi_gray,
                subset_size=self.subset_size,
                spacing=self.poi_spacing,
                noise_variance=noise_variance,
                desired_accuracy=self.desired_accuracy,
            )
            predicted_accuracy = (sssig_result.predicted_accuracy
                                  if sssig_result else None)
        else:
            sssig_result = compute_sssig_map(
                roi_gray,
                subset_size=self.subset_size,
                spacing=self.poi_spacing,
                noise_variance=noise_variance,
                desired_accuracy=self.desired_accuracy,
            )
            recommended_size = self.subset_size
            size_found = len(sssig_result.bad_points) == 0
            predicted_accuracy = (sssig_result.predicted_accuracy
                                  if sssig_result else None)

        if sssig_result and len(sssig_result.points_y) == 0:
            warnings.append("유효한 POI가 없음 - ROI 확인 필요")

        sssig_pass = (sssig_result.n_bad_points == 0
                      if sssig_result else False)

        if not sssig_pass and sssig_result and sssig_result.n_bad_points > 0:
            warnings.append(
                f"불량 POI {sssig_result.n_bad_points}개 "
                f"(min: {sssig_result.min:.2e})")

        if not size_found:
            warnings.append("최대 크기(61px)로도 모든 POI 통과 불가")

        if predicted_accuracy and predicted_accuracy > self.desired_accuracy:
            warnings.append(
                f"예상 정확도 {predicted_accuracy:.4f}px > "
                f"목표 {self.desired_accuracy}px")

        processing_time = time.time() - start_time
        logger.info(
            f"평가 완료: MIG={mig:.2f}, "
            f"D(η)={noise_variance:.2f} [{noise_method}], "
            f"처리시간={processing_time:.3f}s")

        return QualityReport(
            mig=mig,
            mig_pass=mig_pass,
            mig_threshold=self.mig_threshold,
            sssig_result=sssig_result,
            sssig_pass=sssig_pass,
            sssig_threshold=calculated_threshold,
            recommended_subset_size=recommended_size,
            subset_size_found=size_found,
            current_subset_size=self.subset_size,
            analyzable=mig_pass and sssig_pass,
            warnings=warnings,
            image_shape=image_shape,
            processing_time=processing_time,
            noise_variance=noise_variance,
            predicted_accuracy=predicted_accuracy,
            noise_method=noise_method,
        )

    # ── 배치 평가 ──

    def evaluate_batch(self,
                       images: Dict[str, np.ndarray],
                       roi: Optional[Tuple[int, int, int, int]] = None,
                       progress_callback: Optional[
                           Callable[[int, int, str], None]] = None
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
        noise_variances = []
        predicted_accuracies = []

        logger.info(f"배치 평가 시작: {total}개 이미지")

        for idx, (filename, image) in enumerate(images.items()):
            if progress_callback:
                progress_callback(idx + 1, total, filename)

            try:
                report = self.evaluate(image, roi)
                individual_reports[filename] = report

                mig_values.append(report.mig)

                if report.noise_variance is not None:
                    noise_variances.append(report.noise_variance)

                if report.predicted_accuracy is not None:
                    predicted_accuracies.append(report.predicted_accuracy)

                if (report.sssig_result
                        and len(report.sssig_result.points_y) > 0):
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
        logger.info(
            f"배치 평가 완료: {passed}/{total} 통과, "
            f"처리시간={total_time:.2f}s")

        def _stats(values):
            if not values:
                return {'mean': 0.0, 'min': 0.0, 'max': 0.0}
            return {
                'mean': float(np.mean(values)),
                'min': float(np.min(values)),
                'max': float(np.max(values)),
            }

        return BatchReport(
            total_images=total,
            passed_images=passed,
            failed_images=failed,
            global_recommended_size=max_recommended_size,
            global_size_found=global_size_found,
            worst_case_file=worst_file,
            individual_reports=individual_reports,
            mig_stats=_stats(mig_values),
            sssig_stats=_stats(sssig_min_values),
            noise_stats=_stats(noise_variances),
            accuracy_stats=_stats(predicted_accuracies),
            total_processing_time=total_time,
        )
