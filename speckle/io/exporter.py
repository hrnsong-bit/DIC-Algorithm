"""결과 내보내기 모듈"""

import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple
import numpy as np
import cv2

from ..models.reports import QualityReport, SSSIGResult


class ResultExporter:
    """평가 결과 내보내기"""
    
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def export_csv(self, 
                   reports: Dict[str, QualityReport],
                   filename: Optional[str] = None) -> Path:
        """
        CSV로 배치 결과 내보내기
        
        Args:
            reports: {파일명: QualityReport} 딕셔너리
            filename: 출력 파일명 (없으면 자동 생성)
        
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            filename = f"speckle_results_{self.timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            
            # 헤더
            writer.writerow([
                'Filename',
                'MIG',
                'MIG_Pass',
                'MIG_Threshold',
                'SSSIG_Mean',
                'SSSIG_Min',
                'SSSIG_Max',
                'SSSIG_Pass',
                'SSSIG_Threshold',
                'Total_POI',
                'Bad_POI',
                'Current_Subset',
                'Recommended_Subset',
                'Subset_Found',
                'Quality_Grade',
                'Analyzable',
                'Processing_Time_s',
                'Warnings'
            ])
            
            # 데이터
            for fname, report in reports.items():
                sr = report.sssig_result
                
                writer.writerow([
                    fname,
                    f"{report.mig:.4f}",
                    report.mig_pass,
                    report.mig_threshold,
                    f"{sr.mean:.2e}" if sr else "",
                    f"{sr.min:.2e}" if sr else "",
                    f"{sr.max:.2e}" if sr else "",
                    report.sssig_pass,
                    report.sssig_threshold,
                    len(sr.points_y) if sr else 0,
                    sr.n_bad_points if sr else 0,
                    report.current_subset_size,
                    report.recommended_subset_size,
                    report.subset_size_found,
                    report.quality_grade,
                    report.analyzable,
                    f"{report.processing_time:.4f}",
                    "; ".join(report.warnings) if report.warnings else ""
                ])
        
        return output_path
    
    def export_json(self,
                    reports: Dict[str, QualityReport],
                    parameters: Dict,
                    roi: Optional[Tuple[int, int, int, int]] = None,
                    filename: Optional[str] = None) -> Path:
        """
        JSON으로 상세 결과 내보내기
        
        Args:
            reports: 평가 결과
            parameters: 사용된 파라미터
            roi: ROI 설정
            filename: 출력 파일명
        
        Returns:
            저장된 파일 경로
        """
        if filename is None:
            filename = f"speckle_results_{self.timestamp}.json"
        
        output_path = self.output_dir / filename
        
        # 통계 계산
        mig_values = [r.mig for r in reports.values()]
        sssig_mins = [r.sssig_result.min for r in reports.values() 
                      if r.sssig_result and len(r.sssig_result.points_y) > 0]
        
        passed = sum(1 for r in reports.values() if r.analyzable)
        warning = sum(1 for r in reports.values() 
                      if r.analyzable and r.recommended_subset_size > r.current_subset_size)
        failed = sum(1 for r in reports.values() if not r.analyzable)
        
        data = {
            "metadata": {
                "export_time": datetime.now().isoformat(),
                "total_images": len(reports),
                "passed": passed,
                "warning": warning,
                "failed": failed
            },
            "parameters": parameters,
            "roi": {
                "x": roi[0],
                "y": roi[1],
                "width": roi[2],
                "height": roi[3]
            } if roi else None,
            "statistics": {
                "mig": {
                    "mean": float(np.mean(mig_values)) if mig_values else 0,
                    "min": float(np.min(mig_values)) if mig_values else 0,
                    "max": float(np.max(mig_values)) if mig_values else 0,
                    "std": float(np.std(mig_values)) if mig_values else 0
                },
                "sssig_min": {
                    "mean": float(np.mean(sssig_mins)) if sssig_mins else 0,
                    "min": float(np.min(sssig_mins)) if sssig_mins else 0,
                    "max": float(np.max(sssig_mins)) if sssig_mins else 0,
                    "std": float(np.std(sssig_mins)) if sssig_mins else 0
                },
                "recommended_subset_size": max(
                    (r.recommended_subset_size for r in reports.values()), 
                    default=21
                )
            },
            "results": {
                fname: self._report_to_dict(report)
                for fname, report in reports.items()
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        return output_path
    
    def _report_to_dict(self, report: QualityReport) -> dict:
        """QualityReport를 딕셔너리로 변환"""
        sr = report.sssig_result
        
        result = {
            "mig": report.mig,
            "mig_pass": report.mig_pass,
            "sssig_pass": report.sssig_pass,
            "current_subset_size": report.current_subset_size,
            "recommended_subset_size": report.recommended_subset_size,
            "subset_size_found": report.subset_size_found,
            "quality_grade": report.quality_grade,
            "analyzable": report.analyzable,
            "processing_time": report.processing_time,
            "warnings": report.warnings
        }
        
        if sr:
            result["sssig"] = {
                "subset_size": sr.subset_size,
                "spacing": sr.spacing,
                "mean": sr.mean,
                "min": sr.min,
                "max": sr.max,
                "total_points": len(sr.points_y),
                "bad_points": sr.n_bad_points,
                "bad_point_coords": [
                    {"y": bp.y, "x": bp.x, "sssig": bp.sssig}
                    for bp in sr.bad_points
                ]
            }
        
        return result
    
    def export_summary_txt(self,
                           reports: Dict[str, QualityReport],
                           parameters: Dict,
                           roi: Optional[Tuple[int, int, int, int]] = None,
                           filename: Optional[str] = None) -> Path:
        """
        TXT 요약 리포트 내보내기
        """
        if filename is None:
            filename = f"speckle_summary_{self.timestamp}.txt"
        
        output_path = self.output_dir / filename
        
        # 통계
        total = len(reports)
        passed = sum(1 for r in reports.values() if r.quality_grade == "PASS")
        warning = sum(1 for r in reports.values() if r.quality_grade == "NEEDS_LARGER_SUBSET")
        failed = sum(1 for r in reports.values() if r.quality_grade == "FAIL")
        
        mig_values = [r.mig for r in reports.values()]
        max_subset = max((r.recommended_subset_size for r in reports.values()), default=21)
        
        lines = [
            "=" * 60,
            "SPECKLE QUALITY ASSESSMENT REPORT",
            "=" * 60,
            "",
            f"Export Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "--- PARAMETERS ---",
            f"MIG Threshold: {parameters.get('mig_threshold', 'N/A')}",
            f"SSSIG Threshold: {parameters.get('sssig_threshold', 'N/A')}",
            f"Initial Subset Size: {parameters.get('subset_size', 'N/A')}px",
            f"POI Spacing: {parameters.get('spacing', 'N/A')}px",
            "",
        ]
        
        if roi:
            lines.extend([
                "--- ROI ---",
                f"Position: ({roi[0]}, {roi[1]})",
                f"Size: {roi[2]} x {roi[3]} px",
                "",
            ])
        
        lines.extend([
            "--- SUMMARY ---",
            f"Total Images: {total}",
            f"  ✓ Passed: {passed} ({100*passed/total:.1f}%)" if total > 0 else "  ✓ Passed: 0",
            f"  △ Needs Larger Subset: {warning} ({100*warning/total:.1f}%)" if total > 0 else "  △ Warning: 0",
            f"  ✗ Failed: {failed} ({100*failed/total:.1f}%)" if total > 0 else "  ✗ Failed: 0",
            "",
            "--- STATISTICS ---",
            f"MIG Mean: {np.mean(mig_values):.2f}" if mig_values else "MIG Mean: N/A",
            f"MIG Range: [{np.min(mig_values):.2f}, {np.max(mig_values):.2f}]" if mig_values else "MIG Range: N/A",
            "",
            f"Recommended Subset Size: {max_subset}px",
            "",
            "--- INDIVIDUAL RESULTS ---",
            "",
        ])
        
        # 개별 결과
        for fname, report in sorted(reports.items()):
            grade_symbol = {"PASS": "✓", "NEEDS_LARGER_SUBSET": "△", "FAIL": "✗"}.get(
                report.quality_grade, "?"
            )
            lines.append(
                f"{grade_symbol} {fname}: MIG={report.mig:.2f}, "
                f"Subset={report.recommended_subset_size}px, "
                f"{report.quality_grade}"
            )
        
        lines.extend([
            "",
            "=" * 60,
            "END OF REPORT",
            "=" * 60,
        ])
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        return output_path
    
    def export_image(self,
                     image: np.ndarray,
                     report: QualityReport,
                     roi: Optional[Tuple[int, int, int, int]] = None,
                     filename: Optional[str] = None,
                     show_all_poi: bool = True,
                     show_bad_poi: bool = True) -> Path:
        """
        POI 오버레이 이미지 저장
        """
        if filename is None:
            filename = f"speckle_overlay_{self.timestamp}.png"
        
        output_path = self.output_dir / filename
        
        # 이미지 준비
        if len(image.shape) == 2:
            display_img = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display_img = image.copy()
        
        # POI 그리기
        sr = report.sssig_result
        if sr and (show_all_poi or show_bad_poi):
            roi_x, roi_y = (roi[0], roi[1]) if roi else (0, 0)
            bad_coords = set((bp.y, bp.x) for bp in sr.bad_points)
            
            for idx in range(len(sr.points_y)):
                py_local = int(sr.points_y[idx])
                px_local = int(sr.points_x[idx])
                py = py_local + roi_y
                px = px_local + roi_x
                
                is_bad = (py_local, px_local) in bad_coords
                
                if is_bad and show_bad_poi:
                    color = (0, 255, 255) if report.subset_size_found else (0, 0, 255)
                    cv2.circle(display_img, (px, py), 5, color, -1)
                    cv2.drawMarker(display_img, (px, py), color, cv2.MARKER_CROSS, 10, 2)
                elif not is_bad and show_all_poi:
                    cv2.circle(display_img, (px, py), 2, (0, 255, 0), -1)
        
        # ROI 표시
        if roi:
            x, y, w, h = roi
            cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        
        # 정보 텍스트 추가
        info_text = [
            f"MIG: {report.mig:.2f} ({'PASS' if report.mig_pass else 'FAIL'})",
            f"Subset: {report.recommended_subset_size}px",
            f"Grade: {report.quality_grade}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(display_img, text, (10, y_offset), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 25
        
        cv2.imwrite(str(output_path), display_img)
        
        return output_path
    
    def export_all_images(self,
                          images: Dict[str, np.ndarray],
                          reports: Dict[str, QualityReport],
                          roi: Optional[Tuple[int, int, int, int]] = None,
                          subdir: Optional[str] = None) -> Path:
        """
        모든 이미지에 오버레이 저장
        """
        if subdir is None:
            subdir = f"overlay_images_{self.timestamp}"
        
        output_dir = self.output_dir / subdir
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for fname, image in images.items():
            if fname in reports:
                report = reports[fname]
                output_name = f"overlay_{Path(fname).stem}.png"
                
                self.export_image(
                    image=image,
                    report=report,
                    roi=roi,
                    filename=str(output_dir / output_name)
                )
        
        return output_dir
    
    def export_all(self,
                   images: Dict[str, np.ndarray],
                   reports: Dict[str, QualityReport],
                   parameters: Dict,
                   roi: Optional[Tuple[int, int, int, int]] = None,
                   include_images: bool = False) -> Dict[str, Path]:
        """
        모든 형식으로 한번에 내보내기
        
        Returns:
            {형식: 파일경로} 딕셔너리
        """
        results = {}
        
        results['csv'] = self.export_csv(reports)
        results['json'] = self.export_json(reports, parameters, roi)
        results['summary'] = self.export_summary_txt(reports, parameters, roi)
        
        if include_images and images:
            results['images'] = self.export_all_images(images, reports, roi)
        
        return results
