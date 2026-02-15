"""
DIC 결과 처리 모듈

분석 완료 후 UI 업데이트, 결과 텍스트 생성
"""

import logging
import numpy as np
from tkinter import messagebox

logger = logging.getLogger(__name__)


class ResultHandler:
    """결과 처리 및 UI 업데이트 담당"""

    def __init__(self, ctrl):
        self.ctrl = ctrl

    @property
    def state(self):
        return self.ctrl.state

    @property
    def view(self):
        return self.ctrl.view

    def update_result_ui(self, result):
        """결과 UI 업데이트"""
        if result is None:
            return

        u = result.disp_u
        v = result.disp_v
        valid = result.valid_mask if hasattr(result, 'valid_mask') else np.ones(len(u), bool)

        u_valid = u[valid]
        v_valid = v[valid]

        is_icgn = hasattr(result, 'converged')
        filename = self.state.def_path.name if self.state.def_path else "N/A"

        if len(u_valid) == 0:
            self.view.update_result_text(
                f"유효한 POI가 없습니다.\n"
                f"전체 POI: {len(u)}개\n"
                f"유효 POI: 0개\n\n"
                f"원인: search_range 부족 또는 이미지 품질 문제"
            )
            return

        if is_icgn:
            lines = []
            lines.append(f"Shape: {result.shape_function}")
            lines.append("")
            lines.append(f"POI: {result.n_converged}/{result.n_points} ({result.convergence_rate*100:.1f}%)")
            lines.append(f"반복: {result.mean_iterations:.1f}회 | ZNCC: {result.mean_zncc:.4f}")
            lines.append("")
            lines.append("── 변위 ──")
            lines.append(f"U: {np.mean(u_valid):.4f} ± {np.std(u_valid):.4f}")
            lines.append(f"   [{np.min(u_valid):.4f} ~ {np.max(u_valid):.4f}]")
            lines.append(f"V: {np.mean(v_valid):.4f} ± {np.std(v_valid):.4f}")
            lines.append(f"   [{np.min(v_valid):.4f} ~ {np.max(v_valid):.4f}]")

            if hasattr(result, 'disp_ux') and result.disp_ux is not None:
                ux = result.disp_ux[valid]
                uy = result.disp_uy[valid]
                vx = result.disp_vx[valid]
                vy = result.disp_vy[valid]

                lines.append("")
                lines.append("── 변형률 ──")
                lines.append(f"ux: {np.mean(ux): .5f} ± {np.std(ux):.5f}")
                lines.append(f"uy: {np.mean(uy): .5f} ± {np.std(uy):.5f}")
                lines.append(f"vx: {np.mean(vx): .5f} ± {np.std(vx):.5f}")
                lines.append(f"vy: {np.mean(vy): .5f} ± {np.std(vy):.5f}")

            lines.append("")
            lines.append(f"시간: {result.processing_time:.2f}초")
            text = "\n".join(lines)
        else:
            text = (
                f"파일: {filename}\n\n"
                f"POI: {result.n_valid}/{result.n_points} ({result.valid_ratio*100:.1f}%)\n"
                f"ZNCC: {result.mean_zncc:.4f}\n\n"
                f"── 변위 ──\n"
                f"U: {np.mean(u_valid):.4f} ± {np.std(u_valid):.4f}\n"
                f"[{np.min(u_valid):.4f} ~ {np.max(u_valid):.4f}]\n"
                f"V: {np.mean(v_valid):.4f} ± {np.std(v_valid):.4f}\n"
                f"[{np.min(v_valid):.4f} ~ {np.max(v_valid):.4f}]\n\n"
                f"시간: {result.processing_time:.2f}초"
            )

        self.view.update_result_text(text)

    def on_icgn_complete(self, result):
        """IC-GN 분석 완료 처리"""
        self.view.update_progress(100, "완료")

        # PLS 변형률 캐시 무효화 (새 결과)
        if hasattr(self.ctrl, 'renderer'):
            self.ctrl.renderer.invalidate_strain_cache()

        self.state.fft_cc_result = result
        self.ctrl._refresh_display()
        self.update_result_ui(result)

        valid_text = (
            f"수렴율: {result.convergence_rate * 100:.1f}%\n"
            f"유효 POI: {result.n_valid}/{result.n_points}\n"
            f"평균 반복: {result.mean_iterations:.1f}회\n"
            f"평균 ZNCC: {result.mean_zncc:.4f}"
        )
        self.view.update_validation_text(valid_text)

    def on_batch_complete(self):
        """배치 분석 완료"""
        n_results = len(self.state.batch_results)
        self.view.update_progress(100, f"완료: {n_results} 파일 처리됨")

        # PLS 변형률 캐시 무효화 (새 결과)
        if hasattr(self.ctrl, 'renderer'):
            self.ctrl.renderer.invalidate_strain_cache()

        self.ctrl._refresh_display()

        if self.state.batch_results:
            last_key = list(self.state.batch_results.keys())[-1]
            last_result = self.state.batch_results[last_key]
            self.update_result_ui(last_result)

        total_time = sum(r.processing_time for r in self.state.batch_results.values())
        avg_time = total_time / n_results if n_results > 0 else 0

        first_result = list(self.state.batch_results.values())[0] if self.state.batch_results else None
        is_icgn = first_result and hasattr(first_result, 'convergence_rate')

        if is_icgn:
            avg_conv_rate = np.mean([r.convergence_rate for r in self.state.batch_results.values()]) * 100
            messagebox.showinfo(
                "배치 분석 완료",
                f"처리된 파일: {n_results}개\n"
                f"총 소요 시간: {total_time:.1f}초\n"
                f"평균 처리 시간: {avg_time:.2f}초/파일\n"
                f"평균 수렴율: {avg_conv_rate:.1f}%"
            )
        else:
            messagebox.showinfo(
                "배치 분석 완료",
                f"처리된 파일: {n_results}개\n"
                f"총 소요 시간: {total_time:.1f}초\n"
                f"평균 처리 시간: {avg_time:.2f}초/파일"
            )

    def show_error(self, msg: str):
        messagebox.showerror("오류", f"분석 실패: {msg}")
