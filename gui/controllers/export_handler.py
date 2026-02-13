"""
DIC 결과 내보내기 모듈

CSV 및 이미지 내보내기
"""

import logging
import numpy as np
import cv2
from pathlib import Path
from tkinter import filedialog, messagebox

logger = logging.getLogger(__name__)


class ExportHandler:
    """결과 내보내기 담당"""

    def __init__(self, ctrl):
        self.ctrl = ctrl

    @property
    def state(self):
        return self.ctrl.state

    @property
    def view(self):
        return self.ctrl.view

    def export_csv(self):
        """결과를 CSV로 내보내기"""
        result = self.state.icgn_result or self.state.fft_cc_result
        if result is None:
            messagebox.showwarning("경고", "내보낼 결과가 없습니다.")
            return

        path = filedialog.asksaveasfilename(
            title="CSV 저장",
            defaultextension=".csv",
            filetypes=[("CSV 파일", "*.csv"), ("모든 파일", "*.*")]
        )

        if not path:
            return

        try:
            import csv
            valid = result.valid_mask
            is_icgn = hasattr(result, 'converged')

            with open(path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)

                if is_icgn:
                    header = ['y', 'x', 'u', 'v', 'zncc', 'converged', 'valid',
                              'ux', 'uy', 'vx', 'vy', 'iterations']
                    writer.writerow(header)

                    for i in range(result.n_points):
                        row = [
                            result.points_y[i], result.points_x[i],
                            f"{result.disp_u[i]:.6f}", f"{result.disp_v[i]:.6f}",
                            f"{result.zncc_values[i]:.6f}",
                            result.converged[i], result.valid_mask[i],
                            f"{result.disp_ux[i]:.8f}" if result.disp_ux is not None else "",
                            f"{result.disp_uy[i]:.8f}" if result.disp_uy is not None else "",
                            f"{result.disp_vx[i]:.8f}" if result.disp_vx is not None else "",
                            f"{result.disp_vy[i]:.8f}" if result.disp_vy is not None else "",
                            result.iterations[i]
                        ]
                        writer.writerow(row)
                else:
                    header = ['y', 'x', 'u', 'v', 'zncc', 'valid']
                    writer.writerow(header)

                    for i in range(result.n_points):
                        row = [
                            result.points_y[i], result.points_x[i],
                            result.disp_u[i], result.disp_v[i],
                            f"{result.zncc_values[i]:.6f}",
                            result.valid_mask[i]
                        ]
                        writer.writerow(row)

            messagebox.showinfo("완료", f"CSV 저장 완료: {path}")
            logger.info(f"CSV 내보내기: {path}")

        except Exception as e:
            messagebox.showerror("오류", f"CSV 저장 실패: {e}")

    def export_image(self):
        """현재 표시 이미지를 파일로 저장"""
        img = self.state.def_image if self.state.def_image is not None else self.state.ref_image
        if img is None:
            messagebox.showwarning("경고", "내보낼 이미지가 없습니다.")
            return

        path = filedialog.asksaveasfilename(
            title="이미지 저장",
            defaultextension=".png",
            filetypes=[
                ("PNG", "*.png"), ("TIFF", "*.tif *.tiff"),
                ("JPEG", "*.jpg *.jpeg"), ("모든 파일", "*.*")
            ]
        )

        if not path:
            return

        try:
            if self.state.fft_cc_result is not None:
                save_img = self.ctrl.renderer.create_overlay_image(
                    img, self.state.fft_cc_result
                )
            else:
                save_img = img

            cv2.imwrite(path, save_img)
            messagebox.showinfo("완료", f"이미지 저장 완료: {path}")
            logger.info(f"이미지 내보내기: {path}")

        except Exception as e:
            messagebox.showerror("오류", f"이미지 저장 실패: {e}")
