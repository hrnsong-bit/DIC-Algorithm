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
            is_icgn = hasattr(result, 'converged')
            n = result.n_points

            if is_icgn:
                # 벡터화된 CSV 생성: NumPy column_stack으로 한번에 조립
                columns = [
                    result.points_y,
                    result.points_x,
                    result.disp_u,
                    result.disp_v,
                    result.zncc_values,
                    result.converged.astype(int),
                    result.valid_mask.astype(int),
                ]
                # gradient columns (있으면 추가, 없으면 빈 열)
                for arr in [result.disp_ux, result.disp_uy,
                            result.disp_vx, result.disp_vy]:
                    if arr is not None:
                        columns.append(arr)
                    else:
                        columns.append(np.full(n, np.nan))
                columns.append(result.iterations)

                header = 'y,x,u,v,zncc,converged,valid,ux,uy,vx,vy,iterations'
                data = np.column_stack(columns)
                fmt = ['%d', '%d', '%.6f', '%.6f', '%.6f', '%d', '%d',
                       '%.8f', '%.8f', '%.8f', '%.8f', '%d']
            else:
                data = np.column_stack([
                    result.points_y,
                    result.points_x,
                    result.disp_u.astype(np.float64),
                    result.disp_v.astype(np.float64),
                    result.zncc_values,
                    result.valid_mask.astype(int),
                ])
                header = 'y,x,u,v,zncc,valid'
                fmt = ['%d', '%d', '%.6f', '%.6f', '%.6f', '%d']

            np.savetxt(path, data, delimiter=',', header=header,
                       comments='', fmt=fmt)

            messagebox.showinfo("완료", f"CSV 저장 완료: {path}")
            logger.info(f"CSV 내보내기: {path} ({n} POIs)")

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
