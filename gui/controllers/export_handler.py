"""
DIC 결과 내보내기 모듈

CSV, PLS strain, ground truth 비교, 분석 요약, 오버레이 이미지 내보내기
"""

import json
import logging
import numpy as np
import cv2
from pathlib import Path
from datetime import datetime
from tkinter import filedialog, messagebox

logger = logging.getLogger(__name__)


class ExportHandler:
    """DIC 결과 내보내기 담당"""

    def __init__(self, ctrl):
        self.ctrl = ctrl

    @property
    def state(self):
        return self.ctrl.state

    @property
    def view(self):
        return self.ctrl.view

    # ===== 기존: 단일 CSV 내보내기 =====

    def export_csv(self):
        """결과를 CSV로 내보내기 (기존 기능 유지)"""
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
                columns = [
                    result.points_y,
                    result.points_x,
                    result.disp_u,
                    result.disp_v,
                    result.zncc_values,
                    result.converged.astype(int),
                    result.valid_mask.astype(int),
                ]
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

    # ===== 기존: 이미지 내보내기 =====

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

    # ===== 신규: DIC 결과 일괄 내보내기 =====

    def export_dic_result(self):
        """DIC 분석 결과 일괄 내보내기
        
        저장 구조:
            <folder>/
            ├── displacement.csv       # POI별 변위/ZNCC/수렴 데이터
            ├── strain_pls.csv         # PLS 변형률 (IC-GN 결과 시)
            ├── gt_comparison.csv      # Ground truth 비교 (GT 파일 존재 시)
            ├── accuracy_metrics.json  # 정확도 지표 (GT 파일 존재 시)
            ├── analysis_summary.json  # 분석 파라미터 및 통계 요약
            └── overlay.png            # 현재 오버레이 이미지
        """
        result = self.state.icgn_result or self.state.fft_cc_result
        if result is None:
            messagebox.showwarning("경고", "내보낼 DIC 결과가 없습니다.")
            return

        folder = filedialog.askdirectory(title="DIC 결과 저장 폴더 선택")
        if not folder:
            return

        out_dir = Path(folder)
        out_dir.mkdir(parents=True, exist_ok=True)

        saved_files = []
        errors = []

        # 1) displacement.csv
        try:
            path = self._save_displacement_csv(result, out_dir)
            saved_files.append(path.name)
        except Exception as e:
            errors.append(f"displacement.csv: {e}")
            logger.error(f"변위 CSV 저장 실패: {e}")

        # 2) strain_pls.csv (IC-GN only)
        try:
            path = self._save_strain_csv(result, out_dir)
            if path:
                saved_files.append(path.name)
        except Exception as e:
            errors.append(f"strain_pls.csv: {e}")
            logger.error(f"PLS 변형률 CSV 저장 실패: {e}")

        # 3) ground truth 비교 (GT 파일 존재 시)
        try:
            paths = self._save_ground_truth_comparison(result, out_dir)
            for p in paths:
                saved_files.append(p.name)
        except Exception as e:
            errors.append(f"ground truth: {e}")
            logger.error(f"Ground truth 비교 저장 실패: {e}")

        # 4) analysis_summary.json
        try:
            path = self._save_analysis_summary(result, out_dir, saved_files)
            saved_files.append(path.name)
        except Exception as e:
            errors.append(f"analysis_summary.json: {e}")
            logger.error(f"분석 요약 저장 실패: {e}")

        # 5) overlay.png
        try:
            path = self._save_overlay_image(result, out_dir)
            if path:
                saved_files.append(path.name)
        except Exception as e:
            errors.append(f"overlay.png: {e}")
            logger.error(f"오버레이 이미지 저장 실패: {e}")

        # 결과 보고
        msg = f"저장 완료: {len(saved_files)}개 파일\n폴더: {out_dir}\n\n"
        msg += "\n".join(f"  ✓ {f}" for f in saved_files)
        if errors:
            msg += f"\n\n오류 {len(errors)}건:\n"
            msg += "\n".join(f"  ✗ {e}" for e in errors)

        messagebox.showinfo("DIC 결과 내보내기", msg)
        logger.info(f"DIC 결과 내보내기: {out_dir} ({len(saved_files)} files)")

    # ===== 내부 헬퍼 =====

    def _save_displacement_csv(self, result, out_dir: Path) -> Path:
        """POI별 변위/ZNCC/수렴 데이터 CSV"""
        path = out_dir / "displacement.csv"
        is_icgn = hasattr(result, 'converged')
        n = result.n_points

        if is_icgn:
            columns = [
                result.points_y,
                result.points_x,
                result.disp_u,
                result.disp_v,
                result.zncc_values,
                result.converged.astype(int),
                result.valid_mask.astype(int),
            ]
            for arr in [result.disp_ux, result.disp_uy,
                        result.disp_vx, result.disp_vy]:
                columns.append(arr if arr is not None else np.full(n, np.nan))
            columns.append(result.iterations)

            # failure_reason 추가
            if hasattr(result, 'failure_reason') and result.failure_reason is not None:
                columns.append(result.failure_reason)
                header = 'y,x,u,v,zncc,converged,valid,ux,uy,vx,vy,iterations,failure_reason'
                fmt = ['%d', '%d', '%.6f', '%.6f', '%.6f', '%d', '%d',
                       '%.8f', '%.8f', '%.8f', '%.8f', '%d', '%d']
            else:
                header = 'y,x,u,v,zncc,converged,valid,ux,uy,vx,vy,iterations'
                fmt = ['%d', '%d', '%.6f', '%.6f', '%.6f', '%d', '%d',
                       '%.8f', '%.8f', '%.8f', '%.8f', '%d']
        else:
            columns = [
                result.points_y,
                result.points_x,
                result.disp_u.astype(np.float64),
                result.disp_v.astype(np.float64),
                result.zncc_values,
                result.valid_mask.astype(int),
            ]
            header = 'y,x,u,v,zncc,valid'
            fmt = ['%d', '%d', '%.6f', '%.6f', '%.6f', '%d']

        data = np.column_stack(columns)
        np.savetxt(str(path), data, delimiter=',', header=header,
                   comments='', fmt=fmt)
        logger.info(f"displacement.csv: {n} POIs")
        return path

    def _save_strain_csv(self, result, out_dir: Path) -> Path:
        """PLS 변형률 CSV (IC-GN 결과만)"""
        is_icgn = hasattr(result, 'disp_ux') and result.disp_ux is not None
        if not is_icgn:
            logger.info("FFT-CC 결과 — strain_pls.csv 생략")
            return None

        valid = result.valid_mask
        px_valid = result.points_x[valid]
        py_valid = result.points_y[valid]
        unique_x = np.unique(px_valid)
        unique_y = np.unique(py_valid)

        if len(unique_x) < 5 or len(unique_y) < 5:
            logger.warning("그리드 너무 작음 — strain_pls.csv 생략")
            return None

        grid_step = float(np.median(np.diff(unique_x))) if len(unique_x) > 1 else 1.0

        # 변위 그리드 구성
        nx_g, ny_g = len(unique_x), len(unique_y)
        x_to_idx = {x: i for i, x in enumerate(unique_x)}
        y_to_idx = {y: i for i, y in enumerate(unique_y)}

        disp_u_grid = np.full((ny_g, nx_g), np.nan)
        disp_v_grid = np.full((ny_g, nx_g), np.nan)

        for i in range(len(result.points_x)):
            if valid[i]:
                xi = x_to_idx.get(result.points_x[i])
                yi = y_to_idx.get(result.points_y[i])
                if xi is not None and yi is not None:
                    disp_u_grid[yi, xi] = result.disp_u[i]
                    disp_v_grid[yi, xi] = result.disp_v[i]

        # PLS 계산
        from speckle.core.postprocess.strain_pls import compute_strain_pls
        strain = compute_strain_pls(
            disp_u_grid, disp_v_grid,
            window_size=11, poly_order=2, grid_step=grid_step
        )

        # CSV 저장: 각 그리드점의 좌표 + 변형률
        path = out_dir / "strain_pls.csv"
        rows = []
        for iy in range(ny_g):
            for ix in range(nx_g):
                exx = strain.exx[iy, ix]
                eyy = strain.eyy[iy, ix]
                exy = strain.exy[iy, ix]
                e1 = strain.e1[iy, ix]
                e2 = strain.e2[iy, ix]
                vm = strain.von_mises[iy, ix]
                # NaN인 경우에도 저장 (빈 값은 NaN으로 표시)
                rows.append([
                    unique_y[iy], unique_x[ix],
                    disp_u_grid[iy, ix], disp_v_grid[iy, ix],
                    exx, eyy, exy, e1, e2, vm
                ])

        data = np.array(rows)
        header = 'y,x,u,v,exx,eyy,exy,e1,e2,von_mises'
        fmt = ['%.4f', '%.4f', '%.6f', '%.6f', '%.8f', '%.8f', '%.8f', '%.8f', '%.8f', '%.8f']
        np.savetxt(str(path), data, delimiter=',', header=header,
                   comments='', fmt=fmt)

        # 유효 변형률 통계 로깅
        valid_exx = strain.exx[~np.isnan(strain.exx)]
        logger.info(f"strain_pls.csv: {ny_g}x{nx_g} grid, "
                     f"{len(valid_exx)}/{ny_g*nx_g} valid strain points, "
                     f"window={11}, poly_order=2, grid_step={grid_step:.1f}")
        return path

    def _save_ground_truth_comparison(self, result, out_dir: Path) -> list:
        """Ground truth 비교 (GT .npy 파일 존재 시)"""
        saved = []

        # GT 파일 탐색: 이미지 폴더 또는 부모 폴더
        search_dirs = []
        if self.state.def_path:
            search_dirs.append(self.state.def_path.parent)
        if self.state.ref_path:
            search_dirs.append(self.state.ref_path.parent)
        if self.state.sequence_folder:
            search_dirs.append(self.state.sequence_folder)

        gt_u_path = None
        gt_v_path = None
        for d in search_dirs:
            u_candidate = d / "ground_truth_u.npy"
            v_candidate = d / "ground_truth_v.npy"
            if u_candidate.exists() and v_candidate.exists():
                gt_u_path = u_candidate
                gt_v_path = v_candidate
                break

        if gt_u_path is None:
            logger.info("Ground truth 파일 없음 — 비교 생략")
            return saved

        # GT 로드
        gt_u = np.load(str(gt_u_path))
        gt_v = np.load(str(gt_v_path))
        logger.info(f"Ground truth 로드: u={gt_u.shape}, v={gt_v.shape}")

        # POI 좌표에서 GT 값 추출
        valid = result.valid_mask
        n = result.n_points
        gt_h, gt_w = gt_u.shape

        rows = []
        errors_u = []
        errors_v = []

        for i in range(n):
            px = int(round(result.points_x[i]))
            py = int(round(result.points_y[i]))

            # GT 범위 체크
            if 0 <= py < gt_h and 0 <= px < gt_w:
                gt_u_val = gt_u[py, px]
                gt_v_val = gt_v[py, px]
            else:
                gt_u_val = np.nan
                gt_v_val = np.nan

            dic_u = result.disp_u[i]
            dic_v = result.disp_v[i]
            err_u = dic_u - gt_u_val
            err_v = dic_v - gt_v_val
            err_mag = np.sqrt(err_u**2 + err_v**2)

            rows.append([
                result.points_y[i], result.points_x[i],
                dic_u, dic_v,
                gt_u_val, gt_v_val,
                err_u, err_v, err_mag,
                result.zncc_values[i],
                int(valid[i])
            ])

            if valid[i] and not np.isnan(gt_u_val):
                errors_u.append(err_u)
                errors_v.append(err_v)

        # gt_comparison.csv
        path_csv = out_dir / "gt_comparison.csv"
        data = np.array(rows)
        header = 'y,x,dic_u,dic_v,gt_u,gt_v,err_u,err_v,err_mag,zncc,valid'
        fmt = ['%.4f', '%.4f', '%.6f', '%.6f', '%.6f', '%.6f',
               '%.6f', '%.6f', '%.6f', '%.6f', '%d']
        np.savetxt(str(path_csv), data, delimiter=',', header=header,
                   comments='', fmt=fmt)
        saved.append(path_csv)
        logger.info(f"gt_comparison.csv: {n} POIs")

        # accuracy_metrics.json
        if len(errors_u) > 0:
            eu = np.array(errors_u)
            ev = np.array(errors_v)
            em = np.sqrt(eu**2 + ev**2)

            metrics = {
                "ground_truth_files": {
                    "u": str(gt_u_path),
                    "v": str(gt_v_path),
                    "shape": list(gt_u.shape)
                },
                "n_compared": len(eu),
                "n_valid_poi": int(np.sum(valid)),
                "n_total_poi": n,
                "displacement_u": {
                    "mean_bias": float(np.mean(eu)),
                    "std": float(np.std(eu)),
                    "rms": float(np.sqrt(np.mean(eu**2))),
                    "max_abs": float(np.max(np.abs(eu))),
                    "p95": float(np.percentile(np.abs(eu), 95)),
                    "p99": float(np.percentile(np.abs(eu), 99))
                },
                "displacement_v": {
                    "mean_bias": float(np.mean(ev)),
                    "std": float(np.std(ev)),
                    "rms": float(np.sqrt(np.mean(ev**2))),
                    "max_abs": float(np.max(np.abs(ev))),
                    "p95": float(np.percentile(np.abs(ev), 95)),
                    "p99": float(np.percentile(np.abs(ev), 99))
                },
                "displacement_magnitude": {
                    "mean": float(np.mean(em)),
                    "std": float(np.std(em)),
                    "rms": float(np.sqrt(np.mean(em**2))),
                    "max": float(np.max(em)),
                    "p95": float(np.percentile(em, 95)),
                    "p99": float(np.percentile(em, 99))
                }
            }

            path_json = out_dir / "accuracy_metrics.json"
            with open(str(path_json), 'w', encoding='utf-8') as f:
                json.dump(metrics, f, indent=2, ensure_ascii=False)
            saved.append(path_json)
            logger.info(f"accuracy_metrics.json: u_rms={metrics['displacement_u']['rms']:.6f}, "
                         f"v_rms={metrics['displacement_v']['rms']:.6f}")

        return saved

    def _save_analysis_summary(self, result, out_dir: Path, saved_files: list) -> Path:
        """분석 파라미터 및 통계 요약 JSON"""
        path = out_dir / "analysis_summary.json"
        is_icgn = hasattr(result, 'converged')
        valid = result.valid_mask

        # 파라미터 수집
        params = {}
        try:
            gui_params = self.view.get_parameters()
            params = {
                'subset_size': gui_params.get('subset_size', None),
                'spacing': gui_params.get('spacing', None),
                'search_range': gui_params.get('search_range', None),
                'zncc_threshold': gui_params.get('zncc_threshold', None),
            }
        except Exception:
            pass

        if is_icgn:
            params.update({
                'shape_function': getattr(result, 'shape_function', None),
                'max_iterations': getattr(result, 'max_iterations', None),
                'convergence_threshold': getattr(result, 'convergence_threshold', None),
            })

        # 유효 변위 통계
        valid_u = result.disp_u[valid]
        valid_v = result.disp_v[valid]

        summary = {
            "export_time": datetime.now().isoformat(),
            "saved_files": saved_files,
            "images": {
                "reference": str(self.state.ref_path) if self.state.ref_path else None,
                "deformed": str(self.state.def_path) if self.state.def_path else None,
            },
            "parameters": params,
            "result_type": "IC-GN" if is_icgn else "FFT-CC",
            "statistics": {
                "total_poi": int(result.n_points),
                "valid_poi": int(np.sum(valid)),
                "valid_ratio": float(np.sum(valid) / max(result.n_points, 1)),
                "mean_zncc": float(np.mean(result.zncc_values[valid])) if np.any(valid) else 0.0,
                "processing_time_s": float(getattr(result, 'processing_time', 0.0)),
            },
            "displacement": {
                "u": {
                    "mean": float(np.mean(valid_u)) if len(valid_u) > 0 else None,
                    "std": float(np.std(valid_u)) if len(valid_u) > 0 else None,
                    "min": float(np.min(valid_u)) if len(valid_u) > 0 else None,
                    "max": float(np.max(valid_u)) if len(valid_u) > 0 else None,
                },
                "v": {
                    "mean": float(np.mean(valid_v)) if len(valid_v) > 0 else None,
                    "std": float(np.std(valid_v)) if len(valid_v) > 0 else None,
                    "min": float(np.min(valid_v)) if len(valid_v) > 0 else None,
                    "max": float(np.max(valid_v)) if len(valid_v) > 0 else None,
                },
            },
        }

        # ===== IC-GN 전용 통계 =====
        if is_icgn:
            summary["statistics"]["convergence_rate"] = float(
                np.sum(result.converged) / max(result.n_points, 1)
            )
            summary["statistics"]["mean_iterations"] = float(
                np.mean(result.iterations[valid])
            ) if np.any(valid) else 0.0

            # failure reason 분포
            if hasattr(result, 'failure_reason') and result.failure_reason is not None:
                FAILURE_REASON_NAMES = {
                    0: 'success',
                    1: 'low_zncc',
                    2: 'diverged',
                    3: 'out_of_bounds',
                    4: 'singular_hessian',
                    5: 'flat_subset',
                    6: 'max_displacement',
                    7: 'flat_target',
                }
                
                reasons = {}
                for code in np.unique(result.failure_reason):
                    count = int(np.sum(result.failure_reason == code))
                    name = FAILURE_REASON_NAMES.get(int(code), f"UNKNOWN_{code}")
                    reasons[name] = count
                summary["failure_reasons"] = reasons

        # ===== 자기 검증 지표 (Ground Truth 없이도 사용 가능) =====
        validation = {}

        if np.any(valid):
            zncc_valid = result.zncc_values[valid]

            # (1) ZNCC 분포
            validation["zncc_distribution"] = {
                "mean": float(np.mean(zncc_valid)),
                "std": float(np.std(zncc_valid)),
                "min": float(np.min(zncc_valid)),
                "median": float(np.median(zncc_valid)),
                "p5": float(np.percentile(zncc_valid, 5)),
                "above_0.99": float(np.mean(zncc_valid >= 0.99)),
                "above_0.95": float(np.mean(zncc_valid >= 0.95)),
                "below_0.95": float(np.mean(zncc_valid < 0.95)),
            }

            # (2) 변위 연속성 — 인접 POI 간 차이의 MAD 기반 이상치 탐지
            u_valid = result.disp_u[valid]
            v_valid = result.disp_v[valid]
            px_valid = result.points_x[valid]
            py_valid = result.points_y[valid]

            # 그리드 구조 확인 후 인접 차분 계산
            unique_x = np.unique(px_valid)
            unique_y = np.unique(py_valid)

            if len(unique_x) > 1 and len(unique_y) > 1:
                # 변위를 2D 그리드로 변환
                nx_g, ny_g = len(unique_x), len(unique_y)
                x_to_idx = {x: i for i, x in enumerate(unique_x)}
                y_to_idx = {y: i for i, y in enumerate(unique_y)}

                u_grid = np.full((ny_g, nx_g), np.nan)
                v_grid = np.full((ny_g, nx_g), np.nan)
                for i in range(len(px_valid)):
                    xi = x_to_idx.get(px_valid[i])
                    yi = y_to_idx.get(py_valid[i])
                    if xi is not None and yi is not None:
                        u_grid[yi, xi] = u_valid[i]
                        v_grid[yi, xi] = v_valid[i]

                # x방향, y방향 차분
                du_dx_diff = np.diff(u_grid, axis=1)  # (ny, nx-1)
                du_dy_diff = np.diff(u_grid, axis=0)  # (ny-1, nx)
                dv_dx_diff = np.diff(v_grid, axis=1)
                dv_dy_diff = np.diff(v_grid, axis=0)

                all_diffs = np.concatenate([
                    du_dx_diff[~np.isnan(du_dx_diff)],
                    du_dy_diff[~np.isnan(du_dy_diff)],
                    dv_dx_diff[~np.isnan(dv_dx_diff)],
                    dv_dy_diff[~np.isnan(dv_dy_diff)],
                ])

                if len(all_diffs) > 0:
                    med = np.median(np.abs(all_diffs))
                    mad = np.median(np.abs(all_diffs - np.median(all_diffs)))
                    threshold = med + 5 * max(mad, 1e-6)
                    n_outlier = int(np.sum(np.abs(all_diffs) > threshold))

                    validation["displacement_continuity"] = {
                        "neighbor_diff_median": float(med),
                        "neighbor_diff_mad": float(mad),
                        "outlier_threshold": float(threshold),
                        "n_outlier_pairs": n_outlier,
                        "total_pairs": len(all_diffs),
                        "outlier_ratio": float(n_outlier / max(len(all_diffs), 1)),
                    }

        # (3) 수렴 반복 횟수 분포 (IC-GN만)
        if is_icgn and np.any(valid):
            iters_valid = result.iterations[valid].astype(float)
            validation["iteration_distribution"] = {
                "mean": float(np.mean(iters_valid)),
                "std": float(np.std(iters_valid)),
                "median": float(np.median(iters_valid)),
                "max": float(np.max(iters_valid)),
                "above_10": float(np.mean(iters_valid > 10)),
                "above_20": float(np.mean(iters_valid > 20)),
            }

        # (4) 전체 판정
        if validation:
            issues = []
            zncc_d = validation.get("zncc_distribution", {})
            if zncc_d.get("below_0.95", 0) > 0.1:
                issues.append("ZNCC < 0.95인 POI가 10% 이상")
            if zncc_d.get("mean", 1.0) < 0.98:
                issues.append("평균 ZNCC < 0.98")

            cont_d = validation.get("displacement_continuity", {})
            if cont_d.get("outlier_ratio", 0) > 0.05:
                issues.append("변위 불연속 이상치 5% 이상")

            iter_d = validation.get("iteration_distribution", {})
            if iter_d.get("above_20", 0) > 0.05:
                issues.append("20회 이상 반복 POI 5% 이상")

            validation["overall"] = {
                "status": "PASS" if len(issues) == 0 else "WARNING",
                "issues": issues,
                "n_issues": len(issues),
            }

        summary["validation"] = validation

        with open(str(path), 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"analysis_summary.json saved")
        return path

    def _save_overlay_image(self, result, out_dir: Path) -> Path:
        """현재 표시 모드의 오버레이 이미지 저장"""
        img = self.state.def_image if self.state.def_image is not None else self.state.ref_image
        if img is None:
            return None

        try:
            overlay = self.ctrl.renderer.create_overlay_image(img, result)
            path = out_dir / "overlay.png"
            cv2.imwrite(str(path), overlay)
            logger.info(f"overlay.png saved")
            return path
        except Exception as e:
            logger.warning(f"오버레이 이미지 저장 실패: {e}")
            return None
