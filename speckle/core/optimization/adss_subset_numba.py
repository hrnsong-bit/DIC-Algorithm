import numpy as np
from numba import jit, float64, prange

from .variable_subset_numba import extract_reference_subset_variable
from .interpolation_numba import (
    is_inside_batch,
    _interp2d_cubic,
    _interp2d_quintic,
)
from .shape_function_numba import (
    AFFINE, QUADRATIC,
    warp,
    get_num_params,
    compute_steepest_descent,
    compute_hessian,
)
from .icgn_core_numba import (
    compute_znssd,
    icgn_iterate,
    ICGN_SUCCESS,
    ICGN_FAIL_LOW_ZNCC,
    ICGN_FAIL_FLAT_SUBSET,
    ICGN_FAIL_SINGULAR_HESSIAN,
)

"""
ADSS-DIC v2 (Adaptive Subset-Subdivision) Numba 가속 모듈

사분면(Q5~Q8) 복수 채택 방식:
    1. 각 불량 POI에서 4개 사분면의 1-warp ZNCC를 평가
    2. pre_threshold를 넘는 모든 사분면에 IC-GN을 수행
    3. zncc_threshold를 넘는 모든 사분면 결과를 sub-POI로 반환

Quarter-subset 정의 (사분면만 사용):
    Q5: Upper-left   ξ ∈ [-M, 0],  η ∈ [-M, 0]   → (M+1)×(M+1) 픽셀
    Q6: Upper-right  ξ ∈ [0, +M],  η ∈ [-M, 0]   → (M+1)×(M+1) 픽셀
    Q7: Lower-left   ξ ∈ [-M, 0],  η ∈ [0, +M]   → (M+1)×(M+1) 픽셀
    Q8: Lower-right  ξ ∈ [0, +M],  η ∈ [0, +M]   → (M+1)×(M+1) 픽셀

여기서 ξ는 x(col) 방향, η는 y(row) 방향이다.

Reference:
    Zhao, J., & Pan, B. "Adaptive subset-subdivision for automatic
    digital image correlation calculation on discontinuous shape
    and deformation." Applied Optics, 2025.
"""

# === Quarter-subset 상수 ===
Q5 = 5  # Upper-left
Q6 = 6  # Upper-right
Q7 = 7  # Lower-left
Q8 = 8  # Lower-right

# 각 사분면에 대응하는 이웃 방향 (dy, dx)
QUARTER_TO_NEIGHBOR = {
    Q5: (-1, -1),  # upper-left  → 좌상 이웃
    Q6: (-1,  1),  # upper-right → 우상 이웃
    Q7: ( 1, -1),  # lower-left  → 좌하 이웃
    Q8: ( 1,  1),  # lower-right → 우하 이웃
}


@jit(nopython=True, cache=True)
def predict_initial_params(
    neighbor_x, neighbor_y,
    neighbor_params,
    target_x, target_y,
    shape_type,
    out_params
):
    """
    ADSS-DIC Eq.(3): 이웃 POI의 IC-GN 결과로부터 불량 POI의 초기값을 예측.

    1차 Taylor 전개를 사용하여 well-matched 이웃의 변위·변형률 파라미터를
    대상 POI 위치로 외삽한다.
    """
    dx = target_x - neighbor_x
    dy = target_y - neighbor_y

    if shape_type == 0:  # AFFINE
        u_k = neighbor_params[0]
        ux_k = neighbor_params[1]
        uy_k = neighbor_params[2]
        v_k = neighbor_params[3]
        vx_k = neighbor_params[4]
        vy_k = neighbor_params[5]

        out_params[0] = u_k + ux_k * dx + uy_k * dy
        out_params[1] = ux_k
        out_params[2] = uy_k
        out_params[3] = v_k + vx_k * dx + vy_k * dy
        out_params[4] = vx_k
        out_params[5] = vy_k

    else:  # QUADRATIC
        u_k = neighbor_params[0]
        ux_k = neighbor_params[1]
        uy_k = neighbor_params[2]
        v_k = neighbor_params[6]
        vx_k = neighbor_params[7]
        vy_k = neighbor_params[8]

        out_params[0] = u_k + ux_k * dx + uy_k * dy
        out_params[1] = ux_k
        out_params[2] = uy_k
        out_params[3] = neighbor_params[3]
        out_params[4] = neighbor_params[4]
        out_params[5] = neighbor_params[5]
        out_params[6] = v_k + vx_k * dx + vy_k * dy
        out_params[7] = vx_k
        out_params[8] = vy_k
        out_params[9] = neighbor_params[9]
        out_params[10] = neighbor_params[10]
        out_params[11] = neighbor_params[11]


@jit(nopython=True, cache=True)
def evaluate_quarter_zncc(
    ref_image, grad_x, grad_y,
    coeffs, order,
    cx, cy,
    init_params,
    xsi_min, xsi_max, eta_min, eta_max,
    xsi, eta,
    shape_type,
    n_pixels,
    f, dfdx, dfdy,
    xsi_w, eta_w,
    x_def, y_def,
    g
):
    """단일 quarter-subset에 대해 1회 warp ZNCC를 계산."""
    img_h = ref_image.shape[0]
    img_w = ref_image.shape[1]

    f_mean, f_tilde, valid = extract_reference_subset_variable(
        ref_image, grad_x, grad_y, cx, cy,
        xsi_min, xsi_max, eta_min, eta_max,
        f, dfdx, dfdy
    )
    if not valid:
        return -1.0

    warp(init_params, xsi, eta, xsi_w, eta_w, shape_type)

    for i in range(n_pixels):
        x_def[i] = float64(cx) + xsi_w[i]
        y_def[i] = float64(cy) + eta_w[i]

    if not is_inside_batch(y_def[:n_pixels], x_def[:n_pixels], img_h, img_w, order):
        return -1.0

    if order == 3:
        for i in range(n_pixels):
            g[i] = _interp2d_cubic(coeffs, y_def[i], x_def[i])
    else:
        for i in range(n_pixels):
            g[i] = _interp2d_quintic(coeffs, y_def[i], x_def[i])

    g_sum = 0.0
    for i in range(n_pixels):
        g_sum += g[i]
    g_mean = g_sum / float64(n_pixels)

    g_tilde_sq = 0.0
    for i in range(n_pixels):
        diff = g[i] - g_mean
        g_tilde_sq += diff * diff
    g_tilde = np.sqrt(g_tilde_sq)

    if g_tilde < 1e-10:
        return -1.0

    znssd = compute_znssd(f, f_mean, f_tilde, g, g_mean, g_tilde, n_pixels)
    zncc = 1.0 - 0.5 * znssd
    if zncc > 1.0 or zncc < -1.0:
        return -1.0
    return zncc


@jit(nopython=True, cache=True)
def process_poi_adss_multi(
    ref_image, grad_x, grad_y,
    coeffs, order,
    cx, cy,
    subset_size,
    max_iterations,
    convergence_threshold,
    shape_type,
    neighbor_valid,       # (4, 3) bool      ← shape 변경
    neighbor_params,      # (4, 3, n_params)  ← shape 변경
    neighbor_x,           # (4, 3)            ← shape 변경
    neighbor_y,           # (4, 3)            ← shape 변경
    zncc_threshold,
    zncc_pre_threshold,
    out_p,
    out_zncc,
    out_iter,
    out_qt,
    out_candidate_zncc,
    out_fail_info,
    f, dfdx, dfdy,
    J, H, H_inv_buf,
    p, xsi_w, eta_w,
    x_def, y_def,
    g, b, dp, p_new,
    xsi_local, eta_local,
    init_params_buf
):
    M = subset_size // 2
    n_params = get_num_params(shape_type)

    xsi_mins = np.array([-M,  0, -M,  0], dtype=np.int64)
    xsi_maxs = np.array([ 0,  M,  0,  M], dtype=np.int64)
    eta_mins = np.array([-M, -M,  0,  0], dtype=np.int64)
    eta_maxs = np.array([ 0,  0,  M,  M], dtype=np.int64)
    qt_codes = np.array([5, 6, 7, 8], dtype=np.int32)

    for i in range(4):
        out_candidate_zncc[i] = -1.0
        out_zncc[i] = 0.0
        out_iter[i] = 0
        out_qt[i] = -1
        out_fail_info[i, 0] = -1.0
        out_fail_info[i, 1] = 0.0
        out_fail_info[i, 2] = -1.0

    # === 임시 버퍼: 각 사분면의 최적 이웃 초기값 저장 ===
    best_init_params = np.zeros((4, n_params), dtype=np.float64)

    # === Step 1: 각 사분면에 대해 3이웃 후보 평가, 최고 ZNCC 이웃 선택 ===
    for i in range(4):
        # 사분면 픽셀 좌표 생성
        idx = 0
        for row in range(eta_mins[i], eta_maxs[i] + 1):
            for col in range(xsi_mins[i], xsi_maxs[i] + 1):
                xsi_local[idx] = float64(col)
                eta_local[idx] = float64(row)
                idx += 1
        n_pix = idx

        best_zncc_for_quarter = -1.0

        for c in range(3):  # 3개 이웃 후보 순회
            if not neighbor_valid[i, c]:
                continue

            predict_initial_params(
                neighbor_x[i, c], neighbor_y[i, c],
                neighbor_params[i, c],
                float64(cx), float64(cy),
                shape_type, init_params_buf
            )

            zncc_1warp = evaluate_quarter_zncc(
                ref_image, grad_x, grad_y,
                coeffs, order, cx, cy,
                init_params_buf,
                xsi_mins[i], xsi_maxs[i], eta_mins[i], eta_maxs[i],
                xsi_local, eta_local,
                shape_type, n_pix,
                f, dfdx, dfdy, xsi_w, eta_w, x_def, y_def, g
            )

            if zncc_1warp > best_zncc_for_quarter:
                best_zncc_for_quarter = zncc_1warp
                for k in range(n_params):
                    best_init_params[i, k] = init_params_buf[k]

        out_candidate_zncc[i] = best_zncc_for_quarter

    # === Step 2: pre_threshold 이상인 사분면 모두 IC-GN ===
    n_recovered = 0
    quarter_subset_size = M + 1

    for i in range(4):
        if out_candidate_zncc[i] < zncc_pre_threshold:
            continue

        idx = 0
        for row in range(eta_mins[i], eta_maxs[i] + 1):
            for col in range(xsi_mins[i], xsi_maxs[i] + 1):
                xsi_local[idx] = float64(col)
                eta_local[idx] = float64(row)
                idx += 1
        n_pixels = idx

        # Step 1에서 선택된 최적 이웃의 초기값 사용
        for k in range(n_params):
            init_params_buf[k] = best_init_params[i, k]

        f_mean, f_tilde, valid = extract_reference_subset_variable(
            ref_image, grad_x, grad_y, cx, cy,
            xsi_mins[i], xsi_maxs[i], eta_mins[i], eta_maxs[i],
            f, dfdx, dfdy
        )
        if not valid:
            out_fail_info[i, 0] = -2.0
            out_fail_info[i, 1] = 0.0
            out_fail_info[i, 2] = 99.0
            continue

        compute_steepest_descent(
            dfdx[:n_pixels], dfdy[:n_pixels],
            xsi_local[:n_pixels], eta_local[:n_pixels],
            J[:n_pixels], shape_type
        )

        compute_hessian(J[:n_pixels], H)
        det = np.linalg.det(H)
        if abs(det) < 1e-30:
            out_fail_info[i, 0] = -3.0
            out_fail_info[i, 1] = 0.0
            out_fail_info[i, 2] = 4.0
            continue
        H_inv = np.linalg.inv(H)
        for ii in range(n_params):
            for jj in range(n_params):
                H_inv_buf[ii, jj] = H_inv[ii, jj]

        for k in range(n_params):
            p[k] = init_params_buf[k]

        zncc, n_iter, conv, fail_code = icgn_iterate(
            f[:n_pixels], f_mean, f_tilde,
            J[:n_pixels], H_inv_buf,
            coeffs, order,
            ref_image.shape[0], ref_image.shape[1],
            cx, cy,
            xsi_local[:n_pixels], eta_local[:n_pixels],
            p, quarter_subset_size,
            max_iterations, convergence_threshold,
            shape_type, n_pixels, n_params,
            xsi_w[:n_pixels], eta_w[:n_pixels],
            x_def[:n_pixels], y_def[:n_pixels],
            g[:n_pixels], b, dp, p_new
        )

        out_fail_info[i, 0] = zncc
        out_fail_info[i, 1] = float64(n_iter)
        out_fail_info[i, 2] = float64(fail_code)

        if conv and zncc >= zncc_threshold:
            for k in range(n_params):
                out_p[n_recovered, k] = p[k]
            out_zncc[n_recovered] = zncc
            out_iter[n_recovered] = n_iter
            out_qt[n_recovered] = qt_codes[i]
            n_recovered += 1
        elif (fail_code == 2
              and n_iter <= 1
              and out_candidate_zncc[i] >= zncc_threshold):
            for k in range(n_params):
                out_p[n_recovered, k] = best_init_params[i, k]
            out_zncc[n_recovered] = out_candidate_zncc[i]
            out_iter[n_recovered] = 0
            out_qt[n_recovered] = qt_codes[i]
            n_recovered += 1

    return n_recovered


@jit(nopython=True, parallel=True, cache=True)
def process_bad_pois_adss_multi_parallel(
    ref_image, grad_x, grad_y,
    coeffs, order,
    points_x, points_y,
    subset_size,
    max_iterations,
    convergence_threshold,
    shape_type,
    bad_indices,
    all_neighbor_valid,
    all_neighbor_params,
    all_neighbor_x,
    all_neighbor_y,
    zncc_threshold,
    zncc_pre_threshold,
    result_p,
    result_zncc,
    result_iter,
    result_qt,
    result_parent,
    result_count,
    result_candidate_zncc,
    result_fail_info,
    buf_f, buf_dfdx, buf_dfdy,
    buf_J, buf_H, buf_H_inv,
    buf_p, buf_xsi_w, buf_eta_w,
    buf_x_def, buf_y_def,
    buf_g, buf_b, buf_dp, buf_p_new,
    buf_xsi_local, buf_eta_local,
    buf_init_params,
    buf_out_p,
    buf_out_zncc,
    buf_out_iter,
    buf_out_qt,
    buf_out_cand_zncc,
    buf_out_fail_info,
):
    """
    불량 POI들을 prange 병렬 처리.
    각 POI에서 0~4개의 sub-POI가 생성되며,
    결과는 k*4 ~ k*4+3 슬롯에 저장된다.
    """
    n_bad = len(bad_indices)
    n_params = buf_p.shape[1]

    for k in prange(n_bad):
        poi_idx = bad_indices[k]
        cx = points_x[poi_idx]
        cy = points_y[poi_idx]

        n_rec = process_poi_adss_multi(
            ref_image, grad_x, grad_y,
            coeffs, order, cx, cy,
            subset_size, max_iterations, convergence_threshold,
            shape_type,
            all_neighbor_valid[k],
            all_neighbor_params[k],
            all_neighbor_x[k],
            all_neighbor_y[k],
            zncc_threshold,
            zncc_pre_threshold,
            buf_out_p[k],
            buf_out_zncc[k],
            buf_out_iter[k],
            buf_out_qt[k],
            buf_out_cand_zncc[k],
            buf_out_fail_info[k],
            buf_f[k], buf_dfdx[k], buf_dfdy[k],
            buf_J[k], buf_H[k], buf_H_inv[k],
            buf_p[k], buf_xsi_w[k], buf_eta_w[k],
            buf_x_def[k], buf_y_def[k],
            buf_g[k], buf_b[k], buf_dp[k], buf_p_new[k],
            buf_xsi_local[k], buf_eta_local[k],
            buf_init_params[k]
        )

        result_count[k] = n_rec
        for d in range(4):
            result_candidate_zncc[k, d] = buf_out_cand_zncc[k, d]
            result_fail_info[k, d, 0] = buf_out_fail_info[k, d, 0]
            result_fail_info[k, d, 1] = buf_out_fail_info[k, d, 1]
            result_fail_info[k, d, 2] = buf_out_fail_info[k, d, 2]

        offset = k * 4
        for j in range(n_rec):
            for m in range(n_params):
                result_p[offset + j, m] = buf_out_p[k, j, m]
            result_zncc[offset + j] = buf_out_zncc[k, j]
            result_iter[offset + j] = buf_out_iter[k, j]
            result_qt[offset + j] = buf_out_qt[k, j]
            result_parent[offset + j] = poi_idx


def allocate_adss_multi_batch_buffers(n_bad, n_pixels_max, n_params):
    """
    ADSS-DIC v2 배치 처리용 버퍼 할당.
    """
    return {
        'f': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'dfdx': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'dfdy': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'J': np.zeros((n_bad, n_pixels_max, n_params), dtype=np.float64),
        'H': np.zeros((n_bad, n_params, n_params), dtype=np.float64),
        'H_inv': np.zeros((n_bad, n_params, n_params), dtype=np.float64),
        'p': np.zeros((n_bad, n_params), dtype=np.float64),
        'xsi_w': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'eta_w': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'x_def': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'y_def': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'g': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'b': np.zeros((n_bad, n_params), dtype=np.float64),
        'dp': np.zeros((n_bad, n_params), dtype=np.float64),
        'p_new': np.zeros((n_bad, n_params), dtype=np.float64),
        'xsi_local': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'eta_local': np.zeros((n_bad, n_pixels_max), dtype=np.float64),
        'init_params': np.zeros((n_bad, n_params), dtype=np.float64),
        'out_p': np.zeros((n_bad, 4, n_params), dtype=np.float64),
        'out_zncc': np.zeros((n_bad, 4), dtype=np.float64),
        'out_iter': np.zeros((n_bad, 4), dtype=np.int32),
        'out_qt': np.zeros((n_bad, 4), dtype=np.int32),
        'out_cand_zncc': np.zeros((n_bad, 4), dtype=np.float64),
        'out_fail_info': np.full((n_bad, 4, 3), -1.0, dtype=np.float64),
    }
