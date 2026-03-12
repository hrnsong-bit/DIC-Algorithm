from types import SimpleNamespace

import numpy as np

from gui.controllers.renderers import FieldRenderer
from speckle.core.optimization.results import ADSSResult


def _make_renderer():
    ctrl = SimpleNamespace(
        view=SimpleNamespace(
            display_mode_var=SimpleNamespace(get=lambda: "invalid"),
            display_source_var=SimpleNamespace(get=lambda: "auto"),
            get_color_range=lambda: None,
            update_colorbar=lambda *args, **kwargs: None,
        )
    )
    return FieldRenderer(ctrl)


def _make_adss_result(quarter_types, sub_u_values):
    n_sub = len(quarter_types)
    parameters = np.zeros((n_sub, 6), dtype=np.float64)
    parameters[:, 0] = np.asarray(sub_u_values, dtype=np.float64)
    return ADSSResult(
        parent_indices=np.zeros(n_sub, dtype=np.int64),
        quarter_types=np.asarray(quarter_types, dtype=np.int32),
        points_x=np.full(n_sub, 10, dtype=np.int64),
        points_y=np.full(n_sub, 10, dtype=np.int64),
        parameters=parameters,
        zncc_values=np.full(n_sub, 0.99, dtype=np.float64),
        iterations=np.full(n_sub, 8, dtype=np.int32),
        xsi_mins=np.zeros(n_sub, dtype=np.int32),
        xsi_maxs=np.zeros(n_sub, dtype=np.int32),
        eta_mins=np.zeros(n_sub, dtype=np.int32),
        eta_maxs=np.zeros(n_sub, dtype=np.int32),
        n_bad_original=1,
        n_sub_total=n_sub,
        n_parent_recovered=1,
        n_unrecoverable=0,
    )


def _make_result(adss_result):
    points_x = np.array([10, 20, 10, 20], dtype=np.int64)
    points_y = np.array([10, 10, 20, 20], dtype=np.int64)
    return SimpleNamespace(
        n_points=4,
        points_x=points_x,
        points_y=points_y,
        valid_mask=np.array([True, True, True, True], dtype=np.bool_),
        disp_u=np.array([999.0, 2.0, 3.0, 4.0], dtype=np.float64),
        disp_v=np.zeros(4, dtype=np.float64),
        zncc_values=np.full(4, 0.99, dtype=np.float64),
        fft_valid_mask=np.ones(4, dtype=np.bool_),
        adss_result=adss_result,
    )


def test_to_grid_keeps_parent_fill_outside_recovered_triangle_cells():
    renderer = _make_renderer()
    adss_result = _make_adss_result([1], [111.0])
    result = _make_result(adss_result)

    grid, _, _ = renderer._to_grid(result, result.disp_u)

    assert grid[0, 0] == 111.0
    assert grid[0, 1] == 111.0
    assert grid[1, 0] == 999.0
    assert grid[1, 1] == 999.0

    assert grid[0, 2] == 2.0
    assert grid[0, 3] == 2.0
    assert grid[1, 2] == 2.0
    assert grid[1, 3] == 2.0


def test_invalid_overlay_draws_adss_triangle_for_reflected_parent():
    renderer = _make_renderer()
    adss_result = _make_adss_result([1], [111.0])
    result = _make_result(adss_result)
    img = np.zeros((32, 32, 3), dtype=np.uint8)

    rendered = renderer._draw_invalid_points(img, result)

    np.testing.assert_array_equal(rendered[6, 10], np.array([255, 180, 0], dtype=np.uint8))


def test_adss_display_mask_keeps_only_triangle_region():
    renderer = _make_renderer()
    adss_result = _make_adss_result([1], [111.0])
    result = _make_result(adss_result)
    base_img = np.zeros((32, 32, 3), dtype=np.uint8)
    rendered_img = np.full((32, 32, 3), 255, dtype=np.uint8)

    masked = renderer._apply_adss_display_mask(base_img, rendered_img, result)

    np.testing.assert_array_equal(masked[6, 10], np.array([255, 255, 255], dtype=np.uint8))
    np.testing.assert_array_equal(masked[14, 10], np.array([0, 0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(masked[20, 20], np.array([255, 255, 255], dtype=np.uint8))


def test_display_source_auto_defaults_to_raw_for_displacement():
    renderer = _make_renderer()

    assert renderer._resolve_display_source("u_field") == "raw"
    assert renderer._resolve_display_source("magnitude") == "raw"


def test_display_source_auto_defaults_to_processed_for_strain():
    renderer = _make_renderer()

    assert renderer._resolve_display_source("exx") == "processed"
    assert renderer._resolve_display_source("von_mises") == "processed"


def test_display_source_manual_override_is_respected():
    ctrl = SimpleNamespace(
        view=SimpleNamespace(
            display_mode_var=SimpleNamespace(get=lambda: "u_field"),
            display_source_var=SimpleNamespace(get=lambda: "processed"),
            get_color_range=lambda: None,
            update_colorbar=lambda *args, **kwargs: None,
        )
    )
    renderer = FieldRenderer(ctrl)

    assert renderer._resolve_display_source("u_field") == "processed"


def test_raw_magnitude_draws_adss_subsets():
    renderer = _make_renderer()
    adss_result = _make_adss_result([1], [111.0])
    result = _make_result(adss_result)
    img = np.zeros((32, 32, 3), dtype=np.uint8)

    rendered = renderer._draw_magnitude(img, result)

    assert np.any(rendered[6, 10] != np.array([0, 0, 0], dtype=np.uint8))
    np.testing.assert_array_equal(rendered[14, 10], np.array([0, 0, 0], dtype=np.uint8))
