from __future__ import annotations

from pathlib import Path

import pytest
from PySide6.QtCore import Qt

from app.dimensions import (
    dimension_entry_text,
    parse_dimension_entry,
    parse_displacement_entry,
    parse_scalar_entry,
)
from app.viewport.gl_widget import (
    GLWidget,
    apply_typed_create_dimensions,
    apply_typed_move_delta,
    apply_typed_start_point,
    bottom_center_overlay_position,
    centered_create_endpoints,
    constrain_move_point,
    constrain_reference_point,
    create_effective_endpoints,
    create_input_label,
    create_measurement_components,
    create_modifier_status_text,
    create_preview_torus_minor_radius,
    create_size_prompt,
    create_start_prompt,
    create_typed_parameters,
    cursor_preview_active,
    empty_scene_source,
    keyboard_move_delta,
    move_preview_delta,
    move_dimension_prompt,
    next_reference_plane,
    point_shape_minimum_points,
    reference_plane_coordinate_components,
    reference_plane_context,
    reference_plane_label,
    reference_view_for_key,
    should_clear_idle_selection_for_key,
    should_cycle_reference_plane_for_key,
    should_defer_create_release,
    should_frame_scene_for_key,
    should_refresh_create_modifier_preview_for_key,
    should_snap_reference_point,
    snap_status_text,
)
from app.signals import signals
from core.scene import SceneDocument
from core.sdf import Box, Cylinder, PlacedSDF2D, Sphere, SquareProfile, Torus


def test_shift_constraint_equalizes_shape_extents_on_reference_plane() -> None:
    constrained = constrain_reference_point(
        point=(0.25, 0.0, -0.75),
        start=(0.0, 0.0, 0.0),
        reference_plane="xz",
        kind="rectangle",
    )

    assert constrained == (0.75, 0.0, -0.75)


def test_bottom_center_overlay_position_keeps_toolbar_inside_viewport() -> None:
    assert bottom_center_overlay_position(640, 480, 220, 36, 12) == (
        210,
        432,
    )
    assert bottom_center_overlay_position(120, 80, 220, 36, 12) == (
        12,
        32,
    )


def test_point_shape_minimum_points_match_geometry() -> None:
    assert point_shape_minimum_points("polyline") == 2
    assert point_shape_minimum_points("bezier_curve") == 3
    assert point_shape_minimum_points("bezier_polycurve") == 3
    assert point_shape_minimum_points("polygon") == 3


def test_shift_constraint_locks_segment_to_dominant_axis() -> None:
    constrained = constrain_reference_point(
        point=(0.25, 0.9, 0.0),
        start=(0.0, 0.0, 0.0),
        reference_plane="xy",
        kind="segment",
    )

    assert constrained == (0.0, 0.9, 0.0)


def test_shift_move_constraint_locks_to_dominant_reference_axis() -> None:
    constrained = constrain_move_point(
        point=(0.25, 0.9, 0.0),
        start=(0.0, 0.0, 0.0),
        reference_plane="xy",
    )

    assert constrained == (0.0, 0.9, 0.0)


def test_shift_move_constraint_uses_active_reference_plane_axes() -> None:
    constrained = constrain_move_point(
        point=(2.0, 0.0, -0.5),
        start=(1.0, 0.0, 1.0),
        reference_plane="xz",
    )

    assert constrained == (1.0, 0.0, -0.5)


def test_move_preview_delta_applies_shift_lock_without_losing_raw_drag() -> None:
    unlocked = move_preview_delta(
        accumulated_delta=(0.1, 0.0, 0.0),
        start=(0.0, 0.0, 0.0),
        current=(0.25, 0.9, 0.0),
        reference_plane="xy",
        modifiers=Qt.KeyboardModifier.NoModifier,
    )
    locked = move_preview_delta(
        accumulated_delta=(0.1, 0.0, 0.0),
        start=(0.0, 0.0, 0.0),
        current=(0.25, 0.9, 0.0),
        reference_plane="xy",
        modifiers=Qt.KeyboardModifier.ShiftModifier,
    )

    assert unlocked == pytest.approx((0.35, 0.9, 0.0))
    assert locked == pytest.approx((0.1, 0.9, 0.0))


def test_click_create_release_defers_for_cursor_preview_entry() -> None:
    assert should_defer_create_release(
        start_screen=(100, 100),
        end_screen=(102, 101),
        has_typed_input=False,
    )


def test_create_drag_or_typed_release_commits_immediately() -> None:
    assert not should_defer_create_release(
        start_screen=(100, 100),
        end_screen=(140, 100),
        has_typed_input=False,
    )
    assert not should_defer_create_release(
        start_screen=(100, 100),
        end_screen=(100, 100),
        has_typed_input=True,
    )


def test_create_preview_reset_keeps_drawing_tool_active() -> None:
    class FakeWidget:
        def __init__(self) -> None:
            self._interaction_tool = None
            self._tool_start_screen = object()
            self._tool_start_world = (1.0, 2.0, 3.0)
            self._tool_current_world = (4.0, 5.0, 6.0)
            self._tool_hover_world = (7.0, 8.0, 9.0)
            self._dimension_input = "50mm"
            self.cursor = None
            self.measurement_updated = False
            self.repaint_requested = False

        def setCursor(self, cursor: object) -> None:
            self.cursor = cursor

        def _update_measurement_readout(self) -> None:
            self.measurement_updated = True

        def update(self) -> None:
            self.repaint_requested = True

    widget = FakeWidget()

    GLWidget._reset_create_preview(widget, "box")

    assert widget._interaction_tool == ("create", "box")
    assert widget._tool_start_screen is None
    assert widget._tool_start_world is None
    assert widget._tool_current_world is None
    assert widget._tool_hover_world is None
    assert widget._dimension_input == ""
    assert widget.cursor == Qt.CursorShape.CrossCursor
    assert widget.measurement_updated
    assert widget.repaint_requested


def test_begin_create_tool_clears_previous_preview_state() -> None:
    class FakeWidget:
        def __init__(self) -> None:
            self._interaction_tool = ("create", "circle")
            self._tool_start_screen = object()
            self._tool_start_world = (1.0, 2.0, 3.0)
            self._tool_current_world = (4.0, 5.0, 6.0)
            self._tool_hover_world = (7.0, 8.0, 9.0)
            self._move_preview_delta = (0.2, 0.0, 0.0)
            self._dimension_input = "50mm"
            self.cursor = None
            self.focused = False
            self.measurement_updated = False
            self.repaint_requested = False

        def _clear_scene_hover(self) -> None:
            pass

        def setCursor(self, cursor: object) -> None:
            self.cursor = cursor

        def setFocus(self) -> None:
            self.focused = True

        def _update_measurement_readout(self) -> None:
            self.measurement_updated = True

        def update(self) -> None:
            self.repaint_requested = True

    widget = FakeWidget()

    GLWidget.begin_create_tool(widget, "box")

    assert widget._interaction_tool == ("create", "box")
    assert widget._tool_start_screen is None
    assert widget._tool_start_world is None
    assert widget._tool_current_world is None
    assert widget._tool_hover_world is None
    assert widget._move_preview_delta == (0.0, 0.0, 0.0)
    assert widget._dimension_input == ""
    assert widget.cursor == Qt.CursorShape.CrossCursor
    assert widget.focused
    assert widget.measurement_updated
    assert widget.repaint_requested


def test_begin_move_tool_clears_previous_typed_input_and_preview_state() -> None:
    class FakeWidget:
        def __init__(self) -> None:
            self._interaction_tool = ("create", "box")
            self._move_preview_delta = (0.1, 0.2, 0.3)
            self._tool_start_screen = object()
            self._tool_start_world = (1.0, 2.0, 3.0)
            self._tool_current_world = (4.0, 5.0, 6.0)
            self._tool_hover_world = (7.0, 8.0, 9.0)
            self._dimension_input = "50mm"
            self.cursor = None
            self.focused = False
            self.measurement_updated = False
            self.repaint_requested = False

        def _clear_scene_hover(self) -> None:
            pass

        def setCursor(self, cursor: object) -> None:
            self.cursor = cursor

        def setFocus(self) -> None:
            self.focused = True

        def _update_measurement_readout(self) -> None:
            self.measurement_updated = True

        def update(self) -> None:
            self.repaint_requested = True

    widget = FakeWidget()

    GLWidget.begin_move_tool(widget, 42)

    assert widget._interaction_tool == ("move", 42)
    assert widget._move_preview_delta == (0.0, 0.0, 0.0)
    assert widget._tool_start_screen is None
    assert widget._tool_start_world is None
    assert widget._tool_current_world is None
    assert widget._tool_hover_world is None
    assert widget._dimension_input == ""
    assert widget.cursor == Qt.CursorShape.SizeAllCursor
    assert widget.focused
    assert widget.measurement_updated
    assert widget.repaint_requested


def test_begin_boundary_region_tool_clears_previous_generic_preview_state() -> None:
    class FakeRoot:
        object_id = 42

    class FakeWidget:
        def __init__(self) -> None:
            self._interaction_tool = ("create", "box")
            self._move_preview_delta = (0.1, 0.2, 0.3)
            self._tool_start_screen = object()
            self._tool_start_world = (1.0, 2.0, 3.0)
            self._tool_current_world = (4.0, 5.0, 6.0)
            self._tool_hover_world = (7.0, 8.0, 9.0)
            self._dimension_input = "50mm"
            self.cursor = None
            self.focused = False

        def _clear_scene_hover(self) -> None:
            pass

        def setCursor(self, cursor: object) -> None:
            self.cursor = cursor

        def setFocus(self) -> None:
            self.focused = True

    widget = FakeWidget()

    GLWidget.begin_boundary_region_tool(widget, FakeRoot())

    assert widget._boundary_pick_root.object_id == 42
    assert widget._boundary_selection_active
    assert widget._interaction_tool == ("boundary_region", 42)
    assert widget._tool_start_screen is None
    assert widget._tool_start_world is None
    assert widget._tool_current_world is None
    assert widget._tool_hover_world is None
    assert widget._move_preview_delta == (0.0, 0.0, 0.0)
    assert widget._dimension_input == ""
    assert widget.cursor == Qt.CursorShape.CrossCursor
    assert widget.focused


def test_parse_dimension_entry_accepts_cad_separators() -> None:
    assert parse_dimension_entry("2x1.5") == (2.0, 1.5)
    assert parse_dimension_entry("2, 1.5") == (2.0, 1.5)


def test_parse_dimension_entry_accepts_cad_units_in_meters() -> None:
    assert parse_dimension_entry("50mm x 2.5cm") == (0.05, 0.025)
    assert parse_dimension_entry("1m, 2 meters") == (1.0, 2.0)
    assert parse_dimension_entry('2in; 1ft') == (0.0508, 0.3048)


def test_parse_dimension_entry_accepts_safe_formulas_with_units() -> None:
    assert parse_dimension_entry("50mm*2") == (0.1,)
    assert parse_dimension_entry("1m/4 x (2ft + 6in)") == pytest.approx(
        (0.25, 0.762),
    )


def test_parse_scalar_entry_accepts_unitless_cad_formulas() -> None:
    assert parse_scalar_entry("45 + 15") == pytest.approx(60.0)
    assert parse_scalar_entry("1/2") == pytest.approx(0.5)
    assert parse_scalar_entry("1e-3") == pytest.approx(0.001)


def test_parse_scalar_entry_rejects_units_and_unsafe_expressions() -> None:
    with pytest.raises(ValueError, match="scalar"):
        parse_scalar_entry("10mm")
    with pytest.raises(ValueError, match="scalar"):
        parse_scalar_entry("__import__('os')")
    with pytest.raises(ValueError, match="scalar"):
        parse_scalar_entry("1/0")


def test_parse_dimension_entry_rejects_non_positive_values() -> None:
    with pytest.raises(ValueError):
        parse_dimension_entry("2, 0")
    with pytest.raises(ValueError, match="dimension entries"):
        parse_dimension_entry("2kg")
    with pytest.raises(ValueError, match="dimension entries"):
        parse_dimension_entry("2m/0")


def test_parse_displacement_entry_accepts_signed_values() -> None:
    assert parse_displacement_entry("-2, +1.5, 0") == (-2.0, 1.5, 0.0)


def test_parse_displacement_entry_accepts_signed_cad_units() -> None:
    assert parse_displacement_entry("-25mm, +1cm, 0") == (-0.025, 0.01, 0.0)


def test_parse_displacement_entry_accepts_safe_formulas() -> None:
    assert parse_displacement_entry("-(25mm*2), 1m/4") == (-0.05, 0.25)


def test_dimension_entry_text_allows_cad_unit_abbreviations() -> None:
    text = ""
    for character in "50MM x 2.5cm":
        text += dimension_entry_text(character, "create", True, text)
    assert text == "50mm x 2.5cm"

    text = ""
    for character in '2IN; 1ft':
        text += dimension_entry_text(character, "move", False, text)
    assert text == '2in; 1ft'


def test_dimension_entry_text_allows_long_units_after_entry_starts() -> None:
    text = ""
    for character in "2 meters":
        text += dimension_entry_text(character, "create", True, text)

    assert text == "2 meters"
    assert parse_dimension_entry(text) == (2.0,)


def test_dimension_entry_text_keeps_feet_unit_available_after_digit() -> None:
    text = "1"
    for character in "ft":
        text += dimension_entry_text(character, "create", True, text)

    assert text == "1ft"
    assert parse_dimension_entry(text) == pytest.approx((0.3048,))


def test_dimension_entry_text_allows_long_signed_move_units() -> None:
    text = ""
    for character in "-25 millimeters":
        text += dimension_entry_text(character, "move", False, text)

    assert text == "-25 millimeters"
    assert parse_displacement_entry(text) == (-0.025,)


def test_dimension_entry_text_allows_scientific_notation_signs() -> None:
    text = ""
    for character in "1e-3 m":
        text += dimension_entry_text(character, "create", True, text)

    assert text == "1e-3 m"
    assert parse_dimension_entry(text) == (0.001,)


def test_dimension_entry_text_allows_formula_characters() -> None:
    text = ""
    for character in "(50mm*2)+1cm":
        text += dimension_entry_text(character, "create", True, text)

    assert text == "(50mm*2)+1cm"
    assert parse_dimension_entry(text) == pytest.approx((0.11,))


def test_dimension_entry_text_keeps_move_keys_available() -> None:
    assert dimension_entry_text("w", "move", False) == ""
    assert dimension_entry_text("a", "move", False) == ""
    assert dimension_entry_text("s", "move", False) == ""
    assert dimension_entry_text("d", "move", False) == ""
    assert dimension_entry_text("q", "move", False) == ""
    assert dimension_entry_text("e", "move", False) == ""
    assert dimension_entry_text("e", "move", False, "1") == "e"


def test_dimension_entry_text_rejects_invalid_entry_starts() -> None:
    assert dimension_entry_text("m", "create", True) == ""
    assert dimension_entry_text("f", "move", False) == ""
    assert dimension_entry_text("x", "create", True) == ""
    assert dimension_entry_text(",", "create", True) == ""
    assert dimension_entry_text(";", "create", True) == ""
    assert dimension_entry_text('"', "create", True) == ""
    assert dimension_entry_text(":", "create", True, "1") == ""


def test_dimension_entry_text_allows_decimal_start() -> None:
    text = ""
    for character in ".5m":
        text += dimension_entry_text(character, "create", True, text)

    assert text == ".5m"
    assert parse_dimension_entry(text) == (0.5,)


def test_dimension_entry_text_allows_signs_only_for_start_or_move() -> None:
    assert dimension_entry_text("-1", "create", False) == "-1"
    assert dimension_entry_text("-1", "move", False) == "-1"
    assert dimension_entry_text("-1", "create", True) == ""


def test_typed_dimensions_apply_to_active_reference_plane() -> None:
    end = apply_typed_create_dimensions(
        start=(1.0, 0.0, 2.0),
        current=(0.25, 0.0, 3.0),
        reference_plane="xz",
        kind="rectangle",
        dimensions=(4.0, 2.0),
    )

    assert end == (-3.0, 0.0, 4.0)


def test_typed_segment_dimension_uses_dominant_drag_axis() -> None:
    end = apply_typed_create_dimensions(
        start=(0.0, 0.0, 0.0),
        current=(0.2, -1.0, 0.0),
        reference_plane="xy",
        kind="segment",
        dimensions=(3.0,),
    )

    assert end == (0.0, -3.0, 0.0)


def test_typed_radial_dimension_uses_diameter_along_drag_direction() -> None:
    end = apply_typed_create_dimensions(
        start=(0.0, 0.0, 0.0),
        current=(1.0, 1.0, 0.0),
        reference_plane="xy",
        kind="circle",
        dimensions=(2.0,),
    )
    diameter = (end[0] * end[0] + end[1] * end[1]) ** 0.5

    assert diameter == pytest.approx(2.0)


def test_typed_single_diameter_radial_shapes_ignore_extra_values() -> None:
    end = apply_typed_create_dimensions(
        start=(0.0, 0.0, 0.0),
        current=(1.0, 1.0, 0.0),
        reference_plane="xy",
        kind="circle",
        dimensions=(2.0, 9.0),
    )
    diameter = (end[0] * end[0] + end[1] * end[1]) ** 0.5

    assert diameter == pytest.approx(2.0)


def test_typed_sphere_extra_values_keep_first_value_as_diameter() -> None:
    start, end = create_effective_endpoints(
        anchor=(0.0, 0.0, 0.0),
        current=(1.0, 1.0, 0.0),
        reference_plane="xy",
        kind="sphere",
        dimensions=(2.0, 9.0),
        centered=False,
    )
    document = SceneDocument()
    handle = document.add_primitive_from_drag("sphere", start, end)
    sphere = document.node(handle)

    assert isinstance(sphere, Sphere)
    assert 2.0 * sphere.radius == pytest.approx(2.0)


def test_typed_polygon_extra_values_keep_first_value_as_diameter() -> None:
    start, end = create_effective_endpoints(
        anchor=(0.0, 0.0, 0.0),
        current=(1.0, 0.0, 0.0),
        reference_plane="xy",
        kind="regular_polygon",
        dimensions=(2.0, 9.0),
        centered=False,
    )
    document = SceneDocument()
    handle = document.add_primitive_from_drag("regular_polygon", start, end)
    polygon = document.node(handle)

    assert isinstance(polygon, PlacedSDF2D)
    assert polygon.profile.radius == pytest.approx(1.0)


def test_typed_square_extra_values_keep_first_value_as_size() -> None:
    start, end = create_effective_endpoints(
        anchor=(0.0, 0.0, 0.0),
        current=(1.0, -1.0, 0.0),
        reference_plane="xy",
        kind="square",
        dimensions=(2.0, 9.0),
        centered=False,
    )
    document = SceneDocument()
    handle = document.add_primitive_from_drag("square", start, end)
    square = document.node(handle)

    assert isinstance(square, PlacedSDF2D)
    assert isinstance(square.profile, SquareProfile)
    assert 2.0 * square.profile.half_size == pytest.approx(2.0)


def test_centered_create_endpoints_keep_anchor_as_center() -> None:
    start, end = centered_create_endpoints(
        center=(1.0, 2.0, 0.0),
        corner=(1.5, 3.0, 0.0),
    )

    assert start == (0.5, 1.0, 0.0)
    assert end == (1.5, 3.0, 0.0)


def test_create_modifier_keys_request_preview_refresh() -> None:
    assert should_refresh_create_modifier_preview_for_key(Qt.Key.Key_Control)
    assert should_refresh_create_modifier_preview_for_key(Qt.Key.Key_Shift)
    assert not should_refresh_create_modifier_preview_for_key(Qt.Key.Key_Alt)


def test_plain_f_frames_scene_only_as_idle_viewport_command() -> None:
    assert should_frame_scene_for_key(Qt.Key.Key_F, Qt.KeyboardModifier.NoModifier)
    assert not should_frame_scene_for_key(Qt.Key.Key_F, Qt.KeyboardModifier.ShiftModifier)
    assert not should_frame_scene_for_key(Qt.Key.Key_G, Qt.KeyboardModifier.NoModifier)


def test_create_modifier_status_text_names_active_cad_constraints() -> None:
    assert create_modifier_status_text(Qt.KeyboardModifier.NoModifier) == ""
    assert create_modifier_status_text(Qt.KeyboardModifier.ControlModifier) == "Center"
    assert create_modifier_status_text(Qt.KeyboardModifier.ShiftModifier) == "Lock"
    assert create_modifier_status_text(
        Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier
    ) == "Center  Lock"


def test_create_modifier_refresh_repaints_active_preview() -> None:
    class FakeWidget:
        def __init__(self) -> None:
            self._interaction_tool = ("create", "rectangle")
            self._tool_start_world = (0.0, 0.0, 0.0)
            self._tool_current_world = (1.0, 2.0, 0.0)
            self.measurement_updated = False
            self.repaint_requested = False

        def _update_measurement_readout(self) -> None:
            self.measurement_updated = True

        def update(self) -> None:
            self.repaint_requested = True

    widget = FakeWidget()

    assert GLWidget._refresh_interaction_modifier_preview_for_key(
        widget,
        Qt.Key.Key_Control,
    )
    assert widget.measurement_updated
    assert widget.repaint_requested

    widget.measurement_updated = False
    widget.repaint_requested = False
    assert GLWidget._refresh_interaction_modifier_preview_for_key(
        widget,
        Qt.Key.Key_Shift,
    )
    assert widget.measurement_updated
    assert widget.repaint_requested


def test_centered_typed_dimensions_remain_full_model_size() -> None:
    start, end = create_effective_endpoints(
        anchor=(1.0, 2.0, 0.0),
        current=(2.0, 3.0, 0.0),
        reference_plane="xy",
        kind="rectangle",
        dimensions=(4.0, 2.0),
        centered=True,
    )

    assert start == (-1.0, 1.0, 0.0)
    assert end == (3.0, 3.0, 0.0)


def test_typed_box_three_dimensions_create_exact_xyz_size() -> None:
    start, end = create_effective_endpoints(
        anchor=(0.0, 0.0, 0.0),
        current=(1.0, 1.0, 0.0),
        reference_plane="xy",
        kind="box",
        dimensions=(2.0, 4.0, 6.0),
        centered=False,
    )
    document = SceneDocument()
    handle = document.add_primitive_from_drag("box", start, end)
    box = document.node(handle)

    assert isinstance(box, Box)
    assert box.half_size == pytest.approx((1.0, 2.0, 3.0))


def test_centered_radial_typed_dimension_remains_full_diameter() -> None:
    start, end = create_effective_endpoints(
        anchor=(1.0, 1.0, 0.0),
        current=(2.0, 2.0, 0.0),
        reference_plane="xy",
        kind="sphere",
        dimensions=(4.0,),
        centered=True,
    )
    diameter = ((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) ** 0.5

    assert diameter == pytest.approx(4.0)


def test_typed_cylinder_single_dimension_creates_exact_diameter() -> None:
    start, end = create_effective_endpoints(
        anchor=(0.0, 0.0, 0.0),
        current=(1.0, 1.0, 0.0),
        reference_plane="xy",
        kind="cylinder",
        dimensions=(2.0,),
        centered=False,
    )
    document = SceneDocument()
    handle = document.add_primitive_from_drag("cylinder", start, end)
    cylinder = document.node(handle)

    assert isinstance(cylinder, Cylinder)
    assert 2.0 * cylinder.radius == pytest.approx(2.0)


def test_typed_cylinder_side_view_dimensions_create_diameter_and_height() -> None:
    start, end = create_effective_endpoints(
        anchor=(0.0, 0.0, 0.0),
        current=(1.0, 0.0, 1.0),
        reference_plane="xz",
        kind="cylinder",
        dimensions=(2.0, 5.0),
        centered=False,
    )
    document = SceneDocument()
    handle = document.add_primitive_from_drag("cylinder", start, end)
    cylinder = document.node(handle)

    assert isinstance(cylinder, Cylinder)
    assert 2.0 * cylinder.radius == pytest.approx(2.0)
    assert 2.0 * cylinder.half_height == pytest.approx(5.0)


def test_typed_cylinder_top_view_dimensions_create_diameter_and_height() -> None:
    start, end = create_effective_endpoints(
        anchor=(0.0, 0.0, 0.0),
        current=(1.0, 1.0, 0.0),
        reference_plane="xy",
        kind="cylinder",
        dimensions=(2.0, 5.0),
        centered=False,
    )
    document = SceneDocument()
    handle = document.add_primitive_from_drag("cylinder", start, end)
    cylinder = document.node(handle)

    assert isinstance(cylinder, Cylinder)
    assert 2.0 * cylinder.radius == pytest.approx(2.0)
    assert 2.0 * cylinder.half_height == pytest.approx(5.0)


def test_cylinder_preview_shader_matches_committed_radius_and_height_axes() -> None:
    shader = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "viewport"
        / "shaders"
        / "raymarch.frag"
    ).read_text(encoding="utf-8")

    assert "0.5 * length(world_drag.xy)" in shader
    assert "0.5 * abs(world_drag.z)" in shader


def test_box_preview_shader_matches_committed_three_axis_size_rule() -> None:
    shader = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "viewport"
        / "shaders"
        / "raymarch.frag"
    ).read_text(encoding="utf-8")

    assert "vec3 actual_half_size = max(0.5 * abs(world_drag), vec3(0.05));" in shader
    assert "return actual_half_size;" in shader


def test_torus_preview_shader_accepts_typed_minor_radius_uniform() -> None:
    shader = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "viewport"
        / "shaders"
        / "raymarch.frag"
    ).read_text(encoding="utf-8")

    assert "uniform float u_preview_torus_minor_radius;" in shader
    assert "u_preview_torus_minor_radius > 0.0" in shader
    assert "- torus_minor_radius" in shader


def test_centered_typed_cylinder_keeps_anchor_at_center() -> None:
    start, end = create_effective_endpoints(
        anchor=(1.0, 2.0, 3.0),
        current=(2.0, 2.0, 3.0),
        reference_plane="xy",
        kind="cylinder",
        dimensions=(2.0, 4.0),
        centered=True,
    )
    document = SceneDocument()
    handle = document.add_primitive_from_drag("cylinder", start, end)
    cylinder = document.node(handle)

    assert isinstance(cylinder, Cylinder)
    assert cylinder.center == pytest.approx((1.0, 2.0, 3.0))
    assert 2.0 * cylinder.radius == pytest.approx(2.0)
    assert 2.0 * cylinder.half_height == pytest.approx(4.0)


def test_typed_torus_single_dimension_creates_exact_major_diameter() -> None:
    start, end = create_effective_endpoints(
        anchor=(0.0, 0.0, 0.0),
        current=(1.0, -1.0, 0.0),
        reference_plane="xy",
        kind="torus",
        dimensions=(2.0,),
        centered=False,
    )
    document = SceneDocument()
    handle = document.add_primitive_from_drag("torus", start, end)
    torus = document.node(handle)

    assert isinstance(torus, Torus)
    assert 2.0 * torus.major_radius == pytest.approx(2.0)


def test_typed_torus_two_dimensions_create_major_and_minor_diameters() -> None:
    dimensions = (4.0, 0.8)
    start, end = create_effective_endpoints(
        anchor=(0.0, 0.0, 0.0),
        current=(1.0, -1.0, 0.0),
        reference_plane="xy",
        kind="torus",
        dimensions=dimensions,
        centered=False,
    )
    document = SceneDocument()
    handle = document.add_primitive_from_drag(
        "torus",
        start,
        end,
        parameters=create_typed_parameters("torus", dimensions),
    )
    torus = document.node(handle)

    assert isinstance(torus, Torus)
    assert 2.0 * torus.major_radius == pytest.approx(4.0)
    assert 2.0 * torus.minor_radius == pytest.approx(0.8)


def test_typed_create_parameters_are_only_needed_for_parameterized_shapes() -> None:
    assert create_typed_parameters("torus", (4.0, 0.8)) == {
        "minor_diameter": 0.8,
    }
    assert create_typed_parameters("box", (1.0, 2.0, 3.0)) == {}
    assert create_typed_parameters("torus", None) == {}


def test_torus_preview_uses_typed_minor_diameter_when_available() -> None:
    assert create_preview_torus_minor_radius("torus", (4.0, 0.8)) == pytest.approx(
        0.4
    )
    assert create_preview_torus_minor_radius("torus", (4.0,)) == -1.0
    assert create_preview_torus_minor_radius("sphere", (4.0, 0.8)) == -1.0


def test_create_measurement_components_use_diameter_for_radial_shapes() -> None:
    assert create_measurement_components("circle", "X", 3.0, "Y", 4.0) == (
        ("X", 3.0),
        ("Y", 4.0),
        ("D", 5.0),
    )
    assert create_measurement_components("torus", "X", 0.0, "Z", 2.0) == (
        ("X", 0.0),
        ("Z", 2.0),
        ("D", 2.0),
    )


def test_typed_torus_measurement_components_show_major_and_minor_diameters() -> None:
    assert create_measurement_components(
        "torus",
        "X",
        4.0,
        "Y",
        0.0,
        (4.0, 0.0, 0.0),
        (4.0, 0.8),
    ) == (
        ("X", 4.0),
        ("Y", 0.0),
        ("D", 4.0),
        ("d", 0.8),
    )


def test_create_measurement_components_show_cylinder_diameter_and_height() -> None:
    assert create_measurement_components(
        "cylinder",
        "X",
        1.4,
        "Y",
        1.4,
        (1.4, 1.4, 5.0),
    ) == (
        ("X", 1.4),
        ("Y", 1.4),
        ("D", pytest.approx(1.979898987322333)),
        ("H", 5.0),
    )
    assert create_measurement_components(
        "cylinder",
        "X",
        2.0,
        "Z",
        5.0,
        (2.0, 0.0, 5.0),
    ) == (
        ("X", 2.0),
        ("Z", 5.0),
        ("D", 2.0),
        ("H", 5.0),
    )


def test_create_measurement_components_show_box_xyz_when_available() -> None:
    assert create_measurement_components(
        "box",
        "X",
        2.0,
        "Y",
        4.0,
        (2.0, 4.0, 6.0),
    ) == (("X", 2.0), ("Y", 4.0), ("Z", 6.0))


def test_create_measurement_components_show_square_committed_size() -> None:
    assert create_measurement_components("square", "X", 3.0, "Y", 4.0) == (
        ("X", 3.0),
        ("Y", 4.0),
        ("Size", 4.0),
    )


def test_create_measurement_components_keep_active_dimensions_for_box_style_shapes() -> None:
    assert create_measurement_components("rectangle", "X", 3.0, "Y", 4.0) == (
        ("X", 3.0),
        ("Y", 4.0),
    )
    assert create_measurement_components(
        "box",
        "X",
        3.0,
        "Y",
        4.0,
        (3.0, 4.0, 0.0),
    ) == (
        ("X", 3.0),
        ("Y", 4.0),
    )


def test_typed_move_single_value_follows_current_preview_direction() -> None:
    delta = apply_typed_move_delta(
        current_delta=(0.0, -0.4, 0.0),
        reference_plane="xy",
        dimensions=(2.5,),
    )

    assert delta == (0.0, -2.5, 0.0)


def test_typed_move_two_values_use_active_reference_plane() -> None:
    delta = apply_typed_move_delta(
        current_delta=(0.0, 0.0, 0.0),
        reference_plane="xz",
        dimensions=(1.25, -0.5),
    )

    assert delta == (1.25, 0.0, -0.5)


def test_typed_move_three_values_are_explicit_xyz() -> None:
    delta = apply_typed_move_delta(
        current_delta=(0.0, 0.0, 0.0),
        reference_plane="yz",
        dimensions=(1.0, 2.0, 3.0),
    )

    assert delta == (1.0, 2.0, 3.0)


def test_keyboard_move_delta_uses_wasd_qe_axis_mapping() -> None:
    assert keyboard_move_delta(Qt.Key.Key_W, 0.25, Qt.KeyboardModifier.NoModifier) == (
        0.0,
        0.25,
        0.0,
    )
    assert keyboard_move_delta(Qt.Key.Key_S, 0.25, Qt.KeyboardModifier.NoModifier) == (
        0.0,
        -0.25,
        0.0,
    )
    assert keyboard_move_delta(Qt.Key.Key_A, 0.25, Qt.KeyboardModifier.NoModifier) == (
        -0.25,
        0.0,
        0.0,
    )
    assert keyboard_move_delta(Qt.Key.Key_D, 0.25, Qt.KeyboardModifier.NoModifier) == (
        0.25,
        0.0,
        0.0,
    )
    assert keyboard_move_delta(Qt.Key.Key_Q, 0.25, Qt.KeyboardModifier.NoModifier) == (
        0.0,
        0.0,
        -0.25,
    )
    assert keyboard_move_delta(Qt.Key.Key_E, 0.25, Qt.KeyboardModifier.NoModifier) == (
        0.0,
        0.0,
        0.25,
    )


def test_keyboard_move_delta_supports_coarse_and_fine_modifiers() -> None:
    assert keyboard_move_delta(
        Qt.Key.Key_D,
        0.25,
        Qt.KeyboardModifier.ShiftModifier,
    ) == (2.5, 0.0, 0.0)
    assert keyboard_move_delta(
        Qt.Key.Key_D,
        0.25,
        Qt.KeyboardModifier.AltModifier,
    ) == (0.025, 0.0, 0.0)
    assert keyboard_move_delta(
        Qt.Key.Key_D,
        0.25,
        Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.AltModifier,
    ) == (0.25, 0.0, 0.0)


def test_typed_create_start_two_values_use_active_reference_plane() -> None:
    assert apply_typed_start_point("xy", (1.0, -2.0)) == (1.0, -2.0, 0.0)
    assert apply_typed_start_point("xz", (1.0, -2.0)) == (1.0, 0.0, -2.0)
    assert apply_typed_start_point("yz", (1.0, -2.0)) == (0.0, 1.0, -2.0)


def test_typed_create_start_three_values_are_explicit_xyz() -> None:
    assert apply_typed_start_point("xy", (1.0, -2.0, 3.0)) == (1.0, -2.0, 3.0)


def test_typed_create_start_rejects_incomplete_coordinate() -> None:
    with pytest.raises(ValueError, match="start point"):
        apply_typed_start_point("xy", (1.0,))


def test_reference_plane_coordinate_components_follow_active_plane() -> None:
    assert reference_plane_coordinate_components((1.0, 2.0, 3.0), "xy") == (
        ("X", 1.0),
        ("Y", 2.0),
    )
    assert reference_plane_coordinate_components((1.0, 2.0, 3.0), "xz") == (
        ("X", 1.0),
        ("Z", 3.0),
    )
    assert reference_plane_coordinate_components((1.0, 2.0, 3.0), "yz") == (
        ("Y", 2.0),
        ("Z", 3.0),
    )


def test_cursor_preview_is_only_active_before_first_create_point() -> None:
    hover = (1.0, 2.0, 0.0)

    assert cursor_preview_active("create", None, hover)
    assert not cursor_preview_active("create", hover, hover)
    assert not cursor_preview_active("move", None, hover)
    assert not cursor_preview_active("create", None, None)


def test_create_input_label_distinguishes_start_from_size() -> None:
    assert create_input_label(False) == "Start"
    assert create_input_label(True) == "Size"


def test_create_start_prompt_names_plane_and_xyz_entry() -> None:
    assert create_start_prompt("X", "Z") == "Type Start X,Z or X,Y,Z"


def test_create_size_prompt_names_shape_specific_dimensions() -> None:
    assert create_size_prompt("circle", "X", "Y") == "Type D"
    assert create_size_prompt("rectangle", "U", "V") == "Type U x V"
    assert create_size_prompt("square", "U", "V") == "Type Size"
    assert create_size_prompt("box", "X", "Y") == "Type X x Y x Z"
    assert create_size_prompt("cylinder", "X", "Z") == "Type D x H"
    assert create_size_prompt("torus", "X", "Y") == "Type D x d"
    assert create_size_prompt("segment", "X", "Y") == "Type Length"
    assert create_size_prompt("bezier_curve", "X", "Y") == "Type Length"
    assert create_size_prompt("bezier_polycurve", "X", "Y") == "Type Length"


def test_move_dimension_prompt_names_delta_entry() -> None:
    assert move_dimension_prompt() == "Type dX,dY or dX,dY,dZ"
    assert move_dimension_prompt("X", "Z") == "Type dX,dZ or dX,dY,dZ"
    assert move_dimension_prompt("Y", "Z") == "Type dY,dZ or dX,dY,dZ"


def test_reference_view_keyboard_mapping() -> None:
    assert reference_view_for_key(Qt.Key.Key_1) == "3d"
    assert reference_view_for_key(Qt.Key.Key_2) == "xy"
    assert reference_view_for_key(Qt.Key.Key_3) == "xz"
    assert reference_view_for_key(Qt.Key.Key_4) == "yz"
    assert reference_view_for_key(Qt.Key.Key_5) is None


def test_reference_plane_cycle_advances_between_drawing_planes() -> None:
    assert next_reference_plane("xy") == "xz"
    assert next_reference_plane("xz") == "yz"
    assert next_reference_plane("yz") == "xy"
    assert next_reference_plane("xy", reverse=True) == "yz"
    assert next_reference_plane("xz", reverse=True) == "xy"
    assert next_reference_plane("yz", reverse=True) == "xz"
    with pytest.raises(ValueError, match="unknown reference plane"):
        next_reference_plane("3d")


def test_reference_plane_cycle_keyboard_mapping_is_tool_scoped() -> None:
    assert should_cycle_reference_plane_for_key(
        Qt.Key.Key_Tab,
        Qt.KeyboardModifier.NoModifier,
    )
    assert should_cycle_reference_plane_for_key(
        Qt.Key.Key_Tab,
        Qt.KeyboardModifier.ShiftModifier,
    )
    assert not should_cycle_reference_plane_for_key(
        Qt.Key.Key_2,
        Qt.KeyboardModifier.NoModifier,
    )


def test_idle_selection_clear_keyboard_mapping_uses_plain_escape() -> None:
    assert should_clear_idle_selection_for_key(
        Qt.Key.Key_Escape,
        Qt.KeyboardModifier.NoModifier,
    )
    assert not should_clear_idle_selection_for_key(
        Qt.Key.Key_Escape,
        Qt.KeyboardModifier.ShiftModifier,
    )
    assert not should_clear_idle_selection_for_key(
        Qt.Key.Key_Delete,
        Qt.KeyboardModifier.NoModifier,
    )


def test_idle_selection_clear_ignores_active_tools() -> None:
    class FakeWidget:
        def __init__(self, interaction_tool: tuple[str, object] | None) -> None:
            self._interaction_tool = interaction_tool
            self.hover_cleared = False

        def _clear_scene_hover(self) -> None:
            self.hover_cleared = True

    active = FakeWidget(("move", 7))
    idle = FakeWidget(None)

    assert not GLWidget._clear_idle_selection_for_key(
        active,
        Qt.Key.Key_Escape,
        Qt.KeyboardModifier.NoModifier,
    )
    assert not active.hover_cleared
    assert GLWidget._clear_idle_selection_for_key(
        idle,
        Qt.Key.Key_Escape,
        Qt.KeyboardModifier.NoModifier,
    )
    assert idle.hover_cleared


def test_tool_reference_plane_cycle_keeps_create_anchor() -> None:
    class FakeWidget:
        def __init__(self) -> None:
            self._interaction_tool = ("create", "rectangle")
            self._reference_plane = "xy"
            self._tool_start_world = (1.0, 2.0, 0.0)
            self._tool_current_world = (3.0, 4.0, 0.0)
            self._tool_hover_world = (3.0, 4.0, 0.0)
            self.measurement_updated = False
            self.reference_view = None

        def set_reference_view(self, view: str) -> None:
            self.reference_view = view
            self._reference_plane = view

        def _update_measurement_readout(self) -> None:
            self.measurement_updated = True

    widget = FakeWidget()

    assert GLWidget._cycle_reference_plane_for_key(
        widget,
        Qt.Key.Key_Tab,
        Qt.KeyboardModifier.NoModifier,
    )
    assert widget.reference_view == "xz"
    assert widget._tool_current_world == widget._tool_start_world
    assert widget._tool_hover_world == widget._tool_start_world
    assert widget.measurement_updated


def test_tool_reference_plane_cycle_keeps_move_drag_anchor() -> None:
    class FakeWidget:
        def __init__(self) -> None:
            self._interaction_tool = ("move", 7)
            self._reference_plane = "xy"
            self._tool_start_world = (1.0, 2.0, 0.0)
            self._tool_current_world = (2.0, 4.0, 0.0)
            self.measurement_updated = False
            self.reference_view = None

        def set_reference_view(self, view: str) -> None:
            self.reference_view = view
            self._reference_plane = view

        def _update_measurement_readout(self) -> None:
            self.measurement_updated = True

    widget = FakeWidget()

    assert GLWidget._cycle_reference_plane_for_key(
        widget,
        Qt.Key.Key_Tab,
        Qt.KeyboardModifier.ShiftModifier,
    )
    assert widget.reference_view == "yz"
    assert widget._tool_current_world == widget._tool_start_world
    assert widget.measurement_updated


def test_reference_plane_labels_are_user_facing_uppercase() -> None:
    assert reference_plane_label("xy") == "XY"
    assert reference_plane_label("xz") == "XZ"
    assert reference_plane_label("yz") == "YZ"
    with pytest.raises(ValueError, match="unknown reference plane"):
        reference_plane_label("3d")


def test_reference_plane_context_names_active_plane() -> None:
    assert reference_plane_context("xy") == "Plane XY"
    assert reference_plane_context("xz") == "Plane XZ"
    assert reference_plane_context("yz") == "Plane YZ"


def test_snap_reference_point_respects_toggle_and_alt_bypass() -> None:
    assert should_snap_reference_point(True, Qt.KeyboardModifier.NoModifier)
    assert not should_snap_reference_point(False, Qt.KeyboardModifier.NoModifier)
    assert not should_snap_reference_point(True, Qt.KeyboardModifier.AltModifier)


def test_snap_status_text_names_precision_mode() -> None:
    assert snap_status_text(True) == "Snap On"
    assert snap_status_text(False) == "Snap Off"


def test_grid_measurement_text_includes_snap_status() -> None:
    class FakeWidget:
        def __init__(self) -> None:
            self._grid_spacing = 0.25
            self._snap_enabled = False

        def _format_measure(self, value: float) -> str:
            return f"{value:.5g} m"

    assert GLWidget._grid_measurement_text(FakeWidget()) == (
        "Grid 0.25 m  Snap Off"
    )


def test_active_move_measurement_text_keeps_grid_and_snap_context() -> None:
    class FakeWidget:
        def __init__(self) -> None:
            self._committed_move_object_id = 0
            self._interaction_tool = ("move", 7)
            self._dimension_input = ""
            self._reference_plane = "xy"
            self._grid_spacing = 0.5
            self._snap_enabled = True
            self._tool_current_world = None
            self._tool_hover_world = (2.0, 3.0, 0.0)

        def _preview_move_delta(self) -> tuple[float, float, float]:
            return (1.0, 0.0, 0.0)

        def _reference_axes(self) -> tuple[tuple[int, str], tuple[int, str]]:
            return ((0, "X"), (1, "Y"))

        def _format_measure(self, value: float) -> str:
            return f"{value:.5g} m"

        def _grid_measurement_text(self) -> str:
            return GLWidget._grid_measurement_text(self)

        def _xyz_measurement_text(
            self,
            point: tuple[float, float, float] | None,
        ) -> str:
            return GLWidget._xyz_measurement_text(self, point)

        def _move_measurement_text(
            self,
            delta: tuple[float, float, float],
            *,
            cursor_point: tuple[float, float, float] | None = None,
            input_text: str = "",
            modifier_text: str = "",
        ) -> str:
            return GLWidget._move_measurement_text(
                self,
                delta,
                cursor_point=cursor_point,
                input_text=input_text,
                modifier_text=modifier_text,
            )

    text = GLWidget._measurement_text(FakeWidget())

    assert "Tool          Move" in text
    assert "Reference     Plane XY" in text
    assert "Cursor point  X 2 m  Y 3 m  Z 0 m" in text
    assert "Move delta    X 1 m  Y 0 m  Z 0 m" in text
    assert "Grid spacing  0.5 m" in text
    assert "Snap          On" in text
    assert "Type dX,dY or dX,dY,dZ" in text


def test_commit_move_preview_applies_rotation_preview_on_enter_path() -> None:
    class FakeWidget:
        def __init__(self) -> None:
            self._interaction_tool = ("move", 42)
            self._rotation_drag_axis = "z"
            self._rotation_drag_center = (1.0, 2.0, 3.0)
            self._rotation_drag_move_delta = (0.0, 0.0, 0.0)
            self._rotation_preview_angle = 15.0
            self.cancelled = False

        def _preview_move_delta(self) -> tuple[float, float, float]:
            return (0.0, 0.0, 0.0)

        def _preview_rotation_pivot(self) -> tuple[float, float, float]:
            return GLWidget._preview_rotation_pivot(self)

        def cancel_interaction_tool(self) -> None:
            self.cancelled = True

    widget = FakeWidget()
    emitted: list[tuple[int, str, float, tuple[float, float, float]]] = []

    def record(
        handle: int,
        axis: str,
        angle: float,
        pivot: tuple[float, float, float],
    ) -> None:
        emitted.append((handle, axis, angle, pivot))

    signals.viewport_rotate_requested.connect(record)
    try:
        GLWidget._commit_move_preview(widget)
    finally:
        signals.viewport_rotate_requested.disconnect(record)

    assert widget.cancelled
    assert emitted == [(42, "z", 15.0, (1.0, 2.0, 3.0))]


def test_commit_move_preview_emits_combined_transform_on_enter_path() -> None:
    class FakeWidget:
        def __init__(self) -> None:
            self._interaction_tool = ("move", 42)
            self._rotation_drag_axis = "z"
            self._rotation_drag_center = (1.0, 2.0, 3.0)
            self._rotation_drag_move_delta = (0.25, 0.0, 0.0)
            self._rotation_preview_angle = 15.0
            self.cancelled = False

        def _preview_move_delta(self) -> tuple[float, float, float]:
            return (0.25, 0.5, 0.0)

        def _preview_rotation_pivot(self) -> tuple[float, float, float]:
            return GLWidget._preview_rotation_pivot(self)

        def cancel_interaction_tool(self) -> None:
            self.cancelled = True

    widget = FakeWidget()
    emitted: list[
        tuple[
            int,
            tuple[float, float, float],
            tuple[tuple[str, float, tuple[float, float, float]], ...],
        ]
    ] = []

    def record(
        handle: int,
        delta: tuple[float, float, float],
        rotations: tuple[tuple[str, float, tuple[float, float, float]], ...],
    ) -> None:
        emitted.append((handle, delta, rotations))

    signals.viewport_transform_requested.connect(record)
    try:
        GLWidget._commit_move_preview(widget)
    finally:
        signals.viewport_transform_requested.disconnect(record)

    assert widget.cancelled
    assert emitted == [
        (
            42,
            (0.25, 0.5, 0.0),
            (("z", 15.0, (1.0, 2.5, 3.0)),),
        )
    ]


def test_commit_move_preview_keeps_multiple_rotation_axes() -> None:
    class FakeWidget:
        def __init__(self) -> None:
            self._interaction_tool = ("move", 42)
            self._rotation_preview_steps = [
                (
                    "x",
                    10.0,
                    (1.0, 2.0, 3.0),
                    (0.0, 0.0, 0.0),
                ),
                (
                    "y",
                    20.0,
                    (1.0, 2.0, 3.0),
                    (0.5, 0.0, 0.0),
                ),
            ]
            self._rotation_drag_axis = None
            self._rotation_drag_center = None
            self._rotation_drag_move_delta = (0.0, 0.0, 0.0)
            self._rotation_preview_angle = 0.0
            self.cancelled = False

        def _preview_move_delta(self) -> tuple[float, float, float]:
            return (1.0, 0.0, 0.0)

        def cancel_interaction_tool(self) -> None:
            self.cancelled = True

    widget = FakeWidget()
    emitted: list[
        tuple[
            int,
            tuple[float, float, float],
            tuple[tuple[str, float, tuple[float, float, float]], ...],
        ]
    ] = []

    def record(
        handle: int,
        delta: tuple[float, float, float],
        rotations: tuple[tuple[str, float, tuple[float, float, float]], ...],
    ) -> None:
        emitted.append((handle, delta, rotations))

    signals.viewport_transform_requested.connect(record)
    try:
        GLWidget._commit_move_preview(widget)
    finally:
        signals.viewport_transform_requested.disconnect(record)

    assert widget.cancelled
    assert emitted == [
        (
            42,
            (1.0, 0.0, 0.0),
            (
                ("x", 10.0, (2.0, 2.0, 3.0)),
                ("y", 20.0, (1.5, 2.0, 3.0)),
            ),
        )
    ]


def test_rotation_gizmo_center_follows_move_preview_delta() -> None:
    box = Box(name="box", object_id=7, center=(1.0, 2.0, 3.0))

    class FakeSceneTree:
        nodes = (box,)

    class FakeWidget:
        _interaction_tool = ("move", 1)
        _scene_tree = FakeSceneTree()
        _scene_selected_object_id = box.object_id
        _box_center_and_radius = staticmethod(GLWidget._box_center_and_radius)

        def _preview_move_delta(self) -> tuple[float, float, float]:
            return (0.25, -0.5, 1.0)

    visible, center, _radius = GLWidget._rotation_gizmo_state(FakeWidget())

    assert visible
    assert center == (1.25, 1.5, 4.0)


def test_empty_scene_source_defines_renderer_contract() -> None:
    source = empty_scene_source()

    for required_name in (
        "sceneSDF",
        "sceneBoundaryOwnerId",
        "sceneObjectId",
        "sceneSelectionOwnsBoundary",
        "sceneSelectedObjectSDF",
        "sceneSelectedObjectDimension",
        "componentSDF",
        "componentObjectId",
    ):
        assert required_name in source
