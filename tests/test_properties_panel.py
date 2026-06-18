from __future__ import annotations

import os

import pytest
from PySide6.QtWidgets import QApplication, QLabel

from app.panels.properties import (
    PropertiesPanel,
    bounding_box_center,
    bounding_box_size,
    format_point_list_text,
    full_length_from_half_length,
    full_size_from_half_size,
    half_length_from_full_length,
    half_size_from_full_size,
    parse_point_list_text,
    placed_profile_local_points,
    placed_profile_world_points,
    segment_profile_center_label,
    segment_profile_length_label,
    property_dimension_value,
    read_only_vector_text,
    rounded_rectangle_corner_radius_maximum,
    rounded_rectangle_full_size_minimum,
    standard_axis_label,
    standard_workplane_label,
    sweep_end_from_length,
    vector_distance,
    vector_component_labels,
    values_equal,
)
from core.scene import SceneDocument
from core.sdf.base import BoundingBox3D


def test_full_dimension_helpers_convert_from_canonical_sdf_half_values() -> None:
    assert full_length_from_half_length(0.75) == 1.5
    assert half_length_from_full_length(1.5) == 0.75
    assert full_size_from_half_size((0.5, 1.0, 1.5)) == (1.0, 2.0, 3.0)
    assert half_size_from_full_size((1.0, 2.0, 3.0)) == (0.5, 1.0, 1.5)


def test_full_dimension_helpers_reject_non_positive_dimensions() -> None:
    with pytest.raises(ValueError, match="dimension"):
        half_length_from_full_length(0.0)
    with pytest.raises(ValueError, match="dimensions"):
        half_size_from_full_size((1.0, -2.0))


def test_property_value_equality_handles_float_vectors() -> None:
    assert values_equal((1.0, 2.0, 3.0), (1.0, 2.0, 3.0 + 1e-13))
    assert not values_equal((1.0, 2.0, 3.0), (1.0, 2.0, 3.1))


def test_property_dimension_value_accepts_units_formulas_and_spinbox_suffix() -> None:
    assert property_dimension_value("50mm*2") == pytest.approx(0.1)
    assert property_dimension_value("(2ft + 6in) m") == pytest.approx(0.762)
    assert property_dimension_value("1m/4") == pytest.approx(0.25)


def test_property_dimension_value_rejects_vector_entries() -> None:
    with pytest.raises(ValueError, match="one value"):
        property_dimension_value("1m x 2m")


def test_rounded_rectangle_corner_radius_maximum_uses_smallest_half_size() -> None:
    assert rounded_rectangle_corner_radius_maximum((0.6, 0.25)) == 0.25


def test_rounded_rectangle_full_size_minimum_uses_corner_radius_diameter() -> None:
    assert rounded_rectangle_full_size_minimum(0.25) == 0.5


def test_sweep_path_length_helpers_preserve_current_normal_direction() -> None:
    origin = (1.0, 2.0, 3.0)
    normal = (0.0, 0.0, 1.0)

    assert vector_distance(origin, (1.0, 2.0, 5.5)) == 2.5
    assert sweep_end_from_length(origin, normal, (1.0, 2.0, 5.5), 4.0) == (
        1.0,
        2.0,
        7.0,
    )
    assert sweep_end_from_length(origin, normal, (1.0, 2.0, 0.5), 4.0) == (
        1.0,
        2.0,
        -1.0,
    )


def test_sweep_path_length_helper_normalizes_direction() -> None:
    assert sweep_end_from_length(
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 2.0),
        (0.0, 0.0, 1.0),
        3.0,
    ) == (0.0, 0.0, 3.0)


def test_sweep_path_length_helper_rejects_non_positive_lengths() -> None:
    with pytest.raises(ValueError, match="sweep path length"):
        sweep_end_from_length((0.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0), 0.0)


def test_sweep_path_length_helper_rejects_invalid_normals() -> None:
    with pytest.raises(ValueError, match="sweep path normal"):
        sweep_end_from_length(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            1.0,
        )


def test_standard_workplane_label_recognizes_reference_planes() -> None:
    assert standard_workplane_label((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)) == "XY"
    assert standard_workplane_label((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)) == "XZ"
    assert standard_workplane_label((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)) == "YZ"


def test_standard_workplane_label_ignores_axis_direction_for_plane_name() -> None:
    assert standard_workplane_label((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)) == "XZ"
    assert standard_workplane_label((0.0, 0.0, 1.0), (0.0, -1.0, 0.0)) == "YZ"


def test_standard_workplane_label_reports_custom_for_nonstandard_axes() -> None:
    assert standard_workplane_label((0.707, 0.707, 0.0), (0.0, 0.0, 1.0)) == (
        "Custom"
    )
    assert standard_workplane_label((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)) == "Custom"


def test_standard_axis_label_recognizes_reference_axes() -> None:
    assert standard_axis_label((1.0, 0.0, 0.0)) == "X"
    assert standard_axis_label((0.0, -1.0, 0.0)) == "Y"
    assert standard_axis_label((0.0, 0.0, 1.0)) == "Z"


def test_standard_axis_label_reports_none_for_custom_axes() -> None:
    assert standard_axis_label((0.707, 0.707, 0.0)) is None
    assert standard_axis_label((0.5, 0.0, 0.0)) is None


def test_vector_component_labels_default_to_xyz() -> None:
    assert vector_component_labels(2) == ("X", "Y")
    assert vector_component_labels(3) == ("X", "Y", "Z")


def test_vector_component_labels_support_profile_uv_labels() -> None:
    assert vector_component_labels(2, ("U", "V")) == ("U", "V")


def test_bounding_box_readout_helpers_report_center_size_and_units() -> None:
    box = BoundingBox3D(-1.0, 3.0, -2.0, 2.0, 0.5, 1.5)

    assert bounding_box_center(box) == (1.0, 0.0, 1.0)
    assert bounding_box_size(box) == (4.0, 4.0, 1.0)
    assert read_only_vector_text(bounding_box_size(box)) == (
        "X 4 m  Y 4 m  Z 1 m"
    )


def test_vector_component_labels_reject_too_few_labels() -> None:
    with pytest.raises(ValueError, match="component labels"):
        vector_component_labels(3, ("U", "V"))


def test_segment_profile_labels_make_local_axis_explicit() -> None:
    assert segment_profile_center_label() == "Profile center U"
    assert segment_profile_length_label() == "Length along U"


def test_point_list_text_uses_ordered_world_xyz_entries() -> None:
    points = ((0.0, 1.0, 1.0), (0.5, 1.5, 1.5), (-1.0, 0.0, 2.0))

    assert format_point_list_text(points) == "{0;1;1} {0.5;1.5;1.5} {-1;0;2}"
    assert parse_point_list_text(
        "{0;1;1} {0.5;1.5;1.5} {-1;0;2}",
        2,
    ) == points


def test_point_list_text_accepts_units_and_rejects_too_few_points() -> None:
    assert parse_point_list_text("{1000mm;0;0} {0;50cm;0}", 2) == (
        (1.0, 0.0, 0.0),
        (0.0, 0.5, 0.0),
    )
    with pytest.raises(ValueError, match="at least 3 points"):
        parse_point_list_text("{0;0;0} {1;0;0}", 3)
    with pytest.raises(ValueError, match="exactly three"):
        parse_point_list_text("{0;0} {1;0}", 2)
    with pytest.raises(ValueError, match=r"\{x;y;z\}"):
        parse_point_list_text("{0;0;0} invalid {1;0;0}", 2)


def test_placed_profile_point_helpers_convert_between_local_uv_and_world_xyz() -> None:
    origin = (10.0, 20.0, 30.0)
    axis_u = (0.0, 1.0, 0.0)
    axis_v = (0.0, 0.0, 1.0)
    local_points = ((0.0, 0.0), (1.5, 2.0), (-0.5, 0.25))
    world_points = placed_profile_world_points(
        origin,
        axis_u,
        axis_v,
        local_points,
    )

    assert world_points == (
        (10.0, 20.0, 30.0),
        (10.0, 21.5, 32.0),
        (10.0, 19.5, 30.25),
    )
    assert placed_profile_local_points(
        origin,
        axis_u,
        axis_v,
        world_points,
    ) == local_points


def _ensure_qapplication() -> QApplication:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    return application


def _property_panel_labels(panel: PropertiesPanel) -> tuple[str, ...]:
    labels = []
    for row in range(panel.layout().rowCount()):
        item = panel.layout().itemAt(row, panel.layout().ItemRole.LabelRole)
        if item is None:
            continue
        widget = item.widget()
        if isinstance(widget, QLabel):
            labels.append(widget.text())
    return tuple(labels)


def test_point_defined_profiles_hide_internal_placement_frame_fields() -> None:
    _ensure_qapplication()
    document = SceneDocument()
    polyline_handle = document.add_primitive("polyline")
    bezier_handle = document.add_primitive("bezier_curve")
    bezier_polycurve_handle = document.add_primitive("bezier_polycurve")
    polygon_handle = document.add_primitive("polygon")
    rectangle_handle = document.add_primitive("rectangle")
    panel = PropertiesPanel()

    try:
        panel._document = document
        for handle in (
            polyline_handle,
            bezier_handle,
            bezier_polycurve_handle,
            polygon_handle,
        ):
            panel._node = document.node(handle)
            panel._build_form()
            labels = _property_panel_labels(panel)

            assert "Points" in labels
            assert "Workplane" not in labels
            assert "Origin" not in labels
            assert "Axis U" not in labels
            assert "Axis V" not in labels

        panel._node = document.node(rectangle_handle)
        panel._build_form()
        rectangle_labels = _property_panel_labels(panel)

        assert "Workplane" in rectangle_labels
        assert "Origin" in rectangle_labels
        assert "Axis U" in rectangle_labels
        assert "Axis V" in rectangle_labels
    finally:
        panel.deleteLater()
