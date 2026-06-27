from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
import re

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QValidator
from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from app.axis_labels import world_axis_label
from app.dimensions import parse_measurement_entry, parse_scalar_entry
from app.panels.display import display_kind
from app.signals import signals
from core.boundary import BoundaryRegion
from core.scene import SceneDocument
from core.sdf import (
    QuadraticBezierCurveProfile,
    QuadraticBezierSurfaceProfile,
    QuadraticBezierTube,
    Box,
    BoxFrame,
    CappedCone,
    CircleProfile,
    Cone,
    Cylinder,
    EllipseProfile,
    Extrude,
    PolygonProfile,
    PolylineTube,
    PolylineProfile,
    PlacedSDF1D,
    PlacedSDF2D,
    Pyramid,
    RectangleProfile,
    RegularPolygonProfile,
    Revolve,
    Rotate,
    RoundedRectangleProfile,
    Scale,
    SegmentProfile,
    Sphere,
    SquareProfile,
    Torus,
    Translate,
    PlacedPolyline1D,
)
from core.sdf.base import BoundingBox3D, SDFNode


def full_size_from_half_size(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(2.0 * value for value in values)


def half_size_from_full_size(values: tuple[float, ...]) -> tuple[float, ...]:
    if any(value <= 0.0 for value in values):
        raise ValueError("dimensions must be positive")
    return tuple(0.5 * value for value in values)


def full_length_from_half_length(value: float) -> float:
    return 2.0 * value


def half_length_from_full_length(value: float) -> float:
    if value <= 0.0:
        raise ValueError("dimension must be positive")
    return 0.5 * value


def rounded_rectangle_corner_radius_maximum(
    half_size: tuple[float, float],
) -> float:
    return min(half_size)


def rounded_rectangle_full_size_minimum(corner_radius: float) -> float:
    return 2.0 * corner_radius


def standard_workplane_label(
    axis_u: tuple[float, float, float],
    axis_v: tuple[float, float, float],
    tolerance: float = 1e-6,
) -> str:
    axis_names = ("X", "Y", "Z")

    first = standard_axis_label(axis_u, tolerance)
    second = standard_axis_label(axis_v, tolerance)
    if first is None or second is None or first == second:
        return "Custom"
    return "".join(axis for axis in axis_names if axis in {first, second})


def standard_axis_label(
    axis: tuple[float, float, float],
    tolerance: float = 1e-6,
) -> str | None:
    axis_names = ("X", "Y", "Z")
    absolute = tuple(abs(component) for component in axis)
    dominant = max(range(3), key=lambda index: absolute[index])
    if abs(absolute[dominant] - 1.0) > tolerance:
        return None
    if any(
        absolute[index] > tolerance
        for index in range(3)
        if index != dominant
    ):
        return None
    return axis_names[dominant]


def vector_component_labels(
    component_count: int,
    axis_labels: tuple[str, ...] | None = None,
) -> tuple[str, ...]:
    labels = axis_labels or ("X", "Y", "Z")
    if len(labels) < component_count:
        raise ValueError("not enough vector component labels")
    return labels[:component_count]


def bounding_box_center(box: BoundingBox3D) -> tuple[float, float, float]:
    return (
        0.5 * (box.x_min + box.x_max),
        0.5 * (box.y_min + box.y_max),
        0.5 * (box.z_min + box.z_max),
    )


def bounding_box_size(box: BoundingBox3D) -> tuple[float, float, float]:
    return (
        box.x_max - box.x_min,
        box.y_max - box.y_min,
        box.z_max - box.z_min,
    )


def read_only_vector_text(
    values: tuple[float, ...],
    axis_labels: tuple[str, ...] | None = None,
) -> str:
    return "  ".join(
        f"{label} {value:.5g} m"
        for label, value in zip(
            vector_component_labels(len(values), axis_labels),
            values,
            strict=True,
        )
    )


def segment_profile_center_label() -> str:
    return "Profile center U"


def segment_profile_length_label() -> str:
    return "Length along U"


POINT_ENTRY_PATTERN = re.compile(r"\{([^{}]+)\}")


def format_point_list_text(
    points: tuple[tuple[float, float, float], ...],
) -> str:
    return " ".join(
        "{" + ";".join(f"{component:.6g}" for component in point) + "}"
        for point in points
    )


def parse_point_list_text(
    text: str,
    minimum_points: int,
) -> tuple[tuple[float, float, float], ...]:
    matches = tuple(POINT_ENTRY_PATTERN.finditer(text))
    if not matches:
        raise ValueError("points must use {x;y;z} entries")
    cursor = 0
    for match in matches:
        if text[cursor:match.start()].strip():
            raise ValueError("points must use {x;y;z} entries")
        cursor = match.end()
    if text[cursor:].strip():
        raise ValueError("points must use {x;y;z} entries")
    points = []
    for match in matches:
        values = parse_measurement_entry(match.group(1))
        if len(values) != 3:
            raise ValueError("each point must have exactly three coordinates")
        points.append(tuple(float(value) for value in values))
    if len(points) < minimum_points:
        raise ValueError(f"at least {minimum_points} points are required")
    return tuple(points)


def placed_profile_world_points(
    origin: tuple[float, float, float],
    axis_u: tuple[float, float, float],
    axis_v: tuple[float, float, float],
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float, float], ...]:
    origin_array = np.asarray(origin, dtype=np.float64)
    axis_u_array = np.asarray(axis_u, dtype=np.float64)
    axis_v_array = np.asarray(axis_v, dtype=np.float64)
    return tuple(
        tuple(
            float(value)
            for value in (
                origin_array
                + point[0] * axis_u_array
                + point[1] * axis_v_array
            )
        )
        for point in points
    )


def placed_profile_local_points(
    origin: tuple[float, float, float],
    axis_u: tuple[float, float, float],
    axis_v: tuple[float, float, float],
    points: tuple[tuple[float, float, float], ...],
) -> tuple[tuple[float, float], ...]:
    origin_array = np.asarray(origin, dtype=np.float64)
    axis_u_array = np.asarray(axis_u, dtype=np.float64)
    axis_v_array = np.asarray(axis_v, dtype=np.float64)
    return tuple(
        (
            float(
                np.dot(
                    np.asarray(point, dtype=np.float64) - origin_array,
                    axis_u_array,
                )
            ),
            float(
                np.dot(
                    np.asarray(point, dtype=np.float64) - origin_array,
                    axis_v_array,
                )
            ),
        )
        for point in points
    )


def values_equal(left: object, right: object) -> bool:
    if isinstance(left, tuple) and isinstance(right, tuple):
        return len(left) == len(right) and all(
            values_equal(left_value, right_value)
            for left_value, right_value in zip(left, right, strict=True)
        )
    if isinstance(left, int | float) and isinstance(right, int | float):
        return abs(float(left) - float(right)) <= 1e-12
    return left == right


def property_dimension_value(text: str) -> float:
    expression = text.strip()
    if expression.endswith("m") and not expression.endswith(
        ("mm", "cm"),
    ):
        expression = expression[:-1].strip()
    values = parse_measurement_entry(expression)
    if len(values) != 1:
        raise ValueError("property dimension fields accept one value")
    value = float(values[0])
    if not np.isfinite(value):
        raise ValueError("property dimension value must be finite")
    return value


class CadDimensionSpinBox(QDoubleSpinBox):
    def valueFromText(self, text: str) -> float:
        try:
            return property_dimension_value(text)
        except ValueError:
            return self.value()

    def validate(self, text: str, position: int) -> tuple[QValidator.State, str, int]:
        expression = text.strip()
        if expression.endswith("m") and not expression.endswith(("mm", "cm")):
            expression = expression[:-1].strip()
        if not expression or expression in {"-", "+", ".", "(", "-(", "+("}:
            return (QValidator.State.Intermediate, text, position)
        try:
            property_dimension_value(text)
        except ValueError:
            return (QValidator.State.Intermediate, text, position)
        return (QValidator.State.Acceptable, text, position)


class CadScalarSpinBox(QDoubleSpinBox):
    def __init__(
        self,
        parent: QWidget | None = None,
        parse_suffixes: tuple[str, ...] = (),
    ) -> None:
        super().__init__(parent)
        self._parse_suffixes = tuple(
            suffix.strip().lower()
            for suffix in parse_suffixes
            if suffix.strip()
        )

    def _expression_text(self, text: str) -> str:
        expression = text.strip()
        lowered = expression.lower()
        for suffix in sorted(self._parse_suffixes, key=len, reverse=True):
            if lowered.endswith(suffix):
                return expression[: -len(suffix)].strip()
        return expression

    def valueFromText(self, text: str) -> float:
        try:
            return parse_scalar_entry(self._expression_text(text))
        except ValueError:
            return self.value()

    def validate(self, text: str, position: int) -> tuple[QValidator.State, str, int]:
        expression = self._expression_text(text)
        if not expression or expression in {"-", "+", ".", "(", "-(", "+("}:
            return (QValidator.State.Intermediate, text, position)
        try:
            parse_scalar_entry(expression)
        except ValueError:
            return (QValidator.State.Intermediate, text, position)
        return (QValidator.State.Acceptable, text, position)


class PropertiesPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._layout = QFormLayout(self)
        self._document: SceneDocument | None = None
        self._node: SDFNode | BoundaryRegion | None = None
        self._show_empty("Select one scene node.")
        signals.document_changed.connect(self._on_document_changed)
        signals.node_selected.connect(self._on_node_selected)

    def _on_document_changed(self, document: SceneDocument) -> None:
        self._document = document

    def _on_node_selected(self, handle: int | None) -> None:
        if handle is None or self._document is None:
            self._node = None
            self._show_empty("Select one scene node.")
            return
        self._node = self._document.node(handle)
        self._build_form()

    def focus_name_editor(self) -> bool:
        editor = self.findChild(QLineEdit, "nodeNameEdit")
        if editor is None:
            return False
        editor.setFocus(Qt.FocusReason.ShortcutFocusReason)
        editor.selectAll()
        return True

    def _clear(self) -> None:
        while self._layout.rowCount():
            self._layout.removeRow(0)

    def _show_empty(self, text: str) -> None:
        self._clear()
        label = QLabel(text)
        label.setWordWrap(True)
        self._layout.addRow(label)

    def _build_form(self) -> None:
        node = self._node
        if node is None:
            return
        self._clear()
        self._layout.addRow("Kind", QLabel(display_kind(node)))
        self._layout.addRow("Dimension", QLabel(f"{node.dimension}D"))
        if node.object_id:
            self._layout.addRow("Object ID", QLabel(str(node.object_id)))
        name = QLineEdit(node.name)
        name.setObjectName("nodeNameEdit")
        name.editingFinished.connect(
            lambda: self._set_value("name", name.text().strip() or node.name)
        )
        self._layout.addRow("Name", name)
        if isinstance(node, BoundaryRegion):
            self._layout.addRow(
                "Boundary owner ID", QLabel(str(node.owner_object_id))
            )
            self._layout.addRow(
                "Outside direction",
                QLabel(
                    str(node.outside_direction)
                    if node.outside_direction is not None
                    else "all owner surfaces"
                ),
            )
            return
        self._add_bounding_box_summary(node)
        if isinstance(node, (Sphere, Box, BoxFrame, CappedCone, Cone, Cylinder, Pyramid, Torus)):
            self._add_vector(
                "Center",
                node.center,
                lambda value: self._set_value("center", value),
            )
        if isinstance(node, Sphere):
            self._add_full_length(
                "Diameter",
                node.radius,
                lambda value: self._set_value("radius", value),
            )
        elif isinstance(node, Box):
            self._add_full_size(
                "Size",
                node.half_size,
                lambda value: self._set_value("half_size", value),
            )
        elif isinstance(node, Cylinder):
            self._add_full_length(
                "Diameter",
                node.radius,
                lambda value: self._set_value("radius", value),
            )
            self._add_float(
                "Height",
                full_length_from_half_length(node.half_height),
                lambda value: self._set_value(
                    "half_height",
                    half_length_from_full_length(value),
                ),
                0.001,
            )
        elif isinstance(node, CappedCone):
            self._add_full_length(
                "Bottom diameter",
                node.radius_a,
                lambda value: self._set_value("radius_a", value),
            )
            self._add_full_length(
                "Top diameter",
                node.radius_b,
                lambda value: self._set_value("radius_b", value),
            )
            self._add_float(
                "Height",
                full_length_from_half_length(node.half_height),
                lambda value: self._set_value(
                    "half_height",
                    half_length_from_full_length(value),
                ),
                0.001,
            )
        elif isinstance(node, Cone):
            self._add_full_length(
                "Base diameter",
                node.radius,
                lambda value: self._set_value("radius", value),
            )
            self._add_float(
                "Height",
                full_length_from_half_length(node.half_height),
                lambda value: self._set_value(
                    "half_height",
                    half_length_from_full_length(value),
                ),
                0.001,
            )
        elif isinstance(node, Pyramid):
            self._add_full_length(
                "Base size",
                node.base_half_size,
                lambda value: self._set_value("base_half_size", value),
            )
            self._add_float(
                "Height",
                full_length_from_half_length(node.half_height),
                lambda value: self._set_value(
                    "half_height",
                    half_length_from_full_length(value),
                ),
                0.001,
            )
        elif isinstance(node, BoxFrame):
            self._add_full_size(
                "Size",
                node.half_size,
                lambda value: self._set_value("half_size", value),
            )
            self._add_float(
                "Edge thickness",
                node.thickness,
                lambda value: self._set_value("thickness", value),
                0.001,
            )
        elif isinstance(node, Torus):
            self._add_float(
                "Major diameter",
                full_length_from_half_length(node.major_radius),
                lambda value: self._set_value(
                    "major_radius",
                    half_length_from_full_length(value),
                ),
                0.001,
            )
            self._add_float(
                "Minor diameter",
                full_length_from_half_length(node.minor_radius),
                lambda value: self._set_value(
                    "minor_radius",
                    half_length_from_full_length(value),
                ),
                0.001,
            )
        elif isinstance(node, Translate):
            self._add_vector(
                "Offset",
                node.offset,
                lambda value: self._set_value("offset", value),
            )
        elif isinstance(node, Rotate):
            axis = QComboBox()
            axis.addItems(("x", "y", "z"))
            axis.setCurrentText(node.axis)
            axis.currentTextChanged.connect(
                lambda value: self._set_value("axis", value)
            )
            self._layout.addRow("Axis", axis)
            angle = CadScalarSpinBox(parse_suffixes=("deg",))
            angle.setDecimals(3)
            angle.setRange(-360.0, 360.0)
            angle.setSuffix(" deg")
            angle.setKeyboardTracking(False)
            angle.setValue(node.angle_degrees)
            angle.editingFinished.connect(
                lambda control=angle: self._set_value(
                    "angle_degrees",
                    control.value(),
                )
            )
            self._layout.addRow("Angle", angle)
        elif isinstance(node, Scale):
            scale = CadScalarSpinBox()
            scale.setDecimals(5)
            scale.setRange(0.001, 1000.0)
            scale.setKeyboardTracking(False)
            scale.setValue(node.factor)
            scale.editingFinished.connect(
                lambda control=scale: self._set_value("factor", control.value())
            )
            self._layout.addRow("Factor", scale)
        elif isinstance(node, Revolve):
            if (
                node.axis_origin is not None
                or node.axis_direction is not None
                or node.radial_direction is not None
            ):
                self._layout.addRow("Revolve axis", QLabel("Custom"))
                if node.axis_origin is not None:
                    self._layout.addRow(
                        "Axis origin",
                        QLabel(read_only_vector_text(node.axis_origin)),
                    )
                if node.axis_direction is not None:
                    self._layout.addRow(
                        "Axis direction",
                        QLabel(read_only_vector_text(node.axis_direction)),
                    )
            else:
                axis = QComboBox()
                section = node.section
                axis_u_label = (
                    world_axis_label(section.axis_u)
                    if section is not None
                    else "first profile axis"
                )
                axis_v_label = (
                    world_axis_label(section.axis_v)
                    if section is not None
                    else "second profile axis"
                )
                axis.addItem(axis_u_label, "u")
                axis.addItem(axis_v_label, "v")
                axis.setCurrentIndex(0 if node.axis == "u" else 1)
                axis.currentIndexChanged.connect(
                    lambda _index, control=axis: self._set_value(
                        "axis",
                        control.currentData(),
                    )
                )
                self._layout.addRow("Revolve axis", axis)
            angle = CadScalarSpinBox(parse_suffixes=("deg",))
            angle.setDecimals(3)
            angle.setRange(-360.0, 360.0)
            angle.setSuffix(" deg")
            angle.setKeyboardTracking(False)
            angle.setValue(node.angle_degrees)
            angle.editingFinished.connect(
                lambda control=angle: self._set_value(
                    "angle_degrees",
                    control.value(),
                )
            )
            self._layout.addRow("Angle", angle)
        elif isinstance(node, (PolylineTube, QuadraticBezierTube)):
            minimum = 3 if isinstance(node, QuadraticBezierTube) else 2
            self._add_tube_point_fields(node, minimum)
            caps = QComboBox()
            caps.addItems(("round", "flat"))
            caps.setCurrentText(node.caps)
            caps.currentTextChanged.connect(
                lambda value: self._set_value("caps", value)
            )
            self._layout.addRow("Caps", caps)
            self._add_float(
                "Radius",
                node.radius,
                lambda value: self._set_value("radius", value),
                0.001,
            )
            self._add_float(
                "Inner radius",
                node.inner_radius,
                lambda value: self._set_value("inner_radius", value),
                0.0,
            )
        elif isinstance(node, PlacedSDF1D):
            axis_label = standard_axis_label(node.axis_u) or "Custom"
            self._layout.addRow("Reference axis", QLabel(axis_label))
            self._add_vector(
                "Origin",
                node.origin,
                lambda value: self._set_value("origin", value),
            )
            self._add_vector(
                "Axis U",
                node.axis_u,
                lambda value: self._set_placed_axes("axis_u", value),
            )
            self._add_profile_1d_fields(node)
        elif isinstance(node, PlacedPolyline1D):
            if node.profile is not None:
                self._layout.addRow("Profile", QLabel(node.profile.kind))
                minimum = 3 if isinstance(node.profile, QuadraticBezierCurveProfile) else 2
                self._add_point_profile_fields(node, node.profile, minimum)
        elif isinstance(node, PlacedSDF2D):
            if not isinstance(node.profile, (QuadraticBezierSurfaceProfile, PolygonProfile)):
                self._layout.addRow(
                    "Workplane",
                    QLabel(standard_workplane_label(node.axis_u, node.axis_v)),
                )
                self._add_vector(
                    "Origin",
                    node.origin,
                    lambda value: self._set_value("origin", value),
                )
                self._add_vector(
                    "Axis U",
                    node.axis_u,
                    lambda value: self._set_placed_axes("axis_u", value),
                )
                self._add_vector(
                    "Axis V",
                    node.axis_v,
                    lambda value: self._set_placed_axes("axis_v", value),
                )
            self._add_profile_fields(node)
        elif isinstance(node, Extrude):
            self._add_float(
                "Height",
                node.height,
                lambda value: self._set_value("height", value),
                0.001,
            )
            self._add_float(
                "Center offset",
                node.center_offset,
                lambda value: self._set_value("center_offset", value),
            )

    def _add_profile_1d_fields(self, node: PlacedSDF1D) -> None:
        profile = node.profile
        if profile is None:
            return
        self._layout.addRow("Profile", QLabel(profile.kind))
        if isinstance(profile, SegmentProfile):
            self._add_float(
                segment_profile_center_label(),
                profile.center,
                lambda value: self._set_profile_value("center", value),
            )
            self._add_float(
                segment_profile_length_label(),
                full_length_from_half_length(profile.half_length),
                lambda value: self._set_profile_value(
                    "half_length",
                    half_length_from_full_length(value),
                ),
                0.001,
            )

    def _add_profile_fields(self, node: PlacedSDF2D) -> None:
        profile = node.profile
        if profile is None:
            return
        self._layout.addRow("Profile", QLabel(profile.kind))
        if hasattr(profile, "center"):
            self._add_vector(
                "Profile center",
                profile.center,
                lambda value: self._set_profile_value("center", value),
                axis_labels=("U", "V"),
            )
        if isinstance(profile, (CircleProfile, RegularPolygonProfile)):
            self._add_full_length(
                "Diameter",
                profile.radius,
                lambda value: self._set_profile_value("radius", value),
            )
        if isinstance(profile, RectangleProfile) and not isinstance(
            profile, SquareProfile
        ):
            minimum = (
                rounded_rectangle_full_size_minimum(profile.corner_radius)
                if isinstance(profile, RoundedRectangleProfile)
                else 0.001
            )
            self._add_full_size(
                "Size",
                profile.half_size,
                lambda value: self._set_profile_value("half_size", value),
                minimum,
                axis_labels=("U", "V"),
            )
        if isinstance(profile, SquareProfile):
            self._add_full_length(
                "Size",
                profile.half_size,
                lambda value: self._set_profile_value("half_size", value),
            )
        if isinstance(profile, RoundedRectangleProfile):
            self._add_float(
                "Corner radius",
                profile.corner_radius,
                lambda value: self._set_profile_value("corner_radius", value),
                0.001,
                rounded_rectangle_corner_radius_maximum(profile.half_size),
            )
        if isinstance(profile, EllipseProfile):
            self._add_full_size(
                "Axes",
                profile.semi_axes,
                lambda value: self._set_profile_value("semi_axes", value),
                axis_labels=("U", "V"),
            )
        if isinstance(profile, RegularPolygonProfile):
            sides = QDoubleSpinBox()
            sides.setDecimals(0)
            sides.setRange(3, 64)
            sides.setValue(profile.side_count)
            sides.editingFinished.connect(
                lambda control=sides: self._set_profile_value(
                    "side_count",
                    int(control.value()),
                )
            )
            self._layout.addRow("Sides", sides)
        if isinstance(profile, (PolygonProfile, QuadraticBezierSurfaceProfile)):
            self._add_point_profile_fields(node, profile, 3)

    def _add_point_profile_fields(
        self,
        node: PlacedPolyline1D | PlacedSDF2D,
        profile: QuadraticBezierCurveProfile | QuadraticBezierSurfaceProfile | PolylineProfile | PolygonProfile,
        minimum_points: int,
    ) -> None:
        editor = QLineEdit(
            format_point_list_text(
                placed_profile_world_points(
                    node.origin,
                    node.axis_u,
                    node.axis_v,
                    profile.points,
                )
            )
        )
        editor.setToolTip(
            "Ordered world points as {x;y;z} entries. "
            "Quadratic Bezier points alternate anchor, control, anchor. "
            "Polyline closure is explicit; polygon closure is automatic."
        )
        editor.editingFinished.connect(
            lambda control=editor, minimum=minimum_points: self._set_point_profile(
                control.text(),
                minimum,
            )
        )
        self._layout.addRow("Points", editor)

    def _add_tube_point_fields(
        self,
        node: PolylineTube | QuadraticBezierTube,
        minimum_points: int,
    ) -> None:
        editor = QLineEdit(format_point_list_text(node.points))
        editor.setToolTip(
            "Ordered world points as {x;y;z} entries. "
            "Quadratic Bezier tube points alternate anchor, control, anchor."
        )
        editor.editingFinished.connect(
            lambda control=editor, minimum=minimum_points: self._set_tube_points(
                control.text(),
                minimum,
            )
        )
        self._layout.addRow("Points", editor)

    def _set_tube_points(self, text: str, minimum_points: int) -> None:
        if not isinstance(self._node, (PolylineTube, QuadraticBezierTube)):
            return
        undo_snapshot = self._undo_snapshot()
        previous = self._node.points
        try:
            self._node.points = parse_point_list_text(text, minimum_points)
            self._node.__post_init__()
        except ValueError as error:
            self._node.points = previous
            signals.log_message.emit("warning", str(error))
            self._build_form()
            return
        self._emit_undo_snapshot(undo_snapshot)
        signals.node_edited.emit()

    def _set_point_profile(self, text: str, minimum_points: int) -> None:
        if not isinstance(self._node, (PlacedPolyline1D, PlacedSDF2D)):
            return
        if self._node.profile is None:
            return
        undo_snapshot = self._undo_snapshot()
        try:
            world_points = parse_point_list_text(text, minimum_points)
            local_points = placed_profile_local_points(
                self._node.origin,
                self._node.axis_u,
                self._node.axis_v,
                world_points,
            )
            self._node.profile = replace(self._node.profile, points=local_points)
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            self._build_form()
            return
        self._emit_undo_snapshot(undo_snapshot)
        signals.node_edited.emit()

    def _add_bounding_box_summary(self, node: SDFNode) -> None:
        try:
            box = node.bounding_box()
        except (NotImplementedError, ValueError):
            return
        self._layout.addRow(
            "Bounds center",
            QLabel(read_only_vector_text(bounding_box_center(box))),
        )
        self._layout.addRow(
            "Bounds size",
            QLabel(read_only_vector_text(bounding_box_size(box))),
        )

    def _spin(self, value: float, minimum: float = -1000.0) -> QDoubleSpinBox:
        spin = CadDimensionSpinBox()
        spin.setDecimals(5)
        spin.setRange(minimum, 1000.0)
        spin.setSingleStep(0.01)
        spin.setSuffix(" m")
        spin.setKeyboardTracking(False)
        spin.setValue(value)
        return spin

    def _add_float(
        self,
        label: str,
        value: float,
        setter: Callable[[float], None],
        minimum: float = -1000.0,
        maximum: float = 1000.0,
    ) -> None:
        spin = self._spin(value, minimum)
        spin.setMaximum(maximum)
        spin.editingFinished.connect(lambda control=spin: setter(control.value()))
        self._layout.addRow(label, spin)

    def _add_full_length(
        self,
        label: str,
        half_value: float,
        setter: Callable[[float], None],
    ) -> None:
        self._add_float(
            label,
            full_length_from_half_length(half_value),
            lambda value: setter(half_length_from_full_length(value)),
            0.001,
        )

    def _add_vector(
        self,
        label: str,
        value: tuple[float, ...],
        setter: Callable[[tuple[float, ...]], None],
        minimum: float = -1000.0,
        axis_labels: tuple[str, ...] | None = None,
    ) -> None:
        spins = [self._spin(component, minimum) for component in value]
        container = QWidget()
        layout = QFormLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        for axis, spin in zip(
            vector_component_labels(len(spins), axis_labels),
            spins,
            strict=True,
        ):
            layout.addRow(axis, spin)
            spin.editingFinished.connect(
                lambda controls=spins: setter(
                    tuple(control.value() for control in controls)
                )
            )
        self._layout.addRow(label, container)

    def _add_full_size(
        self,
        label: str,
        half_size: tuple[float, ...],
        setter: Callable[[tuple[float, ...]], None],
        minimum: float = 0.001,
        axis_labels: tuple[str, ...] | None = None,
    ) -> None:
        self._add_vector(
            label,
            full_size_from_half_size(half_size),
            lambda value: setter(half_size_from_full_size(value)),
            minimum=minimum,
            axis_labels=axis_labels,
        )

    def _set_value(self, attribute: str, value: object) -> None:
        if self._node is None:
            return
        previous = getattr(self._node, attribute)
        if values_equal(previous, value):
            return
        undo_snapshot = self._undo_snapshot()
        try:
            setattr(self._node, attribute, value)
            if hasattr(self._node, "__post_init__"):
                self._node.__post_init__()
        except ValueError as error:
            setattr(self._node, attribute, previous)
            signals.log_message.emit("warning", str(error))
            return
        self._emit_undo_snapshot(undo_snapshot)
        signals.node_edited.emit()

    def _set_profile_value(self, attribute: str, value: object) -> None:
        if (
            not isinstance(self._node, (PlacedSDF1D, PlacedSDF2D))
            or self._node.profile is None
        ):
            return
        previous = getattr(self._node.profile, attribute)
        if values_equal(previous, value):
            return
        undo_snapshot = self._undo_snapshot()
        try:
            self._node.profile = replace(self._node.profile, **{attribute: value})
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        self._emit_undo_snapshot(undo_snapshot)
        signals.node_edited.emit()

    def _set_placed_axes(self, attribute: str, value: tuple[float, ...]) -> None:
        if not isinstance(self._node, (PlacedSDF1D, PlacedPolyline1D, PlacedSDF2D)):
            return
        previous = getattr(self._node, attribute)
        if values_equal(previous, value):
            return
        undo_snapshot = self._undo_snapshot()
        try:
            setattr(self._node, attribute, value)
            self._node.__post_init__()
        except ValueError:
            setattr(self._node, attribute, previous)
            return
        self._emit_undo_snapshot(undo_snapshot)
        signals.node_edited.emit()

    def _undo_snapshot(self) -> SceneDocument | None:
        if self._document is None:
            return None
        try:
            return self._document.snapshot()
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return None

    def _emit_undo_snapshot(self, snapshot: SceneDocument | None) -> None:
        if snapshot is not None:
            signals.undo_snapshot_ready.emit(snapshot)
