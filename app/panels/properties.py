from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

from PySide6.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QWidget,
)

from app.signals import signals
from core.boundary import BoundaryRegion
from core.scene import SceneDocument
from core.sdf import (
    Box,
    CircleProfile,
    Cylinder,
    EllipseProfile,
    Extrude,
    IntervalProfile,
    PlacedSDF1D,
    PlacedSDF2D,
    RectangleProfile,
    RegularPolygonProfile,
    Rotate,
    RoundedRectangleProfile,
    Scale,
    SmoothUnion,
    Sphere,
    SquareProfile,
    Sweep,
    Torus,
    Translate,
)
from core.sdf.base import SDFNode

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
        self._layout.addRow("Kind", QLabel(node.kind))
        self._layout.addRow("Dimension", QLabel(f"{node.dimension}D"))
        if node.object_id:
            self._layout.addRow("Object ID", QLabel(str(node.object_id)))
        name = QLineEdit(node.name)
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
        if isinstance(node, (Sphere, Box, Cylinder, Torus)):
            self._add_vector(
                "Center",
                node.center,
                lambda value: self._set_value("center", value),
            )
        if isinstance(node, Sphere):
            self._add_float("Radius", node.radius, lambda value: self._set_value("radius", value), 0.001)
        elif isinstance(node, Box):
            self._add_vector(
                "Half size",
                node.half_size,
                lambda value: self._set_value("half_size", value),
                minimum=0.001,
            )
        elif isinstance(node, Cylinder):
            self._add_float("Radius", node.radius, lambda value: self._set_value("radius", value), 0.001)
            self._add_float(
                "Half height",
                node.half_height,
                lambda value: self._set_value("half_height", value),
                0.001,
            )
        elif isinstance(node, Torus):
            self._add_float(
                "Major radius",
                node.major_radius,
                lambda value: self._set_value("major_radius", value),
                0.001,
            )
            self._add_float(
                "Minor radius",
                node.minor_radius,
                lambda value: self._set_value("minor_radius", value),
                0.001,
            )
        elif isinstance(node, SmoothUnion):
            self._add_float(
                "Smoothing",
                node.smoothing,
                lambda value: self._set_value("smoothing", value),
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
            angle = QDoubleSpinBox()
            angle.setDecimals(3)
            angle.setRange(-360.0, 360.0)
            angle.setSuffix(" deg")
            angle.setValue(node.angle_degrees)
            angle.valueChanged.connect(
                lambda value: self._set_value("angle_degrees", value)
            )
            self._layout.addRow("Angle", angle)
        elif isinstance(node, Scale):
            scale = QDoubleSpinBox()
            scale.setDecimals(5)
            scale.setRange(0.001, 1000.0)
            scale.setValue(node.factor)
            scale.valueChanged.connect(
                lambda value: self._set_value("factor", value)
            )
            self._layout.addRow("Factor", scale)
        elif isinstance(node, PlacedSDF1D):
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
        elif isinstance(node, PlacedSDF2D):
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
        elif isinstance(node, Sweep):
            self._add_vector(
                "Path end",
                node.end,
                lambda value: self._set_value("end", value),
            )
        elif isinstance(node, Extrude):
            self._add_float(
                "Height",
                node.height,
                lambda value: self._set_value("height", value),
                0.001,
            )

    def _add_profile_1d_fields(self, node: PlacedSDF1D) -> None:
        profile = node.profile
        if profile is None:
            return
        self._layout.addRow("Profile", QLabel(profile.kind))
        if isinstance(profile, IntervalProfile):
            self._add_float(
                "Center",
                profile.center,
                lambda value: self._set_profile_value("center", value),
            )
            self._add_float(
                "Half length",
                profile.half_length,
                lambda value: self._set_profile_value("half_length", value),
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
            )
        if isinstance(profile, (CircleProfile, RegularPolygonProfile)):
            self._add_float(
                "Radius",
                profile.radius,
                lambda value: self._set_profile_value("radius", value),
                0.001,
            )
        if isinstance(profile, RectangleProfile) and not isinstance(
            profile, SquareProfile
        ):
            self._add_vector(
                "Half size",
                profile.half_size,
                lambda value: self._set_profile_value("half_size", value),
                0.001,
            )
        if isinstance(profile, SquareProfile):
            self._add_float(
                "Half size",
                profile.half_size,
                lambda value: self._set_profile_value("half_size", value),
                0.001,
            )
        if isinstance(profile, RoundedRectangleProfile):
            self._add_float(
                "Corner radius",
                profile.corner_radius,
                lambda value: self._set_profile_value("corner_radius", value),
                0.001,
            )
        if isinstance(profile, EllipseProfile):
            self._add_vector(
                "Semi axes",
                profile.semi_axes,
                lambda value: self._set_profile_value("semi_axes", value),
                0.001,
            )
        if isinstance(profile, RegularPolygonProfile):
            sides = QDoubleSpinBox()
            sides.setDecimals(0)
            sides.setRange(3, 64)
            sides.setValue(profile.side_count)
            sides.valueChanged.connect(
                lambda value: self._set_profile_value("side_count", int(value))
            )
            self._layout.addRow("Sides", sides)

    def _spin(self, value: float, minimum: float = -1000.0) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(5)
        spin.setRange(minimum, 1000.0)
        spin.setSingleStep(0.01)
        spin.setSuffix(" m")
        spin.setValue(value)
        return spin

    def _add_float(
        self,
        label: str,
        value: float,
        setter: Callable[[float], None],
        minimum: float = -1000.0,
    ) -> None:
        spin = self._spin(value, minimum)
        spin.valueChanged.connect(setter)
        self._layout.addRow(label, spin)

    def _add_vector(
        self,
        label: str,
        value: tuple[float, ...],
        setter: Callable[[tuple[float, ...]], None],
        minimum: float = -1000.0,
    ) -> None:
        spins = [self._spin(component, minimum) for component in value]
        container = QWidget()
        layout = QFormLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        for axis, spin in zip("XYZ"[: len(spins)], spins, strict=True):
            layout.addRow(axis, spin)
            spin.valueChanged.connect(
                lambda _value, controls=spins: setter(
                    tuple(control.value() for control in controls)
                )
            )
        self._layout.addRow(label, container)

    def _set_value(self, attribute: str, value: object) -> None:
        if self._node is None:
            return
        setattr(self._node, attribute, value)
        signals.node_edited.emit()

    def _set_profile_value(self, attribute: str, value: object) -> None:
        if (
            not isinstance(self._node, (PlacedSDF1D, PlacedSDF2D))
            or self._node.profile is None
        ):
            return
        self._node.profile = replace(self._node.profile, **{attribute: value})
        signals.node_edited.emit()

    def _set_placed_axes(self, attribute: str, value: tuple[float, ...]) -> None:
        if not isinstance(self._node, (PlacedSDF1D, PlacedSDF2D)):
            return
        previous = getattr(self._node, attribute)
        try:
            setattr(self._node, attribute, value)
            self._node.__post_init__()
        except ValueError:
            setattr(self._node, attribute, previous)
            return
        signals.node_edited.emit()
