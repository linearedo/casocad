from __future__ import annotations

import logging
from copy import deepcopy
from math import atan2, degrees, radians

import moderngl
import numpy as np
from PySide6.QtCore import QElapsedTimer, QPoint, QTimer, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QApplication, QFrame, QHBoxLayout, QLabel, QPushButton

from app.artifacts import RenderArtifactTimings
from app.dimensions import (
    dimension_entry_text,
    parse_dimension_entry,
    parse_displacement_entry,
    parse_scalar_entry,
)
from app.viewport.renderer_base import ViewportRenderer
from app.signals import signals
from core.boundary import BoundaryRegion
from core.mesher.classifier import pick_boundary_owner, pick_sdf_surface
from core.render_ir import RenderIR, build_render_ir
from core.scene import SceneDocument
from core.sdf import PlacedSDF2D, SDFTree
from core.sdf.base import BoundingBox3D, SDFNode

from .camera import OrbitCamera
from .renderers.opengl.renderer import OpenGLRenderer
from .renderers.opengl.renderer import (
    MAX_SELECTED_BOUNDARY_OWNERS,
    POINT_VERTEX_WIDTH,
    SDFRenderer,
    SQUARE_INSTANCE_WIDTH,
    X_AXIS_COLOR,
    Y_AXIS_COLOR,
    Z_AXIS_COLOR,
    WORLD_AXIS_LENGTH,
    ROTATION_GIZMO_SEGMENTS,
)

logger = logging.getLogger(__name__)
MAX_SELECTED_BOUNDARY_REGIONS = 128
FPS_COUNTER_UPDATE_MS = 500
VIEW_ANIMATION_DURATION_MS = 180
VIEW_ANIMATION_INTERVAL_MS = 16
LATTICE_UPLOAD_BUDGET_MS = 2
LATTICE_POINT_UPLOAD_CHUNK = 100_000
LATTICE_SQUARE_UPLOAD_CHUNK = 25_000
CREATE_CLICK_MAX_MANHATTAN = 4
SCENE_CLICK_MAX_MANHATTAN = 4
ROTATION_GIZMO_PICK_TOLERANCE_PX = 12.0
ROTATION_GIZMO_MIN_RADIUS = 0.25
BOOLEAN_PREVIEW_OPERATIONS = {
    "union": 1,
    "intersection": 2,
    "difference": 3,
    "smooth_union": 4,
}
REFERENCE_PLANE_IDS = {"xy": 0, "xz": 1, "yz": 2}
REFERENCE_PLANE_SEQUENCE = ("xy", "xz", "yz")
REFERENCE_PLANE_AXES = {
    "xy": ((0, "X"), (1, "Y")),
    "xz": ((0, "X"), (2, "Z")),
    "yz": ((1, "Y"), (2, "Z")),
}
CREATE_LABELS = {
    "segment": "Segment",
    "polyline": "Polyline",
    "bezier_curve": "Bezier Curve",
    "bezier_polycurve": "Bezier Polycurve",
    "circle": "Circle",
    "rectangle": "Rectangle",
    "square": "Square",
    "rounded_rectangle": "Rounded Rectangle",
    "ellipse": "Ellipse",
    "regular_polygon": "Regular Polygon",
    "polygon": "Polygon",
    "bezier_surface": "Bezier Surface",
    "sphere": "Sphere",
    "box": "Box",
    "box_frame": "Box Frame",
    "cylinder": "Cylinder",
    "capped_cone": "Capped Cone",
    "cone": "Cone",
    "pyramid": "Pyramid",
    "torus": "Torus",
    "polyline_tube": "Polyline Tube",
    "bezier_tube": "Bezier Tube",
}
RADIAL_CREATE_KINDS = {"circle", "regular_polygon", "sphere", "cylinder", "capped_cone", "cone", "torus"}
SINGLE_DIAMETER_CREATE_KINDS = {"circle", "regular_polygon", "sphere"}
POINT_CREATE_KINDS = {
    "polyline",
    "bezier_curve",
    "bezier_polycurve",
    "polyline_tube",
    "bezier_tube",
    "bezier_surface",
    "polygon",
}

EMPTY_RENDER_SCENE_KEY = ("empty-render-ir-scene",)


def render_scene_key(
    render_ir: RenderIR | None,
) -> object:
    if render_ir is None:
        return EMPTY_RENDER_SCENE_KEY
    return (render_ir.topology_signature, render_ir.parameter_values)


def _format_render_artifact_timings(timings: RenderArtifactTimings) -> str:
    ir_support = "yes" if timings.render_ir_supported else "no"
    return (
        "Render artifact built: "
        f"total={timings.total_ms:.1f} ms, "
        f"render_ir={timings.render_ir_ms:.1f} ms, "
        f"tree_nodes={timings.tree_node_count}, "
        f"ir_nodes={timings.render_ir_node_count}, "
        f"ir_supported={ir_support}"
    )


def _format_scene_update_stats(stats: object) -> str:
    path = getattr(stats, "path", "unknown")
    return (
        "Viewport scene update: "
        f"path={path}, "
        f"total={getattr(stats, 'total_ms', 0.0):.1f} ms, "
        f"shader_build={getattr(stats, 'shader_build_ms', 0.0):.1f} ms, "
        f"program_compile={getattr(stats, 'program_compile_ms', 0.0):.1f} ms, "
        f"vao_build={getattr(stats, 'vao_build_ms', 0.0):.1f} ms, "
        f"ir_nodes={getattr(stats, 'render_ir_nodes', 0)}, "
        f"reused_program={getattr(stats, 'reused_program', False)}"
    )


def constrain_reference_point(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    reference_plane: str,
    kind: str,
) -> tuple[float, float, float]:
    values = list(point)
    start_values = list(start)
    (first_axis, _first_label), (second_axis, _second_label) = (
        REFERENCE_PLANE_AXES[reference_plane]
    )
    first_delta = values[first_axis] - start_values[first_axis]
    second_delta = values[second_axis] - start_values[second_axis]
    if kind in {
        "segment",
        "polyline",
        "bezier_curve",
        "bezier_polycurve",
        "polyline_tube",
        "bezier_tube",
    }:
        if abs(first_delta) >= abs(second_delta):
            values[second_axis] = start_values[second_axis]
        else:
            values[first_axis] = start_values[first_axis]
        return tuple(float(value) for value in values)
    extent = max(abs(first_delta), abs(second_delta))
    if extent <= 1e-12:
        return point
    first_sign = -1.0 if first_delta < 0.0 else 1.0
    second_sign = -1.0 if second_delta < 0.0 else 1.0
    values[first_axis] = start_values[first_axis] + first_sign * extent
    values[second_axis] = start_values[second_axis] + second_sign * extent
    return tuple(float(value) for value in values)


def constrain_move_point(
    point: tuple[float, float, float],
    start: tuple[float, float, float],
    reference_plane: str,
) -> tuple[float, float, float]:
    values = list(point)
    start_values = list(start)
    (first_axis, _first_label), (second_axis, _second_label) = (
        REFERENCE_PLANE_AXES[reference_plane]
    )
    first_delta = values[first_axis] - start_values[first_axis]
    second_delta = values[second_axis] - start_values[second_axis]
    if abs(first_delta) >= abs(second_delta):
        values[second_axis] = start_values[second_axis]
    else:
        values[first_axis] = start_values[first_axis]
    return tuple(float(value) for value in values)


def should_defer_create_release(
    start_screen: tuple[int, int],
    end_screen: tuple[int, int],
    has_typed_input: bool,
) -> bool:
    if has_typed_input:
        return False
    return (
        abs(end_screen[0] - start_screen[0])
        + abs(end_screen[1] - start_screen[1])
        <= CREATE_CLICK_MAX_MANHATTAN
    )


def apply_typed_create_dimensions(
    start: tuple[float, float, float],
    current: tuple[float, float, float],
    reference_plane: str,
    kind: str,
    dimensions: tuple[float, ...],
) -> tuple[float, float, float]:
    values = list(start)
    current_values = list(current)
    (first_axis, _first_label), (second_axis, _second_label) = (
        REFERENCE_PLANE_AXES[reference_plane]
    )

    def apply_diameter_along_drag(diameter: float) -> tuple[float, float, float]:
        first_delta = current_values[first_axis] - values[first_axis]
        second_delta = current_values[second_axis] - values[second_axis]
        length = float((first_delta * first_delta + second_delta * second_delta) ** 0.5)
        if length <= 1e-12:
            values[first_axis] += diameter
        else:
            values[first_axis] += diameter * first_delta / length
            values[second_axis] += diameter * second_delta / length
        return tuple(float(value) for value in values)

    if kind in {
        "segment",
        "polyline",
        "bezier_curve",
        "bezier_polycurve",
        "polyline_tube",
        "bezier_tube",
    }:
        length = dimensions[0]
        first_delta = current_values[first_axis] - values[first_axis]
        second_delta = current_values[second_axis] - values[second_axis]
        axis = first_axis if abs(first_delta) >= abs(second_delta) else second_axis
        sign = -1.0 if current_values[axis] < values[axis] else 1.0
        values[axis] += sign * length
        return tuple(float(value) for value in values)
    if kind in SINGLE_DIAMETER_CREATE_KINDS:
        return apply_diameter_along_drag(dimensions[0])
    if kind in RADIAL_CREATE_KINDS and len(dimensions) == 1:
        return apply_diameter_along_drag(dimensions[0])
    if kind == "torus" and len(dimensions) >= 2:
        return apply_diameter_along_drag(dimensions[0])
    if kind == "capped_cone" and len(dimensions) >= 3:
        radial_delta_x = current_values[0] - values[0]
        radial_delta_y = current_values[1] - values[1]
        radial_length = float(
            (radial_delta_x * radial_delta_x + radial_delta_y * radial_delta_y)
            ** 0.5
        )
        if radial_length <= 1e-12:
            fallback_axis = first_axis if first_axis in {0, 1} else second_axis
            radial_delta_x = 1.0 if fallback_axis == 0 else 0.0
            radial_delta_y = 1.0 if fallback_axis == 1 else 0.0
            radial_length = 1.0
        height_delta = current_values[2] - values[2]
        height_sign = -1.0 if height_delta < 0.0 else 1.0
        values[0] += dimensions[0] * radial_delta_x / radial_length
        values[1] += dimensions[0] * radial_delta_y / radial_length
        values[2] += height_sign * dimensions[2]
        return tuple(float(value) for value in values)
    if kind in {"cylinder", "cone", "pyramid"} and len(dimensions) >= 2:
        radial_delta_x = current_values[0] - values[0]
        radial_delta_y = current_values[1] - values[1]
        radial_length = float(
            (radial_delta_x * radial_delta_x + radial_delta_y * radial_delta_y)
            ** 0.5
        )
        if radial_length <= 1e-12:
            fallback_axis = first_axis if first_axis in {0, 1} else second_axis
            radial_delta_x = 1.0 if fallback_axis == 0 else 0.0
            radial_delta_y = 1.0 if fallback_axis == 1 else 0.0
            radial_length = 1.0
        height_delta = current_values[2] - values[2]
        height_sign = -1.0 if height_delta < 0.0 else 1.0
        values[0] += dimensions[0] * radial_delta_x / radial_length
        values[1] += dimensions[0] * radial_delta_y / radial_length
        values[2] += height_sign * dimensions[1]
        return tuple(float(value) for value in values)
    if kind in {"box", "box_frame"} and len(dimensions) >= 3:
        inactive_axis = next(
            axis for axis in range(3) if axis not in {first_axis, second_axis}
        )
        first_delta = current_values[first_axis] - values[first_axis]
        second_delta = current_values[second_axis] - values[second_axis]
        third_delta = current_values[inactive_axis] - values[inactive_axis]
        first_sign = -1.0 if first_delta < 0.0 else 1.0
        second_sign = -1.0 if second_delta < 0.0 else 1.0
        third_sign = -1.0 if third_delta < 0.0 else 1.0
        values[first_axis] += first_sign * dimensions[0]
        values[second_axis] += second_sign * dimensions[1]
        values[inactive_axis] += third_sign * dimensions[2]
        return tuple(float(value) for value in values)
    if kind == "square":
        first_delta = current_values[first_axis] - values[first_axis]
        second_delta = current_values[second_axis] - values[second_axis]
        first_sign = -1.0 if first_delta < 0.0 else 1.0
        second_sign = -1.0 if second_delta < 0.0 else 1.0
        values[first_axis] += first_sign * dimensions[0]
        values[second_axis] += second_sign * dimensions[0]
        return tuple(float(value) for value in values)
    first = dimensions[0]
    second = dimensions[1] if len(dimensions) >= 2 else first
    first_delta = current_values[first_axis] - values[first_axis]
    second_delta = current_values[second_axis] - values[second_axis]
    first_sign = -1.0 if first_delta < 0.0 else 1.0
    second_sign = -1.0 if second_delta < 0.0 else 1.0
    values[first_axis] += first_sign * first
    values[second_axis] += second_sign * second
    return tuple(float(value) for value in values)


def centered_create_endpoints(
    center: tuple[float, float, float],
    corner: tuple[float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    opposite = tuple(
        float(center[index] - (corner[index] - center[index]))
        for index in range(3)
    )
    return opposite, corner


def create_effective_endpoints(
    anchor: tuple[float, float, float],
    current: tuple[float, float, float],
    reference_plane: str,
    kind: str,
    dimensions: tuple[float, ...] | None,
    centered: bool,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    end = current
    if dimensions is not None:
        effective_dimensions = (
            tuple(0.5 * value for value in dimensions)
            if centered
            else dimensions
        )
        end = apply_typed_create_dimensions(
            anchor,
            current,
            reference_plane,
            kind,
            effective_dimensions,
        )
    if centered:
        return centered_create_endpoints(anchor, end)
    return anchor, end


def create_typed_parameters(
    kind: str,
    dimensions: tuple[float, ...] | None,
) -> dict[str, float]:
    if kind == "capped_cone" and dimensions is not None and len(dimensions) >= 3:
        return {"top_diameter": float(dimensions[1])}
    if kind == "torus" and dimensions is not None and len(dimensions) >= 2:
        return {"minor_diameter": float(dimensions[1])}
    return {}


def create_preview_secondary_radius(
    kind: str,
    dimensions: tuple[float, ...] | None,
) -> float:
    if kind == "capped_cone" and dimensions is not None and len(dimensions) >= 3:
        return max(0.5 * float(dimensions[1]), 0.02)
    return -1.0


def create_preview_torus_minor_radius(
    kind: str,
    dimensions: tuple[float, ...] | None,
) -> float:
    if kind == "torus" and dimensions is not None and len(dimensions) >= 2:
        return max(0.5 * float(dimensions[1]), 0.02)
    return -1.0


def create_measurement_components(
    kind: str,
    first_label: str,
    first_size: float,
    second_label: str,
    second_size: float,
    delta: tuple[float, float, float] | None = None,
    typed_dimensions: tuple[float, ...] | None = None,
) -> tuple[tuple[str, float], ...]:
    if kind in {"box", "box_frame"} and delta is not None:
        x_size = abs(float(delta[0]))
        y_size = abs(float(delta[1]))
        z_size = abs(float(delta[2]))
        if min(x_size, y_size, z_size) > 1e-12:
            return (("X", x_size), ("Y", y_size), ("Z", z_size))
    if kind in {"cylinder", "cone"} and delta is not None:
        diameter = float((delta[0] * delta[0] + delta[1] * delta[1]) ** 0.5)
        height = abs(float(delta[2]))
        return (
            (first_label, first_size),
            (second_label, second_size),
            ("D", diameter),
            ("H", height),
        )
    if kind == "capped_cone" and delta is not None:
        diameter = float((delta[0] * delta[0] + delta[1] * delta[1]) ** 0.5)
        top_diameter = (
            typed_dimensions[1]
            if typed_dimensions is not None and len(typed_dimensions) >= 3
            else max(diameter * 0.45, 0.05)
        )
        height = abs(float(delta[2]))
        return (
            (first_label, first_size),
            (second_label, second_size),
            ("D1", diameter),
            ("D2", top_diameter),
            ("H", height),
        )
    if kind == "pyramid" and delta is not None:
        base = max(abs(float(delta[0])), abs(float(delta[1])))
        height = abs(float(delta[2]))
        return (
            (first_label, first_size),
            (second_label, second_size),
            ("Base", base),
            ("H", height),
        )
    diagonal = float((first_size * first_size + second_size * second_size) ** 0.5)
    if kind == "torus" and typed_dimensions is not None and len(typed_dimensions) >= 2:
        return (
            (first_label, first_size),
            (second_label, second_size),
            ("D", typed_dimensions[0]),
            ("d", typed_dimensions[1]),
        )
    if kind == "square":
        return (
            (first_label, first_size),
            (second_label, second_size),
            ("Size", max(first_size, second_size)),
        )
    if kind in RADIAL_CREATE_KINDS:
        return (
            (first_label, first_size),
            (second_label, second_size),
            ("D", diagonal),
        )
    return (
        (first_label, first_size),
        (second_label, second_size),
    )


def apply_typed_move_delta(
    current_delta: tuple[float, float, float],
    reference_plane: str,
    dimensions: tuple[float, ...],
) -> tuple[float, float, float]:
    if len(dimensions) >= 3:
        return tuple(float(value) for value in dimensions[:3])
    values = [0.0, 0.0, 0.0]
    (first_axis, _first_label), (second_axis, _second_label) = (
        REFERENCE_PLANE_AXES[reference_plane]
    )
    if len(dimensions) == 2:
        values[first_axis] = dimensions[0]
        values[second_axis] = dimensions[1]
        return tuple(float(value) for value in values)
    length = dimensions[0]
    axis = max(range(3), key=lambda index: abs(current_delta[index]))
    if abs(current_delta[axis]) <= 1e-12:
        axis = first_axis
    sign = -1.0 if current_delta[axis] < 0.0 else 1.0
    if length < 0.0:
        sign *= -1.0
    values[axis] = sign * abs(length)
    return tuple(float(value) for value in values)


def move_preview_delta(
    accumulated_delta: tuple[float, float, float],
    start: tuple[float, float, float] | None,
    current: tuple[float, float, float] | None,
    reference_plane: str,
    modifiers: Qt.KeyboardModifier | Qt.KeyboardModifiers,
) -> tuple[float, float, float]:
    drag_delta = (0.0, 0.0, 0.0)
    if start is not None and current is not None:
        effective_current = (
            constrain_move_point(current, start, reference_plane)
            if modifiers & Qt.KeyboardModifier.ShiftModifier
            else current
        )
        drag_delta = tuple(
            effective_current[index] - start[index] for index in range(3)
        )
    return tuple(
        accumulated_delta[index] + drag_delta[index]
        for index in range(3)
    )


def apply_typed_start_point(
    reference_plane: str,
    coordinates: tuple[float, ...],
) -> tuple[float, float, float]:
    if len(coordinates) >= 3:
        return tuple(float(value) for value in coordinates[:3])
    if len(coordinates) != 2:
        raise ValueError("start point requires two plane coordinates or XYZ")
    values = [0.0, 0.0, 0.0]
    for coordinate, (axis, _label) in zip(
        coordinates,
        REFERENCE_PLANE_AXES[reference_plane],
        strict=True,
    ):
        values[axis] = coordinate
    return tuple(float(value) for value in values)


def reference_plane_coordinate_components(
    point: tuple[float, float, float],
    reference_plane: str,
) -> tuple[tuple[str, float], tuple[str, float]]:
    return tuple(
        (label, float(point[axis]))
        for axis, label in REFERENCE_PLANE_AXES[reference_plane]
    )


def cursor_preview_active(
    action: str | None,
    start: tuple[float, float, float] | None,
    hover: tuple[float, float, float] | None,
) -> bool:
    return action == "create" and start is None and hover is not None


def create_input_label(has_start: bool) -> str:
    return "Size" if has_start else "Start"


def create_start_prompt(first_label: str, second_label: str) -> str:
    return f"Type Start {first_label},{second_label} or X,Y,Z"


def create_size_prompt(kind: str, first_label: str, second_label: str) -> str:
    if kind in {
        "segment",
        "polyline",
        "bezier_curve",
        "bezier_polycurve",
        "polyline_tube",
        "bezier_tube",
    }:
        return "Type Length"
    if kind == "bezier_surface":
        return "Type point coordinates"
    if kind in {"circle", "regular_polygon", "sphere"}:
        return "Type D"
    if kind == "square":
        return "Type Size"
    if kind in {"rectangle", "rounded_rectangle", "ellipse", "polygon"}:
        return f"Type {first_label} x {second_label}"
    if kind in {"box", "box_frame"}:
        return "Type X x Y x Z"
    if kind in {"cylinder", "cone"}:
        return "Type D x H"
    if kind == "capped_cone":
        return "Type D1 x D2 x H"
    if kind == "pyramid":
        return "Type Base x H"
    if kind == "torus":
        return "Type D x d"
    return "Type Size"


def create_modifier_status_text(
    modifiers: Qt.KeyboardModifier | Qt.KeyboardModifiers,
) -> str:
    labels: list[str] = []
    if modifiers & Qt.KeyboardModifier.ControlModifier:
        labels.append("Center")
    if modifiers & Qt.KeyboardModifier.ShiftModifier:
        labels.append("Lock")
    return "  ".join(labels)


def move_dimension_prompt(first_label: str = "X", second_label: str = "Y") -> str:
    return f"Type d{first_label},d{second_label} or dX,dY,dZ"


def extrude_dimension_prompt() -> str:
    return "Type signed length or drag/QE along normal"


def reference_view_for_key(key: int) -> str | None:
    mapping = {
        Qt.Key.Key_1: "3d",
        Qt.Key.Key_2: "xy",
        Qt.Key.Key_3: "xz",
        Qt.Key.Key_4: "yz",
    }
    return mapping.get(key)


def next_reference_plane(reference_plane: str, reverse: bool = False) -> str:
    try:
        index = REFERENCE_PLANE_SEQUENCE.index(reference_plane)
    except ValueError as error:
        raise ValueError(f"unknown reference plane: {reference_plane}") from error
    step = -1 if reverse else 1
    return REFERENCE_PLANE_SEQUENCE[
        (index + step) % len(REFERENCE_PLANE_SEQUENCE)
    ]


def should_cycle_reference_plane_for_key(
    key: int,
    modifiers: Qt.KeyboardModifier | Qt.KeyboardModifiers,
) -> bool:
    return key == Qt.Key.Key_Tab and modifiers in {
        Qt.KeyboardModifier.NoModifier,
        Qt.KeyboardModifier.ShiftModifier,
    }


def keyboard_move_delta(
    key: int,
    step: float,
    modifiers: Qt.KeyboardModifier | Qt.KeyboardModifiers,
) -> tuple[float, float, float] | None:
    multiplier = 1.0
    if modifiers & Qt.KeyboardModifier.ShiftModifier:
        multiplier *= 10.0
    if modifiers & Qt.KeyboardModifier.AltModifier:
        multiplier *= 0.1
    distance = float(step) * multiplier
    deltas = {
        Qt.Key.Key_W: (0.0, distance, 0.0),
        Qt.Key.Key_S: (0.0, -distance, 0.0),
        Qt.Key.Key_A: (-distance, 0.0, 0.0),
        Qt.Key.Key_D: (distance, 0.0, 0.0),
        Qt.Key.Key_Q: (0.0, 0.0, -distance),
        Qt.Key.Key_E: (0.0, 0.0, distance),
    }
    return deltas.get(key)


def keyboard_extrude_delta(
    key: int,
    step: float,
    modifiers: Qt.KeyboardModifier | Qt.KeyboardModifiers,
) -> float | None:
    if key not in {Qt.Key.Key_Q, Qt.Key.Key_E}:
        return None
    multiplier = 1.0
    if modifiers & Qt.KeyboardModifier.ShiftModifier:
        multiplier *= 10.0
    if modifiers & Qt.KeyboardModifier.AltModifier:
        multiplier *= 0.1
    direction = -1.0 if key == Qt.Key.Key_Q else 1.0
    return direction * float(step) * multiplier


def snap_scalar(
    value: float,
    spacing: float,
    enabled: bool,
    modifiers: Qt.KeyboardModifier | Qt.KeyboardModifiers,
) -> float:
    if not enabled or modifiers & Qt.KeyboardModifier.AltModifier:
        return float(value)
    step = max(float(spacing), 1.0e-9)
    return float(round(value / step) * step)


def reference_plane_label(reference_plane: str) -> str:
    if reference_plane not in REFERENCE_PLANE_AXES:
        raise ValueError(f"unknown reference plane: {reference_plane}")
    return reference_plane.upper()


def reference_plane_context(reference_plane: str) -> str:
    return f"Plane {reference_plane_label(reference_plane)}"


def snap_status_text(snap_enabled: bool) -> str:
    return "Snap On" if snap_enabled else "Snap Off"


def bottom_center_overlay_position(
    viewport_width: int,
    viewport_height: int,
    overlay_width: int,
    overlay_height: int,
    margin: int,
) -> tuple[int, int]:
    return (
        max(margin, (viewport_width - overlay_width) // 2),
        max(margin, viewport_height - overlay_height - margin),
    )


def should_snap_reference_point(
    snap_enabled: bool,
    modifiers: Qt.KeyboardModifier,
) -> bool:
    return snap_enabled and not bool(modifiers & Qt.KeyboardModifier.AltModifier)


def should_refresh_create_modifier_preview_for_key(key: int) -> bool:
    return key in {Qt.Key.Key_Control, Qt.Key.Key_Shift}


def point_shape_minimum_points(kind: str) -> int:
    return 2 if kind in {"polyline", "polyline_tube"} else 3


def should_clear_idle_selection_for_key(
    key: int,
    modifiers: Qt.KeyboardModifier | Qt.KeyboardModifiers,
) -> bool:
    return key == Qt.Key.Key_Escape and modifiers == Qt.KeyboardModifier.NoModifier


def should_frame_scene_for_key(
    key: int,
    modifiers: Qt.KeyboardModifier | Qt.KeyboardModifiers,
) -> bool:
    return key == Qt.Key.Key_F and modifiers == Qt.KeyboardModifier.NoModifier


class ViewportWidget(QOpenGLWidget):
    def __init__(self, parent: object | None = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(640, 480)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._fps_label = QLabel("FPS --", self)
        self._fps_label.setObjectName("viewportFpsCounter")
        self._fps_label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self._fps_label.setStyleSheet(
            "QLabel#viewportFpsCounter {"
            " color: #d7f3ff;"
            " background: rgba(6, 10, 16, 165);"
            " border: 1px solid rgba(120, 210, 255, 120);"
            " padding: 3px 6px;"
            " font: 11px monospace;"
            "}"
        )
        self._fps_label.move(10, 10)
        self._fps_label.adjustSize()
        self._fps_label.raise_()
        self._measure_label = QLabel("", self)
        self._measure_label.setObjectName("viewportMeasureReadout")
        self._measure_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self._measure_label.setStyleSheet(
            "QLabel#viewportMeasureReadout {"
            " color: #f2fbff;"
            " background: rgba(6, 10, 16, 180);"
            " border: 1px solid rgba(120, 210, 255, 120);"
            " padding: 4px 7px;"
            " font: 11px monospace;"
            "}"
        )
        self._measure_label.hide()
        self._viewport_error_label = QLabel("", self)
        self._viewport_error_label.setObjectName("viewportError")
        self._viewport_error_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents
        )
        self._viewport_error_label.setWordWrap(True)
        self._viewport_error_label.setStyleSheet(
            "QLabel#viewportError {"
            " color: #ffdede;"
            " background: rgba(58, 8, 14, 210);"
            " border: 1px solid rgba(255, 116, 116, 150);"
            " padding: 6px 8px;"
            " font: 12px sans-serif;"
            "}"
        )
        self._viewport_error_label.hide()
        self._fps_timer = QElapsedTimer()
        self._fps_timer.start()
        self._fps_frame_count = 0
        self.camera = OrbitCamera()
        self._reference_view = "3d"
        self._reference_plane = "xy"
        self._view_animation: tuple[float, float, float, float] | None = None
        self._view_animation_clock = QElapsedTimer()
        self._view_animation_timer = QTimer(self)
        self._view_animation_timer.setInterval(VIEW_ANIMATION_INTERVAL_MS)
        self._view_animation_timer.timeout.connect(self._advance_view_animation)
        self._view_buttons: dict[str, QPushButton] = {}
        self._view_panel = self._build_view_panel()
        self._command_panel = self._build_command_panel()
        self._position_overlays()
        self._context: moderngl.Context | None = None
        self._renderer: ViewportRenderer | None = None
        self._pending_scene: object | None = EMPTY_RENDER_SCENE_KEY
        self._pending_render_ir: RenderIR | None = None
        self._compiled_scene: object | None = None
        self._compiled_preview_scene: object | None = EMPTY_RENDER_SCENE_KEY
        self._scene_tree: SDFTree | None = None
        self._scene_hover_object_id = 0
        self._scene_selected_object_id = 0
        self._committed_move_object_id = 0
        self._committed_move_delta = (0.0, 0.0, 0.0)
        self._committed_create_preview: tuple[
            str,
            tuple[float, float, float],
            tuple[float, float, float],
            tuple[tuple[float, float, float], ...],
            bool,
            float,
            float,
        ] | None = None
        self._boolean_preview: tuple[int, int, int, float, bool] | None = None
        self._pending_lattice: object | None = None
        self._pending_lattice_upload: tuple[np.ndarray, np.ndarray, float] | None = None
        self._pending_lattice_stream_chunks: list[object] = []
        self._lattice_upload_started = False
        self._lattice_point_upload_cursor = 0
        self._lattice_square_upload_cursor = 0
        self._lattice_result: object | None = None
        self._lattice_filter_ids: set[int] | None = None
        self._lattice_filter_sdf: SDFNode | None = None
        self._lattice_filter_color_id: int | None = None
        self._lattice_filter_enabled = True
        self._grid_spacing = 0.1
        self._auto_grid_spacing = 0.1
        self._manual_grid_spacing: float | None = None
        self._snap_enabled = True
        self._last_mouse_position: QPoint | None = None
        self._scene_press_position: QPoint | None = None
        self._interaction_tool: tuple[str, object] | None = None
        self._boundary_pick_root: SDFNode | None = None
        self._boundary_selection_active = False
        self._boundary_hover_owner_id = 0
        self._boundary_hover_direction = -1
        self._boundary_hover_normal = (0.0, 0.0, 0.0)
        self._selected_boundary_regions: tuple[tuple[int, int], ...] = ()
        self._selected_boundary_normals: tuple[
            tuple[float, float, float], ...
        ] = ()
        self._boundary_press_position: QPoint | None = None
        self._boundary_camera_dragged = False
        self._tool_start_screen: QPoint | None = None
        self._tool_start_world: tuple[float, float, float] | None = None
        self._tool_current_world: tuple[float, float, float] | None = None
        self._tool_hover_world: tuple[float, float, float] | None = None
        self._point_shape_points: list[tuple[float, float, float]] = []
        self._move_preview_delta = (0.0, 0.0, 0.0)
        self._extrude_preview_height = 0.0
        self._extrude_drag_start_height = 0.0
        self._revolve_axis = "v"
        self._revolve_axis_angle_degrees = 90.0
        self._revolve_angle_degrees = 360.0
        self._rotation_drag_axis: str | None = None
        self._rotation_drag_start: tuple[float, float, float] | None = None
        self._rotation_drag_center: tuple[float, float, float] | None = None
        self._rotation_drag_move_delta = (0.0, 0.0, 0.0)
        self._rotation_preview_angle = 0.0
        self._rotation_preview_steps: list[
            tuple[
                str,
                float,
                tuple[float, float, float],
                tuple[float, float, float],
            ]
        ] = []
        self._dimension_input = ""
        self.mode = "sdf"
        self.grid_visible = True
        self.components_visible = False
        self.sdf_opacity = 0.4
        self.background_color = (
            36.0 / 255.0,
            31.0 / 255.0,
            50.0 / 255.0,
        )
        self.gizmo_visible = True
        signals.scene_changed.connect(self.set_scene)
        signals.mesh_ready.connect(self.set_lattice)
        signals.preview_ready.connect(self.set_lattice)
        signals.viewport_create_requested.connect(self.begin_create_tool)
        self.configure_default_grid()

    def _build_view_panel(self) -> QFrame:
        panel = QFrame(self)
        panel.setObjectName("viewPlanePanel")
        panel.setStyleSheet(
            "QFrame#viewPlanePanel {"
            " background: rgba(6, 10, 16, 165);"
            " border: 1px solid rgba(120, 210, 255, 120);"
            "}"
            "QPushButton {"
            " color: #d7f3ff;"
            " background: transparent;"
            " border: 0;"
            " padding: 3px 6px;"
            " font: 11px monospace;"
            "}"
            "QPushButton:checked {"
            " color: #ffffff;"
            " background: rgba(20, 145, 190, 160);"
            "}"
        )
        panel.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(1)
        for key, label in (
            ("3d", "3D"),
            ("xy", "XY"),
            ("xz", "XZ"),
            ("yz", "YZ"),
        ):
            button = QPushButton(label, panel)
            button.setCheckable(True)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.clicked.connect(
                lambda checked=False, value=key: self.set_reference_view(value)
            )
            layout.addWidget(button)
            self._view_buttons[key] = button
        panel.adjustSize()
        self._position_view_panel()
        self._sync_view_buttons("3d")
        return panel

    def _build_command_panel(self) -> QFrame:
        panel = QFrame(self)
        panel.setObjectName("viewportCommandPanel")
        panel.setStyleSheet(
            "QFrame#viewportCommandPanel {"
            " background: rgba(6, 10, 16, 185);"
            " border: 1px solid rgba(120, 210, 255, 130);"
            " border-radius: 4px;"
            "}"
            "QPushButton {"
            " color: #e9f8ff;"
            " background: rgba(20, 145, 190, 95);"
            " border: 1px solid rgba(160, 230, 255, 120);"
            " border-radius: 4px;"
            " padding: 5px 10px;"
            " font: 12px sans-serif;"
            "}"
            "QPushButton:hover {"
            " background: rgba(30, 170, 215, 145);"
            "}"
            "QPushButton:pressed {"
            " background: rgba(10, 100, 150, 170);"
            "}"
        )
        panel.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
        layout = QHBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        move_button = QPushButton("Move", panel)
        move_button.setObjectName("viewportMoveButton")
        move_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        move_button.setToolTip(
            "Select one Scene object, then drag it on the active reference plane"
        )
        move_button.clicked.connect(signals.viewport_move_tool_requested.emit)
        layout.addWidget(move_button)

        boundary_button = QPushButton("Boundary Region", panel)
        boundary_button.setObjectName("viewportBoundaryRegionButton")
        boundary_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        boundary_button.setToolTip(
            "Click the FluidDomain boundary to create a dimension-aware tag"
        )
        boundary_button.clicked.connect(
            signals.viewport_boundary_tool_requested.emit
        )
        layout.addWidget(boundary_button)

        panel.adjustSize()
        self._position_command_panel()
        return panel

    def _position_view_panel(self) -> None:
        if not hasattr(self, "_view_panel"):
            return
        self._view_panel.adjustSize()
        self._view_panel.move(10, max(10, self.height() - 34))
        self._view_panel.raise_()

    def _position_command_panel(self) -> None:
        if not hasattr(self, "_command_panel"):
            return
        self._command_panel.adjustSize()
        x, y = bottom_center_overlay_position(
            self.width(),
            self.height(),
            self._command_panel.width(),
            self._command_panel.height(),
            12,
        )
        self._command_panel.move(x, y)
        self._command_panel.raise_()

    def _position_measure_label(self) -> None:
        if not hasattr(self, "_measure_label"):
            return
        self._measure_label.adjustSize()
        self._measure_label.move(
            10,
            self._fps_label.y() + self._fps_label.height() + 6,
        )
        self._measure_label.raise_()

    def _position_overlays(self) -> None:
        self._position_view_panel()
        self._position_command_panel()
        self._position_measure_label()
        self._position_error_label()

    def _position_error_label(self) -> None:
        if not hasattr(self, "_viewport_error_label"):
            return
        width = min(520, max(240, self.width() - 20))
        self._viewport_error_label.setFixedWidth(width)
        self._viewport_error_label.adjustSize()
        self._viewport_error_label.move(
            10,
            max(
                self._measure_label.y() + self._measure_label.height() + 6,
                42,
            ),
        )
        self._viewport_error_label.raise_()

    def _show_viewport_error(self, message: str) -> None:
        self._viewport_error_label.setText(message)
        self._viewport_error_label.show()
        self._position_error_label()

    def _clear_viewport_error(self) -> None:
        self._viewport_error_label.hide()

    def _sync_view_buttons(self, active: str) -> None:
        for key, button in self._view_buttons.items():
            button.setChecked(key == active)

    def set_reference_view(self, view: str) -> None:
        if view == "3d":
            target_yaw, target_pitch = self.camera.standard_view_angles()
            self._reference_view = "3d"
            self._reference_plane = "xy"
            self._sync_view_buttons("3d")
        elif view in REFERENCE_PLANE_IDS:
            target_yaw, target_pitch = self.camera.plane_view_angles(view)
            self._reference_view = view
            self._reference_plane = view
            self._sync_view_buttons(view)
        else:
            raise ValueError(f"unknown reference view: {view}")
        self._animate_view_to(target_yaw, target_pitch)
        self.update()

    def _animate_view_to(self, target_yaw: float, target_pitch: float) -> None:
        start_yaw = self.camera.yaw_degrees
        start_pitch = self.camera.pitch_degrees
        yaw_delta = ((target_yaw - start_yaw + 180.0) % 360.0) - 180.0
        pitch_delta = target_pitch - start_pitch
        if max(abs(yaw_delta), abs(pitch_delta)) <= 1e-6:
            self._stop_view_animation()
            self.camera.yaw_degrees = target_yaw
            self.camera.pitch_degrees = target_pitch
            return
        self._view_animation = (start_yaw, start_pitch, yaw_delta, pitch_delta)
        self._view_animation_clock.restart()
        self._view_animation_timer.start()

    def _advance_view_animation(self) -> None:
        if self._view_animation is None:
            self._view_animation_timer.stop()
            return
        elapsed = self._view_animation_clock.elapsed()
        progress = min(1.0, elapsed / float(VIEW_ANIMATION_DURATION_MS))
        eased = progress * progress * (3.0 - 2.0 * progress)
        start_yaw, start_pitch, yaw_delta, pitch_delta = self._view_animation
        self.camera.yaw_degrees = start_yaw + yaw_delta * eased
        self.camera.pitch_degrees = start_pitch + pitch_delta * eased
        if progress >= 1.0:
            self._stop_view_animation()
        self.update()

    def _stop_view_animation(self) -> None:
        self._view_animation = None
        self._view_animation_timer.stop()

    def _leave_planar_view_for_orbit(self) -> None:
        if self._reference_view == "3d":
            return
        self._reference_view = "3d"
        self._reference_plane = "xy"
        self._sync_view_buttons("3d")

    def resizeEvent(self, event: object) -> None:
        self._position_overlays()
        super().resizeEvent(event)

    def begin_create_tool(self, kind: str) -> None:
        self._clear_scene_hover()
        self._interaction_tool = ("create", kind)
        self._tool_start_screen = None
        self._tool_start_world = None
        self._tool_current_world = None
        self._dimension_input = ""
        self._tool_hover_world = None
        self._point_shape_points = []
        self._move_preview_delta = (0.0, 0.0, 0.0)
        ViewportWidget._clear_rotation_drag(self)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        if kind in POINT_CREATE_KINDS:
            if kind == "bezier_curve":
                message = (
                    "Click start, control, and end points to create Bezier Curve. "
                    "Backspace removes the last point, Esc cancels."
                )
            else:
                message = (
                    f"Click grid points to draw {kind}. Enter creates, "
                    "Backspace removes the last point, Esc cancels."
                )
        else:
            message = f"Drag on the reference grid to create {kind}. Esc cancels."
        signals.log_message.emit("info", message)
        self._update_measurement_readout()
        self.update()

    def begin_move_tool(self, handle: int) -> None:
        self._clear_scene_hover()
        self._interaction_tool = ("move", handle)
        self._move_preview_delta = (0.0, 0.0, 0.0)
        self._tool_start_screen = None
        self._tool_start_world = None
        self._tool_current_world = None
        self._tool_hover_world = None
        self._point_shape_points = []
        self._dimension_input = ""
        ViewportWidget._clear_rotation_drag(self)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setFocus()
        signals.log_message.emit(
            "info",
            "Move preview active. Drag or use WASD/QE, Enter applies, Esc cancels.",
        )
        self._update_measurement_readout()
        self.update()

    def begin_extrude_tool(self, handle: int, node: PlacedSDF2D) -> None:
        self._clear_scene_hover()
        self._interaction_tool = ("extrude", handle)
        self._scene_selected_object_id = node.object_id
        self._tool_start_screen = None
        self._tool_start_world = None
        self._tool_current_world = None
        self._tool_hover_world = None
        self._point_shape_points = []
        self._move_preview_delta = (0.0, 0.0, 0.0)
        self._extrude_preview_height = 0.0
        self._extrude_drag_start_height = 0.0
        ViewportWidget._clear_rotation_drag(self)
        self._dimension_input = ""
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setFocus()
        signals.log_message.emit(
            "info",
            "Extrude preview active. Drag the normal handle or use Q/E, "
            "Enter applies, Esc cancels.",
        )
        self._update_measurement_readout()
        self.update()

    def begin_revolve_tool(self, handle: int, node: PlacedSDF2D) -> None:
        self._clear_scene_hover()
        self._interaction_tool = ("revolve", handle)
        self._scene_selected_object_id = node.object_id
        self._tool_start_screen = None
        self._tool_start_world = None
        self._tool_current_world = None
        self._tool_hover_world = None
        self._point_shape_points = []
        self._move_preview_delta = (0.0, 0.0, 0.0)
        self._revolve_axis = "v"
        self._revolve_axis_angle_degrees = 90.0
        self._revolve_angle_degrees = 360.0
        ViewportWidget._clear_rotation_drag(self)
        self._dimension_input = ""
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        signals.log_message.emit(
            "info",
            "Revolve preview active. A/D rotates the axis, Q/E changes angle, "
            "U/V snaps the axis, Enter applies, Esc cancels.",
        )
        self._update_measurement_readout()
        self.update()

    def begin_boundary_region_tool(self, root: SDFNode) -> None:
        self._clear_scene_hover()
        self._boundary_pick_root = root
        self._boundary_selection_active = True
        self._boundary_hover_owner_id = 0
        self._boundary_hover_direction = -1
        self._boundary_hover_normal = (0.0, 0.0, 0.0)
        self._boundary_press_position = None
        self._boundary_camera_dragged = False
        self._tool_start_screen = None
        self._tool_start_world = None
        self._tool_current_world = None
        self._tool_hover_world = None
        self._point_shape_points = []
        self._move_preview_delta = (0.0, 0.0, 0.0)
        self._extrude_preview_height = 0.0
        self._extrude_drag_start_height = 0.0
        ViewportWidget._clear_rotation_drag(self)
        self._dimension_input = ""
        self._interaction_tool = ("boundary_region", root.object_id)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        signals.log_message.emit(
            "info",
            "Move over the fluid boundary and click to create a boundary tag.",
        )

    def cancel_interaction_tool(self) -> None:
        self._interaction_tool = None
        self._tool_start_screen = None
        self._tool_start_world = None
        self._tool_current_world = None
        self._tool_hover_world = None
        self._point_shape_points = []
        self._move_preview_delta = (0.0, 0.0, 0.0)
        self._revolve_axis = "v"
        self._revolve_axis_angle_degrees = 90.0
        self._revolve_angle_degrees = 360.0
        ViewportWidget._clear_rotation_drag(self)
        self._dimension_input = ""
        self.unsetCursor()
        self._boundary_selection_active = False
        self._boundary_hover_owner_id = 0
        self._boundary_hover_direction = -1
        self._boundary_hover_normal = (0.0, 0.0, 0.0)
        signals.viewport_boundary_hovered.emit(None)
        self._update_measurement_readout()
        self.update()

    def cancel_active_interaction_tool(self) -> bool:
        if self._interaction_tool is None:
            return False
        self.cancel_interaction_tool()
        signals.log_message.emit("info", "Viewport tool cancelled.")
        return True

    def set_boundary_hover(
        self,
        owner_object_id: int,
        outside_direction: int | None,
        direction_normal: tuple[float, float, float] | None = None,
    ) -> None:
        self._boundary_hover_owner_id = owner_object_id
        self._boundary_hover_direction = (
            outside_direction if outside_direction is not None else -1
        )
        self._boundary_hover_normal = (
            direction_normal if direction_normal is not None else (0.0, 0.0, 0.0)
        )
        self.update()

    def set_boundary_region_selection(
        self,
        regions: list[BoundaryRegion],
    ) -> None:
        selectors: list[tuple[int, int]] = []
        normals: list[tuple[float, float, float]] = []
        for region in regions:
            selectors.append(
                (
                    region.owner_object_id,
                    1 if region.outside_direction is None else 0,
                )
            )
            normals.append((0.0, 0.0, 0.0))
        self.set_boundary_region_selection_entries(
            tuple(selectors),
            tuple(normals),
        )

    def set_boundary_region_selection_entries(
        self,
        selectors: tuple[tuple[int, int], ...],
        normals: tuple[tuple[float, float, float], ...],
    ) -> None:
        if len(selectors) > MAX_SELECTED_BOUNDARY_REGIONS:
            signals.log_message.emit(
                "warning",
                "BoundaryRegion viewport highlighting is limited to "
                f"{MAX_SELECTED_BOUNDARY_REGIONS} selected regions.",
            )
        self._selected_boundary_regions = selectors[:MAX_SELECTED_BOUNDARY_REGIONS]
        self._selected_boundary_normals = normals[:MAX_SELECTED_BOUNDARY_REGIONS]
        self.update()

    def _pick_boundary(
        self, position: object
    ) -> tuple[np.ndarray, int, np.ndarray] | None:
        if self._boundary_pick_root is None:
            return None
        origin, direction = self.camera.screen_ray(
            position.x(),
            position.y(),
            self.width(),
            self.height(),
        )
        return pick_boundary_owner(
            self._boundary_pick_root,
            origin,
            direction,
        )

    def _pick_scene_object(self, position: object) -> int:
        if self._scene_tree is None or self.mode != "sdf":
            return 0
        origin, direction = self.camera.screen_ray(
            position.x(),
            position.y(),
            self.width(),
            self.height(),
        )
        point = pick_sdf_surface(self._scene_tree.root, origin, direction)
        if point is None:
            return 0
        coordinates = tuple(
            np.asarray([point[index]], dtype=np.float64) for index in range(3)
        )
        candidates = tuple(
            node for node in self._scene_tree.components if node.dimension == 3
        ) or (self._scene_tree.root,)
        return min(
            candidates,
            key=lambda node: abs(float(node.to_numpy(*coordinates)[0])),
        ).object_id

    def _clear_scene_hover(self) -> None:
        if self._scene_hover_object_id != 0:
            self._scene_hover_object_id = 0
            self.update()

    def set_scene_selection(self, node: SDFNode | None) -> None:
        self._scene_selected_object_id = (
            node.object_id if node is not None else 0
        )
        self.update()

    def set_scene(self, tree: SDFTree | None) -> None:
        render_ir = build_render_ir(tree) if tree is not None else None
        self.set_scene_artifact(
            tree,
            render_ir if render_ir is not None and render_ir.supported else None,
        )

    def set_scene_artifact(
        self,
        tree: SDFTree | None,
        render_ir: RenderIR | None = None,
    ) -> None:
        if tree is not None and render_ir is None:
            logger.warning("ignored non-RenderIR artifact for a non-empty scene")
            signals.log_message.emit(
                "warning",
                "Ignored a non-RenderIR render artifact and kept the previous viewport.",
            )
            return
        self._scene_tree = tree
        self._scene_hover_object_id = 0
        self._update_measurement_readout()
        self._pending_scene = render_scene_key(render_ir)
        self._pending_render_ir = render_ir
        self.update()

    def set_lattice(self, result: object) -> None:
        self._lattice_result = result
        self._queue_lattice_upload()
        self.mode = "lattice"
        self.update()

    def append_lattice_preview_chunk(self, chunk: object) -> None:
        self._pending_lattice_stream_chunks.append(chunk)
        self.mode = "lattice"
        self.update()

    def set_lattice_filter(
        self,
        object_ids: set[int] | None,
        geometry: SDFNode | None = None,
    ) -> None:
        self._lattice_filter_ids = set(object_ids) if object_ids else None
        self._lattice_filter_sdf = geometry
        self._lattice_filter_color_id = (
            geometry.object_id if geometry is not None else None
        )
        self._queue_lattice_upload()
        self.update()

    def set_lattice_filter_enabled(self, enabled: bool) -> None:
        self._lattice_filter_enabled = enabled
        self._queue_lattice_upload()
        self.update()

    def _queue_lattice_upload(self) -> None:
        if self._lattice_result is None:
            return
        result = self._lattice_result
        positions = result.preview_positions
        node_types = result.preview_node_types
        boundary_faces = result.preview_boundary_faces
        primary_ids = result.preview_primary_tag_ids
        source_ids = result.preview_source_object_ids
        tag_ids = result.preview_tag_ids
        tag_axis_u = result.preview_tag_axis_u
        tag_axis_v = result.preview_tag_axis_v
        mask = None
        if self._lattice_filter_enabled and self._lattice_filter_sdf is not None:
            geometry_mask = (
                self._lattice_filter_sdf.to_numpy(
                    positions[:, 0].astype(np.float64),
                    positions[:, 1].astype(np.float64),
                    positions[:, 2].astype(np.float64),
                )
                <= 0.0
            )
            attribution_mask = np.fromiter(
                (
                    int(source_id) in (self._lattice_filter_ids or ())
                    or bool(
                        (self._lattice_filter_ids or set()).intersection(items)
                    )
                    for source_id, items in zip(source_ids, tag_ids, strict=True)
                ),
                dtype=np.bool_,
                count=len(tag_ids),
            )
            mask = geometry_mask | attribution_mask
        elif self._lattice_filter_enabled and self._lattice_filter_ids:
            mask = np.fromiter(
                (
                    int(source_id) in self._lattice_filter_ids
                    or bool(self._lattice_filter_ids.intersection(items))
                    for source_id, items in zip(source_ids, tag_ids, strict=True)
                ),
                dtype=np.bool_,
                count=len(tag_ids),
            )
        if mask is not None:
            positions = positions[mask]
            node_types = node_types[mask]
            boundary_faces = boundary_faces[mask]
            primary_ids = primary_ids[mask]
            source_ids = source_ids[mask]
            tag_axis_u = tag_axis_u[mask]
            tag_axis_v = tag_axis_v[mask]
        if self._lattice_filter_color_id is not None and mask is not None:
            source_ids = source_ids.copy()
            source_ids[node_types != np.uint8(1)] = self._lattice_filter_color_id
            primary_ids = np.zeros(primary_ids.shape, dtype=np.uint16)
        self._pending_lattice = (
            positions,
            node_types,
            boundary_faces,
            source_ids,
            primary_ids,
            tag_axis_u,
            tag_axis_v,
            result.preview_cell_size,
            result.dimension,
            getattr(result, "preview_axis_i", (1.0, 0.0, 0.0)),
            getattr(result, "preview_axis_j", (0.0, 1.0, 0.0)),
        )
        point_vertices, square_instances = SDFRenderer.prepare_lattice_upload(
            positions,
            node_types,
            boundary_faces,
            source_ids,
            primary_ids,
            result.preview_cell_size,
            dimension=result.dimension,
            axis_i=getattr(result, "preview_axis_i", (1.0, 0.0, 0.0)),
            axis_j=getattr(result, "preview_axis_j", (0.0, 1.0, 0.0)),
        )
        self._pending_lattice_upload = (
            point_vertices,
            square_instances,
            result.preview_cell_size,
        )
        self._lattice_upload_started = False
        self._lattice_point_upload_cursor = 0
        self._lattice_square_upload_cursor = 0
        self.update()

    def set_mode(self, mode: str) -> None:
        if mode not in {"sdf", "lattice"}:
            raise ValueError(mode)
        self.mode = mode
        if mode != "sdf":
            self._clear_scene_hover()
        self.update()

    def set_grid_visible(self, visible: bool) -> None:
        self.grid_visible = visible
        self.update()

    @property
    def snap_enabled(self) -> bool:
        return self._snap_enabled

    def set_snap_enabled(self, enabled: bool) -> None:
        self._snap_enabled = bool(enabled)
        self._update_measurement_readout()
        self.update()

    @property
    def grid_spacing(self) -> float:
        return self._grid_spacing

    @property
    def reference_plane_label(self) -> str:
        return reference_plane_label(self._reference_plane)

    def set_grid_spacing(self, spacing: float) -> None:
        self._manual_grid_spacing = max(1e-6, float(spacing))
        self._grid_spacing = self._manual_grid_spacing
        self._update_measurement_readout()
        self.update()

    def reset_grid_spacing(self) -> None:
        self._manual_grid_spacing = None
        self._grid_spacing = self._auto_grid_spacing
        self._update_measurement_readout()
        self.update()

    def set_components_visible(self, visible: bool) -> None:
        self.components_visible = visible
        self.update()

    def set_sdf_opacity(self, opacity: float) -> None:
        self.sdf_opacity = min(1.0, max(0.05, float(opacity)))
        self.update()

    def set_background_color(self, color: tuple[float, float, float]) -> None:
        self.background_color = tuple(
            min(1.0, max(0.0, float(component)))
            for component in color
        )
        self.update()

    def paste_offset(self) -> tuple[float, float, float]:
        step = max(self._grid_spacing, 0.05)
        if self._reference_plane == "xz":
            return (step, 0.0, step)
        if self._reference_plane == "yz":
            return (0.0, step, step)
        return (step, step, 0.0)

    def _clear_rotation_drag(self) -> None:
        self._rotation_drag_axis = None
        self._rotation_drag_start = None
        self._rotation_drag_center = None
        self._rotation_drag_move_delta = (0.0, 0.0, 0.0)
        self._rotation_preview_angle = 0.0
        self._rotation_preview_steps = []

    def _clear_active_rotation_drag(self) -> None:
        self._rotation_drag_axis = None
        self._rotation_drag_start = None
        self._rotation_drag_center = None
        self._rotation_drag_move_delta = (0.0, 0.0, 0.0)
        self._rotation_preview_angle = 0.0

    @staticmethod
    def _axis_vector(axis: str) -> np.ndarray:
        vectors = {
            "x": (1.0, 0.0, 0.0),
            "y": (0.0, 1.0, 0.0),
            "z": (0.0, 0.0, 1.0),
        }
        return np.asarray(vectors[axis], dtype=np.float64)

    @staticmethod
    def _box_center_and_radius(
        box: BoundingBox3D,
    ) -> tuple[tuple[float, float, float], float]:
        center = (
            (box.x_min + box.x_max) * 0.5,
            (box.y_min + box.y_max) * 0.5,
            (box.z_min + box.z_max) * 0.5,
        )
        size = (
            box.x_max - box.x_min,
            box.y_max - box.y_min,
            box.z_max - box.z_min,
        )
        radius = max(
            ROTATION_GIZMO_MIN_RADIUS,
            0.62 * float(sum(component * component for component in size) ** 0.5),
        )
        return center, radius

    def _rotation_gizmo_state(
        self,
    ) -> tuple[bool, tuple[float, float, float], float]:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] != "move"
            or self._scene_tree is None
            or self._scene_selected_object_id == 0
        ):
            return False, (0.0, 0.0, 0.0), 1.0
        try:
            node = next(
                node
                for node in self._scene_tree.nodes
                if node.object_id == self._scene_selected_object_id
            )
            center, radius = self._box_center_and_radius(node.bounding_box())
            delta = self._preview_move_delta()
            center = tuple(center[index] + delta[index] for index in range(3))
        except (StopIteration, NotImplementedError, ValueError):
            return False, (0.0, 0.0, 0.0), 1.0
        return True, center, radius

    def _extrude_gizmo_state(
        self,
    ) -> tuple[bool, tuple[float, float, float], tuple[float, float, float]]:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] != "extrude"
            or self._scene_tree is None
            or self._scene_selected_object_id == 0
        ):
            return False, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
        try:
            node = next(
                item
                for item in self._scene_tree.nodes
                if item.object_id == self._scene_selected_object_id
            )
        except StopIteration:
            return False, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
        if not isinstance(node, PlacedSDF2D):
            return False, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)
        center, _radius = self._box_center_and_radius(node.bounding_box())
        return True, center, node.normal

    def _revolve_section(self) -> PlacedSDF2D | None:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] != "revolve"
            or self._scene_tree is None
            or self._scene_selected_object_id == 0
        ):
            return None
        try:
            node = next(
                item
                for item in self._scene_tree.nodes
                if item.object_id == self._scene_selected_object_id
            )
        except StopIteration:
            return None
        return node if isinstance(node, PlacedSDF2D) else None

    def _revolve_axis_frame(
        self,
    ) -> tuple[
        bool,
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ]:
        node = self._revolve_section()
        if node is None or node.profile is None:
            return (
                False,
                (0.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            )
        angle = radians(self._revolve_axis_angle_degrees)
        axis_u = np.asarray(node.axis_u, dtype=np.float64)
        axis_v = np.asarray(node.axis_v, dtype=np.float64)
        axis = np.cos(angle) * axis_u + np.sin(angle) * axis_v
        axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
        u_min, u_max, v_min, v_max = node.profile.bounds()
        profile_center = (
            np.asarray(node.origin, dtype=np.float64)
            + 0.5 * (u_min + u_max) * axis_u
            + 0.5 * (v_min + v_max) * axis_v
        )
        origin = np.asarray(node.origin, dtype=np.float64)
        center_delta = profile_center - origin
        radial = center_delta - axis * float(np.dot(center_delta, axis))
        if float(np.linalg.norm(radial)) <= 1.0e-9:
            radial = -np.sin(angle) * axis_u + np.cos(angle) * axis_v
        radial /= max(float(np.linalg.norm(radial)), 1.0e-12)
        return (
            True,
            tuple(float(value) for value in origin),
            tuple(float(value) for value in axis),
            tuple(float(value) for value in radial),
        )

    def _revolve_axis_segment(
        self,
    ) -> tuple[bool, tuple[float, float, float], tuple[float, float, float]]:
        node = self._revolve_section()
        if node is None or node.profile is None:
            return False, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        u_min, u_max, v_min, v_max = node.profile.bounds()
        active, origin_tuple, axis_tuple, _radial = self._revolve_axis_frame()
        if not active:
            return False, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        origin = np.asarray(origin_tuple, dtype=np.float64)
        axis = np.asarray(axis_tuple, dtype=np.float64)
        axis_u = np.asarray(node.axis_u, dtype=np.float64)
        axis_v = np.asarray(node.axis_v, dtype=np.float64)
        corners = np.asarray(
            [
                origin + u * axis_u + v * axis_v
                for u in (u_min, u_max)
                for v in (v_min, v_max)
            ],
            dtype=np.float64,
        )
        coordinates = (corners - origin) @ axis
        minimum = float(coordinates.min())
        maximum = float(coordinates.max())
        length = max(abs(maximum - minimum), self._grid_spacing)
        padding = max(0.15 * length, self._grid_spacing)
        start = origin + (minimum - padding) * axis
        end = origin + (maximum + padding) * axis
        return (
            True,
            tuple(float(value) for value in start),
            tuple(float(value) for value in end),
        )

    def _project_world_to_screen(
        self,
        point: tuple[float, float, float] | np.ndarray,
    ) -> tuple[float, float] | None:
        width = max(1.0, float(self.width()))
        height = max(1.0, float(self.height()))
        projection = self.camera.view_projection(width / height).astype(np.float64)
        clip = projection @ np.append(np.asarray(point, dtype=np.float64), 1.0)
        if abs(float(clip[3])) <= 1.0e-9:
            return None
        ndc = clip[:3] / clip[3]
        if ndc[2] < -1.0 or ndc[2] > 1.0:
            return None
        return (
            float((ndc[0] + 1.0) * 0.5 * width),
            float((1.0 - ndc[1]) * 0.5 * height),
        )

    def _pick_rotation_gizmo_axis(self, x: float, y: float) -> str | None:
        visible, center, radius = self._rotation_gizmo_state()
        if not visible:
            return None
        vertices = SDFRenderer.build_rotation_gizmo_vertices(center, radius)
        best_axis: str | None = None
        best_distance = ROTATION_GIZMO_PICK_TOLERANCE_PX
        for axis_index, axis in enumerate(("x", "y", "z")):
            start = axis_index * ROTATION_GIZMO_SEGMENTS * 2
            stop = start + ROTATION_GIZMO_SEGMENTS * 2
            for point in vertices[start:stop, :3]:
                projected = self._project_world_to_screen(point)
                if projected is None:
                    continue
                distance = ((projected[0] - x) ** 2 + (projected[1] - y) ** 2) ** 0.5
                if distance < best_distance:
                    best_distance = distance
                    best_axis = axis
        return best_axis

    def _screen_to_rotation_plane(
        self,
        axis: str,
        x: float,
        y: float,
        center: tuple[float, float, float],
    ) -> tuple[float, float, float] | None:
        origin, direction = self.camera.screen_ray(x, y, self.width(), self.height())
        normal = self._axis_vector(axis)
        denominator = float(np.dot(direction, normal))
        if abs(denominator) <= 1.0e-9:
            return None
        travel = float(np.dot(np.asarray(center, dtype=np.float64) - origin, normal))
        travel /= denominator
        if travel <= 0.0:
            return None
        point = origin + direction * travel
        return tuple(float(value) for value in point)

    def _rotation_drag_angle(
        self,
        current: tuple[float, float, float],
    ) -> float:
        assert self._rotation_drag_axis is not None
        assert self._rotation_drag_center is not None
        assert self._rotation_drag_start is not None
        axis = self._axis_vector(self._rotation_drag_axis)
        center = np.asarray(self._rotation_drag_center, dtype=np.float64)
        start = np.asarray(self._rotation_drag_start, dtype=np.float64) - center
        end = np.asarray(current, dtype=np.float64) - center
        start -= axis * float(np.dot(start, axis))
        end -= axis * float(np.dot(end, axis))
        if np.linalg.norm(start) <= 1.0e-9 or np.linalg.norm(end) <= 1.0e-9:
            return 0.0
        cross = np.cross(start, end)
        return degrees(atan2(float(np.dot(axis, cross)), float(np.dot(start, end))))

    def apply_committed_move_preview(
        self,
        object_id: int,
        delta: tuple[float, float, float],
    ) -> None:
        current = (
            self._committed_move_delta
            if self._committed_move_object_id == object_id
            else (0.0, 0.0, 0.0)
        )
        updated = tuple(current[index] + delta[index] for index in range(3))
        if max(abs(component) for component in updated) <= 1e-12:
            self._committed_move_object_id = 0
            self._committed_move_delta = (0.0, 0.0, 0.0)
        else:
            self._committed_move_object_id = object_id
            self._committed_move_delta = updated
        self._update_measurement_readout()
        self.update()

    def _clear_committed_move_preview(self) -> None:
        if self._committed_move_object_id == 0:
            return
        self._committed_move_object_id = 0
        self._committed_move_delta = (0.0, 0.0, 0.0)
        self._update_measurement_readout()

    def apply_committed_create_preview(
        self,
        kind: str,
        start: tuple[float, float, float],
        current: tuple[float, float, float],
        *,
        points: tuple[tuple[float, float, float], ...] = (),
        polygon_closed: bool = False,
        secondary_radius: float = -1.0,
        torus_minor_radius: float = -1.0,
    ) -> None:
        self._committed_create_preview = (
            kind,
            start,
            current,
            points[:32],
            polygon_closed,
            float(secondary_radius),
            float(torus_minor_radius),
        )
        self.update()

    def _clear_committed_create_preview(self) -> None:
        if self._committed_create_preview is None:
            return
        self._committed_create_preview = None
        self.update()

    def set_boolean_preview(
        self,
        operation: str,
        first_object_id: int,
        second_object_id: int,
        *,
        smoothing: float = 0.1,
    ) -> None:
        opcode = BOOLEAN_PREVIEW_OPERATIONS.get(operation, 0)
        if opcode == 0 or first_object_id <= 0 or second_object_id <= 0:
            self.clear_boolean_preview()
            return
        self._boolean_preview = (
            opcode,
            int(first_object_id),
            int(second_object_id),
            float(smoothing),
            False,
        )
        self.update()

    def apply_committed_boolean_preview(
        self,
        operation: str,
        first_object_id: int,
        second_object_id: int,
        *,
        smoothing: float = 0.1,
    ) -> None:
        opcode = BOOLEAN_PREVIEW_OPERATIONS.get(operation, 0)
        if opcode == 0 or first_object_id <= 0 or second_object_id <= 0:
            return
        self._boolean_preview = (
            opcode,
            int(first_object_id),
            int(second_object_id),
            float(smoothing),
            True,
        )
        self.update()

    def clear_boolean_preview(self, *, include_committed: bool = False) -> None:
        if self._boolean_preview is None:
            return
        if self._boolean_preview[4] and not include_committed:
            return
        self._boolean_preview = None
        self.update()

    def _clear_committed_boolean_preview(self) -> None:
        self.clear_boolean_preview(include_committed=True)

    def _boolean_preview_values(self) -> tuple[int, int, int, float]:
        if self._boolean_preview is None:
            return (0, 0, 0, 0.1)
        operation, first, second, smoothing, _committed = self._boolean_preview
        return (operation, first, second, smoothing)

    def _committed_boolean_preview_object_ids(self) -> tuple[int, int] | None:
        if self._boolean_preview is None or not self._boolean_preview[4]:
            return None
        _operation, first, second, _smoothing, _committed = self._boolean_preview
        return (first, second)

    def _accept_committed_boolean_pending_scene(self) -> bool:
        if (
            self._renderer is None
            or self._pending_scene is None
            or self._pending_render_ir is None
        ):
            return False
        object_ids = self._committed_boolean_preview_object_ids()
        if object_ids is None:
            return False
        if not self._renderer.update_render_ir_object_parameters(
            self._pending_render_ir,
            object_ids,
        ):
            return False
        stats = self._renderer.last_scene_update_stats()
        if stats is not None:
            logger.info(_format_scene_update_stats(stats))
            signals.log_message.emit("info", _format_scene_update_stats(stats))
        self._compiled_scene = self._pending_scene
        self._pending_scene = None
        self._pending_render_ir = None
        self._clear_viewport_error()
        return True

    def has_scene_object_id(self, object_id: int) -> bool:
        return (
            self._scene_tree is not None
            and any(node.object_id == object_id for node in self._scene_tree.nodes)
        )

    def can_defer_committed_move(self, object_id: int) -> bool:
        return (
            self._committed_move_object_id == 0
            or self._committed_move_object_id == object_id
        )

    def set_gizmo_visible(self, visible: bool) -> None:
        self.gizmo_visible = visible
        self.update()

    def nudge_move_preview(
        self,
        delta: tuple[float, float, float],
    ) -> None:
        self._move_preview_delta = tuple(
            self._move_preview_delta[index] + delta[index]
            for index in range(3)
        )
        self._update_measurement_readout()
        self.update()

    def nudge_extrude_preview(self, delta: float) -> None:
        self._extrude_preview_height = snap_scalar(
            self._extrude_preview_height + delta,
            self._grid_spacing,
            self._snap_enabled,
            QApplication.keyboardModifiers(),
        )
        self._update_measurement_readout()
        self.update()

    def _keyboard_move_delta(
        self,
        key: int,
        modifiers: Qt.KeyboardModifier | Qt.KeyboardModifiers,
    ) -> tuple[float, float, float] | None:
        return keyboard_move_delta(key, self._grid_spacing, modifiers)

    def _commit_move_preview(self) -> None:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] != "move"
        ):
            return
        _action, value = self._interaction_tool
        delta = self._preview_move_delta()
        rotations = ViewportWidget._rotation_preview_commands(self)
        if rotations:
            self.cancel_interaction_tool()
            has_move = max(abs(component) for component in delta) > 1.0e-12
            if has_move or len(rotations) > 1:
                signals.viewport_transform_requested.emit(
                    int(value),
                    delta,
                    rotations,
                )
                return
            axis, angle, center = rotations[0]
            signals.viewport_rotate_requested.emit(int(value), axis, angle, center)
            return
        self.cancel_interaction_tool()
        if max(abs(component) for component in delta) <= 1e-12:
            signals.log_message.emit("info", "Move preview had no displacement.")
            return
        signals.viewport_move_requested.emit(int(value), delta)

    def _commit_extrude_preview(self) -> None:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] != "extrude"
        ):
            return
        _action, value = self._interaction_tool
        height = float(self._extrude_preview_height)
        if abs(height) <= 1.0e-9:
            signals.log_message.emit("warning", "Extrude length must be nonzero.")
            return
        self.cancel_interaction_tool()
        signals.viewport_extrude_requested.emit(int(value), height)

    def _commit_revolve_preview(self) -> None:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] != "revolve"
        ):
            return
        _action, value = self._interaction_tool
        active, origin, axis, radial = self._revolve_axis_frame()
        if not active:
            signals.log_message.emit("warning", "Revolve axis is not available.")
            return
        angle = float(self._revolve_angle_degrees)
        self.cancel_interaction_tool()
        signals.viewport_revolve_requested.emit(int(value), origin, axis, radial, angle)

    def _apply_typed_move_preview(self) -> bool:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] != "move"
            or not self._dimension_input
        ):
            return False
        try:
            self._move_preview_delta = apply_typed_move_delta(
                self._preview_move_delta(),
                self._reference_plane,
                parse_displacement_entry(self._dimension_input),
            )
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return False
        self._tool_start_screen = None
        self._tool_start_world = None
        self._tool_current_world = None
        self._update_measurement_readout()
        self.update()
        return True

    def _commit_create_preview(self, centered: bool | None = None) -> None:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] != "create"
            or self._tool_start_world is None
        ):
            return
        _action, kind = self._interaction_tool
        if str(kind) in POINT_CREATE_KINDS:
            self._commit_point_shape_preview()
            return
        try:
            dimensions = (
                parse_dimension_entry(self._dimension_input)
                if self._dimension_input
                else None
            )
            start, end = self._create_effective_points(centered=centered)
            parameters = create_typed_parameters(
                str(kind),
                dimensions,
            )
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        self.apply_committed_create_preview(
            str(kind),
            start,
            end,
            secondary_radius=create_preview_secondary_radius(str(kind), dimensions),
            torus_minor_radius=create_preview_torus_minor_radius(str(kind), dimensions),
        )
        self._reset_create_preview(str(kind))
        signals.viewport_shape_drawn.emit(str(kind), start, end, parameters)

    def _reset_create_preview(self, kind: str) -> None:
        self._interaction_tool = ("create", kind)
        self._tool_start_screen = None
        self._tool_start_world = None
        self._tool_current_world = None
        self._tool_hover_world = None
        self._point_shape_points = []
        self._dimension_input = ""
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._update_measurement_readout()
        self.update()

    def _point_shape_kind(self) -> str | None:
        if self._interaction_tool is None or self._interaction_tool[0] != "create":
            return None
        kind = str(self._interaction_tool[1])
        return kind if kind in POINT_CREATE_KINDS else None

    def _point_shape_preview_points(
        self,
    ) -> tuple[tuple[float, float, float], ...]:
        points = list(self._point_shape_points)
        if self._tool_hover_world is not None:
            if not points or points[-1] != self._tool_hover_world:
                points.append(self._tool_hover_world)
        return tuple(points[:32])

    @staticmethod
    def _preview_render_ir_from_document(document: SceneDocument) -> RenderIR | None:
        tree = document.visual_tree()
        render_ir = build_render_ir(tree)
        if render_ir is None or not render_ir.supported:
            return None
        return render_ir

    def _scene_node_by_object_id(self, object_id: int) -> SDFNode | None:
        if self._scene_tree is None or object_id <= 0:
            return None
        return next(
            (
                node
                for node in self._scene_tree.nodes
                if int(node.object_id) == int(object_id)
            ),
            None,
        )

    def _preview_document_for_node(self, node: SDFNode) -> tuple[SceneDocument, int]:
        clone = deepcopy(node)
        document = SceneDocument(objects=[clone])
        return document, document.handle_for(clone)

    def _active_create_preview_render_ir(self) -> RenderIR | None:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] != "create"
        ):
            return None
        kind = str(self._interaction_tool[1])
        if kind in POINT_CREATE_KINDS:
            points = self._point_shape_preview_points()
            minimum = point_shape_minimum_points(kind)
            if len(points) < minimum:
                return None
            if kind == "bezier_curve" and len(points) != 3:
                return None
            if kind in {"bezier_polycurve", "bezier_tube", "bezier_surface"} and len(
                points
            ) % 2 == 0:
                return None
            try:
                document = SceneDocument()
                document.add_point_shape_from_world_points(
                    kind,
                    points,
                    self._reference_plane,
                )
            except ValueError:
                return None
            return self._preview_render_ir_from_document(document)
        if self._tool_start_world is None or self._tool_current_world is None:
            return None
        try:
            dimensions = (
                parse_dimension_entry(self._dimension_input)
                if self._dimension_input
                else None
            )
            start, end = self._create_effective_points()
            parameters = create_typed_parameters(kind, dimensions)
            document = SceneDocument()
            document.add_primitive_from_drag(
                kind,
                start,
                end,
                parameters=parameters,
            )
        except ValueError:
            return None
        return self._preview_render_ir_from_document(document)

    def _committed_create_preview_render_ir(self) -> RenderIR | None:
        if self._committed_create_preview is None:
            return None
        (
            kind,
            start,
            end,
            points,
            _polygon_closed,
            secondary_radius,
            torus_minor_radius,
        ) = self._committed_create_preview
        try:
            document = SceneDocument()
            if points:
                document.add_point_shape_from_world_points(
                    kind,
                    points,
                    self._reference_plane,
                )
            else:
                parameters: dict[str, float] = {}
                if kind == "capped_cone" and secondary_radius > 0.0:
                    parameters["top_diameter"] = 2.0 * secondary_radius
                if kind == "torus" and torus_minor_radius > 0.0:
                    parameters["minor_diameter"] = 2.0 * torus_minor_radius
                document.add_primitive_from_drag(
                    kind,
                    start,
                    end,
                    parameters=parameters,
                )
        except ValueError:
            return None
        return self._preview_render_ir_from_document(document)

    def _active_move_preview_render_ir(self) -> RenderIR | None:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] != "move"
        ):
            return None
        node = self._scene_node_by_object_id(self._scene_selected_object_id)
        if node is None:
            return None
        delta = self._preview_move_delta()
        rotations = ViewportWidget._rotation_preview_commands(self)
        if (
            max(abs(component) for component in delta) <= 1.0e-12
            and not rotations
        ):
            return None
        try:
            document, handle = self._preview_document_for_node(node)
            if max(abs(component) for component in delta) > 1.0e-12:
                handle = document.move_object(handle, delta)
            for axis, angle_degrees, pivot in rotations:
                if abs(angle_degrees) > 1.0e-6:
                    handle = document.rotate_object(
                        handle,
                        axis,
                        angle_degrees,
                        pivot,
                    )
        except (KeyError, ValueError, NotImplementedError):
            return None
        return self._preview_render_ir_from_document(document)

    def _active_extrude_preview_render_ir(self) -> RenderIR | None:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] != "extrude"
        ):
            return None
        node = self._scene_node_by_object_id(self._scene_selected_object_id)
        if not isinstance(node, PlacedSDF2D):
            return None
        height = float(self._extrude_preview_height)
        if abs(height) <= 1.0e-9:
            return None
        try:
            document, handle = self._preview_document_for_node(node)
            solid_handle = document.solid_from_2d(
                [handle],
                "extrude",
                signed_height=height,
            )
            solid = deepcopy(document.node(solid_handle))
            preview_document = SceneDocument(objects=[solid])
        except (KeyError, ValueError):
            return None
        return self._preview_render_ir_from_document(preview_document)

    def _active_revolve_preview_render_ir(self) -> RenderIR | None:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] != "revolve"
        ):
            return None
        node = self._scene_node_by_object_id(self._scene_selected_object_id)
        if not isinstance(node, PlacedSDF2D):
            return None
        active, origin, axis, radial = self._revolve_axis_frame()
        if not active:
            return None
        try:
            document, handle = self._preview_document_for_node(node)
            solid_handle = document.solid_from_2d(
                [handle],
                "revolve",
                revolve_axis_origin=origin,
                revolve_axis_direction=axis,
                revolve_radial_direction=radial,
                revolve_angle_degrees=float(self._revolve_angle_degrees),
            )
            solid = deepcopy(document.node(solid_handle))
            preview_document = SceneDocument(objects=[solid])
        except (KeyError, ValueError):
            return None
        return self._preview_render_ir_from_document(preview_document)

    def _boolean_preview_render_ir(self) -> RenderIR | None:
        if self._boolean_preview is None:
            return None
        opcode, first_id, second_id, _smoothing, _committed = self._boolean_preview
        operation = next(
            (
                name
                for name, value in BOOLEAN_PREVIEW_OPERATIONS.items()
                if value == opcode
            ),
            "",
        )
        if not operation:
            return None
        first = self._scene_node_by_object_id(first_id)
        second = self._scene_node_by_object_id(second_id)
        if first is None or second is None:
            return None
        try:
            first_clone = deepcopy(first)
            second_clone = deepcopy(second)
            document = SceneDocument(objects=[first_clone, second_clone])
            combined_handle = document.combine(
                document.handle_for(first_clone),
                document.handle_for(second_clone),
                operation,
            )
            combined = deepcopy(document.node(combined_handle))
            preview_document = SceneDocument(objects=[combined])
        except (KeyError, ValueError):
            return None
        return self._preview_render_ir_from_document(preview_document)

    def _preview_render_ir(self) -> RenderIR | None:
        return (
            self._active_create_preview_render_ir()
            or self._active_move_preview_render_ir()
            or self._active_extrude_preview_render_ir()
            or self._active_revolve_preview_render_ir()
            or self._boolean_preview_render_ir()
            or self._committed_create_preview_render_ir()
        )

    def _commit_point_shape_preview(self) -> None:
        kind = self._point_shape_kind()
        if kind is None:
            return
        minimum = point_shape_minimum_points(kind)
        if len(self._point_shape_points) < minimum:
            signals.log_message.emit(
                "warning",
                f"{CREATE_LABELS[kind]} requires at least {minimum} points.",
            )
            return
        if kind == "bezier_curve" and len(self._point_shape_points) != 3:
            signals.log_message.emit(
                "warning",
                "Bezier Curve requires exactly three points.",
            )
            return
        if kind in {"bezier_polycurve", "bezier_tube", "bezier_surface"} and len(
            self._point_shape_points
        ) % 2 == 0:
            signals.log_message.emit(
                "warning",
                f"{CREATE_LABELS[kind]} requires an odd point count: "
                "anchor, control, anchor.",
            )
            return
        points = tuple(self._point_shape_points)
        self.apply_committed_create_preview(
            kind,
            points[0],
            points[-1],
            points=points,
            polygon_closed=kind in {"bezier_surface", "polygon"},
        )
        self._reset_create_preview(kind)
        signals.viewport_point_shape_drawn.emit(kind, points, self._reference_plane)

    def _remove_last_point_shape_point(self) -> bool:
        kind = self._point_shape_kind()
        if kind is None or not self._point_shape_points:
            return False
        self._point_shape_points.pop()
        self._tool_start_world = (
            self._point_shape_points[0] if self._point_shape_points else None
        )
        self._tool_current_world = (
            self._point_shape_points[-1] if self._point_shape_points else None
        )
        self._update_measurement_readout()
        self.update()
        signals.log_message.emit("info", "Removed last point.")
        return True

    def _place_typed_create_start(self) -> bool:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] != "create"
            or self._tool_start_world is not None
            or not self._dimension_input
        ):
            return False
        try:
            point = apply_typed_start_point(
                self._reference_plane,
                parse_displacement_entry(self._dimension_input),
            )
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return False
        self._tool_start_screen = None
        self._tool_start_world = point
        self._tool_current_world = point
        self._tool_hover_world = point
        self._dimension_input = ""
        self._update_measurement_readout()
        self.update()
        signals.log_message.emit(
            "info",
            "Shape start placed. Move the cursor, type dimensions, "
            "or press Enter to create.",
        )
        return True

    def _handle_dimension_key(self, event: QKeyEvent) -> bool:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] not in {"create", "move", "extrude", "revolve"}
        ):
            return False
        action = self._interaction_tool[0]
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
            if action == "revolve" and self._dimension_input:
                try:
                    angle = parse_scalar_entry(self._dimension_input)
                except ValueError as error:
                    signals.log_message.emit("warning", str(error))
                    return True
                self._revolve_angle_degrees = min(360.0, max(0.1, angle))
                self._dimension_input = ""
                self._update_measurement_readout()
                self.update()
                self._commit_revolve_preview()
                return True
            if action == "extrude" and self._dimension_input:
                try:
                    height = parse_displacement_entry(self._dimension_input)[0]
                except ValueError as error:
                    signals.log_message.emit("warning", str(error))
                    return True
                self._extrude_preview_height = height
                self._dimension_input = ""
                self._update_measurement_readout()
                self.update()
                self._commit_extrude_preview()
                return True
            if action == "move" and self._dimension_input:
                if self._apply_typed_move_preview():
                    self._commit_move_preview()
                return True
            if (
                action == "create"
                and self._tool_start_world is None
                and self._dimension_input
            ):
                self._place_typed_create_start()
                return True
            if action == "create" and self._tool_start_world is not None:
                self._commit_create_preview()
                return True
            return False
        if event.key() == Qt.Key.Key_Backspace and self._dimension_input:
            self._dimension_input = self._dimension_input[:-1]
            self._update_measurement_readout()
            self.update()
            return True
        if event.modifiers() & (
            Qt.KeyboardModifier.ControlModifier
            | Qt.KeyboardModifier.MetaModifier
        ):
            return False
        text = dimension_entry_text(
            event.text(),
            "move" if action == "extrude" else action,
            self._tool_start_world is not None,
            self._dimension_input,
        )
        if text:
            self._dimension_input += text
            self._update_measurement_readout()
            self.update()
            return True
        return False

    def frame_box(self, box: BoundingBox3D) -> None:
        self._stop_view_animation()
        self.camera.frame(box)
        self.update()

    def configure_default_grid(self) -> None:
        self.configure_grid(
            BoundingBox3D(-1.0, 1.0, -1.0, 1.0, 0.0, 0.0),
            0.1,
        )

    def frame_default_grid(self) -> None:
        self.frame_box(BoundingBox3D(-1.0, 1.0, -1.0, 1.0, 0.0, 0.0))

    def configure_grid(self, box: BoundingBox3D, dx: float) -> None:
        span = max(
            box.x_max - box.x_min,
            box.y_max - box.y_min,
            box.z_max - box.z_min,
            dx * 10.0,
        )
        half_extent = max(1.0, span * 2.5)
        self._auto_grid_spacing = self._nice_grid_spacing(
            (2.0 * half_extent) / 40.0
        )
        self._grid_spacing = self._manual_grid_spacing or self._auto_grid_spacing
        self.update()

    @staticmethod
    def _nice_grid_spacing(raw_spacing: float) -> float:
        exponent = np.floor(np.log10(max(raw_spacing, 1e-9)))
        fraction = raw_spacing / (10.0**exponent)
        if fraction <= 1.0:
            nice_fraction = 1.0
        elif fraction <= 2.0:
            nice_fraction = 2.0
        elif fraction <= 5.0:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
        return float(nice_fraction * (10.0**exponent))

    def initializeGL(self) -> None:
        try:
            self._context = moderngl.create_context(require=330)
            self._renderer = OpenGLRenderer(self._context)
            self._compiled_scene = None
            self._compiled_preview_scene = EMPTY_RENDER_SCENE_KEY
            logger.info(
                "OpenGL initialized: %s",
                self._context.info.get("GL_RENDERER", "unknown renderer"),
            )
        except Exception as error:
            logger.exception("OpenGL initialization failed")
            signals.log_message.emit(
                "error", f"OpenGL initialization failed: {error}"
            )
            self._show_viewport_error(f"OpenGL initialization failed: {error}")

    def paintGL(self) -> None:
        if self._renderer is None:
            if self._context is not None:
                try:
                    self._context.clear(*self.background_color, 1.0)
                except Exception:
                    logger.exception("fallback viewport clear failed")
            if not self._viewport_error_label.isVisible():
                self._show_viewport_error("Viewport renderer is unavailable.")
            return
        try:
            self._renderer.bind_framebuffer(self.defaultFramebufferObject())
        except moderngl.Error as error:
            logger.exception("could not bind the Qt OpenGL framebuffer")
            signals.log_message.emit(
                "error", f"Viewport framebuffer failed: {error}"
            )
            return
        if self._pending_scene is not None:
            if self._accept_committed_boolean_pending_scene():
                pass
            elif self._pending_scene == self._compiled_scene:
                self._pending_scene = None
                self._pending_render_ir = None
                self._clear_committed_move_preview()
                self._clear_committed_create_preview()
                self._clear_committed_boolean_preview()
                self._clear_viewport_error()
            else:
                try:
                    if self._scene_tree is None and self._pending_render_ir is None:
                        self._renderer.clear_scene()
                    elif not self._renderer.upload_render_ir(self._pending_render_ir):
                        raise RuntimeError("RenderIR upload failed")
                    stats = self._renderer.last_scene_update_stats()
                    if stats is not None:
                        logger.info(_format_scene_update_stats(stats))
                        signals.log_message.emit(
                            "info",
                            _format_scene_update_stats(stats),
                        )
                    self._compiled_scene = self._pending_scene
                    self._pending_scene = None
                    self._pending_render_ir = None
                    self._clear_committed_move_preview()
                    self._clear_committed_create_preview()
                    self._clear_committed_boolean_preview()
                    self._clear_viewport_error()
                except Exception as error:
                    logger.exception("shader compilation failed")
                    signals.log_message.emit(
                        "error", f"Shader compilation failed: {error}"
                    )
                    self._show_viewport_error(f"Shader compilation failed: {error}")
                    self._pending_scene = None
                    self._pending_render_ir = None
                    return
        preview_render_ir = self._preview_render_ir()
        preview_scene_key = render_scene_key(preview_render_ir)
        if preview_scene_key != self._compiled_preview_scene:
            try:
                if preview_render_ir is None:
                    self._renderer.clear_preview_render_ir()
                elif not self._renderer.upload_preview_render_ir(preview_render_ir):
                    raise RuntimeError("Preview RenderIR upload failed")
                self._compiled_preview_scene = preview_scene_key
            except Exception as error:
                logger.exception("preview shader compilation failed")
                signals.log_message.emit(
                    "error", f"Preview shader compilation failed: {error}"
                )
                self._renderer.clear_preview_render_ir()
                self._compiled_preview_scene = EMPTY_RENDER_SCENE_KEY
        self._append_lattice_stream_chunks()
        self._advance_lattice_upload()
        width = max(1, round(self.width() * self.devicePixelRatio()))
        height = max(1, round(self.height() * self.devicePixelRatio()))
        scene_selected_object_id = (
            self._committed_move_object_id
            if self._committed_move_object_id != 0
            else self._scene_selected_object_id
        )
        rotation_gizmo_visible, rotation_gizmo_center, rotation_gizmo_radius = (
            self._rotation_gizmo_state()
        )
        try:
            self._renderer.render(
                width,
                height,
                self.camera.position,
                self.camera.target,
                self.camera.focal_length,
                self.camera.view_projection(width / height),
                self.mode,
                self.grid_visible,
                self.components_visible,
                self.sdf_opacity,
                self.background_color,
                self.camera.view_rotation(),
                self.gizmo_visible,
                self._grid_spacing,
                REFERENCE_PLANE_IDS[self._reference_plane],
                self._boundary_selection_active,
                self._boundary_hover_owner_id,
                self._boundary_hover_direction,
                self._boundary_hover_normal,
                self._scene_hover_object_id,
                scene_selected_object_id,
                tuple(self._selected_boundary_regions),
                tuple(self._selected_boundary_normals),
                self._preview_point_count(),
                self._preview_points(),
                rotation_gizmo_visible,
                rotation_gizmo_center,
                rotation_gizmo_radius,
            )
        except Exception as error:
            logger.exception("viewport render failed")
            signals.log_message.emit("error", f"Viewport render failed: {error}")
            self._show_viewport_error(f"Viewport render failed: {error}")
            return
        self._update_fps_counter()

    def _append_lattice_stream_chunks(self) -> None:
        if self._renderer is None or not self._pending_lattice_stream_chunks:
            return
        if self._pending_lattice_upload is not None:
            self._pending_lattice_stream_chunks.clear()
            return
        while self._pending_lattice_stream_chunks:
            chunk = self._pending_lattice_stream_chunks.pop(0)
            self._renderer.append_lattice_preview_chunk(
                chunk.preview_positions,
                chunk.preview_node_types,
                chunk.preview_boundary_faces,
                chunk.preview_source_object_ids,
                chunk.preview_primary_tag_ids,
                chunk.preview_cell_size,
                dimension=chunk.dimension,
                axis_i=getattr(chunk, "preview_axis_i", (1.0, 0.0, 0.0)),
                axis_j=getattr(chunk, "preview_axis_j", (0.0, 1.0, 0.0)),
            )

    def _advance_lattice_upload(self) -> None:
        if self._renderer is None or self._pending_lattice_upload is None:
            return
        point_vertices, square_instances, cell_size = self._pending_lattice_upload
        if not self._lattice_upload_started:
            self._renderer.begin_lattice_upload(
                point_vertices.shape[0],
                square_instances.shape[0],
                cell_size,
            )
            self._lattice_upload_started = True
        timer = QElapsedTimer()
        timer.start()
        while (
            self._lattice_point_upload_cursor < point_vertices.shape[0]
            and timer.elapsed() < LATTICE_UPLOAD_BUDGET_MS
        ):
            start = self._lattice_point_upload_cursor
            stop = min(start + LATTICE_POINT_UPLOAD_CHUNK, point_vertices.shape[0])
            self._renderer.write_lattice_points(start, point_vertices[start:stop])
            self._lattice_point_upload_cursor = stop
        while (
            self._lattice_point_upload_cursor >= point_vertices.shape[0]
            and self._lattice_square_upload_cursor < square_instances.shape[0]
            and timer.elapsed() < LATTICE_UPLOAD_BUDGET_MS
        ):
            start = self._lattice_square_upload_cursor
            stop = min(
                start + LATTICE_SQUARE_UPLOAD_CHUNK,
                square_instances.shape[0],
            )
            self._renderer.write_lattice_squares(start, square_instances[start:stop])
            self._lattice_square_upload_cursor = stop
        if (
            self._lattice_point_upload_cursor >= point_vertices.shape[0]
            and self._lattice_square_upload_cursor >= square_instances.shape[0]
        ):
            self._pending_lattice_upload = None
            self._lattice_upload_started = False
            return
        self.update()

    def _update_fps_counter(self) -> None:
        self._fps_frame_count += 1
        elapsed_ms = self._fps_timer.elapsed()
        if elapsed_ms < FPS_COUNTER_UPDATE_MS:
            return
        fps = 1000.0 * self._fps_frame_count / float(elapsed_ms)
        self._fps_label.setText(f"FPS {fps:4.1f}")
        self._fps_label.adjustSize()
        self._fps_label.raise_()
        self._fps_frame_count = 0
        self._fps_timer.restart()
        self._position_measure_label()

    def _preview_points(self) -> tuple[tuple[float, float, float], ...]:
        if self._point_shape_kind() is None:
            if self._committed_create_preview is not None:
                return self._committed_create_preview[3]
            return ()
        return self._point_shape_preview_points()

    def _preview_point_count(self) -> int:
        return len(self._preview_points())

    def _preview_move_delta(self) -> tuple[float, float, float]:
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "move"
        ):
            return move_preview_delta(
                self._move_preview_delta,
                self._tool_start_world,
                self._tool_current_world,
                self._reference_plane,
                QApplication.keyboardModifiers(),
            )
        if self._committed_move_object_id != 0:
            return self._committed_move_delta
        return (0.0, 0.0, 0.0)

    def _preview_rotation_pivot(self) -> tuple[float, float, float]:
        if self._rotation_drag_center is None:
            return (0.0, 0.0, 0.0)
        return ViewportWidget._rotation_step_pivot(
            self,
            self._rotation_drag_center,
            self._rotation_drag_move_delta,
        )

    def _rotation_step_pivot(
        self,
        center: tuple[float, float, float],
        move_delta: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        current_delta = self._preview_move_delta()
        return tuple(
            center[index]
            + current_delta[index]
            - move_delta[index]
            for index in range(3)
        )

    def _rotation_preview_commands(
        self,
    ) -> tuple[tuple[str, float, tuple[float, float, float]], ...]:
        commands: list[tuple[str, float, tuple[float, float, float]]] = [
            (
                axis,
                angle,
                ViewportWidget._rotation_step_pivot(self, center, move_delta),
            )
            for axis, angle, center, move_delta in getattr(
                self,
                "_rotation_preview_steps",
                [],
            )
            if abs(angle) > 1.0e-6
        ]
        if (
            self._rotation_drag_axis is not None
            and self._rotation_drag_center is not None
            and abs(self._rotation_preview_angle) > 1.0e-6
        ):
            commands.append(
                (
                    self._rotation_drag_axis,
                    self._rotation_preview_angle,
                    self._preview_rotation_pivot(),
                )
        )
        return tuple(commands)

    def _reference_axes(self) -> tuple[tuple[int, str], tuple[int, str]]:
        return REFERENCE_PLANE_AXES[self._reference_plane]

    def _format_measure(self, value: float) -> str:
        return f"{value:.5g} m"

    def _grid_measurement_text(self) -> str:
        return (
            f"Grid {self._format_measure(self._grid_spacing)}  "
            f"{snap_status_text(self._snap_enabled)}"
        )

    def _xyz_measurement_text(
        self,
        point: tuple[float, float, float] | None,
    ) -> str:
        if point is None:
            return "X --  Y --  Z --"
        return (
            f"X {self._format_measure(point[0])}  "
            f"Y {self._format_measure(point[1])}  "
            f"Z {self._format_measure(point[2])}"
        )

    def _snap_reference_point(
        self,
        point: tuple[float, float, float],
        modifiers: Qt.KeyboardModifier,
    ) -> tuple[float, float, float]:
        if not should_snap_reference_point(self._snap_enabled, modifiers):
            return point
        spacing = max(self._grid_spacing, 1e-9)
        values = list(point)
        active_axes = {axis for axis, _label in self._reference_axes()}
        for axis in active_axes:
            values[axis] = round(values[axis] / spacing) * spacing
        for axis in set(range(3)) - active_axes:
            values[axis] = 0.0
        return tuple(
            0.0 if abs(value) <= 1e-12 else float(value)
            for value in values
        )

    def _constrain_reference_point(
        self,
        point: tuple[float, float, float],
        modifiers: Qt.KeyboardModifier,
    ) -> tuple[float, float, float]:
        if (
            not modifiers & Qt.KeyboardModifier.ShiftModifier
            or self._interaction_tool is None
            or self._interaction_tool[0] != "create"
            or self._tool_start_world is None
        ):
            return point
        _action, kind = self._interaction_tool
        return constrain_reference_point(
            point,
            self._tool_start_world,
            self._reference_plane,
            str(kind),
        )

    def _tool_point_from_event(
        self,
        event: QMouseEvent,
    ) -> tuple[float, float, float] | None:
        point = self.camera.screen_to_plane(
            self._reference_plane,
            event.position().x(),
            event.position().y(),
            self.width(),
            self.height(),
        )
        if point is None:
            return None
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "move"
        ):
            return point
        snapped = self._snap_reference_point(point, event.modifiers())
        return self._constrain_reference_point(snapped, event.modifiers())

    def _update_measurement_readout(self) -> None:
        text = self._measurement_text()
        if not text:
            self._measure_label.hide()
            return
        self._measure_label.setText(text)
        self._measure_label.adjustSize()
        self._measure_label.show()
        self._position_measure_label()

    def _measurement_text(self) -> str:
        if self._committed_move_object_id != 0:
            return self._move_measurement_text(self._committed_move_delta)
        if self._interaction_tool is None:
            return ""
        action, value = self._interaction_tool
        if action == "move":
            delta = self._preview_move_delta()
            (_first_axis, first_label), (_second_axis, second_label) = (
                self._reference_axes()
            )
            input_text = self._dimension_input or move_dimension_prompt(
                first_label,
                second_label,
            )
            modifier_status = create_modifier_status_text(
                QApplication.keyboardModifiers()
            )
            cursor_point = self._tool_current_world or self._tool_hover_world
            if (
                getattr(self, "_rotation_drag_axis", None) is not None
                and getattr(self, "_rotation_drag_center", None) is not None
            ):
                return self._rotation_measurement_text(
                    self._rotation_drag_axis,
                    self._rotation_preview_angle,
                    self._rotation_drag_center,
                )
            return self._move_measurement_text(
                delta,
                cursor_point=cursor_point,
                input_text=input_text,
                modifier_text=modifier_status,
            )
        if action == "extrude":
            return self._extrude_measurement_text(
                self._extrude_preview_height,
                input_text=self._dimension_input or extrude_dimension_prompt(),
            )
        if action == "revolve":
            return self._revolve_measurement_text()
        if (
            action != "create"
            or self._tool_start_world is None
            or self._tool_current_world is None
        ):
            (_first_axis, first_label), (_second_axis, second_label) = (
                self._reference_axes()
            )
            base = (
                f"{CREATE_LABELS.get(str(value), str(value))} | "
                f"{reference_plane_context(self._reference_plane)} | "
                f"{self._grid_measurement_text()}"
            )
            if action == "create" and self._tool_hover_world is not None:
                base = (
                    f"{base} | Cursor "
                    f"{self._reference_coordinate_text(self._tool_hover_world)}"
                )
            return (
                f"{base} | {create_input_label(False)} {self._dimension_input}"
                if self._dimension_input
                else f"{base} | {create_start_prompt(first_label, second_label)}"
            )
        point_kind = self._point_shape_kind()
        if point_kind is not None:
            minimum = point_shape_minimum_points(point_kind)
            label = CREATE_LABELS.get(point_kind, point_kind)
            count = len(self._point_shape_points)
            limit_suffix = "" if point_kind == "bezier_curve" else "+"
            base = (
                f"{label} | {reference_plane_context(self._reference_plane)} | "
                f"Points {count}/{minimum}{limit_suffix}  {self._grid_measurement_text()}"
            )
            if self._tool_hover_world is not None:
                base = (
                    f"{base} | Cursor "
                    f"{self._reference_coordinate_text(self._tool_hover_world)}"
                )
            if point_kind == "bezier_curve":
                return f"{base} | Third point creates  Backspace removes"
            return f"{base} | Enter creates  Backspace removes"
        try:
            start, current = self._create_effective_points()
            typed_dimensions = (
                parse_dimension_entry(self._dimension_input)
                if self._dimension_input
                else None
            )
        except ValueError:
            start = self._tool_start_world
            current = self._tool_current_world
            typed_dimensions = None
        assert start is not None
        assert current is not None
        delta = tuple(current[index] - start[index] for index in range(3))
        (first_axis, first_label), (second_axis, second_label) = self._reference_axes()
        first = abs(delta[first_axis])
        second = abs(delta[second_axis])
        label = CREATE_LABELS.get(str(value), str(value))
        measurements = "  ".join(
            f"{name} {self._format_measure(measurement)}"
            for name, measurement in create_measurement_components(
                str(value),
                first_label,
                first,
                second_label,
                second,
                delta,
                typed_dimensions,
            )
        )
        text = (
            f"{label} | {reference_plane_context(self._reference_plane)} | "
            f"{measurements}  {self._grid_measurement_text()}"
        )
        modifier_status = create_modifier_status_text(QApplication.keyboardModifiers())
        if modifier_status:
            text = f"{text}  {modifier_status}"
        if self._dimension_input:
            text = f"{text}  {create_input_label(True)} {self._dimension_input}"
        else:
            text = f"{text}  {create_size_prompt(str(value), first_label, second_label)}"
        return text

    def _create_centered_from_modifiers(
        self,
        modifiers: Qt.KeyboardModifier | Qt.KeyboardModifiers,
    ) -> bool:
        return bool(modifiers & Qt.KeyboardModifier.ControlModifier)

    def _refresh_interaction_modifier_preview_for_key(self, key: int) -> bool:
        if not should_refresh_create_modifier_preview_for_key(key):
            return False
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] not in {"create", "move"}
            or self._tool_start_world is None
            or self._tool_current_world is None
        ):
            return False
        self._update_measurement_readout()
        self.update()
        return True

    def _clear_idle_selection_for_key(
        self,
        key: int,
        modifiers: Qt.KeyboardModifier | Qt.KeyboardModifiers,
    ) -> bool:
        if not should_clear_idle_selection_for_key(key, modifiers):
            return False
        if self._interaction_tool is not None:
            return False
        self._clear_scene_hover()
        signals.viewport_scene_object_selected.emit(0)
        return True

    def _cycle_reference_plane_for_key(
        self,
        key: int,
        modifiers: Qt.KeyboardModifier | Qt.KeyboardModifiers,
    ) -> bool:
        if not should_cycle_reference_plane_for_key(key, modifiers):
            return False
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] not in {"create", "move"}
        ):
            return False
        self.set_reference_view(
            next_reference_plane(
                self._reference_plane,
                reverse=bool(modifiers & Qt.KeyboardModifier.ShiftModifier),
            )
        )
        if self._interaction_tool[0] == "create" and self._tool_start_world is not None:
            self._tool_current_world = self._tool_start_world
            self._tool_hover_world = self._tool_start_world
        if self._interaction_tool[0] == "move" and self._tool_start_world is not None:
            self._tool_current_world = self._tool_start_world
        self._update_measurement_readout()
        return True

    def _create_effective_points(
        self,
        centered: bool | None = None,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] != "create"
            or self._tool_start_world is None
        ):
            return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
        _action, kind = self._interaction_tool
        anchor = self._tool_start_world
        current = self._tool_current_world or anchor
        is_centered = (
            self._create_centered_from_modifiers(QApplication.keyboardModifiers())
            if centered is None
            else centered
        )
        dimensions = (
            parse_dimension_entry(self._dimension_input)
            if self._dimension_input
            else None
        )
        return create_effective_endpoints(
            anchor,
            current,
            self._reference_plane,
            str(kind),
            dimensions,
            is_centered,
        )

    def _move_measurement_text(
        self,
        delta: tuple[float, float, float],
        *,
        cursor_point: tuple[float, float, float] | None = None,
        input_text: str = "",
        modifier_text: str = "",
    ) -> str:
        distance = float(sum(component * component for component in delta) ** 0.5)
        rows = [
            ("Tool", "Move"),
            ("Reference", reference_plane_context(self._reference_plane)),
            ("Cursor point", self._xyz_measurement_text(cursor_point)),
            ("Move delta", self._xyz_measurement_text(delta)),
            ("Distance", self._format_measure(distance)),
            ("Grid spacing", self._format_measure(self._grid_spacing)),
            ("Snap", "On" if self._snap_enabled else "Off"),
        ]
        if modifier_text:
            rows.append(("Constraint", modifier_text))
        if input_text:
            rows.append(("Entry", input_text))
        label_width = max(len(label) for label, _value in rows)
        return "\n".join(
            f"{label:<{label_width}}  {value}"
            for label, value in rows
        )

    def _rotation_measurement_text(
        self,
        axis: str,
        angle_degrees: float,
        center: tuple[float, float, float],
    ) -> str:
        rows = [
            ("Tool", "Rotate"),
            ("Axis", axis.upper()),
            ("Angle", f"{angle_degrees:.5g} deg"),
            ("Pivot point", self._xyz_measurement_text(center)),
        ]
        label_width = max(len(label) for label, _value in rows)
        return "\n".join(
            f"{label:<{label_width}}  {value}"
            for label, value in rows
        )

    def _extrude_measurement_text(
        self,
        height: float,
        *,
        input_text: str = "",
    ) -> str:
        active, anchor, normal = self._extrude_gizmo_state()
        rows = [
            ("Tool", "Extrude"),
            ("Base point", self._xyz_measurement_text(anchor if active else None)),
            ("Direction", self._xyz_measurement_text(normal if active else None)),
            ("Signed length", self._format_measure(height)),
            ("Grid spacing", self._format_measure(self._grid_spacing)),
            ("Snap", "On" if self._snap_enabled else "Off"),
        ]
        if input_text:
            rows.append(("Entry", input_text))
        label_width = max(len(label) for label, _value in rows)
        return "\n".join(
            f"{label:<{label_width}}  {value}"
            for label, value in rows
        )

    def _revolve_measurement_text(self) -> str:
        active, start, end = self._revolve_axis_segment()
        frame_active, _origin, axis, radial = self._revolve_axis_frame()
        rows = [
            ("Tool", "Revolve"),
            ("Axis", self._revolve_axis.upper()),
            ("Axis angle", f"{self._revolve_axis_angle_degrees:.5g} deg"),
                ("Revolve angle", f"{self._revolve_angle_degrees:.5g} deg"),
            ("Start", self._xyz_measurement_text(start if active else None)),
            ("End", self._xyz_measurement_text(end if active else None)),
            ("Direction", self._xyz_measurement_text(axis if frame_active else None)),
            ("Radial ref", self._xyz_measurement_text(radial if frame_active else None)),
            ("Grid spacing", self._format_measure(self._grid_spacing)),
            ("Entry", "Type angle in degrees"),
        ]
        label_width = max(len(label) for label, _value in rows)
        return "\n".join(
            f"{label:<{label_width}}  {value}"
            for label, value in rows
        )

    def _extrude_height_from_screen(
        self,
        x: float,
        y: float,
    ) -> float | None:
        if self._tool_start_screen is None:
            return None
        active, anchor, normal = self._extrude_gizmo_state()
        if not active:
            return None
        start = self._project_world_to_screen(anchor)
        end = self._project_world_to_screen(
            tuple(anchor[index] + normal[index] * self._grid_spacing for index in range(3))
        )
        if start is None or end is None:
            axis = np.asarray((0.0, -1.0), dtype=np.float64)
            pixels_per_step = 40.0
        else:
            axis = np.asarray((end[0] - start[0], end[1] - start[1]), dtype=np.float64)
            pixels_per_step = float(np.linalg.norm(axis))
            if pixels_per_step <= 1.0e-6:
                axis = np.asarray((0.0, -1.0), dtype=np.float64)
                pixels_per_step = 40.0
            else:
                axis /= pixels_per_step
        drag = np.asarray(
            (
                x - float(self._tool_start_screen.x()),
                y - float(self._tool_start_screen.y()),
            ),
            dtype=np.float64,
        )
        height = self._extrude_drag_start_height + (
            float(np.dot(drag, axis)) / max(pixels_per_step, 1.0)
        ) * self._grid_spacing
        return snap_scalar(
            height,
            self._grid_spacing,
            self._snap_enabled,
            QApplication.keyboardModifiers(),
        )

    def _reference_coordinate_text(
        self,
        point: tuple[float, float, float],
    ) -> str:
        return "  ".join(
            f"{label} {self._format_measure(value)}"
            for label, value in reference_plane_coordinate_components(
                point,
                self._reference_plane,
            )
        )

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._stop_view_animation()
        self._clear_scene_hover()
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "boundary_region"
            and event.button() == Qt.MouseButton.LeftButton
        ):
            position = event.position().toPoint()
            self._boundary_press_position = position
            self._boundary_camera_dragged = False
            self._last_mouse_position = position
            return
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "move"
            and event.button() == Qt.MouseButton.LeftButton
        ):
            axis = self._pick_rotation_gizmo_axis(
                event.position().x(),
                event.position().y(),
            )
            visible, center, _radius = self._rotation_gizmo_state()
            if axis is not None and visible:
                start = self._screen_to_rotation_plane(
                    axis,
                    event.position().x(),
                    event.position().y(),
                    center,
                )
                if start is not None:
                    self._rotation_drag_axis = axis
                    self._rotation_drag_start = start
                    self._rotation_drag_center = center
                    self._rotation_drag_move_delta = self._preview_move_delta()
                    self._rotation_preview_angle = 0.0
                    self._tool_start_screen = event.position().toPoint()
                    self._update_measurement_readout()
                    self.update()
                    signals.log_message.emit(
                        "info",
                        f"Rotate around {axis.upper()}. Drag to set the angle.",
                    )
                    return
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "extrude"
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._tool_start_screen = event.position().toPoint()
            self._extrude_drag_start_height = self._extrude_preview_height
            self._update_measurement_readout()
            self.update()
            return
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "revolve"
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._revolve_axis = "u" if self._revolve_axis == "v" else "v"
            self._revolve_axis_angle_degrees = (
                0.0 if self._revolve_axis == "u" else 90.0
            )
            self._update_measurement_readout()
            self.update()
            signals.log_message.emit(
                "info",
                f"Revolve axis set to {self._revolve_axis.upper()}. "
                "Press Enter to apply.",
            )
            return
        if (
            self._interaction_tool is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            if self._point_shape_kind() is not None:
                point = self._tool_point_from_event(event)
                if point is None:
                    signals.log_message.emit(
                        "warning",
                        "The current camera ray does not reach the reference plane.",
                    )
                    return
                self._point_shape_points.append(point)
                self._tool_start_screen = event.position().toPoint()
                self._tool_start_world = self._point_shape_points[0]
                self._tool_current_world = point
                self._tool_hover_world = point
                self._update_measurement_readout()
                self.update()
                if (
                    self._point_shape_kind() == "bezier_curve"
                    and len(self._point_shape_points) == 3
                ):
                    self._commit_point_shape_preview()
                    return
                if self._point_shape_kind() == "bezier_curve":
                    point_message = (
                        "Point added. Click the remaining Bezier point, "
                        "Backspace removes last point, Esc cancels."
                    )
                else:
                    point_message = (
                        "Point added. Move to preview next edge, Enter creates, "
                        "Backspace removes last point, Esc cancels."
                    )
                signals.log_message.emit("info", point_message)
                return
            point = self._tool_point_from_event(event)
            if point is None:
                signals.log_message.emit(
                    "warning",
                    "The current camera ray does not reach the reference plane.",
                )
                return
            if (
                self._interaction_tool[0] == "create"
                and self._tool_start_world is not None
                and self._tool_start_screen is None
            ):
                self._tool_current_world = point
                self._tool_hover_world = point
                self._commit_create_preview(
                    centered=self._create_centered_from_modifiers(
                        event.modifiers()
                    )
                )
                return
            self._tool_start_screen = event.position().toPoint()
            self._tool_start_world = point
            self._tool_current_world = point
            self._tool_hover_world = point
            self._update_measurement_readout()
            self.update()
            return
        position = event.position().toPoint()
        self._last_mouse_position = position
        self._scene_press_position = (
            position if event.button() == Qt.MouseButton.LeftButton else None
        )

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if (
            self._rotation_drag_axis is not None
            and self._rotation_drag_center is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            current = self._screen_to_rotation_plane(
                self._rotation_drag_axis,
                event.position().x(),
                event.position().y(),
                self._rotation_drag_center,
            )
            if current is not None:
                self._rotation_preview_angle = self._rotation_drag_angle(current)
                self._update_measurement_readout()
                self.update()
            return
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "extrude"
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            height = self._extrude_height_from_screen(
                event.position().x(),
                event.position().y(),
            )
            if height is not None:
                self._extrude_preview_height = height
                self._update_measurement_readout()
                self.update()
            return
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "boundary_region"
            and not event.buttons()
        ):
            hit = self._pick_boundary(event.position())
            signals.viewport_boundary_hovered.emit(
                (
                    hit[1],
                    tuple(float(value) for value in hit[2]),
                )
                if hit is not None
                else None
            )
            return
        if self._interaction_tool is None and not event.buttons():
            object_id = self._pick_scene_object(event.position())
            if object_id != self._scene_hover_object_id:
                self._scene_hover_object_id = object_id
                self.update()
            return
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "create"
            and not event.buttons()
        ):
            point = self._tool_point_from_event(event)
            if point is not None:
                self._tool_hover_world = point
                if self._tool_start_world is not None:
                    self._tool_current_world = point
            else:
                self._tool_hover_world = None
                if self._tool_start_world is None:
                    self._tool_current_world = None
            self._update_measurement_readout()
            self.update()
            return
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "extrude"
            and event.button() == Qt.MouseButton.LeftButton
        ):
            height = self._extrude_height_from_screen(
                event.position().x(),
                event.position().y(),
            )
            if height is not None:
                self._extrude_preview_height = height
            self._tool_start_screen = None
            self._update_measurement_readout()
            self.update()
            signals.log_message.emit(
                "info",
                "Extrude preview updated. Press Enter to apply or Esc to cancel.",
            )
            return
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "move"
            and not event.buttons()
        ):
            point = self._tool_point_from_event(event)
            self._tool_hover_world = point
            self._update_measurement_readout()
            self.update()
            return
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "boundary_region"
            and self._boundary_press_position is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            displacement = event.position().toPoint() - self._boundary_press_position
            if displacement.manhattanLength() > 4:
                self._boundary_camera_dragged = True
        if (
            self._interaction_tool is not None
            and self._tool_start_screen is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            point = self._tool_point_from_event(event)
            if point is not None:
                self._tool_current_world = point
                self._update_measurement_readout()
                self.update()
            return
        if self._last_mouse_position is None or not event.buttons():
            return
        current = event.position().toPoint()
        delta = current - self._last_mouse_position
        if event.buttons() & Qt.MouseButton.RightButton:
            self.camera.pan(delta.x(), delta.y())
        else:
            self._leave_planar_view_for_orbit()
            self.camera.orbit(delta.x(), delta.y())
        self._last_mouse_position = current
        self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "move"
            and self._rotation_drag_axis is not None
            and self._rotation_drag_center is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            if abs(self._rotation_preview_angle) > 1.0e-6:
                self._rotation_preview_steps.append(
                    (
                        self._rotation_drag_axis,
                        self._rotation_preview_angle,
                        self._rotation_drag_center,
                        self._rotation_drag_move_delta,
                    )
                )
            ViewportWidget._clear_active_rotation_drag(self)
            self._tool_start_screen = None
            self._update_measurement_readout()
            self.update()
            signals.log_message.emit(
                "info",
                "Rotation preview updated. Press Enter to apply or Esc to cancel.",
            )
            return
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "boundary_region"
            and event.button() == Qt.MouseButton.LeftButton
            and self._boundary_press_position is not None
        ):
            dragged = self._boundary_camera_dragged
            self._boundary_press_position = None
            self._boundary_camera_dragged = False
            self._last_mouse_position = None
            if dragged:
                hit = self._pick_boundary(event.position())
                signals.viewport_boundary_hovered.emit(
                    (
                        hit[1],
                        tuple(float(value) for value in hit[2]),
                    )
                    if hit is not None
                    else None
                )
                return
            hit = self._pick_boundary(event.position())
            if hit is None:
                signals.log_message.emit(
                    "warning", "No FluidDomain boundary is under the cursor."
                )
                return
            _point, owner_object_id, normal = hit
            self.cancel_interaction_tool()
            signals.viewport_boundary_region_requested.emit(
                (owner_object_id, tuple(float(value) for value in normal))
            )
            return
        if (
            self._interaction_tool is not None
            and self._tool_start_world is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            if self._point_shape_kind() is not None:
                return
            end = self._tool_point_from_event(event)
            action, _value = self._interaction_tool
            start = self._tool_start_world
            if end is None:
                signals.log_message.emit(
                    "warning",
                    "The current camera ray does not reach the reference plane.",
                )
                return
            if action == "create":
                if self._tool_start_screen is None:
                    self._tool_current_world = end
                    self._tool_hover_world = end
                    self._commit_create_preview(
                        centered=self._create_centered_from_modifiers(
                            event.modifiers()
                        )
                    )
                    return
                release_screen = event.position().toPoint()
                if should_defer_create_release(
                    (self._tool_start_screen.x(), self._tool_start_screen.y()),
                    (release_screen.x(), release_screen.y()),
                    bool(self._dimension_input),
                ):
                    self._tool_current_world = end
                    self._update_measurement_readout()
                    self.update()
                    signals.log_message.emit(
                        "info",
                        "Shape start placed. Move the cursor, type dimensions, "
                        "or press Enter to create.",
                    )
                    return
                self._tool_current_world = end
                self._commit_create_preview(
                    centered=self._create_centered_from_modifiers(
                        event.modifiers()
                    )
                )
            else:
                delta = tuple(end[index] - start[index] for index in range(3))
                self._move_preview_delta = tuple(
                    self._move_preview_delta[index] + delta[index]
                    for index in range(3)
                )
                self._tool_start_screen = None
                self._tool_start_world = None
                self._tool_current_world = None
                self._tool_hover_world = end
                self._update_measurement_readout()
                self.update()
                signals.log_message.emit(
                    "info", "Move preview updated. Press Enter to apply or Esc to cancel."
                )
            return
        if (
            self._interaction_tool is None
            and event.button() == Qt.MouseButton.LeftButton
            and self._scene_press_position is not None
        ):
            release_position = event.position().toPoint()
            displacement = release_position - self._scene_press_position
            self._last_mouse_position = None
            self._scene_press_position = None
            if displacement.manhattanLength() <= SCENE_CLICK_MAX_MANHATTAN:
                signals.viewport_scene_object_selected.emit(
                    self._pick_scene_object(event.position())
                )
                return
            super().mouseReleaseEvent(event)
            return
        self._last_mouse_position = None
        self._scene_press_position = None
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: object) -> None:
        self._clear_scene_hover()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._handle_dimension_key(event):
            return
        if self._cycle_reference_plane_for_key(event.key(), event.modifiers()):
            return
        if self._refresh_interaction_modifier_preview_for_key(event.key()):
            return
        if (
            event.key() == Qt.Key.Key_Backspace
            and self._remove_last_point_shape_point()
        ):
            return
        if self._interaction_tool is not None and event.key() in {
            Qt.Key.Key_Escape,
            Qt.Key.Key_Delete,
            Qt.Key.Key_Backspace,
        }:
            self.cancel_active_interaction_tool()
            return
        if self._clear_idle_selection_for_key(event.key(), event.modifiers()):
            return
        if (
            self._interaction_tool is None
            and should_frame_scene_for_key(event.key(), event.modifiers())
        ):
            signals.viewport_frame_requested.emit()
            return
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "move"
        ):
            if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
                self._commit_move_preview()
                return
            allowed_modifiers = (
                Qt.KeyboardModifier.NoModifier
                | Qt.KeyboardModifier.ShiftModifier
                | Qt.KeyboardModifier.AltModifier
            )
            if event.modifiers() & ~allowed_modifiers == Qt.KeyboardModifier.NoModifier:
                delta = self._keyboard_move_delta(event.key(), event.modifiers())
                if delta is not None:
                    self.nudge_move_preview(delta)
                    return
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "extrude"
        ):
            if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
                self._commit_extrude_preview()
                return
            allowed_modifiers = (
                Qt.KeyboardModifier.NoModifier
                | Qt.KeyboardModifier.ShiftModifier
                | Qt.KeyboardModifier.AltModifier
            )
            if event.modifiers() & ~allowed_modifiers == Qt.KeyboardModifier.NoModifier:
                delta = keyboard_extrude_delta(
                    event.key(),
                    self._grid_spacing,
                    event.modifiers(),
                )
                if delta is not None:
                    self.nudge_extrude_preview(delta)
                    return
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "revolve"
        ):
            if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
                self._commit_revolve_preview()
                return
            if event.modifiers() == Qt.KeyboardModifier.NoModifier:
                if event.key() == Qt.Key.Key_U:
                    self._revolve_axis = "u"
                    self._revolve_axis_angle_degrees = 0.0
                    self._update_measurement_readout()
                    self.update()
                    signals.log_message.emit(
                        "info",
                        "Revolve axis set to U. Press Enter to apply.",
                    )
                    return
                if event.key() == Qt.Key.Key_V:
                    self._revolve_axis = "v"
                    self._revolve_axis_angle_degrees = 90.0
                    self._update_measurement_readout()
                    self.update()
                    signals.log_message.emit(
                        "info",
                        "Revolve axis set to V. Press Enter to apply.",
                    )
                    return
            allowed_modifiers = (
                Qt.KeyboardModifier.NoModifier
                | Qt.KeyboardModifier.ShiftModifier
                | Qt.KeyboardModifier.AltModifier
            )
            if event.modifiers() & ~allowed_modifiers == Qt.KeyboardModifier.NoModifier:
                step = 5.0
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    step = 15.0
                if event.modifiers() & Qt.KeyboardModifier.AltModifier:
                    step = 1.0
                if event.key() in {Qt.Key.Key_A, Qt.Key.Key_D}:
                    direction = -1.0 if event.key() == Qt.Key.Key_A else 1.0
                    self._revolve_axis = "custom"
                    self._revolve_axis_angle_degrees = (
                        self._revolve_axis_angle_degrees + direction * step
                    ) % 180.0
                    self._update_measurement_readout()
                    self.update()
                    return
                if event.key() in {Qt.Key.Key_Q, Qt.Key.Key_E}:
                    direction = -1.0 if event.key() == Qt.Key.Key_Q else 1.0
                    self._revolve_angle_degrees = min(
                        360.0,
                        max(0.1, self._revolve_angle_degrees + direction * step),
                    )
                    self._update_measurement_readout()
                    self.update()
                    return
        if (
            self._interaction_tool is None
            and event.modifiers() == Qt.KeyboardModifier.NoModifier
        ):
            reference_view = reference_view_for_key(event.key())
            if reference_view is not None:
                self.set_reference_view(reference_view)
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        if self._refresh_interaction_modifier_preview_for_key(event.key()):
            return
        super().keyReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._stop_view_animation()
        self.camera.zoom(event.angleDelta().y() / 120.0)
        self.update()

    def closeEvent(self, event: object) -> None:
        if self._renderer is not None:
            try:
                self._renderer.release()
            except Exception:
                logger.exception("failed to release renderer resources")
            self._renderer = None
        self._compiled_scene = None
        self._context = None
        super().closeEvent(event)


# Compatibility alias while imports migrate.
GLWidget = ViewportWidget
