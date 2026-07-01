"""Drag/point create tool for the QRhi viewport.

Owns the create-tool state machine — arm/cancel, click-collected point
shapes, the anchored click + typed-dimension flow, and boundary-cutter
routing. The widget routes input events here and renders the previews
(rubber band, point polyline) from this state; committed geometry is
emitted through the usual signals (viewport_shape_drawn /
viewport_point_shape_drawn) exactly as before the extraction.
"""
from __future__ import annotations

from PySide6.QtCore import Qt

from app.dimensions import parse_dimension_entry
from app.signals import signals

POINT_CREATE_KINDS = {
    "polyline", "quadratic_bezier_curve", "quadratic_bezier_polycurve",
    "polyline_tube", "quadratic_bezier_tube", "quadratic_bezier_surface", "polygon",
}


class CreateTool:
    def __init__(self, viewport) -> None:
        self._viewport = viewport
        self.kind: str | None = None          # armed shape kind, None = inactive
        self.start_world = None               # drag start (drag-kinds)
        self.anchor = None                    # locked start (click -> typed size)
        self.hover = None                     # hover for the rubber-band preview
        self.dimension_input = ""             # typed dimension buffer
        self.points: list | None = None       # multi-click point list (point kinds)
        self.point_hover = None               # hover point for the preview
        self.boundary_cutter = None           # (cutter_kind, shape) when routing

    @property
    def active(self) -> bool:
        return self.kind is not None

    @property
    def point_mode(self) -> bool:
        return self.points is not None

    def begin(self, kind: str) -> None:
        """Arm create: drag-kinds sketch with one drag (viewport_shape_drawn);
        point-kinds collect clicks then commit on Enter (viewport_point_shape_drawn)."""
        viewport = self._viewport
        viewport.end_move_tool()
        viewport.end_rotate_tool()
        viewport._end_extrude_tool()
        viewport._end_revolve_tool()
        self.kind = str(kind)
        self.start_world = None
        self.anchor = None
        self.hover = None
        self.dimension_input = ""
        point_mode = kind in POINT_CREATE_KINDS
        self.points = [] if point_mode else None
        self.point_hover = None
        viewport._renderer.prewarm_for_tool(
            viewport._committed_surface_scene,
            kind,
            compile_pipeline=(
                not point_mode
                or viewport._renderer.should_prewarm_tool_pipeline()
            ),
        )
        viewport.setCursor(Qt.CursorShape.CrossCursor)
        viewport.setFocus()
        if point_mode:
            msg = (f"Click points on the {viewport.reference_plane_label} grid to "
                   f"draw {kind}. Enter creates, Backspace undoes, Esc cancels.")
        else:
            msg = (f"Drag on the {viewport.reference_plane_label} grid to create "
                   f"{kind}. Esc cancels.")
        signals.log_message.emit("info", msg)

    def cancel(self) -> None:
        if self.kind is None:
            return
        self.kind = None
        self.start_world = None
        self.anchor = None
        self.hover = None
        self.dimension_input = ""
        self.points = None
        self.point_hover = None
        self.boundary_cutter = None
        self._viewport.unsetCursor()
        self._viewport._dirty = True

    def commit_point_shape(self) -> None:
        """Emit the collected point-shape if it has enough points."""
        if not self.points or len(self.points) < 2:
            signals.log_message.emit("warning", "Add at least two points first.")
            return
        kind = self.kind
        points = tuple(self.points)
        plane = self._viewport._plane_id()
        # Emit before resetting so a synchronous handler can still read
        # active_boundary_cutter_tool (cutter routing).
        signals.viewport_point_shape_drawn.emit(kind, points, plane)
        self.cancel()

    def emit_drag_shape(self, start, end) -> None:
        signals.viewport_shape_drawn.emit(self.kind, start, end, None)
        self.cancel()

    def commit_typed_dimension(self) -> None:
        """Create the anchored shape at an exact typed size (W or W x H),
        read in the working unit."""
        if self.anchor is None or not self.dimension_input:
            return
        viewport = self._viewport
        unit = viewport._working_unit
        try:
            dims = parse_dimension_entry(self.dimension_input, unit.factor)
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        w = dims[0] * unit.factor
        h = (dims[1] if len(dims) > 1 else dims[0]) * unit.factor
        ui, vi = viewport._PLANE_AXIS_INDICES[viewport._grid_plane]
        end = list(self.anchor)
        end[ui] += w
        end[vi] += h
        self.emit_drag_shape(self.anchor, (end[0], end[1], end[2]))

    def begin_boundary_cutter(self, cutter_kind: str, shape_kind=None) -> None:
        """Arm a boundary cutter: draw `shape_kind` on the grid; the drawn shape
        is routed as a planar/surface cutter for the selected BoundaryRegion."""
        if not self._viewport._boundary_region_selected:
            signals.log_message.emit(
                "warning", "Select a BoundaryRegion before creating a cutter.")
            return
        if cutter_kind == "planar":
            shape = shape_kind or "polyline"
        elif cutter_kind == "surface":
            shape = shape_kind or "sphere"
        else:
            raise ValueError(f"unknown boundary cutter kind: {cutter_kind}")
        self.begin(shape)
        self.boundary_cutter = (cutter_kind, shape)
        signals.log_message.emit(
            "info",
            f"{cutter_kind.title()} cutter armed — draw the {shape} to cut the "
            "selected BoundaryRegion.")
