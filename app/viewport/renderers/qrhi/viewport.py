from __future__ import annotations

"""QRhi viewport widget backed by generated viewport surfaces.

One QRhi renderer consumes stable vertex/index buffers for each backend. Backend
coordinate conventions (fb_y_up / clip_y_sign) are handled at initialize, and no
backend is pinned. Set ``QRHI_BACKEND`` (vulkan|opengl|metal|d3d11) to choose
one; otherwise QRhi picks the platform default.
"""

import math
import os
import random
import time
from dataclasses import replace

import numpy as np
from PySide6.QtCore import QPointF, Qt, QElapsedTimer, QTimer
from PySide6.QtGui import QColor, QMouseEvent, QPainter, QPen, QWheelEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QMenu,
    QPushButton,
    QRhiWidget,
    QWidget,
)

from app.axis_labels import world_axis_label
from app.dimensions import DEFAULT_LENGTH_UNIT
from app.signals import signals
from app.viewport.camera import DEFAULT_VIEW_DISTANCE, ViewportCamera
from app.viewport.performance_governor import (
    ViewportPerformanceGovernor,
    ViewportRenderBudget,
)

from .boundary_tool import BoundaryTool
from .create_tool import CreateTool
from .surface_renderer import QRhiSurfaceRenderer

_BACKENDS = {
    "vulkan": QRhiWidget.Api.Vulkan,
    "opengl": QRhiWidget.Api.OpenGL,
    "metal": getattr(QRhiWidget.Api, "Metal", None),
    "d3d11": getattr(QRhiWidget.Api, "Direct3D11", None),
}
_REVOLVE_PREVIEW_INTERVAL_MS = 80.0


def _revolve_signal_frame(
    axis: np.ndarray | tuple[float, float, float],
    radial: np.ndarray | tuple[float, float, float],
    signed_degrees: float,
) -> tuple[tuple[float, float, float], float] | None:
    degrees = float(signed_degrees)
    if not math.isfinite(degrees):
        return None
    if abs(degrees) <= 1.0e-6:
        return None
    angle = max(-360.0, min(360.0, degrees))
    axis_vec = np.asarray(axis, dtype=np.float64)
    axis_len = float(np.linalg.norm(axis_vec))
    radial_vec = np.asarray(radial, dtype=np.float64)
    if axis_len <= 1.0e-12 or not math.isfinite(axis_len):
        return None
    axis_vec = axis_vec / axis_len
    radial_vec = radial_vec - axis_vec * float(np.dot(radial_vec, axis_vec))
    radial_len = float(np.linalg.norm(radial_vec))
    if radial_len <= 1.0e-12 or not math.isfinite(radial_len):
        return None
    radial_vec = radial_vec / radial_len
    return (
        (
            float(radial_vec[0]),
            float(radial_vec[1]),
            float(radial_vec[2]),
        ),
        float(angle),
    )


def _choose_api() -> "QRhiWidget.Api | None":
    """Backend-aware: honour an explicit QRHI_BACKEND, else let QRhi choose the
    platform-native default (None). The renderer is backend-agnostic, so no backend
    is pinned here — the OpenGL hard-pin that the bytecode VM required is gone."""
    want = os.environ.get("QRHI_BACKEND", "").lower()
    if want in _BACKENDS and _BACKENDS[want] is not None:
        return _BACKENDS[want]
    return None


def _translated_preview_surface(surface, delta: tuple[float, float, float]):
    offset = np.asarray(delta, dtype=np.float32)
    vertices = np.asarray(surface.vertices, dtype=np.float32) + offset
    bounds_delta = tuple(float(value) for value in offset)
    return replace(
        surface,
        vertices=vertices,
        bounds_min=tuple(
            float(surface.bounds_min[index] + bounds_delta[index])
            for index in range(3)
        ),
        bounds_max=tuple(
            float(surface.bounds_max[index] + bounds_delta[index])
            for index in range(3)
        ),
    )


def _camera_basis_from_angles(
    yaw: float,
    pitch: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cp = math.cos(pitch)
    camera_offset = np.array(
        [cp * math.cos(yaw), cp * math.sin(yaw), math.sin(pitch)],
        dtype=np.float64,
    )
    fwd = -camera_offset
    fwd /= max(float(np.linalg.norm(fwd)), 1e-9)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if abs(float(np.dot(fwd, world_up))) > 0.99:
        world_up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    right = np.cross(fwd, world_up)
    right /= max(float(np.linalg.norm(right)), 1e-9)
    up = np.cross(right, fwd)
    return right, up, fwd


def _orientation_axis_projection(
    yaw: float,
    pitch: float,
) -> tuple[tuple[str, float, float, float], ...]:
    right, up, fwd = _camera_basis_from_angles(yaw, pitch)
    axes = (
        ("X", np.array([1.0, 0.0, 0.0], dtype=np.float64)),
        ("Y", np.array([0.0, 1.0, 0.0], dtype=np.float64)),
        ("Z", np.array([0.0, 0.0, 1.0], dtype=np.float64)),
    )
    return tuple(
        (
            label,
            float(np.dot(axis, right)),
            float(-np.dot(axis, up)),
            float(np.dot(axis, fwd)),
        )
        for label, axis in axes
    )


class _OrientationWidget(QWidget):
    _COLORS = {
        "X": QColor(255, 86, 65),
        "Y": QColor(85, 235, 105),
        "Z": QColor(92, 145, 255),
    }

    def __init__(self, viewport: "QRhiViewportWidget") -> None:
        super().__init__(viewport)
        self._viewport = viewport
        self.setFixedSize(96, 96)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setAutoFillBackground(False)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(6, 10, 16, 145))
        painter.drawRoundedRect(0, 0, self.width() - 1, self.height() - 1, 6, 6)
        origin = QPointF(38.0, 58.0)
        length = 30.0
        axes = sorted(
            _orientation_axis_projection(
                self._viewport._camera.yaw,
                self._viewport._camera.pitch,
            ),
            key=lambda item: item[3],
        )
        for label, dx, dy, depth in axes:
            color = QColor(self._COLORS[label])
            alpha = 115 if depth > 0.0 else 235
            color.setAlpha(alpha)
            end = QPointF(origin.x() + dx * length, origin.y() + dy * length)
            painter.setPen(QPen(color, 4.0, Qt.PenStyle.SolidLine,
                                Qt.PenCapStyle.RoundCap))
            painter.drawLine(origin, end)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(color)
            painter.drawEllipse(end, 4.5, 4.5)
            painter.setPen(color)
            painter.drawText(end + QPointF(6.0, -5.0), label)


class QRhiViewportWidget(QRhiWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        api = _choose_api()
        if api is not None:
            self.setApi(api)
        self.setMinimumSize(480, 360)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._renderer = QRhiSurfaceRenderer()
        self._performance_governor = ViewportPerformanceGovernor()
        self._refinement_cb = None
        self._renderer_ready = False
        # orbit camera: owns view state and the working-scale envelope (the
        # grid stays a truthful world-space snap reference; the camera and
        # tool floors scale with camera.view_scale, never the grid)
        self._camera = ViewportCamera()
        self._bg = (0.07, 0.08, 0.10)
        self._show_grid = True
        self._grid_spacing = 1.0
        self._default_grid_spacing = 1.0
        self._sdf_opacity = 1.0
        self._gizmo_visible = True
        self._components_visible = False
        self._grid_plane = 0  # 0=XY, 1=XZ, 2=YZ
        self._view_anim = None  # (start_yaw, start_pitch, dyaw, dpitch)
        self._view_anim_clock = QElapsedTimer()
        self._view_anim_timer = QTimer(self)
        self._view_anim_timer.timeout.connect(self._advance_view_anim)
        self._view_panel = None
        self._last_pos = None
        self._press_pos = None
        self._tree = None        # SDFTree, for CPU click-picking
        self._scene_center = None
        self._scene_radius = 0.0
        self._selected_id = 0
        self._selected_anchor_cache = None
        self._hovered_id = 0
        self._last_hover_pick_t = 0.0
        self._move_active = False
        self._move_handle = None
        self._move_start_grid = None       # grid point under cursor at drag start
        self._move_preview_delta = (0.0, 0.0, 0.0)
        self._move_commit_delta = (0.0, 0.0, 0.0)
        # rotate tool state (grid-plane rotation around the object pivot)
        self._rotate_active = False
        self._rotate_handle = None
        self._rotate_pivot = None
        self._rotate_axis = "z"
        self._rotate_start_angle = None
        self._rotate_preview_deg = 0.0
        # extrude / revolve tool state (2D profile -> 3D solid)
        self._extrude_active = False
        self._extrude_handle = None
        self._extrude_origin = None
        self._extrude_normal = None
        self._extrude_start_h = None
        self._extrude_height = 0.0
        self._revolve_active = False
        self._revolve_handle = None
        self._revolve_axis_name = "v"
        self._revolve_axis_label = "Y axis"
        self._revolve_phase = "axis"
        self._revolve_section_origin = None
        self._revolve_section_normal = None
        self._revolve_section_center = None
        self._revolve_axis_start = None
        self._revolve_axis_end = None
        self._revolve_origin = None
        self._revolve_axis = None
        self._revolve_radial = None
        self._revolve_start_angle = None
        self._revolve_last_preview_ms = 0.0
        self._revolve_last_preview_deg = None
        self._revolve_deg = 0.0
        self._boundary_region_selected = False
        self._committed_surface_scene = None   # last committed scene (restore target)
        self._committed_scene_payload = None
        self._preview_kind = None
        self._boolean_preview_commit_pending = False
        # drag/point create tool (Draw button -> viewport_create_requested);
        # also carries the boundary-cutter routing consumed by main_window
        self._create_tool = CreateTool(self)
        # hover/select boundary tool (BoundaryRegion button)
        self._boundary_tool = BoundaryTool(self)
        self._boundary_hover_id = 0
        self._working_unit = DEFAULT_LENGTH_UNIT  # unit for bare typed sizes
        self._snap_enabled = True  # snap grid-plane points to grid spacing
        self._command_panel = None
        self._orientation_widget = None
        # render FPS overlay (measures real frame cadence in render())
        self._fps_ema = 0.0
        self._last_frame_t = None
        self._last_render_submit_t = None
        self._fps_label_t = 0.0
        self._fps_label = None
        # Event-driven rendering: state changes mark the viewport dirty and
        # schedule a single coalesced Qt update. Idle scenes do not poll.
        self._dirty_flag = False
        self._update_pending = False
        self._dirty = True
        # Interactive-quality state: while the camera/tool is actively moving the
        # renderer uses a cheaper per-pixel budget (see u_interacting); a short
        # idle delay then triggers one full-quality frame once motion stops.
        self._interacting = False
        self._interaction_idle = QTimer(self)
        self._interaction_idle.setSingleShot(True)
        self._interaction_idle.timeout.connect(self._end_interaction)
        self._idle_refine_timer = QTimer(self)
        self._idle_refine_timer.setSingleShot(True)
        self._idle_refine_timer.timeout.connect(self._advance_idle_refinement)
        self.setMouseTracking(True)  # deliver hover moves for the coord readout
        self._readout_label = None
        self._build_command_panel()
        self._build_view_panel()
        self._build_orientation_widget()
        self._build_readout_label()
        self._build_error_label()
        self._build_fps_label()
        signals.viewport_create_requested.connect(self.begin_create_tool)
        signals.working_unit_changed.connect(self._set_working_unit)
        signals.log_message.connect(self._on_log_message)

    def _set_working_unit(self, unit) -> None:
        """Full workspace rescale: snap grid to one working unit and reproduce
        the familiar startup framing at the new scale. Committed geometry is
        world-space and never moves; only the camera does."""
        self._working_unit = unit
        self._camera.view_scale = float(unit.factor)
        self._camera.reframe_to_working_scale()
        self._default_grid_spacing = float(unit.factor)
        self._grid_spacing = float(unit.factor)
        self._dirty = True

    def _build_command_panel(self) -> None:
        """The bottom-center viewport command overlay (recovered from the old
        viewport). Holds the Move button for now; more tools to follow."""
        panel = QFrame(self)
        panel.setObjectName("viewportCommandPanel")
        panel.setStyleSheet(
            "QFrame#viewportCommandPanel {"
            " background: rgba(6, 10, 16, 185);"
            " border: 1px solid rgba(120, 210, 255, 130);"
            " border-radius: 4px; }"
            "QPushButton {"
            " color: #e9f8ff; background: rgba(20, 145, 190, 95);"
            " border: 1px solid rgba(160, 230, 255, 120);"
            " border-radius: 4px; padding: 5px 10px; font: 12px sans-serif; }"
            "QPushButton:hover { background: rgba(30, 170, 215, 145); }"
            "QPushButton:pressed { background: rgba(10, 100, 150, 170); }"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setHorizontalSpacing(6)
        layout.setVerticalSpacing(6)
        move_button = QPushButton("Move", panel)
        move_button.setObjectName("viewportMoveButton")
        move_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        move_button.setToolTip(
            "Select one Scene object, then drag it in the viewport")
        move_button.clicked.connect(signals.viewport_move_tool_requested.emit)
        layout.addWidget(move_button, 0, 0)
        rotate_button = QPushButton("Rotate", panel)
        rotate_button.setObjectName("viewportRotateButton")
        rotate_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        rotate_button.setToolTip(
            "Select one Scene object, then drag to rotate it on the grid plane")
        rotate_button.clicked.connect(signals.viewport_rotate_tool_requested.emit)
        layout.addWidget(rotate_button, 0, 1)

        boundary_button = QPushButton("BoundaryRegion", panel)
        boundary_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        boundary_button.setToolTip(
            "Tag a FluidDomain boundary patch under the cursor")
        boundary_button.clicked.connect(
            signals.viewport_boundary_tool_requested.emit)
        layout.addWidget(boundary_button, 1, 0)

        cutter_button = QPushButton("BoundaryCutter", panel)
        cutter_button.setObjectName("viewportBoundaryCutterButton")
        cutter_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        cutter_button.setToolTip(
            "Draw any shape as a knife to split the selected BoundaryRegion "
            "in two (the knife never becomes a scene object)")
        cutter_menu = QMenu(cutter_button)
        for label, shape in self._CUTTER_KINDS:
            cutter_menu.addAction(label).triggered.connect(
                lambda _=False, s=shape: self.begin_boundary_cutter_tool(s))
        cutter_button.setMenu(cutter_menu)
        layout.addWidget(cutter_button, 1, 1)

        self._cutter_buttons = (cutter_button,)
        self._update_cutter_buttons_enabled()
        panel.adjustSize()
        self._command_panel = panel
        self._position_command_panel()

    def _update_cutter_buttons_enabled(self) -> None:
        for button in getattr(self, "_cutter_buttons", ()):
            button.setEnabled(self._boundary_region_selected)

    def _build_readout_label(self) -> None:
        """Top-center overlay: live cursor coordinates on the active grid plane,
        and tool-specific info (create size, move delta, rotation angle/axis)."""
        label = QLabel("", self)
        label.setObjectName("viewportReadout")
        label.setStyleSheet(
            "QLabel#viewportReadout {"
            " color: #e9f8ff; background: rgba(6, 10, 16, 175);"
            " border: 1px solid rgba(120, 210, 255, 90);"
            " border-radius: 3px; padding: 3px 9px;"
            " font: 12px monospace; }"
        )
        label.setVisible(False)
        self._readout_label = label
        self._update_readout()

    def _position_readout_label(self) -> None:
        p = getattr(self, "_readout_label", None)
        if p is None:
            return
        p.adjustSize()
        p.move(max(8, (self.width() - p.width()) // 2), 8)
        p.raise_()

    def _update_readout(self, hover=None) -> None:
        label = getattr(self, "_readout_label", None)
        if label is None:
            return
        plane = self.reference_plane_label
        if self._extrude_active:
            text = f"Extrude   height {self._extrude_height:+.3f}"
        elif self._revolve_active:
            if self._revolve_phase == "axis":
                if self._revolve_axis_start is not None and self._revolve_axis_end is not None:
                    delta = self._revolve_axis_end - self._revolve_axis_start
                    length = float(np.linalg.norm(delta))
                    text = f"Revolve   axis vector length {length:.3f}"
                else:
                    text = "Revolve   draw axis vector"
            else:
                text = (
                    f"Revolve {self._revolve_axis_label}   "
                    f"{self._revolve_deg:+.1f}°"
                )
        elif self._rotate_active and self._rotate_pivot is not None:
            text = (f"Rotate   {self._rotate_preview_deg:+.1f}°   "
                    f"axis {self._rotate_axis.upper()}")
        elif self._move_active:
            dx, dy, dz = self._move_preview_delta
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            text = (f"Move   Δ({dx:+.3f}, {dy:+.3f}, {dz:+.3f})   "
                    f"|Δ|={dist:.3f}")
        elif self._create_tool.anchor is not None:
            typed = self._create_tool.dimension_input or "_"
            text = (f"{self._create_tool.kind}   size: {typed} "
                    f"[{self._working_unit.key}]   "
                    "(Enter creates, or click end)")
        elif self._create_tool.points is not None:
            text = (f"{self._create_tool.kind}   points: {len(self._create_tool.points)}   "
                    "(Enter creates, Backspace undoes)")
        elif self._create_tool.kind is not None and \
                self._create_tool.start_world is not None and hover is not None:
            sx, sy, sz = self._create_tool.start_world
            text = (f"{self._create_tool.kind}   "
                    f"size ({abs(hover[0]-sx):.3f}, {abs(hover[1]-sy):.3f}, "
                    f"{abs(hover[2]-sz):.3f})")
        elif hover is not None:
            text = (f"[{plane}]   x={hover[0]:+.3f}  "
                    f"y={hover[1]:+.3f}  z={hover[2]:+.3f}")
        else:
            text = f"[{plane}]"
        label.setText(text)
        label.setVisible(True)
        self._position_readout_label()

    def _build_error_label(self) -> None:
        """Transient center overlay surfacing warnings/errors in the viewport
        (auto-hides), so tool feedback isn't buried in the log panel."""
        label = QLabel("", self)
        label.setObjectName("viewportError")
        label.setStyleSheet(
            "QLabel#viewportError {"
            " color: #fff2f0; background: rgba(120, 24, 18, 220);"
            " border: 1px solid rgba(255, 150, 130, 200);"
            " border-radius: 4px; padding: 5px 12px;"
            " font: 12px sans-serif; }"
        )
        label.setWordWrap(True)
        label.setVisible(False)
        self._error_label = label
        self._error_timer = QTimer(self)
        self._error_timer.setSingleShot(True)
        self._error_timer.timeout.connect(self._clear_viewport_error)

    def _on_log_message(self, level: str, message: str) -> None:
        if str(level).lower() in ("warning", "error"):
            self._show_viewport_error(message)

    def _show_viewport_error(self, message: str) -> None:
        label = getattr(self, "_error_label", None)
        if label is None:
            return
        label.setText(str(message))
        label.adjustSize()
        label.setMaximumWidth(max(240, self.width() - 40))
        label.adjustSize()
        label.move(max(8, (self.width() - label.width()) // 2),
                   max(40, self.height() // 3))
        label.setVisible(True)
        label.raise_()
        self._error_timer.start(4000)

    def _clear_viewport_error(self) -> None:
        label = getattr(self, "_error_label", None)
        if label is not None:
            label.setVisible(False)

    def _build_fps_label(self) -> None:
        """Small top-left overlay showing the viewport's real render FPS."""
        label = QLabel("FPS: --", self)
        label.setObjectName("viewportFpsLabel")
        label.setStyleSheet(
            "QLabel#viewportFpsLabel {"
            " color: #aef6c8; background: rgba(6, 10, 16, 160);"
            " border: 1px solid rgba(120, 210, 255, 90);"
            " border-radius: 3px; padding: 2px 6px;"
            " font: 11px monospace; }"
        )
        label.adjustSize()
        label.move(8, 8)
        self._fps_label = label

    # (grid_plane, yaw_deg, pitch_deg) per reference view
    _VIEW_TABLE = {
        "top": (0, -90.0, 90.0),
        "front": (1, -90.0, 0.0),
        "side": (2, 0.0, 0.0),
        "iso": (0, 35.0, 28.0),
    }

    def _build_view_panel(self) -> None:
        """Bottom-right overlay: 3D / {x,y} / {x,z} / {y,z} reference-view buttons.
        Each sets the active grid (draw) plane and flies the camera to it."""
        panel = QFrame(self)
        panel.setObjectName("viewportViewPanel")
        panel.setStyleSheet(
            "QFrame#viewportViewPanel {"
            " background: rgba(6, 10, 16, 170);"
            " border: 1px solid rgba(120, 210, 255, 110); border-radius: 4px; }"
            "QPushButton {"
            " color: #e9f8ff; background: rgba(20, 145, 190, 80);"
            " border: 1px solid rgba(160, 230, 255, 110);"
            " border-radius: 3px; padding: 3px 8px; font: 11px sans-serif; }"
            "QPushButton:hover { background: rgba(30, 170, 215, 140); }"
        )
        layout = QGridLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setHorizontalSpacing(4)
        layout.setVerticalSpacing(4)
        for col, (label, view) in enumerate(
                (("3D", "iso"), ("{x,y}", "top"),
                 ("{x,z}", "front"), ("{y,z}", "side"))):
            button = QPushButton(label, panel)
            button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
            button.setToolTip(f"{label} view (key {col + 1})")
            button.clicked.connect(
                lambda _=False, v=view: self.set_reference_view(v))
            layout.addWidget(button, 0, col)
        panel.adjustSize()
        self._view_panel = panel
        self._position_view_panel()

    def _position_view_panel(self) -> None:
        p = self._view_panel
        if p is None:
            return
        orientation = self._orientation_widget
        if orientation is not None:
            x = orientation.x() + orientation.width() + 8
            y = orientation.y() + (orientation.height() - p.height()) // 2
        else:
            x = 114
            y = self.height() - p.height() - 10
        p.move(min(max(8, x), max(8, self.width() - p.width() - 8)),
               max(8, y))
        p.raise_()

    def _build_orientation_widget(self) -> None:
        widget = _OrientationWidget(self)
        self._orientation_widget = widget
        self._position_orientation_widget()

    def _position_orientation_widget(self) -> None:
        widget = self._orientation_widget
        if widget is None:
            return
        widget.move(10, max(10, self.height() - widget.height() - 10))
        widget.raise_()
        self._position_view_panel()

    def set_reference_view(self, view: str) -> None:
        """Switch the active grid/draw plane and fly the camera to that view."""
        if view not in self._VIEW_TABLE:
            raise ValueError(f"unknown reference view: {view}")
        plane, yaw_deg, pitch_deg = self._VIEW_TABLE[view]
        self._grid_plane = plane
        self._animate_view_to(math.radians(yaw_deg), math.radians(pitch_deg))

    def _animate_view_to(self, target_yaw: float, target_pitch: float) -> None:
        camera = self._camera
        dyaw = ((target_yaw - camera.yaw + math.pi) % (2 * math.pi)) - math.pi
        dpitch = target_pitch - camera.pitch
        if max(abs(dyaw), abs(dpitch)) <= 1e-6:
            self._dirty = True
            return
        self._view_anim = (camera.yaw, camera.pitch, dyaw, dpitch)
        self._view_anim_clock.restart()
        self._view_anim_timer.start(16)

    def _advance_view_anim(self) -> None:
        if self._view_anim is None:
            self._view_anim_timer.stop()
            return
        t = min(1.0, self._view_anim_clock.elapsed() / 260.0)
        eased = t * t * (3.0 - 2.0 * t)  # smoothstep
        y0, p0, dy, dp = self._view_anim
        self._camera.yaw = y0 + dy * eased
        self._camera.pitch = p0 + dp * eased
        if t >= 1.0:
            self._view_anim = None
            self._view_anim_timer.stop()
            self._end_interaction()  # settle to a full-quality frame
        else:
            self._begin_interaction()

    def _position_command_panel(self) -> None:
        p = self._command_panel
        if p is None:
            return
        x = max(8, (self.width() - p.width()) // 2)
        y = max(8, self.height() - p.height() - 12)
        p.move(x, y)
        p.raise_()

    @property
    def _dirty(self) -> bool:
        return bool(getattr(self, "_dirty_flag", False))

    @_dirty.setter
    def _dirty(self, dirty: bool) -> None:
        self._dirty_flag = bool(dirty)
        if dirty:
            self._schedule_update()

    def _schedule_update(self) -> None:
        if self._orientation_widget is not None:
            self._orientation_widget.update()
        if self._update_pending:
            return
        self._update_pending = True
        self.update()

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._position_command_panel()
        self._position_view_panel()
        self._position_orientation_widget()
        self._position_readout_label()
        self._dirty = True

    def _request_render(self) -> None:
        self._dirty = True

    def _begin_interaction(self) -> None:
        """Mark the viewport as actively moving (cheap interactive frames) and
        (re)arm the idle timer that snaps back to a full-quality frame shortly
        after motion stops."""
        self._performance_governor.begin_interaction()
        self._idle_refine_timer.stop()
        self._interacting = True
        mark_interaction = getattr(self._renderer, "mark_interaction", None)
        if callable(mark_interaction):
            mark_interaction()
        self._dirty = True
        self._interaction_idle.start(140)

    def _end_interaction(self) -> None:
        """Motion stopped: render one full-quality frame."""
        if self._interacting:
            self._interacting = False
            self._performance_governor.end_interaction()
            self._dirty = True
            self._schedule_idle_refinement()

    # -- public API ----------------------------------------------------------

    def set_scene(self, surface_scene) -> None:
        self._renderer.set_surface_scene(surface_scene)
        self._dirty = True

    def frame_target(self, target=(0.0, 0.0, 0.0), distance: float = 6.0) -> None:
        self._camera.frame_target(target, distance)
        self._dirty = True

    # --- ViewportWidget drop-in compatibility ------------------------------
    # The RENDER methods below are real. The in-3D-viewport TOOLS (gizmos,
    # drawing, grid, click-selection, boundary tools, boolean preview) are
    # intentionally DROPPED in this QRhi refactor — the document stays fully
    # editable via the side panels and the viewport renders it live. Early-stage
    # feature loss by design; rebuild on the clean QRhi foundation later.

    def set_scene_artifact(self, tree, surface_scene=None, timings=None) -> None:
        """The app's render hook: show the freshly built scene."""
        self._tree = tree  # kept for CPU click-picking
        self._committed_scene_payload = surface_scene
        if timings is not None:
            self._performance_governor.record_artifact_ms(
                float(getattr(timings, "total_ms", 0.0)),
                int(
                    getattr(
                        timings,
                        "total_object_count",
                        int(getattr(timings, "exact_object_count", 0)),
                    )
                ),
            )
            self._performance_governor.record_render_wait_ms(
                float(getattr(timings, "render_wait_ms", 0.0))
            )
        self._update_scene_bounds(tree)
        self._refresh_selected_anchor_cache()
        self._preview_kind = None
        self._boolean_preview_commit_pending = False
        self._move_commit_delta = (0.0, 0.0, 0.0)
        if self._publish_surface_scene(surface_scene):
            self._committed_surface_scene = surface_scene
        self._schedule_idle_refinement()

    def set_refinement_callback(self, cb) -> None:
        self._refinement_cb = cb
        timer = getattr(self, "_idle_refine_timer", None)
        if cb is None and timer is not None:
            timer.stop()

    def render_budget_for_tree(self, tree) -> ViewportRenderBudget:
        camera = self._camera_values()
        return self._performance_governor.budget_for_tree(
            tree,
            selected_object_id=self._selected_id,
            edited_object_ids=self._active_edit_object_ids(),
            hovered_object_id=self._hovered_id,
            camera_position=camera.get("u_camera_position"),
        )

    def _schedule_idle_refinement(self) -> None:
        if (
            getattr(self, "_interacting", False)
            or getattr(self, "_refinement_cb", None) is None
        ):
            return
        governor = getattr(self, "_performance_governor", None)
        if governor is None or not governor.can_refine_idle(getattr(self, "_tree", None)):
            return
        timer = getattr(self, "_idle_refine_timer", None)
        if timer is not None:
            timer.start(int(governor.config.idle_refine_interval_ms))

    def _advance_idle_refinement(self) -> None:
        if (
            getattr(self, "_interacting", False)
            or getattr(self, "_refinement_cb", None) is None
        ):
            return
        governor = getattr(self, "_performance_governor", None)
        if governor is None:
            return
        if not governor.advance_idle_refinement(getattr(self, "_tree", None)):
            return
        self._refinement_cb()

    def _update_scene_bounds(self, tree) -> None:
        root = getattr(tree, "root", None)
        if root is None:
            self._scene_center = None
            self._scene_radius = 0.0
            return
        try:
            box = root.bounding_box()
        except Exception:
            self._scene_center = None
            self._scene_radius = 0.0
            return
        self._scene_center = np.array(
            [
                (box.x_min + box.x_max) * 0.5,
                (box.y_min + box.y_max) * 0.5,
                (box.z_min + box.z_max) * 0.5,
            ],
            dtype=np.float64,
        )
        half = np.array(
            [
                (box.x_max - box.x_min) * 0.5,
                (box.y_max - box.y_min) * 0.5,
                (box.z_max - box.z_min) * 0.5,
            ],
            dtype=np.float64,
        )
        self._scene_radius = max(float(np.linalg.norm(half)), 1.0)

    def show_scene_preview(self, surface_scene, *, preview_kind: str = "tool") -> None:
        """Render a non-committing ghost during a move/rotate drag (main_window
        builds it). The committed scene is restored on cancel/commit."""
        self._preview_kind = preview_kind
        self._publish_surface_scene(surface_scene)
        self._dirty = True

    def _restore_committed_scene(self) -> None:
        self._preview_kind = None
        self._publish_surface_scene(self._committed_surface_scene)
        self._dirty = True

    def _publish_surface_scene(self, surface_scene) -> bool:
        visible_scene = self._visible_surface_scene(surface_scene)
        failed = tuple(getattr(visible_scene, "failed_messages", ()) or ())
        has_geometry = bool(getattr(visible_scene, "has_geometry", False))
        if failed:
            signals.log_message.emit(
                "warning",
                "Viewport surface update incomplete; failed objects are hidden. "
                + " | ".join(failed[:2]),
            )
            if not has_geometry:
                return False
        self.set_scene(visible_scene)
        return True

    def _visible_surface_scene(self, surface_scene):
        if surface_scene is None:
            return None
        highlight_id = self._boundary_hover_id or self._selected_id
        return surface_scene.with_components_visible(
            self._components_visible
        ).with_selected_highlight(highlight_id)

    def frame_box(self, box) -> None:
        self._camera.frame_box(box)
        self._dirty = True

    def set_background_color(self, color) -> None:
        self._bg = tuple(float(c) for c in color)
        self._dirty = True

    # --- move tool (real, deferred preview) ---
    def begin_move_tool(self, handle) -> None:
        """Enter move mode for a document handle. Left-drag on the grid plane
        previews a snapped translation; release commits it once. Esc cancels."""
        self.cancel_create_tool()
        self.end_rotate_tool()
        self._boundary_tool.cancel()
        self._move_handle = handle
        self._move_active = True
        self._move_start_grid = None
        self._move_preview_delta = (0.0, 0.0, 0.0)
        self._move_commit_delta = (0.0, 0.0, 0.0)
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setFocus()
        signals.log_message.emit(
            "info",
            "Drag the selected object on the "
            f"{self.reference_plane_label} grid. Release to apply, Esc cancels.",
        )

    def end_move_tool(self, *a, preserve_preview: bool = False, **k) -> None:
        self._move_active = False
        self._move_handle = None
        self._move_start_grid = None
        if preserve_preview:
            self._move_commit_delta = self._move_preview_delta
        else:
            self._move_preview_delta = (0.0, 0.0, 0.0)
            self._move_commit_delta = (0.0, 0.0, 0.0)
        self.unsetCursor()

    # --- rotate tool (real, deferred preview) ---
    _PLANE_AXIS_INDICES = {0: (0, 1), 1: (0, 2), 2: (1, 2)}  # in-plane (u, v)
    _PLANE_NORMAL_NAME = {0: "z", 1: "y", 2: "x"}

    def begin_rotate_tool(self, handle) -> None:
        """Enter rotate mode: left-drag spins the object around the grid-plane
        normal through its pivot, snapped to 15deg; release commits once."""
        self.end_move_tool()
        self.cancel_create_tool()
        self._boundary_tool.cancel()
        pivot = self._selected_anchor()
        if pivot is None:
            signals.log_message.emit(
                "warning", "Select a Scene object before using Rotate.")
            return
        self._rotate_active = True
        self._rotate_handle = handle
        self._rotate_pivot = tuple(float(v) for v in pivot)
        self._rotate_axis = self._PLANE_NORMAL_NAME[self._grid_plane]
        self._rotate_start_angle = None
        self._rotate_preview_deg = 0.0
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        signals.log_message.emit(
            "info",
            "Grab a colored ring (X/Y/Z) and drag to rotate about that axis. "
            "Alt = free angle, Esc cancels.",
        )

    def end_rotate_tool(self) -> None:
        self._rotate_active = False
        self._rotate_handle = None
        self._rotate_pivot = None
        self._rotate_start_angle = None
        self._rotate_preview_deg = 0.0
        self.unsetCursor()

    # in-plane (u, v) basis for rotation about each axis (u x v = +axis)
    _AXIS_BASIS = {
        "x": (np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])),
        "y": (np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0])),
        "z": (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])),
    }
    _AXIS_NORMAL = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }

    def _gizmo_length(self) -> float:
        camera = self._camera
        return max(camera.distance * 0.18, 0.4 * camera.view_scale)

    def _ray_plane_hit(self, pos, axis: str):
        """Intersect the cursor ray with the plane through the rotate pivot whose
        normal is `axis`; return the world hit point or None."""
        if self._rotate_pivot is None:
            return None
        origin, direction = self._screen_ray(pos)
        normal = self._AXIS_NORMAL[axis]
        denom = float(np.dot(direction, normal))
        if abs(denom) < 1e-9:
            return None
        pivot = np.array(self._rotate_pivot)
        t = float(np.dot(pivot - origin, normal)) / denom
        if t <= 0.0:
            return None
        return origin + direction * t

    def _axis_angle(self, pos, axis: str):
        """Angle (radians) of the cursor about the pivot in `axis`'s ring plane."""
        hit = self._ray_plane_hit(pos, axis)
        if hit is None:
            return None
        u, v = self._AXIS_BASIS[axis]
        rel = hit - np.array(self._rotate_pivot)
        if float(np.dot(rel, rel)) < 1e-12:
            return None
        return math.atan2(float(np.dot(rel, v)), float(np.dot(rel, u)))

    def _pick_rotation_axis(self, pos) -> str:
        """Choose the ring (x/y/z) whose circle the cursor ray passes nearest.
        Falls back to the current grid-plane normal when no ring is close."""
        radius = self._gizmo_length() * 1.5
        best, best_score = None, radius * 0.4  # tolerance band
        for axis in ("x", "y", "z"):
            hit = self._ray_plane_hit(pos, axis)
            if hit is None:
                continue
            r = float(np.linalg.norm(hit - np.array(self._rotate_pivot)))
            score = abs(r - radius)
            if score < best_score:
                best, best_score = axis, score
        return best or self._PLANE_NORMAL_NAME[self._grid_plane]

    # --- extrude / revolve tools (real, deferred preview) ---
    def _end_extrude_tool(self) -> None:
        self._extrude_active = False
        self._extrude_handle = None
        self._extrude_origin = self._extrude_normal = None
        self._extrude_start_h = None
        self._extrude_height = 0.0
        self.unsetCursor()

    def _end_revolve_tool(self) -> None:
        self._revolve_active = False
        self._revolve_handle = None
        self._revolve_axis_name = "v"
        self._revolve_axis_label = "Y axis"
        self._revolve_phase = "axis"
        self._revolve_section_origin = None
        self._revolve_section_normal = None
        self._revolve_section_center = None
        self._revolve_axis_start = None
        self._revolve_axis_end = None
        self._revolve_origin = self._revolve_axis = self._revolve_radial = None
        self._revolve_start_angle = None
        self._revolve_last_preview_ms = 0.0
        self._revolve_last_preview_deg = None
        self._revolve_deg = 0.0
        self.unsetCursor()

    def begin_extrude_tool(self, handle, node) -> None:
        """Drag along the profile normal to set extrude height; release applies."""
        self.end_move_tool(); self.end_rotate_tool(); self.cancel_create_tool()
        self._end_revolve_tool()
        self._extrude_active = True
        self._extrude_handle = handle
        self._extrude_origin = np.array(node.origin, dtype=np.float64)
        n = np.array(node.normal, dtype=np.float64)
        self._extrude_normal = n / max(np.linalg.norm(n), 1e-9)
        self._extrude_start_h = None
        self._extrude_height = 0.0
        self.setCursor(Qt.CursorShape.SizeVerCursor)
        self.setFocus()
        signals.log_message.emit(
            "info", "Drag to set extrude height. Release applies, Esc cancels.")

    def begin_revolve_tool(self, handle, node, axis_name: str = "custom") -> None:
        """Draw an axis vector, then drag the angle gizmo to create a revolve."""
        self.end_move_tool(); self.end_rotate_tool(); self.cancel_create_tool()
        self._end_extrude_tool()
        axis_name = axis_name if axis_name in {"u", "v"} else "custom"
        self._revolve_active = True
        self._revolve_handle = handle
        self._revolve_axis_name = axis_name
        self._revolve_phase = "axis" if axis_name == "custom" else "angle"
        self._revolve_section_origin = np.array(node.origin, dtype=np.float64)
        normal = np.array(node.normal, dtype=np.float64)
        self._revolve_section_normal = normal / max(np.linalg.norm(normal), 1e-9)
        self._revolve_section_center = self._section_profile_center(node)
        self._revolve_axis_start = None
        self._revolve_axis_end = None
        if axis_name == "custom":
            self._revolve_origin = None
            self._revolve_axis = None
            self._revolve_radial = None
            self._revolve_axis_label = "drawn vector"
        else:
            self._revolve_origin = np.array(node.origin, dtype=np.float64)
            if axis_name == "u":
                axis = np.array(node.axis_u, dtype=np.float64)
                radial = np.array(node.axis_v, dtype=np.float64)
            else:
                axis = np.array(node.axis_v, dtype=np.float64)
                radial = np.array(node.axis_u, dtype=np.float64)
            self._revolve_axis = axis / max(np.linalg.norm(axis), 1e-9)
            self._revolve_radial = radial / max(np.linalg.norm(radial), 1e-9)
            self._revolve_axis_label = world_axis_label(tuple(self._revolve_axis))
        self._revolve_start_angle = None
        self._revolve_last_preview_ms = 0.0
        self._revolve_last_preview_deg = None
        self._revolve_deg = 0.0
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        if self._revolve_phase == "axis":
            signals.log_message.emit(
                "info",
                "Drag on the selected profile plane to draw the revolve axis vector.",
            )
        else:
            signals.log_message.emit(
                "info",
                f"Drag to revolve around {self._revolve_axis_label}. "
                "Release applies, Esc cancels.",
            )

    def _section_profile_center(self, node):
        profile = getattr(node, "profile", None)
        if profile is None:
            return np.array(node.origin, dtype=np.float64)
        try:
            u_min, u_max, v_min, v_max = profile.bounds()
        except Exception:  # noqa: BLE001 - viewport hint only.
            return np.array(node.origin, dtype=np.float64)
        origin = np.array(node.origin, dtype=np.float64)
        axis_u = np.array(node.axis_u, dtype=np.float64)
        axis_v = np.array(node.axis_v, dtype=np.float64)
        return origin + 0.5 * (u_min + u_max) * axis_u + 0.5 * (v_min + v_max) * axis_v

    def _ray_plane_point(self, pos, plane_point, plane_normal):
        """Generic ray vs. plane(point, normal) intersection, or None."""
        origin, direction = self._screen_ray(pos)
        denom = float(np.dot(direction, plane_normal))
        if abs(denom) < 1e-9:
            return None
        t = float(np.dot(plane_point - origin, plane_normal)) / denom
        if t <= 0.0:
            return None
        return origin + direction * t

    def _revolve_plane_point(self, pos):
        if self._revolve_section_origin is None or self._revolve_section_normal is None:
            return None
        return self._ray_plane_point(
            pos,
            self._revolve_section_origin,
            self._revolve_section_normal,
        )

    def _set_revolve_axis_vector(self, start, end) -> bool:
        if start is None or end is None or self._revolve_section_normal is None:
            return False
        start = np.asarray(start, dtype=np.float64)
        end = np.asarray(end, dtype=np.float64)
        axis = end - start
        axis_length = float(np.linalg.norm(axis))
        if axis_length <= max(self._grid_spacing * 0.1, 1.0e-4):
            return False
        axis = axis / axis_length
        center = (
            self._revolve_section_center
            if self._revolve_section_center is not None
            else start
        )
        radial = np.asarray(center, dtype=np.float64) - start
        radial = radial - axis * float(np.dot(radial, axis))
        radial_length = float(np.linalg.norm(radial))
        if radial_length <= 1.0e-9:
            normal = np.asarray(self._revolve_section_normal, dtype=np.float64)
            radial = np.cross(axis, normal)
            radial_length = float(np.linalg.norm(radial))
        if radial_length <= 1.0e-9:
            return False
        radial = radial / radial_length
        self._revolve_origin = start
        self._revolve_axis = axis
        self._revolve_radial = radial
        self._revolve_axis_label = world_axis_label(tuple(axis))
        return True

    def _finish_revolve_axis_phase(self) -> bool:
        if not self._set_revolve_axis_vector(
            self._revolve_axis_start,
            self._revolve_axis_end,
        ):
            signals.log_message.emit("warning", "Draw a longer revolve axis vector.")
            return False
        self._revolve_phase = "angle"
        self._revolve_start_angle = None
        self._revolve_last_preview_ms = 0.0
        self._revolve_last_preview_deg = None
        self._revolve_deg = 0.0
        self._update_readout()
        self._dirty = True
        signals.log_message.emit(
            "info",
            f"Axis set to {self._revolve_axis_label}. Drag the angle gizmo; "
            "release applies, Esc cancels.",
        )
        return True

    def _project_ray_to_axis(self, pos, origin, axis):
        """Signed distance along `axis` of the closest point on the axis line to
        the cursor ray (used to read extrude height from a drag)."""
        ro, d = self._screen_ray(pos)
        w0 = origin - ro
        b = float(np.dot(axis, d))
        denom = 1.0 - b * b
        if abs(denom) < 1e-6:
            return float(np.dot(axis, w0))  # parallel: project directly
        return float((b * float(np.dot(d, w0)) - float(np.dot(axis, w0))) / denom)

    def _revolve_angle(self, pos):
        hit = self._ray_plane_point(pos, self._revolve_origin, self._revolve_axis)
        if hit is None:
            return None
        u = self._revolve_radial
        w = np.cross(self._revolve_axis, u)
        rel = hit - self._revolve_origin
        if float(np.dot(rel, rel)) < 1e-12:
            return None
        return math.atan2(float(np.dot(rel, w)), float(np.dot(rel, u)))

    # --- draw / create tool (state machine lives in create_tool.CreateTool) ---

    def _plane_id(self) -> str:
        return ("xy", "xz", "yz")[self._grid_plane]

    def begin_create_tool(self, kind: str) -> None:
        self._create_tool.begin(kind)

    def cancel_create_tool(self) -> None:
        self._create_tool.cancel()

    # --- snapping (real) ---
    def set_snap_enabled(self, enabled: bool) -> None:
        """Toolbar 'Snap' toggle: snap grid-plane points to the grid spacing."""
        self._snap_enabled = bool(enabled)

    def _snap_world(self, point, modifiers=Qt.KeyboardModifier.NoModifier):
        """Snap a grid-plane world point to the grid spacing on the active
        plane's two axes. Holding Alt bypasses snapping (fine placement)."""
        if point is None:
            return None
        if not self._snap_enabled or (modifiers & Qt.KeyboardModifier.AltModifier):
            return point
        spacing = max(self._grid_spacing, 1e-9)
        const_axis = {0: 2, 1: 1, 2: 0}[self._grid_plane]
        out = list(point)
        for axis in range(3):
            if axis == const_axis:
                continue
            out[axis] = round(out[axis] / spacing) * spacing
        return (float(out[0]), float(out[1]), float(out[2]))

    # --- grid controls (real) ---
    @property
    def grid_spacing(self) -> float:
        return self._grid_spacing

    @property
    def reference_plane_label(self) -> str:
        return ("XY", "XZ", "YZ")[self._grid_plane]

    def set_grid_visible(self, visible: bool) -> None:
        self._show_grid = bool(visible)
        self._dirty = True

    def set_grid_spacing(self, spacing: float) -> None:
        self._grid_spacing = max(
            float(spacing), 1e-4 * min(1.0, self._camera.view_scale)
        )
        self._dirty = True

    def configure_grid(self, box=None, dx=None, *a, **k) -> None:
        if dx is not None and dx > 0:
            self._grid_spacing = float(dx)
        self._dirty = True

    # value-returning stubs (safe defaults so main_window's logic doesn't crash)

    def has_scene_object_id(self, object_id: int) -> bool:
        scene = self._committed_surface_scene
        if scene is None:
            return False
        return any(
            int(surface.key.object_id) == int(object_id)
            for surface in scene.surfaces
        )

    def can_defer_committed_move(self, object_id: int) -> bool:
        return self.has_scene_object_id(object_id)

    def show_move_preview(
        self,
        object_id: int,
        delta: tuple[float, float, float],
    ) -> bool:
        scene = self._committed_surface_scene
        if scene is None:
            return False
        shifted: list[object] = []
        matched = False
        for surface in scene.surfaces:
            if int(surface.key.object_id) != int(object_id):
                shifted.append(surface)
                continue
            matched = True
            shifted.append(_translated_preview_surface(surface, delta))
        if not matched:
            return False
        preview = replace(scene, surfaces=tuple(shifted), build_ms=0.0)
        self.show_scene_preview(preview)
        return True

    def committed_surface_scene(self):
        return self._visible_surface_scene(self._committed_surface_scene)

    def paste_offset(self, *a):
        """Offset pasted/duplicated objects on the active grid plane so they
        don't stack exactly on the original. A grid-step diagonal nudge plus a
        small random jitter keeps repeated pastes visibly separated."""
        ui, vi = self._PLANE_AXIS_INDICES[self._grid_plane]
        step = max(self._grid_spacing, 0.1 * min(1.0, self._camera.view_scale))
        off = [0.0, 0.0, 0.0]
        off[ui] = step * (1.0 + random.uniform(-0.25, 0.25))
        off[vi] = step * (1.0 + random.uniform(-0.25, 0.25))
        return (off[0], off[1], off[2])

    def active_boundary_cutter_tool(self, *a):
        return self._create_tool.boundary_cutter

    def set_boundary_region_selection_entries(self, *a, **k) -> None:
        """main_window passes (selectors, normals, regions); we only need to know
        whether a BoundaryRegion is selected so the cutter buttons enable."""
        regions = a[2] if len(a) >= 3 else ()
        self._boundary_region_selected = bool(regions)
        self._update_cutter_buttons_enabled()

    # One cutter, any knife shape: lines/profiles extrude through the scene,
    # 3D shapes cut by their own volume (boundary_region_v2 §3).
    _CUTTER_KINDS = (
        ("Segment", "segment"),
        ("Polyline", "polyline"),
        ("Quadratic Bezier Polycurve", "quadratic_bezier_polycurve"),
        ("Sphere", "sphere"),
        ("Box", "box"),
        ("Cylinder", "cylinder"),
        ("Cone", "cone"),
    )

    def begin_boundary_cutter_tool(self, shape_kind: str) -> None:
        self._create_tool.begin_boundary_cutter(shape_kind)

    # --- boundary region hover/select tool (real; boundary_region_v2 §7) ---

    def begin_boundary_region_tool(self, root, selectors=()) -> None:
        self._boundary_tool.begin(root, selectors)

    def set_boundary_hover(self, owner_object_id, _outside_direction, _normal, _hit) -> None:
        """Track the hovered boundary owner (tints its chunk where one is
        visible; the patch-shell overlay is the primary visual)."""
        hover_id = int(owner_object_id or 0)
        if hover_id == self._boundary_hover_id:
            return
        self._boundary_hover_id = hover_id
        if self._committed_surface_scene is not None and self._preview_kind is None:
            self._publish_surface_scene(self._committed_surface_scene)
        self._dirty = True

    def show_boundary_patch_highlight(self, surfaces) -> None:
        """Overlay classifier-derived highlight surfaces (hovered patch, or
        the cutter's inside/outside split preview) on top of the committed
        scene. Accepts one surface, a sequence, or None/empty to clear."""
        committed = self._committed_surface_scene
        if surfaces is None:
            overlay = ()
        elif isinstance(surfaces, (tuple, list)):
            overlay = tuple(s for s in surfaces if s is not None)
        else:
            overlay = (surfaces,)
        if not overlay or committed is None:
            if self._preview_kind == "boundary_hover":
                self._restore_committed_scene()
            return
        scene = replace(
            committed,
            surfaces=(*committed.surfaces, *overlay),
            primary_object_ids=(
                committed.primary_object_ids
                | {int(surface.key.object_id) for surface in overlay}
            ),
            build_ms=0.0,
        )
        self.show_scene_preview(scene, preview_kind="boundary_hover")

    # no-op stubs for the dropped viewport tools
    def _noop(self, *a, **k) -> None:
        return None
    def apply_committed_move_preview(self, *a, **k) -> None:
        del a, k
    # --- boolean preview (real; main_window builds the combined render) ---
    def apply_committed_boolean_preview(self, *a, **k) -> None:
        if self._preview_kind == "boolean":
            self._boolean_preview_commit_pending = True

    def set_boolean_preview(self, *a, **k) -> None:
        # main_window pushes the combined ghost via show_scene_preview; nothing
        # to do viewport-side beyond that.
        return None

    def clear_boolean_preview(self) -> None:
        if self._preview_kind == "boolean":
            if self._boolean_preview_commit_pending:
                return
            self._restore_committed_scene()

    def set_sdf_opacity(self, opacity: float) -> None:
        self._sdf_opacity = max(0.0, min(1.0, float(opacity)))
        self._dirty = True

    def set_gizmo_visible(self, visible: bool) -> None:
        self._gizmo_visible = bool(visible)
        self._dirty = True

    def set_components_visible(self, visible: bool) -> None:
        visible = bool(visible)
        if self._components_visible == visible:
            return
        self._components_visible = visible
        self._publish_surface_scene(self._committed_surface_scene)
        self._dirty = True

    def set_scene_selection(self, node) -> None:
        """Sync the viewport highlight/gizmo with the tree-panel selection."""
        self._selected_id = int(getattr(node, "object_id", 0) or 0)
        self._selected_anchor_cache = self._anchor_from_node(node)
        self._publish_surface_scene(self._committed_surface_scene)
        self._dirty = True

    def _anchor_from_node(self, node):
        if node is None:
            return None
        try:
            bb = node.bounding_box()
        except Exception:
            return None
        return np.array(
            [
                (bb.x_min + bb.x_max) * 0.5,
                (bb.y_min + bb.y_max) * 0.5,
                (bb.z_min + bb.z_max) * 0.5,
            ],
            dtype=np.float64,
        )

    def _refresh_selected_anchor_cache(self) -> None:
        tree = self._tree
        if tree is None or self._selected_id <= 0:
            self._selected_anchor_cache = None
            return
        node = next(
            (
                n
                for n in getattr(tree, "nodes", ())
                if int(getattr(n, "object_id", 0) or 0) == int(self._selected_id)
            ),
            None,
        )
        self._selected_anchor_cache = self._anchor_from_node(node)

    def _active_edit_object_ids(self) -> tuple[int, ...]:
        if self._selected_id <= 0:
            return ()
        if (
            self._move_active
            or self._rotate_active
            or self._extrude_active
            or self._revolve_active
        ):
            return (self._selected_id,)
        return ()

    def reset_grid_spacing(self) -> None:
        self._grid_spacing = self._default_grid_spacing
        self._dirty = True

    def configure_default_grid(self) -> None:
        self._grid_spacing = self._default_grid_spacing
        self._dirty = True

    def frame_default_grid(self) -> None:
        self.frame_target(
            (0.0, 0.0, 0.0), DEFAULT_VIEW_DISTANCE * self._camera.view_scale
        )

    # -- QRhiWidget hooks ----------------------------------------------------

    def initialize(self, cb) -> None:
        try:
            if not self._renderer_ready:
                self._renderer.set_telemetry_callback(self._record_renderer_telemetry)
                self._renderer.initialize(self.rhi(), self.renderTarget())
                self._renderer.set_update_callback(self._request_render)
                self._renderer_ready = True
                self._dirty = True
                self.update()
        except BaseException:
            import traceback
            traceback.print_exc()
            raise

    def render(self, cb) -> None:
        now = time.perf_counter()
        self._last_render_submit_t = now
        self._update_pending = False
        self._dirty_flag = False
        render_start = time.perf_counter()
        self._renderer.render(
            cb, self.renderTarget(), self._camera_values(),
            self._overlay_geometry())
        self._performance_governor.record_render_call_ms(
            (time.perf_counter() - render_start) * 1000.0
        )
        self._update_fps(now)

    def _record_renderer_telemetry(self, name: str, value_ms: float) -> None:
        if name == "cull_grid_ms":
            self._performance_governor.record_cull_grid_ms(float(value_ms))

    def closeEvent(self, event) -> None:
        # Best-effort: persist the driver pipeline cache for next launch (no-op if
        # the backend doesn't collect it).
        try:
            self._renderer.save_pipeline_cache()
        except Exception:  # noqa: BLE001
            pass
        super().closeEvent(event)

    # -- gizmo overlay -------------------------------------------------------

    def _selected_anchor(self):
        """World anchor (bbox center) of the selected object, shifted by any
        in-progress move preview, or None."""
        if self._selected_id <= 0:
            return None
        center = self._selected_anchor_cache
        if center is None:
            self._refresh_selected_anchor_cache()
            center = self._selected_anchor_cache
        if center is None:
            return None
        delta = (
            self._move_preview_delta
            if self._move_active
            else self._move_commit_delta
        )
        return center + np.array(delta)

    # quad-segment triangulation: (endpoint_select, side) per vertex
    _SEG_CORNERS = ((0.0, 1.0), (0.0, -1.0), (1.0, 1.0),
                    (1.0, 1.0), (0.0, -1.0), (1.0, -1.0))

    def _seg(self, a, b, color, out) -> None:
        """Append one thick line segment (6 verts, 11 floats each) to `out`.
        Each vertex carries both endpoints; the shader expands to pixel width."""
        a = (float(a[0]), float(a[1]), float(a[2]))
        b = (float(b[0]), float(b[1]), float(b[2]))
        for sel, side in self._SEG_CORNERS:
            out.extend([*a, *b, *color, sel, side])

    def _overlay_geometry(self):
        """Build the move gizmo (3 colored axes at the selected object) — plus a
        rotation ring while rotating — as packed (bytes, vertex_count) thick-line
        triangles, or None when nothing's selected."""
        length = self._gizmo_length()
        verts: list[float] = []
        if not self._gizmo_visible:
            return None
        if self._create_tool.points is not None:
            self._append_point_shape_preview(verts)
            if not verts:
                return None
            arr = np.asarray(verts, dtype=np.float32)
            return (arr.tobytes(), arr.size // 11)
        band_start = self._create_tool.anchor or self._create_tool.start_world
        if band_start is not None and self._create_tool.hover is not None:
            self._append_rubber_band(band_start, verts)
            if not verts:
                return None
            arr = np.asarray(verts, dtype=np.float32)
            return (arr.tobytes(), arr.size // 11)
        if self._extrude_active and self._extrude_origin is not None:
            self._append_extrude_gizmo(verts)
        elif (
            self._revolve_active
            and (
                self._revolve_origin is not None
                or self._revolve_axis_start is not None
            )
        ):
            self._append_revolve_gizmo(length * 1.5, verts)
        else:
            anchor = self._selected_anchor()
            if anchor is None:
                return None
            axes = (
                ((1.0, 0.0, 0.0), (1.00, 0.25, 0.20)),
                ((0.0, 1.0, 0.0), (0.25, 0.95, 0.35)),
                ((0.0, 0.0, 1.0), (0.30, 0.55, 1.00)),
            )
            for axis, color in axes:
                self._seg(anchor, anchor + np.array(axis) * length, color, verts)
            if self._rotate_active and self._rotate_pivot is not None:
                self._append_rotation_ring(length * 1.5, verts)
        if not verts:
            return None
        arr = np.asarray(verts, dtype=np.float32)
        return (arr.tobytes(), arr.size // 11)

    def _append_rubber_band(self, start, out: list) -> None:
        """Rectangle from the drag/anchor start to the hover, on the active plane."""
        ui, vi = self._PLANE_AXIS_INDICES[self._grid_plane]
        a = np.array(start)
        c = np.array(self._create_tool.hover)
        color = (0.20, 0.95, 0.95)
        p00 = a.copy()
        p10 = a.copy(); p10[ui] = c[ui]
        p11 = a.copy(); p11[ui] = c[ui]; p11[vi] = c[vi]
        p01 = a.copy(); p01[vi] = c[vi]
        for s, t in ((p00, p10), (p10, p11), (p11, p01), (p01, p00)):
            self._seg(s, t, color, out)

    def _append_point_shape_preview(self, out: list) -> None:
        """Cyan polyline through the collected points, with a dashed-feel segment
        to the current hover; small ticks mark each placed point."""
        pts = list(self._create_tool.points or ())
        preview = list(pts)
        if self._create_tool.point_hover is not None:
            preview.append(self._create_tool.point_hover)
        color = (0.20, 0.95, 0.95)
        for a, b in zip(preview, preview[1:]):
            self._seg(a, b, color, out)
        tick = self._gizmo_length() * 0.08
        u, v = self._AXIS_BASIS[self._PLANE_NORMAL_NAME[self._grid_plane]]
        for p in pts:
            p = np.array(p)
            self._seg(p - u * tick, p + u * tick, (1.0, 0.9, 0.3), out)
            self._seg(p - v * tick, p + v * tick, (1.0, 0.9, 0.3), out)

    def _append_extrude_gizmo(self, out: list) -> None:
        o = self._extrude_origin
        n = self._extrude_normal
        color = (0.20, 0.95, 0.95)
        self._seg(o, o + n * self._extrude_height, color, out)
        # small cross at the current top to mark the height
        tip = o + n * self._extrude_height
        s = self._gizmo_length() * 0.12
        u, v = self._AXIS_BASIS["z"]
        self._seg(tip - u * s, tip + u * s, color, out)
        self._seg(tip - v * s, tip + v * s, color, out)

    def _append_revolve_gizmo(self, radius: float, out: list) -> None:
        if self._revolve_phase == "axis":
            if self._revolve_axis_start is None or self._revolve_axis_end is None:
                return
            start = self._revolve_axis_start
            end = self._revolve_axis_end
            color = (1.0, 0.35, 0.95)
            self._seg(start, end, color, out)
            tick = self._gizmo_length() * 0.08
            if self._revolve_section_normal is not None:
                axis = end - start
                axis_len = float(np.linalg.norm(axis))
                if axis_len > 1.0e-9:
                    axis = axis / axis_len
                    side = np.cross(self._revolve_section_normal, axis)
                    side_len = float(np.linalg.norm(side))
                    if side_len > 1.0e-9:
                        side = side / side_len
                        self._seg(end - side * tick, end + side * tick, color, out)
            return
        o = self._revolve_origin
        axis = self._revolve_axis
        u = self._revolve_radial
        if o is None or axis is None or u is None:
            return
        w = np.cross(axis, u)
        color = (1.0, 0.35, 0.95)
        self._seg(o - axis * radius, o + axis * radius, color, out)  # axis line
        segments = 48
        span = math.radians(self._revolve_deg) if abs(self._revolve_deg) > 1e-6 \
            else 2.0 * math.pi
        for i in range(segments):
            a0 = span * i / segments
            a1 = span * (i + 1) / segments
            p0 = o + radius * (math.cos(a0) * u + math.sin(a0) * w)
            p1 = o + radius * (math.cos(a1) * u + math.sin(a1) * w)
            self._seg(p0, p1, color, out)

    _AXIS_RING_COLOR = {
        "x": (1.00, 0.25, 0.20),
        "y": (0.25, 0.95, 0.35),
        "z": (0.30, 0.55, 1.00),
    }

    def _append_rotation_ring(self, radius: float, out: list) -> None:
        """Three colored rings (X/Y/Z) around the rotate pivot. The axis being
        dragged is brightened, with a spoke at the current preview angle."""
        pivot = np.array(self._rotate_pivot)
        segments = 48
        for axis in ("x", "y", "z"):
            u, v = self._AXIS_BASIS[axis]
            base = self._AXIS_RING_COLOR[axis]
            active = self._rotate_start_angle is not None \
                and axis == self._rotate_axis
            color = (1.0, 0.95, 0.55) if active else base
            for i in range(segments):
                a0 = 2.0 * math.pi * i / segments
                a1 = 2.0 * math.pi * (i + 1) / segments
                p0 = pivot + radius * (math.cos(a0) * u + math.sin(a0) * v)
                p1 = pivot + radius * (math.cos(a1) * u + math.sin(a1) * v)
                self._seg(p0, p1, color, out)
            if active:
                spoke = math.radians(self._rotate_preview_deg)
                tip = pivot + radius * (math.cos(spoke) * u + math.sin(spoke) * v)
                self._seg(pivot, tip, (1.0, 1.0, 0.7), out)

    def _update_fps(self, now: float | None = None) -> None:
        if now is None:
            now = time.perf_counter()
        last, self._last_frame_t = self._last_frame_t, now
        if last is None:
            return
        dt = now - last
        if dt <= 0.0:
            return
        max_sample = (
            float(self._performance_governor.config.max_frame_sample_ms)
            / 1000.0
        )
        if dt > max_sample:
            return
        self._performance_governor.record_frame_ms(dt * 1000.0)
        inst = 1.0 / dt
        # Exponential smoothing; seed on the first real sample.
        self._fps_ema = inst if self._fps_ema == 0.0 else (
            0.9 * self._fps_ema + 0.1 * inst)
        if self._fps_label is not None and now - self._fps_label_t >= 0.25:
            self._fps_label_t = now
            self._fps_label.setText(f"FPS: {self._fps_ema:4.0f}")
            self._fps_label.adjustSize()

    # -- camera --------------------------------------------------------------

    def _camera_values(self) -> dict:
        pos, _fwd, right, up = self._camera.basis()
        return {
            "u_camera_position": tuple(pos),
            "u_camera_target": tuple(self._camera.target),
            "u_camera_right": tuple(right),
            "u_camera_up": tuple(up),
            "u_focal_length": self._camera.focal,
            "u_max_ray_distance": self._max_ray_distance(pos),
            "u_surface_opacity": self._sdf_opacity,
            "u_background_color": self._bg,
            "u_show_grid": 1 if self._show_grid else 0,
            "u_grid_spacing": self._grid_spacing,
            "u_grid_plane": self._grid_plane,
            "u_selected_object_id": self._selected_id,
            "u_interacting": 1 if self._interacting else 0,
        }

    # -- interaction ---------------------------------------------------------

    def mousePressEvent(self, e: QMouseEvent) -> None:
        self._last_pos = e.position()
        self._press_pos = e.position()
        if self._boundary_tool.active and e.button() == Qt.MouseButton.LeftButton:
            self._boundary_tool.commit()
            self._last_pos = None
            self._press_pos = None
            return
        if self._create_tool.kind is not None and e.button() == Qt.MouseButton.LeftButton:
            point = self._snap_world(
                self._screen_to_grid(e.position()), e.modifiers())
            if point is None:
                signals.log_message.emit(
                    "warning",
                    "The camera ray does not reach the reference grid plane.",
                )
            elif self._create_tool.points is not None:  # point-shape: collect clicks
                self._create_tool.points.append(point)
                self._create_tool.point_hover = point
                self._update_readout(point)
                self._dirty = True
            elif self._create_tool.anchor is not None:  # anchored: this click is the end
                self._create_tool.emit_drag_shape(self._create_tool.anchor, point)
            else:  # drag-shape: record the drag start
                self._create_tool.start_world = point
        elif self._move_active and e.button() == Qt.MouseButton.LeftButton:
            self._move_start_grid = self._snap_world(
                self._screen_to_grid(e.position()), e.modifiers())
            self._move_preview_delta = (0.0, 0.0, 0.0)
        elif self._rotate_active and e.button() == Qt.MouseButton.LeftButton:
            self._rotate_axis = self._pick_rotation_axis(e.position())
            self._rotate_start_angle = self._axis_angle(
                e.position(), self._rotate_axis)
            self._rotate_preview_deg = 0.0
        elif self._extrude_active and e.button() == Qt.MouseButton.LeftButton:
            self._extrude_start_h = self._project_ray_to_axis(
                e.position(), self._extrude_origin, self._extrude_normal)
            self._extrude_height = 0.0
        elif self._revolve_active and e.button() == Qt.MouseButton.LeftButton:
            if self._revolve_phase == "axis":
                point = self._revolve_plane_point(e.position())
                self._revolve_axis_start = point
                self._revolve_axis_end = point
                self._revolve_origin = point
                self._dirty = point is not None
            else:
                self._revolve_start_angle = self._revolve_angle(e.position())
                self._revolve_last_preview_ms = 0.0
                self._revolve_last_preview_deg = None
                self._revolve_deg = 0.0

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        # Live coordinate readout (also fires on hover via mouse tracking).
        hover = self._snap_world(self._screen_to_grid(e.position()), e.modifiers())
        self._update_readout(hover)
        if self._boundary_tool.active and not (
            e.buttons() & Qt.MouseButton.LeftButton
        ):
            self._boundary_tool.update_hover(e.position())
        if self._create_tool.points is not None:  # live point-shape preview follows cursor
            self._create_tool.point_hover = hover
            self._dirty = True
        if self._create_tool.anchor is not None:  # rubber-band to the locked anchor
            self._create_tool.hover = hover
            self._dirty = True
        if self._last_pos is None or not (e.buttons() & Qt.MouseButton.LeftButton):
            self._update_hovered_object(e.position())
            return
        # A left-drag is in progress (orbit or any tool): use cheap interactive
        # frames until motion stops.
        self._begin_interaction()
        # While a create tool is armed, a left-drag sketches the shape — don't
        # orbit; show a live ghost + rubber-band from the drag start.
        if self._create_tool.kind is not None:
            self._last_pos = e.position()
            if self._create_tool.points is None and self._create_tool.start_world is not None:
                cur = self._snap_world(
                    self._screen_to_grid(e.position()), e.modifiers())
                if cur is not None and cur != self._create_tool.hover:
                    self._create_tool.hover = cur
                    self._dirty = True
                    if cur != self._create_tool.start_world:
                        signals.viewport_shape_preview_requested.emit(
                            self._create_tool.kind, self._create_tool.start_world, cur, None)
            return
        if self._move_active and self._move_handle is not None:
            # Snapped translation across the active grid plane; preview only.
            self._last_pos = e.position()
            if self._move_start_grid is None:
                return
            current = self._snap_world(
                self._screen_to_grid(e.position()), e.modifiers())
            if current is None:
                return
            delta = tuple(
                float(current[i] - self._move_start_grid[i]) for i in range(3))
            if delta == self._move_preview_delta:
                return
            self._move_preview_delta = delta
            self._update_readout()
            signals.viewport_move_preview_requested.emit(
                int(self._move_handle), delta)
            return
        if self._rotate_active and self._rotate_handle is not None:
            self._last_pos = e.position()
            if self._rotate_start_angle is None:
                self._rotate_start_angle = self._axis_angle(
                    e.position(), self._rotate_axis)
                return
            angle = self._axis_angle(e.position(), self._rotate_axis)
            if angle is None:
                return
            deg = math.degrees(angle - self._rotate_start_angle)
            if not (e.modifiers() & Qt.KeyboardModifier.AltModifier):
                deg = round(deg / 15.0) * 15.0  # snap to 15deg
            deg = (deg + 180.0) % 360.0 - 180.0  # normalize to (-180, 180]
            if deg == self._rotate_preview_deg:
                return
            self._rotate_preview_deg = deg
            self._update_readout()
            signals.viewport_rotate_preview_requested.emit(
                int(self._rotate_handle), self._rotate_axis, deg,
                self._rotate_pivot)
            return
        if self._extrude_active and self._extrude_handle is not None:
            self._last_pos = e.position()
            proj = self._project_ray_to_axis(
                e.position(), self._extrude_origin, self._extrude_normal)
            if self._extrude_start_h is None:
                self._extrude_start_h = proj
                return
            height = proj - self._extrude_start_h
            if abs(height - self._extrude_height) < 1e-6:
                return
            self._extrude_height = height
            self._update_readout()
            signals.viewport_extrude_preview_requested.emit(
                int(self._extrude_handle), float(height))
            return
        if self._revolve_active and self._revolve_handle is not None:
            self._last_pos = e.position()
            if self._revolve_phase == "axis":
                if self._revolve_axis_start is None:
                    return
                point = self._revolve_plane_point(e.position())
                if point is None:
                    return
                self._revolve_axis_end = point
                self._set_revolve_axis_vector(self._revolve_axis_start, point)
                self._update_readout()
                self._dirty = True
                return
            angle = self._revolve_angle(e.position())
            if angle is None:
                return
            if self._revolve_start_angle is None:
                self._revolve_start_angle = angle
                return
            deg = math.degrees(angle - self._revolve_start_angle)
            if not (e.modifiers() & Qt.KeyboardModifier.AltModifier):
                deg = round(deg / 15.0) * 15.0
            deg = max(-360.0, min(360.0, deg))
            if abs(deg - self._revolve_deg) < 1e-6:
                return
            self._revolve_deg = deg
            self._update_readout()
            self._dirty = True
            now_ms = time.monotonic() * 1000.0
            if (
                self._revolve_last_preview_deg is not None
                and now_ms - self._revolve_last_preview_ms
                < _REVOLVE_PREVIEW_INTERVAL_MS
            ):
                return
            frame = _revolve_signal_frame(
                self._revolve_axis,
                self._revolve_radial,
                deg,
            )
            if frame is None:
                self._restore_committed_scene()
                return
            self._revolve_last_preview_ms = now_ms
            self._revolve_last_preview_deg = deg
            radial_direction, angle_degrees = frame
            signal_axis_name = (
                self._revolve_axis_name
                if self._revolve_axis_name in {"u", "v"}
                else "v"
            )
            signals.viewport_revolve_preview_requested.emit(
                int(self._revolve_handle),
                signal_axis_name,
                tuple(self._revolve_origin),
                tuple(self._revolve_axis),
                radial_direction,
                angle_degrees,
            )
            return
        d = e.position() - self._last_pos
        self._last_pos = e.position()
        self._camera.orbit(d.x(), d.y())
        self._dirty = True

    def _nudge_with_arrows(self, key) -> bool:
        """Arrow keys nudge the active tool: Move translates one grid step along
        the plane axes; Rotate steps 15deg about the active axis. Each nudge is
        a single committed step."""
        arrows = (Qt.Key.Key_Left, Qt.Key.Key_Right,
                  Qt.Key.Key_Up, Qt.Key.Key_Down)
        if key not in arrows:
            return False
        if self._move_active and self._move_handle is not None:
            ui, vi = self._PLANE_AXIS_INDICES[self._grid_plane]
            step = self._grid_spacing
            delta = [0.0, 0.0, 0.0]
            if key == Qt.Key.Key_Left:
                delta[ui] = -step
            elif key == Qt.Key.Key_Right:
                delta[ui] = step
            elif key == Qt.Key.Key_Up:
                delta[vi] = step
            else:
                delta[vi] = -step
            signals.viewport_move_requested.emit(
                int(self._move_handle), (delta[0], delta[1], delta[2]))
            return True
        if self._rotate_active and self._rotate_handle is not None \
                and key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            step = 15.0 if key == Qt.Key.Key_Right else -15.0
            signals.viewport_rotate_requested.emit(
                int(self._rotate_handle), self._rotate_axis, step,
                self._rotate_pivot)
            return True
        return False

    def keyPressEvent(self, e) -> None:
        key = e.key()
        if key == Qt.Key.Key_Escape and self._boundary_tool.active:
            self._boundary_tool.cancel()
            signals.log_message.emit("info", "Boundary tool cancelled.")
            return
        if key == Qt.Key.Key_Escape and self._create_tool.kind is not None:
            self.cancel_create_tool()
            self._restore_committed_scene()
            signals.log_message.emit("info", "Create tool cancelled.")
            return
        if self._create_tool.points is not None and key in (
                Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._create_tool.commit_point_shape()
            return
        if self._create_tool.points is not None and key == Qt.Key.Key_Backspace:
            if self._create_tool.points:
                self._create_tool.points.pop()
                self._dirty = True
            return
        # Typed dimensions while an anchored drag-create is pending.
        if self._create_tool.anchor is not None:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._create_tool.commit_typed_dimension()
                return
            if key == Qt.Key.Key_Backspace:
                self._create_tool.dimension_input = self._create_tool.dimension_input[:-1]
                self._update_readout()
                return
            ch = e.text()
            if ch and ch in "0123456789.x, ":
                self._create_tool.dimension_input += ch
                self._update_readout()
                return
        if key == Qt.Key.Key_Escape and self._move_active:
            self.end_move_tool()
            self._restore_committed_scene()
            signals.log_message.emit("info", "Move cancelled.")
            return
        if key == Qt.Key.Key_Escape and self._rotate_active:
            self.end_rotate_tool()
            self._restore_committed_scene()
            signals.log_message.emit("info", "Rotate cancelled.")
            return
        if key == Qt.Key.Key_Escape and self._extrude_active:
            self._end_extrude_tool()
            self._restore_committed_scene()
            signals.log_message.emit("info", "Extrude cancelled.")
            return
        if key == Qt.Key.Key_Escape and self._revolve_active:
            self._end_revolve_tool()
            self._restore_committed_scene()
            signals.log_message.emit("info", "Revolve cancelled.")
            return
        view = {Qt.Key.Key_1: "iso", Qt.Key.Key_2: "top",
                Qt.Key.Key_3: "front", Qt.Key.Key_4: "side"}.get(key)
        if view is not None:
            self.set_reference_view(view)
            return
        if self._nudge_with_arrows(key):
            return
        # WASD + QE fly: translate the view (target + camera) together.
        camera = self._camera
        _pos, fwd, right, _up = camera.basis()
        world_up = np.array([0.0, 0.0, 1.0])
        move = {
            Qt.Key.Key_W: fwd, Qt.Key.Key_S: -fwd,
            Qt.Key.Key_D: right, Qt.Key.Key_A: -right,
            Qt.Key.Key_E: world_up, Qt.Key.Key_Q: -world_up,
        }.get(key)
        if move is not None:
            camera.target = camera.target + move * camera.fly_step()
            self._dirty = True
        else:
            super().keyPressEvent(e)

    def mouseReleaseEvent(self, e: QMouseEvent) -> None:
        self._end_interaction()  # drag finished -> render the settled frame sharp
        if self._extrude_active and e.button() == Qt.MouseButton.LeftButton:
            handle, height = self._extrude_handle, self._extrude_height
            self._end_extrude_tool()
            if abs(height) > 1e-4 and handle is not None:
                signals.viewport_extrude_requested.emit(int(handle), float(height))
            else:
                self._restore_committed_scene()
            self._last_pos = self._press_pos = None
            return
        if self._revolve_active and e.button() == Qt.MouseButton.LeftButton:
            if self._revolve_phase == "axis":
                self._finish_revolve_axis_phase()
                self._last_pos = self._press_pos = None
                return
            handle, deg = self._revolve_handle, self._revolve_deg
            axis_name = (
                self._revolve_axis_name
                if self._revolve_axis_name in {"u", "v"}
                else "v"
            )
            origin = self._revolve_origin
            axis, radial = self._revolve_axis, self._revolve_radial
            self._end_revolve_tool()
            frame = (
                _revolve_signal_frame(axis, radial, deg)
                if axis is not None and radial is not None
                else None
            )
            if frame is not None and handle is not None and origin is not None:
                radial_direction, angle_degrees = frame
                signals.viewport_revolve_requested.emit(
                    int(handle),
                    axis_name,
                    tuple(origin),
                    tuple(axis),
                    radial_direction,
                    angle_degrees,
                )
            else:
                self._restore_committed_scene()
            self._last_pos = self._press_pos = None
            return
        if self._rotate_active and e.button() == Qt.MouseButton.LeftButton:
            handle = self._rotate_handle
            axis = self._rotate_axis
            pivot = self._rotate_pivot
            deg = self._rotate_preview_deg
            self.end_rotate_tool()
            if abs(deg) > 1e-6 and handle is not None:
                signals.viewport_rotate_requested.emit(
                    int(handle), axis, deg, pivot)
            else:
                self._restore_committed_scene()
            self._last_pos = None
            self._press_pos = None
            return
        if self._move_active and e.button() == Qt.MouseButton.LeftButton:
            handle = self._move_handle
            delta = self._move_preview_delta
            moved = max(abs(c) for c in delta) > 1e-9
            if moved and handle is not None:
                # Single commit (one undo step); main_window re-publishes,
                # which restores the real scene over the preview.
                self.end_move_tool(preserve_preview=True)
                signals.viewport_move_requested.emit(int(handle), delta)
            else:
                self.end_move_tool()
                self._restore_committed_scene()  # no displacement -> drop preview
            self._last_pos = None
            self._press_pos = None
            return
        if (
            self._create_tool.kind is not None
            and self._create_tool.points is None
            and e.button() == Qt.MouseButton.LeftButton
        ):
            start = self._create_tool.start_world
            end = self._snap_world(self._screen_to_grid(e.position()), e.modifiers())
            moved = (self._press_pos is not None
                     and (e.position() - self._press_pos).manhattanLength() >= 4)
            if start is not None and end is not None and moved:
                self._create_tool.emit_drag_shape(start, end)  # real drag -> commit
            elif start is not None:
                # click: lock the anchor; type a size + Enter, or click the end.
                self._create_tool.anchor = start
                self._create_tool.start_world = None
                self._restore_committed_scene()  # drop any drag-preview ghost
                signals.log_message.emit(
                    "info",
                    "Type a size and press Enter, or click the end point. "
                    "Esc cancels.")
            self._last_pos = None
            self._press_pos = None
            return
        # A click (negligible drag) selects; a drag was an orbit.
        if self._press_pos is not None and e.button() == Qt.MouseButton.LeftButton:
            moved = (e.position() - self._press_pos).manhattanLength()
            if moved < 4:
                self._pick(self._press_pos)
        self._last_pos = None
        self._press_pos = None

    def _screen_ray(self, pos):
        """Return (origin, direction) of the camera ray through screen `pos`.
        Matches the QRhi surface/grid camera convention."""
        return self._camera.screen_ray(
            pos.x(), pos.y(), self.width(), self.height()
        )

    def _screen_to_grid(self, pos):
        """Intersect the screen ray with the active grid plane (0=XY,1=XZ,2=YZ).
        Returns a world (x, y, z) tuple, or None if the ray is parallel/behind."""
        cp, rd = self._screen_ray(pos)
        axis = {0: 2, 1: 1, 2: 0}[self._grid_plane]  # plane's constant axis
        if abs(rd[axis]) < 1e-9:
            return None
        t = -cp[axis] / rd[axis]
        if t <= 0.0:
            return None
        p = cp + rd * t
        return (float(p[0]), float(p[1]), float(p[2]))

    def _update_hovered_object(self, pos) -> None:
        now = time.perf_counter()
        if now - self._last_hover_pick_t < 0.08:
            return
        self._last_hover_pick_t = now
        hovered_id = self._pick_bounds_object_id(pos)
        if hovered_id != self._hovered_id:
            self._hovered_id = hovered_id
            self._dirty = True

    def _pick_bounds_object_id(self, pos) -> int:
        tree = self._tree
        if tree is None:
            return 0
        cp, rd = self._screen_ray(pos)
        best_t = math.inf
        best_id = 0
        for node in getattr(tree, "components", ()):
            object_id = int(getattr(node, "object_id", 0) or 0)
            if object_id <= 0:
                continue
            try:
                box = node.bounding_box()
            except Exception:  # noqa: BLE001 - hover is best-effort.
                continue
            hit_t = self._ray_box_hit(cp, rd, box)
            if hit_t is not None and hit_t < best_t:
                best_t = hit_t
                best_id = object_id
        return best_id

    def _ray_box_hit(self, origin, direction, box) -> float | None:
        t_min = 0.0
        t_max = self._max_ray_distance(origin)
        bounds = (
            (float(box.x_min), float(box.x_max)),
            (float(box.y_min), float(box.y_max)),
            (float(box.z_min), float(box.z_max)),
        )
        for axis, (lower, upper) in enumerate(bounds):
            rd = float(direction[axis])
            ro = float(origin[axis])
            if abs(rd) < 1.0e-12:
                if ro < lower or ro > upper:
                    return None
                continue
            inv = 1.0 / rd
            near = (lower - ro) * inv
            far = (upper - ro) * inv
            if near > far:
                near, far = far, near
            t_min = max(t_min, near)
            t_max = min(t_max, far)
            if t_min > t_max:
                return None
        return t_min if t_max >= 0.0 else None

    def _pick(self, pos) -> None:
        """Evaluate the canonical SDF tree under the cursor for picking only."""
        tree = self._tree
        if tree is None or getattr(tree, "root", None) is None:
            return
        from core.sdf_attribution import evaluate_with_attribution
        cp, rd = self._screen_ray(pos)
        travel, hit_id = 0.0, 0
        maximum_travel = self._max_ray_distance(cp)
        for _ in range(200):
            p = cp + rd * travel
            d, owner = evaluate_with_attribution(
                tree.root, np.array([p[0]]), np.array([p[1]]), np.array([p[2]]))
            dist = float(d[0])
            if dist < 0.002:
                hit_id = int(owner[0])
                break
            travel += max(dist, 0.002)
            if travel > maximum_travel:
                break
        self._selected_id = hit_id
        signals.viewport_scene_object_selected.emit(hit_id)
        self._dirty = True

    def _max_ray_distance(self, camera_position) -> float:
        base = 100.0 * max(1.0, self._camera.view_scale)
        if self._scene_center is None:
            return base
        camera = np.asarray(camera_position, dtype=np.float64)
        to_scene = float(np.linalg.norm(camera - self._scene_center))
        return max(base, to_scene + self._scene_radius * 4.0)

    def wheelEvent(self, e: QWheelEvent) -> None:
        # angleDelta (mouse wheel, ~120/notch) or pixelDelta (touchpad, small).
        # Zoom proportional to the scroll amount so a touchpad's many tiny events
        # don't compound into a runaway.
        dy = e.angleDelta().y() or e.pixelDelta().y()
        if dy == 0:
            return
        self._camera.zoom_by(dy)
        self._begin_interaction()


__all__ = ["QRhiViewportWidget"]
