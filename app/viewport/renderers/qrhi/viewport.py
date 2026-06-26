from __future__ import annotations

"""QRhi viewport widget — a QRhiWidget that renders a scene via the QRhi codegen
renderer, with mouse orbit / zoom.

ONE renderer + ONE shader source drive every backend. Backend-aware: the renderer
adapts to each backend's coordinate conventions (fb_y_up / clip_y_sign) at
initialize, and the backend is a clean, overridable choice — no backend is pinned
(the bytecode VM that used to force OpenGL has been removed). Set ``QRHI_BACKEND``
(vulkan|opengl|metal|d3d11) to choose one; otherwise QRhi picks the platform
default.
"""

import math
import os
import random
import time

import numpy as np
from PySide6.QtCore import Qt, QElapsedTimer, QTimer
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import (
    QFrame,
    QGridLayout,
    QLabel,
    QMenu,
    QPushButton,
    QRhiWidget,
)

from app.dimensions import parse_dimension_entry
from app.signals import signals

from .renderer import QRhiInterpreterRenderer

_BACKENDS = {
    "vulkan": QRhiWidget.Api.Vulkan,
    "opengl": QRhiWidget.Api.OpenGL,
    "metal": getattr(QRhiWidget.Api, "Metal", None),
    "d3d11": getattr(QRhiWidget.Api, "Direct3D11", None),
}
_MAX_VIEWPORT_FPS = 60
_FRAME_INTERVAL_MS = max(1, round(1000 / _MAX_VIEWPORT_FPS))


def _choose_api() -> "QRhiWidget.Api | None":
    """Backend-aware: honour an explicit QRHI_BACKEND, else let QRhi choose the
    platform-native default (None). The renderer is backend-agnostic, so no backend
    is pinned here — the OpenGL hard-pin that the bytecode VM required is gone."""
    want = os.environ.get("QRHI_BACKEND", "").lower()
    if want in _BACKENDS and _BACKENDS[want] is not None:
        return _BACKENDS[want]
    return None


class QRhiViewportWidget(QRhiWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        api = _choose_api()
        if api is not None:
            self.setApi(api)
        self.setMinimumSize(480, 360)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self._renderer = QRhiInterpreterRenderer()
        self._renderer_ready = False
        # orbit camera state
        self._target = np.array([0.0, 0.0, 0.0])
        self._distance = 6.0
        self._yaw = math.radians(35.0)
        self._pitch = math.radians(28.0)
        self._focal = 1.5
        self._bg = (0.07, 0.08, 0.10)
        self._show_grid = True
        self._grid_spacing = 1.0
        self._default_grid_spacing = 1.0
        self._sdf_opacity = 1.0
        self._gizmo_visible = True
        self._components_visible = True
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
        self._revolve_origin = None
        self._revolve_axis = None
        self._revolve_radial = None
        self._revolve_start_angle = None
        self._revolve_deg = 0.0
        # boundary cutter tool state (cutter = a drawn shape routed via
        # active_boundary_cutter_tool, consumed in main_window._on_viewport_shape_drawn)
        self._boundary_cutter_tool = None
        self._boundary_region_selected = False
        self._committed_render_ir = None   # last committed scene (restore target)
        self._preview_kind = None
        self._boolean_preview_commit_pending = False
        # drag-to-create tool state (Draw button -> viewport_create_requested)
        self._create_kind = None
        self._create_start_world = None
        self._point_pts = None       # multi-click point list (point-shape kinds)
        self._point_hover = None     # current hover point for the preview
        self._create_anchor = None   # locked start (deferred click -> typed size)
        self._create_hover = None    # hover for the rubber-band preview
        self._dimension_input = ""   # typed dimension buffer
        self._snap_enabled = True  # snap grid-plane points to grid spacing
        self._command_panel = None
        # render FPS overlay (measures real frame cadence in render())
        self._fps_ema = 0.0
        self._last_frame_t = None
        self._last_render_submit_t = None
        self._fps_label_t = 0.0
        self._fps_label = None
        # Render throttle: input marks dirty; a 60fps timer renders only when
        # something changed (decouples render rate from the touchpad's flood).
        # Created ONCE here — not per resizeEvent, which leaked a timer each call.
        self._dirty = True
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(_FRAME_INTERVAL_MS)
        # Interactive-quality state: while the camera/tool is actively moving the
        # renderer uses a cheaper per-pixel budget (see u_interacting); a short
        # idle delay then triggers one full-quality frame once motion stops.
        self._interacting = False
        self._interaction_idle = QTimer(self)
        self._interaction_idle.setSingleShot(True)
        self._interaction_idle.timeout.connect(self._end_interaction)
        self.setMouseTracking(True)  # deliver hover moves for the coord readout
        self._readout_label = None
        self._build_command_panel()
        self._build_view_panel()
        self._build_readout_label()
        self._build_error_label()
        self._build_fps_label()
        signals.viewport_create_requested.connect(self.begin_create_tool)
        signals.log_message.connect(self._on_log_message)

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

        planar_button = QPushButton("PlanarCutter", panel)
        planar_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        planar_button.setToolTip(
            "Draw a planar cutter on the selected BoundaryRegion")
        planar_menu = QMenu(planar_button)
        for label, shape in self._PLANAR_CUTTER_KINDS:
            planar_menu.addAction(label).triggered.connect(
                lambda _=False, s=shape: self.begin_boundary_cutter_tool(
                    "planar", s))
        planar_button.setMenu(planar_menu)
        layout.addWidget(planar_button, 1, 1)

        surface_button = QPushButton("SurfaceCutter", panel)
        surface_button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        surface_button.setToolTip(
            "Draw a surface cutter for the selected BoundaryRegion")
        surface_menu = QMenu(surface_button)
        for label, shape in self._SURFACE_CUTTER_KINDS:
            surface_menu.addAction(label).triggered.connect(
                lambda _=False, s=shape: self.begin_boundary_cutter_tool(
                    "surface", s))
        surface_button.setMenu(surface_menu)
        layout.addWidget(surface_button, 1, 2)

        self._cutter_buttons = (planar_button, surface_button)
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
            text = f"Revolve   {self._revolve_deg:+.1f}°"
        elif self._rotate_active and self._rotate_pivot is not None:
            text = (f"Rotate   {self._rotate_preview_deg:+.1f}°   "
                    f"axis {self._rotate_axis.upper()}")
        elif self._move_active:
            dx, dy, dz = self._move_preview_delta
            dist = math.sqrt(dx * dx + dy * dy + dz * dz)
            text = (f"Move   Δ({dx:+.3f}, {dy:+.3f}, {dz:+.3f})   "
                    f"|Δ|={dist:.3f}")
        elif self._create_anchor is not None:
            typed = self._dimension_input or "_"
            text = (f"{self._create_kind}   size: {typed}   "
                    "(Enter creates, or click end)")
        elif self._point_pts is not None:
            text = (f"{self._create_kind}   points: {len(self._point_pts)}   "
                    "(Enter creates, Backspace undoes)")
        elif self._create_kind is not None and \
                self._create_start_world is not None and hover is not None:
            sx, sy, sz = self._create_start_world
            text = (f"{self._create_kind}   "
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
        # bottom-right, raised above the corner orientation-axis gizmo (~130px).
        p.move(max(8, self.width() - p.width() - 8),
               max(8, self.height() - p.height() - 132))
        p.raise_()

    def set_reference_view(self, view: str) -> None:
        """Switch the active grid/draw plane and fly the camera to that view."""
        if view not in self._VIEW_TABLE:
            raise ValueError(f"unknown reference view: {view}")
        plane, yaw_deg, pitch_deg = self._VIEW_TABLE[view]
        self._grid_plane = plane
        self._animate_view_to(math.radians(yaw_deg), math.radians(pitch_deg))

    def _animate_view_to(self, target_yaw: float, target_pitch: float) -> None:
        dyaw = ((target_yaw - self._yaw + math.pi) % (2 * math.pi)) - math.pi
        dpitch = target_pitch - self._pitch
        if max(abs(dyaw), abs(dpitch)) <= 1e-6:
            self._dirty = True
            return
        self._view_anim = (self._yaw, self._pitch, dyaw, dpitch)
        self._view_anim_clock.restart()
        self._view_anim_timer.start(16)

    def _advance_view_anim(self) -> None:
        if self._view_anim is None:
            self._view_anim_timer.stop()
            return
        t = min(1.0, self._view_anim_clock.elapsed() / 260.0)
        eased = t * t * (3.0 - 2.0 * t)  # smoothstep
        y0, p0, dy, dp = self._view_anim
        self._yaw = y0 + dy * eased
        self._pitch = p0 + dp * eased
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

    def resizeEvent(self, e) -> None:
        super().resizeEvent(e)
        self._position_command_panel()
        self._position_view_panel()
        self._position_readout_label()
        self._dirty = True  # the throttle timer (built in __init__) repaints

    def _tick(self) -> None:
        if self._dirty:
            self._dirty = False
            self.update()

    def _request_render(self) -> None:
        self._dirty = True

    def _begin_interaction(self) -> None:
        """Mark the viewport as actively moving (cheap interactive frames) and
        (re)arm the idle timer that snaps back to a full-quality frame shortly
        after motion stops."""
        self._interacting = True
        self._dirty = True
        self._interaction_idle.start(140)

    def _end_interaction(self) -> None:
        """Motion stopped: render one full-quality frame."""
        if self._interacting:
            self._interacting = False
            self._dirty = True

    # -- public API ----------------------------------------------------------

    def set_scene(self, render_ir) -> None:
        self._renderer.set_scene(render_ir)
        self._dirty = True

    def frame_target(self, target=(0.0, 0.0, 0.0), distance: float = 6.0) -> None:
        self._target = np.array(target, dtype=np.float64)
        self._distance = float(distance)
        self._dirty = True

    # --- ViewportWidget drop-in compatibility ------------------------------
    # The RENDER methods below are real. The in-3D-viewport TOOLS (gizmos,
    # drawing, grid, click-selection, boundary tools, boolean preview) are
    # intentionally DROPPED in this QRhi refactor — the document stays fully
    # editable via the side panels and the viewport renders it live. Early-stage
    # feature loss by design; rebuild on the clean QRhi foundation later.

    def set_scene_artifact(self, tree, render_ir=None) -> None:
        """The app's render hook: show the freshly built scene."""
        self._tree = tree  # kept for CPU click-picking
        self._committed_render_ir = render_ir  # restore target after a preview
        self._update_scene_bounds(tree)
        self._preview_kind = None
        self._boolean_preview_commit_pending = False
        self._move_commit_delta = (0.0, 0.0, 0.0)
        self.set_scene(render_ir)

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

    def show_scene_preview(self, render_ir, *, preview_kind: str = "tool") -> None:
        """Render a non-committing ghost during a move/rotate drag (main_window
        builds it). The committed scene is restored on cancel/commit."""
        self._preview_kind = preview_kind
        self._renderer.set_scene(render_ir)
        self._dirty = True

    def _restore_committed_scene(self) -> None:
        self._preview_kind = None
        self._renderer.set_scene(self._committed_render_ir)
        self._dirty = True

    def frame_box(self, box) -> None:
        cx = (box.x_min + box.x_max) * 0.5
        cy = (box.y_min + box.y_max) * 0.5
        cz = (box.z_min + box.z_max) * 0.5
        extent = max(box.x_max - box.x_min, box.y_max - box.y_min,
                     box.z_max - box.z_min, 1e-3)
        self._target = np.array([cx, cy, cz], dtype=np.float64)
        self._distance = max(extent * 1.6, 1.0)
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
        return max(self._distance * 0.18, 0.4)

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
        self._revolve_origin = self._revolve_axis = self._revolve_radial = None
        self._revolve_start_angle = None
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

    def begin_revolve_tool(self, handle, node) -> None:
        """Drag to sweep the profile around its V axis; release applies."""
        self.end_move_tool(); self.end_rotate_tool(); self.cancel_create_tool()
        self._end_extrude_tool()
        self._revolve_active = True
        self._revolve_handle = handle
        self._revolve_origin = np.array(node.origin, dtype=np.float64)
        axis = np.array(node.axis_v, dtype=np.float64)
        radial = np.array(node.axis_u, dtype=np.float64)
        self._revolve_axis = axis / max(np.linalg.norm(axis), 1e-9)
        self._revolve_radial = radial / max(np.linalg.norm(radial), 1e-9)
        self._revolve_start_angle = None
        self._revolve_deg = 0.0
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        signals.log_message.emit(
            "info", "Drag to revolve the profile. Release applies, Esc cancels.")

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

    # --- draw / create tool (real) ---
    # kinds built from a click sequence rather than a single drag
    _POINT_CREATE_KINDS = {
        "polyline", "quadratic_bezier_curve", "quadratic_bezier_polycurve",
        "polyline_tube", "quadratic_bezier_tube", "quadratic_bezier_surface", "polygon",
    }

    def _plane_id(self) -> str:
        return ("xy", "xz", "yz")[self._grid_plane]

    def begin_create_tool(self, kind: str) -> None:
        """Arm create: drag-kinds sketch with one drag (viewport_shape_drawn);
        point-kinds collect clicks then commit on Enter (viewport_point_shape_drawn)."""
        self.end_move_tool()
        self.end_rotate_tool()
        self._end_extrude_tool()
        self._end_revolve_tool()
        self._create_kind = str(kind)
        self._create_start_world = None
        self._create_anchor = None
        self._create_hover = None
        self._dimension_input = ""
        point_mode = kind in self._POINT_CREATE_KINDS
        self._point_pts = [] if point_mode else None
        self._point_hover = None
        self._renderer.prewarm_for_tool(
            self._committed_render_ir,
            kind,
            compile_pipeline=(
                not point_mode or self._renderer.should_prewarm_tool_pipeline()
            ),
        )
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        if point_mode:
            msg = (f"Click points on the {self.reference_plane_label} grid to "
                   f"draw {kind}. Enter creates, Backspace undoes, Esc cancels.")
        else:
            msg = (f"Drag on the {self.reference_plane_label} grid to create "
                   f"{kind}. Esc cancels.")
        signals.log_message.emit("info", msg)

    def cancel_create_tool(self) -> None:
        if self._create_kind is None:
            return
        self._create_kind = None
        self._create_start_world = None
        self._create_anchor = None
        self._create_hover = None
        self._dimension_input = ""
        self._point_pts = None
        self._point_hover = None
        self._boundary_cutter_tool = None
        self.unsetCursor()
        self._dirty = True

    def _commit_point_shape(self) -> None:
        """Emit the collected point-shape if it has enough points."""
        if not self._point_pts or len(self._point_pts) < 2:
            signals.log_message.emit("warning", "Add at least two points first.")
            return
        kind = self._create_kind
        points = tuple(self._point_pts)
        plane = self._plane_id()
        # Emit before resetting so a synchronous handler can still read
        # active_boundary_cutter_tool (cutter routing).
        signals.viewport_point_shape_drawn.emit(kind, points, plane)
        self.cancel_create_tool()

    def _emit_drag_shape(self, start, end) -> None:
        kind = self._create_kind
        signals.viewport_shape_drawn.emit(kind, start, end, None)
        self.cancel_create_tool()

    def _commit_typed_dimension(self) -> None:
        """Create the anchored shape at an exact typed size (W or W x H)."""
        if self._create_anchor is None or not self._dimension_input:
            return
        try:
            dims = parse_dimension_entry(self._dimension_input)
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        w = dims[0]
        h = dims[1] if len(dims) > 1 else dims[0]
        ui, vi = self._PLANE_AXIS_INDICES[self._grid_plane]
        end = list(self._create_anchor)
        end[ui] += w
        end[vi] += h
        self._emit_drag_shape(self._create_anchor, (end[0], end[1], end[2]))

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
        self._grid_spacing = max(float(spacing), 1e-4)
        self._dirty = True

    def configure_grid(self, box=None, dx=None, *a, **k) -> None:
        if dx is not None and dx > 0:
            self._grid_spacing = float(dx)
        self._dirty = True

    # value-returning stubs (safe defaults so main_window's logic doesn't crash)

    def has_scene_object_id(self, *a) -> bool:
        return False

    def can_defer_committed_move(self, *a) -> bool:
        return False

    def paste_offset(self, *a):
        """Offset pasted/duplicated objects on the active grid plane so they
        don't stack exactly on the original. A grid-step diagonal nudge plus a
        small random jitter keeps repeated pastes visibly separated."""
        ui, vi = self._PLANE_AXIS_INDICES[self._grid_plane]
        step = max(self._grid_spacing, 0.1)
        off = [0.0, 0.0, 0.0]
        off[ui] = step * (1.0 + random.uniform(-0.25, 0.25))
        off[vi] = step * (1.0 + random.uniform(-0.25, 0.25))
        return (off[0], off[1], off[2])

    def active_boundary_cutter_tool(self, *a):
        return self._boundary_cutter_tool

    def set_boundary_region_selection_entries(self, *a, **k) -> None:
        """main_window passes (selectors, normals, regions); we only need to know
        whether a BoundaryRegion is selected so the cutter buttons enable."""
        regions = a[2] if len(a) >= 3 else ()
        self._boundary_region_selected = bool(regions)
        self._update_cutter_buttons_enabled()

    _PLANAR_CUTTER_KINDS = (("Segment", "segment"), ("Polyline", "polyline"),
                            ("Quadratic Bezier Polycurve", "quadratic_bezier_polycurve"))
    _SURFACE_CUTTER_KINDS = (("Sphere", "sphere"), ("Box", "box"),
                             ("Cylinder", "cylinder"), ("Cone", "cone"))

    def begin_boundary_cutter_tool(self, cutter_kind: str, shape_kind=None) -> None:
        """Arm a boundary cutter: draw `shape_kind` on the grid; the drawn shape
        is routed as a planar/surface cutter for the selected BoundaryRegion."""
        if not self._boundary_region_selected:
            signals.log_message.emit(
                "warning", "Select a BoundaryRegion before creating a cutter.")
            return
        if cutter_kind == "planar":
            shape = shape_kind or "polyline"
        elif cutter_kind == "surface":
            shape = shape_kind or "sphere"
        else:
            raise ValueError(f"unknown boundary cutter kind: {cutter_kind}")
        self.begin_create_tool(shape)
        self._boundary_cutter_tool = (cutter_kind, shape)
        signals.log_message.emit(
            "info",
            f"{cutter_kind.title()} cutter armed — draw the {shape} to cut the "
            "selected BoundaryRegion.")

    # no-op stubs for the dropped viewport tools
    def _noop(self, *a, **k) -> None:
        return None

    set_boundary_hover = _noop
    begin_boundary_region_tool = _noop
    apply_committed_move_preview = _noop
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
        self._components_visible = bool(visible)
        self._dirty = True

    def set_scene_selection(self, node) -> None:
        """Sync the viewport highlight/gizmo with the tree-panel selection."""
        self._selected_id = int(getattr(node, "object_id", 0) or 0)
        self._dirty = True

    def reset_grid_spacing(self) -> None:
        self._grid_spacing = self._default_grid_spacing
        self._dirty = True

    def configure_default_grid(self) -> None:
        self._grid_spacing = self._default_grid_spacing
        self._dirty = True

    def frame_default_grid(self) -> None:
        self.frame_target((0.0, 0.0, 0.0), 6.0)

    # -- QRhiWidget hooks ----------------------------------------------------

    def initialize(self, cb) -> None:
        try:
            if not self._renderer_ready:
                self._renderer.initialize(self.rhi(), self.renderTarget())
                self._renderer.set_update_callback(self._request_render)
                self._renderer_ready = True
        except BaseException:
            import traceback
            traceback.print_exc()
            raise

    def render(self, cb) -> None:
        now = time.perf_counter()
        if self._last_render_submit_t is not None:
            elapsed_ms = (now - self._last_render_submit_t) * 1000.0
            if elapsed_ms < _FRAME_INTERVAL_MS:
                self._dirty = True
                return
        self._last_render_submit_t = now
        self._renderer.render(
            cb, self.renderTarget(), self._camera_values(),
            self._overlay_geometry())
        self._update_fps(now)

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
        tree = self._tree
        if tree is None or self._selected_id <= 0:
            return None
        node = next((n for n in getattr(tree, "nodes", ())
                     if int(n.object_id) == int(self._selected_id)), None)
        if node is None:
            return None
        try:
            bb = node.bounding_box()
        except Exception:
            return None
        center = np.array([(bb.x_min + bb.x_max) * 0.5,
                           (bb.y_min + bb.y_max) * 0.5,
                           (bb.z_min + bb.z_max) * 0.5])
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
        if not self._gizmo_visible:
            return None
        length = self._gizmo_length()
        verts: list[float] = []
        if self._point_pts is not None:
            self._append_point_shape_preview(verts)
            if not verts:
                return None
            arr = np.asarray(verts, dtype=np.float32)
            return (arr.tobytes(), arr.size // 11)
        band_start = self._create_anchor or self._create_start_world
        if band_start is not None and self._create_hover is not None:
            self._append_rubber_band(band_start, verts)
            if not verts:
                return None
            arr = np.asarray(verts, dtype=np.float32)
            return (arr.tobytes(), arr.size // 11)
        if self._extrude_active and self._extrude_origin is not None:
            self._append_extrude_gizmo(verts)
        elif self._revolve_active and self._revolve_origin is not None:
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
        c = np.array(self._create_hover)
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
        pts = list(self._point_pts or ())
        preview = list(pts)
        if self._point_hover is not None:
            preview.append(self._point_hover)
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
        o = self._revolve_origin
        axis = self._revolve_axis
        u = self._revolve_radial
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
        cp = math.cos(self._pitch)
        offset = np.array([cp * math.cos(self._yaw),
                           cp * math.sin(self._yaw),
                           math.sin(self._pitch)])
        pos = self._target + self._distance * offset
        fwd = self._target - pos
        fwd = fwd / max(np.linalg.norm(fwd), 1e-9)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(fwd, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(fwd, world_up); right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(right, fwd)
        return {
            "u_camera_position": tuple(pos),
            "u_camera_target": tuple(self._target),
            "u_camera_right": tuple(right),
            "u_camera_up": tuple(up),
            "u_focal_length": self._focal,
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
        if self._create_kind is not None and e.button() == Qt.MouseButton.LeftButton:
            point = self._snap_world(
                self._screen_to_grid(e.position()), e.modifiers())
            if point is None:
                signals.log_message.emit(
                    "warning",
                    "The camera ray does not reach the reference grid plane.",
                )
            elif self._point_pts is not None:  # point-shape: collect clicks
                self._point_pts.append(point)
                self._point_hover = point
                self._update_readout(point)
                self._dirty = True
            elif self._create_anchor is not None:  # anchored: this click is the end
                self._emit_drag_shape(self._create_anchor, point)
            else:  # drag-shape: record the drag start
                self._create_start_world = point
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
            self._revolve_start_angle = self._revolve_angle(e.position())
            self._revolve_deg = 0.0

    def mouseMoveEvent(self, e: QMouseEvent) -> None:
        # Live coordinate readout (also fires on hover via mouse tracking).
        hover = self._snap_world(self._screen_to_grid(e.position()), e.modifiers())
        self._update_readout(hover)
        if self._point_pts is not None:  # live point-shape preview follows cursor
            self._point_hover = hover
            self._dirty = True
        if self._create_anchor is not None:  # rubber-band to the locked anchor
            self._create_hover = hover
            self._dirty = True
        if self._last_pos is None or not (e.buttons() & Qt.MouseButton.LeftButton):
            return
        # A left-drag is in progress (orbit or any tool): use cheap interactive
        # frames until motion stops.
        self._begin_interaction()
        # While a create tool is armed, a left-drag sketches the shape — don't
        # orbit; show a live ghost + rubber-band from the drag start.
        if self._create_kind is not None:
            self._last_pos = e.position()
            if self._point_pts is None and self._create_start_world is not None:
                cur = self._snap_world(
                    self._screen_to_grid(e.position()), e.modifiers())
                if cur is not None and cur != self._create_hover:
                    self._create_hover = cur
                    self._dirty = True
                    if cur != self._create_start_world:
                        signals.viewport_shape_preview_requested.emit(
                            self._create_kind, self._create_start_world, cur, None)
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
            signals.viewport_revolve_preview_requested.emit(
                int(self._revolve_handle), tuple(self._revolve_origin),
                tuple(self._revolve_axis), tuple(self._revolve_radial), deg)
            return
        d = e.position() - self._last_pos
        self._last_pos = e.position()
        self._yaw -= d.x() * 0.01
        self._pitch = max(-1.5, min(1.5, self._pitch + d.y() * 0.01))
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
        if key == Qt.Key.Key_Escape and self._create_kind is not None:
            self.cancel_create_tool()
            self._restore_committed_scene()
            signals.log_message.emit("info", "Create tool cancelled.")
            return
        if self._point_pts is not None and key in (
                Qt.Key.Key_Return, Qt.Key.Key_Enter):
            self._commit_point_shape()
            return
        if self._point_pts is not None and key == Qt.Key.Key_Backspace:
            if self._point_pts:
                self._point_pts.pop()
                self._dirty = True
            return
        # Typed dimensions while an anchored drag-create is pending.
        if self._create_anchor is not None:
            if key in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
                self._commit_typed_dimension()
                return
            if key == Qt.Key.Key_Backspace:
                self._dimension_input = self._dimension_input[:-1]
                self._update_readout()
                return
            ch = e.text()
            if ch and ch in "0123456789.x, ":
                self._dimension_input += ch
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
        cam = self._camera_values()
        right = np.array(cam["u_camera_right"], dtype=np.float64)
        fwd = np.array(cam["u_camera_target"], dtype=np.float64) \
            - np.array(cam["u_camera_position"], dtype=np.float64)
        n = np.linalg.norm(fwd)
        fwd = fwd / n if n > 1e-9 else fwd
        world_up = np.array([0.0, 0.0, 1.0])
        step = max(self._distance * 0.06, 0.05)
        move = {
            Qt.Key.Key_W: fwd, Qt.Key.Key_S: -fwd,
            Qt.Key.Key_D: right, Qt.Key.Key_A: -right,
            Qt.Key.Key_E: world_up, Qt.Key.Key_Q: -world_up,
        }.get(key)
        if move is not None:
            self._target = self._target + move * step
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
            handle, deg = self._revolve_handle, self._revolve_deg
            origin = self._revolve_origin
            axis, radial = self._revolve_axis, self._revolve_radial
            self._end_revolve_tool()
            if abs(deg) > 1e-3 and handle is not None:
                signals.viewport_revolve_requested.emit(
                    int(handle), tuple(origin), tuple(axis), tuple(radial), deg)
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
            self._create_kind is not None
            and self._point_pts is None
            and e.button() == Qt.MouseButton.LeftButton
        ):
            start = self._create_start_world
            end = self._snap_world(self._screen_to_grid(e.position()), e.modifiers())
            moved = (self._press_pos is not None
                     and (e.position() - self._press_pos).manhattanLength() >= 4)
            if start is not None and end is not None and moved:
                self._emit_drag_shape(start, end)  # real drag -> commit
            elif start is not None:
                # click: lock the anchor; type a size + Enter, or click the end.
                self._create_anchor = start
                self._create_start_world = None
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
        Matches the fragment shader's ray construction (incl. the Vulkan y-flip)."""
        cam = self._camera_values()
        cp = np.array(cam["u_camera_position"], dtype=np.float64)
        right = np.array(cam["u_camera_right"], dtype=np.float64)
        up = np.array(cam["u_camera_up"], dtype=np.float64)
        fwd = np.array(cam["u_camera_target"], dtype=np.float64) - cp
        fwd /= max(np.linalg.norm(fwd), 1e-9)
        w, h = max(self.width(), 1), max(self.height(), 1)
        suvx = (pos.x() - 0.5 * w) / h
        suvy = -((pos.y() - 0.5 * h) / h)  # match the shader's Vulkan y-flip
        rd = 2.0 * suvx * right + 2.0 * suvy * up + self._focal * fwd
        rd /= max(np.linalg.norm(rd), 1e-9)
        return cp, rd

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

    def _pick(self, pos) -> None:
        """CPU-raymarch the scene tree under the cursor; select the hit object."""
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
        if self._scene_center is None:
            return 100.0
        camera = np.asarray(camera_position, dtype=np.float64)
        to_scene = float(np.linalg.norm(camera - self._scene_center))
        return max(100.0, to_scene + self._scene_radius * 4.0)

    def wheelEvent(self, e: QWheelEvent) -> None:
        # angleDelta (mouse wheel, ~120/notch) or pixelDelta (touchpad, small).
        # Zoom proportional to the scroll amount so a touchpad's many tiny events
        # don't compound into a runaway.
        dy = e.angleDelta().y() or e.pixelDelta().y()
        if dy == 0:
            return
        self._distance = max(0.5, min(200.0, self._distance * math.exp(-dy * 0.0012)))
        self._begin_interaction()


__all__ = ["QRhiViewportWidget"]
