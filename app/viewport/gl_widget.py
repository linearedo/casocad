from __future__ import annotations

import logging

import moderngl
import numpy as np
from PySide6.QtCore import QElapsedTimer, QPoint, QTimer, Qt
from PySide6.QtGui import QKeyEvent, QMouseEvent, QWheelEvent
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QPushButton

from app.signals import signals
from core.boundary import BoundaryRegion
from core.mesher.classifier import pick_boundary_owner, pick_sdf_surface
from core.sdf import SDFTree
from core.sdf.base import BoundingBox3D, SDFNode

from .camera import OrbitCamera
from .renderer import SDFRenderer

logger = logging.getLogger(__name__)
MAX_SELECTED_BOUNDARY_OWNERS = 128
FPS_COUNTER_UPDATE_MS = 500
VIEW_ANIMATION_DURATION_MS = 180
VIEW_ANIMATION_INTERVAL_MS = 16
CREATE_PREVIEW_KINDS = {
    "interval": 1,
    "circle": 2,
    "rectangle": 3,
    "square": 4,
    "rounded_rectangle": 5,
    "ellipse": 6,
    "regular_polygon": 7,
    "sphere": 8,
    "box": 9,
    "cylinder": 10,
    "torus": 11,
}
REFERENCE_PLANE_IDS = {"xy": 0, "xz": 1, "yz": 2}


class GLWidget(QOpenGLWidget):
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
        self._position_view_panel()
        self._context: moderngl.Context | None = None
        self._renderer: SDFRenderer | None = None
        self._pending_scene: str | None = None
        self._scene_tree: SDFTree | None = None
        self._scene_hover_object_id = 0
        self._scene_selected_object_id = 0
        self._pending_lattice: object | None = None
        self._lattice_result: object | None = None
        self._lattice_filter_ids: set[int] | None = None
        self._lattice_filter_sdf: SDFNode | None = None
        self._lattice_filter_color_id: int | None = None
        self._lattice_filter_enabled = True
        self._grid_spacing = 0.1
        self._last_mouse_position: QPoint | None = None
        self._interaction_tool: tuple[str, object] | None = None
        self._boundary_pick_root: SDFNode | None = None
        self._boundary_selection_active = False
        self._boundary_hover_owner_id = 0
        self._boundary_hover_direction = -1
        self._selected_boundary_regions: tuple[tuple[int, int], ...] = ()
        self._boundary_press_position: QPoint | None = None
        self._boundary_camera_dragged = False
        self._tool_start_screen: QPoint | None = None
        self._tool_start_world: tuple[float, float, float] | None = None
        self._tool_current_world: tuple[float, float, float] | None = None
        self._move_preview_delta = (0.0, 0.0, 0.0)
        self.mode = "sdf"
        self.grid_visible = True
        self.components_visible = False
        self.sdf_opacity = 0.4
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

    def _position_view_panel(self) -> None:
        if not hasattr(self, "_view_panel"):
            return
        self._view_panel.adjustSize()
        self._view_panel.move(10, max(10, self.height() - 34))
        self._view_panel.raise_()

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
        self._position_view_panel()
        super().resizeEvent(event)

    def begin_create_tool(self, kind: str) -> None:
        self._clear_scene_hover()
        self._interaction_tool = ("create", kind)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setFocus()
        signals.log_message.emit(
            "info", f"Drag on the reference grid to create {kind}. Esc cancels."
        )

    def begin_move_tool(self, handle: int) -> None:
        self._clear_scene_hover()
        self._interaction_tool = ("move", handle)
        self._move_preview_delta = (0.0, 0.0, 0.0)
        self._tool_start_screen = None
        self._tool_start_world = None
        self._tool_current_world = None
        self.setCursor(Qt.CursorShape.SizeAllCursor)
        self.setFocus()
        signals.log_message.emit(
            "info",
            "Move preview active. Drag or use WASD/QE, Enter applies, Esc cancels.",
        )

    def begin_boundary_region_tool(self, root: SDFNode) -> None:
        self._clear_scene_hover()
        self._boundary_pick_root = root
        self._boundary_selection_active = True
        self._boundary_hover_owner_id = 0
        self._boundary_hover_direction = -1
        self._boundary_press_position = None
        self._boundary_camera_dragged = False
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
        self._move_preview_delta = (0.0, 0.0, 0.0)
        self.unsetCursor()
        self._boundary_selection_active = False
        self._boundary_hover_owner_id = 0
        self._boundary_hover_direction = -1
        signals.viewport_boundary_hovered.emit(None)
        self.update()

    def set_boundary_hover(
        self,
        owner_object_id: int,
        outside_direction: int | None,
    ) -> None:
        self._boundary_hover_owner_id = owner_object_id
        self._boundary_hover_direction = (
            outside_direction if outside_direction is not None else -1
        )
        self.update()

    def set_boundary_region_selection(
        self,
        regions: list[BoundaryRegion],
    ) -> None:
        direction_masks: dict[int, int] = {}
        for region in regions:
            mask = (
                0b11_1111
                if region.outside_direction is None
                else 1 << region.outside_direction
            )
            direction_masks[region.owner_object_id] = (
                direction_masks.get(region.owner_object_id, 0) | mask
            )
        selectors = tuple(sorted(direction_masks.items()))
        if len(selectors) > MAX_SELECTED_BOUNDARY_OWNERS:
            signals.log_message.emit(
                "warning",
                "BoundaryRegion viewport highlighting is limited to "
                f"{MAX_SELECTED_BOUNDARY_OWNERS} distinct boundary owners.",
            )
        self._selected_boundary_regions = selectors[
            :MAX_SELECTED_BOUNDARY_OWNERS
        ]
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
        self._scene_tree = tree
        self._scene_hover_object_id = 0
        self._pending_scene = (
            f"{tree.to_glsl()}\n{tree.components_to_glsl()}"
            if tree is not None
            else (
                "float sceneSDF(vec3 p) { return 1000000.0; }\n"
                "int sceneBoundaryOwnerId(vec3 p) { return 0; }\n"
                "int sceneObjectId(vec3 p) { return 0; }\n"
                "bool sceneSelectionOwnsBoundary(int selected_object_id, "
                "int boundary_owner_id) { return false; }\n"
                "float sceneMovedSDF(vec3 p, int selected_object_id, "
                "vec3 preview_offset) { return 1000000.0; }\n"
                "float sceneSelectedObjectSDF(vec3 p, int selected_object_id) "
                "{ return 1000000.0; }\n"
                "int sceneSelectedObjectDimension(int selected_object_id) "
                "{ return 0; }\n"
                "const int COMPONENT_COUNT = 0;\n"
                "float componentSDF(vec3 p, int component) { return 1000000.0; }\n"
                "int componentObjectId(int component) { return 0; }"
            )
        )
        self.update()

    def set_lattice(self, result: object) -> None:
        self._lattice_result = result
        self._queue_lattice_upload()
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

    def set_components_visible(self, visible: bool) -> None:
        self.components_visible = visible
        self.update()

    def set_sdf_opacity(self, opacity: float) -> None:
        self.sdf_opacity = min(1.0, max(0.05, float(opacity)))
        self.update()

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
        self.update()

    def _keyboard_move_delta(self, key: int) -> tuple[float, float, float] | None:
        step = self._grid_spacing
        deltas = {
            Qt.Key.Key_W: (0.0, step, 0.0),
            Qt.Key.Key_S: (0.0, -step, 0.0),
            Qt.Key.Key_A: (-step, 0.0, 0.0),
            Qt.Key.Key_D: (step, 0.0, 0.0),
            Qt.Key.Key_Q: (0.0, 0.0, -step),
            Qt.Key.Key_E: (0.0, 0.0, step),
        }
        return deltas.get(key)

    def _commit_move_preview(self) -> None:
        if (
            self._interaction_tool is None
            or self._interaction_tool[0] != "move"
        ):
            return
        _action, value = self._interaction_tool
        delta = self._preview_move_delta()
        self.cancel_interaction_tool()
        if max(abs(component) for component in delta) <= 1e-12:
            signals.log_message.emit("info", "Move preview had no displacement.")
            return
        signals.viewport_move_requested.emit(int(value), delta)

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
        self._grid_spacing = self._nice_grid_spacing((2.0 * half_extent) / 40.0)
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
            self._renderer = SDFRenderer(self._context)
            logger.info(
                "OpenGL initialized: %s",
                self._context.info.get("GL_RENDERER", "unknown renderer"),
            )
        except Exception as error:
            logger.exception("OpenGL initialization failed")
            signals.log_message.emit(
                "error", f"OpenGL initialization failed: {error}"
            )

    def paintGL(self) -> None:
        if self._renderer is None:
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
            try:
                self._renderer.compile_scene(self._pending_scene)
                self._pending_scene = None
            except moderngl.Error as error:
                logger.exception("shader compilation failed")
                signals.log_message.emit("error", f"Shader compilation failed: {error}")
        if self._pending_lattice is not None:
            (
                positions,
                node_types,
                boundary_faces,
                source_ids,
                primary_tag_ids,
                tag_axis_u,
                tag_axis_v,
                cell_size,
                dimension,
                axis_i,
                axis_j,
            ) = self._pending_lattice
            self._renderer.upload_lattice(
                positions,
                node_types,
                boundary_faces,
                source_ids,
                primary_tag_ids,
                tag_axis_u,
                tag_axis_v,
                cell_size,
                dimension,
                axis_i,
                axis_j,
            )
            self._pending_lattice = None
        width = max(1, round(self.width() * self.devicePixelRatio()))
        height = max(1, round(self.height() * self.devicePixelRatio()))
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
            self.camera.view_rotation(),
            self.gizmo_visible,
            self._grid_spacing,
            REFERENCE_PLANE_IDS[self._reference_plane],
            self._boundary_selection_active,
            self._boundary_hover_owner_id,
            self._boundary_hover_direction,
            self._scene_hover_object_id,
            self._scene_selected_object_id,
            self._selected_boundary_regions,
            self._preview_kind(),
            self._preview_start(),
            self._preview_current(),
            self._preview_move_delta(),
        )
        self._update_fps_counter()

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

    def _preview_kind(self) -> int:
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "create"
            and self._tool_start_world is not None
            and self._tool_current_world is not None
        ):
            return CREATE_PREVIEW_KINDS.get(str(self._interaction_tool[1]), 0)
        return 0

    def _preview_start(self) -> tuple[float, float, float]:
        return self._tool_start_world or (0.0, 0.0, 0.0)

    def _preview_current(self) -> tuple[float, float, float]:
        return self._tool_current_world or self._preview_start()

    def _preview_move_delta(self) -> tuple[float, float, float]:
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "move"
        ):
            drag_delta = (0.0, 0.0, 0.0)
            if (
                self._tool_start_world is not None
                and self._tool_current_world is not None
            ):
                drag_delta = tuple(
                    self._tool_current_world[index]
                    - self._tool_start_world[index]
                    for index in range(3)
                )
            return tuple(
                self._move_preview_delta[index] + drag_delta[index]
                for index in range(3)
            )
        return (0.0, 0.0, 0.0)

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
            and event.button() == Qt.MouseButton.LeftButton
        ):
            point = self.camera.screen_to_plane(
                self._reference_plane,
                event.position().x(),
                event.position().y(),
                self.width(),
                self.height(),
            )
            if point is None:
                signals.log_message.emit(
                    "warning",
                    "The current camera ray does not reach the reference plane.",
                )
                return
            self._tool_start_screen = event.position().toPoint()
            self._tool_start_world = point
            self._tool_current_world = point
            self.update()
            return
        self._last_mouse_position = event.position().toPoint()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
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
            point = self.camera.screen_to_plane(
                self._reference_plane,
                event.position().x(),
                event.position().y(),
                self.width(),
                self.height(),
            )
            if point is not None:
                self._tool_current_world = point
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
            end = self.camera.screen_to_plane(
                self._reference_plane,
                event.position().x(),
                event.position().y(),
                self.width(),
                self.height(),
            )
            action, value = self._interaction_tool
            start = self._tool_start_world
            if end is None:
                signals.log_message.emit(
                    "warning",
                    "The current camera ray does not reach the reference plane.",
                )
                return
            if action == "create":
                self.cancel_interaction_tool()
                signals.viewport_shape_drawn.emit(str(value), start, end)
            else:
                delta = tuple(end[index] - start[index] for index in range(3))
                self._move_preview_delta = tuple(
                    self._move_preview_delta[index] + delta[index]
                    for index in range(3)
                )
                self._tool_start_screen = None
                self._tool_start_world = None
                self._tool_current_world = None
                self.update()
                signals.log_message.emit(
                    "info", "Move preview updated. Press Enter to apply or Esc to cancel."
                )
            return
        self._last_mouse_position = None
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event: object) -> None:
        self._clear_scene_hover()
        super().leaveEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._interaction_tool is not None and event.key() in {
            Qt.Key.Key_Escape,
            Qt.Key.Key_Delete,
            Qt.Key.Key_Backspace,
        }:
            self.cancel_interaction_tool()
            signals.log_message.emit("info", "Viewport tool cancelled.")
            return
        if (
            self._interaction_tool is not None
            and self._interaction_tool[0] == "move"
        ):
            if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter}:
                self._commit_move_preview()
                return
            if event.modifiers() == Qt.KeyboardModifier.NoModifier:
                delta = self._keyboard_move_delta(event.key())
                if delta is not None:
                    self.nudge_move_preview(delta)
                    return
        super().keyPressEvent(event)

    def wheelEvent(self, event: QWheelEvent) -> None:
        self._stop_view_animation()
        self.camera.zoom(event.angleDelta().y() / 120.0)
        self.update()

    def closeEvent(self, event: object) -> None:
        self.makeCurrent()
        if self._renderer is not None:
            self._renderer.release()
        self.doneCurrent()
        super().closeEvent(event)
