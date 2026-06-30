from __future__ import annotations

import logging
import os
from dataclasses import replace
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QAction, QColor, QKeySequence
from PySide6.QtWidgets import (
    QColorDialog,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMenuBar,
    QMessageBox,
    QLabel,
    QPushButton,
    QSlider,
    QToolBar,
)

from app.title_bar import install_title_bar

from app.axis_labels import world_axis_label
from app.artifacts import (
    ArtifactManager,
    BOOLEAN_DRAW_RESOLUTION,
    COARSE_VIEWPORT_SURFACE_RESOLUTION,
    REFINED_VIEWPORT_SURFACE_RESOLUTION,
    REVOLVE_VIEWPORT_SURFACE_RESOLUTION,
    RenderArtifact,
    RenderSceneSnapshot,
    build_render_artifact,
)
from app.meshing import MeshingWorkspace
from app.panels.log_panel import LogPanel
from app.panels.properties import CadDimensionSpinBox, PropertiesPanel
from app.panels.scene_tree import SceneTreePanel
from app.signals import signals
from app.viewport.renderers.qrhi.viewport import QRhiViewportWidget as ViewportWidget
from core.boundary import BoundaryRegion
from core.boundary_direction import owner_outside_direction_vector
from core.boundary_patches import BoundaryPatchHit
from core.model import (
    disjointness_violations,
    grammar_violations,
    model_from_document,
)
from core.sdf import (
    Difference,
    Intersection,
    PlacedSDF2D,
    Revolve,
    SDFTree,
    Union,
    Xor,
)
from core.sdf.base import BoundingBox3D, SDFNode
from core.sdf.roles import DomainKind
from core.serialization import load_scene, save_scene
from core.scene import INTERNAL_BOUNDARY_SELECTOR_PREFIX, SceneDocument

logger = logging.getLogger(__name__)
UNDO_HISTORY_LIMIT = 50
SNAP_TOGGLE_SHORTCUT = "G"
REDO_SHORTCUTS = ("Ctrl+Y", "Ctrl+Shift+Z")
SELECT_ALL_SHORTCUT = "Ctrl+A"
DUPLICATE_SHORTCUT = "Ctrl+D"
RENAME_SHORTCUT = "F2"


def _format_render_artifact_ready_message(artifact: RenderArtifact) -> str:
    timings = artifact.timings
    large_scene = "yes" if timings.large_scene_mode else "no"
    return (
        "Render artifact built: "
        f"total={timings.total_ms:.1f} ms, "
        f"surface={timings.surface_ms:.1f} ms, "
        f"render_wait={timings.render_wait_ms:.1f} ms, "
        f"tree_nodes={timings.tree_node_count}, "
        f"surface_resolution={timings.surface_resolution}, "
        f"surface_vertices={timings.surface_vertex_count}, "
        f"surface_triangles={timings.surface_triangle_count}, "
        f"large_scene={large_scene}, "
        f"objects={timings.total_object_count}, "
        f"exact={timings.exact_object_count}, "
        f"no_blur={timings.no_blur}, "
        f"reason={timings.large_scene_reason or 'none'}"
    )
FRAME_SHORTCUTS = ("Home",)
FRAME_VIEW_KEY = "F"
CLEAR_SELECTION_SHORTCUT = "Esc"
DEFAULT_BACKGROUND_HEX = "#241f32"


def color_to_rgb_tuple(color: QColor) -> tuple[float, float, float]:
    return (
        color.redF(),
        color.greenF(),
        color.blueF(),
    )


def rgb_tuple_to_hex(color: tuple[float, float, float]) -> str:
    red, green, blue = (
        max(0, min(255, round(float(component) * 255.0)))
        for component in color
    )
    return f"#{red:02x}{green:02x}{blue:02x}"


def viewport_shape_created_message(name: str) -> str:
    return f"Created {name}. Draw tool remains active; press Esc to finish."


def parse_solid_from_2d_method(method: str) -> tuple[str, str]:
    parts = str(method).split(":", 1)
    base = parts[0]
    axis = parts[1] if len(parts) == 2 else "v"
    if base != "revolve":
        return base, "v"
    if len(parts) == 1:
        return base, "custom"
    if axis not in {"u", "v"}:
        raise ValueError("revolve axis must be 'u' or 'v'")
    return base, axis


def viewport_render_resolution_for_tree(tree: object) -> tuple[int, bool]:
    """Pick the first-build resolution and whether to climb the refinement ladder.

    Returns ``(resolution, refine_after)``. Booleans and primitives start at the
    interactive COARSE tier and refine progressively to full precision off-thread
    (outcomes 0 and 0.5); 2D fills are cheap and rendered at full resolution
    immediately; revolves keep their dedicated profile resolution.
    """
    components = tuple(getattr(tree, "components", ()) or ())
    root = getattr(tree, "root", None)
    if isinstance(root, (Union, Intersection, Difference, Xor)):
        return BOOLEAN_DRAW_RESOLUTION, True
    nodes = tuple(getattr(tree, "nodes", ()) or ())
    if any(isinstance(node, Revolve) for node in (*components, *nodes)):
        return REVOLVE_VIEWPORT_SURFACE_RESOLUTION, False
    if components and all(
        int(getattr(component, "dimension", 3)) <= 2
        for component in components
    ):
        return REFINED_VIEWPORT_SURFACE_RESOLUTION, False
    return COARSE_VIEWPORT_SURFACE_RESOLUTION, True


def scene_item_handles(document: SceneDocument) -> list[int]:
    return [handle for handle, _node, _parent in document.walk()]


def selected_sdf_bounding_box(
    document: SceneDocument,
    handles: list[int],
) -> BoundingBox3D | None:
    selected_box = None
    for handle in handles:
        try:
            node = document.node(handle)
        except KeyError:
            continue
        if not isinstance(node, SDFNode):
            continue
        try:
            box = node.bounding_box()
        except (NotImplementedError, ValueError):
            continue
        selected_box = box if selected_box is None else selected_box.union(box)
    return selected_box


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("casoCAD - Programmable SDF CAD")
        self.resize(1400, 860)
        self._menubar = QMenuBar(self)
        self.document = SceneDocument()
        self._undo_stack: list[SceneDocument] = []
        self._redo_stack: list[SceneDocument] = []
        self._clipboard_nodes: list[SDFNode] = []
        self.viewport = ViewportWidget()
        self._background_color = QColor(DEFAULT_BACKGROUND_HEX)
        self.setCentralWidget(self.viewport)
        self.artifacts = ArtifactManager(self)
        self.scene_tree = SceneTreePanel()
        self.properties = PropertiesPanel()
        self.log_panel = LogPanel()
        self._meshing_workspace: MeshingWorkspace | None = None
        self._render_request_timer = QTimer(self)
        self._render_request_timer.setSingleShot(True)
        self._render_request_timer.timeout.connect(self._flush_render_request)
        self.viewport.set_refinement_callback(self._schedule_render_artifact)
        self._build_docks()
        self._build_menu()
        self._install_title_bar()
        self._build_toolbar()
        self._connect_signals()
        self._seed_initial_viewport_scene()
        self._publish_document(frame=True)

    def _build_docks(self) -> None:
        scene_dock = self._dock("Scene", self.scene_tree)
        properties_dock = self._dock("Properties", self.properties)
        log_dock = self._dock("Log", self.log_panel)
        self._scene_dock = scene_dock
        self._properties_dock = properties_dock
        self._log_dock = log_dock
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, properties_dock)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, scene_dock)
        self.tabifyDockWidget(scene_dock, properties_dock)
        scene_dock.raise_()
        self.resizeDocks([scene_dock], [320], Qt.Orientation.Horizontal)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)
        log_dock.hide()

    def _dock(self, title: str, widget: object) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"{title.lower()}Dock")
        dock.setWidget(widget)
        return dock

    def _install_title_bar(self) -> None:
        """Replace the native OS title bar with casoCAD's night-blue one (drawn by
        the app, so it looks the same on every OS / desktop). The central widget,
        docks and toolbars are untouched, so panel layout is unchanged."""
        self._title_bar, self._resizer = install_title_bar(self, self._menubar)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("CAD", self)
        toolbar.setObjectName("cadToolbar")
        self.addToolBar(toolbar)
        fit_action = toolbar.addAction("Frame Scene")
        fit_action.setShortcuts(
            tuple(QKeySequence(shortcut) for shortcut in FRAME_SHORTCUTS)
        )
        fit_action.triggered.connect(self._frame_scene)
        grid_action = QAction("Grid", self, checkable=True)
        grid_action.setChecked(True)
        grid_action.toggled.connect(self.viewport.set_grid_visible)
        toolbar.addAction(grid_action)
        snap_action = QAction("Snap", self, checkable=True)
        snap_action.setObjectName("snapEnabledAction")
        snap_action.setChecked(True)
        snap_action.setShortcut(SNAP_TOGGLE_SHORTCUT)
        snap_action.setToolTip("Enable snap to the active reference grid")
        snap_action.toggled.connect(self.viewport.set_snap_enabled)
        toolbar.addAction(snap_action)
        self._grid_spacing_spin = CadDimensionSpinBox(toolbar)
        self._grid_spacing_spin.setObjectName("gridSpacingSpin")
        self._grid_spacing_spin.setDecimals(4)
        self._grid_spacing_spin.setRange(0.001, 1000.0)
        self._grid_spacing_spin.setSingleStep(0.01)
        self._grid_spacing_spin.setSuffix(" m")
        self._grid_spacing_spin.setKeyboardTracking(False)
        self._grid_spacing_spin.setValue(self.viewport.grid_spacing)
        self._grid_spacing_spin.setFixedWidth(120)
        self._grid_spacing_spin.setToolTip(
            "Set reference-grid spacing with units or formulas"
        )
        self._grid_spacing_spin.valueChanged.connect(
            self.viewport.set_grid_spacing
        )
        toolbar.addWidget(self._grid_spacing_spin)
        auto_snap_action = toolbar.addAction("Auto")
        auto_snap_action.setObjectName("autoGridSpacingAction")
        auto_snap_action.setToolTip("Return snap spacing to the scene-based grid")
        auto_snap_action.triggered.connect(self._reset_grid_spacing)
        components_action = QAction("Components", self, checkable=True)
        components_action.setChecked(False)
        components_action.setToolTip(
            "Show construction operands as X-ray overlays; this can visually cover holes"
        )
        components_action.toggled.connect(self.viewport.set_components_visible)
        toolbar.addAction(components_action)
        toolbar.addWidget(QLabel("Opacity"))
        opacity_slider = QSlider(Qt.Orientation.Horizontal, toolbar)
        opacity_slider.setObjectName("sdfOpacitySlider")
        opacity_slider.setRange(5, 100)
        opacity_slider.setValue(40)
        opacity_slider.setFixedWidth(110)
        opacity_slider.setToolTip(
            "Make the final SDF shell transparent to reveal internal objects"
        )
        opacity_slider.valueChanged.connect(
            lambda value: self.viewport.set_sdf_opacity(value / 100.0)
        )
        self.viewport.set_sdf_opacity(opacity_slider.value() / 100.0)
        toolbar.addWidget(opacity_slider)
        self._background_color_button = QPushButton("Background", toolbar)
        self._background_color_button.setObjectName("backgroundColorButton")
        self._background_color_button.setToolTip("Set viewport background color")
        self._background_color_button.clicked.connect(self._choose_background_color)
        toolbar.addWidget(self._background_color_button)
        self._set_background_color(self._background_color)

    def _build_menu(self) -> None:
        file_menu = self._menubar.addMenu("&File")
        new_default = file_menu.addAction("New Scene")
        new_default.triggered.connect(self._new_default_scene)
        file_menu.addSeparator()
        open_action = file_menu.addAction("Open Scene...")
        open_action.setShortcut("Ctrl+O")
        open_action.triggered.connect(self._open_scene)
        save_action = file_menu.addAction("Save Scene As...")
        save_action.setShortcut("Ctrl+Shift+S")
        save_action.triggered.connect(self._save_scene)
        file_menu.addSeparator()
        quit_action = file_menu.addAction("Quit")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)

        edit_menu = self._menubar.addMenu("&Edit")
        copy_action = edit_menu.addAction("Copy")
        copy_action.setShortcut("Ctrl+C")
        copy_action.triggered.connect(self._copy_selection)
        paste_action = edit_menu.addAction("Paste")
        paste_action.setShortcut("Ctrl+V")
        paste_action.triggered.connect(self._paste_clipboard)
        duplicate_action = edit_menu.addAction("Duplicate")
        duplicate_action.setShortcut(DUPLICATE_SHORTCUT)
        duplicate_action.triggered.connect(self._duplicate_selection)
        rename_action = edit_menu.addAction("Rename")
        rename_action.setShortcut(RENAME_SHORTCUT)
        rename_action.triggered.connect(self._rename_selection)
        select_all_action = edit_menu.addAction("Select All")
        select_all_action.setShortcut(SELECT_ALL_SHORTCUT)
        select_all_action.triggered.connect(self._select_all_scene_items)
        clear_selection_action = edit_menu.addAction("Clear Selection")
        clear_selection_action.setShortcut(CLEAR_SELECTION_SHORTCUT)
        clear_selection_action.triggered.connect(self._clear_selection)
        delete_action = edit_menu.addAction("Delete Selection")
        delete_action.setShortcuts(
            (QKeySequence(Qt.Key.Key_Delete), QKeySequence(Qt.Key.Key_Backspace))
        )
        delete_action.triggered.connect(self._delete_selection)
        edit_menu.addSeparator()
        self._undo_action = edit_menu.addAction("Undo")
        self._undo_action.setShortcut("Ctrl+Z")
        self._undo_action.triggered.connect(self._undo_document_edit)
        self._redo_action = edit_menu.addAction("Redo")
        self._redo_action.setShortcuts(
            tuple(QKeySequence(shortcut) for shortcut in REDO_SHORTCUTS)
        )
        self._redo_action.triggered.connect(self._redo_document_edit)
        self._update_history_actions()

        view_menu = self._menubar.addMenu("&View")
        for dock in (self._scene_dock, self._properties_dock, self._log_dock):
            view_menu.addAction(dock.toggleViewAction())

        domains_menu = self._menubar.addMenu("&Domains")
        validate_action = domains_menu.addAction("Validate Domains (disjointness)")
        validate_action.triggered.connect(self._validate_domains_disjoint)

        meshing_menu = self._menubar.addMenu("&Meshing")
        workspace_action = meshing_menu.addAction("Open Meshing Workspace...")
        workspace_action.triggered.connect(self._open_meshing_workspace)

    def _open_meshing_workspace(self) -> None:
        if self._meshing_workspace is None:
            self._meshing_workspace = MeshingWorkspace(self)
            self._meshing_workspace.destroyed.connect(self._on_meshing_workspace_destroyed)
        self._meshing_workspace.show()
        self._meshing_workspace.raise_()
        self._meshing_workspace.activateWindow()

    def _on_meshing_workspace_destroyed(self) -> None:
        self._meshing_workspace = None

    def _model_or_none(self):
        """Adapt explicitly declared document Domains to a compiler Model."""
        try:
            return model_from_document(self.document)
        except ValueError:
            return None

    def _update_grammar_diagnostics(self) -> None:
        """Live, non-blocking role-grammar check (exact-SDF spec §4).

        Quiet by design: it only fires when operators are wired in an
        exactness-breaking way (e.g. intersecting a union result), which makes
        the interior distance field non-exact. Disjointness (§7) is deferred to
        the Validate Domains action / future mesh-time gate, not run live.
        """
        model = self._model_or_none()
        issues = grammar_violations(model) if model is not None else []
        last = getattr(self, "_last_grammar_issues", [])
        if issues == last:
            return
        self._last_grammar_issues = issues
        if issues:
            signals.log_message.emit(
                "warning",
                "Exact-SDF grammar violated (interior distance no longer exact): "
                + issues[0],
            )
            self.statusBar().showMessage("⚠ Exact-SDF grammar violated", 5000)
        elif last:
            signals.log_message.emit("info", "Exact-SDF grammar OK")

    @Slot()
    def _validate_domains_disjoint(self) -> None:
        """On-demand exactness + disjointness check (spec §4 + §7).

        Runs the full compile gate (role grammar + the sampled disjointness
        probe). The hard gate will also run automatically at mesh time once a
        mesher exists; for now it is user-triggered from the Domains menu.
        """
        model = self._model_or_none()
        if model is None:
            QMessageBox.warning(
                self,
                "Validate Domains",
                "The declared Domain names must be unique.",
            )
            return
        if not model.domains:
            QMessageBox.warning(
                self,
                "Validate Domains",
                "No Domains are defined. Select an SDF object and choose "
                "'Set as Domain' before validating solver-ready geometry.",
            )
            return
        problems = grammar_violations(model) + disjointness_violations(model)
        if not problems:
            QMessageBox.information(
                self,
                "Validate Domains",
                f"All {len(model.domains)} domain(s) are exact and mutually "
                "disjoint.",
            )
            return
        QMessageBox.warning(
            self,
            "Validate Domains",
            "Model is not compilable:\n\n- " + "\n- ".join(problems),
        )

    def _connect_signals(self) -> None:
        signals.add_primitive_requested.connect(self._on_add_primitive)
        signals.viewport_shape_drawn.connect(self._on_viewport_shape_drawn)
        signals.viewport_shape_preview_requested.connect(
            self._on_viewport_shape_preview
        )
        signals.viewport_point_shape_drawn.connect(self._on_viewport_point_shape_drawn)
        signals.viewport_move_tool_requested.connect(self._start_viewport_move)
        signals.viewport_boundary_tool_requested.connect(
            self._start_boundary_region_tool
        )
        signals.viewport_move_requested.connect(self._on_viewport_move_requested)
        signals.viewport_move_preview_requested.connect(
            self._on_viewport_move_preview
        )
        signals.viewport_rotate_tool_requested.connect(self._start_viewport_rotate)
        signals.viewport_rotate_requested.connect(self._on_viewport_rotate_requested)
        signals.viewport_rotate_preview_requested.connect(
            self._on_viewport_rotate_preview
        )
        signals.viewport_extrude_preview_requested.connect(
            self._on_viewport_extrude_preview
        )
        signals.viewport_revolve_preview_requested.connect(
            self._on_viewport_revolve_preview
        )
        signals.viewport_transform_requested.connect(
            self._on_viewport_transform_requested
        )
        signals.viewport_extrude_requested.connect(
            self._on_viewport_extrude_requested
        )
        signals.viewport_revolve_requested.connect(
            self._on_viewport_revolve_requested
        )
        signals.viewport_frame_requested.connect(self._frame_scene)
        signals.viewport_scene_object_selected.connect(
            self._on_viewport_scene_object_selected
        )
        signals.viewport_boundary_hovered.connect(
            self._on_viewport_boundary_hovered
        )
        signals.viewport_boundary_region_requested.connect(
            self._on_viewport_boundary_region_requested
        )
        signals.delete_nodes_requested.connect(self._on_delete_nodes)
        signals.sdf_op_requested.connect(self._on_sdf_op_requested)
        signals.sdf_op_preview_requested.connect(self._on_sdf_op_preview_requested)
        signals.transform_requested.connect(self._on_transform_requested)
        signals.solid_from_2d_requested.connect(self._on_solid_from_2d)
        signals.set_domain_requested.connect(self._on_set_domain)
        signals.set_fluid_root_requested.connect(self._on_set_fluid_root)
        signals.set_tag_enabled_requested.connect(self._on_set_tag_enabled)
        signals.create_boundary_region_requested.connect(
            self._on_create_boundary_region
        )
        signals.create_polygon_from_polyline_requested.connect(
            self._on_create_polygon_from_polyline
        )
        signals.undo_snapshot_ready.connect(self._record_undo_snapshot)
        signals.node_edited.connect(self._on_node_edited)
        signals.selection_changed.connect(self._on_selection_changed)
        signals.log_message.connect(self._on_log_message)
        self.artifacts.render_ready.connect(self._on_render_artifact_ready)
        self.artifacts.render_failed.connect(self._on_render_artifact_failed)

    def _publish_document(
        self,
        frame: bool = False,
        clear_selection: bool = True,
        render: bool = True,
        notify_document: bool = True,
    ) -> None:
        if notify_document:
            signals.document_changed.emit(self.document)
        self._update_grammar_diagnostics()
        if clear_selection:
            signals.node_selected.emit(None)
        if not self.document.objects:
            render_timer = getattr(self, "_render_request_timer", None)
            if render_timer is not None:
                render_timer.stop()
            self.viewport.set_scene_artifact(None, None)
            self.viewport.configure_default_grid()
            self.viewport.frame_default_grid()
            self._sync_grid_spacing_control()
            return
        if render:
            self._schedule_render_artifact()
        if self.document.fluid_domain is not None:
            domain = self.document.fluid_domain
            assert domain is not None
            box = domain.bounding_box()
            self.viewport.configure_grid(box)
            self._sync_grid_spacing_control()
            if frame:
                self.viewport.frame_box(box)

    def _schedule_render_artifact(self) -> None:
        if not self._render_request_timer.isActive():
            self._render_request_timer.start(0)

    @Slot()
    def _flush_render_request(self) -> None:
        if not self.document.objects:
            return
        try:
            version, tree = self.document.visual_snapshot()
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        budget = self.viewport.render_budget_for_tree(tree)
        surface_resolution, refine_after = viewport_render_resolution_for_tree(tree)
        self.artifacts.request_render(
            RenderSceneSnapshot(
                version=version,
                tree=tree,
                budget=budget,
                surface_resolution=surface_resolution,
                refine_after=refine_after,
            )
        )

    def _seed_initial_viewport_scene(self) -> None:
        if not self.document.objects:
            return
        try:
            version, tree = self.document.visual_snapshot()
            artifact = build_render_artifact(
                RenderSceneSnapshot(
                    version=version,
                    tree=tree,
                )
            )
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        self.viewport.set_scene_artifact(
            artifact.tree,
            artifact.surface_scene,
            artifact.timings,
        )

    def _sync_grid_spacing_control(self) -> None:
        if not hasattr(self, "_grid_spacing_spin"):
            return
        self._grid_spacing_spin.blockSignals(True)
        self._grid_spacing_spin.setValue(self.viewport.grid_spacing)
        self._grid_spacing_spin.blockSignals(False)

    @Slot()
    def _reset_grid_spacing(self) -> None:
        self.viewport.reset_grid_spacing()
        self._sync_grid_spacing_control()

    def _set_background_color(self, color: QColor) -> None:
        self._background_color = QColor(color)
        self.viewport.set_background_color(color_to_rgb_tuple(self._background_color))
        if hasattr(self, "_background_color_button"):
            color_hex = self._background_color.name()
            self._background_color_button.setStyleSheet(
                "QPushButton#backgroundColorButton {"
                f" background: {color_hex};"
                " color: #ffffff;"
                " border: 1px solid #20242b;"
                " padding: 3px 8px;"
                "}"
            )
            self._background_color_button.setToolTip(
                f"Set viewport background color ({color_hex})"
            )

    @Slot()
    def _choose_background_color(self) -> None:
        color = QColorDialog.getColor(
            self._background_color,
            self,
            "Viewport background color",
        )
        if not color.isValid():
            return
        self._set_background_color(color)

    def _history_snapshot(self) -> SceneDocument | None:
        try:
            return self.document.snapshot()
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return None

    @Slot(object)
    def _record_undo_snapshot(self, snapshot: SceneDocument | None) -> None:
        if snapshot is None:
            return
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > UNDO_HISTORY_LIMIT:
            del self._undo_stack[0]
        self._redo_stack.clear()
        self._update_history_actions()

    @Slot()
    def _undo_document_edit(self) -> None:
        if not self._undo_stack:
            return
        current = self._history_snapshot()
        if current is not None:
            self._redo_stack.append(current)
        snapshot = self._undo_stack.pop()
        self._restore_history_snapshot(snapshot)
        self.statusBar().showMessage("Undo", 3000)

    @Slot()
    def _redo_document_edit(self) -> None:
        if not self._redo_stack:
            return
        current = self._history_snapshot()
        if current is not None:
            self._undo_stack.append(current)
        snapshot = self._redo_stack.pop()
        self._restore_history_snapshot(snapshot)
        self.statusBar().showMessage("Redo", 3000)

    def _restore_history_snapshot(self, snapshot: SceneDocument) -> None:
        current_version = self.document.version
        restored = snapshot.snapshot()
        restored.version = current_version + 1
        self.document = restored
        self._publish_document()
        self._update_history_actions()

    def _update_history_actions(self) -> None:
        if hasattr(self, "_undo_action"):
            self._undo_action.setEnabled(bool(self._undo_stack))
        if hasattr(self, "_redo_action"):
            self._redo_action.setEnabled(bool(self._redo_stack))

    @Slot(object)
    def _on_render_artifact_ready(self, artifact: RenderArtifact) -> None:
        if artifact.version != self.document.version:
            return
        signals.log_message.emit(
            "info",
            _format_render_artifact_ready_message(artifact),
        )
        self.viewport.set_scene_artifact(
            artifact.tree,
            artifact.surface_scene,
            artifact.timings,
        )

    @Slot(int, str)
    def _on_render_artifact_failed(self, version: int, message: str) -> None:
        if version != self.document.version:
            return
        signals.log_message.emit("error", f"Render update failed: {message}")

    @Slot(str)
    def _on_add_primitive(self, kind: str) -> None:
        undo_snapshot = self._history_snapshot()
        handle = self.document.add_primitive(kind)
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()
        self.scene_tree.select_handle(handle)

    def _on_viewport_shape_preview(self, kind, start, end, parameters) -> None:
        """Live ghost of a shape being drawn (non-committing). Boundary cutters
        are skipped — they need a selected region and special handling."""
        if self.viewport.active_boundary_cutter_tool() is not None:
            return
        try:
            preview = self.document.snapshot()
            handle = preview.add_primitive_from_drag(
                kind,
                start,
                end,
                parameters=parameters,
            )
            node = preview.node(handle)
            if not isinstance(node, SDFNode):
                return
            tree = SDFTree(root=node, components=(node,))
            surface_resolution, _refine_after = viewport_render_resolution_for_tree(tree)
            artifact = build_render_artifact(
                RenderSceneSnapshot(
                    version=preview.version,
                    tree=tree,
                    surface_resolution=surface_resolution,
                    refine_after=False,
                )
            )
        except (KeyError, ValueError):
            return
        base_scene = self.viewport.committed_surface_scene()
        if base_scene is None or artifact.surface_scene is None:
            self.viewport.show_scene_preview(artifact.surface_scene)
            return
        self.viewport.show_scene_preview(
            replace(
                base_scene,
                revision=preview.version,
                surfaces=(*base_scene.surfaces, *artifact.surface_scene.surfaces),
                build_ms=artifact.surface_scene.build_ms,
            )
        )

    @Slot(str, object, object, object)
    def _on_viewport_shape_drawn(
        self,
        kind: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        parameters: dict[str, float] | None,
    ) -> None:
        undo_snapshot = self._history_snapshot()
        base_region = self._selected_boundary_region_for_cutter()
        surface_cutter_active = self._active_surface_cutter_kind(kind) is not None
        try:
            if self._active_planar_cutter_kind(kind) == "segment":
                if base_region is None:
                    raise ValueError(
                        "Select a BoundaryRegion before creating a planar cutter."
                    )
                handle = self._add_planar_segment_cutter_region(
                    base_region,
                    start,
                    end,
                )
            else:
                handle = self.document.add_primitive_from_drag(
                    kind,
                    start,
                    end,
                    parameters=parameters,
                )
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        split_handles = self._try_create_boundary_selector_split(
            base_region,
            self.document.node(handle),
        )
        if split_handles and surface_cutter_active:
            self._mark_internal_boundary_selector(
                self.document.node(handle),
                "surface_cutter",
            )
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()
        if split_handles:
            self.scene_tree.select_handles(list(split_handles))
        else:
            self.scene_tree.select_handle(handle)
        self.viewport.setFocus()
        if split_handles:
            self.statusBar().showMessage(
                "Created cutter and split BoundaryRegion into inside/outside.",
                5000,
            )
        else:
            self.statusBar().showMessage(
                viewport_shape_created_message(self.document.node(handle).name),
                5000,
            )

    @Slot(str, object, str)
    def _on_viewport_point_shape_drawn(
        self,
        kind: str,
        points: tuple[tuple[float, float, float], ...],
        reference_plane: str,
    ) -> None:
        undo_snapshot = self._history_snapshot()
        base_region = self._selected_boundary_region_for_cutter()
        effective_kind = self._planar_point_cutter_kind(kind)
        try:
            handle = self.document.add_point_shape_from_world_points(
                effective_kind,
                points,
                reference_plane,
            )
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        split_handles = self._try_create_boundary_selector_split(
            base_region,
            self.document.node(handle),
        )
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()
        if split_handles:
            self.scene_tree.select_handles(list(split_handles))
        else:
            self.scene_tree.select_handle(handle)
        self.viewport.setFocus()
        if split_handles:
            self.statusBar().showMessage(
                "Created cutter and split BoundaryRegion into inside/outside.",
                5000,
            )
        else:
            self.statusBar().showMessage(
                viewport_shape_created_message(self.document.node(handle).name),
                5000,
            )

    def _selected_boundary_region_for_cutter(self) -> BoundaryRegion | None:
        handles = self.scene_tree.selected_handles()
        if len(handles) != 1:
            return None
        try:
            node = self.document.node(handles[0])
        except KeyError:
            return None
        if isinstance(node, BoundaryRegion) and node.patch_id is not None:
            return node
        return None

    def _active_planar_cutter_kind(self, kind: str) -> str | None:
        active = self.viewport.active_boundary_cutter_tool()
        if active is None:
            return None
        cutter_kind, shape_kind = active
        if cutter_kind == "planar" and shape_kind == kind:
            return shape_kind
        return None

    def _active_surface_cutter_kind(self, kind: str) -> str | None:
        active = self.viewport.active_boundary_cutter_tool()
        if active is None:
            return None
        cutter_kind, shape_kind = active
        if cutter_kind == "surface" and shape_kind == kind:
            return shape_kind
        return None

    def _planar_point_cutter_kind(self, kind: str) -> str:
        active = self._active_planar_cutter_kind(kind)
        if active == "polyline":
            return "polygon"
        if active == "quadratic_bezier_polycurve":
            return "quadratic_bezier_surface"
        return kind

    def _mark_internal_boundary_selector(
        self,
        selector: SDFNode,
        label: str,
    ) -> None:
        if self.document.is_internal_scene_node(selector):
            return
        selector.name = f"{INTERNAL_BOUNDARY_SELECTOR_PREFIX}{label}_{selector.name}"
        self.document.mark_changed()

    def _add_planar_segment_cutter_region(
        self,
        base_region: BoundaryRegion,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> int:
        _selectors, normals = self._viewport_boundary_region_entries([base_region])
        normal = np.asarray(normals[0], dtype=np.float64)
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= 1.0e-9:
            raise ValueError(
                "The selected BoundaryRegion has no planar normal for a segment cutter."
            )
        normal /= normal_length
        start_array = np.asarray(start, dtype=np.float64)
        end_array = np.asarray(end, dtype=np.float64)
        line = end_array - start_array
        line_length = float(np.linalg.norm(line))
        if line_length <= 1.0e-9:
            raise ValueError("Planar segment cutter length must be nonzero.")
        line_axis = line / line_length
        side_axis = np.cross(normal, line_axis)
        side_length = float(np.linalg.norm(side_axis))
        if side_length <= 1.0e-9:
            raise ValueError(
                "Planar segment cutter must lie across the selected boundary."
            )
        side_axis /= side_length
        assert self.document.fluid_domain is not None
        bounds = self.document.fluid_domain.root.bounding_box()
        span = max(
            bounds.x_max - bounds.x_min,
            bounds.y_max - bounds.y_min,
            bounds.z_max - bounds.z_min,
            line_length,
            1.0,
        ) * 2.0
        midpoint = 0.5 * (start_array + end_array)
        origin = midpoint - line_axis * span
        return self.document.add_polygon(
            (
                (0.0, 0.0),
                (2.0 * span, 0.0),
                (2.0 * span, 2.0 * span),
                (0.0, 2.0 * span),
            ),
            name=f"{INTERNAL_BOUNDARY_SELECTOR_PREFIX}planar_segment_cutter",
            origin=tuple(float(value) for value in origin),
            axis_u=tuple(float(value) for value in line_axis),
            axis_v=tuple(float(value) for value in side_axis),
        )

    def _try_create_boundary_selector_split(
        self,
        base_region: BoundaryRegion | None,
        selector: SDFNode,
    ) -> tuple[int, int] | None:
        if base_region is None:
            return None
        try:
            handles = self.document.add_boundary_selector_split_regions(
                base_region,
                selector,
            )
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return None
        signals.log_message.emit(
            "info",
            "Boundary cutter split created: inside and outside regions.",
        )
        return handles

    def _start_viewport_move(self) -> None:
        handles = self.scene_tree.selected_handles()
        if len(handles) != 1:
            signals.log_message.emit(
                "warning", "Select exactly one Scene object before using Move."
            )
            return
        self.viewport.begin_move_tool(handles[0])

    def _start_viewport_rotate(self) -> None:
        handles = self.scene_tree.selected_handles()
        if len(handles) != 1:
            signals.log_message.emit(
                "warning", "Select exactly one Scene object before using Rotate."
            )
            return
        self.viewport.begin_rotate_tool(handles[0])

    @Slot()
    def _copy_selection(self) -> None:
        handles = self.scene_tree.selected_handles()
        if not handles:
            signals.log_message.emit(
                "warning",
                "Select one or more SDF objects to copy.",
            )
            return
        try:
            self._clipboard_nodes = self.document.copy_nodes(handles)
        except (KeyError, ValueError) as error:
            signals.log_message.emit("warning", str(error))
            return
        count = len(self._clipboard_nodes)
        signals.log_message.emit(
            "info",
            f"Copied {count} SDF object{'s' if count != 1 else ''}.",
        )

    @Slot()
    def _paste_clipboard(self) -> None:
        if not self._clipboard_nodes:
            signals.log_message.emit(
                "warning",
                "Copy an SDF object before pasting.",
            )
            return
        self._paste_nodes(self._clipboard_nodes, "Pasted")

    def _paste_nodes(self, nodes: list[SDFNode], action_name: str) -> list[int]:
        undo_snapshot = self._history_snapshot()
        handles = self.document.paste_nodes(
            nodes,
            self.viewport.paste_offset(),
        )
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document(clear_selection=False)
        self.scene_tree.select_handles(handles)
        count = len(handles)
        self.statusBar().showMessage(
            f"{action_name} {count} SDF object{'s' if count != 1 else ''}",
            5000,
        )
        return handles

    @Slot()
    def _duplicate_selection(self) -> None:
        handles = self.scene_tree.selected_handles()
        if not handles:
            signals.log_message.emit(
                "warning",
                "Select one or more SDF objects to duplicate.",
            )
            return
        try:
            copied_nodes = self.document.copy_nodes(handles)
        except (KeyError, ValueError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self._paste_nodes(copied_nodes, "Duplicated")

    @Slot()
    def _rename_selection(self) -> None:
        handles = self.scene_tree.selected_handles()
        if len(handles) != 1:
            signals.log_message.emit(
                "warning",
                "Select exactly one scene item to rename.",
            )
            return
        signals.node_selected.emit(handles[0])
        if not self.properties.focus_name_editor():
            signals.log_message.emit("warning", "No editable name field is available.")
            return
        self.statusBar().showMessage("Rename selected scene item", 3000)

    @Slot()
    def _delete_selection(self) -> None:
        handles = self.scene_tree.selected_handles()
        if handles:
            self._on_delete_nodes(handles)
            return
        if MainWindow._cancel_viewport_tool(self):
            return

    @Slot()
    def _select_all_scene_items(self) -> None:
        handles = scene_item_handles(self.document)
        if not handles:
            return
        self.scene_tree.select_handles(handles)
        self.statusBar().showMessage(f"Selected {len(handles)} scene items", 3000)

    @Slot()
    def _clear_selection(self) -> None:
        if MainWindow._cancel_viewport_tool(self):
            return
        if not self.scene_tree.selected_handles():
            return
        self.scene_tree.select_handles([])
        self.statusBar().showMessage("Selection cleared", 3000)

    def _cancel_viewport_tool(self) -> bool:
        viewport = getattr(self, "viewport", None)
        cancel_tool = getattr(viewport, "cancel_active_interaction_tool", None)
        if not callable(cancel_tool):
            return False
        if not cancel_tool():
            return False
        self.statusBar().showMessage("Viewport tool cancelled", 3000)
        return True

    @Slot(int, object)
    def _on_viewport_move_preview(
        self,
        handle: int,
        delta: tuple[float, float, float],
    ) -> None:
        """Render a non-committing ghost of the move during the drag. Builds a
        throwaway document so the live tree/undo history stay untouched; the
        viewport restores the committed scene on cancel or after commit."""
        try:
            node = self.document.node(handle)
            if (
                isinstance(node, SDFNode)
                and self.viewport.show_move_preview(node.object_id, delta)
            ):
                return
            preview = self.document.snapshot()
            preview.move_object(handle, delta)
            version, tree = preview.visual_snapshot()
            artifact = build_render_artifact(
                RenderSceneSnapshot(
                    version=version,
                    tree=tree,
                )
            )
        except (KeyError, ValueError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self.viewport.show_scene_preview(artifact.surface_scene)

    def _on_viewport_move_requested(
        self,
        handle: int,
        delta: tuple[float, float, float],
    ) -> None:
        undo_snapshot = self._history_snapshot()
        try:
            moved_handle = self.document.move_object(handle, delta)
        except (KeyError, ValueError) as error:
            signals.log_message.emit("warning", str(error))
            return
        moved_node = self.document.node(moved_handle)
        if (
            moved_handle == handle
            and isinstance(moved_node, SDFNode)
            and self.viewport.has_scene_object_id(moved_node.object_id)
            and self.viewport.can_defer_committed_move(moved_node.object_id)
        ):
            self.viewport.apply_committed_move_preview(moved_node.object_id, delta)
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document(
            clear_selection=False,
            render=True,
            notify_document=False,
        )
        self.scene_tree.select_handle(moved_handle)
        self.statusBar().showMessage(
            f"Object moved on the {self.viewport.reference_plane_label} "
            "reference plane",
            5000,
        )

    def _on_viewport_rotate_preview(
        self,
        handle: int,
        axis: str,
        angle_degrees: float,
        pivot: tuple[float, float, float],
    ) -> None:
        """Non-committing ghost of a rotation drag (see _on_viewport_move_preview)."""
        try:
            preview = self.document.snapshot()
            preview.rotate_object(handle, axis, angle_degrees, pivot)
            version, tree = preview.visual_snapshot()
            artifact = build_render_artifact(
                RenderSceneSnapshot(
                    version=version,
                    tree=tree,
                )
            )
        except (KeyError, ValueError, NotImplementedError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self.viewport.show_scene_preview(artifact.surface_scene)

    @Slot(int, str, float, object)
    def _on_viewport_rotate_requested(
        self,
        handle: int,
        axis: str,
        angle_degrees: float,
        pivot: tuple[float, float, float],
    ) -> None:
        undo_snapshot = self._history_snapshot()
        try:
            rotated_handle = self.document.rotate_object(
                handle,
                axis,
                angle_degrees,
                pivot,
            )
        except (KeyError, ValueError, NotImplementedError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document(
            clear_selection=False,
            render=True,
            notify_document=False,
        )
        self.scene_tree.select_handle(rotated_handle)
        self.statusBar().showMessage(
            f"Object rotated {angle_degrees:.1f} deg around {axis.upper()}",
            5000,
        )

    def _on_viewport_extrude_preview(self, handle: int, signed_height: float) -> None:
        try:
            preview = self.document.snapshot()
            new_handle = preview.solid_from_2d(
                [handle], "extrude", signed_height=signed_height)
            version, tree = preview.visual_snapshot()
            artifact = build_render_artifact(
                RenderSceneSnapshot(
                    version=version,
                    tree=tree,
                )
            )
        except (KeyError, ValueError) as error:
            signals.log_message.emit("warning", str(error))
            return
        del new_handle
        self.viewport.show_scene_preview(artifact.surface_scene)

    def _on_viewport_revolve_preview(
        self,
        handle,
        axis_name,
        axis_origin,
        axis_direction,
        radial_direction,
        angle_degrees,
    ) -> None:
        if (
            axis_name not in {"u", "v"}
            or not np.isfinite(float(angle_degrees))
            or abs(float(angle_degrees)) <= 1.0e-6
        ):
            return
        try:
            preview = self.document.snapshot()
            preview.solid_from_2d(
                [handle], "revolve",
                revolve_axis=axis_name,
                revolve_axis_origin=axis_origin,
                revolve_axis_direction=axis_direction,
                revolve_radial_direction=radial_direction,
                revolve_angle_degrees=angle_degrees,
            )
            version, tree = preview.visual_snapshot()
            artifact = build_render_artifact(
                RenderSceneSnapshot(
                    version=version,
                    tree=tree,
                )
            )
        except (KeyError, ValueError) as error:
            if "revolve angle magnitude must be finite" in str(error):
                return
            signals.log_message.emit("warning", str(error))
            return
        self.viewport.show_scene_preview(artifact.surface_scene)

    @Slot(int, object, object)
    def _on_viewport_transform_requested(
        self,
        handle: int,
        delta: tuple[float, float, float],
        rotations: tuple[tuple[str, float, tuple[float, float, float]], ...],
    ) -> None:
        undo_snapshot = self._history_snapshot()
        try:
            transformed_handle = handle
            if max(abs(component) for component in delta) > 1.0e-12:
                transformed_handle = self.document.move_object(
                    transformed_handle,
                    delta,
                )
            for axis, angle_degrees, pivot in rotations:
                if abs(angle_degrees) > 1.0e-6:
                    transformed_handle = self.document.rotate_object(
                        transformed_handle,
                        axis,
                        angle_degrees,
                        pivot,
                    )
        except (KeyError, ValueError, NotImplementedError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document(
            clear_selection=False,
            render=True,
            notify_document=False,
        )
        self.scene_tree.select_handle(transformed_handle)
        self.statusBar().showMessage("Object transform applied", 5000)

    @Slot(int, float)
    def _on_viewport_extrude_requested(
        self,
        handle: int,
        signed_height: float,
    ) -> None:
        undo_snapshot = self._history_snapshot()
        try:
            solid_handle = self.document.solid_from_2d(
                [handle],
                "extrude",
                signed_height=signed_height,
            )
        except (ValueError, KeyError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document(clear_selection=False)
        self.scene_tree.select_handle(solid_handle)
        self.statusBar().showMessage(
            f"Extruded {abs(signed_height):.5g} m",
            5000,
        )

    @Slot(int, str, object, object, object, float)
    def _on_viewport_revolve_requested(
        self,
        handle: int,
        axis_name: str,
        axis_origin: tuple[float, float, float],
        axis_direction: tuple[float, float, float],
        radial_direction: tuple[float, float, float],
        angle_degrees: float,
    ) -> None:
        undo_snapshot = self._history_snapshot()
        try:
            solid_handle = self.document.solid_from_2d(
                [handle],
                "revolve",
                revolve_axis=axis_name,
                revolve_axis_origin=axis_origin,
                revolve_axis_direction=axis_direction,
                revolve_radial_direction=radial_direction,
                revolve_angle_degrees=angle_degrees,
            )
        except (ValueError, KeyError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document(clear_selection=False)
        self.scene_tree.select_handle(solid_handle)
        self.statusBar().showMessage(
            f"Revolved around {world_axis_label(axis_direction)} "
            f"{angle_degrees:.5g} degrees",
            5000,
        )

    @Slot(int)
    def _on_viewport_scene_object_selected(self, object_id: int) -> None:
        if object_id == 0:
            self.scene_tree.tree.clearSelection()
            return
        for handle, node, _parent in self.document.walk():
            if isinstance(node, SDFNode) and node.object_id == object_id:
                self.scene_tree.select_handle(handle)
                return

    def _start_boundary_region_tool(self) -> None:
        if self.document.fluid_domain is None:
            signals.log_message.emit(
                "warning",
                "Select a FluidDomain root before creating a boundary tag.",
            )
            return
        self.viewport.begin_boundary_region_tool(
            self.document.fluid_domain.root,
            tuple(
                selector
                for selector in self.document.fluid_domain.selector_objects
                if not self.document.is_internal_scene_node(selector)
            ),
        )

    @Slot(object)
    def _on_viewport_boundary_hovered(
        self,
        selection: BoundaryPatchHit | None,
    ) -> None:
        if selection is None:
            self.viewport.set_boundary_hover(0, None, None, None)
            self.statusBar().showMessage("No FluidDomain boundary under cursor")
            return
        owner_object_id = selection.owner_object_id
        outside_direction = selection.outside_direction
        owner = next(
            (
                node
                for _handle, node, _parent in self.document.walk()
                if node.object_id == owner_object_id
            ),
            None,
        )
        self.viewport.set_boundary_hover(
            owner_object_id,
            outside_direction,
            selection.normal,
            selection,
        )
        if owner is not None:
            selector_suffix = (
                f" / {selection.selector.side}"
                if selection.selector is not None
                else ""
            )
            self.statusBar().showMessage(
                f"Boundary patch: {owner.name} {selection.patch_id}{selector_suffix}"
            )

    @Slot(object)
    def _on_viewport_boundary_region_requested(
        self,
        selection: BoundaryPatchHit,
    ) -> None:
        self._create_boundary_region_from_hit(selection)

    @Slot(object)
    def _on_create_boundary_region(self, handles: list[int]) -> None:
        if len(handles) == 2:
            self._create_boundary_selector_region(handles)
            return
        if len(handles) != 1:
            return
        try:
            owner = self.document.node(handles[0])
        except KeyError:
            return
        if not hasattr(owner, "children"):
            signals.log_message.emit(
                "warning", "Boundary tag owners must be SDF objects."
            )
            return
        self._create_boundary_region(owner.object_id, None)

    def _create_boundary_selector_region(self, handles: list[int]) -> None:
        try:
            first = self.document.node(handles[0])
            second = self.document.node(handles[1])
        except KeyError:
            return
        if isinstance(first, BoundaryRegion) and isinstance(second, SDFNode):
            base_region = first
            selector = second
        elif isinstance(second, BoundaryRegion) and isinstance(first, SDFNode):
            base_region = second
            selector = first
        else:
            return
        undo_snapshot = self._history_snapshot()
        try:
            handles = self.document.add_boundary_selector_split_regions(
                base_region,
                selector,
            )
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()
        self.scene_tree.select_handles(list(handles))

    def _create_boundary_region(
        self,
        owner_object_id: int,
        outside_direction: int | None,
    ) -> None:
        undo_snapshot = self._history_snapshot()
        try:
            handle = self.document.add_boundary_region(
                owner_object_id,
                outside_direction,
            )
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()
        self.scene_tree.select_handle(handle)

    def _create_boundary_region_from_hit(self, hit: BoundaryPatchHit) -> None:
        undo_snapshot = self._history_snapshot()
        try:
            handle = self.document.add_boundary_region_from_hit(hit)
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()
        self.scene_tree.select_handle(handle)

    @Slot(object)
    def _on_create_polygon_from_polyline(self, handles: list[int]) -> None:
        if len(handles) != 1:
            signals.log_message.emit(
                "warning", "Select exactly one polyline to create a polygon."
            )
            return
        undo_snapshot = self._history_snapshot()
        try:
            handle = self.document.create_polygon_from_polyline(handles[0])
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()
        self.scene_tree.select_handle(handle)

    def _viewport_outside_direction_normal(
        self,
        owner: object,
        outside_direction: int | None,
    ) -> tuple[float, float, float] | None:
        if not isinstance(owner, SDFNode) or outside_direction is None:
            return None
        direction = owner_outside_direction_vector(owner, outside_direction)
        if direction is None:
            return None
        return tuple(float(value) for value in direction)

    def _viewport_boundary_region_entries(
        self,
        regions: list[BoundaryRegion],
    ) -> tuple[
        tuple[tuple[int, int], ...],
        tuple[tuple[float, float, float], ...],
    ]:
        owner_by_id = {
            node.object_id: node
            for _handle, node, _parent in self.document.walk()
            if isinstance(node, SDFNode)
        }
        selectors: list[tuple[int, int]] = []
        normals: list[tuple[float, float, float]] = []
        for region in regions:
            normal = self._viewport_outside_direction_normal(
                owner_by_id.get(region.owner_object_id),
                region.outside_direction,
            )
            selectors.append(
                (
                    region.owner_object_id,
                    1 if region.outside_direction is None else 0,
                )
            )
            normals.append(normal if normal is not None else (0.0, 0.0, 0.0))
        return tuple(selectors), tuple(normals)

    @Slot(object)
    def _on_delete_nodes(self, handles: list[int]) -> None:
        if not handles:
            return
        undo_snapshot = self._history_snapshot()
        deleted = self.document.delete_many(handles)
        if deleted <= 0:
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()

    @Slot(str, object)
    def _on_sdf_op_preview_requested(self, operation: str, handles: list[int]) -> None:
        if not operation:
            QTimer.singleShot(0, self.viewport.clear_boolean_preview)
            return
        if len(handles) != 2:
            self.viewport.clear_boolean_preview()
            return
        try:
            first = self.document.node(handles[0])
            second = self.document.node(handles[1])
        except KeyError:
            self.viewport.clear_boolean_preview()
            return
        if (
            not isinstance(first, SDFNode)
            or not isinstance(second, SDFNode)
            or first.dimension != 3
            or second.dimension != 3
            or not self.document.can_combine(handles[0], handles[1])
        ):
            self.viewport.clear_boolean_preview()
            return
        try:
            preview = self.document.snapshot()
            preview.combine(handles[0], handles[1], operation)
            version, tree = preview.visual_snapshot()
            artifact = build_render_artifact(
                RenderSceneSnapshot(
                    version=version,
                    tree=tree,
                )
            )
        except (KeyError, ValueError):
            self.viewport.clear_boolean_preview()
            return
        self.viewport.show_scene_preview(artifact.surface_scene, preview_kind="boolean")

    @Slot(str, object)
    def _on_sdf_op_requested(self, operation: str, handles: list[int]) -> None:
        if len(handles) != 2:
            signals.log_message.emit(
                "warning", "Select exactly two independent SDF nodes."
            )
            return
        preview_ids: tuple[int, int] | None = None
        try:
            first = self.document.node(handles[0])
            second = self.document.node(handles[1])
            if (
                isinstance(first, SDFNode)
                and isinstance(second, SDFNode)
                and first.dimension == 3
                and second.dimension == 3
            ):
                preview_ids = (first.object_id, second.object_id)
        except KeyError:
            preview_ids = None
        undo_snapshot = self._history_snapshot()
        try:
            handle = self.document.combine(handles[0], handles[1], operation)
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        if preview_ids is not None:
            self.viewport.apply_committed_boolean_preview(
                operation,
                preview_ids[0],
                preview_ids[1],
            )
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()
        self.scene_tree.select_handle(handle)

    @Slot(object)
    def _on_selection_changed(self, handles: list[int]) -> None:
        self.viewport.clear_boolean_preview()
        selected_names: list[str] = []
        scene_selection = None
        boundary_regions: list[BoundaryRegion] = []
        for handle in handles:
            try:
                node = self.document.node(handle)
            except KeyError:
                continue
            selected_names.append(node.name)
            if isinstance(node, BoundaryRegion):
                boundary_regions.append(node)
                continue
            if not isinstance(node, SDFNode):
                continue
            if len(handles) == 1:
                scene_selection = node
        boundary_selectors, boundary_normals = (
            self._viewport_boundary_region_entries(boundary_regions)
        )
        self.viewport.set_boundary_region_selection_entries(
            boundary_selectors,
            boundary_normals,
            tuple(boundary_regions),
        )
        self.viewport.set_scene_selection(scene_selection)
        if selected_names:
            self.statusBar().showMessage(
                f"Selection: {', '.join(selected_names)}"
            )
        else:
            self.statusBar().clearMessage()

    @Slot(str, object)
    def _on_transform_requested(
        self, transform: str, handles: list[int]
    ) -> None:
        if len(handles) != 1:
            return
        undo_snapshot = self._history_snapshot()
        try:
            if transform == "translate":
                handle = self.document.move_object(handles[0], (0.1, 0.0, 0.0))
            elif transform == "rotate":
                handle = self.document.rotate_object(handles[0], "y", 15.0)
            else:
                handle = self.document.wrap_transform(handles[0], transform)
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()
        self.scene_tree.select_handle(handle)

    @Slot(str, object)
    def _on_solid_from_2d(self, method: str, handles: list[int]) -> None:
        try:
            base_method, revolve_axis = parse_solid_from_2d_method(method)
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        if base_method in {"extrude", "revolve"}:
            if len(handles) != 1:
                signals.log_message.emit(
                    "warning",
                    f"Select exactly one placed 2D SDF before using "
                    f"{base_method.title()}.",
                )
                return
            try:
                node = self.document.node(handles[0])
            except KeyError as error:
                signals.log_message.emit("warning", str(error))
                return
            if not isinstance(node, PlacedSDF2D):
                signals.log_message.emit(
                    "warning",
                    f"{base_method.title()} requires one placed 2D SDF.",
                )
                return
            if node not in self.document.objects:
                signals.log_message.emit(
                    "warning",
                    f"{base_method.title()} requires a top-level placed 2D SDF.",
                )
                return
            if base_method == "extrude":
                self.viewport.begin_extrude_tool(handles[0], node)
            else:
                self.viewport.begin_revolve_tool(
                    handles[0],
                    node,
                    axis_name=revolve_axis,
                )
            return
        undo_snapshot = self._history_snapshot()
        try:
            handle = self.document.solid_from_2d(handles, base_method)
        except (ValueError, KeyError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()
        self.scene_tree.select_handle(handle)

    @Slot(object)
    def _on_set_fluid_root(self, handles: list[int]) -> None:
        self._on_set_domain(handles, DomainKind.FLUID.value)

    @Slot(object, str)
    def _on_set_domain(self, handles: list[int], kind: str) -> None:
        if len(handles) != 1:
            return
        try:
            domain_kind = DomainKind(kind)
        except ValueError:
            signals.log_message.emit("warning", f"Unknown Domain kind: {kind}")
            return
        undo_snapshot = self._history_snapshot()
        try:
            self.document.set_domain_root(handles[0], domain_kind)
        except (ValueError, KeyError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document(clear_selection=False)

    @Slot(object, bool)
    def _on_set_tag_enabled(self, handles: list[int], enabled: bool) -> None:
        if len(handles) != 1:
            return
        undo_snapshot = self._history_snapshot()
        try:
            self.document.set_tag_enabled(handles[0], enabled)
        except (ValueError, KeyError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document(clear_selection=False)

    @Slot()
    def _on_node_edited(self) -> None:
        # Property editors emit from inside Qt widget callbacks. Preserve the
        # editor and selection until that callback returns; deleting the
        # emitting widget here can crash Qt's accessibility event handling.
        self.document.mark_changed()
        self._publish_document(clear_selection=False)

    @Slot()
    def _frame_scene(self) -> None:
        selected_box = selected_sdf_bounding_box(
            self.document,
            self.scene_tree.selected_handles(),
        )
        if selected_box is not None:
            self.viewport.frame_box(selected_box)
            self.statusBar().showMessage("Framed selected scene item", 3000)
            return
        if self.document.fluid_domain is not None:
            self.viewport.frame_box(self.document.fluid_domain.bounding_box())
            self.statusBar().showMessage("Framed fluid domain", 3000)
            return
        self.viewport.frame_default_grid()
        self.statusBar().showMessage("Framed reference grid", 3000)

    @Slot()
    def _new_default_scene(self) -> None:
        undo_snapshot = self._history_snapshot()
        self.document = SceneDocument()
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document(frame=True)

    @Slot()
    def _new_empty_scene(self) -> None:
        undo_snapshot = self._history_snapshot()
        self.document = SceneDocument()
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()

    @Slot()
    def _open_scene(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open casoCAD scene", "", "casoCAD Scene (*.json)"
        )
        if not path:
            return
        undo_snapshot = self._history_snapshot()
        try:
            self.document = load_scene(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.critical(self, "Open failed", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document(frame=True)
        self.statusBar().showMessage(f"Opened {path}", 5000)

    @Slot()
    def _save_scene(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save casoCAD scene",
            "scene.json",
            "casoCAD Scene (*.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path = f"{path}.json"
        try:
            save_scene(self.document, path)
        except OSError as error:
            QMessageBox.critical(self, "Save failed", str(error))
            return
        self.statusBar().showMessage(f"Saved {path}", 5000)

    @Slot(str, str)
    def _on_log_message(self, level: str, message: str) -> None:
        getattr(logger, level, logger.info)(message)
        self.statusBar().showMessage(message, 8000)

    def closeEvent(self, event: object) -> None:
        self._render_request_timer.stop()
        if self._meshing_workspace is not None:
            self._meshing_workspace.close()
        self.artifacts.shutdown()
        super().closeEvent(event)
