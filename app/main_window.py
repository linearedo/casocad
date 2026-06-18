from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import pickle
import sys
import tempfile

import numpy as np
from PySide6.QtCore import QProcess, Qt, Slot
from PySide6.QtGui import QAction, QActionGroup, QColor, QKeySequence
from PySide6.QtWidgets import (
    QColorDialog,
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QLabel,
    QPushButton,
    QSlider,
    QToolBar,
)

from app.artifacts import (
    ArtifactManager,
    RenderArtifact,
    build_render_artifact,
    empty_render_scene_source,
)
from app.panels.export_panel import ExportPanel
from app.panels.log_panel import LogPanel
from app.panels.mesher_panel import MesherPanel
from app.panels.properties import CadDimensionSpinBox, PropertiesPanel
from app.panels.scene_tree import SceneTreePanel
from app.signals import signals
from app.viewport.gl_widget import GLWidget
from core.boundary import BoundaryRegion
from core.boundary_direction import (
    owner_outside_direction_from_normal,
    owner_outside_direction_vector,
)
from core.sdf import PlacedSDF2D
from core.sdf.base import BoundingBox3D, SDFNode
from core.serialization import load_scene, save_scene
from core.scene import SceneDocument
from scenes.boolean_operations import build_scene as build_boolean_scene
from scenes.lattice_benchmark import build_scene as build_benchmark_scene
from scenes.pipe_3d import build_scene as build_pipe_scene
from scenes.placed_section_tags import build_scene as build_tagging_scene

logger = logging.getLogger(__name__)
UNDO_HISTORY_LIMIT = 50
SNAP_TOGGLE_SHORTCUT = "G"
REDO_SHORTCUTS = ("Ctrl+Y", "Ctrl+Shift+Z")
SELECT_ALL_SHORTCUT = "Ctrl+A"
DUPLICATE_SHORTCUT = "Ctrl+D"
RENAME_SHORTCUT = "F2"
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
        self.document = SceneDocument.default()
        self._undo_stack: list[SceneDocument] = []
        self._redo_stack: list[SceneDocument] = []
        self._clipboard_nodes: list[SDFNode] = []
        self.viewport = GLWidget()
        self._background_color = QColor(DEFAULT_BACKGROUND_HEX)
        self.setCentralWidget(self.viewport)
        self.artifacts = ArtifactManager(self)
        self.scene_tree = SceneTreePanel()
        self.properties = PropertiesPanel()
        self.mesher_panel = MesherPanel()
        self.export_panel = ExportPanel()
        self.log_panel = LogPanel()
        self._thread: QProcess | None = None
        self._meshing_mode: str | None = None
        self._meshing_version: int | None = None
        self._temporary_mesh_path: Path | None = None
        self._mesh_input_path: Path | None = None
        self._mesh_result_path: Path | None = None
        self._mesh_preview_chunk_paths: list[Path] = []
        self._mesh_stdout_buffer = ""
        self._mesh_error_message: str | None = None
        self._build_docks()
        self._build_menu()
        self._build_toolbar()
        self._connect_signals()
        self._seed_initial_viewport_scene()
        self._publish_document(frame=True)

    def _build_docks(self) -> None:
        scene_dock = self._dock("Scene", self.scene_tree)
        properties_dock = self._dock("Properties", self.properties)
        mesher_dock = self._dock("Mesher", self.mesher_panel)
        export_dock = self._dock("Export", self.export_panel)
        log_dock = self._dock("Log", self.log_panel)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, scene_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, properties_dock)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, mesher_dock)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, export_dock)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, log_dock)
        self.tabifyDockWidget(export_dock, log_dock)
        self.tabifyDockWidget(properties_dock, mesher_dock)
        properties_dock.raise_()

    def _dock(self, title: str, widget: object) -> QDockWidget:
        dock = QDockWidget(title, self)
        dock.setObjectName(f"{title.lower()}Dock")
        dock.setWidget(widget)
        return dock

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("CAD", self)
        toolbar.setObjectName("cadToolbar")
        self.addToolBar(toolbar)
        fit_action = toolbar.addAction("Frame Scene")
        fit_action.setShortcuts(
            tuple(QKeySequence(shortcut) for shortcut in FRAME_SHORTCUTS)
        )
        fit_action.triggered.connect(self._frame_scene)
        toolbar.addSeparator()
        mode_group = QActionGroup(self)
        mode_group.setExclusive(True)
        self._sdf_action = QAction("SDF", self, checkable=True)
        self._sdf_action.setChecked(True)
        self._lattice_action = QAction("Lattice", self, checkable=True)
        mode_group.addAction(self._sdf_action)
        mode_group.addAction(self._lattice_action)
        toolbar.addAction(self._sdf_action)
        toolbar.addAction(self._lattice_action)
        self._sdf_action.triggered.connect(lambda: self.viewport.set_mode("sdf"))
        self._lattice_action.triggered.connect(
            lambda: self.viewport.set_mode("lattice")
        )
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
        file_menu = self.menuBar().addMenu("&File")
        new_default = file_menu.addAction("New von Karman Scene")
        new_default.triggered.connect(self._new_default_scene)
        new_empty = file_menu.addAction("New Empty Scene")
        new_empty.triggered.connect(self._new_empty_scene)
        examples_menu = file_menu.addMenu("New Example")
        for label, factory in (
            ("Pipe with inlet/outlet", build_pipe_scene),
            ("von Karman obstacle", SceneDocument.default),
            ("Boolean operations", build_boolean_scene),
            ("Placed section tags", build_tagging_scene),
            ("Smooth union benchmark", build_benchmark_scene),
        ):
            action = examples_menu.addAction(label)
            action.triggered.connect(
                lambda checked=False, value=factory: self._load_example(value)
            )
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

        edit_menu = self.menuBar().addMenu("&Edit")
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

    def _connect_signals(self) -> None:
        signals.add_primitive_requested.connect(self._on_add_primitive)
        signals.viewport_shape_drawn.connect(self._on_viewport_shape_drawn)
        signals.viewport_point_shape_drawn.connect(self._on_viewport_point_shape_drawn)
        signals.viewport_move_tool_requested.connect(self._start_viewport_move)
        signals.viewport_boundary_tool_requested.connect(
            self._start_boundary_region_tool
        )
        signals.viewport_move_requested.connect(self._on_viewport_move_requested)
        signals.viewport_rotate_requested.connect(self._on_viewport_rotate_requested)
        signals.viewport_transform_requested.connect(
            self._on_viewport_transform_requested
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
        signals.csg_requested.connect(self._on_csg_requested)
        signals.transform_requested.connect(self._on_transform_requested)
        signals.solid_from_2d_requested.connect(self._on_solid_from_2d)
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
        signals.export_requested.connect(self._on_export_requested)
        signals.mesh_requested.connect(self._on_mesh_requested)
        signals.log_message.connect(self._on_log_message)
        self.artifacts.render_ready.connect(self._on_render_artifact_ready)
        self.artifacts.render_failed.connect(self._on_render_artifact_failed)

    def _publish_document(
        self,
        frame: bool = False,
        clear_selection: bool = True,
        render: bool = True,
    ) -> None:
        signals.document_changed.emit(self.document)
        if clear_selection:
            signals.node_selected.emit(None)
        if not self.document.objects:
            self.viewport.set_scene_artifact(None, empty_render_scene_source())
            self._sdf_action.setChecked(True)
            self.viewport.set_mode("sdf")
            self.viewport.configure_default_grid()
            self.viewport.frame_default_grid()
            self._sync_grid_spacing_control()
            return
        snapshot = None
        if render:
            try:
                snapshot = self.document.snapshot()
            except ValueError as error:
                signals.log_message.emit("warning", str(error))
                return
            self.artifacts.request_render(snapshot)
        if self.document.fluid_domain is not None:
            domain = (
                snapshot.fluid_domain
                if snapshot is not None
                else self.document.fluid_domain
            )
            assert domain is not None
            box = domain.bounding_box()
            self.viewport.configure_grid(box, self.mesher_panel.config().dx)
            self._sync_grid_spacing_control()
            if frame:
                self.viewport.frame_box(box)

    def _seed_initial_viewport_scene(self) -> None:
        if not self.document.objects:
            return
        try:
            artifact = build_render_artifact(self.document.snapshot())
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        self.viewport.set_scene_artifact(artifact.tree, artifact.scene_source)

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
        self.viewport.set_scene_artifact(artifact.tree, artifact.scene_source)

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

    @Slot(str, object, object, object)
    def _on_viewport_shape_drawn(
        self,
        kind: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        parameters: dict[str, float] | None,
    ) -> None:
        undo_snapshot = self._history_snapshot()
        handle = self.document.add_primitive_from_drag(
            kind,
            start,
            end,
            parameters=parameters,
        )
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()
        self.scene_tree.select_handle(handle)
        self.viewport.setFocus()
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
        try:
            handle = self.document.add_point_shape_from_world_points(
                kind,
                points,
                reference_plane,
            )
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()
        self.scene_tree.select_handle(handle)
        self.viewport.setFocus()
        self.statusBar().showMessage(
            viewport_shape_created_message(self.document.node(handle).name),
            5000,
        )

    def _start_viewport_move(self) -> None:
        handles = self.scene_tree.selected_handles()
        if len(handles) != 1:
            signals.log_message.emit(
                "warning", "Select exactly one Scene object before using Move."
            )
            return
        self.viewport.begin_move_tool(handles[0])

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
        if MainWindow._cancel_viewport_tool(self):
            return
        self._on_delete_nodes(self.scene_tree.selected_handles())

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
    def _on_viewport_move_requested(
        self,
        handle: int,
        delta: tuple[float, float, float],
    ) -> None:
        undo_snapshot = self._history_snapshot()
        try:
            moved_handle = self.document.move_object(handle, delta)
            self.document.refresh_derived_geometry()
        except (KeyError, ValueError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document(
            clear_selection=False,
            render=True,
        )
        self.scene_tree.select_handle(moved_handle)
        self.statusBar().showMessage(
            f"Object moved on the {self.viewport.reference_plane_label} "
            "reference plane",
            5000,
        )

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
            self.document.refresh_derived_geometry()
        except (KeyError, ValueError, NotImplementedError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document(
            clear_selection=False,
            render=True,
        )
        self.scene_tree.select_handle(rotated_handle)
        self.statusBar().showMessage(
            f"Object rotated {angle_degrees:.1f} deg around {axis.upper()}",
            5000,
        )

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
            self.document.refresh_derived_geometry()
        except (KeyError, ValueError, NotImplementedError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document(
            clear_selection=False,
            render=True,
        )
        self.scene_tree.select_handle(transformed_handle)
        self.statusBar().showMessage("Object transform applied", 5000)

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
        self.viewport.begin_boundary_region_tool(self.document.fluid_domain.root)

    @Slot(object)
    def _on_viewport_boundary_hovered(
        self,
        selection: tuple[int, tuple[float, float, float]] | None,
    ) -> None:
        if selection is None:
            self.statusBar().showMessage("No FluidDomain boundary under cursor")
            return
        owner_object_id, normal = selection
        outside_direction = self._planar_outside_direction(
            owner_object_id,
            normal,
        )
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
            self._viewport_outside_direction_normal(owner, outside_direction),
        )
        if owner is not None:
            self.statusBar().showMessage(
                f"Boundary owner: {owner.name} [ID {owner_object_id}]"
            )

    @Slot(object)
    def _on_viewport_boundary_region_requested(
        self,
        selection: tuple[int, tuple[float, float, float]],
    ) -> None:
        owner_object_id, normal = selection
        outside_direction = self._planar_outside_direction(
            owner_object_id,
            normal,
        )
        self._create_boundary_region(owner_object_id, outside_direction)

    @Slot(object)
    def _on_create_boundary_region(self, handles: list[int]) -> None:
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

    def _planar_outside_direction(
        self,
        owner_object_id: int,
        normal: tuple[float, float, float],
    ) -> int | None:
        owner = next(
            (
                node
                for _handle, node, _parent in self.document.walk()
                if node.object_id == owner_object_id
            ),
            None,
        )
        if (
            isinstance(owner, PlacedSDF2D)
            and self.document.fluid_domain is not None
            and self.document.fluid_domain.root.dimension == 2
        ):
            normal_array = np.asarray(normal, dtype=np.float64)
            coordinates = (
                float(np.dot(normal_array, owner.axis_u)),
                float(np.dot(normal_array, owner.axis_v)),
            )
            axis = int(np.argmax(np.abs(coordinates)))
            if abs(coordinates[axis]) < 0.95:
                return None
            return 2 * axis + int(coordinates[axis] > 0.0)
        if owner is None:
            return None
        return owner_outside_direction_from_normal(owner, normal)

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
        selected_nodes = []
        for handle in handles:
            try:
                selected_nodes.append(self.document.node(handle))
            except KeyError:
                continue
        selected_ids = {id(node) for node in selected_nodes}
        roots = [
            node
            for node in selected_nodes
            if not self._has_selected_ancestor(node, selected_ids)
        ]
        if not roots:
            return
        undo_snapshot = self._history_snapshot()
        deleted = False
        for node in roots:
            try:
                self.document.delete(self.document.handle_for(node))
                deleted = True
            except KeyError:
                continue
        if not deleted:
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()

    def _has_selected_ancestor(
        self, target: object, selected_ids: set[int]
    ) -> bool:
        parent_by_child: dict[int, object] = {}
        for handle, node, parent_handle in self.document.walk():
            if parent_handle is not None:
                parent_by_child[id(node)] = self.document.node(parent_handle)
        parent = parent_by_child.get(id(target))
        while parent is not None:
            if id(parent) in selected_ids:
                return True
            parent = parent_by_child.get(id(parent))
        return False

    @Slot(str, object)
    def _on_csg_requested(self, operation: str, handles: list[int]) -> None:
        if len(handles) != 2:
            signals.log_message.emit(
                "warning", "Select exactly two independent SDF nodes."
            )
            return
        undo_snapshot = self._history_snapshot()
        try:
            handle = self.document.combine(handles[0], handles[1], operation)
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()
        self.scene_tree.select_handle(handle)

    @Slot(object)
    def _on_selection_changed(self, handles: list[int]) -> None:
        object_ids: set[int] = set()
        selected_names: list[str] = []
        geometry_filter = None
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
                object_ids.add(node.object_id)
                continue
            if not isinstance(node, SDFNode):
                continue
            if node.children():
                object_ids.update(self._attribution_ids(node))
                if len(handles) == 1:
                    geometry_filter = node
            else:
                object_ids.add(node.object_id)
            if len(handles) == 1:
                scene_selection = node
        boundary_selectors, boundary_normals = (
            self._viewport_boundary_region_entries(boundary_regions)
        )
        self.viewport.set_boundary_region_selection_entries(
            boundary_selectors,
            boundary_normals,
        )
        self.viewport.set_scene_selection(scene_selection)
        self.viewport.set_lattice_filter(
            object_ids or None,
            geometry=geometry_filter,
        )
        if selected_names:
            self.statusBar().showMessage(
                f"Lattice selection: {', '.join(selected_names)}"
            )
        else:
            self.statusBar().clearMessage()

    def _attribution_ids(self, node: SDFNode) -> set[int]:
        children = node.children()
        if not children:
            return {node.object_id}
        result: set[int] = set()
        for child in children:
            result.update(self._attribution_ids(child))
        return result or {node.object_id}

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
        undo_snapshot = self._history_snapshot()
        try:
            handle = self.document.solid_from_2d(handles, method)
        except (ValueError, KeyError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()
        self.scene_tree.select_handle(handle)

    @Slot(object)
    def _on_set_fluid_root(self, handles: list[int]) -> None:
        if len(handles) != 1:
            return
        undo_snapshot = self._history_snapshot()
        try:
            self.document.set_fluid_root(handles[0])
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
        try:
            self.document.refresh_derived_geometry()
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        self.document.mark_changed()
        self._publish_document(clear_selection=False)

    @Slot(str)
    def _on_export_requested(self, path: str) -> None:
        if not path.strip():
            signals.log_message.emit("warning", "Choose an Arrow output path.")
            return
        output = Path(path)
        if output.suffix.lower() != ".arrow":
            output = output.with_suffix(".arrow")
            self.export_panel.path.setText(str(output))
        self._start_meshing(output, "export")

    @Slot()
    def _on_mesh_requested(self) -> None:
        if self._thread is not None:
            return
        descriptor, path = tempfile.mkstemp(
            prefix="casocad-preview-", suffix=".arrow"
        )
        os.close(descriptor)
        self._temporary_mesh_path = Path(path)
        self._start_meshing(self._temporary_mesh_path, "preview")

    def _start_meshing(self, output: Path, mode: str) -> None:
        if self._thread is not None:
            if mode == "preview":
                self._remove_temporary_mesh()
            return
        if self.document.fluid_domain is None:
            signals.log_message.emit(
                "warning",
                "Select one 2D or 3D SDF as the Fluid Domain before meshing.",
            )
            self._remove_temporary_mesh()
            return
        snapshot = self.document.snapshot()
        config = self.mesher_panel.config()
        assert snapshot.fluid_domain is not None
        self.scene_tree.tree.clearSelection()
        self.viewport.set_lattice_filter(None)
        self.viewport.configure_grid(snapshot.fluid_domain.bounding_box(), config.dx)
        self._meshing_mode = mode
        self.mesher_panel.set_busy(True)
        self.export_panel.set_actions_enabled(False)
        if mode == "export":
            self.export_panel.set_busy(True)
        self._meshing_version = snapshot.version
        input_descriptor, input_path = tempfile.mkstemp(
            prefix="casocad-mesh-input-", suffix=".pickle"
        )
        os.close(input_descriptor)
        result_descriptor, result_path = tempfile.mkstemp(
            prefix="casocad-mesh-result-", suffix=".pickle"
        )
        os.close(result_descriptor)
        self._mesh_input_path = Path(input_path)
        self._mesh_result_path = Path(result_path)
        self._mesh_preview_chunk_paths = []
        self._mesh_stdout_buffer = ""
        self._mesh_error_message = None
        with self._mesh_input_path.open("wb") as stream:
            pickle.dump(
                (snapshot.version, snapshot.fluid_domain, config),
                stream,
                protocol=pickle.HIGHEST_PROTOCOL,
            )
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(
            [
                "-m",
                "app.mesher_process",
                str(self._mesh_input_path),
                str(output),
                str(self._mesh_result_path),
            ]
        )
        process.readyReadStandardOutput.connect(self._on_mesh_process_stdout)
        process.readyReadStandardError.connect(self._on_mesh_process_stderr)
        process.errorOccurred.connect(self._on_mesh_process_error)
        process.finished.connect(self._on_mesh_process_finished)
        self._thread = process
        process.start()

    @Slot()
    def _on_mesh_process_stdout(self) -> None:
        process = self._thread
        if process is None:
            return
        data = bytes(process.readAllStandardOutput()).decode(
            "utf-8",
            errors="replace",
        )
        self._mesh_stdout_buffer += data
        while "\n" in self._mesh_stdout_buffer:
            line, self._mesh_stdout_buffer = self._mesh_stdout_buffer.split(
                "\n",
                1,
            )
            self._handle_mesh_process_message(line)

    @Slot()
    def _on_mesh_process_stderr(self) -> None:
        process = self._thread
        if process is None:
            return
        message = bytes(process.readAllStandardError()).decode(
            "utf-8",
            errors="replace",
        ).strip()
        if message:
            logger.warning("mesher process stderr: %s", message)

    def _handle_mesh_process_message(self, line: str) -> None:
        if not line.strip():
            return
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("invalid mesher process message: %s", line)
            return
        version = int(message.get("version", -1))
        if version != self._meshing_version or version != self.document.version:
            if message.get("type") == "preview_chunk":
                path_text = str(message.get("path", ""))
                if path_text:
                    Path(path_text).unlink(missing_ok=True)
            return
        message_type = message.get("type")
        if message_type == "progress":
            signals.mesh_progress.emit(int(message.get("value", 0)))
        elif message_type == "preview_chunk":
            self._load_mesh_preview_chunk(Path(str(message.get("path", ""))))
        elif message_type == "failed":
            self._mesh_error_message = str(message.get("message", "meshing failed"))

    def _load_mesh_preview_chunk(self, path: Path) -> None:
        if not path.exists():
            return
        self._mesh_preview_chunk_paths.append(path)
        try:
            with path.open("rb") as stream:
                chunk = pickle.load(stream)
        except (OSError, pickle.PickleError, EOFError) as error:
            logger.warning("could not read mesh preview chunk %s: %s", path, error)
            return
        try:
            path.unlink(missing_ok=True)
        except OSError:
            logger.warning("could not remove mesh preview chunk %s", path)
        self.viewport.append_lattice_preview_chunk(chunk)

    @Slot(QProcess.ProcessError)
    def _on_mesh_process_error(self, error: QProcess.ProcessError) -> None:
        self._mesh_error_message = f"mesher process error: {error.name}"

    @Slot(int, QProcess.ExitStatus)
    def _on_mesh_process_finished(
        self,
        exit_code: int,
        exit_status: QProcess.ExitStatus,
    ) -> None:
        if self._mesh_stdout_buffer.strip():
            remaining = self._mesh_stdout_buffer
            self._mesh_stdout_buffer = ""
            self._handle_mesh_process_message(remaining)
        version = self._meshing_version
        result_path = self._mesh_result_path
        if version is None:
            self._clear_worker()
            return
        if (
            exit_status != QProcess.ExitStatus.NormalExit
            or exit_code != 0
            or result_path is None
            or not result_path.exists()
            or result_path.stat().st_size == 0
        ):
            self._on_mesh_failed(
                version,
                self._mesh_error_message
                or f"mesher process exited with code {exit_code}",
            )
            self._cleanup_mesh_process_files()
            self._clear_worker()
            return
        try:
            with result_path.open("rb") as stream:
                result = pickle.load(stream)
        except (OSError, pickle.PickleError, EOFError) as error:
            self._on_mesh_failed(version, f"could not read mesh result: {error}")
            self._cleanup_mesh_process_files()
            self._clear_worker()
            return
        self._on_mesh_completed(version, result)
        self._cleanup_mesh_process_files()
        self._clear_worker()

    @Slot(int, object)
    def _on_mesh_completed(self, version: int, result: object) -> None:
        if version != self.document.version:
            self._remove_temporary_mesh()
            self.mesher_panel.set_busy(False)
            self.export_panel.set_actions_enabled(True)
            signals.log_message.emit(
                "warning",
                "Discarded stale mesh result because the scene changed.",
            )
            return
        if self._meshing_mode == "preview":
            signals.preview_ready.emit(result)
            self._remove_temporary_mesh()
        else:
            signals.mesh_ready.emit(result)
        self.mesher_panel.set_busy(False)
        self.export_panel.set_actions_enabled(True)

    @Slot(int, str)
    def _on_mesh_failed(self, version: int, message: str) -> None:
        if version != self.document.version:
            self._remove_temporary_mesh()
            self.mesher_panel.set_busy(False)
            self.export_panel.set_actions_enabled(True)
            signals.log_message.emit(
                "warning",
                "Ignored stale mesh failure because the scene changed.",
            )
            return
        self._remove_temporary_mesh()
        self.mesher_panel.set_busy(False)
        self.export_panel.set_actions_enabled(True)
        action = "Preview" if self._meshing_mode == "preview" else "Export"
        signals.log_message.emit("error", f"{action} failed: {message}")
        QMessageBox.critical(self, f"{action} failed", message)

    @Slot()
    def _clear_worker(self) -> None:
        self._thread = None
        self._meshing_mode = None
        self._meshing_version = None
        self._mesh_stdout_buffer = ""
        self._mesh_error_message = None

    def _cleanup_mesh_process_files(self) -> None:
        for path in (
            self._mesh_input_path,
            self._mesh_result_path,
            *self._mesh_preview_chunk_paths,
        ):
            if path is None:
                continue
            try:
                path.unlink(missing_ok=True)
            except OSError:
                logger.warning("could not remove mesh worker file %s", path)
        self._mesh_input_path = None
        self._mesh_result_path = None
        self._mesh_preview_chunk_paths = []

    def _remove_temporary_mesh(self) -> None:
        if self._temporary_mesh_path is None:
            return
        try:
            self._temporary_mesh_path.unlink(missing_ok=True)
        except OSError:
            logger.warning(
                "could not remove preview lattice %s", self._temporary_mesh_path
            )
        self._temporary_mesh_path = None

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
        self.document = SceneDocument.default()
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document(frame=True)

    @Slot()
    def _new_empty_scene(self) -> None:
        undo_snapshot = self._history_snapshot()
        self.document = SceneDocument()
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document()

    def _load_example(self, factory: object) -> None:
        undo_snapshot = self._history_snapshot()
        self.document = factory()
        self._record_undo_snapshot(undo_snapshot)
        self._publish_document(frame=True)

    @Slot()
    def _open_scene(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open casoCAD scene", "", "casoCAD Scene (*.casocad.json *.json)"
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
            "scene.casocad.json",
            "casoCAD Scene (*.casocad.json)",
        )
        if not path:
            return
        if not path.lower().endswith(".json"):
            path = f"{path}.casocad.json"
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
        if (
            self._thread is not None
            and self._thread.state() != QProcess.ProcessState.NotRunning
        ):
            QMessageBox.information(
                self,
                "Meshing in progress",
                "Wait for the current Arrow export to finish before closing.",
            )
            event.ignore()
            return
        self.artifacts.shutdown()
        super().closeEvent(event)
