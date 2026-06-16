from __future__ import annotations

import logging
import os
from pathlib import Path
import tempfile

import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QDockWidget,
    QFileDialog,
    QMainWindow,
    QMessageBox,
    QLabel,
    QSlider,
    QToolBar,
)

from app.panels.export_panel import ExportPanel
from app.panels.log_panel import LogPanel
from app.panels.mesher_panel import MesherPanel
from app.panels.properties import PropertiesPanel
from app.panels.scene_tree import SceneTreePanel
from app.signals import signals
from app.viewport.gl_widget import GLWidget
from core.boundary import BoundaryRegion
from core.mesher import FluidDomain, LatticeMesher, MesherConfig
from core.sdf import PlacedSDF2D
from core.sdf.base import SDFNode
from core.serialization import load_scene, save_scene
from core.scene import SceneDocument
from scenes.boolean_operations import build_scene as build_boolean_scene
from scenes.lattice_benchmark import build_scene as build_benchmark_scene
from scenes.pipe_3d import build_scene as build_pipe_scene
from scenes.placed_section_tags import build_scene as build_tagging_scene

logger = logging.getLogger(__name__)


class MesherWorker(QObject):
    progress = Signal(int)
    completed = Signal(object)
    failed = Signal(str)

    def __init__(
        self, domain: FluidDomain, config: MesherConfig, path: str
    ) -> None:
        super().__init__()
        self.domain = domain
        self.config = config
        self.path = path

    @Slot()
    def run(self) -> None:
        try:
            result = LatticeMesher(self.domain, self.config).mesh(
                self.path,
                lambda done, total: self.progress.emit(
                    round(100.0 * done / total)
                ),
            )
        except Exception as error:
            logger.exception("lattice export failed")
            self.failed.emit(str(error))
            return
        self.completed.emit(result)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("casoCAD - Programmable SDF CAD")
        self.resize(1400, 860)
        self.document = SceneDocument.default()
        self.viewport = GLWidget()
        self.setCentralWidget(self.viewport)
        self.scene_tree = SceneTreePanel()
        self.properties = PropertiesPanel()
        self.mesher_panel = MesherPanel()
        self.export_panel = ExportPanel()
        self.log_panel = LogPanel()
        self._thread: QThread | None = None
        self._worker: MesherWorker | None = None
        self._meshing_mode: str | None = None
        self._temporary_mesh_path: Path | None = None
        self._build_docks()
        self._build_menu()
        self._build_toolbar()
        self._connect_signals()
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
        move_action = toolbar.addAction("Move")
        move_action.setToolTip(
            "Select one Scene object, then drag it on the Z=0 XY reference grid"
        )
        move_action.triggered.connect(self._start_viewport_move)
        boundary_action = toolbar.addAction("Boundary Region")
        boundary_action.setToolTip(
            "Click the FluidDomain boundary to create a dimension-aware tag"
        )
        boundary_action.triggered.connect(self._start_boundary_region_tool)
        toolbar.addSeparator()
        fit_action = toolbar.addAction("Frame Scene")
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

    def _connect_signals(self) -> None:
        signals.add_primitive_requested.connect(self._on_add_primitive)
        signals.viewport_shape_drawn.connect(self._on_viewport_shape_drawn)
        signals.viewport_move_requested.connect(self._on_viewport_move_requested)
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
        signals.node_edited.connect(self._on_node_edited)
        signals.selection_changed.connect(self._on_selection_changed)
        signals.export_requested.connect(self._on_export_requested)
        signals.mesh_requested.connect(self._on_mesh_requested)
        signals.log_message.connect(self._on_log_message)

    def _publish_document(
        self, frame: bool = False, clear_selection: bool = True
    ) -> None:
        signals.document_changed.emit(self.document)
        if clear_selection:
            signals.node_selected.emit(None)
        if not self.document.objects:
            signals.scene_changed.emit(None)
            self._sdf_action.setChecked(True)
            self.viewport.set_mode("sdf")
            self.viewport.configure_default_grid()
            self.viewport.frame_default_grid()
            return
        try:
            tree = self.document.visual_tree()
        except ValueError:
            signals.scene_changed.emit(None)
            return
        signals.scene_changed.emit(tree)
        if self.document.fluid_domain is not None:
            box = self.document.fluid_domain.bounding_box()
            self.viewport.configure_grid(box, self.mesher_panel.config().dx)
            if frame:
                self.viewport.frame_box(box)

    @Slot(str)
    def _on_add_primitive(self, kind: str) -> None:
        handle = self.document.add_primitive(kind)
        self._publish_document()
        self.scene_tree.select_handle(handle)

    @Slot(str, object, object)
    def _on_viewport_shape_drawn(
        self,
        kind: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> None:
        handle = self.document.add_primitive_from_drag(kind, start, end)
        self._publish_document()
        self.scene_tree.select_handle(handle)
        self.statusBar().showMessage(
            f"Created {self.document.node(handle).name} from viewport drag",
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

    @Slot(int, object)
    def _on_viewport_move_requested(
        self,
        handle: int,
        delta: tuple[float, float, float],
    ) -> None:
        try:
            moved_handle = self.document.move_object(handle, delta)
            self.document.refresh_derived_geometry()
        except (KeyError, ValueError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self._publish_document(clear_selection=False)
        self.scene_tree.select_handle(moved_handle)
        self.statusBar().showMessage(
            "Object moved on the Z=0 XY reference grid", 5000
        )

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
        self.viewport.set_boundary_hover(owner_object_id, outside_direction)
        owner = next(
            (
                node
                for _handle, node, _parent in self.document.walk()
                if node.object_id == owner_object_id
            ),
            None,
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
        try:
            handle = self.document.add_boundary_region(
                owner_object_id,
                outside_direction,
            )
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
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
        if owner is None or owner.kind not in {"box", "cylinder"}:
            return None
        normal_array = np.asarray(normal, dtype=np.float64)
        axis = int(np.argmax(np.abs(normal_array)))
        if abs(normal_array[axis]) < 0.999:
            return None
        if owner.kind == "cylinder" and axis != 2:
            return None
        return 2 * axis + int(normal_array[axis] > 0.0)

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
        for node in roots:
            try:
                self.document.delete(self.document.handle_for(node))
            except KeyError:
                continue
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
        try:
            handle = self.document.combine(handles[0], handles[1], operation)
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
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
        self.viewport.set_boundary_region_selection(boundary_regions)
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
        try:
            handle = self.document.wrap_transform(handles[0], transform)
        except ValueError as error:
            signals.log_message.emit("warning", str(error))
            return
        self._publish_document()
        self.scene_tree.select_handle(handle)

    @Slot(str, object)
    def _on_solid_from_2d(self, method: str, handles: list[int]) -> None:
        try:
            handle = self.document.solid_from_2d(handles, method)
        except (ValueError, KeyError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self._publish_document()
        self.scene_tree.select_handle(handle)

    @Slot(object)
    def _on_set_fluid_root(self, handles: list[int]) -> None:
        if len(handles) != 1:
            return
        try:
            self.document.set_fluid_root(handles[0])
        except (ValueError, KeyError) as error:
            signals.log_message.emit("warning", str(error))
            return
        self._publish_document(clear_selection=False)

    @Slot(object, bool)
    def _on_set_tag_enabled(self, handles: list[int], enabled: bool) -> None:
        if len(handles) != 1:
            return
        try:
            self.document.set_tag_enabled(handles[0], enabled)
        except (ValueError, KeyError) as error:
            signals.log_message.emit("warning", str(error))
            return
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
        thread = QThread(self)
        worker = MesherWorker(snapshot.fluid_domain, config, str(output))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.progress.connect(signals.mesh_progress.emit)
        worker.completed.connect(self._on_mesh_completed)
        worker.failed.connect(self._on_mesh_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_worker)
        self._thread = thread
        self._worker = worker
        thread.start()

    @Slot(object)
    def _on_mesh_completed(self, result: object) -> None:
        if self._meshing_mode == "preview":
            signals.preview_ready.emit(result)
            self._remove_temporary_mesh()
        else:
            signals.mesh_ready.emit(result)
        self.mesher_panel.set_busy(False)
        self.export_panel.set_actions_enabled(True)

    @Slot(str)
    def _on_mesh_failed(self, message: str) -> None:
        self._remove_temporary_mesh()
        self.mesher_panel.set_busy(False)
        self.export_panel.set_actions_enabled(True)
        action = "Preview" if self._meshing_mode == "preview" else "Export"
        signals.log_message.emit("error", f"{action} failed: {message}")
        QMessageBox.critical(self, f"{action} failed", message)

    @Slot()
    def _clear_worker(self) -> None:
        self._thread = None
        self._worker = None
        self._meshing_mode = None

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
        if self.document.fluid_domain is not None:
            self.viewport.frame_box(self.document.fluid_domain.bounding_box())

    @Slot()
    def _new_default_scene(self) -> None:
        self.document = SceneDocument.default()
        self._publish_document(frame=True)

    @Slot()
    def _new_empty_scene(self) -> None:
        self.document = SceneDocument()
        self._publish_document()

    def _load_example(self, factory: object) -> None:
        self.document = factory()
        self._publish_document(frame=True)

    @Slot()
    def _open_scene(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open casoCAD scene", "", "casoCAD Scene (*.casocad.json *.json)"
        )
        if not path:
            return
        try:
            self.document = load_scene(path)
        except (OSError, ValueError, KeyError, TypeError) as error:
            QMessageBox.critical(self, "Open failed", str(error))
            return
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
        if self._thread is not None and self._thread.isRunning():
            QMessageBox.information(
                self,
                "Meshing in progress",
                "Wait for the current Arrow export to finish before closing.",
            )
            event.ignore()
            return
        super().closeEvent(event)
