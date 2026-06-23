from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QItemSelectionModel, Qt
from PySide6.QtGui import QAction, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QMenu,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.signals import signals
from core.boundary import BoundaryRegion
from core.scene import SceneDocument
from core.sdf import PlacedPolyline2D, PlacedSDF1D, PlacedSDF2D, PolylineProfile
from core.sdf.base import SDFNode
HANDLE_ROLE = Qt.ItemDataRole.UserRole
SDF_ICON_DIR = Path(__file__).resolve().parents[2] / "assets" / "icons"
SDF_MENU_ITEMS: tuple[tuple[str, str], ...] = (
    ("Segment 1D", "segment"),
    ("Polyline 1D", "polyline"),
    ("Bezier Curve 1D", "bezier_curve"),
    ("Bezier Polycurve 1D", "bezier_polycurve"),
    ("Circle 2D", "circle"),
    ("Rectangle 2D", "rectangle"),
    ("Square 2D", "square"),
    ("Rounded Rectangle 2D", "rounded_rectangle"),
    ("Ellipse 2D", "ellipse"),
    ("Regular Polygon 2D", "regular_polygon"),
    ("Polygon 2D", "polygon"),
    ("Bezier Surface 2D", "bezier_surface"),
    ("Sphere", "sphere"),
    ("Box", "box"),
    ("Box Frame", "box_frame"),
    ("Cylinder", "cylinder"),
    ("Capped Cone", "capped_cone"),
    ("Cone", "cone"),
    ("Pyramid", "pyramid"),
    ("Torus", "torus"),
    ("Polyline Tube", "polyline_tube"),
    ("Bezier Tube", "bezier_tube"),
)
SDF_MENU_SECTIONS: tuple[tuple[str, tuple[tuple[str, str], ...]], ...] = (
    ("1D", SDF_MENU_ITEMS[:4]),
    ("2D", SDF_MENU_ITEMS[4:12]),
    ("3D", SDF_MENU_ITEMS[12:]),
)


def sdf_icon_path(kind: str) -> Path:
    return SDF_ICON_DIR / f"sdf_{kind}.svg"


def sdf_icon(kind: str) -> QIcon:
    return QIcon(str(sdf_icon_path(kind)))


def add_sdf_menu_actions(menu: QMenu, signal: object) -> None:
    for index, (section, items) in enumerate(SDF_MENU_SECTIONS):
        if index > 0:
            menu.addSeparator()
        header = menu.addSection(section)
        header.setEnabled(False)
        for label, kind in items:
            action = QAction(sdf_icon(kind), label, menu)
            action.setIconVisibleInMenu(True)
            action.triggered.connect(
                lambda checked=False, value=kind: signal.emit(value)
            )
            menu.addAction(action)


class SceneTreePanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._document: SceneDocument | None = None
        self._context_submenus: list[QMenu] = []
        self._items_by_handle: dict[int, QTreeWidgetItem] = {}
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        add_button = QPushButton("Add SDF")
        add_menu = QMenu(add_button)
        add_sdf_menu_actions(add_menu, signals.add_primitive_requested)
        add_button.setMenu(add_menu)
        layout.addWidget(add_button)
        draw_button = QPushButton("Draw on Grid")
        draw_menu = QMenu(draw_button)
        add_sdf_menu_actions(draw_menu, signals.viewport_create_requested)
        draw_button.setMenu(draw_menu)
        layout.addWidget(draw_button)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Scene", "Kind", "Dim", "Role"])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.tree)
        signals.document_changed.connect(self.set_document)

    def set_document(self, document: SceneDocument) -> None:
        self._document = document
        selected = set(self.selected_handles())
        expanded = {
            handle
            for handle, item in self._items_by_handle.items()
            if item.isExpanded()
        }
        self.tree.blockSignals(True)
        self.tree.clear()
        items: dict[int, QTreeWidgetItem] = {}
        for handle, node, parent_handle in document.walk():
            if isinstance(node, SDFNode) and document.is_internal_scene_node(node):
                continue
            role = ""
            if document.fluid_domain is not None:
                if node is document.fluid_domain.root:
                    role = "Fluid root"
                elif node in document.fluid_domain.tag_objects:
                    role = (
                        "Boundary tag"
                        if isinstance(
                            node,
                            (PlacedSDF1D, PlacedPolyline2D, BoundaryRegion),
                        )
                        else "Section tag"
                    )
            item = QTreeWidgetItem(
                [node.name, node.kind, f"{node.dimension}D", role]
            )
            item.setData(0, HANDLE_ROLE, handle)
            if parent_handle is None:
                self.tree.addTopLevelItem(item)
            else:
                items[parent_handle].addChild(item)
            items[handle] = item
            if handle in selected:
                item.setSelected(True)
        self._items_by_handle = items
        if expanded:
            for handle in expanded:
                item = self._items_by_handle.get(handle)
                if item is not None:
                    item.setExpanded(True)
        elif len(items) <= 64:
            self.tree.expandAll()
        else:
            for index in range(self.tree.topLevelItemCount()):
                self.tree.topLevelItem(index).setExpanded(True)
        self.tree.blockSignals(False)
        if len(items) <= 128:
            self.tree.resizeColumnToContents(0)

    def selected_handles(self) -> list[int]:
        return [
            int(item.data(0, HANDLE_ROLE))
            for item in self.tree.selectedItems()
        ]

    def select_handle(self, handle: int) -> None:
        self.select_handles([handle])

    def select_handles(self, handles: list[int]) -> None:
        targets = set(handles)
        first_item: QTreeWidgetItem | None = None
        self.tree.blockSignals(True)
        self.tree.clearSelection()
        for handle in handles:
            item = self._items_by_handle.get(handle)
            if item is not None and handle in targets:
                if first_item is None:
                    first_item = item
                item.setSelected(True)
        if first_item is not None:
            self.tree.selectionModel().setCurrentIndex(
                self.tree.indexFromItem(first_item),
                QItemSelectionModel.SelectionFlag.NoUpdate,
            )
        self.tree.blockSignals(False)
        self._on_selection_changed()

    def _on_selection_changed(self) -> None:
        handles = self.selected_handles()
        signals.selection_changed.emit(handles)
        signals.node_selected.emit(handles[0] if len(handles) == 1 else None)

    def _show_context_menu(self, position: object) -> None:
        self._context_submenus.clear()
        clicked_item = self.tree.itemAt(position)
        if clicked_item is not None and not clicked_item.isSelected():
            self.tree.clearSelection()
            self.tree.setCurrentItem(clicked_item)
            clicked_item.setSelected(True)
        menu = QMenu(self)
        menu.aboutToHide.connect(
            lambda: signals.sdf_op_preview_requested.emit("", [])
        )
        add_menu = menu.addMenu("Add SDF")
        add_sdf_menu_actions(add_menu, signals.add_primitive_requested)
        selected = self.selected_handles()
        boolean_menu = menu.addMenu("Boolean")
        selected_node = (
            self._document.node(selected[0])
            if len(selected) == 1 and self._document is not None
            else None
        )
        if isinstance(selected_node, SDFNode):
            self._populate_object_boolean_menu(boolean_menu, selected[0])
        else:
            for label, operation in (
                ("Union selected pair", "union"),
                ("Intersect selected pair", "intersection"),
                ("Difference: first - second", "difference"),
            ):
                action = boolean_menu.addAction(label)
                action.setEnabled(len(selected) == 2)
                action.hovered.connect(
                    lambda value=operation: signals.sdf_op_preview_requested.emit(
                        value, self.selected_handles()
                    )
                )
                action.triggered.connect(
                    lambda checked=False, value=operation: signals.sdf_op_requested.emit(
                        value, self.selected_handles()
                    )
                )
        transform_menu = menu.addMenu("Transform")
        for label, transform in (
            ("Translate", "translate"),
            ("Rotate", "rotate"),
            ("Scale", "scale"),
        ):
            action = transform_menu.addAction(label)
            action.setEnabled(isinstance(selected_node, SDFNode))
            action.triggered.connect(
                lambda checked=False, value=transform: signals.transform_requested.emit(
                    value, self.selected_handles()
                )
            )
        solid_menu = menu.addMenu("Solid From 2D")
        for label, method in (
            ("Extrude", "extrude"),
            ("Revolve", "revolve"),
        ):
            action = solid_menu.addAction(label)
            action.setEnabled(
                bool(selected)
                and self._document is not None
                and all(
                    isinstance(self._document.node(handle), SDFNode)
                    and self._document.node(handle).dimension == 2
                    for handle in selected
                )
            )
            action.triggered.connect(
                lambda checked=False, value=method: signals.solid_from_2d_requested.emit(
                    value, self.selected_handles()
                )
            )
        set_root = menu.addAction("Set as Fluid Domain")
        set_root.setEnabled(
            isinstance(selected_node, SDFNode)
            and selected_node.dimension in {2, 3}
        )
        set_root.triggered.connect(
            lambda: signals.set_fluid_root_requested.emit(self.selected_handles())
        )
        enable_tag = menu.addAction("Enable Lattice Tag")
        fluid_root = (
            self._document.fluid_domain.root
            if self._document is not None
            and self._document.fluid_domain is not None
            else None
        )
        enable_tag.setEnabled(
            fluid_root is not None
            and (
                (
                    isinstance(selected_node, BoundaryRegion)
                    and fluid_root.dimension in {2, 3}
                )
                or (
                    isinstance(selected_node, SDFNode)
                    and selected_node.dimension == fluid_root.dimension - 1
                )
            )
        )
        enable_tag.triggered.connect(
            lambda: signals.set_tag_enabled_requested.emit(
                self.selected_handles(), True
            )
        )
        disable_tag = menu.addAction("Disable Lattice Tag")
        disable_tag.setEnabled(enable_tag.isEnabled())
        disable_tag.triggered.connect(
            lambda: signals.set_tag_enabled_requested.emit(
                self.selected_handles(), False
            )
        )
        create_boundary = menu.addAction("Create Boundary Region")
        create_boundary.setEnabled(
            self._can_create_boundary_region(selected)
        )
        create_boundary.triggered.connect(
            lambda: signals.create_boundary_region_requested.emit(
                self.selected_handles()
            )
        )
        polygon_from_polyline = menu.addAction("Create Polygon from Polyline")
        polygon_from_polyline.setEnabled(
            len(selected) == 1
            and isinstance(selected_node, PlacedPolyline2D)
            and isinstance(selected_node.profile, PolylineProfile)
        )
        polygon_from_polyline.triggered.connect(
            lambda: signals.create_polygon_from_polyline_requested.emit(
                self.selected_handles()
            )
        )
        delete_action = QAction("Delete", menu)
        delete_action.setEnabled(bool(selected))
        delete_action.triggered.connect(
            lambda: signals.delete_nodes_requested.emit(self.selected_handles())
        )
        menu.addAction(delete_action)
        menu.exec(self.tree.viewport().mapToGlobal(position))
        self._context_submenus.clear()

    def _can_create_boundary_region(self, handles: list[int]) -> bool:
        if self._document is None:
            return False
        if len(handles) == 1:
            node = self._document.node(handles[0])
            return isinstance(node, SDFNode) and node.dimension == 3
        if len(handles) != 2:
            return False
        first = self._document.node(handles[0])
        second = self._document.node(handles[1])
        if isinstance(first, BoundaryRegion):
            region = first
            selector = second
        elif isinstance(second, BoundaryRegion):
            region = second
            selector = first
        else:
            return False
        if region.patch_id is None or fluid_root is None:
            return False
        if fluid_root.dimension == 2:
            return isinstance(selector, (PlacedSDF1D, PlacedPolyline2D))
        return isinstance(selector, SDFNode) and (
            selector.dimension == 3
            or isinstance(selector, (PlacedSDF1D, PlacedPolyline2D, PlacedSDF2D))
        )

    def _populate_object_boolean_menu(
        self,
        menu: QMenu,
        base_handle: int,
    ) -> None:
        assert self._document is not None
        base = self._document.node(base_handle)
        candidates = [
            (handle, node)
            for handle, node, _parent in self._document.walk()
            if handle != base_handle
            and isinstance(node, SDFNode)
            and node.dimension == base.dimension
            and self._document.can_combine(base_handle, handle)
        ]
        if not candidates:
            disabled = menu.addAction("No compatible SDF objects")
            disabled.setEnabled(False)
            return
        for label, operation in (
            ("Union with", "union"),
            ("Intersect with", "intersection"),
        ):
            operation_menu = menu.addMenu(label)
            self._context_submenus.append(operation_menu)
            for other_handle, other in candidates:
                action = operation_menu.addAction(
                    f"{other.name}  [ID {other.object_id}]"
                )
                action.hovered.connect(
                    lambda op=operation,
                    first=base_handle,
                    second=other_handle: signals.sdf_op_preview_requested.emit(
                        op, [first, second]
                    )
                )
                action.triggered.connect(
                    lambda checked=False,
                    op=operation,
                    first=base_handle,
                    second=other_handle: signals.sdf_op_requested.emit(
                        op, [first, second]
                    )
                )
        subtract_menu = menu.addMenu("Subtract from this")
        reverse_menu = menu.addMenu("Subtract this from")
        self._context_submenus.extend((subtract_menu, reverse_menu))
        for other_handle, other in candidates:
            label = f"{other.name}  [ID {other.object_id}]"
            subtract = subtract_menu.addAction(label)
            subtract.hovered.connect(
                lambda first=base_handle,
                second=other_handle: signals.sdf_op_preview_requested.emit(
                    "difference", [first, second]
                )
            )
            subtract.triggered.connect(
                lambda checked=False,
                first=base_handle,
                second=other_handle: signals.sdf_op_requested.emit(
                    "difference", [first, second]
                )
            )
            reverse = reverse_menu.addAction(label)
            reverse.hovered.connect(
                lambda first=other_handle,
                second=base_handle: signals.sdf_op_preview_requested.emit(
                    "difference", [first, second]
                )
            )
            reverse.triggered.connect(
                lambda checked=False,
                first=other_handle,
                second=base_handle: signals.sdf_op_requested.emit(
                    "difference", [first, second]
                )
            )
