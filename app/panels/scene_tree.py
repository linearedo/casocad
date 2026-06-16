from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
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
from core.sdf import PlacedSDF1D
from core.sdf.base import SDFNode
HANDLE_ROLE = Qt.ItemDataRole.UserRole


class SceneTreePanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._document: SceneDocument | None = None
        self._context_submenus: list[QMenu] = []
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        add_button = QPushButton("Add SDF")
        add_menu = QMenu(add_button)
        for label, kind in (
            ("Interval 1D", "interval"),
            ("Circle 2D", "circle"),
            ("Rectangle 2D", "rectangle"),
            ("Square 2D", "square"),
            ("Rounded Rectangle 2D", "rounded_rectangle"),
            ("Ellipse 2D", "ellipse"),
            ("Regular Polygon 2D", "regular_polygon"),
            ("Sphere", "sphere"),
            ("Box", "box"),
            ("Cylinder", "cylinder"),
            ("Torus", "torus"),
        ):
            action = add_menu.addAction(label)
            action.triggered.connect(
                lambda checked=False, value=kind: signals.add_primitive_requested.emit(
                    value
                )
            )
        add_button.setMenu(add_menu)
        layout.addWidget(add_button)
        draw_button = QPushButton("Draw on Grid")
        draw_menu = QMenu(draw_button)
        for label, kind in (
            ("Interval 1D", "interval"),
            ("Circle 2D", "circle"),
            ("Rectangle 2D", "rectangle"),
            ("Square 2D", "square"),
            ("Rounded Rectangle 2D", "rounded_rectangle"),
            ("Ellipse 2D", "ellipse"),
            ("Regular Polygon 2D", "regular_polygon"),
            ("Sphere", "sphere"),
            ("Box", "box"),
            ("Cylinder", "cylinder"),
            ("Torus", "torus"),
        ):
            action = draw_menu.addAction(label)
            action.triggered.connect(
                lambda checked=False, value=kind: signals.viewport_create_requested.emit(
                    value
                )
            )
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
        self.tree.blockSignals(True)
        self.tree.clear()
        items: dict[int, QTreeWidgetItem] = {}
        for handle, node, parent_handle in document.walk():
            role = ""
            if document.fluid_domain is not None:
                if node is document.fluid_domain.root:
                    role = "Fluid root"
                elif node in document.fluid_domain.tag_objects:
                    role = (
                        "Boundary tag"
                        if isinstance(node, (PlacedSDF1D, BoundaryRegion))
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
        self.tree.expandAll()
        self.tree.resizeColumnToContents(0)
        self.tree.blockSignals(False)

    def selected_handles(self) -> list[int]:
        return [
            int(item.data(0, HANDLE_ROLE))
            for item in self.tree.selectedItems()
        ]

    def select_handle(self, handle: int) -> None:
        iterator = self.tree.findItems(
            "*", Qt.MatchFlag.MatchWildcard | Qt.MatchFlag.MatchRecursive, 0
        )
        for item in iterator:
            if item.data(0, HANDLE_ROLE) == handle:
                self.tree.clearSelection()
                self.tree.setCurrentItem(item)
                item.setSelected(True)
                break

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
        add_menu = menu.addMenu("Add SDF")
        for label, kind in (
            ("Interval 1D", "interval"),
            ("Circle 2D", "circle"),
            ("Rectangle 2D", "rectangle"),
            ("Square 2D", "square"),
            ("Rounded Rectangle 2D", "rounded_rectangle"),
            ("Ellipse 2D", "ellipse"),
            ("Regular Polygon 2D", "regular_polygon"),
            ("Sphere", "sphere"),
            ("Box", "box"),
            ("Cylinder", "cylinder"),
            ("Torus", "torus"),
        ):
            action = add_menu.addAction(label)
            action.triggered.connect(
                lambda checked=False, value=kind: signals.add_primitive_requested.emit(
                    value
                )
            )
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
                ("Smooth Union selected pair", "smooth_union"),
            ):
                action = boolean_menu.addAction(label)
                action.setEnabled(len(selected) == 2)
                action.triggered.connect(
                    lambda checked=False, value=operation: signals.csg_requested.emit(
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
            ("Sweep", "sweep"),
            ("Loft Implicit", "loft_implicit"),
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
                    and fluid_root.dimension == 3
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
        create_boundary = menu.addAction("Create Boundary SDF from Owner")
        create_boundary.setEnabled(
            isinstance(selected_node, SDFNode)
            and selected_node.dimension == 3
        )
        create_boundary.triggered.connect(
            lambda: signals.create_boundary_region_requested.emit(
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
            ("Smooth Union with", "smooth_union"),
        ):
            operation_menu = menu.addMenu(label)
            self._context_submenus.append(operation_menu)
            for other_handle, other in candidates:
                action = operation_menu.addAction(
                    f"{other.name}  [ID {other.object_id}]"
                )
                action.triggered.connect(
                    lambda checked=False,
                    op=operation,
                    first=base_handle,
                    second=other_handle: signals.csg_requested.emit(
                        op, [first, second]
                    )
                )
        subtract_menu = menu.addMenu("Subtract from this")
        reverse_menu = menu.addMenu("Subtract this from")
        self._context_submenus.extend((subtract_menu, reverse_menu))
        for other_handle, other in candidates:
            label = f"{other.name}  [ID {other.object_id}]"
            subtract = subtract_menu.addAction(label)
            subtract.triggered.connect(
                lambda checked=False,
                first=base_handle,
                second=other_handle: signals.csg_requested.emit(
                    "difference", [first, second]
                )
            )
            reverse = reverse_menu.addAction(label)
            reverse.triggered.connect(
                lambda checked=False,
                first=other_handle,
                second=base_handle: signals.csg_requested.emit(
                    "difference", [first, second]
                )
            )
