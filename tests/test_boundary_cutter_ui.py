from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication, QPushButton

from app.main_window import MainWindow
from app.viewport.viewport_widget import ViewportWidget
from core.boundary import BoundaryRegion
from core.boundary_patches import pick_boundary_patch
from core.scene import SceneDocument


def _application() -> QApplication:
    application = QApplication.instance()
    if application is None:
        application = QApplication([])
    return application


def test_boundary_cutter_buttons_require_selected_boundary_region() -> None:
    _application()
    viewport = ViewportWidget()
    try:
        planar = viewport.findChild(QPushButton, "viewportPlanarBoundaryCutterButton")
        surface = viewport.findChild(QPushButton, "viewportSurfaceBoundaryCutterButton")
        assert planar is not None
        assert surface is not None
        assert not planar.isEnabled()
        assert not surface.isEnabled()

        region = BoundaryRegion(
            name="inlet",
            object_id=10,
            owner_object_id=1,
            outside_direction=1,
            patch_id="+X",
            patch_type="face",
        )
        viewport.set_boundary_region_selection_entries(
            ((region.owner_object_id, 0),),
            ((1.0, 0.0, 0.0),),
            (region,),
        )

        assert planar.isEnabled()
        assert surface.isEnabled()
    finally:
        viewport.deleteLater()


def test_surface_cutter_button_exposes_shape_menu() -> None:
    _application()
    viewport = ViewportWidget()
    try:
        surface = viewport.findChild(QPushButton, "viewportSurfaceBoundaryCutterButton")
        assert surface is not None
        menu = surface.menu()
        assert menu is not None

        labels = {action.text() for action in menu.actions()}

        assert labels == {
            "Sphere",
            "Box",
            "Cylinder",
            "Cone",
            "Capped Cone",
            "Torus",
        }
    finally:
        viewport.deleteLater()


def test_planar_cutter_button_exposes_curve_menu() -> None:
    _application()
    viewport = ViewportWidget()
    try:
        planar = viewport.findChild(QPushButton, "viewportPlanarBoundaryCutterButton")
        assert planar is not None
        menu = planar.menu()
        assert menu is not None

        labels = {action.text() for action in menu.actions()}

        assert labels == {"Segment", "Polyline", "Bezier Polycurve"}
    finally:
        viewport.deleteLater()


def test_planar_cutter_tool_starts_selected_curve_cutter() -> None:
    _application()
    viewport = ViewportWidget()
    try:
        region = BoundaryRegion(
            name="inlet",
            object_id=10,
            owner_object_id=1,
            outside_direction=1,
            patch_id="+X",
            patch_type="face",
        )
        viewport.set_boundary_region_selection_entries(
            ((region.owner_object_id, 0),),
            ((1.0, 0.0, 0.0),),
            (region,),
        )

        viewport.begin_boundary_cutter_tool("planar", "polyline")

        assert viewport._interaction_tool == ("create", "polyline")
        assert viewport.active_boundary_cutter_tool() == ("planar", "polyline")
    finally:
        viewport.deleteLater()


def test_planar_polyline_cutter_draw_flow_creates_two_scene_tree_regions() -> None:
    application = _application()
    window, base_handle = _window_with_selected_box_boundary()
    try:
        window.viewport.begin_boundary_cutter_tool("planar", "polyline")
        window._on_viewport_point_shape_drawn(
            "polyline",
            (
                (0.5, -0.25, -0.25),
                (0.5, 0.25, -0.25),
                (0.5, 0.25, 0.25),
            ),
            "yz",
        )
        application.processEvents()

        selected = set(window.scene_tree.selected_handles())
        split_regions = _selector_regions_for_base(window.document, base_handle)
        split_handles = {window.document.handle_for(region) for region in split_regions}

        assert len(split_regions) == 2
        assert {region.selector_side for region in split_regions} == {
            "inside",
            "outside",
        }
        assert {region.selector_type for region in split_regions} == {
            "surface_split_profile"
        }
        assert selected == split_handles
    finally:
        window.close()
        window.deleteLater()


def test_planar_segment_cutter_draw_flow_creates_two_scene_tree_regions() -> None:
    application = _application()
    window, base_handle = _window_with_selected_box_boundary()
    try:
        window.viewport.begin_boundary_cutter_tool("planar", "segment")
        window._on_viewport_shape_drawn(
            "segment",
            (0.5, -0.25, 0.0),
            (0.5, 0.25, 0.0),
            None,
        )
        application.processEvents()

        selected = set(window.scene_tree.selected_handles())
        split_regions = _selector_regions_for_base(window.document, base_handle)
        split_handles = {window.document.handle_for(region) for region in split_regions}

        assert len(split_regions) == 2
        assert {region.selector_side for region in split_regions} == {
            "inside",
            "outside",
        }
        assert {region.selector_type for region in split_regions} == {
            "surface_split_profile"
        }
        assert selected == split_handles
        tree_names = {
            window.scene_tree.tree.topLevelItem(index).text(0)
            for index in range(window.scene_tree.tree.topLevelItemCount())
        }
        assert "__boundary_selector_planar_segment_cutter" not in tree_names
        assert any(
            node.name == "__boundary_selector_planar_segment_cutter"
            for node in window.document.objects
        )
        tree = window.document.visual_tree()
        assert not any(
            component.name == "__boundary_selector_planar_segment_cutter"
            for component in tree.components
        )
        assert any(
            selector.name == "__boundary_selector_planar_segment_cutter"
            for selector in tree.selector_objects
        )
    finally:
        window.close()
        window.deleteLater()


def test_planar_bezier_polycurve_cutter_draw_flow_creates_two_scene_tree_regions() -> None:
    application = _application()
    window, base_handle = _window_with_selected_box_boundary()
    try:
        window.viewport.begin_boundary_cutter_tool("planar", "bezier_polycurve")
        window._on_viewport_point_shape_drawn(
            "bezier_polycurve",
            (
                (0.5, -0.25, -0.25),
                (0.5, 0.0, 0.30),
                (0.5, 0.25, -0.25),
            ),
            "yz",
        )
        application.processEvents()

        selected = set(window.scene_tree.selected_handles())
        split_regions = _selector_regions_for_base(window.document, base_handle)
        split_handles = {window.document.handle_for(region) for region in split_regions}

        assert len(split_regions) == 2
        assert {region.selector_side for region in split_regions} == {
            "inside",
            "outside",
        }
        assert {region.selector_type for region in split_regions} == {
            "surface_split_profile"
        }
        assert selected == split_handles
    finally:
        window.close()
        window.deleteLater()


def test_surface_cutter_draw_flow_creates_two_scene_tree_regions() -> None:
    application = _application()
    window, base_handle = _window_with_selected_box_boundary()
    try:
        window._on_viewport_shape_drawn(
            "sphere",
            (0.5, -0.25, -0.25),
            (0.5, 0.25, 0.25),
            None,
        )
        application.processEvents()

        selected = set(window.scene_tree.selected_handles())
        split_regions = _selector_regions_for_base(window.document, base_handle)
        split_handles = {window.document.handle_for(region) for region in split_regions}

        assert len(split_regions) == 2
        assert {region.selector_side for region in split_regions} == {
            "inside",
            "outside",
        }
        assert {region.selector_type for region in split_regions} == {
            "surface_sdf_subregion"
        }
        assert selected == split_handles
    finally:
        window.close()
        window.deleteLater()


def _window_with_selected_box_boundary() -> tuple[MainWindow, int]:
    window = MainWindow()
    window._render_request_timer.stop()
    document = SceneDocument()
    root_handle = document.add_primitive("box")
    root = document.node(root_handle)
    document.set_fluid_root(root_handle)
    hit = pick_boundary_patch(
        root,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    )
    assert hit is not None
    base_handle = document.add_boundary_region_from_hit(hit)
    window.document = document
    window._publish_document(render=False)
    window.scene_tree.select_handle(base_handle)
    _application().processEvents()
    return window, base_handle


def _selector_regions_for_base(
    document: SceneDocument,
    base_handle: int,
) -> list[BoundaryRegion]:
    base = document.node(base_handle)
    assert isinstance(base, BoundaryRegion)
    return [
        region
        for region in document.boundary_regions
        if region is not base
        and region.owner_object_id == base.owner_object_id
        and region.patch_id == base.patch_id
        and region.selector_id is not None
    ]
