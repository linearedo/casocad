"""Single Boundary Cutter flow (boundary_region_v2 §7 Phase 5): arm, draw any
shape, the selected region splits in two, and the knife never becomes a scene
object."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.signals import signals
from core.boundary import BoundaryRegion
from core.scene import SceneDocument
from core.sdf import Box


_SHARED_WINDOW = None


def _window():
    """One window per process: MainWindow instances all listen on the global
    signal bus, so a second live window would double-handle every emit."""
    global _SHARED_WINDOW
    if QApplication.instance() is None:
        QApplication([])
    if _SHARED_WINDOW is None:
        from app.main_window import MainWindow

        _SHARED_WINDOW = MainWindow()
    window = _SHARED_WINDOW
    window.document = SceneDocument.default()
    window.scene_tree.tree.clearSelection()
    window.viewport.cancel_create_tool()
    window._publish_document()
    return window


def _select_whole_surface_region(window):
    box = next(
        n for _h, n, _p in window.document.walk() if isinstance(n, Box)
    )
    handle = window.document.add_boundary_region(box.object_id)
    window._publish_document()
    window.scene_tree.select_handle(handle)
    return window.document.node(handle)


def test_cutter_splits_selected_region_without_scene_ghost() -> None:
    window = _window()
    region = _select_whole_surface_region(window)
    objects_before = list(window.document.objects)

    window.viewport._boundary_region_selected = True
    window.viewport.begin_boundary_cutter_tool("sphere")
    assert window.viewport.active_boundary_cutter_tool() == "sphere"

    # commit through the tool, exactly as a real drag release does
    window.viewport._create_tool.emit_drag_shape(
        (-1.9, -0.3, -0.3), (-1.3, 0.3, 0.3)
    )

    assert region not in window.document.boundary_regions
    children = [r for r in window.document.boundary_regions if r.cuts]
    assert len(children) == 2
    assert {c.cuts[-1].side for c in children} == {"inside", "outside"}
    assert window.document.objects == objects_before
    assert not any(
        window.document.is_internal_scene_node(node)
        for node in window.document.objects
    )
    # the cutter disarms after the commit
    assert window.viewport.active_boundary_cutter_tool() is None


def test_point_shape_cutter_splits_region() -> None:
    window = _window()
    region = _select_whole_surface_region(window)
    objects_before = list(window.document.objects)

    window.viewport._boundary_region_selected = True
    window.viewport.begin_boundary_cutter_tool("polyline")
    tool = window.viewport._create_tool
    tool.points.extend(
        ((-1.6, -0.5, -0.3), (-1.6, 0.5, -0.3), (-1.6, 0.0, 0.4))
    )
    tool.commit_point_shape()

    assert region not in window.document.boundary_regions
    children = [r for r in window.document.boundary_regions if r.cuts]
    assert len(children) == 2
    assert window.document.objects == objects_before


def test_normal_draw_still_creates_objects_when_cutter_idle() -> None:
    window = _window()
    _select_whole_surface_region(window)
    count_before = len(window.document.objects)
    regions_before = list(window.document.boundary_regions)

    signals.viewport_shape_drawn.emit(
        "sphere", (-0.4, -0.4, 0.0), (0.4, 0.4, 0.0), None
    )

    assert len(window.document.objects) == count_before + 1
    assert window.document.boundary_regions == regions_before


def test_boundary_tool_hover_highlights_and_click_tags() -> None:
    import numpy as np

    from core.boundary_patches import pick_boundary_patch

    window = _window()
    root = window.document.fluid_domain.root
    window._start_boundary_region_tool()
    tool = window.viewport._boundary_tool
    assert tool.active

    hit = pick_boundary_patch(
        root, np.array([-5.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])
    )
    assert hit is not None
    tool.hover = hit
    signals.viewport_boundary_hovered.emit(hit)
    assert window.viewport._boundary_hover_id == hit.owner_object_id

    regions_before = len(window.document.boundary_regions)
    assert tool.commit() is True
    assert not tool.active
    assert len(window.document.boundary_regions) == regions_before + 1
    assert window.viewport._boundary_hover_id == 0


def test_escape_cancels_boundary_tool() -> None:
    from PySide6.QtCore import QEvent, Qt
    from PySide6.QtGui import QKeyEvent

    window = _window()
    window._start_boundary_region_tool()
    assert window.viewport._boundary_tool.active

    window.viewport.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
        )
    )

    assert not window.viewport._boundary_tool.active


def _split_minus_x_face(window):
    import numpy as np
    from core.sdf import Sphere

    box = next(n for _h, n, _p in window.document.walk() if isinstance(n, Box))
    region = window.document.node(
        window.document.add_boundary_region(box.object_id, patch_id="-X")
    )
    knife = Sphere(name="k", object_id=0, center=(-1.6, 0.0, 0.0), radius=0.4)
    handles, _ = window.document.split_boundary_region(region, knife)
    window._publish_document()
    return handles


def test_hover_resolves_cutter_made_children_and_click_selects() -> None:
    import numpy as np
    from core.boundary_patches import pick_boundary_patch

    window = _window()
    inside_h, outside_h = _split_minus_x_face(window)
    root = window.document.fluid_domain.root
    window._start_boundary_region_tool()

    hit_disk = pick_boundary_patch(
        root, np.array([-5.0, 0.3, 0.0]), np.array([1.0, 0.0, 0.0])
    )
    hit_ring = pick_boundary_patch(
        root, np.array([-5.0, 0.6, 0.3]), np.array([1.0, 0.0, 0.0])
    )
    disk = window._hovered_boundary_region(hit_disk)
    ring = window._hovered_boundary_region(hit_ring)
    assert disk is window.document.node(inside_h)
    assert ring is window.document.node(outside_h)

    tool = window.viewport._boundary_tool
    tool.hover = hit_disk
    regions_before = len(window.document.boundary_regions)
    tool.commit()
    # selects the existing child instead of tagging a duplicate
    assert window.scene_tree.selected_handles() == [inside_h]
    assert len(window.document.boundary_regions) == regions_before


def _wait_for_committed_scene(window, timeout: float = 30.0) -> None:
    import time

    app = QApplication.instance()
    viewport = window.viewport
    deadline = time.monotonic() + timeout
    while (
        viewport._committed_surface_scene is None
        or viewport._committed_surface_scene.revision < window.document.version
    ) and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.02)
    assert viewport._committed_surface_scene is not None


def test_cutter_drag_shows_two_color_split_preview() -> None:
    window = _window()
    region = _select_whole_surface_region(window)
    _wait_for_committed_scene(window)
    window.viewport._boundary_region_selected = True
    window.viewport.begin_boundary_cutter_tool("box")

    ghost = window._drag_cutter_ghost(
        region, "box", (-1.9, -0.8, -0.5), (-1.3, 0.0, 0.5), None
    )
    surfaces = window._boundary_cut_preview_surfaces(region, ghost)

    assert len(surfaces) == 2
    colors = {s.color for s in surfaces}
    assert window._BOUNDARY_HIGHLIGHT_COLOR in colors
    assert window._BOUNDARY_CUT_OUTSIDE_COLOR in colors
    assert all(s.indices.size > 0 for s in surfaces)
    window.viewport.cancel_create_tool()
