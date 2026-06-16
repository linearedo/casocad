from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMenu, QSlider

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow
from app.signals import signals
from core.io import read_lattice
from core.sdf import Box, Cylinder, PlacedSDF2D, RectangleProfile


def main() -> int:
    application = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    state: dict[str, object] = {"phase": "3d"}
    export_path = Path("/tmp/casocad-gui-workflow.arrow")
    export_path.unlink(missing_ok=True)

    def fail(message: str) -> None:
        sys.stderr.write(f"GUI workflow smoke failed: {message}\n")
        application.exit(1)

    def build_and_mesh() -> None:
        window._new_empty_scene()
        opacity_slider = window.findChild(QSlider, "sdfOpacitySlider")
        if opacity_slider is None:
            fail("SDF opacity control is missing")
            return
        if opacity_slider.value() != 40:
            fail("SDF scene does not default to transparent mode")
            return
        opacity_slider.setValue(45)
        if not np.isclose(window.viewport.sdf_opacity, 0.45):
            fail("SDF opacity control did not update the viewport")
            return
        opacity_slider.setValue(100)
        if window.viewport._grid_spacing <= 0.0:
            fail("empty scene did not retain a valid reference-grid spacing")
            return
        viewport = window.viewport
        first = QPoint(viewport.width() // 2 - 120, viewport.height() // 2 - 50)
        second = QPoint(viewport.width() // 2 + 80, viewport.height() // 2 + 70)
        viewport.begin_create_tool("box")
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=first)
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=second)

        first = QPoint(viewport.width() // 2 - 40, viewport.height() // 2 - 30)
        second = QPoint(viewport.width() // 2 + 30, viewport.height() // 2 + 40)
        viewport.begin_create_tool("cylinder")
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=first)
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=second)
        box, cylinder = window.document.objects
        if not isinstance(box, Box) or not isinstance(cylinder, Cylinder):
            fail("mouse creation did not produce box and cylinder")
            return
        box.half_size = (1.2, 0.6, 0.5)
        cylinder.radius = 0.22
        cylinder.half_height = 0.5
        state["obstacle_id"] = cylinder.object_id

        old_center = box.center
        viewport.begin_move_tool(window.document.handle_for(box))
        start = QPoint(viewport.width() // 2, viewport.height() // 2)
        end = QPoint(viewport.width() // 2 + 30, viewport.height() // 2)
        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, pos=start)
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, pos=end)
        QTest.keyClick(viewport, Qt.Key.Key_Return)
        if box.center == old_center:
            fail("mouse move tool did not move the box")
            return

        boolean_menu = QMenu()
        window.scene_tree._populate_object_boolean_menu(
            boolean_menu, window.document.handle_for(box)
        )
        subtract_menu = next(
            (
                submenu
                for submenu in window.scene_tree._context_submenus
                if submenu.title() == "Subtract from this"
            ),
            None,
        )
        if subtract_menu is None:
            fail("per-object subtract menu is missing")
            return
        subtract_action = next(
            (
                action
                for action in subtract_menu.actions()
                if action.text().startswith(cylinder.name)
            ),
            None,
        )
        if subtract_action is None:
            fail("per-object boolean operand is missing")
            return
        subtract_action.trigger()
        root = window.document.objects[0]
        root_handle = window.document.handle_for(root)
        state["root_handle"] = root_handle
        window._on_set_fluid_root([root_handle])

        for name, x_position in (("inlet", -1.2), ("outlet", 1.2)):
            window._on_add_primitive("rectangle")
            tag = window.document.objects[-1]
            if not isinstance(tag, PlacedSDF2D):
                fail("placed section creation failed")
                return
            tag.name = name
            tag.origin = (x_position, 0.0, 0.0)
            tag.axis_u = (0.0, 1.0, 0.0)
            tag.axis_v = (0.0, 0.0, 1.0)
            tag.profile = RectangleProfile(half_size=(0.6, 0.5))
            tag.__post_init__()
            window._on_set_tag_enabled([window.document.handle_for(tag)], True)

        window.mesher_panel.dx.setValue(0.12)
        window.mesher_panel.boundary_error_tolerance.setValue(0.04)
        window._on_mesh_requested()

    def verify_result(result: object) -> None:
        if state["phase"] == "2d":
            verify_2d_result(result)
            return
        if not result.boundary_error_tolerance_met:
            fail("automatic dx refinement did not meet the error target")
            return
        if result.boundary_error_maximum > 0.04:
            fail("reported maximum boundary error exceeds the GUI target")
            return
        if "maximum" not in window.mesher_panel.status.text():
            fail("Mesh panel did not report boundary error statistics")
            return
        window._on_selection_changed([state["root_handle"]])
        pending = window.viewport._pending_lattice
        if pending is None or pending[0].shape[0] != result.preview_positions.shape[0]:
            fail("boolean result selection did not retain its complete lattice")
            return
        selected_boundary_sources = pending[3][pending[1] == 1]
        if state["obstacle_id"] not in selected_boundary_sources:
            fail("boolean result selection hid subtractive obstacle boundary cells")
            return
        selected_internal_sources = pending[3][pending[1] == 0]
        if not np.all(
            selected_internal_sources
            == window.document.node(state["root_handle"]).object_id
        ):
            fail("boolean result internal lattice did not use the result object color")
            return
        boundary_sources = result.preview_source_object_ids[
            result.preview_node_types == 1
        ]
        if state["obstacle_id"] not in boundary_sources:
            fail("subtractive obstacle has no boundary preview cells")
            return
        obstacle_sources = (
            result.preview_source_object_ids == state["obstacle_id"]
        )
        if np.any(result.preview_node_types[obstacle_sources] != 1):
            fail("subtractive obstacle owns internal fluid preview cells")
            return
        if not np.any(result.preview_primary_tag_ids):
            fail("placed sections produced no tagged preview cells")
            return

        def verify_renderer() -> None:
            renderer = window.viewport._renderer
            if renderer is None:
                fail("OpenGL renderer was not initialized")
                return
            if renderer._point_count <= 0:
                fail("internal lattice points were not uploaded")
                return
            if renderer._square_count <= 0:
                fail("boundary lattice faces were not uploaded as squares")
                return
            if renderer._gizmo_label_vertex_count <= 0:
                fail("camera gyroscope axis labels were not uploaded")
                return
            start_export()

        QTimer.singleShot(750, verify_renderer)

    def start_export() -> None:
        if window._thread is not None:
            QTimer.singleShot(100, start_export)
            return
        window._on_export_requested(str(export_path))

    def verify_export(result: object) -> None:
        if result.path != export_path:
            return
        table, metadata = read_lattice(export_path)
        if table.num_rows != result.row_count:
            fail("Arrow row count does not match mesher result")
            return
        if "tag_ids" not in table.schema.names:
            fail("Arrow tag_ids column is missing")
            return
        if metadata["fluid_domain"]["root_object_id"] <= 0:
            fail("Arrow FluidDomain metadata is invalid")
            return
        export_path.unlink(missing_ok=True)
        start_2d_when_idle()

    def start_2d_when_idle() -> None:
        if window._thread is not None:
            QTimer.singleShot(100, start_2d_when_idle)
            return
        window._new_empty_scene()
        window._on_add_primitive("rectangle")
        rectangle = window.document.objects[0]
        if not isinstance(rectangle, PlacedSDF2D):
            fail("2D FluidDomain creation failed")
            return
        rectangle.profile = RectangleProfile(half_size=(0.5, 0.25))
        rectangle.__post_init__()
        window._on_set_fluid_root([window.document.handle_for(rectangle)])
        if (
            window.document.fluid_domain is None
            or window.document.fluid_domain.root is not rectangle
        ):
            fail("2D SDF could not be selected as the FluidDomain")
            return
        boundary_handle = window.document.add_boundary_region(
            rectangle.object_id,
            outside_direction=0,
        )
        boundary = window.document.node(boundary_handle)
        if boundary.dimension != 1:
            fail("2D boundary creation did not produce a 1D SDF")
            return
        state["2d_tag_id"] = boundary.object_id
        window._publish_document(clear_selection=False)
        window.mesher_panel.dx.setValue(0.25)
        window.mesher_panel.boundary_error_tolerance.setValue(0.02)
        if " x " not in window.mesher_panel.estimate.text():
            fail("2D candidate-node estimate is missing")
            return
        state["phase"] = "2d"
        window._on_mesh_requested()

    def verify_2d_result(result: object) -> None:
        if result.dimension != 2:
            fail("2D FluidDomain produced a non-2D lattice result")
            return
        if result.grid_node_count != 15 or result.row_count != 15:
            fail(
                "2D rectangle lattice did not produce the expected "
                f"15 nodes (grid={result.grid_node_count}, rows={result.row_count})"
            )
            return
        if np.any(result.boundary_sample_directions > 3):
            fail("2D lattice used a non-planar boundary direction")
            return
        tag_id = int(state["2d_tag_id"])
        tagged = np.asarray(
            [tag_id in items for items in result.preview_tag_ids],
            dtype=np.bool_,
        )
        if np.count_nonzero(tagged) != 3:
            fail("1D boundary SDF did not tag the expected 2D side nodes")
            return
        finish_when_idle()

    def finish_when_idle() -> None:
        if window._thread is not None:
            QTimer.singleShot(100, finish_when_idle)
            return
        window.close()
        application.exit(0)

    signals.preview_ready.connect(verify_result)
    signals.mesh_ready.connect(verify_export)
    QTimer.singleShot(500, build_and_mesh)
    QTimer.singleShot(15_000, lambda: fail("workflow timed out"))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
