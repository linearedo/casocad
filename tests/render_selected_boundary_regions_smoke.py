from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow
from app.panels.scene_tree import HANDLE_ROLE
from core.mesher import FluidDomain
from core.scene import SceneDocument
from core.sdf import Box, PlacedSDF2D, RectangleProfile


def image_difference(first: object, second: object) -> tuple[int, int]:
    difference = 0
    cyan = 0
    height = min(first.height(), second.height())
    width = min(first.width(), second.width())
    for y in range(0, height, 2):
        for x in range(0, width, 2):
            before = first.pixelColor(x, y)
            after = second.pixelColor(x, y)
            difference += (
                abs(after.red() - before.red())
                + abs(after.green() - before.green())
                + abs(after.blue() - before.blue())
            )
            if (
                after.green() > 100
                and after.blue() > 100
                and after.green() > after.red() + 20
            ):
                cyan += 1
    return difference, cyan


def main() -> int:
    application = QApplication(sys.argv)
    window = MainWindow()
    box = Box(
        name="box",
        object_id=1,
        half_size=(0.8, 0.6, 0.5),
    )
    inlet = PlacedSDF2D(
        name="inlet",
        object_id=2,
        profile=RectangleProfile(half_size=(0.45, 0.35)),
        origin=(0.0, 0.6, 0.0),
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 0.0, 1.0),
    )
    window.document = SceneDocument(
        objects=[box, inlet],
        fluid_domain=FluidDomain(box, (inlet,)),
    )
    first_handle = window.document.add_boundary_region(
        box.object_id,
        outside_direction=2,
    )
    second_handle = window.document.add_boundary_region(
        box.object_id,
        outside_direction=3,
    )
    window._publish_document(frame=True)
    window.show()
    window.viewport.set_grid_visible(False)
    window.viewport.set_components_visible(False)
    window.viewport.set_sdf_opacity(0.4)
    window.viewport.camera.yaw_degrees = 0.0
    window.viewport.camera.pitch_degrees = 0.0
    window.viewport.camera.distance = 3.0
    window.viewport.camera.target = (0.0, 0.0, 0.0)
    state: dict[str, object] = {}

    def fail(message: str) -> None:
        sys.stderr.write(
            f"Selected BoundaryRegions render smoke failed: {message}\n"
        )
        application.exit(1)

    def select_inlet() -> None:
        state["before"] = window.viewport.grabFramebuffer()
        window.scene_tree.select_handle(window.document.handle_for(inlet))
        QTimer.singleShot(300, capture_inlet)

    def capture_inlet() -> None:
        image = window.viewport.grabFramebuffer()
        state["inlet_center"] = image.pixelColor(
            image.width() // 2,
            image.height() // 2,
        )
        items = window.scene_tree.tree.findItems(
            "*",
            Qt.MatchFlag.MatchWildcard | Qt.MatchFlag.MatchRecursive,
            0,
        )
        window.scene_tree.tree.blockSignals(True)
        window.scene_tree.tree.clearSelection()
        for item in items:
            if item.data(0, HANDLE_ROLE) == second_handle:
                item.setSelected(True)
        window.scene_tree.tree.blockSignals(False)
        window.scene_tree._on_selection_changed()
        QTimer.singleShot(300, verify_appearance)

    def verify_appearance() -> None:
        expected = ((box.object_id, 1 << 3),)
        if window.viewport._selected_boundary_regions != expected:
            fail(
                "the selected Scene region did not reach the viewport "
                f"(got={window.viewport._selected_boundary_regions})"
            )
            return
        if window.viewport._interaction_tool is not None:
            fail("panel selection unexpectedly activated a viewport tool")
            return
        difference, cyan = image_difference(
            state["before"],
            window.viewport.grabFramebuffer(),
        )
        if difference < 50_000 or cyan < 100:
            fail(
                "selected regions did not visibly highlight the exposed box "
                f"face (difference={difference}, cyan={cyan})"
            )
            return
        image = window.viewport.grabFramebuffer()
        inlet_center = state["inlet_center"]
        boundary_center = image.pixelColor(
            image.width() // 2,
            image.height() // 2,
        )
        channel_difference = sum(
            abs(first - second)
            for first, second in zip(
                (
                    inlet_center.red(),
                    inlet_center.green(),
                    inlet_center.blue(),
                ),
                (
                    boundary_center.red(),
                    boundary_center.green(),
                    boundary_center.blue(),
                ),
                strict=True,
            )
        )
        if channel_difference > 12:
            fail(
                "BoundaryRegion and PlacedSDF2D selection overlays do not "
                "match at the same surface point "
                f"({channel_difference=}, "
                f"inlet={inlet_center.getRgb()[:3]}, "
                f"boundary={boundary_center.getRgb()[:3]})"
            )
            return
        select_both_regions()

    def select_both_regions() -> None:
        items = window.scene_tree.tree.findItems(
            "*",
            Qt.MatchFlag.MatchWildcard | Qt.MatchFlag.MatchRecursive,
            0,
        )
        window.scene_tree.tree.blockSignals(True)
        window.scene_tree.tree.clearSelection()
        for item in items:
            if item.data(0, HANDLE_ROLE) in {first_handle, second_handle}:
                item.setSelected(True)
        window.scene_tree.tree.blockSignals(False)
        window.scene_tree._on_selection_changed()
        QTimer.singleShot(300, verify_multiple_selection)

    def verify_multiple_selection() -> None:
        expected = ((box.object_id, (1 << 2) | (1 << 3)),)
        if window.viewport._selected_boundary_regions != expected:
            fail(
                "multiple Scene selections did not merge their direction masks "
                f"(got={window.viewport._selected_boundary_regions})"
            )
            return
        window.close()
        application.exit(0)

    QTimer.singleShot(700, select_inlet)
    QTimer.singleShot(10_000, lambda: fail("workflow timed out"))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
