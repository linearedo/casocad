from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QPoint, Qt, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow
from app.signals import signals
from core.boundary import BoundaryRegion
from core.mesher import LatticeMesher, MesherConfig


def cyan_pixels(image: object) -> int:
    count = 0
    for y in range(0, image.height(), 2):
        for x in range(0, image.width(), 2):
            color = image.pixelColor(x, y)
            if (
                color.green() > 100
                and color.blue() > 100
                and color.green() > color.red() + 20
            ):
                count += 1
    return count


def main() -> int:
    application = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    hovered_owner_ids: list[int] = []
    state: dict[str, object] = {}
    output = Path("/tmp/casocad-boundary-region-smoke.arrow")
    output.unlink(missing_ok=True)

    def fail(message: str) -> None:
        sys.stderr.write(f"BoundaryRegion workflow smoke failed: {message}\n")
        application.exit(1)

    def record_hover(
        selection: tuple[int, tuple[float, float, float]] | None,
    ) -> None:
        if selection is not None:
            hovered_owner_ids.append(selection[0])

    def create_region() -> None:
        assert window.document.fluid_domain is not None
        window.viewport.set_grid_visible(False)
        window.viewport.set_components_visible(False)
        window.viewport.set_sdf_opacity(1.0)
        window.viewport.update()
        QTimer.singleShot(300, begin_hover)

    def begin_hover() -> None:
        state["before"] = window.viewport.grabFramebuffer()
        window.viewport.begin_boundary_region_tool(
            window.document.fluid_domain.root
        )
        center = QPoint(
            window.viewport.width() // 2,
            window.viewport.height() // 2,
        )
        original_yaw = window.viewport.camera.yaw_degrees
        original_target = window.viewport.camera.target
        original_pitch = window.viewport.camera.pitch_degrees
        original_distance = window.viewport.camera.distance
        QTest.mousePress(
            window.viewport,
            Qt.MouseButton.LeftButton,
            pos=center,
        )
        QTest.mouseMove(window.viewport, center + QPoint(35, 12), delay=50)
        QTest.mouseRelease(
            window.viewport,
            Qt.MouseButton.LeftButton,
            pos=center + QPoint(35, 12),
        )
        if window.viewport.camera.yaw_degrees == original_yaw:
            fail("left-drag did not orbit while BoundaryRegion tool was active")
            return
        if window.document.boundary_regions:
            fail("camera orbit accidentally created a BoundaryRegion")
            return
        QTest.mousePress(
            window.viewport,
            Qt.MouseButton.RightButton,
            pos=center,
        )
        QTest.mouseMove(window.viewport, center + QPoint(25, 15), delay=50)
        QTest.mouseRelease(
            window.viewport,
            Qt.MouseButton.RightButton,
            pos=center + QPoint(25, 15),
        )
        if window.viewport.camera.target == original_target:
            fail("right-drag did not pan while BoundaryRegion tool was active")
            return
        if window.viewport._interaction_tool is None:
            fail("camera navigation cancelled the BoundaryRegion tool")
            return
        window.viewport.camera.yaw_degrees = original_yaw
        window.viewport.camera.pitch_degrees = original_pitch
        window.viewport.camera.distance = original_distance
        window.viewport.camera.target = original_target
        window.viewport.update()
        QTest.qWait(100)
        QTest.mouseMove(window.viewport, center + QPoint(10, 0))
        QTest.qWait(100)
        QTest.mouseMove(window.viewport, center)
        QTest.qWait(100)
        state["center"] = center
        QTimer.singleShot(300, verify_highlight)

    def verify_highlight() -> None:
        attempts = int(state.get("highlight_attempts", 0))
        if window.viewport._boundary_hover_owner_id == 0 and attempts < 10:
            state["highlight_attempts"] = attempts + 1
            QTimer.singleShot(100, verify_highlight)
            return
        before = state["before"]
        after = window.viewport.grabFramebuffer()
        difference = 0
        yellow_pixels = 0
        for y in range(0, after.height(), 2):
            for x in range(0, after.width(), 2):
                first = before.pixelColor(x, y)
                second = after.pixelColor(x, y)
                difference += (
                    abs(second.red() - first.red())
                    + abs(second.green() - first.green())
                    + abs(second.blue() - first.blue())
                )
                if (
                    second.red() > 150
                    and second.green() > 100
                    and second.blue() < 100
                ):
                    yellow_pixels += 1
        if (difference < 50_000 or yellow_pixels < 50) and attempts < 10:
            state["highlight_attempts"] = attempts + 1
            QTimer.singleShot(100, verify_highlight)
            return
        if difference < 50_000 or yellow_pixels < 50:
            fail(
                "hover did not visibly color boundary owners and highlight the "
                f"candidate region (difference={difference}, yellow={yellow_pixels})"
            )
            return
        QTest.mouseClick(
            window.viewport,
            Qt.MouseButton.LeftButton,
            pos=state["center"],
        )
        QTimer.singleShot(500, verify_region)

    def verify_region() -> None:
        if not hovered_owner_ids:
            fail("hovering the final boundary did not resolve an owner")
            return
        if len(window.document.boundary_regions) != 1:
            fail("clicking the final boundary did not create one region")
            return
        region = window.document.boundary_regions[0]
        if not isinstance(region, BoundaryRegion):
            fail("created scene item is not a BoundaryRegion")
            return
        if window.document.fluid_domain is None:
            fail("FluidDomain was lost")
            return
        if region not in window.document.fluid_domain.tag_objects:
            fail("created BoundaryRegion was not enabled as a lattice tag")
            return
        state["selected_region"] = region
        state["selected_region_cyan"] = cyan_pixels(
            window.viewport.grabFramebuffer()
        )
        window.scene_tree.tree.clearSelection()
        QTimer.singleShot(300, verify_panel_selection_highlight)

    def verify_panel_selection_highlight() -> None:
        region = state["selected_region"]
        assert isinstance(region, BoundaryRegion)
        selected_cyan = int(state["selected_region_cyan"])
        unselected_cyan = cyan_pixels(window.viewport.grabFramebuffer())
        if selected_cyan < unselected_cyan + 50:
            fail(
                "the created BoundaryRegion was not highlighted through its "
                "selected Scene entry "
                f"(selected={selected_cyan}, unselected={unselected_cyan})"
            )
            return
        window.scene_tree.select_handle(window.document.handle_for(region))
        result = LatticeMesher(
            window.document.fluid_domain,
            MesherConfig(dx=0.2, chunk_size=1_000),
        ).mesh(output)
        tagged = np.asarray(
            [region.object_id in items for items in result.preview_tag_ids],
            dtype=np.bool_,
        )
        if not tagged.any():
            fail("created BoundaryRegion matched no preview boundary nodes")
            return
        if np.any(result.preview_node_types[tagged] != 1):
            fail("BoundaryRegion tagged an internal lattice node")
            return
        output.unlink(missing_ok=True)
        window.close()
        application.exit(0)

    signals.viewport_boundary_hovered.connect(record_hover)
    QTimer.singleShot(500, create_region)
    QTimer.singleShot(15_000, lambda: fail("workflow timed out"))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
