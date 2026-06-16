from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QPoint, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow
from core.mesher import FluidDomain
from core.scene import SceneDocument
from core.sdf import Sphere


def color_counts(image: object) -> tuple[int, int]:
    cyan = 0
    yellow = 0
    for y in range(0, image.height(), 2):
        for x in range(0, image.width(), 2):
            color = image.pixelColor(x, y)
            red, green, blue = color.red(), color.green(), color.blue()
            if green > 100 and blue > 100 and green > red + 20:
                cyan += 1
            if red > 120 and green > 90 and blue < 100:
                yellow += 1
    return cyan, yellow


def main() -> int:
    application = QApplication(sys.argv)
    window = MainWindow()
    sphere = Sphere(name="highlight_sphere", object_id=1, radius=0.7)
    window.document = SceneDocument(
        objects=[sphere],
        fluid_domain=FluidDomain(sphere),
    )
    window._publish_document(frame=True)
    window.show()
    window.viewport.set_grid_visible(False)
    window.viewport.set_components_visible(False)
    window.viewport.set_sdf_opacity(1.0)
    state: dict[str, object] = {}

    def fail(message: str) -> None:
        sys.stderr.write(f"Scene highlight render smoke failed: {message}\n")
        application.exit(1)

    def capture_base() -> None:
        state["base"] = window.viewport.grabFramebuffer()
        window.scene_tree.select_handle(window.document.handle_for(sphere))
        QTimer.singleShot(300, verify_selection)

    def verify_selection() -> None:
        selected = window.viewport.grabFramebuffer()
        cyan, _yellow = color_counts(selected)
        if cyan < 100:
            fail(f"panel selection did not produce a cyan highlight ({cyan=})")
            return
        window.scene_tree.tree.clearSelection()
        center = QPoint(
            window.viewport.width() // 2,
            window.viewport.height() // 2,
        )
        QTest.mouseMove(window.viewport, center + QPoint(40, 0))
        QTest.mouseMove(window.viewport, center)
        QTimer.singleShot(300, verify_hover)

    def verify_hover() -> None:
        hovered = window.viewport.grabFramebuffer()
        _cyan, yellow = color_counts(hovered)
        if window.viewport._scene_hover_object_id != sphere.object_id:
            fail("cursor hover did not resolve the rendered sphere")
            return
        if yellow < 100:
            fail(f"cursor hover did not produce a yellow highlight ({yellow=})")
            return
        window.scene_tree.select_handle(window.document.handle_for(sphere))
        window.viewport.begin_boundary_region_tool(sphere)
        QTimer.singleShot(300, verify_boundary_suppression)

    def verify_boundary_suppression() -> None:
        boundary_mode = window.viewport.grabFramebuffer()
        cyan, _yellow = color_counts(boundary_mode)
        if cyan >= 25:
            fail(
                "normal scene selection highlight remained visible in "
                f"Boundary Region mode ({cyan=})"
            )
            return
        window.close()
        application.exit(0)

    QTimer.singleShot(700, capture_base)
    QTimer.singleShot(10_000, lambda: fail("workflow timed out"))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
