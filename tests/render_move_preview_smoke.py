from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow
from core.mesher import FluidDomain
from core.scene import SceneDocument
from core.sdf import Sphere


def visible_pixels(image: object) -> tuple[int, int]:
    bright = 0
    cyan = 0
    for y in range(0, image.height(), 2):
        for x in range(0, image.width(), 2):
            color = image.pixelColor(x, y)
            red, green, blue = color.red(), color.green(), color.blue()
            if red + green + blue > 60:
                bright += 1
            if green > 100 and blue > 100 and green > red + 20:
                cyan += 1
    return bright, cyan


def main() -> int:
    application = QApplication(sys.argv)
    sphere = Sphere(name="move_preview_sphere", object_id=1, radius=0.7)
    window = MainWindow()
    window.document = SceneDocument(
        objects=[sphere],
        fluid_domain=FluidDomain(sphere),
    )
    window._publish_document(frame=True)
    window.show()
    window.viewport.set_grid_visible(False)
    window.viewport.set_components_visible(False)
    window.viewport.set_sdf_opacity(1.0)
    window.scene_tree.select_handle(window.document.handle_for(sphere))

    def fail(message: str) -> None:
        sys.stderr.write(f"Move preview render smoke failed: {message}\n")
        application.exit(1)

    def activate_preview() -> None:
        window.viewport.begin_move_tool(window.document.handle_for(sphere))
        window.viewport.nudge_move_preview((0.35, 0.0, 0.0))
        QTimer.singleShot(300, verify)

    def verify() -> None:
        image = window.viewport.grabFramebuffer()
        bright, cyan = visible_pixels(image)
        if bright < 1_000:
            fail(f"viewport rendered too few visible pixels ({bright=})")
            return
        if cyan < 100:
            fail(f"move preview did not draw a cyan preview ({cyan=})")
            return
        window.close()
        application.exit(0)

    QTimer.singleShot(700, activate_preview)
    QTimer.singleShot(10_000, lambda: fail("workflow timed out"))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
