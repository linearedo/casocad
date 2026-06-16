from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow
from core.mesher import FluidDomain
from core.scene import SceneDocument
from core.sdf import Box, Sphere


def main() -> int:
    application = QApplication(sys.argv)
    window = MainWindow()
    outer = Box(
        name="outer",
        object_id=1,
        half_size=(0.9, 0.9, 0.9),
    )
    inner = Sphere(
        name="inner",
        object_id=2,
        radius=0.42,
    )
    window.document = SceneDocument(
        objects=[outer, inner],
        fluid_domain=FluidDomain(outer),
    )
    window._publish_document(frame=True)
    window.show()
    window.viewport.set_grid_visible(False)
    window.viewport.set_components_visible(False)
    window.viewport.set_sdf_opacity(1.0)
    window.viewport.update()
    state: dict[str, object] = {}

    def capture_opaque() -> None:
        state["opaque"] = window.viewport.grabFramebuffer()
        window.viewport.set_sdf_opacity(0.25)
        window.viewport.update()
        QTimer.singleShot(500, verify_transparent)

    def verify_transparent() -> None:
        opaque = state["opaque"]
        transparent = window.viewport.grabFramebuffer()
        difference = 0
        blue_gain = 0
        for y in range(0, transparent.height(), 3):
            for x in range(0, transparent.width(), 3):
                before = opaque.pixelColor(x, y)
                after = transparent.pixelColor(x, y)
                difference += (
                    abs(after.red() - before.red())
                    + abs(after.green() - before.green())
                    + abs(after.blue() - before.blue())
                )
                before_blue = (
                    before.blue() > before.red() + 25
                    and before.blue() > before.green() + 10
                )
                after_blue = (
                    after.blue() > after.red() + 25
                    and after.blue() > after.green() + 10
                )
                blue_gain += int(after_blue and not before_blue)
        if difference < 100_000 or blue_gain < 40:
            sys.stderr.write(
                "Transparency render smoke failed: enclosed object was not "
                f"revealed (difference={difference}, blue_gain={blue_gain})\n"
            )
            application.exit(1)
            return
        window.close()
        application.exit(0)

    QTimer.singleShot(800, capture_opaque)
    QTimer.singleShot(10_000, lambda: application.exit(1))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
