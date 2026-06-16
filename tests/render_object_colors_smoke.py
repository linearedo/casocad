from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow
from core.mesher import FluidDomain
from core.scene import SceneDocument
from core.sdf import Box, Sphere, Union


def main() -> int:
    application = QApplication(sys.argv)
    window = MainWindow()
    box = Box(
        name="box",
        object_id=1,
        center=(-0.7, 0.0, 0.0),
        half_size=(0.45, 0.45, 0.45),
    )
    sphere = Sphere(
        name="sphere",
        object_id=2,
        center=(0.7, 0.0, 0.0),
        radius=0.48,
    )
    domain = Union(
        name="two_object_domain",
        object_id=3,
        left=box,
        right=sphere,
    )
    window.document = SceneDocument(
        objects=[box, sphere],
        fluid_domain=FluidDomain(domain),
    )
    window._publish_document(frame=True)
    window.show()
    window.viewport.set_grid_visible(False)
    window.viewport.set_components_visible(False)
    window.viewport.set_sdf_opacity(1.0)
    window.viewport.update()

    def verify() -> None:
        image = window.viewport.grabFramebuffer()
        red_pixels = 0
        blue_pixels = 0
        for y in range(0, image.height(), 2):
            for x in range(0, image.width(), 2):
                color = image.pixelColor(x, y)
                red, green, blue = color.red(), color.green(), color.blue()
                if red > 55 and red > blue + 18 and red > green + 8:
                    red_pixels += 1
                if blue > 55 and blue > red + 18 and blue > green:
                    blue_pixels += 1
        if red_pixels < 100 or blue_pixels < 25:
            sys.stderr.write(
                "Object color render smoke failed: expected distinct box and "
                f"cylinder colors, got red={red_pixels}, blue={blue_pixels}\n"
            )
            application.exit(1)
            return
        window.close()
        application.exit(0)

    QTimer.singleShot(1_000, verify)
    QTimer.singleShot(10_000, lambda: application.exit(1))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
