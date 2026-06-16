from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow


def main() -> int:
    application = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.viewport.set_grid_visible(False)
    window.viewport.set_components_visible(False)
    window.viewport.set_sdf_opacity(1.0)
    window.viewport.camera.yaw_degrees = 0.0
    window.viewport.camera.pitch_degrees = 89.0
    window.viewport.camera.distance = 4.0
    window.viewport.camera.target = (0.0, 0.0, 0.0)
    window.viewport.update()

    def verify() -> None:
        image = window.viewport.grabFramebuffer()
        center_x = image.width() // 2
        center_y = image.height() // 2
        center = image.pixelColor(center_x, center_y)
        face = image.pixelColor(center_x + 90, center_y)
        center_light = center.red() + center.green() + center.blue()
        face_light = face.red() + face.green() + face.blue()
        if center_light >= face_light:
            sys.stderr.write(
                "Difference render smoke failed: center does not show the hole "
                f"(center={center_light}, face={face_light})\n"
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
