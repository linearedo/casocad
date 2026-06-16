from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow


def main() -> int:
    application = QApplication(sys.argv)
    window = MainWindow()
    window.show()

    def empty_scene() -> None:
        window.viewport.camera.target = (1_000.0, 1_000.0, 1_000.0)
        window.viewport.set_mode("lattice")
        handles = [handle for handle, _node, _parent in window.document.walk()]
        window._on_delete_nodes(handles)
        if window.viewport.mode != "sdf":
            fail("viewport did not return to SDF mode")
            return
        if np.linalg.norm(np.asarray(window.viewport.camera.target)) > 1e-6:
            fail("camera did not frame the default grid")
            return
        window.viewport.camera.target = (100.0, 100.0, 0.0)
        window.viewport.update()
        QTimer.singleShot(750, verify)

    def fail(message: str) -> None:
        sys.stderr.write(f"Empty-scene grid smoke failed: {message}\n")
        application.exit(1)

    def verify() -> None:
        image = window.viewport.grabFramebuffer().convertToFormat(
            QImage.Format.Format_RGB888
        )
        pixels = np.frombuffer(
            image.constBits(),
            dtype=np.uint8,
            count=image.sizeInBytes(),
        ).reshape(image.height(), image.bytesPerLine())[:, : image.width() * 3]
        pixels = pixels.reshape(image.height(), image.width(), 3)
        color_span = pixels.max(axis=(0, 1)).astype(np.int16) - pixels.min(
            axis=(0, 1)
        ).astype(np.int16)
        if int(color_span.max()) < 40:
            fail(
                "framebuffer has no visible infinite grid at world position "
                f"(100, 100, 0) (RGB span={color_span.tolist()})"
            )
            return
        window.close()
        application.exit(0)

    QTimer.singleShot(500, empty_scene)
    QTimer.singleShot(10_000, lambda: application.exit(1))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
