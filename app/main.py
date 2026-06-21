from __future__ import annotations

import logging
import sys

from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    surface_format = QSurfaceFormat()
    # The SDF interpreter is the only renderer; it needs OpenGL 4.6 core for SSBOs.
    surface_format.setVersion(4, 6)
    surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    surface_format.setDepthBufferSize(24)
    QSurfaceFormat.setDefaultFormat(surface_format)

    application = QApplication(sys.argv)
    application.setApplicationName("casoCAD")
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
