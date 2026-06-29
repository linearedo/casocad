from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from .main_window import MainWindow

_THEME_QSS = Path(__file__).resolve().parent / "assets" / "theme.qss"


def _load_theme(application: QApplication) -> None:
    """Apply the global appearance theme. Styling only — never touches layout,
    so dock/splitter/window positions are unaffected."""
    # Ask the platform for a dark color scheme, so the native window title bar
    # (drawn by the OS/WM, not by our QSS) renders dark instead of white.
    # Honoring it is up to the desktop environment; it's a no-op where unsupported.
    style_hints = application.styleHints()
    if hasattr(style_hints, "setColorScheme"):
        style_hints.setColorScheme(Qt.ColorScheme.Dark)
    try:
        application.setStyleSheet(_THEME_QSS.read_text(encoding="utf-8"))
    except OSError as exc:  # pragma: no cover - falls back to default Qt look
        logging.getLogger(__name__).warning("Could not load theme: %s", exc)


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
    _load_theme(application)
    window = MainWindow()
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
