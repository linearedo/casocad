"""Custom frameless-window title bar.

casoCAD removes the native OS title bar and draws its own so the dark "night
blue" look is identical on every operating system and Linux desktop environment
(native title bars are drawn by the OS/WM and cannot be portably recolored).

Because the native bar is gone, this module also restores the behaviour it
provided: window dragging, double-click maximize, the min/max/close buttons,
and edge resizing. Move and resize are delegated to the window manager via
``QWindow.startSystemMove`` / ``startSystemResize`` so native snapping and
multi-monitor behaviour are preserved.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, QObject, QSize, Qt
from PySide6.QtGui import QCursor, QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QWidget,
)
from shiboken6 import isValid

_WORDMARK = Path(__file__).resolve().parent / "assets" / "casocad-wordmark.png"


class CustomTitleBar(QWidget):
    """The night-blue title bar shown at the top of the main window."""

    HEIGHT = 36
    _LOGO_HEIGHT = 22

    def __init__(self, window: QWidget, suffix: str = "") -> None:
        super().__init__(window)
        self._window = window
        self.setObjectName("customTitleBar")
        self.setFixedHeight(self.HEIGHT)
        self.setAutoFillBackground(True)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 0, 6, 0)
        layout.setSpacing(8)

        logo = QLabel(self)
        logo.setObjectName("titleBarLogo")
        pixmap = QPixmap(str(_WORDMARK))
        if not pixmap.isNull():
            pixmap.setDevicePixelRatio(pixmap.height() / self._LOGO_HEIGHT)
            logo.setPixmap(pixmap)
        layout.addWidget(logo)

        if suffix:
            suffix_label = QLabel(suffix, self)
            suffix_label.setObjectName("titleBarSuffix")
            layout.addWidget(suffix_label)

        layout.addStretch(1)

        # Keep the maximize/restore glyph in sync no matter how the window state
        # changes (our button, the WM, or a keyboard shortcut).
        window.installEventFilter(self)

        self._min_button = self._make_button(
            "titleBarMinButton", "–", window.showMinimized)
        self._max_button = self._make_button(
            "titleBarMaxButton", "□", self._toggle_maximized)
        self._close_button = self._make_button(
            "titleBarCloseButton", "✕", window.close)
        for button in (self._min_button, self._max_button, self._close_button):
            layout.addWidget(button)

    def _make_button(self, name: str, glyph: str, slot) -> QPushButton:
        button = QPushButton(glyph, self)
        button.setObjectName(name)
        button.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        button.setFixedSize(QSize(42, self.HEIGHT))
        button.clicked.connect(slot)
        return button

    def _toggle_maximized(self) -> None:
        if self._window.isMaximized():
            self._window.showNormal()
        else:
            self._window.showMaximized()
        self.sync_state()

    def sync_state(self) -> None:
        """Refresh the maximize/restore glyph to match the window state."""
        restore = self._window.isMaximized()
        self._max_button.setText("❐" if restore else "□")
        self._max_button.setToolTip("Restore" if restore else "Maximize")

    # -- drag-to-move (delegated to the window manager) --------------------
    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            handle = self._window.windowHandle()
            if handle is not None and handle.startSystemMove():
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._toggle_maximized()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def eventFilter(self, obj, event) -> bool:
        if (obj is self._window
                and event.type() == QEvent.Type.WindowStateChange):
            self.sync_state()
        return False


class FramelessResizer(QObject):
    """Application-level event filter that resizes a frameless window from its
    edges, handing the actual resize to the window manager so snapping works."""

    _CURSORS = {
        Qt.Edge.LeftEdge: Qt.CursorShape.SizeHorCursor,
        Qt.Edge.RightEdge: Qt.CursorShape.SizeHorCursor,
        Qt.Edge.TopEdge: Qt.CursorShape.SizeVerCursor,
        Qt.Edge.BottomEdge: Qt.CursorShape.SizeVerCursor,
        Qt.Edge.LeftEdge | Qt.Edge.TopEdge: Qt.CursorShape.SizeFDiagCursor,
        Qt.Edge.RightEdge | Qt.Edge.BottomEdge: Qt.CursorShape.SizeFDiagCursor,
        Qt.Edge.RightEdge | Qt.Edge.TopEdge: Qt.CursorShape.SizeBDiagCursor,
        Qt.Edge.LeftEdge | Qt.Edge.BottomEdge: Qt.CursorShape.SizeBDiagCursor,
    }

    def __init__(self, window: QWidget, margin: int = 6) -> None:
        super().__init__(window)
        self._window = window
        self._margin = margin
        self._cursor_overridden = False

    def _edges_at(self, gpos) -> Qt.Edge:
        geom = self._window.geometry()
        margin = self._margin
        edges = Qt.Edge(0)
        if abs(gpos.x() - geom.left()) <= margin:
            edges |= Qt.Edge.LeftEdge
        if abs(gpos.x() - geom.right()) <= margin:
            edges |= Qt.Edge.RightEdge
        if abs(gpos.y() - geom.top()) <= margin:
            edges |= Qt.Edge.TopEdge
        if abs(gpos.y() - geom.bottom()) <= margin:
            edges |= Qt.Edge.BottomEdge
        return edges

    def _set_cursor(self, edges: Qt.Edge) -> None:
        shape = self._CURSORS.get(edges)
        if shape is not None:
            self._window.setCursor(QCursor(shape))
            self._cursor_overridden = True
        elif self._cursor_overridden:
            self._window.unsetCursor()
            self._cursor_overridden = False

    def eventFilter(self, obj, event) -> bool:
        window = self._window
        if not isValid(window):  # window torn down while the filter is installed
            return False
        if window.isMaximized() or window.isFullScreen():
            if self._cursor_overridden:
                window.unsetCursor()
                self._cursor_overridden = False
            return False

        etype = event.type()
        if etype == QEvent.Type.MouseMove:
            if not (event.buttons() & Qt.MouseButton.LeftButton):
                gpos = event.globalPosition().toPoint()
                if window.geometry().contains(gpos):
                    self._set_cursor(self._edges_at(gpos))
                elif self._cursor_overridden:
                    window.unsetCursor()
                    self._cursor_overridden = False
        elif (etype == QEvent.Type.MouseButtonPress
              and event.button() == Qt.MouseButton.LeftButton):
            gpos = event.globalPosition().toPoint()
            if window.geometry().contains(gpos):
                edges = self._edges_at(gpos)
                if edges:
                    handle = window.windowHandle()
                    if handle is not None and handle.startSystemResize(edges):
                        return True
        return False


def install_title_bar(window, menubar, suffix: str = ""):
    """Make a ``QMainWindow`` frameless and give it casoCAD's night-blue title
    bar, stacked above ``menubar`` in the window's top (menu) row.

    Returns ``(title_bar, resizer)``; the window already parents both, so the
    caller only needs to keep them alive if it wants to reference them later.
    Layout of the central widget, docks and toolbars is left untouched.
    """
    # Imported here to keep the module's top-level imports focused on the widgets.
    from PySide6.QtWidgets import QApplication, QVBoxLayout, QWidget

    window.setWindowFlag(Qt.WindowType.FramelessWindowHint, True)
    title_bar = CustomTitleBar(window, suffix)
    container = QWidget(window)
    container.setObjectName("titleBarContainer")
    layout = QVBoxLayout(container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(0)
    layout.addWidget(title_bar)
    layout.addWidget(menubar)
    window.setMenuWidget(container)

    resizer = FramelessResizer(window)
    QApplication.instance().installEventFilter(resizer)
    return title_bar, resizer
