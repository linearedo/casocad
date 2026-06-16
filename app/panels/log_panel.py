from __future__ import annotations

import logging

from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QPlainTextEdit, QVBoxLayout, QWidget


class _LogEmitter(QObject):
    message = Signal(str)


class _QtLogHandler(logging.Handler):
    def __init__(self, emitter: _LogEmitter) -> None:
        super().__init__()
        self._emitter = emitter
        self.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )

    def emit(self, record: logging.LogRecord) -> None:
        self._emitter.message.emit(self.format(record))


class LogPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        self.output = QPlainTextEdit()
        self.output.setReadOnly(True)
        self.output.setMaximumBlockCount(2_000)
        layout.addWidget(self.output)
        self._emitter = _LogEmitter(self)
        self._emitter.message.connect(self.output.appendPlainText)
        self._handler = _QtLogHandler(self._emitter)
        logging.getLogger().addHandler(self._handler)

    def closeEvent(self, event: object) -> None:
        logging.getLogger().removeHandler(self._handler)
        super().closeEvent(event)
