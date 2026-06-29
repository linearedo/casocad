from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.panels.properties import CadDimensionSpinBox


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def test_dimension_spinbox_click_and_digits_replace_current_value() -> None:
    _app()
    spin = CadDimensionSpinBox()
    spin.setDecimals(5)
    spin.setRange(-1000.0, 1000.0)
    spin.setSuffix(" m")
    spin.setKeyboardTracking(False)
    spin.setValue(12.34)
    spin.show()

    QTest.mouseClick(spin.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClicks(spin, "56")
    QTest.keyClick(spin, Qt.Key.Key_Return)

    assert spin.value() == 56.0
