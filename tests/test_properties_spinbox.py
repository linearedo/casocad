from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.dimensions import length_unit
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


def test_dimension_spinbox_value_stays_meters_in_working_unit() -> None:
    _app()
    spin = CadDimensionSpinBox()
    spin.setRange(0.0, 1.0e9)
    spin.setKeyboardTracking(False)
    spin.set_unit(length_unit("mm"))
    spin.setValue(0.5)

    assert spin.text() == "500 mm"

    spin.show()
    QTest.mouseClick(spin.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClicks(spin, "20")
    QTest.keyClick(spin, Qt.Key.Key_Return)
    assert spin.value() == 0.02

    QTest.mouseClick(spin.lineEdit(), Qt.MouseButton.LeftButton)
    QTest.keyClicks(spin, "3cm")
    QTest.keyClick(spin, Qt.Key.Key_Return)
    assert spin.value() == 0.03


def test_dimension_spinbox_switching_unit_keeps_value() -> None:
    _app()
    spin = CadDimensionSpinBox()
    spin.setRange(0.0, 1.0e9)
    spin.setValue(1.25)
    assert spin.text() == "1.25 m"

    spin.set_unit(length_unit("cm"))
    assert spin.value() == 1.25
    assert spin.text() == "125 cm"
