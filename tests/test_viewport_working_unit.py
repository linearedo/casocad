"""Unit switch = full workspace rescale: the grid stays a truthful
world-space snap reference; the camera envelope and grid spacing adapt."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QRect
from PySide6.QtGui import QPaintEvent
from PySide6.QtWidgets import QApplication

from app.dimensions import length_unit
from app.viewport.renderers.qrhi.viewport import QRhiViewportWidget


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _viewport() -> QRhiViewportWidget:
    _app()
    return QRhiViewportWidget()


def test_unit_switch_snaps_grid_and_reframes_camera() -> None:
    viewport = _viewport()
    assert viewport.grid_spacing == 1.0
    assert viewport._camera.distance == 6.0

    viewport._set_working_unit(length_unit("mm"))
    assert viewport.grid_spacing == 0.001
    assert viewport._default_grid_spacing == 0.001
    assert viewport._camera.distance == 0.006

    viewport._set_working_unit(length_unit("km"))
    assert viewport.grid_spacing == 1000.0
    assert viewport._camera.distance == 6000.0


def test_orientation_overlay_paints_from_camera_state() -> None:
    """Regression: Qt swallows exceptions raised inside paintEvent overrides,
    so a stale attribute there crash-loops the real app while unit tests of
    the projection math stay green. Calling paintEvent directly propagates."""
    viewport = _viewport()
    overlay = viewport._orientation_widget
    assert overlay is not None
    overlay.paintEvent(QPaintEvent(QRect(0, 0, overlay.width(), overlay.height())))
