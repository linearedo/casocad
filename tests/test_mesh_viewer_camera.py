"""Mesh-viewer camera floors scale with the framed mesh's own size, so a
mm-scale FEA/CFD mesh frames and zooms correctly instead of hitting the old
meter floors (2 m frame minimum, 1 cm zoom minimum)."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.meshing.viewer.widget import QRhiMeshViewerWidget


def _viewer() -> QRhiMeshViewerWidget:
    if QApplication.instance() is None:
        QApplication([])
    return QRhiMeshViewerWidget()


def test_framing_small_mesh_gets_close_instead_of_meter_floor() -> None:
    viewer = _viewer()
    viewer._frame_bounds((0.0, 0.0, 0.0), (0.005, 0.005, 0.005))  # ~8.7 mm diagonal
    assert viewer._distance < 0.02  # old floor parked the camera 2 m away
    assert viewer._zoom_limits()[0] < 0.001  # zoom can reach the mesh


def test_framing_meter_mesh_keeps_old_limits() -> None:
    viewer = _viewer()
    viewer._frame_bounds((0.0, 0.0, 0.0), (2.0, 1.0, 1.0))
    assert viewer._distance == 2.449489742783178 * 1.8  # diagonal * 1.8
    assert viewer._zoom_limits() == (0.01, 1.0e6)


def test_degenerate_bounds_keep_default_view() -> None:
    viewer = _viewer()
    viewer._frame_bounds((1.0, 1.0, 1.0), (1.0, 1.0, 1.0))
    assert viewer._distance == 2.0
    assert viewer._view_scale == 1.0
