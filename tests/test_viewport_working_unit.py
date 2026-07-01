"""Unit switch = full workspace rescale: the grid stays a truthful
world-space snap reference; the camera envelope and grid spacing adapt."""
from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.dimensions import length_unit
from app.viewport.renderers.qrhi.viewport import QRhiViewportWidget
from core.sdf.base import BoundingBox3D


def _app() -> QApplication:
    return QApplication.instance() or QApplication([])


def _viewport() -> QRhiViewportWidget:
    _app()
    return QRhiViewportWidget()


def test_unit_switch_snaps_grid_and_reframes_camera() -> None:
    viewport = _viewport()
    assert viewport.grid_spacing == 1.0
    assert viewport._distance == 6.0

    viewport._set_working_unit(length_unit("mm"))
    assert viewport.grid_spacing == 0.001
    assert viewport._default_grid_spacing == 0.001
    assert viewport._distance == 0.006

    viewport._set_working_unit(length_unit("km"))
    assert viewport.grid_spacing == 1000.0
    assert viewport._distance == 6000.0


def test_zoom_envelope_widens_but_never_shrinks() -> None:
    viewport = _viewport()
    assert viewport._zoom_limits() == (0.5, 200.0)

    viewport._set_working_unit(length_unit("mm"))
    minimum, maximum = viewport._zoom_limits()
    assert minimum == 0.5 * 0.001  # close enough to see a mm grid
    assert maximum == 200.0  # meter-scale scenes stay reachable

    viewport._set_working_unit(length_unit("km"))
    minimum, maximum = viewport._zoom_limits()
    assert minimum == 0.5
    assert maximum == 200.0 * 1000.0


def test_frame_box_floor_scales_down_for_small_parts() -> None:
    viewport = _viewport()
    part = BoundingBox3D(0.0, 0.002, 0.0, 0.002, 0.0, 0.002)

    viewport.frame_box(part)
    assert viewport._distance == 1.0  # meter floor dwarfs a 2 mm part

    viewport._set_working_unit(length_unit("mm"))
    viewport.frame_box(part)
    assert viewport._distance == 0.002 * 1.6
