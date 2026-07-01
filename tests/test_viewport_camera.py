"""ViewportCamera is pure math/state — no Qt required. These cover the
working-scale envelope and framing floors that used to be hardcoded meter
constants scattered across the widget."""
from __future__ import annotations

import numpy as np

from app.viewport.camera import DEFAULT_VIEW_DISTANCE, ViewportCamera
from core.sdf.base import BoundingBox3D


def test_zoom_envelope_widens_but_never_shrinks() -> None:
    camera = ViewportCamera()
    assert camera.zoom_limits() == (0.5, 200.0)

    camera.view_scale = 0.001  # mm work
    minimum, maximum = camera.zoom_limits()
    assert minimum == 0.5 * 0.001  # close enough to see a mm grid
    assert maximum == 200.0  # meter-scale scenes stay reachable

    camera.view_scale = 1000.0  # km work
    minimum, maximum = camera.zoom_limits()
    assert minimum == 0.5
    assert maximum == 200.0 * 1000.0


def test_zoom_by_clamps_to_envelope() -> None:
    camera = ViewportCamera()
    for _ in range(200):
        camera.zoom_by(1000.0)
    assert camera.distance == camera.zoom_limits()[0]
    for _ in range(200):
        camera.zoom_by(-1000.0)
    assert camera.distance == camera.zoom_limits()[1]


def test_frame_box_floor_scales_down_for_small_parts() -> None:
    camera = ViewportCamera()
    part = BoundingBox3D(0.0, 0.002, 0.0, 0.002, 0.0, 0.002)

    camera.frame_box(part)
    assert camera.distance == 1.0  # meter floor dwarfs a 2 mm part

    camera.view_scale = 0.001
    camera.frame_box(part)
    assert camera.distance == 0.002 * 1.6


def test_reframe_to_working_scale_keeps_target() -> None:
    camera = ViewportCamera()
    camera.target = np.array([5.0, 3.0, 0.0])
    camera.view_scale = 0.001
    camera.reframe_to_working_scale()
    assert camera.distance == DEFAULT_VIEW_DISTANCE * 0.001
    assert tuple(camera.target) == (5.0, 3.0, 0.0)


def test_screen_ray_center_points_at_target() -> None:
    camera = ViewportCamera()
    origin, direction = camera.screen_ray(400.0, 300.0, 800.0, 600.0)
    to_target = camera.target - origin
    to_target = to_target / np.linalg.norm(to_target)
    assert float(np.dot(direction, to_target)) > 0.999
