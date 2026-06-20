from __future__ import annotations

import numpy as np

from app.viewport.viewport_widget import pick_2d_scene_object_from_ray
from core.scene import SceneDocument


def test_viewport_pick_selects_filled_2d_component() -> None:
    document = SceneDocument()
    rectangle_handle = document.add_primitive("rectangle")
    rectangle = document.node(rectangle_handle)
    _version, tree = document.visual_snapshot()

    hit = pick_2d_scene_object_from_ray(
        tree,
        np.asarray((0.0, 0.0, 3.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
        lambda _travel: 0.02,
    )

    assert hit is not None
    assert hit[0] == rectangle.object_id


def test_viewport_pick_rejects_missed_2d_component() -> None:
    document = SceneDocument()
    document.add_primitive("rectangle")
    _version, tree = document.visual_snapshot()

    hit = pick_2d_scene_object_from_ray(
        tree,
        np.asarray((2.0, 2.0, 3.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
        lambda _travel: 0.02,
    )

    assert hit is None


def test_viewport_pick_selects_2d_curve_with_tolerance() -> None:
    document = SceneDocument()
    polyline_handle = document.add_primitive("polyline")
    polyline = document.node(polyline_handle)
    _version, tree = document.visual_snapshot()

    hit = pick_2d_scene_object_from_ray(
        tree,
        np.asarray((0.0, -0.4, 3.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
        lambda _travel: 0.05,
    )

    assert hit is not None
    assert hit[0] == polyline.object_id
