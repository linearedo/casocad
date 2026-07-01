"""New-primitive defaults scale with the working unit; committed SDFs never
rescale (the document always stores meters)."""
from __future__ import annotations

from core.scene import SceneDocument
from core.sdf import (
    Box,
    CircleProfile,
    PlacedSDF1D,
    PlacedSDF2D,
    PolylineTube,
    SegmentProfile,
    Sphere,
)

MILLIMETER = 0.001


def test_add_primitive_scales_3d_defaults() -> None:
    document = SceneDocument()
    sphere = document.node(document.add_primitive("sphere", scale=MILLIMETER))
    assert isinstance(sphere, Sphere)
    assert sphere.radius == 0.5 * MILLIMETER
    box = document.node(document.add_primitive("box", scale=MILLIMETER))
    assert isinstance(box, Box)
    assert box.half_size == (0.5 * MILLIMETER,) * 3
    tube = document.node(document.add_primitive("polyline_tube", scale=MILLIMETER))
    assert isinstance(tube, PolylineTube)
    assert tube.radius == 0.12 * MILLIMETER
    assert all(
        abs(component) <= MILLIMETER for point in tube.points for component in point
    )


def test_add_primitive_scales_2d_and_1d_defaults() -> None:
    document = SceneDocument()
    circle = document.node(document.add_primitive("circle", scale=MILLIMETER))
    assert isinstance(circle, PlacedSDF2D)
    assert isinstance(circle.profile, CircleProfile)
    assert circle.profile.radius == 0.5 * MILLIMETER
    segment = document.node(document.add_primitive("segment", scale=MILLIMETER))
    assert isinstance(segment, PlacedSDF1D)
    assert isinstance(segment.profile, SegmentProfile)
    assert segment.profile.half_length == 0.5 * MILLIMETER


def test_committed_nodes_keep_their_size_when_defaults_change() -> None:
    document = SceneDocument()
    committed = document.node(document.add_primitive("sphere"))
    assert isinstance(committed, Sphere)
    later = document.node(document.add_primitive("sphere", scale=MILLIMETER))
    assert isinstance(later, Sphere)
    assert committed.radius == 0.5
    assert later.radius == 0.5 * MILLIMETER


def test_drag_creation_minimum_sizes_scale_with_working_unit() -> None:
    document = SceneDocument()
    tiny = 0.0001  # a 0.1 mm drag: meter-scale floors would inflate it 500x
    handle = document.add_primitive_from_drag(
        "box", (0.0, 0.0, 0.0), (tiny, tiny, tiny), scale=MILLIMETER
    )
    box = document.node(handle)
    assert isinstance(box, Box)
    assert box.half_size == (0.5 * tiny,) * 3
    degenerate = document.add_primitive_from_drag(
        "sphere", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), scale=MILLIMETER
    )
    sphere = document.node(degenerate)
    assert isinstance(sphere, Sphere)
    assert sphere.radius == 0.05 * MILLIMETER
