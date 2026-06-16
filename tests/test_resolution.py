from __future__ import annotations

import numpy as np

from core.mesher import minimum_feature_size, recommended_max_dx
from core.sdf import (
    Box,
    Cylinder,
    Difference,
    PlacedSDF2D,
    RectangleProfile,
    Scale,
    Sphere,
    Torus,
)


def test_recommended_dx_uses_smallest_csg_feature() -> None:
    box = Box(
        name="box",
        object_id=1,
        half_size=(1.0, 0.5, 0.4),
    )
    cylinder = Cylinder(
        name="cylinder",
        object_id=2,
        radius=0.18,
        half_height=0.4,
    )
    root = Difference(
        name="cut",
        object_id=3,
        left=box,
        right=cylinder,
    )

    assert np.isclose(minimum_feature_size(root), 0.36)
    assert np.isclose(recommended_max_dx(root), 0.06)


def test_recommended_dx_tracks_scale_and_torus_tube() -> None:
    sphere = Scale(
        name="scaled",
        object_id=2,
        factor=0.5,
        child=Sphere(name="sphere", object_id=1, radius=0.6),
    )
    torus = Torus(
        name="torus",
        object_id=3,
        major_radius=1.0,
        minor_radius=0.12,
    )

    assert np.isclose(minimum_feature_size(sphere), 0.6)
    assert np.isclose(recommended_max_dx(torus), 0.04)


def test_recommended_dx_uses_2d_profile_size_not_render_thickness() -> None:
    rectangle = PlacedSDF2D(
        name="rectangle",
        object_id=1,
        profile=RectangleProfile(half_size=(1.0, 0.3)),
    )

    assert np.isclose(minimum_feature_size(rectangle), 0.6)
    assert np.isclose(recommended_max_dx(rectangle), 0.1)
