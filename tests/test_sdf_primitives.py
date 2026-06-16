from __future__ import annotations

import numpy as np

from core.sdf import Box, Cylinder, Sphere, Torus


def point(x: float, y: float, z: float) -> tuple[np.ndarray, ...]:
    return (
        np.asarray([x], dtype=np.float64),
        np.asarray([y], dtype=np.float64),
        np.asarray([z], dtype=np.float64),
    )


def test_sphere_dual_contract() -> None:
    sphere = Sphere(name="sphere", radius=2.0)
    assert sphere.to_numpy(*point(0.0, 0.0, 0.0))[0] == -2.0
    assert sphere.to_numpy(*point(2.0, 0.0, 0.0))[0] == 0.0
    assert "length(" in sphere.to_glsl()


def test_box_distance() -> None:
    box = Box(name="box", half_size=(1.0, 2.0, 3.0))
    assert box.to_numpy(*point(0.0, 0.0, 0.0))[0] == -1.0
    assert box.to_numpy(*point(2.0, 2.0, 3.0))[0] == 1.0
    assert "max(" in box.to_glsl()


def test_cylinder_and_torus_return_float64_arrays() -> None:
    cylinder = Cylinder(name="cylinder", radius=1.0, half_height=2.0)
    torus = Torus(name="torus", major_radius=2.0, minor_radius=0.5)
    assert cylinder.to_numpy(*point(0.0, 0.0, 0.0)).dtype == np.float64
    assert torus.to_numpy(*point(2.0, 0.0, 0.0))[0] == -0.5
