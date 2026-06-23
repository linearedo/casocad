from __future__ import annotations

"""Exactness regression guard (spec §11, step 8).

The whole migration exists to keep the **interior** distance field exact:
``f(p) = -d(p, boundary)`` for every point inside a Domain. This test pins that
down on analytic cases by comparing the kernel's field (``to_numpy``) against the
true Euclidean distance computed **independently** from geometry. If a future
change makes the box/sphere primitives or the boolean operators non-exact inside
(e.g. a cheap bound, or a smooth blend creeping back), this fails.
"""

import numpy as np

from core.sdf import Box, Difference, Intersection, Sphere

# Geometry: a sphere small and centred inside a larger box, so the nearest point
# on the full box boundary and the nearest point on the full sphere both lie on
# the resulting boundary -> d(p, boundary) = min(box_dist, sphere_dist) exactly.
_HALF = 1.0
_RADIUS = 0.3
_EPS = 1e-9


def _interior_grid(n: int = 40) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coords = np.linspace(-0.97 * _HALF, 0.97 * _HALF, n)
    return np.meshgrid(coords, coords, coords, indexing="ij")


def _box_interior_distance(x, y, z):
    # True distance from an interior point to an axis-aligned box boundary.
    return np.minimum.reduce(
        [_HALF - np.abs(x), _HALF - np.abs(y), _HALF - np.abs(z)]
    )


def _radius(x, y, z):
    return np.sqrt(x * x + y * y + z * z)


def test_box_minus_sphere_interior_is_exact_distance() -> None:
    box = Box(name="box", half_size=(_HALF, _HALF, _HALF))
    sphere = Sphere(name="hole", radius=_RADIUS)
    fluid = Difference(name="F", left=box, right=sphere)

    x, y, z = _interior_grid()
    field = fluid.to_numpy(x, y, z)

    # Independent ground truth: inside F, distance to leave is the nearer of the
    # box wall or the sphere wall.
    box_dist = _box_interior_distance(x, y, z)
    sphere_dist = _radius(x, y, z) - _RADIUS  # >0 outside the sphere
    true_distance = np.minimum(box_dist, sphere_dist)

    interior = field < 0.0
    assert interior.any()
    error = np.abs(field[interior] - (-true_distance[interior]))
    assert float(error.max()) < _EPS


def test_box_intersect_sphere_interior_is_exact_distance() -> None:
    # Intersection interior exactness (the §4 union-of-complements result).
    box = Box(name="box", half_size=(_HALF, _HALF, _HALF))
    sphere = Sphere(name="ball", radius=0.8)
    region = Intersection(name="I", left=box, right=sphere)

    x, y, z = _interior_grid()
    field = region.to_numpy(x, y, z)

    box_dist = _box_interior_distance(x, y, z)
    sphere_interior_dist = 0.8 - _radius(x, y, z)  # >0 inside the sphere
    true_distance = np.minimum(box_dist, sphere_interior_dist)

    interior = field < 0.0
    assert interior.any()
    error = np.abs(field[interior] - (-true_distance[interior]))
    assert float(error.max()) < _EPS


def test_difference_interior_never_underestimates_wall_distance() -> None:
    # A weaker, always-true invariant kept as a guard even outside the analytic
    # sweet spot: the field magnitude inside must not exceed the true distance
    # (an exact field equals it; a broken one would over- or under-shoot).
    box = Box(name="box", half_size=(_HALF, _HALF, _HALF))
    sphere = Sphere(name="hole", radius=_RADIUS)
    fluid = Difference(name="F", left=box, right=sphere)

    x, y, z = _interior_grid()
    field = fluid.to_numpy(x, y, z)
    box_dist = _box_interior_distance(x, y, z)
    sphere_dist = _radius(x, y, z) - _RADIUS
    true_distance = np.minimum(box_dist, sphere_dist)

    interior = field < 0.0
    # |f| <= true distance everywhere inside (equality when exact).
    assert float((-field[interior] - true_distance[interior]).max()) < _EPS
