"""Boundary-region membership classifier (design_docs/boundary_region_v2.md §2).

Fixture is the default von Karman scene: root = Difference(flow box, cylinder
obstacle), box half_size (1.6, 0.7, 0.45), cylinder radius 0.24 through Z.
"""
from __future__ import annotations

import numpy as np

from core.boundary import BoundaryCut, BoundaryRegion
from core.boundary_region import (
    boundary_region_mask,
    owner_active_mask,
    region_tolerance,
)
from core.scene import SceneDocument
from core.sdf import Box, Cylinder, Sphere


def _default_scene():
    document = SceneDocument.default()
    root = document.fluid_domain.root
    box = next(n for _h, n, _p in document.walk() if isinstance(n, Box))
    cylinder = next(n for _h, n, _p in document.walk() if isinstance(n, Cylinder))
    return document, root, box, cylinder


def _face_points(x: float, count: int = 7) -> np.ndarray:
    ys = np.linspace(-0.6, 0.6, count)
    zs = np.linspace(-0.35, 0.35, count)
    grid = np.array([(x, y, z) for y in ys for z in zs])
    return grid


def _cylinder_wall_points(radius: float = 0.24, count: int = 24) -> np.ndarray:
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
    zs = np.linspace(-0.3, 0.3, 5)
    return np.array(
        [(radius * np.cos(a), radius * np.sin(a), z) for a in angles for z in zs]
    )


def _region(owner_id: int, **kwargs) -> BoundaryRegion:
    return BoundaryRegion(
        name="r", object_id=999, owner_object_id=owner_id, **kwargs
    )


def test_obstacle_owns_the_cut_surface() -> None:
    _doc, root, box, cylinder = _default_scene()
    wall = _cylinder_wall_points()
    tol = region_tolerance(root, _region(cylinder.object_id))

    assert owner_active_mask(root, cylinder.object_id, wall, tie_tolerance=tol).all()
    cylinder_region = _region(cylinder.object_id)
    box_region = _region(box.object_id)
    assert boundary_region_mask(root, cylinder_region, wall).all()
    assert not boundary_region_mask(root, box_region, wall).any()


def test_direction_region_selects_one_face() -> None:
    _doc, root, box, _cyl = _default_scene()
    inlet = _region(box.object_id, outside_direction=0)   # -X face
    minus_x = _face_points(-1.6)
    plus_x = _face_points(1.6)
    wall = _cylinder_wall_points()

    assert boundary_region_mask(root, inlet, minus_x).all()
    assert not boundary_region_mask(root, inlet, plus_x).any()
    assert not boundary_region_mask(root, inlet, wall).any()


def test_one_cut_partitions_the_parent_exactly() -> None:
    _doc, root, box, _cyl = _default_scene()
    samples = np.concatenate(
        (_face_points(-1.6), _face_points(1.6), _cylinder_wall_points())
    )
    parent = _region(box.object_id)
    ghost = Sphere(name="ghost", object_id=0, center=(-1.6, 0.0, 0.0), radius=0.5)
    inside = _region(box.object_id, cuts=(BoundaryCut("inside", ghost),))
    outside = _region(box.object_id, cuts=(BoundaryCut("outside", ghost),))

    parent_mask = boundary_region_mask(root, parent, samples)
    inside_mask = boundary_region_mask(root, inside, samples)
    outside_mask = boundary_region_mask(root, outside, samples)

    assert parent_mask.any() and inside_mask.any() and outside_mask.any()
    assert not (inside_mask & outside_mask).any()          # no double-tagging
    assert ((inside_mask | outside_mask) == parent_mask).all()  # no gaps
    # the inside child hugs the sphere: all its points near the face center
    picked = samples[inside_mask]
    assert (np.linalg.norm(picked - np.array([-1.6, 0.0, 0.0]), axis=1) < 0.55).all()


def test_cut_chain_is_a_conjunction() -> None:
    _doc, root, box, _cyl = _default_scene()
    samples = _face_points(-1.6, count=13)
    sphere = Sphere(name="g1", object_id=0, center=(-1.6, 0.0, 0.0), radius=0.4)
    upper = Box(
        name="g2", object_id=0, center=(-1.6, 1.0, 0.0), half_size=(1.0, 1.0, 1.0)
    )
    chained = _region(
        box.object_id,
        cuts=(BoundaryCut("outside", sphere), BoundaryCut("inside", upper)),
    )

    mask = boundary_region_mask(root, chained, samples)
    ring = ~boundary_region_mask(
        root, _region(box.object_id, cuts=(BoundaryCut("inside", sphere),)), samples
    )
    parent = boundary_region_mask(root, _region(box.object_id), samples)
    expected = parent & ring & (samples[:, 1] >= 0.0)

    assert mask.any()
    assert (mask == expected).all()


def test_lower_dimensional_ghost_extrudes_through_the_scene() -> None:
    document, root, box, _cyl = _default_scene()
    handle = document.add_primitive_from_drag(
        "segment", (-1.6, -0.2, 0.0), (-1.6, 0.2, 0.0)
    )
    segment = document.node(handle)
    region = _region(box.object_id, cuts=(BoundaryCut("inside", segment),))
    samples = _face_points(-1.6, count=13)

    mask = boundary_region_mask(root, region, samples)

    assert mask.any()
    assert (np.abs(samples[mask][:, 1]) <= 0.25).all()
    assert not mask[np.abs(samples[:, 1]) > 0.3].any()


def test_tolerance_scales_with_owner_size() -> None:
    scale = 0.001  # a mm-scale scene
    box = Box(name="b", object_id=1, half_size=(1.6 * scale, 0.7 * scale, 0.45 * scale))
    region = _region(box.object_id)
    face = np.array([[-1.6 * scale, 0.0, 0.0], [-1.6 * scale, 0.0002, 0.0001]])
    off_surface = np.array([[-1.55 * scale, 0.0, 0.0]])

    assert boundary_region_mask(box, region, face).all()
    assert not boundary_region_mask(box, region, off_surface).any()


def test_generic_leaf_patches_make_pyramid_pickable() -> None:
    """boundary_region_v2 §7: every 3D leaf owns at least its whole surface,
    so primitives without hand-written patch tables (Pyramid & co.) are
    hover-selectable."""
    from core.boundary_patches import pick_boundary_patch, surface_patches_for_root
    from core.sdf import Pyramid

    pyramid = Pyramid(
        name="p", object_id=7, base_half_size=0.45, half_height=0.6
    )
    patches = surface_patches_for_root(pyramid)
    assert len(patches) == 1
    assert patches[0].owner_object_id == 7
    assert patches[0].patch_id == "surface"

    hit = pick_boundary_patch(
        pyramid,
        np.array([0.1, 0.05, 3.0]),
        np.array([0.0, 0.0, -1.0]),
    )
    assert hit is not None
    assert hit.owner_object_id == 7
    assert hit.patch_id == "surface"
