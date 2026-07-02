"""Smooth-polyline knife: shortest on-boundary paths + the NormalCurtain
classification field (boundary cutter, surface-following knife)."""
from __future__ import annotations

import numpy as np
import pytest

from core.boundary_paths import (
    smooth_polyline_knife,
    surface_shortest_path,
)
from core.boundary_region import boundary_region_mask, sample_boundary_points
from core.scene import SceneDocument
from core.sdf import NormalCurtain
from core.serialization import _ghost_from_record, _ghost_to_record


def _root():
    return SceneDocument.default().fluid_domain.root


def _surface_error(root, path):
    return float(
        np.abs(
            np.asarray(root.to_numpy(path[:, 0], path[:, 1], path[:, 2]))
        ).max()
    )


def test_shortest_path_on_flat_face_is_straight() -> None:
    root = _root()
    start, end = (-1.6, -0.5, -0.3), (-1.6, 0.5, 0.3)
    path = surface_shortest_path(root, start, end)
    assert _surface_error(root, path) < 1.0e-6
    length = float(np.linalg.norm(path[1:] - path[:-1], axis=1).sum())
    chord = float(np.linalg.norm(np.subtract(end, start)))
    assert length == pytest.approx(chord, rel=1.0e-3)
    np.testing.assert_allclose(path[0], start)
    np.testing.assert_allclose(path[-1], end)


def test_shortest_path_follows_cylinder_wall() -> None:
    root = _root()
    radius = 0.24  # default-scene obstacle wall
    start = (radius * np.cos(np.pi * 1.25), radius * np.sin(np.pi * 1.25), -0.2)
    end = (radius * np.cos(np.pi * 1.75), radius * np.sin(np.pi * 1.75), 0.25)
    path = surface_shortest_path(root, start, end)
    assert _surface_error(root, path) < 1.0e-6
    radii = np.sqrt(path[:, 0] ** 2 + path[:, 1] ** 2)
    np.testing.assert_allclose(radii, radius, atol=1.0e-6)
    length = float(np.linalg.norm(path[1:] - path[:-1], axis=1).sum())
    ideal_helix = float(np.hypot(radius * np.pi * 0.5, 0.45))
    assert length == pytest.approx(ideal_helix, rel=2.0e-2)


def test_shortest_path_crosses_box_edge_on_surface() -> None:
    root = _root()
    path = surface_shortest_path(root, (-1.6, -0.3, 0.1), (-1.2, -0.7, 0.1))
    assert _surface_error(root, path) < 1.0e-6


def test_normal_curtain_signs_split_sides() -> None:
    curtain = NormalCurtain(
        name="k",
        object_id=0,
        points=((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        normals=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
        extent=4.0,
    )
    values = curtain.to_numpy(
        np.asarray([0.0, 0.0]), np.asarray([0.5, -0.5]), np.asarray([0.0, 0.0])
    )
    assert values[0] * values[1] < 0.0
    assert np.abs(values) == pytest.approx((0.5, 0.5))


def test_smooth_polyline_knife_splits_curved_region() -> None:
    document = SceneDocument.default()
    root = document.fluid_domain.root
    from core.boundary_patches import pick_boundary_patch

    hit = pick_boundary_patch(
        root, np.array([0.0, -5.0, 0.0]), np.array([0.0, 1.0, 0.0])
    )
    assert hit is not None
    region = document.node(document.add_boundary_region_from_hit(hit))
    radius = 0.24
    knife = smooth_polyline_knife(
        root,
        (
            (radius * np.cos(np.pi * 1.2), radius * np.sin(np.pi * 1.2), -0.3),
            (radius * np.cos(np.pi * 1.8), radius * np.sin(np.pi * 1.8), 0.3),
        ),
    )
    handles = document.split_boundary_region(region, knife)
    samples, band = sample_boundary_points(root, resolution=48)
    for handle in handles:
        child = document.node(handle)
        mask = boundary_region_mask(root, child, samples, tolerance=band)
        assert mask.any()


def test_normal_curtain_ghost_serialization_roundtrip() -> None:
    root = _root()
    knife = smooth_polyline_knife(
        root, ((-1.6, -0.4, -0.2), (-1.6, 0.4, 0.2))
    )
    record = _ghost_to_record(knife)
    restored = _ghost_from_record(record)
    assert isinstance(restored, NormalCurtain)
    assert restored.points == knife.points
    assert restored.normals == knife.normals
    assert restored.extent == knife.extent
    probe = np.asarray([(-1.6, -0.5, 0.3), (-1.6, 0.5, -0.3)])
    original = knife.to_numpy(probe[:, 0], probe[:, 1], probe[:, 2])
    loaded = restored.to_numpy(probe[:, 0], probe[:, 1], probe[:, 2])
    np.testing.assert_allclose(loaded, original)


def test_smooth_polyline_knife_ignores_repeated_clicks() -> None:
    root = _root()
    point = (-1.6, -0.4, -0.2)
    knife = smooth_polyline_knife(
        root, (point, point, (-1.6, 0.4, 0.2))
    )
    assert len(knife.points) >= 2
    with pytest.raises(ValueError):
        smooth_polyline_knife(root, (point, point))
