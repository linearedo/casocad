from __future__ import annotations

import numpy as np
import pytest

from app.viewport.renderers.qrhi.viewport import _revolve_signal_frame
from app.viewport.surface_builder import (
    ViewportSurfaceCache,
    build_viewport_surface_scene,
)
from core.scene import SceneDocument
from core.sdf import Extrude, PlacedSDF2D, Revolve


def _visual_components(document: SceneDocument):
    _version, tree = document.visual_snapshot()
    return tree.components


def test_extrude_replaces_top_level_2d_source_in_visual_scene() -> None:
    document = SceneDocument()
    source_handle = document.add_primitive_from_drag(
        "rectangle",
        (-0.5, -0.25, 0.0),
        (0.5, 0.25, 0.0),
    )
    source = document.node(source_handle)

    solid_handle = document.solid_from_2d(
        [source_handle],
        "extrude",
        signed_height=0.75,
    )
    solid = document.node(solid_handle)

    assert isinstance(source, PlacedSDF2D)
    assert isinstance(solid, Extrude)
    assert document.objects == [solid]
    assert solid.section is source
    assert document.node(source_handle) is source
    assert _visual_components(document) == (solid,)


def test_revolve_replaces_top_level_2d_source_in_visual_scene() -> None:
    document = SceneDocument()
    source_handle = document.add_primitive_from_drag(
        "circle",
        (-0.25, 0.0, 0.0),
        (0.25, 0.0, 0.0),
    )
    source = document.node(source_handle)

    solid_handle = document.solid_from_2d(
        [source_handle],
        "revolve",
        revolve_axis_origin=(0.0, 0.0, 0.0),
        revolve_axis_direction=(0.0, 1.0, 0.0),
        revolve_radial_direction=(1.0, 0.0, 0.0),
    )
    solid = document.node(solid_handle)

    assert isinstance(source, PlacedSDF2D)
    assert isinstance(solid, Revolve)
    assert document.objects == [solid]
    assert solid.section is source
    assert document.node(source_handle) is source
    assert _visual_components(document) == (solid,)


def test_revolve_can_use_either_profile_axis() -> None:
    document_u = SceneDocument()
    handle_u = document_u.add_primitive_from_drag(
        "rectangle",
        (-0.5, -0.25, 0.0),
        (0.5, 0.25, 0.0),
    )
    solid_u = document_u.node(
        document_u.solid_from_2d([handle_u], "revolve", revolve_axis="u")
    )

    document_v = SceneDocument()
    handle_v = document_v.add_primitive_from_drag(
        "rectangle",
        (-0.5, -0.25, 0.0),
        (0.5, 0.25, 0.0),
    )
    solid_v = document_v.node(
        document_v.solid_from_2d([handle_v], "revolve", revolve_axis="v")
    )

    assert isinstance(solid_u, Revolve)
    assert isinstance(solid_v, Revolve)
    assert solid_u.axis == "u"
    assert solid_v.axis == "v"
    _origin_u, axis_u, radial_u, _tangent_u = solid_u._axis_frame()
    _origin_v, axis_v, radial_v, _tangent_v = solid_v._axis_frame()
    np.testing.assert_allclose(axis_u, (1.0, 0.0, 0.0))
    np.testing.assert_allclose(radial_u, (0.0, 1.0, 0.0))
    np.testing.assert_allclose(axis_v, (0.0, 1.0, 0.0))
    np.testing.assert_allclose(radial_v, (1.0, 0.0, 0.0))


def test_repeat_revolve_after_snapshot_restore_keeps_single_visual_source() -> None:
    document = SceneDocument()
    source_handle = document.add_primitive_from_drag(
        "circle",
        (-0.25, 0.0, 0.0),
        (0.25, 0.0, 0.0),
    )
    snapshot = document.snapshot()

    first_handle = document.solid_from_2d([source_handle], "revolve")
    first = document.node(first_handle)
    assert isinstance(first, Revolve)
    assert document.objects == [first]
    assert _visual_components(document) == (first,)

    restored = snapshot.snapshot()
    second_handle = restored.solid_from_2d(
        [source_handle],
        "revolve",
        revolve_axis="u",
    )
    second = restored.node(second_handle)
    assert isinstance(second, Revolve)
    assert restored.objects == [second]
    assert second.axis == "u"
    assert _visual_components(restored) == (second,)


def test_repeat_revolve_after_undo_keeps_stable_surface_cache_contract() -> None:
    document = SceneDocument()
    source_handle = document.add_primitive_from_drag(
        "circle",
        (-0.25, 0.0, 0.0),
        (0.25, 0.0, 0.0),
    )
    undo_snapshot = document.snapshot()
    cache = ViewportSurfaceCache(resolution=16)
    object_ids = []
    cache_keys = []
    triangle_counts = []
    vertex_counts = []

    for index, axis in enumerate(("v", "u", "v"), start=1):
        rebuilt = undo_snapshot.snapshot()
        solid_handle = rebuilt.solid_from_2d(
            [source_handle],
            "revolve",
            revolve_axis=axis,
        )
        rebuilt.version = 100 + index
        solid = rebuilt.node(solid_handle)
        assert isinstance(solid, Revolve)
        assert rebuilt.objects == [solid]
        assert _visual_components(rebuilt) == (solid,)

        version, tree = rebuilt.visual_snapshot()
        surface_scene = build_viewport_surface_scene(tree, version, cache=cache)
        assert surface_scene is not None
        ready = [surface for surface in surface_scene.surfaces if surface.status == "ready"]
        assert ready
        assert all(surface.indices.size > 0 for surface in ready)
        object_ids.append(solid.object_id)
        cache_keys.extend(surface.key for surface in ready)
        triangle_counts.append(surface_scene.triangle_count)
        vertex_counts.append(surface_scene.vertex_count)

    assert object_ids[0] == object_ids[1] == object_ids[2]
    assert [key.scene_revision for key in cache_keys] == [101, 102, 103]
    assert triangle_counts[0] == triangle_counts[2]
    assert vertex_counts[0] == vertex_counts[2]


def test_negative_revolve_drag_keeps_profile_radial_axis_fixed() -> None:
    frame = _revolve_signal_frame((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), -90.0)

    assert frame is not None
    radial_direction, angle_degrees = frame
    np.testing.assert_allclose(radial_direction, (1.0, 0.0, 0.0))
    assert angle_degrees == -90.0


def test_asymmetric_bezier_revolve_after_undo_is_rejected_before_surface_build() -> None:
    document = SceneDocument()
    source_handle = document.add_primitive("quadratic_bezier_surface")
    undo_snapshot = document.snapshot()
    solid_handle = document.solid_from_2d(
        [source_handle],
        "extrude",
        signed_height=0.5,
    )
    assert isinstance(document.node(solid_handle), Extrude)

    restored = undo_snapshot.snapshot()
    with pytest.raises(ValueError, match="non-symmetric profile crosses"):
        restored.solid_from_2d(
            [source_handle],
            "revolve",
            revolve_axis="u",
        )
