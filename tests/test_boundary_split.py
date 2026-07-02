"""SceneDocument.split_boundary_region (boundary_region_v2 §2 Phase 2):
children partition and replace the parent, ghosts never enter the scene
graph, legacy selector regions convert into the chain, empty sides warn."""
from __future__ import annotations

import numpy as np

from core.boundary import BoundaryRegion
from core.boundary_region import boundary_region_mask
from core.scene import SceneDocument
from core.sdf import Box, Sphere


def _scene_with_whole_surface_region():
    document = SceneDocument.default()
    box = next(n for _h, n, _p in document.walk() if isinstance(n, Box))
    handle = document.add_boundary_region(box.object_id)
    region = document.node(handle)
    assert isinstance(region, BoundaryRegion)
    return document, box, region


def _face_samples(x: float, count: int = 9) -> np.ndarray:
    ys = np.linspace(-0.6, 0.6, count)
    zs = np.linspace(-0.35, 0.35, count)
    return np.array([(x, y, z) for y in ys for z in zs])


def test_split_replaces_parent_and_keeps_ghost_out_of_scene() -> None:
    document, _box, region = _scene_with_whole_surface_region()
    objects_before = list(document.objects)
    ghost = Sphere(name="knife", object_id=0, center=(-1.6, 0.0, 0.0), radius=0.5)

    (inside_handle, outside_handle), _empty = document.split_boundary_region(
        region, ghost
    )

    assert region not in document.boundary_regions
    inside = document.node(inside_handle)
    outside = document.node(outside_handle)
    assert {inside.cuts[-1].side, outside.cuts[-1].side} == {"inside", "outside"}
    assert len(inside.cuts) == 1 and len(outside.cuts) == 1
    assert document.objects == objects_before          # ghost never became a node
    assert region not in document.fluid_domain.tag_objects
    assert inside in document.fluid_domain.tag_objects
    assert outside in document.fluid_domain.tag_objects


def test_nested_split_composes_chains_and_partitions() -> None:
    document, box, region = _scene_with_whole_surface_region()
    root = document.fluid_domain.root
    sphere = Sphere(name="k1", object_id=0, center=(-1.6, 0.0, 0.0), radius=0.5)
    (inside_handle, outside_handle), _ = document.split_boundary_region(region, sphere)
    outside = document.node(outside_handle)

    upper = Box(name="k2", object_id=0, center=(0.0, 1.0, 0.0), half_size=(4.0, 1.0, 1.0))
    (top_handle, bottom_handle), _ = document.split_boundary_region(outside, upper)
    top = document.node(top_handle)
    bottom = document.node(bottom_handle)

    assert len(top.cuts) == 2 and len(bottom.cuts) == 2
    samples = _face_samples(-1.6)
    inside_mask = boundary_region_mask(root, document.node(inside_handle), samples)
    top_mask = boundary_region_mask(root, top, samples)
    bottom_mask = boundary_region_mask(root, bottom, samples)
    # the three leaves partition the face samples exactly
    total = inside_mask.astype(int) + top_mask.astype(int) + bottom_mask.astype(int)
    assert (total == 1).all()


def test_empty_side_is_reported_not_forbidden() -> None:
    document, _box, region = _scene_with_whole_surface_region()
    faraway = Sphere(name="k", object_id=0, center=(50.0, 50.0, 50.0), radius=0.1)

    (_handles), empty = document.split_boundary_region(region, faraway)

    assert empty == ("inside",)


def test_legacy_selector_region_converts_into_chain() -> None:
    document = SceneDocument.default()
    box = next(n for _h, n, _p in document.walk() if isinstance(n, Box))
    handle = document.add_boundary_region(box.object_id, patch_id="-X")
    region = document.node(handle)
    selector_handle = document.add_primitive("sphere")
    selector = document.node(selector_handle)
    selector.center = (-1.6, 0.0, 0.0)
    legacy_handles = document.add_boundary_selector_split_regions(region, selector)
    legacy_inside = document.node(legacy_handles[0])
    assert legacy_inside.selector_id is not None and not legacy_inside.cuts

    knife = Box(name="k2", object_id=0, center=(0.0, 1.0, 0.0), half_size=(4.0, 1.0, 1.0))
    (top_handle, _bottom_handle), _ = document.split_boundary_region(
        legacy_inside, knife
    )
    top = document.node(top_handle)

    assert len(top.cuts) == 2               # converted legacy cut + the new one
    assert top.cuts[0].ghost.name == selector.name
    assert top.selector_id is None or top.cuts[0].side == legacy_inside.selector_side
