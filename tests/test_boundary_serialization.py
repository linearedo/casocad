"""Boundary-region persistence (boundary_region_v2 §5 Phase 3): the new
self-contained cuts format round-trips, and legacy selector scenes migrate on
load with identical classification."""
from __future__ import annotations

import json

import numpy as np

from core.boundary import BoundaryRegion
from core.boundary_region import boundary_region_mask
from core.scene import SceneDocument
from core.sdf import Box, Sphere
from core.serialization import load_scene, save_scene


def _face_samples(x: float, count: int = 9) -> np.ndarray:
    ys = np.linspace(-0.6, 0.6, count)
    zs = np.linspace(-0.35, 0.35, count)
    return np.array([(x, y, z) for y in ys for z in zs])


def test_cut_chain_round_trips(tmp_path) -> None:
    document = SceneDocument.default()
    box = next(n for _h, n, _p in document.walk() if isinstance(n, Box))
    region = document.node(document.add_boundary_region(box.object_id))
    ghost = Sphere(name="knife", object_id=0, center=(-1.6, 0.0, 0.0), radius=0.5)
    (inside_handle, _), _ = document.split_boundary_region(region, ghost)
    inside = document.node(inside_handle)
    inside.tag = "inlet"

    path = tmp_path / "scene.json"
    save_scene(document, path)
    payload = json.loads(path.read_text())
    record = next(
        r for r in payload["boundary_regions"].values() if r.get("tag") == "inlet"
    )
    assert record["cuts"][0]["ghost"]["type"] == "sphere"

    loaded = load_scene(path)
    loaded_region = next(
        r for r in loaded.boundary_regions if r.tag == "inlet"
    )
    assert len(loaded_region.cuts) == 1
    assert loaded_region.cuts[0].side == "inside"
    samples = _face_samples(-1.6)
    original = boundary_region_mask(document.fluid_domain.root, inside, samples)
    reloaded = boundary_region_mask(loaded.fluid_domain.root, loaded_region, samples)
    assert (original == reloaded).all()
    assert original.any()


def test_in_session_legacy_selector_saves_as_cuts(tmp_path) -> None:
    document = SceneDocument.default()
    box = next(n for _h, n, _p in document.walk() if isinstance(n, Box))
    region = document.node(document.add_boundary_region(box.object_id, patch_id="-X"))
    selector_handle = document.add_primitive("sphere")
    selector = document.node(selector_handle)
    selector.center = (-1.6, 0.0, 0.0)
    selector.name = "__boundary_selector_test_sphere"
    document.add_boundary_selector_split_regions(region, selector)

    path = tmp_path / "scene.json"
    save_scene(document, path)
    payload = json.loads(path.read_text())
    selector_backed = [
        r for r in payload["boundary_regions"].values() if r.get("cuts")
    ]
    assert len(selector_backed) == 2      # inside + outside, both inlined
    for record in selector_backed:
        assert record["cuts"][0]["ghost"]["type"] == "sphere"
        assert "selector" not in record

    loaded = load_scene(path)
    # the hidden selector node is not resurrected as a scene object
    assert not any(
        loaded.is_internal_scene_node(node) for node in loaded.objects
    )
    assert loaded.fluid_domain.selector_objects == ()
    chained = [r for r in loaded.boundary_regions if r.cuts]
    assert len(chained) == 2
    assert all(r.selector_id is None for r in chained)


_LEGACY_SCENE = {
    "format": "casocad",
    "version": 1,
    "unit": "m",
    "root_objects": ["fluid", "__boundary_selector_knife"],
    "objects": {
        "flow": {"type": "box", "center": [0, 0, 0], "size": [3.2, 1.4, 0.9]},
        "hole": {
            "type": "cylinder",
            "center": [0, 0, 0],
            "radius": 0.24,
            "height": 1.1,
        },
        "fluid": {"type": "difference", "left": "flow", "right": "hole"},
        "__boundary_selector_knife": {
            "type": "sphere",
            "center": [-1.6, 0.0, 0.0],
            "radius": 0.5,
        },
    },
    "boundary_regions": {
        "inlet_spot": {
            "owner": "flow",
            "selector": "__boundary_selector_knife",
            "selector_type": "surface_sdf_subregion",
        }
    },
    "domains": {
        "fluid": {
            "type": "fluid",
            "root": "fluid",
            "tags": ["inlet_spot"],
            "selectors": ["__boundary_selector_knife"],
        }
    },
}


def test_legacy_file_migrates_on_load(tmp_path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(_LEGACY_SCENE))

    document = load_scene(path)

    region = next(iter(document.boundary_regions))
    assert isinstance(region, BoundaryRegion)
    assert len(region.cuts) == 1
    assert region.cuts[0].side == "inside"
    assert region.selector_id is None
    assert not any(document.is_internal_scene_node(n) for n in document.objects)
    assert document.fluid_domain.selector_objects == ()

    samples = _face_samples(-1.6)
    mask = boundary_region_mask(document.fluid_domain.root, region, samples)
    assert mask.any()
    picked = samples[mask]
    assert (np.linalg.norm(picked - np.array([-1.6, 0.0, 0.0]), axis=1) < 0.55).all()
