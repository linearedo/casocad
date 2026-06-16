from __future__ import annotations

import json

import numpy as np

from core.boundary import BoundaryRegion
from core.scene import SceneDocument
from core.serialization import load_scene, save_scene
from core.sdf import PlacedSDF1D


def test_scene_json_roundtrip_preserves_geometry_and_domain(tmp_path) -> None:
    document = SceneDocument.default()
    root_handle = document.handle_for(document.bodies[0])
    document.wrap_transform(root_handle, "rotate")
    path = tmp_path / "scene.casocad.json"
    save_scene(document, path)
    restored = load_scene(path)

    coordinates = np.asarray([-0.5, 0.0, 0.5], dtype=np.float64)
    zero = np.zeros(3, dtype=np.float64)
    np.testing.assert_allclose(
        restored.tree().to_numpy(coordinates, zero, zero),
        document.tree().to_numpy(coordinates, zero, zero),
    )
    assert restored.fluid_domain is not None
    assert document.fluid_domain is not None
    assert restored.fluid_domain.root.object_id == document.fluid_domain.root.object_id
    assert [
        (tag.object_id, tag.name, tag.dimension)
        for tag in restored.fluid_domain.tag_objects
    ] == [
        (tag.object_id, tag.name, tag.dimension)
        for tag in document.fluid_domain.tag_objects
    ]


def test_load_rejects_unknown_format(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"format": "something_else", "version": 1}', encoding="utf-8")
    try:
        load_scene(path)
    except ValueError as error:
        assert "not a casoCAD" in str(error)
    else:
        raise AssertionError("invalid scene format was accepted")


def test_scene_json_roundtrip_preserves_boundary_regions(tmp_path) -> None:
    document = SceneDocument.default()
    obstacle = next(
        node
        for _handle, node, _parent in document.walk()
        if node.name == "cylinder_obstacle"
    )
    document.add_boundary_region(obstacle.object_id, outside_direction=4)
    path = tmp_path / "boundary-region.casocad.json"

    save_scene(document, path)
    restored = load_scene(path)

    region = next(
        item
        for item in restored.boundary_regions
        if item.name == "cylinder_obstacle boundary 4"
    )
    assert isinstance(region, BoundaryRegion)
    assert region.owner_object_id == obstacle.object_id
    assert region.outside_direction == 4
    assert restored.fluid_domain is not None
    assert region in restored.fluid_domain.tag_objects
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["version"] == 4


def test_scene_json_roundtrip_preserves_2d_fluid_domain(tmp_path) -> None:
    document = SceneDocument()
    rectangle_handle = document.add_primitive("rectangle")
    document.set_fluid_root(rectangle_handle)
    assert document.fluid_domain is not None
    document.add_boundary_region(
        document.fluid_domain.root.object_id,
        outside_direction=2,
    )
    path = tmp_path / "fluid-2d.casocad.json"

    save_scene(document, path)
    restored = load_scene(path)

    assert restored.fluid_domain is not None
    assert restored.fluid_domain.root.dimension == 2
    assert len(restored.fluid_domain.tag_objects) == 1
    region = restored.fluid_domain.tag_objects[0]
    assert isinstance(region, PlacedSDF1D)
    assert region.origin == (0.0, -0.35, 0.0)
    assert region.axis_u == (1.0, 0.0, 0.0)


def test_version_2_scene_loads_without_boundary_regions(tmp_path) -> None:
    document = SceneDocument.default()
    path = tmp_path / "version-2.casocad.json"
    save_scene(document, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 2
    payload.pop("boundary_regions")
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = load_scene(path)

    assert not restored.boundary_regions


def test_version_3_2d_boundary_region_migrates_to_1d_sdf(tmp_path) -> None:
    document = SceneDocument()
    rectangle_handle = document.add_primitive("rectangle")
    document.set_fluid_root(rectangle_handle)
    path = tmp_path / "legacy-2d-boundary.casocad.json"
    save_scene(document, path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["version"] = 3
    payload["boundary_regions"] = [
        {
            "object_id": 2,
            "name": "legacy inlet",
            "owner_object_id": 1,
            "outside_direction": 0,
        }
    ]
    payload["fluid_domain"]["tag_object_ids"] = [2]
    path.write_text(json.dumps(payload), encoding="utf-8")

    restored = load_scene(path)

    assert restored.fluid_domain is not None
    assert not restored.boundary_regions
    tag = restored.fluid_domain.tag_objects[0]
    assert isinstance(tag, PlacedSDF1D)
    assert tag.object_id == 2
    assert tag.origin == (-0.5, 0.0, 0.0)
