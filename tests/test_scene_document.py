from __future__ import annotations

import numpy as np

from core.boundary import BoundaryRegion
from core.scene import SceneDocument
from core.sdf import (
    Difference,
    Extrude,
    PlacedSDF1D,
    PlacedSDF2D,
    Translate,
    Union,
)


def test_document_add_combine_transform_and_snapshot() -> None:
    document = SceneDocument()
    sphere_handle = document.add_primitive("sphere")
    box_handle = document.add_primitive("box")
    combined_handle = document.combine(sphere_handle, box_handle, "difference")
    assert isinstance(document.node(combined_handle), Difference)
    transformed_handle = document.wrap_transform(combined_handle, "translate")
    transformed = document.node(transformed_handle)
    assert isinstance(transformed, Translate)
    document.set_fluid_root(transformed_handle)

    snapshot = document.snapshot()
    transformed.offset = (2.0, 0.0, 0.0)
    snapshot_transform = snapshot.bodies[0]
    assert isinstance(snapshot_transform, Translate)
    assert snapshot_transform.offset != transformed.offset
    assert snapshot.fluid_domain is not None
    assert snapshot.fluid_domain.root is snapshot_transform


def test_document_delete_nested_child_collapses_boolean() -> None:
    document = SceneDocument()
    sphere_handle = document.add_primitive("sphere")
    box_handle = document.add_primitive("box")
    document.combine(sphere_handle, box_handle, "union")
    nested_box = next(
        handle
        for handle, node, _parent in document.walk()
        if node.name.startswith("box")
    )
    document.delete(nested_box)
    assert len(document.bodies) == 1
    assert document.bodies[0].name.startswith("sphere")


def test_boolean_with_fluid_root_promotes_result_to_domain() -> None:
    document = SceneDocument.default()
    assert document.fluid_domain is not None
    old_root = document.fluid_domain.root
    sphere_handle = document.add_primitive("sphere")
    result_handle = document.combine(
        document.handle_for(old_root),
        sphere_handle,
        "union",
    )
    assert document.fluid_domain is not None
    assert document.fluid_domain.root is document.node(result_handle)
    assert document.fluid_domain.root is not old_root


def test_2d_boolean_remains_2d_and_can_be_extruded() -> None:
    document = SceneDocument()
    circle = document.add_primitive("circle")
    rectangle = document.add_primitive("rectangle")
    combined = document.combine(circle, rectangle, "union")
    profile = document.node(combined)
    assert isinstance(profile, PlacedSDF2D)
    assert profile.dimension == 2

    solid_handle = document.solid_from_2d([combined], "extrude")
    solid = document.node(solid_handle)
    assert isinstance(solid, Extrude)
    assert solid.dimension == 3
    document.set_fluid_root(solid_handle)
    document.set_tag_enabled(combined, True)
    assert document.fluid_domain is not None
    assert document.fluid_domain.tag_objects == (profile,)


def test_2d_object_can_be_selected_as_fluid_domain() -> None:
    document = SceneDocument()
    rectangle_handle = document.add_primitive("rectangle")

    document.set_fluid_root(rectangle_handle)
    assert document.fluid_domain is not None
    assert document.fluid_domain.root is document.node(rectangle_handle)
    assert document.fluid_domain.root.dimension == 2

    region_handle = document.add_boundary_region(
        document.fluid_domain.root.object_id,
        outside_direction=1,
    )
    region = document.node(region_handle)
    assert isinstance(region, PlacedSDF1D)
    assert region.dimension == 1
    assert region.origin == (0.5, 0.0, 0.0)
    assert region.axis_u == (0.0, 1.0, 0.0)
    assert region in document.fluid_domain.tag_objects


def test_generated_2d_boundary_tag_is_in_visual_tree_components() -> None:
    document = SceneDocument()
    rectangle_handle = document.add_primitive("rectangle")

    document.set_fluid_root(rectangle_handle)
    assert document.fluid_domain is not None

    region_handle = document.add_boundary_region(
        document.fluid_domain.root.object_id,
        outside_direction=1,
    )
    region = document.node(region_handle)
    assert isinstance(region, PlacedSDF1D)

    tree = document.visual_tree()

    assert any(node is region for node in tree.nodes)


def test_1d_intervals_can_be_combined() -> None:
    document = SceneDocument()
    first = document.add_primitive("interval")
    second = document.add_primitive("interval")
    second_node = document.node(second)
    assert isinstance(second_node, PlacedSDF1D)
    second_node.origin = (0.75, 0.0, 0.0)

    result_handle = document.combine(first, second, "union")
    result = document.node(result_handle)

    assert isinstance(result, PlacedSDF1D)
    assert result.dimension == 1
    assert result.to_numpy(
        np.asarray((0.0, 1.0, 2.0)),
        np.zeros(3),
        np.zeros(3),
    ).tolist() == [-0.5, -0.25, 0.75]


def test_transformed_fluid_root_evaluates_in_batches() -> None:
    document = SceneDocument()
    sphere = document.add_primitive("sphere")
    transformed = document.wrap_transform(sphere, "translate")
    node = document.node(transformed)
    node.offset = (1.0, 0.0, 0.0)
    document.set_fluid_root(transformed)
    assert document.fluid_domain is not None
    x = np.asarray([1.0, 2.0], dtype=np.float64)
    zero = np.zeros(2, dtype=np.float64)
    distances = document.fluid_domain.to_numpy(x, zero, zero)
    assert distances.tolist() == [-0.5, 0.5]


def test_drag_creation_and_move_keep_fluid_root_consistent() -> None:
    document = SceneDocument()
    handle = document.add_primitive_from_drag(
        "rectangle", (-1.0, -0.5, 0.0), (1.0, 0.5, 0.0)
    )
    placed = document.node(handle)
    assert isinstance(placed, PlacedSDF2D)
    assert placed.origin == (0.0, 0.0, 0.0)
    assert placed.axis_u == (1.0, 0.0, 0.0)
    assert placed.axis_v == (0.0, 1.0, 0.0)

    box_handle = document.add_primitive_from_drag(
        "box", (-1.0, -1.0, 0.0), (1.0, 1.0, 0.0)
    )
    document.set_fluid_root(box_handle)
    moved_handle = document.move_object(box_handle, (2.0, 0.0, 3.0))
    assert document.fluid_domain is not None
    assert document.fluid_domain.root is document.node(moved_handle)


def test_boundary_region_is_created_as_enabled_owner_tag_and_can_be_deleted() -> None:
    document = SceneDocument.default()
    assert document.fluid_domain is not None
    obstacle = next(
        node
        for _handle, node, _parent in document.walk()
        if node.name == "cylinder_obstacle"
    )

    handle = document.add_boundary_region(obstacle.object_id)
    region = document.node(handle)

    assert isinstance(region, BoundaryRegion)
    assert region.owner_object_id == obstacle.object_id
    assert region in document.fluid_domain.tag_objects

    document.delete(handle)

    assert region not in document.boundary_regions
    assert region not in document.fluid_domain.tag_objects


def test_boundary_region_rejects_structural_csg_owner() -> None:
    document = SceneDocument.default()
    assert document.fluid_domain is not None

    with np.testing.assert_raises_regex(ValueError, "does not directly control"):
        document.add_boundary_region(document.fluid_domain.root.object_id)
