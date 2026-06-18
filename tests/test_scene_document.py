from __future__ import annotations

import numpy as np

from core.boundary import BoundaryRegion
from core.scene import SceneDocument
from core.sdf import (
    Box,
    Difference,
    Extrude,
    PlacedSDF1D,
    PlacedSDF2D,
    RectangleProfile,
    Sphere,
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


def test_rotate_object_updates_placed_2d_axes_about_pivot() -> None:
    document = SceneDocument()
    handle = document.add_primitive("rectangle")
    rectangle = document.node(handle)
    assert isinstance(rectangle, PlacedSDF2D)
    rectangle.origin = (1.0, 0.0, 0.0)
    rectangle.axis_u = (1.0, 0.0, 0.0)
    rectangle.axis_v = (0.0, 1.0, 0.0)
    rectangle.__post_init__()

    rotated_handle = document.rotate_object(
        handle,
        "z",
        90.0,
        (0.0, 0.0, 0.0),
    )

    rotated = document.node(rotated_handle)
    assert rotated is rectangle
    assert np.allclose(rotated.origin, (0.0, 1.0, 0.0), atol=1e-12)
    assert np.allclose(rotated.axis_u, (0.0, 1.0, 0.0), atol=1e-12)
    assert np.allclose(rotated.axis_v, (-1.0, 0.0, 0.0), atol=1e-12)


def test_rotate_object_updates_3d_primitive_orientation_in_place() -> None:
    document = SceneDocument()
    handle = document.add_primitive("box")

    rotated_handle = document.rotate_object(
        handle,
        "z",
        45.0,
        (0.0, 0.0, 0.0),
    )

    rotated = document.node(rotated_handle)
    assert isinstance(rotated, Box)
    assert rotated_handle == handle
    assert np.allclose(
        rotated.axis_u,
        (np.sqrt(0.5), np.sqrt(0.5), 0.0),
        atol=1e-12,
    )
    assert np.allclose(
        rotated.axis_v,
        (-np.sqrt(0.5), np.sqrt(0.5), 0.0),
        atol=1e-12,
    )


def test_move_boolean_tree_moves_children_without_transform_wrapper() -> None:
    document = SceneDocument()
    sphere = document.add_primitive("sphere")
    box = document.add_primitive("box")
    combined = document.combine(sphere, box, "difference")
    root_before = document.node(combined)
    assert isinstance(root_before, Difference)
    assert isinstance(root_before.left, Sphere)
    assert isinstance(root_before.right, Box)
    left_center = root_before.left.center
    right_center = root_before.right.center

    moved = document.move_object(combined, (1.0, 2.0, 3.0))
    root = document.node(moved)

    assert isinstance(root, Difference)
    assert moved == combined
    assert isinstance(root.left, Sphere)
    assert isinstance(root.right, Box)
    delta = (1.0, 2.0, 3.0)
    assert root.left.center == tuple(
        left_center[index] + delta[index] for index in range(3)
    )
    assert root.right.center == tuple(
        right_center[index] + delta[index] for index in range(3)
    )
    assert all(
        not isinstance(node, Translate)
        for _handle, node, _parent in document.walk()
    )


def test_rotate_boolean_tree_rotates_child_placement_without_transform_wrapper() -> None:
    document = SceneDocument()
    sphere = document.add_primitive("sphere")
    box = document.add_primitive("box")
    combined = document.combine(sphere, box, "difference")

    rotated = document.rotate_object(combined, "z", 90.0, (0.0, 0.0, 0.0))
    root = document.node(rotated)

    assert isinstance(root, Difference)
    assert rotated == combined
    assert isinstance(root.left, Sphere)
    assert isinstance(root.right, Box)
    assert np.allclose(root.right.axis_u, (0.0, 1.0, 0.0), atol=1e-12)
    assert np.allclose(root.right.axis_v, (-1.0, 0.0, 0.0), atol=1e-12)
    assert all(
        not isinstance(node, Translate)
        for _handle, node, _parent in document.walk()
    )


def test_document_version_tracks_editable_scene_changes() -> None:
    document = SceneDocument()
    initial = document.version

    sphere = document.add_primitive("sphere")
    after_add = document.version
    document.move_object(sphere, (0.1, 0.0, 0.0))
    after_move = document.version
    snapshot = document.snapshot()

    assert after_add == initial + 1
    assert after_move == after_add + 1
    assert snapshot.version == document.version


def test_document_copy_paste_assigns_fresh_ids_and_offsets_copy() -> None:
    document = SceneDocument()
    sphere_handle = document.add_primitive("sphere")
    sphere = document.node(sphere_handle)
    copied = document.copy_nodes([sphere_handle])

    pasted_handles = document.paste_nodes(copied, (0.25, 0.5, 0.0))
    pasted = document.node(pasted_handles[0])

    assert pasted is not sphere
    assert pasted.name == f"{sphere.name} copy"
    assert pasted.object_id != sphere.object_id
    assert getattr(pasted, "center") == (0.25, 0.5, 0.0)


def test_document_copy_paste_nested_selection_copies_only_selected_root() -> None:
    document = SceneDocument()
    sphere_handle = document.add_primitive("sphere")
    box_handle = document.add_primitive("box")
    union_handle = document.combine(sphere_handle, box_handle, "union")
    nested_handle = next(
        handle
        for handle, node, _parent in document.walk()
        if node.name.startswith("sphere")
    )

    copied = document.copy_nodes([union_handle, nested_handle])
    pasted_handles = document.paste_nodes(copied)

    assert len(pasted_handles) == 1
    assert len(document.objects) == 2
    assert isinstance(document.node(pasted_handles[0]), Union)


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


def test_1d_segments_can_be_combined() -> None:
    document = SceneDocument()
    first = document.add_primitive("segment")
    second = document.add_primitive("segment")
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


def test_legacy_interval_primitive_alias_creates_segment() -> None:
    document = SceneDocument()

    handle = document.add_primitive("interval")
    node = document.node(handle)

    assert isinstance(node, PlacedSDF1D)
    assert node.name.startswith("segment_")


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


def test_drag_creation_uses_reference_plane_axes() -> None:
    document = SceneDocument()

    rectangle_handle = document.add_primitive_from_drag(
        "rectangle",
        (-1.0, 0.0, -0.5),
        (1.0, 0.0, 0.5),
    )
    rectangle = document.node(rectangle_handle)

    assert isinstance(rectangle, PlacedSDF2D)
    assert rectangle.origin == (0.0, 0.0, 0.0)
    assert rectangle.axis_u == (1.0, 0.0, 0.0)
    assert rectangle.axis_v == (0.0, 0.0, 1.0)
    assert isinstance(rectangle.profile, RectangleProfile)
    assert rectangle.profile.half_size == (1.0, 0.5)

    box_handle = document.add_primitive_from_drag(
        "box",
        (0.0, -2.0, -0.5),
        (0.0, 2.0, 0.5),
    )
    box = document.node(box_handle)

    assert isinstance(box, Box)
    assert box.center == (0.0, 0.0, 0.0)
    assert box.half_size == (2.0, 2.0, 0.5)


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
