from __future__ import annotations

import numpy as np

from core.boundary import BoundaryRegion
from core.io import read_lattice
from core.mesher import FluidDomain, LatticeMesher, MesherConfig
from core.sdf import (
    Box,
    CircleProfile,
    Cylinder,
    Difference,
    IntervalProfile,
    PlacedSDF1D,
    PlacedSDF2D,
    RectangleProfile,
    Sphere,
)
from core.scene import SceneDocument


def test_arrow_roundtrip(tmp_path) -> None:
    sphere = Sphere(name="fluid_ball", radius=0.5, object_id=1)
    equator = PlacedSDF2D(
        name="equator",
        object_id=2,
        profile=CircleProfile(radius=0.5),
    )
    domain = FluidDomain(sphere, (equator,))
    config = MesherConfig(dx=0.25, chunk_size=20)
    path = tmp_path / "sphere.arrow"
    result = LatticeMesher(domain, config).mesh(path)
    table, metadata = read_lattice(path)

    assert result.row_count == table.num_rows
    assert table.schema.names == [
        "x",
        "y",
        "z",
        "i",
        "j",
        "k",
        "node_type",
        "tag_ids",
        "level",
    ]
    assert set(table["node_type"].to_pylist()) <= {0, 1}
    assert metadata["grid"]["n_levels"] == 0
    assert metadata["grid"]["dx"] == config.dx
    assert set(table["level"].to_pylist()) == {0}
    assert metadata["fluid_domain"]["root_object_id"] == sphere.object_id
    assert metadata["fluid_domain"]["tag_object_ids"] == [equator.object_id]
    assert result.preview_cell_size == config.dx
    assert result.preview_positions.dtype.name == "float32"
    assert result.preview_positions.shape[1] == 3
    assert result.preview_positions.shape[0] == result.preview_node_types.size
    assert result.preview_positions.shape[0] == result.preview_boundary_faces.size
    assert np.all(
        result.preview_boundary_faces[result.preview_node_types == 0] == 0
    )
    assert np.all(
        result.preview_boundary_faces[result.preview_node_types == 1] != 0
    )
    assert result.preview_positions.shape[0] == result.preview_source_object_ids.size
    assert result.preview_positions.shape[0] == len(result.preview_tag_ids)
    assert result.preview_tag_axis_u.shape == result.preview_positions.shape
    assert result.preview_tag_axis_v.shape == result.preview_positions.shape
    assert result.boundary_sample_indices.shape[1] == 3
    assert result.boundary_sample_positions.shape[1] == 3
    assert result.boundary_sample_normals.shape == (
        result.boundary_sample_positions.shape
    )
    assert result.boundary_sample_directions.shape[0] == (
        result.boundary_sample_positions.shape[0]
    )
    assert result.boundary_sample_owner_object_ids.shape[0] == (
        result.boundary_sample_positions.shape[0]
    )
    assert len(result.boundary_sample_region_ids) == (
        result.boundary_sample_positions.shape[0]
    )
    assert result.boundary_sample_errors.shape[0] == (
        result.boundary_sample_positions.shape[0]
    )
    assert result.boundary_error_maximum >= result.boundary_error_mean
    assert result.boundary_error_rms >= result.boundary_error_mean
    assert result.boundary_error_percentile_95 <= result.boundary_error_maximum
    assert metadata["grid"]["recommended_max_dx"] > 0.0
    assert metadata["boundary_sampling"]["ownership"] == (
        "evaluated per exposed lattice direction"
    )


def test_2d_arrow_lattice_uses_four_neighbors_and_one_k_layer(
    tmp_path,
) -> None:
    rectangle = PlacedSDF2D(
        name="fluid_2d",
        object_id=1,
        profile=RectangleProfile(half_size=(1.0, 0.5)),
        origin=(1.0, 2.0, 3.0),
        axis_u=(0.0, 1.0, 0.0),
        axis_v=(0.0, 0.0, 1.0),
    )
    negative_u = PlacedSDF1D(
        name="inlet",
        object_id=2,
        profile=IntervalProfile(half_length=0.5),
        origin=(1.0, 1.0, 3.0),
        axis_u=(0.0, 0.0, 1.0),
    )
    result = LatticeMesher(
        FluidDomain(rectangle, (negative_u,)),
        MesherConfig(
            dx=0.5,
            chunk_size=4,
            internal_preview_density=1.0,
        ),
    ).mesh(tmp_path / "rectangle-2d.arrow")
    table, metadata = read_lattice(result.path)
    positions = np.column_stack(
        (
            np.asarray(table["x"]),
            np.asarray(table["y"]),
            np.asarray(table["z"]),
        )
    )
    node_types = np.asarray(table["node_type"])
    tags = table["tag_ids"].to_pylist()

    assert result.dimension == 2
    assert result.grid_node_count == 15
    assert result.row_count == 15
    assert metadata["dimension"] == 2
    assert metadata["grid"]["dimension"] == 2
    assert metadata["grid"]["nz"] == 1
    assert metadata["boundary_rule"].endswith("outside 4-neighbor")
    assert set(table["k"].to_pylist()) == {0}
    np.testing.assert_allclose(positions[:, 0], 1.0)
    assert np.count_nonzero(node_types == 1) == 12
    assert np.count_nonzero(node_types == 0) == 3
    assert set(result.boundary_sample_directions.tolist()) == {0, 1, 2, 3}
    assert np.allclose(result.boundary_sample_normals[:, 0], 0.0)

    inlet_nodes = np.asarray(
        [negative_u.object_id in items for items in tags],
        dtype=np.bool_,
    )
    assert np.count_nonzero(inlet_nodes) == 3
    np.testing.assert_allclose(positions[inlet_nodes, 1], 1.0)
    assert all(
        region_ids == (negative_u.object_id,)
        for direction, region_ids in zip(
            result.boundary_sample_directions,
            result.boundary_sample_region_ids,
            strict=True,
        )
        if direction == 0
    )


def test_2d_lattice_is_chunk_partition_independent(tmp_path) -> None:
    rectangle = PlacedSDF2D(
        name="fluid_2d",
        object_id=1,
        profile=RectangleProfile(half_size=(0.9, 0.6)),
    )

    def rows(chunk_size: int) -> set[tuple[float, float, int, int]]:
        result = LatticeMesher(
            FluidDomain(rectangle),
            MesherConfig(dx=0.3, chunk_size=chunk_size),
        ).mesh(tmp_path / f"rectangle-2d-{chunk_size}.arrow")
        table, _metadata = read_lattice(result.path)
        return set(
            zip(
                table["x"].to_pylist(),
                table["y"].to_pylist(),
                table["node_type"].to_pylist(),
                table["k"].to_pylist(),
                strict=True,
            )
        )

    assert rows(5) == rows(1_000)


def test_preview_keeps_boundaries_and_samples_interior_across_domain(
    tmp_path,
) -> None:
    box = Box(
        name="long_box",
        object_id=1,
        half_size=(5.0, 1.0, 1.0),
    )
    result = LatticeMesher(
        FluidDomain(box),
        MesherConfig(
            dx=0.25,
            chunk_size=100,
            internal_preview_density=0.0,
        ),
        preview_limit=150,
    ).mesh(tmp_path / "sampled-preview.arrow")

    boundary = result.preview_node_types == 1
    interior = result.preview_node_types == 0
    expected_boundary_count = 2 * (9 * 9) + 2 * (39 * 9) + 2 * (39 * 7)

    assert np.count_nonzero(boundary) == expected_boundary_count
    assert result.preview_positions.shape[0] == expected_boundary_count
    assert not interior.any()
    assert np.isclose(result.preview_positions[boundary, 0].min(), -5.0)
    assert np.isclose(result.preview_positions[boundary, 0].max(), 5.0)

    sampled = LatticeMesher(
        FluidDomain(box),
        MesherConfig(
            dx=0.25,
            chunk_size=100,
            internal_preview_density=1.0,
        ),
        preview_limit=40,
    ).mesh(tmp_path / "sampled-preview-with-interior.arrow")
    sampled_interior = sampled.preview_positions[
        sampled.preview_node_types == 0
    ]
    assert sampled_interior.shape[0] == 40
    assert sampled_interior[:, 0].min() < -3.0
    assert sampled_interior[:, 0].max() > 3.0

    ten_percent = LatticeMesher(
        FluidDomain(box),
        MesherConfig(
            dx=0.25,
            chunk_size=100,
            internal_preview_density=0.1,
        ),
    ).mesh(tmp_path / "ten-percent-preview.arrow")
    total_internal = ten_percent.row_count - expected_boundary_count
    assert np.count_nonzero(ten_percent.preview_node_types == 0) == round(
        total_internal * 0.1
    )

    no_interior = LatticeMesher(
        FluidDomain(box),
        MesherConfig(
            dx=0.25,
            chunk_size=100,
            internal_preview_density=0.0,
        ),
    ).mesh(tmp_path / "boundary-only-preview.arrow")
    assert np.all(no_interior.preview_node_types == 1)


def test_overlapping_tags_are_preserved_on_one_node(tmp_path) -> None:
    sphere = Sphere(name="fluid", radius=0.5, object_id=1)
    first = PlacedSDF2D(
        name="first",
        object_id=2,
        profile=CircleProfile(radius=0.5),
    )
    second = PlacedSDF2D(
        name="second",
        object_id=3,
        profile=CircleProfile(radius=0.5),
    )
    domain = FluidDomain(sphere, (first, second))
    config = MesherConfig(
        dx=0.25,
        chunk_size=20,
    )
    path = tmp_path / "overlap.arrow"
    LatticeMesher(domain, config).mesh(path)
    table, _metadata = read_lattice(path)
    tags = table["tag_ids"].to_pylist()
    assert [2, 3] in tags


def test_boundary_edges_and_corners_are_unique_multi_tag_nodes(tmp_path) -> None:
    box = Box(
        name="fluid",
        object_id=1,
        half_size=(0.5, 0.5, 0.5),
    )
    regions = (
        BoundaryRegion(
            name="negative_x",
            object_id=2,
            owner_object_id=box.object_id,
            outside_direction=0,
        ),
        BoundaryRegion(
            name="negative_y",
            object_id=3,
            owner_object_id=box.object_id,
            outside_direction=2,
        ),
        BoundaryRegion(
            name="negative_z",
            object_id=4,
            owner_object_id=box.object_id,
            outside_direction=4,
        ),
    )
    result = LatticeMesher(
        FluidDomain(box, regions),
        MesherConfig(
            dx=0.25,
            chunk_size=7,
            internal_preview_density=1.0,
        ),
    ).mesh(tmp_path / "boundary-edge-corner.arrow")
    table, _metadata = read_lattice(result.path)
    indices = np.column_stack(
        (
            np.asarray(table["i"]),
            np.asarray(table["j"]),
            np.asarray(table["k"]),
        )
    )
    positions = np.column_stack(
        (
            np.asarray(table["x"]),
            np.asarray(table["y"]),
            np.asarray(table["z"]),
        )
    )
    tags = table["tag_ids"].to_pylist()

    assert np.unique(indices, axis=0).shape[0] == len(table)
    assert np.unique(positions, axis=0).shape[0] == len(table)
    assert np.unique(result.preview_positions, axis=0).shape[0] == len(
        result.preview_positions
    )
    assert all(items == sorted(set(items)) for items in tags)

    corner = np.all(
        np.isclose(positions, (-0.5, -0.5, -0.5)),
        axis=1,
    )
    assert np.count_nonzero(corner) == 1
    assert tags[int(np.flatnonzero(corner)[0])] == [2, 3, 4]

    edge = (
        np.isclose(positions[:, 0], -0.5)
        & np.isclose(positions[:, 1], -0.5)
        & np.isclose(positions[:, 2], 0.0)
    )
    assert np.count_nonzero(edge) == 1
    assert tags[int(np.flatnonzero(edge)[0])] == [2, 3]


def test_refined_boundary_samples_are_chunk_partition_independent(
    tmp_path,
) -> None:
    sphere = Sphere(name="fluid", object_id=1, radius=0.65)
    region = BoundaryRegion(
        name="wall",
        object_id=2,
        owner_object_id=sphere.object_id,
    )

    def samples(chunk_size: int) -> dict[tuple[int, int, int, int], tuple[object, ...]]:
        result = LatticeMesher(
            FluidDomain(sphere, (region,)),
            MesherConfig(dx=0.2, chunk_size=chunk_size),
        ).mesh(tmp_path / f"refined-{chunk_size}.arrow")
        return {
            (
                int(index[0]),
                int(index[1]),
                int(index[2]),
                int(direction),
            ): (
                position,
                normal,
                int(owner),
                region_ids,
            )
            for index, direction, position, normal, owner, region_ids in zip(
                result.boundary_sample_indices,
                result.boundary_sample_directions,
                result.boundary_sample_positions,
                result.boundary_sample_normals,
                result.boundary_sample_owner_object_ids,
                result.boundary_sample_region_ids,
                strict=True,
            )
        }

    small = samples(7)
    large = samples(10_000)
    assert small.keys() == large.keys()
    for key in small:
        small_position, small_normal, small_owner, small_regions = small[key]
        large_position, large_normal, large_owner, large_regions = large[key]
        np.testing.assert_allclose(small_position, large_position, atol=1e-12)
        np.testing.assert_allclose(small_normal, large_normal, atol=1e-12)
        assert small_owner == large_owner
        assert small_regions == large_regions


def test_von_karman_preview_contains_subtractive_boundary_and_flat_tags(
    tmp_path,
) -> None:
    document = SceneDocument.default()
    assert document.fluid_domain is not None
    config = MesherConfig(
        dx=0.08,
        chunk_size=1_000_000,
    )
    result = LatticeMesher(document.fluid_domain, config).mesh(
        tmp_path / "von_karman.arrow"
    )
    obstacle = next(
        node
        for _handle, node, _parent in document.walk()
        if node.name == "cylinder_obstacle"
    )
    boundary_sources = result.preview_source_object_ids[
        result.preview_node_types == 1
    ]
    assert obstacle.object_id in boundary_sources
    obstacle_sources = result.preview_source_object_ids == obstacle.object_id
    assert obstacle_sources.any()
    assert np.all(result.preview_node_types[obstacle_sources] == 1)
    assert np.all(
        result.preview_source_object_ids[result.preview_node_types == 0] != 0
    )
    tagged = result.preview_primary_tag_ids != 0
    assert tagged.any()
    assert np.all(result.preview_node_types[tagged] == 1)
    tagged_ids = set(result.preview_primary_tag_ids[tagged].tolist())
    sample_region_ids = {
        region_id
        for region_ids in result.boundary_sample_region_ids
        for region_id in region_ids
    }
    assert tagged_ids <= sample_region_ids


def test_face_tags_remain_on_boundary_across_dx_values(tmp_path) -> None:
    for dx in (0.2, 0.16, 0.11):
        document = SceneDocument.default()
        assert document.fluid_domain is not None
        result = LatticeMesher(
            document.fluid_domain,
            MesherConfig(dx=dx, chunk_size=1_000_000),
        ).mesh(tmp_path / f"face-tags-{dx}.arrow")
        inlet_id = next(
            tag.object_id
            for tag in document.fluid_domain.tag_objects
            if tag.name == "inlet"
        )
        inlet_mask = np.asarray(
            [inlet_id in tags for tags in result.preview_tag_ids],
            dtype=np.bool_,
        )
        inlet = next(
            tag
            for tag in document.fluid_domain.tag_objects
            if tag.object_id == inlet_id
        )
        assert isinstance(inlet, BoundaryRegion)
        owner = next(
            node
            for _handle, node, _parent in document.walk()
            if node.object_id == inlet.owner_object_id
        )
        face_x = owner.bounding_box().x_min
        face_mask = (
            result.preview_node_types == 1
        ) & np.isclose(result.preview_positions[:, 0], face_x)
        assert inlet_mask.any()
        assert np.all(result.preview_node_types[inlet_mask] == 1)
        assert np.array_equal(inlet_mask, face_mask)


def test_added_sphere_subtraction_is_present_in_meshed_domain(tmp_path) -> None:
    document = SceneDocument.default()
    assert document.fluid_domain is not None
    old_root = document.fluid_domain.root
    sphere_handle = document.add_primitive("sphere")
    sphere = document.node(sphere_handle)
    assert isinstance(sphere, Sphere)
    sphere.center = (0.72, 0.0, 0.0)
    sphere.radius = 0.26
    result_handle = document.combine(
        document.handle_for(old_root),
        sphere_handle,
        "difference",
    )
    assert document.fluid_domain is not None
    assert document.fluid_domain.root is document.node(result_handle)
    result = LatticeMesher(
        document.fluid_domain,
        MesherConfig(dx=0.08, chunk_size=1_000_000),
    ).mesh(tmp_path / "sphere-cavity.arrow")
    sphere_boundary = (
        (result.preview_node_types == 1)
        & (result.preview_source_object_ids == sphere.object_id)
    )
    assert sphere_boundary.any()


def test_2d_difference_preview_preserves_subtractive_boundary_owner(tmp_path) -> None:
    document = SceneDocument()
    rectangle_handle = document.add_primitive("rectangle")
    circle_handle = document.add_primitive("circle")
    rectangle = document.node(rectangle_handle)
    circle = document.node(circle_handle)
    assert isinstance(rectangle, PlacedSDF2D)
    assert isinstance(circle, PlacedSDF2D)
    rectangle.profile = RectangleProfile(half_size=(0.5, 0.5))
    rectangle.__post_init__()
    circle.profile = CircleProfile(radius=0.2)
    circle.origin = (0.1, 0.0, 0.0)
    circle.__post_init__()
    root_handle = document.combine(rectangle_handle, circle_handle, "difference")
    document.set_fluid_root(root_handle)
    assert document.fluid_domain is not None

    result = LatticeMesher(
        document.fluid_domain,
        MesherConfig(dx=0.1, chunk_size=1_000_000, internal_preview_density=1.0),
    ).mesh(tmp_path / "difference-2d.arrow")

    circle_boundary = (
        (result.preview_node_types == 1)
        & (result.preview_source_object_ids == circle.object_id)
    )
    assert circle_boundary.any()
    circle_owned = result.preview_source_object_ids == circle.object_id
    assert np.all(result.preview_node_types[circle_owned] == 1)


def test_boundary_region_tags_only_owned_final_boundary_nodes(tmp_path) -> None:
    box = Box(
        name="fluid",
        object_id=1,
        half_size=(1.0, 1.0, 1.0),
    )
    obstacle = Cylinder(
        name="obstacle",
        object_id=2,
        radius=0.35,
        half_height=1.0,
    )
    root = Difference(
        name="fluid_with_obstacle",
        object_id=3,
        left=box,
        right=obstacle,
    )
    region = BoundaryRegion(
        name="obstacle_wall",
        object_id=4,
        owner_object_id=obstacle.object_id,
    )
    path = tmp_path / "boundary-region.arrow"

    result = LatticeMesher(
        FluidDomain(root, (region,)),
        MesherConfig(dx=0.2, chunk_size=200),
    ).mesh(path)
    table, metadata = read_lattice(path)
    tagged = np.asarray(
        [region.object_id in items for items in table["tag_ids"].to_pylist()],
        dtype=np.bool_,
    )

    assert tagged.any()
    assert np.all(np.asarray(table["node_type"])[tagged] == 1)
    preview_tagged = np.asarray(
        [region.object_id in items for items in result.preview_tag_ids],
        dtype=np.bool_,
    )
    sample_owned = (
        result.boundary_sample_owner_object_ids == obstacle.object_id
    )
    sample_tagged = np.asarray(
        [
            region.object_id in items
            for items in result.boundary_sample_region_ids
        ],
        dtype=np.bool_,
    )
    assert np.array_equal(sample_tagged, sample_owned)
    tagged_sample_nodes = np.unique(
        result.boundary_sample_indices[sample_tagged],
        axis=0,
    )
    preview_tagged_positions = result.preview_positions[preview_tagged]
    assert preview_tagged_positions.shape[0] == tagged_sample_nodes.shape[0]
    directory = {
        item["object_id"]: item for item in metadata["object_directory"]
    }
    assert directory[region.object_id]["kind"] == "boundary_region"


def test_directional_boundary_region_selects_one_box_face(tmp_path) -> None:
    box = Box(
        name="fluid",
        object_id=1,
        half_size=(0.5, 0.5, 0.5),
    )
    negative_x = BoundaryRegion(
        name="negative_x",
        object_id=2,
        owner_object_id=box.object_id,
        outside_direction=0,
    )
    result = LatticeMesher(
        FluidDomain(box, (negative_x,)),
        MesherConfig(dx=0.25, chunk_size=100),
    ).mesh(tmp_path / "negative-x.arrow")
    tagged = np.asarray(
        [negative_x.object_id in items for items in result.preview_tag_ids],
        dtype=np.bool_,
    )

    assert tagged.any()
    assert np.all(result.preview_positions[tagged, 0] == -0.5)
    assert np.all((result.preview_boundary_faces[tagged] & 1) != 0)


def test_directional_box_region_excludes_remote_csg_intersection(
    tmp_path,
) -> None:
    box = Box(
        name="fluid",
        object_id=1,
        half_size=(1.6, 0.7, 0.45),
    )
    obstacle = Cylinder(
        name="obstacle",
        object_id=2,
        radius=0.24,
        half_height=0.55,
    )
    root = Difference(
        name="fluid_with_obstacle",
        object_id=3,
        left=box,
        right=obstacle,
    )
    inlet = BoundaryRegion(
        name="inlet",
        object_id=4,
        owner_object_id=box.object_id,
        outside_direction=0,
    )
    result = LatticeMesher(
        FluidDomain(root, (inlet,)),
        MesherConfig(dx=0.2, chunk_size=10_000),
    ).mesh(tmp_path / "inlet.arrow")
    tagged = np.asarray(
        [inlet.object_id in items for items in result.preview_tag_ids],
        dtype=np.bool_,
    )

    assert tagged.any()
    np.testing.assert_allclose(result.preview_positions[tagged, 0], -1.6)
