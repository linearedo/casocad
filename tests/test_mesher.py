from __future__ import annotations

import numpy as np

from core.boundary import BoundaryRegion
from core.mesher import FluidDomain, LatticeMesher, MesherConfig
from core.mesher.classifier import (
    boundary_owner_ids,
    classify_boundary_faces,
    classify_nodes,
    evaluate_volume_attribution,
    evaluate_with_attribution,
    nearest_tag_mask,
    pick_boundary_owner,
    pick_sdf_surface,
    retained_mask,
    sample_boundary_faces,
)
from core.mesher.grid import derive_grid, derive_lattice_grid, generate_chunks
from core.sdf import (
    BinaryProfile,
    Box,
    CircleProfile,
    Cylinder,
    Difference,
    IntervalProfile,
    PlacedSDF1D,
    PlacedSDF2D,
    RectangleProfile,
    OffsetProfile,
    Sphere,
    Union,
)


def test_grid_chunks_cover_envelope_once() -> None:
    sphere = Sphere(name="sphere", radius=0.5, object_id=1)
    grid = derive_grid(sphere.bounding_box(), 0.25)
    chunks = list(generate_chunks(grid, 17))
    assert sum(chunk.x.size for chunk in chunks) == grid.node_count
    assert max(chunk.x.size for chunk in chunks) <= 17


def test_2d_grid_follows_the_placed_sdf_workplane() -> None:
    rectangle = PlacedSDF2D(
        name="fluid",
        object_id=1,
        profile=RectangleProfile(half_size=(1.0, 0.5)),
        origin=(1.0, 2.0, 3.0),
        axis_u=(0.0, 1.0, 0.0),
        axis_v=(0.0, 0.0, 1.0),
    )
    grid = derive_lattice_grid(rectangle, 0.5)
    chunk = next(generate_chunks(grid, grid.node_count))

    assert grid.dimension == 2
    assert (grid.nx, grid.ny, grid.nz) == (5, 3, 1)
    assert np.all(chunk.k == 0)
    np.testing.assert_allclose(chunk.x, 1.0)
    assert np.isclose(chunk.y.min(), 1.0)
    assert np.isclose(chunk.y.max(), 3.0)
    assert np.isclose(chunk.z.min(), 2.5)
    assert np.isclose(chunk.z.max(), 3.5)


def test_classification() -> None:
    box = Box(name="box", object_id=1, half_size=(1.0, 1.0, 1.0))
    x = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5], dtype=np.float64)
    zero = np.zeros(x.shape, dtype=np.float64)
    sdf = box.to_numpy(x, zero, zero)
    keep = retained_mask(sdf)
    assert keep.tolist() == [True, True, True, True, True, False]
    node_type = classify_nodes(box, x, zero, zero, keep, 0.5)
    assert node_type.tolist() == [1, 0, 0, 0, 1]
    boundary_faces = classify_boundary_faces(box, x, zero, zero, keep, 0.5)
    assert boundary_faces.tolist() == [1, 0, 0, 0, 2]


def test_boundary_samples_refine_sphere_crossings_and_normals() -> None:
    sphere = Sphere(name="sphere", object_id=1, radius=0.75)
    x = np.asarray([0.5], dtype=np.float64)
    zero = np.zeros(1, dtype=np.float64)
    samples = sample_boundary_faces(
        sphere,
        x,
        zero,
        zero,
        np.asarray([True]),
        0.5,
    )

    positive_x = samples.directions == 1
    assert np.count_nonzero(positive_x) == 1
    np.testing.assert_allclose(
        samples.positions[positive_x][0],
        (0.75, 0.0, 0.0),
        atol=0.001,
    )
    np.testing.assert_allclose(
        samples.normals[positive_x][0],
        (1.0, 0.0, 0.0),
        atol=0.001,
    )
    assert samples.owner_object_ids[positive_x].tolist() == [sphere.object_id]
    np.testing.assert_allclose(
        samples.approximation_errors[positive_x],
        (0.25,),
        atol=0.001,
    )


def test_boundary_error_is_zero_for_grid_aligned_box_surface() -> None:
    box = Box(
        name="box",
        object_id=1,
        half_size=(0.5, 0.5, 0.5),
    )
    coordinate = np.asarray([0.5], dtype=np.float64)
    zero = np.zeros(1, dtype=np.float64)
    samples = sample_boundary_faces(
        box,
        coordinate,
        zero,
        zero,
        np.asarray([True]),
        0.25,
    )

    assert samples.directions.tolist() == [1]
    np.testing.assert_allclose(samples.approximation_errors, (0.0,), atol=1e-14)


def test_boundary_sample_owner_comes_from_outside_side_of_csg_seam() -> None:
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
    samples = sample_boundary_faces(
        root,
        np.asarray([0.4], dtype=np.float64),
        np.asarray([0.1], dtype=np.float64),
        np.asarray([-0.45], dtype=np.float64),
        np.asarray([True]),
        0.2,
    )

    negative_x = samples.directions == 0
    assert np.count_nonzero(negative_x) == 1
    assert samples.owner_object_ids[negative_x].tolist() == [
        obstacle.object_id
    ]


def test_fluid_domain_has_one_root_and_optional_tags() -> None:
    sphere = Sphere(name="fluid", radius=1.0, object_id=1)
    inlet = PlacedSDF2D(
        name="inlet",
        object_id=2,
        profile=CircleProfile(radius=0.5),
        origin=(-1.0, 0.0, 0.0),
        axis_u=(0.0, 1.0, 0.0),
        axis_v=(0.0, 0.0, 1.0),
    )
    domain = FluidDomain(root=sphere, tag_objects=(inlet,))
    config = MesherConfig(dx=0.1)
    assert domain.bounding_box().x_min == -1.0
    assert domain.tag_objects == (inlet,)
    assert config.n_levels == 0


def test_fluid_domain_accepts_2d_root_and_uses_four_neighbors() -> None:
    rectangle = PlacedSDF2D(
        name="fluid",
        object_id=1,
        profile=RectangleProfile(half_size=(1.0, 0.5)),
        axis_u=(0.0, 1.0, 0.0),
        axis_v=(0.0, 0.0, 1.0),
    )
    region = PlacedSDF1D(
        name="negative_u",
        object_id=2,
        profile=IntervalProfile(half_length=0.5),
        origin=(0.0, -1.0, 0.0),
        axis_u=(0.0, 0.0, 1.0),
    )
    domain = FluidDomain(rectangle, (region,))

    assert domain.root.dimension == 2
    assert len(domain.boundary_offsets()) == 4
    assert domain.boundary_offsets()[0] == (0.0, -1.0, 0.0)
    assert domain.boundary_offsets()[3] == (0.0, 0.0, 1.0)


def test_2d_fluid_domain_rejects_non_coplanar_1d_tags() -> None:
    root = PlacedSDF2D(
        name="fluid",
        object_id=1,
        profile=RectangleProfile(),
    )
    tag = PlacedSDF1D(
        name="tag",
        object_id=2,
        profile=IntervalProfile(),
        origin=(0.0, 0.0, 1.0),
    )

    with np.testing.assert_raises_regex(ValueError, "must lie"):
        FluidDomain(root, (tag,))


def test_refinement_levels_are_rejected() -> None:
    with np.testing.assert_raises_regex(ValueError, "n_levels must be 0"):
        MesherConfig(dx=0.1, n_levels=1)


def test_mesher_config_preserves_positional_n_levels_contract() -> None:
    with np.testing.assert_raises_regex(ValueError, "n_levels must be 0"):
        MesherConfig(0.1, 1)


def test_invalid_internal_preview_density_is_rejected() -> None:
    with np.testing.assert_raises_regex(ValueError, "between 0 and 1"):
        MesherConfig(dx=0.1, internal_preview_density=1.1)


def test_boundary_error_target_refines_dx_until_met(tmp_path) -> None:
    sphere = Sphere(name="sphere", object_id=1, radius=0.75)
    result = LatticeMesher(
        FluidDomain(sphere),
        MesherConfig(
            dx=0.5,
            boundary_error_tolerance=0.08,
            max_error_refinements=4,
            chunk_size=1_000,
        ),
    ).mesh(tmp_path / "error-refined.arrow")

    assert result.preview_cell_size == 0.0625
    assert result.refinement_count == 3
    assert result.boundary_error_tolerance_met
    assert result.boundary_error_maximum <= 0.08


def test_unreachable_boundary_error_target_fails_without_export(tmp_path) -> None:
    sphere = Sphere(name="sphere", object_id=1, radius=0.75)
    path = tmp_path / "error-unmet.arrow"

    with np.testing.assert_raises_regex(
        ValueError,
        "exceeds the requested",
    ):
        LatticeMesher(
            FluidDomain(sphere),
            MesherConfig(
                dx=0.5,
                boundary_error_tolerance=0.01,
                max_error_refinements=1,
                chunk_size=1_000,
            ),
        ).mesh(path)

    assert not path.exists()


def test_nearest_tag_mask_selects_one_lattice_layer() -> None:
    tag = PlacedSDF2D(
        name="section",
        object_id=2,
        profile=CircleProfile(radius=2.0),
        origin=(0.0, 0.0, 0.12),
    )
    grid = derive_grid(
        Box(name="box", object_id=1, half_size=(1.0, 1.0, 1.0)).bounding_box(),
        0.25,
    )
    chunks = list(generate_chunks(grid, grid.node_count))
    chunk = chunks[0]
    matched = nearest_tag_mask(
        tag,
        grid,
        chunk.i,
        chunk.j,
        chunk.k,
        chunk.x,
        chunk.y,
        chunk.z,
    )
    selected = np.column_stack((chunk.i[matched], chunk.j[matched], chunk.k[matched]))
    pairs = selected[:, :2]
    assert np.unique(pairs, axis=0).shape[0] == selected.shape[0]
    assert np.unique(selected[:, 2]).size == 1


def test_difference_attributes_cut_surface_to_subtractive_operand() -> None:
    box = Box(
        name="fluid",
        object_id=1,
        half_size=(1.0, 1.0, 1.0),
    )
    cylinder = Cylinder(
        name="obstacle",
        object_id=2,
        radius=0.25,
        half_height=1.0,
    )
    domain = Difference(
        name="cut",
        object_id=3,
        left=box,
        right=cylinder,
    )
    x = np.asarray([0.25, 1.0, 0.5], dtype=np.float64)
    zero = np.zeros(3, dtype=np.float64)
    _distance, source_ids = evaluate_with_attribution(domain, x, zero, zero)
    assert source_ids.tolist() == [2, 1, 2]
    volume_ids = evaluate_volume_attribution(domain, x, zero, zero)
    assert volume_ids.tolist() == [1, 1, 1]
    assert boundary_owner_ids(domain) == {1, 2}


def test_2d_difference_boundary_owners_include_subtractive_operand() -> None:
    rectangle = PlacedSDF2D(
        name="fluid",
        object_id=1,
        profile=RectangleProfile(half_size=(1.0, 0.5)),
    )
    circle = PlacedSDF2D(
        name="hole",
        object_id=2,
        profile=CircleProfile(radius=0.2),
    )
    domain = PlacedSDF2D(
        name="cut",
        object_id=3,
        profile=BinaryProfile(
            rectangle.profile,
            OffsetProfile(circle.profile),
            "difference",
        ),
        sources=(rectangle, circle),
    )

    assert boundary_owner_ids(domain) == {rectangle.object_id, circle.object_id}


def test_ray_pick_returns_controlling_boundary_owner() -> None:
    sphere = Sphere(name="sphere", object_id=7, radius=0.5)
    hit = pick_boundary_owner(
        sphere,
        np.asarray((2.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((-1.0, 0.0, 0.0), dtype=np.float64),
    )

    assert hit is not None
    point, owner_object_id, normal = hit
    assert np.isclose(point[0], 0.5)
    assert owner_object_id == sphere.object_id
    np.testing.assert_allclose(normal, (1.0, 0.0, 0.0), atol=1e-6)


def test_ray_pick_surface_returns_first_visible_point() -> None:
    sphere = Sphere(name="sphere", object_id=7, radius=1.0)
    hit = pick_sdf_surface(
        sphere,
        np.asarray((0.0, 0.0, 3.0), dtype=np.float64),
        np.asarray((0.0, 0.0, -1.0), dtype=np.float64),
    )

    assert hit is not None
    np.testing.assert_allclose(hit, (0.0, 0.0, 1.0), atol=0.001)


def test_fluid_domain_accepts_boundary_region_owner_tag() -> None:
    sphere = Sphere(name="sphere", object_id=1, radius=0.5)
    region = BoundaryRegion(
        name="wall",
        object_id=2,
        owner_object_id=sphere.object_id,
    )

    domain = FluidDomain(sphere, (region,))

    assert domain.tag_objects == (region,)


def test_union_volume_attribution_distinguishes_constituent_sdfs() -> None:
    left = Sphere(
        name="left",
        object_id=1,
        center=(-0.75, 0.0, 0.0),
        radius=1.0,
    )
    right = Sphere(
        name="right",
        object_id=2,
        center=(0.75, 0.0, 0.0),
        radius=1.0,
    )
    domain = Union(name="union", object_id=3, left=left, right=right)
    x = np.asarray((-1.0, 1.0), dtype=np.float64)
    zero = np.zeros(2, dtype=np.float64)
    assert evaluate_volume_attribution(domain, x, zero, zero).tolist() == [1, 2]
