from __future__ import annotations

import numpy as np

from app.viewport.renderer import (
    ROTATION_GIZMO_SEGMENTS,
    SDFRenderer,
    WORLD_AXIS_LENGTH,
    X_AXIS_COLOR,
    Y_AXIS_COLOR,
    Z_AXIS_COLOR,
)


def test_gizmo_uses_red_x_green_y_blue_z_axis_colors() -> None:
    labels = SDFRenderer._build_gizmo_labels()

    assert X_AXIS_COLOR == (1.0, 0.0, 0.0)
    assert Y_AXIS_COLOR == (0.0, 1.0, 0.0)
    assert Z_AXIS_COLOR == (0.1, 0.45, 1.0)
    assert np.allclose(labels[0, -3:], X_AXIS_COLOR)
    assert np.allclose(labels[4, -3:], Y_AXIS_COLOR)
    assert np.allclose(labels[10, -3:], Z_AXIS_COLOR)


def test_world_axis_draws_blue_z_axis_through_scene_origin() -> None:
    vertices = SDFRenderer._build_world_axis_vertices()

    assert vertices.shape == (2, 6)
    assert np.allclose(vertices[:, :2], 0.0)
    assert np.allclose(vertices[:, 2], (-WORLD_AXIS_LENGTH, WORLD_AXIS_LENGTH))
    assert np.allclose(vertices[:, 3:], Z_AXIS_COLOR)


def test_rotation_gizmo_builds_xyz_rings_around_center() -> None:
    vertices = SDFRenderer.build_rotation_gizmo_vertices((1.0, 2.0, 3.0), 0.5)

    assert vertices.shape == (3 * ROTATION_GIZMO_SEGMENTS * 2, 6)
    assert np.allclose(vertices[0, 0], 1.0)
    assert np.allclose(vertices[0, 3:], X_AXIS_COLOR)
    y_start = ROTATION_GIZMO_SEGMENTS * 2
    z_start = ROTATION_GIZMO_SEGMENTS * 4
    assert np.allclose(vertices[y_start, 3:], Y_AXIS_COLOR)
    assert np.allclose(vertices[z_start, 3:], Z_AXIS_COLOR)


def test_lattice_colors_distinguish_objects_and_keep_fluid_blue() -> None:
    colors = SDFRenderer._lattice_colors(
        node_types=np.asarray((0, 0, 1, 1), dtype=np.uint8),
        source_object_ids=np.asarray((1, 2, 1, 2), dtype=np.uint16),
        primary_tag_ids=np.asarray((0, 7, 0, 0), dtype=np.uint16),
    )
    assert np.allclose(colors[0], (0.12, 0.42, 1.00))
    assert not np.allclose(colors[1], colors[0])
    assert not np.allclose(colors[2], colors[3])


def test_lattice_upload_preparation_packs_points_and_squares() -> None:
    positions = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float32,
    )
    point_vertices, square_instances = SDFRenderer.prepare_lattice_upload(
        positions,
        np.ones(4, dtype=np.uint8),
        np.ones(4, dtype=np.uint8),
        np.ones(4, dtype=np.uint16),
        np.zeros(4, dtype=np.uint16),
        1.0,
    )

    assert point_vertices.shape == (4, 7)
    assert square_instances.shape == (1, 12)
    assert np.all(point_vertices[:, -1] == 5.0)


def test_boundary_square_uses_four_lattice_points_as_vertices() -> None:
    positions = np.asarray(
        (
            (0.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.0, 1.0, 1.0),
        ),
        dtype=np.float32,
    )
    boundary_faces = np.full(4, 1, dtype=np.uint8)
    colors = np.tile(
        np.asarray((1.0, 0.35, 0.18), dtype=np.float32),
        (4, 1),
    )

    instances = SDFRenderer._build_boundary_square_instances(
        positions,
        boundary_faces,
        colors,
        cell_size=1.0,
    )

    assert instances.shape == (1, 12)
    center = instances[0, :3]
    axis_u = instances[0, 6:9]
    axis_v = instances[0, 9:12]
    corners = np.asarray(
        [
            center + sign_u * 0.5 * axis_u + sign_v * 0.5 * axis_v
            for sign_u, sign_v in ((-1, -1), (1, -1), (1, 1), (-1, 1))
        ]
    )
    assert {
        tuple(point) for point in np.round(corners, decimals=6)
    } == {
        tuple(point) for point in positions
    }
    assert np.allclose(instances[0, 3:6], colors[0])


def test_incomplete_boundary_vertices_do_not_create_square() -> None:
    positions = np.asarray(
        ((0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        dtype=np.float32,
    )
    instances = SDFRenderer._build_boundary_square_instances(
        positions,
        np.full(3, 1, dtype=np.uint8),
        np.ones((3, 3), dtype=np.float32),
        cell_size=1.0,
    )
    assert instances.shape == (0, 12)


def test_2d_boundary_nodes_do_not_create_cell_squares() -> None:
    positions = np.asarray(
        (
            (-0.5, 0.0, 0.0),
            (0.0, -0.5, 0.0),
            (0.5, 0.0, 0.0),
        ),
        dtype=np.float32,
    )
    colors = np.tile(
        np.asarray((1.0, 0.35, 0.18), dtype=np.float32),
        (3, 1),
    )

    instances = SDFRenderer._build_boundary_square_instances(
        positions,
        np.asarray((1, 0, 2), dtype=np.uint8),
        colors,
        cell_size=0.25,
        dimension=2,
        axis_i=(1.0, 0.0, 0.0),
        axis_j=(0.0, 1.0, 0.0),
    )

    assert instances.shape == (0, 12)
