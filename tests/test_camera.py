from __future__ import annotations

import numpy as np

from app.viewport.camera import OrbitCamera


def _matrix_project(
    camera: OrbitCamera, point: np.ndarray, aspect_ratio: float
) -> np.ndarray:
    clip = camera.view_projection(aspect_ratio).astype(np.float64) @ np.append(
        point, 1.0
    )
    return clip[:2] / clip[3]


def _raymarch_project(
    camera: OrbitCamera, point: np.ndarray, aspect_ratio: float
) -> np.ndarray:
    eye = np.asarray(camera.position, dtype=np.float64)
    target = np.asarray(camera.target, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0)))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    to_point = point - eye
    depth = np.dot(to_point, forward)
    screen_uv = camera.focal_length * np.asarray(
        (np.dot(to_point, right), np.dot(to_point, up))
    ) / (2.0 * depth)
    return np.asarray(
        (2.0 * screen_uv[0] / aspect_ratio, 2.0 * screen_uv[1])
    )


def test_raymarch_and_matrix_projection_share_focal_length() -> None:
    camera = OrbitCamera(field_of_view_degrees=45.0)
    eye = np.asarray(camera.position, dtype=np.float64)
    target = np.asarray(camera.target, dtype=np.float64)
    forward = target - eye
    forward /= np.linalg.norm(forward)
    camera_up = camera.view_rotation()[1].astype(np.float64)
    point_one_unit_above_target = target + camera_up
    view_projection = camera.view_projection(1.0).astype(np.float64)
    clip = view_projection @ np.append(point_one_unit_above_target, 1.0)
    projected_y = clip[1] / clip[3]
    expected_y = camera.focal_length / np.linalg.norm(target - eye)
    assert np.isclose(projected_y, expected_y, rtol=1e-5)
    assert np.isclose(camera.focal_length, 1.0 / np.tan(np.pi / 8.0))


def test_pan_keeps_raymarch_and_grid_projection_aligned() -> None:
    camera = OrbitCamera()
    point = np.asarray((0.4, 0.15, -0.2), dtype=np.float64)
    aspect_ratio = 16.0 / 9.0
    before_matrix = _matrix_project(camera, point, aspect_ratio)
    before_raymarch = _raymarch_project(camera, point, aspect_ratio)
    camera.pan(120.0, -55.0)
    after_matrix = _matrix_project(camera, point, aspect_ratio)
    after_raymarch = _raymarch_project(camera, point, aspect_ratio)
    assert np.allclose(before_matrix, before_raymarch, rtol=1e-5, atol=1e-6)
    assert np.allclose(after_matrix, after_raymarch, rtol=1e-5, atol=1e-6)
    assert np.allclose(
        after_matrix - before_matrix,
        after_raymarch - before_raymarch,
        rtol=1e-5,
        atol=1e-6,
    )


def test_screen_to_ground_hits_reference_plane() -> None:
    camera = OrbitCamera()
    point = camera.screen_to_ground(400.0, 300.0, 800.0, 600.0)
    assert point is not None
    assert np.isclose(point[2], 0.0)
    assert np.allclose(point, camera.target, atol=1e-6)


def test_screen_to_reference_planes_hits_target() -> None:
    camera = OrbitCamera(target=(0.25, -0.4, 0.75))

    for plane, axis in (("xy", 2), ("xz", 1), ("yz", 0)):
        camera.set_plane_view(plane)
        point = camera.screen_to_plane(plane, 400.0, 300.0, 800.0, 600.0)

        assert point is not None
        assert np.isclose(point[axis], 0.0)


def test_planar_camera_views_have_valid_projection() -> None:
    camera = OrbitCamera()

    for plane in ("xy", "xz", "yz"):
        camera.set_plane_view(plane)
        projection = camera.view_projection(1.0)
        rotation = camera.view_rotation()

        assert np.all(np.isfinite(projection))
        assert np.all(np.isfinite(rotation))


def test_xz_view_places_positive_x_on_screen_right() -> None:
    camera = OrbitCamera()
    camera.set_plane_view("xz")

    left = _matrix_project(camera, np.asarray((-0.5, 0.0, 0.0)), 1.0)
    right = _matrix_project(camera, np.asarray((0.5, 0.0, 0.0)), 1.0)

    assert right[0] > left[0]


def test_orbit_from_top_plane_does_not_snap_far_from_plane() -> None:
    camera = OrbitCamera()
    camera.set_plane_view("xy")

    camera.orbit(12.0, 0.0)

    assert camera.pitch_degrees > 89.0
