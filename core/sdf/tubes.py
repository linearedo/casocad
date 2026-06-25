from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .base import BoundingBox3D, FloatArray, SDFNode


Point3D = tuple[float, float, float]
CapStyle = Literal["round", "flat"]


def _as_points(
    points: tuple[Point3D, ...] | list[Point3D] | list[list[float]],
) -> tuple[Point3D, ...]:
    return tuple(
        (float(point[0]), float(point[1]), float(point[2]))
        for point in points
    )


def _validate_tube_radius(radius: float, inner_radius: float) -> None:
    if radius <= 0.0 or not np.isfinite(radius):
        raise ValueError("tube radius must be finite and positive")
    if inner_radius < 0.0 or not np.isfinite(inner_radius):
        raise ValueError("tube inner radius must be finite and non-negative")
    if inner_radius >= radius:
        raise ValueError("tube inner radius must be smaller than radius")


def _validate_caps(caps: str) -> None:
    if caps not in {"round", "flat"}:
        raise ValueError("tube caps must be 'round' or 'flat'")


def _tube_signed_distance(
    centerline_distance: FloatArray,
    radius: float,
    inner_radius: float,
) -> FloatArray:
    outer = centerline_distance - radius
    if inner_radius <= 0.0:
        return np.asarray(outer, dtype=np.float64)
    return np.asarray(np.maximum(outer, inner_radius - centerline_distance), dtype=np.float64)


def _exact_extrusion(profile_distance: FloatArray, axial: FloatArray) -> FloatArray:
    outside = np.sqrt(
        np.maximum(profile_distance, 0.0) ** 2 + np.maximum(axial, 0.0) ** 2
    )
    inside = np.minimum(np.maximum(profile_distance, axial), 0.0)
    return np.asarray(outside + inside, dtype=np.float64)


def _segment_distance_numpy(
    X: FloatArray,
    Y: FloatArray,
    Z: FloatArray,
    first: Point3D,
    second: Point3D,
) -> FloatArray:
    ax, ay, az = first
    bx, by, bz = second
    bax = bx - ax
    bay = by - ay
    baz = bz - az
    denominator = bax * bax + bay * bay + baz * baz
    if denominator <= 1.0e-24:
        return np.asarray(
            np.sqrt((X - ax) ** 2 + (Y - ay) ** 2 + (Z - az) ** 2),
            dtype=np.float64,
        )
    h = np.clip(
        ((X - ax) * bax + (Y - ay) * bay + (Z - az) * baz) / denominator,
        0.0,
        1.0,
    )
    dx = X - ax - h * bax
    dy = Y - ay - h * bay
    dz = Z - az - h * baz
    return np.asarray(np.sqrt(dx * dx + dy * dy + dz * dz), dtype=np.float64)


def _flat_capped_segment_tube_numpy(
    X: FloatArray,
    Y: FloatArray,
    Z: FloatArray,
    first: Point3D,
    second: Point3D,
    radius: float,
) -> FloatArray:
    ax, ay, az = first
    bx, by, bz = second
    bax = bx - ax
    bay = by - ay
    baz = bz - az
    length = float(np.sqrt(bax * bax + bay * bay + baz * baz))
    if length <= 1.0e-12:
        return np.asarray(
            np.sqrt((X - ax) ** 2 + (Y - ay) ** 2 + (Z - az) ** 2) - radius,
            dtype=np.float64,
        )
    direction = (bax / length, bay / length, baz / length)
    projection = (
        (X - ax) * direction[0]
        + (Y - ay) * direction[1]
        + (Z - az) * direction[2]
    )
    radial_x = X - ax - projection * direction[0]
    radial_y = Y - ay - projection * direction[1]
    radial_z = Z - az - projection * direction[2]
    radial = np.sqrt(radial_x * radial_x + radial_y * radial_y + radial_z * radial_z)
    profile = radial - radius
    axial = np.abs(projection - 0.5 * length) - 0.5 * length
    return _exact_extrusion(profile, axial)


def _quadratic_bezier_spans(
    points: tuple[Point3D, ...],
) -> tuple[tuple[Point3D, Point3D, Point3D], ...]:
    return tuple(
        (points[index], points[index + 1], points[index + 2])
        for index in range(0, len(points) - 2, 2)
    )


def _quadratic_bezier_distance_numpy(
    X: FloatArray,
    Y: FloatArray,
    Z: FloatArray,
    start: Point3D,
    control: Point3D,
    end: Point3D,
) -> FloatArray:
    ax, ay, az = start
    bx, by, bz = control
    cx, cy, cz = end
    a_x = bx - ax
    a_y = by - ay
    a_z = bz - az
    b_x = ax - 2.0 * bx + cx
    b_y = ay - 2.0 * by + cy
    b_z = az - 2.0 * bz + cz
    c_x = 2.0 * a_x
    c_y = 2.0 * a_y
    c_z = 2.0 * a_z
    b_dot_b = b_x * b_x + b_y * b_y + b_z * b_z
    if b_dot_b <= 1.0e-24:
        return _segment_distance_numpy(X, Y, Z, start, end)

    d_x = ax - X
    d_y = ay - Y
    d_z = az - Z
    kk = 1.0 / b_dot_b
    kx = kk * (a_x * b_x + a_y * b_y + a_z * b_z)
    ky = kk * (
        2.0 * (a_x * a_x + a_y * a_y + a_z * a_z)
        + d_x * b_x
        + d_y * b_y
        + d_z * b_z
    ) / 3.0
    kz = kk * (d_x * a_x + d_y * a_y + d_z * a_z)
    p = ky - kx * kx
    q = kx * (2.0 * kx * kx - 3.0 * ky) + kz
    h = q * q + 4.0 * p * p * p
    result = np.full(np.shape(X), np.inf, dtype=np.float64)

    single_root = h >= 0.0
    if np.any(single_root):
        h_root = np.sqrt(np.maximum(h[single_root], 0.0))
        x_0 = 0.5 * (h_root - q[single_root])
        x_1 = 0.5 * (-h_root - q[single_root])
        t = np.clip(np.cbrt(x_0) + np.cbrt(x_1) - kx, 0.0, 1.0)
        w_x = d_x[single_root] + (c_x + b_x * t) * t
        w_y = d_y[single_root] + (c_y + b_y * t) * t
        w_z = d_z[single_root] + (c_z + b_z * t) * t
        result[single_root] = w_x * w_x + w_y * w_y + w_z * w_z

    three_roots = ~single_root
    if np.any(three_roots):
        p_values = p[three_roots]
        q_values = q[three_roots]
        z = np.sqrt(np.maximum(-p_values, 0.0))
        denominator = 2.0 * p_values * z
        angle_argument = np.divide(
            q_values,
            denominator,
            out=np.zeros_like(q_values),
            where=np.abs(denominator) > 1.0e-24,
        )
        angle = np.arccos(np.clip(angle_argument, -1.0, 1.0)) / 3.0
        m = np.cos(angle)
        n = np.sin(angle) * 1.732050808
        t_0 = np.clip((m + m) * z - kx, 0.0, 1.0)
        t_1 = np.clip((-n - m) * z - kx, 0.0, 1.0)
        d_x_values = d_x[three_roots]
        d_y_values = d_y[three_roots]
        d_z_values = d_z[three_roots]
        w_0_x = d_x_values + (c_x + b_x * t_0) * t_0
        w_0_y = d_y_values + (c_y + b_y * t_0) * t_0
        w_0_z = d_z_values + (c_z + b_z * t_0) * t_0
        w_1_x = d_x_values + (c_x + b_x * t_1) * t_1
        w_1_y = d_y_values + (c_y + b_y * t_1) * t_1
        w_1_z = d_z_values + (c_z + b_z * t_1) * t_1
        result[three_roots] = np.minimum(
            w_0_x * w_0_x + w_0_y * w_0_y + w_0_z * w_0_z,
            w_1_x * w_1_x + w_1_y * w_1_y + w_1_z * w_1_z,
        )

    return np.asarray(np.sqrt(np.maximum(result, 0.0)), dtype=np.float64)


def _unit_vector(first: Point3D, second: Point3D) -> np.ndarray:
    vector = np.asarray(second, dtype=np.float64) - np.asarray(first, dtype=np.float64)
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-12 or not np.isfinite(length):
        raise ValueError("tube cap tangent must be finite and nonzero")
    return vector / length


def _quadratic_bezier_endpoint_tangents(points: tuple[Point3D, ...]) -> tuple[np.ndarray, np.ndarray]:
    start = points[0]
    first_control = points[1]
    first_end = points[2]
    end = points[-1]
    last_control = points[-2]
    last_start = points[-3]
    try:
        start_tangent = _unit_vector(start, first_control)
    except ValueError:
        start_tangent = _unit_vector(start, first_end)
    try:
        end_tangent = _unit_vector(last_control, end)
    except ValueError:
        end_tangent = _unit_vector(last_start, end)
    return start_tangent, end_tangent


def _flat_cap_planes_numpy(
    X: FloatArray,
    Y: FloatArray,
    Z: FloatArray,
    start: Point3D,
    start_tangent: np.ndarray,
    end: Point3D,
    end_tangent: np.ndarray,
) -> tuple[FloatArray, FloatArray]:
    start_plane = (
        (start[0] - X) * start_tangent[0]
        + (start[1] - Y) * start_tangent[1]
        + (start[2] - Z) * start_tangent[2]
    )
    end_plane = (
        (X - end[0]) * end_tangent[0]
        + (Y - end[1]) * end_tangent[1]
        + (Z - end[2]) * end_tangent[2]
    )
    return np.asarray(start_plane, dtype=np.float64), np.asarray(end_plane, dtype=np.float64)


def _points_bounds(points: tuple[Point3D, ...], radius: float) -> BoundingBox3D:
    array = np.asarray(points, dtype=np.float64)
    minimum = array.min(axis=0) - radius
    maximum = array.max(axis=0) + radius
    return BoundingBox3D(
        minimum[0],
        maximum[0],
        minimum[1],
        maximum[1],
        minimum[2],
        maximum[2],
    )


@dataclass
class PolylineTube(SDFNode):
    points: tuple[Point3D, ...] = (
        (-0.75, 0.0, 0.0),
        (0.0, 0.5, 0.0),
        (0.75, 0.0, 0.0),
    )
    radius: float = 0.12
    inner_radius: float = 0.0
    caps: CapStyle = "round"

    def __post_init__(self) -> None:
        points = _as_points(self.points)
        if len(points) < 2:
            raise ValueError("polyline tube requires at least two points")
        if all(
            np.linalg.norm(np.asarray(second) - np.asarray(first)) <= 1.0e-12
            for first, second in zip(points, points[1:])
        ):
            raise ValueError("polyline tube requires at least one nonzero segment")
        _validate_tube_radius(self.radius, self.inner_radius)
        _validate_caps(self.caps)
        self.points = points

    @property
    def dimension(self) -> int:
        return 3

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        pairs = zip(self.points, self.points[1:])
        centerline_distances = [
            _segment_distance_numpy(X, Y, Z, first, second)
            for first, second in pairs
        ]
        centerline = np.asarray(np.minimum.reduce(centerline_distances), dtype=np.float64)
        if self.caps == "round":
            return _tube_signed_distance(centerline, self.radius, self.inner_radius)
        outer_distances = [
            _flat_capped_segment_tube_numpy(X, Y, Z, first, second, self.radius)
            for first, second in zip(self.points, self.points[1:])
        ]
        outer = np.asarray(np.minimum.reduce(outer_distances), dtype=np.float64)
        if self.inner_radius <= 0.0:
            return outer
        return np.asarray(np.maximum(outer, self.inner_radius - centerline), dtype=np.float64)

    def bounding_box(self) -> BoundingBox3D:
        return _points_bounds(self.points, self.radius)


@dataclass
class QuadraticBezierTube(SDFNode):
    points: tuple[Point3D, ...] = (
        (-0.75, 0.0, 0.0),
        (0.0, 0.55, 0.0),
        (0.75, 0.0, 0.0),
    )
    radius: float = 0.12
    inner_radius: float = 0.0
    caps: CapStyle = "round"

    def __post_init__(self) -> None:
        points = _as_points(self.points)
        if len(points) < 3:
            raise ValueError("quadratic Bezier tube requires at least three points")
        if len(points) % 2 == 0:
            raise ValueError(
                "quadratic Bezier tube requires an odd point count: anchor, control, anchor"
            )
        if all(
            np.linalg.norm(np.asarray(control) - np.asarray(start)) <= 1.0e-12
            and np.linalg.norm(np.asarray(end) - np.asarray(start)) <= 1.0e-12
            for start, control, end in _quadratic_bezier_spans(points)
        ):
            raise ValueError("quadratic Bezier tube requires at least one nonzero span")
        _validate_tube_radius(self.radius, self.inner_radius)
        _validate_caps(self.caps)
        if self.caps == "flat":
            _quadratic_bezier_endpoint_tangents(points)
        self.points = points

    @property
    def dimension(self) -> int:
        return 3

    @property
    def kind(self) -> str:
        return "quadratic_bezier_polycurve_tube" if len(self.points) > 3 else "quadratic_bezier_tube"

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        distances = [
            _quadratic_bezier_distance_numpy(X, Y, Z, start, control, end)
            for start, control, end in _quadratic_bezier_spans(self.points)
        ]
        centerline = np.asarray(np.minimum.reduce(distances), dtype=np.float64)
        outer = centerline - self.radius
        if self.caps == "flat":
            start_tangent, end_tangent = _quadratic_bezier_endpoint_tangents(self.points)
            start_plane, end_plane = _flat_cap_planes_numpy(
                X,
                Y,
                Z,
                self.points[0],
                start_tangent,
                self.points[-1],
                end_tangent,
            )
            outer = np.maximum.reduce((outer, start_plane, end_plane))
        if self.inner_radius <= 0.0:
            return np.asarray(outer, dtype=np.float64)
        return np.asarray(np.maximum(outer, self.inner_radius - centerline), dtype=np.float64)

    def bounding_box(self) -> BoundingBox3D:
        return _points_bounds(self.points, self.radius)
