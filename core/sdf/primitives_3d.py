from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BoundingBox3D, FloatArray, SDFNode, glsl_float, glsl_vec3


def _normalized(vector: tuple[float, float, float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    length = np.linalg.norm(array)
    if length <= 1e-12:
        raise ValueError("orientation axis must be nonzero")
    return array / length


def _orthonormal_frame(
    axis_u: tuple[float, float, float],
    axis_v: tuple[float, float, float],
    axis_w: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    u = _normalized(axis_u)
    v = _normalized(axis_v)
    w = _normalized(axis_w)
    if (
        abs(float(np.dot(u, v))) > 1e-6
        or abs(float(np.dot(u, w))) > 1e-6
        or abs(float(np.dot(v, w))) > 1e-6
    ):
        raise ValueError("orientation axes must be orthogonal")
    return (
        tuple(float(value) for value in u),
        tuple(float(value) for value in v),
        tuple(float(value) for value in w),
    )


def _oriented_local_glsl(
    p_var: str,
    center: tuple[float, float, float],
    axis_u: tuple[float, float, float],
    axis_v: tuple[float, float, float],
    axis_w: tuple[float, float, float],
) -> str:
    local = f"({p_var} - {glsl_vec3(center)})"
    return (
        "vec3("
        f"dot({local}, {glsl_vec3(axis_u)}), "
        f"dot({local}, {glsl_vec3(axis_v)}), "
        f"dot({local}, {glsl_vec3(axis_w)})"
        ")"
    )


def _oriented_local_numpy(
    X: FloatArray,
    Y: FloatArray,
    Z: FloatArray,
    center: tuple[float, float, float],
    axis_u: tuple[float, float, float],
    axis_v: tuple[float, float, float],
    axis_w: tuple[float, float, float],
) -> tuple[FloatArray, FloatArray, FloatArray]:
    rx = X - center[0]
    ry = Y - center[1]
    rz = Z - center[2]
    u = np.asarray(axis_u, dtype=np.float64)
    v = np.asarray(axis_v, dtype=np.float64)
    w = np.asarray(axis_w, dtype=np.float64)
    return (
        np.asarray(rx * u[0] + ry * u[1] + rz * u[2], dtype=np.float64),
        np.asarray(rx * v[0] + ry * v[1] + rz * v[2], dtype=np.float64),
        np.asarray(rx * w[0] + ry * w[1] + rz * w[2], dtype=np.float64),
    )


def _oriented_box_bounds(
    center: tuple[float, float, float],
    axis_u: tuple[float, float, float],
    axis_v: tuple[float, float, float],
    axis_w: tuple[float, float, float],
    half_size: tuple[float, float, float],
) -> BoundingBox3D:
    center_array = np.asarray(center, dtype=np.float64)
    axes = (
        np.asarray(axis_u, dtype=np.float64),
        np.asarray(axis_v, dtype=np.float64),
        np.asarray(axis_w, dtype=np.float64),
    )
    corners = np.asarray(
        [
            center_array
            + sign_u * half_size[0] * axes[0]
            + sign_v * half_size[1] * axes[1]
            + sign_w * half_size[2] * axes[2]
            for sign_u in (-1.0, 1.0)
            for sign_v in (-1.0, 1.0)
            for sign_w in (-1.0, 1.0)
        ],
        dtype=np.float64,
    )
    minimum = corners.min(axis=0)
    maximum = corners.max(axis=0)
    return BoundingBox3D(
        minimum[0], maximum[0], minimum[1], maximum[1], minimum[2], maximum[2]
    )


@dataclass
class Sphere(SDFNode):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.5

    @property
    def dimension(self) -> int:
        return 3

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("sphere radius must be positive")

    def to_glsl(self, p_var: str = "p") -> str:
        return (
            f"(length({p_var} - {glsl_vec3(self.center)})"
            f" - {glsl_float(self.radius)})"
        )

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        cx, cy, cz = self.center
        return np.asarray(
            np.sqrt((X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2)
            - self.radius,
            dtype=np.float64,
        )

    def bounding_box(self) -> BoundingBox3D:
        cx, cy, cz = self.center
        r = self.radius
        return BoundingBox3D(cx - r, cx + r, cy - r, cy + r, cz - r, cz + r)


@dataclass
class Box(SDFNode):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_size: tuple[float, float, float] = (0.5, 0.5, 0.5)
    axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0)
    axis_v: tuple[float, float, float] = (0.0, 1.0, 0.0)
    axis_w: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def dimension(self) -> int:
        return 3

    def __post_init__(self) -> None:
        if any(value <= 0.0 for value in self.half_size):
            raise ValueError("box half sizes must be positive")
        self.axis_u, self.axis_v, self.axis_w = _orthonormal_frame(
            self.axis_u,
            self.axis_v,
            self.axis_w,
        )

    def to_glsl(self, p_var: str = "p") -> str:
        local = _oriented_local_glsl(
            p_var,
            self.center,
            self.axis_u,
            self.axis_v,
            self.axis_w,
        )
        q = f"(abs({local}) - {glsl_vec3(self.half_size)})"
        return f"(length(max({q}, vec3(0.0))) + min(max({q}.x, max({q}.y, {q}.z)), 0.0))"

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        half_size = np.asarray(self.half_size, dtype=np.float64)
        local_x, local_y, local_z = _oriented_local_numpy(
            X,
            Y,
            Z,
            self.center,
            self.axis_u,
            self.axis_v,
            self.axis_w,
        )
        qx = np.abs(local_x) - half_size[0]
        qy = np.abs(local_y) - half_size[1]
        qz = np.abs(local_z) - half_size[2]
        outside = np.sqrt(
            np.maximum(qx, 0.0) ** 2
            + np.maximum(qy, 0.0) ** 2
            + np.maximum(qz, 0.0) ** 2
        )
        inside = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0.0)
        return np.asarray(outside + inside, dtype=np.float64)

    def bounding_box(self) -> BoundingBox3D:
        return _oriented_box_bounds(
            self.center,
            self.axis_u,
            self.axis_v,
            self.axis_w,
            self.half_size,
        )


@dataclass
class Cylinder(SDFNode):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.5
    half_height: float = 0.5
    axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0)
    axis_v: tuple[float, float, float] = (0.0, 1.0, 0.0)
    axis_w: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def dimension(self) -> int:
        return 3

    def __post_init__(self) -> None:
        if self.radius <= 0.0 or self.half_height <= 0.0:
            raise ValueError("cylinder dimensions must be positive")
        self.axis_u, self.axis_v, self.axis_w = _orthonormal_frame(
            self.axis_u,
            self.axis_v,
            self.axis_w,
        )

    def to_glsl(self, p_var: str = "p") -> str:
        local = _oriented_local_glsl(
            p_var,
            self.center,
            self.axis_u,
            self.axis_v,
            self.axis_w,
        )
        d = (
            f"(abs(vec2(length({local}.xy), {local}.z))"
            f" - vec2({glsl_float(self.radius)}, {glsl_float(self.half_height)}))"
        )
        return f"(min(max({d}.x, {d}.y), 0.0) + length(max({d}, vec2(0.0))))"

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        local_x, local_y, local_z = _oriented_local_numpy(
            X,
            Y,
            Z,
            self.center,
            self.axis_u,
            self.axis_v,
            self.axis_w,
        )
        radial = np.sqrt(local_x**2 + local_y**2) - self.radius
        axial = np.abs(local_z) - self.half_height
        outside = np.sqrt(np.maximum(radial, 0.0) ** 2 + np.maximum(axial, 0.0) ** 2)
        inside = np.minimum(np.maximum(radial, axial), 0.0)
        return np.asarray(outside + inside, dtype=np.float64)

    def bounding_box(self) -> BoundingBox3D:
        return _oriented_box_bounds(
            self.center,
            self.axis_u,
            self.axis_v,
            self.axis_w,
            (self.radius, self.radius, self.half_height),
        )


@dataclass
class Torus(SDFNode):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    major_radius: float = 0.5
    minor_radius: float = 0.15
    axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0)
    axis_v: tuple[float, float, float] = (0.0, 1.0, 0.0)
    axis_w: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def dimension(self) -> int:
        return 3

    def __post_init__(self) -> None:
        if self.major_radius <= 0.0 or self.minor_radius <= 0.0:
            raise ValueError("torus radii must be positive")
        self.axis_u, self.axis_v, self.axis_w = _orthonormal_frame(
            self.axis_u,
            self.axis_v,
            self.axis_w,
        )

    def to_glsl(self, p_var: str = "p") -> str:
        local = _oriented_local_glsl(
            p_var,
            self.center,
            self.axis_u,
            self.axis_v,
            self.axis_w,
        )
        return (
            f"(length(vec2(length({local}.xy) - {glsl_float(self.major_radius)},"
            f" {local}.z)) - {glsl_float(self.minor_radius)})"
        )

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        local_x, local_y, local_z = _oriented_local_numpy(
            X,
            Y,
            Z,
            self.center,
            self.axis_u,
            self.axis_v,
            self.axis_w,
        )
        qx = np.sqrt(local_x**2 + local_y**2) - self.major_radius
        return np.asarray(
            np.sqrt(qx**2 + local_z**2) - self.minor_radius,
            dtype=np.float64,
        )

    def bounding_box(self) -> BoundingBox3D:
        outer = self.major_radius + self.minor_radius
        return _oriented_box_bounds(
            self.center,
            self.axis_u,
            self.axis_v,
            self.axis_w,
            (outer, outer, self.minor_radius),
        )
