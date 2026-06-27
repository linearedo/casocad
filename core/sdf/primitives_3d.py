from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BoundingBox3D, FloatArray, SDFNode


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
    if (
        axis_u == (1.0, 0.0, 0.0)
        and axis_v == (0.0, 1.0, 0.0)
        and axis_w == (0.0, 0.0, 1.0)
    ):
        cx, cy, cz = center
        hx, hy, hz = half_size
        return BoundingBox3D(cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz)
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
class Cone(SDFNode):
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
            raise ValueError("cone dimensions must be positive")
        self.axis_u, self.axis_v, self.axis_w = _orthonormal_frame(
            self.axis_u,
            self.axis_v,
            self.axis_w,
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
        wx = np.sqrt(local_x**2 + local_y**2)
        wy = local_z - self.half_height
        qx = self.radius
        qy = -2.0 * self.half_height
        denominator = qx * qx + qy * qy
        h = np.clip((wx * qx + wy * qy) / denominator, 0.0, 1.0)
        ax = wx - qx * h
        ay = wy - qy * h
        bx = wx - qx * np.clip(wx / qx, 0.0, 1.0)
        by = wy - qy
        d = np.minimum(ax * ax + ay * ay, bx * bx + by * by)
        s = np.maximum(-(wx * qy - wy * qx), -(wy - qy))
        return np.asarray(np.sqrt(d) * np.sign(s), dtype=np.float64)

    def bounding_box(self) -> BoundingBox3D:
        return _oriented_box_bounds(
            self.center,
            self.axis_u,
            self.axis_v,
            self.axis_w,
            (self.radius, self.radius, self.half_height),
        )


@dataclass
class CappedCone(SDFNode):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius_a: float = 0.5
    radius_b: float = 0.25
    half_height: float = 0.5
    axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0)
    axis_v: tuple[float, float, float] = (0.0, 1.0, 0.0)
    axis_w: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def dimension(self) -> int:
        return 3

    def __post_init__(self) -> None:
        if self.radius_a <= 0.0 or self.radius_b <= 0.0 or self.half_height <= 0.0:
            raise ValueError("capped cone dimensions must be positive")
        self.axis_u, self.axis_v, self.axis_w = _orthonormal_frame(
            self.axis_u,
            self.axis_v,
            self.axis_w,
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
        qx = np.sqrt(local_x**2 + local_y**2)
        qy = local_z
        k1x = self.radius_b
        k1y = self.half_height
        k2x = self.radius_b - self.radius_a
        k2y = 2.0 * self.half_height
        cax = qx - np.minimum(qx, np.where(qy < 0.0, self.radius_a, self.radius_b))
        cay = np.abs(qy) - self.half_height
        dot_k2 = k2x * k2x + k2y * k2y
        projection = ((k1x - qx) * k2x + (k1y - qy) * k2y) / dot_k2
        f = np.clip(projection, 0.0, 1.0)
        cbx = qx - k1x + k2x * f
        cby = qy - k1y + k2y * f
        sign = np.where((cbx < 0.0) & (cay < 0.0), -1.0, 1.0)
        distance_squared = np.minimum(
            cax * cax + cay * cay,
            cbx * cbx + cby * cby,
        )
        return np.asarray(sign * np.sqrt(distance_squared), dtype=np.float64)

    def bounding_box(self) -> BoundingBox3D:
        radius = max(self.radius_a, self.radius_b)
        return _oriented_box_bounds(
            self.center,
            self.axis_u,
            self.axis_v,
            self.axis_w,
            (radius, radius, self.half_height),
        )


@dataclass
class Pyramid(SDFNode):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    base_half_size: float = 0.5
    half_height: float = 0.5
    axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0)
    axis_v: tuple[float, float, float] = (0.0, 1.0, 0.0)
    axis_w: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def dimension(self) -> int:
        return 3

    def __post_init__(self) -> None:
        if self.base_half_size <= 0.0 or self.half_height <= 0.0:
            raise ValueError("pyramid dimensions must be positive")
        self.axis_u, self.axis_v, self.axis_w = _orthonormal_frame(
            self.axis_u,
            self.axis_v,
            self.axis_w,
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
        scale = 2.0 * self.base_half_size
        px = np.abs(local_x / scale)
        py = (local_z + self.half_height) / scale
        pz = np.abs(local_y / scale)
        swap = pz > px
        old_px = px
        px = np.where(swap, pz, px)
        pz = np.where(swap, old_px, pz)
        px = px - 0.5
        pz = pz - 0.5
        h = 2.0 * self.half_height / scale
        m2 = h * h + 0.25
        qx = pz
        qy = h * py - 0.5 * px
        qz = h * px + 0.5 * py
        s = np.maximum(-qx, 0.0)
        t = np.clip((qy - 0.5 * pz) / (m2 + 0.25), 0.0, 1.0)
        a = m2 * (qx + s) * (qx + s) + qy * qy
        b = m2 * (qx + 0.5 * t) * (qx + 0.5 * t) + (qy - m2 * t) * (qy - m2 * t)
        d2 = np.where(
            np.minimum(qy, -qx * m2 - qy * 0.5) > 0.0,
            0.0,
            np.minimum(a, b),
        )
        return np.asarray(
            scale * np.sqrt((d2 + qz * qz) / m2) * np.sign(np.maximum(qz, -py)),
            dtype=np.float64,
        )

    def bounding_box(self) -> BoundingBox3D:
        return _oriented_box_bounds(
            self.center,
            self.axis_u,
            self.axis_v,
            self.axis_w,
            (self.base_half_size, self.base_half_size, self.half_height),
        )


@dataclass
class BoxFrame(SDFNode):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    half_size: tuple[float, float, float] = (0.5, 0.5, 0.5)
    thickness: float = 0.08
    axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0)
    axis_v: tuple[float, float, float] = (0.0, 1.0, 0.0)
    axis_w: tuple[float, float, float] = (0.0, 0.0, 1.0)

    @property
    def dimension(self) -> int:
        return 3

    def __post_init__(self) -> None:
        if any(value <= 0.0 for value in self.half_size) or self.thickness <= 0.0:
            raise ValueError("box frame dimensions must be positive")
        self.axis_u, self.axis_v, self.axis_w = _orthonormal_frame(
            self.axis_u,
            self.axis_v,
            self.axis_w,
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
        px = np.abs(local_x) - self.half_size[0]
        py = np.abs(local_y) - self.half_size[1]
        pz = np.abs(local_z) - self.half_size[2]
        qx = np.abs(px + self.thickness) - self.thickness
        qy = np.abs(py + self.thickness) - self.thickness
        qz = np.abs(pz + self.thickness) - self.thickness

        def box_distance(
            ax: FloatArray,
            ay: FloatArray,
            az: FloatArray,
        ) -> FloatArray:
            outside = np.sqrt(
                np.maximum(ax, 0.0) ** 2
                + np.maximum(ay, 0.0) ** 2
                + np.maximum(az, 0.0) ** 2
            )
            inside = np.minimum(np.maximum(ax, np.maximum(ay, az)), 0.0)
            return np.asarray(outside + inside, dtype=np.float64)

        return np.asarray(
            np.minimum(
                np.minimum(
                    box_distance(px, qy, qz),
                    box_distance(qx, py, qz),
                ),
                box_distance(qx, qy, pz),
            ),
            dtype=np.float64,
        )

    def bounding_box(self) -> BoundingBox3D:
        return _oriented_box_bounds(
            self.center,
            self.axis_u,
            self.axis_v,
            self.axis_w,
            self.half_size,
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
