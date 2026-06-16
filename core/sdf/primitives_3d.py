from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BoundingBox3D, FloatArray, SDFNode, glsl_float, glsl_vec3


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

    @property
    def dimension(self) -> int:
        return 3

    def __post_init__(self) -> None:
        if any(value <= 0.0 for value in self.half_size):
            raise ValueError("box half sizes must be positive")

    def to_glsl(self, p_var: str = "p") -> str:
        q = f"(abs({p_var} - {glsl_vec3(self.center)}) - {glsl_vec3(self.half_size)})"
        return f"(length(max({q}, vec3(0.0))) + min(max({q}.x, max({q}.y, {q}.z)), 0.0))"

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        center = np.asarray(self.center, dtype=np.float64)
        half_size = np.asarray(self.half_size, dtype=np.float64)
        qx = np.abs(X - center[0]) - half_size[0]
        qy = np.abs(Y - center[1]) - half_size[1]
        qz = np.abs(Z - center[2]) - half_size[2]
        outside = np.sqrt(
            np.maximum(qx, 0.0) ** 2
            + np.maximum(qy, 0.0) ** 2
            + np.maximum(qz, 0.0) ** 2
        )
        inside = np.minimum(np.maximum(qx, np.maximum(qy, qz)), 0.0)
        return np.asarray(outside + inside, dtype=np.float64)

    def bounding_box(self) -> BoundingBox3D:
        cx, cy, cz = self.center
        hx, hy, hz = self.half_size
        return BoundingBox3D(cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz)


@dataclass
class Cylinder(SDFNode):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    radius: float = 0.5
    half_height: float = 0.5

    @property
    def dimension(self) -> int:
        return 3

    def __post_init__(self) -> None:
        if self.radius <= 0.0 or self.half_height <= 0.0:
            raise ValueError("cylinder dimensions must be positive")

    def to_glsl(self, p_var: str = "p") -> str:
        local = f"({p_var} - {glsl_vec3(self.center)})"
        d = (
            f"(abs(vec2(length({local}.xy), {local}.z))"
            f" - vec2({glsl_float(self.radius)}, {glsl_float(self.half_height)}))"
        )
        return f"(min(max({d}.x, {d}.y), 0.0) + length(max({d}, vec2(0.0))))"

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        cx, cy, cz = self.center
        radial = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - self.radius
        axial = np.abs(Z - cz) - self.half_height
        outside = np.sqrt(np.maximum(radial, 0.0) ** 2 + np.maximum(axial, 0.0) ** 2)
        inside = np.minimum(np.maximum(radial, axial), 0.0)
        return np.asarray(outside + inside, dtype=np.float64)

    def bounding_box(self) -> BoundingBox3D:
        cx, cy, cz = self.center
        return BoundingBox3D(
            cx - self.radius,
            cx + self.radius,
            cy - self.radius,
            cy + self.radius,
            cz - self.half_height,
            cz + self.half_height,
        )


@dataclass
class Torus(SDFNode):
    center: tuple[float, float, float] = (0.0, 0.0, 0.0)
    major_radius: float = 0.5
    minor_radius: float = 0.15

    @property
    def dimension(self) -> int:
        return 3

    def __post_init__(self) -> None:
        if self.major_radius <= 0.0 or self.minor_radius <= 0.0:
            raise ValueError("torus radii must be positive")

    def to_glsl(self, p_var: str = "p") -> str:
        local = f"({p_var} - {glsl_vec3(self.center)})"
        return (
            f"(length(vec2(length({local}.xy) - {glsl_float(self.major_radius)},"
            f" {local}.z)) - {glsl_float(self.minor_radius)})"
        )

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        cx, cy, cz = self.center
        qx = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - self.major_radius
        return np.asarray(
            np.sqrt(qx**2 + (Z - cz) ** 2) - self.minor_radius,
            dtype=np.float64,
        )

    def bounding_box(self) -> BoundingBox3D:
        cx, cy, cz = self.center
        outer = self.major_radius + self.minor_radius
        return BoundingBox3D(
            cx - outer,
            cx + outer,
            cy - outer,
            cy + outer,
            cz - self.minor_radius,
            cz + self.minor_radius,
        )
