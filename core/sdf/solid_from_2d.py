from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BoundingBox3D, FloatArray, SDFNode, glsl_float, glsl_vec3
from .placed_2d import PlacedSDF2D


def _exact_extrusion(profile_distance: FloatArray, axial: FloatArray) -> FloatArray:
    outside = np.sqrt(
        np.maximum(profile_distance, 0.0) ** 2 + np.maximum(axial, 0.0) ** 2
    )
    inside = np.minimum(np.maximum(profile_distance, axial), 0.0)
    return np.asarray(outside + inside, dtype=np.float64)


@dataclass
class Extrude(SDFNode):
    section: PlacedSDF2D | None = None
    height: float = 1.0

    def __post_init__(self) -> None:
        if self.section is None:
            raise ValueError("extrude requires a placed 2D section")
        if self.height <= 0.0 or not np.isfinite(self.height):
            raise ValueError("extrude height must be finite and positive")

    @property
    def dimension(self) -> int:
        return 3

    def children(self) -> tuple[SDFNode, ...]:
        assert self.section is not None
        return (self.section,)

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        assert self.section is not None and self.section.profile is not None
        u, v, plane = self.section.project_numpy(X, Y, Z)
        profile_distance = self.section.profile.to_numpy(u, v)
        axial = np.abs(plane) - self.height * 0.5
        return _exact_extrusion(profile_distance, axial)

    def to_glsl(self, p_var: str = "p") -> str:
        assert self.section is not None and self.section.profile is not None
        u, v, plane = self.section._project_glsl(p_var)
        profile = self.section.profile.to_glsl(f"vec2({u}, {v})")
        axial = f"(abs({plane}) - {glsl_float(self.height * 0.5)})"
        pair = f"vec2({profile}, {axial})"
        return (
            f"(length(max({pair}, vec2(0.0)))"
            f" + min(max({profile}, {axial}), 0.0))"
        )

    def bounding_box(self) -> BoundingBox3D:
        assert self.section is not None and self.section.profile is not None
        u_min, u_max, v_min, v_max = self.section.profile.bounds()
        origin = np.asarray(self.section.origin)
        axis_u = np.asarray(self.section.axis_u)
        axis_v = np.asarray(self.section.axis_v)
        normal = np.asarray(self.section.normal)
        half = self.height * 0.5
        corners = np.asarray(
            [
                origin + u * axis_u + v * axis_v + n * normal
                for u in (u_min, u_max)
                for v in (v_min, v_max)
                for n in (-half, half)
            ]
        )
        minimum = corners.min(axis=0)
        maximum = corners.max(axis=0)
        return BoundingBox3D(
            minimum[0],
            maximum[0],
            minimum[1],
            maximum[1],
            minimum[2],
            maximum[2],
        )


@dataclass
class Sweep(SDFNode):
    section: PlacedSDF2D | None = None
    end: tuple[float, float, float] = (0.0, 0.0, 1.0)

    def _path(self) -> tuple[np.ndarray, np.ndarray, float]:
        if self.section is None:
            raise ValueError("sweep requires a placed 2D section")
        path = np.asarray(self.end) - np.asarray(self.section.origin)
        length = float(np.linalg.norm(path))
        if length <= 1e-12 or not np.isfinite(length):
            raise ValueError("sweep path must have finite nonzero length")
        direction = path / length
        if abs(float(np.dot(direction, self.section.normal))) < 1.0 - 1e-6:
            raise ValueError(
                "current sweep supports only a straight path normal to the section"
            )
        midpoint = (np.asarray(self.section.origin) + np.asarray(self.end)) * 0.5
        return midpoint, direction, length

    def __post_init__(self) -> None:
        self._path()

    @property
    def dimension(self) -> int:
        return 3

    def children(self) -> tuple[SDFNode, ...]:
        assert self.section is not None
        return (self.section,)

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        assert self.section is not None and self.section.profile is not None
        midpoint, direction, length = self._path()
        u, v, _ = self.section.project_numpy(X, Y, Z)
        profile_distance = self.section.profile.to_numpy(u, v)
        axial_position = (
            (X - midpoint[0]) * direction[0]
            + (Y - midpoint[1]) * direction[1]
            + (Z - midpoint[2]) * direction[2]
        )
        axial_distance = np.abs(axial_position) - length * 0.5
        return _exact_extrusion(profile_distance, axial_distance)

    def to_glsl(self, p_var: str = "p") -> str:
        assert self.section is not None and self.section.profile is not None
        midpoint, direction, length = self._path()
        u, v, _ = self.section._project_glsl(p_var)
        profile = self.section.profile.to_glsl(f"vec2({u}, {v})")
        axial = (
            f"(abs(dot(({p_var} - {glsl_vec3(tuple(midpoint))}),"
            f" {glsl_vec3(tuple(direction))})) - {glsl_float(length * 0.5)})"
        )
        pair = f"vec2({profile}, {axial})"
        return (
            f"(length(max({pair}, vec2(0.0)))"
            f" + min(max({profile}, {axial}), 0.0))"
        )

    def bounding_box(self) -> BoundingBox3D:
        assert self.section is not None and self.section.profile is not None
        u_min, u_max, v_min, v_max = self.section.profile.bounds()
        origin = np.asarray(self.section.origin)
        end = np.asarray(self.end)
        axis_u = np.asarray(self.section.axis_u)
        axis_v = np.asarray(self.section.axis_v)
        corners = np.asarray(
            [
                path_point + u * axis_u + v * axis_v
                for path_point in (origin, end)
                for u in (u_min, u_max)
                for v in (v_min, v_max)
            ]
        )
        minimum = corners.min(axis=0)
        maximum = corners.max(axis=0)
        return BoundingBox3D(
            minimum[0],
            maximum[0],
            minimum[1],
            maximum[1],
            minimum[2],
            maximum[2],
        )


@dataclass
class Revolve(SDFNode):
    section: PlacedSDF2D | None = None

    def __post_init__(self) -> None:
        if self.section is None:
            raise ValueError("revolve requires a placed 2D section")

    @property
    def dimension(self) -> int:
        return 3

    def children(self) -> tuple[SDFNode, ...]:
        assert self.section is not None
        return (self.section,)

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        assert self.section is not None and self.section.profile is not None
        origin = np.asarray(self.section.origin)
        rx, ry, rz = X - origin[0], Y - origin[1], Z - origin[2]
        axis = np.asarray(self.section.axis_v)
        axial = rx * axis[0] + ry * axis[1] + rz * axis[2]
        squared = rx**2 + ry**2 + rz**2
        radial = np.sqrt(np.maximum(squared - axial**2, 0.0))
        return self.section.profile.to_numpy(radial, axial)

    def to_glsl(self, p_var: str = "p") -> str:
        assert self.section is not None and self.section.profile is not None
        local = f"({p_var} - {glsl_vec3(self.section.origin)})"
        axis = glsl_vec3(self.section.axis_v)
        axial = f"dot({local}, {axis})"
        radial = f"sqrt(max(dot({local}, {local}) - ({axial}) * ({axial}), 0.0))"
        return self.section.profile.to_glsl(f"vec2({radial}, {axial})")

    def bounding_box(self) -> BoundingBox3D:
        assert self.section is not None and self.section.profile is not None
        u_min, u_max, v_min, v_max = self.section.profile.bounds()
        radius = max(abs(u_min), abs(u_max))
        origin = np.asarray(self.section.origin)
        axis = np.asarray(self.section.axis_v)
        endpoints = np.asarray((origin + v_min * axis, origin + v_max * axis))
        minimum = endpoints.min(axis=0) - radius
        maximum = endpoints.max(axis=0) + radius
        return BoundingBox3D(
            minimum[0],
            maximum[0],
            minimum[1],
            maximum[1],
            minimum[2],
            maximum[2],
        )


@dataclass
class LoftImplicit(SDFNode):
    sections: tuple[PlacedSDF2D, ...] = ()

    def __post_init__(self) -> None:
        if len(self.sections) < 2:
            raise ValueError("loft requires at least two placed sections")
        reference = self.sections[0]
        for section in self.sections[1:]:
            if not np.allclose(section.axis_u, reference.axis_u, atol=1e-6):
                raise ValueError("loft sections must use compatible axis_u")
            if not np.allclose(section.axis_v, reference.axis_v, atol=1e-6):
                raise ValueError("loft sections must use compatible axis_v")
        normal = np.asarray(reference.normal)
        parameters = [
            float(np.dot(np.asarray(section.origin), normal))
            for section in self.sections
        ]
        if any(
            right <= left
            for left, right in zip(parameters, parameters[1:])
        ):
            raise ValueError("loft section planes must be strictly ordered")

    @property
    def dimension(self) -> int:
        return 3

    def children(self) -> tuple[SDFNode, ...]:
        return tuple(self.sections)

    def _parameters(self) -> NDArray[np.float64]:
        normal = np.asarray(self.sections[0].normal)
        return np.asarray(
            [np.dot(np.asarray(section.origin), normal) for section in self.sections],
            dtype=np.float64,
        )

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        normal = np.asarray(self.sections[0].normal)
        world_parameter = X * normal[0] + Y * normal[1] + Z * normal[2]
        parameters = self._parameters()
        result = np.full(np.broadcast_shapes(X.shape, Y.shape, Z.shape), np.inf)
        for index, (first, second) in enumerate(
            zip(self.sections, self.sections[1:])
        ):
            assert first.profile is not None and second.profile is not None
            lower, upper = parameters[index], parameters[index + 1]
            interval = (world_parameter >= lower) & (world_parameter <= upper)
            u0, v0, _ = first.project_numpy(X, Y, Z)
            u1, v1, _ = second.project_numpy(X, Y, Z)
            d0 = first.profile.to_numpy(u0, v0)
            d1 = second.profile.to_numpy(u1, v1)
            t = np.clip((world_parameter - lower) / (upper - lower), 0.0, 1.0)
            blended = (1.0 - t) * d0 + t * d1
            result = np.where(interval, blended, result)
        cap_distance = np.maximum(parameters[0] - world_parameter, world_parameter - parameters[-1])
        return np.asarray(np.maximum(result, cap_distance), dtype=np.float64)

    def to_glsl(self, p_var: str = "p") -> str:
        normal = self.sections[0].normal
        world_parameter = f"dot({p_var}, {glsl_vec3(normal)})"
        parameters = self._parameters()
        expression = "1000000.0"
        for index in reversed(range(len(self.sections) - 1)):
            first = self.sections[index]
            second = self.sections[index + 1]
            assert first.profile is not None and second.profile is not None
            lower, upper = parameters[index], parameters[index + 1]
            u0, v0, _ = first._project_glsl(p_var)
            u1, v1, _ = second._project_glsl(p_var)
            d0 = first.profile.to_glsl(f"vec2({u0}, {v0})")
            d1 = second.profile.to_glsl(f"vec2({u1}, {v1})")
            t = (
                f"clamp(({world_parameter} - {glsl_float(lower)})"
                f" / {glsl_float(upper - lower)}, 0.0, 1.0)"
            )
            blended = f"mix({d0}, {d1}, {t})"
            condition = (
                f"({world_parameter} >= {glsl_float(lower)}"
                f" && {world_parameter} <= {glsl_float(upper)})"
            )
            expression = f"({condition} ? {blended} : {expression})"
        caps = (
            f"max({glsl_float(parameters[0])} - {world_parameter},"
            f" {world_parameter} - {glsl_float(parameters[-1])})"
        )
        return f"max({expression}, {caps})"

    def bounding_box(self) -> BoundingBox3D:
        boxes = [section.bounding_box() for section in self.sections]
        box = boxes[0]
        for other in boxes[1:]:
            box = box.union(other)
        return box
