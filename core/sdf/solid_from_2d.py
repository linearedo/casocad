from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BoundingBox3D, FloatArray, SDFNode
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
    center_offset: float = 0.0

    def __post_init__(self) -> None:
        if self.section is None:
            raise ValueError("extrude requires a placed 2D section")
        if self.height <= 0.0 or not np.isfinite(self.height):
            raise ValueError("extrude height must be finite and positive")
        if not np.isfinite(self.center_offset):
            raise ValueError("extrude center offset must be finite")

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
        axial = np.abs(plane - self.center_offset) - self.height * 0.5
        return _exact_extrusion(profile_distance, axial)

    def bounding_box(self) -> BoundingBox3D:
        assert self.section is not None and self.section.profile is not None
        u_min, u_max, v_min, v_max = self.section.profile.bounds()
        origin = np.asarray(self.section.origin)
        axis_u = np.asarray(self.section.axis_u)
        axis_v = np.asarray(self.section.axis_v)
        normal = np.asarray(self.section.normal)
        half = self.height * 0.5
        center = origin + self.center_offset * normal
        corners = np.asarray(
            [
                center + u * axis_u + v * axis_v + n * normal
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
class Revolve(SDFNode):
    section: PlacedSDF2D | None = None
    axis: str = "v"
    axis_origin: tuple[float, float, float] | None = None
    axis_direction: tuple[float, float, float] | None = None
    radial_direction: tuple[float, float, float] | None = None
    angle_degrees: float = 360.0

    def __post_init__(self) -> None:
        if self.section is None:
            raise ValueError("revolve requires a placed 2D section")
        if self.axis not in {"u", "v"}:
            raise ValueError("revolve axis must be 'u' or 'v'")
        if not np.isfinite(self.angle_degrees) or not 0.0 < self.angle_degrees <= 360.0:
            raise ValueError("revolve angle must be finite and in (0, 360]")
        self._axis_frame()

    @property
    def dimension(self) -> int:
        return 3

    def children(self) -> tuple[SDFNode, ...]:
        assert self.section is not None
        return (self.section,)

    def _axis_frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert self.section is not None
        origin = np.asarray(
            self.axis_origin if self.axis_origin is not None else self.section.origin,
            dtype=np.float64,
        )
        axis = np.asarray(
            self.axis_direction
            if self.axis_direction is not None
            else (self.section.axis_u if self.axis == "u" else self.section.axis_v),
            dtype=np.float64,
        )
        axis_length = float(np.linalg.norm(axis))
        if axis_length <= 1.0e-12 or not np.isfinite(axis_length):
            raise ValueError("revolve axis direction must be finite and nonzero")
        axis = axis / axis_length
        radial = np.asarray(
            self.radial_direction
            if self.radial_direction is not None
            else (self.section.axis_v if self.axis == "u" else self.section.axis_u),
            dtype=np.float64,
        )
        radial = radial - axis * float(np.dot(radial, axis))
        radial_length = float(np.linalg.norm(radial))
        if radial_length <= 1.0e-12 or not np.isfinite(radial_length):
            raise ValueError("revolve radial direction must not be parallel to axis")
        radial = radial / radial_length
        tangent = np.cross(axis, radial)
        tangent_length = float(np.linalg.norm(tangent))
        if tangent_length <= 1.0e-12 or not np.isfinite(tangent_length):
            raise ValueError("revolve axis frame is degenerate")
        tangent = tangent / tangent_length
        return origin, axis, radial, tangent

    @staticmethod
    def _angular_sector_sdf_numpy(
        x: FloatArray,
        y: FloatArray,
        angle_degrees: float,
    ) -> FloatArray:
        if angle_degrees >= 360.0 - 1.0e-9:
            return np.full_like(x, -1.0e6, dtype=np.float64)
        angle = np.deg2rad(angle_degrees)
        radius = np.sqrt(x * x + y * y)
        theta = np.arctan2(y, x)
        theta = np.where(theta < 0.0, theta + 2.0 * np.pi, theta)
        inside = theta <= angle

        def ray_distance(ray_x: float, ray_y: float) -> FloatArray:
            projection = x * ray_x + y * ray_y
            cross = ray_x * y - ray_y * x
            return np.where(projection >= 0.0, np.abs(cross), radius)

        start_distance = ray_distance(1.0, 0.0)
        end_distance = ray_distance(float(np.cos(angle)), float(np.sin(angle)))
        distance = np.minimum(start_distance, end_distance)
        return np.asarray(np.where(inside, -distance, distance), dtype=np.float64)

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        assert self.section is not None and self.section.profile is not None
        origin, axis, radial_axis, tangent_axis = self._axis_frame()
        rx, ry, rz = X - origin[0], Y - origin[1], Z - origin[2]
        axial = rx * axis[0] + ry * axis[1] + rz * axis[2]
        radial_x = rx * radial_axis[0] + ry * radial_axis[1] + rz * radial_axis[2]
        radial_y = rx * tangent_axis[0] + ry * tangent_axis[1] + rz * tangent_axis[2]
        radial = np.sqrt(np.maximum(radial_x**2 + radial_y**2, 0.0))
        sample_x = origin[0] + axial * axis[0] + radial * radial_axis[0]
        sample_y = origin[1] + axial * axis[1] + radial * radial_axis[1]
        sample_z = origin[2] + axial * axis[2] + radial * radial_axis[2]
        u, v, _plane = self.section.project_numpy(sample_x, sample_y, sample_z)
        profile = self.section.profile.to_numpy(u, v)
        if self.angle_degrees >= 360.0 - 1.0e-9:
            return profile
        angular = self._angular_sector_sdf_numpy(
            radial_x,
            radial_y,
            self.angle_degrees,
        )
        outside = np.sqrt(np.maximum(profile, 0.0) ** 2 + np.maximum(angular, 0.0) ** 2)
        inside = np.minimum(np.maximum(profile, angular), 0.0)
        return np.asarray(outside + inside, dtype=np.float64)

    def bounding_box(self) -> BoundingBox3D:
        assert self.section is not None and self.section.profile is not None
        u_min, u_max, v_min, v_max = self.section.profile.bounds()
        origin, axis, _radial_axis, _tangent_axis = self._axis_frame()
        section_origin = np.asarray(self.section.origin, dtype=np.float64)
        axis_u = np.asarray(self.section.axis_u, dtype=np.float64)
        axis_v = np.asarray(self.section.axis_v, dtype=np.float64)
        corners = np.asarray(
            [
                section_origin + u * axis_u + v * axis_v
                for u in (u_min, u_max)
                for v in (v_min, v_max)
            ],
            dtype=np.float64,
        )
        local = corners - origin
        axial = local @ axis
        radial_vectors = local - np.outer(axial, axis)
        radius = float(np.linalg.norm(radial_vectors, axis=1).max())
        endpoints = np.asarray((origin + axial.min() * axis, origin + axial.max() * axis))
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
