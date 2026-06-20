from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin

import numpy as np

from .base import BoundingBox3D, FloatArray, SDFNode


@dataclass
class UnaryTransform(SDFNode):
    child: SDFNode | None = None

    def __post_init__(self) -> None:
        if self.child is None:
            raise ValueError("transform requires a child node")

    @property
    def dimension(self) -> int:
        assert self.child is not None
        return self.child.dimension

    def children(self) -> tuple[SDFNode, ...]:
        assert self.child is not None
        return (self.child,)


@dataclass
class Translate(UnaryTransform):
    offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        assert self.child is not None
        ox, oy, oz = self.offset
        return self.child.to_numpy(X - ox, Y - oy, Z - oz)

    def bounding_box(self) -> BoundingBox3D:
        assert self.child is not None
        box = self.child.bounding_box()
        ox, oy, oz = self.offset
        return BoundingBox3D(
            box.x_min + ox,
            box.x_max + ox,
            box.y_min + oy,
            box.y_max + oy,
            box.z_min + oz,
            box.z_max + oz,
        )


@dataclass
class Scale(UnaryTransform):
    factor: float = 1.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.factor <= 0.0:
            raise ValueError("scale factor must be positive")

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        assert self.child is not None
        return np.asarray(
            self.child.to_numpy(X / self.factor, Y / self.factor, Z / self.factor)
            * self.factor,
            dtype=np.float64,
        )

    def bounding_box(self) -> BoundingBox3D:
        assert self.child is not None
        box = self.child.bounding_box()
        return BoundingBox3D(
            box.x_min * self.factor,
            box.x_max * self.factor,
            box.y_min * self.factor,
            box.y_max * self.factor,
            box.z_min * self.factor,
            box.z_max * self.factor,
        )


@dataclass
class Rotate(UnaryTransform):
    axis: str = "y"
    angle_degrees: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.axis not in {"x", "y", "z"}:
            raise ValueError("rotation axis must be x, y, or z")

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        assert self.child is not None
        c = cos(radians(self.angle_degrees))
        s = sin(radians(self.angle_degrees))
        if self.axis == "x":
            local = (X, c * Y + s * Z, -s * Y + c * Z)
        elif self.axis == "y":
            local = (c * X - s * Z, Y, s * X + c * Z)
        else:
            local = (c * X + s * Y, -s * X + c * Y, Z)
        return self.child.to_numpy(*local)

    def bounding_box(self) -> BoundingBox3D:
        assert self.child is not None
        box = self.child.bounding_box()
        corners = np.asarray(
            [
                (x, y, z)
                for x in (box.x_min, box.x_max)
                for y in (box.y_min, box.y_max)
                for z in (box.z_min, box.z_max)
            ],
            dtype=np.float64,
        )
        angle = radians(self.angle_degrees)
        c, s = cos(angle), sin(angle)
        if self.axis == "x":
            matrix = np.asarray(((1, 0, 0), (0, c, -s), (0, s, c)))
        elif self.axis == "y":
            matrix = np.asarray(((c, 0, s), (0, 1, 0), (-s, 0, c)))
        else:
            matrix = np.asarray(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        rotated = corners @ matrix.T
        minima = rotated.min(axis=0)
        maxima = rotated.max(axis=0)
        return BoundingBox3D(
            minima[0], maxima[0], minima[1], maxima[1], minima[2], maxima[2]
        )
