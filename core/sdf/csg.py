from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BoundingBox3D, FloatArray, SDFNode


@dataclass
class BinaryCSG(SDFNode):
    left: SDFNode | None = None
    right: SDFNode | None = None

    def __post_init__(self) -> None:
        if self.left is None or self.right is None:
            raise ValueError("CSG operations require two child nodes")
        if self.left.dimension != self.right.dimension:
            raise ValueError("boolean operands must have the same dimension")

    @property
    def dimension(self) -> int:
        assert self.left is not None and self.right is not None
        return self.left.dimension

    def children(self) -> tuple[SDFNode, ...]:
        assert self.left is not None and self.right is not None
        return (self.left, self.right)


@dataclass
class Union(BinaryCSG):
    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        assert self.left is not None and self.right is not None
        return np.minimum(
            self.left.to_numpy(X, Y, Z), self.right.to_numpy(X, Y, Z)
        )

    def bounding_box(self) -> BoundingBox3D:
        assert self.left is not None and self.right is not None
        return self.left.bounding_box().union(self.right.bounding_box())


@dataclass
class Intersection(BinaryCSG):
    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        assert self.left is not None and self.right is not None
        return np.maximum(
            self.left.to_numpy(X, Y, Z), self.right.to_numpy(X, Y, Z)
        )

    def bounding_box(self) -> BoundingBox3D:
        assert self.left is not None and self.right is not None
        left = self.left.bounding_box()
        right = self.right.bounding_box()
        try:
            return left.intersection(right)
        except ValueError:
            return left.union(right)


@dataclass
class Difference(BinaryCSG):
    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        assert self.left is not None and self.right is not None
        return np.maximum(
            self.left.to_numpy(X, Y, Z), -self.right.to_numpy(X, Y, Z)
        )

    def bounding_box(self) -> BoundingBox3D:
        assert self.left is not None
        return self.left.bounding_box()


@dataclass
class SmoothUnion(BinaryCSG):
    smoothing: float = 0.1

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.smoothing <= 0.0:
            raise ValueError("smooth union radius must be positive")

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        assert self.left is not None and self.right is not None
        a = self.left.to_numpy(X, Y, Z)
        b = self.right.to_numpy(X, Y, Z)
        h = np.clip(0.5 + 0.5 * (b - a) / self.smoothing, 0.0, 1.0)
        return np.asarray(
            b * (1.0 - h) + a * h - self.smoothing * h * (1.0 - h),
            dtype=np.float64,
        )

    def bounding_box(self) -> BoundingBox3D:
        assert self.left is not None and self.right is not None
        box = self.left.bounding_box().union(self.right.bounding_box())
        k = self.smoothing
        return BoundingBox3D(
            box.x_min - k,
            box.x_max + k,
            box.y_min - k,
            box.y_max + k,
            box.z_min - k,
            box.z_max + k,
        )
