from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .base import BoundingBox3D, FloatArray, SDFNode


@dataclass
class BinarySDFOperator(SDFNode):
    left: SDFNode | None = None
    right: SDFNode | None = None

    def __post_init__(self) -> None:
        if self.left is None or self.right is None:
            raise ValueError("SDF operations require two child nodes")
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
class Union(BinarySDFOperator):
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
class Intersection(BinarySDFOperator):
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
class Difference(BinarySDFOperator):
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
