from __future__ import annotations

from dataclasses import dataclass

from core.boundary import BoundaryRegion
from core.sdf.base import FloatArray
from core.sdf.base import BoundingBox3D, SDFNode
from core.sdf.placed_1d import PlacedSDF1D
from core.sdf.placed_2d import PlacedSDF2D
from .classifier import boundary_owner_ids

LatticeTag = PlacedSDF1D | PlacedSDF2D | BoundaryRegion


@dataclass(frozen=True)
class FluidDomain:
    root: SDFNode
    tag_objects: tuple[LatticeTag, ...] = ()

    def __post_init__(self) -> None:
        if self.root.dimension not in {2, 3}:
            raise ValueError("FluidDomain root must be a 2D or 3D SDF")
        if self.root.dimension == 2 and not isinstance(self.root, PlacedSDF2D):
            raise ValueError("2D FluidDomain root must be a PlacedSDF2D")
        graph_nodes: list[SDFNode] = []
        seen: set[int] = set()

        def visit(node: SDFNode) -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            graph_nodes.append(node)
            for child in node.children():
                visit(child)

        visit(self.root)
        ids_to_objects: dict[int, SDFNode | BoundaryRegion] = {}
        for node in graph_nodes:
            if node.object_id <= 0:
                raise ValueError("FluidDomain objects require stable nonzero IDs")
            existing = ids_to_objects.get(node.object_id)
            if existing is not None and existing is not node:
                raise ValueError(
                    f"duplicate FluidDomain object_id {node.object_id}"
                )
            ids_to_objects[node.object_id] = node
        root_object_ids = set(ids_to_objects)
        valid_boundary_owner_ids = boundary_owner_ids(self.root)
        for tag in self.tag_objects:
            if tag.object_id <= 0:
                raise ValueError("FluidDomain objects require stable nonzero IDs")
            existing = ids_to_objects.get(tag.object_id)
            if existing is not None and existing is not tag:
                raise ValueError(f"duplicate FluidDomain object_id {tag.object_id}")
            if (
                self.root.dimension == 3
                and isinstance(tag, BoundaryRegion)
                and tag.owner_object_id not in valid_boundary_owner_ids
            ):
                raise ValueError(
                    "BoundaryRegion owner must control values in the FluidDomain root"
                )
            if (
                self.root.dimension == 2
                and not isinstance(tag, PlacedSDF1D)
            ):
                raise ValueError(
                    "2D FluidDomain tags must be PlacedSDF1D objects"
                )
            if (
                self.root.dimension == 2
                and isinstance(tag, PlacedSDF1D)
                and not tag.lies_in_plane_of(self.root)
            ):
                raise ValueError(
                    "1D FluidDomain tags must lie in the 2D root workplane"
                )
            if (
                self.root.dimension == 3
                and not isinstance(tag, (PlacedSDF2D, BoundaryRegion))
            ):
                raise ValueError(
                    "3D FluidDomain tags must be PlacedSDF2D or BoundaryRegion"
                )
            ids_to_objects[tag.object_id] = tag
        tag_ids = [tag.object_id for tag in self.tag_objects]
        if len(tag_ids) != len(set(tag_ids)):
            raise ValueError("FluidDomain tag object IDs must be unique")

    def bounding_box(self) -> BoundingBox3D:
        # Provisional traversal strategy. Bounds are not SDF semantics.
        return self.root.bounding_box()

    def to_numpy(
        self, X: FloatArray, Y: FloatArray, Z: FloatArray
    ) -> FloatArray:
        return self.root.to_numpy(X, Y, Z)

    def boundary_offsets(self) -> tuple[tuple[float, float, float], ...]:
        if self.root.dimension == 3:
            return (
                (-1.0, 0.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, -1.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, -1.0),
                (0.0, 0.0, 1.0),
            )
        assert isinstance(self.root, PlacedSDF2D)
        axis_u = tuple(float(value) for value in self.root.axis_u)
        axis_v = tuple(float(value) for value in self.root.axis_v)
        return (
            tuple(-value for value in axis_u),
            axis_u,
            tuple(-value for value in axis_v),
            axis_v,
        )


@dataclass(frozen=True)
class MesherConfig:
    dx: float
    n_levels: int = 0
    chunk_size: int = 10_000_000
    unit_label: str = "m"
    internal_preview_density: float = 0.1
    boundary_error_tolerance: float | None = None
    max_error_refinements: int = 4
    max_candidate_nodes: int = 10_000_000

    def __post_init__(self) -> None:
        if self.dx <= 0.0:
            raise ValueError("dx must be positive")
        if (
            self.boundary_error_tolerance is not None
            and self.boundary_error_tolerance <= 0.0
        ):
            raise ValueError("boundary_error_tolerance must be positive")
        if self.max_error_refinements < 0:
            raise ValueError("max_error_refinements must be nonnegative")
        if self.max_candidate_nodes <= 0:
            raise ValueError("max_candidate_nodes must be positive")
        if self.n_levels != 0:
            raise ValueError("refinement is not implemented; n_levels must be 0")
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if not 0.0 <= self.internal_preview_density <= 1.0:
            raise ValueError(
                "internal_preview_density must be between 0 and 1"
            )
