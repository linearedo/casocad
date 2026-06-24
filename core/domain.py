from __future__ import annotations

from dataclasses import dataclass

from core.boundary import BoundaryRegion
from core.sdf.base import BoundingBox3D, FloatArray, SDFNode
from core.sdf.placed_1d import PlacedSDF1D
from core.sdf.placed_2d import PlacedPolyline2D, PlacedSDF2D
from core.sdf_attribution import boundary_owner_ids

DomainTag = PlacedSDF1D | PlacedPolyline2D | PlacedSDF2D | BoundaryRegion


@dataclass(frozen=True)
class FluidDomain:
    root: SDFNode
    tag_objects: tuple[DomainTag, ...] = ()
    selector_objects: tuple[SDFNode, ...] = ()

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
                raise ValueError(f"duplicate FluidDomain object_id {node.object_id}")
            ids_to_objects[node.object_id] = node
        valid_boundary_owner_ids = boundary_owner_ids(self.root)
        for tag in self.tag_objects:
            if tag.object_id <= 0:
                raise ValueError("FluidDomain objects require stable nonzero IDs")
            existing = ids_to_objects.get(tag.object_id)
            if existing is not None and existing is not tag:
                raise ValueError(f"duplicate FluidDomain object_id {tag.object_id}")
            if (
                isinstance(tag, BoundaryRegion)
                and tag.owner_object_id not in valid_boundary_owner_ids
            ):
                raise ValueError(
                    "BoundaryRegion owner must control values in the FluidDomain root"
                )
            if (
                self.root.dimension == 2
                and not isinstance(tag, (PlacedSDF1D, PlacedPolyline2D, BoundaryRegion))
            ):
                raise ValueError(
                    "2D FluidDomain tags must be PlacedSDF1D, PlacedPolyline2D, "
                    "or BoundaryRegion objects"
                )
            if (
                self.root.dimension == 2
                and isinstance(tag, (PlacedSDF1D, PlacedPolyline2D))
                and not tag.lies_in_plane_of(self.root)
            ):
                raise ValueError("1D FluidDomain tags must lie in the 2D root workplane")
            if (
                self.root.dimension == 3
                and not isinstance(tag, (PlacedSDF2D, BoundaryRegion))
            ):
                raise ValueError("3D FluidDomain tags must be PlacedSDF2D or BoundaryRegion")
            ids_to_objects[tag.object_id] = tag
        for selector in self.selector_objects:
            if selector.object_id <= 0:
                raise ValueError("FluidDomain objects require stable nonzero IDs")
            existing = ids_to_objects.get(selector.object_id)
            if existing is not None and existing is not selector:
                raise ValueError(f"duplicate FluidDomain object_id {selector.object_id}")
            if self.root.dimension == 3:
                valid_selector = isinstance(selector, SDFNode)
            else:
                valid_selector = isinstance(selector, (PlacedSDF1D, PlacedPolyline2D))
            if not valid_selector:
                raise ValueError("FluidDomain boundary selectors must be SDF cutter objects")
            if (
                self.root.dimension == 2
                and isinstance(self.root, PlacedSDF2D)
                and not selector.lies_in_plane_of(self.root)
            ):
                raise ValueError(
                    "2D FluidDomain boundary selectors must lie in the root workplane"
                )
            ids_to_objects[selector.object_id] = selector
        tag_ids = [tag.object_id for tag in self.tag_objects]
        if len(tag_ids) != len(set(tag_ids)):
            raise ValueError("FluidDomain tag object IDs must be unique")
        selector_ids = [selector.object_id for selector in self.selector_objects]
        if len(selector_ids) != len(set(selector_ids)):
            raise ValueError("FluidDomain selector object IDs must be unique")

    def bounding_box(self) -> BoundingBox3D:
        return self.root.bounding_box()

    def to_numpy(
        self,
        X: FloatArray,
        Y: FloatArray,
        Z: FloatArray,
    ) -> FloatArray:
        return self.root.to_numpy(X, Y, Z)


__all__ = ["DomainTag", "FluidDomain"]
