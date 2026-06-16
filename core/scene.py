from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from .boundary import BoundaryRegion
from .mesher import FluidDomain
from .mesher.classifier import boundary_owner_ids
from .sdf import (
    BinaryProfile1D,
    BinaryProfile,
    Box,
    CircleProfile,
    Cylinder,
    Difference,
    EllipseProfile,
    Extrude,
    Intersection,
    IntervalProfile,
    LoftImplicit,
    OffsetProfile,
    OffsetProfile1D,
    PlacedSDF1D,
    PlacedSDF2D,
    RectangleProfile,
    RegularPolygonProfile,
    Rotate,
    RoundedRectangleProfile,
    SDFNode,
    SDFTree,
    Scale,
    SmoothUnion,
    Sphere,
    SquareProfile,
    Torus,
    Translate,
    Union,
)
from .sdf.csg import BinaryCSG
from .sdf.solid_from_2d import Revolve, Sweep
from .sdf.transforms import UnaryTransform

Primitive3D = Sphere | Box | Cylinder | Torus
SceneItem = SDFNode | BoundaryRegion


@dataclass
class SceneDocument:
    objects: list[SDFNode] = field(default_factory=list)
    fluid_domain: FluidDomain | None = None
    boundary_regions: list[BoundaryRegion] = field(default_factory=list)
    _next_object_id: int = field(default=1, init=False, repr=False)
    _handles: dict[int, SceneItem] = field(default_factory=dict, init=False, repr=False)
    _node_handles: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _next_handle: int = field(default=1, init=False, repr=False)

    def __post_init__(self) -> None:
        maximum_id = max(
            (
                node.object_id
                for node in (
                    *(
                        item
                        for root in self.objects
                        for item in self._iter_nodes(root)
                    ),
                    *self.boundary_regions,
                )
            ),
            default=0,
        )
        self._next_object_id = maximum_id + 1
        self._reindex()
        self._refresh_fluid_domain()

    @property
    def bodies(self) -> list[SDFNode]:
        """Compatibility alias while UI code migrates to objects."""
        return self.objects

    @classmethod
    def default(cls) -> SceneDocument:
        document = cls()
        outer = document.create_primitive("box", name="flow_volume")
        assert isinstance(outer, Box)
        outer.center = (0.0, 0.0, 0.0)
        outer.half_size = (1.6, 0.7, 0.45)
        obstacle = document.create_primitive("cylinder", name="cylinder_obstacle")
        assert isinstance(obstacle, Cylinder)
        obstacle.radius = 0.24
        obstacle.half_height = 0.55
        root = Difference(
            name="von_karman_fluid",
            object_id=document._allocate_object_id(),
            left=outer,
            right=obstacle,
        )
        inlet = BoundaryRegion(
            name="inlet",
            object_id=document._allocate_object_id(),
            owner_object_id=outer.object_id,
            outside_direction=0,
        )
        outlet = BoundaryRegion(
            name="outlet",
            object_id=document._allocate_object_id(),
            owner_object_id=outer.object_id,
            outside_direction=1,
        )
        document.objects = [root]
        document.boundary_regions = [inlet, outlet]
        document.fluid_domain = FluidDomain(root, (inlet, outlet))
        document._reindex()
        return document

    def _allocate_object_id(self) -> int:
        object_id = self._next_object_id
        self._next_object_id += 1
        if object_id > 65_535:
            raise ValueError("maximum SDF object count exceeded")
        return object_id

    def create_primitive(
        self, kind: str, name: str | None = None
    ) -> Primitive3D:
        object_id = self._allocate_object_id()
        common = {"name": name or f"{kind}_{object_id}", "object_id": object_id}
        factories = {
            "sphere": lambda: Sphere(**common, radius=0.5),
            "box": lambda: Box(**common, half_size=(0.5, 0.5, 0.5)),
            "cylinder": lambda: Cylinder(**common, radius=0.4, half_height=0.6),
            "torus": lambda: Torus(**common, major_radius=0.5, minor_radius=0.15),
        }
        if kind not in factories:
            raise ValueError(f"unknown 3D primitive type: {kind}")
        return factories[kind]()

    def create_placed_2d(
        self, kind: str, name: str | None = None
    ) -> PlacedSDF2D:
        object_id = self._allocate_object_id()
        factories = {
            "circle": CircleProfile,
            "rectangle": RectangleProfile,
            "square": SquareProfile,
            "rounded_rectangle": RoundedRectangleProfile,
            "ellipse": EllipseProfile,
            "regular_polygon": RegularPolygonProfile,
        }
        if kind not in factories:
            raise ValueError(f"unknown 2D profile type: {kind}")
        return PlacedSDF2D(
            name=name or f"{kind}_{object_id}",
            object_id=object_id,
            profile=factories[kind](),
        )

    def create_placed_1d(
        self,
        name: str | None = None,
    ) -> PlacedSDF1D:
        object_id = self._allocate_object_id()
        return PlacedSDF1D(
            name=name or f"interval_{object_id}",
            object_id=object_id,
            profile=IntervalProfile(),
        )

    def add_primitive(self, kind: str) -> int:
        if kind == "interval":
            node: SDFNode = self.create_placed_1d()
        elif kind in {
            "circle",
            "rectangle",
            "square",
            "rounded_rectangle",
            "ellipse",
            "regular_polygon",
        }:
            node = self.create_placed_2d(kind)
        else:
            node = self.create_primitive(kind)
            offset = 0.25 * len(self.objects)
            if hasattr(node, "center"):
                node.center = (offset, 0.0, 0.0)
        self.objects.append(node)
        self._reindex()
        return self.handle_for(node)

    def add_primitive_from_drag(
        self,
        kind: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
    ) -> int:
        start_array = np.asarray(start, dtype=np.float64)
        end_array = np.asarray(end, dtype=np.float64)
        center = 0.5 * (start_array + end_array)
        extent_x = max(abs(end_array[0] - start_array[0]) * 0.5, 0.05)
        extent_y = max(abs(end_array[1] - start_array[1]) * 0.5, 0.05)
        radius = max(float(np.linalg.norm(end_array[[0, 1]] - start_array[[0, 1]])) * 0.5, 0.05)
        if kind == "interval":
            direction = end_array - start_array
            length = float(np.linalg.norm(direction))
            node = self.create_placed_1d()
            node.origin = tuple(float(value) for value in center)
            node.axis_u = (
                tuple(float(value) for value in direction / length)
                if length > 1e-12
                else (1.0, 0.0, 0.0)
            )
            node.profile = IntervalProfile(half_length=max(0.5 * length, 0.05))
            node.__post_init__()
        elif kind in {
            "circle",
            "rectangle",
            "square",
            "rounded_rectangle",
            "ellipse",
            "regular_polygon",
        }:
            node = self.create_placed_2d(kind)
            assert isinstance(node, PlacedSDF2D)
            node.origin = (float(center[0]), float(center[1]), 0.0)
            node.axis_u = (1.0, 0.0, 0.0)
            node.axis_v = (0.0, 1.0, 0.0)
            if kind == "circle":
                node.profile = CircleProfile(radius=radius)
            elif kind == "rectangle":
                node.profile = RectangleProfile(half_size=(extent_x, extent_y))
            elif kind == "square":
                node.profile = SquareProfile(half_size=max(extent_x, extent_y))
            elif kind == "rounded_rectangle":
                half_size = (extent_x, extent_y)
                node.profile = RoundedRectangleProfile(
                    half_size=half_size,
                    corner_radius=max(0.01, min(half_size) * 0.2),
                )
            elif kind == "ellipse":
                node.profile = EllipseProfile(semi_axes=(extent_x, extent_y))
            else:
                node.profile = RegularPolygonProfile(radius=radius)
            node.__post_init__()
        else:
            node = self.create_primitive(kind)
            world_center = (float(center[0]), float(center[1]), 0.0)
            if isinstance(node, Sphere):
                node.center = world_center
                node.radius = radius
            elif isinstance(node, Box):
                node.center = world_center
                node.half_size = (extent_x, extent_y, max(extent_x, extent_y))
            elif isinstance(node, Cylinder):
                node.center = world_center
                node.radius = radius
                node.half_height = max(extent_x, extent_y)
            elif isinstance(node, Torus):
                node.center = world_center
                node.major_radius = radius
                node.minor_radius = max(radius * 0.25, 0.02)
        self.objects.append(node)
        self._reindex()
        return self.handle_for(node)

    def move_object(
        self,
        handle: int,
        delta: tuple[float, float, float],
    ) -> int:
        node = self.node(handle)
        if isinstance(node, (PlacedSDF1D, PlacedSDF2D)):
            node.origin = tuple(
                node.origin[index] + delta[index] for index in range(3)
            )
            node.__post_init__()
            return handle
        if isinstance(node, (Sphere, Box, Cylinder, Torus)):
            node.center = tuple(
                node.center[index] + delta[index] for index in range(3)
            )
            return handle
        if isinstance(node, Translate):
            node.offset = tuple(
                node.offset[index] + delta[index] for index in range(3)
            )
            return handle
        wrapped_handle = self.wrap_transform(handle, "translate")
        wrapped = self.node(wrapped_handle)
        assert isinstance(wrapped, Translate)
        wrapped.offset = delta
        return wrapped_handle

    def combine(self, first_handle: int, second_handle: int, operation: str) -> int:
        first = self.node(first_handle)
        second = self.node(second_handle)
        if first is second:
            raise ValueError("select two different SDF nodes")
        if self._contains(first, second) or self._contains(second, first):
            raise ValueError("an SDF cannot be combined with its own descendant")
        if first.dimension != second.dimension:
            raise ValueError("boolean operands must have the same dimension")
        current_domain = self.fluid_domain
        replaces_fluid_root = (
            current_domain is not None
            and (
                current_domain.root is first
                or current_domain.root is second
            )
        )
        domain_tags = current_domain.tag_objects if current_domain is not None else ()

        label = operation.replace("_", " ")
        object_id = self._allocate_object_id()
        if isinstance(first, PlacedSDF1D) and isinstance(second, PlacedSDF1D):
            if not first.is_collinear_with(second):
                raise ValueError("1D boolean operands must be collinear")
            assert first.profile is not None and second.profile is not None
            displacement = np.asarray(second.origin) - np.asarray(first.origin)
            second_offset = float(
                np.dot(displacement, np.asarray(first.axis_u))
            )
            combined: SDFNode = PlacedSDF1D(
                name=f"{label}: {first.name}, {second.name}",
                object_id=object_id,
                profile=BinaryProfile1D(
                    first.profile,
                    OffsetProfile1D(second.profile, second_offset),
                    operation,
                ),
                origin=first.origin,
                axis_u=first.axis_u,
                sources=(first, second),
            )
        elif isinstance(first, PlacedSDF2D) and isinstance(second, PlacedSDF2D):
            if not first.is_coplanar_with(second):
                raise ValueError("2D boolean operands must be coplanar")
            assert first.profile is not None and second.profile is not None
            displacement = (
                np.asarray(second.origin, dtype=np.float64)
                - np.asarray(first.origin, dtype=np.float64)
            )
            second_offset = (
                float(np.dot(displacement, np.asarray(first.axis_u))),
                float(np.dot(displacement, np.asarray(first.axis_v))),
            )
            combined = PlacedSDF2D(
                name=f"{label}: {first.name}, {second.name}",
                object_id=object_id,
                profile=BinaryProfile(
                    first.profile,
                    OffsetProfile(second.profile, second_offset),
                    operation,
                ),
                origin=first.origin,
                axis_u=first.axis_u,
                axis_v=first.axis_v,
                sources=(first, second),
            )
        elif first.dimension == 3:
            constructors = {
                "union": Union,
                "intersection": Intersection,
                "difference": Difference,
                "smooth_union": SmoothUnion,
            }
            if operation not in constructors:
                raise ValueError(f"unknown CSG operation: {operation}")
            combined = constructors[operation](
                name=f"{label}: {first.name}, {second.name}",
                object_id=object_id,
                left=first,
                right=second,
            )
        else:
            raise ValueError("unsupported boolean operand types")

        first_index = self._detach(first)
        second_index = self._detach(second)
        self.objects.insert(min(first_index, second_index, len(self.objects)), combined)
        self._reindex()
        if replaces_fluid_root:
            self.fluid_domain = FluidDomain(
                combined,
                self._compatible_domain_tags(combined, domain_tags),
            )
        else:
            self._refresh_fluid_domain()
        return self.handle_for(combined)

    def can_combine(self, first_handle: int, second_handle: int) -> bool:
        first = self.node(first_handle)
        second = self.node(second_handle)
        return (
            first is not second
            and first.dimension == second.dimension
            and not self._contains(first, second)
            and not self._contains(second, first)
        )

    def wrap_transform(self, handle: int, transform: str) -> int:
        node = self.node(handle)
        if node.dimension != 3:
            raise ValueError(
                "edit the placed SDF origin and axes to transform 1D or 2D objects"
            )
        common = {
            "name": f"{transform}: {node.name}",
            "object_id": self._allocate_object_id(),
            "child": node,
        }
        constructors = {
            "translate": lambda: Translate(**common, offset=(0.1, 0.0, 0.0)),
            "rotate": lambda: Rotate(**common, axis="y", angle_degrees=15.0),
            "scale": lambda: Scale(**common, factor=1.1),
        }
        if transform not in constructors:
            raise ValueError(f"unknown transform: {transform}")
        wrapped = constructors[transform]()
        was_fluid_root = (
            self.fluid_domain is not None and self.fluid_domain.root is node
        )
        tags = self.fluid_domain.tag_objects if self.fluid_domain is not None else ()
        index = self._detach(node)
        self.objects.insert(min(index, len(self.objects)), wrapped)
        self._reindex()
        if was_fluid_root:
            self.fluid_domain = FluidDomain(wrapped, tags)
        else:
            self._refresh_fluid_domain()
        return self.handle_for(wrapped)

    def solid_from_2d(self, handles: list[int], method: str) -> int:
        sections = tuple(self.node(handle) for handle in handles)
        if not sections or not all(isinstance(node, PlacedSDF2D) for node in sections):
            raise ValueError("Solid From 2D requires placed 2D objects")
        placed = tuple(node for node in sections if isinstance(node, PlacedSDF2D))
        common = {
            "name": f"{method}: {', '.join(node.name for node in placed)}",
            "object_id": self._allocate_object_id(),
        }
        if method == "extrude" and len(placed) == 1:
            result: SDFNode = Extrude(**common, section=placed[0], height=1.0)
        elif method == "revolve" and len(placed) == 1:
            result = Revolve(**common, section=placed[0])
        elif method == "sweep" and len(placed) == 1:
            normal = placed[0].normal
            end = tuple(
                placed[0].origin[index] + normal[index]
                for index in range(3)
            )
            result = Sweep(**common, section=placed[0], end=end)
        elif method == "loft_implicit" and len(placed) >= 2:
            result = LoftImplicit(**common, sections=placed)
        else:
            raise ValueError(f"invalid section count for {method}")
        self.objects.append(result)
        self._reindex()
        return self.handle_for(result)

    def set_fluid_root(self, handle: int) -> None:
        node = self.node(handle)
        if node.dimension not in {2, 3}:
            raise ValueError("FluidDomain root must be a 2D or 3D SDF")
        tags = (
            self._compatible_domain_tags(
                node,
                self.fluid_domain.tag_objects,
            )
            if self.fluid_domain is not None
            else ()
        )
        self.fluid_domain = FluidDomain(node, tags)

    @staticmethod
    def _compatible_domain_tags(
        root: SDFNode,
        tags: tuple[PlacedSDF1D | PlacedSDF2D | BoundaryRegion, ...],
    ) -> tuple[PlacedSDF1D | PlacedSDF2D | BoundaryRegion, ...]:
        valid_owner_ids = boundary_owner_ids(root)
        return tuple(
            tag
            for tag in tags
            if (
                root.dimension == 2
                and isinstance(root, PlacedSDF2D)
                and isinstance(tag, PlacedSDF1D)
                and tag.lies_in_plane_of(root)
            )
            or (
                root.dimension == 3
                and isinstance(tag, BoundaryRegion)
                and tag.owner_object_id in valid_owner_ids
            )
            or (
                root.dimension == 3
                and isinstance(tag, PlacedSDF2D)
            )
        )

    def set_tag_enabled(self, handle: int, enabled: bool) -> None:
        node = self.node(handle)
        if not isinstance(node, (PlacedSDF1D, PlacedSDF2D, BoundaryRegion)):
            raise ValueError(
                "only dimension-compatible placed SDFs and BoundaryRegion "
                "objects can tag lattice nodes"
            )
        if self.fluid_domain is None:
            raise ValueError("select a FluidDomain root first")
        tags = list(self.fluid_domain.tag_objects)
        if enabled and node not in tags:
            tags.append(node)
        elif not enabled and node in tags:
            tags.remove(node)
        self.fluid_domain = FluidDomain(self.fluid_domain.root, tuple(tags))

    def add_boundary_region(
        self,
        owner_object_id: int,
        outside_direction: int | None = None,
    ) -> int:
        if self.fluid_domain is None:
            raise ValueError("select a FluidDomain root first")
        if self.fluid_domain.root.dimension == 2:
            return self._add_2d_boundary_region(
                owner_object_id,
                outside_direction,
            )
        owners = {
            node.object_id: node
            for node in self._iter_nodes(self.fluid_domain.root)
        }
        owner = owners.get(owner_object_id)
        if (
            owner is None
            or owner_object_id not in boundary_owner_ids(self.fluid_domain.root)
        ):
            raise ValueError(
                "selected object does not directly control the FluidDomain boundary"
            )
        region = BoundaryRegion(
            name=(
                f"{owner.name} boundary {outside_direction}"
                if outside_direction is not None
                else f"{owner.name} boundary"
            ),
            object_id=self._allocate_object_id(),
            owner_object_id=owner.object_id,
            outside_direction=outside_direction,
        )
        self.boundary_regions.append(region)
        self._reindex()
        self.fluid_domain = FluidDomain(
            self.fluid_domain.root,
            (*self.fluid_domain.tag_objects, region),
        )
        return self.handle_for(region)

    def _add_2d_boundary_region(
        self,
        owner_object_id: int,
        outside_direction: int | None,
    ) -> int:
        assert self.fluid_domain is not None
        root = self.fluid_domain.root
        if not isinstance(root, PlacedSDF2D):
            raise ValueError("2D FluidDomain root must be a PlacedSDF2D")
        if owner_object_id not in boundary_owner_ids(root):
            raise ValueError(
                "selected object does not directly control the FluidDomain boundary"
            )
        if outside_direction is None or not 0 <= outside_direction < 4:
            raise ValueError(
                "2D boundary regions require an outside direction in the range 0..3"
            )
        owner = next(
            (
                node
                for node in self._iter_nodes(root)
                if node.object_id == owner_object_id
            ),
            None,
        )
        if not isinstance(owner, PlacedSDF2D) or owner.profile is None:
            raise ValueError(
                "2D boundary owners must be placed 2D SDF objects"
            )
        u_min, u_max, v_min, v_max = owner.profile.bounds()
        axis_u = np.asarray(owner.axis_u, dtype=np.float64)
        axis_v = np.asarray(owner.axis_v, dtype=np.float64)
        origin = np.asarray(owner.origin, dtype=np.float64)
        if outside_direction in {0, 1}:
            side_u = u_min if outside_direction == 0 else u_max
            center_v = 0.5 * (v_min + v_max)
            line_origin = origin + side_u * axis_u + center_v * axis_v
            line_axis = owner.axis_v
            half_length = 0.5 * (v_max - v_min)
        else:
            side_v = v_min if outside_direction == 2 else v_max
            center_u = 0.5 * (u_min + u_max)
            line_origin = origin + center_u * axis_u + side_v * axis_v
            line_axis = owner.axis_u
            half_length = 0.5 * (u_max - u_min)
        region = PlacedSDF1D(
            name=f"{owner.name} boundary {outside_direction}",
            object_id=self._allocate_object_id(),
            profile=IntervalProfile(half_length=half_length),
            origin=tuple(float(value) for value in line_origin),
            axis_u=line_axis,
        )
        self.objects.append(region)
        self._reindex()
        self.fluid_domain = FluidDomain(
            root,
            (*self.fluid_domain.tag_objects, region),
        )
        return self.handle_for(region)

    def delete(self, handle: int) -> None:
        target = self.node(handle)
        if isinstance(target, BoundaryRegion):
            self.boundary_regions.remove(target)
            self._reindex()
            self._refresh_fluid_domain()
            return
        self._detach(target)
        self._reindex()
        self._refresh_fluid_domain()

    def _remove_from(
        self, current: SDFNode, target: SDFNode
    ) -> tuple[SDFNode | None, bool]:
        if isinstance(current, UnaryTransform):
            assert current.child is not None
            if current.child is target:
                return None, True
            replacement, removed = self._remove_from(current.child, target)
            if removed:
                current.child = replacement
                return current if replacement is not None else None, True
            return current, False
        if isinstance(current, BinaryCSG):
            assert current.left is not None and current.right is not None
            if current.left is target:
                return current.right, True
            if current.right is target:
                return current.left, True
            replacement, removed = self._remove_from(current.left, target)
            if removed:
                current.left = replacement
                return current, True
            replacement, removed = self._remove_from(current.right, target)
            if removed:
                current.right = replacement
                return current, True
        return current, False

    def _detach(self, target: SDFNode) -> int:
        if target in self.objects:
            index = self.objects.index(target)
            self.objects.pop(index)
            return index
        for index, root in enumerate(tuple(self.objects)):
            replacement, removed = self._remove_from(root, target)
            if removed:
                if replacement is None:
                    self.objects.pop(index)
                else:
                    self.objects[index] = replacement
                return index
        raise KeyError("SDF node is not part of this document")

    @staticmethod
    def _contains(root: SDFNode, target: SDFNode) -> bool:
        return root is target or any(
            SceneDocument._contains(child, target) for child in root.children()
        )

    def visual_tree(self) -> SDFTree:
        self.refresh_derived_geometry()
        three_dimensional = [node for node in self.objects if node.dimension == 3]
        if not three_dimensional:
            root: SDFNode = Sphere(
                name="empty_visual_root",
                center=(1_000_000.0, 1_000_000.0, 1_000_000.0),
                radius=0.001,
            )
        else:
            root = three_dimensional[0]
            for node in three_dimensional[1:]:
                root = Union(name="visual_union", left=root, right=node)
        components = list(self.objects)
        if self.fluid_domain is not None:
            for tag in self.fluid_domain.tag_objects:
                if (
                    isinstance(tag, SDFNode)
                    and all(existing is not tag for existing in components)
                ):
                    components.append(tag)
        return SDFTree(root, components=tuple(components))

    def tree(self) -> SDFTree:
        return self.visual_tree()

    def snapshot(self) -> SceneDocument:
        self.refresh_derived_geometry()
        snapshot = deepcopy(self)
        snapshot._handles.clear()
        snapshot._node_handles.clear()
        snapshot._reindex()
        snapshot._refresh_fluid_domain()
        return snapshot

    def node(self, handle: int) -> SceneItem:
        try:
            return self._handles[handle]
        except KeyError as error:
            raise KeyError(f"unknown scene handle {handle}") from error

    def handle_for(self, node: SceneItem) -> int:
        return self._node_handles[id(node)]

    def walk(self) -> Iterator[tuple[int, SceneItem, int | None]]:
        seen: set[int] = set()
        for root in self.objects:
            yield from self._walk_node(root, None, seen)
        for region in self.boundary_regions:
            yield self.handle_for(region), region, None

    def _walk_node(
        self,
        node: SDFNode,
        parent_handle: int | None,
        seen: set[int],
    ) -> Iterator[tuple[int, SDFNode, int | None]]:
        if id(node) in seen:
            return
        seen.add(id(node))
        handle = self.handle_for(node)
        yield handle, node, parent_handle
        for child in node.children():
            yield from self._walk_node(child, handle, seen)

    def refresh_derived_geometry(self) -> None:
        seen: set[int] = set()

        def refresh(node: SDFNode) -> None:
            if id(node) in seen:
                return
            seen.add(id(node))
            for child in node.children():
                refresh(child)
            if isinstance(node, PlacedSDF1D) and len(node.sources) == 2:
                first, second = node.sources
                if not isinstance(first, PlacedSDF1D) or not isinstance(
                    second,
                    PlacedSDF1D,
                ):
                    return
                if not first.is_collinear_with(second):
                    raise ValueError(
                        f"1D boolean '{node.name}' has non-collinear operands"
                    )
                assert first.profile is not None and second.profile is not None
                operation = (
                    node.profile.operation
                    if isinstance(node.profile, BinaryProfile1D)
                    else "union"
                )
                smoothing = (
                    node.profile.smoothing
                    if isinstance(node.profile, BinaryProfile1D)
                    else 0.1
                )
                node.origin = first.origin
                node.axis_u = first.axis_u
                displacement = np.asarray(second.origin) - np.asarray(
                    first.origin
                )
                offset = float(
                    np.dot(displacement, np.asarray(first.axis_u))
                )
                node.profile = BinaryProfile1D(
                    first.profile,
                    OffsetProfile1D(second.profile, offset),
                    operation,
                    smoothing,
                )
            elif isinstance(node, PlacedSDF2D) and len(node.sources) == 2:
                first, second = node.sources
                if not isinstance(first, PlacedSDF2D) or not isinstance(
                    second, PlacedSDF2D
                ):
                    return
                if not first.is_coplanar_with(second):
                    raise ValueError(
                        f"2D boolean '{node.name}' has non-coplanar operands"
                    )
                assert first.profile is not None and second.profile is not None
                operation = (
                    node.profile.operation
                    if isinstance(node.profile, BinaryProfile)
                    else "union"
                )
                smoothing = (
                    node.profile.smoothing
                    if isinstance(node.profile, BinaryProfile)
                    else 0.1
                )
                node.origin = first.origin
                node.axis_u = first.axis_u
                node.axis_v = first.axis_v
                displacement = np.asarray(second.origin) - np.asarray(first.origin)
                offset = (
                    float(np.dot(displacement, np.asarray(first.axis_u))),
                    float(np.dot(displacement, np.asarray(first.axis_v))),
                )
                node.profile = BinaryProfile(
                    first.profile,
                    OffsetProfile(second.profile, offset),
                    operation,
                    smoothing,
                )

        for root in self.objects:
            refresh(root)

    def _reindex(self) -> None:
        old_handles = dict(self._node_handles)
        self._handles.clear()
        self._node_handles.clear()
        seen: set[int] = set()
        for root in self.objects:
            for node in self._iter_nodes(root):
                if id(node) in seen:
                    continue
                seen.add(id(node))
                if node.object_id <= 0:
                    node.object_id = self._allocate_object_id()
                handle = old_handles.get(id(node))
                if handle is None:
                    handle = self._next_handle
                    self._next_handle += 1
                self._handles[handle] = node
                self._node_handles[id(node)] = handle
        for region in self.boundary_regions:
            if region.object_id <= 0:
                region.object_id = self._allocate_object_id()
            handle = old_handles.get(id(region))
            if handle is None:
                handle = self._next_handle
                self._next_handle += 1
            self._handles[handle] = region
            self._node_handles[id(region)] = handle

    def _iter_nodes(self, node: SDFNode) -> Iterator[SDFNode]:
        yield node
        for child in node.children():
            yield from self._iter_nodes(child)

    def _refresh_fluid_domain(self) -> None:
        live_sdf_nodes = tuple(
            node for root in self.objects for node in self._iter_nodes(root)
        )
        live_sdf_ids = {node.object_id for node in live_sdf_nodes}
        self.boundary_regions = [
            region
            for region in self.boundary_regions
            if region.owner_object_id in live_sdf_ids
        ]
        self._reindex()
        if self.fluid_domain is None:
            return
        live = {id(node) for node in (*live_sdf_nodes, *self.boundary_regions)}
        if id(self.fluid_domain.root) not in live:
            self.fluid_domain = None
            return
        tags = tuple(
            tag for tag in self.fluid_domain.tag_objects if id(tag) in live
        )
        self.fluid_domain = FluidDomain(self.fluid_domain.root, tags)
