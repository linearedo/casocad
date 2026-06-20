from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field, replace
from math import cos, radians, sin
from typing import Iterator

import numpy as np

from .boundary import BoundaryRegion
from .boundary_patches import (
    BoundaryPatchHit,
    BoundaryCurvePatch,
    BoundaryIntervalSelector,
    BoundarySelector,
    boundary_owner_ids,
    boundary_interval_selector_from_node,
    boundary_patches,
    boundary_selector_from_node,
)
from .mesher import FluidDomain
from .sdf import (
    BezierCurveProfile,
    BezierSurfaceProfile,
    BezierTube,
    BinaryProfile1D,
    BinaryProfile,
    BoundingBox3D,
    Box,
    BoxFrame,
    CappedCone,
    Cone,
    CircleProfile,
    Cylinder,
    Difference,
    EllipseProfile,
    Extrude,
    Intersection,
    OffsetProfile,
    OffsetProfile1D,
    PlacedSDF1D,
    PlacedPolyline2D,
    PlacedSDF2D,
    PolygonProfile,
    PolylineTube,
    PolylineProfile,
    Pyramid,
    RectangleProfile,
    RegularPolygonProfile,
    Rotate,
    RoundedRectangleProfile,
    SDFNode,
    SDFTree,
    Scale,
    SegmentProfile,
    SmoothUnion,
    Sphere,
    SquareProfile,
    Torus,
    Translate,
    Union,
)
from .sdf.csg import BinaryCSG
from .sdf.solid_from_2d import Revolve
from .sdf.transforms import UnaryTransform

Primitive3D = (
    Sphere
    | Box
    | BoxFrame
    | CappedCone
    | Cone
    | Cylinder
    | Pyramid
    | Torus
    | PolylineTube
    | BezierTube
)
SceneItem = SDFNode | BoundaryRegion
DEFAULT_BEZIER_POLYCURVE_POINTS = (
    (-0.65, -0.35),
    (-0.3, 0.55),
    (0.0, -0.05),
    (0.3, -0.65),
    (0.65, 0.35),
)
REFERENCE_PLANE_AXES_3D = {
    "xy": ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    "xz": ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    "yz": ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
}


@dataclass
class SceneDocument:
    objects: list[SDFNode] = field(default_factory=list)
    fluid_domain: FluidDomain | None = None
    boundary_regions: list[BoundaryRegion] = field(default_factory=list)
    version: int = 0
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

    def mark_changed(self) -> None:
        self.version += 1

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
            "capped_cone": lambda: CappedCone(
                **common,
                radius_a=0.45,
                radius_b=0.25,
                half_height=0.6,
            ),
            "cone": lambda: Cone(**common, radius=0.45, half_height=0.6),
            "pyramid": lambda: Pyramid(
                **common,
                base_half_size=0.45,
                half_height=0.6,
            ),
            "box_frame": lambda: BoxFrame(
                **common,
                half_size=(0.5, 0.5, 0.5),
                thickness=0.08,
            ),
            "torus": lambda: Torus(**common, major_radius=0.5, minor_radius=0.15),
            "polyline_tube": lambda: PolylineTube(**common),
            "bezier_tube": lambda: BezierTube(**common),
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
            "polygon": PolygonProfile,
            "bezier_surface": BezierSurfaceProfile,
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
            name=name or f"segment_{object_id}",
            object_id=object_id,
            profile=SegmentProfile(),
        )

    def create_polyline(
        self,
        points: tuple[tuple[float, float], ...] | list[tuple[float, float]],
        name: str | None = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0),
        axis_v: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> PlacedPolyline2D:
        object_id = self._allocate_object_id()
        return PlacedPolyline2D(
            name=name or f"polyline_{object_id}",
            object_id=object_id,
            profile=PolylineProfile(points=tuple(points)),
            origin=origin,
            axis_u=axis_u,
            axis_v=axis_v,
        )

    def create_bezier_curve(
        self,
        points: tuple[tuple[float, float], ...] | list[tuple[float, float]],
        name: str | None = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0),
        axis_v: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> PlacedPolyline2D:
        object_id = self._allocate_object_id()
        return PlacedPolyline2D(
            name=name or f"bezier_curve_{object_id}",
            object_id=object_id,
            profile=BezierCurveProfile(points=tuple(points)),
            origin=origin,
            axis_u=axis_u,
            axis_v=axis_v,
        )

    def create_polygon(
        self,
        points: tuple[tuple[float, float], ...] | list[tuple[float, float]],
        name: str | None = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0),
        axis_v: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> PlacedSDF2D:
        object_id = self._allocate_object_id()
        return PlacedSDF2D(
            name=name or f"polygon_{object_id}",
            object_id=object_id,
            profile=PolygonProfile(points=tuple(points)),
            origin=origin,
            axis_u=axis_u,
            axis_v=axis_v,
        )

    def add_placed_2d_profile(
        self,
        profile: PolygonProfile | BezierSurfaceProfile,
        name: str | None = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0),
        axis_v: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> int:
        object_id = self._allocate_object_id()
        node = PlacedSDF2D(
            name=name or f"{profile.kind}_{object_id}",
            object_id=object_id,
            profile=profile,
            origin=origin,
            axis_u=axis_u,
            axis_v=axis_v,
        )
        self.objects.append(node)
        self._reindex()
        self.mark_changed()
        return self.handle_for(node)

    def add_polyline(
        self,
        points: tuple[tuple[float, float], ...] | list[tuple[float, float]],
        name: str | None = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0),
        axis_v: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> int:
        node = self.create_polyline(points, name, origin, axis_u, axis_v)
        self.objects.append(node)
        self._reindex()
        self.mark_changed()
        return self.handle_for(node)

    def add_bezier_curve(
        self,
        points: tuple[tuple[float, float], ...] | list[tuple[float, float]],
        name: str | None = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0),
        axis_v: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> int:
        node = self.create_bezier_curve(points, name, origin, axis_u, axis_v)
        self.objects.append(node)
        self._reindex()
        self.mark_changed()
        return self.handle_for(node)

    def add_polyline_tube(
        self,
        points: tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]],
        name: str | None = None,
        radius: float = 0.12,
        inner_radius: float = 0.0,
    ) -> int:
        object_id = self._allocate_object_id()
        node = PolylineTube(
            name=name or f"polyline_tube_{object_id}",
            object_id=object_id,
            points=tuple(points),
            radius=radius,
            inner_radius=inner_radius,
        )
        self.objects.append(node)
        self._reindex()
        self.mark_changed()
        return self.handle_for(node)

    def add_bezier_tube(
        self,
        points: tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]],
        name: str | None = None,
        radius: float = 0.12,
        inner_radius: float = 0.0,
    ) -> int:
        object_id = self._allocate_object_id()
        node = BezierTube(
            name=name or f"bezier_tube_{object_id}",
            object_id=object_id,
            points=tuple(points),
            radius=radius,
            inner_radius=inner_radius,
        )
        self.objects.append(node)
        self._reindex()
        self.mark_changed()
        return self.handle_for(node)

    def add_polygon(
        self,
        points: tuple[tuple[float, float], ...] | list[tuple[float, float]],
        name: str | None = None,
        origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
        axis_u: tuple[float, float, float] = (1.0, 0.0, 0.0),
        axis_v: tuple[float, float, float] = (0.0, 1.0, 0.0),
    ) -> int:
        node = self.create_polygon(points, name, origin, axis_u, axis_v)
        self.objects.append(node)
        self._reindex()
        self.mark_changed()
        return self.handle_for(node)

    def add_point_shape_from_world_points(
        self,
        kind: str,
        points: tuple[tuple[float, float, float], ...] | list[tuple[float, float, float]],
        reference_plane: str,
    ) -> int:
        if reference_plane not in REFERENCE_PLANE_AXES_3D:
            raise ValueError(f"unknown reference plane: {reference_plane}")
        if kind not in {
            "polyline",
            "bezier_curve",
            "bezier_polycurve",
            "polyline_tube",
            "bezier_tube",
            "bezier_surface",
            "polygon",
        }:
            raise ValueError(f"unsupported point shape: {kind}")
        minimum_points = 2 if kind in {"polyline", "polyline_tube"} else 3
        if len(points) < minimum_points:
            label = "polyline tube" if kind == "polyline_tube" else "polyline"
            raise ValueError(
                f"{label} requires at least two points"
                if kind in {"polyline", "polyline_tube"}
                else f"{kind} requires at least three points"
            )
        if kind == "bezier_curve" and len(points) != 3:
            raise ValueError("bezier curve requires exactly three points")
        if kind == "bezier_polycurve" and len(points) % 2 == 0:
            raise ValueError(
                "bezier polycurve requires an odd point count: "
                "anchor, control, anchor"
            )
        if kind == "bezier_tube" and len(points) % 2 == 0:
            raise ValueError(
                "bezier tube requires an odd point count: "
                "anchor, control, anchor"
            )
        if kind == "bezier_surface" and len(points) % 2 == 0:
            raise ValueError(
                "bezier surface requires an odd point count: "
                "anchor, control, anchor"
            )
        axis_u, axis_v = REFERENCE_PLANE_AXES_3D[reference_plane]
        origin = tuple(float(value) for value in points[0])
        origin_array = np.asarray(origin, dtype=np.float64)
        axis_u_array = np.asarray(axis_u, dtype=np.float64)
        axis_v_array = np.asarray(axis_v, dtype=np.float64)
        local_points = tuple(
            (
                float(np.dot(np.asarray(point, dtype=np.float64) - origin_array, axis_u_array)),
                float(np.dot(np.asarray(point, dtype=np.float64) - origin_array, axis_v_array)),
            )
            for point in points
        )
        if kind == "polyline":
            return self.add_polyline(
                local_points,
                origin=origin,
                axis_u=axis_u,
                axis_v=axis_v,
            )
        if kind == "polyline_tube":
            return self.add_polyline_tube(points)
        if kind == "bezier_tube":
            return self.add_bezier_tube(points)
        if kind in {"bezier_curve", "bezier_polycurve"}:
            return self.add_bezier_curve(
                local_points,
                origin=origin,
                axis_u=axis_u,
                axis_v=axis_v,
            )
        if kind == "bezier_surface":
            return self.add_placed_2d_profile(
                BezierSurfaceProfile(points=local_points),
                name=None,
                origin=origin,
                axis_u=axis_u,
                axis_v=axis_v,
            )
        return self.add_polygon(
            local_points,
            origin=origin,
            axis_u=axis_u,
            axis_v=axis_v,
        )

    def add_primitive(self, kind: str) -> int:
        if kind in {"segment", "interval"}:
            node: SDFNode = self.create_placed_1d()
        elif kind == "polyline":
            node = self.create_polyline(PolylineProfile().points)
        elif kind == "bezier_curve":
            node = self.create_bezier_curve(BezierCurveProfile().points)
        elif kind == "bezier_polycurve":
            node = self.create_bezier_curve(DEFAULT_BEZIER_POLYCURVE_POINTS)
        elif kind == "polyline_tube":
            node = self.create_primitive(kind)
        elif kind == "bezier_tube":
            node = self.create_primitive(kind)
        elif kind == "bezier_surface":
            node = self.create_placed_2d(kind)
        elif kind in {
            "circle",
            "rectangle",
            "square",
            "rounded_rectangle",
            "ellipse",
            "regular_polygon",
            "polygon",
        }:
            node = self.create_placed_2d(kind)
        else:
            node = self.create_primitive(kind)
            offset = 0.25 * len(self.objects)
            if hasattr(node, "center"):
                node.center = (offset, 0.0, 0.0)
        self.objects.append(node)
        self._reindex()
        self.mark_changed()
        return self.handle_for(node)

    def add_primitive_from_drag(
        self,
        kind: str,
        start: tuple[float, float, float],
        end: tuple[float, float, float],
        parameters: dict[str, float] | None = None,
    ) -> int:
        parameters = parameters or {}
        start_array = np.asarray(start, dtype=np.float64)
        end_array = np.asarray(end, dtype=np.float64)
        center = 0.5 * (start_array + end_array)
        axis_a, axis_b = self._drag_plane_axes(start_array, end_array)
        axis_u = tuple(1.0 if index == axis_a else 0.0 for index in range(3))
        axis_v = tuple(1.0 if index == axis_b else 0.0 for index in range(3))
        extent_a = float(
            max(abs(end_array[axis_a] - start_array[axis_a]) * 0.5, 0.05)
        )
        extent_b = float(
            max(abs(end_array[axis_b] - start_array[axis_b]) * 0.5, 0.05)
        )
        planar_delta = end_array[[axis_a, axis_b]] - start_array[[axis_a, axis_b]]
        radius = max(float(np.linalg.norm(planar_delta)) * 0.5, 0.05)
        if kind in {"segment", "interval"}:
            direction = end_array - start_array
            length = float(np.linalg.norm(direction))
            node = self.create_placed_1d()
            node.origin = tuple(float(value) for value in center)
            node.axis_u = (
                tuple(float(value) for value in direction / length)
                if length > 1e-12
                else (1.0, 0.0, 0.0)
            )
            node.profile = SegmentProfile(half_length=max(0.5 * length, 0.05))
            node.__post_init__()
        elif kind == "polyline":
            node = self.create_polyline(
                ((-extent_a, -extent_b), (extent_a, extent_b)),
                origin=tuple(float(value) for value in center),
                axis_u=axis_u,
                axis_v=axis_v,
            )
        elif kind == "bezier_curve":
            node = self.create_bezier_curve(
                ((-extent_a, 0.0), (0.0, extent_b), (extent_a, 0.0)),
                origin=tuple(float(value) for value in center),
                axis_u=axis_u,
                axis_v=axis_v,
            )
        elif kind == "bezier_polycurve":
            node = self.create_bezier_curve(
                (
                    (-extent_a, 0.0),
                    (-0.5 * extent_a, extent_b),
                    (0.0, 0.0),
                    (0.5 * extent_a, -extent_b),
                    (extent_a, 0.0),
                ),
                origin=tuple(float(value) for value in center),
                axis_u=axis_u,
                axis_v=axis_v,
            )
        elif kind == "polyline_tube":
            node = PolylineTube(
                name=f"polyline_tube_{self._next_object_id}",
                object_id=self._allocate_object_id(),
                points=(
                    tuple(float(value) for value in start_array),
                    tuple(float(value) for value in end_array),
                ),
                radius=float(parameters.get("radius", 0.12)),
            )
        elif kind == "bezier_tube":
            control = 0.5 * (start_array + end_array)
            axis_offset = np.zeros(3, dtype=np.float64)
            axis_offset[axis_b] = extent_b
            node = BezierTube(
                name=f"bezier_tube_{self._next_object_id}",
                object_id=self._allocate_object_id(),
                points=(
                    tuple(float(value) for value in start_array),
                    tuple(float(value) for value in control + axis_offset),
                    tuple(float(value) for value in end_array),
                ),
                radius=float(parameters.get("radius", 0.12)),
            )
        elif kind in {
            "circle",
            "rectangle",
            "square",
            "rounded_rectangle",
            "ellipse",
            "regular_polygon",
            "polygon",
        }:
            node = self.create_placed_2d(kind)
            assert isinstance(node, PlacedSDF2D)
            node.origin = tuple(float(value) for value in center)
            node.axis_u = axis_u
            node.axis_v = axis_v
            if kind == "circle":
                node.profile = CircleProfile(radius=radius)
            elif kind == "rectangle":
                node.profile = RectangleProfile(half_size=(extent_a, extent_b))
            elif kind == "square":
                node.profile = SquareProfile(half_size=max(extent_a, extent_b))
            elif kind == "rounded_rectangle":
                half_size = (extent_a, extent_b)
                node.profile = RoundedRectangleProfile(
                    half_size=half_size,
                    corner_radius=max(0.01, min(half_size) * 0.2),
                )
            elif kind == "ellipse":
                node.profile = EllipseProfile(semi_axes=(extent_a, extent_b))
            elif kind == "polygon":
                node.profile = PolygonProfile(
                    points=(
                        (-extent_a, -extent_b),
                        (extent_a, -extent_b),
                        (extent_a, extent_b),
                        (-extent_a, extent_b),
                    )
                )
            elif kind == "bezier_surface":
                node.profile = BezierSurfaceProfile(
                    points=(
                        (-extent_a, -extent_b),
                        (-0.35 * extent_a, extent_b),
                        (0.25 * extent_a, 0.35 * extent_b),
                        (0.85 * extent_a, -0.25 * extent_b),
                        (extent_a, -extent_b),
                    )
                )
            else:
                node.profile = RegularPolygonProfile(radius=radius)
            node.__post_init__()
        else:
            node = self.create_primitive(kind)
            world_center = tuple(float(value) for value in center)
            if isinstance(node, Sphere):
                node.center = world_center
                node.radius = radius
            elif isinstance(node, Box):
                node.center = world_center
                box_delta = np.abs(end_array - start_array)
                if np.count_nonzero(box_delta > 1e-9) == 3:
                    half_size = [
                        float(max(0.5 * value, 0.05))
                        for value in box_delta
                    ]
                else:
                    fallback = max(extent_a, extent_b)
                    half_size = [fallback, fallback, fallback]
                    half_size[axis_a] = extent_a
                    half_size[axis_b] = extent_b
                node.half_size = tuple(half_size)
            elif isinstance(node, Cylinder):
                node.center = world_center
                radial_delta = float(np.linalg.norm((end_array - start_array)[:2]))
                height_delta = abs(float(end_array[2] - start_array[2]))
                node.radius = max(0.5 * radial_delta, 0.05)
                node.half_height = (
                    max(0.5 * height_delta, 0.05)
                    if height_delta > 1e-9
                    else max(extent_a, extent_b)
                )
            elif isinstance(node, CappedCone):
                node.center = world_center
                radial_delta = float(np.linalg.norm((end_array - start_array)[:2]))
                height_delta = abs(float(end_array[2] - start_array[2]))
                node.radius_a = max(0.5 * radial_delta, 0.05)
                top_diameter = parameters.get("top_diameter")
                node.radius_b = (
                    max(0.5 * float(top_diameter), 0.02)
                    if top_diameter is not None
                    else max(node.radius_a * 0.45, 0.025)
                )
                node.half_height = (
                    max(0.5 * height_delta, 0.05)
                    if height_delta > 1e-9
                    else max(extent_a, extent_b)
                )
            elif isinstance(node, Cone):
                node.center = world_center
                radial_delta = float(np.linalg.norm((end_array - start_array)[:2]))
                height_delta = abs(float(end_array[2] - start_array[2]))
                node.radius = max(0.5 * radial_delta, 0.05)
                node.half_height = (
                    max(0.5 * height_delta, 0.05)
                    if height_delta > 1e-9
                    else max(extent_a, extent_b)
                )
            elif isinstance(node, Pyramid):
                node.center = world_center
                box_delta = np.abs(end_array - start_array)
                node.base_half_size = max(extent_a, extent_b, 0.05)
                node.half_height = (
                    max(0.5 * float(box_delta[2]), 0.05)
                    if box_delta[2] > 1e-9
                    else max(extent_a, extent_b)
                )
            elif isinstance(node, BoxFrame):
                node.center = world_center
                box_delta = np.abs(end_array - start_array)
                if np.count_nonzero(box_delta > 1e-9) == 3:
                    half_size = [
                        float(max(0.5 * value, 0.05))
                        for value in box_delta
                    ]
                else:
                    fallback = max(extent_a, extent_b)
                    half_size = [fallback, fallback, fallback]
                    half_size[axis_a] = extent_a
                    half_size[axis_b] = extent_b
                node.half_size = tuple(half_size)
                node.thickness = max(min(node.half_size) * 0.14, 0.015)
            elif isinstance(node, Torus):
                node.center = world_center
                node.major_radius = radius
                minor_diameter = parameters.get("minor_diameter")
                node.minor_radius = (
                    max(0.5 * float(minor_diameter), 0.02)
                    if minor_diameter is not None
                    else max(radius * 0.25, 0.02)
                )
        self.objects.append(node)
        self._reindex()
        self.mark_changed()
        return self.handle_for(node)

    def create_polygon_from_polyline(self, handle: int) -> int:
        node = self.node(handle)
        if (
            not isinstance(node, PlacedPolyline2D)
            or not isinstance(node.profile, PolylineProfile)
        ):
            raise ValueError("select one polyline to create a polygon")
        polygon = self.create_polygon(
            node.profile.points,
            name=f"polygon_from_{node.name}",
            origin=node.origin,
            axis_u=node.axis_u,
            axis_v=node.axis_v,
        )
        self.objects.append(polygon)
        self._reindex()
        self.mark_changed()
        return self.handle_for(polygon)

    @staticmethod
    def _drag_plane_axes(
        start: np.ndarray,
        end: np.ndarray,
    ) -> tuple[int, int]:
        delta = np.abs(end - start)
        tolerance = 1e-9
        if delta[2] <= tolerance:
            return 0, 1
        if delta[1] <= tolerance:
            return 0, 2
        if delta[0] <= tolerance:
            return 1, 2
        axes = tuple(int(axis) for axis in np.argsort(delta)[-2:])
        return tuple(sorted(axes))

    def copy_nodes(self, handles: list[int]) -> list[SDFNode]:
        selected = [
            node
            for handle in handles
            if isinstance((node := self.node(handle)), SDFNode)
        ]
        selected_ids = {id(node) for node in selected}
        roots = [
            node
            for node in selected
            if not any(
                other is not node
                and id(other) in selected_ids
                and self._contains(other, node)
                for other in selected
            )
        ]
        if not roots:
            raise ValueError("select at least one SDF object to copy")
        return [deepcopy(node) for node in roots]

    def paste_nodes(
        self,
        nodes: list[SDFNode],
        offset: tuple[float, float, float] = (0.1, 0.1, 0.0),
    ) -> list[int]:
        pasted: list[SDFNode] = []
        for node in nodes:
            clone = deepcopy(node)
            self._assign_fresh_object_ids(clone)
            clone.name = f"{clone.name} copy"
            if not self._translate_copy_in_place(clone, offset):
                clone = Translate(
                    name=clone.name,
                    object_id=self._allocate_object_id(),
                    child=clone,
                    offset=offset,
                )
            pasted.append(clone)
        self.objects.extend(pasted)
        self._reindex()
        self._refresh_fluid_domain()
        self.mark_changed()
        return [self.handle_for(node) for node in pasted]

    def move_object(
        self,
        handle: int,
        delta: tuple[float, float, float],
    ) -> int:
        node = self.node(handle)
        if isinstance(node, (PlacedSDF1D, PlacedPolyline2D, PlacedSDF2D)):
            node.origin = tuple(
                node.origin[index] + delta[index] for index in range(3)
            )
            node.__post_init__()
            self.mark_changed()
            return handle
        if isinstance(node, (Sphere, Box, BoxFrame, CappedCone, Cone, Cylinder, Pyramid, Torus)):
            node.center = tuple(
                node.center[index] + delta[index] for index in range(3)
            )
            self.mark_changed()
            return handle
        if isinstance(node, (PolylineTube, BezierTube)):
            node.points = tuple(
                tuple(point[index] + delta[index] for index in range(3))
                for point in node.points
            )
            self.mark_changed()
            return handle
        if isinstance(node, Translate):
            node.offset = tuple(
                node.offset[index] + delta[index] for index in range(3)
            )
            self.mark_changed()
            return handle
        if self._translate_copy_in_place(node, delta):
            self.mark_changed()
            return handle
        wrapped_handle = self.wrap_transform(handle, "translate")
        wrapped = self.node(wrapped_handle)
        assert isinstance(wrapped, Translate)
        wrapped.offset = delta
        return wrapped_handle

    @staticmethod
    def _rotation_matrix(axis: str, angle_degrees: float) -> np.ndarray:
        angle = radians(angle_degrees)
        c = cos(angle)
        s = sin(angle)
        if axis == "x":
            return np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)))
        if axis == "y":
            return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)))
        if axis == "z":
            return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)))
        raise ValueError("rotation axis must be x, y, or z")

    @staticmethod
    def _box_center(box: BoundingBox3D) -> tuple[float, float, float]:
        return (
            (box.x_min + box.x_max) * 0.5,
            (box.y_min + box.y_max) * 0.5,
            (box.z_min + box.z_max) * 0.5,
        )

    @classmethod
    def _rotate_point(
        cls,
        point: tuple[float, float, float],
        axis: str,
        angle_degrees: float,
        pivot: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        matrix = cls._rotation_matrix(axis, angle_degrees)
        rotated = (
            matrix
            @ (
                np.asarray(point, dtype=np.float64)
                - np.asarray(pivot, dtype=np.float64)
            )
            + np.asarray(pivot, dtype=np.float64)
        )
        return tuple(float(value) for value in rotated)

    @classmethod
    def _rotate_vector(
        cls,
        vector: tuple[float, float, float],
        axis: str,
        angle_degrees: float,
    ) -> tuple[float, float, float]:
        rotated = cls._rotation_matrix(axis, angle_degrees) @ np.asarray(
            vector,
            dtype=np.float64,
        )
        return tuple(float(value) for value in rotated)

    def rotate_object(
        self,
        handle: int,
        axis: str,
        angle_degrees: float,
        pivot: tuple[float, float, float] | None = None,
    ) -> int:
        if abs(angle_degrees) <= 1e-9:
            return handle
        node = self.node(handle)
        if not isinstance(node, SDFNode):
            raise ValueError("only SDF objects can be rotated")
        if axis not in {"x", "y", "z"}:
            raise ValueError("rotation axis must be x, y, or z")
        if pivot is None:
            pivot = self._box_center(node.bounding_box())
        if self._rotate_node_in_place(node, axis, angle_degrees, pivot):
            self.mark_changed()
            return handle
        raise ValueError("only SDF objects with editable placement can be rotated")

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
        domain_selectors = (
            current_domain.selector_objects if current_domain is not None else ()
        )

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
                self._compatible_domain_selectors(combined, domain_selectors),
            )
        else:
            self._refresh_fluid_domain()
        self.mark_changed()
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
        selectors = (
            self.fluid_domain.selector_objects if self.fluid_domain is not None else ()
        )
        index = self._detach(node)
        self.objects.insert(min(index, len(self.objects)), wrapped)
        self._reindex()
        if was_fluid_root:
            self.fluid_domain = FluidDomain(wrapped, tags, selectors)
        else:
            self._refresh_fluid_domain()
        self.mark_changed()
        return self.handle_for(wrapped)

    def solid_from_2d(
        self,
        handles: list[int],
        method: str,
        signed_height: float | None = None,
        revolve_axis: str = "v",
        revolve_axis_origin: tuple[float, float, float] | None = None,
        revolve_axis_direction: tuple[float, float, float] | None = None,
        revolve_radial_direction: tuple[float, float, float] | None = None,
        revolve_angle_degrees: float = 360.0,
    ) -> int:
        sections = tuple(self.node(handle) for handle in handles)
        if not sections or not all(isinstance(node, PlacedSDF2D) for node in sections):
            raise ValueError("Solid From 2D requires placed 2D objects")
        placed = tuple(node for node in sections if isinstance(node, PlacedSDF2D))
        common = {
            "name": f"{method}: {', '.join(node.name for node in placed)}",
            "object_id": self._allocate_object_id(),
        }
        if method == "extrude" and len(placed) == 1:
            height = 1.0 if signed_height is None else abs(float(signed_height))
            if height <= 0.0 or not np.isfinite(height):
                raise ValueError("extrude height must be finite and positive")
            center_offset = 0.0 if signed_height is None else float(signed_height) * 0.5
            result: SDFNode = Extrude(
                **common,
                section=placed[0],
                height=height,
                center_offset=center_offset,
            )
        elif method == "revolve" and len(placed) == 1:
            result = Revolve(
                **common,
                section=placed[0],
                axis=revolve_axis,
                axis_origin=revolve_axis_origin,
                axis_direction=revolve_axis_direction,
                radial_direction=revolve_radial_direction,
                angle_degrees=revolve_angle_degrees,
            )
        else:
            raise ValueError(f"invalid section count for {method}")
        self.objects.append(result)
        self._reindex()
        self.mark_changed()
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
        selectors = (
            self._compatible_domain_selectors(
                node,
                self.fluid_domain.selector_objects,
            )
            if self.fluid_domain is not None
            else ()
        )
        self.fluid_domain = FluidDomain(node, tags, selectors)
        self.mark_changed()

    @staticmethod
    def _compatible_domain_tags(
        root: SDFNode,
        tags: tuple[PlacedSDF1D | PlacedPolyline2D | PlacedSDF2D | BoundaryRegion, ...],
    ) -> tuple[PlacedSDF1D | PlacedPolyline2D | PlacedSDF2D | BoundaryRegion, ...]:
        valid_owner_ids = boundary_owner_ids(root)
        return tuple(
            tag
            for tag in tags
            if (
                root.dimension == 2
                and isinstance(root, PlacedSDF2D)
                and (
                    (
                        isinstance(tag, (PlacedSDF1D, PlacedPolyline2D))
                        and tag.lies_in_plane_of(root)
                    )
                    or (
                        isinstance(tag, BoundaryRegion)
                        and tag.owner_object_id in valid_owner_ids
                    )
                )
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

    @staticmethod
    def _compatible_domain_selectors(
        root: SDFNode,
        selectors: tuple[SDFNode, ...],
    ) -> tuple[SDFNode, ...]:
        return tuple(
            selector
            for selector in selectors
            if (
                root.dimension == 3
                and isinstance(selector, SDFNode)
            )
            or (
                root.dimension == 2
                and isinstance(root, PlacedSDF2D)
                and isinstance(selector, (PlacedSDF1D, PlacedPolyline2D))
                and selector.lies_in_plane_of(root)
            )
        )

    def set_tag_enabled(self, handle: int, enabled: bool) -> None:
        node = self.node(handle)
        if not isinstance(
            node,
            (PlacedSDF1D, PlacedPolyline2D, PlacedSDF2D, BoundaryRegion),
        ):
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
        self.fluid_domain = FluidDomain(
            self.fluid_domain.root,
            tuple(tags),
            self.fluid_domain.selector_objects,
        )
        self.mark_changed()

    def add_boundary_region(
        self,
        owner_object_id: int,
        outside_direction: int | None = None,
        patch_id: str | None = None,
        patch_type: str | None = None,
        selector: BoundarySelector | None = None,
    ) -> int:
        if self.fluid_domain is None:
            raise ValueError("select a FluidDomain root first")
        if self.fluid_domain.root.dimension == 2:
            return self._add_2d_boundary_region(
                owner_object_id,
                outside_direction,
                patch_id,
                patch_type,
            )
        owners = {
            node.object_id: node
            for node in self._iter_nodes(self.fluid_domain.root)
        }
        owner = owners.get(owner_object_id)
        available_patches = {
            (patch.owner_object_id, patch.patch_id): patch
            for patch in boundary_patches(self.fluid_domain.root)
        }
        patch = (
            available_patches.get((owner_object_id, patch_id))
            if patch_id is not None
            else None
        )
        if (
            owner is None
            or owner_object_id not in boundary_owner_ids(self.fluid_domain.root)
        ):
            raise ValueError(
                "selected object does not directly control the FluidDomain boundary"
            )
        if patch_id is not None and patch is None:
            raise ValueError("selected boundary patch is not part of the FluidDomain")
        region = BoundaryRegion(
            name=(
                f"{owner.name} {patch_id}"
                if patch_id is not None
                else f"{owner.name} boundary {outside_direction}"
                if outside_direction is not None
                else f"{owner.name} boundary"
            ),
            object_id=self._allocate_object_id(),
            owner_object_id=owner.object_id,
            outside_direction=(
                outside_direction
                if outside_direction is not None
                else patch.outside_direction
                if patch is not None
                else None
            ),
            patch_id=patch_id,
            patch_type=(
                patch_type
                if patch_type is not None
                else patch.patch_type
                if patch is not None
                else None
            ),
            selector_id=selector.selector_id if selector is not None else None,
            selector_type=selector.selector_type if selector is not None else None,
            selector_side=selector.side if selector is not None else "inside",
            selector_start=(
                selector.start
                if isinstance(selector, BoundaryIntervalSelector)
                else None
            ),
            selector_end=(
                selector.end
                if isinstance(selector, BoundaryIntervalSelector)
                else None
            ),
        )
        self.boundary_regions.append(region)
        self._reindex()
        self.fluid_domain = FluidDomain(
            self.fluid_domain.root,
            (*self.fluid_domain.tag_objects, region),
            self.fluid_domain.selector_objects,
        )
        self.mark_changed()
        return self.handle_for(region)

    def _add_2d_boundary_region(
        self,
        owner_object_id: int,
        outside_direction: int | None,
        patch_id: str | None,
        patch_type: str | None,
    ) -> int:
        assert self.fluid_domain is not None
        root = self.fluid_domain.root
        if not isinstance(root, PlacedSDF2D):
            raise ValueError("2D FluidDomain root must be a PlacedSDF2D")
        if owner_object_id not in boundary_owner_ids(root):
            raise ValueError(
                "selected object does not directly control the FluidDomain boundary"
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
        available_patches = {
            (patch.owner_object_id, patch.patch_id): patch
            for patch in boundary_patches(root)
        }
        patch = (
            available_patches.get((owner_object_id, patch_id))
            if patch_id is not None
            else next(
                (
                    candidate
                    for candidate in available_patches.values()
                    if candidate.owner_object_id == owner_object_id
                    and candidate.outside_direction == outside_direction
                ),
                None,
            )
        )
        if patch is None:
            raise ValueError("selected boundary patch is not part of the FluidDomain")
        region = BoundaryRegion(
            name=(
                f"{owner.name} {patch.patch_id}"
                if patch.patch_id is not None
                else f"{owner.name} boundary"
            ),
            object_id=self._allocate_object_id(),
            owner_object_id=owner.object_id,
            outside_direction=patch.outside_direction,
            patch_id=patch.patch_id,
            patch_type=patch_type if patch_type is not None else patch.patch_type,
        )
        self.boundary_regions.append(region)
        self._reindex()
        self.fluid_domain = FluidDomain(
            root,
            (*self.fluid_domain.tag_objects, region),
            self.fluid_domain.selector_objects,
        )
        self.mark_changed()
        return self.handle_for(region)

    def add_boundary_region_from_hit(self, hit: BoundaryPatchHit) -> int:
        return self.add_boundary_region(
            hit.owner_object_id,
            hit.outside_direction,
            hit.patch_id,
            hit.patch_type,
            hit.selector,
        )

    def add_boundary_selector_region(
        self,
        base_region: BoundaryRegion,
        selector: SDFNode,
    ) -> int:
        if self.fluid_domain is None:
            raise ValueError("select a FluidDomain root first")
        if base_region not in self.boundary_regions:
            raise ValueError("base boundary region is not part of this document")
        if base_region.patch_id is None:
            raise ValueError("base boundary region must identify a boundary patch")
        live_sdf_nodes = tuple(
            node for root in self.objects for node in self._iter_nodes(root)
        )
        if all(node is not selector for node in live_sdf_nodes):
            raise ValueError("boundary selector object is not part of this document")
        selector_metadata = self._boundary_selector_metadata(
            base_region,
            selector,
        )
        if selector_metadata is None:
            raise ValueError(
                "boundary selectors must be compatible SDF cutter objects"
            )
        region = BoundaryRegion(
            name=f"{base_region.name} / {selector.name}",
            object_id=self._allocate_object_id(),
            owner_object_id=base_region.owner_object_id,
            outside_direction=base_region.outside_direction,
            patch_id=base_region.patch_id,
            patch_type=base_region.patch_type,
            selector_id=selector_metadata.selector_id,
            selector_type=selector_metadata.selector_type,
            selector_side=selector_metadata.side,
            selector_start=(
                selector_metadata.start
                if isinstance(selector_metadata, BoundaryIntervalSelector)
                else None
            ),
            selector_end=(
                selector_metadata.end
                if isinstance(selector_metadata, BoundaryIntervalSelector)
                else None
            ),
        )
        self.boundary_regions.append(region)
        self._reindex()
        selectors = self.fluid_domain.selector_objects
        if selector not in selectors:
            selectors = (*selectors, selector)
        self.fluid_domain = FluidDomain(
            self.fluid_domain.root,
            (*self.fluid_domain.tag_objects, region),
            selectors,
        )
        self.mark_changed()
        return self.handle_for(region)

    def add_boundary_selector_split_regions(
        self,
        base_region: BoundaryRegion,
        selector: SDFNode,
    ) -> tuple[int, int]:
        if self.fluid_domain is None:
            raise ValueError("select a FluidDomain root first")
        if base_region not in self.boundary_regions:
            raise ValueError("base boundary region is not part of this document")
        if base_region.patch_id is None:
            raise ValueError("base boundary region must identify a boundary patch")
        live_sdf_nodes = tuple(
            node for root in self.objects for node in self._iter_nodes(root)
        )
        if all(node is not selector for node in live_sdf_nodes):
            raise ValueError("boundary selector object is not part of this document")
        selector_metadata = self._boundary_selector_metadata(
            base_region,
            selector,
        )
        if selector_metadata is None:
            raise ValueError(
                "boundary selectors must be compatible SDF cutter objects"
            )
        regions: list[BoundaryRegion] = []
        for side in ("inside", "outside"):
            side_selector = replace(selector_metadata, side=side)
            regions.append(
                BoundaryRegion(
                    name=f"{base_region.name} / {selector.name} {side}",
                    object_id=self._allocate_object_id(),
                    owner_object_id=base_region.owner_object_id,
                    outside_direction=base_region.outside_direction,
                    patch_id=base_region.patch_id,
                    patch_type=base_region.patch_type,
                    selector_id=side_selector.selector_id,
                    selector_type=side_selector.selector_type,
                    selector_side=side_selector.side,
                    selector_start=(
                        side_selector.start
                        if isinstance(side_selector, BoundaryIntervalSelector)
                        else None
                    ),
                    selector_end=(
                        side_selector.end
                        if isinstance(side_selector, BoundaryIntervalSelector)
                        else None
                    ),
                )
            )
        self.boundary_regions.extend(regions)
        self._reindex()
        selectors = self.fluid_domain.selector_objects
        if selector not in selectors:
            selectors = (*selectors, selector)
        self.fluid_domain = FluidDomain(
            self.fluid_domain.root,
            (*self.fluid_domain.tag_objects, *regions),
            selectors,
        )
        self.mark_changed()
        return tuple(self.handle_for(region) for region in regions)

    def _boundary_selector_metadata(
        self,
        base_region: BoundaryRegion,
        selector: SDFNode,
    ) -> BoundarySelector | None:
        assert self.fluid_domain is not None
        root = self.fluid_domain.root
        if root.dimension == 2:
            patch = next(
                (
                    patch
                    for patch in boundary_patches(root)
                    if isinstance(patch, BoundaryCurvePatch)
                    and patch.owner_object_id == base_region.owner_object_id
                    and patch.patch_id == base_region.patch_id
                ),
                None,
            )
            if patch is None:
                return None
            interval = boundary_interval_selector_from_node(patch, selector)
            if interval is not None:
                return interval
        return boundary_selector_from_node(
            selector,
            domain_dimension=root.dimension,
        )

    def delete(self, handle: int) -> None:
        self.delete_many((handle,))

    def delete_many(self, handles: list[int] | tuple[int, ...]) -> int:
        selected_nodes: list[SceneItem] = []
        for handle in handles:
            try:
                selected_nodes.append(self.node(handle))
            except KeyError:
                continue
        if not selected_nodes:
            return 0

        selected_ids = {id(node) for node in selected_nodes}
        boundary_region_ids = {
            id(node)
            for node in selected_nodes
            if isinstance(node, BoundaryRegion)
        }
        root_targets = [
            node
            for node in selected_nodes
            if isinstance(node, SDFNode)
            and not self._contains_selected_ancestor(node, selected_ids)
        ]
        target_ids = {id(node) for node in root_targets}
        deleted_count = 0

        if boundary_region_ids:
            original_count = len(self.boundary_regions)
            self.boundary_regions = [
                region
                for region in self.boundary_regions
                if id(region) not in boundary_region_ids
            ]
            deleted_count += original_count - len(self.boundary_regions)

        if target_ids:
            remaining_objects: list[SDFNode] = []
            for root in self.objects:
                replacement, removed = self._remove_targets_from(root, target_ids)
                deleted_count += removed
                if replacement is not None:
                    remaining_objects.append(replacement)
            self.objects = remaining_objects

        if deleted_count <= 0:
            return 0

        self._refresh_fluid_domain()
        self.mark_changed()
        return deleted_count

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

    def _remove_targets_from(
        self,
        current: SDFNode,
        target_ids: set[int],
    ) -> tuple[SDFNode | None, int]:
        if id(current) in target_ids:
            return None, 1
        if isinstance(current, UnaryTransform):
            assert current.child is not None
            replacement, removed = self._remove_targets_from(
                current.child,
                target_ids,
            )
            if removed <= 0:
                return current, 0
            if replacement is None:
                return None, removed
            current.child = replacement
            return current, removed
        if isinstance(current, BinaryCSG):
            assert current.left is not None and current.right is not None
            left, left_removed = self._remove_targets_from(current.left, target_ids)
            right, right_removed = self._remove_targets_from(
                current.right,
                target_ids,
            )
            removed = left_removed + right_removed
            if removed <= 0:
                return current, 0
            if left is None and right is None:
                return None, removed
            if left is None:
                return right, removed
            if right is None:
                return left, removed
            current.left = left
            current.right = right
            return current, removed
        if isinstance(current, (PlacedSDF1D, PlacedSDF2D)):
            if not current.sources:
                return current, 0
            replacements: list[SDFNode] = []
            removed = 0
            for source in current.sources:
                replacement, source_removed = self._remove_targets_from(
                    source,
                    target_ids,
                )
                removed += source_removed
                if replacement is not None:
                    replacements.append(replacement)
            if removed <= 0:
                return current, 0
            if not replacements:
                return None, removed
            if len(replacements) == 1:
                return replacements[0], removed
            current.sources = tuple(replacements)
            return current, removed
        if isinstance(current, (Extrude, Revolve)):
            assert current.section is not None
            replacement, removed = self._remove_targets_from(
                current.section,
                target_ids,
            )
            if removed <= 0:
                return current, 0
            if replacement is None:
                return None, removed
            if not isinstance(replacement, PlacedSDF2D):
                return None, removed
            current.section = replacement
            return current, removed
        return current, 0

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

    def _contains_selected_ancestor(
        self,
        target: SDFNode,
        selected_ids: set[int],
    ) -> bool:
        for root in self.objects:
            if self._contains_selected_ancestor_in_subtree(
                root,
                target,
                selected_ids,
                ancestor_selected=False,
            ):
                return True
        return False

    def _contains_selected_ancestor_in_subtree(
        self,
        current: SDFNode,
        target: SDFNode,
        selected_ids: set[int],
        ancestor_selected: bool,
    ) -> bool:
        current_selected = id(current) in selected_ids
        if current is target:
            return ancestor_selected
        return any(
            self._contains_selected_ancestor_in_subtree(
                child,
                target,
                selected_ids,
                ancestor_selected or current_selected,
            )
            for child in current.children()
        )

    def _assign_fresh_object_ids(self, root: SDFNode) -> None:
        seen: set[int] = set()
        for node in self._iter_nodes(root):
            if id(node) in seen:
                continue
            seen.add(id(node))
            node.object_id = self._allocate_object_id()

    def _translate_copy_in_place(
        self,
        node: SDFNode,
        delta: tuple[float, float, float],
        seen: set[int] | None = None,
    ) -> bool:
        if seen is None:
            seen = set()
        if id(node) in seen:
            return True
        seen.add(id(node))
        if isinstance(node, (Rotate, Scale)):
            return False
        if isinstance(node, Translate):
            node.offset = tuple(
                node.offset[index] + delta[index] for index in range(3)
            )
            return True
        if isinstance(node, (PlacedSDF1D, PlacedPolyline2D, PlacedSDF2D)):
            node.origin = tuple(
                node.origin[index] + delta[index] for index in range(3)
            )
            for child in node.children():
                if not self._translate_copy_in_place(child, delta, seen):
                    return False
            node.__post_init__()
            return True
        if isinstance(node, (Sphere, Box, BoxFrame, CappedCone, Cone, Cylinder, Pyramid, Torus)):
            node.center = tuple(
                node.center[index] + delta[index] for index in range(3)
            )
            return True
        if isinstance(node, (PolylineTube, BezierTube)):
            node.points = tuple(
                tuple(point[index] + delta[index] for index in range(3))
                for point in node.points
            )
            return True
        if isinstance(node, Revolve):
            if node.axis_origin is not None:
                node.axis_origin = tuple(
                    node.axis_origin[index] + delta[index] for index in range(3)
                )
            for child in node.children():
                if not self._translate_copy_in_place(child, delta, seen):
                    return False
            node.__post_init__()
            return True
        for child in node.children():
            if not self._translate_copy_in_place(child, delta, seen):
                return False
        return True

    def _rotate_node_in_place(
        self,
        node: SDFNode,
        axis: str,
        angle_degrees: float,
        pivot: tuple[float, float, float],
        seen: set[int] | None = None,
    ) -> bool:
        if seen is None:
            seen = set()
        if id(node) in seen:
            return True
        seen.add(id(node))
        if isinstance(node, Translate):
            node.offset = self._rotate_vector(node.offset, axis, angle_degrees)
            for child in node.children():
                if not self._rotate_node_in_place(
                    child,
                    axis,
                    angle_degrees,
                    pivot,
                    seen,
                ):
                    return False
            return True
        if isinstance(node, Scale):
            for child in node.children():
                if not self._rotate_node_in_place(
                    child,
                    axis,
                    angle_degrees,
                    pivot,
                    seen,
                ):
                    return False
            return True
        if isinstance(node, Rotate):
            for child in node.children():
                if not self._rotate_node_in_place(
                    child,
                    axis,
                    angle_degrees,
                    pivot,
                    seen,
                ):
                    return False
            return True
        if isinstance(node, (PlacedSDF1D, PlacedPolyline2D, PlacedSDF2D)):
            node.origin = self._rotate_point(node.origin, axis, angle_degrees, pivot)
            node.axis_u = self._rotate_vector(node.axis_u, axis, angle_degrees)
            if isinstance(node, (PlacedPolyline2D, PlacedSDF2D)):
                node.axis_v = self._rotate_vector(node.axis_v, axis, angle_degrees)
            for child in node.children():
                if not self._rotate_node_in_place(
                    child,
                    axis,
                    angle_degrees,
                    pivot,
                    seen,
                ):
                    return False
            node.__post_init__()
            return True
        if isinstance(node, Sphere):
            node.center = self._rotate_point(node.center, axis, angle_degrees, pivot)
            return True
        if isinstance(node, (Box, BoxFrame, CappedCone, Cone, Cylinder, Pyramid, Torus)):
            node.center = self._rotate_point(node.center, axis, angle_degrees, pivot)
            node.axis_u = self._rotate_vector(node.axis_u, axis, angle_degrees)
            node.axis_v = self._rotate_vector(node.axis_v, axis, angle_degrees)
            node.axis_w = self._rotate_vector(node.axis_w, axis, angle_degrees)
            node.__post_init__()
            return True
        if isinstance(node, (PolylineTube, BezierTube)):
            node.points = tuple(
                self._rotate_point(point, axis, angle_degrees, pivot)
                for point in node.points
            )
            return True
        if isinstance(node, Revolve):
            if node.axis_origin is not None:
                node.axis_origin = self._rotate_point(
                    node.axis_origin,
                    axis,
                    angle_degrees,
                    pivot,
                )
            if node.axis_direction is not None:
                node.axis_direction = self._rotate_vector(
                    node.axis_direction,
                    axis,
                    angle_degrees,
                )
            if node.radial_direction is not None:
                node.radial_direction = self._rotate_vector(
                    node.radial_direction,
                    axis,
                    angle_degrees,
                )
            for child in node.children():
                if not self._rotate_node_in_place(
                    child,
                    axis,
                    angle_degrees,
                    pivot,
                    seen,
                ):
                    return False
            node.__post_init__()
            return True
        for child in node.children():
            if not self._rotate_node_in_place(child, axis, angle_degrees, pivot, seen):
                return False
        return True

    def _build_visual_tree(self) -> SDFTree:
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

    def visual_tree(self) -> SDFTree:
        self.refresh_derived_geometry()
        return self._build_visual_tree()

    def tree(self) -> SDFTree:
        return self.visual_tree()

    def _snapshot_bundle(
        self,
    ) -> tuple[
        list[SDFNode],
        FluidDomain | None,
        list[BoundaryRegion],
        int,
        int,
    ]:
        self.refresh_derived_geometry()
        return deepcopy(
            (
                self.objects,
                self.fluid_domain,
                self.boundary_regions,
                self.version,
                self._next_object_id,
            )
        )

    def snapshot(self) -> SceneDocument:
        (
            objects,
            fluid_domain,
            boundary_regions,
            version,
            next_object_id,
        ) = self._snapshot_bundle()
        snapshot = SceneDocument(
            objects=objects,
            fluid_domain=fluid_domain,
            boundary_regions=boundary_regions,
            version=version,
        )
        snapshot._next_object_id = next_object_id
        return snapshot

    def visual_snapshot(self) -> tuple[int, SDFTree | None]:
        self.refresh_derived_geometry()
        tree = deepcopy(self._build_visual_tree()) if self.objects else None
        return self.version, tree

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
            and self._boundary_region_selector_is_live(region, live_sdf_ids)
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
        selectors = tuple(
            selector
            for selector in self.fluid_domain.selector_objects
            if id(selector) in live
        )
        self.fluid_domain = FluidDomain(self.fluid_domain.root, tags, selectors)

    @staticmethod
    def _boundary_region_selector_is_live(
        region: BoundaryRegion,
        live_sdf_ids: set[int],
    ) -> bool:
        if region.selector_id is None:
            return True
        prefix = "selector:"
        if not region.selector_id.startswith(prefix):
            return True
        try:
            selector_object_id = int(region.selector_id[len(prefix):])
        except ValueError:
            return True
        return selector_object_id in live_sdf_ids
