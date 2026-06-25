from __future__ import annotations

"""Backend-agnostic render intermediate representation for SDF scenes."""

from dataclasses import dataclass, replace
from math import cos, radians, sin

import numpy as np

from .sdf import (
    QuadraticBezierTube,
    QuadraticBezierCurveProfile,
    QuadraticBezierSurfaceProfile,
    CircleProfile,
    Box,
    BoxFrame,
    EllipseProfile,
    CappedCone,
    Cone,
    Difference,
    DistanceOffsetProfile,
    Extrude,
    Intersection,
    Cylinder,
    OffsetProfile,
    OffsetProfile1D,
    PolygonProfile,
    PolylineTube,
    PlacedPolyline1D,
    PlacedSDF1D,
    PlacedSDF2D,
    PolylineProfile,
    RectangleProfile,
    RoundedRectangleProfile,
    SquareProfile,
    Pyramid,
    RegularPolygonProfile,
    Revolve,
    Rotate,
    Scale,
    SegmentProfile,
    Sphere,
    Torus,
    Translate,
    Union,
    BinaryProfile,
    BinaryProfile1D,
    SDFTree,
)


@dataclass(frozen=True)
class RenderIRNode:
    kind: str
    object_id: int
    dimension: int
    children: tuple[int, ...]
    params: tuple[float, ...] = ()
    component_indices: tuple[int, ...] = ()
    # Bit-flag payload packed into GpuNode.flags. For Layer 2 region_selector
    # nodes this carries the region id to assign on a match (design §5.2);
    # for geometry leaves it stays 0 (reserved for Layer 1 intrinsic bits).
    flags: int = 0
    # World-space (cx, cy, cz, radius) enclosing sphere for geometry leaves, from
    # the SDF node's authoritative bounding_box() (None for operators/unsupported).
    # Used by the GPU spatial cull (core/gpu_cull) to bin leaves into grid cells.
    bound: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class RenderIRObjectRef:
    object_id: int
    dimension: int
    node_index: int


@dataclass(frozen=True)
class RenderIR:
    nodes: tuple[RenderIRNode, ...]
    root_indices: tuple[int, ...]
    component_indices: tuple[int, ...]
    aliases: tuple[RenderIRObjectRef, ...] = ()
    material_refs: tuple[RenderIRObjectRef, ...] = ()
    component_refs: tuple[RenderIRObjectRef, ...] = ()
    unsupported_kinds: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.unsupported_kinds

    @property
    def parameter_values(self) -> tuple[float, ...]:
        return tuple(value for node in self.nodes for value in node.params)

    @property
    def topology_signature(self) -> tuple[object, ...]:
        return (
            tuple(_node_topology_signature(node) for node in self.nodes),
            self.root_indices,
            self.unsupported_kinds,
        )


@dataclass(frozen=True)
class _AffineTransform:
    scale: float
    rotation: tuple[float, ...]
    translation: tuple[float, float, float]

    @staticmethod
    def identity() -> _AffineTransform:
        return _AffineTransform(
            scale=1.0,
            rotation=(
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
                1.0,
            ),
            translation=(0.0, 0.0, 0.0),
        )

    @property
    def rotation_matrix(self) -> np.ndarray:
        return np.asarray(self.rotation, dtype=np.float64).reshape(3, 3)

    @property
    def cache_key(self) -> tuple[object, ...]:
        return (self.scale, *self.rotation, *self.translation)

    def apply_point(
        self,
        point: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        transformed = (
            self.scale * (self.rotation_matrix @ np.asarray(point, dtype=np.float64))
            + np.asarray(self.translation, dtype=np.float64)
        )
        return tuple(float(value) for value in transformed)

    def apply_direction(
        self,
        vector: tuple[float, float, float],
    ) -> tuple[float, float, float]:
        transformed = self.rotation_matrix @ np.asarray(vector, dtype=np.float64)
        length = float(np.linalg.norm(transformed))
        if length <= 1.0e-12:
            raise ValueError("transform produced a degenerate direction")
        return tuple(float(value) for value in transformed / length)

    def apply_length(self, value: float) -> float:
        return float(self.scale * value)

    def apply_profile_point(
        self,
        point: tuple[float, float],
    ) -> tuple[float, float]:
        return tuple(float(self.scale * value) for value in point)

    def translated(self, offset: tuple[float, float, float]) -> _AffineTransform:
        translated = np.asarray(self.translation, dtype=np.float64) + self.scale * (
            self.rotation_matrix @ np.asarray(offset, dtype=np.float64)
        )
        return _AffineTransform(
            scale=self.scale,
            rotation=self.rotation,
            translation=tuple(float(value) for value in translated),
        )

    def scaled(self, factor: float) -> _AffineTransform:
        return _AffineTransform(
            scale=self.scale * float(factor),
            rotation=self.rotation,
            translation=self.translation,
        )

    def rotated(self, axis: str, angle_degrees: float) -> _AffineTransform:
        angle_radians = radians(angle_degrees)
        c = cos(angle_radians)
        s = sin(angle_radians)
        if axis == "x":
            local_rotation = np.asarray(
                ((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)),
                dtype=np.float64,
            )
        elif axis == "y":
            local_rotation = np.asarray(
                ((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)),
                dtype=np.float64,
            )
        else:
            local_rotation = np.asarray(
                ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)),
                dtype=np.float64,
            )
        rotation = self.rotation_matrix @ local_rotation
        return _AffineTransform(
            scale=self.scale,
            rotation=tuple(float(value) for value in rotation.reshape(-1)),
            translation=self.translation,
        )


def _node_topology_signature(node: RenderIRNode) -> tuple[object, ...]:
    semantic_flags: tuple[object, ...] = ()
    if node.kind in {"polyline_tube", "quadratic_bezier_tube"} and node.params:
        semantic_flags = ("flat_caps" if node.params[-1] > 0.5 else "round_caps",)
    return (
        node.kind,
        int(node.dimension),
        node.children,
        len(node.params),
        node.component_indices,
        semantic_flags,
    )


def _tuple_floats(*values: object) -> tuple[float, ...]:
    return tuple(float(value) for value in values)


def _emit_points(points: tuple[tuple[float, float, float], ...]) -> tuple[float, ...]:
    flat: list[float] = []
    for x, y, z in points:
        flat.extend((float(x), float(y), float(z)))
    return tuple(flat)


def _emit_points_2d(points: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
    flat: list[float] = []
    for u, v in points:
        flat.extend((float(u), float(v)))
    return tuple(flat)


def _build_profile_1d_ir_node(
    profile,
    nodes: dict[tuple[object, ...], int],
    ir_nodes: list[RenderIRNode],
    unsupported: list[str],
    scale: float,
    offset: float = 0.0,
) -> int:
    profile_key = ("profile_1d", id(profile), float(scale), float(offset))
    cached = nodes.get(profile_key)
    if cached is not None:
        return cached
    if isinstance(profile, SegmentProfile):
        payload = RenderIRNode(
            kind="profile_segment_1d",
            object_id=0,
            dimension=1,
            children=(),
            params=_tuple_floats(
                float(scale) * float(profile.center) + float(offset),
                float(scale) * float(profile.half_length),
            ),
        )
    elif isinstance(profile, OffsetProfile1D):
        return _build_profile_1d_ir_node(
            profile.child,
            nodes,
            ir_nodes,
            unsupported,
            scale,
            offset + float(scale) * float(profile.offset),
        )
    elif isinstance(profile, BinaryProfile1D):
        payload = RenderIRNode(
            kind={
                "union": "profile_union_1d",
                "intersection": "profile_intersection_1d",
                "difference": "profile_difference_1d",
            }[profile.operation],
            object_id=0,
            dimension=1,
            children=(
                _build_profile_1d_ir_node(
                    profile.left,
                    nodes,
                    ir_nodes,
                    unsupported,
                    scale,
                    offset,
                ),
                _build_profile_1d_ir_node(
                    profile.right,
                    nodes,
                    ir_nodes,
                    unsupported,
                    scale,
                    offset,
                ),
            ),
            params=(),
        )
    else:
        unsupported.append(type(profile).__name__)
        payload = RenderIRNode(
            kind="unsupported",
            object_id=0,
            dimension=1,
            children=(),
            params=(),
        )
    index = len(ir_nodes)
    ir_nodes.append(payload)
    nodes[profile_key] = index
    return index


def _build_profile_ir_node(
    profile,
    nodes: dict[tuple[object, ...], int],
    ir_nodes: list[RenderIRNode],
    unsupported: list[str],
    scale: float,
    offset: tuple[float, float] = (0.0, 0.0),
) -> int:
    profile_key = ("profile", id(profile), float(scale), offset)
    cached = nodes.get(profile_key)
    if cached is not None:
        return cached
    if isinstance(profile, CircleProfile):
        payload = RenderIRNode(
            kind="profile_circle_2d",
            object_id=0,
            dimension=2,
            children=(),
            params=_tuple_floats(
                *(float(scale) * value + offset[index] for index, value in enumerate(profile.center)),
                float(scale) * profile.radius,
            ),
        )
    elif isinstance(profile, RectangleProfile) and not isinstance(
        profile,
        (SquareProfile, RoundedRectangleProfile),
    ):
        payload = RenderIRNode(
            kind="profile_rectangle_2d",
            object_id=0,
            dimension=2,
            children=(),
            params=_tuple_floats(
                *(float(scale) * value + offset[index] for index, value in enumerate(profile.center)),
                *(float(scale) * value for value in profile.half_size),
            ),
        )
    elif isinstance(profile, SquareProfile):
        payload = RenderIRNode(
            kind="profile_square_2d",
            object_id=0,
            dimension=2,
            children=(),
            params=_tuple_floats(
                *(float(scale) * value + offset[index] for index, value in enumerate(profile.center)),
                float(scale) * profile.half_size,
            ),
        )
    elif isinstance(profile, RoundedRectangleProfile):
        payload = RenderIRNode(
            kind="profile_rounded_rectangle_2d",
            object_id=0,
            dimension=2,
            children=(),
            params=_tuple_floats(
                *(float(scale) * value + offset[index] for index, value in enumerate(profile.center)),
                *(float(scale) * value for value in profile.half_size),
                float(scale) * profile.corner_radius,
            ),
        )
    elif isinstance(profile, EllipseProfile):
        payload = RenderIRNode(
            kind="profile_ellipse_2d",
            object_id=0,
            dimension=2,
            children=(),
            params=_tuple_floats(
                *(float(scale) * value + offset[index] for index, value in enumerate(profile.center)),
                *(float(scale) * value for value in profile.semi_axes),
            ),
        )
    elif isinstance(profile, (PolygonProfile, RegularPolygonProfile)):
        points = (
            tuple(tuple(float(value) for value in point) for point in profile._vertices())
            if isinstance(profile, RegularPolygonProfile)
            else profile.points
        )
        payload = RenderIRNode(
            kind="profile_polygon_2d",
            object_id=0,
            dimension=2,
            children=(),
            params=tuple(
                value
                for point in points
                for value in (
                    float(scale) * float(point[0]) + offset[0],
                    float(scale) * float(point[1]) + offset[1],
                )
            ),
        )
    elif isinstance(profile, PolylineProfile):
        payload = RenderIRNode(
            kind="profile_polyline_2d",
            object_id=0,
            dimension=2,
            children=(),
            params=tuple(
                value
                for point in profile.points
                for value in (
                    float(scale) * float(point[0]) + offset[0],
                    float(scale) * float(point[1]) + offset[1],
                )
            ),
        )
    elif isinstance(profile, QuadraticBezierCurveProfile):
        payload = RenderIRNode(
            kind="profile_quadratic_bezier_curve_1d",
            object_id=0,
            dimension=2,
            children=(),
            params=tuple(
                value
                for point in profile.points
                for value in (
                    float(scale) * float(point[0]) + offset[0],
                    float(scale) * float(point[1]) + offset[1],
                )
            ),
        )
    elif isinstance(profile, QuadraticBezierSurfaceProfile):
        payload = RenderIRNode(
            kind="profile_quadratic_bezier_surface_2d",
            object_id=0,
            dimension=2,
            children=(),
            params=tuple(
                value
                for point in profile.points
                for value in (
                    float(scale) * float(point[0]) + offset[0],
                    float(scale) * float(point[1]) + offset[1],
                )
            ),
        )
    elif isinstance(profile, OffsetProfile):
        return _build_profile_ir_node(
            profile.child,
            nodes,
            ir_nodes,
            unsupported,
            scale,
            (
                offset[0] + float(scale) * float(profile.offset[0]),
                offset[1] + float(scale) * float(profile.offset[1]),
            ),
        )
    elif isinstance(profile, DistanceOffsetProfile):
        payload = RenderIRNode(
            kind="profile_distance_offset_2d",
            object_id=0,
            dimension=2,
            children=(
                _build_profile_ir_node(
                    profile.child,
                    nodes,
                    ir_nodes,
                    unsupported,
                    scale,
                    offset,
                ),
            ),
            params=(float(scale) * float(profile.offset),),
        )
    elif isinstance(profile, BinaryProfile):
        payload = RenderIRNode(
            kind={
                "union": "profile_union_2d",
                "intersection": "profile_intersection_2d",
                "difference": "profile_difference_2d",
            }[profile.operation],
            object_id=0,
            dimension=2,
            children=(
                _build_profile_ir_node(
                    profile.left,
                    nodes,
                    ir_nodes,
                    unsupported,
                    scale,
                    offset,
                ),
                _build_profile_ir_node(
                    profile.right,
                    nodes,
                    ir_nodes,
                    unsupported,
                    scale,
                    offset,
                ),
            ),
            params=(),
        )
    else:
        unsupported.append(type(profile).__name__)
        payload = RenderIRNode(
            kind="unsupported",
            object_id=0,
            dimension=2,
            children=(),
            params=(),
        )
    index = len(ir_nodes)
    ir_nodes.append(payload)
    nodes[profile_key] = index
    return index


def _world_bound_sphere(node, transform: _AffineTransform):
    """World-space ``(cx, cy, cz, radius)`` enclosing ``node`` under ``transform``,
    derived from the node's authoritative ``bounding_box()`` (transform applied to
    the box corners). Returns None if the node has no usable box. This is what lets
    the GPU cull bin *every* geometry kind, not just the closed-form primitives."""
    try:
        bb = node.bounding_box()
        corners = np.array(
            [(x, y, z)
             for x in (bb.x_min, bb.x_max)
             for y in (bb.y_min, bb.y_max)
             for z in (bb.z_min, bb.z_max)],
            dtype=np.float64,
        )
        world = np.array([transform.apply_point(tuple(c)) for c in corners])
    except Exception:
        return None
    lo = world.min(axis=0)
    hi = world.max(axis=0)
    center = (lo + hi) * 0.5
    radius = float(np.linalg.norm(hi - center)) + 1.0e-4
    return (float(center[0]), float(center[1]), float(center[2]), radius)


def _build_render_ir_node(
    node,
    nodes: dict[tuple[object, ...], int],
    ir_nodes: list[RenderIRNode],
    unsupported: list[str],
    transform: _AffineTransform,
    aliases: list[RenderIRObjectRef],
) -> int:
    node_key = ("node", id(node), transform.cache_key)
    cached = nodes.get(node_key)
    if cached is not None:
        return cached
    if isinstance(node, Sphere):
        payload = RenderIRNode(
            kind="sphere",
            object_id=node.object_id,
            dimension=3,
            children=(),
            params=_tuple_floats(
                *transform.apply_point(node.center),
                transform.apply_length(node.radius),
            ),
        )
    elif isinstance(node, Box):
        payload = RenderIRNode(
            kind="box",
            object_id=node.object_id,
            dimension=3,
            children=(),
            params=_tuple_floats(
                *transform.apply_point(node.center),
                *transform.apply_direction(node.axis_u),
                *transform.apply_direction(node.axis_v),
                *transform.apply_direction(node.axis_w),
                *(transform.apply_length(value) for value in node.half_size),
            ),
        )
    elif isinstance(node, CappedCone):
        payload = RenderIRNode(
            kind="capped_cone",
            object_id=node.object_id,
            dimension=3,
            children=(),
            params=_tuple_floats(
                *transform.apply_point(node.center),
                *transform.apply_direction(node.axis_u),
                *transform.apply_direction(node.axis_v),
                *transform.apply_direction(node.axis_w),
                transform.apply_length(node.radius_a),
                transform.apply_length(node.radius_b),
                transform.apply_length(node.half_height),
            ),
        )
    elif isinstance(node, Cone):
        payload = RenderIRNode(
            kind="cone",
            object_id=node.object_id,
            dimension=3,
            children=(),
            params=_tuple_floats(
                *transform.apply_point(node.center),
                *transform.apply_direction(node.axis_u),
                *transform.apply_direction(node.axis_v),
                *transform.apply_direction(node.axis_w),
                transform.apply_length(node.radius),
                transform.apply_length(node.half_height),
            ),
        )
    elif isinstance(node, Cylinder):
        payload = RenderIRNode(
            kind="cylinder",
            object_id=node.object_id,
            dimension=3,
            children=(),
            params=_tuple_floats(
                *transform.apply_point(node.center),
                *transform.apply_direction(node.axis_u),
                *transform.apply_direction(node.axis_v),
                *transform.apply_direction(node.axis_w),
                transform.apply_length(node.radius),
                transform.apply_length(node.half_height),
            ),
        )
    elif isinstance(node, BoxFrame):
        payload = RenderIRNode(
            kind="box_frame",
            object_id=node.object_id,
            dimension=3,
            children=(),
            params=_tuple_floats(
                *transform.apply_point(node.center),
                *transform.apply_direction(node.axis_u),
                *transform.apply_direction(node.axis_v),
                *transform.apply_direction(node.axis_w),
                *(transform.apply_length(value) for value in node.half_size),
                transform.apply_length(node.thickness),
            ),
        )
    elif isinstance(node, Pyramid):
        payload = RenderIRNode(
            kind="pyramid",
            object_id=node.object_id,
            dimension=3,
            children=(),
            params=_tuple_floats(
                *transform.apply_point(node.center),
                *transform.apply_direction(node.axis_u),
                *transform.apply_direction(node.axis_v),
                *transform.apply_direction(node.axis_w),
                transform.apply_length(node.base_half_size),
                transform.apply_length(node.half_height),
            ),
        )
    elif isinstance(node, Torus):
        payload = RenderIRNode(
            kind="torus",
            object_id=node.object_id,
            dimension=3,
            children=(),
            params=_tuple_floats(
                *transform.apply_point(node.center),
                *transform.apply_direction(node.axis_u),
                *transform.apply_direction(node.axis_v),
                *transform.apply_direction(node.axis_w),
                transform.apply_length(node.major_radius),
                transform.apply_length(node.minor_radius),
            ),
        )
    elif isinstance(node, Union):
        payload = RenderIRNode(
            kind="union",
            object_id=max(node.object_id, 0),
            dimension=node.dimension,
            children=tuple(
                _build_render_ir_node(
                    child,
                    nodes,
                    ir_nodes,
                    unsupported,
                    transform,
                    aliases,
                ) for child in node.children()
            ),
            params=(),
        )
    elif isinstance(node, Intersection):
        payload = RenderIRNode(
            kind="intersection",
            object_id=max(node.object_id, 0),
            dimension=node.dimension,
            children=tuple(
                _build_render_ir_node(
                    child,
                    nodes,
                    ir_nodes,
                    unsupported,
                    transform,
                    aliases,
                ) for child in node.children()
            ),
        )
    elif isinstance(node, Difference):
        payload = RenderIRNode(
            kind="difference",
            object_id=max(node.object_id, 0),
            dimension=node.dimension,
            children=tuple(
                _build_render_ir_node(
                    child,
                    nodes,
                    ir_nodes,
                    unsupported,
                    transform,
                    aliases,
                ) for child in node.children()
            ),
        )
    elif isinstance(node, PlacedSDF2D):
        payload = _build_placed_sdf_2d_ir_node(
            node,
            nodes,
            ir_nodes,
            unsupported,
            transform,
        )
    elif isinstance(node, PlacedSDF1D):
        assert node.profile is not None
        payload = RenderIRNode(
            kind="placed_profile_1d",
            object_id=node.object_id,
            dimension=1,
            children=(
                _build_profile_1d_ir_node(
                    node.profile,
                    nodes,
                    ir_nodes,
                    unsupported,
                    transform.scale,
                ),
            ),
            params=_tuple_floats(
                *transform.apply_point(node.origin),
                *transform.apply_direction(node.axis_u),
            ),
        )
    elif isinstance(node, PlacedPolyline1D):
        assert node.profile is not None
        kind = "placed_polyline_1d"
        if isinstance(node.profile, QuadraticBezierCurveProfile):
            kind = (
                "placed_quadratic_bezier_curve_1d"
                if len(node.profile.points) == 3
                else "placed_quadratic_bezier_polycurve_1d"
            )
        payload = RenderIRNode(
            kind=kind,
            object_id=node.object_id,
            dimension=1,
            children=(),
            params=_tuple_floats(
                *transform.apply_point(node.origin),
                *transform.apply_direction(node.axis_u),
                *transform.apply_direction(node.axis_v),
                *transform.apply_direction(node.normal),
                *(
                    transform.scale * value
                    for value in _emit_points_2d(node.profile.points)
                ),
            ),
        )
    elif isinstance(node, Extrude):
        assert node.section is not None and node.section.profile is not None
        payload = RenderIRNode(
            kind="extrude_profile_2d",
            object_id=node.object_id,
            dimension=3,
            children=(
                _build_profile_ir_node(
                    node.section.profile,
                    nodes,
                    ir_nodes,
                    unsupported,
                    transform.scale,
                ),
            ),
            params=_tuple_floats(
                *transform.apply_point(node.section.origin),
                *transform.apply_direction(node.section.axis_u),
                *transform.apply_direction(node.section.axis_v),
                *transform.apply_direction(node.section.normal),
                transform.apply_length(node.height),
                transform.apply_length(node.center_offset),
            ),
        )
    elif isinstance(node, Revolve):
        assert node.section is not None and node.section.profile is not None
        axis_origin, axis_direction, radial_direction, tangent_direction = (
            node._axis_frame()
        )
        payload = RenderIRNode(
            kind="revolve_profile_2d",
            object_id=node.object_id,
            dimension=3,
            children=(
                _build_profile_ir_node(
                    node.section.profile,
                    nodes,
                    ir_nodes,
                    unsupported,
                    transform.scale,
                ),
            ),
            params=_tuple_floats(
                *transform.apply_point(node.section.origin),
                *transform.apply_direction(node.section.axis_u),
                *transform.apply_direction(node.section.axis_v),
                *transform.apply_direction(node.section.normal),
                *transform.apply_point(tuple(float(value) for value in axis_origin)),
                *transform.apply_direction(
                    tuple(float(value) for value in axis_direction)
                ),
                *transform.apply_direction(
                    tuple(float(value) for value in radial_direction)
                ),
                *transform.apply_direction(
                    tuple(float(value) for value in tangent_direction)
                ),
                radians(node.angle_degrees),
            ),
        )
    elif isinstance(node, Translate):
        assert node.child is not None
        child_index = _build_render_ir_node(
            node.child,
            nodes,
            ir_nodes,
            unsupported,
            transform.translated(node.offset),
            aliases,
        )
        aliases.append(
            RenderIRObjectRef(
                object_id=max(node.object_id, 0),
                dimension=max(node.dimension, 0),
                node_index=child_index,
            )
        )
        return child_index
    elif isinstance(node, Scale):
        assert node.child is not None
        child_index = _build_render_ir_node(
            node.child,
            nodes,
            ir_nodes,
            unsupported,
            transform.scaled(node.factor),
            aliases,
        )
        aliases.append(
            RenderIRObjectRef(
                object_id=max(node.object_id, 0),
                dimension=max(node.dimension, 0),
                node_index=child_index,
            )
        )
        return child_index
    elif isinstance(node, Rotate):
        assert node.child is not None
        child_index = _build_render_ir_node(
            node.child,
            nodes,
            ir_nodes,
            unsupported,
            transform.rotated(node.axis, node.angle_degrees),
            aliases,
        )
        aliases.append(
            RenderIRObjectRef(
                object_id=max(node.object_id, 0),
                dimension=max(node.dimension, 0),
                node_index=child_index,
            )
        )
        return child_index
    elif isinstance(node, PolylineTube):
        payload = RenderIRNode(
            kind="polyline_tube",
            object_id=node.object_id,
            dimension=3,
            children=(),
            params=(
                *_emit_points(
                    tuple(transform.apply_point(point) for point in node.points)
                ),
                transform.apply_length(node.radius),
                transform.apply_length(node.inner_radius),
                1.0 if node.caps == "flat" else 0.0,
            ),
        )
    elif isinstance(node, QuadraticBezierTube):
        payload = RenderIRNode(
            kind="quadratic_bezier_tube",
            object_id=node.object_id,
            dimension=3,
            children=(),
            params=(
                *_emit_points(
                    tuple(transform.apply_point(point) for point in node.points)
                ),
                transform.apply_length(node.radius),
                transform.apply_length(node.inner_radius),
                1.0 if node.caps == "flat" else 0.0,
            ),
        )
    elif isinstance(
        node,
        (
            PlacedSDF2D,
            PolygonProfile,
            QuadraticBezierCurveProfile,
            PolylineProfile,
        ),
    ):
        unsupported.append(node.__class__.__name__)
        payload = RenderIRNode(
            kind="unsupported",
            object_id=max(node.object_id, 0),
            dimension=max(node.dimension, 0),
            children=(),
            params=(),
        )
    else:
        unsupported.append(node.__class__.__name__)
        payload = RenderIRNode(
            kind="unsupported",
            object_id=max(node.object_id, 0),
            dimension=max(getattr(node, "dimension", 0), 0),
            children=(),
            params=(),
        )
    index = len(ir_nodes)
    if payload.kind != "unsupported" and not isinstance(
            node, (Union, Intersection, Difference)):
        payload = replace(payload, bound=_world_bound_sphere(node, transform))
    ir_nodes.append(payload)
    nodes[node_key] = index
    return index


def _build_placed_sdf_2d_ir_node(
    node: PlacedSDF2D,
    nodes: dict[tuple[object, ...], int],
    ir_nodes: list[RenderIRNode],
    unsupported: list[str],
    transform: _AffineTransform,
) -> RenderIRNode:
    profile = node.profile
    if isinstance(profile, CircleProfile):
        return RenderIRNode(
            kind="placed_circle_2d",
            object_id=node.object_id,
            dimension=2,
            children=(),
            params=_tuple_floats(
                *transform.apply_point(node.origin),
                *transform.apply_direction(node.axis_u),
                *transform.apply_direction(node.axis_v),
                *transform.apply_direction(node.normal),
                *(transform.scale * value for value in profile.center),
                transform.apply_length(profile.radius),
            ),
        )
    if isinstance(profile, RectangleProfile) and not isinstance(
        profile,
        (SquareProfile, RoundedRectangleProfile),
    ):
        return RenderIRNode(
            kind="placed_rectangle_2d",
            object_id=node.object_id,
            dimension=2,
            children=(),
            params=_tuple_floats(
                *transform.apply_point(node.origin),
                *transform.apply_direction(node.axis_u),
                *transform.apply_direction(node.axis_v),
                *transform.apply_direction(node.normal),
                *(transform.scale * value for value in profile.center),
                *(transform.scale * value for value in profile.half_size),
            ),
        )
    if isinstance(profile, SquareProfile):
        return RenderIRNode(
            kind="placed_square_2d",
            object_id=node.object_id,
            dimension=2,
            children=(),
            params=_tuple_floats(
                *transform.apply_point(node.origin),
                *transform.apply_direction(node.axis_u),
                *transform.apply_direction(node.axis_v),
                *transform.apply_direction(node.normal),
                *(transform.scale * value for value in profile.center),
                transform.apply_length(profile.half_size),
            ),
        )
    if isinstance(profile, RoundedRectangleProfile):
        return RenderIRNode(
            kind="placed_rounded_rectangle_2d",
            object_id=node.object_id,
            dimension=2,
            children=(),
            params=_tuple_floats(
                *transform.apply_point(node.origin),
                *transform.apply_direction(node.axis_u),
                *transform.apply_direction(node.axis_v),
                *transform.apply_direction(node.normal),
                *(transform.scale * value for value in profile.center),
                *(transform.scale * value for value in profile.half_size),
                transform.apply_length(profile.corner_radius),
            ),
        )
    if isinstance(profile, EllipseProfile):
        return RenderIRNode(
            kind="placed_ellipse_2d",
            object_id=node.object_id,
            dimension=2,
            children=(),
            params=_tuple_floats(
                *transform.apply_point(node.origin),
                *transform.apply_direction(node.axis_u),
                *transform.apply_direction(node.axis_v),
                *transform.apply_direction(node.normal),
                *(transform.scale * value for value in profile.center),
                *(transform.scale * value for value in profile.semi_axes),
            ),
        )
    if isinstance(profile, (PolygonProfile, RegularPolygonProfile)):
        points = (
            tuple(tuple(float(value) for value in point) for point in profile._vertices())
            if isinstance(profile, RegularPolygonProfile)
            else profile.points
        )
        return RenderIRNode(
            kind="placed_polygon_2d",
            object_id=node.object_id,
            dimension=2,
            children=(),
            params=tuple(
                [
                    *transform.apply_point(node.origin),
                    *transform.apply_direction(node.axis_u),
                    *transform.apply_direction(node.axis_v),
                    *transform.apply_direction(node.normal),
                    *(
                        float(value)
                        for point in points
                        for value in (
                            transform.scale * float(point[0]),
                            transform.scale * float(point[1]),
                        )
                    ),
                ]
            ),
        )
    if isinstance(profile, QuadraticBezierSurfaceProfile):
        return RenderIRNode(
            kind="placed_quadratic_bezier_surface_2d",
            object_id=node.object_id,
            dimension=2,
            children=(),
            params=tuple(
                [
                    *transform.apply_point(node.origin),
                    *transform.apply_direction(node.axis_u),
                    *transform.apply_direction(node.axis_v),
                    *transform.apply_direction(node.normal),
                    *(
                        float(value)
                        for point in profile.points
                        for value in (
                            transform.scale * float(point[0]),
                            transform.scale * float(point[1]),
                        )
                    ),
                ]
            ),
        )
    if isinstance(
        profile,
        (
            OffsetProfile,
            BinaryProfile,
        ),
    ):
        return RenderIRNode(
            kind="placed_profile_2d",
            object_id=node.object_id,
            dimension=2,
            children=(
                _build_profile_ir_node(
                    profile,
                    nodes,
                    ir_nodes,
                    unsupported,
                    transform.scale,
                ),
            ),
            params=_tuple_floats(
                *transform.apply_point(node.origin),
                *transform.apply_direction(node.axis_u),
                *transform.apply_direction(node.axis_v),
                *transform.apply_direction(node.normal),
            ),
        )
    unsupported.append(type(profile).__name__)
    return RenderIRNode(
        kind="unsupported",
        object_id=max(node.object_id, 0),
        dimension=2,
        children=(),
        params=(),
    )


def _reachable_node_indices(nodes: list[RenderIRNode], start: int) -> set[int]:
    """Node indices reachable from ``start`` by walking children (inclusive)."""
    seen: set[int] = set()
    stack = [start]
    while stack:
        index = stack.pop()
        if index in seen:
            continue
        seen.add(index)
        stack.extend(int(child) for child in nodes[index].children)
    return seen


def build_render_ir(tree: SDFTree | None) -> RenderIR:
    if tree is None:
        return RenderIR((), (), (), (), (), (), ())
    nodes: dict[tuple[object, ...], int] = {}
    ir_nodes: list[RenderIRNode] = []
    unsupported: list[str] = []
    aliases: list[RenderIRObjectRef] = []
    transform = _AffineTransform.identity()
    component_refs = tuple(
        RenderIRObjectRef(
            object_id=max(component.object_id, 0),
            dimension=max(component.dimension, 0),
            node_index=_build_render_ir_node(
                component,
                nodes,
                ir_nodes,
                unsupported,
                transform,
                aliases,
            ),
        )
        for component in tree.components
    )
    root_index = _build_render_ir_node(
        tree.root,
        nodes,
        ir_nodes,
        unsupported,
        transform,
        aliases,
    )
    component_indices = tuple(ref.node_index for ref in component_refs)
    # tree.root is the solid boolean. Standalone components — placed 2D/1D
    # sections — are NOT part of it, so the renderer (which walks only the root)
    # would never reach them and they would never render. Union those non-root
    # components onto the root so the viewport shows them. Components already
    # inside the root are reachable and skipped, so boolean scenes are unchanged.
    reachable = _reachable_node_indices(ir_nodes, root_index)
    extra_roots = tuple(idx for idx in component_indices if idx not in reachable)
    if extra_roots:
        root_indices = (len(ir_nodes),)
        ir_nodes.append(
            RenderIRNode(
                kind="union",
                object_id=0,
                dimension=3,
                children=(root_index, *extra_roots),
            )
        )
    else:
        root_indices = (root_index,)
    material_refs = component_refs or (
        RenderIRObjectRef(
            object_id=max(tree.root.object_id, 0),
            dimension=max(tree.root.dimension, 0),
            node_index=root_index,
        ),
    )
    unique_aliases = tuple(
        dict.fromkeys(alias for alias in aliases if alias.object_id > 0)
    )
    unique_material_refs = tuple(
        dict.fromkeys(ref for ref in material_refs if ref.object_id > 0)
    )
    unique_component_refs = tuple(
        dict.fromkeys(ref for ref in component_refs if ref.object_id > 0)
    )
    return RenderIR(
        nodes=tuple(ir_nodes),
        root_indices=root_indices,
        component_indices=component_indices,
        aliases=unique_aliases,
        material_refs=unique_material_refs,
        component_refs=unique_component_refs,
        unsupported_kinds=tuple(dict.fromkeys(unsupported)),
    )


__all__ = [
    "RenderIR",
    "RenderIRObjectRef",
    "RenderIRNode",
    "build_render_ir",
]
