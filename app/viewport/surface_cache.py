from __future__ import annotations

from dataclasses import dataclass, replace
import math
from threading import RLock
from time import perf_counter
from typing import Callable, Literal

import numpy as np
from numpy.typing import NDArray

from core.sdf import (
    Box,
    BoxFrame,
    CappedCone,
    CircleProfile,
    Cone,
    Cylinder,
    Difference,
    EllipseProfile,
    Extrude,
    Intersection,
    OffsetProfile,
    PlacedPolyline1D,
    PlacedSDF1D,
    PlacedSDF2D,
    PolygonProfile,
    PolylineTube,
    PolylineProfile,
    Profile2D,
    QuadraticBezierCurveProfile,
    QuadraticBezierSurfaceProfile,
    QuadraticBezierTube,
    Pyramid,
    RectangleProfile,
    RegularPolygonProfile,
    Revolve,
    RoundedRectangleProfile,
    SDFNode,
    SDFTree,
    Sphere,
    SquareProfile,
    Translate,
    Torus,
    Union,
)


SurfaceStatus = Literal["ready", "outline", "empty", "failed"]

_DEFAULT_RESOLUTION = 12
# Truncation ratio for the sharp-feature QEF: eigenvalues of A^T A (= squared
# singular values) below this fraction of the largest are treated as flat
# directions and dropped, so planar faces/edges/corners get rank 1/2/3 solves.
_QEF_SINGULAR_RATIO = 0.03
# At/above this requested resolution, contour via the sparse narrow band
# (manifold, memory-bounded) using base = resolution / subdiv.
_NARROW_BAND_MIN_RES = 96
_NARROW_BAND_SUBDIV = 2
_MIN_AXIS_CELLS = 8
# Precision ceiling for generic dual contouring (booleans and unsupported SDFs).
# The vectorised vertex/index passes make this affordable on the worker thread
# (~0.7s for a 96^3 boolean), reached progressively so interaction stays smooth.
_MAX_AXIS_CELLS = 96
_MAX_CONTOURED_2D_CELLS = 96
_MAX_SAMPLED_2D_CELLS = 48
_MAX_REVOLVE_VIEWPORT_RESOLUTION = 8
_MAX_DUAL_CONTOUR_WIREFRAME_TRIANGLES = 3000
_CORNER_OFFSETS = np.asarray(
    (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    ),
    dtype=np.int32,
)
_CELL_EDGE_CORNERS = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
)
_CELL_EDGE_A = np.asarray([edge[0] for edge in _CELL_EDGE_CORNERS], dtype=np.int32)
_CELL_EDGE_B = np.asarray([edge[1] for edge in _CELL_EDGE_CORNERS], dtype=np.int32)
_CORNER_OFFSETS_F64 = _CORNER_OFFSETS.astype(np.float64)


@dataclass(frozen=True)
class ViewportSurfaceKey:
    object_id: int
    scene_revision: int
    resolution: int = _DEFAULT_RESOLUTION


@dataclass(frozen=True)
class ViewportSurface:
    key: ViewportSurfaceKey
    object_kind: str
    status: SurfaceStatus
    vertices: NDArray[np.float32]
    normals: NDArray[np.float32]
    indices: NDArray[np.uint32]
    wire_indices: NDArray[np.uint32]
    color: tuple[float, float, float]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    message: str = ""

    @property
    def has_geometry(self) -> bool:
        return bool(self.vertices.size and (self.indices.size or self.wire_indices.size))

    @property
    def triangle_count(self) -> int:
        return int(self.indices.size // 3)

    @property
    def vertex_count(self) -> int:
        return int(self.vertices.shape[0])


@dataclass(frozen=True)
class ViewportSurfaceScene:
    revision: int
    surfaces: tuple[ViewportSurface, ...]
    build_ms: float

    @property
    def has_geometry(self) -> bool:
        return any(surface.has_geometry for surface in self.surfaces)

    @property
    def vertex_count(self) -> int:
        return sum(surface.vertex_count for surface in self.surfaces)

    @property
    def triangle_count(self) -> int:
        return sum(surface.triangle_count for surface in self.surfaces)

    @property
    def failed_messages(self) -> tuple[str, ...]:
        return tuple(
            surface.message
            for surface in self.surfaces
            if surface.status == "failed" and surface.message
        )


class ViewportSurfaceCache:
    """Viewport-only CPU cache for generated draw surfaces.

    The key deliberately includes the scene revision. Geometry remains canonical
    in the SDF tree; cached surfaces are disposable viewport data.
    """

    def __init__(self, *, resolution: int = _DEFAULT_RESOLUTION) -> None:
        self.resolution = int(resolution)
        self._surfaces: dict[ViewportSurfaceKey, ViewportSurface] = {}
        # Reuse slots are bounded to one entry per live object. They hold the
        # *latest* surface only; keying by node repr/signature would retain a
        # surface (and its vertex/normal/index arrays) for every historical edit
        # of every object, an unbounded leak across an editing session.
        self._latest_by_signature: dict[int, tuple[str, ViewportSurface]] = {}
        self._latest_by_translation_signature: dict[
            int,
            tuple[str, ViewportSurface, tuple[float, float, float]],
        ] = {}
        self._lock = RLock()

    def get_or_build(self, node: SDFNode, revision: int) -> ViewportSurface:
        key = ViewportSurfaceKey(
            object_id=int(node.object_id),
            scene_revision=int(revision),
            resolution=self.resolution,
        )
        translation_signature, anchor = _translation_cache_signature(node)
        signature = repr(node)
        with self._lock:
            cached = self._surfaces.get(key)
            sig_entry = self._latest_by_signature.get(key.object_id)
            reusable = (
                sig_entry[1]
                if sig_entry is not None and sig_entry[0] == signature
                else None
            )
            trans_entry = (
                self._latest_by_translation_signature.get(key.object_id)
                if translation_signature is not None
                else None
            )
            moved_reusable = (
                (trans_entry[1], trans_entry[2])
                if trans_entry is not None and trans_entry[0] == translation_signature
                else None
            )
        if cached is not None:
            return cached
        if reusable is not None:
            surface = replace(reusable, key=key)
            self._store(key, signature, translation_signature, anchor, surface)
            return surface
        if moved_reusable is not None:
            reusable_surface, reusable_anchor = moved_reusable
            surface = _translated_surface(reusable_surface, key, anchor, reusable_anchor)
            self._store(key, signature, translation_signature, anchor, surface)
            return surface
        surface = build_viewport_surface(node, key)
        with self._lock:
            stored = self._surfaces.setdefault(key, surface)
        self._store(key, signature, translation_signature, anchor, stored)
        return stored

    def _store(
        self,
        key: ViewportSurfaceKey,
        signature: str,
        translation_signature: str | None,
        anchor: tuple[float, float, float],
        surface: ViewportSurface,
    ) -> None:
        with self._lock:
            self._surfaces[key] = surface
            self._latest_by_signature[key.object_id] = (signature, surface)
            if translation_signature is not None and surface.has_geometry:
                self._latest_by_translation_signature[key.object_id] = (
                    translation_signature,
                    surface,
                    anchor,
                )

    def prune_before(self, revision: int) -> None:
        with self._lock:
            stale = [
                key for key in self._surfaces if key.scene_revision < int(revision)
            ]
            for key in stale:
                del self._surfaces[key]

    def prune_to_object_ids(self, live_object_ids: frozenset[int]) -> None:
        """Drop reuse slots for objects no longer present in the scene.

        Bounds the reuse dictionaries to the live object set so deleted objects
        do not retain their surface arrays for the lifetime of the cache.
        """
        with self._lock:
            for store in (
                self._latest_by_signature,
                self._latest_by_translation_signature,
            ):
                stale = [oid for oid in store if oid not in live_object_ids]
                for oid in stale:
                    del store[oid]


def build_viewport_surface_scene(
    tree: SDFTree | None,
    revision: int,
    *,
    cache: ViewportSurfaceCache | None = None,
) -> ViewportSurfaceScene | None:
    if tree is None:
        return None
    start = perf_counter()
    surface_cache = cache or ViewportSurfaceCache()
    components = tuple(getattr(tree, "components", ())) or (tree.root,)
    live = tuple(
        component
        for component in components
        if int(getattr(component, "object_id", 0)) > 0
    )
    surfaces = tuple(
        surface_cache.get_or_build(component, revision) for component in live
    )
    surface_cache.prune_before(int(revision) - 2)
    surface_cache.prune_to_object_ids(
        frozenset(int(component.object_id) for component in live)
    )
    return ViewportSurfaceScene(
        revision=int(revision),
        surfaces=surfaces,
        build_ms=(perf_counter() - start) * 1000.0,
    )


def _translation_cache_signature(
    node: SDFNode,
) -> tuple[str | None, tuple[float, float, float]]:
    anchor = _translation_anchor(node)
    if anchor is None:
        return None, (0.0, 0.0, 0.0)
    shape = _translation_shape_signature(node, anchor)
    if shape is None:
        return None, (0.0, 0.0, 0.0)
    return repr(shape), anchor


def _translation_anchor(node: SDFNode) -> tuple[float, float, float] | None:
    if isinstance(node, (Sphere, Box, BoxFrame, CappedCone, Cone, Cylinder, Pyramid, Torus)):
        return _tuple3(node.center)
    if isinstance(node, (PlacedSDF1D, PlacedPolyline1D, PlacedSDF2D)):
        return _tuple3(node.origin)
    if isinstance(node, (PolylineTube, QuadraticBezierTube)):
        return _tuple3(node.points[0])
    if isinstance(node, Extrude) and node.section is not None:
        return _tuple3(node.section.origin)
    if isinstance(node, Revolve) and node.section is not None:
        return _tuple3(node.section.origin)
    if isinstance(node, Translate):
        return _tuple3(node.offset)
    return None


def _translation_shape_signature(
    node: SDFNode,
    anchor: tuple[float, float, float],
) -> object | None:
    del anchor
    if isinstance(node, Sphere):
        return ("Sphere", _sig_float(node.radius))
    if isinstance(node, Box):
        return (
            "Box",
            _sig_tuple(node.half_size),
            _sig_tuple(node.axis_u),
            _sig_tuple(node.axis_v),
            _sig_tuple(node.axis_w),
        )
    if isinstance(node, BoxFrame):
        return (
            "BoxFrame",
            _sig_tuple(node.half_size),
            _sig_float(node.thickness),
            _sig_tuple(node.axis_u),
            _sig_tuple(node.axis_v),
            _sig_tuple(node.axis_w),
        )
    if isinstance(node, Cylinder):
        return (
            "Cylinder",
            _sig_float(node.radius),
            _sig_float(node.half_height),
            _sig_tuple(node.axis_u),
            _sig_tuple(node.axis_v),
            _sig_tuple(node.axis_w),
        )
    if isinstance(node, Cone):
        return (
            "Cone",
            _sig_float(node.radius),
            _sig_float(node.half_height),
            _sig_tuple(node.axis_u),
            _sig_tuple(node.axis_v),
            _sig_tuple(node.axis_w),
        )
    if isinstance(node, CappedCone):
        return (
            "CappedCone",
            _sig_float(node.radius_a),
            _sig_float(node.radius_b),
            _sig_float(node.half_height),
            _sig_tuple(node.axis_u),
            _sig_tuple(node.axis_v),
            _sig_tuple(node.axis_w),
        )
    if isinstance(node, Pyramid):
        return (
            "Pyramid",
            _sig_float(node.base_half_size),
            _sig_float(node.half_height),
            _sig_tuple(node.axis_u),
            _sig_tuple(node.axis_v),
            _sig_tuple(node.axis_w),
        )
    if isinstance(node, Torus):
        return (
            "Torus",
            _sig_float(node.major_radius),
            _sig_float(node.minor_radius),
            _sig_tuple(node.axis_u),
            _sig_tuple(node.axis_v),
            _sig_tuple(node.axis_w),
        )
    if isinstance(node, PlacedSDF1D):
        return ("PlacedSDF1D", repr(node.profile), _sig_tuple(node.axis_u))
    if isinstance(node, PlacedPolyline1D):
        return (
            "PlacedPolyline1D",
            repr(node.profile),
            _sig_tuple(node.axis_u),
            _sig_tuple(node.axis_v),
        )
    if isinstance(node, PlacedSDF2D):
        return (
            "PlacedSDF2D",
            repr(node.profile),
            _sig_tuple(node.axis_u),
            _sig_tuple(node.axis_v),
        )
    if isinstance(node, PolylineTube):
        return (
            "PolylineTube",
            _relative_points(node.points, node.points[0]),
            _sig_float(node.radius),
            _sig_float(node.inner_radius),
            node.caps,
        )
    if isinstance(node, QuadraticBezierTube):
        return (
            "QuadraticBezierTube",
            _relative_points(node.points, node.points[0]),
            _sig_float(node.radius),
            _sig_float(node.inner_radius),
            node.caps,
        )
    if isinstance(node, Extrude) and node.section is not None:
        return (
            "Extrude",
            _translation_shape_signature(node.section, _tuple3(node.section.origin)),
            _sig_float(node.height),
            _sig_float(node.center_offset),
        )
    if isinstance(node, Revolve) and node.section is not None:
        section_origin = _tuple3(node.section.origin)
        return (
            "Revolve",
            _translation_shape_signature(node.section, section_origin),
            node.axis,
            (
                _relative_tuple(node.axis_origin, section_origin)
                if node.axis_origin is not None
                else None
            ),
            _sig_tuple(node.axis_direction) if node.axis_direction is not None else None,
            _sig_tuple(node.radial_direction) if node.radial_direction is not None else None,
            _sig_float(node.angle_degrees),
        )
    if isinstance(node, Translate):
        return ("Translate", repr(node.child))
    return None


def _translated_surface(
    surface: ViewportSurface,
    key: ViewportSurfaceKey,
    anchor: tuple[float, float, float],
    previous_anchor: tuple[float, float, float],
) -> ViewportSurface:
    delta = np.asarray(anchor, dtype=np.float32) - np.asarray(
        previous_anchor,
        dtype=np.float32,
    )
    vertices = np.asarray(surface.vertices, dtype=np.float32) + delta
    bounds_min = tuple(
        float(value + delta[index]) for index, value in enumerate(surface.bounds_min)
    )
    bounds_max = tuple(
        float(value + delta[index]) for index, value in enumerate(surface.bounds_max)
    )
    return replace(
        surface,
        key=key,
        vertices=vertices,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
    )


def _tuple3(values: tuple[float, float, float]) -> tuple[float, float, float]:
    return (float(values[0]), float(values[1]), float(values[2]))


def _sig_float(value: float) -> float:
    rounded = round(float(value), 12)
    return 0.0 if rounded == 0.0 else rounded


def _sig_tuple(values: tuple[float, ...]) -> tuple[float, ...]:
    return tuple(_sig_float(value) for value in values)


def _relative_tuple(
    values: tuple[float, float, float],
    anchor: tuple[float, float, float],
) -> tuple[float, float, float]:
    return tuple(
        _sig_float(float(values[index]) - float(anchor[index]))
        for index in range(3)
    )


def _relative_points(
    points: tuple[tuple[float, float, float], ...],
    anchor: tuple[float, float, float],
) -> tuple[tuple[float, float, float], ...]:
    return tuple(_relative_tuple(point, anchor) for point in points)


def build_viewport_surface(node: SDFNode, key: ViewportSurfaceKey) -> ViewportSurface:
    color = _object_color(key.object_id)
    try:
        primitive = _primitive_surface(node, key, color)
        if primitive is not None:
            return primitive
        if node.dimension == 3:
            # Booleans of meshable operands: clip the smooth analytic operand
            # meshes against each other's exact SDF. Smooth curved faces + an
            # exact root-found seam, with no grid polygonization. Falls back to
            # dual contouring when an operand has no analytic mesh (e.g. nested).
            clipped = _boolean_clip_surface(node, key, color)
            if clipped is not None:
                return clipped
            return _dual_contour_surface(node, key, color)
        if isinstance(node, PlacedSDF1D):
            return _placed_1d_line(node, key, color)
        if isinstance(node, PlacedPolyline1D):
            return _placed_polyline_1d(node, key, color)
        if isinstance(node, PlacedSDF2D):
            return _placed_2d_outline(node, key, color)
        return _empty_surface(node, key, color, "no viewport surface for dimension")
    except Exception as exc:  # noqa: BLE001
        return _failed_surface(node, key, color, str(exc))


SurfaceBuilder = Callable[
    [SDFNode, "ViewportSurfaceKey", "tuple[float, float, float]"],
    "ViewportSurface",
]

# Optional analytic fast-path meshers keyed by SDF node type. The generic
# dual-contour path already renders any bounded 3D SDF from bounding_box() +
# to_numpy() alone, so this registry is a pure acceleration layer: adding a new
# primitive's exact mesh is a co-located @_register_surface_builder decorator,
# never an edit to central dispatch (outcome 7).
_SURFACE_BUILDERS: dict[type, SurfaceBuilder] = {}


def _register_surface_builder(
    node_type: type,
) -> Callable[[SurfaceBuilder], SurfaceBuilder]:
    def decorate(builder: SurfaceBuilder) -> SurfaceBuilder:
        _SURFACE_BUILDERS[node_type] = builder
        return builder

    return decorate


def _primitive_surface(
    node: SDFNode,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface | None:
    # MRO walk preserves the isinstance-style subclass match the ladder had,
    # while keeping dispatch O(depth) and table-driven.
    for cls in type(node).__mro__:
        builder = _SURFACE_BUILDERS.get(cls)
        if builder is not None:
            return builder(node, key, color)
    return None


@_register_surface_builder(Sphere)
def _sphere_surface(
    node: Sphere,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface:
    segments = max(64, int(key.resolution) * 2)
    rings = max(32, int(key.resolution))
    center = np.asarray(node.center, dtype=np.float64)
    vertices: list[NDArray[np.float64]] = []
    normals: list[NDArray[np.float64]] = []
    for ring in range(rings + 1):
        phi = math.pi * ring / rings
        sin_phi = math.sin(phi)
        cos_phi = math.cos(phi)
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            normal = np.asarray(
                (
                    math.cos(theta) * sin_phi,
                    math.sin(theta) * sin_phi,
                    cos_phi,
                ),
                dtype=np.float64,
            )
            vertices.append(center + float(node.radius) * normal)
            normals.append(normal)
    indices: list[int] = []
    for ring in range(rings):
        for segment in range(segments):
            next_segment = (segment + 1) % segments
            a = ring * segments + segment
            b = ring * segments + next_segment
            c = (ring + 1) * segments + next_segment
            d = (ring + 1) * segments + segment
            if ring > 0:
                indices.extend((a, b, d))
            if ring < rings - 1:
                indices.extend((b, c, d))
    return _surface_from_arrays(node, key, color, vertices, normals, indices)


@_register_surface_builder(Box)
def _box_surface(
    node: Box,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface:
    center = np.asarray(node.center, dtype=np.float64)
    axes = (
        np.asarray(node.axis_u, dtype=np.float64),
        np.asarray(node.axis_v, dtype=np.float64),
        np.asarray(node.axis_w, dtype=np.float64),
    )
    half = np.asarray(node.half_size, dtype=np.float64)
    face_specs = (
        ((1, 0, 0), (1, 2), 1.0),
        ((-1, 0, 0), (1, 2), -1.0),
        ((0, 1, 0), (0, 2), 1.0),
        ((0, -1, 0), (0, 2), -1.0),
        ((0, 0, 1), (0, 1), 1.0),
        ((0, 0, -1), (0, 1), -1.0),
    )
    vertices: list[NDArray[np.float64]] = []
    normals: list[NDArray[np.float64]] = []
    indices: list[int] = []
    for normal_signs, tangent_axes, side in face_specs:
        normal = sum(
            sign * axis for sign, axis in zip(normal_signs, axes) if sign != 0
        )
        fixed_axis = next(index for index, sign in enumerate(normal_signs) if sign)
        local = np.zeros((4, 3), dtype=np.float64)
        local[:, fixed_axis] = side * half[fixed_axis]
        axis_a, axis_b = tangent_axes
        local[:, axis_a] = (-half[axis_a], half[axis_a], half[axis_a], -half[axis_a])
        local[:, axis_b] = (-half[axis_b], -half[axis_b], half[axis_b], half[axis_b])
        base = len(vertices)
        for corner in local:
            vertices.append(center + corner[0] * axes[0] + corner[1] * axes[1] + corner[2] * axes[2])
            normals.append(np.asarray(normal, dtype=np.float64))
        if side > 0.0:
            indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
        else:
            indices.extend((base, base + 2, base + 1, base, base + 3, base + 2))
    return _surface_from_arrays(node, key, color, vertices, normals, indices)


@_register_surface_builder(BoxFrame)
def _box_frame_surface(
    node: BoxFrame,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface:
    center = np.asarray(node.center, dtype=np.float64)
    axes = (
        np.asarray(node.axis_u, dtype=np.float64),
        np.asarray(node.axis_v, dtype=np.float64),
        np.asarray(node.axis_w, dtype=np.float64),
    )
    half = np.asarray(node.half_size, dtype=np.float64)
    radius = min(float(node.thickness) * 0.5, float(half.min()))
    vertices: list[NDArray[np.float64]] = []
    normals: list[NDArray[np.float64]] = []
    indices: list[int] = []
    for axis_index in range(3):
        tangent_axes = [index for index in range(3) if index != axis_index]
        beam_half = np.asarray((radius, radius, radius), dtype=np.float64)
        beam_half[axis_index] = half[axis_index]
        for sign_a in (-1.0, 1.0):
            for sign_b in (-1.0, 1.0):
                offset = np.zeros(3, dtype=np.float64)
                offset[tangent_axes[0]] = sign_a * (half[tangent_axes[0]] - radius)
                offset[tangent_axes[1]] = sign_b * (half[tangent_axes[1]] - radius)
                beam_center = (
                    center
                    + offset[0] * axes[0]
                    + offset[1] * axes[1]
                    + offset[2] * axes[2]
                )
                _append_oriented_box(vertices, normals, indices, beam_center, axes, beam_half)
    return _surface_from_arrays(node, key, color, vertices, normals, indices)


def _append_oriented_box(
    vertices: list[NDArray[np.float64]],
    normals: list[NDArray[np.float64]],
    indices: list[int],
    center: NDArray[np.float64],
    axes: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    half: NDArray[np.float64],
) -> None:
    face_specs = (
        ((1, 0, 0), (1, 2), 1.0),
        ((-1, 0, 0), (1, 2), -1.0),
        ((0, 1, 0), (0, 2), 1.0),
        ((0, -1, 0), (0, 2), -1.0),
        ((0, 0, 1), (0, 1), 1.0),
        ((0, 0, -1), (0, 1), -1.0),
    )
    for normal_signs, tangent_axes, side in face_specs:
        normal = sum(
            sign * axis for sign, axis in zip(normal_signs, axes) if sign != 0
        )
        fixed_axis = next(index for index, sign in enumerate(normal_signs) if sign)
        local = np.zeros((4, 3), dtype=np.float64)
        local[:, fixed_axis] = side * half[fixed_axis]
        axis_a, axis_b = tangent_axes
        local[:, axis_a] = (-half[axis_a], half[axis_a], half[axis_a], -half[axis_a])
        local[:, axis_b] = (-half[axis_b], -half[axis_b], half[axis_b], half[axis_b])
        base = len(vertices)
        for corner in local:
            vertices.append(
                center
                + corner[0] * axes[0]
                + corner[1] * axes[1]
                + corner[2] * axes[2]
            )
            normals.append(np.asarray(normal, dtype=np.float64))
        if side > 0.0:
            indices.extend((base, base + 1, base + 2, base, base + 2, base + 3))
        else:
            indices.extend((base, base + 2, base + 1, base, base + 3, base + 2))


@_register_surface_builder(Cylinder)
def _cylinder_surface(
    node: Cylinder,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface:
    segments = max(64, int(key.resolution) * 2)
    center = np.asarray(node.center, dtype=np.float64)
    u = np.asarray(node.axis_u, dtype=np.float64)
    v = np.asarray(node.axis_v, dtype=np.float64)
    w = np.asarray(node.axis_w, dtype=np.float64)
    vertices: list[NDArray[np.float64]] = []
    normals: list[NDArray[np.float64]] = []
    for z_sign in (-1.0, 1.0):
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            radial = math.cos(theta) * u + math.sin(theta) * v
            vertices.append(
                center
                + float(node.radius) * radial
                + z_sign * float(node.half_height) * w
            )
            normals.append(radial)
    top_center = len(vertices)
    vertices.append(center + float(node.half_height) * w)
    normals.append(w)
    bottom_center = len(vertices)
    vertices.append(center - float(node.half_height) * w)
    normals.append(-w)
    top_ring = len(vertices)
    for segment in range(segments):
        theta = 2.0 * math.pi * segment / segments
        radial = math.cos(theta) * u + math.sin(theta) * v
        vertices.append(center + float(node.radius) * radial + float(node.half_height) * w)
        normals.append(w)
    bottom_ring = len(vertices)
    for segment in range(segments):
        theta = 2.0 * math.pi * segment / segments
        radial = math.cos(theta) * u + math.sin(theta) * v
        vertices.append(center + float(node.radius) * radial - float(node.half_height) * w)
        normals.append(-w)
    indices: list[int] = []
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        bottom_a = segment
        bottom_b = next_segment
        top_a = segments + segment
        top_b = segments + next_segment
        indices.extend((bottom_a, bottom_b, top_b, bottom_a, top_b, top_a))
        indices.extend((top_center, top_ring + segment, top_ring + next_segment))
        indices.extend((bottom_center, bottom_ring + next_segment, bottom_ring + segment))
    return _surface_from_arrays(node, key, color, vertices, normals, indices)


@_register_surface_builder(Cone)
def _cone_surface(
    node: Cone,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface:
    segments = max(64, int(key.resolution) * 2)
    center = np.asarray(node.center, dtype=np.float64)
    u = np.asarray(node.axis_u, dtype=np.float64)
    v = np.asarray(node.axis_v, dtype=np.float64)
    w = np.asarray(node.axis_w, dtype=np.float64)
    vertices: list[NDArray[np.float64]] = []
    normals: list[NDArray[np.float64]] = []
    for segment in range(segments):
        theta = 2.0 * math.pi * segment / segments
        radial = math.cos(theta) * u + math.sin(theta) * v
        vertices.append(center + float(node.radius) * radial - float(node.half_height) * w)
        normals.append(_normalize(radial + float(node.radius) / (2.0 * float(node.half_height)) * w))
    apex = len(vertices)
    vertices.append(center + float(node.half_height) * w)
    normals.append(w)
    bottom_center = len(vertices)
    vertices.append(center - float(node.half_height) * w)
    normals.append(-w)
    bottom_ring = len(vertices)
    for segment in range(segments):
        theta = 2.0 * math.pi * segment / segments
        radial = math.cos(theta) * u + math.sin(theta) * v
        vertices.append(center + float(node.radius) * radial - float(node.half_height) * w)
        normals.append(-w)
    indices: list[int] = []
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        indices.extend((segment, next_segment, apex))
        indices.extend((bottom_center, bottom_ring + next_segment, bottom_ring + segment))
    return _surface_from_arrays(node, key, color, vertices, normals, indices)


@_register_surface_builder(CappedCone)
def _capped_cone_surface(
    node: CappedCone,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface:
    return _frustum_surface(
        node,
        key,
        color,
        center=np.asarray(node.center, dtype=np.float64),
        axes=(
            np.asarray(node.axis_u, dtype=np.float64),
            np.asarray(node.axis_v, dtype=np.float64),
            np.asarray(node.axis_w, dtype=np.float64),
        ),
        bottom_radius=float(node.radius_a),
        top_radius=float(node.radius_b),
        half_height=float(node.half_height),
    )


def _frustum_surface(
    node: SDFNode,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
    *,
    center: NDArray[np.float64],
    axes: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    bottom_radius: float,
    top_radius: float,
    half_height: float,
) -> ViewportSurface:
    segments = max(64, int(key.resolution) * 2)
    u, v, w = axes
    vertices: list[NDArray[np.float64]] = []
    normals: list[NDArray[np.float64]] = []
    slope = (bottom_radius - top_radius) / max(2.0 * half_height, 1.0e-12)
    for z_sign, radius in ((-1.0, bottom_radius), (1.0, top_radius)):
        for segment in range(segments):
            theta = 2.0 * math.pi * segment / segments
            radial = math.cos(theta) * u + math.sin(theta) * v
            vertices.append(center + radius * radial + z_sign * half_height * w)
            normals.append(_normalize(radial + slope * w))
    top_center = len(vertices)
    vertices.append(center + half_height * w)
    normals.append(w)
    bottom_center = len(vertices)
    vertices.append(center - half_height * w)
    normals.append(-w)
    top_ring = len(vertices)
    for segment in range(segments):
        theta = 2.0 * math.pi * segment / segments
        radial = math.cos(theta) * u + math.sin(theta) * v
        vertices.append(center + top_radius * radial + half_height * w)
        normals.append(w)
    bottom_ring = len(vertices)
    for segment in range(segments):
        theta = 2.0 * math.pi * segment / segments
        radial = math.cos(theta) * u + math.sin(theta) * v
        vertices.append(center + bottom_radius * radial - half_height * w)
        normals.append(-w)
    indices: list[int] = []
    for segment in range(segments):
        next_segment = (segment + 1) % segments
        bottom_a = segment
        bottom_b = next_segment
        top_a = segments + segment
        top_b = segments + next_segment
        indices.extend((bottom_a, bottom_b, top_b, bottom_a, top_b, top_a))
        indices.extend((top_center, top_ring + segment, top_ring + next_segment))
        indices.extend((bottom_center, bottom_ring + next_segment, bottom_ring + segment))
    return _surface_from_arrays(node, key, color, vertices, normals, indices)


@_register_surface_builder(Pyramid)
def _pyramid_surface(
    node: Pyramid,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface:
    center = np.asarray(node.center, dtype=np.float64)
    axes = (
        np.asarray(node.axis_u, dtype=np.float64),
        np.asarray(node.axis_v, dtype=np.float64),
        np.asarray(node.axis_w, dtype=np.float64),
    )
    half = float(node.base_half_size)
    height = float(node.half_height)
    local = (
        (-half, -half, -height),
        (half, -half, -height),
        (half, half, -height),
        (-half, half, -height),
        (0.0, 0.0, height),
    )
    world = [
        center + x * axes[0] + y * axes[1] + z * axes[2]
        for x, y, z in local
    ]
    faces = (
        (0, 2, 1),
        (0, 3, 2),
        (0, 1, 4),
        (1, 2, 4),
        (2, 3, 4),
        (3, 0, 4),
    )
    vertices: list[NDArray[np.float64]] = []
    indices: list[int] = []
    for face in faces:
        base = len(vertices)
        vertices.extend(world[index] for index in face)
        indices.extend((base, base + 1, base + 2))
    return _surface_from_arrays(node, key, color, vertices, None, indices)


@_register_surface_builder(Torus)
def _torus_surface(
    node: Torus,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface:
    major_segments = max(96, int(key.resolution) * 3)
    minor_segments = max(32, int(key.resolution))
    center = np.asarray(node.center, dtype=np.float64)
    u = np.asarray(node.axis_u, dtype=np.float64)
    v = np.asarray(node.axis_v, dtype=np.float64)
    w = np.asarray(node.axis_w, dtype=np.float64)
    vertices: list[NDArray[np.float64]] = []
    normals: list[NDArray[np.float64]] = []
    for major in range(major_segments):
        theta = 2.0 * math.pi * major / major_segments
        radial = math.cos(theta) * u + math.sin(theta) * v
        ring_center = center + float(node.major_radius) * radial
        for minor in range(minor_segments):
            phi = 2.0 * math.pi * minor / minor_segments
            normal = math.cos(phi) * radial + math.sin(phi) * w
            vertices.append(ring_center + float(node.minor_radius) * normal)
            normals.append(normal)
    indices: list[int] = []
    for major in range(major_segments):
        next_major = (major + 1) % major_segments
        for minor in range(minor_segments):
            next_minor = (minor + 1) % minor_segments
            a = major * minor_segments + minor
            b = next_major * minor_segments + minor
            c = next_major * minor_segments + next_minor
            d = major * minor_segments + next_minor
            indices.extend((a, b, c, a, c, d))
    return _surface_from_arrays(node, key, color, vertices, normals, indices)


@_register_surface_builder(PolylineTube)
def _polyline_tube_surface(
    node: PolylineTube,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface | None:
    if float(node.inner_radius) > 0.0:
        return None
    points = _drop_duplicate_points_3d(
        [np.asarray(point, dtype=np.float64) for point in node.points]
    )
    if len(points) < 2:
        return None
    return _tube_centerline_surface(node, key, color, points, float(node.radius))


@_register_surface_builder(QuadraticBezierTube)
def _quadratic_bezier_tube_surface(
    node: QuadraticBezierTube,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface | None:
    if float(node.inner_radius) > 0.0:
        return None
    points = _sample_quadratic_points_3d(node.points, key.resolution)
    if len(points) < 2:
        return None
    return _tube_centerline_surface(node, key, color, points, float(node.radius))


def _tube_centerline_surface(
    node: SDFNode,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
    points: list[NDArray[np.float64]],
    radius: float,
) -> ViewportSurface:
    ring_segments = max(32, int(key.resolution))
    frames = _tube_frames(points)
    vertices: list[NDArray[np.float64]] = []
    normals: list[NDArray[np.float64]] = []
    for point, (normal, binormal, _tangent) in zip(points, frames, strict=True):
        for segment in range(ring_segments):
            theta = 2.0 * math.pi * segment / ring_segments
            radial = math.cos(theta) * normal + math.sin(theta) * binormal
            vertices.append(point + radius * radial)
            normals.append(radial)
    indices: list[int] = []
    for ring in range(len(points) - 1):
        ring_base = ring * ring_segments
        next_base = (ring + 1) * ring_segments
        for segment in range(ring_segments):
            next_segment = (segment + 1) % ring_segments
            a = ring_base + segment
            b = ring_base + next_segment
            c = next_base + next_segment
            d = next_base + segment
            indices.extend((a, b, c, a, c, d))
    _append_tube_cap(vertices, normals, indices, points[0], frames[0], radius, ring_segments, flip=True)
    end_ring = (len(points) - 1) * ring_segments
    _append_tube_cap(
        vertices,
        normals,
        indices,
        points[-1],
        frames[-1],
        radius,
        ring_segments,
        ring_start=end_ring,
    )
    return _surface_from_arrays(node, key, color, vertices, normals, indices)


def _tube_frames(
    points: list[NDArray[np.float64]],
) -> list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
    frames: list[tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = []
    previous_normal: NDArray[np.float64] | None = None
    for index, point in enumerate(points):
        if index == 0:
            tangent = _normalize(points[1] - point)
        elif index == len(points) - 1:
            tangent = _normalize(point - points[index - 1])
        else:
            tangent = _normalize(points[index + 1] - points[index - 1])
        if previous_normal is None:
            normal = _perpendicular_axis(tangent)
        else:
            normal = previous_normal - tangent * float(np.dot(previous_normal, tangent))
            if float(np.linalg.norm(normal)) <= 1.0e-12:
                normal = _perpendicular_axis(tangent)
            else:
                normal = _normalize(normal)
        binormal = _normalize(np.cross(tangent, normal))
        normal = _normalize(np.cross(binormal, tangent))
        previous_normal = normal
        frames.append((normal, binormal, tangent))
    return frames


def _perpendicular_axis(tangent: NDArray[np.float64]) -> NDArray[np.float64]:
    candidates = (
        np.asarray((1.0, 0.0, 0.0), dtype=np.float64),
        np.asarray((0.0, 1.0, 0.0), dtype=np.float64),
        np.asarray((0.0, 0.0, 1.0), dtype=np.float64),
    )
    reference = min(candidates, key=lambda axis: abs(float(np.dot(axis, tangent))))
    return _normalize(np.cross(tangent, reference))


def _append_tube_cap(
    vertices: list[NDArray[np.float64]],
    normals: list[NDArray[np.float64]],
    indices: list[int],
    center: NDArray[np.float64],
    frame: tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]],
    radius: float,
    ring_segments: int,
    *,
    ring_start: int | None = None,
    flip: bool = False,
) -> None:
    normal, binormal, tangent = frame
    cap_normal = -tangent if flip else tangent
    center_index = len(vertices)
    vertices.append(center)
    normals.append(cap_normal)
    cap_ring = len(vertices)
    for segment in range(ring_segments):
        if ring_start is not None:
            vertices.append(vertices[ring_start + segment])
        else:
            theta = 2.0 * math.pi * segment / ring_segments
            radial = math.cos(theta) * normal + math.sin(theta) * binormal
            vertices.append(center + radius * radial)
        normals.append(cap_normal)
    for segment in range(ring_segments):
        next_segment = (segment + 1) % ring_segments
        if flip:
            indices.extend((center_index, cap_ring + next_segment, cap_ring + segment))
        else:
            indices.extend((center_index, cap_ring + segment, cap_ring + next_segment))


def _sample_quadratic_points_3d(
    points: tuple[tuple[float, float, float], ...],
    resolution: int,
) -> list[NDArray[np.float64]]:
    steps = max(8, int(resolution) * 2)
    sampled: list[NDArray[np.float64]] = []
    for span_start in range(0, len(points) - 2, 2):
        a = np.asarray(points[span_start], dtype=np.float64)
        b = np.asarray(points[span_start + 1], dtype=np.float64)
        c = np.asarray(points[span_start + 2], dtype=np.float64)
        for step in range(steps + 1):
            if sampled and step == 0:
                continue
            t = float(step) / float(steps)
            sampled.append((1.0 - t) ** 2 * a + 2.0 * (1.0 - t) * t * b + t**2 * c)
    return _drop_duplicate_points_3d(sampled)


def _drop_duplicate_points_3d(
    points: list[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    deduped: list[NDArray[np.float64]] = []
    for point in points:
        if not deduped or float(np.linalg.norm(point - deduped[-1])) > 1.0e-9:
            deduped.append(point)
    return deduped


@_register_surface_builder(Extrude)
def _extrude_profile_surface(
    node: Extrude,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface | None:
    section = node.section
    profile = getattr(section, "profile", None)
    if section is None or not isinstance(profile, Profile2D):
        return None
    outline = _profile_outline(profile, key.resolution)
    if len(outline) < 3:
        return None
    origin = np.asarray(section.origin, dtype=np.float64)
    axis_u = np.asarray(section.axis_u, dtype=np.float64)
    axis_v = np.asarray(section.axis_v, dtype=np.float64)
    normal = np.asarray(section.normal, dtype=np.float64)
    bottom_offset = float(node.center_offset) - float(node.height) * 0.5
    top_offset = float(node.center_offset) + float(node.height) * 0.5
    bottom = [
        origin + u * axis_u + v * axis_v + bottom_offset * normal
        for u, v in outline
    ]
    top = [
        origin + u * axis_u + v * axis_v + top_offset * normal
        for u, v in outline
    ]
    vertices = [*bottom, *top]
    count = len(outline)
    indices: list[int] = []
    for index in range(count):
        next_index = (index + 1) % count
        indices.extend(
            (
                index,
                next_index,
                count + next_index,
                index,
                count + next_index,
                count + index,
            )
        )
    bottom_center = len(vertices)
    vertices.append(np.asarray(bottom, dtype=np.float64).mean(axis=0))
    top_center = len(vertices)
    vertices.append(np.asarray(top, dtype=np.float64).mean(axis=0))
    for index in range(count):
        next_index = (index + 1) % count
        indices.extend((bottom_center, next_index, index))
        indices.extend((top_center, count + index, count + next_index))
    return _surface_from_arrays(node, key, color, vertices, None, indices)


@_register_surface_builder(Revolve)
def _revolve_profile_surface(
    node: Revolve,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface | None:
    section = node.section
    profile = getattr(section, "profile", None)
    if section is None or not isinstance(profile, Profile2D):
        return None
    resolution = min(int(key.resolution), _MAX_REVOLVE_VIEWPORT_RESOLUTION)
    outline = _profile_outline(profile, resolution)
    if len(outline) < 3:
        return None
    axis_origin, axis, radial_axis, tangent_axis = node._axis_frame()
    section_origin = np.asarray(section.origin, dtype=np.float64)
    section_u = np.asarray(section.axis_u, dtype=np.float64)
    section_v = np.asarray(section.axis_v, dtype=np.float64)
    outline_points = np.asarray(
        [section_origin + u * section_u + v * section_v for u, v in outline],
        dtype=np.float64,
    )
    local_outline = outline_points - axis_origin
    axial_values = local_outline @ axis
    radius_vectors = local_outline - np.outer(axial_values, axis)
    radius_values = np.linalg.norm(radius_vectors, axis=1)
    axial_points = axis_origin + np.outer(axial_values, axis)
    angle = math.radians(float(node.angle_degrees))
    closed = abs(float(node.angle_degrees)) >= 360.0 - 1.0e-9
    segments = max(32, resolution * 3)
    if not closed:
        segments = max(4, int(math.ceil(segments * abs(angle) / (2.0 * math.pi))))
    vertices: list[NDArray[np.float64]] = []
    for sweep_index in range(segments + (0 if closed else 1)):
        t = float(sweep_index) / float(segments)
        theta = (2.0 * math.pi * t) if closed else angle * t
        radial = math.cos(theta) * radial_axis + math.sin(theta) * tangent_axis
        ring = axial_points + radius_values[:, None] * radial
        vertices.extend(np.asarray(point, dtype=np.float64) for point in ring)
    ring_count = segments if closed else segments + 1
    outline_count = len(outline)
    indices: list[int] = []
    sweep_limit = segments if closed else segments
    for sweep_index in range(sweep_limit):
        next_sweep = (sweep_index + 1) % ring_count
        for index in range(outline_count):
            next_index = (index + 1) % outline_count
            a = sweep_index * outline_count + index
            b = next_sweep * outline_count + index
            c = next_sweep * outline_count + next_index
            d = sweep_index * outline_count + next_index
            indices.extend((a, b, c, a, c, d))
    if not closed:
        _append_revolve_cap(indices, 0, outline_count, vertices)
        _append_revolve_cap(
            indices,
            segments * outline_count,
            outline_count,
            vertices,
            flip=True,
        )
    vertices, indices = _deduplicate_indexed_mesh(vertices, indices)
    return _surface_from_arrays(
        node,
        key,
        color,
        vertices,
        None,
        indices,
        build_wire=False,
    )


def _append_revolve_cap(
    indices: list[int],
    ring_start: int,
    outline_count: int,
    vertices: list[NDArray[np.float64]],
    *,
    flip: bool = False,
) -> None:
    center = np.asarray(
        vertices[ring_start : ring_start + outline_count],
        dtype=np.float64,
    ).mean(axis=0)
    center_index = len(vertices)
    vertices.append(center)
    for index in range(outline_count):
        next_index = (index + 1) % outline_count
        if flip:
            indices.extend((center_index, ring_start + next_index, ring_start + index))
        else:
            indices.extend((center_index, ring_start + index, ring_start + next_index))


def _profile_outline(
    profile: Profile2D,
    resolution: int,
) -> list[tuple[float, float]]:
    if isinstance(profile, QuadraticBezierSurfaceProfile):
        return _quadratic_surface_outline(profile, resolution)
    if isinstance(profile, CircleProfile):
        return _ellipse_outline(profile.center, (profile.radius, profile.radius), resolution)
    if isinstance(profile, EllipseProfile):
        return _ellipse_outline(profile.center, profile.semi_axes, resolution)
    if isinstance(profile, RoundedRectangleProfile):
        return _rounded_rectangle_outline(profile, resolution)
    if isinstance(profile, SquareProfile):
        half = float(profile.half_size)
        return _rectangle_outline(profile.center, (half, half))
    if isinstance(profile, RectangleProfile):
        return _rectangle_outline(profile.center, profile.half_size)
    if isinstance(profile, RegularPolygonProfile):
        return _regular_polygon_outline(profile)
    if isinstance(profile, PolygonProfile):
        return _closed_points(profile.points)
    if isinstance(profile, OffsetProfile):
        child_outline = _profile_outline(profile.child, resolution)
        return [
            (u + float(profile.offset[0]), v + float(profile.offset[1]))
            for u, v in child_outline
        ]
    return []


def _ellipse_outline(
    center: tuple[float, float],
    semi_axes: tuple[float, float],
    resolution: int,
) -> list[tuple[float, float]]:
    segments = max(40, int(resolution) * 4)
    cu, cv = center
    au, av = semi_axes
    return [
        (
            float(cu + au * math.cos(2.0 * math.pi * index / segments)),
            float(cv + av * math.sin(2.0 * math.pi * index / segments)),
        )
        for index in range(segments)
    ]


def _rectangle_outline(
    center: tuple[float, float],
    half_size: tuple[float, float],
) -> list[tuple[float, float]]:
    cu, cv = center
    hu, hv = half_size
    return [
        (float(cu - hu), float(cv - hv)),
        (float(cu + hu), float(cv - hv)),
        (float(cu + hu), float(cv + hv)),
        (float(cu - hu), float(cv + hv)),
    ]


def _rounded_rectangle_outline(
    profile: RoundedRectangleProfile,
    resolution: int,
) -> list[tuple[float, float]]:
    cu, cv = profile.center
    radius = float(profile.corner_radius)
    inner_u = float(profile.half_size[0]) - radius
    inner_v = float(profile.half_size[1]) - radius
    arc_steps = max(6, int(resolution) // 2)
    corners = (
        (cu + inner_u, cv - inner_v, -0.5 * math.pi, 0.0),
        (cu + inner_u, cv + inner_v, 0.0, 0.5 * math.pi),
        (cu - inner_u, cv + inner_v, 0.5 * math.pi, math.pi),
        (cu - inner_u, cv - inner_v, math.pi, 1.5 * math.pi),
    )
    outline: list[tuple[float, float]] = []
    for corner_u, corner_v, start, end in corners:
        for step in range(arc_steps + 1):
            if outline and step == 0:
                continue
            theta = start + (end - start) * float(step) / float(arc_steps)
            outline.append(
                (
                    float(corner_u + radius * math.cos(theta)),
                    float(corner_v + radius * math.sin(theta)),
                )
            )
    return outline


def _regular_polygon_outline(
    profile: RegularPolygonProfile,
) -> list[tuple[float, float]]:
    cu, cv = profile.center
    return [
        (
            float(cu + profile.radius * math.cos(angle)),
            float(cv + profile.radius * math.sin(angle)),
        )
        for angle in (
            profile.rotation + index * 2.0 * math.pi / profile.side_count
            for index in range(profile.side_count)
        )
    ]


def _closed_points(
    points: tuple[tuple[float, float], ...],
) -> list[tuple[float, float]]:
    outline = [(float(u), float(v)) for u, v in points]
    if len(outline) >= 2 and outline[0] == outline[-1]:
        outline.pop()
    return outline


def _quadratic_surface_outline(
    profile: QuadraticBezierSurfaceProfile,
    resolution: int,
) -> list[tuple[float, float]]:
    steps = max(12, int(resolution) * 2)
    sampled: list[tuple[float, float]] = []
    points = profile.points
    for span_start in range(0, len(points) - 2, 2):
        a = np.asarray(points[span_start], dtype=np.float64)
        b = np.asarray(points[span_start + 1], dtype=np.float64)
        c = np.asarray(points[span_start + 2], dtype=np.float64)
        for step in range(steps + 1):
            if sampled and step == 0:
                continue
            t = float(step) / float(steps)
            point = (1.0 - t) ** 2 * a + 2.0 * (1.0 - t) * t * b + t**2 * c
            sampled.append((float(point[0]), float(point[1])))
    first = np.asarray(sampled[0], dtype=np.float64)
    last = np.asarray(sampled[-1], dtype=np.float64)
    if float(np.linalg.norm(first - last)) > 1.0e-9:
        for step in range(1, steps + 1):
            t = float(step) / float(steps)
            point = (1.0 - t) * last + t * first
            sampled.append((float(point[0]), float(point[1])))
    return _drop_duplicate_points(sampled)


def _drop_duplicate_points(
    points: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    deduped: list[tuple[float, float]] = []
    for point in points:
        if not deduped:
            deduped.append(point)
            continue
        if float(np.linalg.norm(np.asarray(point) - np.asarray(deduped[-1]))) > 1.0e-9:
            deduped.append(point)
    if len(deduped) > 1:
        if float(np.linalg.norm(np.asarray(deduped[0]) - np.asarray(deduped[-1]))) <= 1.0e-9:
            deduped.pop()
    return deduped


def _surface_from_arrays(
    node: SDFNode,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
    vertices: list[NDArray[np.float64]],
    normals: list[NDArray[np.float64]] | None,
    indices: list[int],
    *,
    build_wire: bool = True,
) -> ViewportSurface:
    vertex_array = np.asarray(vertices, dtype=np.float32)
    index_array = np.asarray(indices, dtype=np.uint32)
    normal_array = (
        _mesh_normals(vertex_array, index_array)
        if normals is None
        else np.asarray(normals, dtype=np.float32)
    )
    wire = (
        _wire_indices_from_triangles(index_array)
        if build_wire
        else np.zeros(0, dtype=np.uint32)
    )
    return ViewportSurface(
        key=key,
        object_kind=type(node).__name__,
        status="ready",
        vertices=vertex_array,
        normals=normal_array,
        indices=index_array,
        wire_indices=wire,
        color=color,
        bounds_min=tuple(float(value) for value in vertex_array.min(axis=0)),
        bounds_max=tuple(float(value) for value in vertex_array.max(axis=0)),
    )


def _deduplicate_indexed_mesh(
    vertices: list[NDArray[np.float64]],
    indices: list[int],
) -> tuple[list[NDArray[np.float64]], list[int]]:
    if not vertices or not indices:
        return vertices, indices
    vertex_array = np.asarray(vertices, dtype=np.float64)
    rounded = np.round(vertex_array, decimals=12)
    welded_array, inverse = np.unique(rounded, axis=0, return_inverse=True)
    triangle_array = inverse[np.asarray(indices, dtype=np.int64)].reshape(-1, 3)
    distinct = (
        (triangle_array[:, 0] != triangle_array[:, 1])
        & (triangle_array[:, 0] != triangle_array[:, 2])
        & (triangle_array[:, 1] != triangle_array[:, 2])
    )
    triangle_array = triangle_array[distinct]
    if triangle_array.size == 0:
        return [row.copy() for row in welded_array], []

    pa = welded_array[triangle_array[:, 0]]
    pb = welded_array[triangle_array[:, 1]]
    pc = welded_array[triangle_array[:, 2]]
    areas = np.linalg.norm(np.cross(pb - pa, pc - pa), axis=1)
    triangle_array = triangle_array[areas > 1.0e-14]
    if triangle_array.size == 0:
        return [row.copy() for row in welded_array], []

    sorted_faces = np.sort(triangle_array, axis=1)
    _unique, keep_indices = np.unique(sorted_faces, axis=0, return_index=True)
    keep_indices.sort()
    cleaned = triangle_array[keep_indices].reshape(-1).astype(int).tolist()
    return [row.copy() for row in welded_array], cleaned


def _placed_1d_line(
    node: PlacedSDF1D,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface:
    assert node.profile is not None
    t_min, t_max = node.profile.bounds()
    resolution = max(24, min(256, int(key.resolution) * 8))
    samples = np.linspace(t_min, t_max, resolution + 1, dtype=np.float64)
    mids = (samples[:-1] + samples[1:]) * 0.5
    inside = np.asarray(node.profile.to_numpy(mids) <= 0.0, dtype=np.bool_)
    spans: list[tuple[float, float]] = []
    start: float | None = None
    for index, is_inside in enumerate(inside):
        if is_inside and start is None:
            start = float(samples[index])
        if start is not None and (not is_inside or index == len(inside) - 1):
            end = float(samples[index] if not is_inside else samples[index + 1])
            if end > start:
                spans.append((start, end))
            start = None
    if not spans:
        return _empty_surface(node, key, color, "1D profile produced no line spans")
    origin = np.asarray(node.origin, dtype=np.float64)
    axis = np.asarray(node.axis_u, dtype=np.float64)
    vertices: list[NDArray[np.float64]] = []
    for start_t, end_t in spans:
        vertices.append(origin + start_t * axis)
        vertices.append(origin + end_t * axis)
    return _wire_surface_from_points(
        node,
        key,
        color,
        vertices,
        [(index, index + 1) for index in range(0, len(vertices), 2)],
    )


def _placed_polyline_1d(
    node: PlacedPolyline1D,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface:
    assert node.profile is not None
    local_points: list[tuple[float, float]]
    if isinstance(node.profile, PolylineProfile):
        local_points = list(node.profile.points)
    elif isinstance(node.profile, QuadraticBezierCurveProfile):
        local_points = _sample_quadratic_curve(node.profile, key.resolution)
    else:
        return _empty_surface(node, key, color, "unsupported 1D curve profile")
    if len(local_points) < 2:
        return _empty_surface(node, key, color, "1D curve has too few points")
    origin = np.asarray(node.origin, dtype=np.float64)
    axis_u = np.asarray(node.axis_u, dtype=np.float64)
    axis_v = np.asarray(node.axis_v, dtype=np.float64)
    vertices = [
        origin + float(u) * axis_u + float(v) * axis_v
        for u, v in local_points
    ]
    segments = [(index, index + 1) for index in range(len(vertices) - 1)]
    return _wire_surface_from_points(node, key, color, vertices, segments)


def _sample_quadratic_curve(
    profile: QuadraticBezierCurveProfile,
    resolution: int,
) -> list[tuple[float, float]]:
    steps = max(12, int(resolution) * 2)
    sampled: list[tuple[float, float]] = []
    points = profile.points
    for span_start in range(0, len(points) - 2, 2):
        a = np.asarray(points[span_start], dtype=np.float64)
        b = np.asarray(points[span_start + 1], dtype=np.float64)
        c = np.asarray(points[span_start + 2], dtype=np.float64)
        for step in range(steps + 1):
            if sampled and step == 0:
                continue
            t = float(step) / float(steps)
            point = (1.0 - t) ** 2 * a + 2.0 * (1.0 - t) * t * b + t**2 * c
            sampled.append((float(point[0]), float(point[1])))
    return sampled


def _wire_surface_from_points(
    node: SDFNode,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
    vertices: list[NDArray[np.float64]],
    segments: list[tuple[int, int]],
) -> ViewportSurface:
    vertex_array = np.asarray(vertices, dtype=np.float32)
    normals = np.zeros_like(vertex_array)
    normals[:, 2] = 1.0
    wire = np.asarray([index for segment in segments for index in segment], dtype=np.uint32)
    return ViewportSurface(
        key=key,
        object_kind=type(node).__name__,
        status="outline",
        vertices=vertex_array,
        normals=normals,
        indices=np.zeros(0, dtype=np.uint32),
        wire_indices=wire,
        color=color,
        bounds_min=tuple(float(value) for value in vertex_array.min(axis=0)),
        bounds_max=tuple(float(value) for value in vertex_array.max(axis=0)),
        message="1D object rendered as line geometry",
    )


def _clip_mesh_to_sdf(
    verts: NDArray[np.float64],
    normals: NDArray[np.float64],
    tris: NDArray[np.int64],
    clip: SDFNode,
    keep_inside: bool,
    eps: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Clip a triangle mesh against an SDF half-space (marching triangles).

    Keeps the portion of the mesh where ``clip`` is inside (<=0) or outside
    (>=0); triangles straddling the boundary are split, and the new cut vertices
    are root-found exactly onto ``clip``'s zero isosurface so the seam is exact
    and smooth, not grid-polygonised. Cut vertices keep the *original* mesh's
    interpolated normal (the kept surface), preserving smooth shading. The cut
    point on a shared edge is identical for both incident triangles, so the seam
    is gap-free.
    """
    sv = np.asarray(clip.to_numpy(verts[:, 0], verts[:, 1], verts[:, 2]), dtype=np.float64)
    keep = sv <= 0.0 if keep_inside else sv >= 0.0
    ktri = keep[tris]
    k0, k1, k2 = ktri[:, 0], ktri[:, 1], ktri[:, 2]
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]

    new_pos: list[NDArray[np.float64]] = []
    new_nrm: list[NDArray[np.float64]] = []
    cut_index = [np.full(tris.shape[0], -1, dtype=np.int64) for _ in range(3)]
    cursor = verts.shape[0]
    for ei, (i, j) in enumerate(((0, 1), (1, 2), (2, 0))):
        a = tris[:, i]
        b = tris[:, j]
        cross = keep[a] != keep[b]
        if not np.any(cross):
            continue
        ai, bi = a[cross], b[cross]
        fa, fb = sv[ai], sv[bi]
        # Cut exactly onto clip's zero isosurface. The edge endpoints are on the
        # operand surface, so the cut already lands on the seam {original≈0,
        # clip=0} to root tolerance; both clipped pieces meet there.
        pts, _ = _refine_edge_hermite(clip, verts[ai], verts[bi], fa, fb, eps)
        t = np.clip(
            fa / np.where(np.abs(fa - fb) > 1.0e-12, fa - fb, 1.0), 0.0, 1.0
        )
        nrm = _normalize_rows(normals[ai] + t[:, None] * (normals[bi] - normals[ai]))
        ids = cursor + np.arange(ai.shape[0], dtype=np.int64)
        cut_index[ei][cross] = ids
        new_pos.append(pts)
        new_nrm.append(nrm)
        cursor += ai.shape[0]
    c01, c12, c20 = cut_index

    out: list[NDArray[np.int64]] = []

    def add(mask: NDArray[np.bool_], *cols: NDArray[np.int64]) -> None:
        if np.any(mask):
            out.append(np.stack([col[mask] for col in cols], axis=1))

    add(k0 & k1 & k2, v0, v1, v2)
    add(k0 & ~k1 & ~k2, v0, c01, c20)
    add(~k0 & k1 & ~k2, v1, c12, c01)
    add(~k0 & ~k1 & k2, v2, c20, c12)
    m = k0 & k1 & ~k2
    add(m, v0, v1, c12)
    add(m, v0, c12, c20)
    m = ~k0 & k1 & k2
    add(m, v1, v2, c20)
    add(m, v1, c20, c01)
    m = k0 & ~k1 & k2
    add(m, v2, v0, c01)
    add(m, v2, c01, c12)

    all_pos = np.concatenate([verts, *new_pos]) if new_pos else verts
    all_nrm = np.concatenate([normals, *new_nrm]) if new_nrm else normals
    faces = np.concatenate(out) if out else np.zeros((0, 3), dtype=np.int64)
    return all_pos, all_nrm, faces


def _grid_box_mesh(
    node: Box, n: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """A box tessellated with an n x n grid per face.

    Flat-face primitives must be subdivided for mesh-SDF clipping: a curved cut
    through a face is captured only where the face has vertices. (The analytic
    2-triangle box face would miss the cut entirely.)
    """
    center = np.asarray(node.center, dtype=np.float64)
    axes = (
        np.asarray(node.axis_u, dtype=np.float64),
        np.asarray(node.axis_v, dtype=np.float64),
        np.asarray(node.axis_w, dtype=np.float64),
    )
    half = np.asarray(node.half_size, dtype=np.float64)
    grid = np.linspace(-1.0, 1.0, n + 1)
    gu, gv = np.meshgrid(grid, grid, indexing="ij")
    verts: list[NDArray[np.float64]] = []
    norms: list[NDArray[np.float64]] = []
    faces: list[NDArray[np.int64]] = []
    base = 0
    # (fixed axis, sign) for each of the 6 faces; the other two axes span the grid.
    for fixed, side in ((0, 1.0), (0, -1.0), (1, 1.0), (1, -1.0), (2, 1.0), (2, -1.0)):
        a, b = [ax for ax in range(3) if ax != fixed]
        normal = side * axes[fixed]
        local = (
            center
            + side * half[fixed] * axes[fixed]
            + (gu[..., None] * half[a]) * axes[a]
            + (gv[..., None] * half[b]) * axes[b]
        ).reshape(-1, 3)
        verts.append(local)
        norms.append(np.broadcast_to(normal, local.shape).copy())
        # two triangles per grid cell, wound to face outward (+side).
        ii, jj = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        v00 = (ii * (n + 1) + jj).reshape(-1) + base
        v10 = v00 + (n + 1)
        v01 = v00 + 1
        v11 = v10 + 1
        if side > 0.0:
            faces.append(np.stack([v00, v10, v11], axis=1))
            faces.append(np.stack([v00, v11, v01], axis=1))
        else:
            faces.append(np.stack([v00, v11, v10], axis=1))
            faces.append(np.stack([v00, v01, v11], axis=1))
        base += local.shape[0]
    return (
        np.concatenate(verts),
        np.concatenate(norms),
        np.concatenate(faces),
    )


def _split_marked_triangles(
    tris: NDArray[np.int64],
    m01: NDArray[np.int64],
    m12: NDArray[np.int64],
    m20: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Edge-based 1/2/3-split subdivision (red-green, crack-free).

    ``mXY`` is the inserted midpoint vertex id for edge (vX, vY) or -1. The split
    count per triangle selects the sub-triangulation; the shared midpoint of a
    split edge is identical for both incident triangles, so no T-junctions form.
    """
    v0, v1, v2 = tris[:, 0], tris[:, 1], tris[:, 2]
    s01, s12, s20 = m01 >= 0, m12 >= 0, m20 >= 0
    count = s01.astype(np.int64) + s12.astype(np.int64) + s20.astype(np.int64)
    out: list[NDArray[np.int64]] = [tris[count == 0]]

    def emit(mask: NDArray[np.bool_], *triangles: tuple) -> None:
        if np.any(mask):
            for tri in triangles:
                out.append(np.stack(tri, axis=1))

    m = count == 3
    a, b, c, p, q, r = v0[m], v1[m], v2[m], m01[m], m12[m], m20[m]
    emit(m, (a, p, r), (p, b, q), (r, q, c), (p, q, r))
    m = (count == 1) & s01
    a, b, c, p = v0[m], v1[m], v2[m], m01[m]
    emit(m, (a, p, c), (p, b, c))
    m = (count == 1) & s12
    a, b, c, p = v0[m], v1[m], v2[m], m12[m]
    emit(m, (a, b, p), (a, p, c))
    m = (count == 1) & s20
    a, b, c, p = v0[m], v1[m], v2[m], m20[m]
    emit(m, (a, b, p), (b, c, p))
    m = (count == 2) & s01 & s12
    a, b, c, p, q = v0[m], v1[m], v2[m], m01[m], m12[m]
    emit(m, (a, p, c), (p, b, q), (p, q, c))
    m = (count == 2) & s12 & s20
    a, b, c, q, r = v0[m], v1[m], v2[m], m12[m], m20[m]
    emit(m, (b, q, a), (q, c, r), (q, r, a))
    m = (count == 2) & s20 & s01
    a, b, c, r, p = v0[m], v1[m], v2[m], m20[m], m01[m]
    emit(m, (c, r, b), (r, a, p), (r, p, b))

    parts = [part for part in out if part.shape[0] > 0]
    if not parts:
        return np.zeros((0, 3), dtype=np.int64)
    return np.concatenate(parts, axis=0)


def _tessellate_for_clip(
    verts: NDArray[np.float64],
    normals: NDArray[np.float64],
    tris: NDArray[np.int64],
    clip: SDFNode,
    target_edge: float,
    max_passes: int = 9,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]]:
    """Refine mesh edges that are long AND near the ``clip`` cut (crack-free).

    Extrude/revolve meshes have tall un-subdivided wall quads and fan caps; a
    curved SDF cut through them is only captured where the mesh has vertices.
    Refinement is concentrated in a narrow band around the cut (|clip| small or a
    sign change across the edge), so the seam is smooth without exploding the flat
    regions far from it. New vertices are edge midpoints with averaged normals.
    """
    for _ in range(max_passes):
        a = verts[tris]
        edge_len = np.stack(
            (
                np.linalg.norm(a[:, 1] - a[:, 0], axis=1),
                np.linalg.norm(a[:, 2] - a[:, 1], axis=1),
                np.linalg.norm(a[:, 0] - a[:, 2], axis=1),
            ),
            axis=1,
        )
        cv = np.asarray(clip.to_numpy(verts[:, 0], verts[:, 1], verts[:, 2]))
        band = 2.0 * target_edge
        ct = cv[tris]
        # An edge is in the cut band if it crosses clip=0 or an endpoint is near it.
        near = np.stack(
            (
                (np.sign(ct[:, 0]) != np.sign(ct[:, 1]))
                | (np.minimum(np.abs(ct[:, 0]), np.abs(ct[:, 1])) < band),
                (np.sign(ct[:, 1]) != np.sign(ct[:, 2]))
                | (np.minimum(np.abs(ct[:, 1]), np.abs(ct[:, 2])) < band),
                (np.sign(ct[:, 2]) != np.sign(ct[:, 0]))
                | (np.minimum(np.abs(ct[:, 2]), np.abs(ct[:, 0])) < band),
            ),
            axis=1,
        )
        long_edge = (edge_len > target_edge) & near
        if not np.any(long_edge):
            break
        edges = np.concatenate((tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]))
        long_flat = np.concatenate(
            (long_edge[:, 0], long_edge[:, 1], long_edge[:, 2])
        )
        lo = np.minimum(edges[:, 0], edges[:, 1])[long_flat]
        hi = np.maximum(edges[:, 0], edges[:, 1])[long_flat]
        scale = np.int64(verts.shape[0] + 1)
        key = lo * scale + hi
        unique_key, first = np.unique(key, return_index=True)
        ulo, uhi = lo[first], hi[first]
        mid = 0.5 * (verts[ulo] + verts[uhi])
        mnorm = _normalize_rows(normals[ulo] + normals[uhi])
        new_ids = verts.shape[0] + np.arange(unique_key.shape[0], dtype=np.int64)
        order = np.argsort(unique_key)
        sorted_key, sorted_vid = unique_key[order], new_ids[order]

        def lookup(u: NDArray[np.int64], v: NDArray[np.int64]) -> NDArray[np.int64]:
            klo = np.minimum(u, v)
            khi = np.maximum(u, v)
            k = klo * scale + khi
            pos = np.clip(np.searchsorted(sorted_key, k), 0, sorted_key.shape[0] - 1)
            return np.where(sorted_key[pos] == k, sorted_vid[pos], np.int64(-1))

        m01 = lookup(tris[:, 0], tris[:, 1])
        m12 = lookup(tris[:, 1], tris[:, 2])
        m20 = lookup(tris[:, 2], tris[:, 0])
        verts = np.concatenate((verts, mid))
        normals = np.concatenate((normals, mnorm))
        tris = _split_marked_triangles(tris, m01, m12, m20)
    return verts, normals, tris


def _clip_operand_mesh(
    node: SDFNode, key: ViewportSurfaceKey, color: tuple[float, float, float]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]] | None:
    """Adequately tessellated operand mesh for SDF clipping, or None.

    Curved primitives come pre-tessellated from the analytic meshers; the box is
    rebuilt as a per-face grid; extrude/revolve meshes are adaptively refined so
    their flat caps and tall walls can capture a curved cut. Operands without a
    fine analytic mesh return None so the boolean falls back to dual contouring.
    """
    n = max(8, int(key.resolution))
    if isinstance(node, Box):
        return _grid_box_mesh(node, n)
    if isinstance(node, (Sphere, Cylinder, Cone, CappedCone, Torus, Extrude, Revolve)):
        surface = _primitive_surface(node, key, color)
        if surface is None or not surface.has_geometry:
            return None
        return (
            surface.vertices.astype(np.float64),
            surface.normals.astype(np.float64),
            surface.indices.reshape(-1, 3).astype(np.int64),
        )
    return None


def _boolean_clip_surface(
    node: SDFNode,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface | None:
    if not isinstance(node, (Union, Intersection, Difference)):
        return None
    left = node.left
    right = node.right
    if left is None or right is None:
        return None
    operand_l = _clip_operand_mesh(left, key, color)
    operand_r = _clip_operand_mesh(right, key, color)
    if operand_l is None or operand_r is None:
        return None

    # keep_inside for each operand mesh, and whether the right operand's facing
    # is inverted (difference exposes B as an inner wall).
    if isinstance(node, Intersection):
        keep_l, keep_r, flip_r = True, True, False
    elif isinstance(node, Union):
        keep_l, keep_r, flip_r = False, False, False
    else:  # Difference: A - B
        keep_l, keep_r, flip_r = False, True, True

    box = node.bounding_box()
    extent = max(
        box.x_max - box.x_min, box.y_max - box.y_min, box.z_max - box.z_min, 1.0
    )
    eps = np.full(3, max(extent * 1.0e-4, 1.0e-6), dtype=np.float64)
    # Refine each operand mesh only in the band where the *other* operand cuts it,
    # so a curved cut through a coarse flat region (extrude/revolve wall or cap) is
    # captured smoothly without inflating the rest of the mesh.
    target = extent / max(8.0, float(key.resolution))
    operand_l = _tessellate_for_clip(*operand_l, right, target)
    operand_r = _tessellate_for_clip(*operand_r, left, target)

    vl, nl, fl = _clip_mesh_to_sdf(*operand_l, right, keep_l, eps)
    vr, nr, fr = _clip_mesh_to_sdf(*operand_r, left, keep_r, eps)
    if flip_r:
        nr = -nr
        fr = fr[:, [0, 2, 1]]

    vertices = np.concatenate([vl, vr]).astype(np.float32)
    normals = np.concatenate([nl, nr]).astype(np.float32)
    faces = np.concatenate([fl, fr + vl.shape[0]])
    if faces.shape[0] == 0:
        return _empty_surface(node, key, color, "boolean clip produced no geometry")
    index_array = _orient_triangles(
        vertices, normals, faces.reshape(-1).astype(np.uint32)
    )
    # Drop the operand verts the clip discarded (roughly half), remapping indices.
    used = np.unique(index_array)
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    vertices = vertices[used]
    normals = normals[used]
    index_array = remap[index_array].astype(np.uint32)
    triangle_count = int(index_array.size // 3)
    wire = (
        _wire_indices_from_triangles(index_array)
        if triangle_count <= _MAX_DUAL_CONTOUR_WIREFRAME_TRIANGLES
        else np.zeros(0, dtype=np.uint32)
    )
    return ViewportSurface(
        key=key,
        object_kind=type(node).__name__,
        status="ready",
        vertices=vertices,
        normals=normals,
        indices=index_array,
        wire_indices=wire,
        color=color,
        bounds_min=tuple(float(v) for v in vertices.min(axis=0)),
        bounds_max=tuple(float(v) for v in vertices.max(axis=0)),
        message="boolean rendered as SDF-clipped analytic meshes",
    )


def _dual_contour_surface(
    node: SDFNode,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface:
    # Top precision tier: sparse narrow band (manifold/watertight by construction,
    # no dense fine grid). Lower tiers use the dense path for fast first paints.
    if int(key.resolution) >= _NARROW_BAND_MIN_RES:
        base_res = max(_MIN_AXIS_CELLS, int(key.resolution) // _NARROW_BAND_SUBDIV)
        return _narrow_band_dual_contour(
            node, key, color, base_res, _NARROW_BAND_SUBDIV
        )
    mins, maxs = _sampling_bounds(node, key.resolution)
    dims = _axis_cell_counts(mins, maxs, key.resolution)
    xs = np.linspace(mins[0], maxs[0], dims[0] + 1, dtype=np.float64)
    ys = np.linspace(mins[1], maxs[1], dims[1] + 1, dtype=np.float64)
    zs = np.linspace(mins[2], maxs[2], dims[2] + 1, dtype=np.float64)
    xg, yg, zg = np.meshgrid(xs, ys, zs, indexing="ij")
    values = node.to_numpy(xg, yg, zg)
    if not (np.any(values <= 0.0) and np.any(values >= 0.0)):
        return _empty_surface(node, key, color, "no zero crossing in viewport bounds")

    cell_vertex, vertex_array, normal_array = _dual_contour_cell_vertices(
        node,
        values,
        xs,
        ys,
        zs,
        dims,
    )

    if vertex_array.size == 0:
        return _empty_surface(node, key, color, "dual contour produced no cells")

    # Grid-gradient normals (np.gradient over the value field) are central
    # differences across whole cells; at a CSG seam they average the two
    # incident surfaces and round the crease. Resample the analytic SDF gradient
    # at the final vertex positions for crisp, feature-accurate shading.
    step = np.asarray(
        (xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]), dtype=np.float64
    )
    normal_array = _analytic_vertex_normals(node, vertex_array, step, normal_array)

    index_array = _dual_contour_indices(values, cell_vertex)
    index_array = _orient_triangles(vertex_array, normal_array, index_array)
    triangle_count = int(index_array.size // 3)
    wire = (
        _wire_indices_from_triangles(index_array)
        if triangle_count <= _MAX_DUAL_CONTOUR_WIREFRAME_TRIANGLES
        else np.zeros(0, dtype=np.uint32)
    )
    status: SurfaceStatus = "ready" if index_array.size else "empty"
    return ViewportSurface(
        key=key,
        object_kind=type(node).__name__,
        status=status,
        vertices=vertex_array,
        normals=normal_array,
        indices=index_array,
        wire_indices=wire,
        color=color,
        bounds_min=tuple(float(value) for value in mins),
        bounds_max=tuple(float(value) for value in maxs),
        message="" if index_array.size else "dual contour produced no faces",
    )


# Per-cell minimal-edge quad specs for sparse dual contouring. Each crossing cell
# "owns" the x/y/z edge at its corner 0; the quad joins it to three neighbours.
# (corner_a, corner_b, neighbour offsets in dense quad order, flip-on-corner0>0).
_DC_EDGE_SPECS = (
    (0, 1, ((0, -1, -1), (0, 0, -1), (0, 0, 0), (0, -1, 0)), True),
    (0, 2, ((-1, 0, -1), (0, 0, -1), (0, 0, 0), (-1, 0, 0)), False),
    (0, 4, ((-1, -1, 0), (0, -1, 0), (0, 0, 0), (-1, 0, 0)), True),
)


def _cell_vid_lookup(
    cell_coords: NDArray[np.int64],
    dims_fine: NDArray[np.int64],
    sorted_lid: NDArray[np.int64],
    sorted_vid: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Vertex index for each requested fine cell, or -1 if it has no vertex."""
    in_bounds = np.all((cell_coords >= 0) & (cell_coords < dims_fine), axis=1)
    lid = (
        cell_coords[:, 0] * dims_fine[1] + cell_coords[:, 1]
    ) * dims_fine[2] + cell_coords[:, 2]
    lid = np.where(in_bounds, lid, -1)
    pos = np.clip(np.searchsorted(sorted_lid, lid), 0, sorted_lid.shape[0] - 1)
    hit = in_bounds & (sorted_lid[pos] == lid)
    return np.where(hit, sorted_vid[pos], -1)


def _narrow_band_dual_contour(
    node: SDFNode,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
    base_res: int,
    subdiv: int,
) -> ViewportSurface:
    """Quasi-exact dual contouring on a sparse narrow band around the surface.

    A coarse base grid locates the surface; only the crossing coarse cells are
    subdivided (uniformly, factor ``subdiv``) and contoured. Effective resolution
    is ``base_res * subdiv`` at the surface, but cost and memory scale with the
    O(n^2) surface band, not the O(n^3) volume — no dense fine grid is ever
    materialised. The band is a single uniform fine level, so the mesh is
    crack-free/manifold by construction (no T-junctions to stitch).
    """
    mins, maxs = _sampling_bounds(node, base_res)
    cdims = _axis_cell_counts(mins, maxs, base_res).astype(np.int64)
    cxs = np.linspace(mins[0], maxs[0], int(cdims[0]) + 1, dtype=np.float64)
    cys = np.linspace(mins[1], maxs[1], int(cdims[1]) + 1, dtype=np.float64)
    czs = np.linspace(mins[2], maxs[2], int(cdims[2]) + 1, dtype=np.float64)
    cxg, cyg, czg = np.meshgrid(cxs, cys, czs, indexing="ij")
    cvals = node.to_numpy(cxg, cyg, czg)
    if not (np.any(cvals <= 0.0) and np.any(cvals >= 0.0)):
        return _empty_surface(node, key, color, "no zero crossing in viewport bounds")

    corner = _cell_corner_stack(cvals, cdims)
    cmin = corner.min(axis=-1)
    cmax = corner.max(axis=-1)
    coarse_cross = (cmin <= 0.0) & (cmax >= 0.0) & ((cmax - cmin) > 1.0e-12)
    coarse_cells = np.argwhere(coarse_cross).astype(np.int64)
    if coarse_cells.shape[0] == 0:
        return _empty_surface(node, key, color, "dual contour produced no cells")

    # Dilate the band so coarse crossing detection cannot miss surface that clips
    # a same-sign cell and leave a surface fine cell with an inactive neighbour
    # (a crack). DC quads only join face/edge neighbours (never the 8 cube
    # corners), so the 18-neighbourhood — offsets with |dx|+|dy|+|dz| <= 2 —
    # suffices for watertight/manifold output while subdividing fewer cells than
    # the full 26-neighbourhood.
    offsets = np.stack(
        np.meshgrid(*(np.arange(-1, 2),) * 3, indexing="ij"), axis=-1
    ).reshape(-1, 3).astype(np.int64)
    neighbourhood = offsets[np.abs(offsets).sum(axis=1) <= 2]
    dilated = (coarse_cells[:, None, :] + neighbourhood[None, :, :]).reshape(-1, 3)
    dilated = dilated[np.all((dilated >= 0) & (dilated < cdims), axis=1)]
    clid = (dilated[:, 0] * cdims[1] + dilated[:, 1]) * cdims[2] + dilated[:, 2]
    coarse_cells = dilated[np.unique(clid, return_index=True)[1]]

    s = int(subdiv)
    fdims = cdims * s
    child = np.stack(
        np.meshgrid(np.arange(s), np.arange(s), np.arange(s), indexing="ij"),
        axis=-1,
    ).reshape(-1, 3).astype(np.int64)
    fine = (coarse_cells[:, None, :] * s + child[None, :, :]).reshape(-1, 3)
    flid = (fine[:, 0] * fdims[1] + fine[:, 1]) * fdims[2] + fine[:, 2]
    _, keep_idx = np.unique(flid, return_index=True)
    fine_cells = fine[keep_idx]

    # Evaluate the SDF once per unique fine corner point in the band. Dedup by
    # integer linear id (sort of int64) rather than np.unique(axis=0) row-sort.
    corner_pts = fine_cells[:, None, :] + _CORNER_OFFSETS[None, :, :].astype(np.int64)
    flat_pts = corner_pts.reshape(-1, 3)
    py = int(fdims[1]) + 1
    pz = int(fdims[2]) + 1
    pid = (flat_pts[:, 0] * py + flat_pts[:, 1]) * pz + flat_pts[:, 2]
    unique_pid, inverse = np.unique(pid, return_inverse=True)
    upk = unique_pid % pz
    upj = (unique_pid // pz) % py
    upi = unique_pid // (py * pz)
    step_f = (maxs - mins) / fdims.astype(np.float64)
    world = mins + np.column_stack((upi, upj, upk)).astype(np.float64) * step_f
    field = np.asarray(
        node.to_numpy(world[:, 0], world[:, 1], world[:, 2]), dtype=np.float64
    )
    corner_values = field[inverse].reshape(-1, 8)

    value_min = corner_values.min(axis=1)
    value_max = corner_values.max(axis=1)
    crossing = (value_min <= 0.0) & (value_max >= 0.0) & ((value_max - value_min) > 1.0e-12)
    cells = fine_cells[crossing]
    if cells.shape[0] == 0:
        return _empty_surface(node, key, color, "dual contour produced no cells")
    cell_corner_values = corner_values[crossing]
    bases = mins + cells.astype(np.float64) * step_f
    vertices, normals = _solve_cells_hermite_qef(
        node, bases, step_f, cell_corner_values
    )

    clid = (cells[:, 0] * fdims[1] + cells[:, 1]) * fdims[2] + cells[:, 2]
    order = np.argsort(clid)
    sorted_lid = clid[order]
    sorted_vid = order.astype(np.int64)
    parts: list[NDArray[np.uint32]] = []
    for corner_a, corner_b, offsets, flip_positive in _DC_EDGE_SPECS:
        fa = cell_corner_values[:, corner_a]
        fb = cell_corner_values[:, corner_b]
        flip = (cell_corner_values[:, 0] > 0.0) if flip_positive else (
            cell_corner_values[:, 0] <= 0.0
        )
        vids = [
            _cell_vid_lookup(
                cells + np.asarray(offset, dtype=np.int64),
                fdims,
                sorted_lid,
                sorted_vid,
            )
            for offset in offsets
        ]
        parts.append(
            _quad_triangles(vids[0], vids[1], vids[2], vids[3], fa, fb, flip)
        )
    parts = [part for part in parts if part.size]
    index_array = (
        np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint32)
    )

    vertex_array = vertices.astype(np.float32)
    normal_array = _analytic_vertex_normals(
        node, vertex_array, step_f, normals.astype(np.float32)
    )
    index_array = _orient_triangles(vertex_array, normal_array, index_array)
    triangle_count = int(index_array.size // 3)
    wire = (
        _wire_indices_from_triangles(index_array)
        if triangle_count <= _MAX_DUAL_CONTOUR_WIREFRAME_TRIANGLES
        else np.zeros(0, dtype=np.uint32)
    )
    status: SurfaceStatus = "ready" if index_array.size else "empty"
    return ViewportSurface(
        key=key,
        object_kind=type(node).__name__,
        status=status,
        vertices=vertex_array,
        normals=normal_array,
        indices=index_array,
        wire_indices=wire,
        color=color,
        bounds_min=tuple(float(value) for value in mins),
        bounds_max=tuple(float(value) for value in maxs),
        message="" if index_array.size else "dual contour produced no faces",
    )


def _analytic_gradient(
    node: SDFNode,
    points: NDArray[np.float64],
    eps: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Central-difference gradient of the analytic SDF at scattered points.

    ``points`` is (N, 3); ``eps`` a per-axis step (3,). The six axis samples are
    batched into one ``to_numpy`` call (a single tree walk) to amortise cost.
    Returns the raw (unnormalised) gradient (N, 3).
    """
    offsets = np.asarray(
        (
            (eps[0], 0.0, 0.0),
            (-eps[0], 0.0, 0.0),
            (0.0, eps[1], 0.0),
            (0.0, -eps[1], 0.0),
            (0.0, 0.0, eps[2]),
            (0.0, 0.0, -eps[2]),
        ),
        dtype=np.float64,
    )
    samples = points[None, :, :] + offsets[:, None, :]
    flat = samples.reshape(-1, 3)
    field = np.asarray(
        node.to_numpy(flat[:, 0], flat[:, 1], flat[:, 2]),
        dtype=np.float64,
    ).reshape(6, -1)
    return np.column_stack(
        (
            (field[0] - field[1]) / (2.0 * eps[0]),
            (field[2] - field[3]) / (2.0 * eps[1]),
            (field[4] - field[5]) / (2.0 * eps[2]),
        )
    )


def _refine_edge_hermite(
    node: SDFNode,
    point_a: NDArray[np.float64],
    point_b: NDArray[np.float64],
    fa: NDArray[np.float64],
    fb: NDArray[np.float64],
    eps: NDArray[np.float64],
    iterations: int = 4,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Exact Hermite data for a batch of sign-crossing edges.

    Finds the analytic zero of ``node`` along each edge with the Illinois variant
    of regula falsi — one ``to_numpy`` eval per iteration, always bracketed by the
    [a, b] sign change, with superlinear convergence even on curved fields — then
    samples the exact analytic gradient at the root. Returns (points (N,3), unit
    normals (N,3)) accurate to SDF tolerance independent of grid resolution.
    """
    n = point_a.shape[0]
    if n == 0:
        return point_a.copy(), point_a.copy()
    direction = point_b - point_a
    lo = np.zeros(n, dtype=np.float64)
    hi = np.ones(n, dtype=np.float64)
    f_lo = fa.astype(np.float64).copy()
    f_hi = fb.astype(np.float64).copy()
    t = lo.copy()
    kept_lo_last = np.zeros(n, dtype=np.bool_)
    kept_hi_last = np.zeros(n, dtype=np.bool_)
    for _ in range(iterations):
        denom = f_hi - f_lo
        usable = np.abs(denom) > 1.0e-300
        denom_safe = np.where(usable, denom, 1.0)
        # False position; fall back to the bisection midpoint when the bracket
        # values collapse. Always stays inside [lo, hi].
        t = np.where(
            usable,
            lo - f_lo * (hi - lo) / denom_safe,
            0.5 * (lo + hi),
        )
        point = point_a + t[:, None] * direction
        f = np.asarray(
            node.to_numpy(point[:, 0], point[:, 1], point[:, 2]),
            dtype=np.float64,
        )
        # Keep the half of the bracket that still straddles the sign change.
        keep_lo = (f >= 0.0) == (f_hi >= 0.0)
        hi = np.where(keep_lo, t, hi)
        f_hi = np.where(keep_lo, f, f_hi)
        lo = np.where(keep_lo, lo, t)
        f_lo = np.where(keep_lo, f_lo, f)
        # Illinois: when the same endpoint is retained twice, halve the stale
        # endpoint's value to break one-sided stalling -> superlinear rate.
        halve_lo = keep_lo & kept_lo_last
        halve_hi = (~keep_lo) & kept_hi_last
        f_lo = np.where(halve_lo, f_lo * 0.5, f_lo)
        f_hi = np.where(halve_hi, f_hi * 0.5, f_hi)
        kept_lo_last = keep_lo
        kept_hi_last = ~keep_lo
    point = point_a + t[:, None] * direction
    grad = _analytic_gradient(node, point, eps)
    return point, grad


def _orient_triangles(
    vertices: NDArray[np.float32],
    normals: NDArray[np.float32],
    indices: NDArray[np.uint32],
) -> NDArray[np.uint32]:
    """Make winding consistent and drop degenerate triangles.

    Dual contouring can wind a few triangles backwards at concave seams (worst on
    union); with no backface culling those overlap their neighbours and z-fight,
    reading as a torn seam. The analytic per-vertex normals are a reliable
    orientation reference: flip any triangle whose geometric normal opposes its
    averaged vertex normal, and remove zero-area triangles. Vectorised and
    topology-preserving (only winding / removal), so the mesh stays watertight.
    """
    if indices.size == 0:
        return indices
    idx = indices.reshape(-1, 3)
    tri = vertices[idx].astype(np.float64)
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    keep = np.linalg.norm(face, axis=1) > 1.0e-14
    idx = idx[keep]
    face = face[keep]
    vnorm = normals[idx].astype(np.float64).mean(axis=1)
    flip = np.einsum("ij,ij->i", face, vnorm) < 0.0
    idx = idx.copy()
    idx[flip] = idx[flip][:, [0, 2, 1]]
    return idx.reshape(-1).astype(np.uint32)


def _analytic_vertex_normals(
    node: SDFNode,
    vertices: NDArray[np.float32],
    step: NDArray[np.float64],
    fallback: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Shading normals from the analytic SDF gradient at each vertex.

    Degenerate (near-zero) gradients fall back to the supplied normal so seams
    never produce black/undefined shading.
    """
    if vertices.shape[0] == 0:
        return fallback
    eps = np.maximum(step * 0.25, 1.0e-5)
    try:
        grad = _analytic_gradient(node, vertices.astype(np.float64), eps)
    except Exception:  # noqa: BLE001 - analytic eval optional; keep grid normals
        return fallback
    lengths = np.linalg.norm(grad, axis=1)
    valid = lengths > 1.0e-9
    out = fallback.astype(np.float64).copy()
    out[valid] = grad[valid] / lengths[valid, None]
    return out.astype(np.float32)


def _cell_corner_stack(
    values: NDArray[np.float64],
    dims: NDArray[np.int32],
) -> NDArray[np.float64]:
    nx, ny, nz = (int(value) for value in dims)
    return np.stack(
        [
            values[dx : dx + nx, dy : dy + ny, dz : dz + nz]
            for dx, dy, dz in _CORNER_OFFSETS
        ],
        axis=-1,
    )


def _solve_cells_hermite_qef(
    node: SDFNode,
    bases: NDArray[np.float64],
    step: NDArray[np.float64],
    corner_values: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Place one DC vertex + normal per cell from exact Hermite data.

    ``bases`` (M,3) are cell-origin world coords, ``step`` (3,) the uniform cell
    size, ``corner_values`` (M,8) the field at each cell's 8 corners in
    ``_CORNER_OFFSETS`` order. Shared by the dense and narrow-band paths. Phase A
    (exact edge roots) + Phase B (sharp-feature QEF). Returns (vertices (M,3),
    normals (M,3)).
    """
    edge_a = _CELL_EDGE_A
    edge_b = _CELL_EDGE_B
    fa = corner_values[:, edge_a]
    fb = corner_values[:, edge_b]
    delta = fa - fb
    edge_crossing = (
        (((fa <= 0.0) & (fb >= 0.0)) | ((fb <= 0.0) & (fa >= 0.0)))
        & (np.abs(delta) > 1.0e-12)
    )

    point_a = bases[:, None, :] + _CORNER_OFFSETS_F64[edge_a][None, :, :] * step
    point_b = bases[:, None, :] + _CORNER_OFFSETS_F64[edge_b][None, :, :] * step

    # Phase A: exact Hermite data. Root-find the analytic zero along every
    # crossing edge and sample the exact gradient there, instead of linearly
    # interpolating grid corner values/gradients.
    edge_points = 0.5 * (point_a + point_b)
    edge_normals = np.zeros_like(point_a)
    normal_valid = np.zeros(edge_crossing.shape, dtype=np.bool_)
    grad_eps = np.maximum(step * 0.05, 1.0e-6)
    pa_flat = point_a[edge_crossing]
    if pa_flat.shape[0] > 0:
        pb_flat = point_b[edge_crossing]
        refined_pts, refined_grad = _refine_edge_hermite(
            node,
            pa_flat,
            pb_flat,
            fa[edge_crossing],
            fb[edge_crossing],
            grad_eps,
        )
        grad_len = np.linalg.norm(refined_grad, axis=1)
        good = grad_len > 1.0e-12
        refined_unit = np.zeros_like(refined_grad)
        refined_unit[good] = refined_grad[good] / grad_len[good, None]
        edge_points[edge_crossing] = refined_pts
        edge_normals[edge_crossing] = refined_unit
        normal_valid[edge_crossing] = good

    point_counts = edge_crossing.sum(axis=1)
    average_points = np.divide(
        (edge_points * edge_crossing[:, :, None]).sum(axis=1),
        point_counts[:, None],
        out=bases + step * 0.5,
        where=point_counts[:, None] > 0,
    )

    # Phase B: sharp-feature QEF (Lindstrom/Schaefer). Solve for the vertex as
    # the mass-point plus the minimum-norm correction in the feature subspace,
    # using a truncated eigendecomposition of A^T A. Singular directions (flat
    # faces -> rank 1, seams -> rank 2, corners -> rank 3) are dropped, so the
    # solver places the EXACT sharp edge/corner instead of an averaged point.
    qef_normals = edge_normals * normal_valid[:, :, None]
    ata = np.einsum("mei,mej->mij", qef_normals, qef_normals)
    ndotp = (qef_normals * edge_points).sum(axis=2)
    atb = np.einsum("mei,me->mi", qef_normals, ndotp)
    qef_counts = normal_valid.sum(axis=1)
    mass_point = average_points
    # A^T b' with the system recentred at the mass point: b'_i = n_i.(p_i - c).
    rhs = atb - np.einsum("mij,mj->mi", ata, mass_point)
    eigvals, eigvecs = np.linalg.eigh(ata)
    eig_max = np.maximum(eigvals[:, -1:], 1.0e-30)
    keep = eigvals > (_QEF_SINGULAR_RATIO * eig_max)
    inv_eig = np.where(keep, 1.0 / np.where(keep, eigvals, 1.0), 0.0)
    pinv = np.einsum("mik,mk,mjk->mij", eigvecs, inv_eig, eigvecs)
    candidates = mass_point + np.einsum("mij,mj->mi", pinv, rhs)
    finite = np.isfinite(candidates).all(axis=1)
    candidates = np.where(finite[:, None], candidates, mass_point)
    cell_min = bases
    cell_max = bases + step
    # Clamp to the cell so a degenerate solve cannot spike outside it.
    candidates = np.minimum(np.maximum(candidates, cell_min), cell_max)
    vertices = np.where((qef_counts > 0)[:, None], candidates, mass_point)
    normals = _normalize_rows((edge_normals * normal_valid[:, :, None]).sum(axis=1))
    return vertices, normals


def _dual_contour_cell_vertices(
    node: SDFNode,
    values: NDArray[np.float64],
    xs: NDArray[np.float64],
    ys: NDArray[np.float64],
    zs: NDArray[np.float64],
    dims: NDArray[np.int32],
) -> tuple[NDArray[np.int32], NDArray[np.float32], NDArray[np.float32]]:
    cell_shape = tuple(int(value) for value in dims)
    cell_vertex = np.full(cell_shape, -1, dtype=np.int32)
    corner_values = _cell_corner_stack(values, dims)
    value_min = corner_values.min(axis=-1)
    value_max = corner_values.max(axis=-1)
    crossing = (
        (value_min <= 0.0)
        & (value_max >= 0.0)
        & ((value_max - value_min) > 1.0e-12)
    )
    if not np.any(crossing):
        empty = np.zeros((0, 3), dtype=np.float32)
        return cell_vertex, empty, empty

    cells = np.argwhere(crossing)
    corner_values = corner_values[crossing]
    bases = np.column_stack(
        (xs[cells[:, 0]], ys[cells[:, 1]], zs[cells[:, 2]])
    )
    step = np.asarray((xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]), dtype=np.float64)
    vertices, normals = _solve_cells_hermite_qef(node, bases, step, corner_values)
    cell_vertex[tuple(cells.T)] = np.arange(cells.shape[0], dtype=np.int32)
    return (
        cell_vertex,
        np.asarray(vertices, dtype=np.float32),
        np.asarray(normals, dtype=np.float32),
    )


def _placed_2d_outline(
    node: PlacedSDF2D,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface:
    assert node.profile is not None
    ordered = _ordered_placed_2d_surface(node, key, color)
    if ordered is not None:
        return ordered
    contoured = _contoured_placed_2d_surface(node, key, color)
    if contoured is not None:
        return contoured
    return _sampled_placed_2d_surface(node, key, color)


def _contoured_placed_2d_surface(
    node: PlacedSDF2D,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface | None:
    assert node.profile is not None
    u_min, u_max, v_min, v_max = node.profile.bounds()
    span = max(u_max - u_min, v_max - v_min, 1.0e-6)
    pad = span * 0.025
    u_min -= pad
    u_max += pad
    v_min -= pad
    v_max += pad
    resolution = max(64, min(_MAX_CONTOURED_2D_CELLS, key.resolution * 5))
    us = np.linspace(u_min, u_max, resolution + 1, dtype=np.float64)
    vs = np.linspace(v_min, v_max, resolution + 1, dtype=np.float64)
    ug, vg = np.meshgrid(us, vs, indexing="ij")
    values = np.asarray(node.profile.to_numpy(ug, vg), dtype=np.float64)
    local_rings = _marching_squares_rings(values, us, vs)
    if not local_rings:
        return None

    cleaned_rings: list[list[tuple[float, float]]] = []
    for ring in local_rings:
        cleaned = _clean_polygon_ring(ring)
        if len(cleaned) < 3:
            continue
        if _signed_area_2d(cleaned) < 0.0:
            cleaned = list(reversed(cleaned))
        cleaned_rings.append(cleaned)
    if not cleaned_rings or _contour_rings_have_holes(cleaned_rings):
        return None

    local_vertices: list[tuple[float, float]] = []
    indices: list[int] = []
    wire: list[int] = []
    for cleaned in cleaned_rings:
        triangles = _triangulate_simple_polygon(cleaned)
        if not triangles:
            continue
        base = len(local_vertices)
        local_vertices.extend(cleaned)
        indices.extend(base + index for triangle in triangles for index in triangle)
        count = len(cleaned)
        for index in range(count):
            wire.extend((base + index, base + ((index + 1) % count)))

    if not local_vertices or not indices:
        return None

    origin = np.asarray(node.origin, dtype=np.float64)
    axis_u = np.asarray(node.axis_u, dtype=np.float64)
    axis_v = np.asarray(node.axis_v, dtype=np.float64)
    normal = np.asarray(node.normal, dtype=np.float64)
    local = np.asarray(local_vertices, dtype=np.float64)
    vertices = np.asarray(
        origin + local[:, :1] * axis_u + local[:, 1:] * axis_v,
        dtype=np.float32,
    )
    index_array = np.asarray(indices, dtype=np.uint32)
    normals = np.broadcast_to(normal.astype(np.float32), vertices.shape).copy()
    return ViewportSurface(
        key=key,
        object_kind=type(node).__name__,
        status="ready",
        vertices=vertices,
        normals=normals,
        indices=index_array,
        wire_indices=np.asarray(wire, dtype=np.uint32),
        color=color,
        bounds_min=tuple(float(value) for value in vertices.min(axis=0)),
        bounds_max=tuple(float(value) for value in vertices.max(axis=0)),
        message="2D profile rendered as contoured filled surface",
    )


def _sampled_placed_2d_surface(
    node: PlacedSDF2D,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface:
    assert node.profile is not None
    u_min, u_max, v_min, v_max = node.profile.bounds()
    span = max(u_max - u_min, v_max - v_min, 1.0e-6)
    pad = span * 0.025
    u_min -= pad
    u_max += pad
    v_min -= pad
    v_max += pad
    resolution = max(24, min(_MAX_SAMPLED_2D_CELLS, key.resolution * 3))
    us = np.linspace(u_min, u_max, resolution + 1, dtype=np.float64)
    vs = np.linspace(v_min, v_max, resolution + 1, dtype=np.float64)
    mid_u = (us[:-1] + us[1:]) * 0.5
    mid_v = (vs[:-1] + vs[1:]) * 0.5
    ug, vg = np.meshgrid(mid_u, mid_v, indexing="ij")
    inside = np.asarray(node.profile.to_numpy(ug, vg) <= 0.0, dtype=np.bool_)
    filled_cells = np.argwhere(inside)
    if filled_cells.size == 0:
        return _empty_surface(node, key, color, "2D profile produced no filled cells")

    origin = np.asarray(node.origin, dtype=np.float64)
    axis_u = np.asarray(node.axis_u, dtype=np.float64)
    axis_v = np.asarray(node.axis_v, dtype=np.float64)
    normal = np.asarray(node.normal, dtype=np.float64)
    uu, vv = np.meshgrid(us, vs, indexing="ij")
    local = np.column_stack((uu.reshape(-1), vv.reshape(-1)))
    vertex_count_per_axis = resolution + 1
    world = np.asarray(
        origin + local[:, :1] * axis_u + local[:, 1:] * axis_v,
        dtype=np.float32,
    )
    cell_base = filled_cells[:, 0] * vertex_count_per_axis + filled_cells[:, 1]
    a = cell_base
    b = cell_base + vertex_count_per_axis
    c = cell_base + vertex_count_per_axis + 1
    d = cell_base + 1
    raw_indices = np.column_stack((a, b, c, a, c, d)).reshape(-1)
    used_vertices, inverse = np.unique(raw_indices, return_inverse=True)
    vertices = world[used_vertices]
    indices = inverse.astype(np.uint32)
    wire = _sampled_2d_boundary_wire(inside, used_vertices, vertex_count_per_axis)
    normals = np.broadcast_to(normal.astype(np.float32), vertices.shape).copy()
    return ViewportSurface(
        key=key,
        object_kind=type(node).__name__,
        status="ready",
        vertices=vertices,
        normals=normals,
        indices=indices,
        wire_indices=wire,
        color=color,
        bounds_min=tuple(float(value) for value in vertices.min(axis=0)),
        bounds_max=tuple(float(value) for value in vertices.max(axis=0)),
        message="2D profile rendered as sampled filled surface",
    )


def _sampled_2d_boundary_wire(
    inside: NDArray[np.bool_],
    used_vertices: NDArray[np.int64],
    vertex_count_per_axis: int,
) -> NDArray[np.uint32]:
    remap = {
        int(original): index
        for index, original in enumerate(np.asarray(used_vertices, dtype=np.int64))
    }
    resolution = int(inside.shape[0])
    wire: list[int] = []

    def append_edge(first: int, second: int) -> None:
        mapped_first = remap.get(int(first))
        mapped_second = remap.get(int(second))
        if mapped_first is None or mapped_second is None:
            return
        wire.extend((mapped_first, mapped_second))

    for i, j in np.argwhere(inside):
        base = int(i) * vertex_count_per_axis + int(j)
        left = int(i) == 0 or not bool(inside[int(i) - 1, int(j)])
        right = int(i) == resolution - 1 or not bool(inside[int(i) + 1, int(j)])
        bottom = int(j) == 0 or not bool(inside[int(i), int(j) - 1])
        top = int(j) == resolution - 1 or not bool(inside[int(i), int(j) + 1])
        if bottom:
            append_edge(base, base + vertex_count_per_axis)
        if right:
            append_edge(
                base + vertex_count_per_axis,
                base + vertex_count_per_axis + 1,
            )
        if top:
            append_edge(base + vertex_count_per_axis + 1, base + 1)
        if left:
            append_edge(base + 1, base)
    return np.asarray(wire, dtype=np.uint32)


def _marching_squares_rings(
    values: NDArray[np.float64],
    us: NDArray[np.float64],
    vs: NDArray[np.float64],
) -> list[list[tuple[float, float]]]:
    resolution = int(len(us) - 1)
    horizontal = np.full((resolution, resolution + 1), -1, dtype=np.int32)
    vertical = np.full((resolution + 1, resolution), -1, dtype=np.int32)
    vertices: list[tuple[float, float]] = []

    def add_vertex(u: float, v: float) -> int:
        vertices.append((float(u), float(v)))
        return len(vertices) - 1

    for i in range(resolution):
        for j in range(resolution + 1):
            first = float(values[i, j])
            second = float(values[i + 1, j])
            if not _edge_has_crossing(first, second):
                continue
            t = first / (first - second)
            horizontal[i, j] = add_vertex(
                float(us[i] + np.clip(t, 0.0, 1.0) * (us[i + 1] - us[i])),
                float(vs[j]),
            )

    for i in range(resolution + 1):
        for j in range(resolution):
            first = float(values[i, j])
            second = float(values[i, j + 1])
            if not _edge_has_crossing(first, second):
                continue
            t = first / (first - second)
            vertical[i, j] = add_vertex(
                float(us[i]),
                float(vs[j] + np.clip(t, 0.0, 1.0) * (vs[j + 1] - vs[j])),
            )

    segments: list[tuple[int, int]] = []
    for i in range(resolution):
        for j in range(resolution):
            mask = 0
            if float(values[i, j]) <= 0.0:
                mask |= 1
            if float(values[i + 1, j]) <= 0.0:
                mask |= 2
            if float(values[i + 1, j + 1]) <= 0.0:
                mask |= 4
            if float(values[i, j + 1]) <= 0.0:
                mask |= 8
            if mask == 0 or mask == 15:
                continue
            edge_vertices = (
                int(horizontal[i, j]),
                int(vertical[i + 1, j]),
                int(horizontal[i, j + 1]),
                int(vertical[i, j]),
            )
            for first, second in _marching_square_pairs(mask, values, i, j):
                first_vertex = edge_vertices[first]
                second_vertex = edge_vertices[second]
                if first_vertex >= 0 and second_vertex >= 0:
                    segments.append((first_vertex, second_vertex))

    return _stitch_contour_rings(vertices, segments)


def _marching_square_pairs(
    mask: int,
    values: NDArray[np.float64],
    i: int,
    j: int,
) -> tuple[tuple[int, int], ...]:
    table: dict[int, tuple[tuple[int, int], ...]] = {
        1: ((3, 0),),
        2: ((0, 1),),
        3: ((3, 1),),
        4: ((1, 2),),
        6: ((0, 2),),
        7: ((3, 2),),
        8: ((2, 3),),
        9: ((0, 2),),
        11: ((1, 2),),
        12: ((1, 3),),
        13: ((0, 1),),
        14: ((3, 0),),
    }
    pairs = table.get(mask)
    if pairs is not None:
        return pairs
    center_inside = float(values[i : i + 2, j : j + 2].mean()) <= 0.0
    if mask == 5:
        return ((0, 1), (2, 3)) if center_inside else ((3, 0), (1, 2))
    if mask == 10:
        return ((0, 3), (1, 2)) if center_inside else ((0, 1), (2, 3))
    return ()


def _stitch_contour_rings(
    vertices: list[tuple[float, float]],
    segments: list[tuple[int, int]],
) -> list[list[tuple[float, float]]]:
    adjacency: dict[int, list[int]] = {}
    unused: set[tuple[int, int]] = set()
    for first, second in segments:
        if first == second:
            continue
        edge = (min(first, second), max(first, second))
        if edge in unused:
            continue
        unused.add(edge)
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)

    rings: list[list[tuple[float, float]]] = []
    while unused:
        first, second = next(iter(unused))
        unused.remove((first, second))
        ring = [first, second]
        previous = first
        current = second
        closed = False
        for _ in range(len(segments) + 1):
            candidates = [
                candidate
                for candidate in adjacency.get(current, ())
                if (min(current, candidate), max(current, candidate)) in unused
            ]
            if not candidates:
                closed = ring[0] in adjacency.get(current, ())
                break
            next_vertex = candidates[0]
            if len(candidates) > 1:
                next_vertex = next(
                    (candidate for candidate in candidates if candidate != previous),
                    candidates[0],
                )
            unused.remove((min(current, next_vertex), max(current, next_vertex)))
            if next_vertex == ring[0]:
                closed = True
                break
            ring.append(next_vertex)
            previous, current = current, next_vertex
        if closed and len(ring) >= 3:
            rings.append([vertices[index] for index in ring])
    rings.sort(key=lambda ring: abs(_signed_area_2d(ring)), reverse=True)
    return rings


def _clean_polygon_ring(
    ring: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    if len(ring) < 3:
        return ring
    cleaned: list[tuple[float, float]] = []
    for point in ring:
        if cleaned and _points_close_2d(cleaned[-1], point):
            continue
        cleaned.append(point)
    if len(cleaned) > 1 and _points_close_2d(cleaned[0], cleaned[-1]):
        cleaned.pop()
    changed = True
    while changed and len(cleaned) >= 3:
        changed = False
        reduced: list[tuple[float, float]] = []
        count = len(cleaned)
        for index, point in enumerate(cleaned):
            previous = cleaned[(index - 1) % count]
            next_point = cleaned[(index + 1) % count]
            if abs(_cross_2d(previous, point, next_point)) <= 1.0e-12:
                changed = True
                continue
            reduced.append(point)
        cleaned = reduced
    return cleaned


def _contour_rings_have_holes(
    rings: list[list[tuple[float, float]]],
) -> bool:
    for index, ring in enumerate(rings):
        probe = ring[0]
        depth = 0
        for other_index, other in enumerate(rings):
            if index == other_index:
                continue
            if _point_in_polygon_2d(probe, other):
                depth += 1
        if depth % 2 == 1:
            return True
    return False


def _point_in_polygon_2d(
    point: tuple[float, float],
    polygon: list[tuple[float, float]],
) -> bool:
    inside = False
    x, y = point
    previous_x, previous_y = polygon[-1]
    for current_x, current_y in polygon:
        if (current_y > y) != (previous_y > y):
            slope = (previous_x - current_x) / (previous_y - current_y)
            crossing_x = current_x + (y - current_y) * slope
            if x < crossing_x:
                inside = not inside
        previous_x, previous_y = current_x, current_y
    return inside


def _points_close_2d(
    first: tuple[float, float],
    second: tuple[float, float],
) -> bool:
    return bool(
        abs(first[0] - second[0]) <= 1.0e-12
        and abs(first[1] - second[1]) <= 1.0e-12
    )


def _signed_area_2d(points: list[tuple[float, float]]) -> float:
    area = 0.0
    for index, point in enumerate(points):
        next_point = points[(index + 1) % len(points)]
        area += point[0] * next_point[1] - next_point[0] * point[1]
    return area * 0.5


def _cross_2d(
    previous: tuple[float, float],
    point: tuple[float, float],
    next_point: tuple[float, float],
) -> float:
    return (
        (point[0] - previous[0]) * (next_point[1] - point[1])
        - (point[1] - previous[1]) * (next_point[0] - point[0])
    )


def _triangulate_simple_polygon(
    points: list[tuple[float, float]],
) -> list[tuple[int, int, int]]:
    if len(points) < 3:
        return []
    polygon = points
    if _signed_area_2d(polygon) < 0.0:
        polygon = list(reversed(polygon))
    if _polygon_is_convex(polygon):
        return [(0, index, index + 1) for index in range(1, len(polygon) - 1)]
    remaining = list(range(len(polygon)))
    triangles: list[tuple[int, int, int]] = []
    guard = len(polygon) * len(polygon)
    while len(remaining) > 3 and guard > 0:
        guard -= 1
        clipped = False
        for position, index in enumerate(tuple(remaining)):
            previous = remaining[(position - 1) % len(remaining)]
            next_index = remaining[(position + 1) % len(remaining)]
            if _cross_2d(polygon[previous], polygon[index], polygon[next_index]) <= 1.0e-12:
                continue
            if _triangle_contains_any_point(
                polygon,
                previous,
                index,
                next_index,
                remaining,
            ):
                continue
            triangles.append((previous, index, next_index))
            del remaining[position]
            clipped = True
            break
        if clipped:
            continue
        for position, index in enumerate(tuple(remaining)):
            previous = remaining[(position - 1) % len(remaining)]
            next_index = remaining[(position + 1) % len(remaining)]
            if abs(_cross_2d(polygon[previous], polygon[index], polygon[next_index])) <= 1.0e-12:
                del remaining[position]
                clipped = True
                break
        if not clipped:
            return []
    if len(remaining) == 3:
        triangles.append((remaining[0], remaining[1], remaining[2]))
    return triangles


def _polygon_is_convex(points: list[tuple[float, float]]) -> bool:
    if len(points) <= 3:
        return True
    for index, point in enumerate(points):
        previous = points[(index - 1) % len(points)]
        next_point = points[(index + 1) % len(points)]
        if _cross_2d(previous, point, next_point) < -1.0e-12:
            return False
    return True


def _triangle_contains_any_point(
    points: list[tuple[float, float]],
    first: int,
    second: int,
    third: int,
    candidates: list[int],
) -> bool:
    a = points[first]
    b = points[second]
    c = points[third]
    for index in candidates:
        if index in {first, second, third}:
            continue
        if _point_in_triangle_2d(points[index], a, b, c):
            return True
    return False


def _point_in_triangle_2d(
    point: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    ab = _cross_2d(a, b, point)
    bc = _cross_2d(b, c, point)
    ca = _cross_2d(c, a, point)
    return bool(ab >= -1.0e-12 and bc >= -1.0e-12 and ca >= -1.0e-12)


def _ordered_placed_2d_surface(
    node: PlacedSDF2D,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
) -> ViewportSurface | None:
    assert node.profile is not None
    outline = _profile_outline(node.profile, key.resolution)
    if len(outline) < 2:
        return None
    origin = np.asarray(node.origin, dtype=np.float64)
    axis_u = np.asarray(node.axis_u, dtype=np.float64)
    axis_v = np.asarray(node.axis_v, dtype=np.float64)
    normal = np.asarray(node.normal, dtype=np.float64)
    world = np.asarray(
        [origin + u * axis_u + v * axis_v for u, v in outline],
        dtype=np.float32,
    )
    center = np.asarray(world, dtype=np.float64).mean(axis=0).astype(np.float32)
    vertices = np.vstack((world, center))
    normals = np.broadcast_to(normal.astype(np.float32), world.shape).copy()
    wire = np.asarray(
        [
            index
            for current in range(len(outline))
            for index in (current, (current + 1) % len(outline))
        ],
        dtype=np.uint32,
    )
    center_index = len(outline)
    indices = np.asarray(
        [
            index
            for current in range(len(outline))
            for index in (center_index, current, (current + 1) % len(outline))
        ],
        dtype=np.uint32,
    )
    return ViewportSurface(
        key=key,
        object_kind=type(node).__name__,
        status="ready",
        vertices=vertices,
        normals=np.vstack((normals, normal.astype(np.float32))),
        indices=indices,
        wire_indices=wire,
        color=color,
        bounds_min=tuple(float(value) for value in vertices.min(axis=0)),
        bounds_max=tuple(float(value) for value in vertices.max(axis=0)),
        message="2D profile rendered as filled ordered surface",
    )


def _sampling_bounds(
    node: SDFNode,
    resolution: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    box = node.bounding_box()
    mins = np.asarray((box.x_min, box.y_min, box.z_min), dtype=np.float64)
    maxs = np.asarray((box.x_max, box.y_max, box.z_max), dtype=np.float64)
    if not (np.all(np.isfinite(mins)) and np.all(np.isfinite(maxs))):
        raise ValueError("viewport surface requires finite SDF bounds")
    extents = np.maximum(maxs - mins, 0.0)
    base = max(float(extents.max()), 1.0)
    margin = max(base / max(float(resolution), 1.0), base * 0.025, 1.0e-3)
    thin = extents <= 1.0e-9
    mins = mins - margin
    maxs = maxs + margin
    mins[thin] -= margin
    maxs[thin] += margin
    return mins, maxs


def _axis_cell_counts(
    mins: NDArray[np.float64],
    maxs: NDArray[np.float64],
    resolution: int,
) -> NDArray[np.int32]:
    extents = np.maximum(maxs - mins, 1.0e-9)
    scaled = np.ceil(float(resolution) * extents / float(extents.max()))
    return np.asarray(
        np.clip(scaled, _MIN_AXIS_CELLS, _MAX_AXIS_CELLS),
        dtype=np.int32,
    )


def _cell_has_crossing(values: NDArray[np.float64]) -> bool:
    return bool(
        np.any(values <= 0.0)
        and np.any(values >= 0.0)
        and float(values.max() - values.min()) > 1.0e-12
    )


def _edge_has_crossing(first: float, second: float) -> bool:
    return bool(
        (first <= 0.0 <= second or second <= 0.0 <= first)
        and abs(first - second) > 1.0e-12
    )


def _cell_edge_intersections(
    points: NDArray[np.float64],
    values: NDArray[np.float64],
    normals: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    out_points: list[NDArray[np.float64]] = []
    out_normals: list[NDArray[np.float64]] = []
    for a, b in _CELL_EDGE_CORNERS:
        fa = float(values[a])
        fb = float(values[b])
        if not _edge_has_crossing(fa, fb):
            continue
        t = fa / (fa - fb) if abs(fa - fb) > 1.0e-12 else 0.5
        t = float(np.clip(t, 0.0, 1.0))
        out_points.append(points[a] + t * (points[b] - points[a]))
        out_normals.append(_normalize(normals[a] + t * (normals[b] - normals[a])))
    if not out_points:
        empty = np.zeros((0, 3), dtype=np.float64)
        return empty, empty
    return (
        np.asarray(out_points, dtype=np.float64),
        np.asarray(out_normals, dtype=np.float64),
    )


def _solve_qef(
    points: NDArray[np.float64],
    normals: NDArray[np.float64],
    cell_min: NDArray[np.float64],
    cell_max: NDArray[np.float64],
) -> NDArray[np.float64]:
    average = points.mean(axis=0)
    try:
        b = np.einsum("ij,ij->i", normals, points)
        candidate = np.linalg.lstsq(normals, b, rcond=None)[0]
    except np.linalg.LinAlgError:
        return average
    if not np.all(np.isfinite(candidate)):
        return average
    if np.any(candidate < cell_min) or np.any(candidate > cell_max):
        return np.minimum(np.maximum(candidate, cell_min), cell_max)
    return np.asarray(candidate, dtype=np.float64)


def _quad_triangles(
    c0: NDArray[np.int32],
    c1: NDArray[np.int32],
    c2: NDArray[np.int32],
    c3: NDArray[np.int32],
    fa: NDArray[np.float64],
    fb: NDArray[np.float64],
    flip: NDArray[np.bool_],
) -> NDArray[np.uint32]:
    """Vectorised dual-contour quad -> two triangles for one edge axis.

    ``cN`` are the cell-vertex indices of the four cells sharing each edge;
    ``fa``/``fb`` the edge endpoint field values; ``flip`` the winding selector.
    Mirrors the scalar ``append_quad`` (no-flip ``a,b,c / a,c,d``; flip
    ``a,c,b / a,d,c``) but emits the whole grid in one pass.
    """
    cross = (((fa <= 0.0) & (fb >= 0.0)) | ((fb <= 0.0) & (fa >= 0.0))) & (
        np.abs(fa - fb) > 1.0e-12
    )
    a = c0[cross]
    b = c1[cross]
    c = c2[cross]
    d = c3[cross]
    f = flip[cross]
    # All four cells around an edge are distinct by construction; only drop
    # quads where a corner cell has no emitted vertex (boundary band).
    keep = (a >= 0) & (b >= 0) & (c >= 0) & (d >= 0)
    a, b, c, d, f = a[keep], b[keep], c[keep], d[keep], f[keep]
    if a.size == 0:
        return np.zeros(0, dtype=np.uint32)
    tri = np.empty((a.size, 6), dtype=np.uint32)
    tri[:, 0] = a
    tri[:, 1] = np.where(f, c, b)
    tri[:, 2] = np.where(f, b, c)
    tri[:, 3] = a
    tri[:, 4] = np.where(f, d, c)
    tri[:, 5] = np.where(f, c, d)
    return tri.reshape(-1)


def _dual_contour_indices(
    values: NDArray[np.float64],
    cell_vertex: NDArray[np.int32],
) -> NDArray[np.uint32]:
    nx, ny, nz = cell_vertex.shape
    parts: list[NDArray[np.uint32]] = []

    # X-parallel edges: shared by four cells in the (y, z) plane.
    if nx >= 1 and ny >= 2 and nz >= 2:
        fa = values[0:nx, 1:ny, 1:nz]
        fb = values[1 : nx + 1, 1:ny, 1:nz]
        parts.append(
            _quad_triangles(
                cell_vertex[0:nx, 0 : ny - 1, 0 : nz - 1],
                cell_vertex[0:nx, 1:ny, 0 : nz - 1],
                cell_vertex[0:nx, 1:ny, 1:nz],
                cell_vertex[0:nx, 0 : ny - 1, 1:nz],
                fa,
                fb,
                fa > 0.0,
            )
        )

    # Y-parallel edges: shared by four cells in the (x, z) plane.
    if nx >= 2 and ny >= 1 and nz >= 2:
        fa = values[1:nx, 0:ny, 1:nz]
        fb = values[1:nx, 1 : ny + 1, 1:nz]
        parts.append(
            _quad_triangles(
                cell_vertex[0 : nx - 1, 0:ny, 0 : nz - 1],
                cell_vertex[1:nx, 0:ny, 0 : nz - 1],
                cell_vertex[1:nx, 0:ny, 1:nz],
                cell_vertex[0 : nx - 1, 0:ny, 1:nz],
                fa,
                fb,
                fa <= 0.0,
            )
        )

    # Z-parallel edges: shared by four cells in the (x, y) plane.
    if nx >= 2 and ny >= 2 and nz >= 1:
        fa = values[1:nx, 1:ny, 0:nz]
        fb = values[1:nx, 1:ny, 1 : nz + 1]
        parts.append(
            _quad_triangles(
                cell_vertex[0 : nx - 1, 0 : ny - 1, 0:nz],
                cell_vertex[1:nx, 0 : ny - 1, 0:nz],
                cell_vertex[1:nx, 1:ny, 0:nz],
                cell_vertex[0 : nx - 1, 1:ny, 0:nz],
                fa,
                fb,
                fa > 0.0,
            )
        )

    parts = [part for part in parts if part.size]
    if not parts:
        return np.zeros(0, dtype=np.uint32)
    return np.concatenate(parts)


def _wire_indices_from_triangles(indices: NDArray[np.uint32]) -> NDArray[np.uint32]:
    if indices.size == 0:
        return np.zeros(0, dtype=np.uint32)
    edges: set[tuple[int, int]] = set()
    for a, b, c in indices.reshape(-1, 3):
        ia = int(a)
        ib = int(b)
        ic = int(c)
        edges.add(tuple(sorted((ia, ib))))
        edges.add(tuple(sorted((ib, ic))))
        edges.add(tuple(sorted((ic, ia))))
    return np.asarray(
        [index for edge in sorted(edges) for index in edge],
        dtype=np.uint32,
    )


def _mesh_normals(
    vertices: NDArray[np.float32],
    indices: NDArray[np.uint32],
) -> NDArray[np.float32]:
    normals = np.zeros(vertices.shape, dtype=np.float64)
    if indices.size:
        vertex64 = np.asarray(vertices, dtype=np.float64)
        triangles = np.asarray(indices.reshape(-1, 3), dtype=np.int64)
        face = np.cross(
            vertex64[triangles[:, 1]] - vertex64[triangles[:, 0]],
            vertex64[triangles[:, 2]] - vertex64[triangles[:, 0]],
        )
        lengths = np.linalg.norm(face, axis=1)
        valid = lengths > 1.0e-12
        if np.any(valid):
            face = face[valid] / lengths[valid, None]
            triangles = triangles[valid]
            np.add.at(normals, triangles[:, 0], face)
            np.add.at(normals, triangles[:, 1], face)
            np.add.at(normals, triangles[:, 2], face)
    lengths = np.linalg.norm(normals, axis=1)
    fallback = lengths <= 1.0e-12
    lengths[fallback] = 1.0
    normals = normals / lengths[:, None]
    normals[fallback] = (0.0, 0.0, 1.0)
    return np.asarray(normals, dtype=np.float32)


def _normalize(vector: NDArray[np.float64]) -> NDArray[np.float64]:
    length = float(np.linalg.norm(vector))
    if length <= 1.0e-12 or not math.isfinite(length):
        return np.asarray((0.0, 0.0, 1.0), dtype=np.float64)
    return np.asarray(vector / length, dtype=np.float64)


def _normalize_rows(vectors: NDArray[np.float64]) -> NDArray[np.float64]:
    lengths = np.linalg.norm(vectors, axis=1)
    out = np.divide(
        vectors,
        lengths[:, None],
        out=np.zeros_like(vectors),
        where=lengths[:, None] > 1.0e-12,
    )
    fallback = lengths <= 1.0e-12
    if np.any(fallback):
        out[fallback] = (0.0, 0.0, 1.0)
    return out


def _object_color(object_id: int) -> tuple[float, float, float]:
    value = (int(object_id) * 2_654_435_761) & 0xFFFFFFFF
    hue = (value % 360) / 360.0
    saturation = 0.48
    value_luma = 0.92
    return _hsv_to_rgb(hue, saturation, value_luma)


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[float, float, float]:
    h = (hue % 1.0) * 6.0
    c = value * saturation
    x = c * (1.0 - abs(h % 2.0 - 1.0))
    m = value - c
    if h < 1.0:
        rgb = (c, x, 0.0)
    elif h < 2.0:
        rgb = (x, c, 0.0)
    elif h < 3.0:
        rgb = (0.0, c, x)
    elif h < 4.0:
        rgb = (0.0, x, c)
    elif h < 5.0:
        rgb = (x, 0.0, c)
    else:
        rgb = (c, 0.0, x)
    return tuple(float(component + m) for component in rgb)


def _empty_surface(
    node: SDFNode,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
    message: str,
) -> ViewportSurface:
    mins, maxs = _safe_node_bounds(node)
    return ViewportSurface(
        key=key,
        object_kind=type(node).__name__,
        status="empty",
        vertices=np.zeros((0, 3), dtype=np.float32),
        normals=np.zeros((0, 3), dtype=np.float32),
        indices=np.zeros(0, dtype=np.uint32),
        wire_indices=np.zeros(0, dtype=np.uint32),
        color=color,
        bounds_min=mins,
        bounds_max=maxs,
        message=message,
    )


def _failed_surface(
    node: SDFNode,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
    message: str,
) -> ViewportSurface:
    mins, maxs = _safe_node_bounds(node)
    return ViewportSurface(
        key=key,
        object_kind=type(node).__name__,
        status="failed",
        vertices=np.zeros((0, 3), dtype=np.float32),
        normals=np.zeros((0, 3), dtype=np.float32),
        indices=np.zeros(0, dtype=np.uint32),
        wire_indices=np.zeros(0, dtype=np.uint32),
        color=color,
        bounds_min=mins,
        bounds_max=maxs,
        message=message,
    )


def _safe_node_bounds(node: SDFNode) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    try:
        box = node.bounding_box()
    except Exception:  # noqa: BLE001
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0)
    return (
        (float(box.x_min), float(box.y_min), float(box.z_min)),
        (float(box.x_max), float(box.y_max), float(box.z_max)),
    )


__all__ = [
    "ViewportSurface",
    "ViewportSurfaceCache",
    "ViewportSurfaceKey",
    "ViewportSurfaceScene",
    "build_viewport_surface",
    "build_viewport_surface_scene",
]
