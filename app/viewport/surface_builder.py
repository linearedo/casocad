"""Analytic primitive surface builders, the per-object surface cache, and the dispatcher.

`build_viewport_surface` is the one routing point between the two rendering
strategies:

    primitive/sweep leaf      -> analytic mesh (this module's `_primitive_surface`)
    sharp boolean (meshable)  -> Strategy A, exact clip  (`surface_clipping`)
    anything else / field SDF -> Strategy B, dual contour (`surface_contouring`)

The two strategies are independent modules behind a clean boundary; this module owns
the primitive mesh library, the per-object/per-revision cache, and the dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from threading import RLock
from time import perf_counter
from typing import Callable, Iterator, Literal

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

from app.viewport.surface_types import (
    SurfaceStatus,
    ViewportSurface,
    ViewportSurfaceKey,
    ViewportSurfaceScene,
    _DEFAULT_RESOLUTION,
    _empty_surface,
    _failed_surface,
    _hsv_to_rgb,
    _object_color,
    _safe_node_bounds,
)
from app.viewport.surface_geomops import (
    _MAX_DUAL_CONTOUR_WIREFRAME_TRIANGLES,
    _analytic_gradient,
    _mesh_normals,
    _normalize,
    _normalize_rows,
    _orient_triangles,
    _refine_edge_hermite,
    _split_marked_triangles,
    _wire_indices_from_triangles,
)
from app.viewport.surface_contouring import (
    _NARROW_BAND_MIN_RES,
    _dual_contour_surface,
    _edge_has_crossing,
    contour_surface,
)
from app.viewport.surface_clipping import clip_surface



# Truncation ratio for the sharp-feature QEF: eigenvalues of A^T A (= squared
# singular values) below this fraction of the largest are treated as flat
# directions and dropped, so planar faces/edges/corners get rank 1/2/3 solves.
# At/above this requested resolution, contour via the sparse narrow band
# (manifold, memory-bounded) using base = resolution / subdiv.
# Precision ceiling for generic dual contouring (booleans and unsupported SDFs).
# The vectorised vertex/index passes make this affordable on the worker thread
# (~0.7s for a 96^3 boolean), reached progressively so interaction stays smooth.
_MAX_CONTOURED_2D_CELLS = 96
_MAX_SAMPLED_2D_CELLS = 48
# Quality ceiling for the revolve sweep surface builder (profile x angular samples). The
# cost grows ~quadratically, so this is a deliberate balance: 48 -> ~144 angular
# + 48 profile segments (smooth) at ~0.3s, vs the old octagonal 8.
_MAX_REVOLVE_VIEWPORT_RESOLUTION = 48








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
    include_component_surfaces: bool = False,
) -> ViewportSurfaceScene | None:
    if tree is None:
        return None
    start = perf_counter()
    surface_cache = cache or ViewportSurfaceCache()
    components = tuple(getattr(tree, "components", ())) or (tree.root,)
    primary_object_ids = frozenset(
        int(component.object_id)
        for component in components
        if int(getattr(component, "object_id", 0)) > 0
    )
    candidates = (
        _surface_component_nodes(components)
        if include_component_surfaces
        else components
    )
    live = tuple(
        component
        for component in candidates
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
        primary_object_ids=primary_object_ids,
    )


def _surface_component_nodes(components: tuple[SDFNode, ...]) -> tuple[SDFNode, ...]:
    nodes: list[SDFNode] = []
    seen: set[int] = set()
    for component in components:
        for node in _walk_surface_component(component):
            object_id = int(getattr(node, "object_id", 0) or 0)
            if object_id <= 0 or object_id in seen:
                continue
            seen.add(object_id)
            nodes.append(node)
    return tuple(nodes)


def _walk_surface_component(node: SDFNode) -> Iterator[SDFNode]:
    yield node
    for child in node.children():
        yield from _walk_surface_component(child)


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


def _operand_primitive_mesh(
    node: SDFNode, key: ViewportSurfaceKey, color: tuple[float, float, float]
):
    """Provide a meshable leaf's analytic mesh arrays to the clip module."""
    surface = _primitive_surface(node, key, color)
    if surface is None or not surface.has_geometry:
        return None
    return (
        surface.vertices.astype(np.float64),
        surface.normals.astype(np.float64),
        surface.indices.reshape(-1, 3).astype(np.int64),
    )


def build_viewport_surface(node: SDFNode, key: ViewportSurfaceKey) -> ViewportSurface:
    color = _object_color(key.object_id)
    try:
        primitive = _primitive_surface(node, key, color)
        if primitive is not None:
            return primitive
        if node.dimension == 3:
            # Strategy A (exact): clip analytic operand meshes against each other's
            # SDF -> smooth curved faces + an exact root-found seam, no grid
            # polygonization. Strategy B (fallback): dual contouring, for operands
            # with no analytic mesh (field SDFs, nested non-meshable, etc.).
            clipped = clip_surface(node, key, color, _operand_primitive_mesh)
            if clipped is not None:
                return clipped
            return contour_surface(node, key, color)
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

# Optional analytic fast-path surface builders keyed by SDF node type. The generic
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


















# Per-cell minimal-edge quad specs for sparse dual contouring. Each crossing cell
# "owns" the x/y/z edge at its corner 0; the quad joins it to three neighbours.
# (corner_a, corner_b, neighbour offsets in dense quad order, flip-on-corner0>0).




















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




































__all__ = [
    "ViewportSurface",
    "ViewportSurfaceCache",
    "ViewportSurfaceKey",
    "ViewportSurfaceScene",
    "build_viewport_surface",
    "build_viewport_surface_scene",
]
