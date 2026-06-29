from __future__ import annotations

from collections import Counter
from time import perf_counter

import numpy as np
import pytest

from app.viewport.renderers.qrhi.surface_renderer import (
    _LINE_STRIDE,
    _chunk_from_surface,
    _dynamic_line_payload,
)
from app.viewport.surface_cache import (
    ViewportSurface,
    ViewportSurfaceCache,
    build_viewport_surface_scene,
)
from app.artifacts import RenderSceneSnapshot, build_render_artifact
from core.scene import SceneDocument
from core.sdf import Extrude, Revolve


_FILLED_2D_KINDS = (
    "circle",
    "rectangle",
    "square",
    "rounded_rectangle",
    "ellipse",
    "regular_polygon",
    "polygon",
    "quadratic_bezier_surface",
)
_DRAG_START = (-0.35, -0.25, 0.0)
_DRAG_END = (0.35, 0.25, 0.0)
_SYMMETRIC_BEZIER_POINTS = (
    (-0.45, 0.0, 0.0),
    (-0.45, 0.45, 0.0),
    (0.0, 0.45, 0.0),
    (0.45, 0.45, 0.0),
    (0.45, 0.0, 0.0),
    (0.45, -0.45, 0.0),
    (0.0, -0.45, 0.0),
    (-0.45, -0.45, 0.0),
    (-0.45, 0.0, 0.0),
)


def _surface_scene(document: SceneDocument):
    version, tree = document.visual_snapshot()
    return build_viewport_surface_scene(
        tree,
        version,
        cache=ViewportSurfaceCache(resolution=14),
    )


def _add_filled_2d(document: SceneDocument, kind: str) -> int:
    if kind == "quadratic_bezier_surface":
        return document.add_point_shape_from_world_points(
            kind,
            _SYMMETRIC_BEZIER_POINTS,
            "xy",
        )
    return document.add_primitive_from_drag(kind, _DRAG_START, _DRAG_END)


def _solid_from_2d_revolve(document: SceneDocument, handle: int) -> int:
    return document.solid_from_2d(
        [handle],
        "revolve",
        revolve_axis_origin=(0.0, 0.0, 0.0),
        revolve_axis_direction=(0.0, 1.0, 0.0),
        revolve_radial_direction=(1.0, 0.0, 0.0),
        revolve_angle_degrees=270.0,
    )


def _point_covered_by_triangles(
    surface: ViewportSurface,
    point: tuple[float, float],
) -> bool:
    vertices = np.asarray(surface.vertices[:, :2], dtype=np.float64)
    triangles = np.asarray(surface.indices.reshape(-1, 3), dtype=np.int64)
    target = np.asarray(point, dtype=np.float64)
    for triangle in triangles:
        a, b, c = vertices[triangle]
        if _point_in_triangle(target, a, b, c):
            return True
    return False


def _point_in_triangle(
    point: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> bool:
    def cross(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
        return float(
            (second[0] - first[0]) * (third[1] - second[1])
            - (second[1] - first[1]) * (third[0] - second[0])
        )

    area = cross(a, b, c)
    if abs(area) <= 1.0e-12:
        return False
    sign = 1.0 if area > 0.0 else -1.0
    return (
        sign * cross(a, b, point) >= -1.0e-12
        and sign * cross(b, c, point) >= -1.0e-12
        and sign * cross(c, a, point) >= -1.0e-12
    )


@pytest.mark.parametrize("kind", _FILLED_2D_KINDS)
def test_filled_2d_create_renders_filled_surface(kind: str) -> None:
    document = SceneDocument()
    handle = _add_filled_2d(document, kind)
    source = document.node(handle)

    scene = _surface_scene(document)

    assert scene is not None
    assert len(scene.surfaces) == 1
    surface = scene.surfaces[0]
    assert surface.key.object_id == source.object_id
    assert surface.status == "ready"
    assert surface.indices.size > 0
    assert surface.wire_indices.size > 0
    assert surface.vertex_count >= 5


def test_multiple_2d_profiles_use_ordered_outlines_at_refined_resolution() -> None:
    document = SceneDocument()
    document.add_primitive_from_drag("rectangle", (-0.5, -0.3, 0.0), (0.5, 0.3, 0.0))
    document.add_primitive_from_drag("circle", (-0.25, 0.0, 0.0), (0.25, 0.0, 0.0))
    document.add_primitive_from_drag("square", (-0.2, -0.2, 0.0), (0.2, 0.2, 0.0))
    version, tree = document.visual_snapshot()

    scene = build_viewport_surface_scene(
        tree,
        version,
        cache=ViewportSurfaceCache(resolution=32),
    )

    assert scene is not None
    assert len(scene.surfaces) == 3
    assert scene.triangle_count == 136
    assert scene.vertex_count <= 145
    assert [surface.vertex_count for surface in scene.surfaces] == [5, 129, 5]


def test_2d_boolean_profile_renders_smooth_contoured_surface() -> None:
    document = SceneDocument()
    circle_handle = document.add_primitive_from_drag(
        "circle",
        (-0.35, -0.35, 0.0),
        (0.35, 0.35, 0.0),
    )
    bezier_handle = document.add_point_shape_from_world_points(
        "quadratic_bezier_surface",
        _SYMMETRIC_BEZIER_POINTS,
        "xy",
    )
    result_handle = document.combine(circle_handle, bezier_handle, "intersection")
    version, tree = document.visual_snapshot()

    scene = build_viewport_surface_scene(
        tree,
        version,
        cache=ViewportSurfaceCache(resolution=32),
    )

    assert scene is not None
    assert len(scene.surfaces) == 1
    surface = scene.surfaces[0]
    assert surface.key.object_id == result_handle
    assert surface.status == "ready"
    assert surface.indices.size > 0
    assert surface.wire_indices.size > 0
    assert surface.message == "2D profile rendered as contoured filled surface"
    assert surface.vertex_count > 250
    assert surface.triangle_count > 250


def test_2d_difference_profile_preserves_hole_fill() -> None:
    document = SceneDocument()
    outer_handle = document.add_primitive_from_drag(
        "circle",
        (-0.5, 0.0, 0.0),
        (0.5, 0.0, 0.0),
    )
    inner_handle = document.add_primitive_from_drag(
        "circle",
        (-0.2, 0.0, 0.0),
        (0.2, 0.0, 0.0),
    )
    result_handle = document.combine(outer_handle, inner_handle, "difference")
    version, tree = document.visual_snapshot()

    scene = build_viewport_surface_scene(
        tree,
        version,
        cache=ViewportSurfaceCache(resolution=32),
    )

    assert scene is not None
    assert len(scene.surfaces) == 1
    surface = scene.surfaces[0]
    assert surface.key.object_id == result_handle
    assert surface.status == "ready"
    assert surface.indices.size > 0
    assert not _point_covered_by_triangles(surface, (0.0, 0.0))


@pytest.mark.parametrize("kind", _FILLED_2D_KINDS)
def test_filled_2d_extrude_generates_ordered_surface(kind: str) -> None:
    document = SceneDocument()
    source_handle = _add_filled_2d(document, kind)
    solid_handle = document.solid_from_2d(
        [source_handle],
        "extrude",
        signed_height=0.5,
    )
    solid = document.node(solid_handle)

    scene = _surface_scene(document)

    assert isinstance(solid, Extrude)
    assert scene is not None
    assert len(scene.surfaces) == 1
    surface = scene.surfaces[0]
    assert surface.key.object_id == solid.object_id
    assert surface.status == "ready"
    assert surface.indices.size > 0
    assert surface.wire_indices.size > 0
    assert 0 < surface.vertex_count < 600


@pytest.mark.parametrize("kind", _FILLED_2D_KINDS)
def test_filled_2d_revolve_generates_ordered_surface_after_undo(kind: str) -> None:
    document = SceneDocument()
    source_handle = _add_filled_2d(document, kind)
    undo_snapshot = document.snapshot()
    extrude_handle = document.solid_from_2d(
        [source_handle],
        "extrude",
        signed_height=0.5,
    )
    assert isinstance(document.node(extrude_handle), Extrude)

    restored = undo_snapshot.snapshot()
    solid_handle = _solid_from_2d_revolve(restored, source_handle)
    solid = restored.node(solid_handle)
    scene = _surface_scene(restored)

    assert isinstance(solid, Revolve)
    assert scene is not None
    assert len(scene.surfaces) == 1
    surface = scene.surfaces[0]
    assert surface.key.object_id == solid.object_id
    assert surface.status == "ready"
    assert surface.indices.size > 0
    assert 0 < surface.vertex_count < 5000


def test_revolve_box_intersection_refined_surface_stays_interactive() -> None:
    document = SceneDocument()
    source_handle = document.add_primitive_from_drag(
        "rectangle",
        (-0.5, -0.3, 0.0),
        (0.5, 0.3, 0.0),
    )
    axis = np.asarray((0.701, 0.713, 0.0), dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    origin = np.asarray((-1.0, 0.8, 0.0), dtype=np.float64)
    radial = -origin
    radial = radial - axis * float(np.dot(radial, axis))
    radial = radial / np.linalg.norm(radial)
    revolve_handle = document.solid_from_2d(
        [source_handle],
        "revolve",
        revolve_axis_origin=tuple(float(value) for value in origin),
        revolve_axis_direction=tuple(float(value) for value in axis),
        revolve_radial_direction=tuple(float(value) for value in radial),
        revolve_angle_degrees=270.0,
    )
    box_handle = document.add_primitive_from_drag(
        "box",
        (-0.3, -0.2, 0.0),
        (0.3, 0.2, 0.0),
    )
    document.combine(revolve_handle, box_handle, "intersection")
    version, tree = document.visual_snapshot()

    scene = build_viewport_surface_scene(
        tree,
        version,
        cache=ViewportSurfaceCache(resolution=32),
    )

    assert scene is not None
    assert len(scene.surfaces) == 1
    surface = scene.surfaces[0]
    assert surface.status == "ready"
    assert surface.vertex_count > 1000
    assert surface.triangle_count > 3000
    assert scene.build_ms < 600.0


def test_repeated_2d_solid_operations_do_not_regress_to_grid_extraction() -> None:
    cache = ViewportSurfaceCache(resolution=14)
    durations_ms: list[float] = []
    vertex_counts: list[int] = []

    for cycle in range(3):
        for kind in _FILLED_2D_KINDS:
            document = SceneDocument()
            source_handle = _add_filled_2d(document, kind)
            document.solid_from_2d(
                [source_handle],
                "extrude",
                signed_height=0.4 + 0.05 * cycle,
            )
            version, tree = document.visual_snapshot()

            started = perf_counter()
            scene = build_viewport_surface_scene(tree, version, cache=cache)
            durations_ms.append((perf_counter() - started) * 1000.0)

            assert scene is not None
            assert len(scene.surfaces) == 1
            surface = scene.surfaces[0]
            assert surface.status == "ready"
            assert surface.triangle_count > 0
            assert surface.vertex_count < 600
            vertex_counts.append(surface.vertex_count)

    assert max(vertex_counts) < 600
    assert sum(durations_ms) < 2000.0


def test_quadratic_bezier_surface_create_renders_real_outline() -> None:
    document = SceneDocument()
    document.add_primitive("quadratic_bezier_surface")

    scene = _surface_scene(document)

    assert scene is not None
    assert len(scene.surfaces) == 1
    surface = scene.surfaces[0]
    assert surface.status == "ready"
    assert surface.indices.size > 0
    assert surface.wire_indices.size > 8
    assert surface.vertex_count > 4


def test_segment_1d_renders_real_line() -> None:
    document = SceneDocument()
    document.add_primitive("segment")

    scene = _surface_scene(document)

    assert scene is not None
    assert len(scene.surfaces) == 1
    surface = scene.surfaces[0]
    assert surface.status == "outline"
    assert surface.indices.size == 0
    assert surface.wire_indices.size >= 2
    assert surface.vertex_count >= 2


def test_quadratic_bezier_curve_1d_renders_real_polyline() -> None:
    document = SceneDocument()
    document.add_primitive("quadratic_bezier_curve")

    scene = _surface_scene(document)

    assert scene is not None
    assert len(scene.surfaces) == 1
    surface = scene.surfaces[0]
    assert surface.status == "outline"
    assert surface.indices.size == 0
    assert surface.wire_indices.size > 16
    assert surface.vertex_count > 8


def test_quadratic_bezier_surface_extrude_generates_indexed_surface() -> None:
    document = SceneDocument()
    source_handle = document.add_primitive("quadratic_bezier_surface")
    solid_handle = document.solid_from_2d(
        [source_handle],
        "extrude",
        signed_height=0.6,
    )
    solid = document.node(solid_handle)

    scene = _surface_scene(document)

    assert isinstance(solid, Extrude)
    assert scene is not None
    assert len(scene.surfaces) == 1
    surface = scene.surfaces[0]
    assert surface.key.object_id == solid.object_id
    assert surface.key.scene_revision == document.version
    assert surface.status == "ready"
    assert surface.indices.size > 0
    assert surface.wire_indices.size > 0


def test_circle_revolve_generates_indexed_surface_after_2d_create() -> None:
    document = SceneDocument()
    source_handle = document.add_primitive_from_drag(
        "circle",
        (-0.25, 0.0, 0.0),
        (0.25, 0.0, 0.0),
    )
    solid_handle = document.solid_from_2d([source_handle], "revolve")
    solid = document.node(solid_handle)

    scene = _surface_scene(document)

    assert isinstance(solid, Revolve)
    assert scene is not None
    ready = [surface for surface in scene.surfaces if surface.status == "ready"]
    assert len(ready) == 1
    assert ready[0].key.object_id == solid.object_id
    assert ready[0].indices.size > 0


def test_axis_crossing_revolve_deduplicates_folded_surface_faces() -> None:
    document = SceneDocument()
    source_handle = document.add_primitive_from_drag(
        "circle",
        (-0.25, 0.0, 0.0),
        (0.25, 0.0, 0.0),
    )
    document.solid_from_2d([source_handle], "revolve")

    scene = _surface_scene(document)

    assert scene is not None
    surface = scene.surfaces[0]
    quantized = np.round(surface.vertices, 6)
    face_keys = [
        tuple(sorted(tuple(quantized[int(index)]) for index in triangle))
        for triangle in surface.indices.reshape(-1, 3)
    ]
    counts = Counter(face_keys)

    assert all(count == 1 for count in counts.values())


def test_circle_extrude_uses_ordered_profile_surface() -> None:
    document = SceneDocument()
    source_handle = document.add_primitive_from_drag(
        "circle",
        (-0.25, 0.0, 0.0),
        (0.25, 0.0, 0.0),
    )
    solid_handle = document.solid_from_2d(
        [source_handle],
        "extrude",
        signed_height=0.5,
    )

    scene = _surface_scene(document)

    assert scene is not None
    surface = scene.surfaces[0]
    assert surface.key.object_id == solid_handle
    assert surface.status == "ready"
    assert 0 < surface.vertex_count < 200


def test_unchanged_object_reuses_surface_arrays_across_scene_revisions() -> None:
    document = SceneDocument()
    document.add_primitive("sphere")
    _version, tree = document.visual_snapshot()
    cache = ViewportSurfaceCache(resolution=12)

    first = build_viewport_surface_scene(tree, 10, cache=cache)
    second = build_viewport_surface_scene(tree, 11, cache=cache)

    assert first is not None
    assert second is not None
    assert first.surfaces[0].key.scene_revision == 10
    assert second.surfaces[0].key.scene_revision == 11
    assert first.surfaces[0].vertices is second.surfaces[0].vertices
    assert first.surfaces[0].indices is second.surfaces[0].indices


def test_moved_object_reuses_topology_and_shifts_vertices_from_cache() -> None:
    document = SceneDocument()
    handle = document.add_primitive("sphere")
    version, tree = document.visual_snapshot()
    cache = ViewportSurfaceCache(resolution=12)

    first = build_viewport_surface_scene(tree, version, cache=cache)
    document.move_object(handle, (1.25, -0.5, 0.25))
    moved_version, moved_tree = document.visual_snapshot()
    second = build_viewport_surface_scene(moved_tree, moved_version, cache=cache)

    assert first is not None
    assert second is not None
    first_surface = first.surfaces[0]
    second_surface = second.surfaces[0]
    assert second_surface.vertices is not first_surface.vertices
    assert second_surface.normals is first_surface.normals
    assert second_surface.indices is first_surface.indices
    np.testing.assert_allclose(
        second_surface.vertices,
        first_surface.vertices + np.asarray((1.25, -0.5, 0.25), dtype=np.float32),
    )


def test_new_sdf_renders_via_generic_path_without_renderer_changes() -> None:
    """Outcome 7: a new SDF needs only eval + bounds, not renderer surgery."""
    from dataclasses import dataclass

    from core.sdf.base import BoundingBox3D, SDFNode
    from app.viewport.surface_cache import (
        _SURFACE_BUILDERS,
        ViewportSurfaceKey,
        build_viewport_surface,
    )

    @dataclass
    class _Egg(SDFNode):
        radius: float = 1.0

        @property
        def dimension(self):  # type: ignore[override]
            return 3

        def to_numpy(self, X, Y, Z):  # noqa: N803
            return np.sqrt(X * X + (Y * 1.4) ** 2 + Z * Z) - self.radius

        def bounding_box(self) -> BoundingBox3D:
            r = self.radius
            return BoundingBox3D(-r, r, -r, r, -r, r)

    node = _Egg("egg", object_id=11, radius=1.0)
    assert type(node) not in _SURFACE_BUILDERS  # no registered fast path
    key = ViewportSurfaceKey(object_id=11, scene_revision=1, resolution=24)
    surface = build_viewport_surface(node, key)

    # Generic dual contouring produced real, watertight-ish geometry, not a
    # bounding-box stand-in or a failure.
    assert surface.status == "ready"
    assert surface.triangle_count > 100
    assert np.all(np.isfinite(surface.vertices))
    # Vertices lie on the egg surface (|f| ~ 0), proving it is the real SDF.
    v = surface.vertices.astype(np.float64)
    field = np.sqrt(v[:, 0] ** 2 + (v[:, 1] * 1.4) ** 2 + v[:, 2] ** 2) - 1.0
    assert np.max(np.abs(field)) < 0.1


def test_hermite_edge_roots_are_exact_and_grid_independent() -> None:
    """Phase A: analytic root-finding lands edge points on the true surface and
    yields exact normals, at any cell size (grid-independent precision)."""
    from core.sdf.primitives_3d import Sphere
    from app.viewport.surface_cache import _refine_edge_hermite

    node = Sphere("s", object_id=0, center=(0.0, 0.0, 0.0), radius=0.73)
    rng = np.random.default_rng(0)
    axes = np.eye(3)
    eps = np.array([1.0e-4, 1.0e-4, 1.0e-4])
    # Coarse (res ~14) and fine (res ~96) cell sizes; accuracy must hold at both.
    for half_edge in (0.065, 0.009):
        normals = rng.normal(size=(3000, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        on_surface = normals * 0.73
        axis = axes[rng.integers(0, 3, size=on_surface.shape[0])]
        point_a = on_surface - axis * half_edge
        point_b = on_surface + axis * half_edge
        fa = node.to_numpy(point_a[:, 0], point_a[:, 1], point_a[:, 2])
        fb = node.to_numpy(point_b[:, 0], point_b[:, 1], point_b[:, 2])
        cross = np.sign(fa) != np.sign(fb)
        point_a, point_b = point_a[cross], point_b[cross]
        fa, fb = fa[cross], fb[cross]

        pts, grad = _refine_edge_hermite(node, point_a, point_b, fa, fb, eps)
        field = node.to_numpy(pts[:, 0], pts[:, 1], pts[:, 2])
        assert np.mean(np.abs(field)) < 1.0e-4
        # Exact outward sphere normal.
        expected = pts / np.linalg.norm(pts, axis=1, keepdims=True)
        unit = grad / np.linalg.norm(grad, axis=1, keepdims=True)
        assert np.min(np.einsum("ij,ij->i", unit, expected)) > 0.999


def test_non_clippable_boolean_falls_back_to_watertight_band() -> None:
    """When an operand has no clip mesh (here a Pyramid, not in the clip set),
    the boolean falls back to the dual-contour band, which is watertight/manifold."""
    from collections import Counter

    from core.sdf.operators import Intersection
    from core.sdf.primitives_3d import Box, Pyramid
    from app.viewport.surface_cache import (
        _NARROW_BAND_MIN_RES,
        ViewportSurfaceKey,
        build_viewport_surface,
    )

    box = Box(
        "b", object_id=0, center=(0.0, 0.0, 0.0),
        axis_u=(1.0, 0.0, 0.0), axis_v=(0.0, 1.0, 0.0), axis_w=(0.0, 0.0, 1.0),
        half_size=(0.7, 0.7, 0.7),
    )
    pyramid = Pyramid("p", object_id=0, center=(0.0, 0.0, -0.2), base_half_size=0.8, half_height=0.9)
    node = Intersection("i", object_id=12, left=box, right=pyramid)
    key = ViewportSurfaceKey(
        object_id=12, scene_revision=1, resolution=_NARROW_BAND_MIN_RES
    )
    surface = build_viewport_surface(node, key)
    # Not a clip ("clipped" message) -> the watertight band path.
    assert "clipped" not in surface.message
    assert surface.status == "ready"
    assert surface.triangle_count > 1000

    edge_use: Counter = Counter()
    for a, b, c in surface.indices.reshape(-1, 3):
        for u, v in ((a, b), (b, c), (c, a)):
            edge_use[(min(int(u), int(v)), max(int(u), int(v)))] += 1
    # Essentially watertight: at most a handful of boundary edges (the band can
    # leave a few at an extreme singular apex), all others shared by two faces.
    boundary = sum(1 for count in edge_use.values() if count != 2)
    assert boundary <= 8
    assert boundary < 0.001 * len(edge_use)


def test_nested_boolean_clips_recursively() -> None:
    """A nested boolean of clippable operands renders through the exact recursive
    clip path (not grid dual contouring), placing vertices on the true surface."""
    from core.sdf.operators import Difference, Intersection
    from core.sdf.primitives_3d import Box, Cylinder, Sphere
    from app.viewport.surface_cache import ViewportSurfaceKey, build_viewport_surface

    sphere = Sphere("s", object_id=0, center=(0.0, 0.0, 0.0), radius=0.9)
    box = Box(
        "b", object_id=0, center=(0.0, 0.0, 0.0),
        axis_u=(1.0, 0.0, 0.0), axis_v=(0.0, 1.0, 0.0), axis_w=(0.0, 0.0, 1.0),
        half_size=(0.7, 0.7, 0.7),
    )
    cyl = Cylinder("c", object_id=0, center=(0.0, 0.0, 0.0), radius=0.35, half_height=1.2)
    # (sphere ∩ box) − cylinder: a rounded cube with a cylindrical hole.
    node = Difference(
        "d", object_id=20, left=Intersection("i", 0, sphere, box), right=cyl
    )
    surface = build_viewport_surface(
        node, ViewportSurfaceKey(object_id=20, scene_revision=1, resolution=96)
    )
    assert "clipped" in surface.message
    idx = surface.indices.reshape(-1, 3)
    v = surface.vertices.astype(np.float64)
    f = node.to_numpy(v[:, 0], v[:, 1], v[:, 2])
    assert np.max(np.abs(f[np.unique(idx)])) < 5.0e-3
    # Consistent winding (no folds/tears).
    tri = v[idx]
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(face, axis=1)
    assert np.all(length > 1.0e-12)
    vnorm = surface.normals.astype(np.float64)[idx].mean(axis=1)
    vnorm /= np.maximum(np.linalg.norm(vnorm, axis=1), 1.0e-12)[:, None]
    assert np.min(np.einsum("ij,ij->i", face / length[:, None], vnorm)) > -0.3


@pytest.mark.parametrize("op", ["intersection", "difference", "union"])
def test_primitive_boolean_clips_to_exact_surface(op: str) -> None:
    """Booleans of primitives clip smooth analytic operand meshes against each
    other's SDF: rendered vertices lie on the exact boolean surface and the
    winding is consistent (no folds/tears)."""
    from core.sdf.operators import Difference, Intersection, Union
    from core.sdf.primitives_3d import Box, Sphere
    from app.viewport.surface_cache import ViewportSurfaceKey, build_viewport_surface

    sphere = Sphere("s", object_id=0, center=(0.0, 0.0, 0.0), radius=0.9)
    box = Box(
        "b", object_id=0, center=(0.0, 0.0, 0.0),
        axis_u=(1.0, 0.0, 0.0), axis_v=(0.0, 1.0, 0.0), axis_w=(0.0, 0.0, 1.0),
        half_size=(0.7, 0.7, 0.7),
    )
    ctor = {"intersection": Intersection, "difference": Difference, "union": Union}[op]
    node = ctor("n", object_id=15, left=box, right=sphere)
    surface = build_viewport_surface(
        node, ViewportSurfaceKey(object_id=15, scene_revision=1, resolution=96)
    )
    assert "clipped" in surface.message
    idx = surface.indices.reshape(-1, 3)
    v = surface.vertices.astype(np.float64)
    # Every rendered vertex lies on the exact boolean surface (max(...) ~ 0).
    f = node.to_numpy(v[:, 0], v[:, 1], v[:, 2])
    assert np.max(np.abs(f[np.unique(idx)])) < 1.0e-3
    # Consistent winding, no degenerate triangles (no tears).
    tri = v[idx]
    face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    length = np.linalg.norm(face, axis=1)
    assert np.all(length > 1.0e-12)
    vnorm = surface.normals.astype(np.float64)[idx].mean(axis=1)
    vnorm /= np.maximum(np.linalg.norm(vnorm, axis=1), 1.0e-12)[:, None]
    assert np.min(np.einsum("ij,ij->i", face / length[:, None], vnorm)) > -0.3


def test_boolean_mesh_has_no_folded_or_degenerate_triangles() -> None:
    """Clean geometry beats 'exact' geometry that tears. The contoured boolean
    mesh must have consistent winding (no folds) and no zero-area triangles, so
    the seam never shows tears/holes."""
    from core.sdf.operators import Difference, Intersection, Union
    from core.sdf.primitives_3d import Box, Sphere
    from app.viewport.surface_cache import (
        _NARROW_BAND_MIN_RES,
        ViewportSurfaceKey,
        build_viewport_surface,
    )

    sphere = Sphere("s", object_id=0, center=(0.0, 0.0, 0.0), radius=0.9)
    box = Box(
        "b", object_id=0, center=(0.0, 0.0, 0.0),
        axis_u=(1.0, 0.0, 0.0), axis_v=(0.0, 1.0, 0.0), axis_w=(0.0, 0.0, 1.0),
        half_size=(0.7, 0.7, 0.7),
    )
    for ctor in (Intersection, Difference, Union):
        node = ctor("n", object_id=16, left=sphere, right=box)
        key = ViewportSurfaceKey(
            object_id=16, scene_revision=1, resolution=_NARROW_BAND_MIN_RES
        )
        surface = build_viewport_surface(node, key)
        idx = surface.indices.reshape(-1, 3)
        v = surface.vertices.astype(np.float64)
        tri = v[idx]
        face = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
        length = np.linalg.norm(face, axis=1)
        # No zero-area triangles.
        assert np.all(length > 1.0e-12)
        # Consistent winding: no triangle is folded backwards relative to the
        # smooth vertex-normal field (which would z-fight and read as a tear).
        vnorm = surface.normals.astype(np.float64)[idx].mean(axis=1)
        vnorm /= np.maximum(np.linalg.norm(vnorm, axis=1), 1.0e-12)[:, None]
        normalized_dot = np.einsum("ij,ij->i", face / length[:, None], vnorm)
        assert np.min(normalized_dot) > -0.3


def test_sharp_feature_qef_places_exact_edges_grid_independently() -> None:
    """Phase B: the SVD/Lindstrom QEF puts box-intersection edges and corners
    exactly on the surface, at coarse and fine resolution alike."""
    from core.sdf.operators import Intersection
    from core.sdf.primitives_3d import Box
    from app.viewport.surface_cache import ViewportSurfaceKey, build_viewport_surface

    box_a = Box(
        "a", object_id=0, center=(0.0, 0.0, 0.0),
        axis_u=(1.0, 0.0, 0.0), axis_v=(0.0, 1.0, 0.0), axis_w=(0.0, 0.0, 1.0),
        half_size=(0.7, 0.7, 0.7),
    )
    box_b = Box(
        "b", object_id=0, center=(0.3, 0.3, 0.3),
        axis_u=(1.0, 0.0, 0.0), axis_v=(0.0, 1.0, 0.0), axis_w=(0.0, 0.0, 1.0),
        half_size=(0.6, 0.6, 0.6),
    )
    node = Intersection("i", object_id=9, left=box_a, right=box_b)

    for res in (24, 96):
        key = ViewportSurfaceKey(object_id=9, scene_revision=1, resolution=res)
        surface = build_viewport_surface(node, key)
        assert surface.status == "ready"
        v = surface.vertices.astype(np.float64)
        f = node.to_numpy(v[:, 0], v[:, 1], v[:, 2])
        # Piecewise-planar booleans are reconstructed essentially exactly,
        # independent of grid resolution (sharp edges, not averaged).
        assert np.max(np.abs(f)) < 1.0e-5


def test_clipped_boolean_uses_exact_operand_normals() -> None:
    """Clipped booleans carry the exact analytic operand normals: the sphere
    portion of a sphere-intersection has perfectly radial normals (smooth, not
    grid-faceted shading)."""
    from core.sdf.operators import Intersection
    from core.sdf.primitives_3d import Box, Sphere
    from app.viewport.surface_cache import ViewportSurfaceKey, build_viewport_surface

    sphere = Sphere("s", object_id=0, center=(0.0, 0.0, 0.0), radius=0.9)
    box = Box(
        "b", object_id=0, center=(0.0, 0.0, 0.0),
        axis_u=(1.0, 0.0, 0.0), axis_v=(0.0, 1.0, 0.0), axis_w=(0.0, 0.0, 1.0),
        half_size=(0.7, 0.7, 0.7),
    )
    node = Intersection("i", object_id=5, left=sphere, right=box)
    surface = build_viewport_surface(
        node, ViewportSurfaceKey(object_id=5, scene_revision=1, resolution=64)
    )
    assert surface.status == "ready"
    n = surface.normals.astype(np.float64)
    v = surface.vertices.astype(np.float64)
    lengths = np.linalg.norm(n, axis=1)
    assert np.all(np.isfinite(n))
    np.testing.assert_allclose(lengths, 1.0, atol=1.0e-3)

    # On the sphere face (sphere ~ 0, strictly inside the box, away from the
    # seam) the normal is the exact radial normal.
    on_sphere_face = (
        np.abs(sphere.to_numpy(v[:, 0], v[:, 1], v[:, 2])) < 1.0e-4
    ) & (box.to_numpy(v[:, 0], v[:, 1], v[:, 2]) < -0.02)
    assert np.any(on_sphere_face)
    radial = v[on_sphere_face] / np.linalg.norm(v[on_sphere_face], axis=1, keepdims=True)
    assert np.min(np.einsum("ij,ij->i", n[on_sphere_face], radial)) > 0.999


def test_editing_session_does_not_leak_reuse_slots() -> None:
    from core.sdf.primitives_3d import Sphere

    cache = ViewportSurfaceCache(resolution=12)
    for revision in range(1, 251):
        node = Sphere("s", object_id=7, center=(0.0, 0.0, 0.0), radius=1.0 + revision * 0.001)
        cache.get_or_build(node, revision)
        cache.prune_before(revision - 2)
        cache.prune_to_object_ids(frozenset({7}))

    # Reuse slots stay bounded to the live object set regardless of edit count.
    assert len(cache._latest_by_signature) == 1
    assert len(cache._latest_by_translation_signature) == 1
    # Revision-keyed surfaces remain bounded by prune_before.
    assert len(cache._surfaces) <= 3


def test_deleted_object_reuse_slots_are_pruned() -> None:
    from core.sdf.primitives_3d import Sphere

    cache = ViewportSurfaceCache(resolution=12)
    cache.get_or_build(Sphere("a", object_id=1, radius=1.0), 1)
    cache.get_or_build(Sphere("b", object_id=2, radius=1.0), 1)
    assert len(cache._latest_by_signature) == 2

    # Object 2 deleted from the scene; only object 1 survives.
    cache.prune_to_object_ids(frozenset({1}))
    assert set(cache._latest_by_signature) == {1}
    assert set(cache._latest_by_translation_signature) == {1}


def test_outline_only_surface_builds_thick_line_vertices() -> None:
    document = SceneDocument()
    document.add_primitive("segment")
    scene = _surface_scene(document)

    assert scene is not None
    chunk = _chunk_from_surface(scene.surfaces[0])

    assert chunk is not None
    assert chunk.index_count == 0
    assert chunk.thick_line_vertex_count >= 6
    assert len(chunk.thick_line_bytes) == chunk.thick_line_vertex_count * _LINE_STRIDE
    payload, count = _dynamic_line_payload([chunk], None)
    assert count == chunk.thick_line_vertex_count
    assert len(payload) == count * _LINE_STRIDE


def test_render_artifact_uses_requested_surface_resolution() -> None:
    document = SceneDocument()
    document.add_primitive("sphere")
    version, tree = document.visual_snapshot()

    # Resolutions above the smooth-primitive tessellation floor, where the
    # triangle count scales with the requested resolution.
    coarse = build_render_artifact(
        RenderSceneSnapshot(
            version=version,
            tree=tree,
            surface_resolution=40,
            refine_after=False,
        )
    )
    refined = build_render_artifact(
        RenderSceneSnapshot(
            version=version,
            tree=tree,
            surface_resolution=80,
            refine_after=False,
        )
    )

    assert coarse.timings.surface_resolution == 40
    assert refined.timings.surface_resolution == 80
    assert refined.timings.surface_triangle_count > coarse.timings.surface_triangle_count
