"""Strategy A — exact boolean rendering by SDF-clipping analytic meshes.

This is the precise, fast path for *sharp* booleans (union/intersection/difference)
whose operands have an analytic mesh (primitives, sweeps, and — recursively — nested
booleans of those). Each operand's smooth mesh is clipped against the *other*
operand's exact SDF, giving exact curved faces and a root-found seam with no grid
polygonisation.

It renders nothing it cannot do exactly: when an operand has no analytic mesh (a
field/smooth SDF, or a primitive outside the clip set) `clip_surface` returns None and
the dispatcher falls back to Strategy B (`surface_contouring`). This module is
self-contained — it only depends on `surface_types` and `surface_meshops`, and receives
operand meshes through an injected `OperandMeshProvider`, so it never imports the
primitive meshers or the contouring fallback.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray

from core.sdf import (
    Box,
    CappedCone,
    Cone,
    Cylinder,
    Difference,
    Extrude,
    Intersection,
    Revolve,
    SDFNode,
    Sphere,
    Torus,
    Union,
)

from app.viewport.surface_types import (
    ViewportSurface,
    ViewportSurfaceKey,
    _empty_surface,
)
from app.viewport.surface_meshops import (
    _MAX_DUAL_CONTOUR_WIREFRAME_TRIANGLES,
    _normalize_rows,
    _orient_triangles,
    _refine_edge_hermite,
    _split_marked_triangles,
    _wire_indices_from_triangles,
)

# Supplies the analytic mesh (verts f64, normals f64, tris i64) of a meshable
# primitive/sweep leaf, or None. Injected by the dispatcher so this module stays
# independent of the primitive meshers.
OperandMeshProvider = Callable[
    [SDFNode, "ViewportSurfaceKey", "tuple[float, float, float]"],
    "tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]] | None",
]


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
    node: SDFNode,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
    operand_mesh: OperandMeshProvider,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.int64]] | None:
    """Adequately tessellated operand mesh for SDF clipping, or None.

    Curved primitives come pre-tessellated from the analytic meshers (supplied by
    ``operand_mesh``); the box is rebuilt as a per-face grid; nested booleans recurse.
    Operands without a fine analytic mesh return None so the boolean falls back to
    dual contouring.
    """
    n = max(8, int(key.resolution))
    if isinstance(node, (Union, Intersection, Difference)):
        nested = _clip_boolean_mesh(node, key, color, operand_mesh)
        if nested is None:
            return None
        verts, normals, faces = nested
        return (
            verts.astype(np.float64),
            normals.astype(np.float64),
            faces.reshape(-1, 3).astype(np.int64),
        )
    if isinstance(node, Box):
        return _grid_box_mesh(node, n)
    if isinstance(node, (Sphere, Cylinder, Cone, CappedCone, Torus, Extrude, Revolve)):
        return operand_mesh(node, key, color)
    return None


def _clip_boolean_mesh(
    node: SDFNode,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
    operand_mesh: OperandMeshProvider,
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.uint32]] | None:
    """Recursively build the SDF-clipped mesh of a boolean as raw arrays.

    Each operand mesh is built by `_clip_operand_mesh` (a primitive/sweep mesh,
    or — for a nested boolean — this function again), then clipped against the
    *other* operand's SDF (always available from the tree). So an arbitrarily
    nested tree of sharp booleans over meshable leaves renders through the exact
    clip path. Returns (vertices f32, normals f32, indices u32) or None when an
    operand has no analytic mesh (then the caller falls back to dual contouring).
    """
    if not isinstance(node, (Union, Intersection, Difference)):
        return None
    left = node.left
    right = node.right
    if left is None or right is None:
        return None
    operand_l = _clip_operand_mesh(left, key, color, operand_mesh)
    operand_r = _clip_operand_mesh(right, key, color, operand_mesh)
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
    # so a curved cut through a coarse flat region (extrude/revolve wall/cap, or a
    # nested-boolean face) is captured smoothly without inflating the rest.
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
        return None
    index_array = _orient_triangles(
        vertices, normals, faces.reshape(-1).astype(np.uint32)
    )
    # Drop the operand verts the clip discarded (roughly half), remapping indices.
    used = np.unique(index_array)
    remap = np.full(vertices.shape[0], -1, dtype=np.int64)
    remap[used] = np.arange(used.shape[0], dtype=np.int64)
    return vertices[used], normals[used], remap[index_array].astype(np.uint32)


def clip_surface(
    node: SDFNode,
    key: ViewportSurfaceKey,
    color: tuple[float, float, float],
    operand_mesh: OperandMeshProvider,
) -> ViewportSurface | None:
    """Render a sharp boolean by clipping analytic operand meshes, or None.

    Returns None when an operand has no analytic mesh, so the caller can fall
    back to dual contouring. ``operand_mesh`` supplies the analytic mesh arrays
    of a primitive/sweep leaf (the clip module owns the box grid and recursion).
    """
    result = _clip_boolean_mesh(node, key, color, operand_mesh)
    if result is None:
        return None
    vertices, normals, index_array = result
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
