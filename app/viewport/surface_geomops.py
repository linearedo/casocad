"""Shared surface-geometry primitives used by both rendering strategies.

Pure-numpy helpers with no dependency on either strategy or the primitive surface builders:
triangle orientation, normals, wireframe edges, SDF gradient/edge root-finding, and
edge-split subdivision. Both `surface_clipping` and `surface_contouring` import from
here; this module imports nothing from the viewport package, so it is a leaf.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from core.sdf import SDFNode


_MAX_DUAL_CONTOUR_WIREFRAME_TRIANGLES = 3000


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
