"""Strategy B (fallback) — dual contouring of an arbitrary SDF.

The universal isosurface extractor: it renders *any* bounded 3D SDF from
`node.to_numpy` + `node.bounding_box` alone, with no parametric mesh. This is what
makes casoCAD a real SDF engine — it is the only path that can render field-based
geometry that has no analytic mesh (future smooth/rounded booleans, shells, offsets,
deformations), and it produces watertight/manifold output.

It is the fallback for `surface_clipping`: the dispatcher tries the exact clip first
and calls `contour_surface` only when an operand has no analytic mesh. Exact-Hermite
edges + sharp-feature QEF + a memory-bounded narrow band keep it precise. This module
depends only on `surface_types` and `surface_meshops`; it never imports the clip path
or the primitive meshers.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from core.sdf import SDFNode

from app.viewport.surface_types import (
    SurfaceStatus,
    ViewportSurface,
    ViewportSurfaceKey,
    _empty_surface,
)
from app.viewport.surface_meshops import (
    _MAX_DUAL_CONTOUR_WIREFRAME_TRIANGLES,
    _analytic_gradient,
    _normalize,
    _normalize_rows,
    _orient_triangles,
    _refine_edge_hermite,
    _wire_indices_from_triangles,
)


_QEF_SINGULAR_RATIO = 0.03


_NARROW_BAND_MIN_RES = 96


_NARROW_BAND_SUBDIV = 2


_MIN_AXIS_CELLS = 8


_MAX_AXIS_CELLS = 96


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




def _edge_has_crossing(first: float, second: float) -> bool:
    return bool(
        (first <= 0.0 <= second or second <= 0.0 <= first)
        and abs(first - second) > 1.0e-12
    )






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

contour_surface = _dual_contour_surface
