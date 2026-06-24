from __future__ import annotations

"""Per-leaf bounds + uniform-grid binning for the codegen spatial cull.

Backend-agnostic, pure data. ``leaf_bounds`` gives each IR leaf a conservative
world-space bounding sphere (from the SDF node's authoritative ``bounding_box()``,
or a cheap closed-form fallback for hand-built IRs); ``build_term_grid`` bins the
codegen term list into a world-space uniform grid so the shader's DDA march only
evaluates the few terms whose bound reaches each cell. (The old bytecode-VM
flatten/cull machinery was removed with the VM.)
"""

import numpy as np

from .gpu_node_types import is_operator
from .render_ir import RenderIR, RenderIRNode

_INF_RADIUS = 1.0e9  # sentinel "unbounded" leaf: never culled


def _leaf_bounding_sphere(node: RenderIRNode) -> tuple[float, float, float, float]:
    """Conservative world-space (cx, cy, cz, radius) for a leaf.

    Params are already world-space (build_render_ir applies the transform).
    Kinds without a cheap closed-form bound get an unbounded sentinel radius so
    they are simply never culled (correct, just not accelerated).
    """

    p = node.params
    k = node.kind
    if k == "sphere":
        return (p[0], p[1], p[2], abs(p[3]))
    if k in ("box", "box_frame"):
        # centre + diagonal of the (oriented) half-size box.
        hs = np.asarray(p[12:15], dtype=np.float64)
        return (p[0], p[1], p[2], float(np.linalg.norm(hs)))
    if k in ("cylinder", "cone"):
        radius, half_h = abs(p[12]), abs(p[13])
        return (p[0], p[1], p[2], float(np.hypot(radius, half_h)))
    if k == "capped_cone":
        radius = max(abs(p[12]), abs(p[13]))
        return (p[0], p[1], p[2], float(np.hypot(radius, abs(p[14]))))
    if k == "pyramid":
        return (p[0], p[1], p[2], float(np.hypot(abs(p[12]), abs(p[13]))))
    if k == "torus":
        return (p[0], p[1], p[2], float(abs(p[12]) + abs(p[13])))
    # Placed 2D sections (closed-form): thin discs/quads on a world plane. World
    # centre = origin + cu*axis_u + cv*axis_v (orthonormal basis); radius = the
    # profile extent + the section's small half-thickness. Giving these a bound lets
    # the cull grid include them, so adding a 2D section doesn't disable culling for
    # the whole scene. (Params are already world-space; see build_render_ir.)
    if k in ("placed_circle_2d", "placed_rectangle_2d", "placed_square_2d",
             "placed_rounded_rectangle_2d", "placed_ellipse_2d"):
        origin = np.asarray(p[0:3], dtype=np.float64)
        center = origin + p[12] * np.asarray(p[3:6], dtype=np.float64) \
            + p[13] * np.asarray(p[6:9], dtype=np.float64)
        if k == "placed_circle_2d":
            r = abs(p[14])
        elif k == "placed_square_2d":
            r = abs(p[14]) * 1.4142135623730951
        elif k == "placed_ellipse_2d":
            r = max(abs(p[14]), abs(p[15]))
        else:  # rectangle / rounded_rectangle: half-extents at indices 14, 15
            r = float(np.hypot(p[14], p[15]))
        return (float(center[0]), float(center[1]), float(center[2]), r + 0.01)
    # polyline/bezier sections, extrude/revolve/profile sub-graphs, sweeps, placed
    # 1D: no cheap closed-form bound -> unbounded sentinel (never culled).
    return (0.0, 0.0, 0.0, _INF_RADIUS)


def leaf_bounds(render_ir: RenderIR | None) -> np.ndarray:
    """(N, 4) float32 bounding spheres, one per IR node (operators -> unbounded)."""

    nodes = () if render_ir is None else render_ir.nodes
    out = np.zeros((max(len(nodes), 1), 4), dtype=np.float32)
    for i, node in enumerate(nodes):
        if is_operator(node.kind):
            out[i] = (0.0, 0.0, 0.0, _INF_RADIUS)
        elif getattr(node, "bound", None) is not None:
            # Authoritative world bound captured from the SDF node's bounding_box()
            # (build_render_ir) — covers every geometry kind, incl. sections, tubes,
            # extrude/revolve. Falls back to the param formula for hand-built IRs.
            out[i] = node.bound
        else:
            out[i] = _leaf_bounding_sphere(node)
    return out


def build_term_grid(
    positive_leaves: np.ndarray,
    bounds: np.ndarray,
    dim: int = 16,
):
    """Uniform grid for the codegen term model: bin each term by its positive
    leaf's bound; items are TERM indices.

    A codegen term is one positive solid with local carves; carves only shrink the
    solid, so the solid's bound conservatively contains the term's surface → bin by
    it. The shader looks up a cell, then for each term index evaluates the carved
    term and accumulates the DNF groups. Returns
    ``(origin, cell, dim, offsets, counts, items)`` or None when any term's positive
    leaf is unbounded (sentinel radius) — that scene keeps the brute-force map().
    """
    pl = np.asarray(positive_leaves, dtype=np.int64)
    if pl.size == 0:
        return None
    b = bounds[pl].astype(np.float64)
    if np.any(b[:, 3] >= _INF_RADIUS):
        return None
    lo_all = np.min(b[:, :3] - b[:, 3:4], axis=0)
    hi_all = np.max(b[:, :3] + b[:, 3:4], axis=0)
    span = np.maximum(hi_all - lo_all, 1e-4)
    pad = span * 0.01
    origin = lo_all - pad
    cell = (span + 2 * pad) / dim
    ncells = dim * dim * dim
    inv = 1.0 / cell
    center, radius = b[:, :3], b[:, 3:4]
    lo = np.clip(np.floor((center - radius - origin) * inv).astype(np.int64), 0, dim - 1)
    hi = np.clip(np.floor((center + radius - origin) * inv).astype(np.int64), 0, dim - 1)
    sp = hi - lo + 1
    per = sp[:, 0] * sp[:, 1] * sp[:, 2]
    total = int(per.sum())
    if total == 0:
        return None
    starts = np.zeros(len(pl) + 1, np.int64)
    starts[1:] = np.cumsum(per)
    local = np.arange(total, dtype=np.int64) - np.repeat(starts[:-1], per)
    nx = np.repeat(sp[:, 0], per)
    ny = np.repeat(sp[:, 1], per)
    ix = np.repeat(lo[:, 0], per) + (local % nx)
    iy = np.repeat(lo[:, 1], per) + ((local // nx) % ny)
    iz = np.repeat(lo[:, 2], per) + (local // (nx * ny))
    ci = (iz * dim + iy) * dim + ix
    counts = np.bincount(ci, minlength=ncells).astype(np.int32)
    offsets = np.zeros(ncells, np.int32)
    offsets[1:] = np.cumsum(counts)[:-1]
    order = np.argsort(ci, kind="stable")
    items = np.repeat(np.arange(len(pl)), per)[order].astype(np.uint32)
    return (tuple(float(v) for v in origin), tuple(float(v) for v in cell),
            int(dim), offsets, counts, items)


__all__ = ["leaf_bounds", "build_term_grid", "_INF_RADIUS"]
