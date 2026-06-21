from __future__ import annotations

"""Scene flattening + per-leaf bounds for GPU spatial culling (design §13.5).

Original culling design (not derived from any external SDF library): when the
scene's operator graph is a union/difference composition it flattens to two flat
leaf sets — *additive* (contributes by union/min) and *subtractive* (carves by
difference) — so the field is

    dist(p) = max( min_{a in ADD} sdf_a(p),  max_{s in SUB} -sdf_s(p) )

That flat form lets a per-tile broadphase keep only the few leaves whose bounding
sphere reaches a tile, instead of evaluating every node at every pixel. Scenes
whose operators don't flatten (intersection, smooth_union, differences nested
under a subtraction) are reported non-cullable and the renderer keeps the exact
full VM. This module is backend-agnostic: pure data + a CPU reference evaluator
used to prove the flattened field equals the tree.
"""

from dataclasses import dataclass

import numpy as np

from .gpu_node_types import is_operator
from .render_ir import RenderIR, RenderIRNode

_INF_RADIUS = 1.0e9  # sentinel "unbounded" leaf: never culled


@dataclass(frozen=True)
class CullPlan:
    """Flat additive/subtractive leaf index sets for a cullable scene."""

    add: tuple[int, ...]
    sub: tuple[int, ...]


def flatten_scene(render_ir: RenderIR | None) -> CullPlan | None:
    """Flatten a union/difference scene into ADD/SUB leaf sets, or None.

    Returns None when the operator graph cannot be expressed as
    ``max(min(ADD), max(-SUB))`` (intersection, smooth_union, or a difference
    appearing under an already-subtractive branch).
    """

    if render_ir is None or not render_ir.nodes or len(render_ir.root_indices) != 1:
        return None

    nodes = render_ir.nodes
    add: list[int] = []
    sub: list[int] = []
    ok = True

    # Iterative walk carrying a sign: +1 additive, -1 subtractive.
    stack: list[tuple[int, int]] = [(render_ir.root_indices[0], 1)]
    while stack:
        index, sign = stack.pop()
        node = nodes[index]
        if not is_operator(node.kind):
            (add if sign > 0 else sub).append(index)
            continue
        if node.kind == "union":
            for child in node.children:
                stack.append((int(child), sign))
        elif node.kind == "difference":
            if sign < 0:
                # difference under a subtraction does not flatten — bail out.
                ok = False
                break
            children = node.children
            if children:
                stack.append((int(children[0]), sign))
                for child in children[1:]:
                    stack.append((int(child), -sign))
        else:  # intersection, smooth_union, anything else
            ok = False
            break

    if not ok:
        return None
    return CullPlan(add=tuple(add), sub=tuple(sub))


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
    # profiles / sweeps / placed: unbounded (never culled).
    return (0.0, 0.0, 0.0, _INF_RADIUS)


def leaf_bounds(render_ir: RenderIR | None) -> np.ndarray:
    """(N, 4) float32 bounding spheres, one per IR node (operators -> unbounded)."""

    nodes = () if render_ir is None else render_ir.nodes
    out = np.zeros((max(len(nodes), 1), 4), dtype=np.float32)
    for i, node in enumerate(nodes):
        if is_operator(node.kind):
            out[i] = (0.0, 0.0, 0.0, _INF_RADIUS)
        else:
            out[i] = _leaf_bounding_sphere(node)
    return out


@dataclass(frozen=True)
class GridPlan:
    """A world-space uniform grid binning leaves into cells (design §13.5).

    ``origin`` + ``cell`` + ``dim`` define a regular GxGxG grid. For each cell,
    ``add_offsets``/``add_counts`` index into ``add_items`` (leaf node indices);
    likewise for sub. A leaf is binned into every cell its bounding sphere
    overlaps, so a DDA sphere-trace that re-fetches the cell each step and clamps
    steps to cell boundaries evaluates only nearby leaves yet stays exact.
    """

    origin: tuple[float, float, float]
    cell: tuple[float, float, float]
    dim: int
    add_offsets: np.ndarray  # int32 (dim^3,)
    add_counts: np.ndarray   # int32 (dim^3,)
    add_items: np.ndarray    # uint32 (total,)
    sub_offsets: np.ndarray
    sub_counts: np.ndarray
    sub_items: np.ndarray


def _bin_leaves(
    leaves: tuple[int, ...],
    bounds: np.ndarray,
    origin: np.ndarray,
    cell: np.ndarray,
    dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin leaves into the GxGxG grid by sphere-vs-cell-AABB overlap."""

    cells: list[list[int]] = [[] for _ in range(dim * dim * dim)]
    inv = 1.0 / cell
    for leaf in leaves:
        cx, cy, cz, r = bounds[leaf]
        center = np.array([cx, cy, cz])
        lo = np.floor((center - r - origin) * inv).astype(int)
        hi = np.floor((center + r - origin) * inv).astype(int)
        lo = np.clip(lo, 0, dim - 1)
        hi = np.clip(hi, 0, dim - 1)
        for iz in range(lo[2], hi[2] + 1):
            for iy in range(lo[1], hi[1] + 1):
                for ix in range(lo[0], hi[0] + 1):
                    cells[(iz * dim + iy) * dim + ix].append(int(leaf))
    counts = np.array([len(c) for c in cells], dtype=np.int32)
    offsets = np.zeros(dim * dim * dim, dtype=np.int32)
    if counts.size:
        offsets[1:] = np.cumsum(counts)[:-1]
    items = np.array([i for c in cells for i in c], dtype=np.uint32)
    if items.size == 0:
        items = np.zeros(1, dtype=np.uint32)
    return offsets, counts, items


def build_grid(
    plan: CullPlan,
    bounds: np.ndarray,
    dim: int = 16,
) -> GridPlan | None:
    """Build a uniform grid over the additive leaves' extent, or None.

    Returns None when any additive leaf is unbounded (sentinel radius) — such a
    scene can't be reliably gridded, so the renderer keeps the full VM.
    """

    if not plan.add:
        return None
    add = np.asarray(plan.add, dtype=np.int64)
    add_b = bounds[add]
    if np.any(add_b[:, 3] >= _INF_RADIUS):
        return None  # unbounded additive leaf -> not griddable
    lo = np.min(add_b[:, :3] - add_b[:, 3:4], axis=0)
    hi = np.max(add_b[:, :3] + add_b[:, 3:4], axis=0)
    span = np.maximum(hi - lo, 1e-4)
    pad = span * 0.01
    origin = lo - pad
    cell = (span + 2 * pad) / dim

    ao, ac, ai = _bin_leaves(plan.add, bounds, origin, cell, dim)
    so, sc, si = _bin_leaves(plan.sub, bounds, origin, cell, dim)
    return GridPlan(
        origin=tuple(float(v) for v in origin),
        cell=tuple(float(v) for v in cell),
        dim=dim,
        add_offsets=ao, add_counts=ac, add_items=ai,
        sub_offsets=so, sub_counts=sc, sub_items=si,
    )


def combine_flat(
    plan: CullPlan,
    leaf_distance: dict[int, np.ndarray],
) -> np.ndarray:
    """Evaluate ``max(min(ADD), max(-SUB))`` given each leaf's distances.

    ``leaf_distance`` maps node index -> per-point distances. Used by parity
    tests to compare the flattened field against the full VM.
    """

    first = next(iter(leaf_distance.values()))
    if plan.add:
        dist = leaf_distance[plan.add[0]].copy()
        for idx in plan.add[1:]:
            dist = np.minimum(dist, leaf_distance[idx])
    else:
        dist = np.full_like(first, 1.0e6)
    for idx in plan.sub:
        dist = np.maximum(dist, -leaf_distance[idx])
    return dist


__all__ = [
    "CullPlan", "GridPlan", "flatten_scene", "leaf_bounds", "build_grid",
    "combine_flat", "_INF_RADIUS",
]
