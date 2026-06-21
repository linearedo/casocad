from __future__ import annotations

import sys

import numpy as np
import pytest

sys.setrecursionlimit(10000)

from core.gpu_cull import (
    build_grid,
    combine_flat,
    flatten_scene,
    leaf_bounds,
    _INF_RADIUS,
)
from core.render_ir import build_render_ir
from core.sdf import (
    Box,
    Difference,
    Intersection,
    SDFTree,
    SmoothUnion,
    Sphere,
    Union,
)


def _ir(root):
    return build_render_ir(SDFTree(root=root))


# ---- flatten classification (pure CPU) -------------------------------------

def test_union_chain_is_all_additive() -> None:
    root = Union(name="u", object_id=0,
                 left=Union(name="u2", object_id=0,
                            left=Sphere(name="a", object_id=1, radius=1.0),
                            right=Sphere(name="b", object_id=2, radius=1.0)),
                 right=Sphere(name="c", object_id=3, radius=1.0))
    ir = _ir(root)
    plan = flatten_scene(ir)
    assert plan is not None
    add_owners = sorted(ir.nodes[i].object_id for i in plan.add)
    assert add_owners == [1, 2, 3]
    assert plan.sub == ()


def test_difference_chain_splits_add_sub() -> None:
    root = Difference(name="d2", object_id=0,
                      left=Difference(name="d1", object_id=0,
                                      left=Box(name="box", object_id=1,
                                               half_size=(2.0, 2.0, 2.0)),
                                      right=Sphere(name="s1", object_id=2, radius=0.5)),
                      right=Sphere(name="s2", object_id=3, radius=0.5))
    ir = _ir(root)
    plan = flatten_scene(ir)
    assert plan is not None
    assert sorted(ir.nodes[i].object_id for i in plan.add) == [1]
    assert sorted(ir.nodes[i].object_id for i in plan.sub) == [2, 3]


def test_intersection_and_smooth_union_are_not_cullable() -> None:
    a = Sphere(name="a", object_id=1, radius=1.0)
    b = Sphere(name="b", object_id=2, center=(0.5, 0, 0), radius=1.0)
    assert flatten_scene(_ir(Intersection(name="i", object_id=0, left=a, right=b))) is None
    assert flatten_scene(
        _ir(SmoothUnion(name="su", object_id=0, left=a, right=b, smoothing=0.3))
    ) is None


def test_difference_under_subtraction_is_not_cullable() -> None:
    # Difference(A, Difference(B, C)) does not flatten to max(min(ADD), max(-SUB)).
    a = Box(name="A", object_id=1, half_size=(2.0, 2.0, 2.0))
    b = Sphere(name="B", object_id=2, radius=1.0)
    c = Sphere(name="C", object_id=3, radius=0.5)
    root = Difference(name="outer", object_id=0,
                      left=a, right=Difference(name="inner", object_id=0, left=b, right=c))
    assert flatten_scene(_ir(root)) is None


# ---- bounds conservativeness (CPU via to_numpy) ----------------------------

@pytest.mark.parametrize("node", [
    Sphere(name="s", object_id=1, center=(0.3, -0.2, 0.1), radius=1.4),
    Box(name="b", object_id=1, center=(0.1, 0, -0.2), half_size=(1.2, 0.8, 0.6)),
])
def test_bounds_contain_interior(node) -> None:
    ir = _ir(node)
    bounds = leaf_bounds(ir)
    cx, cy, cz, r = bounds[0]
    assert r < _INF_RADIUS
    rng = np.random.default_rng(0)
    pts = rng.uniform(-3, 3, size=(20000, 3))
    inside = node.to_numpy(pts[:, 0], pts[:, 1], pts[:, 2]) <= 0.0
    d = np.linalg.norm(pts[inside] - np.array([cx, cy, cz]), axis=1)
    assert np.all(d <= r + 1e-4)


# ---- world grid binning invariant (CPU) ------------------------------------

def test_grid_bins_cover_every_overlapping_leaf() -> None:
    # Correctness invariant: any leaf whose bounding sphere contains a point must
    # be present in that point's grid cell (necessary for an exact DDA march).
    rng = np.random.default_rng(3)
    root = Box(name="domain", object_id=1, half_size=(2.5, 2.5, 2.5))
    for i in range(2, 24):
        c = rng.uniform(-2.0, 2.0, size=3)
        root = Difference(name=f"d{i}", object_id=0, left=root,
                          right=Sphere(name=f"s{i}", object_id=i,
                                       center=tuple(c), radius=0.4))
    ir = _ir(root)
    plan = flatten_scene(ir)
    bounds = leaf_bounds(ir)
    grid = build_grid(plan, bounds, dim=8)
    assert grid is not None

    origin = np.array(grid.origin)
    cell = np.array(grid.cell)
    pts = rng.uniform(-2.5, 2.5, size=(3000, 3))
    for p in pts[:300]:
        ci = np.floor((p - origin) / cell).astype(int)
        if np.any(ci < 0) or np.any(ci >= grid.dim):
            continue
        flat = (ci[2] * grid.dim + ci[1]) * grid.dim + ci[0]
        cell_subs = set(
            grid.sub_items[grid.sub_offsets[flat]:
                           grid.sub_offsets[flat] + grid.sub_counts[flat]].tolist()
        )
        for s in plan.sub:
            cx, cy, cz, r = bounds[s]
            if np.linalg.norm(p - np.array([cx, cy, cz])) <= r:
                assert s in cell_subs, "overlapping leaf missing from cell"


def test_grid_rebins_when_object_moves() -> None:
    # The move bug: a moved object must land in different cells after a rebuild,
    # otherwise it's evaluated at its old location (appears not to move).
    def grid_for(cx):
        root = Union(name="u", object_id=0,
                     left=Sphere(name="a", object_id=1, center=(0.0, 0.0, 0.0), radius=0.4),
                     right=Sphere(name="b", object_id=2, center=(cx, 0.0, 0.0), radius=0.4))
        ir = _ir(root)
        plan = flatten_scene(ir)
        grid = build_grid(plan, leaf_bounds(ir), dim=16)
        # node index of sphere "b" (object_id 2)
        b_idx = next(i for i, n in enumerate(ir.nodes) if n.object_id == 2)
        cells_with_b = {
            c for c in range(grid.dim ** 3)
            if b_idx in grid.add_items[grid.add_offsets[c]:
                                       grid.add_offsets[c] + grid.add_counts[c]].tolist()
        }
        return cells_with_b

    # Cell indices aren't comparable across grids of different extent, so just
    # assert the binning changed when the object moved (end-to-end move
    # correctness is covered by the offscreen render test).
    near = grid_for(0.5)
    far = grid_for(3.0)
    assert near and far
    assert near != far, "moved object must re-bin to different cells"


def test_grid_none_when_additive_leaf_unbounded() -> None:
    # An extrude (profile) additive leaf is unbounded -> not griddable.
    from core.sdf import Extrude, PlacedSDF2D, RectangleProfile
    root = Extrude(name="ex", object_id=1,
                   section=PlacedSDF2D(name="sec", object_id=2,
                                       profile=RectangleProfile(half_size=(0.7, 0.5)),
                                       origin=(0, 0, 0), axis_u=(1, 0, 0), axis_v=(0, 1, 0)),
                   height=1.0)
    ir = _ir(root)
    plan = flatten_scene(ir)
    assert plan is not None
    assert build_grid(plan, leaf_bounds(ir), dim=8) is None


# ---- flattened field == full VM (GPU) --------------------------------------

moderngl = pytest.importorskip("moderngl")
try:
    _ctx = moderngl.create_context(standalone=True, require=460)
except Exception as exc:  # pragma: no cover
    pytest.skip(f"no headless GL 4.6 context: {exc}", allow_module_level=True)

from app.viewport.renderers.opengl_interpreter import SdfEvaluator


@pytest.fixture(scope="module")
def evaluator() -> SdfEvaluator:
    ev = SdfEvaluator(ctx=_ctx)
    yield ev
    ev.release()


def test_flattened_field_matches_full_vm(evaluator: SdfEvaluator) -> None:
    rng = np.random.default_rng(7)
    root = Box(name="domain", object_id=1, half_size=(2.5, 2.5, 2.5))
    for i in range(2, 30):
        c = rng.uniform(-2.0, 2.0, size=3)
        root = Difference(name=f"d{i}", object_id=0, left=root,
                          right=Sphere(name=f"s{i}", object_id=i,
                                       center=tuple(c), radius=0.3))
    ir = _ir(root)
    plan = flatten_scene(ir)
    assert plan is not None

    evaluator.upload_render_ir(ir)
    pts = rng.uniform(-3.0, 3.0, size=(6000, 3))
    full = evaluator.evaluate_scene(pts).dist

    leaf_d = {idx: evaluator.evaluate_leaf(idx, pts).dist for idx in (*plan.add, *plan.sub)}
    flat = combine_flat(plan, leaf_d)

    np.testing.assert_allclose(flat, full, atol=2e-4)
