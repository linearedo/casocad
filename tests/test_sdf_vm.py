from __future__ import annotations

import sys

import numpy as np
import pytest

# The GPU interpreter is iterative; only the recursive CPU oracle (SDFTree walk
# / to_numpy) needs head-room to validate a deep carve chain.
sys.setrecursionlimit(10000)

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


def _points(seed: int, n: int = 8192) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-3.0, 3.0, size=(n, 3)).astype(np.float64)


def _oracle(node, points: np.ndarray) -> np.ndarray:
    return node.to_numpy(points[:, 0], points[:, 1], points[:, 2])


def test_union_distance_and_owner(evaluator: SdfEvaluator) -> None:
    a = Sphere(name="a", object_id=1, center=(-0.7, 0, 0), radius=1.0)
    b = Sphere(name="b", object_id=2, center=(0.8, 0, 0), radius=1.1)
    root = Union(name="u", object_id=0, left=a, right=b)
    ir = build_render_ir(SDFTree(root=root))
    evaluator.upload_render_ir(ir)

    pts = _points(1)
    res = evaluator.evaluate_scene(pts)
    np.testing.assert_allclose(res.dist, _oracle(root, pts), atol=2e-4)

    # union keeps the nearer operand's owner.
    da = _oracle(a, pts)
    db = _oracle(b, pts)
    expected_owner = np.where(da <= db, 1, 2)
    assert np.all(res.owner == expected_owner)


def test_intersection_distance(evaluator: SdfEvaluator) -> None:
    a = Box(name="a", object_id=1, half_size=(1.2, 1.2, 1.2))
    b = Sphere(name="b", object_id=2, radius=1.5)
    root = Intersection(name="i", object_id=0, left=a, right=b)
    ir = build_render_ir(SDFTree(root=root))
    evaluator.upload_render_ir(ir)
    pts = _points(2)
    res = evaluator.evaluate_scene(pts)
    np.testing.assert_allclose(res.dist, _oracle(root, pts), atol=2e-4)


def test_difference_threads_tool_owner_on_cavity_wall(evaluator: SdfEvaluator) -> None:
    solid = Box(name="solid", object_id=1, half_size=(1.5, 1.5, 1.5))
    tool = Sphere(name="tool", object_id=2, center=(0.0, 0.0, 0.0), radius=1.0)
    root = Difference(name="d", object_id=0, left=solid, right=tool)
    ir = build_render_ir(SDFTree(root=root))
    evaluator.upload_render_ir(ir)

    pts = _points(3)
    res = evaluator.evaluate_scene(pts)
    np.testing.assert_allclose(res.dist, _oracle(root, pts), atol=2e-4)

    # Where the carving tool wins (-tool dominates the box term), the cavity
    # wall must inherit the tool's owner id (design §6 SDF threading).
    box_d = _oracle(solid, pts)
    carved = -_oracle(tool, pts)
    tool_wins = carved > box_d
    assert np.all(res.owner[tool_wins] == 2)
    assert np.all(res.owner[~tool_wins] == 1)


def test_smooth_union_distance(evaluator: SdfEvaluator) -> None:
    a = Sphere(name="a", object_id=1, center=(-0.6, 0, 0), radius=0.9)
    b = Sphere(name="b", object_id=2, center=(0.6, 0, 0), radius=0.9)
    root = SmoothUnion(name="su", object_id=0, left=a, right=b, smoothing=0.4)
    ir = build_render_ir(SDFTree(root=root))
    evaluator.upload_render_ir(ir)
    pts = _points(4)
    res = evaluator.evaluate_scene(pts)
    np.testing.assert_allclose(res.dist, _oracle(root, pts), atol=2e-4)


def test_thousand_op_carve_chain(evaluator: SdfEvaluator) -> None:
    # A long flat difference chain: the case the design must handle effortlessly.
    rng = np.random.default_rng(99)
    root = Box(name="base", object_id=1, half_size=(2.5, 2.5, 2.5))
    for i in range(2, 502):  # ~1001 bytecode ops (500 tools: push + eval each)
        c = rng.uniform(-2.0, 2.0, size=3)
        tool = Sphere(name=f"t{i}", object_id=i,
                      center=tuple(c), radius=0.18)
        root = Difference(name=f"d{i}", object_id=0, left=root, right=tool)
    ir = build_render_ir(SDFTree(root=root))
    evaluator.upload_render_ir(ir)

    pts = _points(5, n=16384)
    res = evaluator.evaluate_scene(pts)
    np.testing.assert_allclose(res.dist, _oracle(root, pts), atol=2e-4)
