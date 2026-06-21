from __future__ import annotations

import numpy as np
import pytest

from core.render_ir import build_render_ir
from core.sdf import (
    Box,
    BoxFrame,
    CappedCone,
    Cone,
    Cylinder,
    Pyramid,
    SDFTree,
    Sphere,
    Torus,
)

moderngl = pytest.importorskip("moderngl")

try:
    _ctx = moderngl.create_context(standalone=True, require=460)
except Exception as exc:  # pragma: no cover - environment without GL 4.6
    pytest.skip(f"no headless GL 4.6 context: {exc}", allow_module_level=True)

from app.viewport.renderers.opengl_interpreter import SdfEvaluator


@pytest.fixture(scope="module")
def evaluator() -> SdfEvaluator:
    ev = SdfEvaluator(ctx=_ctx)
    yield ev
    ev.release()


def _sample_points(seed: int, n: int = 4096) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-3.0, 3.0, size=(n, 3)).astype(np.float64)


def _oracle(node, points: np.ndarray) -> np.ndarray:
    return node.to_numpy(points[:, 0], points[:, 1], points[:, 2])


_PRIMITIVES = [
    Sphere(name="s", object_id=7, center=(0.3, -0.2, 0.1), radius=1.4),
    Box(name="b", object_id=8, center=(0.1, 0.0, -0.2),
        half_size=(1.2, 0.8, 0.6)),
    Cylinder(name="c", object_id=9, center=(0.0, 0.0, 0.0),
             radius=0.9, half_height=1.1),
    Cone(name="co", object_id=10, center=(0.0, 0.0, 0.0),
         radius=1.0, half_height=1.2),
    CappedCone(name="cc", object_id=11, center=(0.0, 0.0, 0.0),
               radius_a=1.1, radius_b=0.4, half_height=1.0),
    BoxFrame(name="bf", object_id=12, center=(0.0, 0.0, 0.0),
             half_size=(1.0, 0.9, 0.8), thickness=0.15),
    Pyramid(name="py", object_id=13, center=(0.0, 0.0, 0.0),
            base_half_size=1.0, half_height=1.3),
    Torus(name="to", object_id=14, center=(0.0, 0.0, 0.0),
          major_radius=1.0, minor_radius=0.3),
]


@pytest.mark.parametrize("node", _PRIMITIVES, ids=lambda n: n.kind)
def test_leaf_sdf_matches_to_numpy(evaluator: SdfEvaluator, node) -> None:
    ir = build_render_ir(SDFTree(root=node))
    assert ir.supported and len(ir.nodes) == 1
    evaluator.upload_render_ir(ir)

    points = _sample_points(seed=hash(node.kind) & 0xFFFF)
    result = evaluator.evaluate_leaf(0, points)
    expected = _oracle(node, points)

    np.testing.assert_allclose(result.dist, expected, atol=1.5e-4, rtol=0)
    # Leaf owner id flows through; pure primitives are a single region.
    assert np.all(result.owner == node.object_id)
    assert np.all(result.region == 0)
