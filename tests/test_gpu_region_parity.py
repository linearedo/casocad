from __future__ import annotations

import numpy as np
import pytest

from core.boundary_patches import PATCH_TOLERANCE, surface_selector_values
from core.gpu_selector import (
    RegionSelectorSpec,
    attach_region_selectors,
    selector_volume_ir,
)
from core.render_ir import build_render_ir
from core.sdf import (
    Box,
    CircleProfile,
    PlacedPolyline2D,
    PlacedSDF1D,
    PlacedSDF2D,
    PolylineProfile,
    RectangleProfile,
    SDFTree,
    SegmentProfile,
    Sphere,
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


def _domain() -> Box:
    return Box(name="domain", object_id=1, center=(0.0, 0.0, 0.0),
               half_size=(1.5, 1.5, 1.5))


def _selectors() -> dict[str, object]:
    return {
        # full 3D SDF selector
        "sdf3d": Sphere(name="sel3d", object_id=2,
                        center=(0.5, 0.0, 0.0), radius=0.7),
        # extruded 2D profile (Surface Cutter)
        "extrude2d": PlacedSDF2D(
            name="selsurf", object_id=3,
            profile=CircleProfile(center=(0.0, 0.0), radius=0.6),
            origin=(0.0, 0.0, 0.0), axis_u=(1.0, 0.0, 0.0), axis_v=(0.0, 1.0, 0.0)),
        # offset polyline band
        "polyline": PlacedPolyline2D(
            name="selband", object_id=4,
            profile=PolylineProfile(points=((-0.8, -0.3), (0.0, 0.4), (0.8, -0.2))),
            origin=(0.0, 0.0, 0.0), axis_u=(1.0, 0.0, 0.0), axis_v=(0.0, 1.0, 0.0)),
        # oriented segment slab (Planar Cutter)
        "segment": PlacedSDF1D(
            name="selslab", object_id=5,
            profile=SegmentProfile(center=0.0, half_length=0.4),
            origin=(0.0, 0.0, 0.0), axis_u=(1.0, 0.0, 0.0)),
    }


def _points(seed: int, n: int = 8192) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-2.0, 2.0, size=(n, 3)).astype(np.float64)


@pytest.mark.parametrize("name", list(_selectors()))
def test_selector_volume_matches_cpu(evaluator: SdfEvaluator, name: str) -> None:
    """GPU evaluation of the selector volume == CPU surface_selector_values."""
    root = _domain()
    selector = _selectors()[name]

    volume_ir = selector_volume_ir(root, selector)
    assert volume_ir is not None and volume_ir.supported
    evaluator.upload_render_ir(volume_ir)

    pts = _points(seed=hash(name) & 0xFFFF)
    gpu = evaluator.evaluate_scene(pts).dist
    cpu = surface_selector_values(root, selector, pts)
    np.testing.assert_allclose(gpu, cpu, atol=2e-4)


@pytest.mark.parametrize("name", list(_selectors()))
@pytest.mark.parametrize("side", ["inside", "outside"])
def test_region_assign_matches_cpu_mask(
    evaluator: SdfEvaluator, name: str, side: str
) -> None:
    """End-to-end REGION_ASSIGN region_id == CPU inside/outside mask, both sides."""
    root = _domain()
    selector = _selectors()[name]
    region_id = 777

    scene_ir = build_render_ir(SDFTree(root=root))
    spec = RegionSelectorSpec(
        base_owner_id=1, region_id=region_id, selector=selector, side=side,
        tolerance=PATCH_TOLERANCE,
    )
    augmented = attach_region_selectors(scene_ir, root, (spec,))
    evaluator.upload_render_ir(augmented)

    pts = _points(seed=(hash(name) ^ hash(side)) & 0xFFFF)
    res = evaluator.evaluate_scene(pts)

    # The domain is a single box: owner is 1 everywhere, so the region is
    # assigned wherever the inside/outside test passes.
    values = surface_selector_values(root, selector, pts)
    if side == "inside":
        cpu_in = values <= PATCH_TOLERANCE
    else:
        cpu_in = values > PATCH_TOLERANCE

    expected = np.where(cpu_in, region_id, 0).astype(np.uint32)
    # Owner threading must be intact: distance still equals the bare box.
    np.testing.assert_array_equal(res.owner, np.ones_like(res.owner))
    # Allow a hair of disagreement only on points within float32 epsilon of the
    # tolerance band edge.
    near_edge = np.abs(values - PATCH_TOLERANCE) < 5e-5
    mismatches = (res.region != expected) & ~near_edge
    assert not np.any(mismatches), int(np.count_nonzero(mismatches))
