from __future__ import annotations

import numpy as np
import pytest

from core.gpu_node_types import PROFILE_1D_KINDS, PROFILE_2D_KINDS
from core.render_ir import build_render_ir
from core.sdf import (
    BezierTube,
    BinaryProfile,
    CircleProfile,
    DistanceOffsetProfile,
    EllipseProfile,
    Extrude,
    PlacedSDF1D,
    PlacedSDF2D,
    PolygonProfile,
    PolylineProfile,
    PolylineTube,
    RectangleProfile,
    Revolve,
    RoundedRectangleProfile,
    SDFTree,
    SegmentProfile,
    BinaryProfile1D,
    SquareProfile,
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


def _find_kind(ir, kind: str) -> int:
    for index, node in enumerate(ir.nodes):
        if node.kind == kind:
            return index
    raise AssertionError(f"no {kind} node in IR")


def _profile_root_2d(ir, leaf_kind: str) -> int:
    leaf = ir.nodes[_find_kind(ir, leaf_kind)]
    return leaf.children[0]


def _placed_section(profile) -> PlacedSDF2D:
    return PlacedSDF2D(
        name="sec", object_id=1, profile=profile,
        origin=(0.0, 0.0, 0.0), axis_u=(1.0, 0.0, 0.0), axis_v=(0.0, 1.0, 0.0),
    )


# --- 2D profile sub-VM parity (Phase B) -------------------------------------

_PROFILES_2D = {
    "circle": CircleProfile(center=(0.1, -0.2), radius=0.9),
    "rectangle": RectangleProfile(center=(0.0, 0.1), half_size=(0.8, 0.5)),
    "square": SquareProfile(center=(0.0, 0.0), half_size=0.7),
    "rounded_rect": RoundedRectangleProfile(
        center=(0.0, 0.0), half_size=(0.8, 0.6), corner_radius=0.2),
    "ellipse": EllipseProfile(center=(0.0, 0.0), semi_axes=(0.9, 0.5)),
    "polygon": PolygonProfile(points=((-0.6, -0.5), (0.7, -0.4),
                                      (0.5, 0.6), (-0.5, 0.5))),
    "polyline": PolylineProfile(points=((-0.8, -0.3), (0.0, 0.4), (0.8, -0.2))),
    "union": BinaryProfile(
        left=CircleProfile(center=(-0.3, 0.0), radius=0.6),
        right=RectangleProfile(center=(0.3, 0.0), half_size=(0.5, 0.4)),
        operation="union"),
    "difference": BinaryProfile(
        left=RectangleProfile(center=(0.0, 0.0), half_size=(0.8, 0.8)),
        right=CircleProfile(center=(0.0, 0.0), radius=0.5),
        operation="difference"),
    "smooth_union": BinaryProfile(
        left=CircleProfile(center=(-0.3, 0.0), radius=0.5),
        right=CircleProfile(center=(0.3, 0.0), radius=0.5),
        operation="smooth_union", smoothing=0.3),
    "distance_offset": DistanceOffsetProfile(
        child=CircleProfile(center=(0.0, 0.0), radius=0.6), offset=0.15),
}


@pytest.mark.parametrize("name", list(_PROFILES_2D))
def test_profile_2d_vm_matches_to_numpy(evaluator: SdfEvaluator, name: str) -> None:
    profile = _PROFILES_2D[name]
    ir = build_render_ir(SDFTree(root=Extrude(
        name="ex", object_id=2, section=_placed_section(profile), height=1.0)))
    root = _profile_root_2d(ir, "extrude_profile_2d")
    assert ir.nodes[root].kind in PROFILE_2D_KINDS
    evaluator.upload_render_ir(ir)

    rng = np.random.default_rng(hash(name) & 0xFFFF)
    q = rng.uniform(-1.5, 1.5, size=(6000, 2))
    got = evaluator.evaluate_profile_2d(root, q)
    expected = profile.to_numpy(q[:, 0], q[:, 1])
    # The exact-ellipse formula loses float32 precision right at its zero level
    # set; everywhere else parity is tight.
    atol = 1.5e-3 if name == "ellipse" else 2e-4
    np.testing.assert_allclose(got, expected, atol=atol)


# --- 1D profile sub-VM parity ------------------------------------------------

def test_profile_1d_vm_matches_to_numpy(evaluator: SdfEvaluator) -> None:
    profile = BinaryProfile1D(
        left=SegmentProfile(center=-0.4, half_length=0.5),
        right=SegmentProfile(center=0.5, half_length=0.3),
        operation="union")
    placed = PlacedSDF1D(name="p1", object_id=1, profile=profile,
                         origin=(0.0, 0.0, 0.0), axis_u=(1.0, 0.0, 0.0))
    ir = build_render_ir(SDFTree(root=placed))
    leaf = ir.nodes[_find_kind(ir, "placed_profile_1d")]
    root = leaf.children[0]
    assert ir.nodes[root].kind in PROFILE_1D_KINDS
    evaluator.upload_render_ir(ir)

    t = np.linspace(-2.0, 2.0, 4000)
    got = evaluator.evaluate_profile_1d(root, t)
    np.testing.assert_allclose(got, profile.to_numpy(t), atol=2e-4)


# --- genuine 3D sweeps: full evalSceneSDF vs to_numpy ------------------------

def _points(seed: int, n: int = 8192) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-2.5, 2.5, size=(n, 3)).astype(np.float64)


def _oracle(node, pts):
    return node.to_numpy(pts[:, 0], pts[:, 1], pts[:, 2])


def test_extrude_3d(evaluator: SdfEvaluator) -> None:
    section = _placed_section(RectangleProfile(center=(0.0, 0.0),
                                               half_size=(0.7, 0.5)))
    root = Extrude(name="ex", object_id=3, section=section,
                   height=1.4, center_offset=0.2)
    ir = build_render_ir(SDFTree(root=root))
    evaluator.upload_render_ir(ir)
    pts = _points(1)
    res = evaluator.evaluate_scene(pts)
    np.testing.assert_allclose(res.dist, _oracle(root, pts), atol=2e-4)
    assert np.all(res.owner == 3)


@pytest.mark.parametrize("angle", [360.0, 270.0])
def test_revolve_3d(evaluator: SdfEvaluator, angle: float) -> None:
    section = _placed_section(CircleProfile(center=(1.2, 0.0), radius=0.35))
    root = Revolve(name="rev", object_id=4, section=section,
                   axis="v", angle_degrees=angle)
    ir = build_render_ir(SDFTree(root=root))
    evaluator.upload_render_ir(ir)
    pts = _points(2)
    res = evaluator.evaluate_scene(pts)
    np.testing.assert_allclose(res.dist, _oracle(root, pts), atol=2e-4)


@pytest.mark.parametrize("caps", ["round", "flat"])
def test_polyline_tube_3d(evaluator: SdfEvaluator, caps: str) -> None:
    root = PolylineTube(name="pt", object_id=5,
                        points=((-1.0, 0.0, 0.0), (0.0, 0.8, 0.0), (1.0, 0.0, 0.5)),
                        radius=0.3, inner_radius=0.0, caps=caps)
    ir = build_render_ir(SDFTree(root=root))
    evaluator.upload_render_ir(ir)
    pts = _points(3)
    res = evaluator.evaluate_scene(pts)
    np.testing.assert_allclose(res.dist, _oracle(root, pts), atol=2e-4)


@pytest.mark.parametrize("caps", ["round", "flat"])
def test_bezier_tube_3d(evaluator: SdfEvaluator, caps: str) -> None:
    root = BezierTube(name="bt", object_id=6,
                      points=((-1.0, 0.0, 0.0), (-0.3, 1.0, 0.0),
                              (0.4, 0.0, 0.0), (1.0, -0.6, 0.4), (1.6, 0.0, 0.4)),
                      radius=0.25, inner_radius=0.0, caps=caps)
    ir = build_render_ir(SDFTree(root=root))
    evaluator.upload_render_ir(ir)
    pts = _points(4)
    res = evaluator.evaluate_scene(pts)
    np.testing.assert_allclose(res.dist, _oracle(root, pts), atol=2e-4)
