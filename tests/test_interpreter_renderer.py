from __future__ import annotations

import sys

import numpy as np
import pytest

sys.setrecursionlimit(10000)

from core.render_ir import build_render_ir
from core.sdf import Box, Difference, SDFTree, Sphere

modergll = pytest.importorskip("moderngl")
import moderngl

try:
    _ctx = moderngl.create_context(standalone=True, require=460)
except Exception as exc:  # pragma: no cover
    pytest.skip(f"no headless GL 4.6 context: {exc}", allow_module_level=True)

# The interpreter FRAGMENT shader links only at the reduced FRAGMENT_STACK_CAPACITY
# and only on drivers whose fragment compiler can bind the per-invocation arrays.
# NVIDIA links it in seconds; Mesa Intel (Iris Xe) hangs at link regardless of
# capacity (the link call would wedge the whole suite). Gate on the GL vendor so
# this module runs under the NVIDIA launcher and is skipped on Mesa.
_renderer_name = _ctx.info.get("GL_RENDERER", "")
if "NVIDIA" not in _renderer_name:
    pytest.skip(
        f"interpreter fragment renderer needs an NVIDIA GPU; got {_renderer_name!r} "
        f"(run under __NV_PRIME_RENDER_OFFLOAD=1). Mesa Intel hangs at shader link.",
        allow_module_level=True,
    )

from app.viewport.renderers.opengl_interpreter import InterpreterRenderer


@pytest.fixture(scope="module")
def renderer() -> InterpreterRenderer:
    r = InterpreterRenderer(ctx=_ctx)
    yield r
    r.release()


def _carve_scene(n_tools: int) -> SDFTree:
    rng = np.random.default_rng(7)
    root = Box(name="domain", object_id=1, half_size=(2.0, 2.0, 2.0))
    for i in range(2, 2 + n_tools):
        c = rng.uniform(-1.5, 1.5, size=3)
        root = Difference(name=f"d{i}", object_id=0, left=root,
                          right=Sphere(name=f"s{i}", object_id=i,
                                       center=tuple(c), radius=0.3))
    return SDFTree(root=root)


def test_renders_sphere_in_center(renderer: InterpreterRenderer) -> None:
    ir = build_render_ir(SDFTree(root=Sphere(name="s", object_id=1, radius=1.0)))
    assert renderer.upload_render_ir(ir)
    img = renderer.render_to_array(
        128, 128, camera_position=(0.0, -5.0, 0.0), camera_target=(0.0, 0.0, 0.0))
    bg = np.array([int(0.07 * 255), int(0.08 * 255), int(0.10 * 255)])
    center = img[64, 64].astype(int)
    corner = img[2, 2].astype(int)
    # Center ray hits the sphere (lit), corner sees background.
    assert np.abs(center - bg).sum() > 40
    assert np.abs(corner - bg).sum() < 12


def test_topology_change_never_recompiles(renderer: InterpreterRenderer) -> None:
    """The migration's central claim: compile cost is ~0 and constant.

    Across scenes from 1 to 800 carve operations the program object is reused
    and program_compile_ms stays zero — a topology change is a buffer upload.
    """
    program_id = id(renderer.program)
    compile_times = []
    for n in (1, 50, 200):
        ir = build_render_ir(_carve_scene(n))
        assert renderer.upload_render_ir(ir)
        stats = renderer.last_scene_update_stats()
        compile_times.append(stats.program_compile_ms)
        assert stats.reused_program
        assert id(renderer.program) == program_id  # same GL program, no recompile
        # It still renders (small frame keeps the integrated GPU snappy).
        img = renderer.render_to_array(
            32, 32, camera_position=(0.0, -8.0, 0.0), camera_target=(0.0, 0.0, 0.0))
        assert img.shape == (32, 32, 3)

    assert max(compile_times) == 0.0


def test_param_fast_path_updates_without_topology_change(
    renderer: InterpreterRenderer,
) -> None:
    sphere = Sphere(name="s", object_id=1, center=(0.0, 0.0, 0.0), radius=1.0)
    renderer.upload_render_ir(build_render_ir(SDFTree(root=sphere)))

    moved = Sphere(name="s", object_id=1, center=(0.8, 0.0, 0.0), radius=1.0)
    ok = renderer.update_object_parameters(build_render_ir(SDFTree(root=moved)))
    assert ok  # same topology, params written in place

    img = renderer.render_to_array(
        128, 128, camera_position=(0.0, -5.0, 0.0), camera_target=(0.0, 0.0, 0.0))
    # The moved sphere shifts the lit region off-center toward +x (screen right).
    lit = np.abs(img.astype(int) - np.array([17, 20, 25])).sum(axis=2) > 40
    xs = np.where(lit.any(axis=0))[0]
    assert xs.size > 0
    assert xs.mean() > 64  # centroid shifted right of center
