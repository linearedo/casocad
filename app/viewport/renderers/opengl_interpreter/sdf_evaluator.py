from __future__ import annotations

"""Off-screen compute-shader SDF sampler used as the interpreter's test oracle.

This runs the *same* interpreter GLSL the on-screen renderer uses, but as a
compute dispatch over an arbitrary batch of points, so tests can diff the GPU
result against ``SDFNode.to_numpy`` (geometry) and ``surface_selector_values``
(Layer 2 regions). Keeping the validation path on the real shader means a parity
regression is caught here rather than on screen (design §7.1).
"""

from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np

from core.gpu_features import FULL_FEATURES
from core.gpu_program import GpuProgram, emit_program
from core.gpu_scene import GpuSceneBuffers, serialize_scene
from core.render_ir import RenderIR

from .shader_assembly import build_program_source

_SHADER_DIR = Path(__file__).parent / "shaders"

# SSBO binding points — must match sdf_core.glsl / sdf_eval.comp.
_BIND_NODES = 0
_BIND_PARAMS = 1
_BIND_CHILDREN = 2
_BIND_BYTECODE = 3
_BIND_POINTS = 4
_BIND_OUT_DIST = 5
_BIND_OUT_OWNER = 6
_BIND_OUT_REGION = 7

_LOCAL_SIZE = 64

_MODE_LEAF = 0
_MODE_SCENE = 1
_MODE_PROFILE_2D = 2
_MODE_PROFILE_1D = 3


@dataclass(frozen=True)
class SdfEvalResult:
    dist: np.ndarray   # float32, shape (N,)
    owner: np.ndarray  # uint32, shape (N,)
    region: np.ndarray  # uint32, shape (N,)


def _build_source() -> str:
    # Validation oracle: full feature set at the design cap (compute compiles it
    # fine on every GPU — only the fragment/raymarch forms hit the Mesa wall).
    comp_main = (_SHADER_DIR / "sdf_eval.comp").read_text()
    return build_program_source(FULL_FEATURES, comp_main)


class SdfEvaluator:
    """Compiles the interpreter compute shader and evaluates point batches.

    Pass an existing ``moderngl.Context`` or let it create a standalone headless
    GL 4.6 context (design §4 — GL 4.6 is the hard floor).
    """

    def __init__(self, ctx: moderngl.Context | None = None) -> None:
        self._owns_ctx = ctx is None
        self.ctx = ctx or moderngl.create_context(standalone=True, require=460)
        self.program = self.ctx.compute_shader(_build_source())
        self._buffers: dict[str, moderngl.Buffer] = {}

    # -- scene upload --------------------------------------------------------

    def upload(self, scene: GpuSceneBuffers, program: GpuProgram) -> None:
        self._release_buffers()
        self._buffers["nodes"] = self.ctx.buffer(scene.nodes_bytes)
        self._buffers["params"] = self.ctx.buffer(scene.params_bytes)
        self._buffers["children"] = self.ctx.buffer(scene.children_bytes)
        # Bytecode may be empty; reserve a minimal slot so the binding is valid.
        self._buffers["bytecode"] = self.ctx.buffer(
            program.bytecode_bytes or np.zeros(1, dtype=np.uint32).tobytes()
        )
        self._program_length = program.program_length

    def upload_render_ir(self, render_ir: RenderIR) -> None:
        self.upload(serialize_scene(render_ir), emit_program(render_ir))

    # -- evaluation ----------------------------------------------------------

    def _evaluate(self, points: np.ndarray, mode: int, node_index: int) -> SdfEvalResult:
        pts = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 3)
        count = pts.shape[0]

        point_buf = self.ctx.buffer(pts.tobytes())
        dist_buf = self.ctx.buffer(reserve=count * 4)
        owner_buf = self.ctx.buffer(reserve=count * 4)
        region_buf = self.ctx.buffer(reserve=count * 4)

        self._buffers["nodes"].bind_to_storage_buffer(_BIND_NODES)
        self._buffers["params"].bind_to_storage_buffer(_BIND_PARAMS)
        self._buffers["children"].bind_to_storage_buffer(_BIND_CHILDREN)
        self._buffers["bytecode"].bind_to_storage_buffer(_BIND_BYTECODE)
        point_buf.bind_to_storage_buffer(_BIND_POINTS)
        dist_buf.bind_to_storage_buffer(_BIND_OUT_DIST)
        owner_buf.bind_to_storage_buffer(_BIND_OUT_OWNER)
        region_buf.bind_to_storage_buffer(_BIND_OUT_REGION)

        self.program["u_point_count"].value = count
        self.program["u_node_index"].value = node_index
        self.program["u_mode"].value = mode
        if "u_program_length" in self.program:
            self.program["u_program_length"].value = self._program_length

        groups = (count + _LOCAL_SIZE - 1) // _LOCAL_SIZE
        self.program.run(group_x=groups)
        self.ctx.finish()

        dist = np.frombuffer(dist_buf.read(), dtype=np.float32).copy()
        owner = np.frombuffer(owner_buf.read(), dtype=np.uint32).copy()
        region = np.frombuffer(region_buf.read(), dtype=np.uint32).copy()

        for buf in (point_buf, dist_buf, owner_buf, region_buf):
            buf.release()

        return SdfEvalResult(dist=dist, owner=owner, region=region)

    def evaluate_leaf(self, node_index: int, points: np.ndarray) -> SdfEvalResult:
        """Evaluate ``irNodeSDF(node_index, p)`` for each point (Stage 2)."""
        return self._evaluate(points, _MODE_LEAF, node_index)

    def evaluate_scene(self, points: np.ndarray) -> SdfEvalResult:
        """Run the full bytecode program ``evalSceneSDF(p)`` (Stage 3+)."""
        return self._evaluate(points, _MODE_SCENE, 0)

    def evaluate_profile_2d(self, root: int, q: np.ndarray) -> np.ndarray:
        """Evaluate the 2D profile sub-VM at query points ``q`` (N,2)."""
        q = np.asarray(q, dtype=np.float64).reshape(-1, 2)
        pts = np.zeros((q.shape[0], 3), dtype=np.float64)
        pts[:, 0:2] = q
        return self._evaluate(pts, _MODE_PROFILE_2D, root).dist

    def evaluate_profile_1d(self, root: int, t: np.ndarray) -> np.ndarray:
        """Evaluate the 1D profile sub-VM at parameters ``t`` (N,)."""
        t = np.asarray(t, dtype=np.float64).reshape(-1)
        pts = np.zeros((t.shape[0], 3), dtype=np.float64)
        pts[:, 0] = t
        return self._evaluate(pts, _MODE_PROFILE_1D, root).dist

    # -- lifecycle -----------------------------------------------------------

    def _release_buffers(self) -> None:
        for buf in self._buffers.values():
            buf.release()
        self._buffers.clear()

    def release(self) -> None:
        self._release_buffers()
        self.program.release()
        if self._owns_ctx:
            self.ctx.release()


__all__ = ["SdfEvaluator", "SdfEvalResult"]
