from __future__ import annotations

"""Interpreter-backed SDF renderer (design §7).

Implements the scene-upload + raymarch core of the ViewportRenderer seam using
one fixed interpreter program. The headline property: ``upload_render_ir`` on a
topology change is a buffer upload, never a GLSL recompile, so
``program_compile_ms`` stays ~0 and constant regardless of scene size.

Overlays (grid, gizmos, lattice preview, boundary highlight) are intentionally
out of scope for this core; they are unchanged screen-space passes that the
existing backend already provides and can be layered on once the default flips.
"""

import time
from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np

from core.gpu_features import FULL_FEATURES
from core.gpu_program import GpuProgramError, emit_program
from core.gpu_scene import pack_param_update, serialize_scene
from core.render_ir import RenderIR

from .scene_buffers import SceneBuffers
from .shader_assembly import build_program_source

_SHADER_DIR = Path(__file__).parent / "shaders"

# The fragment-stage GLSL compiler cannot bind the interpreter's per-invocation
# value-stack arrays at the design's 64 cap (NVIDIA errors C5041 "possibly large
# array"; Mesa Intel hangs entirely). Compiling once at a smaller capacity keeps
# the fragment path viable on NVIDIA. This is a capability limit only — the max
# SDF-operator nesting depth the renderer accepts; flat carve chains are depth ~2, so it
# bites only deeply-nested trees, which are rejected cleanly by emit_program.
# The compute/validation path keeps the full 64 cap (it compiles fine there).
FRAGMENT_STACK_CAPACITY = 16

_FULLSCREEN_VERT = """#version 460
void main() {
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0,
                  (gl_VertexID == 2) ? 3.0 : -1.0);
    gl_Position = vec4(p, 0.0, 1.0);
}
"""


@dataclass(frozen=True)
class InterpreterUpdateStats:
    path: str
    total_ms: float
    program_compile_ms: float
    upload_ms: float
    render_ir_nodes: int
    reused_program: bool


def _fragment_source(features: frozenset[str] = FULL_FEATURES) -> str:
    return build_program_source(
        features,
        (_SHADER_DIR / "raymarch_interpreter.frag").read_text(),
        stack_capacity=FRAGMENT_STACK_CAPACITY,
    )


class InterpreterRenderer:
    def __init__(self, ctx: moderngl.Context | None = None) -> None:
        self._owns_ctx = ctx is None
        self.ctx = ctx or moderngl.create_context(standalone=True, require=460)
        # The one and only scene program. Compiled once, ever.
        self.program = self.ctx.program(
            vertex_shader=_FULLSCREEN_VERT,
            fragment_shader=_fragment_source(),
        )
        self.vao = self.ctx.vertex_array(self.program, [])
        self.buffers = SceneBuffers(self.ctx)
        self._has_scene = False
        self._last_stats: InterpreterUpdateStats | None = None
        self._fbo: moderngl.Framebuffer | None = None
        self._fbo_size: tuple[int, int] = (0, 0)

    # -- scene upload --------------------------------------------------------

    def upload_render_ir(self, render_ir: RenderIR | None) -> bool:
        start = time.perf_counter()
        if render_ir is None or not render_ir.supported or not render_ir.nodes:
            self.buffers.release()
            self._has_scene = False
            self._last_stats = InterpreterUpdateStats(
                "interpreter", 0.0, 0.0, 0.0, 0, True
            )
            return False

        scene = serialize_scene(render_ir)
        # Validate against the capacity the fragment shader was compiled with,
        # so a too-deeply-nested scene is rejected cleanly instead of overrunning
        # the (smaller) GPU value stack.
        try:
            program = emit_program(
                render_ir,
                stack_capacity=FRAGMENT_STACK_CAPACITY,
                profile_capacity=FRAGMENT_STACK_CAPACITY,
            )
        except GpuProgramError:
            # Too deeply nested for this backend; caller may fall back.
            self.buffers.release()
            self._has_scene = False
            self._last_stats = InterpreterUpdateStats(
                "interpreter", 0.0, 0.0, 0.0, len(render_ir.nodes), True
            )
            return False
        upload_start = time.perf_counter()
        self.buffers.upload(scene, program)
        upload_ms = (time.perf_counter() - upload_start) * 1000.0
        self._has_scene = True

        total_ms = (time.perf_counter() - start) * 1000.0
        self._last_stats = InterpreterUpdateStats(
            path="interpreter",
            total_ms=total_ms,
            program_compile_ms=0.0,  # fixed shader — no per-topology recompile
            upload_ms=upload_ms,
            render_ir_nodes=len(render_ir.nodes),
            reused_program=True,
        )
        return True

    def update_object_parameters(self, render_ir: RenderIR | None) -> bool:
        """Parameter-only fast path (move/resize) — no bytecode regeneration."""
        if render_ir is None or not self.buffers.ready:
            return False
        return self.buffers.update_params(pack_param_update(render_ir))

    def has_scene_program(self) -> bool:
        return self._has_scene

    def last_scene_update_stats(self) -> InterpreterUpdateStats | None:
        return self._last_stats

    # -- rendering -----------------------------------------------------------

    def _ensure_fbo(self, width: int, height: int) -> moderngl.Framebuffer:
        if self._fbo is None or self._fbo_size != (width, height):
            if self._fbo is not None:
                self._fbo.release()
            self._fbo = self.ctx.simple_framebuffer((width, height), components=3)
            self._fbo_size = (width, height)
        return self._fbo

    def render_to_array(
        self,
        width: int,
        height: int,
        camera_position: tuple[float, float, float],
        camera_target: tuple[float, float, float],
        focal_length: float = 1.5,
        surface_opacity: float = 1.0,
        background_color: tuple[float, float, float] = (0.07, 0.08, 0.10),
    ) -> np.ndarray:
        """Render off-screen and return an (H, W, 3) uint8 image (row 0 = bottom)."""
        fbo = self._ensure_fbo(width, height)
        fbo.use()
        self.ctx.clear(*background_color)
        if not self._has_scene:
            return self._read_fbo(fbo, width, height)

        forward = np.asarray(camera_target, np.float64) - np.asarray(camera_position, np.float64)
        forward /= max(np.linalg.norm(forward), 1e-9)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up); right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(right, forward)

        self.program["u_resolution"].value = (float(width), float(height))
        self.program["u_camera_position"].value = tuple(map(float, camera_position))
        self.program["u_camera_target"].value = tuple(map(float, camera_target))
        self.program["u_camera_right"].value = tuple(map(float, right))
        self.program["u_camera_up"].value = tuple(map(float, up))
        self.program["u_focal_length"].value = float(focal_length)
        self.program["u_surface_opacity"].value = float(surface_opacity)
        self.program["u_background_color"].value = tuple(map(float, background_color))
        if "u_program_length" in self.program:
            self.program["u_program_length"].value = self.buffers.program_length

        self.buffers.bind()
        self.vao.render(mode=moderngl.TRIANGLES, vertices=3)
        self.ctx.finish()
        return self._read_fbo(fbo, width, height)

    @staticmethod
    def _read_fbo(fbo: moderngl.Framebuffer, width: int, height: int) -> np.ndarray:
        raw = fbo.read(components=3, dtype="f1")
        return np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)

    def release(self) -> None:
        self.buffers.release()
        self.vao.release()
        self.program.release()
        if self._fbo is not None:
            self._fbo.release()
        if self._owns_ctx:
            self.ctx.release()


__all__ = ["InterpreterRenderer", "InterpreterUpdateStats"]
