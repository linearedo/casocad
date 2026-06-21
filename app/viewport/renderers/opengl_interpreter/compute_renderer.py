from __future__ import annotations

"""Compute-prepass interpreter renderer — the cross-vendor path (Stage 8 #1).

The fragment-stage GLSL compiler cannot handle the interpreter's value-stack
arrays (Mesa Intel hangs; NVIDIA errors C5041). A COMPUTE shader compiles the
exact same VM at the full IR_STACK_CAPACITY on Intel/AMD/NVIDIA alike, so this is
the path that can drive the viewport on any modern GPU. Each compute invocation
raymarches one pixel and writes the shaded colour to an image; the host reads it
back (offscreen) or a trivial blit puts it on screen (GUI integration to follow).
"""

import time
from dataclasses import dataclass
from pathlib import Path

import moderngl
import numpy as np

from core.gpu_features import FULL_FEATURES
from core.gpu_program import emit_program
from core.gpu_scene import pack_param_update, serialize_scene
from core.render_ir import RenderIR

from .scene_buffers import SceneBuffers
from .shader_assembly import build_program_source

_SHADER_DIR = Path(__file__).parent / "shaders"
_LOCAL = 8
_IMAGE_UNIT = 0


@dataclass(frozen=True)
class ComputeUpdateStats:
    path: str
    total_ms: float
    program_compile_ms: float
    upload_ms: float
    render_ir_nodes: int
    reused_program: bool


def _compute_source() -> str:
    return build_program_source(
        FULL_FEATURES, (_SHADER_DIR / "raymarch_interpreter.comp").read_text()
    )


class ComputeInterpreterRenderer:
    def __init__(self, ctx: moderngl.Context | None = None) -> None:
        self._owns_ctx = ctx is None
        self.ctx = ctx or moderngl.create_context(standalone=True, require=460)
        # One compute program, compiled once at the full value-stack capacity.
        self.program = self.ctx.compute_shader(_compute_source())
        self.buffers = SceneBuffers(self.ctx)
        self._has_scene = False
        self._last_stats: ComputeUpdateStats | None = None
        self._out_tex: moderngl.Texture | None = None
        self._out_size: tuple[int, int] = (0, 0)

    # -- scene upload --------------------------------------------------------

    def upload_render_ir(self, render_ir: RenderIR | None) -> bool:
        start = time.perf_counter()
        if render_ir is None or not render_ir.supported or not render_ir.nodes:
            self.buffers.release()
            self._has_scene = False
            self._last_stats = ComputeUpdateStats("compute", 0.0, 0.0, 0.0, 0, True)
            return False
        scene = serialize_scene(render_ir)
        program = emit_program(render_ir)  # full cap 64 — compute handles it
        upload_start = time.perf_counter()
        self.buffers.upload(scene, program)
        upload_ms = (time.perf_counter() - upload_start) * 1000.0
        self._has_scene = True
        self._last_stats = ComputeUpdateStats(
            path="compute",
            total_ms=(time.perf_counter() - start) * 1000.0,
            program_compile_ms=0.0,
            upload_ms=upload_ms,
            render_ir_nodes=len(render_ir.nodes),
            reused_program=True,
        )
        return True

    def update_object_parameters(self, render_ir: RenderIR | None) -> bool:
        if render_ir is None or not self.buffers.ready:
            return False
        return self.buffers.update_params(pack_param_update(render_ir))

    def has_scene_program(self) -> bool:
        return self._has_scene

    def last_scene_update_stats(self) -> ComputeUpdateStats | None:
        return self._last_stats

    # -- rendering -----------------------------------------------------------

    def _ensure_texture(self, width: int, height: int) -> moderngl.Texture:
        if self._out_tex is None or self._out_size != (width, height):
            if self._out_tex is not None:
                self._out_tex.release()
            self._out_tex = self.ctx.texture((width, height), 4, dtype="f1")
            self._out_size = (width, height)
        return self._out_tex

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
        tex = self._ensure_texture(width, height)
        if not self._has_scene:
            return np.frombuffer(tex.read(), dtype=np.uint8).reshape(height, width, 4)[..., :3]

        forward = np.asarray(camera_target, np.float64) - np.asarray(camera_position, np.float64)
        forward /= max(np.linalg.norm(forward), 1e-9)
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, world_up); right /= max(np.linalg.norm(right), 1e-9)
        up = np.cross(right, forward)

        p = self.program
        p["u_resolution"].value = (float(width), float(height))
        p["u_camera_position"].value = tuple(map(float, camera_position))
        p["u_camera_target"].value = tuple(map(float, camera_target))
        p["u_camera_right"].value = tuple(map(float, right))
        p["u_camera_up"].value = tuple(map(float, up))
        p["u_focal_length"].value = float(focal_length)
        p["u_surface_opacity"].value = float(surface_opacity)
        p["u_background_color"].value = tuple(map(float, background_color))
        if "u_program_length" in p:
            p["u_program_length"].value = self.buffers.program_length

        self.buffers.bind()
        tex.bind_to_image(_IMAGE_UNIT, read=False, write=True)
        groups_x = (width + _LOCAL - 1) // _LOCAL
        groups_y = (height + _LOCAL - 1) // _LOCAL
        p.run(groups_x, groups_y, 1)
        self.ctx.finish()

        raw = np.frombuffer(tex.read(), dtype=np.uint8).reshape(height, width, 4)
        return raw[..., :3]

    def release(self) -> None:
        self.buffers.release()
        self.program.release()
        if self._out_tex is not None:
            self._out_tex.release()
        if self._owns_ctx:
            self.ctx.release()


__all__ = ["ComputeInterpreterRenderer", "ComputeUpdateStats"]
