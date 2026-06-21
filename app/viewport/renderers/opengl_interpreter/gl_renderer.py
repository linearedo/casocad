from __future__ import annotations

"""Interpreter viewport renderer — the sole SDF backend (design §13.3, §13.6).

Subclasses the generic GL infrastructure (gizmos, grid pass, lattice, world
axis, framebuffer, the render loop) and provides the scene + preview SDF passes
via the fixed interpreter program + SSBOs. The per-scene GLSL codegen path has
been deleted; this renderer is the only one.

The shader is assembled per-GPU into a feature tier (NVIDIA -> full; Mesa/other
-> lean primitives+operators) because Mesa's compiler hangs on the full shader.
A scene needing a feature the active tier can't compile is skipped (the viewport
shows the grid/overlays without that geometry) and reported — there is no
codegen fallback.
"""

import logging
import time

import moderngl

from app.viewport.renderer import SceneUpdateStats, _RenderIRLayerState
from core.gpu_features import CULL, FULL_FEATURES, LEAN_FEATURES, scene_fits_tier
from core.gpu_program import GpuProgramError, emit_program
from core.gpu_scene import pack_param_update, serialize_scene
from core.render_ir import RenderIR

from ..opengl.renderer import OpenGLRenderer
from .renderer import FRAGMENT_STACK_CAPACITY, _fragment_source
from .scene_buffers import SceneBuffers
from .scene_cull import SceneCull

logger = logging.getLogger(__name__)

_FULLSCREEN_VERT = """#version 460
in vec2 in_position;
void main() { gl_Position = vec4(in_position, 0.0, 1.0); }
"""


def _feature_tier_for(renderer_name: str) -> frozenset[str]:
    """Pick the shader feature tier the GPU's compiler can build (design §13.3).

    NVIDIA compiles the full interpreter; Mesa Intel hangs at link, so it gets
    the lean core (primitives + operators). Detection is an allow-list by
    GL_RENDERER — Mesa *hangs* rather than failing fast, so we must never hand it
    the big shader to "try". Widen as other drivers are confirmed.
    """

    name = renderer_name.lower()
    # The world-grid cull (core-only chunk) compiles on every tier, so both get
    # it; only the profile/sweep/selector features are gated by GPU capability.
    if any(tag in name for tag in ("nvidia", "geforce", "quadro", "rtx")):
        return frozenset(FULL_FEATURES | {CULL})
    return frozenset(LEAN_FEATURES | {CULL})


class InterpreterOpenGLRenderer(OpenGLRenderer):
    def __init__(self, context: moderngl.Context) -> None:
        super().__init__(context)
        renderer_name = context.info.get("GL_RENDERER", "")
        self._features = _feature_tier_for(renderer_name)
        self._interp_program = context.program(
            vertex_shader=_FULLSCREEN_VERT,
            fragment_shader=_fragment_source(self._features),
        )
        self._interp_vao = context.vertex_array(
            self._interp_program, [(self._vertex_buffer, "2f", "in_position")]
        )
        self._scene_buffers = SceneBuffers(context)
        self._preview_buffers = SceneBuffers(context)
        self._cull = SceneCull(context) if CULL in self._features else None
        self._interp_active = False
        self._preview_active = False
        self._frame_count = 0
        self._frame_clock = time.perf_counter()
        tier = "full" if self._features == FULL_FEATURES else "lean"
        logger.info("SDF interpreter backend active — %s tier on %s", tier, renderer_name)

    # -- helpers -------------------------------------------------------------

    def _try_serialize(self, render_ir: RenderIR):
        """Return (scene, program) or None if the scene can't run on this tier."""
        if render_ir is None or not render_ir.supported or not render_ir.nodes:
            return None
        if not scene_fits_tier(render_ir, self._features):
            return None
        try:
            scene = serialize_scene(render_ir)
            program = emit_program(
                render_ir,
                stack_capacity=FRAGMENT_STACK_CAPACITY,
                profile_capacity=FRAGMENT_STACK_CAPACITY,
            )
        except GpuProgramError as error:
            logger.info("interpreter cannot serialize scene: %s", error)
            return None
        return scene, program

    # -- scene upload --------------------------------------------------------

    def upload_render_ir(self, render_ir: RenderIR | None) -> bool:
        start = time.perf_counter()
        result = self._try_serialize(render_ir)
        if result is None:
            # Unsupported on this tier — show grid/overlays without this geometry
            # (no codegen fallback). Returning True keeps the viewport quiet
            # rather than raising; the skip is logged.
            if render_ir is not None and render_ir.nodes:
                logger.info(
                    "scene not renderable on the %s tier — skipping geometry",
                    "full" if self._features == FULL_FEATURES else "lean",
                )
            # Empty / unsupported scene: still run the fullscreen pass with a
            # zero-length program so the in-shader grid + background draw (the
            # grid is part of the scene fragment) — just no geometry.
            self._set_grid_only()
            self._last_scene_update_stats = SceneUpdateStats(
                "interpreter", 0.0, 0.0, 0.0, 0.0,
                0 if render_ir is None else len(render_ir.nodes), True,
            )
            return True

        scene, program = result
        self._scene_buffers.upload(scene, program)
        layer = _RenderIRLayerState()
        layer.program = self._interp_program
        layer.vao = self._interp_vao
        layer.render_ir = render_ir
        layer.topology_signature = render_ir.topology_signature
        self._scene_layer = layer
        self._interp_active = True
        cull_active = False
        if self._cull is not None:
            cull_active = self._cull.build(render_ir)  # inactive if not cullable
        logger.info(
            "interpreter upload: ir_nodes=%d cull=%s",
            len(render_ir.nodes), "active" if cull_active else "inactive",
        )
        self._last_scene_update_stats = SceneUpdateStats(
            path="interpreter",
            total_ms=(time.perf_counter() - start) * 1000.0,
            shader_build_ms=0.0,
            program_compile_ms=0.0,
            vao_build_ms=0.0,
            render_ir_nodes=len(render_ir.nodes),
            reused_program=True,
        )
        return True

    def update_render_ir_object_parameters(
        self, render_ir: RenderIR | None, object_ids: tuple[int, ...]
    ) -> bool:
        if self._interp_active and self._scene_buffers.ready and render_ir is not None:
            if self._scene_buffers.update_params(pack_param_update(render_ir)):
                self._scene_layer.render_ir = render_ir
                # Positions changed -> the cull grid binning is now stale; re-bin
                # so a moved/resized object is evaluated in its new cells (else it
                # appears not to move and corrupts neighbouring cells).
                if self._cull is not None:
                    self._cull.build(render_ir)
                return True
            return self.upload_render_ir(render_ir)
        return False

    def _set_grid_only(self) -> None:
        """Empty scene: fullscreen pass with a zero-length program so the
        in-shader grid + background still render, with no geometry."""
        self._scene_buffers.upload(serialize_scene(None), emit_program(None))
        layer = _RenderIRLayerState()
        layer.program = self._interp_program
        layer.vao = self._interp_vao
        self._scene_layer = layer
        self._interp_active = True
        if self._cull is not None:
            self._cull.active = False

    def clear_scene(self) -> None:
        super().clear_scene()
        self._set_grid_only()

    # -- preview layer (interpreter) ----------------------------------------

    def upload_preview_render_ir(self, render_ir: RenderIR | None) -> bool:
        if render_ir is None:
            self.clear_preview_render_ir()
            return True
        result = self._try_serialize(render_ir)
        if result is None:
            self.clear_preview_render_ir()
            return True  # unsupported preview on this tier: skip quietly
        scene, program = result
        self._preview_buffers.upload(scene, program)
        self._preview_active = True
        return True

    def clear_preview_render_ir(self) -> None:
        self._preview_active = False
        self._preview_buffers.release()

    # -- draw (scene + overlays via base; then the interpreter preview ghost) -

    def render(self, *args, **kwargs) -> None:
        if self._interp_active and self._scene_buffers.ready:
            self._scene_buffers.bind()
            if "u_program_length" in self._interp_program:
                self._interp_program["u_program_length"].value = (
                    self._scene_buffers.program_length
                )
        if self._cull is not None:
            # Bind the cull SSBOs (always, since the shader declares them) and
            # enable the cull march only when the scene is cullable; else the
            # frag uses the exact full VM.
            self._cull.bind()
            self._cull.set_uniforms(self._interp_program)
            if "u_cull_enabled" in self._interp_program:
                self._interp_program["u_cull_enabled"].value = bool(
                    self._cull.active and self._interp_active
                )
        super().render(*args, **kwargs)

        # Throttled live diagnostic: real frame rate + which path is running +
        # the rendered pixel count (width,height are render() args 0,1).
        self._frame_count += 1
        if self._frame_count >= 60:
            now = time.perf_counter()
            fps = self._frame_count / max(now - self._frame_clock, 1e-6)
            px = (args[0] * args[1]) if len(args) >= 2 else 0
            cull_on = self._cull is not None and self._cull.active and self._interp_active
            logger.info(
                "interpreter frame: %.1f fps, %d px, cull=%s",
                fps, px, "on" if cull_on else "off",
            )
            self._frame_count = 0
            self._frame_clock = now

        if self._preview_active and self._preview_buffers.ready:
            # Translucent ghost overlay; camera/grid uniforms persist on the
            # program from the scene pass above.
            self._preview_buffers.bind()
            prog = self._interp_program
            if "u_program_length" in prog:
                prog["u_program_length"].value = self._preview_buffers.program_length
            if "u_cull_enabled" in prog:
                prog["u_cull_enabled"].value = False  # preview uses the VM path
            if "u_render_preview_layer" in prog:
                prog["u_render_preview_layer"].value = True
            self.context.disable(moderngl.DEPTH_TEST)
            self.context.enable(moderngl.BLEND)
            self.context.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
            self._interp_vao.render(mode=moderngl.TRIANGLES)
            self.context.disable(moderngl.BLEND)
            self.context.enable(moderngl.DEPTH_TEST)
            if "u_render_preview_layer" in prog:
                prog["u_render_preview_layer"].value = False

    def release(self) -> None:
        self._scene_buffers.release()
        self._preview_buffers.release()
        if self._cull is not None:
            self._cull.release()
        # Avoid the base double-freeing our shared interpreter program/vao.
        self._scene_layer = _RenderIRLayerState()
        self._preview_layer = _RenderIRLayerState()
        self._interp_vao.release()
        self._interp_program.release()
        super().release()


__all__ = ["InterpreterOpenGLRenderer"]
