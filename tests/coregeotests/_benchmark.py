from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, TypeVar

from app.artifacts import RenderArtifact, RenderSceneSnapshot, build_render_artifact
from core.scene import SceneDocument

logger = logging.getLogger(__name__)

T = TypeVar("T")
SECOND_SCALE_THRESHOLD_MS = 1000.0


@dataclass(frozen=True)
class UploadTiming:
    total_ms: float
    path: str
    shader_build_ms: float
    program_compile_ms: float
    vao_build_ms: float
    reused_program: bool


@dataclass(frozen=True)
class BenchmarkTiming:
    label: str
    mutation_ms: float
    visual_snapshot_ms: float
    artifact_total_ms: float
    render_ir_ms: float
    tree_node_count: int
    render_ir_node_count: int
    render_ir_supported: bool
    upload: UploadTiming | None


class RenderUploadProbe:
    def __init__(self, context: object, renderer: object) -> None:
        self._context = context
        self._renderer = renderer

    @classmethod
    def create_optional(cls) -> RenderUploadProbe | None:
        try:
            import moderngl
            from app.viewport.renderers.opengl_interpreter.gl_renderer import (
                InterpreterOpenGLRenderer,
            )

            context = moderngl.create_standalone_context(backend="egl", require=460)
            renderer = InterpreterOpenGLRenderer(context)
        except Exception as error:
            logger.info(
                "coregeotest render upload probe unavailable: %s",
                error,
            )
            return None
        return cls(context, renderer)

    def upload(self, render_ir: object) -> UploadTiming:
        total_start = perf_counter()
        success = self._renderer.upload_render_ir(render_ir)
        if not success:
            raise AssertionError("render IR upload failed")
        stats = self._renderer.last_scene_update_stats()
        if stats is None:
            raise AssertionError("render upload did not report timing stats")
        total_ms = (perf_counter() - total_start) * 1000.0
        return UploadTiming(
            total_ms=total_ms,
            path=stats.path,
            shader_build_ms=stats.shader_build_ms,
            program_compile_ms=stats.program_compile_ms,
            vao_build_ms=stats.vao_build_ms,
            reused_program=stats.reused_program,
        )

    def upload_artifact(self, artifact: RenderArtifact) -> UploadTiming | None:
        if artifact.render_ir is not None:
            return self.upload(artifact.render_ir)
        if artifact.tree is None:
            return None
        raise AssertionError("render artifact has no supported RenderIR")

    def close(self) -> None:
        self._renderer.release()
        self._context.release()


def benchmark_scene_step(
    document: SceneDocument,
    label: str,
    mutate: Callable[[SceneDocument], T],
    upload_probe: RenderUploadProbe | None,
) -> tuple[T, BenchmarkTiming]:
    mutation_start = perf_counter()
    result = mutate(document)
    mutation_ms = (perf_counter() - mutation_start) * 1000.0

    snapshot_start = perf_counter()
    version, tree = document.visual_snapshot()
    visual_snapshot_ms = (perf_counter() - snapshot_start) * 1000.0

    artifact = build_render_artifact(
        RenderSceneSnapshot(version=version, tree=tree)
    )
    upload = upload_probe.upload_artifact(artifact) if upload_probe is not None else None
    timing = BenchmarkTiming(
        label=label,
        mutation_ms=mutation_ms,
        visual_snapshot_ms=visual_snapshot_ms,
        artifact_total_ms=artifact.timings.total_ms,
        render_ir_ms=artifact.timings.render_ir_ms,
        tree_node_count=artifact.timings.tree_node_count,
        render_ir_node_count=artifact.timings.render_ir_node_count,
        render_ir_supported=artifact.timings.render_ir_supported,
        upload=upload,
    )
    logger.info(
        "coregeotest step=%s mutation=%.3f ms visual_snapshot=%.3f ms "
        "artifact_total=%.3f ms render_ir=%.3f ms "
        "tree_nodes=%d ir_nodes=%d ir_supported=%s "
        "upload_total=%s upload_path=%s upload_compile=%s reused_program=%s",
        timing.label,
        timing.mutation_ms,
        timing.visual_snapshot_ms,
        timing.artifact_total_ms,
        timing.render_ir_ms,
        timing.tree_node_count,
        timing.render_ir_node_count,
        "yes" if timing.render_ir_supported else "no",
        _format_optional_ms(timing.upload.total_ms if timing.upload else None),
        timing.upload.path if timing.upload is not None else "n/a",
        _format_optional_ms(
            timing.upload.program_compile_ms if timing.upload else None
        ),
        timing.upload.reused_program if timing.upload is not None else "n/a",
    )
    _assert_no_second_scale_timing(timing)
    return result, timing


def _format_optional_ms(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f} ms"


def _assert_no_second_scale_timing(timing: BenchmarkTiming) -> None:
    fields = {
        "mutation": timing.mutation_ms,
        "visual_snapshot": timing.visual_snapshot_ms,
        "artifact_total": timing.artifact_total_ms,
        "render_ir": timing.render_ir_ms,
    }
    if timing.upload is not None:
        fields.update(
            {
                "upload_total": timing.upload.total_ms,
                "upload_shader_build": timing.upload.shader_build_ms,
                "upload_program_compile": timing.upload.program_compile_ms,
                "upload_vao_build": timing.upload.vao_build_ms,
            }
        )
    slow_fields = {
        name: value
        for name, value in fields.items()
        if value >= SECOND_SCALE_THRESHOLD_MS
    }
    if slow_fields:
        details = ", ".join(
            f"{name}={value:.3f} ms" for name, value in slow_fields.items()
        )
        raise AssertionError(
            f"coregeotest step={timing.label} has second-scale timing: {details}"
        )
