from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, TypeVar

from app.artifacts import RenderSceneSnapshot, build_render_artifact
from core.scene import SceneDocument

logger = logging.getLogger(__name__)

T = TypeVar("T")
SECOND_SCALE_THRESHOLD_MS = 1000.0


@dataclass(frozen=True)
class BenchmarkTiming:
    label: str
    mutation_ms: float
    visual_snapshot_ms: float
    artifact_total_ms: float
    surface_ms: float
    tree_node_count: int
    surface_vertex_count: int
    surface_triangle_count: int
    surface_has_geometry: bool


class RenderUploadProbe:
    @classmethod
    def create_optional(cls) -> RenderUploadProbe | None:
        # The viewport now measures disposable surface artifact generation only.
        # Keep the fixture type so timing tests can share the same setup shape.
        return None

    def close(self) -> None:
        return None


def benchmark_scene_step(
    document: SceneDocument,
    label: str,
    mutate: Callable[[SceneDocument], T],
    upload_probe: RenderUploadProbe | None,
) -> tuple[T, BenchmarkTiming]:
    del upload_probe
    mutation_start = perf_counter()
    result = mutate(document)
    mutation_ms = (perf_counter() - mutation_start) * 1000.0

    snapshot_start = perf_counter()
    version, tree = document.visual_snapshot()
    visual_snapshot_ms = (perf_counter() - snapshot_start) * 1000.0

    artifact = build_render_artifact(
        RenderSceneSnapshot(version=version, tree=tree)
    )
    timing = BenchmarkTiming(
        label=label,
        mutation_ms=mutation_ms,
        visual_snapshot_ms=visual_snapshot_ms,
        artifact_total_ms=artifact.timings.total_ms,
        surface_ms=artifact.timings.surface_ms,
        tree_node_count=artifact.timings.tree_node_count,
        surface_vertex_count=artifact.timings.surface_vertex_count,
        surface_triangle_count=artifact.timings.surface_triangle_count,
        surface_has_geometry=bool(
            artifact.surface_scene is not None and artifact.surface_scene.has_geometry
        ),
    )
    logger.info(
        "coregeotest step=%s mutation=%.3f ms visual_snapshot=%.3f ms "
        "artifact_total=%.3f ms surface=%.3f ms "
        "tree_nodes=%d surface_vertices=%d surface_triangles=%d surface_geometry=%s",
        timing.label,
        timing.mutation_ms,
        timing.visual_snapshot_ms,
        timing.artifact_total_ms,
        timing.surface_ms,
        timing.tree_node_count,
        timing.surface_vertex_count,
        timing.surface_triangle_count,
        "yes" if timing.surface_has_geometry else "no",
    )
    _assert_no_second_scale_timing(timing)
    return result, timing


def _assert_no_second_scale_timing(timing: BenchmarkTiming) -> None:
    fields = {
        "mutation": timing.mutation_ms,
        "visual_snapshot": timing.visual_snapshot_ms,
        "artifact_total": timing.artifact_total_ms,
        "surface": timing.surface_ms,
    }
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
