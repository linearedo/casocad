from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from time import perf_counter

from PySide6.QtCore import QObject, QThread, Signal, Slot

from app.viewport.performance_governor import ViewportRenderBudget
from app.viewport.surface_cache import (
    ViewportSurfaceCache,
    ViewportSurfaceScene,
    build_viewport_surface_scene,
)
from core.scene import SceneDocument
from core.sdf import SDFTree

logger = logging.getLogger(__name__)
COARSE_VIEWPORT_SURFACE_RESOLUTION = 14
# Booleans contour through dual contouring (not a cheap analytic mesh), so their
# interactive/drawing pass needs a higher floor than primitives or they look
# chunky while being drawn. The build is async, so this only adds surface-update
# latency (~0.2s), never a GUI freeze.
BOOLEAN_DRAW_RESOLUTION = 48
REVOLVE_VIEWPORT_SURFACE_RESOLUTION = 48
REFINED_VIEWPORT_SURFACE_RESOLUTION = 128
# Progressive refinement ladder. The first build is COARSE (instant, interactive);
# after each build settles the artifact thread re-requests the next tier, so the
# image sharpens in short background steps instead of one long high-res build.
# Each tier is a separate async build off the GUI thread (outcome 0). The top tier
# crosses into the manifold narrow-band contour (see _NARROW_BAND_MIN_RES).
_REFINEMENT_TIERS = (64, 96, REFINED_VIEWPORT_SURFACE_RESOLUTION)


def _next_surface_resolution(current: int) -> int | None:
    for tier in _REFINEMENT_TIERS:
        if current < tier:
            return tier
    return None


@dataclass(frozen=True)
class RenderArtifactTimings:
    total_ms: float
    surface_ms: float
    render_wait_ms: float
    tree_node_count: int
    surface_vertex_count: int = 0
    surface_triangle_count: int = 0
    surface_resolution: int = COARSE_VIEWPORT_SURFACE_RESOLUTION
    large_scene_mode: bool = False
    large_scene_reason: str = ""
    total_object_count: int = 0
    exact_object_count: int = 0
    no_blur: bool = True


@dataclass(frozen=True)
class RenderArtifact:
    version: int
    tree: SDFTree | None
    surface_scene: ViewportSurfaceScene | None
    budget: ViewportRenderBudget | None
    refine_after: bool
    timings: RenderArtifactTimings


@dataclass(frozen=True)
class RenderSceneSnapshot:
    version: int
    tree: SDFTree | None
    budget: ViewportRenderBudget | None = None
    requested_at: float = 0.0
    surface_resolution: int = COARSE_VIEWPORT_SURFACE_RESOLUTION
    refine_after: bool = True


def _coerce_render_scene_snapshot(
    snapshot: RenderSceneSnapshot | SceneDocument,
) -> RenderSceneSnapshot:
    if isinstance(snapshot, SceneDocument):
        version, tree = snapshot.visual_snapshot()
        return RenderSceneSnapshot(
            version=version,
            tree=tree,
            requested_at=perf_counter(),
        )
    return snapshot


def build_render_artifact(
    snapshot: RenderSceneSnapshot | SceneDocument,
    surface_cache: ViewportSurfaceCache | None = None,
) -> RenderArtifact:
    total_start = perf_counter()
    render_snapshot = _coerce_render_scene_snapshot(snapshot)
    tree = render_snapshot.tree
    surface_start = perf_counter()
    if surface_cache is None:
        surface_cache = ViewportSurfaceCache(
            resolution=render_snapshot.surface_resolution,
        )
    surface_scene = build_viewport_surface_scene(
        tree,
        render_snapshot.version,
        cache=surface_cache,
    )
    surface_ms = (perf_counter() - surface_start) * 1000.0
    total_ms = (perf_counter() - total_start) * 1000.0
    render_wait_ms = (
        (perf_counter() - render_snapshot.requested_at) * 1000.0
        if render_snapshot.requested_at > 0.0
        else total_ms
    )
    budget = render_snapshot.budget
    timings = RenderArtifactTimings(
        total_ms=total_ms,
        surface_ms=surface_ms,
        render_wait_ms=render_wait_ms,
        tree_node_count=len(tree.nodes) if tree is not None else 0,
        surface_vertex_count=surface_scene.vertex_count if surface_scene else 0,
        surface_triangle_count=surface_scene.triangle_count if surface_scene else 0,
        surface_resolution=render_snapshot.surface_resolution,
        large_scene_mode=bool(budget is not None and budget.large_scene_mode),
        large_scene_reason=budget.reason if budget is not None else "",
        total_object_count=len(tree.components) if tree is not None else 0,
        exact_object_count=len(tree.components) if tree is not None else 0,
        no_blur=(
            budget.no_blur
            if budget is not None
            else True
        ),
    )
    if timings.large_scene_mode:
        logger.info(
            "viewport-governor: mode=large exact=%d total=%d no_blur=%s reason=%s",
            timings.exact_object_count,
            timings.total_object_count,
            timings.no_blur,
            timings.large_scene_reason,
        )
    return RenderArtifact(
        version=render_snapshot.version,
        tree=tree,
        surface_scene=surface_scene,
        budget=budget,
        refine_after=render_snapshot.refine_after,
        timings=timings,
    )


class RenderArtifactWorker(QObject):
    completed = Signal(object)
    failed = Signal(int, str)

    def __init__(
        self,
        snapshot: RenderSceneSnapshot,
        surface_cache: ViewportSurfaceCache,
    ) -> None:
        super().__init__()
        self.snapshot = snapshot
        self.surface_cache = surface_cache

    @Slot()
    def run(self) -> None:
        try:
            artifact = build_render_artifact(self.snapshot, self.surface_cache)
        except Exception as error:
            logger.exception("render artifact build failed")
            self.failed.emit(self.snapshot.version, str(error))
            return
        self.completed.emit(artifact)


class ArtifactManager(QObject):
    render_ready = Signal(object)
    render_failed = Signal(int, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._latest_render_version = -1
        self._pending_render_snapshot: RenderSceneSnapshot | None = None
        self._render_thread: QThread | None = None
        self._render_worker: RenderArtifactWorker | None = None
        self._surface_caches: dict[int, ViewportSurfaceCache] = {}

    def request_render(
        self,
        snapshot: RenderSceneSnapshot | SceneDocument,
    ) -> None:
        render_snapshot = _coerce_render_scene_snapshot(snapshot)
        if render_snapshot.requested_at <= 0.0:
            render_snapshot = replace(render_snapshot, requested_at=perf_counter())
        self._latest_render_version = render_snapshot.version
        if self._render_thread is not None:
            self._pending_render_snapshot = render_snapshot
            return
        self._start_render(render_snapshot)

    def shutdown(self, timeout_ms: int = 2000) -> None:
        self._pending_render_snapshot = None
        thread = self._render_thread
        if thread is None:
            return
        thread.quit()
        if not thread.wait(timeout_ms):
            logger.warning("waiting for render artifact thread to finish")
            thread.wait()
        self._render_thread = None
        self._render_worker = None

    def _start_render(self, snapshot: RenderSceneSnapshot) -> None:
        thread = QThread(self)
        worker = RenderArtifactWorker(snapshot, self._surface_cache_for(snapshot))
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.completed.connect(self._on_render_completed)
        worker.failed.connect(self._on_render_failed)
        worker.completed.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.completed.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._on_render_thread_finished)
        self._render_thread = thread
        self._render_worker = worker
        thread.start()

    def _surface_cache_for(
        self,
        snapshot: RenderSceneSnapshot,
    ) -> ViewportSurfaceCache:
        resolution = int(snapshot.surface_resolution)
        cache = self._surface_caches.get(resolution)
        if cache is None:
            cache = ViewportSurfaceCache(resolution=resolution)
            self._surface_caches[resolution] = cache
        return cache

    @Slot(object)
    def _on_render_completed(self, artifact: RenderArtifact) -> None:
        if artifact.version == self._latest_render_version:
            self.render_ready.emit(artifact)
            next_resolution = _next_surface_resolution(
                artifact.timings.surface_resolution
            )
            if (
                artifact.refine_after
                and artifact.tree is not None
                and next_resolution is not None
                and not artifact.timings.large_scene_mode
                and self._pending_render_snapshot is None
            ):
                self._pending_render_snapshot = RenderSceneSnapshot(
                    version=artifact.version,
                    tree=artifact.tree,
                    budget=artifact.budget,
                    requested_at=perf_counter(),
                    surface_resolution=next_resolution,
                    # Keep refining up the ladder until the top tier is reached.
                    refine_after=True,
                )

    @Slot(int, str)
    def _on_render_failed(self, version: int, message: str) -> None:
        if version == self._latest_render_version:
            self.render_failed.emit(version, message)

    @Slot()
    def _on_render_thread_finished(self) -> None:
        self._render_thread = None
        self._render_worker = None
        pending = self._pending_render_snapshot
        self._pending_render_snapshot = None
        if pending is not None:
            self._start_render(pending)
