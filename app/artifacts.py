from __future__ import annotations

import logging
from dataclasses import dataclass
from time import perf_counter

from PySide6.QtCore import QObject, QThread, Signal, Slot

from core.render_ir import RenderIR, build_render_ir
from core.scene import SceneDocument
from core.sdf import SDFTree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderArtifactTimings:
    total_ms: float
    render_ir_ms: float
    tree_node_count: int
    render_ir_node_count: int
    render_ir_supported: bool


@dataclass(frozen=True)
class RenderArtifact:
    version: int
    tree: SDFTree | None
    render_ir: RenderIR | None
    timings: RenderArtifactTimings


@dataclass(frozen=True)
class RenderSceneSnapshot:
    version: int
    tree: SDFTree | None


def _coerce_render_scene_snapshot(
    snapshot: RenderSceneSnapshot | SceneDocument,
) -> RenderSceneSnapshot:
    if isinstance(snapshot, SceneDocument):
        version, tree = snapshot.visual_snapshot()
        return RenderSceneSnapshot(version=version, tree=tree)
    return snapshot


def build_render_artifact(
    snapshot: RenderSceneSnapshot | SceneDocument,
) -> RenderArtifact:
    total_start = perf_counter()
    render_snapshot = _coerce_render_scene_snapshot(snapshot)
    tree = render_snapshot.tree
    render_ir_start = perf_counter()
    render_ir = build_render_ir(tree) if tree is not None else None
    render_ir_ms = (perf_counter() - render_ir_start) * 1000.0
    total_ms = (perf_counter() - total_start) * 1000.0
    timings = RenderArtifactTimings(
        total_ms=total_ms,
        render_ir_ms=render_ir_ms,
        tree_node_count=len(tree.nodes) if tree is not None else 0,
        render_ir_node_count=len(render_ir.nodes) if render_ir is not None else 0,
        render_ir_supported=bool(render_ir is not None and render_ir.supported),
    )
    return RenderArtifact(
        version=render_snapshot.version,
        tree=tree,
        render_ir=render_ir if render_ir is not None and render_ir.supported else None,
        timings=timings,
    )


class RenderArtifactWorker(QObject):
    completed = Signal(object)
    failed = Signal(int, str)

    def __init__(self, snapshot: RenderSceneSnapshot) -> None:
        super().__init__()
        self.snapshot = snapshot

    @Slot()
    def run(self) -> None:
        try:
            artifact = build_render_artifact(self.snapshot)
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

    def request_render(
        self,
        snapshot: RenderSceneSnapshot | SceneDocument,
    ) -> None:
        render_snapshot = _coerce_render_scene_snapshot(snapshot)
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
        worker = RenderArtifactWorker(snapshot)
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

    @Slot(object)
    def _on_render_completed(self, artifact: RenderArtifact) -> None:
        if artifact.version == self._latest_render_version:
            self.render_ready.emit(artifact)

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
