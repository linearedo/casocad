from __future__ import annotations

import logging
from dataclasses import dataclass

from PySide6.QtCore import QObject, QThread, Signal, Slot

from core.scene import SceneDocument
from core.sdf import SDFTree

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RenderArtifact:
    version: int
    tree: SDFTree | None
    scene_source: str


def empty_render_scene_source() -> str:
    return (
        "float sceneSDF(vec3 p) { return 1000000.0; }\n"
        "int sceneBoundaryOwnerId(vec3 p) { return 0; }\n"
        "int sceneObjectId(vec3 p) { return 0; }\n"
        "bool sceneSelectionOwnsBoundary(int selected_object_id, "
        "int boundary_owner_id) { return false; }\n"
        "float sceneSelectedObjectSDF(vec3 p, int selected_object_id) "
        "{ return 1000000.0; }\n"
        "int sceneSelectedObjectDimension(int selected_object_id) "
        "{ return 0; }\n"
        "const int COMPONENT_COUNT = 0;\n"
        "float componentSDF(vec3 p, int component) { return 1000000.0; }\n"
        "int componentObjectId(int component) { return 0; }"
    )


def build_render_artifact(snapshot: SceneDocument) -> RenderArtifact:
    tree = snapshot.visual_tree() if snapshot.objects else None
    source = (
        f"{tree.to_glsl()}\n{tree.components_to_glsl()}"
        if tree is not None
        else empty_render_scene_source()
    )
    return RenderArtifact(
        version=snapshot.version,
        tree=tree,
        scene_source=source,
    )


class RenderArtifactWorker(QObject):
    completed = Signal(object)
    failed = Signal(int, str)

    def __init__(self, snapshot: SceneDocument) -> None:
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
        self._pending_render_snapshot: SceneDocument | None = None
        self._render_thread: QThread | None = None
        self._render_worker: RenderArtifactWorker | None = None

    def request_render(self, snapshot: SceneDocument) -> None:
        self._latest_render_version = snapshot.version
        if self._render_thread is not None:
            self._pending_render_snapshot = snapshot
            return
        self._start_render(snapshot)

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

    def _start_render(self, snapshot: SceneDocument) -> None:
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
