from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
import threading
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, QProcess, Signal

from .render_cache import (
    RenderCacheSummary,
    ensure_render_cache,
    iter_render_cache_chunks,
    render_cache_summary_from_dict,
)


@dataclass(frozen=True)
class MeshPreviewChunk:
    vertices: NDArray[np.float32]
    colors: NDArray[np.float32]
    wire_vertices: NDArray[np.float32]
    wire_colors: NDArray[np.float32]
    element_count: int
    triangle_count: int
    edge_count: int


@dataclass(frozen=True)
class MeshPreviewSummary:
    path: Path
    element_count: int
    preview_vertex_count: int
    preview_triangle_count: int
    preview_edge_count: int
    tag_names: tuple[str, ...]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    truncated: bool


def iter_mesh_preview_chunks_sync(
    path: str | Path,
    *,
    max_rows_per_chunk: int = 4096,
    max_preview_vertices: int = 300_000,
) -> Iterator[MeshPreviewChunk | MeshPreviewSummary]:
    del max_rows_per_chunk
    artifact_path = Path(path)
    cache_path, cache_summary = ensure_render_cache(
        artifact_path,
        max_preview_vertices=max_preview_vertices,
    )
    yield from _iter_cached_preview_chunks(artifact_path, cache_path, cache_summary)


def _iter_cached_preview_chunks(
    artifact_path: Path,
    cache_path: Path,
    cache_summary: RenderCacheSummary,
) -> Iterator[MeshPreviewChunk | MeshPreviewSummary]:
    empty = np.zeros((0, 3), dtype=np.float32)

    for cache_chunk in iter_render_cache_chunks(cache_path):
        if cache_chunk.primitive_type == "triangle":
            yield MeshPreviewChunk(
                vertices=cache_chunk.positions,
                colors=cache_chunk.colors,
                wire_vertices=empty,
                wire_colors=empty,
                element_count=0,
                triangle_count=int(cache_chunk.positions.shape[0] // 3),
                edge_count=0,
            )
        elif cache_chunk.primitive_type == "line":
            yield MeshPreviewChunk(
                vertices=empty,
                colors=empty,
                wire_vertices=cache_chunk.positions,
                wire_colors=cache_chunk.colors,
                element_count=0,
                triangle_count=0,
                edge_count=int(cache_chunk.positions.shape[0] // 2),
            )

    yield MeshPreviewSummary(
        path=artifact_path,
        element_count=cache_summary.element_count,
        preview_vertex_count=cache_summary.preview_vertex_count,
        preview_triangle_count=cache_summary.preview_triangle_count,
        preview_edge_count=cache_summary.preview_edge_count,
        tag_names=cache_summary.tag_names,
        bounds_min=cache_summary.bounds_min,
        bounds_max=cache_summary.bounds_max,
        truncated=cache_summary.truncated,
    )


class MeshArtifactLoader(QObject):
    chunk_loaded = Signal(object)
    finished = Signal(object)
    failed = Signal(str)
    status_changed = Signal(str)

    def __init__(
        self,
        *,
        max_rows_per_chunk: int = 4096,
        max_preview_vertices: int = 300_000,
    ) -> None:
        super().__init__()
        self._max_rows_per_chunk = max_rows_per_chunk
        self._max_preview_vertices = max_preview_vertices
        self._generation = 0
        self._cache_process: QProcess | None = None
        self._cache_stdout_buffer = ""
        self._cache_stderr_buffer = ""
        self._cache_done: tuple[Path, RenderCacheSummary] | None = None
        self._cache_artifact_path: Path | None = None
        self._cache_max_rows_per_chunk = max_rows_per_chunk

    def load(self, path: str | Path) -> None:
        self._generation += 1
        generation = self._generation
        artifact_path = Path(path)
        self._stop_cache_process()
        self._cache_stdout_buffer = ""
        self._cache_stderr_buffer = ""
        self._cache_done = None
        self._cache_artifact_path = artifact_path
        self._cache_max_rows_per_chunk = self._max_rows_per_chunk

        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(
            [
                "-m",
                "app.meshing.viewer.cache_worker",
                str(artifact_path),
                str(self._max_preview_vertices),
            ]
        )
        process.readyReadStandardOutput.connect(
            lambda process=process: self._on_cache_stdout(process)
        )
        process.readyReadStandardError.connect(
            lambda process=process: self._on_cache_stderr(process)
        )
        process.finished.connect(
            lambda code, status, process=process: self._on_cache_finished(
                generation,
                process,
                code,
                status,
            )
        )
        process.errorOccurred.connect(
            lambda error, process=process: self._on_cache_process_error(
                generation,
                process,
                error,
            )
        )
        self._cache_process = process
        self.status_changed.emit("Preparing render preview cache in worker process.")
        process.start()

    def set_preview_limits(
        self,
        *,
        max_rows_per_chunk: int | None = None,
        max_preview_vertices: int | None = None,
    ) -> None:
        if max_rows_per_chunk is not None:
            self._max_rows_per_chunk = max(1, int(max_rows_per_chunk))
        if max_preview_vertices is not None:
            self._max_preview_vertices = max(3, int(max_preview_vertices))

    def _load_worker(
        self,
        generation: int,
        path: Path,
        cache_path: Path,
        cache_summary: RenderCacheSummary,
    ) -> None:
        try:
            for item in _iter_cached_preview_chunks(path, cache_path, cache_summary):
                if generation != self._generation:
                    return
                if isinstance(item, MeshPreviewChunk):
                    self.chunk_loaded.emit(item)
                else:
                    self.finished.emit(item)
        except Exception as exc:  # noqa: BLE001
            if generation == self._generation:
                self.failed.emit(str(exc))

    def _start_cache_stream(
        self,
        generation: int,
        artifact_path: Path,
        cache_path: Path,
        cache_summary: RenderCacheSummary,
    ) -> None:
        threading.Thread(
            target=self._load_worker,
            args=(generation, artifact_path, cache_path, cache_summary),
            daemon=True,
        ).start()

    def _on_cache_stdout(self, process: QProcess) -> None:
        if process is not self._cache_process:
            return
        self._cache_stdout_buffer += bytes(process.readAllStandardOutput()).decode(
            "utf-8",
            errors="replace",
        )
        while "\n" in self._cache_stdout_buffer:
            line, self._cache_stdout_buffer = self._cache_stdout_buffer.split("\n", 1)
            if line:
                self._handle_cache_message(line)

    def _on_cache_stderr(self, process: QProcess) -> None:
        if process is not self._cache_process:
            return
        self._cache_stderr_buffer += bytes(process.readAllStandardError()).decode(
            "utf-8",
            errors="replace",
        )

    def _handle_cache_message(self, line: str) -> None:
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            self._cache_stderr_buffer += line + "\n"
            return
        message_type = message.get("type")
        if message_type == "started":
            self.status_changed.emit("Render preview cache worker started.")
        elif message_type == "done":
            self._cache_done = (
                Path(str(message["cache_path"])),
                render_cache_summary_from_dict(message["summary"]),
            )
        elif message_type == "error":
            details = str(message.get("traceback") or message.get("message") or "unknown error")
            self.failed.emit(details)

    def _on_cache_finished(
        self,
        generation: int,
        process: QProcess,
        exit_code: int,
        exit_status: QProcess.ExitStatus,
    ) -> None:
        if process is not self._cache_process:
            process.deleteLater()
            return
        process.deleteLater()
        self._cache_process = None
        if generation != self._generation:
            return
        if self._cache_stdout_buffer.strip():
            for line in self._cache_stdout_buffer.splitlines():
                if line:
                    self._handle_cache_message(line)
            self._cache_stdout_buffer = ""
        if exit_status != QProcess.ExitStatus.NormalExit or exit_code != 0:
            if self._cache_done is None:
                details = self._cache_stderr_buffer.strip() or f"cache worker exited with {exit_code}"
                self.failed.emit(details)
            return
        if self._cache_done is None:
            details = self._cache_stderr_buffer.strip() or "cache worker did not return a result"
            self.failed.emit(details)
            return
        artifact_path = self._cache_artifact_path
        if artifact_path is None:
            self.failed.emit("cache worker lost artifact path")
            return
        cache_path, cache_summary = self._cache_done
        self.status_changed.emit("Render preview cache ready; streaming preview chunks.")
        self._start_cache_stream(generation, artifact_path, cache_path, cache_summary)

    def _on_cache_process_error(
        self,
        generation: int,
        process: QProcess,
        error: QProcess.ProcessError,
    ) -> None:
        if generation != self._generation or process is not self._cache_process:
            return
        self.failed.emit(f"cache worker process error: {error.name}")

    def _stop_cache_process(self) -> None:
        process = self._cache_process
        if process is None:
            return
        self._cache_process = None
        process.kill()
        process.deleteLater()


__all__ = [
    "MeshArtifactLoader",
    "MeshPreviewChunk",
    "MeshPreviewSummary",
    "iter_mesh_preview_chunks_sync",
]
