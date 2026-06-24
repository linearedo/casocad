from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import threading
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
from PySide6.QtCore import QObject, Signal

from .render_cache import ensure_render_cache, iter_render_cache_chunks


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


def iter_mesh_preview_chunks(
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

    def load(self, path: str | Path) -> None:
        self._generation += 1
        generation = self._generation
        artifact_path = Path(path)
        max_rows_per_chunk = self._max_rows_per_chunk
        max_preview_vertices = self._max_preview_vertices
        threading.Thread(
            target=self._load_worker,
            args=(generation, artifact_path, max_rows_per_chunk, max_preview_vertices),
            daemon=True,
        ).start()

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
        max_rows_per_chunk: int,
        max_preview_vertices: int,
    ) -> None:
        try:
            for item in iter_mesh_preview_chunks(
                path,
                max_rows_per_chunk=max_rows_per_chunk,
                max_preview_vertices=max_preview_vertices,
            ):
                if generation != self._generation:
                    return
                if isinstance(item, MeshPreviewChunk):
                    self.chunk_loaded.emit(item)
                else:
                    self.finished.emit(item)
        except Exception as exc:  # noqa: BLE001
            if generation == self._generation:
                self.failed.emit(str(exc))


__all__ = [
    "MeshArtifactLoader",
    "MeshPreviewChunk",
    "MeshPreviewSummary",
    "iter_mesh_preview_chunks",
]
