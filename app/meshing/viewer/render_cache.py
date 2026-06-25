from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from numpy.typing import NDArray
import pyarrow as pa
import pyarrow.ipc as ipc


POSITION_TYPE = pa.list_(pa.float32(), list_size=3)
RENDER_CACHE_SCHEMA = pa.schema(
    (
        pa.field("chunk_id", pa.int64()),
        pa.field("primitive_type", pa.string()),
        pa.field("position", POSITION_TYPE),
        pa.field("color", POSITION_TYPE),
    )
)


@dataclass(frozen=True)
class RenderCacheSummary:
    source_path: str
    source_mtime_ns: int
    max_preview_vertices: int
    element_count: int
    preview_vertex_count: int
    preview_triangle_count: int
    preview_edge_count: int
    tag_names: tuple[str, ...]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    truncated: bool


@dataclass(frozen=True)
class RenderCacheChunk:
    primitive_type: str
    positions: NDArray[np.float32]
    colors: NDArray[np.float32]


def render_cache_paths(
    mesh_path: str | Path,
    max_preview_vertices: int,
) -> tuple[Path, Path]:
    source = Path(mesh_path)
    stem = f"{source.stem}.preview-{int(max_preview_vertices)}"
    return source.with_name(f"{stem}.arrow"), source.with_name(f"{stem}.json")


def ensure_render_cache(
    mesh_path: str | Path,
    *,
    max_preview_vertices: int,
) -> tuple[Path, RenderCacheSummary]:
    source = Path(mesh_path)
    cache_path, summary_path = render_cache_paths(source, max_preview_vertices)
    summary = _read_summary(summary_path)
    if (
        summary is not None
        and cache_path.exists()
        and summary.source_path == str(source)
        and summary.source_mtime_ns == source.stat().st_mtime_ns
        and summary.max_preview_vertices == int(max_preview_vertices)
    ):
        return cache_path, summary
    summary = build_render_cache(
        source,
        cache_path,
        max_preview_vertices=max_preview_vertices,
    )
    summary_path.write_text(
        json.dumps(asdict(summary), separators=(",", ":"), sort_keys=True),
        encoding="utf-8",
    )
    return cache_path, summary


def render_cache_summary_from_dict(raw: object) -> RenderCacheSummary:
    if not isinstance(raw, dict):
        raise TypeError("render cache summary must be a JSON object")
    return RenderCacheSummary(
        source_path=str(raw["source_path"]),
        source_mtime_ns=int(raw["source_mtime_ns"]),
        max_preview_vertices=int(raw["max_preview_vertices"]),
        element_count=int(raw["element_count"]),
        preview_vertex_count=int(raw["preview_vertex_count"]),
        preview_triangle_count=int(raw["preview_triangle_count"]),
        preview_edge_count=int(raw["preview_edge_count"]),
        tag_names=tuple(str(item) for item in raw["tag_names"]),
        bounds_min=tuple(float(item) for item in raw["bounds_min"]),
        bounds_max=tuple(float(item) for item in raw["bounds_max"]),
        truncated=bool(raw["truncated"]),
    )


def iter_render_cache_chunks(path: str | Path) -> Iterator[RenderCacheChunk]:
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        for index in range(reader.num_record_batches):
            batch = reader.get_batch(index)
            if batch.num_rows == 0:
                continue
            primitive_type = str(batch.column("primitive_type")[0].as_py())
            positions = _fixed_vec3_column(batch.column("position"))
            colors = _fixed_vec3_column(batch.column("color"))
            yield RenderCacheChunk(
                primitive_type=primitive_type,
                positions=positions,
                colors=colors,
            )


def build_render_cache(
    mesh_path: str | Path,
    cache_path: str | Path,
    *,
    max_preview_vertices: int,
    max_rows_per_chunk: int = 4096,
) -> RenderCacheSummary:
    source = Path(mesh_path)
    output = Path(cache_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    element_count = _count_mesh_elements(source)
    preview_vertices = 0
    preview_triangles = 0
    preview_edges = 0
    truncated = False
    tags: set[str] = set()
    bounds_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    bounds_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)

    with pa.OSFile(str(output), "wb") as sink:
        writer = ipc.new_file(
            sink,
            RENDER_CACHE_SCHEMA,
            options=ipc.IpcWriteOptions(compression="zstd"),
        )
        tri_buffer = _RenderPrimitiveBuffer(
            writer,
            primitive_type="triangle",
            max_rows_per_chunk=max_rows_per_chunk,
        )
        line_buffer = _RenderPrimitiveBuffer(
            writer,
            primitive_type="line",
            max_rows_per_chunk=max_rows_per_chunk,
        )
        try:
            for batch in _iter_mesh_batches(source):
                vertex_rows = batch.column("vertices")
                element_types = batch.column("element_type")
                tag_names = batch.column("tag_name")
                for row_index in range(batch.num_rows):
                    element_type = str(element_types[row_index].as_py())
                    vertices = vertex_rows[row_index].as_py()
                    tag_name = str(tag_names[row_index].as_py())
                    source_vertices = np.asarray(vertices, dtype=np.float64)
                    if (
                        source_vertices.ndim != 2
                        or source_vertices.shape[1] != 3
                        or source_vertices.shape[0] == 0
                    ):
                        continue
                    tags.add(tag_name)
                    bounds_min = np.minimum(bounds_min, source_vertices.min(axis=0))
                    bounds_max = np.maximum(bounds_max, source_vertices.max(axis=0))
                    triangles = _triangulate_vertices(element_type, source_vertices)
                    if triangles.size == 0:
                        continue
                    remaining = int(max_preview_vertices) - preview_vertices
                    clipped = False
                    if triangles.shape[0] > remaining:
                        truncated = True
                        clipped = True
                        if remaining <= 0:
                            break
                        keep = remaining - (remaining % 3)
                        triangles = triangles[:keep]
                    if triangles.size == 0:
                        break
                    color = np.asarray(_tag_color(tag_name), dtype=np.float32)
                    tri_buffer.add(triangles, color)
                    preview_vertices += int(triangles.shape[0])
                    preview_triangles += int(triangles.shape[0] // 3)
                    if not clipped:
                        edges = _wire_edges_vertices(element_type, source_vertices)
                        if edges.size:
                            line_buffer.add(edges, color)
                            preview_edges += int(edges.shape[0] // 2)
                    if preview_vertices >= int(max_preview_vertices):
                        truncated = True
                        break
                if truncated:
                    break
            tri_buffer.flush()
            line_buffer.flush()
        finally:
            writer.close()

    if not np.all(np.isfinite(bounds_min)):
        bounds_min = np.zeros(3, dtype=np.float64)
        bounds_max = np.zeros(3, dtype=np.float64)
    return RenderCacheSummary(
        source_path=str(source),
        source_mtime_ns=source.stat().st_mtime_ns,
        max_preview_vertices=int(max_preview_vertices),
        element_count=element_count,
        preview_vertex_count=preview_vertices,
        preview_triangle_count=preview_triangles,
        preview_edge_count=preview_edges,
        tag_names=tuple(sorted(tags)),
        bounds_min=tuple(float(value) for value in bounds_min),
        bounds_max=tuple(float(value) for value in bounds_max),
        truncated=truncated,
    )


def _read_summary(path: Path) -> RenderCacheSummary | None:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return render_cache_summary_from_dict(raw)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _iter_mesh_batches(path: Path) -> Iterator[pa.RecordBatch]:
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        for index in range(reader.num_record_batches):
            yield reader.get_batch(index)


def _count_mesh_elements(path: Path) -> int:
    count = 0
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        for index in range(reader.num_record_batches):
            count += reader.get_batch(index).num_rows
    return count


def _fixed_vec3_column(column: pa.Array) -> NDArray[np.float32]:
    return np.asarray(column.values, dtype=np.float32).reshape(-1, 3)


def _write_cache_vertices(
    writer: ipc.RecordBatchFileWriter,
    *,
    primitive_type: str,
    positions: NDArray[np.float32],
    colors: NDArray[np.float32],
    max_rows_per_chunk: int,
) -> None:
    count = int(positions.shape[0])
    for start in range(0, count, max_rows_per_chunk):
        end = min(start + max_rows_per_chunk, count)
        rows = end - start
        arrays = [
            pa.array([start // max_rows_per_chunk] * rows, type=pa.int64()),
            pa.array([primitive_type] * rows, type=pa.string()),
            pa.array(positions[start:end].tolist(), type=POSITION_TYPE),
            pa.array(colors[start:end].tolist(), type=POSITION_TYPE),
        ]
        writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=RENDER_CACHE_SCHEMA))


class _RenderPrimitiveBuffer:
    def __init__(
        self,
        writer: ipc.RecordBatchFileWriter,
        *,
        primitive_type: str,
        max_rows_per_chunk: int,
    ) -> None:
        self._writer = writer
        self._primitive_type = primitive_type
        self._max_rows_per_chunk = max(1, int(max_rows_per_chunk))
        self._positions: list[NDArray[np.float32]] = []
        self._colors: list[NDArray[np.float32]] = []
        self._row_count = 0

    def add(
        self,
        positions: NDArray[np.float32],
        color: NDArray[np.float32],
    ) -> None:
        if positions.size == 0:
            return
        start = 0
        while start < positions.shape[0]:
            remaining = self._max_rows_per_chunk - self._row_count
            take = min(remaining, positions.shape[0] - start)
            part = positions[start : start + take].astype(np.float32, copy=False)
            self._positions.append(part)
            self._colors.append(np.tile(color, (take, 1)))
            self._row_count += take
            start += take
            if self._row_count >= self._max_rows_per_chunk:
                self.flush()

    def flush(self) -> None:
        if not self._positions:
            return
        _write_cache_vertices(
            self._writer,
            primitive_type=self._primitive_type,
            positions=np.vstack(self._positions),
            colors=np.vstack(self._colors),
            max_rows_per_chunk=self._max_rows_per_chunk,
        )
        self._positions.clear()
        self._colors.clear()
        self._row_count = 0


def _triangulate_vertices(
    element_type: str,
    vertices: NDArray[np.float64],
) -> NDArray[np.float32]:
    if element_type == "point" or vertices.shape[0] == 1:
        return _point_marker(vertices[0], 1.0e-3)
    if vertices.shape[0] < 3:
        return np.zeros((0, 3), dtype=np.float32)
    if vertices.shape[0] == 3:
        return np.asarray(vertices, dtype=np.float32)
    triangles: list[NDArray[np.float64]] = []
    anchor = vertices[0]
    for index in range(1, vertices.shape[0] - 1):
        triangles.extend((anchor, vertices[index], vertices[index + 1]))
    return np.asarray(triangles, dtype=np.float32)


def _wire_edges_vertices(
    element_type: str,
    vertices: NDArray[np.float64],
) -> NDArray[np.float32]:
    if element_type == "point" or vertices.shape[0] < 2:
        return np.zeros((0, 3), dtype=np.float32)
    if element_type == "segment" or vertices.shape[0] == 2:
        return np.asarray(vertices[:2], dtype=np.float32)
    edges: list[NDArray[np.float64]] = []
    for index in range(vertices.shape[0]):
        edges.extend((vertices[index], vertices[(index + 1) % vertices.shape[0]]))
    return np.asarray(edges, dtype=np.float32)


def _point_marker(point: NDArray[np.float64], size: float) -> NDArray[np.float32]:
    offsets = np.array(
        (
            (size, 0.0, 0.0),
            (-size, 0.0, 0.0),
            (0.0, size, 0.0),
            (0.0, 0.0, size),
        ),
        dtype=np.float64,
    )
    a, b, c, d = point + offsets
    return np.asarray(
        (a, c, d, c, b, d, b, a, d, a, b, c),
        dtype=np.float32,
    )


def _tag_color(tag_name: str) -> tuple[float, float, float]:
    value = 2166136261
    for byte in tag_name.encode("utf-8"):
        value ^= byte
        value = (value * 16777619) & 0xFFFFFFFF
    hue = (value % 360) / 360.0
    saturation = 0.58 + ((value >> 9) % 18) / 100.0
    value_channel = 0.72 + ((value >> 17) % 20) / 100.0
    return _hsv_to_rgb(hue, saturation, min(value_channel, 0.92))


def _hsv_to_rgb(hue: float, saturation: float, value: float) -> tuple[float, float, float]:
    sector = int(hue * 6.0)
    fraction = hue * 6.0 - sector
    p = value * (1.0 - saturation)
    q = value * (1.0 - fraction * saturation)
    t = value * (1.0 - (1.0 - fraction) * saturation)
    sector %= 6
    if sector == 0:
        return value, t, p
    if sector == 1:
        return q, value, p
    if sector == 2:
        return p, value, t
    if sector == 3:
        return p, q, value
    if sector == 4:
        return t, p, value
    return value, p, q


__all__ = [
    "RENDER_CACHE_SCHEMA",
    "RenderCacheChunk",
    "RenderCacheSummary",
    "build_render_cache",
    "ensure_render_cache",
    "iter_render_cache_chunks",
    "render_cache_summary_from_dict",
    "render_cache_paths",
]
