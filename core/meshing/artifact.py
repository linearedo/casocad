from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc


VERTEX_TYPE = pa.list_(pa.float64(), list_size=3)
VERTICES_TYPE = pa.list_(VERTEX_TYPE)
MESH_ARTIFACT_SCHEMA = pa.schema(
    (
        pa.field("element_type", pa.string()),
        pa.field("vertices", VERTICES_TYPE),
        pa.field("tag_name", pa.string()),
    )
)


def _normalize_vertices(vertices: Sequence[object]) -> list[list[list[float]]]:
    rows: list[list[list[float]]] = []
    for item in vertices:
        array = np.asarray(item, dtype=np.float64)
        if array.ndim != 2 or array.shape[1] != 3:
            raise ValueError("each mesh element vertices item must have shape (N, 3)")
        rows.append(array.tolist())
    return rows


class MeshArtifactWriter:
    """Stream mesh elements to an Arrow artifact.

    Rows follow the minimal mesher integration schema:
    ``element_type``, ``vertices``, ``tag_name``.
    """

    def __init__(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(metadata or {}, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        self.schema = MESH_ARTIFACT_SCHEMA.with_metadata({b"metadata": encoded})
        self._sink = pa.OSFile(str(self.path), "wb")
        options = ipc.IpcWriteOptions(compression="zstd")
        self._writer = ipc.new_file(self._sink, self.schema, options=options)
        self._closed = False

    def write_batch(
        self,
        *,
        element_type: Sequence[str],
        vertices: Sequence[object],
        tag_name: Sequence[str],
    ) -> None:
        if not (len(element_type) == len(vertices) == len(tag_name)):
            raise ValueError("element_type, vertices, and tag_name lengths must match")
        arrays = [
            pa.array(element_type, type=pa.string()),
            pa.array(_normalize_vertices(vertices), type=VERTICES_TYPE),
            pa.array(tag_name, type=pa.string()),
        ]
        batch = pa.RecordBatch.from_arrays(arrays, schema=self.schema)
        self._writer.write_batch(batch)

    def close(self) -> None:
        if self._closed:
            return
        self._writer.close()
        self._sink.close()
        self._closed = True

    def __enter__(self) -> MeshArtifactWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()


def read_mesh_artifact(path: str | Path) -> tuple[pa.Table, dict[str, Any]]:
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        metadata = reader.schema.metadata or {}
        raw = metadata.get(b"metadata", b"{}")
        return reader.read_all(), json.loads(raw.decode("utf-8"))


__all__ = [
    "MESH_ARTIFACT_SCHEMA",
    "MeshArtifactWriter",
    "read_mesh_artifact",
]
