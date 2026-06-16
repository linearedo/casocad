from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc
from numpy.typing import NDArray


FIELDS = (
    pa.field("x", pa.float64()),
    pa.field("y", pa.float64()),
    pa.field("z", pa.float64()),
    pa.field("i", pa.uint64()),
    pa.field("j", pa.uint64()),
    pa.field("k", pa.uint64()),
    pa.field("node_type", pa.uint8()),
    pa.field("tag_ids", pa.list_(pa.uint16())),
    pa.field("level", pa.uint8()),
)


class ArrowWriter:
    def __init__(self, path: str | Path, metadata: dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(metadata, separators=(",", ":"), sort_keys=True).encode(
            "utf-8"
        )
        self.schema = pa.schema(FIELDS, metadata={b"metadata": encoded})
        self._sink = pa.OSFile(str(self.path), "wb")
        options = ipc.IpcWriteOptions(compression="zstd")
        self._writer = ipc.new_file(self._sink, self.schema, options=options)
        self._closed = False

    def write_batch(
        self,
        *,
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        z: NDArray[np.float64],
        i: NDArray[np.uint64],
        j: NDArray[np.uint64],
        k: NDArray[np.uint64],
        node_type: NDArray[np.uint8],
        tag_ids: Sequence[Sequence[int]],
        level: NDArray[np.uint8],
    ) -> None:
        arrays = [
            pa.array(x, type=pa.float64()),
            pa.array(y, type=pa.float64()),
            pa.array(z, type=pa.float64()),
            pa.array(i, type=pa.uint64()),
            pa.array(j, type=pa.uint64()),
            pa.array(k, type=pa.uint64()),
            pa.array(node_type, type=pa.uint8()),
            pa.array(tag_ids, type=pa.list_(pa.uint16())),
            pa.array(level, type=pa.uint8()),
        ]
        batch = pa.RecordBatch.from_arrays(arrays, schema=self.schema)
        self._writer.write_batch(batch)

    def close(self) -> None:
        if self._closed:
            return
        self._writer.close()
        self._sink.close()
        self._closed = True

    def __enter__(self) -> ArrowWriter:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()
