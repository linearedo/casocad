from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.ipc as ipc


def read_lattice(path: str | Path) -> tuple[pa.Table, dict[str, Any]]:
    with pa.memory_map(str(path), "r") as source:
        reader = ipc.open_file(source)
        raw_metadata = reader.schema.metadata
        if raw_metadata is None or b"metadata" not in raw_metadata:
            raise ValueError("Arrow lattice is missing metadata")
        metadata = json.loads(raw_metadata[b"metadata"].decode("utf-8"))
        return reader.read_all(), metadata
