from __future__ import annotations

import json
import pickle
import subprocess
import sys

from core.mesher import FluidDomain, LatticePreviewChunk, MesherConfig
from core.sdf import Box


def test_mesher_process_writes_versioned_result(tmp_path) -> None:
    domain = FluidDomain(
        Box(
            name="box",
            object_id=1,
            center=(0.0, 0.0, 0.0),
            half_size=(0.5, 0.5, 0.5),
        )
    )
    config = MesherConfig(dx=0.5, chunk_size=1000)
    input_path = tmp_path / "input.pickle"
    output_path = tmp_path / "lattice.arrow"
    result_path = tmp_path / "result.pickle"
    with input_path.open("wb") as stream:
        pickle.dump((7, domain, config), stream)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "app.mesher_process",
            str(input_path),
            str(output_path),
            str(result_path),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=20,
    )

    messages = [
        json.loads(line)
        for line in completed.stdout.splitlines()
        if line.strip()
    ]
    with result_path.open("rb") as stream:
        result = pickle.load(stream)
    chunk_messages = [
        message for message in messages if message["type"] == "preview_chunk"
    ]
    assert chunk_messages
    with open(chunk_messages[0]["path"], "rb") as stream:
        preview_chunk = pickle.load(stream)

    assert isinstance(preview_chunk, LatticePreviewChunk)
    assert preview_chunk.preview_positions.shape[0] > 0
    assert any(
        message["type"] == "completed" and message["version"] == 7
        for message in messages
    )
    assert output_path.exists()
    assert result.row_count > 0
    assert result.path == output_path
