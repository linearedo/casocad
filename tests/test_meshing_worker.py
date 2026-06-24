from __future__ import annotations

import json
import subprocess
import sys


def test_meshing_worker_runs_script_in_subprocess(tmp_path) -> None:
    output = tmp_path / "mesh.arrow"
    job = tmp_path / "job.json"
    script = """\
import numpy as np
emit(
    element_type=["triangle"],
    vertices=[
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
    ],
    tag_name=["worker_test"],
)
"""
    job.write_text(
        json.dumps(
            {
                "scene_path": "scene.json",
                "script_text": script,
                "output_path": str(output),
                "metadata": {"source": "worker-test"},
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [sys.executable, "-m", "app.meshing.worker", str(job)],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    assert messages[0]["type"] == "started"
    assert messages[-1]["type"] == "done"
    assert messages[-1]["element_count"] == 1
    assert messages[-1]["preview_rows"] == [["triangle", 3, "worker_test"]]
    assert output.exists()
