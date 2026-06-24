from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import traceback
from typing import Any

from .script_runner import run_meshing_script


def _emit(message: dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.__stdout__.flush()


def _emit_captured_logs(stdout_text: str, stderr_text: str) -> None:
    for line in stdout_text.splitlines():
        if line:
            _emit({"type": "log", "level": "info", "message": line})
    for line in stderr_text.splitlines():
        if line:
            _emit({"type": "log", "level": "warning", "message": line})


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        _emit({"type": "error", "message": "usage: python -m app.meshing.worker JOB"})
        return 2
    job_path = Path(args[0])
    try:
        job = json.loads(job_path.read_text(encoding="utf-8"))
        _emit({"type": "started"})
        stdout_buffer = io.StringIO()
        stderr_buffer = io.StringIO()
        with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(
            stderr_buffer
        ):
            result = run_meshing_script(
                scene_path=job["scene_path"],
                script_text=job["script_text"],
                output_path=job["output_path"],
                metadata=job.get("metadata"),
                preview_limit=int(job.get("preview_limit", 200)),
            )
        _emit_captured_logs(stdout_buffer.getvalue(), stderr_buffer.getvalue())
        _emit(
            {
                "type": "done",
                "output_path": str(result.output_path),
                "element_count": result.element_count,
                "preview_rows": [list(row) for row in result.preview_rows],
            }
        )
        return 0
    except Exception:  # noqa: BLE001
        _emit({"type": "error", "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
