from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import sys
import traceback
from typing import Any

from .render_cache import ensure_render_cache


def _emit(message: dict[str, Any]) -> None:
    sys.__stdout__.write(json.dumps(message, separators=(",", ":")) + "\n")
    sys.__stdout__.flush()


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        _emit(
            {
                "type": "error",
                "message": (
                    "usage: python -m app.meshing.viewer.cache_worker "
                    "MESH_ARTIFACT MAX_PREVIEW_VERTICES"
                ),
            }
        )
        return 2
    try:
        artifact_path = Path(args[0])
        max_preview_vertices = int(args[1])
        _emit({"type": "started"})
        cache_path, summary = ensure_render_cache(
            artifact_path,
            max_preview_vertices=max_preview_vertices,
        )
        _emit(
            {
                "type": "done",
                "cache_path": str(cache_path),
                "summary": asdict(summary),
            }
        )
        return 0
    except Exception:  # noqa: BLE001
        _emit({"type": "error", "traceback": traceback.format_exc()})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
