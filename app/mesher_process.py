from __future__ import annotations

import json
import pickle
import sys
import traceback
from pathlib import Path

from core.mesher import LatticeMesher


def _emit(message: dict[str, object]) -> None:
    print(json.dumps(message, separators=(",", ":")), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 3:
        _emit(
            {
                "type": "failed",
                "version": -1,
                "message": "usage: mesher_process <input> <arrow-output> <result>",
            }
        )
        return 2
    input_path = Path(args[0])
    output_path = Path(args[1])
    result_path = Path(args[2])
    with input_path.open("rb") as stream:
        version, domain, config = pickle.load(stream)
    preview_index = 0

    def emit_preview_chunk(chunk: object) -> None:
        nonlocal preview_index
        chunk_path = result_path.with_name(
            f"{result_path.stem}-preview-{preview_index}.pickle"
        )
        preview_index += 1
        with chunk_path.open("wb") as stream:
            pickle.dump(chunk, stream, protocol=pickle.HIGHEST_PROTOCOL)
        _emit(
            {
                "type": "preview_chunk",
                "version": version,
                "path": str(chunk_path),
            }
        )

    try:
        result = LatticeMesher(domain, config).mesh(
            output_path,
            lambda done, total: _emit(
                {
                    "type": "progress",
                    "version": version,
                    "value": round(100.0 * done / total),
                }
            ),
            emit_preview_chunk,
        )
        with result_path.open("wb") as stream:
            pickle.dump(result, stream, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as error:
        _emit(
            {
                "type": "failed",
                "version": version,
                "message": str(error),
                "traceback": traceback.format_exc(),
            }
        )
        return 1
    _emit({"type": "completed", "version": version})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
