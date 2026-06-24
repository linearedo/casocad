from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.meshing import MeshArtifactWriter, load_meshable_domains


@dataclass(frozen=True)
class ScriptRunResult:
    output_path: Path
    element_count: int
    preview_rows: tuple[tuple[str, int, str], ...]


class MeshScriptEmitter:
    def __init__(self, writer: MeshArtifactWriter, preview_limit: int = 200) -> None:
        self._writer = writer
        self._preview_limit = preview_limit
        self.count = 0
        self.preview_rows: list[tuple[str, int, str]] = []

    def __call__(
        self,
        *,
        element_type: str | list[str] | tuple[str, ...],
        vertices: object,
        tag_name: str | list[str] | tuple[str, ...],
    ) -> None:
        if isinstance(element_type, str):
            element_types = [element_type]
            vertex_rows = [vertices]
            tag_names = [str(tag_name)]
        else:
            element_types = [str(item) for item in element_type]
            vertex_rows = list(vertices)  # type: ignore[arg-type]
            if isinstance(tag_name, str):
                tag_names = [tag_name] * len(element_types)
            else:
                tag_names = [str(item) for item in tag_name]
        self._writer.write_batch(
            element_type=element_types,
            vertices=vertex_rows,
            tag_name=tag_names,
        )
        for etype, verts, tag in zip(element_types, vertex_rows, tag_names):
            if len(self.preview_rows) < self._preview_limit:
                array = np.asarray(verts, dtype=np.float64)
                self.preview_rows.append((etype, int(array.shape[0]), tag))
        self.count += len(element_types)


def run_meshing_script(
    *,
    scene_path: str | Path,
    script_text: str,
    output_path: str | Path,
    metadata: dict[str, Any] | None = None,
    preview_limit: int = 200,
) -> ScriptRunResult:
    domains = load_meshable_domains(scene_path)
    output = Path(output_path)
    with MeshArtifactWriter(output, metadata) as writer:
        emitter = MeshScriptEmitter(writer, preview_limit=preview_limit)
        env: dict[str, Any] = {
            "__builtins__": __builtins__,
            "domains": domains,
            "np": np,
            "emit": emitter,
        }
        exec(script_text, env, env)
    return ScriptRunResult(
        output_path=output,
        element_count=emitter.count,
        preview_rows=tuple(emitter.preview_rows),
    )


__all__ = ["MeshScriptEmitter", "ScriptRunResult", "run_meshing_script"]
