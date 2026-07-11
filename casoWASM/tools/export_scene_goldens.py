#!/usr/bin/env python3
"""Export scene round-trip goldens from the Python casoCAD serializer.

Loads the repo-root ``scene.json`` (a real saved scene with legacy boundary
selectors) plus the built-in default scene through the *Python* load/save
path, and writes the resaved JSON. The Rust round-trip test must produce
semantically identical JSON from the same inputs.

Run from the casoCAD repo root:

    .venv/bin/python casoWASM/tools/export_scene_goldens.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from core.scene import SceneDocument
from core.serialization import load_scene, save_scene

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "kernel" / "tests" / "goldens"


def resave(source: Path, destination: Path) -> None:
    document = load_scene(source)
    save_scene(document, destination)
    print(f"{source.name} -> {destination}")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resave(REPO_ROOT / "scene.json", OUTPUT_DIR / "scene_python_resave.json")

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as handle:
        default_path = Path(handle.name)
    save_scene(SceneDocument.default(), default_path)
    resave(default_path, OUTPUT_DIR / "default_python_resave.json")
    default_path.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
