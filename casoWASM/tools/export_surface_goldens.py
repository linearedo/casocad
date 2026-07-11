#!/usr/bin/env python3
"""Export display-surface goldens from the Python surface builders.

Reuses the kernel fixture definitions (export_goldens.build_fixtures) and runs
`build_viewport_surface` on a representative subset at two resolutions (dense
12 and narrow-band 96), recording status / vertex count / triangle count and
the max |sdf| at the produced vertices. The Rust surfaces crate must
reproduce these metrics.

Run from the casoCAD repo root:

    .venv/bin/python casoWASM/tools/export_surface_goldens.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from export_goldens import build_fixtures
from app.viewport.surface_builder import build_viewport_surface
from app.viewport.surface_types import ViewportSurfaceKey

FIXTURES = (
    "sphere",
    "box_oriented",
    "cylinder",
    "torus",
    "pyramid",
    "cone",
    "cappedcone",
    "boxframe",
    "polyline_tube_round",
    "bezier_tube_round",
    "extrude",
    "revolve_full",
    "revolve_partial",
    "von_karman",
    "op_union",
    "op_xor",
    "op_nested",
    "placed2d_circle",
    "placed2d_polygon",
    "placed2d_bezier_surface_open",
    "placed1d_segment",
    "placed_polyline_1d",
)
RESOLUTIONS = (12, 96)


def main() -> int:
    fixtures = build_fixtures()
    output = (
        Path(__file__).resolve().parents[1]
        / "surfaces"
        / "tests"
        / "goldens"
        / "surface_goldens.txt"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for name in FIXTURES:
        node = fixtures[name]
        node.object_id = 7  # stable color/id for parity
        for resolution in RESOLUTIONS:
            key = ViewportSurfaceKey(
                object_id=7, scene_revision=1, resolution=resolution
            )
            surface = build_viewport_surface(node, key)
            if surface.vertices.shape[0] and surface.indices.size:
                used = np.unique(surface.indices)
                v = surface.vertices.astype(np.float64)[used]
                max_err = float(
                    np.max(np.abs(node.to_numpy(v[:, 0], v[:, 1], v[:, 2])))
                )
            else:
                max_err = 0.0
            lines.append(
                " ".join(
                    (
                        "s",
                        name,
                        str(resolution),
                        surface.status,
                        str(int(surface.vertices.shape[0])),
                        str(int(surface.indices.size // 3)),
                        str(int(surface.wire_indices.size)),
                        repr(max_err),
                    )
                )
            )
            print(lines[-1])
    output.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(lines)} golden rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
