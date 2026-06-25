from __future__ import annotations

import json
import numpy as np
import subprocess
import sys

from app.meshing.viewer.render_cache import (
    build_render_cache,
    ensure_render_cache,
    iter_render_cache_chunks,
    render_cache_paths,
)
from core.meshing import MeshArtifactWriter


def test_render_cache_packs_semantic_mesh_for_viewer(tmp_path) -> None:
    path = tmp_path / "mesh.arrow"
    triangle = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    quad = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    with MeshArtifactWriter(path) as writer:
        writer.write_batch(
            element_type=["triangle", "quad"],
            vertices=[triangle, quad],
            tag_name=["fluid", "wall"],
        )

    cache_path, summary = ensure_render_cache(path, max_preview_vertices=300)
    cache_arrow, cache_json = render_cache_paths(path, 300)
    chunks = list(iter_render_cache_chunks(cache_path))

    assert cache_path == cache_arrow
    assert cache_json.exists()
    assert summary.element_count == 2
    assert summary.preview_triangle_count == 3
    assert summary.preview_edge_count == 7
    assert summary.tag_names == ("fluid", "wall")
    assert {chunk.primitive_type for chunk in chunks} == {"triangle", "line"}
    assert all(chunk.positions.dtype == np.float32 for chunk in chunks)
    assert all(chunk.positions.shape == chunk.colors.shape for chunk in chunks)

    reused_path, reused_summary = ensure_render_cache(path, max_preview_vertices=300)

    assert reused_path == cache_path
    assert reused_summary == summary


def test_render_cache_worker_builds_cache_in_subprocess(tmp_path) -> None:
    path = tmp_path / "mesh.arrow"
    triangle = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    with MeshArtifactWriter(path) as writer:
        writer.write_batch(
            element_type=["triangle"],
            vertices=[triangle],
            tag_name=["fluid"],
        )

    result = subprocess.run(
        [sys.executable, "-m", "app.meshing.viewer.cache_worker", str(path), "300"],
        check=False,
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0, result.stderr
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    assert messages[0]["type"] == "started"
    assert messages[-1]["type"] == "done"
    assert messages[-1]["summary"]["element_count"] == 1
    assert messages[-1]["summary"]["preview_triangle_count"] == 1
    assert (tmp_path / "mesh.preview-300.arrow").exists()


def test_render_cache_flushes_large_input_batches_incrementally(tmp_path) -> None:
    path = tmp_path / "mesh.arrow"
    cache_path = tmp_path / "mesh.preview.arrow"
    triangle = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    with MeshArtifactWriter(path) as writer:
        writer.write_batch(
            element_type=["triangle"] * 12,
            vertices=[triangle] * 12,
            tag_name=["fluid"] * 12,
        )

    summary = build_render_cache(
        path,
        cache_path,
        max_preview_vertices=36,
        max_rows_per_chunk=5,
    )
    chunks = list(iter_render_cache_chunks(cache_path))

    assert summary.element_count == 12
    assert summary.preview_triangle_count == 12
    assert len(chunks) > 2
    assert all(chunk.positions.shape[0] <= 5 for chunk in chunks)
