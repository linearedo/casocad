from __future__ import annotations

import numpy as np

from app.meshing.viewer.loader import (
    MeshPreviewChunk,
    MeshPreviewSummary,
    iter_mesh_preview_chunks,
)
from core.meshing import MeshArtifactWriter


def test_mesh_preview_loader_triangulates_artifact_rows(tmp_path) -> None:
    path = tmp_path / "mesh.arrow"
    with MeshArtifactWriter(path) as writer:
        writer.write_batch(
            element_type=["point", "triangle", "quad"],
            vertices=[
                np.array([[0.0, 0.0, 0.0]], dtype=np.float64),
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                    dtype=np.float64,
                ),
                np.array(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [1.0, 1.0, 0.0],
                        [0.0, 1.0, 0.0],
                    ],
                    dtype=np.float64,
                ),
            ],
            tag_name=["fluid", "wall", "wall"],
        )

    items = list(iter_mesh_preview_chunks(path, max_rows_per_chunk=2))
    chunks = [item for item in items if isinstance(item, MeshPreviewChunk)]
    summary = items[-1]

    assert isinstance(summary, MeshPreviewSummary)
    assert len(chunks) == 2
    assert sum(chunk.triangle_count for chunk in chunks) == 7
    assert sum(chunk.edge_count for chunk in chunks) == 7
    assert summary.element_count == 3
    assert summary.preview_triangle_count == 7
    assert summary.preview_edge_count == 7
    assert summary.tag_names == ("fluid", "wall")
    assert not summary.truncated
    assert chunks[0].vertices.shape[1] == 3
    assert chunks[0].colors.shape == chunks[0].vertices.shape
    assert chunks[0].wire_vertices.shape[1] == 3
    assert chunks[0].wire_colors.shape == chunks[0].wire_vertices.shape


def test_mesh_preview_loader_respects_vertex_budget(tmp_path) -> None:
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
            element_type=["triangle"] * 10,
            vertices=[triangle] * 10,
            tag_name=["fluid"] * 10,
        )

    items = list(iter_mesh_preview_chunks(path, max_preview_vertices=9))
    chunks = [item for item in items if isinstance(item, MeshPreviewChunk)]
    summary = items[-1]

    assert isinstance(summary, MeshPreviewSummary)
    assert sum(chunk.vertices.shape[0] for chunk in chunks) == 9
    assert summary.preview_triangle_count == 3
    assert summary.preview_edge_count == 9
    assert summary.truncated
