from __future__ import annotations

import numpy as np

from core.meshing import MeshArtifactWriter, read_mesh_artifact


def test_mesh_artifact_round_trip(tmp_path) -> None:
    path = tmp_path / "mesh.arrow"
    with MeshArtifactWriter(path, {"source": "test"}) as writer:
        writer.write_batch(
            element_type=["point", "triangle"],
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
            ],
            tag_name=["fluid_internal", "wall"],
        )

    table, metadata = read_mesh_artifact(path)
    assert metadata == {"source": "test"}
    assert table.num_rows == 2
    assert table.column("element_type").to_pylist() == ["point", "triangle"]
    assert table.column("tag_name").to_pylist() == ["fluid_internal", "wall"]
    vertices = table.column("vertices").to_pylist()
    assert vertices[0] == [[0.0, 0.0, 0.0]]
    assert vertices[1][2] == [0.0, 1.0, 0.0]


def test_mesh_artifact_rejects_invalid_vertices(tmp_path) -> None:
    path = tmp_path / "mesh.arrow"
    with MeshArtifactWriter(path) as writer:
        try:
            writer.write_batch(
                element_type=["bad"],
                vertices=[np.array([0.0, 0.0, 0.0])],
                tag_name=["wall"],
            )
        except ValueError as exc:
            assert "shape (N, 3)" in str(exc)
        else:
            raise AssertionError("expected invalid vertices to be rejected")
