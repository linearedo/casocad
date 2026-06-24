from __future__ import annotations

import numpy as np

from app.meshing.workspace import _DEFAULT_SCRIPT
from core.meshing import load_meshable_domains


class _CaptureEmit:
    def __init__(self) -> None:
        self.element_types: list[str] = []
        self.vertices: list[np.ndarray] = []
        self.tag_names: list[str] = []

    def __call__(
        self,
        *,
        element_type: list[str],
        vertices: list[np.ndarray],
        tag_name: list[str],
    ) -> None:
        self.element_types.extend(element_type)
        self.vertices.extend(vertices)
        self.tag_names.extend(tag_name)


def test_default_meshing_script_emits_conforming_slice_triangles() -> None:
    capture = _CaptureEmit()
    env = {
        "__builtins__": __builtins__,
        "domains": load_meshable_domains("scene.json"),
        "np": np,
        "emit": capture,
    }

    exec(_DEFAULT_SCRIPT, env, env)

    assert capture.element_types
    assert set(capture.element_types) == {"triangle"}
    assert set(capture.tag_names) == {"fluid_slice"}
    vertex_counts = {vertices.shape[0] for vertices in capture.vertices}
    assert vertex_counts == {3}
