from __future__ import annotations

from pathlib import Path

from app.artifacts import build_render_artifact
from core.scene import SceneDocument


def test_build_render_artifact_contains_default_scene_source() -> None:
    document = SceneDocument.default()

    artifact = build_render_artifact(document.snapshot())

    assert artifact.version == document.version
    assert artifact.tree is not None
    assert "float sceneSDF(vec3 p)" in artifact.scene_source
    assert "float componentSDF(vec3 p, int component)" in artifact.scene_source


def test_build_render_artifact_allows_empty_scene() -> None:
    artifact = build_render_artifact(SceneDocument().snapshot())

    assert artifact.tree is None
    assert "float sceneSDF(vec3 p)" in artifact.scene_source
    assert "int sceneBoundaryOwnerId(vec3 p)" in artifact.scene_source
    assert "int sceneObjectId(vec3 p)" in artifact.scene_source
    assert "bool sceneSelectionOwnsBoundary" in artifact.scene_source
    assert "float sceneSelectedObjectSDF" in artifact.scene_source
    assert "int sceneSelectedObjectDimension" in artifact.scene_source
    assert "const int COMPONENT_COUNT = 0;" in artifact.scene_source
    assert "float componentSDF(vec3 p, int component)" in artifact.scene_source
    assert "int componentObjectId(int component)" in artifact.scene_source


def test_bezier_render_helper_is_declared_before_generated_scene_source() -> None:
    template = (
        Path(__file__).resolve().parents[1]
        / "app"
        / "viewport"
        / "shaders"
        / "raymarch.frag"
    ).read_text(encoding="utf-8")

    assert template.index("float quadraticBezierDistance") < template.index(
        "/*__SCENE_SDF__*/"
    )
