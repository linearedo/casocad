from __future__ import annotations

from core.scene import SceneDocument


def build_scene() -> SceneDocument:
    """Return the same editable flow-domain scene used at application startup."""
    return SceneDocument.default()
