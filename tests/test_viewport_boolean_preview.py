from __future__ import annotations

from app.viewport.renderers.qrhi.viewport import QRhiViewportWidget


class _Renderer:
    def __init__(self) -> None:
        self.scenes: list[object] = []

    def set_scene(self, render_ir: object) -> None:
        self.scenes.append(render_ir)


def _viewport() -> QRhiViewportWidget:
    viewport = QRhiViewportWidget.__new__(QRhiViewportWidget)
    viewport._renderer = _Renderer()
    viewport._dirty = False
    viewport._preview_kind = None
    viewport._boolean_preview_commit_pending = False
    viewport._committed_render_ir = "old"
    viewport._move_commit_delta = (0.0, 0.0, 0.0)
    viewport._tree = None
    return viewport


def test_boolean_preview_commit_survives_preview_clear_until_artifact_arrives() -> None:
    viewport = _viewport()

    viewport.show_scene_preview("preview", preview_kind="boolean")
    viewport.apply_committed_boolean_preview("intersection", 1, 2)
    viewport.clear_boolean_preview()

    assert viewport._renderer.scenes == ["preview"]
    assert viewport._preview_kind == "boolean"
    assert viewport._boolean_preview_commit_pending

    viewport.set_scene_artifact(None, "final")

    assert viewport._renderer.scenes == ["preview", "final"]
    assert viewport._preview_kind is None
    assert not viewport._boolean_preview_commit_pending


def test_boolean_preview_cancel_restores_committed_scene() -> None:
    viewport = _viewport()

    viewport.show_scene_preview("preview", preview_kind="boolean")
    viewport.clear_boolean_preview()

    assert viewport._renderer.scenes == ["preview", "old"]
    assert viewport._preview_kind is None
