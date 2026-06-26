from __future__ import annotations

from app import main_window


class _Viewport:
    def __init__(self) -> None:
        self.cleared = 0

    def clear_boolean_preview(self) -> None:
        self.cleared += 1


def test_empty_boolean_preview_clear_is_deferred(monkeypatch) -> None:
    callbacks = []
    window = main_window.MainWindow.__new__(main_window.MainWindow)
    window.viewport = _Viewport()

    monkeypatch.setattr(
        main_window.QTimer,
        "singleShot",
        lambda delay_ms, callback: callbacks.append((delay_ms, callback)),
    )

    window._on_sdf_op_preview_requested("", [])

    assert window.viewport.cleared == 0
    assert len(callbacks) == 1
    assert callbacks[0][0] == 0

    callbacks[0][1]()

    assert window.viewport.cleared == 1
