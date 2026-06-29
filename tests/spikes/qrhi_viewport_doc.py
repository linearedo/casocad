"""Step 1 — document-aware QRhi viewport.

The viewport renders whatever the document is, driven ONLY by the
``document_changed`` signal (no manual set_scene). To prove live updates, a timer
adds a sphere to the document after 2.5s and re-emits — the viewport should update
on its own.

    .venv/bin/python spikes/qrhi_viewport_doc.py
"""
import sys

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from app.signals import signals
from app.viewport.renderers.qrhi.viewport import QRhiViewportWidget
from core.scene import SceneDocument


def main() -> int:
    app = QApplication(sys.argv)
    document = SceneDocument.default()

    w = QRhiViewportWidget()  # auto-connects to signals.document_changed
    w.setWindowTitle("QRhi viewport (document-aware) — drag to orbit")
    w.resize(900, 640)
    w.show()

    # Initial scene: drive it purely through the signal.
    signals.document_changed.emit(document)

    # Live update: add a sphere after 2.5s and re-emit. The viewport should change
    # without any direct call.
    def add_sphere():
        document.add_primitive("sphere")
        print("added a sphere -> emitting document_changed", flush=True)
        signals.document_changed.emit(document)

    QTimer.singleShot(2500, add_sphere)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
