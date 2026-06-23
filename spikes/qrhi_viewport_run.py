"""Phase 3 — run the real QRhi viewport on the default casoCAD scene.

Opens a QRhiWidget viewport (the reusable QRhiInterpreterRenderer) showing the
default von-Karman document scene. Drag to orbit, wheel to zoom. On Linux it uses
Vulkan automatically; override with QRHI_BACKEND=opengl|vulkan.

    .venv/bin/python spikes/qrhi_viewport_run.py
"""
import sys

from PySide6.QtWidgets import QApplication

from core.render_ir import build_render_ir
from core.scene import SceneDocument
from core.sdf import SDFTree
from app.viewport.renderers.qrhi.viewport import QRhiViewportWidget


def main() -> int:
    app = QApplication(sys.argv)
    doc = SceneDocument.default()
    root = doc.objects[0]
    render_ir = build_render_ir(SDFTree(root=root))
    print("scene nodes:", len(render_ir.nodes), "| supported:", render_ir.supported)

    w = QRhiViewportWidget()
    w.set_scene(render_ir)
    w.frame_target((0.0, 0.0, 0.0), 6.0)
    w.setWindowTitle("QRhi viewport — drag = orbit, wheel = zoom")
    w.resize(900, 640)
    w.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
