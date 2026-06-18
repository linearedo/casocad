from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow
from core.mesher import FluidDomain
from core.scene import SceneDocument
from core.sdf import (
    PlacedSDF1D,
    PlacedSDF2D,
    RectangleProfile,
    SegmentProfile,
)


def cyan_pixels(image: object) -> int:
    count = 0
    for y in range(0, image.height(), 2):
        for x in range(0, image.width(), 2):
            color = image.pixelColor(x, y)
            if (
                color.green() > 100
                and color.blue() > 100
                and color.green() > color.red() + 20
            ):
                count += 1
    return count


def main() -> int:
    application = QApplication(sys.argv)
    window = MainWindow()
    domain = PlacedSDF2D(
        name="fluid",
        object_id=1,
        profile=RectangleProfile(half_size=(0.8, 0.5)),
    )
    inlet = PlacedSDF1D(
        name="inlet",
        object_id=2,
        profile=SegmentProfile(half_length=0.5),
        origin=(-0.8, 0.0, 0.0),
        axis_u=(0.0, 1.0, 0.0),
    )
    window.document = SceneDocument(
        objects=[domain, inlet],
        fluid_domain=FluidDomain(domain, (inlet,)),
    )
    window._publish_document(frame=True)
    window.show()
    window.viewport.set_grid_visible(False)
    window.viewport.set_components_visible(False)
    window.viewport.set_sdf_opacity(1.0)
    window.viewport.camera.yaw_degrees = 0.0
    window.viewport.camera.pitch_degrees = -89.0
    window.viewport.camera.distance = 3.0
    window.viewport.camera.target = (0.0, 0.0, 0.0)
    window.viewport.update()

    def fail(message: str) -> None:
        sys.stderr.write(f"Selected 1D render smoke failed: {message}\n")
        application.exit(1)

    def select_inlet() -> None:
        before = cyan_pixels(window.viewport.grabFramebuffer())
        window.scene_tree.select_handle(window.document.handle_for(inlet))
        window.viewport.update()

        def verify() -> None:
            after = cyan_pixels(window.viewport.grabFramebuffer())
            if after < before + 20:
                fail(
                    "selecting the inlet did not highlight its 1D segment "
                    f"(before={before}, after={after})"
                )
                return
            window.close()
            application.exit(0)

        QTimer.singleShot(300, verify)

    QTimer.singleShot(700, select_inlet)
    QTimer.singleShot(10_000, lambda: fail("workflow timed out"))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
