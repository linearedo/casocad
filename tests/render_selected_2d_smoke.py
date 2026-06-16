from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow
from core.mesher import FluidDomain
from core.scene import SceneDocument
from core.sdf import Box, PlacedSDF2D, RectangleProfile


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
    box = Box(
        name="flow_volume",
        object_id=1,
        half_size=(0.8, 0.6, 0.5),
    )
    inlet = PlacedSDF2D(
        name="inlet",
        object_id=2,
        profile=RectangleProfile(half_size=(0.45, 0.35)),
        origin=(0.0, 0.6, 0.0),
        axis_u=(1.0, 0.0, 0.0),
        axis_v=(0.0, 0.0, 1.0),
    )
    window.document = SceneDocument(
        objects=[box, inlet],
        fluid_domain=FluidDomain(box, (inlet,)),
    )
    window._publish_document(frame=True)
    window.show()
    window.viewport.set_grid_visible(False)
    window.viewport.set_components_visible(False)
    window.viewport.set_sdf_opacity(1.0)
    window.viewport.camera.yaw_degrees = 0.0
    window.viewport.camera.pitch_degrees = 0.0
    window.viewport.camera.distance = 3.0
    window.viewport.camera.target = (0.0, 0.0, 0.0)
    window.viewport.update()

    def fail(message: str) -> None:
        sys.stderr.write(f"Selected 2D render smoke failed: {message}\n")
        application.exit(1)

    def select_inlet() -> None:
        before = cyan_pixels(window.viewport.grabFramebuffer())
        window.scene_tree.select_handle(window.document.handle_for(inlet))
        window.viewport.update()

        def verify() -> None:
            after = cyan_pixels(window.viewport.grabFramebuffer())
            if after < before + 100:
                fail(
                    "selecting the inlet did not draw its highlighted profile "
                    f"(before={before}, after={after})"
                )
                return
            window.viewport.begin_boundary_region_tool(box)

            def verify_suppression() -> None:
                suppressed = cyan_pixels(window.viewport.grabFramebuffer())
                if suppressed >= after - 50:
                    fail(
                        "selected inlet highlight remained visible in Boundary "
                        f"Region mode (selected={after}, suppressed={suppressed})"
                    )
                    return
                window.close()
                application.exit(0)

            QTimer.singleShot(300, verify_suppression)

        QTimer.singleShot(300, verify)

    QTimer.singleShot(700, select_inlet)
    QTimer.singleShot(10_000, lambda: fail("workflow timed out"))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
