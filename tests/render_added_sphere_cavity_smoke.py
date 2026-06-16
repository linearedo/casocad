from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.main_window import MainWindow
from core.sdf import Difference, Sphere


def main() -> int:
    application = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    window.viewport.set_grid_visible(False)
    window.viewport.set_components_visible(False)
    window.viewport.set_sdf_opacity(0.4)
    window.viewport.camera.yaw_degrees = 0.0
    window.viewport.camera.pitch_degrees = 82.0
    window.viewport.camera.distance = 3.0
    window.viewport.camera.target = (0.72, 0.0, 0.0)
    window.viewport.update()
    state: dict[str, object] = {}

    def subtract_sphere() -> None:
        state["before"] = window.viewport.grabFramebuffer()
        assert window.document.fluid_domain is not None
        old_root = window.document.fluid_domain.root
        sphere_handle = window.document.add_primitive("sphere")
        sphere = window.document.node(sphere_handle)
        if not isinstance(sphere, Sphere):
            application.exit(1)
            return
        sphere.center = (0.72, 0.0, 0.0)
        sphere.radius = 0.26
        result_handle = window.document.combine(
            window.document.handle_for(old_root),
            sphere_handle,
            "difference",
        )
        result = window.document.node(result_handle)
        if (
            not isinstance(result, Difference)
            or window.document.fluid_domain is None
            or window.document.fluid_domain.root is not result
        ):
            application.exit(1)
            return
        state["sphere_id"] = sphere.object_id
        window._publish_document(frame=False)
        window.viewport.update()
        QTimer.singleShot(700, verify)

    def verify() -> None:
        before = state["before"]
        after = window.viewport.grabFramebuffer()
        center_x = after.width() // 2
        center_y = after.height() // 2
        difference = 0
        for y in range(center_y - 90, center_y + 91, 3):
            for x in range(center_x - 90, center_x + 91, 3):
                first = before.pixelColor(x, y)
                second = after.pixelColor(x, y)
                difference += (
                    abs(second.red() - first.red())
                    + abs(second.green() - first.green())
                    + abs(second.blue() - first.blue())
                )
        if difference < 35_000:
            sys.stderr.write(
                "Added sphere cavity render smoke failed: final Difference "
                f"did not visibly change the scene (difference={difference})\n"
            )
            application.exit(1)
            return
        root = window.document.fluid_domain.root
        center = np.asarray([0.72], dtype=np.float64)
        zero = np.asarray([0.0], dtype=np.float64)
        if root.to_numpy(center, zero, zero)[0] <= 0.0:
            sys.stderr.write(
                "Added sphere cavity render smoke failed: sphere center "
                "was not removed from the final SDF\n"
            )
            application.exit(1)
            return
        window.close()
        application.exit(0)

    QTimer.singleShot(800, subtract_sphere)
    QTimer.singleShot(10_000, lambda: application.exit(1))
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
