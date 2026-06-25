from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
import sys

import numpy as np
from PySide6.QtCore import QProcess, Qt, Slot
from PySide6.QtGui import QAction, QActionGroup
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from core.meshing import MeshableDomains, load_meshable_domains

from .viewer.gpu_memory import (
    GpuRenderDeviceInfo,
    choose_preview_budget_bytes,
    query_gpu_memory_info,
)
from .viewer import MeshPreviewSummary, QRhiMeshViewerWidget

_DEFAULT_PREVIEW_RENDER_TRIANGLE_LIMIT = 100_000


_DEFAULT_SCRIPT = """\
# domains: MeshableDomains
# np: numpy
# emit(element_type, vertices, tag_name): append elements to the Arrow artifact
#
# Default demo: emit a boundary-conforming 2D triangle slice through a 3D fluid domain.
# This is a marching-squares visualization preview, not a full 3D mesher.

domain = domains["fluid"]
bb = domain.bounds
resolution = 64
xs = np.linspace(bb.x_min, bb.x_max, resolution)
ys = np.linspace(bb.y_min, bb.y_max, resolution)
z = 0.5 * (bb.z_min + bb.z_max)

X, Y = np.meshgrid(xs, ys, indexing="xy")
Z = np.full_like(X, z)
points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
phi = domain.domain_sdf(points).reshape(X.shape)

edge_cache = {}


def vertex(row, col):
    return np.array([xs[col], ys[row], z], dtype=np.float64)


def edge_intersection(a, b):
    key = tuple(sorted((a, b)))
    cached = edge_cache.get(key)
    if cached is not None:
        return cached
    da = phi[a]
    db = phi[b]
    if abs(da - db) < 1.0e-15:
        t = 0.5
    else:
        t = float(np.clip(da / (da - db), 0.0, 1.0))
    point = vertex(*a) + t * (vertex(*b) - vertex(*a))
    edge_cache[key] = point
    return point


triangles = []
for row in range(resolution - 1):
    for col in range(resolution - 1):
        corners = (
            (row, col),
            (row, col + 1),
            (row + 1, col + 1),
            (row + 1, col),
        )
        values = tuple(phi[index] for index in corners)
        if all(value > 0.0 for value in values):
            continue
        polygon = []
        for index, a in enumerate(corners):
            b = corners[(index + 1) % 4]
            da = phi[a]
            db = phi[b]
            if da <= 0.0:
                polygon.append(vertex(*a))
            if (da <= 0.0) != (db <= 0.0):
                polygon.append(edge_intersection(a, b))
        if len(polygon) >= 3:
            anchor = polygon[0]
            for index in range(1, len(polygon) - 1):
                triangles.append(
                    np.asarray(
                        [anchor, polygon[index], polygon[index + 1]],
                        dtype=np.float64,
                    )
                )

if triangles:
    emit(
        element_type=["triangle"] * len(triangles),
        vertices=triangles,
        tag_name=["fluid_slice"] * len(triangles),
    )
"""


class MeshingWorkspace(QMainWindow):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("casoCAD - Meshing Workspace")
        self.resize(1200, 760)
        self._scene_path: Path | None = None
        self._domains = MeshableDomains(())
        self._last_artifact_path: Path | None = None
        self._script_process: QProcess | None = None
        self._script_job_path: Path | None = None
        self._script_output_path: Path | None = None
        self._script_stdout_buffer = ""
        self._script_failed = False
        self._build_toolbar()

        self.import_button = QPushButton("Import scene.json")
        self.import_button.clicked.connect(self.import_scene)
        self.open_artifact_button = QPushButton("Open mesh artifact...")
        self.open_artifact_button.clicked.connect(self.open_artifact)
        self.preview_limit = QSpinBox()
        self.preview_limit.setRange(1_000, 6_666_666)
        self.preview_limit.setSingleStep(25_000)
        self.preview_limit.setValue(_DEFAULT_PREVIEW_RENDER_TRIANGLE_LIMIT)
        self.preview_limit.setToolTip("Maximum render triangles loaded into the preview")
        self.preview_limit.valueChanged.connect(self._on_preview_limit_changed)
        self.auto_preview_button = QPushButton("Auto")
        self.auto_preview_button.setToolTip("Choose a conservative render-triangle limit from RAM")
        self.auto_preview_button.clicked.connect(self.auto_preview_limit)
        self.filled_toggle = QCheckBox("Filled")
        self.filled_toggle.setChecked(False)
        self.filled_toggle.setToolTip("Show filled render triangles")
        self.filled_toggle.toggled.connect(self.viewer_filled_toggled)
        self.wireframe_toggle = QCheckBox("Wireframe")
        self.wireframe_toggle.setChecked(True)
        self.wireframe_toggle.setToolTip("Show polygon edges over the mesh preview")
        self.wireframe_toggle.toggled.connect(self.viewer_wireframe_toggled)
        self.run_button = QPushButton("Run Script")
        self.run_button.clicked.connect(self.run_script)
        self.run_button.setEnabled(False)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.cancel_script)
        self.cancel_button.setEnabled(False)
        self.domain_list = QListWidget()
        self.script = QPlainTextEdit()
        self.script.setPlainText(_DEFAULT_SCRIPT)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.output_label = QLabel("No mesh artifact")
        self.viewer = QRhiMeshViewerWidget()
        self.viewer.status_changed.connect(self._log)
        self.viewer.summary_changed.connect(self._on_viewer_summary)
        self.preview = QTableWidget(0, 3)
        self.preview.setHorizontalHeaderLabels(("element_type", "vertex_count", "tag_name"))

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.import_button)
        left_layout.addWidget(self.open_artifact_button)
        limit_controls = QHBoxLayout()
        limit_controls.addWidget(QLabel("Max render triangles"))
        limit_controls.addWidget(self.preview_limit)
        limit_controls.addWidget(self.auto_preview_button)
        left_layout.addLayout(limit_controls)
        left_layout.addWidget(QLabel("Imported Domains"))
        left_layout.addWidget(self.domain_list)

        center = QWidget()
        center_layout = QVBoxLayout(center)
        center_layout.addWidget(self.output_label)
        center_layout.addWidget(self.filled_toggle)
        center_layout.addWidget(self.wireframe_toggle)
        center_layout.addWidget(self.viewer, 1)
        center_layout.addWidget(self.preview)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left)
        split.addWidget(center)
        split.setSizes((280, 920))

        viewer_page = QWidget()
        viewer_layout = QVBoxLayout(viewer_page)
        viewer_layout.addWidget(split)

        script_page = QWidget()
        script_layout = QVBoxLayout(script_page)
        controls = QHBoxLayout()
        controls.addWidget(self.run_button)
        controls.addWidget(self.cancel_button)
        controls.addStretch(1)
        script_layout.addLayout(controls)
        vertical = QSplitter(Qt.Orientation.Vertical)
        vertical.addWidget(self.script)
        vertical.addWidget(self.log)
        vertical.setSizes((520, 220))
        script_layout.addWidget(vertical)

        self.pages = QStackedWidget()
        self.pages.addWidget(viewer_page)
        self.pages.addWidget(script_page)
        self.setCentralWidget(self.pages)

    def _build_toolbar(self) -> None:
        toolbar = QToolBar("Meshing", self)
        toolbar.setObjectName("meshingWorkspaceToolbar")
        self.addToolBar(toolbar)
        group = QActionGroup(self)
        group.setExclusive(True)
        self.viewer_action = QAction("Viewer", self, checkable=True)
        self.viewer_action.setChecked(True)
        self.script_action = QAction("Script", self, checkable=True)
        group.addAction(self.viewer_action)
        group.addAction(self.script_action)
        toolbar.addAction(self.viewer_action)
        toolbar.addAction(self.script_action)
        self.viewer_action.triggered.connect(self._show_viewer_page)
        self.script_action.triggered.connect(self._show_script_page)

    def _show_viewer_page(self) -> None:
        self.pages.setCurrentIndex(0)
        self.viewer_action.setChecked(True)

    def _show_script_page(self) -> None:
        self.pages.setCurrentIndex(1)
        self.script_action.setChecked(True)

    def import_scene(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Import casoCAD Scene",
            "",
            "casoCAD scene (*.json);;All files (*)",
        )
        if path:
            self.load_scene_file(path)

    def open_artifact(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Open Mesh Artifact",
            "",
            "Arrow mesh artifact (*.arrow);;All files (*)",
        )
        if path:
            self.load_artifact_file(path)

    def load_artifact_file(self, path: str | Path) -> None:
        artifact_path = Path(path)
        self.preview.setRowCount(0)
        self._load_artifact_preview(artifact_path)

    def _load_artifact_preview(self, artifact_path: Path) -> None:
        self._last_artifact_path = artifact_path
        self.viewer.set_preview_render_triangle_limit(self.preview_limit.value())
        self.output_label.setText(f"Artifact: {artifact_path}")
        self._log(f"Opening mesh artifact {artifact_path}")
        self.viewer.load_artifact(artifact_path)

    def _on_preview_limit_changed(self, value: int) -> None:
        self.viewer.set_preview_render_triangle_limit(value)

    def viewer_filled_toggled(self, visible: bool) -> None:
        self.viewer.set_filled_visible(visible)

    def viewer_wireframe_toggled(self, visible: bool) -> None:
        self.viewer.set_wireframe_visible(visible)

    def auto_preview_limit(self) -> None:
        limit, source = _auto_preview_render_triangle_limit(
            render_device=self.viewer.render_device_info(),
            wireframe_enabled=self.wireframe_toggle.isChecked(),
        )
        self.preview_limit.setValue(limit)
        self._log(f"Auto max render triangles set to {limit:,} ({source})")
        if self._last_artifact_path is not None:
            self._load_artifact_preview(self._last_artifact_path)

    def load_scene_file(self, path: str | Path) -> None:
        scene_path = Path(path)
        domains = load_meshable_domains(scene_path)
        self._scene_path = scene_path
        self._domains = domains
        self.domain_list.clear()
        for domain in domains:
            self.domain_list.addItem(
                f"{domain.name}  kind={domain.kind}  dim={domain.dimension}  "
                f"tags={len(domain.boundary_tags)}"
            )
        self.run_button.setEnabled(bool(domains))
        self._log(f"Imported {scene_path}")
        self._log(f"Loaded {len(domains)} meshable domain(s)")

    def run_script(self) -> None:
        if not self._domains:
            self._log("No imported domains.")
            return
        if self._script_process is not None:
            self._log("A meshing script is already running.")
            return
        self.preview.setRowCount(0)
        output = Path(tempfile.gettempdir()) / "casocad_mesh_workspace.arrow"
        metadata = {
            "format": "casocad_mesh_artifact",
            "scene_path": str(self._scene_path) if self._scene_path is not None else None,
        }
        descriptor, job_name = tempfile.mkstemp(
            prefix="casocad-meshing-job-",
            suffix=".json",
        )
        os.close(descriptor)
        Path(job_name).write_text(
            json.dumps(
                {
                    "scene_path": str(self._scene_path),
                    "script_text": self.script.toPlainText(),
                    "output_path": str(output),
                    "metadata": metadata,
                    "preview_limit": 200,
                },
                separators=(",", ":"),
            ),
            encoding="utf-8",
        )
        self._script_job_path = Path(job_name)
        self._script_output_path = output
        self._script_stdout_buffer = ""
        self._script_failed = False
        process = QProcess(self)
        process.setProgram(sys.executable)
        process.setArguments(["-m", "app.meshing.worker", str(self._script_job_path)])
        process.readyReadStandardOutput.connect(self._on_script_stdout)
        process.readyReadStandardError.connect(self._on_script_stderr)
        process.finished.connect(self._on_script_finished)
        process.errorOccurred.connect(self._on_script_process_error)
        self._script_process = process
        self.run_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self._log("Starting meshing worker process.")
        process.start()

    def cancel_script(self) -> None:
        process = self._script_process
        if process is None:
            return
        self._log("Cancelling meshing worker process.")
        process.kill()

    @Slot()
    def _on_script_stdout(self) -> None:
        process = self._script_process
        if process is None:
            return
        data = bytes(process.readAllStandardOutput()).decode(
            "utf-8",
            errors="replace",
        )
        self._script_stdout_buffer += data
        while "\n" in self._script_stdout_buffer:
            line, self._script_stdout_buffer = self._script_stdout_buffer.split(
                "\n",
                1,
            )
            self._handle_script_message(line)

    @Slot()
    def _on_script_stderr(self) -> None:
        process = self._script_process
        if process is None:
            return
        data = bytes(process.readAllStandardError()).decode(
            "utf-8",
            errors="replace",
        )
        if data.strip():
            self._log(data.rstrip())

    def _handle_script_message(self, line: str) -> None:
        if not line.strip():
            return
        try:
            message = json.loads(line)
        except json.JSONDecodeError:
            self._log(line)
            return
        message_type = message.get("type")
        if message_type == "started":
            self._log("Meshing worker started.")
            return
        if message_type == "log":
            self._log(str(message.get("message", "")))
            return
        if message_type == "error":
            self._script_failed = True
            self._log(str(message.get("traceback") or message.get("message") or "error"))
            return
        if message_type != "done":
            self._log(line)
            return
        output = Path(str(message["output_path"]))
        element_count = int(message.get("element_count", 0))
        preview_rows = [
            (str(row[0]), int(row[1]), str(row[2]))
            for row in message.get("preview_rows", [])
            if len(row) == 3
        ]
        self._last_artifact_path = output
        self.output_label.setText(f"Artifact: {output}")
        self._populate_preview(preview_rows)
        self._log(f"Script completed: {element_count} mesh element(s) emitted")
        self._log(f"Wrote {output}")
        self._load_artifact_preview(output)
        self._show_viewer_page()

    @Slot(int, QProcess.ExitStatus)
    def _on_script_finished(
        self,
        exit_code: int,
        exit_status: QProcess.ExitStatus,
    ) -> None:
        if self._script_stdout_buffer:
            self._handle_script_message(self._script_stdout_buffer)
            self._script_stdout_buffer = ""
        if exit_status == QProcess.ExitStatus.CrashExit:
            self._script_failed = True
            self._log("Meshing worker crashed.")
        elif exit_code != 0 and not self._script_failed:
            self._script_failed = True
            self._log(f"Meshing worker exited with code {exit_code}.")
        self._clear_script_process()

    @Slot(QProcess.ProcessError)
    def _on_script_process_error(self, error: QProcess.ProcessError) -> None:
        self._script_failed = True
        self._log(f"Meshing worker process error: {error.name}")

    def _clear_script_process(self) -> None:
        process = self._script_process
        if process is not None:
            process.deleteLater()
        self._script_process = None
        if self._script_job_path is not None:
            try:
                self._script_job_path.unlink(missing_ok=True)
            except OSError:
                pass
        self._script_job_path = None
        self._script_output_path = None
        self.run_button.setEnabled(bool(self._domains))
        self.cancel_button.setEnabled(False)

    def _populate_preview(self, rows: list[tuple[str, int, str]]) -> None:
        self.preview.setRowCount(len(rows))
        for row_index, (element_type, vertex_count, tag_name) in enumerate(rows):
            self.preview.setItem(row_index, 0, QTableWidgetItem(element_type))
            self.preview.setItem(row_index, 1, QTableWidgetItem(str(vertex_count)))
            self.preview.setItem(row_index, 2, QTableWidgetItem(tag_name))
        self.preview.resizeColumnsToContents()

    def _log(self, message: str) -> None:
        self.log.append(str(message))

    def _on_viewer_summary(self, summary: MeshPreviewSummary) -> None:
        tag_count = len(summary.tag_names)
        truncated = " truncated" if summary.truncated else ""
        self.output_label.setText(
            f"Artifact: {summary.path} | mesh_elements={summary.element_count} | "
            f"preview_vertices={summary.preview_vertex_count} | "
            f"render_triangles={summary.preview_triangle_count} | "
            f"wire_edges={summary.preview_edge_count} | tags={tag_count}{truncated}"
        )


def _available_memory_bytes() -> int | None:
    try:
        import os

        page_size = os.sysconf("SC_PAGE_SIZE")
        available_pages = os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, OSError, ValueError):
        return None
    if not isinstance(page_size, int) or not isinstance(available_pages, int):
        return None
    return page_size * available_pages


def _auto_preview_render_triangle_limit(
    *,
    render_device: GpuRenderDeviceInfo | None,
    wireframe_enabled: bool,
) -> tuple[int, str]:
    available = _available_memory_bytes()
    gpu_info = query_gpu_memory_info(render_device=render_device)
    budget, source = choose_preview_budget_bytes(
        gpu_info=gpu_info,
        available_ram_bytes=available,
        wireframe_enabled=wireframe_enabled,
    )
    bytes_per_vertex = 3 * 4 * 2
    bytes_per_triangle = bytes_per_vertex * 3
    limit = max(1_000, min(50_000_000, budget // bytes_per_triangle))
    if available is None and source == "fallback":
        limit = _DEFAULT_PREVIEW_RENDER_TRIANGLE_LIMIT
    return limit, source


__all__ = ["MeshingWorkspace"]
