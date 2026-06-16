from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.signals import signals


class ExportPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        path_layout = QHBoxLayout()
        self.path = QLineEdit(str(Path.cwd() / "casocad_lattice.arrow"))
        browse = QPushButton("Browse")
        browse.clicked.connect(self._browse)
        path_layout.addWidget(self.path)
        path_layout.addWidget(browse)
        self.export_button = QPushButton("Mesh and Export .arrow")
        self.export_button.clicked.connect(
            lambda: signals.export_requested.emit(self.path.text())
        )
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        layout.addLayout(path_layout)
        layout.addWidget(self.export_button)
        layout.addWidget(self.progress)
        layout.addWidget(self.status)
        signals.mesh_progress.connect(self.progress.setValue)
        signals.mesh_ready.connect(self._on_mesh_ready)
        signals.log_message.connect(self._on_log_message)

    def set_busy(self, busy: bool) -> None:
        self.export_button.setEnabled(not busy)
        self.export_button.setText(
            "Meshing..." if busy else "Mesh and Export .arrow"
        )
        if busy:
            self.progress.setValue(0)
            self.status.setText("Preparing lattice...")

    def set_actions_enabled(self, enabled: bool) -> None:
        self.export_button.setEnabled(enabled)

    def _browse(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export lattice", self.path.text(), "Apache Arrow (*.arrow)"
        )
        if path:
            self.path.setText(
                path if path.lower().endswith(".arrow") else f"{path}.arrow"
            )

    def _on_mesh_ready(self, result: object) -> None:
        self.set_busy(False)
        self.progress.setValue(100)
        self.status.setText(
            f"{result.row_count:,} retained nodes, "
            f"{result.file_size / (1024 * 1024):.2f} MiB, {result.path}"
        )

    def _on_log_message(self, level: str, message: str) -> None:
        if level in {"error", "warning"}:
            self.set_busy(False)
        self.status.setText(message)
