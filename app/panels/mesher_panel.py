from __future__ import annotations

from math import prod

from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QWidget,
)

from app.signals import signals
from core.mesher import MesherConfig, recommended_max_dx
from core.mesher.grid import derive_lattice_grid
from core.scene import SceneDocument


class MesherPanel(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._document: SceneDocument | None = None
        self._busy = False
        layout = QFormLayout(self)
        self.dx = self._length_spin(0.08, 0.00001)
        self.boundary_error_tolerance = self._length_spin(0.02, 0.000001)
        self.boundary_error_tolerance.setToolTip(
            "Maximum permitted distance from a retained boundary node to its "
            "refined SDF crossing along an exposed lattice edge. casoCAD "
            "halves dx until this target is reached."
        )
        self.internal_preview_density = QDoubleSpinBox()
        self.internal_preview_density.setDecimals(6)
        self.internal_preview_density.setRange(0.0, 1.0)
        self.internal_preview_density.setSingleStep(0.001)
        self.internal_preview_density.setValue(0.1)
        self.internal_preview_density.setToolTip(
            "Fraction of internal nodes shown in the viewport. "
            "Boundary nodes are always shown."
        )
        self.chunk_size = QSpinBox()
        self.chunk_size.setRange(10_000, 10_000_000)
        self.chunk_size.setSingleStep(100_000)
        self.chunk_size.setValue(1_000_000)
        self.estimate = QLabel("No geometry")
        self.estimate.setWordWrap(True)
        self.recommended_dx = QLabel("No geometry")
        self.recommended_dx.setWordWrap(True)
        self.domain_summary = QLabel("No Fluid Domain selected")
        self.domain_summary.setWordWrap(True)
        self.mesh_button = QPushButton("Mesh Preview")
        self.mesh_button.clicked.connect(signals.mesh_requested.emit)
        self.status = QLabel("Ready")
        self.status.setWordWrap(True)
        self.legend = QLabel(
            "Stable object colours: SDF volumes, boundaries, and tags\n"
            "Squares connect four boundary vertices; all four edges share one face colour."
        )
        self.legend.setWordWrap(True)
        layout.addRow("Fluid Domain", self.domain_summary)
        layout.addRow("Initial maximum dx", self.dx)
        layout.addRow("Target maximum boundary error", self.boundary_error_tolerance)
        layout.addRow("Geometry-based dx suggestion", self.recommended_dx)
        layout.addRow(
            "Internal preview density", self.internal_preview_density
        )
        rule = QLabel(
            "Boundary: refined SDF zero crossings on retained-to-OUTSIDE edges.\n"
            "Tags: placed 1D intervals on 2D boundaries; placed 2D profiles "
            "or owner regions on 3D boundaries."
        )
        rule.setWordWrap(True)
        layout.addRow("Classification", rule)
        layout.addRow("Chunk nodes", self.chunk_size)
        layout.addRow("Envelope estimate", self.estimate)
        layout.addRow(self.mesh_button)
        layout.addRow("Preview", self.status)
        layout.addRow("Legend", self.legend)
        signals.document_changed.connect(self.set_document)
        signals.mesh_progress.connect(self._on_mesh_progress)
        signals.preview_ready.connect(self._on_preview_ready)
        for control in (self.dx, self.boundary_error_tolerance):
            control.valueChanged.connect(self._update_estimate)

    def _length_spin(self, value: float, minimum: float) -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setDecimals(6)
        spin.setRange(minimum, 1000.0)
        spin.setSingleStep(0.001)
        spin.setSuffix(" m")
        spin.setValue(value)
        return spin

    def set_document(self, document: SceneDocument) -> None:
        self._document = document
        if document.fluid_domain is None:
            self.domain_summary.setText("No Fluid Domain selected")
            self.recommended_dx.setText("Select a 2D or 3D Fluid Domain root")
        else:
            tags = ", ".join(tag.name for tag in document.fluid_domain.tag_objects)
            self.domain_summary.setText(
                f"{document.fluid_domain.root.name}"
                + (f"\nTags: {tags}" if tags else "\nTags: none")
            )
            recommended = recommended_max_dx(document.fluid_domain.root)
            self.recommended_dx.setText(
                f"{recommended:.6g} m based on the smallest modeled feature. "
                "The error target remains authoritative."
            )
            self.dx.setToolTip(
                "Starting uniform lattice spacing. casoCAD automatically "
                "reduces it when required by the boundary-error target."
            )
        self._update_estimate()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.mesh_button.setEnabled(not busy)
        self.mesh_button.setText("Meshing..." if busy else "Mesh Preview")
        if busy:
            self.status.setText("Preparing lattice preview...")

    def _on_mesh_progress(self, value: int) -> None:
        if self._busy:
            self.status.setText(f"Meshing: {value}%")

    def _on_preview_ready(self, result: object) -> None:
        self.set_busy(False)
        boundary_count = int((result.preview_node_types == 1).sum())
        internal_count = int((result.preview_node_types == 0).sum())
        tagged_count = sum(bool(items) for items in result.preview_tag_ids)
        crossing_count = result.boundary_sample_positions.shape[0]
        tolerance_status = (
            "met" if result.boundary_error_tolerance_met else "not met"
        )
        self.status.setText(
            f"Final dx: {result.preview_cell_size:.6g} m "
            f"after {result.refinement_count} refinement(s); target {tolerance_status}\n"
            f"Boundary error: maximum {result.boundary_error_maximum:.6g} m; "
            f"mean {result.boundary_error_mean:.6g} m; "
            f"RMS {result.boundary_error_rms:.6g} m; "
            f"95th percentile {result.boundary_error_percentile_95:.6g} m\n"
            f"{result.row_count:,} retained nodes; preview shows "
            f"{boundary_count:,} boundary, {internal_count:,} internal, "
            f"and {tagged_count:,} tagged nodes; "
            f"{crossing_count:,} refined boundary crossings"
        )

    def config(self) -> MesherConfig:
        return MesherConfig(
            dx=self.dx.value(),
            boundary_error_tolerance=self.boundary_error_tolerance.value(),
            n_levels=0,
            chunk_size=self.chunk_size.value(),
            unit_label="m",
            internal_preview_density=self.internal_preview_density.value(),
        )

    def _update_estimate(self) -> None:
        if self._document is None or self._document.fluid_domain is None:
            self.estimate.setText("Select a 2D or 3D Fluid Domain root")
            return
        try:
            grid = derive_lattice_grid(
                self._document.fluid_domain.root, self.dx.value()
            )
        except ValueError as error:
            self.estimate.setText(str(error))
            return
        shape = (
            f"{grid.nx} x {grid.ny}"
            if grid.dimension == 2
            else f"{grid.nx} x {grid.ny} x {grid.nz}"
        )
        self.estimate.setText(
            f"{shape} = {prod((grid.nx, grid.ny, grid.nz)):,} candidate nodes"
        )
