"""
flow_sim_panel.py – Self-contained Sim panel window.

Responsibilities
----------------
* Let the user pick / receive an .arrow file path.
* Show simulation controls (particles, velocity, diffusion, streamlines).
* Load the .arrow file via flow_sim_arrow and create a FlowMapSimulation.
* Pass the simulation to FlowSimView; the view handles all rendering.
* Expose color-customisation for lattice layers and particles.

Does NOT touch the core CAD viewport or renderer.
"""
from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt, QThread, QTimer, Signal
from PySide6.QtGui import QColor, QHideEvent, QShowEvent
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from app.flow_sim_arrow import FlowLatticeData, load_arrow_lattice
from app.flow_sim_map import FlowMapSimulation, jax_backend_name
from app.flow_sim_view import (
    DEFAULT_BACKGROUND,
    DEFAULT_BOUNDARY_COLOR,
    DEFAULT_LATTICE_COLOR,
    DEFAULT_PARTICLE_COLOR,
    DEFAULT_STREAMLINE_COLOR,
    DEFAULT_UNTAGGED_BOUNDARY_COLOR,
    TAG_PALETTE,
    FlowSimView,
)


class _SimBuilder(QThread):
    """Builds a FlowMapSimulation off the main thread.

    ``FlowMapSimulation.from_lattice`` re-seeds particles and warms up the JAX
    backend, which is slow enough to freeze the UI when run inline. Running it
    here keeps the window responsive (and the previous simulation animating)
    until the new one is ready to swap in.
    """

    done = Signal(object, str)  # (simulation | None, error_message)

    def __init__(
        self,
        lattice: FlowLatticeData,
        kwargs: dict,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._lattice = lattice
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            sim = FlowMapSimulation.from_lattice(self._lattice, **self._kwargs)
        except Exception as e:  # noqa: BLE001 - reported to the UI verbatim
            self.done.emit(None, str(e))
            return
        self.done.emit(sim, "")


class FlowSimPanel(QWidget):
    """Independent floating window for the casoCAD flow simulation."""

    visibility_changed = Signal(bool)

    def __init__(
        self,
        initial_arrow_path: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("flowSimPanel")
        self.setWindowTitle("Sim")
        self.setWindowFlag(Qt.WindowType.Window, True)
        self.resize(1180, 760)
        self.setMinimumSize(920, 620)

        self._lattice: FlowLatticeData | None = None
        self._simulation: FlowMapSimulation | None = None
        self._running = False
        self._tag_checks: dict[int, QCheckBox] = {}
        self._tag_colors: dict[int, QColor] = {}
        self._tag_color_btns: dict[int, QPushButton] = {}

        # Per-node tag ids packed into a padded int array so lattice toggles can
        # be resolved with vectorised numpy instead of a per-node Python loop.
        self._tags_padded: np.ndarray = np.zeros((0, 1), dtype=np.int32)

        # Coalesce rapid spin-box edits (particle count / streamline length)
        # into a single simulation rebuild once the user settles.
        self._rebuild_timer = QTimer(self)
        self._rebuild_timer.setSingleShot(True)
        self._rebuild_timer.setInterval(220)
        self._rebuild_timer.timeout.connect(self._restart_simulation)
        self._load_pending = False

        # Off-thread simulation builder (so changing particle count doesn't
        # freeze the UI). At most one runs at a time; _build_again coalesces
        # further changes that arrive while a build is in flight.
        self._builder: _SimBuilder | None = None
        self._build_again = False

        # Color state
        self._bg_color = QColor(DEFAULT_BACKGROUND)
        self._lat_color = QColor(DEFAULT_LATTICE_COLOR)
        self._bnd_color = QColor(DEFAULT_BOUNDARY_COLOR)
        self._ubnd_color = QColor(DEFAULT_UNTAGGED_BOUNDARY_COLOR)
        self._par_color = QColor(DEFAULT_PARTICLE_COLOR)
        self._sl_color = QColor(DEFAULT_STREAMLINE_COLOR)

        # Build UI
        root = QHBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(12)

        self._view = FlowSimView()
        root.addWidget(self._view, stretch=1)

        # The whole control column lives inside a scroll area so it stays
        # usable no matter how many tagged objects the lattice contains.
        side = QWidget()
        side_lay = QVBoxLayout(side)
        side_lay.setContentsMargins(0, 0, 6, 0)
        side_lay.setSpacing(10)

        side_scroll = QScrollArea()
        side_scroll.setWidgetResizable(True)
        side_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        side_scroll.setFixedWidth(344)
        side_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        side_scroll.setWidget(side)
        root.addWidget(side_scroll)

        # --- File section ---
        self._path_edit = QLineEdit()
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse)
        load_btn = QPushButton("Load .arrow")
        load_btn.clicked.connect(self._load)
        fl = QFormLayout()
        fl.addRow("Arrow file", self._path_edit)
        btn_row = QVBoxLayout()
        btn_row.addWidget(browse_btn)
        btn_row.addWidget(load_btn)
        fl.addRow("", btn_row)
        side_lay.addLayout(fl)

        # --- Compute backend indicator (GPU/CUDA vs CPU) ---
        self._backend_label = QLabel()
        self._backend_label.setWordWrap(True)
        self._backend_label.setObjectName("simBackendLabel")
        side_lay.addWidget(self._backend_label)
        self._refresh_backend_label()

        # --- Simulation controls ---
        self._particle_count = QSpinBox()
        self._particle_count.setRange(200, 2_000_000)
        self._particle_count.setValue(1800)
        self._particle_count.setSingleStep(100)

        self._velocity = QDoubleSpinBox()
        self._velocity.setRange(0.1, 200.0)
        self._velocity.setDecimals(3)
        self._velocity.setSingleStep(0.5)
        self._velocity.setValue(5.0)

        self._diffusion = QDoubleSpinBox()
        self._diffusion.setRange(0.0, 2.0)
        self._diffusion.setDecimals(4)
        self._diffusion.setSingleStep(0.005)
        self._diffusion.setValue(0.08)

        self._viscosity = QDoubleSpinBox()
        self._viscosity.setRange(0.0, 1.0)
        self._viscosity.setDecimals(3)
        self._viscosity.setSingleStep(0.05)
        self._viscosity.setValue(0.35)

        self._particle_size = QSpinBox()
        self._particle_size.setRange(1, 8)
        self._particle_size.setValue(2)

        self._sl_check = QCheckBox("Streamlines")
        self._sl_check.setChecked(False)
        self._sl_len_label = QLabel("20")
        self._sl_len = QSlider(Qt.Orientation.Horizontal)
        self._sl_len.setRange(4, 80)
        self._sl_len.setValue(20)
        self._sl_len.setTickPosition(QSlider.TickPosition.TicksBelow)
        sl_sub = QVBoxLayout()
        sl_sub.addWidget(self._sl_len)
        sl_sub.addWidget(self._sl_len_label)

        ctrl = QFormLayout()
        ctrl.addRow("Particles", self._particle_count)
        ctrl.addRow("Particle size", self._particle_size)
        ctrl.addRow("Velocity", self._velocity)
        ctrl.addRow("Diffusion", self._diffusion)
        ctrl.addRow("Viscosity", self._viscosity)
        ctrl.addRow("Show streamlines", self._sl_check)
        ctrl.addRow("Streamline length", sl_sub)
        ctrl_box = QGroupBox("Simulation")
        ctrl_box.setLayout(ctrl)
        side_lay.addWidget(ctrl_box)

        # --- Color controls ---
        color_box = QGroupBox("Colors")
        color_lay = QVBoxLayout(color_box)
        color_lay.setContentsMargins(8, 8, 8, 8)
        color_lay.setSpacing(4)
        self._bg_btn = self._color_row(color_lay, "Background", self._bg_color, self._pick_bg)
        self._par_btn = self._color_row(color_lay, "Particles", self._par_color, self._pick_par)
        self._sl_btn = self._color_row(color_lay, "Streamlines", self._sl_color, self._pick_sl)
        side_lay.addWidget(color_box)

        # --- Lattice layer visibility ---
        vis_box = QGroupBox("Visible Lattice")
        vis_lay = QVBoxLayout(vis_box)
        vis_lay.setContentsMargins(8, 8, 8, 8)
        vis_lay.setSpacing(6)

        self._show_lat = QCheckBox("Retained lattice")
        self._show_lat.setChecked(True)
        self._show_bnd = QCheckBox("Boundary")
        self._show_bnd.setChecked(True)
        self._show_ubnd = QCheckBox("Untagged boundary")
        self._show_ubnd.setChecked(True)

        self._lat_btn = self._vis_row(vis_lay, self._show_lat, self._lat_color, self._pick_lat)
        self._bnd_btn = self._vis_row(vis_lay, self._show_bnd, self._bnd_color, self._pick_bnd)
        self._ubnd_btn = self._vis_row(vis_lay, self._show_ubnd, self._ubnd_color, self._pick_ubnd)

        for cb in (self._show_lat, self._show_bnd, self._show_ubnd):
            cb.toggled.connect(lambda _: self._push_lattice_filters())

        self._tag_label = QLabel("Tagged mesh points")
        vis_lay.addWidget(self._tag_label)
        self._tag_host = QWidget()
        self._tag_lay = QVBoxLayout(self._tag_host)
        self._tag_lay.setContentsMargins(0, 0, 0, 0)
        self._tag_lay.setSpacing(4)
        self._tag_lay.addStretch(1)
        # Tag rows grow with the number of tagged objects; the side scroll
        # area scrolls them into view rather than squeezing each row.
        vis_lay.addWidget(self._tag_host)
        side_lay.addWidget(vis_box)

        # --- Status line ---
        self._status = QLabel("Load an .arrow file to start.")
        self._status.setWordWrap(True)
        self._debug_frames_btn = QPushButton("Debug frame compare")
        self._debug_frames_btn.setToolTip("Check whether recent frames are static")
        self._debug_frames_btn.setEnabled(False)
        self._debug_frames_btn.clicked.connect(self._debug_frame_compare)
        side_lay.addWidget(self._status)
        side_lay.addWidget(self._debug_frames_btn)
        side_lay.addStretch(1)

        # Wire signals.
        # Particle count and streamline length change array sizes, so they need
        # a (debounced) rebuild. Particle size is a pure render uniform. Velocity
        # / diffusion / viscosity are applied live on the running simulation.
        self._particle_count.valueChanged.connect(self._schedule_rebuild)
        self._particle_size.valueChanged.connect(self._view.set_particle_size)
        self._velocity.valueChanged.connect(self._apply_live_params)
        self._diffusion.valueChanged.connect(self._apply_live_params)
        self._viscosity.valueChanged.connect(self._apply_live_params)
        self._sl_check.toggled.connect(self._on_streamlines_toggled)
        self._sl_len.valueChanged.connect(self._on_sl_len_changed)

        if initial_arrow_path:
            self._path_edit.setText(initial_arrow_path)
            self._load()

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def show_panel(self, owner: QWidget | None = None) -> None:
        if owner is not None and not self.isVisible():
            geo = self.frameGeometry()
            geo.moveCenter(owner.frameGeometry().center())
            self.move(geo.topLeft())
        self.show()
        self.raise_()
        self.activateWindow()

    def set_arrow_path(self, path: str | None) -> None:
        if not path:
            return
        p = Path(path)
        if self._lattice is not None and self._lattice.path == p:
            self._path_edit.setText(str(p))
            return
        self._path_edit.setText(str(p))
        # Defer the heavy load (arrow parse + simulation build) to the next
        # event-loop tick so the window can paint immediately on open.
        self._schedule_load()

    def _schedule_load(self) -> None:
        if self._load_pending:
            return
        self._load_pending = True
        self._status.setText("Loading…")
        QTimer.singleShot(0, self._load)

    def stop_simulation(self) -> None:
        self._rebuild_timer.stop()
        self._build_again = False
        if self._builder is not None:
            self._builder.wait(5000)
            self._builder = None
        self._running = False
        self._simulation = None
        self._view.set_simulation(None, self._sl_check.isChecked())
        self._status.setText("Simulation stopped.")
        self._refresh_debug_button_state()

    # -----------------------------------------------------------------------
    # File loading
    # -----------------------------------------------------------------------

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open lattice .arrow", "", "Apache Arrow (*.arrow)"
        )
        if path:
            self._path_edit.setText(path)
            self._load()

    def _load(self) -> None:
        self._load_pending = False
        text = self._path_edit.text().strip()
        if not text:
            self._status.setText("No .arrow path provided.")
            return
        path = Path(text)
        if path.suffix.lower() != ".arrow":
            path = path.with_suffix(".arrow")
            self._path_edit.setText(str(path))
        try:
            lattice = load_arrow_lattice(path)
        except (OSError, ValueError) as e:
            self._status.setText(f"Failed to load {path.name}: {e}")
            return
        self._lattice = lattice
        self._precompute_tag_arrays()
        self._rebuild_tag_ui()
        self._push_lattice_filters()
        self._view.frame_points(lattice.positions)
        self._status.setText(
            f"Loaded {lattice.positions.shape[0]:,} nodes from {path.name}"
            f"  ({int(lattice.untagged_boundary_mask.sum()):,} untagged boundary)"
        )
        self._restart_simulation()

    # -----------------------------------------------------------------------
    # Simulation lifecycle
    # -----------------------------------------------------------------------

    def _schedule_rebuild(self) -> None:
        """Debounce a full simulation rebuild for size-changing settings."""
        if self._lattice is None:
            return
        self._rebuild_timer.start()

    def _apply_live_params(self) -> None:
        """Push velocity / diffusion / viscosity to the running sim, no rebuild."""
        if not self._running:
            return
        self._view.set_sim_params(
            velocity=self._velocity.value(),
            diffusion=self._diffusion.value(),
            viscosity=self._viscosity.value(),
        )

    def _on_streamlines_toggled(self, checked: bool) -> None:
        self._view.set_streamlines_visible(checked)
        if self._running and self._simulation is not None:
            self._view.set_simulation(self._simulation, checked)

    def _on_sl_len_changed(self, value: int) -> None:
        self._sl_len_label.setText(str(value))
        self._schedule_rebuild()

    def _debug_frame_compare(self) -> None:
        if self._simulation is None:
            self._status.setText("No simulation running; load and start first.")
            return

        report = self._view.frame_stagnation_report()
        stuck = "yes" if report["stuck"] else "no"
        self._status.setText(
            (
                "Frame debug | window "
                f"{report['window_frames']} | static {report['static_ratio']:.1%} | "
                f"longest streak {report['longest_static_streak']} | moved "
                f"{report['moved_ratio_latest']:.1%} | stuck {stuck}"
            )
        )

    def _refresh_backend_label(self, backend_name: str | None = None) -> None:
        """Show whether the sim computes on the GPU (CUDA), CPU JAX, or NumPy.

        Before a simulation exists we report JAX's default backend; once one is
        running we use its authoritative ``backend_name``. JAX selects CUDA
        automatically when a CUDA-enabled jaxlib is installed, so this is purely
        a status read-out (plus an install hint when the GPU is idle).
        """
        if backend_name is None:
            backend_name = jax_backend_name()
        label, color, hint = self._backend_descriptor(backend_name)
        text = f"Compute: {label}"
        if hint:
            text += f"  ·  {hint}"
        self._backend_label.setText(text)
        self._backend_label.setStyleSheet(f"color:{color};")

    @staticmethod
    def _backend_descriptor(name: str | None) -> tuple[str, str, str]:
        low = (name or "").lower()
        if "gpu" in low or "cuda" in low:
            return ("GPU (CUDA)", "#34d399", "")
        if name is None or "numpy" in low:
            return ("CPU (NumPy)", "#fbbf24", "install jax[cuda12] for GPU")
        # JAX present but running on CPU.
        return ("CPU (JAX)", "#fbbf24", "GPU idle — install jax[cuda12]")

    def _refresh_debug_button_state(self) -> None:
        self._debug_frames_btn.setEnabled(self._running and self._simulation is not None)

    def _restart_simulation(self) -> None:
        self._rebuild_timer.stop()
        if self._lattice is None:
            return
        # If a build is already running, let it finish, then rebuild once more
        # with the latest values — never stack builders or block the UI.
        if self._builder is not None and self._builder.isRunning():
            self._build_again = True
            return
        self._build_again = False
        kwargs = {
            "particle_count": self._particle_count.value(),
            "velocity": self._velocity.value(),
            "diffusion": self._diffusion.value(),
            "viscosity": self._viscosity.value(),
            "streamline_length": self._sl_len.value(),
        }
        self._status.setText("Rebuilding simulation…")
        builder = _SimBuilder(self._lattice, kwargs, self)
        builder.done.connect(self._on_sim_built)
        self._builder = builder
        builder.start()

    def _on_sim_built(self, sim: object, error: str) -> None:
        # Ignore results from a builder that was superseded or stopped (its
        # done signal may already have been queued before we dropped it).
        if self.sender() is not self._builder:
            return
        self._builder.deleteLater()
        self._builder = None

        # Settings changed again mid-build: discard this (stale) sim and rebuild.
        if self._build_again:
            self._restart_simulation()
            return

        if sim is None:
            self._simulation = None
            self._running = False
            self._view.set_simulation(None, self._sl_check.isChecked())
            self._status.setText(f"Simulation error: {error}")
            self._refresh_debug_button_state()
            return

        self._simulation = sim
        self._running = True
        self._view.set_simulation(sim, self._sl_check.isChecked())
        self._refresh_debug_button_state()
        self._refresh_backend_label(sim.backend_name)
        self._status.setText(
            "Simulating"
            + (" with streamlines" if self._sl_check.isChecked() else "")
            + f"  ({sim.backend_name})"
        )

    # -----------------------------------------------------------------------
    # Lattice layer filters → view
    # -----------------------------------------------------------------------

    def _rebuild_tag_ui(self) -> None:
        while self._tag_lay.count() > 1:
            item = self._tag_lay.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()
        self._tag_checks.clear()
        self._tag_color_btns.clear()
        if self._lattice is None:
            self._tag_label.setText("Tagged mesh points")
            return
        tag_ids = sorted({
            int(t) for tags in self._lattice.tag_ids for t in tags if int(t) > 0
        })
        self._tag_label.setText(
            f"Tagged mesh points ({len(tag_ids)})" if tag_ids
            else "Tagged mesh points (none)"
        )
        for tid in tag_ids:
            label = self._lattice.object_names.get(tid, f"Tag {tid}")
            cb = QCheckBox(label)
            cb.setChecked(True)
            cb.toggled.connect(lambda _: self._push_lattice_filters())
            btn = self._make_color_btn(
                label, self._tag_color(tid),
                lambda _, v=tid: self._pick_tag_color(v),
            )
            row = QWidget()
            rl = QHBoxLayout(row)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.addWidget(cb, stretch=1)
            rl.addWidget(btn)
            self._tag_lay.insertWidget(self._tag_lay.count() - 1, row)
            self._tag_checks[tid] = cb
            self._tag_color_btns[tid] = btn

    def _precompute_tag_arrays(self) -> None:
        """Pack per-node tag ids into a padded int matrix once, on load.

        ``tag_ids`` is a tuple of variable-length tuples; resolving it per
        toggle in pure Python is what made lattice toggles lag. With the padded
        matrix the toggle handler becomes a couple of vectorised numpy ops.
        """
        lat = self._lattice
        if lat is None:
            self._tags_padded = np.zeros((0, 1), dtype=np.int32)
            return
        tag_ids = lat.tag_ids
        n = len(tag_ids)
        max_k = max((len(tags) for tags in tag_ids), default=0)
        padded = np.zeros((n, max(max_k, 1)), dtype=np.int32)
        for i, tags in enumerate(tag_ids):
            if tags:
                padded[i, : len(tags)] = tags
        self._tags_padded = padded

    def _push_lattice_filters(self) -> None:
        lat = self._lattice
        if lat is None:
            empty = np.empty((0, 3), dtype=np.float64)
            self._view.set_lattice_points(empty, empty, empty, empty, np.empty(0, np.uint16))
            return

        active = [tid for tid, cb in self._tag_checks.items() if cb.isChecked()]
        padded = self._tags_padded
        n = padded.shape[0]

        if active and padded.size:
            max_tag = int(padded.max())
            lut = np.zeros(max_tag + 1, dtype=np.bool_)
            for tid in active:
                if 0 < tid <= max_tag:
                    lut[tid] = True
            slot_active = lut[padded]              # (N, K) bool
            tagged_mask = slot_active.any(axis=1)  # any active tag → node shown
            # Primary tag = first active tag in the node's stored order.
            first = slot_active.argmax(axis=1)
            primary_ids = padded[np.arange(n), first].astype(np.uint16)
            primary_ids[~tagged_mask] = 0
        else:
            tagged_mask = np.zeros(n, dtype=np.bool_)
            primary_ids = np.zeros(n, dtype=np.uint16)

        self._view.set_lattice_points(
            lat.positions[~lat.boundary_mask] if self._show_lat.isChecked()
            else np.empty((0, 3), np.float64),
            lat.positions[lat.boundary_mask] if self._show_bnd.isChecked()
            else np.empty((0, 3), np.float64),
            lat.positions[lat.untagged_boundary_mask] if self._show_ubnd.isChecked()
            else np.empty((0, 3), np.float64),
            lat.positions[tagged_mask],
            primary_ids[tagged_mask],
        )
        self._view.set_tag_colors(self._tag_colors)

    # -----------------------------------------------------------------------
    # Color pickers
    # -----------------------------------------------------------------------

    def _pick_bg(self) -> None:
        c = self._open_color(self._bg_color)
        if c.isValid():
            self._bg_color = c
            self._apply_btn_color(self._bg_btn, c)
            self._view.set_background_color(c)

    def _pick_par(self) -> None:
        c = self._open_color(self._par_color)
        if c.isValid():
            self._par_color = c
            self._apply_btn_color(self._par_btn, c)
            self._view.set_particle_color(c)

    def _pick_sl(self) -> None:
        c = self._open_color(self._sl_color)
        if c.isValid():
            self._sl_color = c
            self._apply_btn_color(self._sl_btn, c)
            self._view.set_streamline_color(c)

    def _pick_lat(self) -> None:
        c = self._open_color(self._lat_color)
        if c.isValid():
            self._lat_color = c
            self._apply_btn_color(self._lat_btn, c)
            self._view.set_lattice_color(c)

    def _pick_bnd(self) -> None:
        c = self._open_color(self._bnd_color)
        if c.isValid():
            self._bnd_color = c
            self._apply_btn_color(self._bnd_btn, c)
            self._view.set_boundary_color(c)

    def _pick_ubnd(self) -> None:
        c = self._open_color(self._ubnd_color)
        if c.isValid():
            self._ubnd_color = c
            self._apply_btn_color(self._ubnd_btn, c)
            self._view.set_untagged_boundary_color(c)

    def _pick_tag_color(self, tid: int) -> None:
        c = self._open_color(self._tag_color(tid))
        if not c.isValid():
            return
        self._tag_colors[tid] = c
        btn = self._tag_color_btns.get(tid)
        if btn:
            self._apply_btn_color(btn, c)
        self._view.set_tag_colors(self._tag_colors)

    def _open_color(self, current: QColor) -> QColor:
        was_active = self._running
        if was_active:
            self._view.set_active(False)
        try:
            return QColorDialog.getColor(current, self)
        finally:
            if was_active:
                self._view.set_active(True)

    def _tag_color(self, tid: int) -> QColor:
        return self._tag_colors.get(tid, TAG_PALETTE[tid % len(TAG_PALETTE)])

    # -----------------------------------------------------------------------
    # UI builder helpers
    # -----------------------------------------------------------------------

    def _make_color_btn(self, label: str, color: QColor, slot: object) -> QPushButton:
        btn = QPushButton("Color")
        btn.setFixedWidth(62)
        btn.setToolTip(f"Set {label} color")
        btn.clicked.connect(lambda _=False, s=slot: s())
        self._apply_btn_color(btn, color)
        return btn

    def _apply_btn_color(self, btn: QPushButton, color: QColor) -> None:
        fg = "#000000" if color.lightness() > 150 else "#ffffff"
        btn.setStyleSheet(
            f"background-color:{color.name()};color:{fg};border:1px solid #666;"
        )

    def _color_row(
        self,
        parent: QVBoxLayout,
        label: str,
        color: QColor,
        slot: object,
    ) -> QPushButton:
        btn = self._make_color_btn(label, color, slot)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(QLabel(label), stretch=1)
        rl.addWidget(btn)
        parent.addWidget(row)
        return btn

    def _vis_row(
        self,
        parent: QVBoxLayout,
        checkbox: QCheckBox,
        color: QColor,
        slot: object,
    ) -> QPushButton:
        btn = self._make_color_btn(checkbox.text(), color, slot)
        row = QWidget()
        rl = QHBoxLayout(row)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(checkbox, stretch=1)
        rl.addWidget(btn)
        parent.addWidget(row)
        return btn

    # -----------------------------------------------------------------------
    # Qt events
    # -----------------------------------------------------------------------

    def showEvent(self, event: QShowEvent) -> None:
        self._view.set_active(True)
        self.visibility_changed.emit(True)
        super().showEvent(event)

    def hideEvent(self, event: QHideEvent) -> None:
        self._view.set_active(False)
        self.visibility_changed.emit(False)
        super().hideEvent(event)

    def closeEvent(self, event: object) -> None:
        self.stop_simulation()
        self.visibility_changed.emit(False)
        super().closeEvent(event)
