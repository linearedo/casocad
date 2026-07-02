"""Application-wide signal bus.

One global ``AppSignals`` instance (``signals``) decouples the panels and the
viewport from ``MainWindow``, which owns the document and handles most
requests. Conventions:

- ``*_requested`` — user intent emitted by UI (panels, viewport tools);
  handled by ``MainWindow``, which mutates the document and republishes.
- ``*_preview_*`` — non-committing live previews (ghosts); no undo snapshot.
- past tense (``document_changed``, ``node_edited``, ...) — state-change
  notifications fanned out to whoever displays that state.

When adding a signal, put it in its section below and keep the payload types
in the ``Signal(...)`` declaration meaningful (``object`` = Python object
passed through unchanged).
"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal


class AppSignals(QObject):
    # -- document / scene lifecycle -------------------------------------------
    document_changed = Signal(object)
    scene_changed = Signal(object)
    undo_snapshot_ready = Signal(object)
    node_edited = Signal()

    # -- selection --------------------------------------------------------------
    node_selected = Signal(object)
    selection_changed = Signal(object)
    viewport_scene_object_selected = Signal(int)

    # -- SDF creation (Add button + viewport draw tool) --------------------------
    add_primitive_requested = Signal(str)
    viewport_create_requested = Signal(str)
    viewport_shape_drawn = Signal(str, object, object, object)
    viewport_shape_preview_requested = Signal(str, object, object, object)
    viewport_point_shape_preview_requested = Signal(str, object, str)
    viewport_point_shape_drawn = Signal(str, object, str)

    # -- viewport transform tools (move / rotate / extrude / revolve) ------------
    viewport_move_tool_requested = Signal()
    viewport_move_requested = Signal(int, object)
    viewport_move_preview_requested = Signal(int, object)
    viewport_rotate_tool_requested = Signal()
    viewport_rotate_requested = Signal(int, str, float, object)
    viewport_rotate_preview_requested = Signal(int, str, float, object)
    viewport_transform_requested = Signal(int, object, object)
    viewport_extrude_requested = Signal(int, float)
    viewport_extrude_preview_requested = Signal(int, float)
    viewport_revolve_requested = Signal(int, str, object, object, object, float)
    viewport_revolve_preview_requested = Signal(
        int, str, object, object, object, float
    )
    viewport_frame_requested = Signal()

    # -- SDF operators / modeling edits ------------------------------------------
    sdf_op_requested = Signal(str, object)
    sdf_op_preview_requested = Signal(str, object)
    transform_requested = Signal(str, object)
    solid_from_2d_requested = Signal(str, object)
    delete_nodes_requested = Signal(object)

    # -- boundary regions ---------------------------------------------------------
    viewport_boundary_tool_requested = Signal()
    boundary_cutter_armed = Signal()
    viewport_boundary_hovered = Signal(object)
    viewport_boundary_region_requested = Signal(object)
    create_boundary_region_requested = Signal(object)
    create_polygon_from_polyline_requested = Signal(object)

    # -- domain roles (exact-SDF grammar) ------------------------------------------
    set_domain_requested = Signal(object, str)
    unset_domain_requested = Signal(object)
    set_fluid_root_requested = Signal(object)
    set_tag_enabled_requested = Signal(object, bool)

    # -- app-level settings / diagnostics -------------------------------------------
    working_unit_changed = Signal(object)
    log_message = Signal(str, str)


signals = AppSignals()
