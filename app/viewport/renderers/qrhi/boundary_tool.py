"""Hover/select boundary tool for the QRhi viewport (boundary_region_v2 §7).

Armed via the BoundaryRegion button: moving the cursor ray-picks the Domain
boundary on the CPU (core.boundary_patches against the live fluid root — no
GPU round-trip) and highlights the owner surface under the cursor, with the
patch identity in the status bar. A left click requests a BoundaryRegion for
the hovered patch; Esc cancels. Pick tolerances scale with the Domain extent
so mm and km scenes behave alike.
"""
from __future__ import annotations

import numpy as np
from PySide6.QtCore import Qt

from app.signals import signals
from core.boundary_patches import BoundaryPatchHit, pick_boundary_patch

_RELATIVE_HIT_TOLERANCE = 4.0e-4
_RELATIVE_MAX_TRAVEL = 20.0


def _same_hit(a: BoundaryPatchHit | None, b: BoundaryPatchHit | None) -> bool:
    if a is None or b is None:
        return a is b
    a_selector = (a.selector.selector_id, a.selector.side) if a.selector else None
    b_selector = (b.selector.selector_id, b.selector.side) if b.selector else None
    return (
        a.owner_object_id == b.owner_object_id
        and a.patch_id == b.patch_id
        and a_selector == b_selector
    )


class BoundaryTool:
    def __init__(self, viewport) -> None:
        self._viewport = viewport
        self.root = None
        self.selectors: tuple = ()
        self.hover: BoundaryPatchHit | None = None

    @property
    def active(self) -> bool:
        return self.root is not None

    def _scene_extent(self, root=None) -> float:
        root = self.root if root is None else root
        try:
            box = root.bounding_box()
        except (ValueError, NotImplementedError):
            return 1.0
        return max(
            box.x_max - box.x_min,
            box.y_max - box.y_min,
            box.z_max - box.z_min,
            1.0e-9,
        )

    def begin(self, root, selectors=()) -> None:
        viewport = self._viewport
        viewport.cancel_create_tool()
        viewport.end_move_tool()
        viewport.end_rotate_tool()
        viewport._end_extrude_tool()
        viewport._end_revolve_tool()
        self.root = root
        self.selectors = tuple(selectors)
        self.hover = None
        viewport.setCursor(Qt.CursorShape.PointingHandCursor)
        viewport.setFocus()
        signals.log_message.emit(
            "info",
            "Hover a FluidDomain boundary to preview its patch; click to tag "
            "it as a BoundaryRegion. Esc cancels.",
        )

    def cancel(self) -> None:
        if not self.active:
            return
        self.root = None
        self.selectors = ()
        self.hover = None
        self._viewport.set_boundary_hover(0, None, None, None)
        self._viewport.show_boundary_patch_highlight(None)
        self._viewport.unsetCursor()
        self._viewport._dirty = True

    def update_hover(self, pos) -> None:
        if not self.active:
            return
        hit = self.pick(pos)
        if not _same_hit(hit, self.hover):
            self.hover = hit
            signals.viewport_boundary_hovered.emit(hit)

    def pick(self, pos, *, root=None, selectors=()) -> BoundaryPatchHit | None:
        root = self.root if root is None else root
        if root is None:
            return None
        origin, direction = self._viewport._screen_ray(pos)
        extent = self._scene_extent(root)
        return pick_boundary_patch(
            root,
            np.asarray(origin, dtype=np.float64),
            np.asarray(direction, dtype=np.float64),
            selector_objects=self.selectors if root is self.root else tuple(selectors),
            hit_tolerance=_RELATIVE_HIT_TOLERANCE * extent,
            maximum_travel=_RELATIVE_MAX_TRAVEL * extent,
        )

    def commit(self) -> bool:
        """Left click while armed: tag the hovered patch. Returns True when
        the click was consumed by this tool."""
        if not self.active:
            return False
        if self.hover is None:
            signals.log_message.emit(
                "warning", "No FluidDomain boundary under the cursor."
            )
            return True
        hit = self.hover
        self.cancel()
        signals.viewport_boundary_region_requested.emit(hit)
        return True
