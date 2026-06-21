from __future__ import annotations

"""World-grid cull buffers (design §13.5) — isolated optimization.

Owns the six SSBOs (bindings 8-13) for the per-cell additive/subtractive leaf
lists that the FEATURE_CULL shader path (sdf_cull.glsl) marches with DDA. Built
on topology change from core.gpu_cull (flatten + grid). If a scene isn't
cullable (non-flattening operators, or an unbounded additive leaf) ``build``
returns False and the renderer keeps the exact full VM — delete this file +
the FEATURE_CULL flag and the renderer falls back to the plain interpreter.
"""

import moderngl
import numpy as np

from core.gpu_cull import build_grid, flatten_scene, leaf_bounds
from core.render_ir import RenderIR

_FIRST_BINDING = 8  # add_off, add_cnt, add_item, sub_off, sub_cnt, sub_item


class SceneCull:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._buffers: list[moderngl.Buffer] = []
        self.active = False
        self.origin = (0.0, 0.0, 0.0)
        self.cell = (1.0, 1.0, 1.0)
        self.dim = 1
        self._make_empty()

    def _make_empty(self) -> None:
        self._release()
        zero = np.zeros(1, dtype=np.uint32).tobytes()
        self._buffers = [self.ctx.buffer(zero) for _ in range(6)]

    def build(self, render_ir: RenderIR | None) -> bool:
        plan = flatten_scene(render_ir)
        if plan is None:
            self.active = False
            self._make_empty()
            return False
        grid = build_grid(plan, leaf_bounds(render_ir), dim=16)
        if grid is None:
            self.active = False
            self._make_empty()
            return False

        self._release()

        def buf(arr: np.ndarray) -> moderngl.Buffer:
            return self.ctx.buffer(np.ascontiguousarray(arr.astype(np.uint32)).tobytes())

        self._buffers = [
            buf(grid.add_offsets), buf(grid.add_counts), buf(grid.add_items),
            buf(grid.sub_offsets), buf(grid.sub_counts), buf(grid.sub_items),
        ]
        self.origin = grid.origin
        self.cell = grid.cell
        self.dim = grid.dim
        self.active = True
        return True

    def bind(self) -> None:
        for i, b in enumerate(self._buffers):
            b.bind_to_storage_buffer(_FIRST_BINDING + i)

    def set_uniforms(self, program: moderngl.Program) -> None:
        if "u_grid_origin" in program:
            program["u_grid_origin"].value = self.origin
        if "u_grid_cell" in program:
            program["u_grid_cell"].value = self.cell
        if "u_grid_dim" in program:
            program["u_grid_dim"].value = self.dim

    def _release(self) -> None:
        for b in self._buffers:
            b.release()
        self._buffers = []

    def release(self) -> None:
        self._release()


__all__ = ["SceneCull"]
