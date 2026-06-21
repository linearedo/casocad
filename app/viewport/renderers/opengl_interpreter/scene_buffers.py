from __future__ import annotations

"""Owns the four interpreter SSBOs (nodes, params, children, bytecode).

A topology change re-creates the buffers and is the only "expensive" path — and
even that is just an upload, never a shader recompile (design §7). Moving or
resizing an object changes only ``node.params``; ``update_params`` writes the
flat float array straight into the existing params SSBO (``glBufferSubData`` fast
path, §12) with no bytecode regeneration.
"""

import moderngl
import numpy as np

from core.gpu_program import GpuProgram
from core.gpu_scene import GpuSceneBuffers

# SSBO binding points — must match sdf_core.glsl.
BIND_NODES = 0
BIND_PARAMS = 1
BIND_CHILDREN = 2
BIND_BYTECODE = 3


class SceneBuffers:
    def __init__(self, ctx: moderngl.Context) -> None:
        self.ctx = ctx
        self._nodes: moderngl.Buffer | None = None
        self._params: moderngl.Buffer | None = None
        self._children: moderngl.Buffer | None = None
        self._bytecode: moderngl.Buffer | None = None
        self.program_length = 0
        self._params_nbytes = 0

    def upload(self, scene: GpuSceneBuffers, program: GpuProgram) -> None:
        """Full (topology) upload: re-create all four buffers."""
        self.release()
        self._nodes = self.ctx.buffer(scene.nodes_bytes)
        params_bytes = scene.params_bytes
        self._params = self.ctx.buffer(params_bytes)
        self._params_nbytes = len(params_bytes)
        self._children = self.ctx.buffer(scene.children_bytes)
        self._bytecode = self.ctx.buffer(
            program.bytecode_bytes or np.zeros(1, dtype=np.uint32).tobytes()
        )
        self.program_length = program.program_length

    def update_params(self, params: np.ndarray) -> bool:
        """Parameter-only fast path: overwrite the params SSBO in place.

        Returns False if the param buffer size changed (which means topology
        actually changed and a full ``upload`` is required instead).
        """
        if self._params is None:
            return False
        data = np.ascontiguousarray(params, dtype=np.float32).tobytes()
        if len(data) != self._params_nbytes:
            return False
        self._params.write(data)
        return True

    def bind(self) -> None:
        assert self._nodes and self._params and self._children and self._bytecode
        self._nodes.bind_to_storage_buffer(BIND_NODES)
        self._params.bind_to_storage_buffer(BIND_PARAMS)
        self._children.bind_to_storage_buffer(BIND_CHILDREN)
        self._bytecode.bind_to_storage_buffer(BIND_BYTECODE)

    @property
    def ready(self) -> bool:
        return self._nodes is not None

    def release(self) -> None:
        for buf in (self._nodes, self._params, self._children, self._bytecode):
            if buf is not None:
                buf.release()
        self._nodes = self._params = self._children = self._bytecode = None
        self.program_length = 0
        self._params_nbytes = 0


__all__ = ["SceneBuffers", "BIND_NODES", "BIND_PARAMS", "BIND_CHILDREN", "BIND_BYTECODE"]
