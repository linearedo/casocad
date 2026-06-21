from __future__ import annotations

"""Serialize a :class:`RenderIR` into the interpreter's std430 GPU buffers.

This is backend-agnostic (design §5, §7.1): it imports nothing from ``app/``
and emits only raw scalar arrays, so a future Vulkan backend or a CPU reference
evaluator can reuse the exact same bytes.

Layout (design §5.1, the "scalar-only alignment rule"):

* ``nodes``    — N records of exactly 8 ``uint32`` (32 bytes), perfectly aligned
                 on every driver with zero padding.
* ``params``   — one flat ``float32`` array; each node reads a window
                 ``[param_offset, param_offset + param_count)``. Vectors are
                 re-assembled on shader registers from sequential scalars.
* ``children`` — one flat ``uint32`` array of node indices; each node reads
                 ``[child_offset, child_offset + child_count)``.

Complex vector types and nested structs are deliberately banned from the buffers
to dodge driver-specific std430 vec3-padding bugs.
"""

from dataclasses import dataclass

import numpy as np

from .gpu_node_types import code_for
from .render_ir import RenderIR


# Number of uint32 fields per GpuNode record. Mirror of the GLSL struct in
# sdf_interpreter.glsl (design §5.1).
GPU_NODE_UINTS = 8
GPU_NODE_BYTES = GPU_NODE_UINTS * 4

# Field positions inside one GpuNode record.
FIELD_TYPE = 0
FIELD_DIM = 1
FIELD_BASE_OWNER_ID = 2
FIELD_FLAGS = 3
FIELD_PARAM_OFFSET = 4
FIELD_PARAM_COUNT = 5
FIELD_CHILD_OFFSET = 6
FIELD_CHILD_COUNT = 7


@dataclass(frozen=True)
class GpuSceneBuffers:
    """The serialized scalar buffers ready for SSBO upload.

    ``nodes`` is shape ``(node_count, GPU_NODE_UINTS)`` ``uint32``; ``params``
    is 1-D ``float32``; ``children`` is 1-D ``uint32``. Empty scenes still
    produce length-1 buffers so the GL upload always has a valid, bound range.
    """

    nodes: np.ndarray
    params: np.ndarray
    children: np.ndarray
    node_count: int

    @property
    def nodes_bytes(self) -> bytes:
        return self.nodes.tobytes()

    @property
    def params_bytes(self) -> bytes:
        return self.params.tobytes()

    @property
    def children_bytes(self) -> bytes:
        return self.children.tobytes()


def _u32(value: int) -> int:
    """Clamp a Python int into the unsigned 32-bit range for buffer packing.

    Owner ids and region ids are non-negative by construction; a negative
    object id (the IR uses ``-1`` for anonymous operators, already lifted to 0
    by ``build_render_ir``) is treated as 0.
    """

    if value < 0:
        return 0
    return int(value) & 0xFFFFFFFF


def serialize_scene(render_ir: RenderIR | None) -> GpuSceneBuffers:
    """Pack a RenderIR's nodes/params/children into flat scalar arrays.

    Node array index equals the RenderIR node index, so bytecode payloads
    (which carry RenderIR indices) address GPU records directly.
    """

    nodes_list = () if render_ir is None else render_ir.nodes
    node_count = len(nodes_list)

    nodes = np.zeros((max(node_count, 1), GPU_NODE_UINTS), dtype=np.uint32)
    params: list[float] = []
    children: list[int] = []

    for index, node in enumerate(nodes_list):
        param_offset = len(params)
        params.extend(float(value) for value in node.params)
        child_offset = len(children)
        children.extend(int(child) for child in node.children)

        record = nodes[index]
        record[FIELD_TYPE] = code_for(node.kind)
        record[FIELD_DIM] = _u32(node.dimension)
        record[FIELD_BASE_OWNER_ID] = _u32(node.object_id)
        record[FIELD_FLAGS] = _u32(node.flags)
        record[FIELD_PARAM_OFFSET] = param_offset
        record[FIELD_PARAM_COUNT] = len(node.params)
        record[FIELD_CHILD_OFFSET] = child_offset
        record[FIELD_CHILD_COUNT] = len(node.children)

    params_array = np.asarray(params or (0.0,), dtype=np.float32)
    children_array = np.asarray(children or (0,), dtype=np.uint32)

    return GpuSceneBuffers(
        nodes=nodes,
        params=params_array,
        children=children_array,
        node_count=node_count,
    )


def pack_param_update(render_ir: RenderIR | None) -> np.ndarray:
    """Flat ``float32`` params for the parameter-only fast path (design §12).

    Moving/resizing an object changes only ``node.params`` while topology (node
    types, children, offsets) is unchanged, so the renderer can ``glBufferSubData``
    this straight into the params SSBO without re-emitting bytecode.
    """

    if render_ir is None:
        return np.asarray((0.0,), dtype=np.float32)
    values = [float(value) for node in render_ir.nodes for value in node.params]
    return np.asarray(values or (0.0,), dtype=np.float32)


__all__ = [
    "GPU_NODE_UINTS",
    "GPU_NODE_BYTES",
    "GpuSceneBuffers",
    "serialize_scene",
    "pack_param_update",
]
