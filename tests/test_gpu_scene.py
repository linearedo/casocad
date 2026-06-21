from __future__ import annotations

import numpy as np

from core.gpu_node_types import (
    NODE_TYPE_CODES,
    emit_glsl_defines,
    IR_STACK_CAPACITY,
)
from core.gpu_scene import (
    GPU_NODE_BYTES,
    GPU_NODE_UINTS,
    FIELD_BASE_OWNER_ID,
    FIELD_CHILD_COUNT,
    FIELD_CHILD_OFFSET,
    FIELD_PARAM_COUNT,
    FIELD_PARAM_OFFSET,
    FIELD_TYPE,
    pack_param_update,
    serialize_scene,
)
from core.render_ir import RenderIR, RenderIRNode


def _scene() -> RenderIR:
    # sphere (id 1), box (id 2), union of the two (id 0).
    sphere = RenderIRNode(
        kind="sphere", object_id=1, dimension=3, children=(),
        params=(0.0, 0.0, 0.0, 2.0),
    )
    box = RenderIRNode(
        kind="box", object_id=2, dimension=3, children=(),
        params=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 1.0, 0.5, 0.5, 0.5),
    )
    union = RenderIRNode(
        kind="union", object_id=0, dimension=3, children=(0, 1), params=(),
    )
    return RenderIR(nodes=(sphere, box, union), root_indices=(2,),
                    component_indices=())


def test_scalar_alignment_golden_bytes() -> None:
    buffers = serialize_scene(_scene())
    # 8 plain uint32 -> exactly 32 bytes, 16-byte aligned, zero padding.
    assert GPU_NODE_UINTS == 8
    assert GPU_NODE_BYTES == 32
    assert buffers.nodes.dtype == np.uint32
    assert buffers.nodes.shape == (3, 8)
    assert len(buffers.nodes_bytes) == 3 * 32
    assert buffers.params.dtype == np.float32
    assert buffers.children.dtype == np.uint32


def test_node_records_round_trip() -> None:
    buffers = serialize_scene(_scene())
    sphere, box, union = buffers.nodes

    assert sphere[FIELD_TYPE] == NODE_TYPE_CODES["sphere"]
    assert sphere[FIELD_BASE_OWNER_ID] == 1
    assert sphere[FIELD_PARAM_OFFSET] == 0
    assert sphere[FIELD_PARAM_COUNT] == 4
    assert sphere[FIELD_CHILD_COUNT] == 0

    assert box[FIELD_TYPE] == NODE_TYPE_CODES["box"]
    assert box[FIELD_PARAM_OFFSET] == 4  # right after the sphere's 4 floats
    assert box[FIELD_PARAM_COUNT] == 15

    assert union[FIELD_TYPE] == NODE_TYPE_CODES["union"]
    assert union[FIELD_CHILD_OFFSET] == 0
    assert union[FIELD_CHILD_COUNT] == 2


def test_params_and_children_are_flat_concatenations() -> None:
    scene = _scene()
    buffers = serialize_scene(scene)
    expected_params = tuple(
        float(v) for node in scene.nodes for v in node.params
    )
    np.testing.assert_array_equal(
        buffers.params, np.asarray(expected_params, dtype=np.float32)
    )
    np.testing.assert_array_equal(
        buffers.children, np.asarray((0, 1), dtype=np.uint32)
    )


def test_param_only_fast_path_matches_full_serialize() -> None:
    scene = _scene()
    np.testing.assert_array_equal(
        pack_param_update(scene), serialize_scene(scene).params
    )


def test_empty_scene_yields_valid_bound_ranges() -> None:
    buffers = serialize_scene(None)
    assert buffers.node_count == 0
    # Length-1 fallbacks so a GL bind always has a valid range.
    assert buffers.nodes.shape == (1, 8)
    assert buffers.params.size == 1
    assert buffers.children.size == 1


def test_glsl_defines_match_python_codes() -> None:
    glsl = emit_glsl_defines()
    assert f"#define IR_STACK_CAPACITY {IR_STACK_CAPACITY}" in glsl
    for kind, code in NODE_TYPE_CODES.items():
        assert f"#define NODE_{kind.upper()} {code}u" in glsl
