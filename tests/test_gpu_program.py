from __future__ import annotations

import numpy as np
import pytest

from core.gpu_node_types import IR_STACK_CAPACITY, OPCODE_SHIFT, Opcode
from core.gpu_program import (
    GpuProgramError,
    emit_program,
    encode_instruction,
)
from core.render_ir import RenderIR, RenderIRNode


def _leaf(index_owner: int) -> RenderIRNode:
    return RenderIRNode(
        kind="sphere", object_id=index_owner, dimension=3, children=(),
        params=(0.0, 0.0, 0.0, 1.0),
    )


def _decode(instruction: int) -> tuple[int, int]:
    return instruction >> OPCODE_SHIFT, instruction & 0x00FFFFFF


def test_union_of_two_leaves_emits_post_order() -> None:
    nodes = (
        _leaf(1),
        _leaf(2),
        RenderIRNode(kind="union", object_id=0, dimension=3,
                     children=(0, 1), params=()),
    )
    program = emit_program(RenderIR(nodes=nodes, root_indices=(2,),
                                    component_indices=()))
    decoded = [_decode(int(w)) for w in program.bytecode]
    assert decoded == [
        (Opcode.PUSH_LEAF, 0),
        (Opcode.PUSH_LEAF, 1),
        (Opcode.EVAL_OP, 2),
    ]
    assert program.peak_depth == 2
    assert program.program_length == 3


def test_flat_carve_chain_of_1000_passes() -> None:
    # difference(difference(...(box, s1), s2)..., s1000): a left-leaning chain.
    nodes: list[RenderIRNode] = [_leaf(0)]  # node 0 = base solid
    current = 0
    for owner in range(1, 1001):
        nodes.append(_leaf(owner))  # the carving tool
        tool = len(nodes) - 1
        nodes.append(
            RenderIRNode(kind="difference", object_id=0, dimension=3,
                         children=(current, tool), params=())
        )
        current = len(nodes) - 1
    program = emit_program(
        RenderIR(nodes=tuple(nodes), root_indices=(current,),
                 component_indices=())
    )
    # A flat chain never holds more than two unresolved operands at once.
    assert program.peak_depth == 2
    assert program.program_length == 1 + 2 * 1000


def test_nesting_beyond_capacity_raises() -> None:
    # An N-ary union with CAPACITY+1 children pushes that many before EVAL_OP.
    leaves = tuple(_leaf(i + 1) for i in range(IR_STACK_CAPACITY + 1))
    union = RenderIRNode(
        kind="union", object_id=0, dimension=3,
        children=tuple(range(IR_STACK_CAPACITY + 1)), params=(),
    )
    ir = RenderIR(nodes=(*leaves, union),
                  root_indices=(IR_STACK_CAPACITY + 1,), component_indices=())
    with pytest.raises(GpuProgramError, match="value-stack capacity"):
        emit_program(ir)


def test_nesting_exactly_at_capacity_passes() -> None:
    leaves = tuple(_leaf(i + 1) for i in range(IR_STACK_CAPACITY))
    union = RenderIRNode(
        kind="union", object_id=0, dimension=3,
        children=tuple(range(IR_STACK_CAPACITY)), params=(),
    )
    ir = RenderIR(nodes=(*leaves, union),
                  root_indices=(IR_STACK_CAPACITY,), component_indices=())
    program = emit_program(ir)
    assert program.peak_depth == IR_STACK_CAPACITY


def test_single_leaf_program() -> None:
    program = emit_program(
        RenderIR(nodes=(_leaf(1),), root_indices=(0,), component_indices=())
    )
    assert program.peak_depth == 1
    assert [_decode(int(w)) for w in program.bytecode] == [(Opcode.PUSH_LEAF, 0)]


def test_empty_scene_yields_empty_program() -> None:
    program = emit_program(None)
    assert program.program_length == 0
    assert program.bytecode.dtype == np.uint32
    assert program.peak_depth == 0


def test_encode_instruction_rejects_oversized_payload() -> None:
    with pytest.raises(GpuProgramError):
        encode_instruction(Opcode.PUSH_LEAF, 0x01000000)
