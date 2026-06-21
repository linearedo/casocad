from __future__ import annotations

"""Emit the interpreter's bytecode program and validate per-ray stack depth.

Backend-agnostic (design §5, §6, §7.1). The program is a post-order
linearization of the 3D node graph: operator nodes become ``EVAL_OP`` (pop
``child_count`` / push one), every other reachable node becomes ``PUSH_LEAF``
(a self-contained sample; its profile subtree is evaluated by the nested VM, not
on the main value stack, so it is never walked here).

The emitter simulates the exact push/pop of every instruction, computes the peak
concurrent stack depth, and raises :class:`GpuProgramError` if it exceeds
``IR_STACK_CAPACITY``. This is the §1.2(2) / §12 guarantee: the host refuses to
upload mathematically impossible topologies so the shader never needs runtime
safety clamping.
"""

from dataclasses import dataclass

import numpy as np

from .gpu_node_types import (
    IR_PROFILE_STACK_CAPACITY,
    IR_STACK_CAPACITY,
    OPCODE_SHIFT,
    Opcode,
    is_operator,
    is_profile,
)
from .render_ir import RenderIR, RenderIRNode


class GpuProgramError(Exception):
    """Raised when a scene cannot be serialized into a valid bytecode program."""


@dataclass(frozen=True)
class GpuProgram:
    """A validated bytecode stream plus its simulated peak stack depth."""

    bytecode: np.ndarray  # 1-D uint32
    program_length: int
    peak_depth: int

    @property
    def bytecode_bytes(self) -> bytes:
        return self.bytecode.tobytes()


def encode_instruction(opcode: Opcode, payload: int) -> int:
    """Pack ``(opcode << 24) | payload`` into one uint32 instruction word."""

    if payload < 0 or payload > 0x00FFFFFF:
        raise GpuProgramError(
            f"bytecode payload {payload} exceeds the 24-bit node-index range"
        )
    return (int(opcode) << OPCODE_SHIFT) | int(payload)


def _emit_node(
    root_index: int,
    nodes: tuple[RenderIRNode, ...],
    stream: list[int],
) -> None:
    """Append the post-order instructions for ``root_index`` to ``stream``.

    Operators emit their children first (so operands are already on the stack
    when the ``EVAL_OP`` runs); leaves stop the descent — their children are
    profile-subtree references handled by the nested profile VM.

    The traversal is iterative with an explicit work stack: a long flat carve
    chain is exactly the case the design must handle effortlessly (§1.2(1)), and
    a recursive walk would hit Python's recursion limit well before the
    value-stack capacity is ever in question.
    """

    # Each frame is (node_index, is_post_visit). A post-visit frame for an
    # operator emits its EVAL_OP after its children have been emitted.
    work: list[tuple[int, bool]] = [(root_index, False)]
    while work:
        node_index, post_visit = work.pop()
        if post_visit:
            stream.append(encode_instruction(Opcode.EVAL_OP, node_index))
            continue
        node = nodes[node_index]
        if is_operator(node.kind):
            work.append((node_index, True))
            # Push children in reverse so the leftmost is emitted first.
            for child in reversed(node.children):
                work.append((int(child), False))
        else:
            stream.append(encode_instruction(Opcode.PUSH_LEAF, node_index))


def simulate_stack(
    stream: list[int],
    nodes: tuple[RenderIRNode, ...],
    capacity: int = IR_STACK_CAPACITY,
) -> tuple[int, int]:
    """Replay a bytecode stream on a virtual stack.

    Returns ``(peak_depth, final_depth)``. Raises :class:`GpuProgramError` on
    stack underflow (a malformed program) or when the peak exceeds ``capacity``
    (the value-stack array size the target shader was compiled with).
    """

    sp = 0
    peak = 0
    for instruction in stream:
        opcode = instruction >> OPCODE_SHIFT
        payload = instruction & 0x00FFFFFF
        if opcode == Opcode.PUSH_LEAF:
            sp += 1
        elif opcode == Opcode.EVAL_OP:
            arity = len(nodes[payload].children)
            if sp < arity:
                raise GpuProgramError(
                    f"operator node {payload} ({nodes[payload].kind}) pops "
                    f"{arity} operands but only {sp} are on the stack"
                )
            sp -= arity
            sp += 1  # combined result
        elif opcode == Opcode.REGION_ASSIGN:
            # Layer 2 re-tags the top element in place; depth is unchanged but
            # there must be something to re-tag.
            if sp < 1:
                raise GpuProgramError(
                    f"region selector node {payload} has no operand on the stack"
                )
        else:
            raise GpuProgramError(f"unknown opcode {opcode} in program stream")
        peak = max(peak, sp)
        if peak > capacity:
            raise GpuProgramError(
                f"scene exceeds value-stack capacity ({capacity}): peak "
                f"depth reached {peak}. Deep operator nesting must be "
                f"flattened or split."
            )
    return peak, sp


def _profile_subtree_peak(
    root: int,
    nodes: tuple[RenderIRNode, ...],
) -> int:
    """Peak value-stack depth of a profile sub-graph, evaluated post-order.

    Mirrors the shader's profile sub-VM (a profile node with children is a
    combinator that pops them and pushes one result; a childless node is a leaf).
    Iterative so a deep profile graph can't blow Python's recursion limit.
    """

    work: list[tuple[int, bool]] = [(root, False)]
    sp = 0
    peak = 0
    while work:
        idx, post = work.pop()
        node = nodes[idx]
        if post:
            sp -= len(node.children)
            sp += 1
        elif node.children:
            work.append((idx, True))
            for child in reversed(node.children):
                work.append((int(child), False))
            continue
        else:
            sp += 1
        peak = max(peak, sp)
    return peak


def validate_profile_depths(
    render_ir: RenderIR,
    capacity: int = IR_PROFILE_STACK_CAPACITY,
) -> None:
    """Validate every profile sub-graph against the profile value-stack capacity.

    A profile root is any profile-kind node referenced as a child of a
    non-profile leaf (extrude/revolve/placed_*). Each is validated independently
    (design §1.2(2), per profile).
    """

    for node in render_ir.nodes:
        if is_profile(node.kind):
            continue
        for child in node.children:
            child_node = render_ir.nodes[child]
            if not is_profile(child_node.kind):
                continue
            peak = _profile_subtree_peak(int(child), render_ir.nodes)
            if peak > capacity:
                raise GpuProgramError(
                    f"profile sub-graph rooted at node {child} "
                    f"({child_node.kind}) exceeds profile capacity "
                    f"({capacity}): peak depth {peak}"
                )


def emit_program(
    render_ir: RenderIR | None,
    *,
    stack_capacity: int = IR_STACK_CAPACITY,
    profile_capacity: int = IR_PROFILE_STACK_CAPACITY,
) -> GpuProgram:
    """Build and validate the bytecode program for a RenderIR.

    Multiple roots are combined by an implicit union (min) of their resulting
    samples; ``build_render_ir`` currently always yields a single root.
    """

    if render_ir is None or not render_ir.nodes or not render_ir.root_indices:
        empty = np.asarray((), dtype=np.uint32)
        return GpuProgram(bytecode=empty, program_length=0, peak_depth=0)

    validate_profile_depths(render_ir, profile_capacity)

    nodes = render_ir.nodes
    roots = render_ir.root_indices
    if len(roots) != 1:
        raise GpuProgramError(
            "multi-root programs are not supported yet; build_render_ir emits a "
            "single root that already encloses the full SDF operation tree"
        )

    stream: list[int] = []
    _emit_node(roots[0], nodes, stream)

    # Layer 2: region selectors run as downstream ops on the resolved sample
    # (design §1.2 note, §6). Each re-tags the final stack element in place.
    for index, node in enumerate(nodes):
        if node.kind == "region_selector":
            stream.append(encode_instruction(Opcode.REGION_ASSIGN, index))

    peak, final_depth = simulate_stack(stream, nodes, stack_capacity)
    if final_depth != 1:
        raise GpuProgramError(
            f"program must leave exactly one sample on the stack, found "
            f"{final_depth} residual — emitter produced an unbalanced stream"
        )

    bytecode = np.asarray(stream, dtype=np.uint32)
    return GpuProgram(
        bytecode=bytecode,
        program_length=len(stream),
        peak_depth=peak,
    )


__all__ = [
    "GpuProgramError",
    "GpuProgram",
    "encode_instruction",
    "simulate_stack",
    "validate_profile_depths",
    "emit_program",
]
