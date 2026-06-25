from __future__ import annotations

"""Single source of truth for the SDF interpreter's node-type codes and limits.

This module is backend-agnostic: it imports nothing from ``app/`` and only
produces plain Python data plus a GLSL ``#define`` block generated from the same
tables, so the Python host and the shaders can never drift (design §7.1).

The integer ``kind -> code`` mapping, the stack-capacity constant, and the
bytecode opcodes are all defined here once. ``emit_glsl_defines()`` renders them
into a header chunk that the interpreter shader ``#include``s.
"""

from collections.abc import Iterable
from enum import IntEnum


# Peak number of concurrent unresolved operations on the per-ray value stack
# (design §1.2(2), §6). The host refuses to upload topologies whose simulated
# peak depth exceeds this; the shader never does runtime safety clamping.
IR_STACK_CAPACITY = 64

# Same cap, reused by the nested profile VM (design §10 Phase B). A profile
# sub-graph is validated independently against this bound.
IR_PROFILE_STACK_CAPACITY = 64


class Opcode(IntEnum):
    """High 8 bits of each 32-bit bytecode instruction (design §5)."""

    PUSH_LEAF = 0x01      # evaluate leaf SDF, push (dist, owner, region)
    EVAL_OP = 0x02        # pop child_count, push 1 per the node's operator type
    REGION_ASSIGN = 0x03  # Layer 2 selector-volume region split (§6.1)


# Each instruction is (opcode << OPCODE_SHIFT) | payload, where payload is a
# 24-bit node index unless the opcode documents otherwise.
OPCODE_SHIFT = 24
PAYLOAD_MASK = 0x00FFFFFF


# ---------------------------------------------------------------------------
# Node-type registry.
#
# Codes are assigned by declaration order in these tuples and frozen into
# ``NODE_TYPE_CODES``. Append new kinds at the end of the appropriate group to
# keep codes stable across releases. ``"unsupported"`` is code 0 so a zeroed
# buffer slot reads as an explicit no-op leaf.
# ---------------------------------------------------------------------------

# Category tags describe how the bytecode emitter and shader treat a kind:
#   "leaf"        -> PUSH_LEAF, self-contained (may own a profile subtree)
#   "operator"    -> EVAL_OP, pops child_count and pushes one combined sample
#   "profile2d"   -> evaluated only by the nested 2D profile VM, never on the
#                    main value stack
#   "profile1d"   -> evaluated only by the nested 1D profile VM
#   "selector"    -> REGION_ASSIGN target (Layer 2)
#   "special"     -> sentinel (e.g. unsupported)

_SPECIAL_KINDS = (
    "unsupported",
)

_OPERATOR_KINDS = (
    "union",
    "intersection",
    "difference",
)

_LEAF_3D_KINDS = (
    "sphere",
    "box",
    "cylinder",
    "cone",
    "capped_cone",
    "box_frame",
    "pyramid",
    "torus",
    "polyline_tube",
    "quadratic_bezier_tube",
    "extrude_profile_2d",
    "revolve_profile_2d",
)

_LEAF_PLACED_KINDS = (
    "placed_circle_2d",
    "placed_rectangle_2d",
    "placed_square_2d",
    "placed_rounded_rectangle_2d",
    "placed_ellipse_2d",
    "placed_profile_2d",
    "placed_polyline_1d",
    "placed_quadratic_bezier_curve_1d",
    "placed_profile_1d",
)

_LEAF_SPECIALIZED_2D_KINDS = (
    "placed_polygon_2d",
    "placed_quadratic_bezier_surface_2d",
    "placed_quadratic_bezier_polycurve_1d",
)

_PROFILE_2D_KINDS = (
    "profile_circle_2d",
    "profile_rectangle_2d",
    "profile_square_2d",
    "profile_rounded_rectangle_2d",
    "profile_ellipse_2d",
    "profile_polygon_2d",
    "profile_polyline_2d",
    "profile_quadratic_bezier_curve_1d",
    "profile_quadratic_bezier_surface_2d",
    "profile_offset_2d",
    "profile_distance_offset_2d",
    "profile_union_2d",
    "profile_intersection_2d",
    "profile_difference_2d",
)

_PROFILE_1D_KINDS = (
    "profile_segment_1d",
    "profile_union_1d",
    "profile_intersection_1d",
    "profile_difference_1d",
)

_SELECTOR_KINDS = (
    "region_selector",
)


_CATEGORY_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("special", _SPECIAL_KINDS),
    ("operator", _OPERATOR_KINDS),
    ("leaf", _LEAF_3D_KINDS),
    ("leaf", _LEAF_PLACED_KINDS),
    ("profile2d", _PROFILE_2D_KINDS),
    ("profile1d", _PROFILE_1D_KINDS),
    ("selector", _SELECTOR_KINDS),
    ("leaf", _LEAF_SPECIALIZED_2D_KINDS),
)


def _build_tables() -> tuple[dict[str, int], dict[str, str]]:
    codes: dict[str, int] = {}
    categories: dict[str, str] = {}
    next_code = 0
    for category, kinds in _CATEGORY_GROUPS:
        for kind in kinds:
            if kind in codes:
                raise ValueError(f"duplicate node kind in registry: {kind!r}")
            codes[kind] = next_code
            categories[kind] = category
            next_code += 1
    return codes, categories


NODE_TYPE_CODES, NODE_CATEGORIES = _build_tables()

# Reverse map for diagnostics / round-trip tests.
NODE_TYPE_KINDS: dict[int, str] = {code: kind for kind, code in NODE_TYPE_CODES.items()}

OPERATOR_KINDS = frozenset(_OPERATOR_KINDS)
PROFILE_2D_KINDS = frozenset(_PROFILE_2D_KINDS)
PROFILE_1D_KINDS = frozenset(_PROFILE_1D_KINDS)
PROFILE_KINDS = PROFILE_2D_KINDS | PROFILE_1D_KINDS


def code_for(kind: str) -> int:
    """Return the stable uint type code for an IR node ``kind``.

    Unknown kinds map to the ``unsupported`` sentinel (code 0) rather than
    raising, so a partially-supported scene still serializes; the host
    surfaces unsupported kinds separately via ``RenderIR.unsupported_kinds``.
    """

    return NODE_TYPE_CODES.get(kind, NODE_TYPE_CODES["unsupported"])


def is_operator(kind: str) -> bool:
    return kind in OPERATOR_KINDS


def is_profile(kind: str) -> bool:
    return kind in PROFILE_KINDS


def emit_glsl_defines(
    used_kinds: Iterable[str] | None = None,
    *,
    include_stack_defs: bool = True,
    include_opcode_defs: bool = True,
) -> str:
    """Render the GLSL ``#define`` header generated from the Python tables.

    The interpreter shaders ``#include`` this so the kind codes, opcodes, and
    stack capacities are guaranteed identical to the host serializer.

    ``used_kinds`` is an optional source-size trim for specialized codegen
    shaders. The default emits every node kind and support constant for
    interpreter-style callers.
    """

    lines: list[str] = [
        "// AUTO-GENERATED from core/gpu_node_types.py — do not edit by hand.",
        "",
    ]
    if include_stack_defs:
        lines.extend([
            f"#define IR_STACK_CAPACITY {IR_STACK_CAPACITY}",
            f"#define IR_PROFILE_STACK_CAPACITY {IR_PROFILE_STACK_CAPACITY}",
            "",
        ])
    if include_opcode_defs:
        lines.extend([
            f"#define OP_PUSH_LEAF {int(Opcode.PUSH_LEAF)}u",
            f"#define OP_EVAL_NODE {int(Opcode.EVAL_OP)}u",
            f"#define OP_REGION_ASSIGN {int(Opcode.REGION_ASSIGN)}u",
            f"#define OPCODE_SHIFT {OPCODE_SHIFT}u",
            f"#define PAYLOAD_MASK {PAYLOAD_MASK}u",
            "",
        ])
    used = None if used_kinds is None else set(used_kinds)
    for code, kind in sorted(NODE_TYPE_KINDS.items()):
        if used is not None and kind not in used:
            continue
        lines.append(f"#define NODE_{kind.upper()} {code}u")
    lines.append("")
    return "\n".join(lines)


__all__ = [
    "IR_STACK_CAPACITY",
    "IR_PROFILE_STACK_CAPACITY",
    "Opcode",
    "OPCODE_SHIFT",
    "PAYLOAD_MASK",
    "NODE_TYPE_CODES",
    "NODE_TYPE_KINDS",
    "NODE_CATEGORIES",
    "OPERATOR_KINDS",
    "PROFILE_2D_KINDS",
    "PROFILE_1D_KINDS",
    "PROFILE_KINDS",
    "code_for",
    "is_operator",
    "is_profile",
    "emit_glsl_defines",
]
