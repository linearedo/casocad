from __future__ import annotations

"""Exactness system for the exact-SDF CFD kernel.

casoCAD is a *safe geometry compiler*: the only expressible geometry is a set of
named, interior-exact-distance Domains (spec v2). This module is the type
backbone that makes illegal geometry unrepresentable:

* :class:`Exactness` -- which side of its own boundary a field is guaranteed
  exact on.
* :class:`DomainKind` -- the physics tag on a top-level Domain (Fluid vs Solid).
  Geometry rules are identical for both; the tag only matters downstream.
* :func:`result_exactness` -- typed exactness-collapsing operator signatures from
  spec §4.
* :class:`Domain` -- a named, exported top-level cell whose root must be
  inside-exact.

This first migration commit is **purely additive**: nothing else imports this
module yet. Later steps wire it into ``operators.py``, ``scene.py``, and
``serialization.py`` so the operators themselves enforce these signatures.

Reference: ``docs/exact_signed_distance_field_cfd_migration_v2.md`` (§2–§4, §7).
"""

from dataclasses import dataclass
from enum import Enum, Flag, auto

from .base import SDFNode


class Exactness(Flag):
    """Which side of its boundary a field is exact on (§3).

    ``SDF_INSIDE`` -> exact inside the shape; valid as a Domain root.
    ``SDF_OUTSIDE`` -> exact outside the shape; valid as a subtraction cutter.
    ``SDF_BOTH`` -> full exact signed distance field.
    """

    NONE = 0
    SDF_INSIDE = auto()
    SDF_OUTSIDE = auto()
    SDF_BOTH = SDF_INSIDE | SDF_OUTSIDE


class DomainKind(Enum):
    """Physics tag on a top-level Domain (§2).

    The geometry rules are identical for both kinds; the tag only selects which
    solver consumes the exported mesh downstream.
    """

    FLUID = "fluid"
    SOLID = "solid"


class ExactnessError(ValueError):
    """Raised when an exact operator gets operands with the wrong exactness.

    This is the compiler refusing to build non-exact geometry: the typed grammar
    of spec §4 makes the illegal combination unrepresentable rather than merely
    discouraged.
    """


# Typed exactness-collapsing operator signatures: operator -> (left, right, result).
#   intersect(inside, inside) -> inside   f = max(f_A, f_B)
#   subtract (inside, outside)-> inside   f = max(f_R, -f_O)  [non-commutative]
#   union    (outside,outside)-> outside  f = min(f_A, f_B)
_OPERATOR_SIGNATURES: dict[str, tuple[Exactness, Exactness, Exactness]] = {
    "intersect": (Exactness.SDF_INSIDE, Exactness.SDF_INSIDE, Exactness.SDF_INSIDE),
    "subtract": (Exactness.SDF_INSIDE, Exactness.SDF_OUTSIDE, Exactness.SDF_INSIDE),
    "union": (Exactness.SDF_OUTSIDE, Exactness.SDF_OUTSIDE, Exactness.SDF_OUTSIDE),
}


def _exactness_label(exactness: Exactness) -> str:
    if exactness == Exactness.NONE:
        return "none"
    if exactness == Exactness.SDF_BOTH:
        return "both"
    labels: list[str] = []
    if Exactness.SDF_INSIDE in exactness:
        labels.append("inside")
    if Exactness.SDF_OUTSIDE in exactness:
        labels.append("outside")
    return "+".join(labels)


def result_exactness(
    operator: str,
    left: Exactness,
    right: Exactness,
) -> Exactness:
    """Validate operand exactness for an operator and return result exactness.

    Raises :class:`ExactnessError` if ``operator`` is unknown or the operands do
    not provide the exactness required by its slots. ``subtract`` is
    non-commutative: ``(SDF_INSIDE, SDF_OUTSIDE)`` is legal, the reverse is not.
    """

    try:
        want_left, want_right, result = _OPERATOR_SIGNATURES[operator]
    except KeyError:
        raise ExactnessError(
            f"unknown exact operator {operator!r}; "
            f"allowed: {sorted(_OPERATOR_SIGNATURES)}"
        ) from None
    if want_left not in left or want_right not in right:
        raise ExactnessError(
            f"{operator}({_exactness_label(left)}, {_exactness_label(right)}) "
            "is not exact-safe; required "
            f"{operator}({_exactness_label(want_left)}, "
            f"{_exactness_label(want_right)})"
        )
    return result


# --- Slot exactness validation engine (explicit-at-the-operator-slot, spec §4)
#
# Exactness is NOT stored on nodes. Each operator has fixed slots with required
# exactness, and a node's result exactness is determined structurally by its top
# operator. A leaf primitive (or an exact generator / transform thereof) has
# Exactness.SDF_BOTH, so it can fill either slot -- that is what lets the same
# pipe node be a Domain root in one place and a subtraction cutter in another.
#
# This is a validation pass, not constructor enforcement: the raw operator
# dataclasses stay general (the app is also a free SDF renderer), and the Model
# build (§5) / the typed authoring API call validate_exactness() to refuse a
# non-exact CFD scene. node.kind strings are used for dispatch to avoid importing
# operators/transforms here (no import cycle).

_BOTH = Exactness.SDF_BOTH

# operator kind -> (left-slot exactness, right-slot exactness, result exactness)
_KIND_SIGNATURE: dict[str, tuple[Exactness, Exactness, Exactness]] = {
    "intersection": (
        Exactness.SDF_INSIDE,
        Exactness.SDF_INSIDE,
        Exactness.SDF_INSIDE,
    ),
    "difference": (
        Exactness.SDF_INSIDE,
        Exactness.SDF_OUTSIDE,
        Exactness.SDF_INSIDE,
    ),
    "union": (
        Exactness.SDF_OUTSIDE,
        Exactness.SDF_OUTSIDE,
        Exactness.SDF_OUTSIDE,
    ),
}

# XOR is intentionally not part of the exact compiler grammar: coincident or
# touching boundaries can cancel, making the standard XOR SDF non-exact.
_NON_EXACT_OPERATOR_KINDS: frozenset[str] = frozenset({"xor"})

# Exactness-transparent unary transforms (isometry / uniform scale): result
# exactness is the child's exactness (spec §6).
_TRANSFORM_KINDS: frozenset[str] = frozenset({"translate", "rotate", "scale"})


def _node_exactness(node: SDFNode, cache: dict[int, Exactness]) -> Exactness:
    """Exactness this node's result can provide (spec §4).

    * an exact leaf / generator -> ``SDF_BOTH``,
    * ``intersection`` / ``difference`` -> inside-exact only,
    * ``union`` -> outside-exact only,
    * a role-transparent transform -> its child's exactness.
    """

    node_id = id(node)
    cached = cache.get(node_id)
    if cached is not None:
        return cached

    kind = node.kind
    if kind in _NON_EXACT_OPERATOR_KINDS:
        result = Exactness.NONE
    else:
        signature = _KIND_SIGNATURE.get(kind)
        if signature is not None:
            result = signature[2]
        elif kind in _TRANSFORM_KINDS:
            children = node.children()
            result = _node_exactness(children[0], cache) if children else _BOTH
        else:
            result = _BOTH
    cache[node_id] = result
    return result


def node_exactness(node: SDFNode) -> Exactness:
    return _node_exactness(node, {})


def _node_label(node: SDFNode) -> str:
    return f"{node.kind} {node.name!r}"


def _exactness_message(
    *,
    node: SDFNode,
    required: Exactness,
    actual: Exactness,
    context: str,
) -> str:
    if node.kind == "union" and required is Exactness.SDF_INSIDE:
        return (
            f"Union {context} would lose exact interior distance. "
            "Union preserves exact SDF distance outside the combined shapes, "
            "but it does not generally preserve exact distance inside them. "
            "A meshable Domain needs exact interior distance for meshing. "
            "Use Difference/Intersection to build the Domain, or use this "
            "Union only as a subtraction cutter."
        )
    if required is Exactness.SDF_INSIDE:
        return (
            f"{_node_label(node)} {context} is not an exact interior distance "
            "field. A meshable Domain needs exact interior SDF distance for "
            f"meshing; this expression provides {_exactness_label(actual)} "
            "exactness."
        )
    if required is Exactness.SDF_OUTSIDE:
        return (
            f"{_node_label(node)} {context} is not an exact outside distance "
            "field. Subtraction cutters must provide exact outside SDF "
            f"distance; this expression provides {_exactness_label(actual)} "
            "exactness."
        )
    return (
        f"{_node_label(node)} {context} does not provide the required "
        f"{_exactness_label(required)} exactness; it provides "
        f"{_exactness_label(actual)} exactness."
    )


def _non_exact_message(node: SDFNode) -> str:
    if node.kind == "xor":
        return (
            f"XOR {node.name!r} is not accepted for solver-ready Domains yet. "
            "casoCAD does not currently prove the extra boundary-cancellation "
            "conditions needed for XOR to preserve exact SDF distance. XOR is "
            "still available for free SDF modeling, but not for compiled "
            "meshing geometry."
        )
    return (
        f"{_node_label(node)} is not part of the exact-SDF compiler grammar."
    )


def exactness_violations(
    node: SDFNode,
    *,
    required: Exactness | None = Exactness.SDF_INSIDE,
) -> list[str]:
    """Return human-readable exactness violations in the subtree (empty = OK).

    For every operator node, each operand must be able to fill its slot's
    required exactness (its ``node_exactness`` must include that exactness). Leaves
    fill either slot; a ``union`` result cannot fill an inside-exact slot; etc.
    """

    violations: list[str] = []
    exactness_cache: dict[int, Exactness] = {}
    reported_non_exact: set[int] = set()
    root_exactness = _node_exactness(node, exactness_cache)
    if required is not None and required not in root_exactness:
        if node.kind in _NON_EXACT_OPERATOR_KINDS:
            violations.append(_non_exact_message(node))
            reported_non_exact.add(id(node))
        else:
            violations.append(
                _exactness_message(
                    node=node,
                    required=required,
                    actual=root_exactness,
                    context="cannot define a meshable Domain",
                )
            )

    def visit(n: SDFNode) -> None:
        signature = _KIND_SIGNATURE.get(n.kind)
        if signature is not None:
            want_left, want_right, _ = signature
            children = n.children()
            if len(children) == 2:
                left, right = children
                for child, want, slot in (
                    (left, want_left, "left"),
                    (right, want_right, "right"),
                ):
                    child_exactness = _node_exactness(child, exactness_cache)
                    if want not in child_exactness:
                        violations.append(
                            _exactness_message(
                                node=child,
                                required=want,
                                actual=child_exactness,
                                context=(
                                    f"cannot be used as the {slot} operand of "
                                    f"{n.kind} {n.name!r}"
                                ),
                            )
                        )
        elif n.kind in _NON_EXACT_OPERATOR_KINDS and id(n) not in reported_non_exact:
            violations.append(_non_exact_message(n))
            reported_non_exact.add(id(n))
        for child in n.children():
            visit(child)

    visit(node)
    return violations


def validate_exactness(node: SDFNode) -> None:
    """Raise :class:`ExactnessError` if the subtree has any exactness
    violation (spec §4). No-op on a valid tree."""

    violations = exactness_violations(node)
    if violations:
        raise ExactnessError(
            "scene cannot be compiled as solver-ready exact geometry:\n  "
            + "\n  ".join(violations)
        )


@dataclass(frozen=True)
class Domain:
    """A named, exported top-level cell (spec §2).

    ``region`` is the inside-exact SDF root; ``kind`` is the physics tag. The
    Domain-level disjointness invariant (§7) is a *Model*-level check, not
    enforced here.
    """

    name: str
    kind: DomainKind
    region: SDFNode

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Domain requires a non-empty name")


__all__ = [
    "Exactness",
    "DomainKind",
    "ExactnessError",
    "result_exactness",
    "node_exactness",
    "exactness_violations",
    "validate_exactness",
    "Domain",
]
