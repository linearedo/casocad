from __future__ import annotations

"""Semantic role / type system for the exact-SDF CFD kernel.

casoCAD is a *safe geometry compiler*: the only expressible geometry is a set of
named, interior-exact-distance Domains (spec v2). This module is the type
backbone that makes illegal geometry unrepresentable:

* :class:`Role` -- the algebra-side role (Region vs Obstacle), i.e. *which side*
  of its own boundary a field is guaranteed exact on.
* :class:`DomainKind` -- the physics tag on a top-level Domain (Fluid vs Solid).
  Geometry rules are identical for both; the tag only matters downstream.
* :func:`result_role` -- the typed operator signatures of spec §4. The three
  exact operators form a closed algebra; anything else is rejected.
* :class:`Domain` -- a Region promoted to a named, exported top-level cell.

This first migration commit is **purely additive**: nothing else imports this
module yet. Later steps wire it into ``operators.py``, ``scene.py``, and
``serialization.py`` so the operators themselves enforce these signatures.

Reference: ``docs/exact_signed_distance_field_cfd_migration_v2.md`` (§2–§4, §7).
"""

from dataclasses import dataclass
from enum import Enum

from .base import SDFNode


class Role(Enum):
    """Algebra-side role: which side of its boundary a field is exact on (§3).

    ``REGION``   -> inside-exact  (a fluid/solid building block).
    ``OBSTACLE`` -> outside-exact (subtracted to carve a neighbour).
    """

    REGION = "region"
    OBSTACLE = "obstacle"


class DomainKind(Enum):
    """Physics tag on a top-level Domain (§2).

    The geometry rules are identical for both kinds; the tag only selects which
    solver consumes the exported mesh downstream.
    """

    FLUID = "fluid"
    SOLID = "solid"


class IllegalOperandRole(ValueError):
    """Raised when an exact operator gets operands of the wrong :class:`Role`.

    This is the compiler refusing to build non-exact geometry: the typed grammar
    of spec §4 makes the illegal combination unrepresentable rather than merely
    discouraged.
    """


# Typed operator signatures (spec §4): operator -> (left, right, result).
# These three are the *entire* closed algebra; nothing else combines fields.
#   intersect(Region, Region)   -> Region   f = max(f_A, f_B)
#   subtract (Region, Obstacle) -> Region   f = max(f_R, -f_O)  [non-commutative]
#   union    (Obstacle,Obstacle)-> Obstacle f = min(f_A, f_B)
_OPERATOR_SIGNATURES: dict[str, tuple[Role, Role, Role]] = {
    "intersect": (Role.REGION, Role.REGION, Role.REGION),
    "subtract": (Role.REGION, Role.OBSTACLE, Role.REGION),
    "union": (Role.OBSTACLE, Role.OBSTACLE, Role.OBSTACLE),
}


def result_role(operator: str, left: Role, right: Role) -> Role:
    """Validate operand roles for an exact operator and return its result role.

    Raises :class:`IllegalOperandRole` if ``operator`` is unknown or the operand
    roles do not match its signature. ``subtract`` is non-commutative:
    ``(REGION, OBSTACLE)`` is legal, ``(OBSTACLE, REGION)`` is not.
    """

    try:
        want_left, want_right, result = _OPERATOR_SIGNATURES[operator]
    except KeyError:
        raise IllegalOperandRole(
            f"unknown exact operator {operator!r}; "
            f"allowed: {sorted(_OPERATOR_SIGNATURES)}"
        ) from None
    if (left, right) != (want_left, want_right):
        raise IllegalOperandRole(
            f"{operator}({left.value}, {right.value}) is not exact-safe; "
            f"required {operator}({want_left.value}, {want_right.value})"
        )
    return result


# --- Slot-role validation engine (explicit-at-the-operator-slot, spec §4) ----
#
# Roles are NOT stored on nodes. Each operator has fixed *slots* with required
# roles, and a node's *result role* is determined structurally by its top
# operator. A leaf primitive (or an exact generator / transform thereof) is
# exact on BOTH sides, so it can fill EITHER slot — that is what lets the *same*
# pipe node be a Region in one place and an Obstacle in another (role per use).
#
# This is a validation pass, not constructor enforcement: the raw operator
# dataclasses stay general (the app is also a free SDF renderer), and the Model
# build (§5) / the typed authoring API call validate_roles() to refuse a
# non-exact CFD scene. node.kind strings are used for dispatch to avoid importing
# operators/transforms here (no import cycle).

_BOTH: frozenset[Role] = frozenset({Role.REGION, Role.OBSTACLE})

# operator kind -> (left-slot role, right-slot role, result role)
_KIND_SIGNATURE: dict[str, tuple[Role, Role, Role]] = {
    "intersection": (Role.REGION, Role.REGION, Role.REGION),
    "difference": (Role.REGION, Role.OBSTACLE, Role.REGION),
    "union": (Role.OBSTACLE, Role.OBSTACLE, Role.OBSTACLE),
}

# Role-transparent unary transforms (isometry / uniform scale): result role is
# the child's result role (spec §6).
_TRANSFORM_KINDS: frozenset[str] = frozenset({"translate", "rotate", "scale"})


def node_result_roles(node: SDFNode) -> frozenset[Role]:
    """Roles this node's result can serve as (spec §4).

    * an exact leaf / generator -> ``{REGION, OBSTACLE}`` (exact both sides),
    * ``intersection`` / ``difference`` -> ``{REGION}`` (inside-exact only),
    * ``union`` -> ``{OBSTACLE}`` (outside-exact only),
    * a role-transparent transform -> its child's result roles.
    """

    kind = node.kind
    signature = _KIND_SIGNATURE.get(kind)
    if signature is not None:
        return frozenset({signature[2]})
    if kind in _TRANSFORM_KINDS:
        children = node.children()
        if children:
            return node_result_roles(children[0])
    return _BOTH


def role_violations(node: SDFNode) -> list[str]:
    """Return human-readable slot-role violations in the subtree (empty = OK).

    For every operator node, each operand must be able to fill its slot's
    required role (its ``node_result_roles`` must contain that role). Leaves fill
    either slot; a ``union`` result cannot fill a Region slot; etc.
    """

    violations: list[str] = []

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
                    if want not in node_result_roles(child):
                        violations.append(
                            f"{n.kind}({n.name!r}) {slot} slot requires "
                            f"{want.value}, but child {child.kind}({child.name!r}) "
                            f"can only serve {sorted(r.value for r in node_result_roles(child))}"
                        )
        for child in n.children():
            visit(child)

    visit(node)
    return violations


def validate_roles(node: SDFNode) -> None:
    """Raise :class:`IllegalOperandRole` if the subtree has any slot-role
    violation (spec §4). No-op on a valid tree."""

    violations = role_violations(node)
    if violations:
        raise IllegalOperandRole(
            "scene violates the exact-operator grammar:\n  "
            + "\n  ".join(violations)
        )


@dataclass(frozen=True)
class Domain:
    """A Region promoted to a named, exported top-level cell (spec §2).

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
    "Role",
    "DomainKind",
    "IllegalOperandRole",
    "result_role",
    "node_result_roles",
    "role_violations",
    "validate_roles",
    "Domain",
]
