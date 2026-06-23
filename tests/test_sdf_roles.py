from __future__ import annotations

import pytest

from core.sdf import (
    Box,
    Difference,
    Intersection,
    Sphere,
    Translate,
    Union,
)
from core.sdf.roles import (
    Domain,
    DomainKind,
    IllegalOperandRole,
    Role,
    node_result_roles,
    result_role,
    role_violations,
    validate_roles,
)


# --- result_role: the legal closed algebra (spec §4) ------------------------


def test_intersect_region_region_is_region() -> None:
    assert result_role("intersect", Role.REGION, Role.REGION) is Role.REGION


def test_subtract_region_obstacle_is_region() -> None:
    assert result_role("subtract", Role.REGION, Role.OBSTACLE) is Role.REGION


def test_union_obstacle_obstacle_is_obstacle() -> None:
    assert result_role("union", Role.OBSTACLE, Role.OBSTACLE) is Role.OBSTACLE


# --- result_role: illegal combinations are unrepresentable ------------------


def test_intersect_with_obstacle_rejected() -> None:
    # An obstacle is only outside-exact; its interior is a bound, so it can
    # never be an Intersect operand.
    with pytest.raises(IllegalOperandRole):
        result_role("intersect", Role.REGION, Role.OBSTACLE)


def test_subtract_is_non_commutative() -> None:
    # subtract(Region, Obstacle) is legal; the reverse is not a legal expression.
    with pytest.raises(IllegalOperandRole):
        result_role("subtract", Role.OBSTACLE, Role.REGION)


def test_union_of_regions_rejected() -> None:
    # Union is the obstacle composer only; min is exterior-exact, so a union of
    # regions would corrupt the interior.
    with pytest.raises(IllegalOperandRole):
        result_role("union", Role.REGION, Role.REGION)


def test_unknown_operator_rejected() -> None:
    with pytest.raises(IllegalOperandRole):
        result_role("smooth_union", Role.REGION, Role.REGION)


# --- Domain container -------------------------------------------------------


def test_domain_holds_name_kind_region() -> None:
    sphere = Sphere(name="bore", radius=0.3)
    domain = Domain(name="gas", kind=DomainKind.FLUID, region=sphere)
    assert domain.name == "gas"
    assert domain.kind is DomainKind.FLUID
    assert domain.region is sphere


def test_domain_requires_non_empty_name() -> None:
    sphere = Sphere(name="bore", radius=0.3)
    with pytest.raises(ValueError):
        Domain(name="", kind=DomainKind.SOLID, region=sphere)


# --- Slot-role validation engine (spec §4) ----------------------------------


def _box(name: str = "b") -> Box:
    return Box(name=name)


def _sphere(name: str = "s") -> Sphere:
    return Sphere(name=name, radius=0.3)


def test_node_result_roles_leaf_is_both() -> None:
    assert node_result_roles(_box()) == {Role.REGION, Role.OBSTACLE}


def test_node_result_roles_intersection_is_region_only() -> None:
    node = Intersection(name="i", left=_box(), right=_sphere())
    assert node_result_roles(node) == {Role.REGION}


def test_node_result_roles_union_is_obstacle_only() -> None:
    node = Union(name="u", left=_box(), right=_sphere())
    assert node_result_roles(node) == {Role.OBSTACLE}


def test_transform_is_role_transparent() -> None:
    # A transform over a union still serves only the Obstacle role.
    union = Union(name="u", left=_box("a"), right=_box("b"))
    moved = Translate(name="t", child=union, offset=(1.0, 0.0, 0.0))
    assert node_result_roles(moved) == {Role.OBSTACLE}


def test_von_karman_difference_is_valid() -> None:
    # Difference(Region box, Obstacle cylinder) — the canonical fluid carve.
    scene = Difference(name="fluid", left=_box(), right=_sphere())
    assert role_violations(scene) == []
    validate_roles(scene)  # no raise


def test_subtract_union_of_obstacles_is_valid() -> None:
    # Subtracting a union of obstacles is legal (obstacle composer feeds the
    # obstacle slot).
    obstacles = Union(name="obs", left=_sphere("o1"), right=_sphere("o2"))
    scene = Difference(name="fluid", left=_box(), right=obstacles)
    assert role_violations(scene) == []


def test_intersect_with_union_is_rejected() -> None:
    # Union result is Obstacle-only; it cannot fill an Intersect's Region slot.
    obstacles = Union(name="obs", left=_sphere("o1"), right=_sphere("o2"))
    scene = Intersection(name="bad", left=obstacles, right=_box())
    assert role_violations(scene)
    with pytest.raises(IllegalOperandRole):
        validate_roles(scene)


def test_union_of_region_result_is_rejected() -> None:
    # A Difference result is Region-only; it cannot fill a Union's Obstacle slot.
    region = Difference(name="r", left=_box(), right=_sphere())
    scene = Union(name="bad", left=region, right=_sphere("o2"))
    assert role_violations(scene)


def test_difference_left_slot_rejects_obstacle_result() -> None:
    # The left (Region) slot of Subtract cannot take a Union (Obstacle) result.
    obstacles = Union(name="obs", left=_sphere("o1"), right=_sphere("o2"))
    scene = Difference(name="bad", left=obstacles, right=_box())
    violations = role_violations(scene)
    assert any("left slot requires region" in v for v in violations)
