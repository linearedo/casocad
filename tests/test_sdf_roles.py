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
    Exactness,
    ExactnessError,
    exactness_violations,
    node_exactness,
    result_exactness,
    validate_exactness,
)


# --- result_exactness: the legal closed algebra (spec §4) --------------------


def test_intersect_inside_inside_is_inside() -> None:
    assert (
        result_exactness("intersect", Exactness.SDF_INSIDE, Exactness.SDF_INSIDE)
        is Exactness.SDF_INSIDE
    )


def test_subtract_inside_outside_is_inside() -> None:
    assert (
        result_exactness("subtract", Exactness.SDF_INSIDE, Exactness.SDF_OUTSIDE)
        is Exactness.SDF_INSIDE
    )


def test_union_outside_outside_is_outside() -> None:
    assert (
        result_exactness("union", Exactness.SDF_OUTSIDE, Exactness.SDF_OUTSIDE)
        is Exactness.SDF_OUTSIDE
    )


# --- result_exactness: illegal combinations are unrepresentable --------------


def test_intersect_with_obstacle_rejected() -> None:
    # Outside-exact results cannot fill an inside-exact Intersect slot.
    with pytest.raises(ExactnessError):
        result_exactness("intersect", Exactness.SDF_INSIDE, Exactness.SDF_OUTSIDE)


def test_subtract_is_non_commutative() -> None:
    # subtract(inside-exact, outside-exact) is legal; the reverse is not.
    with pytest.raises(ExactnessError):
        result_exactness("subtract", Exactness.SDF_OUTSIDE, Exactness.SDF_INSIDE)


def test_union_of_regions_rejected() -> None:
    # Union composes outside-exact fields; a union of inside-exact-only fields
    # would corrupt the interior.
    with pytest.raises(ExactnessError):
        result_exactness("union", Exactness.SDF_INSIDE, Exactness.SDF_INSIDE)


def test_unknown_operator_rejected() -> None:
    with pytest.raises(ExactnessError):
        result_exactness("smooth_union", Exactness.SDF_INSIDE, Exactness.SDF_INSIDE)


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


def test_node_exactness_leaf_is_both() -> None:
    assert node_exactness(_box()) is Exactness.SDF_BOTH


def test_node_exactness_intersection_is_inside_exact_only() -> None:
    node = Intersection(name="i", left=_box(), right=_sphere())
    assert node_exactness(node) is Exactness.SDF_INSIDE


def test_node_exactness_union_is_outside_exact_only() -> None:
    node = Union(name="u", left=_box(), right=_sphere())
    assert node_exactness(node) is Exactness.SDF_OUTSIDE


def test_transform_is_role_transparent() -> None:
    # A transform over a union still serves only the outside-exact role.
    union = Union(name="u", left=_box("a"), right=_box("b"))
    moved = Translate(name="t", child=union, offset=(1.0, 0.0, 0.0))
    assert node_exactness(moved) is Exactness.SDF_OUTSIDE


def test_von_karman_difference_is_valid() -> None:
    # Difference(inside-exact box, outside-exact cylinder) is the canonical carve.
    scene = Difference(name="fluid", left=_box(), right=_sphere())
    assert exactness_violations(scene) == []
    validate_exactness(scene)  # no raise


def test_subtract_union_of_obstacles_is_valid() -> None:
    # Subtracting a union of obstacles is legal (obstacle composer feeds the
    # obstacle slot).
    obstacles = Union(name="obs", left=_sphere("o1"), right=_sphere("o2"))
    scene = Difference(name="fluid", left=_box(), right=obstacles)
    assert exactness_violations(scene) == []


def test_intersect_with_union_is_rejected() -> None:
    # Union result is outside-exact only; it cannot fill an inside-exact slot.
    obstacles = Union(name="obs", left=_sphere("o1"), right=_sphere("o2"))
    scene = Intersection(name="bad", left=obstacles, right=_box())
    violations = exactness_violations(scene)
    assert any("Union cannot be used" in v for v in violations)
    assert any("A meshable Domain needs exact interior distance" in v for v in violations)
    with pytest.raises(ExactnessError):
        validate_exactness(scene)


def test_union_of_region_result_is_rejected() -> None:
    # A Difference result is inside-exact only; it cannot fill an outside-exact slot.
    region = Difference(name="r", left=_box(), right=_sphere())
    scene = Union(name="bad", left=region, right=_sphere("o2"))
    assert exactness_violations(scene)


def test_difference_left_slot_rejects_obstacle_result() -> None:
    # The left inside-exact slot of Subtract cannot take a Union result.
    obstacles = Union(name="obs", left=_sphere("o1"), right=_sphere("o2"))
    scene = Difference(name="bad", left=obstacles, right=_box())
    violations = exactness_violations(scene)
    assert any("Union cannot be used as the left operand" in v for v in violations)
    assert any("use this Union only as a subtraction cutter" in v for v in violations)
