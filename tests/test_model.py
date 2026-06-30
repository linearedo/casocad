from __future__ import annotations

import pytest

from core.model import (
    Model,
    ModelCompileError,
    compile_model,
    disjointness_violations,
    domains_overlap,
    grammar_violations,
    model_from_document,
)
from core.scene import SceneDocument
from core.sdf import Box, Difference, Intersection, Sphere, Union
from core.sdf.roles import Domain, DomainKind


def _fluid(name: str, region) -> Domain:
    return Domain(name=name, kind=DomainKind.FLUID, region=region)


def _solid(name: str, region) -> Domain:
    return Domain(name=name, kind=DomainKind.SOLID, region=region)


# --- Model container --------------------------------------------------------


def test_model_allows_unique_domain_names() -> None:
    m = Model(
        domains=(
            _fluid("sea", Sphere(name="a", center=(0, 0, 0), radius=0.4)),
            _solid("pipe", Sphere(name="b", center=(5, 0, 0), radius=0.4)),
        )
    )
    assert len(m.domains) == 2


def test_model_rejects_duplicate_domain_names() -> None:
    with pytest.raises(ValueError):
        Model(
            domains=(
                _fluid("dup", Sphere(name="a", radius=0.4)),
                _solid("dup", Sphere(name="b", center=(5, 0, 0), radius=0.4)),
            )
        )


# --- domains_overlap (the §7 disjointness probe) ----------------------------


def test_coincident_spheres_overlap() -> None:
    a = _fluid("a", Sphere(name="a", center=(0, 0, 0), radius=0.5))
    b = _fluid("b", Sphere(name="b", center=(0, 0, 0), radius=0.5))
    assert domains_overlap(a, b) is True


def test_far_apart_spheres_do_not_overlap() -> None:
    a = _fluid("a", Sphere(name="a", center=(0, 0, 0), radius=0.3))
    b = _fluid("b", Sphere(name="b", center=(5, 0, 0), radius=0.3))
    assert domains_overlap(a, b) is False


def test_touching_boxes_do_not_overlap() -> None:
    # Two boxes sharing the plane x=0: disjoint open interiors -> not an overlap.
    a = _fluid(
        "a", Box(name="a", center=(-0.5, 0, 0), half_size=(0.5, 0.5, 0.5))
    )
    b = _solid(
        "b", Box(name="b", center=(0.5, 0, 0), half_size=(0.5, 0.5, 0.5))
    )
    assert domains_overlap(a, b) is False


# --- compile_model: disjointness invariant (§7) -----------------------------


def test_compile_passes_for_disjoint_domains() -> None:
    m = Model(
        domains=(
            _fluid("a", Sphere(name="a", center=(0, 0, 0), radius=0.4)),
            _solid("b", Sphere(name="b", center=(3, 0, 0), radius=0.4)),
        )
    )
    compile_model(m)  # no raise


def test_compile_fails_on_overlap() -> None:
    m = Model(
        domains=(
            _fluid("a", Sphere(name="a", center=(0, 0, 0), radius=0.6)),
            _solid("b", Sphere(name="b", center=(0.2, 0, 0), radius=0.6)),
        )
    )
    assert disjointness_violations(m)
    with pytest.raises(ModelCompileError):
        compile_model(m)


# --- compile_model: role grammar invariant (§4) -----------------------------


def test_compile_passes_for_valid_carve_domain() -> None:
    # Difference(inside-exact, outside-exact) is the canonical fluid carve.
    region = Difference(
        name="fluid",
        left=Box(name="box", half_size=(1.0, 1.0, 1.0)),
        right=Sphere(name="hole", radius=0.3),
    )
    compile_model(Model(domains=(_fluid("only", region),)))  # no raise


def test_compile_fails_on_role_grammar_violation() -> None:
    # Union result is outside-exact only; it cannot fill an inside-exact slot.
    bad = Intersection(
        name="bad",
        left=Union(
            name="obs",
            left=Sphere(name="o1", radius=0.4),
            right=Sphere(name="o2", center=(0.3, 0, 0), radius=0.4),
        ),
        right=Box(name="box", half_size=(1.0, 1.0, 1.0)),
    )
    with pytest.raises(ModelCompileError):
        compile_model(Model(domains=(_fluid("only", bad),)))


# --- model_from_document adapter (§5b bridge) -------------------------------


def test_model_from_default_document_compiles() -> None:
    # The default von-Karman scene declares one FluidDomain root.
    document = SceneDocument.default()
    model = model_from_document(document)
    assert len(model.domains) == 1
    assert document.fluid_domain is not None
    assert model.domains[0].region is document.fluid_domain.root
    assert all(d.kind is DomainKind.FLUID for d in model.domains)
    compile_model(model)  # no raise


def test_model_from_document_honours_solid_kind_override() -> None:
    document = SceneDocument.default()
    assert document.fluid_domain is not None
    name = document.fluid_domain.root.name
    model = model_from_document(document, kinds={name: DomainKind.SOLID})
    assert model.domains[0].kind is DomainKind.SOLID


def test_model_from_document_ignores_undeclared_construction_objects() -> None:
    document = SceneDocument()
    document.add_primitive("sphere")
    document.add_primitive("box")
    model = model_from_document(document)
    assert model.domains == ()


def test_model_from_document_uses_declared_solid_domain() -> None:
    document = SceneDocument()
    handle = document.add_primitive("box")
    box = document.node(handle)
    document.set_domain_root(handle, DomainKind.SOLID)
    model = model_from_document(document)
    assert len(model.domains) == 1
    assert model.domains[0].name == box.name
    assert model.domains[0].kind is DomainKind.SOLID
    assert model.domains[0].region is box


def test_unset_domain_root_removes_declared_solid_domain() -> None:
    document = SceneDocument()
    handle = document.add_primitive("box")
    document.set_domain_root(handle, DomainKind.SOLID)

    document.unset_domain_root(handle)

    assert document.domain_kinds == {}
    assert model_from_document(document).domains == ()


def test_unset_domain_root_clears_fluid_domain() -> None:
    document = SceneDocument()
    handle = document.add_primitive("box")
    document.set_domain_root(handle, DomainKind.FLUID)

    document.unset_domain_root(handle)

    assert document.fluid_domain is None
    assert document.domain_kinds == {}


# --- grammar_violations: the cheap live-diagnostics half --------------------


def test_grammar_violations_empty_for_valid_model() -> None:
    region = Difference(
        name="fluid",
        left=Box(name="box", half_size=(1.0, 1.0, 1.0)),
        right=Sphere(name="hole", radius=0.3),
    )
    assert grammar_violations(Model(domains=(_fluid("only", region),))) == []


def test_grammar_violations_reports_illegal_wiring_with_domain_name() -> None:
    # Intersect of a Union result — the exactness-breaking mis-wire the boolean
    # menus can produce.
    bad = Intersection(
        name="bad",
        left=Union(
            name="obs",
            left=Sphere(name="o1", radius=0.4),
            right=Sphere(name="o2", center=(0.3, 0, 0), radius=0.4),
        ),
        right=Box(name="box", half_size=(1.0, 1.0, 1.0)),
    )
    issues = grammar_violations(Model(domains=(_fluid("widget", bad),)))
    assert issues
    assert all(v.startswith("Domain 'widget':") for v in issues)
