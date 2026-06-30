from __future__ import annotations

"""The exact-geometry Model: a set of disjoint, named Domains (spec v2 §2, §7).

A :class:`Model` is the document-level object of the safe geometry compiler. Two
invariants make a Model *compilable* (:func:`compile_model`):

1. **Exactness grammar** -- every Domain's region satisfies the slot-exactness
   grammar (``core.sdf.roles``, §4): exactness is enforced *by construction*.
2. **Disjointness** -- the Domains are mutually disjoint (§7). Unlike exactness,
   this is a **checked invariant**: overlap is a *compile error*, not something
   the type system can prevent. ``overlap(A, B)`` iff some point is interior to
   both, i.e. ``max(f_A, f_B) < 0`` somewhere.

This module is an **additive layer** -- it does not yet replace ``SceneDocument``.
``compile_model()`` is what a future Model-build / export path calls to refuse a
non-exact or overlapping scene.

Reference: ``docs/exact_signed_distance_field_cfd_migration_v2.md`` (§4, §7).
"""

from dataclasses import dataclass
from itertools import combinations
from typing import Protocol

import numpy as np

from core.preconditions import precondition_violations
from core.sdf.base import SDFNode
from core.sdf.roles import Domain, DomainKind, exactness_violations


class ModelCompileError(ValueError):
    """Raised when a Model fails a compile-time invariant (exactness §4 or
    disjointness §7)."""


# Default samples per axis for the disjointness probe. A dense-ish grid over the
# bounding-box overlap region; the spec notes an interval backstop as future work.
_DEFAULT_RESOLUTION = 32


@dataclass(frozen=True)
class Model:
    """A set of named Domains. Names must be unique; disjointness is checked at
    compile time, not enforced here (§7)."""

    domains: tuple[Domain, ...] = ()

    def __post_init__(self) -> None:
        names = [d.name for d in self.domains]
        if len(names) != len(set(names)):
            raise ValueError("Domain names must be unique within a Model")


class _HasObjects(Protocol):
    """Minimal structural view of a SceneDocument (avoids a core.scene import
    cycle): anything exposing a list of top-level SDF objects."""

    objects: list[SDFNode]


def model_from_document(
    document: _HasObjects,
    *,
    kinds: dict[str, DomainKind] | None = None,
) -> Model:
    """Derive a :class:`Model` from a free-form ``SceneDocument`` (spec §5).

    Each top-level object becomes a named Domain. Existing scenes carry no
    Fluid/Solid tag, so every Domain defaults to ``FLUID``; pass ``kinds`` (by
    object name) to mark Solid domains. This is the additive bridge that lets the
    current document feed :func:`compile_model` -- it does not yet replace
    ``SceneDocument``.

    The Model's unique-name invariant applies: top-level object names must be
    distinct (they are, in normal documents).
    """

    kinds = kinds or {}
    domains = tuple(
        Domain(
            name=obj.name,
            kind=kinds.get(obj.name, DomainKind.FLUID),
            region=obj,
        )
        for obj in document.objects
    )
    return Model(domains=domains)


def domains_overlap(
    a: Domain, b: Domain, *, resolution: int = _DEFAULT_RESOLUTION
) -> bool:
    """Return True if Domains ``a`` and ``b`` share interior volume (§7).

    Overlap can only occur inside *both* regions, hence inside the intersection
    of their bounding boxes. If those boxes are disjoint the domains are disjoint
    (fast path). Otherwise a grid is sampled in that overlap box and the domains
    overlap iff some sample is interior to both (``f_a < 0`` and ``f_b < 0``).

    This is a sampled probe: domains that merely *touch* (share a boundary, with
    disjoint open interiors) are correctly reported as non-overlapping.
    """

    if resolution < 2:
        raise ValueError("resolution must be at least 2")
    try:
        box = a.region.bounding_box().intersection(b.region.bounding_box())
    except ValueError:
        # Disjoint bounding boxes -> regions cannot share interior.
        return False

    xs = np.linspace(box.x_min, box.x_max, resolution)
    ys = np.linspace(box.y_min, box.y_max, resolution)
    zs = np.linspace(box.z_min, box.z_max, resolution)
    grid_x, grid_y, grid_z = np.meshgrid(xs, ys, zs, indexing="ij")

    f_a = a.region.to_numpy(grid_x, grid_y, grid_z)
    f_b = b.region.to_numpy(grid_x, grid_y, grid_z)
    return bool(np.any((f_a < 0.0) & (f_b < 0.0)))


def disjointness_violations(
    model: Model, *, resolution: int = _DEFAULT_RESOLUTION
) -> list[str]:
    """Return human-readable overlap reports for every overlapping Domain pair
    (empty = all disjoint)."""

    violations: list[str] = []
    for a, b in combinations(model.domains, 2):
        if domains_overlap(a, b, resolution=resolution):
            violations.append(
                f"Domains {a.name!r} and {b.name!r} overlap "
                f"(share interior volume); Domains must be disjoint"
            )
    return violations


def grammar_violations(model: Model) -> list[str]:
    """Return exactness-grammar violations across all Domains (empty = OK).

    This is the *cheap half* of :func:`compile_model` -- a pure tree walk (§4),
    with no disjointness sampling. Suitable for **live** per-edit diagnostics:
    it only fires on genuinely illegal operator wiring, which normal scenes never
    produce, so it is quiet. Disjointness (§7) is deferred to an explicit gate
    (a Validate action / future mesh step), not run live.
    """

    violations: list[str] = []
    for domain in model.domains:
        for issue in exactness_violations(domain.region):
            violations.append(f"Domain {domain.name!r}: {issue}")
    return violations


def compile_model(
    model: Model, *, resolution: int = _DEFAULT_RESOLUTION
) -> None:
    """Validate a Model's compile-time invariants; raise on the first failure.

    Checks, in order: (1) every Domain region satisfies the exactness grammar (§4);
    (2) the Domains are mutually disjoint (§7). Raises :class:`ModelCompileError`
    if either fails. No-op on a valid Model.
    """

    for domain in model.domains:
        exactness_issues = exactness_violations(domain.region)
        if exactness_issues:
            raise ModelCompileError(
                f"Domain {domain.name!r} cannot be compiled for meshing:\n  "
                + "\n  ".join(exactness_issues)
            )

    for domain in model.domains:
        precondition_issues = precondition_violations(domain.region)
        if precondition_issues:
            raise ModelCompileError(
                f"Domain {domain.name!r} violates a generator/offset precondition:"
                "\n  " + "\n  ".join(precondition_issues)
            )

    overlaps = disjointness_violations(model, resolution=resolution)
    if overlaps:
        raise ModelCompileError("\n".join(overlaps))


__all__ = [
    "Model",
    "ModelCompileError",
    "model_from_document",
    "domains_overlap",
    "disjointness_violations",
    "grammar_violations",
    "compile_model",
]
