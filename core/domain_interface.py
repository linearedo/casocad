from __future__ import annotations

"""DomainsInterface metadata: the exact shared surface between two Domains (§8).

When two Domains are built from the **same** primitive (e.g. the pipe's bore is
the gas Domain's wall *and* the obstacle subtracted from the steel Domain), their
shared boundary is exactly that primitive's zero-set. So an interface is
identified structurally by a leaf ``object_id`` that appears in *both* Domains'
regions; the leaf itself is the interface's generating SDF.

The Model retains, per interface, the two Domain names + the generating node +
its owner id, so a future mesher isolates the interface **analytically** (evaluate
one known SDF) instead of differencing two volumetric fields. This reuses the
same identity (``object_id`` / owner) that drives surface provenance (§4); §4 and
§8 share the machinery.

Reference: ``docs/exact_signed_distance_field_cfd_migration_v2.md`` (§8).
"""

from dataclasses import dataclass
from itertools import combinations

from core.model import Model
from core.sdf.base import SDFNode
from core.sdf.roles import Domain


@dataclass(frozen=True)
class DomainsInterface:
    """The exact surface shared by two adjacent Domains (spec §8).

    * ``domain_a`` / ``domain_b`` -- the two Domain names (sorted, deterministic).
    * ``owner_object_id`` -- the shared primitive's stable id.
    * ``generating_node`` -- that primitive; its zero-set **is** the interface,
      so the mesher isolates the interface by evaluating this one SDF.
    """

    domain_a: str
    domain_b: str
    owner_object_id: int
    generating_node: SDFNode


def _domain_leaves_by_id(domain: Domain) -> dict[int, SDFNode]:
    """Map each stable leaf ``object_id`` in a Domain's region to its node.

    Leaves with non-positive ids are skipped: id 0 is the unset/unstable
    sentinel and must not be treated as a shared identity.
    """

    return {
        leaf.object_id: leaf
        for leaf in domain.region.leaves()
        if leaf.object_id > 0
    }


def domain_interfaces(model: Model) -> tuple[DomainsInterface, ...]:
    """Return every DomainsInterface in a Model (§8).

    Two Domains share an interface for each leaf ``object_id`` present in both
    their regions -- that shared primitive's surface is the interface. A leaf
    shared by N Domains yields one interface per Domain pair.
    """

    leaves_by_domain = {d.name: _domain_leaves_by_id(d) for d in model.domains}
    interfaces: list[DomainsInterface] = []
    for a, b in combinations(model.domains, 2):
        shared_ids = set(leaves_by_domain[a.name]) & set(leaves_by_domain[b.name])
        name_a, name_b = sorted((a.name, b.name))
        for object_id in sorted(shared_ids):
            interfaces.append(
                DomainsInterface(
                    domain_a=name_a,
                    domain_b=name_b,
                    owner_object_id=object_id,
                    generating_node=leaves_by_domain[a.name][object_id],
                )
            )
    return tuple(interfaces)


__all__ = [
    "DomainsInterface",
    "domain_interfaces",
]
