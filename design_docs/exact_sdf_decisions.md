# Exact-SDF Migration — Decision Log

> One entry per significant design choice: **what** we decided, the **alternatives**
> we weighed, and **why** we rejected them. This exists so the *reasoning* isn't
> trapped in chat history. Companion to the spec
> ([`exact_signed_distance_field_cfd_migration_v2.md`](./exact_signed_distance_field_cfd_migration_v2.md))
> and the build log (`progress/exact_sdf_migration_progress.md`).
>
> Last updated: 2026-06-23.

---

## D1 — Role model: explicit-at-the-operator-slot

**Decision.** A node's Region/Obstacle role is determined by **which operator slot
it is plugged into**, not stored on the node and not inferred loosely. `Subtract`
has a Region slot + an Obstacle slot; `Union` has two Obstacle slots; `Intersect`
has two Region slots. A leaf/primitive is exact on both sides, so it fits either
slot.

**Alternatives rejected.**
- *Role stored on the leaf node.* Breaks the shared-solid case: the pipe must be a
  Domain **and** an Obstacle (to the sea) at the same time — a single stored role
  can't be both, forcing duplication that silently drifts.
- *Infer role purely from position with no slot typing.* Can't catch the
  exactness-breaking combinations that are the whole point: it would relabel a
  union result as a "region" just because it sits in an intersect, hiding the
  error instead of flagging it.

**Why.** Explicit-at-slot is the only option that keeps "role per use" (shared
solids work) **and** rejects exactness-breaking wiring. See D5.

---

## D2 — Enforcement is a validation pass, not constructor-level

**Decision.** The exact-operator grammar is enforced by a **validation pass**
(`core/sdf/roles.py: validate_roles` / `role_violations`), called by the Model
build and the app — *not* by raising inside the operator constructors.

**Alternative rejected.** Make `Intersection`/`Union`/etc. reject illegal operand
roles at construction (truly "unrepresentable").

**Why.** casoCAD is *also* a general free SDF renderer; existing scenes/tests build
operator trees the CFD grammar forbids. Rejecting at construction would break the
working app. The validation pass gives the same guarantee at the points that
matter (compile / export) without breaking everything else.

---

## D3 — SmoothUnion deleted entirely (no quarantine)

**Decision.** Smooth-union (and the 1D/2D profile smooth variants) were **removed
from the whole stack** — kernel, registry, GLSL, UI, serialization.

**Alternative rejected.** Keep it but "quarantine" it (allowed for visual-only,
banned from export).

**Why.** A safe compiler must have **no reachable unsafe corner**. If the node can
exist, an agent or user can wire it toward the fluid field. Smooth blends destroy
the exact distance on both sides, so the only consistent choice is: it cannot be
constructed. Rounded shells survive only as exact constant **offset** (D8).

---

## D4 — Disjointness is a *checked invariant*, not "by construction"

**Decision.** Exactness is enforced *by construction* (the typed grammar).
**Disjointness of domains** is a separate, *global geometric* property enforced
**by compilation**: `compile_model` samples for overlap and refuses to build if
two domains share interior (`max(f_A,f_B) < 0` somewhere).

**Alternative rejected.** A space-partition (BSP-style) authoring model that makes
overlap structurally impossible.

**Why.** Disjointness can't be guaranteed by typing alone (it's global, not local),
and the partition model would force an unfamiliar authoring UX. A compile error —
like a real compiler refusing a type error — is honest and far cheaper.

### D4a — Live = grammar only; disjointness deferred to Validate/mesh

**Decision.** The **role-grammar** check runs **live** on every edit (cheap, quiet).
The **disjointness** check runs only on demand (**Domains ▸ Validate**) and, later,
at mesh time.

**Why.** Disjointness samples the field (expensive) and is **premature live**: until
domains are explicit, every transient overlap during normal editing (two shapes
before you combine them) would false-alarm. Grammar is cheap and only fires on
genuinely illegal wiring.

---

## D5 — "Obstacle" is a role per use, not a stored attribute

**Decision.** There is no "obstacle" flag on objects. A shape *is* an obstacle
**because** it occupies the obstacle slot of a `Subtract`. The same shape can be a
Domain in one place and an Obstacle in another.

**Why.** This is what makes the pipe-in-sea expressible: `pipe_outer` is the steel
Domain *and* the thing subtracted from the sea — same node, two uses. A stored flag
couldn't be both (D1).

---

## D6 — Authoring model: reusable shapes, domains reference them

**Decision (target UX, not yet built).** Shapes are authored once and **not consumed**
when used; **Domains** (named, Fluid/Solid) are built by **referencing** shapes with
the exact operators. The same shape can be referenced by multiple domains.

**Alternative rejected.** The current "consuming boolean" flow (operands disappear
into the result), or duplicating a shape to use it twice.

**Why.** Consuming booleans make the shared-solid case impossible; duplication gives
the copies different ids, so the shared-wall interface (§8) is lost and the copies
drift. Reference-based authoring is the only model where pipe-as-both works and the
interface is detected automatically (shared `object_id`).

---

## D7 — Additive layering; kept vestigial `smoothing` fields

**Decision.** The migration is **additive**: new `core/` modules (`roles`, `model`,
`surface_provenance`, `domain_interface`, `preconditions`) plus thin hooks, leaving
`SceneDocument`/the renderer working. The now-unused `smoothing` fields on
`BinaryProfile`/`BinaryProfile1D` and the viewport preview tuple were **left in place**.

**Why.** Removing the fields would churn `scene.py` + serialization for zero
functional gain; the smooth *operation* is already gone, so the fields are inert.
Additive layering kept the app green at every commit. (Cosmetic cleanup is a noted
carry-forward.)

---

## D8 — Offsets: dilation unconditional, erosion conditional

**Decision.** Dilation (grow, `f - r`) is exact unconditionally. Erosion (shrink) is
exact only while the radius stays below the shape's reach; `compile_model` rejects
erosion that reaches the max inscribed depth.

**Why.** Outer offsets only round convex features (always clean); inner offsets
self-intersect past the medial axis, breaking *exactness* (not just topology). The
enforced check is necessary-not-sufficient (true reach can be stricter at concave
features) — documented in `core/preconditions.py`.

---

## D9 — Multiple disjoint domains; both export modes

**Decision.** A model is a **set of disjoint Domains** (fluid or solid). Shared walls
between adjacent domains are `DomainsInterface`s, and export supports **both**
connected (conformal) and separated meshes — chosen per export, not baked into
geometry.

**Why.** Exact geometry forecloses nothing downstream: it keeps both export modes
open. A non-exact field would silently close the connected (FSI/conjugate) one.

---

## D10 — Boundary regions: one identity (owner + cut chain), one cutter

**Decision.** A `BoundaryRegion` is identified by **owner provenance (§4) + an
ordered chain of `(ghost, side)` cuts + an opaque string tag** — nothing else.
Ghost knives are embedded in the region's JSON record (never scene objects);
one `BoundaryCutter` tool replaces PlanarCutter/SurfaceCutter; splitting
replaces the parent region with two children that partition it exactly; the
same classifier (`core/boundary_region.py`) serves viewport highlighting and
the meshing API (`MeshableBoundaryRegion.contains`/`owner_sdf`). Tags are
never interpreted by the kernel — physics meaning belongs to mesher scripts.

**Why.** Three parallel identity systems (patch/direction, intervals, volume
selectors) made regions ambiguous, non-composable (a second cut silently
dropped the first), and partially invisible to the meshing API. One identity
makes every region savable, reloadable, and callable by construction. Full
design + implementation log: `design_docs/boundary_region_v2.md`.

---

## Open / deferred (decided to defer, not decided against)

- **Mesh-time disjointness gate** — wire when a real mesher exists (currently only
  the manual Validate action).
- **Authoring UX (D6) implementation** — the reusable-shapes + Domains panel; the
  big next phase.
- **`FluidDomain` reconciliation** — fold `core/mesher/domain.py` into `Domain`/`Model`.
- **Sweep/tube self-overlap precondition** — needs curvature analysis.
- **Friendlier validation messages** in the UI.
