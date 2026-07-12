# Exact SDF for CFD — v2: casoCAD as a Safe Geometry Compiler

> **Status:** design / migration spec (no code yet).
> **Builds on:** [`exact_signed_distance_field_cfd_migration.md`](./exact_signed_distance_field_cfd_migration.md) (*v1*).
>
> v1 proves the *math*: union preserves exterior exactness, intersection / subtraction
> preserve interior exactness, so `F = D \ (∪Oᵢ)` is interior-exact. **v2 turns that proof
> into the contract of the tool**: casoCAD must behave like a *safe geometry compiler*, where
> a geometry with a non-exact interior distance field is simply **not expressible**.

---

## 0. Thesis

The original mistake was exposing the **full SDF algebra** (smooth-min, displacement, twist, bend, …). 
That algebra is built to render a *surface*: only the zero-crossing has to be correct, 
and the field away from it may be a mere bound. The picture still looks right.

**CFD/FEA does not consume the surface — it consumes the field.** A solver needs, for every
*interior* point of a meshable region,

```math
f(p) = -d(p,\partial\,\text{region})
```

the *true* distance to the nearest wall (for `y⁺`, wall functions, near-wall turbulence models,
boundary-layer grading, structural meshing). A field that is "right at the surface but wrong
inside" silently corrupts all of that.

So casoCAD is not a modeler that *happens* to stay exact. It is a **compiler whose type system
is "exact interior distance field."** A model that violates it is a program that does not compile.
The guarantee is **by construction**, not by after-the-fact checking:

```math
\boxed{\text{No user — and no future LLM agent — can express a geometry with a non-exact interior distance field.}}
```

We **restrict hard at exactly one layer** (the geometry), and **leave freedom everywhere else**
(physics role, meshing, export). The restriction is the *only* place freedom is removed.

---

## 1. Scope of this migration

**There is no real mesher yet** (the existing lattice mesher is a placeholder and is out of
scope). v2 does **not** build a mesher.

v2's job is to define the **exact-geometry Model** and ensure it **carries every piece of
information a future mesher will need** — for *all* intended uses: separated meshes, connected
(conformal) meshes, fluid meshes, solid meshes, fluid–structure interaction.

In one line:

> **v2 = the exact-geometry Model + the metadata it must carry to feed any future mesher.**

---

## 2. Naming conventions (the vocabulary)

These terms are normative for the code and all later docs.

| Term | Meaning |
|------|---------|
| **Region** | An inside-exact building block — the *expression* level. Combined freely by `Intersect` / `Subtract`; **operand overlap is normal and expected**. Not named, not exported, **not** subject to disjointness. |
| **Domain** | A **Region promoted to a top-level, named, exported cell** — the *binding* level. Carries a **name**, is exported as one mesh, and is **disjoint from every other Domain**. |
| **FluidDomain** / **SolidDomain** | A Domain tagged by the physics that will run on it. *Geometrically identical rules* — the tag only matters downstream. e.g. `FluidDomain(name="sea")`, `SolidDomain(name="pipe")`, `FluidDomain(name="gas")`. |
| **interior-exact-distance** | The one hard invariant: a Region's SDF equals the true distance to its wall for **every interior point**. |
| **inside-exact** / **outside-exact** | Short operand-side adjectives: a field is a true distance on its *inside* / its *outside*. A Region must be inside-exact; an Obstacle must be outside-exact. |
| **Primitive** | An exact leaf shape (box, sphere, cylinder, …). Exact on *both* sides by construction. |
| **Obstacle** | A **role**, *not* a kind of object: a region subtracted from a neighbor to carve its boundary. The *same* solid may be its own Domain **and** act as an Obstacle for another Domain. |
| **Exact generator** | An operation that builds a higher-dimensional exact shape from a lower-dimensional exact SDF (extrude, revolve, sweep). |
| **Exact transform** | An operation that maps an exact SDF to an exact SDF, preserving *both* sides (isometry, uniform scale, dilation). |
| **Model** | The whole document: a **set of named Domains** with the **checked invariant** that they are mutually **disjoint** (§7). (Replaces the generic "Scene".) |
| **DomainsInterface** | The exact surface shared by two adjacent Domains (e.g. gas↔pipe). The Model **retains its generating SDF** (the shared primitive's zero-set), so the mesher isolates it analytically; meshed-as-shared only if export asks. |
| **Connected export** / **Separated export** | The two export modes — conformal shared nodes vs. independent meshes. **Both supported**; a per-export choice. |

The split between **Region** (expression) and **Domain** (named binding) is load-bearing:
disjointness is a property of the *set of Domains*, never of the Region sub-expressions that build
one — those overlap by design (a fluid built as `cylinder ∩ box` needs the two to overlap).

A sentence in this vocabulary:

> *A **Model** is a set of mutually **disjoint**, **named Domains** (each **Fluid** or **Solid**).
> Each Domain is a top-level **Region** built from **Primitives** via **exact generators**, **exact
> transforms**, and the **exact operators**; a solid may also act as an **Obstacle** that
> **subtracts** from a neighbor. Domains meet at **DomainsInterfaces**, exported **connected** or
> **separated**.*

---

## 3. The exactness contract

Every Region (and therefore every Domain) must be **interior-exact-distance**. The contract is
bookkeeping over one classification — every operation either *preserves* exactness on a side,
preserves it on *one* side only, or *destroys* it.

| Role | Must guarantee | Why |
|------|----------------|-----|
| **Domain** (named, exported) | **inside-exact** | it is the **mesher entry point** — the mesh is generated directly from its interior field |
| **Region** (building block) | **inside-exact** | it composes into a Domain; exactness must hold at *every* intermediate step or the Domain cannot inherit it |
| **Obstacle** (a role) | **outside-exact** | it is subtracted; the neighbor lives *outside* it |

Definitions: `f` is *inside-exact* on `A` iff `f(p) = -d(p,∂A) ∀ p∈A`; *outside-exact* iff
`f(p) = +d(p,∂A) ∀ p∉A`. Rigid maps and uniform scale preserve both; `min`/`max` each preserve
exactly one; smoothing preserves neither.

---

## 4. The exact operators (typed, closed algebra)

The operand **types are derived from which side each operand must be exact on** — they are not
arbitrary. This typing is what makes illegal geometry unrepresentable.

### Operators that produce a Region (result is inside-exact)

```
Intersect(Region, Region) -> Region          f = max(f_A, f_B)
Subtract (Region, Obstacle) -> Region         f = max(f_R, -f_O)     [NON-COMMUTATIVE]
```

- **`Intersect` needs *both* operands inside-exact**, so both must be **Regions**. An Obstacle is
  only *outside*-exact — its interior field is a bound — so it can never be an `Intersect` operand.
  (Intersect operands routinely *overlap*; that is the point, and is unrelated to the
  Domain-level disjointness invariant of §7.)
- **`Subtract` is the only way to bring an Obstacle into a Region**, precisely because it uses
  `−f_O`, reading the obstacle on its **exact outside**. That sign-flip is what makes it safe.
  It is **non-commutative**: `Subtract(Region, Obstacle)` only — `Obstacle − Region` is not a legal
  expression.

`Intersect` is *unconditionally* inside-exact (not just a bound), because a point inside `A∩B`
leaves the region by entering `(¬A)∪(¬B)`, and distance to a union of sets is the min of the
distances:

```math
d(p,\partial(A\cap B)) = \min\big(d(p,\partial A),\,d(p,\partial B)\big) = -\max(f_A,f_B)
```

### Operators that produce an Obstacle (result is outside-exact)

```
Union(Obstacle, Obstacle) -> Obstacle         f = min(f_A, f_B)
```

- `min` of two outside-exact fields is outside-exact (v1 §3). This is the **only** obstacle
  composer. The n-obstacle case never materialises `∪Oᵢ`; the top-level domain is
  `max(f_D, −f_{O₁}, …, −f_{Oₙ})` directly (v1 §5).

### The closed grammar

```
Region   := Primitive(role=region)
          | Generator(...)                  # extrude / revolve / sweep of an exact profile
          | Transform(Region)
          | Intersect(Region, Region)       # operands may (and usually must) overlap
          | Subtract(Region, Obstacle)

Obstacle := Primitive(role=obstacle)
          | Generator(...)
          | Transform(Obstacle)
          | Union(Obstacle, Obstacle)

Domain   := FluidDomain(name, Region) | SolidDomain(name, Region)   # a named, exported Region

Model    := { Domain, … }   with the checked invariant:  ∀ i≠j  Domainᵢ, Domainⱼ disjoint   (§7)
```

Nothing else produces a Region, and only a top-level Region becomes a Domain. There is no path to
a non-exact interior field.

### Surface provenance (boundary patches ride the operators)

Every operator is a `min`/`max` **selection**: at any boundary point, exactly one operand is
*active* (the one attaining the `min`/`max`). That selection already exists in the codebase — the
interpreter's `PUSH_LEAF` carries `(dist, owner, region)` and `BoundarySurfacePatch` records
`owner_object_id` / `owner` (`core/boundary_patches.py:74`). v2 makes **surface provenance a
first-class part of operator semantics**, so each cut surface inherits a patch tag from the leaf
that owns it:

- `Subtract(R, O)` — the newly cut surface (where `−f_O` is active) **inherits O's patch tag**
  (`Wall` by default). This is how an inlet/outlet/wall survives a boolean.
- `Intersect(R₁, R₂)` — each boundary point inherits the tag of whichever operand is **active**
  (the `max` selector).
- `Union(O₁, O₂)` — inherits from the **active** obstacle (the `min` selector).

Because provenance rides the *same* selection that computes the field, tagging adds no new
mechanism and cannot perturb exactness. **Boundary patches are therefore part of v2, not deferred**
— a meshable Domain always knows which physics tag each of its faces carries.

---

## 5. Exact generators (build 3D exact shapes from exact profiles)

*Updatable list — extend as new exactness-preserving generators are proven.*

- **Extrude** a 2D exact profile along a straight axis (prism) — exact, **unconditional**.
- **Revolve** a 2D exact profile about an axis — exact **iff the profile does not cross the axis**
  (precondition; must be validated, not assumed).
- **Sweep** a 2D exact profile along a path — exact **iff the tube does not self-overlap**
  (path curvature radius ≥ profile half-extent) (precondition).
- *(candidates to evaluate later: loft, others — add only with a proof/precondition.)*

---

## 6. Exact transforms (exact SDF → exact SDF, both sides)

*Updatable list.*

- **Translate** — isometry.
- **Rotate** — isometry.
- **Mirror / reflect** — isometry.
- **Uniform positive scale** — with the (`f(p/s)·s`, `s>0`) formula. Already
  correct in `core/sdf/transforms.py:55`.
- **Dilation** (grow), `f − r`, `r > 0` — **unconditionally exact.** Outer offsets only round
  convex features and never self-intersect; merging two features is fine.
- **Erosion** (shrink), `f + r`, `r > 0` — **conditionally exact.** Exact **iff `r < local reach`**
  (the medial-axis distance / local feature size). Past that, the inner offset self-intersects and
  `f + r` ceases to be the true distance: this is a **violation of interior-exactness**, not merely
  a topology anomaly. The builder must enforce `r < reach` (or fail compilation).
- *(add as found.)*

> Erosion is the honest, exact replacement for "rounded shells" — but, unlike dilation, it carries
> a precondition, just like the revolve/sweep generators (§5).

### Forbidden operations (destroy exactness — excluded from the kernel entirely)

None of these may exist for Region or Obstacle geometry. Each makes the field stop being a true
distance, so a safe compiler must not offer them.

| Operation | What it is | Why it breaks exactness |
|-----------|------------|-------------------------|
| **Non-uniform scale** | per-axis stretch | gradient no longer unit-length → not a distance |
| **Shear** | slanting space | distorts the metric → not a distance |
| **Twist** | progressive rotation along an axis | non-rigid space warp → not a distance |
| **Bend** | curving space along an axis | non-rigid space warp → not a distance |
| **Displacement** | adding `f(p) + g(p)` bumps | perturbs the field off true distance |
| **Smooth-union / any smooth blend** | `smin` fillet | deliberately pulls the field off true distance near the seam (deleted in §10) |

---

## 7. Multiple Domains, disjointness, and roles

A Model is a set of **disjoint, named Domains**. Worked example — an offshore gas pipeline in the
sea:

```
~~~~~  ┌───────────────┐  ~~~~~     three disjoint Domains:
~~~~~  │███████████████│  ~~~~~       FluidDomain(name="gas")    — bore interior
~~~~~  │███   GAS   ███│  ~~~~~       SolidDomain(name="pipe")   — steel shell  (FSI)
~~~~~  │███████████████│  ~~~~~       FluidDomain(name="sea")    — surrounding water
~~~~~  └───────────────┘  ~~~~~
~~~~~~~~~~~~ SEA ~~~~~~~~~~~~~~~~
```

- `gas`  = bore interior (Primitive / Intersect).
- `pipe` = `Subtract(pipe_outer, bore)` — the steel shell, **inside-exact via the same Subtract
  operator**; meshing the solid costs the guarantee nothing.
- `sea`  = `Subtract(sea_box, pipe_outer)` — the pipe plays the **Obstacle** role here.

Note the **same** solid (`pipe_outer`) is simultaneously a Domain (`pipe`) and an Obstacle (to
`sea`). "Obstacle" is a role per *use*, not a property of the object.

### Disjointness is a *checked invariant*, not "by construction"

Two guarantees in this spec are **different in kind**, and must not be conflated:

- **Exactness** is *local* — an operator either preserves a side or it doesn't — so it is genuinely
  enforced **by construction** (the typed grammar of §4). Nothing can express a non-exact field.
- **Disjointness** is a *global geometric* property of the finished Domains. It **cannot** be made
  structural by typing alone: nothing stops a user authoring an independent `seabed` Region that
  clips `pipe_outer`. So disjointness is **enforced by compilation, not by construction.**

Like a real compiler, casoCAD does not stop you *writing* an overlap — it **refuses to build the
Model** when one exists:

```
overlap(A, B)  ⇔  ∃ p :  max(f_A, f_B) < 0          # A∩B non-empty ⇒ compile error
```

checked across every pair of **top-level Domains** (sampled, with an interval backstop). It is
checked **only** between Domains — never between the Region operands inside one Domain, which
overlap by design (§4).

The authoring style that *avoids* overlaps — derive Domains from a shared set of solids with
consistent roles (`gas ⊆ pipe_outer`, `sea` excludes `pipe_outer`) — is a **recommended practice**,
not a structural guarantee; the compile-time check is the actual safety net.

---

## 8. DomainsInterface and export freedom

Where two Domains are adjacent they share a **DomainsInterface** — the *same* exact surface seen
from both sides (gas↔pipe inner wall; sea↔pipe outer wall). Because the surface is exact and
identical from both sides, the two meshes *can* be made to line up node-for-node.

The Model must **record** each DomainsInterface so a future mesher can isolate it **analytically**,
not by numerically differencing two volumetric fields (which is artifact-prone). For each interface
it retains:

- the **two Domain names** it joins,
- the **generating SDF of the shared surface** — the shared primitive's zero-set itself (e.g. the
  gas↔pipe interface *is* `{f_bore = 0}`), **plus its owner id**.

Because both Domains are built from the *same* primitive, the interface is exactly that primitive's
zero-set, clipped to the adjacency region via the **owner-tracking from §4**. So the mesher isolates
it by evaluating one known SDF — cheap and artifact-free — rather than searching for where two
volumes happen to touch. (This is the same `owner` machinery that propagates boundary patches; §4
and §8 share it.)

Export then offers, from the same Model, **both** modes:

- **Connected** — Domains share nodes at the interface (conformal; required for FSI / conjugate
  problems).
- **Separated** — each Domain meshed independently.

Neither is the default and neither is baked into the geometry. **Both coexist**; the choice is
made per export. Exact geometry forecloses nothing: it keeps *both* doors open. (A non-exact
field would silently close the connected one.)

---

## 9. Why this is the right line to draw

> **An exact interior distance field is the precondition for guaranteed meshability.**

A mesher needs, at every interior point, a correct distance, a correct gradient (= surface
normal), and a correct nearest-wall direction. An exact field provides all three everywhere — so
advancing-front / octree / marching schemes cannot be fed a contradictory value and cannot emit a
tangled or inverted cell from a field that "lied" between surfaces.

> **Perfect interior field ⇒ unbreakable, watertight geometry ⇒ a mesh that is guaranteed to be
> generatable.**

This is exactly why freedom is removed *only* at the geometry layer and left intact everywhere
downstream (physics tag, connected/separated export, …). The restriction *buys* the downstream
freedom.

---

## 10. Where today's code stands (audit)

| Item | File | State | v2 disposition |
|------|------|-------|----------------|
| `Union` (`min`) | `core/sdf/operators.py:32` | outside-exact | keep — **Obstacle** composer only |
| `Intersection` (`max`) | `core/sdf/operators.py:47` | inside-exact | keep — **Region** composer; type to `(Region, Region)` |
| `Difference` (`max(a,−b)`) | `core/sdf/operators.py:67` | inside-exact (result) | keep — rename concept to **Subtract**; type to `(Region, Obstacle)`, non-commutative |
| `SmoothUnion` | `core/sdf/operators.py:82` | **destroys both sides** | **DELETE** (no quarantine — a safe compiler has no reachable unsafe corner) |
| `profile_smooth_union_2d/_1d` | removed with the old GPU interpreter registry | destroys exactness | **DELETE** (lockstep with above) |
| `Translate` / `Rotate` | `core/sdf/transforms.py:30,87` | isometry, exact | keep |
| `Scale` (uniform) | `core/sdf/transforms.py:55` | exact (`·s` correction present) | keep; keep `factor>0` guard |
| Mirror / reflect | — | absent | add (isometry, exact) |
| `DistanceOffsetProfile` (`f−r`) | `core/sdf/primitives_2d.py:664` | exact | keep — exact **dilation**; **erosion** needs the `r < reach` precondition (§6) |
| operator provenance | CPU SDF/operator semantics | tracks active contributor | **promote to operator semantics** — drives boundary-patch provenance (§4) and interface isolation (§8) |
| `BoundarySurfacePatch` (`owner_object_id`) | `core/boundary_patches.py:74` | per-surface patch + owner | keep — the patch tags that ride operator provenance |
| `FluidDomain` | `core/mesher/`, wired `core/scene.py:164` | wraps the *final* root only | generalize → `Domain` = a **named** top-level Region (Fluid/Solid); Model holds N |
| `SceneDocument` | `core/scene.py:96` | free-form node graph, no roles | becomes **Model**: a set of named Domains + the disjointness check (§7) |
| von-Kármán default | `core/scene.py:142` | `Difference(box, cylinder)` | already the right pattern; re-express as `FluidDomain("fluid", Subtract(box_region, cylinder_obstacle))` |

There is **no** non-uniform scale / twist / bend / displacement in the codebase today — good; the
task is to keep them permanently out.

---

## 11. Migration steps

1. **Land this spec.** Agree the vocabulary (§2) and the deletion of `SmoothUnion` (§10).
2. **Typed kernel.** Introduce `Region`, `Domain` (= named Fluid/Solid Region), and the `Obstacle`
   role; make the exact operators (§4) carry their operand types so illegal combinations cannot be
   constructed.
3. **Delete smooth blends** across `operators.py` and the profile kinds — lockstep.
4. **Surface provenance in operators.** Promote the existing `owner` / `PUSH_LEAF` tracking to
   first-class operator output so each cut surface inherits a boundary-patch tag (§4); wire it to
   `BoundarySurfacePatch`.
5. **Model + disjointness check.** Replace the free-form `SceneDocument` with a Model holding N
   named Domains, and add the **compile-time overlap check** across Domain pairs (§7) — fail the
   build on overlap.
6. **DomainsInterface metadata.** Per adjacency, retain the two Domain names + the **generating SDF**
   of the shared surface and its owner id (§8) — so the mesher isolates interfaces analytically.
7. **Offset + generator preconditions.** Enforce `r < reach` for erosion (§6); add validators for
   revolve (no axis crossing) and sweep (no self-overlap) before treating output as exact.
8. **Exactness regression test.** On an analytic case (box minus sphere), assert
   `|f(p) − d(p,∂region)| < ε` for interior points — the guard proving the restriction actually
   delivers exactness.

---

## 12. Open items (carry forward)

- **Connected/Separated export mechanics** — belongs to the future mesher; v2 only guarantees the
  Model *carries* the interface generating-SDF + provenance it needs (§8).
- **Generator / offset preconditions** — exact statements + validators for revolve, sweep (§5), and
  the erosion `r < reach` bound (§6).
- **Patch taxonomy** — the *set* of physics tags (inlet / outlet / wall / symmetry / …) and the UI
  to assign them. The *propagation mechanism* is settled in v2 (§4); only the catalogue is open.
- **XOR** — both-sides exact given exact inputs (v1 §5); admit later if needed. Default: defer.
- **New exact generators/transforms** — extend §5 / §6 as proven.

---

## 13. One-line summary

> casoCAD is a **safe geometry compiler**: the only expressible geometry is a set of **named,
> interior-exact-distance Domains**, each a top-level **Region** built from exact primitives,
> generators, transforms, and the typed operators `Intersect(Region,Region)`,
> `Subtract(Region,Obstacle)`, `Union(Obstacle,Obstacle)` — **no smoothing, no warps, ever**.
> Exactness is enforced **by construction**; **disjointness** of Domains is enforced **by
> compilation** (overlap = build error). Operators carry **surface provenance**, so boundary patches
> and interface SDFs survive every boolean. From that one exact source the future mesher can export
> each Domain **connected or separated**, for fluid, solid, or FSI use.
