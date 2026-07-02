# Boundary Region v2 — one identity, one cutter

> **Status:** IMPLEMENTED 2026-07-02 (all 7 phases; deviations + follow-ups in
> `progress/boundary_region_v2_progress.md`).
> **Builds on:** `exact_signed_distance_field_cfd_migration_v2.md` §4 (surface
> provenance), §8 (analytic interface isolation), §12 (open patch taxonomy).
>
> Goal: reimplement boundary-region selection **once, definitively**. A region
> is identified by one mechanism only; the same exact classifier serves the
> viewport (hover/select/highlight) and the meshing API (custom mesher
> scripts). Everything a region *is* survives save/reload.

---

## 1. Problems with the current implementation (audit)

| # | Problem | Where |
|---|---------|-------|
| 1 | Two buttons (`PlanarCutter`, `SurfaceCutter`) expose the drawing gesture, not the result; both converge on the same ghost-volume mechanism | `viewport._PLANAR_CUTTER_KINDS` / `_SURFACE_CUTTER_KINDS`, cutter routing in `main_window` |
| 2 | Three parallel identity systems: `patch_id`+`outside_direction` (analytic faces), parametric intervals (`selector_start/end`), ghost volumes (`selector_id`) | `core/boundary.py` |
| 3 | A region stores **one** selector — cutting an already-cut region silently drops the first knife (nested cuts don't compose) | `scene.add_boundary_selector_split_regions` |
| 4 | Per-type patch/pick/preview whitelists: any leaf not Box/Cylinder/Cone/Sphere/Torus gets **zero** patches (Pyramid, BoxFrame, tubes, extrudes are not hover-selectable at all) | `_surface_patches_for_node` returns `()` by default |
| 5 | Owner/direction-only regions are **invisible to the meshing API** (`_boundary_region_callable` → `None`), and the callable that exists is the raw unscoped knife field, not region membership | `core/meshing/api.py` |
| 6 | Ghost selectors persist as hidden scene nodes (`__boundary_selector_*` in `objects` + `selectors` list on the fluid record) — the scene graph is polluted with non-geometry | `INTERNAL_BOUNDARY_SELECTOR_PREFIX` machinery |
| 7 | Hover/pick boundary tool is stubbed in the QRhi viewport (`begin_boundary_region_tool = _noop`) — dropped in the migration, never rebuilt | `renderers/qrhi/viewport.py` |
| 8 | `PATCH_TOLERANCE = 1.5e-3` and scope thickness `0.006` are absolute meters — wrong under mm/km working scales | `core/boundary_patches.py` |

---

## 2. The one identity

> **A boundary region = an owner surface + the ordered list of knife-halves
> that carved it + an opaque physics tag.**

```
BoundaryRegion
├── name                    user-facing, unique (JSON map key)
├── object_id               stable int (provenance / references)
├── owner_object_id         the leaf whose surface this region lives on (§4 provenance;
│                           for Subtract, the OBSTACLE owns the cut surface)
├── patch_id  (optional)    analytic sub-face scope where the owner type has one
│                           ("-X" box face, "wall" cylinder side, ...). Coarse scope
│                           only — refinement is always done by cuts.
├── cuts                    ordered list of BoundaryCut
│     BoundaryCut
│     ├── side              "inside" | "outside"   (which half of the knife this region keeps)
│     └── ghost             a serialized SDF node record (the knife geometry)
└── tag       (optional)    opaque string ("wall", "inlet", "clamped", "heat_flux_10W", ...)
```

### Membership (the classifier — single source of truth)

A world point `p` belongs to the region iff **all** of:

1. `|f_root(p)| ≤ tol` — it lies on the Domain's boundary;
2. the region's **owner leaf is the active operand** at `p` (the `min`/`max`
   selection of spec §4 — evaluated exactly by walking operand fields);
3. `p` is within the analytic `patch_id` scope, when present;
4. for **every** cut in the chain: `sign(G_i(p))` matches `side_i`.

Properties that make this "the last time":

- **Partition exactness.** One cut splits a region into two siblings that
  partition it exactly (every point has exactly one sign) — no gaps, no
  double-tagging, no meshes stored.
- **Composability.** Cutting a region only refines *that* region — the chain
  carries the parent's constraints, so a knife crossing other regions cannot
  touch them.
- **Totality.** Every region — including plain whole-face ones with an empty
  chain — compiles to the same classifier. Nothing is special-cased, so
  nothing can fall out of the mesher contract again.
- **Exactness.** Only sign tests of known SDFs (spec §8 style: evaluate known
  fields, never difference volumes numerically).

Tolerances are **scale-relative**: `tol = k · owner_extent` (owner bounding-box
diagonal), never absolute meters. Fixes problem 8 and survives mm/km work.

### Semantics notes

- The chain is stored **ordered** for lineage/undo/naming, but membership is a
  pure conjunction — order does not change the point set.
- The cut **tree**: splitting replaces the parent region with its two children
  (undo restores). The "rest" is never lost — it *is* the `outside` sibling.
- A knife that misses the region (either side empty) is a **no-op cut and
  is refused** — the parent stays untouched. (Revised from "warn, don't
  forbid" after real use: an empty child plus a duplicate-of-parent child is
  pure clutter.) Emptiness is validated against dense display-mesh vertices
  plus coarse grid sampling, before any mutation.

---

## 3. Ghost lifecycle

1. User selects a region (or a whole owner surface) and activates **Boundary
   Cutter** — the *single* tool replacing PlanarCutter/SurfaceCutter.
2. User draws **any** shape from the normal draw vocabulary. Rule:
   - dimension **3** (sphere, box, cylinder, ...) → the ghost volume is the
     shape itself;
   - dimension **< 3** (segment, polyline, profile, bezier) → extruded through
     the scene into a prism/half-space volume (the existing
     `surface_selector_volume` conversions, kept as-is).
3. The ghost is **never** a scene object: not in `objects`, not in the tree,
   not rendered, no `__boundary_selector_*` names, no `selectors` list. Its
   serialized parameters are embedded in the region record (`cuts[i].ghost`),
   reusing the standard node-record format (same serializer as `objects`).
4. On load, ghost nodes are rebuilt from the records on demand (classifier
   construction), and cached per region.

---

## 4. Tags are opaque

- `tag: str | None` on the record. The kernel stores, round-trips, displays —
  **never interprets**. Physics meaning belongs to the mesher script.
- UI offers per-DomainKind *suggestion lists* (editable combo):
  Fluid → `wall` (pre-filled), `inlet`, `outlet`, `symmetry`;
  Solid → nothing pre-filled, suggest `fixed`, `load`, `contact`.
- The untagged remainder convention (uncovered fluid boundary = wall,
  uncovered solid boundary = free) lives in the mesher, not here.
- `PatchTag` in `core/surface_provenance.py` shrinks to the UI suggestion
  list; the kernel-level default-WALL interpretation moves to the fluid-UI
  pre-fill.

---

## 5. JSON schema

```json
"boundary_regions": {
  "jet_inlet": {
    "id": 42,
    "owner": "flow_volume",
    "patch": "-X",
    "tag": "inlet",
    "cuts": [
      {"side": "inside", "ghost": {"kind": "cylinder", "center": [0,0,0], "radius": 0.2, ...}}
    ]
  },
  "upper_wall": {
    "id": 43,
    "owner": "flow_volume",
    "patch": "-X",
    "tag": "wall",
    "cuts": [
      {"side": "outside", "ghost": {"kind": "cylinder", ...}},
      {"side": "inside",  "ghost": {"kind": "box", ...}}
    ]
  }
}
```

**Legacy migration (on load, one-way):**

- `outside_direction: 0..5` → the owner's corresponding `patch_id`.
- `selector_id: "selector:<oid>"` → resolve the hidden node, inline its record
  as a one-item chain `[{side: selector_side, ghost: <record>}]`.
- interval selectors (`selector_start/end`, 2D curve params) → kept readable
  as a legacy `IntervalCut` chain entry (see §9 open items); no new ones are
  created by the UI.
- hidden `__boundary_selector_*` nodes and the fluid `selectors` list are
  dropped after conversion. Saving always writes the new format.

---

## 6. Meshing API (`core/meshing/api.py`)

```python
@dataclass(frozen=True)
class MeshableBoundaryRegion:
    name: str                     # domains["fluid"].boundary_regions["jet_inlet"]
    tag: str | None               # opaque; script decides meaning
    owner_object_id: int
    contains: PointsPredicate     # (N,3) points -> bool mask  — THE classifier (§2)
    owner_sdf: SDFCallable        # exact field of the generating surface
    selector_sdf: SDFCallable | None  # combined signed field of the cut chain
```

- `contains` is the **same function** the viewport uses — what you see
  highlighted is what the script gets.
- `owner_sdf` gives custom meshers the exact distance to *that* boundary
  (y⁺ layers, grading, refinement bands).
- `MeshableDomain.boundary_tags` (name, raw-callable tuples) is replaced by
  `boundary_regions` with name lookup mirroring `MeshableDomains`; a
  deprecation shim may keep `boundary_tags` one release.
- This closes problem 5 twice over: every region is callable (empty-chain
  ones included), and membership is scoped, not a bare knife field.

---

## 7. UI / viewport

**Boundary Region select (hover) — rebuilt on QRhi:**

- New `boundary_tool.py` beside `create_tool.py` (same extraction pattern):
  arm → hover ray-picks the committed surface (`_pick_sdf_surface` +
  provenance attribution) → highlight the region/patch under the cursor
  (overlay surface via `boundary_region_preview_node` → contoured thin shell)
  → click selects the existing region, or creates a whole-surface region if
  none exists there yet.
- **Generic fallback patch:** `_surface_patches_for_node`'s default branch
  returns one whole-surface patch instead of `()` — every leaf becomes
  hover-selectable immediately (fixes problem 4). Analytic per-face patches
  (box faces, cylinder wall/caps) remain as quick-picks where they exist.

**Boundary Cutter (one button):**

- Requires a selected region (or a selected owner surface → implicit
  whole-surface region). Opens the standard draw menu; any shape is legal.
- On commit: split replaces the parent with the two named children
  (`<parent> / cut N inside|outside` default names), selects both, warns if a
  child is empty. Undo restores the parent.
- Delete `_PLANAR_CUTTER_KINDS` / `_SURFACE_CUTTER_KINDS`, the two buttons,
  `_active_planar_cutter_kind` / `_active_surface_cutter_kind`, the
  segment→giant-polygon special case (`_add_planar_segment_cutter_region`)
  if the generic extrusion covers it, and `_mark_internal_boundary_selector`.

**Properties panel:** region shows owner, patch, tag combo (per-kind
suggestions), and the cut list (read-only lineage; "remove last cut" action).

---

## 8. Implementation phases (each lands green, in order)

**Phase 1 — core schema + classifier** (`core/boundary.py`, new
`core/boundary_region.py`)
- `BoundaryCut` dataclass; `BoundaryRegion` gains `cuts: tuple[BoundaryCut, ...]`,
  loses nothing yet (legacy fields kept, deprecated).
- `boundary_region_mask(root, region, points, *, tolerance=None)` — the §2
  classifier, scale-relative tolerance. Owner-active test implemented
  generically (operand field walk), reusing/refactoring
  `boundary_region_scope_mask`.
- Tests: partition exactness (two siblings of any cut partition the parent on
  sampled boundary points), chain conjunction, owner attribution on
  `Subtract` (obstacle owns the cut surface), scale-relative tolerance at
  mm/km.

**Phase 2 — document operations** (`core/scene.py`)
- `split_boundary_region(region, ghost_node) -> (inside, outside)`:
  serializes the ghost into both children's chains, **removes the parent**,
  never inserts the ghost into `objects`.
- Empty-child sampling warning. Undo via existing snapshot mechanism.
- Tests: nested splits compose; parent replaced; ghost absent from `objects`.

**Phase 3 — serialization + migration** (`core/serialization.py`)
- New record format (§5); loader accepts legacy records and converts
  (direction→patch, selector→one-item chain, inline hidden-node records);
  writer emits only the new format; hidden-selector machinery deleted after
  conversion.
- Tests: round-trip new format; load legacy `scene.json` fixture → regions
  classify identically pre/post migration.

**Phase 4 — meshing API** (`core/meshing/api.py`)
- `MeshableBoundaryRegion` + `boundary_regions` collection; `contains` wired
  to the Phase-1 classifier; `owner_sdf`/`selector_sdf` callables.
- Tests: a scripted "custom mesher" classifies face-centroid samples of the
  von Kármán scene: inlet disk vs wall ring vs obstacle wall; whole-face
  (empty-chain) regions are callable.

**Phase 5 — single Boundary Cutter in the UI** (`app/main_window.py`,
`renderers/qrhi/*`)
- One button/action wired to the generic ghost rule (3D as-is, <3D extruded);
  delete the two old buttons and their routing; children replace parent in
  the tree; naming + status messages.
- Tests: headless flow — select region, commit drawn shape, assert two
  children with correct chains and no ghost in `objects`.

**Phase 6 — hover/select boundary tool on QRhi**
- `boundary_tool.py` (viewport), generic fallback patch in
  `_surface_patches_for_node`, highlight overlay, click-to-select/create.
- Tests: pick attribution unit tests (ray → owner/patch) incl. a Pyramid
  (previously unpickable); headless arm/hover/click smoke.

**Phase 7 — cleanup + docs**
- Remove dead code (interval-selector creation paths, `PatchTag`
  interpretation, stale helpers); update `exact_sdf_decisions.md` with the
  settled identity; progress notes in `design_docs/progress/`.

---

## 9. Open items (deliberately deferred)

- **2D domain parity** — curve regions on 2D roots via 2D ghosts; legacy
  `IntervalCut` entries stay readable until then. New-UI creation is 3D-first.
- **Tag suggestion catalogue** — grows with real solver scripts (spec §12).
- **Region area/statistics** for the mesher (`area_estimate()`, sample point
  generators) — easy to add on top of the classifier when a script needs it.
- **DomainsInterface × regions** — interfaces are discovered structurally
  (§8); if a user wants to tag *part* of an interface, the cutter applies
  there too since interfaces are owner surfaces. No extra mechanism expected.
