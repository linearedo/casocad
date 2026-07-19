# Mesher script API reference

Mesher scripts run in [Rhai](https://rhai.rs) from the Meshing panel. Every
script receives two globals:

- `domains` — the document's declared Domains, exposed through the exact-SDF
  meshing API;
- `mesh` — the MeshIR builder the script writes points, cells, faces, zones
  and tags into.

Plus one free function, `sizing(domain, spec)`.

Conventions used by every call:

- Coordinates are world x, y, z (three separate numbers in, `[x, y, z]`
  arrays out).
- Lookups are **keyed by name**; a miss is an error that lists what is
  available. Nothing is positional.
- Failures are script errors (catch with `try { … } catch (error) { … }`);
  no call returns a silently wrong value.

Exactness rules every script can rely on (see
`design_docs/meshing_toolkit.md` §2):

- `d.sdf` is an **exact distance only inside** the domain (negative
  values). Positive values are sign-correct ("outside") but are NOT
  distances — never use them as lengths.
- `boundary_projection` refuses to start from outside the domain for exactly
  that reason, and reports failure honestly on a seam (a point equally far
  from two walls) instead of returning a bad point.
- Every point returned by `boundary_marching_sample` is an exact boundary
  vertex (on the true zero set), never an interpolated chord point.

---

## `domains` — the document's Domains

| call | returns | notes |
| --- | --- | --- |
| `domains.names()` | array of strings | domain names, document order |
| `domains.get(name)` | Domain | by NAME only; error lists the names |
| `domains.len()` | int | number of domains |
| `domains.interface(a, b)` | DomainsInterface | `a`, `b` are two names or two Domains, order-independent; error lists available pairs |
| `domains.interfaces()` | array of DomainsInterface | for iteration |

There is no lookup by kind. To find, say, the fluid domain:

```rhai
for name in domains.names() {
    let d = domains.get(name);
    if d.kind == "fluid" { /* … */ }
}
```

## Domain

Properties: `.name` (string), `.kind` (`"fluid"` or `"solid"`),
`.dimension` (2 or 3).

| call | returns | notes |
| --- | --- | --- |
| `d.bounds()` | map | `x_min, x_max, y_min, y_max, z_min, z_max` |
| `d.sdf(x, y, z)` | float | negative inside (exact), zero on the wall |
| `d.boundary_normal(x, y, z)` | `[x, y, z]` | outward unit normal of the boundary at/near the point |
| `d.boundary_projection(x, y, z)` | `[x, y, z]` | nearest boundary point; **errors** if the start is outside the domain or on a seam |
| `d.boundary_curvature(x, y, z)` | float | boundary curvature at a projected point (2D: in-plane κ; 3D: mean curvature); near creases read large values as "refine here" |
| `d.classify(x, y, z)` | map | `on_boundary` (bool), `owner` (object id), `region` (region name or `()` when untagged) |
| `d.boundaries()` | array of strings | the names `boundary_marching_sample` accepts (2D domains) |
| `d.boundary_marching_sample(name, npoints)` | array of `[x, y, z]` | see below (2D domains) |
| `d.mesh_space()` | MeshSpace | the 2D domain's plane chart; errors on 3D domains |
| `d.regions()` | array of BoundaryRegion | declared boundary regions |
| `d.region(name)` | BoundaryRegion | by name; error lists the names |

### `boundary_marching_sample(name, npoints)`

Marches along one named boundary of a 2D domain and returns `npoints`
ordered points, all exact boundary vertices, at approximately even
arc-length spacing. Accepted names (also returned by `d.boundaries()`):

- `"outer"` — the domain's outer boundary loop;
- a boundary-region name — the tagged piece (`"inlet"`, `"skin"`, …);
- a scene object's name — the boundary contributed by that object (a
  subtracted circle's hole, a rectangle's edges).

Closed boundaries: `npoints` points, head NOT repeated, ordered with the
material on the left (outer loops counter-clockwise, holes clockwise).
Open pieces (for example one tagged edge): `npoints` points including both
endpoints. Errors: unknown names (listing the available ones), a name whose
pieces are not one connected curve, `npoints < 2` (`< 3` for closed loops),
3D domains.

## BoundaryRegion

Properties: `.name`, `.tag` (the physics tag string, may be empty).

| call | returns | notes |
| --- | --- | --- |
| `r.contains(x, y, z)` | bool | is this boundary point in the region |
| `r.owner_sdf(x, y, z)` | float | the owning leaf primitive's field — exact everywhere (both signs) |
| `r.normal(x, y, z)` | `[x, y, z]` | outward normal of the domain boundary here |
| `r.project_to_owner(x, y, z)` | `[x, y, z]` | projection onto the owning primitive's surface (exact from both sides); errors on non-convergence |

## DomainsInterface

The shared wall between two directly nested domains — the same exact node
the outer domain was cut with. Properties: `.domain_a` (outer name),
`.domain_b` (inner name), `.owner` (object id of the shared surface).

| call | returns | notes |
| --- | --- | --- |
| `itf.sdf(x, y, z)` | float | the shared surface's field |
| `itf.contains(x, y, z)` | bool | on the shared wall (within band) and on both domains' boundaries |
| `itf.project(x, y, z)` | `[x, y, z]` | projects onto the wall through whichever domain's interior contains the start; **errors** if the start is inside neither |

## MeshSpace (2D domains)

The plane chart of a 2D domain: coordinates `(a, b)` on the drawing plane.

| call | returns | notes |
| --- | --- | --- |
| `space.bounds()` | map | `a_min, a_max, b_min, b_max` |
| `space.point(a, b)` | `[x, y, z]` | chart → world |
| `space.sdf(a, b)` | float | the domain field at the chart point |

## `sizing(domain, spec)` → SizingField

`spec` is a map; unknown keys are errors. Keys (all optional):
`background` (default diagonal/20), `min_size` (default diagonal × 1e-4),
`gradation` (default 0.3), `curvature_factor` (opt-in), `bands` — an array
of `#{ region: "name", distance: …, size: … }` (unknown region names error,
listing the available ones).

| call | returns |
| --- | --- |
| `s.size_at(x, y, z)` | float — target element size at the point |

## `mesh` — the MeshIR builder

| call | returns | notes |
| --- | --- | --- |
| `mesh.point(x, y, z)` | point id | ids are positive ints |
| `mesh.zone(name, kind)` | zone id | e.g. `mesh.zone("water", "fluid")` |
| `mesh.tag(name, kind)` | tag id | e.g. `mesh.tag("inlet", "inlet")`, `mesh.tag("hull", "wall")` |
| `mesh.cell(type, [point ids], zone_id)` | cell id | types: `"tri3"`, `"quad4"`, `"tet4"`, `"hex8"`, … (MeshIR element types) |
| `mesh.face(type, [point ids])` | face id | explicit face (for `cell_with_faces`) |
| `mesh.cell_with_faces(type, [point ids], [face ids], zone_id)` | cell id | cell with explicit faces |
| `mesh.tag_edge([a, b], tag_id)` | — | tag the edge between two existing points |
| `mesh.tag_face([point ids], tag_id)` | — | tag the face spanned by existing points |
| `mesh.attribute(kind, id, key, json)` | — | attach a JSON attribute to `"point"`, `"cell"`, `"zone"`, … |

Referencing a point id that does not exist is an error at the call site —
the builder never invents topology.

---

The default script in the Meshing panel (a conforming grid mesher over
every domain) exercises most of this API and is kept in sync with it —
use it as a working example.
