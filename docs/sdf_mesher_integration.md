# SDF Mesher Integration

This note defines the minimal API boundary between casoCAD geometry and external
mesher algorithms.

The persistent input is a normal casoCAD `scene.json`. The mesher-facing object
is rebuilt at runtime from that file; it is not saved as an intermediate file.

## MeshableDomain

```python
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from core.sdf.base import BoundingBox3D


@dataclass(frozen=True)
class MeshableDomain:
    name: str
    kind: tuple[str, ...]
    dimension: int
    bounds: BoundingBox3D
    domain_sdf: Callable[[np.ndarray], np.ndarray]
    boundary_tags: tuple[
        tuple[str, Callable[[np.ndarray], np.ndarray]],
        ...
    ] = ()
```

Conventions:

- callable input points have shape `(N, 3)` in world coordinates
- callable return values have shape `(N,)`
- `domain_sdf(points)` returns signed distance for the meshable domain
- each boundary tag is a pair `(tag_name, tag_sdf)`
- `tag_sdf(points)` returns a signed-distance/query field for that named
  boundary region

The API intentionally does not define tolerance, priority, overlap handling,
gradient queries, or tag interpretation. Those choices belong to each mesher
algorithm.

## Runtime Loading

The intended external entry point is:

```python
domains = load_meshable_domains("scene.json")
mesh = my_mesher(domains["fluid"], dx=0.01)
```

`domains` supports lookup by exact domain name, for example
`domains["water_domain"]`, and by unique domain kind, for example
`domains["fluid"]`.

`load_meshable_domains()` should:

1. load `scene.json` with the existing scene deserializer
2. reconstruct the SDF graph
3. select exported/meshable domains
4. wrap each domain root as `domain_sdf`
5. wrap each SDF boundary region as a named `boundary_tag`

Conceptually:

```python
def _sdf_callable(node):
    def sdf(points: np.ndarray) -> np.ndarray:
        return node.to_numpy(points[:, 0], points[:, 1], points[:, 2])

    return sdf
```

The saved file remains the source of truth:

```text
scene.json -> SDF objects -> MeshableDomain -> mesher algorithm
```

`MeshableDomain` is only a temporary runtime adapter. It contains Python
callables, so it should not be serialized directly.

## Meshing Workspace

Meshing should be framed as a separate workspace from CAD editing.

```text
CAD Workspace
  edits scene geometry
  saves scene.json

Meshing Workspace
  imports scene.json
  builds MeshableDomain objects
  lets user run mesher scripts
  visualizes mesher output
```

This keeps meshing fully separated from the live CAD document. The mesher works
from a saved scene file, not from mutable GUI state.

Top-level menu direction:

```text
File | Edit | Domains | Meshing
```

Initial Meshing menu actions:

```text
Meshing -> Open Meshing Workspace...
Meshing -> Import scene.json...
```

Inside the Meshing Workspace:

```text
left: imported domains list
center: meshing visualization viewport
right/bottom: script editor + run button + output log
```

The script editor can expose a small predefined environment:

```python
# provided by casoCAD
domains  # MeshableDomains
np       # numpy
```

User scripts should operate on `domains`, generate mesh chunks, and emit or
write mesh artifact batches for visualization.

## Mesher Author Contract

A custom mesher only needs to accept a `MeshableDomain`:

```python
def my_mesher(domain: MeshableDomain, dx: float):
    points = make_points(domain.bounds, dx)
    distance = domain.domain_sdf(points)

    for tag_name, tag_sdf in domain.boundary_tags:
        tag_values = tag_sdf(points)
        ...
```

The mesher decides how to use `distance` and `tag_values`, including:

- inside/outside thresholds
- boundary tolerance
- tag matching rules
- conflict handling
- grid, lattice, octree, marching, or other algorithm choices

This keeps casoCAD geometry independent from mesher implementation details while
still exposing the SDF fields a mesher needs to query.

## Mesh Artifact Schema

Mesher output should be written as a streamable Arrow artifact, not returned as
one large in-memory NumPy result. This keeps large mesh generation bounded in
RAM while still giving the viewport and exporters a common format to consume.

Minimal Arrow schema:

```text
element_type: string
vertices: list<fixed_size_list<float64, 3>>
tag_name: string
```

Meanings:

- `element_type` describes the emitted element shape, for example `point`,
  `segment`, `triangle`, `quad`, `polygon`, `tetra`, or `hexa`
- `vertices` stores the element vertices as world-coordinate `(x, y, z)` points
- `tag_name` stores the semantic label, for example `fluid_internal`, `wall`,
  `inlet`, or `outlet`

Example rows:

```text
point     [[x, y, z]]                                      fluid_internal
segment   [[x0, y0, z0], [x1, y1, z1]]                     inlet
triangle  [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2]]        wall
quad      [[x0, y0, z0], [x1, y1, z1], [x2, y2, z2],
           [x3, y3, z3]]                                   outlet
polygon   [[x0, y0, z0], [x1, y1, z1], ...]                 wall
```

The intended pipeline is:

```text
scene.json -> MeshableDomain -> mesher chunks -> Arrow artifact -> viewport/export
```

Arrow is the mesh artifact format. `scene.json` remains the geometry source of
truth, and `MeshableDomain` remains a temporary runtime adapter.

## Mesh Visualization Strategy

The viewer should not treat every mesh dimension the same.

For 2D meshes, displaying the full mesh is usually practical and useful:

```text
2D visualization
  show full mesh by default
  color by tag_name
```

For 3D meshes, displaying the full volume mesh is usually both too large and
not useful to inspect. The default 3D view should focus on what a user can
understand:

```text
3D visualization
  show boundary/surface mesh by default
  show zero or more local volume probes on demand
```

A local probe is a viewer-side query such as:

```python
local_mesh_near(point=p, radius=r)
```

The first inclusion rule can be simple:

```text
show an element when any vertex lies inside sphere(point, radius)
```

Multiple probes should be additive:

```text
visible volume elements =
  elements near probe_1
  union elements near probe_2
  union ...
```

Overlapping probes should deduplicate elements in the viewer. This gives a
useful 3D inspection model:

- boundary mesh provides global shape and boundary tags
- local probes expose internal cells only where the user asks to inspect them
- full 3D volume mesh is not loaded or displayed by default

Future viewer state can model this explicitly:

```python
@dataclass(frozen=True)
class LocalMeshProbe:
    name: str
    point: tuple[float, float, float]
    radius: float
    enabled: bool = True
```

This does not require the artifact schema to become complex immediately. The
viewer can initially scan Arrow batches and keep only rows matching active
probes. If large artifacts require faster lookup later, the same model can be
accelerated with chunk bounds or a spatial index.

## Arrow And GPU Upload

Arrow's zero-copy features are useful, but the boundary must be stated
precisely:

```text
Arrow mmap / Arrow buffers
  -> CPU memory views with little or no CPU copying
  -> QRhi upload to GPU buffer
```

Arrow can reduce CPU-side copies and Python object conversion, especially when
data is stored in contiguous primitive buffers. It does not remove the required
CPU-to-GPU upload step for QRhi buffers.

The current semantic schema:

```text
vertices: list<fixed_size_list<float64, 3>>
```

is flexible and solver/interchange friendly, but it is not the ideal GPU upload
layout. A future optional render cache can be Arrow-based and render-ready:

```text
position: fixed_size_list<float32, 3>
color: fixed_size_list<float32, 3>
element_id: int64
role: string or dictionary
```

That future cache would allow:

```text
Arrow mmap -> contiguous CPU float32 view -> QRhi GPU upload
```

The semantic mesh artifact should remain the source of mesh meaning. A
render-ready cache, if added, should be an optimization for visualization, not
the authoritative solver artifact.
