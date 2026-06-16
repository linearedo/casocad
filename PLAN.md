# PLAN.md - casoCAD Product And Code Design

This document defines what casoCAD is, how the code should be organized, and
the order in which the product should be built. Binding contributor rules live
in `AGENTS.md`.

## 1. Project Objective

casoCAD is a Signed Distance Function based CAD application for creating
solver-ready 2D and 3D analysis cases.

The first target is CFD. A user should be able to:

1. create the fluid domain and obstacles
2. edit and combine SDF geometry
3. mark inlet, outlet, wall, symmetry, and other regions
4. enter meshing and simulation parameters
5. inspect the geometry and generated discretization
6. select a supported solver
7. generate a runnable solver case
8. save and reopen the editable project

Example:

```text
Pipe geometry
+ inlet: velocity 1 m/s
+ outlet: pressure 0 Pa
+ walls: no slip
+ lattice spacing: 0.01 m
+ solver: OpenLB
= generated OpenLB case directory
```

The primary solver targets are:

- OpenLB
- Palabos
- OpenFOAM
- SU2

OpenLB and Palabos use a regular lattice. OpenFOAM and SU2 require a surface
and volume mesh. They share the same SDF geometry and user-defined boundary
meaning, but they do not share the same meshing implementation.

Both 2D and 3D SDF scenes are valid analysis inputs. A 2D project must remain
authoritatively 2D; a solver adapter may generate required thin or one-cell
topology without requiring the user to extrude the CAD geometry.

The first product milestone is deliberately narrower:

> Create a 2D or 3D SDF CFD model in the GUI and generate one runnable OpenLB
> or Palabos case from it.

General FEA support is a later extension. It must not delay the first complete
CFD workflow.

## 2. Product Boundaries

casoCAD owns:

- editable SDF geometry
- analysis-region definition
- integrated lattice and mesh generation
- discretization preview and validation
- solver-specific case generation
- project persistence

casoCAD is not:

- a traditional boundary-representation CAD kernel
- a triangle-mesh-first geometry editor
- a CFD or FEA solver
- a universal file converter
- a replacement for every advanced option exposed by each solver

The generated case may expose a solver-specific advanced settings panel, but
the geometry and boundary definitions must remain independent of one solver.

## 3. Core Product Model

The product has four concepts:

### 3.1 Geometry

The SDF graph describes where the domain and objects are.

Examples:

- box
- cylinder
- sphere
- union
- intersection
- difference
- extrusion
- transform

### 3.2 Simulation Setup

The simulation setup describes what the geometry means for an analysis.

Examples:

- which 2D or 3D SDF is the fluid domain
- which region is an inlet
- which region is a wall
- inlet velocity
- outlet pressure
- fluid properties
- lattice or mesh resolution
- selected solver

This is normal typed Python state. It is not required to be a universal
intermediate file.

### 3.3 Discretization

The mesher converts the SDF geometry and region assignments into the data
needed by a solver family.

There are two different paths:

```text
OpenLB / Palabos -> 2D or 3D regular lattice
OpenFOAM / SU2   -> 2D planar mesh or 3D surface and volume mesh
```

### 3.4 Case Generation

A solver case generator combines:

- the simulation setup
- the generated lattice or mesh
- solver-specific numerical settings

It writes the files and directory structure expected by that solver.

## 4. System Architecture

```text
SceneDocument
    |
    +-- SDF geometry graph
    +-- stable object IDs
    +-- fluid-domain selection
    +-- named boundary and internal regions
    +-- simulation settings
    |
    +-----------------------+-------------------------+
    |                       |                         |
    v                       v                         v
CAD viewport            integrated mesher         project save/load
to_glsl()               to_numpy()                scene JSON
                            |
                  +---------+---------+
                  |                   |
                  v                   v
            lattice result        mesh result
                  |                   |
          +-------+-------+       +---+---+
          |               |       |       |
          v               v       v       v
       OpenLB          Palabos  OpenFOAM  SU2
       generator       generator generator generator
```

The SDF graph is authoritative. Rendering, meshing, and case generation must
not maintain independent geometry definitions.

There is no mandatory universal file between the mesher and case generators.
Python objects are the normal in-process interface. Arrow remains an optional
serialization format for the current lattice data.

## 5. Code Organization

Current code:

```text
app/
    main.py                 application entry point
    main_window.py          GUI workflow and worker coordination
    panels/                 scene, properties, mesher, export, logs
    viewport/               camera, Qt OpenGL widget, renderer, shaders

core/
    scene.py                editable document and graph operations
    boundary.py             boundary-region selection
    serialization.py        versioned project persistence
    sdf/                    SDF geometry kernel
    mesher/                 current lattice mesher
    io/                     Arrow reader and writer

scenes/                     example projects
tests/                      unit, workflow, and framebuffer tests
```

Planned additions should follow this structure:

```text
core/
    simulation/
        setup.py            CFD setup and physical parameters
        conditions.py       typed boundary and initial conditions
        validation.py       setup validation before meshing/export

    mesher/
        lattice/            regular-grid discretization
        volume/             future surface and volume meshing

    solvers/
        base.py             shared case-generator protocol and errors
        openlb/             OpenLB validation and case writer
        palabos/            Palabos validation and case writer
        openfoam/           future OpenFOAM case writer
        su2/                future SU2 case writer

    io/
        arrow_reader.py     optional lattice interchange
        arrow_writer.py
```

Do not create all planned directories before they are needed. Add each module
when implementing a complete vertical feature.

## 6. Geometry Kernel Design

### 6.1 SDF Contract

For every SDF:

```text
d(p) < 0  inside
d(p) = 0  boundary
d(p) > 0  outside
```

Every visible SDF object has:

- a stable nonzero `object_id`
- an editable `name`
- dimension `1`, `2`, or `3`
- a `kind`
- graph children where applicable
- `to_numpy()` for CPU evaluation
- `to_glsl()` for GPU visualization

The CPU and GPU implementations must describe the same field.

Bounding boxes are traversal aids only. They must never replace SDF evaluation
for geometric classification.

### 6.2 Graph Rules

The SDF graph is acyclic. Boolean operations require equal dimensions:

```text
Union(A, B)          min(a, b)
Intersection(A, B)   max(a, b)
Difference(A, B)     max(a, -b)
```

A boolean result is a first-class SDF and may be used in later operations.
The final result is rendered as one object; boolean operands remain graph
definitions.

### 6.3 Geometry Evaluation Paths

CAD rendering:

- calls `SDFNode.to_glsl()`
- runs in ModernGL on the main thread
- uses `float32`
- is visualization-only

Meshing:

- calls `SDFNode.to_numpy()`
- runs with NumPy in a worker
- uses `float64`
- is the authoritative discrete geometry evaluation

The renderer must not define solver geometry from pixels. The mesher must not
depend on OpenGL.

## 7. Simulation Setup Design

The current `FluidDomain` and region/tag objects are the starting point for the
simulation setup.

The setup should grow incrementally to contain:

- one active 2D or 3D fluid-domain SDF
- named boundary regions
- optional internal regions
- typed boundary conditions
- fluid properties
- discretization settings
- selected solver and solver-specific settings

Initial CFD boundary-condition types:

- no-slip wall
- velocity inlet
- pressure outlet
- symmetry
- periodic pair

Conditions refer to stable region IDs. They do not inspect GUI handles and do
not alter SDF geometry.

Generic condition meaning belongs in `core/simulation/`. Solver-specific names
and syntax belong in `core/solvers/`.

Example in-memory intent:

```text
region 10: velocity inlet, velocity = (1, 0, 0) m/s
region 11: pressure outlet, pressure = 0 Pa
region 12: no-slip wall
```

The OpenLB, Palabos, OpenFOAM, and SU2 generators may translate that intent
differently. Unsupported mappings must produce explicit validation errors.

## 8. Integrated Meshing

### 8.1 Lattice Mesher

The existing `LatticeMesher` is the first production meshing path.

Its mathematical contract remains:

- uniform spacing `dx`
- retain a candidate node when `root_sdf <= 0`
- classify a retained node as boundary when at least one axis-adjacent
  neighbor has `root_sdf > 0`: four neighbors in 2D and six in 3D
- preserve boundary direction, ownership, and region identity
- process the domain in bounded chunks
- never materialize the full traversal envelope at once

The lattice result must record its dimension and provide the information
required by both OpenLB and Palabos generators. It should not contain OpenLB-
or Palabos-specific syntax.

The current Arrow output may remain available for:

- debugging
- inspection
- interoperability
- downstream experiments

Case generators should be able to consume the mesher result directly without
writing and rereading Arrow.

### 8.2 Planar, Surface, And Volume Mesher

OpenFOAM and SU2 require explicit mesh topology. The mesher must support
planar 2D domains as well as 3D domains:

- vertices
- edges or faces
- cells or elements
- connectivity
- named boundary patches

This is a separate future meshing path derived from the same SDF scene and
region assignments.

The design must preserve:

- the SDF as authoritative geometry
- boundary-region identity during surface extraction
- valid cell topology
- deterministic case generation

Do not implement OpenFOAM or SU2 export by pretending the current lattice is an
unstructured volume mesh.

If a solver represents a 2D case as a thin 3D mesh, that conversion belongs in
its meshing or case-generation adapter. It must preserve the original 2D SDF,
region IDs, physical scale, and boundary-condition meaning.

## 9. Solver Case Generators

A case generator is responsible for one solver.

Conceptual interface:

```python
class CaseGenerator(Protocol):
    def validate(
        self,
        setup: SimulationSetup,
        discretization: DiscretizationResult,
    ) -> list[ValidationIssue]: ...

    def write_case(
        self,
        setup: SimulationSetup,
        discretization: DiscretizationResult,
        output_directory: Path,
    ) -> CaseResult: ...
```

This protocol is a design direction, not a requirement to introduce speculative
base classes before the first generator exists.

Each generator owns:

- solver file syntax
- directory layout
- solver-specific boundary mapping
- numerical-scheme settings
- generated source code when required
- compatibility checks

Each generator must reject:

- unsupported condition types
- missing required physical values
- incompatible discretization
- invalid region assignments
- unsupported dimensionality

Generated output should include provenance:

- casoCAD version
- project or scene identifier
- geometry/setup version when available
- meshing parameters
- selected solver profile

## 10. GUI Workflow

The GUI should expose the product workflow directly:

1. geometry
2. regions
3. conditions
4. meshing
5. solver
6. generate case

The user should be able to inspect:

- the final SDF result
- selected boundary regions
- lattice nodes or mesh cells
- invalid or unassigned boundaries
- meshing statistics
- case-generation validation errors

Long-running meshing and case generation run in workers. Qt widgets and
ModernGL remain on the main thread.

## 11. Persistence

Project persistence stores editable state, not only generated solver files.

The saved project should eventually contain:

- SDF graph and stable object IDs
- top-level scene objects
- active fluid domain
- boundary and internal regions
- boundary-condition assignments
- physical properties
- meshing settings
- selected solver profile

Scene JSON is versioned. Any incompatible shape change requires a version
increment and migration or rejection tests.

Generated lattices, meshes, Arrow files, and solver cases are derived artifacts
and do not replace the editable project.

## 12. Current State

Implemented:

- SDF primitives, transforms, booleans, and selected 2D-to-3D operations
- placed 1D interval SDFs for 2D boundary regions
- `SceneDocument` with stable object identities
- `FluidDomain`
- placed 1D and 2D tags plus 3D owner-based `BoundaryRegion`
- GPU raymarched CAD viewport
- dimension-aware chunked 2D and 3D uniform lattice mesher
- lattice preview
- Arrow lattice export and round-trip tests
- versioned scene serialization
- GUI geometry, meshing, and export workflows

Not yet implemented:

- 2D and 3D solver-case export
- typed CFD boundary conditions with physical values
- complete simulation settings
- solver selection and validation
- OpenLB or Palabos case generator
- direct mesher-result-to-case workflow
- planar, surface, and volume meshing
- OpenFOAM or SU2 case generator
- general FEA setup

## 13. Development Roadmap

### Phase 1 - Complete The CFD Setup

1. define typed wall, velocity-inlet, pressure-outlet, symmetry, and periodic
   conditions
2. assign conditions to existing stable boundary regions
3. add fluid properties and basic simulation parameters
4. persist the setup in the project file
5. validate missing and conflicting assignments in the GUI

Exit criterion:

- a saved casoCAD project fully describes one basic incompressible CFD setup

### Phase 2 - Stabilize The Lattice Result

1. separate the reusable in-memory lattice result from preview-only data
2. support dimension-aware 2D and 3D lattice generation
3. expose boundary ownership, directions, and condition-region IDs
4. keep chunked bounded-memory generation
5. retain Arrow as optional serialization
6. add deterministic 2D and 3D mesher and round-trip tests

Exit criterion:

- 2D and 3D lattice results contain everything needed by a case generator
  without reading GUI state or an Arrow file

### Phase 3 - Generate The First Runnable Case

1. choose OpenLB or Palabos as the first backend
2. implement setup validation for that backend
3. map casoCAD regions and conditions to solver boundary setup
4. generate the complete case directory or source project
5. compile or load the generated case in an automated compatibility test
6. run at least one deterministic 2D and one 3D solver step

Exit criterion:

- saved 2D and 3D casoCAD projects generate and execute verified solver cases

### Phase 4 - Add The Second Lattice Solver

1. implement the second OpenLB/Palabos generator
2. reuse the same CFD setup and lattice result
3. identify and correct assumptions accidentally tied to the first backend

Exit criterion:

- 2D and 3D casoCAD projects can generate valid cases for both lattice solvers

### Phase 5 - Add Mesh-Based CFD

1. design 2D boundary and 3D surface extraction with region preservation
2. implement and validate planar and volume-mesh generation
3. add mesh inspection to the GUI
4. implement an OpenFOAM or SU2 generator
5. add the second mesh-based solver

Exit criterion:

- the same SDF authoring workflow supports verified 2D and 3D mesh-based CFD
  cases

### Phase 6 - Extend Toward General FEA

1. define solid domains, materials, loads, and constraints
2. determine supported element and mesh requirements
3. implement one complete FEA backend before expanding further

Exit criterion:

- casoCAD supports one verified non-CFD analysis workflow without compromising
  the CFD architecture

## 14. Required Testing

Geometry changes require:

- NumPy formula tests
- GLSL/NumPy parity checks where practical
- graph and serialization regression tests

Mesher changes require:

- retention and boundary classification tests
- ownership and region-assignment tests
- chunk-partition independence tests
- bounded-memory behavior
- preview and serialization tests where relevant

Case-generator changes require:

- setup validation tests
- exact generated-file structure tests
- boundary mapping tests
- compatibility tests against the target solver version
- at least one minimal end-to-end case

GUI changes require:

- focused widget or workflow tests
- the native GUI smoke test
- framebuffer tests for rendering behavior

## 15. Definition Of Done

A feature is complete only when:

- it works through the intended GUI workflow
- geometry and simulation meaning remain separate
- meshing derives from the authoritative SDF
- generated solver output is validated
- project persistence retains the editable setup
- focused and full tests pass
- documentation matches the implementation

The central success criterion is simple:

> A user creates one 2D or 3D SDF model, assigns CFD meaning to it, and casoCAD
> reliably generates solver cases from that same model.
