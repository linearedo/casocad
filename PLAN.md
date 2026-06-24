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

### 3.3 CAD Boundary Patch Selection

CAD boundary selection is an SDF-native patch layer derived from the scene
graph, not from lattice classification. It describes selectable parts of the
final fluid boundary without changing the SDF solid geometry:

- 3D domains expose surface patches such as box faces, cylinder side walls,
  caps, and boolean cut surfaces
- 2D domains expose curve patches such as rectangle edges and stable curved
  profile boundaries
- 1D construction objects can act as boundary selectors or split curves over
  selected patches

Boundary patch IDs and selector IDs are CFD metadata over the final boundary.
The viewport may use them for CAD selection and highlighting, and meshers may
consume them later, but `to_numpy()` remains the authoritative geometry path.
Mesher classifier internals must not be the conceptual source of CAD boundary
picking.

# Read the Design Docs under docs/ folder to remain in sync with the update
