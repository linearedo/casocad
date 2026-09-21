# casoCAD

SDF-based CAD for solver-ready analysis cases, written in Rust and
targeting **native desktop and the web browser** from a single codebase
(egui + wgpu).

## Scope

This project started from a simple question: **can a CAD system be designed with the mesher in mind from the very beginning?**

I wanted to explore whether an **SDF-based CAD** could provide geometry that is inherently more favorable for numerical meshing and simulation.

SDFs have useful properties, but some operations, especially certain boolean constructions, can distort the distance field. So I asked: **what if the CAD could prevent constructions that break those properties?**

This is the idea behind **domain validation** in casoCAD. A domain, such as a fluid domain, is represented as an SDF object and its construction history is tracked to determine whether its **internal distance field remains exact**.

The goal is therefore not to represent everything, but to deliberately restrict the modeling language to produce mesher-friendly, solver-ready geometry.

This is a personal exploration of where SDF exactness can be preserved, and how those guarantees can be enforced directly at the CAD level.

🌐 **Website:** [casocad.com](https://casocad.com)

## ⚠️ AI-assisted development:
This project is heavily coded with the assistance of AI. The architecture, concepts, design decisions, and validation remain my responsibility, but a significant portion of the implementation has been generated or refined using AI tools.

## Design
Principles:

- Unsafe code is denied workspace-wide.
- Minimal dependencies
- f64 for all kernel/analysis math; f32 only at GPU upload boundaries.

See [DESIGN.md](DESIGN.md) for the architecture and product scope,
[Console Draw scripting](docs/console_draw_api.md) for transactional CAD
editing, and the [Arrow mesh producer contract](docs/casomesh_arrow_v3.md)
for external mesh interoperability.

## Crates

- `kernel/` — the exact signed-distance-field geometry kernel: primitives
  (1D/2D/3D), SDF operators, exact transforms and generators, the exactness
  role system ("safe geometry compiler", spec
  `design_docs/exact_signed_distance_field_cfd_migration_v2.md`), scene
  document + JSON serialization, meshing API.
- `surfaces/` — display-surface builders (exact boolean clipping + dual
  contouring, 2D profiles, 1D wires).
- `render/` — wgpu renderer (WGSL shaders, native + WebGPU/WebGL).
- `meshing/` — Arrow-native out-of-core mesh storage, queries, quality,
  camera-driven LOD, and 2D DistMesh generation with quad boundary layers.
- `app/` — egui application (viewport, tools, panels); native and wasm entry
  points. `web/` holds the browser shell.

For 2D boundary layers, `hwall_n`, growth, and the derived normal rows are
hard geometry controls. `hwall_t` is a soft tangential target: the mesher may
adjust station count and position when that improves strip validity or cell
quality.

## Build & Run

```bash
cargo run --release -p caso-app           # native app
```
