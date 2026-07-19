# casoCAD

SDF-based CAD for solver-ready analysis cases (CFD first), written in Rust and
targeting **native desktop and the web browser** from a single codebase
(egui + wgpu).

This Rust workspace is the full port of the original Python casoCAD, now the
sole codebase. The final Python snapshot is archived at the `python-final` tag
(branch `final-python`, commit `6185ec6`).

Principles:

- `#![forbid(unsafe_code)]` in every crate (enforced via workspace lints).
- Minimal dependencies: the kernel has **zero**; later crates add only what is
  irreplaceable (`wgpu`, `egui`).
- f64 for all kernel/analysis math; f32 only at GPU upload boundaries.
- SDF terminology: boolean operations are *SDF operators* (never "CSG");
  "meshing" is reserved for FEA/CFD — the viewport makes *surfaces*.

See `DESIGN.md` for the architecture and product scope, and
`docs/mesher_script_api.md` for the mesher scripting reference (the Rhai
API the Meshing panel exposes).

## Crates

- `kernel/` — the exact signed-distance-field geometry kernel: primitives
  (1D/2D/3D), SDF operators, exact transforms and generators, the exactness
  role system ("safe geometry compiler", spec
  `design_docs/exact_signed_distance_field_cfd_migration_v2.md`), scene
  document + JSON serialization, meshing API.
- `surfaces/` — display-surface builders (exact boolean clipping + dual
  contouring, 2D profiles, 1D wires).
- `render/` — wgpu renderer (WGSL shaders, native + WebGPU/WebGL).
- `meshing/` — FEA/CFD meshing workspace.
- `app/` — egui application (viewport, tools, panels); native and wasm entry
  points. `web/` holds the browser shell.

## Build & test

```bash
cargo test --workspace          # includes the Python-parity golden suites
cargo clippy --workspace --all-targets
cargo run -p caso-app           # native app
cargo build --target wasm32-unknown-unknown -p caso-app   # web build check
```

