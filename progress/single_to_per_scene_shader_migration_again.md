# Migration (again): single unified shader → per-scene specialized shaders

Living progress log. Rationale + full context: see
`progress/per_scene_specialized_shader_pivot.md`.

**Why again:** the path has changed before (codegen → interpreter VM). The unified
FULL interpreter shader is now at the NVIDIA GL **link limit** — it cannot take the
per-pixel work heavy scenes need (intersection ≈ 1 FPS, grouped-cull won't link).
We specialize the shader to the node types / features each scene actually uses, so
each baked shader is small (links, faster) and only re-bakes when the scene's
type-set changes.

## Goal / done-when
- [x] Intersection scene renders correctly AND fast on the real RTX 3050
      (`tools/fps_bench.py --scene intersection` links + good FPS + screenshot).
- [x] Union / difference unchanged (no regression).
- [x] Adding/moving objects of an already-present type does NOT recompile.
- [x] Full test suite green (162 passed).

## Checklist
- [x] Decision recorded; pivot committed (checkpoint).
- [x] CPU grouped-cull flattener (DNF groups) + parity tests.
- [x] Vectorised cull-grid binning (`_bin_entries`).
- [x] Grouped `irCellEval` (single call site, 4 scalar groups).
- [x] Real-GPU FPS benchmark tool (`tools/fps_bench.py`).
- [x] Scene → feature signature (`features_for_render_ir`, content-driven).
- [x] Renderer: bake/cache frag shader per feature-set; swap + rebuild pipeline
      only on change; UBO + buffers reused (UBO layout is variant-independent).
- [x] GPU validation: intersection links + renders + fast.
- [x] Removed dead GPU-capability-tier code (`LEAN_FEATURES`, `scene_fits_tier`,
      "weak driver" framing) — specialization is content-driven, NOT per-GPU.
- [ ] (optional) gate individual core primitive cases per scene for more headroom.
- [ ] Final commit.

## Result (real RTX 3050, orbiting)
| scene | nodes | before | after |
|---|---|---|---|
| intersection | 75 | 0.9 FPS (or link FAIL) | **35 FPS** |
| intersection | 151 | ~1 FPS | **27 FPS** |
| union (cullable) | 75 | 34 FPS | 43 FPS |

Screenshot confirms correct geometry (colored intersection blob, grid/axes).

## Log
- Confirmed on real GPU: grouped-cull added to the UNIFIED shader →
  `pipeline create() FAILED` (C5025), nothing renders. Root cause = link limit,
  not the math. User directed the pivot to per-scene specialized shaders.
- Found: optional feature chunks declare NO uniforms → UBO layout is identical
  across variants → specialization only swaps the frag shader + pipeline.
- Implemented per-scene specialization: `_frag_for_features` (cached bake),
  `_activate_features` (swap on change), `set_scene` records the scene's feature
  set. A sphere scene bakes core+cull (small) → LINKS.
- Validated: union 43 FPS (specialized shader links + renders), intersection
  75→35 FPS, 151→27 FPS. Screenshots correct. The intersection-lag problem is
  solved. First-run cold compile is a few seconds (driver-cached after).
- Cleaned the capability-tier remnants per user's directive (all features ship
  in one codebase; specialization is by scene content, never by GPU).
- Follow-on investigation (codegen vs VM compile times, Vulkan, the thin
  data-driven codegen conclusion) recorded in
  `progress/codegen_vs_vm_compile_investigation.md` — recommends evolving toward
  thin per-structure codegen (typed DNF loops) over the generic bytecode VM.
