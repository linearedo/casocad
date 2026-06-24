# Codegen vs interpreter-VM: compile-time investigation (real RTX 3050)

Question raised by the user: is the interpreter VM actually needed, or does
per-scene codegen (Shadertoy-style) compile fast enough to recompile on edit?
All numbers below are **real driver compile** (`pipeline.create()` link step) on
the NVIDIA RTX 3050, measured with `tools/fps_bench.py`-style harnesses
(scratchpad: `codegen_*_bench.py`, `loop_bench.py`).

## Why the VM exists (plain terms)
Two ways to draw the scene on the GPU:
- **Codegen** — emit a shader that *is* this exact scene (like Shadertoy). Small,
  fast to run. But every edit rewrites + **recompiles** it → freeze on edit.
- **Interpreter VM** (current) — one general shader that reads the scene as data
  (SSBOs). Never recompiles on edit (just upload data). But it's a big, slow-to-
  compile, hard-to-optimize shader, and it hit the NVIDIA **link limit** (C5025).

## Measurements

### 1. Unrolled codegen (one giant `map()` with N `min`s) — cold compile
| primitives | OpenGL | Vulkan |
|---|---|---|
| 100 | 160 ms | 64 ms |
| 300 | 850 ms | 180 ms |
| 500 | 2.4 s | 500 ms |
| 1000 | 8.9 s | 1.16 s |
| 2000 | 30 s | 4.2 s |
| 5000 | (minutes) | 30.6 s |

- Both backends are **superlinear (~quadratic)** in N (my earlier "linear" claim
  was wrong). Restructuring the source (chunked helper fns, balanced reduction
  tree) did **NOT** help — the driver inlines anyway (measured).
- **Vulkan compiles codegen 5–8× faster than OpenGL** and pushes the wall out
  ~2.5× (Vulkan@5000 ≈ OpenGL@2000). IMPORTANT: the old "Vulkan 456s, use OpenGL"
  rule is **VM-specific** (the VM's dynamic loops/switch/stack); for straight-line
  codegen Vulkan is the BEST backend. Coupling: VM→OpenGL, codegen→Vulkan.
- The NVIDIA driver disk-caches compiled shaders (revisits ~instant), but each
  edit changes the source → cache miss.

### 2. Data-driven LOOP shader (loop over a primitive array) — cold compile
| approach | compile @ 5000 |
|---|---|
| unrolled codegen | 30 s |
| **loop over data array** | **~30 ms, CONSTANT for any N** |

A shader that loops over an instance array is the **same size** for 100 or
5,000,000 primitives → ~30 ms compile regardless. The add+sub DNF loop form
(additive min-loop + subtractive max-loop) = 32 ms.

## Conclusion — the sweet spot

Three points on a spectrum:
1. **Unrolled codegen** — compile ~quadratic in N → blows up at scale.
2. **Full bytecode VM** (current) — compile O(1), but ONE all-types giant shader
   → link limit, complex (value stack, opcode dispatch, type switch).
3. **THIN data-driven codegen** — emit a small scene-specialized shader = loops
   over TYPED instance arrays, structured by the boolean DNF (add min-loop, sub
   max-loop, a loop per intersection group), parameters in the data buffers.

**Option 3 wins:** ~30 ms compile at any scale, NO link limit (specialized to the
types/ops the scene uses → tiny), far simpler than the bytecode VM, and **moves
don't recompile** (positions live in the buffers; only a structural change
re-bakes, ~30 ms). Target backend: Vulkan.

This is the natural evolution of what already shipped (per-scene specialization +
grouped-cull DNF, see `per_scene_specialized_shader_pivot.md`): keep the
data-driven cull/DNF loops, DROP the generic bytecode VM in favour of a thin
per-structure codegen.

## Status
- The user's original problem (intersection lag) is ALREADY fixed & committed via
  VM specialization + grouped cull (0.9 → 35 FPS).
- Option 3 (thin data-driven codegen) is being implemented. Progress below.

## Implementation progress (thin codegen — Option 3)

**Done — the emitter foundation (`core/gpu_codegen.py`, tests
`tests/test_gpu_codegen.py`):**
- `emit_map_glsl(render_ir)` emits a small shader: GLSL preamble (reuses the
  serialize_scene buffer layout: Nodes@0, Params@1) + `leafDist()` branching ONLY
  the present primitive kinds + a DNF-loop `map()` (max over ≤4 additive groups of
  the group min, carved by -sub), reading AddLeaves@4 (uvec2 node_index,group_id)
  and Subs@5 (node_index) data buffers. Helpers (irOrientedLocal, cone/pyramid/
  box-frame) emitted only when a using kind is present.
- `scene_structure_signature()` = the recompile key (set of primitive kinds);
  `supported()` = core-primitive scenes that flatten to a bounded DNF.
- VALIDATED on the real RTX 3050 (Vulkan): the codegen shader compiles in **~40ms
  cold and is N-INDEPENDENT** — 199-node and 9999-node intersection scenes use the
  identical shader (0ms cache hit after the first bake). Intersection adds ZERO
  shader code (it's the `u_group_count` uniform + the group loop). qsb-compiles for
  sphere/box/intersection/difference scenes. 168 tests green.

**Done — full fragment shader + render proof:**
- `emit_fragment_shader()` adds a raymarch main (camera UBO, sphere-trace map(),
  Lambert + owner palette, grid plane). `tools/codegen_demo.py` renders a scene
  through the codegen path end-to-end (emit → compile → upload data → draw) and
  screenshots it. **Verified on the real GPU: intersection AND difference scenes
  render correctly** (colored blob, shading, grid). One gotcha recorded: parse
  `uniform_block_members` from the PRE-vulkanize source; Vulkan `grabFramebuffer`
  returns black (use OpenGL for screenshots; on-screen Vulkan is fine).

**Remaining — wire it into the live viewport (next):**
- A `CodegenRenderer` behind an env flag (e.g. `CASOCAD_CODEGEN=1`): on set_scene,
  `flatten_scene` → build AddLeaves/Subs buffers + nodes/params (reuse
  serialize_scene); compute the structure signature; re-bake+rebuild pipeline only
  when the signature changes, else just update buffers (moving objects = free).
- Port the raymarch main (camera UBO, shading, grid, gizmo overlay) from
  `raymarch_frag_main.glsl` into the codegen `main`.
- Select Vulkan for the codegen path (codegen compiles far faster on Vulkan).
- A/B against the VM with `tools/fps_bench.py` on the real GPU.
- Fallback to the VM for non-`supported()` scenes (carve-under-union, feature
  nodes) until codegen covers them.
