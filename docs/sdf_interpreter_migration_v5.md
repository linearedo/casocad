# SDF Interpreter Renderer — Migration Design (V5: Region-Parity Hardened)

This is a self-contained design spec. It does not describe the current state of the
codebase; it describes the target architecture and the order in which to build it, so
that an implementer starting from any commit can reproduce every stage from scratch.

V5 supersedes the V3/V4 drafts. The only substantive change vs V4 is Layer 2: the
region-split mechanism is generalized from an axis-aligned-box test into a
**selector-volume SDF** evaluated by the same interpreter, so it reaches full parity
with the existing CPU boundary-cutter feature (`core/boundary_patches.py`,
`surface_selector_volume`) — arbitrary, non-axis-aligned, curved region shapes — and
leaves room to grow.

## 1. Goal & non-goals

**Goal.** Eliminate the per-topology shader compilation freeze by replacing per-scene
GLSL code generation with a single fixed "interpreter" shader driven by a GPU node
buffer. A topology change becomes a buffer upload, not a recompile. Compile cost stops
scaling with scene size — it happens once, ever.

**Non-goals (explicitly out of scope for v1).**
* Spatial acceleration / BVH (added later, only when large scenes make per-frame cost
  bite — see §9).
* Changing geometry/meshing/export. Those use the CPU path (core/mesher, core/io) and
  are untouched. This is a preview-renderer change only — it cannot affect CAD
  correctness.

### 1.1 Domain model the interpreter must serve

This is a CFD-oriented CAD. The scene is not a loose pile of objects; it is a single
fluid-domain SDF built by a sequence of SDF operations (min, max, blends).

FluidDomain = Difference( Difference( BoundingBox, Sphere ), Cylinder ) ∪ …

Two facts about this model drive the whole design:

1. The operation sequence is long and order-dependent. A real domain may chain 1000+
   operations. The interpreter must evaluate arbitrarily long chains effortlessly.
2. Boundary attributes are attached per object AND per region. This is a CFD tool. A
   single geometry leaf (like a tube) can produce several boundary regions (inlet ring,
   outlet ring, tube wall). The SDF preview must return both the Owner ID (which object)
   and the Region ID (which part of the object) so the UI can visually distinguish
   boundary conditions.

*Note on the Two-Layer Region Model:*
* Layer 1 (Intrinsic): Geometric regions naturally emitted by the leaf primitive itself
  (e.g., tube walls vs flat caps). Evaluated directly inside the leaf SDF function.
* Layer 2 (Selector-Driven Split): A region carved out of an existing boundary by a
  separate *selector* — the same mechanism the CPU tool already exposes as the Boundary
  Planar Cutter and Surface Cutter. The selector is an SDF volume (an extruded 2D
  profile, an offset polyline band, an oriented slab, or a full 3D SDF); any point whose
  owner matches and that falls inside (or outside) the selector volume is re-tagged with
  the selector's region id. This is NOT limited to axis-aligned boxes — it is a general
  SDF inside/outside test (§6.1). Layer 2 is applied as downstream operations on the
  value stack via dedicated bytecode (§5, §6).

### 1.2 Core evaluation principles (non-negotiable)

1. Object / operation count is unbounded. The number of shapes and operations lives in
   GPU memory, never in a fixed-size array. A 100,000-operation chain is just a longer
   buffer.
2. The Per-Ray Execution Stack is strictly validated on the host. Evaluating one ray
   needs a scratch "value stack" to hold unresolved intermediate geometry. Its size is
   capped. Because this is a global execution stack, deep tree nesting (e.g.,
   `cap + 1` levels of unresolved brackets) will overflow it, while flat chains of
   10,000 operations will not. The Python host must simulate stack depth during
   serialization and throw a hard exception if the depth exceeds the cap. The shader
   must never be forced to "gracefully fail" on bad data.

   **Implementation note (as built).** The cap is *not* a single global 64. It is the
   largest value-stack array the *target shader stage* can compile:
   * Compute / validation path: `IR_STACK_CAPACITY = 64` (the design target). The
     compute stage binds the arrays fine on Intel/AMD/NVIDIA.
   * GL fragment backend: `FRAGMENT_STACK_CAPACITY = 16`. The fragment-stage GLSL
     compiler cannot bind seven 64-wide local arrays (the scene VM plus the three
     sub-VMs) — NVIDIA errors `C5041 "possibly large array"`, Mesa Intel hangs at
     link. So the fragment renderer compiles at 16 and the host validator is given
     that same 16 via `emit_program(stack_capacity=16)`, keeping host and shader in
     lock-step. `emit_program` / `simulate_stack` / `validate_profile_depths` take the
     capacity as a parameter; each backend passes the one it compiled with.
3. Operators are variable-arity (N in → M out). Nothing hardcodes "pop two push one". A
   future boundary cutter (1 in → 2 out) or fuse (N in → 1 out) must fit without
   redesign. The serialized bytecode explicitly tells the shader how many items to pop
   and push.

## 2. Root cause recap

Today _ParameterizedIRSceneSource.build() generates a bespoke GLSL function per
topology. The driver compiles it synchronously. This stalls 1–4 seconds at 5–8 nodes and
scales super-linearly. Parameters are already in a UBO; only the structure triggers
codegen.

## 3. Target architecture

RenderIR (flat nodes)
   │  serialize & validate max stack depth (backend-agnostic)
   ▼
GPU node buffer    (SSBO)  ┐
GPU param buffer   (SSBO)  ├─►  ONE fixed interpreter shader
GPU child buffer   (SSBO)  │      │ runs the eval bytecode against a small
GPU bytecode buffer(SSBO)  ┘      │ (dist, owner, region) stack  ── §6
                                  ▼
                           raymarch (unchanged loop) ──► colour + boundary tag

The bytecode buffer is the pre-computed evaluation program; the shader executes it
rather than walking a node tree.

## 4. Backend Decision

**Target OpenGL 4.6 core as the hard floor.**
* The interpreter needs SSBOs (GL 4.3).
* 4.6 is the final OpenGL version and universally supported on modern desktop
  Linux/Windows drivers.
* macOS GL is dead. Apple froze OpenGL at 4.1. The future macOS answer is Vulkan/Metal,
  so dropping macOS GL today is the correct strategic move. Request a 4.6 core context.

## 5. The buffers (IR → GPU bytes)

Define a stable std430 binary layout in backend-agnostic code (core/gpu_scene.py).

### 5.1 The Scalar-Only Alignment Rule

To completely bypass driver-specific std430 alignment and padding bugs (such as the
driver padding vec3 structures up to 16 bytes), complex vector types and nested
structures are strictly banned inside the parameter buffers.
* The GpuNode layout consists exclusively of 8 plain uint32 fields (exactly 32 bytes),
  aligning perfectly to 16-byte boundaries on all hardware.
* The u_params buffer is a completely flat array of raw float scalars. Floats in a flat
  list have a stride of exactly 4 bytes with zero padding. The shader manually
  re-assembles vectors on local registers by reading sequential indices starting at
  param_offset.

// Perfect 32-byte alignment, no padding gaps
struct GpuNode {
    uint type;          // node-type code
    uint dim;           // 1 / 2 / 3
    uint base_owner_id; // base owner id (regions are offsets from this)
    uint flags;         // bit flags (intrinsic type bits, or Layer 2 region id payload)
    uint param_offset;  // start index into the flat float array
    uint param_count;   // number of floats
    uint child_offset;  // start index into the child array
    uint child_count;   // number of children
};

layout(std430, binding = 0) readonly buffer Nodes    { GpuNode u_nodes[]; };
layout(std430, binding = 1) readonly buffer Params   { float   u_params[]; };
layout(std430, binding = 2) readonly buffer Children { uint    u_children[]; };
layout(std430, binding = 3) readonly buffer Program  { uint    u_bytecode[]; };

**The Bytecode Program.** Drive evaluation with an explicit instruction stream in
u_bytecode, not a bare list of node indices. A bare node list forces the shader to
re-derive every node's pop/push counts at runtime; an explicit bytecode makes the stack
effect self-describing and leaves room for future opcodes — which matters once operators
are N → M arity.

Opcodes (high 8 bits of each 32-bit instruction; payload is the low 24 bits = node
index unless noted):
* 0x01 PUSH_LEAF      — evaluate leaf SDF, push (dist, owner, region) onto the stack.
* 0x02 EVAL_OP        — pop N / push M per the node's type and child_count.
* 0x03 REGION_ASSIGN  — Layer 2 split. Modify the top stack element by testing the live
                        point against a *selector-volume SDF* and re-tagging region_id.
                        This is the general form; the box test is just one selector kind.

The operator's pop/push arity is carried by the node record itself (child_count and the
node type), so the payload stays a plain node index. The host computes this sequence,
calculates the exact peak stack depth (simulating the specific pushes and pops of every
instruction), and aborts serialization if peak_depth > IR_STACK_CAPACITY.

### 5.2 The Selector node (Layer 2 parity contract)

A REGION_ASSIGN instruction points at a **selector node**. The selector node mirrors,
field-for-field, what `surface_selector_volume()` produces on the CPU today, so the GPU
path has exact parity with the Boundary Planar Cutter / Surface Cutter:

GpuNode (type = REGION_SELECTOR) fields:
* base_owner_id  → only stack elements whose owner_id == base_owner_id are eligible
                   (matches the CPU "owner of the patch" scoping).
* flags          → the region_id to assign on a match.
* child_offset/  → child[0] = node index of the **selector-volume SDF subtree**. This
  child_count       subtree is any SDF the interpreter can already evaluate:
                       - extruded 2D profile  (circle / ellipse / rect / polygon)
                       - offset polyline band (DistanceOffsetProfile → extrude)
                       - oriented segment slab (Box)
                       - a full native 3D SDF selector
                   child[1] (optional) = scope-volume subtree (the patch's own
                       thin slab), matching CPU `scope_region` intersection.
* param_offset   → 1 float: signed inside/outside tolerance band `tol`. A negative tol
                   encodes "side = outside" (test dist > |tol|); positive encodes
                   "side = inside" (test dist <= tol). This reproduces the CPU
                   selector `side` semantics with zero extra fields.

The selector-volume subtree is serialized exactly like any other geometry, so **every
selector the CPU tool can build is expressible with no new primitive.** The only thing
Layer 2 adds over Layer 1 is the opcode that runs the inside test and overwrites
region_id on the top of the stack.

## 6. The interpreter shader — No tree walking

**The Value Stack.** The stack element is a triplet (distance, owner, region). This
guarantees CFD boundary conditions survive the SDF operations.

// Peak concurrent unresolved operations. 64 on the compute/validation path; the
// GL fragment backend compiles this chunk at FRAGMENT_STACK_CAPACITY = 16 because
// the fragment-stage compiler cannot bind the larger arrays (see §1.2(2) note).
const int IR_STACK_CAPACITY = 64;

struct Sample {
    float dist;
    uint owner_id;
    uint region_id;
};

Sample evalSceneSDF(vec3 p) {
    Sample stack[IR_STACK_CAPACITY];
    int sp = 0;

    for (int k = 0; k < u_program_length; k++) {
        uint instruction = u_bytecode[k];
        uint opcode  = instruction >> 24;
        uint payload = instruction & 0x00FFFFFF;

        if (opcode == OP_PUSH_LEAF) {
            // Leaf SDF handles Layer 1 (Intrinsic) region logic internally
            stack[sp++] = irNodeSDF(payload, p);
        }
        else if (opcode == OP_EVAL_NODE) {
            // payload = node_index. Node dictates operator type and N -> M arity.
            sp = applyOperator(payload, stack, sp);
        }
        else if (opcode == OP_REGION_ASSIGN) {
            // Layer 2: general selector-volume region split (§6.1)
            applyRegionSelector(payload, p, stack, sp);
        }
    }
    return stack[0];
}

### 6.1 applyRegionSelector — general selector-volume split

This is the parity-critical routine. It does NOT hardcode a box; it evaluates an
arbitrary selector-volume SDF (the same SDF the CPU cutter builds) and re-tags the top
of the stack. The selector volume is itself evaluated by the interpreter's leaf/VM
dispatch (`evalSubtreeSDF`), so curved, rotated, polyline, and full-3D selectors all
work identically.

void applyRegionSelector(uint node_idx, vec3 p, inout Sample stack[IR_STACK_CAPACITY],
                         int sp) {
    GpuNode node = u_nodes[node_idx];

    // Owner scoping: only re-tag the boundary that owns this selector.
    if (stack[sp - 1].owner_id != node.base_owner_id) return;

    // child[0] = selector-volume SDF subtree (extrude / band / slab / 3D SDF).
    uint sel_root = u_children[node.child_offset];
    float sel_d = evalSubtreeSDF(sel_root, p).dist;

    // Optional child[1] = scope volume (patch thin-slab), matching CPU scope_region.
    if (node.child_count > 1) {
        uint scope_root = u_children[node.child_offset + 1];
        float scope_d = evalSubtreeSDF(scope_root, p).dist;
        if (scope_d > SCOPE_TOL) return;       // outside the owning patch slab
    }

    // param[0] = signed tolerance: >=0 means "inside", <0 means "outside".
    float tol = u_params[node.param_offset];
    bool inside = (tol >= 0.0) ? (sel_d <= tol) : (sel_d > -tol);
    if (inside) {
        stack[sp - 1].region_id = node.flags;  // assign the selector's region id
    }
}

Notes:
* `evalSubtreeSDF` is the same stack VM (or the nested profile VM, §10 Phase B) used for
  any subtree. No selector kind is special-cased in the shader — expressiveness comes
  from the serialized subtree, exactly as on the CPU.
* The axis-aligned box from earlier drafts is simply the degenerate case where the
  selector subtree is a single Box node. It is no longer a limit, just one input.

**SDF Threading:**
* union min(a,b) → keep the nearer operand's owner and region.
* difference max(a,-b) → if the carving tool wins, the result is the cavity wall. It
  strictly inherits the tool's owner_id and the tool's region_id.

## 7. Integration with existing code

The seam already exists: the ViewportRenderer Protocol.
* Add renderers/opengl_interpreter/.
* upload_render_ir serializes the IR into the 4 buffers (Nodes, Params, Children,
  Bytecode).
* If program_compile_ms drops to ~0 on topology changes, the architecture works.

### 7.1 Repository layout

The serializer and the bytecode/validation logic are backend-agnostic and live in
core/ (so a future Vulkan backend reuses them unchanged). Everything GL-specific lives
behind the ViewportRenderer Protocol in its own backend folder, a sibling of the
existing codegen backend.

core/
  render_ir.py            # flat RenderIR node graph (input to serialization)
  gpu_scene.py            # std430 scalar layout: node / param / child buffers (§5)
  gpu_program.py          # bytecode emission + host stack-depth simulator/validator
                          #   - walks the SDF graph, emits PUSH_LEAF / EVAL_OP /
                          #     REGION_ASSIGN stream
                          #   - simulates per-instruction push/pop, computes peak depth
                          #   - raises if peak_depth > IR_STACK_CAPACITY (fail-fast)
  gpu_selector.py         # builds REGION_SELECTOR nodes from boundary selectors;
                          #   reuses core/boundary_patches.surface_selector_volume so
                          #   GPU Layer 2 == CPU cutter, by construction (§5.2)

app/viewport/
  renderer_base.py        # ViewportRenderer Protocol (the seam; unchanged)
  renderers/
    opengl/               # existing codegen backend, kept as fallback during migration
    opengl_interpreter/   # the new backend (this design)
      __init__.py
      renderer.py         # implements ViewportRenderer Protocol; paintGL-facing,
                          #   - no per-topology GLSL codegen
      scene_buffers.py    # owns the 4 SSBOs (nodes, params, children, bytecode);
                          #   - full upload on topology change, glBufferSubData
                          #     param-only fast path on move/resize (§12)
      sdf_evaluator.py    # compute-shader SDF sampler — validation harness that runs
                          #   the shader off-screen to compare against SDFNode.to_numpy
      shaders/
        sdf_interpreter.glsl       # irNodeSDF leaf dispatch (§6) + evalSceneSDF stack
                                   #   VM + applyRegionSelector (§6.1)
        sdf_profile.glsl           # second stack VM for the 2D/1D profile sub-graphs
                                   #   (placed_profile / extrude / revolve), §10 Phase B
        raymarch_interpreter.frag  # fragment shader: #includes the chunk, runs the
                                   #   (unchanged) raymarch loop against evalSceneSDF

tests/
  test_gpu_scene.py       # buffer round-trip + scalar alignment golden byte test
  test_gpu_program.py     # bytecode order is valid; stack-depth simulator raises on
                          #   nesting > 64, passes flat chains of 1000+ (§10 stage 1)
  test_sdf_interpreter.py # GPU-vs-to_numpy per node kind (every geometry stage)
  test_gpu_region_parity.py # GPU REGION_ASSIGN result == CPU surface_selector_values
                          #   for every selector kind (extrude / polyline band / segment
                          #   slab / 3D SDF), inside AND outside side (§5.2, §11)

**Source-of-truth boundaries.**
* The kind → uint node-type codes and the IR_STACK_CAPACITY constant are defined once in
  core/ and the GLSL #defines are generated from them, so Python and the shaders cannot
  drift.
* core/gpu_scene.py + core/gpu_program.py produce only bytes; they import nothing from
  app/. This keeps the data contract reusable across GL, Vulkan, or a CPU reference
  evaluator.

## 8. Multi-backend strategy (Vulkan prep)

1. The node-buffer binary layout is the cross-backend contract.
2. Write portable GLSL (no GL-only built-ins). GLSL → SPIR-V feeds Vulkan directly.
3. Vulkan differs in resource management (descriptor sets) but the data and shader logic
   are shared.

## 9. Performance & scaling

* Dynamic branching causes GPU divergence. V1 accepts this cost.
* Layer 2 selectors add a sub-evaluation per REGION_ASSIGN. They are rare (one per
  user-defined region) and only run after the main field resolves, so cost is bounded by
  region count, not pixel-by-geometry product.
* When scenes get too heavy, add spatial culling / BVH. Do not build this speculatively.

## 10. Staged migration plan

*Do not lump all geometry into one step. It will bottleneck development.*

1. Scaffolding & Bytecode Serializer: Request GL 4.6. Build the Python serializer that
   outputs the 4 buffers. Implement the stack-depth validation simulator. Unit-test that
   nested structures > 64 throw, and flat chains of 1000 pass.
2. Interpreter Shader (Leaves only): Build irNodeSDF dispatch for the 8 pure 3D
   primitives. Output both owner_id and intrinsic region_id. Validate vs to_numpy.
3. The Virtual Machine: Implement the stack execution loop and the fundamental SDF
   combiners (union, diff, intersect, smooth_union). Verify a 1000-operation carve chain
   renders perfectly and tracks CFD boundary tags on cavity walls.
4. Profile Geometry Phase A (Analytic): placed 2D sections (circle, rect, ellipse).
5. Profile Geometry Phase B (Interpreter within Interpreter): the 2D profile sub-tree
   logic. A second stack VM evaluating the profile's own SDF graph, with its own value
   stack and its own host-side peak-depth validation (the §1.2(2) rule, per profile).
6. Profile Geometry Phase C (Sweeps & Lofts): Extrude, revolve, tubes.
7. **Layer 2 Region Parity:** Implement REGION_ASSIGN + applyRegionSelector and
   gpu_selector.py. Drive selector-volume construction through the existing
   `surface_selector_volume`. Ship `test_gpu_region_parity.py` asserting GPU == CPU for
   every selector kind and both sides. **This stage is the explicit parity gate with the
   current Boundary Planar Cutter / Surface Cutter.**
8. Feature Parity & Release: Object-id picking, components/x-ray, opacity. Flip the
   default flag, deprecate the old codegen pipeline.

## 11. Region model — resolution & roadmap

* **Layer 1 (Intrinsic):** Solved natively inside leaf SDFs.
* **Layer 2 (Selector-Driven Split) — full parity:** Solved by REGION_ASSIGN +
  applyRegionSelector (§6.1) operating on an arbitrary selector-volume SDF (§5.2). This
  reaches **1:1 parity with the existing CPU Boundary Planar Cutter and Surface Cutter**
  because it consumes the very same `surface_selector_volume` output — extruded 2D
  profiles, offset polyline bands, oriented segment slabs, and full 3D SDF selectors,
  with inside/outside side and patch scoping. The earlier "AABB-only" framing is retired;
  axis-aligned boxes are merely the simplest selector subtree.
* **Future headroom (beyond current parity):** the same opcode trivially extends to
  - boolean *combinations* of selectors (multiple REGION_ASSIGN in sequence, or selector
    subtrees built from SDF boolean operations) for compound patches;
  - blended/soft region weights (return a weight from the inside test instead of a hard
    overwrite) for gradient boundary conditions;
  - additional selector primitives — anything expressible as an SDF needs no shader
    change, only a serializer addition.
* **The one genuinely out-of-scope item:** *non-SDF* freeform paint — a raw brush stroke
  with no backing geometry. That needs a per-face UV/texture lookup, orthogonal to the
  VM. Note this does NOT include the current cutters, whose output is always SDF-backed
  and therefore in scope.

## 12. Risks & Guardrails

* **Stack Overflow:** Solved by architecture. The shader does not do safety clamping. The
  host refuses to upload mathematically impossible topologies. Selector subtrees count
  toward the same peak-depth simulation.
* **Variable Arity Cost:** Be careful with N-arity operators. A SmoothUnion with 60
  children instantly eats 60 stack slots. Limit N-arity batching in the Python layer if
  needed.
* **Selector divergence vs CPU:** Guard with `test_gpu_region_parity.py` (§7.1). If the
  GPU inside test ever disagrees with `surface_selector_values`, parity is broken — treat
  as a release blocker, not a tolerance tweak.
* **Parameter-only fast path:** Must be maintained. Moving/resizing an object should
  trigger a glBufferSubData to the u_params buffer, bypassing the bytecode generator.
  Selector tolerances live in u_params, so re-tuning a cutter band stays on the fast path.

## 13. Implementation findings & revised rendering strategy (2026-06-22)

This section records what was actually learned building the interpreter and the
architecture decision that follows, so the reasoning is traceable.

### 13.1 What works (validated)

* The interpreter renders correctly. GPU-vs-`to_numpy` parity holds for every node
  kind: the 8 primitives, the SDF combine operators with owner/region tag threading on
  cavity walls, a ~1000-op flat carve chain, every 2D/1D profile kind, extrude / revolve
  / polyline+bezier tubes, and Layer-2 region selectors (GPU == `surface_selector_values`
  for all selector kinds, both sides). 57 GPU tests + the existing suite pass.
* On a **dedicated NVIDIA RTX 3050** the goal is met: creating 250+ objects keeps
  `program_compile_ms = 0` with flat ~1-2 ms scene updates and no crash — versus the
  codegen path, which on the same machine grew to ~5.5 s per edit and **crashed at ~65
  objects** ("RenderIR upload failed"). The per-topology compile freeze is gone.

### 13.2 The cross-vendor wall (root-caused)

The fixed interpreter shader is large (one shader serves all scenes; complexity scales
with *features*, not scene size — see §13.4). On **Mesa Intel (the integrated laptop
iGPU)** that shader **will not compile in any shader stage**:

* Fragment stage: link hangs (Mesa) / errors `C5041 "possibly large array"` (NVIDIA at
  cap 64).
* Compute stage: the lightweight one-eval-per-invocation form compiles fine (that is how
  the validation harness runs on Intel), but the full **raymarch** compute — which calls
  `evalSceneSDF` at ~6 sites (march loop + 4 normal taps + shading) — also hangs at link
  on Mesa, at every stack cap (64/16/8/4/2).

Diagnostics confirming it is a **compiler** problem, not GPU or stack:

* `dmesg` shows **no** GPU hang / reset / i915 error during the hang — the GPU is never
  involved (a shader that never finishes compiling is never submitted).
* CPU sampling shows the process pegged at **100% CPU on the main thread**, stuck inside
  the userspace `glLinkProgram` call — a compiler spin (combinatorial blow-up in an
  optimization pass when flattening the giant function with the nested VMs + big arrays).
* `MESA_GLSL=nopt` does **not** help — it only disables the legacy GLSL-IR optimizer; the
  spin is in a deeper NIR / Intel-backend pass with no public off switch.
* Reducing only the stack cap does **not** help (full interpreter hangs even at cap 2).
  The lever that *works* is **shader size/complexity**.

Empirical compile results on Mesa Intel:

| Interpreter contents                                              | Mesa Intel |
|------------------------------------------------------------------|------------|
| primitives + operators only (minimal)                            | ✅ ~90 ms  |
| all 8 primitives + all 4 SDF operators                           | ✅ ~143 ms |
| + profile sub-VMs + 2D helpers + subtree evaluator (full)        | ❌ hangs   |

So the specific culprit is the **profile / sweep / selector machinery** (the nested 2D/1D
profile VMs, `evalSubtreeSDF`, and the bezier/ellipse/polygon 2D helpers), inlined into
one function. Primitives + operators alone compile everywhere.

Hardware note: the failing GPU is the **integrated** Intel iGPU (weakest hardware); the
working GPU is the **dedicated** RTX. AMD *dedicated* is untested — not known-broken
(RADV/ACO is far more capable than Intel integrated).

### 13.3 Decision: one interpreter codebase, drop codegen

The interpreter is the chosen renderer. The per-scene **codegen path is to be deleted**
(its compile freeze and ~65-object crash are exactly what we are removing). We do **not**
keep two rendering engines.

To stay cross-vendor with a single codebase, the interpreter shader is **assembled
per-GPU from feature modules** (the standard "ubershader with feature flags" pattern):

* Shader is split into chunk files: `sdf_core` (primitives + operators + VM, always) plus
  optional `sdf_profiles`, `sdf_sweeps`, `sdf_selectors`.
* Per-GPU assembly chooses which chunks to concatenate + which `#define FEATURE_*` to
  prepend. A ~10-line `GPU → feature set` tier table drives it (e.g. NVIDIA/dedicated →
  ALL; default/Mesa → PRIMITIVES | OPERATORS).
* No logic is duplicated — each feature is written once; tiers differ only by which blocks
  compile in. Ship 2-3 fixed tiers (not 2^N combinations) and test those.
* A shared `node-kind → required-feature` map lets the host skip/flag a node a tier does
  not support (e.g. an extrude on a lean GPU) — no codegen fallback.
* Because Mesa *hangs* rather than *errors*, the tier must be chosen **up front** by GPU
  detection; we must never "try the big shader and fall back" (the try never returns).

Net effect on maintenance: this is *simpler* than today's codegen + interpreter split —
one engine plus a small feature table — not a new two-path burden.

### 13.4 Two kinds of "growth" (clarification)

* The **interpreter** shader is **fixed-size w.r.t. the scene** — a 5-object and a
  100,000-object scene use the identical shader; scene data lives in buffers. It grows
  only when *new geometry features* are added (slow, developer-controlled). So it never
  becomes NVIDIA-unsuitable as users build bigger scenes.
* The **codegen** shader grows *super-linearly with the scene* (a function per object) —
  that is the freeze, and why it eventually fails even on strong GPUs. (Being deleted.)

### 13.5 Known remaining cost: per-frame framerate (§9 comes due)

With the interpreter, *editing* is instant but *per-frame* raymarch cost grows with object
count: every pixel, every frame, every march step re-walks the whole program (all
objects), with no spatial culling. Measured ~1-2 fps on large scenes (hundreds of
objects) on the RTX. This is the §9 tradeoff, now due. Planned mitigations, in order:

1. **Adaptive resolution while interacting** (render at reduced resolution during camera
   motion, full-res when idle) — large perceived-fps win, renderer-layer change.
2. **Per-object bounding-box early-out in the VM** (return the conservative distance-to-box
   for far objects; safe for sphere tracing) — cheaper per-node work.
3. **Spatial acceleration (BVH / uniform grid)** so a pixel evaluates only nearby objects
   — O(N)→O(few) per pixel, the structural fix for large scenes.

### 13.6 Forward plan

1. Split the interpreter shader into `sdf_core` + optional feature chunks; add the tier
   table + per-GPU assembly; host feature-gating of unsupported node kinds.
2. Prove the lean tier on Mesa and the full tier on NVIDIA, then **delete the codegen
   path** (`_ParameterizedIRSceneSource` and the legacy renderer) in a final commit.
3. Performance pass (§13.5): adaptive resolution, bbox early-out, then BVH.
