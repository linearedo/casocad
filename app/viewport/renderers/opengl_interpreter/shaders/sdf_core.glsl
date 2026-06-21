// SDF interpreter CORE chunk (always compiled) — design §6, §13.3.
//
// Buffers + Sample + flat-scalar readers + the 8 primitive leaves + the SDF
// operators + the value-stack VM. Optional features (profiles, sweeps,
// selectors) live in sibling chunks concatenated after this one, gated by
// FEATURE_* defines so a weak driver can omit what it cannot compile.
//
// The host prepends the generated node-type/opcode/capacity #defines and the
// FEATURE_* block before this source.

struct GpuNode {
    uint type;
    uint dim;
    uint base_owner_id;
    uint flags;
    uint param_offset;
    uint param_count;
    uint child_offset;
    uint child_count;
};

layout(std430, binding = 0) readonly buffer Nodes    { GpuNode u_nodes[]; };
layout(std430, binding = 1) readonly buffer Params   { float   u_params[]; };
layout(std430, binding = 2) readonly buffer Children { uint    u_children[]; };
layout(std430, binding = 3) readonly buffer Program  { uint    u_bytecode[]; };

// One stack element: signed distance plus the CFD boundary tags that must
// survive every SDF operation.
struct Sample {
    float dist;
    uint owner_id;
    uint region_id;
};

const float IR_FAR = 1.0e6;

uniform uint u_program_length;

// ---- flat-scalar param readers (design §5.1) -------------------------------
float irP(uint base, uint i) { return u_params[base + i]; }

vec3 irP3(uint base, uint i) {
    return vec3(u_params[base + i], u_params[base + i + 1u], u_params[base + i + 2u]);
}

vec3 irOrientedLocal(vec3 p, vec3 c, vec3 au, vec3 av, vec3 aw) {
    vec3 local = p - c;
    return vec3(dot(local, au), dot(local, av), dot(local, aw));
}

// ---- primitive analytic helpers --------------------------------------------
float irPyramidUnitSDF(vec3 p, float h) {
    float m2 = h * h + 0.25;
    p.xz = abs(p.xz);
    p.xz = (p.z > p.x) ? p.zx : p.xz;
    p.xz -= 0.5;
    vec3 q = vec3(p.z, h * p.y - 0.5 * p.x, h * p.x + 0.5 * p.y);
    float s = max(-q.x, 0.0);
    float t = clamp((q.y - 0.5 * p.z) / (m2 + 0.25), 0.0, 1.0);
    float a = m2 * (q.x + s) * (q.x + s) + q.y * q.y;
    float b = m2 * (q.x + 0.5 * t) * (q.x + 0.5 * t)
        + (q.y - m2 * t) * (q.y - m2 * t);
    float d2 = min(q.y, -q.x * m2 - q.y * 0.5) > 0.0 ? 0.0 : min(a, b);
    return sqrt((d2 + q.z * q.z) / m2) * sign(max(q.z, -p.y));
}

float irCappedConeSDF(vec3 p, float h, float r1, float r2) {
    vec2 q = vec2(length(p.xy), p.z);
    vec2 k1 = vec2(r2, h);
    vec2 k2 = vec2(r2 - r1, 2.0 * h);
    vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h);
    vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

float irBoxFrameSDF(vec3 p, vec3 b, float e) {
    p = abs(p) - b;
    vec3 q = abs(p + e) - e;
    return min(
        min(
            length(max(vec3(p.x, q.y, q.z), 0.0)) + min(max(p.x, max(q.y, q.z)), 0.0),
            length(max(vec3(q.x, p.y, q.z), 0.0)) + min(max(q.x, max(p.y, q.z)), 0.0)
        ),
        length(max(vec3(q.x, q.y, p.z), 0.0)) + min(max(q.x, max(q.y, p.z)), 0.0)
    );
}

// Returns true and sets `dist` if `node` is one of the 8 core primitives.
bool irPrimitiveLeaf(GpuNode node, vec3 p, out float dist) {
    uint b = node.param_offset;
    uint t = node.type;
    if (t == NODE_SPHERE) {
        dist = length(p - irP3(b, 0u)) - irP(b, 3u);
        return true;
    }
    if (t == NODE_BOX) {
        vec3 local = irOrientedLocal(p, irP3(b, 0u), irP3(b, 3u), irP3(b, 6u), irP3(b, 9u));
        vec3 q = abs(local) - irP3(b, 12u);
        dist = length(max(q, vec3(0.0))) + min(max(q.x, max(q.y, q.z)), 0.0);
        return true;
    }
    if (t == NODE_CYLINDER) {
        vec3 local = irOrientedLocal(p, irP3(b, 0u), irP3(b, 3u), irP3(b, 6u), irP3(b, 9u));
        vec2 d = abs(vec2(length(local.xy), local.z)) - vec2(irP(b, 12u), irP(b, 13u));
        dist = min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)));
        return true;
    }
    if (t == NODE_CONE) {
        vec3 local = irOrientedLocal(p, irP3(b, 0u), irP3(b, 3u), irP3(b, 6u), irP3(b, 9u));
        float radius = irP(b, 12u);
        float half_height = irP(b, 13u);
        float height = 2.0 * half_height;
        vec2 q = vec2(radius, -height);
        vec2 w = vec2(length(local.xy), local.z - half_height);
        vec2 a = w - q * clamp(dot(w, q) / dot(q, q), 0.0, 1.0);
        vec2 bb = w - q * vec2(clamp(w.x / q.x, 0.0, 1.0), 1.0);
        float d = min(dot(a, a), dot(bb, bb));
        float s = max(-(w.x * q.y - w.y * q.x), -(w.y - q.y));
        dist = sqrt(d) * sign(s);
        return true;
    }
    if (t == NODE_CAPPED_CONE) {
        vec3 local = irOrientedLocal(p, irP3(b, 0u), irP3(b, 3u), irP3(b, 6u), irP3(b, 9u));
        dist = irCappedConeSDF(local, irP(b, 14u), irP(b, 12u), irP(b, 13u));
        return true;
    }
    if (t == NODE_BOX_FRAME) {
        vec3 local = irOrientedLocal(p, irP3(b, 0u), irP3(b, 3u), irP3(b, 6u), irP3(b, 9u));
        dist = irBoxFrameSDF(local, irP3(b, 12u), irP(b, 15u));
        return true;
    }
    if (t == NODE_PYRAMID) {
        vec3 local = irOrientedLocal(p, irP3(b, 0u), irP3(b, 3u), irP3(b, 6u), irP3(b, 9u));
        float base_half_size = irP(b, 12u);
        float half_height = irP(b, 13u);
        float scale = 2.0 * base_half_size;
        float height = (2.0 * half_height) / scale;
        vec3 q = vec3(local.x / scale, (local.z + half_height) / scale, local.y / scale);
        dist = scale * irPyramidUnitSDF(q, height);
        return true;
    }
    if (t == NODE_TORUS) {
        vec3 local = irOrientedLocal(p, irP3(b, 0u), irP3(b, 3u), irP3(b, 6u), irP3(b, 9u));
        dist = length(vec2(length(local.xy) - irP(b, 12u), local.z)) - irP(b, 13u);
        return true;
    }
    return false;
}

// Optional feature leaf handlers (defined in their chunks; declared so the core
// dispatch can call them under the matching FEATURE_* guard).
#ifdef FEATURE_PROFILES
bool irProfileLeaf(GpuNode node, vec3 p, out float dist);
#endif
#ifdef FEATURE_SWEEPS
bool irSweepLeaf(GpuNode node, vec3 p, out float dist);
#endif

float irLeafDistance(GpuNode node, vec3 p) {
    float d;
    if (irPrimitiveLeaf(node, p, d)) return d;
#ifdef FEATURE_PROFILES
    if (irProfileLeaf(node, p, d)) return d;
#endif
#ifdef FEATURE_SWEEPS
    if (irSweepLeaf(node, p, d)) return d;
#endif
    return IR_FAR;
}

Sample irNodeSDF(uint node_index, vec3 p) {
    GpuNode node = u_nodes[node_index];
    Sample s;
    s.dist = irLeafDistance(node, p);
    s.owner_id = node.base_owner_id;
    s.region_id = 0u;
    return s;
}

// ---- value-stack VM (design §6) --------------------------------------------
int applyOperator(uint node_index, inout Sample stack[IR_STACK_CAPACITY], int sp) {
    GpuNode node = u_nodes[node_index];
    int n = int(node.child_count);
    int base = sp - n;
    Sample acc = stack[base];
    uint t = node.type;

    if (t == NODE_UNION) {
        for (int i = 1; i < n; i++) {
            Sample s = stack[base + i];
            if (s.dist < acc.dist) acc = s;
        }
    } else if (t == NODE_INTERSECTION) {
        for (int i = 1; i < n; i++) {
            Sample s = stack[base + i];
            if (s.dist > acc.dist) acc = s;
        }
    } else if (t == NODE_DIFFERENCE) {
        for (int i = 1; i < n; i++) {
            Sample s = stack[base + i];
            float carved = -s.dist;
            if (carved > acc.dist) {
                acc.dist = carved;
                acc.owner_id = s.owner_id;
                acc.region_id = s.region_id;
            }
        }
    } else if (t == NODE_SMOOTH_UNION) {
        float k = irP(node.param_offset, 0u);
        for (int i = 1; i < n; i++) {
            Sample s = stack[base + i];
            float a = acc.dist;
            float b = s.dist;
            float h = clamp(0.5 + 0.5 * (b - a) / k, 0.0, 1.0);
            float d = mix(b, a, h) - k * h * (1.0 - h);
            if (b < a) { acc.owner_id = s.owner_id; acc.region_id = s.region_id; }
            acc.dist = d;
        }
    }

    stack[base] = acc;
    return base + 1;
}

// Layer 2 hook. The real implementation lives in sdf_selectors.glsl; when that
// feature is absent it is a no-op so the VM loop compiles unchanged.
void applyRegionSelector(uint node_index, vec3 p,
                         inout Sample stack[IR_STACK_CAPACITY], int sp);
#ifndef FEATURE_SELECTORS
void applyRegionSelector(uint node_index, vec3 p,
                         inout Sample stack[IR_STACK_CAPACITY], int sp) {}
#endif

Sample evalSceneSDF(vec3 p) {
    Sample stack[IR_STACK_CAPACITY];
    int sp = 0;
    for (uint k = 0u; k < u_program_length; k++) {
        uint instruction = u_bytecode[k];
        uint opcode = instruction >> OPCODE_SHIFT;
        uint payload = instruction & PAYLOAD_MASK;
        if (opcode == OP_PUSH_LEAF) {
            stack[sp] = irNodeSDF(payload, p);
            sp++;
        } else if (opcode == OP_EVAL_NODE) {
            sp = applyOperator(payload, stack, sp);
        } else if (opcode == OP_REGION_ASSIGN) {
            applyRegionSelector(payload, p, stack, sp);
        }
    }
    if (sp == 0) {
        Sample empty;
        empty.dist = IR_FAR;
        empty.owner_id = 0u;
        empty.region_id = 0u;
        return empty;
    }
    return stack[0];
}
