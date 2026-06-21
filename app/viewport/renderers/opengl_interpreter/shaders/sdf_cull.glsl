// Culled raymarch via a world-space uniform grid + DDA (original design, §13.5).
//
// Optional, gated by FEATURE_CULL. After sdf_core.glsl (uses irNodeSDF). The
// grid bins additive/subtractive leaves per cell (built on the host, core/
// gpu_cull.build_grid); this marches the ray clamped to cell boundaries so each
// step evaluates only the current cell's few leaves, skips empty cells in one
// jump, and stays exact (a leaf is binned into every cell its bound overlaps).
#ifdef FEATURE_CULL

layout(std430, binding = 8)  readonly buffer AddOff   { uint u_add_off[]; };
layout(std430, binding = 9)  readonly buffer AddCnt   { uint u_add_cnt[]; };
layout(std430, binding = 10) readonly buffer AddItem  { uint u_add_item[]; };
layout(std430, binding = 11) readonly buffer SubOff   { uint u_sub_off[]; };
layout(std430, binding = 12) readonly buffer SubCnt   { uint u_sub_cnt[]; };
layout(std430, binding = 13) readonly buffer SubItem  { uint u_sub_item[]; };

uniform vec3 u_grid_origin;
uniform vec3 u_grid_cell;
uniform int u_grid_dim;

bool irRayGrid(vec3 ro, vec3 rd, out float t0, out float t1) {
    vec3 lo = u_grid_origin;
    vec3 hi = u_grid_origin + u_grid_cell * float(u_grid_dim);
    vec3 inv = 1.0 / rd;
    vec3 a = (lo - ro) * inv;
    vec3 b = (hi - ro) * inv;
    vec3 tmin = min(a, b);
    vec3 tmax = max(a, b);
    t0 = max(max(tmin.x, tmin.y), tmin.z);
    t1 = min(min(tmax.x, tmax.y), tmax.z);
    return t1 >= max(t0, 0.0);
}

ivec3 irCellCoord(vec3 p) {
    ivec3 c = ivec3(floor((p - u_grid_origin) / u_grid_cell));
    return clamp(c, ivec3(0), ivec3(u_grid_dim - 1));
}

Sample irCellEval(int ci, vec3 p) {
    Sample r;
    r.dist = IR_FAR; r.owner_id = 0u; r.region_id = 0u;
    uint ao = u_add_off[ci];
    uint ac = u_add_cnt[ci];
    for (uint i = 0u; i < ac; i++) {
        Sample s = irNodeSDF(u_add_item[ao + i], p);
        if (s.dist < r.dist) r = s;
    }
    uint so = u_sub_off[ci];
    uint sc = u_sub_cnt[ci];
    for (uint i = 0u; i < sc; i++) {
        Sample s = irNodeSDF(u_sub_item[so + i], p);
        float carved = -s.dist;
        if (carved > r.dist) { r.dist = carved; r.owner_id = s.owner_id; r.region_id = s.region_id; }
    }
    return r;
}

// Distance along rd to leave the cell containing p.
float irCellExit(vec3 p, vec3 rd, ivec3 c) {
    vec3 cmin = u_grid_origin + vec3(c) * u_grid_cell;
    vec3 cmax = cmin + u_grid_cell;
    vec3 inv = 1.0 / rd;
    vec3 tb = max((cmin - p) * inv, (cmax - p) * inv);  // far face per axis
    return min(min(tb.x, tb.y), tb.z);
}

// Scene distance at p via the grid cell (for normals / dispatch).
float irCullDist(vec3 p) {
    ivec3 c = irCellCoord(p);
    int ci = (c.z * u_grid_dim + c.y) * u_grid_dim + c.x;
    return irCellEval(ci, p).dist;
}

Sample irMarchCulled(vec3 ro, vec3 rd, out vec3 hit_point, out bool hit) {
    hit = false;
    hit_point = ro;
    Sample empty;
    empty.dist = IR_FAR; empty.owner_id = 0u; empty.region_id = 0u;

    float t0, t1;
    if (!irRayGrid(ro, rd, t0, t1)) return empty;
    float t = max(t0, 0.0);

    for (int i = 0; i < 256; i++) {
        if (t > t1 + 0.01) break;
        vec3 p = ro + rd * t;
        ivec3 c = irCellCoord(p);
        int ci = (c.z * u_grid_dim + c.y) * u_grid_dim + c.x;
        Sample s = irCellEval(ci, p);
        if (s.dist < 0.0008) { hit = true; hit_point = p; return s; }
        // Distance from p to the current cell's exit face.
        float exit_d = irCellExit(p, rd, c);
        // Step by the SDF, but never skip past the cell boundary unchecked
        // (the next cell may hold nearer geometry). Empty cells (dist == IR_FAR)
        // jump straight to the boundary.
        float step = max(s.dist, 0.0002);
        step = min(step, exit_d + 0.0008);
        t += max(step, 0.0008);
    }
    return empty;
}

#endif  // FEATURE_CULL
