// SDF interpreter SELECTORS chunk (optional, FEATURE_SELECTORS) — design §6.1, §13.3.
// Layer 2 region split: an arbitrary SDF subtree VM + applyRegionSelector.
// After sdf_core.glsl (uses irNodeSDF). Provides the real applyRegionSelector
// (core supplies a no-op when this feature is absent).
#ifdef FEATURE_SELECTORS

bool irIsOperator(uint t) {
    return t == NODE_UNION || t == NODE_INTERSECTION
        || t == NODE_DIFFERENCE || t == NODE_SMOOTH_UNION;
}

Sample irCombine(uint t, uint param_offset, Sample a, Sample b) {
    Sample r = a;
    if (t == NODE_UNION) {
        if (b.dist < a.dist) r = b;
    } else if (t == NODE_INTERSECTION) {
        if (b.dist > a.dist) r = b;
    } else if (t == NODE_DIFFERENCE) {
        float carved = -b.dist;
        if (carved > a.dist) { r.dist = carved; r.owner_id = b.owner_id; r.region_id = b.region_id; }
    } else {
        float k = irP(param_offset, 0u);
        float h = clamp(0.5 + 0.5 * (b.dist - a.dist) / k, 0.0, 1.0);
        float d = mix(b.dist, a.dist, h) - k * h * (1.0 - h);
        if (b.dist < a.dist) { r.owner_id = b.owner_id; r.region_id = b.region_id; }
        r.dist = d;
    }
    return r;
}

Sample evalSubtreeSDF(uint root, vec3 p) {
    uint work_node[IR_STACK_CAPACITY];
    int work_state[IR_STACK_CAPACITY];
    Sample vstack[IR_STACK_CAPACITY];
    int wsp = 0;
    int vsp = 0;
    work_node[0] = root;
    work_state[0] = 0;
    wsp = 1;

    while (wsp > 0) {
        wsp--;
        uint idx = work_node[wsp];
        int state = work_state[wsp];
        GpuNode n = u_nodes[idx];
        if (state == 0 && irIsOperator(n.type)) {
            work_node[wsp] = idx;
            work_state[wsp] = 1;
            wsp++;
            for (int c = int(n.child_count) - 1; c >= 0; c--) {
                work_node[wsp] = u_children[n.child_offset + uint(c)];
                work_state[wsp] = 0;
                wsp++;
            }
        } else if (state == 1) {
            int base = vsp - int(n.child_count);
            Sample acc = vstack[base];
            for (int i = 1; i < int(n.child_count); i++) {
                acc = irCombine(n.type, n.param_offset, acc, vstack[base + i]);
            }
            vsp = base;
            vstack[vsp] = acc;
            vsp++;
        } else {
            vstack[vsp] = irNodeSDF(idx, p);
            vsp++;
        }
    }
    return vstack[0];
}

const float IR_SCOPE_TOL = 1.5e-3;  // matches PATCH_TOLERANCE on the CPU

void applyRegionSelector(uint node_index, vec3 p,
                         inout Sample stack[IR_STACK_CAPACITY], int sp) {
    GpuNode node = u_nodes[node_index];
    if (sp < 1) return;
    if (stack[sp - 1].owner_id != node.base_owner_id) return;

    uint sel_root = u_children[node.child_offset];
    float sel_d = evalSubtreeSDF(sel_root, p).dist;

    if (node.child_count > 1u) {
        uint scope_root = u_children[node.child_offset + 1u];
        float scope_d = evalSubtreeSDF(scope_root, p).dist;
        if (scope_d > IR_SCOPE_TOL) return;
    }

    float tol = irP(node.param_offset, 0u);
    bool inside = (tol >= 0.0) ? (sel_d <= tol) : (sel_d > -tol);
    if (inside) {
        stack[sp - 1].region_id = node.flags;
    }
}

#endif  // FEATURE_SELECTORS
