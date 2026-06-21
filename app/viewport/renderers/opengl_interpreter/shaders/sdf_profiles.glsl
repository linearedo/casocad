// SDF interpreter PROFILES chunk (optional, FEATURE_PROFILES) — design §13.3.
//
// 2D/1D profile helpers + the nested profile sub-VMs + the placed-2D / extrude /
// revolve / placed-1D leaf handler. Concatenated after sdf_core.glsl. Omitted on
// drivers whose compiler cannot swallow it (Mesa Intel; §13.2).
#ifdef FEATURE_PROFILES

// ---- 2D helpers ------------------------------------------------------------
float irSegmentDistance2D(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a;
    vec2 ba = b - a;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1.0e-12), 0.0, 1.0);
    return length(pa - ba * h);
}

bool irSegmentRayCrosses(vec2 pos, vec2 A, vec2 B) {
    float denominator = B.y - A.y;
    denominator = abs(denominator) < 1.0e-8
        ? (denominator < 0.0 ? -1.0e-8 : 1.0e-8)
        : denominator;
    return (
        ((A.y > pos.y) != (B.y > pos.y))
        && (pos.x < (B.x - A.x) * (pos.y - A.y) / denominator + A.x)
    );
}

float irSegmentRayCrossValue(vec2 pos, vec2 A, vec2 B) {
    return irSegmentRayCrosses(pos, A, B) ? 1.0 : 0.0;
}

float irExactEllipseDistance(vec2 p, vec2 ab) {
    p = abs(p);
    if (p.x > p.y) {
        p = p.yx;
        ab = ab.yx;
    }
    float l = ab.y * ab.y - ab.x * ab.x;
    float m = ab.x * p.x / l;
    float m2 = m * m;
    float n = ab.y * p.y / l;
    float n2 = n * n;
    float c = (m2 + n2 - 1.0) / 3.0;
    float c3 = c * c * c;
    float d = c3 + m2 * n2;
    float q = d + m2 * n2;
    float g = m + m * n2;
    float co;
    if (d < 0.0) {
        float h = acos(clamp(q / c3, -1.0, 1.0)) / 3.0;
        float s = cos(h);
        float t = sin(h) * sqrt(3.0);
        float rx = sqrt(max(-c * (s + t + 2.0) + m2, 0.0));
        float ry = sqrt(max(-c * (s - t + 2.0) + m2, 0.0));
        co = (ry + sign(l) * rx + abs(g) / (rx * ry) - m) * 0.5;
    } else {
        float h = 2.0 * m * n * sqrt(max(d, 0.0));
        float s = sign(q + h) * pow(abs(q + h), 1.0 / 3.0);
        float t = sign(q - h) * pow(abs(q - h), 1.0 / 3.0);
        float rx = -s - t - c * 4.0 + 2.0 * m2;
        float ry = (s - t) * sqrt(3.0);
        float rm = sqrt(max(rx * rx + ry * ry, 0.0));
        co = (ry / sqrt(max(rm - rx, 1.0e-12)) + 2.0 * g / rm - m) * 0.5;
    }
    co = clamp(co, 0.0, 1.0);
    vec2 r = ab * vec2(co, sqrt(max(1.0 - co * co, 0.0)));
    return length(r - p) * sign(p.y - r.y);
}

float irQuadraticBezierDistance(vec2 pos, vec2 A, vec2 B, vec2 C) {
    vec2 a = B - A;
    vec2 b = A - 2.0 * B + C;
    vec2 c = a * 2.0;
    vec2 d = A - pos;
    float b_dot_b = dot(b, b);
    if (b_dot_b <= 1.0e-12) {
        vec2 pa = pos - A;
        vec2 ba = C - A;
        float hh = clamp(dot(pa, ba) / max(dot(ba, ba), 1.0e-12), 0.0, 1.0);
        return length(pa - ba * hh);
    }
    float kk = 1.0 / b_dot_b;
    float kx = kk * dot(a, b);
    float ky = (kk * (2.0 * dot(a, a) + dot(d, b))) / 3.0;
    float kz = kk * dot(d, a);
    float p = ky - kx * kx;
    float q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    float h = q * q + 4.0 * p * p * p;
    float res = 0.0;
    if (h >= 0.0) {
        h = sqrt(h);
        vec2 x = (vec2(h, -h) - q) * 0.5;
        vec2 uv = sign(x) * pow(abs(x), vec2(1.0 / 3.0));
        float t = clamp(uv.x + uv.y - kx, 0.0, 1.0);
        vec2 w = d + (c + b * t) * t;
        res = dot(w, w);
    } else {
        float z = sqrt(max(-p, 0.0));
        float denominator = 2.0 * p * z;
        float angle_argument = denominator == 0.0 ? 0.0 : q / denominator;
        float v = acos(clamp(angle_argument, -1.0, 1.0)) / 3.0;
        float m = cos(v);
        float n = sin(v) * 1.732050808;
        vec3 t = clamp(vec3(m + m, -n - m, n - m) * z - kx, 0.0, 1.0);
        vec2 qx = d + (c + b * t.x) * t.x;
        vec2 qy = d + (c + b * t.y) * t.y;
        res = min(dot(qx, qx), dot(qy, qy));
    }
    return sqrt(max(res, 0.0));
}

bool irQuadraticBezierRayCrosses(vec2 pos, vec2 A, vec2 B, vec2 C) {
    float qa = A.y - 2.0 * B.y + C.y;
    float qb = 2.0 * (B.y - A.y);
    float qc = A.y - pos.y;
    bool crosses = false;
    if (abs(qa) <= 1.0e-8) {
        if (abs(qb) <= 1.0e-8) return false;
        float t = -qc / qb;
        if (t >= 0.0 && t < 1.0) {
            vec2 q = mix(mix(A, B, t), mix(B, C, t), t);
            crosses = q.x > pos.x;
        }
        return crosses;
    }
    float h = qb * qb - 4.0 * qa * qc;
    if (h <= 1.0e-8) return false;
    float root = sqrt(h);
    float t0 = (-qb - root) / (2.0 * qa);
    float t1 = (-qb + root) / (2.0 * qa);
    if (t0 >= 0.0 && t0 < 1.0) {
        vec2 q0 = mix(mix(A, B, t0), mix(B, C, t0), t0);
        crosses = crosses != (q0.x > pos.x);
    }
    if (t1 >= 0.0 && t1 < 1.0) {
        vec2 q1 = mix(mix(A, B, t1), mix(B, C, t1), t1);
        crosses = crosses != (q1.x > pos.x);
    }
    return crosses;
}

float irQuadraticBezierRayCrossValue(vec2 pos, vec2 A, vec2 B, vec2 C) {
    return irQuadraticBezierRayCrosses(pos, A, B, C) ? 1.0 : 0.0;
}

float irRayDistance2D(vec2 p, vec2 ray) {
    float projection = dot(p, ray);
    float cross_value = ray.x * p.y - ray.y * p.x;
    return projection >= 0.0 ? abs(cross_value) : length(p);
}

float irAngularSectorSDF(vec2 p, float angle) {
    if (angle >= 6.283184) return -1000000.0;
    float theta = atan(p.y, p.x);
    theta = theta < 0.0 ? theta + 6.28318530718 : theta;
    bool inside = theta <= angle;
    vec2 end_ray = vec2(cos(angle), sin(angle));
    float distance_to_edge = min(
        irRayDistance2D(p, vec2(1.0, 0.0)),
        irRayDistance2D(p, end_ray)
    );
    return inside ? -distance_to_edge : distance_to_edge;
}

// ---- profile sub-VMs (design §10 Phase B) ----------------------------------
float irProfileLeafValue(uint node_index, vec2 q) {
    GpuNode n = u_nodes[node_index];
    uint b = n.param_offset;
    uint t = n.type;

    if (t == NODE_PROFILE_CIRCLE_2D) {
        return length(q - vec2(irP(b, 0u), irP(b, 1u))) - irP(b, 2u);
    }
    if (t == NODE_PROFILE_RECTANGLE_2D) {
        vec2 d = abs(q - vec2(irP(b, 0u), irP(b, 1u))) - vec2(irP(b, 2u), irP(b, 3u));
        return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
    }
    if (t == NODE_PROFILE_SQUARE_2D) {
        vec2 d = abs(q - vec2(irP(b, 0u), irP(b, 1u))) - vec2(irP(b, 2u));
        return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
    }
    if (t == NODE_PROFILE_ROUNDED_RECTANGLE_2D) {
        float cr = irP(b, 4u);
        vec2 inner = vec2(irP(b, 2u), irP(b, 3u)) - vec2(cr);
        vec2 d = abs(q - vec2(irP(b, 0u), irP(b, 1u))) - inner;
        return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - cr;
    }
    if (t == NODE_PROFILE_ELLIPSE_2D) {
        return irExactEllipseDistance(q - vec2(irP(b, 0u), irP(b, 1u)),
                                      vec2(irP(b, 2u), irP(b, 3u)));
    }
    if (t == NODE_PROFILE_POLYGON_2D) {
        uint pc = n.param_count / 2u;
        if (pc < 3u) return IR_FAR;
        float d = IR_FAR;
        bool inside = false;
        for (uint i = 0u; i < pc; i++) {
            uint j = (i + 1u) % pc;
            vec2 a = vec2(irP(b, i * 2u), irP(b, i * 2u + 1u));
            vec2 bb = vec2(irP(b, j * 2u), irP(b, j * 2u + 1u));
            d = min(d, irSegmentDistance2D(q, a, bb));
            inside = inside != irSegmentRayCrosses(q, a, bb);
        }
        return inside ? -d : d;
    }
    if (t == NODE_PROFILE_POLYLINE_2D) {
        uint pc = n.param_count / 2u;
        if (pc < 2u) return IR_FAR;
        float d = IR_FAR;
        for (uint i = 0u; i + 1u < pc; i++) {
            vec2 a = vec2(irP(b, i * 2u), irP(b, i * 2u + 1u));
            vec2 bb = vec2(irP(b, (i + 1u) * 2u), irP(b, (i + 1u) * 2u + 1u));
            d = min(d, irSegmentDistance2D(q, a, bb));
        }
        return d;
    }
    if (t == NODE_PROFILE_BEZIER_CURVE_2D) {
        uint pc = n.param_count / 2u;
        if (pc < 3u) return IR_FAR;
        float d = IR_FAR;
        for (uint i = 0u; i + 2u < pc; i += 2u) {
            vec2 a = vec2(irP(b, i * 2u), irP(b, i * 2u + 1u));
            vec2 bb = vec2(irP(b, (i + 1u) * 2u), irP(b, (i + 1u) * 2u + 1u));
            vec2 c = vec2(irP(b, (i + 2u) * 2u), irP(b, (i + 2u) * 2u + 1u));
            d = min(d, irQuadraticBezierDistance(q, a, bb, c));
        }
        return d;
    }
    if (t == NODE_PROFILE_BEZIER_SURFACE_2D) {
        uint pc = n.param_count / 2u;
        if (pc < 3u) return IR_FAR;
        vec2 first = vec2(irP(b, 0u), irP(b, 1u));
        vec2 last = vec2(irP(b, (pc - 1u) * 2u), irP(b, (pc - 1u) * 2u + 1u));
        bool closed = distance(first, last) <= 1.0e-6;
        float d = IR_FAR;
        for (uint i = 0u; i + 2u < pc; i += 2u) {
            vec2 a = vec2(irP(b, i * 2u), irP(b, i * 2u + 1u));
            vec2 bb = vec2(irP(b, (i + 1u) * 2u), irP(b, (i + 1u) * 2u + 1u));
            vec2 c = vec2(irP(b, (i + 2u) * 2u), irP(b, (i + 2u) * 2u + 1u));
            d = min(d, irQuadraticBezierDistance(q, a, bb, c));
        }
        if (!closed) d = min(d, irSegmentDistance2D(q, last, first));
        float crossings = 0.0;
        for (uint i = 0u; i + 2u < pc; i += 2u) {
            vec2 a = vec2(irP(b, i * 2u), irP(b, i * 2u + 1u));
            vec2 bb = vec2(irP(b, (i + 1u) * 2u), irP(b, (i + 1u) * 2u + 1u));
            vec2 c = vec2(irP(b, (i + 2u) * 2u), irP(b, (i + 2u) * 2u + 1u));
            crossings += irQuadraticBezierRayCrossValue(q, a, bb, c);
        }
        if (!closed) crossings += irSegmentRayCrossValue(q, last, first);
        float parity = mod(crossings, 2.0);
        return (1.0 - 2.0 * parity) * d;
    }
    return IR_FAR;
}

bool irIsProfileCombinator(uint t) {
    return t == NODE_PROFILE_UNION_2D
        || t == NODE_PROFILE_INTERSECTION_2D
        || t == NODE_PROFILE_DIFFERENCE_2D
        || t == NODE_PROFILE_SMOOTH_UNION_2D
        || t == NODE_PROFILE_OFFSET_2D
        || t == NODE_PROFILE_DISTANCE_OFFSET_2D;
}

float evalProfileSDF(uint root, vec2 q) {
    uint work_node[IR_PROFILE_STACK_CAPACITY];
    int work_state[IR_PROFILE_STACK_CAPACITY];
    float vstack[IR_PROFILE_STACK_CAPACITY];
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
        if (state == 0 && irIsProfileCombinator(n.type)) {
            work_node[wsp] = idx;
            work_state[wsp] = 1;
            wsp++;
            for (int c = int(n.child_count) - 1; c >= 0; c--) {
                work_node[wsp] = u_children[n.child_offset + uint(c)];
                work_state[wsp] = 0;
                wsp++;
            }
        } else if (state == 1) {
            int cc = int(n.child_count);
            int base = vsp - cc;
            uint t = n.type;
            float r;
            if (t == NODE_PROFILE_OFFSET_2D) {
                r = vstack[base];
            } else if (t == NODE_PROFILE_DISTANCE_OFFSET_2D) {
                r = vstack[base] - irP(n.param_offset, 0u);
            } else if (t == NODE_PROFILE_UNION_2D) {
                r = min(vstack[base], vstack[base + 1]);
            } else if (t == NODE_PROFILE_INTERSECTION_2D) {
                r = max(vstack[base], vstack[base + 1]);
            } else if (t == NODE_PROFILE_DIFFERENCE_2D) {
                r = max(vstack[base], -vstack[base + 1]);
            } else {
                float a = vstack[base];
                float bb = vstack[base + 1];
                float k = irP(n.param_offset, 0u);
                float h = clamp(0.5 + 0.5 * (bb - a) / k, 0.0, 1.0);
                r = mix(bb, a, h) - k * h * (1.0 - h);
            }
            vsp = base;
            vstack[vsp] = r;
            vsp++;
        } else {
            vstack[vsp] = irProfileLeafValue(idx, q);
            vsp++;
        }
    }
    return vstack[0];
}

float irProfile1DLeafValue(uint node_index, float t) {
    GpuNode n = u_nodes[node_index];
    uint b = n.param_offset;
    if (n.type == NODE_PROFILE_SEGMENT_1D) {
        return abs(t - irP(b, 0u)) - irP(b, 1u);
    }
    return IR_FAR;
}

bool irIsProfile1DCombinator(uint t) {
    return t == NODE_PROFILE_UNION_1D
        || t == NODE_PROFILE_INTERSECTION_1D
        || t == NODE_PROFILE_DIFFERENCE_1D
        || t == NODE_PROFILE_SMOOTH_UNION_1D;
}

float evalProfile1DSDF(uint root, float t) {
    uint work_node[IR_PROFILE_STACK_CAPACITY];
    int work_state[IR_PROFILE_STACK_CAPACITY];
    float vstack[IR_PROFILE_STACK_CAPACITY];
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
        if (state == 0 && irIsProfile1DCombinator(n.type)) {
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
            uint ty = n.type;
            float r;
            if (ty == NODE_PROFILE_UNION_1D) {
                r = min(vstack[base], vstack[base + 1]);
            } else if (ty == NODE_PROFILE_INTERSECTION_1D) {
                r = max(vstack[base], vstack[base + 1]);
            } else if (ty == NODE_PROFILE_DIFFERENCE_1D) {
                r = max(vstack[base], -vstack[base + 1]);
            } else {
                float a = vstack[base];
                float bb = vstack[base + 1];
                float k = irP(n.param_offset, 0u);
                float h = clamp(0.5 + 0.5 * (bb - a) / k, 0.0, 1.0);
                r = mix(bb, a, h) - k * h * (1.0 - h);
            }
            vsp = base;
            vstack[vsp] = r;
            vsp++;
        } else {
            vstack[vsp] = irProfile1DLeafValue(idx, t);
            vsp++;
        }
    }
    return vstack[0];
}

// ---- leaf handler: placed-2D sections, extrude, revolve, placed-1D ----------
bool irProfileLeaf(GpuNode node, vec3 p, out float dist) {
    uint b = node.param_offset;
    uint t = node.type;

    bool isPlaced2D =
        t == NODE_PLACED_CIRCLE_2D || t == NODE_PLACED_RECTANGLE_2D ||
        t == NODE_PLACED_SQUARE_2D || t == NODE_PLACED_ROUNDED_RECTANGLE_2D ||
        t == NODE_PLACED_ELLIPSE_2D || t == NODE_PLACED_PROFILE_2D ||
        t == NODE_PLACED_POLYLINE_2D || t == NODE_PLACED_BEZIER_CURVE_2D ||
        t == NODE_EXTRUDE_PROFILE_2D;
    if (isPlaced2D) {
        vec3 local3 = p - irP3(b, 0u);
        float u = dot(local3, irP3(b, 3u));
        float v = dot(local3, irP3(b, 6u));
        float plane = dot(local3, irP3(b, 9u));
        vec2 q = vec2(u, v);

        if (t == NODE_PLACED_CIRCLE_2D) {
            float profile = length(q - vec2(irP(b, 12u), irP(b, 13u))) - irP(b, 14u);
            dist = max(profile, abs(plane) - 0.002); return true;
        }
        if (t == NODE_PLACED_RECTANGLE_2D) {
            vec2 d = abs(q - vec2(irP(b, 12u), irP(b, 13u))) - vec2(irP(b, 14u), irP(b, 15u));
            float profile = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
            dist = max(profile, abs(plane) - 0.002); return true;
        }
        if (t == NODE_PLACED_SQUARE_2D) {
            vec2 d = abs(q - vec2(irP(b, 12u), irP(b, 13u))) - vec2(irP(b, 14u));
            float profile = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
            dist = max(profile, abs(plane) - 0.002); return true;
        }
        if (t == NODE_PLACED_ROUNDED_RECTANGLE_2D) {
            float cr = irP(b, 16u);
            vec2 inner = vec2(irP(b, 14u), irP(b, 15u)) - vec2(cr);
            vec2 d = abs(q - vec2(irP(b, 12u), irP(b, 13u))) - inner;
            float profile = length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - cr;
            dist = max(profile, abs(plane) - 0.002); return true;
        }
        if (t == NODE_PLACED_ELLIPSE_2D) {
            float profile = irExactEllipseDistance(q - vec2(irP(b, 12u), irP(b, 13u)),
                                                   vec2(irP(b, 14u), irP(b, 15u)));
            dist = max(profile, abs(plane) - 0.002); return true;
        }
        if (t == NODE_PLACED_PROFILE_2D) {
            float profile = evalProfileSDF(u_children[node.child_offset], q);
            dist = max(profile, abs(plane) - 0.002); return true;
        }
        if (t == NODE_PLACED_POLYLINE_2D) {
            uint pc = (node.param_count - 12u) / 2u;
            float d = IR_FAR;
            for (uint i = 0u; i + 1u < pc; i++) {
                vec2 a = vec2(irP(b, 12u + i * 2u), irP(b, 12u + i * 2u + 1u));
                vec2 bb = vec2(irP(b, 12u + (i + 1u) * 2u), irP(b, 12u + (i + 1u) * 2u + 1u));
                d = min(d, irSegmentDistance2D(q, a, bb));
            }
            dist = max(d - 0.004, abs(plane) - 0.002); return true;
        }
        if (t == NODE_PLACED_BEZIER_CURVE_2D) {
            uint pc = (node.param_count - 12u) / 2u;
            float d = IR_FAR;
            for (uint i = 0u; i + 2u < pc; i += 2u) {
                vec2 a = vec2(irP(b, 12u + i * 2u), irP(b, 12u + i * 2u + 1u));
                vec2 bb = vec2(irP(b, 12u + (i + 1u) * 2u), irP(b, 12u + (i + 1u) * 2u + 1u));
                vec2 c = vec2(irP(b, 12u + (i + 2u) * 2u), irP(b, 12u + (i + 2u) * 2u + 1u));
                d = min(d, irQuadraticBezierDistance(q, a, bb, c));
            }
            dist = max(d - 0.004, abs(plane) - 0.002); return true;
        }
        // NODE_EXTRUDE_PROFILE_2D
        float profile = evalProfileSDF(u_children[node.child_offset], q);
        float axial = abs(plane - irP(b, 13u)) - irP(b, 12u) * 0.5;
        vec2 pair = vec2(profile, axial);
        dist = length(max(pair, vec2(0.0))) + min(max(profile, axial), 0.0);
        return true;
    }

    if (t == NODE_REVOLVE_PROFILE_2D) {
        vec3 local = p - irP3(b, 12u);
        vec3 axis_dir = irP3(b, 15u);
        vec3 radial_dir = irP3(b, 18u);
        vec3 tangent_dir = irP3(b, 21u);
        float axial = dot(local, axis_dir);
        float radial_x = dot(local, radial_dir);
        float radial_y = dot(local, tangent_dir);
        float radial = sqrt(max(radial_x * radial_x + radial_y * radial_y, 0.0));
        vec3 sample_point = irP3(b, 12u) + axial * axis_dir + radial * radial_dir;
        vec3 section_local = sample_point - irP3(b, 0u);
        float su = dot(section_local, irP3(b, 3u));
        float sv = dot(section_local, irP3(b, 6u));
        float profile = evalProfileSDF(u_children[node.child_offset], vec2(su, sv));
        float angle = irP(b, 24u);
        if (angle >= 6.283184307179586) { dist = profile; return true; }
        float angular = irAngularSectorSDF(vec2(radial_x, radial_y), angle);
        vec2 pair = vec2(profile, angular);
        dist = length(max(pair, vec2(0.0))) + min(max(profile, angular), 0.0);
        return true;
    }

    if (t == NODE_PLACED_PROFILE_1D) {
        vec3 origin = irP3(b, 0u);
        vec3 axis_u = irP3(b, 3u);
        vec3 local = p - origin;
        float tt = dot(local, axis_u);
        vec3 radial = local - tt * axis_u;
        float profile = evalProfile1DSDF(u_children[node.child_offset], tt);
        dist = max(profile, length(radial) - 0.004);
        return true;
    }

    return false;
}

#endif  // FEATURE_PROFILES
