// SDF interpreter SWEEPS chunk (optional, FEATURE_SWEEPS) — design §13.3.
// Polyline / bezier tubes + the 3D centerline helpers. After sdf_core.glsl.
#ifdef FEATURE_SWEEPS

float irSegmentDistance3D(vec3 pos, vec3 A, vec3 B) {
    vec3 pa = pos - A;
    vec3 ba = B - A;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1.0e-12), 0.0, 1.0);
    return length(pa - ba * h);
}

float irQuadraticBezierDistance3D(vec3 pos, vec3 A, vec3 B, vec3 C) {
    vec3 a = B - A;
    vec3 b = A - 2.0 * B + C;
    vec3 c = a * 2.0;
    vec3 d = A - pos;
    float b_dot_b = dot(b, b);
    if (b_dot_b <= 1.0e-12) {
        return irSegmentDistance3D(pos, A, C);
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
        vec3 w = d + (c + b * t) * t;
        res = dot(w, w);
    } else {
        float z = sqrt(max(-p, 0.0));
        float denominator = 2.0 * p * z;
        float angle_argument = denominator == 0.0 ? 0.0 : q / denominator;
        float v = acos(clamp(angle_argument, -1.0, 1.0)) / 3.0;
        float m = cos(v);
        float n = sin(v) * 1.732050808;
        vec3 t = clamp(vec3(m + m, -n - m, n - m) * z - kx, 0.0, 1.0);
        vec3 qx = d + (c + b * t.x) * t.x;
        vec3 qy = d + (c + b * t.y) * t.y;
        res = min(dot(qx, qx), dot(qy, qy));
    }
    return sqrt(max(res, 0.0));
}

float irFlatCappedSegmentTubeSDF3D(vec3 pos, vec3 A, vec3 B, float radius) {
    vec3 ba = B - A;
    float segment_length = length(ba);
    if (segment_length <= 1.0e-8) {
        return length(pos - A) - radius;
    }
    vec3 axis = ba / segment_length;
    vec3 pa = pos - A;
    float projection = dot(pa, axis);
    float radial = length(pa - axis * projection) - radius;
    float axial = abs(projection - 0.5 * segment_length) - 0.5 * segment_length;
    vec2 pair = vec2(radial, axial);
    return length(max(pair, vec2(0.0))) + min(max(radial, axial), 0.0);
}

vec3 irSafeDirection(vec3 preferred, vec3 fallback) {
    float preferred_length = length(preferred);
    if (preferred_length > 1.0e-12) return preferred / preferred_length;
    float fallback_length = max(length(fallback), 1.0e-12);
    return fallback / fallback_length;
}

float irTubeSDF(float centerline_distance, float radius, float inner_radius) {
    float outer = centerline_distance - radius;
    if (inner_radius <= 0.0) return outer;
    return max(outer, inner_radius - centerline_distance);
}

float irFlatTubeSDF(float outer, float centerline_distance, float inner_radius) {
    if (inner_radius <= 0.0) return outer;
    return max(outer, inner_radius - centerline_distance);
}

bool irSweepLeaf(GpuNode node, vec3 p, out float dist) {
    uint b = node.param_offset;
    uint t = node.type;
    if (t != NODE_POLYLINE_TUBE && t != NODE_BEZIER_TUBE) return false;

    uint pc = (node.param_count - 3u) / 3u;
    float radius = irP(b, node.param_count - 3u);
    float inner = irP(b, node.param_count - 2u);
    bool flat_caps = irP(b, node.param_count - 1u) > 0.5;

    if (t == NODE_POLYLINE_TUBE) {
        float centerline = IR_FAR;
        for (uint i = 0u; i + 1u < pc; i++) {
            centerline = min(centerline,
                irSegmentDistance3D(p, irP3(b, i * 3u), irP3(b, (i + 1u) * 3u)));
        }
        if (!flat_caps) { dist = irTubeSDF(centerline, radius, inner); return true; }
        float outer = IR_FAR;
        for (uint i = 0u; i + 1u < pc; i++) {
            outer = min(outer, irFlatCappedSegmentTubeSDF3D(
                p, irP3(b, i * 3u), irP3(b, (i + 1u) * 3u), radius));
        }
        dist = irFlatTubeSDF(outer, centerline, inner);
        return true;
    }

    // bezier tube
    float centerline = IR_FAR;
    for (uint i = 0u; i + 2u < pc; i += 2u) {
        centerline = min(centerline, irQuadraticBezierDistance3D(
            p, irP3(b, i * 3u), irP3(b, (i + 1u) * 3u), irP3(b, (i + 2u) * 3u)));
    }
    if (!flat_caps) { dist = irTubeSDF(centerline, radius, inner); return true; }
    vec3 start = irP3(b, 0u);
    vec3 first_control = irP3(b, 3u);
    vec3 first_end = irP3(b, 6u);
    vec3 last_start = irP3(b, (pc - 3u) * 3u);
    vec3 last_control = irP3(b, (pc - 2u) * 3u);
    vec3 endp = irP3(b, (pc - 1u) * 3u);
    vec3 start_tangent = irSafeDirection(first_control - start, first_end - start);
    vec3 end_tangent = irSafeDirection(endp - last_control, endp - last_start);
    float start_plane = dot(start - p, start_tangent);
    float end_plane = dot(p - endp, end_tangent);
    float outer = max(max(centerline - radius, start_plane), end_plane);
    dist = irFlatTubeSDF(outer, centerline, inner);
    return true;
}

#endif  // FEATURE_SWEEPS
