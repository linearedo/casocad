#version 330

out vec4 frag_color;

uniform vec2 u_resolution;
uniform vec3 u_camera_position;
uniform vec3 u_camera_target;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform float u_focal_length;
uniform bool u_show_components;
uniform float u_surface_opacity;
uniform vec3 u_background_color;
uniform bool u_show_grid;
uniform bool u_render_preview_layer;
uniform float u_grid_spacing;
uniform int u_grid_plane;
uniform int u_scene_selected_object_id;
uniform bool u_boundary_selection_active;
uniform int u_boundary_hover_owner_id;
uniform int u_boundary_hover_node_index;
uniform vec3 u_boundary_hover_normal;
uniform int u_selected_boundary_count;
uniform int u_selected_boundary_owner_ids[128];
uniform int u_selected_boundary_node_indices[128];
uniform int u_selected_boundary_whole_flags[128];
uniform vec3 u_selected_boundary_normals[128];
const vec3 PREVIEW_COLOR = vec3(0.15, 0.92, 1.0);
const float PI = 3.14159265359;

float quadraticBezierDistance(vec2 pos, vec2 A, vec2 B, vec2 C) {
    vec2 a = B - A;
    vec2 b = A - 2.0 * B + C;
    vec2 c = a * 2.0;
    vec2 d = A - pos;
    float b_dot_b = dot(b, b);
    if (b_dot_b <= 1.0e-12) {
        vec2 pa = pos - A;
        vec2 ba = C - A;
        float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1.0e-12), 0.0, 1.0);
        return length(pa - ba * h);
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

float segmentDistance3D(vec3 pos, vec3 A, vec3 B) {
    vec3 pa = pos - A;
    vec3 ba = B - A;
    float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1.0e-12), 0.0, 1.0);
    return length(pa - ba * h);
}

float flatCappedSegmentTubeSDF3D(vec3 pos, vec3 A, vec3 B, float radius) {
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

float quadraticBezierDistance3D(vec3 pos, vec3 A, vec3 B, vec3 C) {
    vec3 a = B - A;
    vec3 b = A - 2.0 * B + C;
    vec3 c = a * 2.0;
    vec3 d = A - pos;
    float b_dot_b = dot(b, b);
    if (b_dot_b <= 1.0e-12) {
        return segmentDistance3D(pos, A, C);
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

bool quadraticBezierRayCrosses(vec2 pos, vec2 A, vec2 B, vec2 C) {
    float qa = A.y - 2.0 * B.y + C.y;
    float qb = 2.0 * (B.y - A.y);
    float qc = A.y - pos.y;
    bool crosses = false;

    if (abs(qa) <= 1.0e-8) {
        if (abs(qb) <= 1.0e-8) {
            return false;
        }
        float t = -qc / qb;
        if (t >= 0.0 && t < 1.0) {
            vec2 q = mix(mix(A, B, t), mix(B, C, t), t);
            crosses = q.x > pos.x;
        }
        return crosses;
    }

    float h = qb * qb - 4.0 * qa * qc;
    if (h <= 1.0e-8) {
        return false;
    }
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

float quadraticBezierRayCrossValue(vec2 pos, vec2 A, vec2 B, vec2 C) {
    return quadraticBezierRayCrosses(pos, A, B, C) ? 1.0 : 0.0;
}

bool segmentRayCrosses(vec2 pos, vec2 A, vec2 B) {
    float denominator = B.y - A.y;
    denominator = abs(denominator) < 1.0e-8
        ? (denominator < 0.0 ? -1.0e-8 : 1.0e-8)
        : denominator;
    return (
        ((A.y > pos.y) != (B.y > pos.y))
        && (
            pos.x
            < (B.x - A.x) * (pos.y - A.y) / denominator + A.x
        )
    );
}

float segmentRayCrossValue(vec2 pos, vec2 A, vec2 B) {
    return segmentRayCrosses(pos, A, B) ? 1.0 : 0.0;
}

float exactEllipseDistance(vec2 p, vec2 ab) {
    p = abs(p);
    if (p.x > p.y) {
        p = p.yx;
        ab = ab.yx;
    }
    if (abs(ab.x - ab.y) <= 1.0e-6) {
        return length(p) - ab.x;
    }

    float l = ab.y * ab.y - ab.x * ab.x;
    float m = ab.x * p.x / l;
    float n = ab.y * p.y / l;
    float m2 = m * m;
    float n2 = n * n;
    float c = (m2 + n2 - 1.0) / 3.0;
    float c3 = c * c * c;
    float q = c3 + m2 * n2 * 2.0;
    float d = c3 + m2 * n2;
    float g = m + m * n2;
    float co;

    if (d < 0.0) {
        float h = acos(clamp(q / c3, -1.0, 1.0)) / 3.0;
        float s = cos(h);
        float t = sin(h) * sqrt(3.0);
        float rx = sqrt(max(-c * (s + t + 2.0) + m2, 0.0));
        float ry = sqrt(max(-c * (s - t + 2.0) + m2, 0.0));
        co = (
            ry
            + sign(l) * rx
            + abs(g) / max(rx * ry, 1.0e-12)
            - m
        ) * 0.5;
    } else {
        float h = 2.0 * m * n * sqrt(max(d, 0.0));
        float s = sign(q + h) * pow(abs(q + h), 1.0 / 3.0);
        float u = sign(q - h) * pow(abs(q - h), 1.0 / 3.0);
        float rx = -s - u - c * 4.0 + 2.0 * m2;
        float ry = (s - u) * sqrt(3.0);
        float rm = sqrt(rx * rx + ry * ry);
        co = (
            ry / sqrt(max(rm - rx, 1.0e-12))
            + 2.0 * g / max(rm, 1.0e-12)
            - m
        ) * 0.5;
    }

    co = clamp(co, 0.0, 1.0);
    vec2 r = ab * vec2(co, sqrt(max(1.0 - co * co, 0.0)));
    return length(r - p) * sign(p.y - r.y);
}

float pyramidUnitSDF(vec3 p, float h) {
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

float cappedConeSDF(vec3 p, float h, float r1, float r2) {
    vec2 q = vec2(length(p.xy), p.z);
    vec2 k1 = vec2(r2, h);
    vec2 k2 = vec2(r2 - r1, 2.0 * h);
    vec2 ca = vec2(q.x - min(q.x, (q.y < 0.0) ? r1 : r2), abs(q.y) - h);
    vec2 cb = q - k1 + k2 * clamp(dot(k1 - q, k2) / dot(k2, k2), 0.0, 1.0);
    float s = (cb.x < 0.0 && ca.y < 0.0) ? -1.0 : 1.0;
    return s * sqrt(min(dot(ca, ca), dot(cb, cb)));
}

float rayDistance2D(vec2 p, vec2 ray) {
    float projection = dot(p, ray);
    float cross_value = ray.x * p.y - ray.y * p.x;
    return projection >= 0.0 ? abs(cross_value) : length(p);
}

float angularSectorSDF(vec2 p, float angle) {
    if (angle >= 6.283184) {
        return -1000000.0;
    }
    float theta = atan(p.y, p.x);
    theta = theta < 0.0 ? theta + 6.28318530718 : theta;
    bool inside = theta <= angle;
    vec2 end_ray = vec2(cos(angle), sin(angle));
    float distance_to_edge = min(
        rayDistance2D(p, vec2(1.0, 0.0)),
        rayDistance2D(p, end_ray)
    );
    return inside ? -distance_to_edge : distance_to_edge;
}

float boxFrameSDF(vec3 p, vec3 b, float e) {
    p = abs(p) - b;
    vec3 q = abs(p + e) - e;
    return min(
        min(
            length(max(vec3(p.x, q.y, q.z), 0.0))
                + min(max(p.x, max(q.y, q.z)), 0.0),
            length(max(vec3(q.x, p.y, q.z), 0.0))
                + min(max(q.x, max(p.y, q.z)), 0.0)
        ),
        length(max(vec3(q.x, q.y, p.z), 0.0))
            + min(max(q.x, max(q.y, p.z)), 0.0)
    );
}

/*__SCENE_SDF__*/
