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
uniform float u_grid_spacing;
uniform int u_grid_plane;
uniform bool u_boundary_selection_active;
uniform int u_boundary_hover_owner_id;
uniform int u_boundary_hover_direction;
uniform vec3 u_boundary_hover_normal;
uniform int u_scene_hover_object_id;
uniform int u_scene_selected_object_id;
uniform int u_preview_kind;
uniform vec3 u_preview_start;
uniform vec3 u_preview_current;
uniform vec3 u_preview_move_delta;
const int MAX_PREVIEW_ROTATIONS = 8;
uniform int u_preview_rotation_count;
uniform int u_preview_rotation_axes[MAX_PREVIEW_ROTATIONS];
uniform float u_preview_rotation_angles[MAX_PREVIEW_ROTATIONS];
uniform vec3 u_preview_rotation_pivots[MAX_PREVIEW_ROTATIONS];
uniform bool u_preview_cursor_active;
uniform vec3 u_preview_cursor;
uniform float u_preview_torus_minor_radius;
uniform int u_preview_point_count;
uniform bool u_preview_polygon_closed;
uniform vec3 u_preview_points[32];
const int MAX_SELECTED_BOUNDARY_OWNERS = 128;
uniform int u_selected_boundary_region_count;
uniform ivec2 u_selected_boundary_regions[MAX_SELECTED_BOUNDARY_OWNERS];
uniform vec3 u_selected_boundary_normals[MAX_SELECTED_BOUNDARY_OWNERS];
const vec3 SELECTION_OVERLAY_COLOR = vec3(0.15, 0.92, 1.0);
const float SELECTION_OVERLAY_ALPHA = 0.92;
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
    float ky = kk * (2.0 * dot(a, a) + dot(d, b)) / 3.0;
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

/*__SCENE_SDF__*/

bool movePreviewActive() {
    return (
        u_scene_selected_object_id != 0
        && length(u_preview_move_delta) > 0.000001
    );
}

bool rotationPreviewActive() {
    return (
        u_scene_selected_object_id != 0
        && u_preview_rotation_count > 0
    );
}

vec3 inverseRotationStepPoint(vec3 p, int axis, float angle, vec3 pivot) {
    float c = cos(angle);
    float s = sin(angle);
    vec3 local = p - pivot;
    vec3 rotated;
    if (axis == 0) {
        rotated = vec3(local.x, c * local.y + s * local.z, -s * local.y + c * local.z);
    } else if (axis == 1) {
        rotated = vec3(c * local.x - s * local.z, local.y, s * local.x + c * local.z);
    } else {
        rotated = vec3(c * local.x + s * local.y, -s * local.x + c * local.y, local.z);
    }
    return pivot + rotated;
}

vec3 inverseRotationPreviewPoint(vec3 p) {
    vec3 sample_point = p;
    for (int offset = 0; offset < MAX_PREVIEW_ROTATIONS; ++offset) {
        int index = u_preview_rotation_count - 1 - offset;
        if (index < 0) {
            break;
        }
        sample_point = inverseRotationStepPoint(
            sample_point,
            u_preview_rotation_axes[index],
            u_preview_rotation_angles[index],
            u_preview_rotation_pivots[index]
        );
    }
    return sample_point;
}

vec3 previewTransformPoint(vec3 p) {
    vec3 sample_point = rotationPreviewActive()
        ? inverseRotationPreviewPoint(p)
        : p;
    return movePreviewActive() ? sample_point - u_preview_move_delta : sample_point;
}

float renderSceneSDF(vec3 p) {
    return sceneSDF(p);
}

vec3 sceneNormal(vec3 p) {
    const float e = 0.0008;
    vec2 h = vec2(e, 0.0);
    return normalize(vec3(
        renderSceneSDF(p + h.xyy) - renderSceneSDF(p - h.xyy),
        renderSceneSDF(p + h.yxy) - renderSceneSDF(p - h.yxy),
        renderSceneSDF(p + h.yyx) - renderSceneSDF(p - h.yyx)
    ));
}

float softShadow(vec3 origin, vec3 direction) {
    float result = 1.0;
    float distance_travelled = 0.02;
    for (int step = 0; step < 48; ++step) {
        float distance_to_scene = renderSceneSDF(
            origin + direction * distance_travelled
        );
        result = min(result, 12.0 * distance_to_scene / distance_travelled);
        distance_travelled += clamp(distance_to_scene, 0.01, 0.15);
        if (distance_to_scene < 0.0005 || distance_travelled > 8.0) {
            break;
        }
    }
    return clamp(result, 0.18, 1.0);
}

vec3 objectPalette(int object_id) {
    int slot = (max(object_id, 1) - 1) % 10;
    if (slot == 0) return vec3(0.95, 0.28, 0.16);
    if (slot == 1) return vec3(0.20, 0.72, 1.00);
    if (slot == 2) return vec3(0.25, 0.88, 0.38);
    if (slot == 3) return vec3(1.00, 0.76, 0.16);
    if (slot == 4) return vec3(0.82, 0.36, 1.00);
    if (slot == 5) return vec3(0.12, 0.90, 0.82);
    if (slot == 6) return vec3(1.00, 0.42, 0.68);
    if (slot == 7) return vec3(0.58, 0.86, 0.18);
    if (slot == 8) return vec3(1.00, 0.52, 0.12);
    return vec3(0.46, 0.55, 1.00);
}

float gridLineCoverage(float coordinate, float spacing) {
    float distance_to_line =
        abs(fract(coordinate / spacing + 0.5) - 0.5) * spacing;
    float pixel_width = max(fwidth(coordinate), 0.000001);
    return 1.0 - smoothstep(pixel_width * 0.55, pixel_width * 1.45, distance_to_line);
}

float gridAxisCoverage(float coordinate) {
    float pixel_width = max(fwidth(coordinate), 0.000001);
    return 1.0 - smoothstep(pixel_width * 0.55, pixel_width * 1.45, abs(coordinate));
}

vec3 gridBackground(vec3 ray_direction, vec3 background) {
    int normal_axis = u_grid_plane == 2 ? 0 : (u_grid_plane == 1 ? 1 : 2);
    float origin_axis = u_camera_position[normal_axis];
    float direction_axis = ray_direction[normal_axis];
    if (!u_show_grid || abs(direction_axis) < 0.000001) {
        return background;
    }
    float travel = -origin_axis / direction_axis;
    if (travel <= 0.0) {
        return background;
    }

    vec3 world_point = u_camera_position + ray_direction * travel;
    vec2 point = (
        u_grid_plane == 2
        ? world_point.yz
        : (u_grid_plane == 1 ? world_point.xz : world_point.xy)
    );
    float minor = max(
        gridLineCoverage(point.x, u_grid_spacing),
        gridLineCoverage(point.y, u_grid_spacing)
    );
    float major = max(
        gridLineCoverage(point.x, u_grid_spacing * 5.0),
        gridLineCoverage(point.y, u_grid_spacing * 5.0)
    );
    float x_axis = gridAxisCoverage(point.y);
    float y_axis = gridAxisCoverage(point.x);
    float fade = clamp(1.0 / (1.0 + travel * travel * 0.002), 0.0, 1.0);

    vec3 color = mix(vec3(0.26, 0.30, 0.38), vec3(0.40, 0.46, 0.56), major);
    vec3 first_axis_color = (
        u_grid_plane == 2 ? vec3(0.0, 1.0, 0.0) : vec3(0.92, 0.24, 0.20)
    );
    vec3 second_axis_color = (
        u_grid_plane == 0 ? vec3(0.0, 1.0, 0.0) : vec3(0.1, 0.45, 1.0)
    );
    color = mix(color, first_axis_color, x_axis);
    color = mix(color, second_axis_color, y_axis);
    float coverage = max(max(minor * 0.55, major * 0.78), max(x_axis, y_axis));
    return mix(background, color, coverage * fade);
}

vec3 componentNormal(vec3 p, int component) {
    const float e = 0.0008;
    vec2 h = vec2(e, 0.0);
    return normalize(vec3(
        componentSDF(p + h.xyy, component) - componentSDF(p - h.xyy, component),
        componentSDF(p + h.yxy, component) - componentSDF(p - h.yxy, component),
        componentSDF(p + h.yyx, component) - componentSDF(p - h.yyx, component)
    ));
}

bool marchComponent(
    vec3 ray_origin,
    vec3 ray_direction,
    int component,
    out vec3 hit_point,
    out float hit_travel
) {
    hit_travel = 0.0;
    for (int step = 0; step < 128; ++step) {
        hit_point = ray_origin + ray_direction * hit_travel;
        float distance_to_component = componentSDF(hit_point, component);
        if (distance_to_component < 0.0008) {
            return true;
        }
        hit_travel += max(distance_to_component, 0.0002);
        if (hit_travel > 100.0) {
            break;
        }
    }
    return false;
}

bool marchSelectedObject(
    vec3 ray_origin,
    vec3 ray_direction,
    out vec3 hit_point,
    out float hit_travel
) {
    hit_travel = 0.0;
    for (int step = 0; step < 128; ++step) {
        hit_point = ray_origin + ray_direction * hit_travel;
        vec3 sample_point = previewTransformPoint(hit_point);
        float distance_to_object = sceneSelectedObjectSDF(
            sample_point,
            u_scene_selected_object_id
        );
        if (distance_to_object < 0.0008) {
            return true;
        }
        hit_travel += max(distance_to_object, 0.0002);
        if (hit_travel > 100.0) {
            break;
        }
    }
    return false;
}

vec3 selectedObjectNormal(vec3 p) {
    const float e = 0.0008;
    vec2 h = vec2(e, 0.0);
    vec3 sample_point = previewTransformPoint(p);
    return normalize(vec3(
        sceneSelectedObjectSDF(sample_point + h.xyy, u_scene_selected_object_id)
            - sceneSelectedObjectSDF(
                sample_point - h.xyy,
                u_scene_selected_object_id
            ),
        sceneSelectedObjectSDF(sample_point + h.yxy, u_scene_selected_object_id)
            - sceneSelectedObjectSDF(
                sample_point - h.yxy,
                u_scene_selected_object_id
            ),
        sceneSelectedObjectSDF(sample_point + h.yyx, u_scene_selected_object_id)
            - sceneSelectedObjectSDF(
                sample_point - h.yyx,
                u_scene_selected_object_id
            )
    ));
}

float box2SDF(vec2 q, vec2 half_size) {
    vec2 d = abs(q) - half_size;
    return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
}

float box3SDF(vec3 q, vec3 half_size) {
    vec3 d = abs(q) - half_size;
    return length(max(d, vec3(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

float roundedBox2SDF(vec2 q, vec2 half_size, float radius) {
    vec2 inner = half_size - vec2(radius);
    vec2 d = abs(q) - inner;
    return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0) - radius;
}

float regularPolygonSDF(vec2 q, float radius) {
    float result = -1000000.0;
    for (int index = 0; index < 6; ++index) {
        float first_angle = float(index) * PI / 3.0;
        float second_angle = float(index + 1) * PI / 3.0;
        vec2 first = radius * vec2(cos(first_angle), sin(first_angle));
        vec2 second = radius * vec2(cos(second_angle), sin(second_angle));
        vec2 edge = second - first;
        vec2 normal = normalize(vec2(edge.y, -edge.x));
        result = max(result, dot(q - first, normal));
    }
    return result;
}

vec2 referencePlaneCoordinates(vec3 value) {
    if (u_grid_plane == 2) return value.yz;
    if (u_grid_plane == 1) return value.xz;
    return value.xy;
}

float referencePlaneDistance(vec3 value) {
    if (u_grid_plane == 2) return value.x;
    if (u_grid_plane == 1) return value.y;
    return value.z;
}

vec3 boxHalfSizeForReferencePlane(vec3 world_drag, float first, float second) {
    vec3 actual_half_size = max(0.5 * abs(world_drag), vec3(0.05));
    if (
        abs(world_drag.x) > 0.000001
        && abs(world_drag.y) > 0.000001
        && abs(world_drag.z) > 0.000001
    ) {
        return actual_half_size;
    }
    float fallback = max(first, second);
    if (u_grid_plane == 2) return vec3(fallback, first, second);
    if (u_grid_plane == 1) return vec3(first, fallback, second);
    return vec3(first, second, fallback);
}

float createPreviewSDF(vec3 p) {
    vec3 center = 0.5 * (u_preview_start + u_preview_current);
    vec3 local = p - center;
    vec2 drag =
        referencePlaneCoordinates(u_preview_current)
        - referencePlaneCoordinates(u_preview_start);
    vec2 local_plane = referencePlaneCoordinates(local);
    float plane_distance = abs(referencePlaneDistance(local));
    float extent_x = max(abs(drag.x) * 0.5, 0.05);
    float extent_y = max(abs(drag.y) * 0.5, 0.05);
    float radius = max(length(drag) * 0.5, 0.05);

    if (u_preview_kind == 12 || u_preview_kind == 13 || u_preview_kind == 14) {
        if (u_preview_point_count < 2) {
            return 1000000.0;
        }
        vec2 q = referencePlaneCoordinates(p);
        float plane_distance_points = abs(referencePlaneDistance(p - u_preview_points[0]));
        float distance_to_edges = 1000000.0;
        if (u_preview_kind == 14) {
            if (u_preview_point_count < 3) {
                return 1000000.0;
            }
            for (int index = 0; index < 30; index += 2) {
                if (index + 2 >= u_preview_point_count) {
                    break;
                }
                vec2 a = referencePlaneCoordinates(u_preview_points[index]);
                vec2 b = referencePlaneCoordinates(u_preview_points[index + 1]);
                vec2 c = referencePlaneCoordinates(u_preview_points[index + 2]);
                distance_to_edges = min(
                    distance_to_edges,
                    quadraticBezierDistance(q, a, b, c)
                );
            }
            return max(distance_to_edges - 0.004, plane_distance_points - 0.002);
        }
        for (int index = 0; index < 31; ++index) {
            if (index + 1 >= u_preview_point_count) {
                break;
            }
            vec2 a = referencePlaneCoordinates(u_preview_points[index]);
            vec2 b = referencePlaneCoordinates(u_preview_points[index + 1]);
            vec2 pa = q - a;
            vec2 ba = b - a;
            float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1.0e-12), 0.0, 1.0);
            distance_to_edges = min(distance_to_edges, length(pa - ba * h));
        }
        bool closed = u_preview_polygon_closed && u_preview_point_count >= 3;
        if (closed) {
            vec2 a = referencePlaneCoordinates(u_preview_points[u_preview_point_count - 1]);
            vec2 b = referencePlaneCoordinates(u_preview_points[0]);
            vec2 pa = q - a;
            vec2 ba = b - a;
            float h = clamp(dot(pa, ba) / max(dot(ba, ba), 1.0e-12), 0.0, 1.0);
            distance_to_edges = min(distance_to_edges, length(pa - ba * h));

            bool inside = false;
            for (int index = 0; index < 32; ++index) {
                if (index >= u_preview_point_count) {
                    break;
                }
                int next_index = index + 1;
                if (next_index >= u_preview_point_count) {
                    next_index = 0;
                }
                a = referencePlaneCoordinates(u_preview_points[index]);
                b = referencePlaneCoordinates(u_preview_points[next_index]);
                float denominator = b.y - a.y;
                denominator = abs(denominator) < 1.0e-8 ? 1.0e-8 : denominator;
                bool crosses = (
                    ((a.y > q.y) != (b.y > q.y))
                    && (q.x < (b.x - a.x) * (q.y - a.y) / denominator + a.x)
                );
                inside = inside != crosses;
            }
            float profile = inside ? -distance_to_edges : distance_to_edges;
            return max(profile, plane_distance_points - 0.002);
        }
        return max(distance_to_edges - 0.004, plane_distance_points - 0.002);
    }

    if (u_preview_kind == 1) {
        vec3 direction = u_preview_current - u_preview_start;
        float length_direction = length(direction);
        vec3 axis = length_direction > 0.000001
            ? direction / length_direction
            : vec3(1.0, 0.0, 0.0);
        float coordinate = dot(p - center, axis);
        float radial = length((p - center) - coordinate * axis);
        return max(
            abs(coordinate) - max(0.5 * length_direction, 0.05),
            radial - 0.004
        );
    }

    float profile = 1000000.0;
    if (u_preview_kind == 2) {
        profile = length(local_plane) - radius;
    } else if (u_preview_kind == 3) {
        profile = box2SDF(local_plane, vec2(extent_x, extent_y));
    } else if (u_preview_kind == 4) {
        profile = box2SDF(local_plane, vec2(max(extent_x, extent_y)));
    } else if (u_preview_kind == 5) {
        vec2 half_size = vec2(extent_x, extent_y);
        profile = roundedBox2SDF(
            local_plane,
            half_size,
            max(0.01, min(half_size.x, half_size.y) * 0.2)
        );
    } else if (u_preview_kind == 6) {
        vec2 axes = vec2(extent_x, extent_y);
        profile = (length(local_plane / axes) - 1.0) * min(axes.x, axes.y);
    } else if (u_preview_kind == 7) {
        profile = regularPolygonSDF(local_plane, radius);
    }
    if (u_preview_kind >= 2 && u_preview_kind <= 7) {
        return max(profile, plane_distance - 0.002);
    }
    if (u_preview_kind == 8) {
        return length(local) - radius;
    }
    if (u_preview_kind == 9) {
        vec3 world_drag = u_preview_current - u_preview_start;
        return box3SDF(
            local,
            boxHalfSizeForReferencePlane(world_drag, extent_x, extent_y)
        );
    }
    if (u_preview_kind == 10) {
        vec3 world_drag = u_preview_current - u_preview_start;
        float cylinder_radius = max(0.5 * length(world_drag.xy), 0.05);
        float cylinder_half_height = (
            abs(world_drag.z) > 0.000001
            ? max(0.5 * abs(world_drag.z), 0.05)
            : max(extent_x, extent_y)
        );
        vec2 d = abs(vec2(length(local.xy), local.z))
            - vec2(cylinder_radius, cylinder_half_height);
        return min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)));
    }
    if (u_preview_kind == 11) {
        float torus_minor_radius = (
            u_preview_torus_minor_radius > 0.0
            ? u_preview_torus_minor_radius
            : max(radius * 0.25, 0.02)
        );
        return length(vec2(length(local.xy) - radius, local.z))
            - torus_minor_radius;
    }
    return 1000000.0;
}

bool createPreviewActive() {
    return u_preview_kind > 0;
}

bool cursorPreviewActive() {
    return u_preview_cursor_active;
}

float referencePlaneMarkerSDF(vec3 p, vec3 anchor, float scale) {
    vec3 local = p - anchor;
    vec2 q = referencePlaneCoordinates(local);
    float plane_distance = abs(referencePlaneDistance(local));
    float arm = clamp(u_grid_spacing * 0.28 * scale, 0.03, 0.16);
    float thickness = clamp(u_grid_spacing * 0.035 * scale, 0.004, 0.02);
    float first_bar = box2SDF(q, vec2(arm, thickness));
    float second_bar = box2SDF(q, vec2(thickness, arm));
    return max(min(first_bar, second_bar), plane_distance - thickness);
}

float cursorPreviewSDF(vec3 p) {
    return referencePlaneMarkerSDF(p, u_preview_cursor, 1.0);
}

float createControlPointSDF(vec3 p) {
    float start_marker = referencePlaneMarkerSDF(p, u_preview_start, 0.78);
    float current_marker = referencePlaneMarkerSDF(p, u_preview_current, 0.78);
    return min(start_marker, current_marker);
}

float previewSDF(vec3 p) {
    float distance_to_preview = 1000000.0;
    if (cursorPreviewActive()) {
        distance_to_preview = min(distance_to_preview, cursorPreviewSDF(p));
    }
    if (createPreviewActive()) {
        distance_to_preview = min(distance_to_preview, createPreviewSDF(p));
        distance_to_preview = min(distance_to_preview, createControlPointSDF(p));
    }
    if (movePreviewActive() || rotationPreviewActive()) {
        distance_to_preview = min(
            distance_to_preview,
            sceneSelectedObjectSDF(
                previewTransformPoint(p),
                u_scene_selected_object_id
            )
        );
    }
    return distance_to_preview;
}

bool marchPreview(
    vec3 ray_origin,
    vec3 ray_direction,
    out vec3 hit_point,
    out float hit_travel
) {
    hit_travel = 0.0;
    for (int step = 0; step < 128; ++step) {
        hit_point = ray_origin + ray_direction * hit_travel;
        float distance_to_preview = previewSDF(hit_point);
        if (distance_to_preview < 0.0008) {
            return true;
        }
        hit_travel += max(distance_to_preview, 0.0002);
        if (hit_travel > 100.0) {
            break;
        }
    }
    return false;
}

vec3 previewNormal(vec3 p) {
    const float e = 0.0008;
    vec2 h = vec2(e, 0.0);
    return normalize(vec3(
        previewSDF(p + h.xyy) - previewSDF(p - h.xyy),
        previewSDF(p + h.yxy) - previewSDF(p - h.yxy),
        previewSDF(p + h.yyx) - previewSDF(p - h.yyx)
    ));
}

vec3 previewSurfaceColor(vec3 point, vec3 ray_direction) {
    vec3 normal = previewNormal(point);
    vec3 light_direction = normalize(vec3(0.7, 1.0, 0.45));
    float diffuse = max(dot(normal, light_direction), 0.0);
    float facing = abs(dot(normal, -ray_direction));
    float rim = pow(1.0 - facing, 0.55);
    vec3 color =
        PREVIEW_COLOR * (0.32 + 0.68 * diffuse)
        + PREVIEW_COLOR * (0.45 * rim)
        + vec3(0.04, 0.10, 0.11);
    return pow(color, vec3(0.86));
}

bool marchNextSceneSurface(
    vec3 ray_origin,
    vec3 ray_direction,
    float start_travel,
    out vec3 hit_point,
    out float hit_travel
) {
    hit_travel = start_travel;
    for (int step = 0; step < 160; ++step) {
        hit_point = ray_origin + ray_direction * hit_travel;
        float distance_to_scene = renderSceneSDF(hit_point);
        if (abs(distance_to_scene) < 0.0008) {
            return true;
        }
        hit_travel += max(abs(distance_to_scene), 0.001);
        if (hit_travel > 100.0) {
            break;
        }
    }
    return false;
}

bool boundaryDirectionMatches(vec3 normal) {
    if (u_boundary_hover_direction < 0) return true;
    if (length(u_boundary_hover_normal) < 0.5) return false;
    return dot(normal, normalize(u_boundary_hover_normal)) > 0.90;
}

bool selectedBoundaryRegionMatches(int boundary_owner_id, vec3 normal) {
    for (int index = 0; index < MAX_SELECTED_BOUNDARY_OWNERS; ++index) {
        if (index >= u_selected_boundary_region_count) {
            break;
        }
        ivec2 selector = u_selected_boundary_regions[index];
        if (
            selector.x == boundary_owner_id
            && (
                selector.y == 1
                || (
                    length(u_selected_boundary_normals[index]) > 0.5
                    && dot(
                        normal,
                        normalize(u_selected_boundary_normals[index])
                    ) > 0.90
                )
            )
        ) {
            return true;
        }
    }
    return false;
}

vec3 selectionOverlayColor(vec3 normal, vec3 ray_direction) {
    vec3 light_direction = normalize(vec3(0.7, 1.0, 0.45));
    float diffuse = max(dot(normal, light_direction), 0.0);
    float facing = abs(dot(normal, -ray_direction));
    float rim = pow(1.0 - facing, 0.65);
    vec3 selected_color =
        SELECTION_OVERLAY_COLOR * (0.42 + 0.58 * diffuse)
        + SELECTION_OVERLAY_COLOR * (0.28 * rim)
        + vec3(0.12, 0.20, 0.20);
    return pow(selected_color, vec3(0.86));
}

vec3 shadeSceneSurface(vec3 point, vec3 ray_direction) {
    vec3 normal = sceneNormal(point);
    vec3 light_direction = normalize(vec3(0.7, 1.0, 0.45));
    float diffuse = max(dot(normal, light_direction), 0.0);
    float shadow = softShadow(point + normal * 0.002, light_direction);
    float rim = pow(1.0 - abs(dot(normal, -ray_direction)), 2.5);
    int boundary_owner_id = sceneBoundaryOwnerId(point);
    int material_id = (
        u_boundary_selection_active
        ? boundary_owner_id
        : sceneObjectId(point)
    );
    vec3 material = objectPalette(material_id);
    bool boundary_hovered = (
        u_boundary_selection_active
        && boundary_owner_id == u_boundary_hover_owner_id
        && boundaryDirectionMatches(normal)
    );
    bool scene_hovered = (
        !u_boundary_selection_active
        && material_id == u_scene_hover_object_id
    );
    bool scene_selected = (
        !u_boundary_selection_active
        && sceneSelectionOwnsBoundary(
            u_scene_selected_object_id,
            boundary_owner_id
        )
    );
    if (boundary_hovered || scene_hovered) {
        material = vec3(1.0, 0.88, 0.10);
    } else if (scene_selected) {
        material = vec3(0.15, 0.92, 1.0);
    }
    vec3 surface_color = material * (0.20 + 0.80 * diffuse * shadow);
    surface_color += material * (0.34 * rim);
    if (
        boundary_hovered
        || scene_hovered
        || scene_selected
    ) {
        surface_color += vec3(0.25, 0.20, 0.02);
    }
    return pow(surface_color, vec3(0.86));
}

void main() {
    vec2 screen_uv =
        (gl_FragCoord.xy - 0.5 * u_resolution.xy) / max(u_resolution.y, 1.0);

    vec3 forward = normalize(u_camera_target - u_camera_position);
    vec3 right = normalize(u_camera_right);
    vec3 up = normalize(u_camera_up);
    vec3 ray_direction = normalize(
        2.0 * screen_uv.x * right
        + 2.0 * screen_uv.y * up
        + u_focal_length * forward
    );

    float travel = 0.0;
    bool hit = false;
    vec3 point = u_camera_position;
    for (int step = 0; step < 160; ++step) {
        point = u_camera_position + ray_direction * travel;
        float distance_to_scene = renderSceneSDF(point);
        if (distance_to_scene < 0.0008) {
            hit = true;
            break;
        }
        travel += max(distance_to_scene, 0.0002);
        if (travel > 100.0) {
            break;
        }
    }

    vec3 background = u_background_color;
    background = gridBackground(ray_direction, background);
    vec3 color = background;
    bool selected_boundary_overlay = false;
    vec3 selected_boundary_overlay_color = vec3(0.0);

    if (hit) {
        vec3 normal = sceneNormal(point);
        vec3 surface_color = shadeSceneSurface(point, ray_direction);
        color = mix(background, surface_color, u_surface_opacity);
        bool selected_boundary_surface = (
            !u_boundary_selection_active
            && selectedBoundaryRegionMatches(
                sceneBoundaryOwnerId(point),
                normal
            )
        );
        if (selected_boundary_surface) {
            selected_boundary_overlay = true;
            selected_boundary_overlay_color = selectionOverlayColor(
                normal,
                ray_direction
            );
        }
    }

    bool transparent_view = u_surface_opacity < 0.999;
    if (transparent_view && hit) {
        float layer_start = travel + 0.004;
        for (int layer = 0; layer < 4; ++layer) {
            vec3 layer_point;
            float layer_travel;
            if (!marchNextSceneSurface(
                u_camera_position,
                ray_direction,
                layer_start,
                layer_point,
                layer_travel
            )) {
                break;
            }
            vec3 layer_color = shadeSceneSurface(layer_point, ray_direction);
            float layer_alpha =
                (1.0 - u_surface_opacity) * (0.72 / (1.0 + 0.22 * float(layer)));
            bool selected_boundary_layer = (
                !u_boundary_selection_active
                && selectedBoundaryRegionMatches(
                    sceneBoundaryOwnerId(layer_point),
                    sceneNormal(layer_point)
                )
            );
            if (selected_boundary_layer) {
                if (!selected_boundary_overlay) {
                    selected_boundary_overlay = true;
                    selected_boundary_overlay_color = selectionOverlayColor(
                        sceneNormal(layer_point),
                        ray_direction
                    );
                }
            }
            color = mix(color, layer_color, layer_alpha);
            layer_start = layer_travel + 0.004;
        }
    }

    for (int component = 0; component < COMPONENT_COUNT; ++component) {
        int component_object_id = componentObjectId(component);
        if (u_show_components || transparent_view) {
            vec3 component_point;
            float component_travel;
            if (marchComponent(
                u_camera_position,
                ray_direction,
                component,
                component_point,
                component_travel
            )) {
                bool duplicates_front_surface =
                    hit
                    && abs(component_travel - travel) < 0.01;
                if (
                    transparent_view
                    && duplicates_front_surface
                ) {
                    continue;
                }
                vec3 normal = componentNormal(component_point, component);
                float facing = abs(dot(normal, -ray_direction));
                float silhouette = pow(1.0 - facing, 0.65);
                float depth_fade = hit && component_travel > travel ? 0.78 : 1.0;
                float manual_alpha = 0.035 + 0.90 * silhouette;
                vec3 light_direction = normalize(vec3(0.7, 1.0, 0.45));
                float diffuse = max(dot(normal, light_direction), 0.0);
                vec3 material = objectPalette(component_object_id);
                vec3 component_color =
                    material * (0.30 + 0.70 * diffuse)
                    + material * (0.22 * silhouette);
                component_color = pow(component_color, vec3(0.86));
                float transparent_alpha =
                    (1.0 - u_surface_opacity) * (0.58 + 0.32 * silhouette);
                float alpha = depth_fade * (
                    u_show_components
                    ? max(manual_alpha, transparent_alpha)
                    : transparent_alpha
                );
                color = mix(color, component_color, alpha);
            }
        }
    }

    if (
        cursorPreviewActive()
        || createPreviewActive()
        || movePreviewActive()
        || rotationPreviewActive()
    ) {
        vec3 preview_point;
        float preview_travel;
        if (marchPreview(
            u_camera_position,
            ray_direction,
            preview_point,
            preview_travel
        )) {
            float preview_alpha = hit && preview_travel > travel ? 0.42 : 0.74;
            color = mix(
                color,
                previewSurfaceColor(preview_point, ray_direction),
                preview_alpha
            );
        }
    }

    if (selected_boundary_overlay) {
        color = mix(
            color,
            selected_boundary_overlay_color,
            SELECTION_OVERLAY_ALPHA
        );
    }

    int selected_dimension = sceneSelectedObjectDimension(
        u_scene_selected_object_id
    );
    bool selected_lower_dimensional_object = (
        !u_boundary_selection_active
        && (selected_dimension == 1 || selected_dimension == 2)
    );
    if (selected_lower_dimensional_object) {
        vec3 selected_point;
        float selected_travel;
        if (marchSelectedObject(
            u_camera_position,
            ray_direction,
            selected_point,
            selected_travel
        )) {
            vec3 normal = selectedObjectNormal(selected_point);
            color = mix(
                color,
                selectionOverlayColor(normal, ray_direction),
                SELECTION_OVERLAY_ALPHA
            );
        }
    }

    frag_color = vec4(color, 1.0);
}
