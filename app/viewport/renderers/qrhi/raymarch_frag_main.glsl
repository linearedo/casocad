// Fragment raymarcher for the QRhi viewport — ONE pass, direct to the render
// target (mirrors the smooth OpenGL fragment path; no compute, no intermediate
// texture, no blit). Appended after the assembled interpreter chunks
// (generated #defines + sdf_core + features), same as raymarch_interpreter.comp.
//
// Uses the SAME camera uniforms as the compute path, so the host's std140 UBO
// packing (uniform_block_members) is unchanged. evalSceneSDF + u_program_length
// come from sdf_core.glsl.

layout(location = 0) out vec4 frag_color;

uniform vec2 u_resolution;
uniform vec3 u_camera_position;
uniform vec3 u_camera_target;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform float u_focal_length;
uniform float u_surface_opacity;
uniform vec3 u_background_color;
// u_show_grid: 0/1   u_grid_plane: 0=XY, 1=XZ, 2=YZ
uniform int u_show_grid;
uniform float u_grid_spacing;
uniform int u_grid_plane;
uniform int u_selected_object_id;

// ---- reference grid (rays that miss geometry hit the grid plane) -----------
float gridLineCoverage(float coordinate, float spacing) {
    float distance_to_line = abs(fract(coordinate / spacing + 0.5) - 0.5) * spacing;
    float pixel_width = max(fwidth(coordinate), 0.000001);
    return 1.0 - smoothstep(pixel_width * 0.55, pixel_width * 1.45, distance_to_line);
}

float gridAxisCoverage(float coordinate) {
    float pixel_width = max(fwidth(coordinate), 0.000001);
    return 1.0 - smoothstep(pixel_width * 0.55, pixel_width * 1.45, abs(coordinate));
}

vec3 gridBackground(vec3 cam_pos, vec3 ray_direction, vec3 background) {
    int normal_axis = u_grid_plane == 2 ? 0 : (u_grid_plane == 1 ? 1 : 2);
    float origin_axis = cam_pos[normal_axis];
    float direction_axis = ray_direction[normal_axis];
    if (u_show_grid == 0 || abs(direction_axis) < 0.000001) return background;
    float travel = -origin_axis / direction_axis;
    if (travel <= 0.0) return background;

    vec3 world_point = cam_pos + ray_direction * travel;
    vec2 point = (
        u_grid_plane == 2 ? world_point.yz
        : (u_grid_plane == 1 ? world_point.xz : world_point.xy)
    );
    float minor = max(gridLineCoverage(point.x, u_grid_spacing),
                      gridLineCoverage(point.y, u_grid_spacing));
    float major = max(gridLineCoverage(point.x, u_grid_spacing * 5.0),
                      gridLineCoverage(point.y, u_grid_spacing * 5.0));
    float x_axis = gridAxisCoverage(point.y);
    float y_axis = gridAxisCoverage(point.x);
    float fade = clamp(1.0 / (1.0 + travel * travel * 0.002), 0.0, 1.0);

    vec3 color = mix(vec3(0.26, 0.30, 0.38), vec3(0.40, 0.46, 0.56), major);
    vec3 first_axis_color = (u_grid_plane == 2 ? vec3(0.0, 1.0, 0.0) : vec3(0.92, 0.24, 0.20));
    vec3 second_axis_color = (u_grid_plane == 0 ? vec3(0.0, 1.0, 0.0) : vec3(0.1, 0.45, 1.0));
    color = mix(color, first_axis_color, x_axis);
    color = mix(color, second_axis_color, y_axis);
    float coverage = max(max(minor * 0.55, major * 0.78), max(x_axis, y_axis));
    return mix(background, color, coverage * fade);
}

vec3 objectPalette(uint object_id) {
    int slot = int((max(object_id, 1u) - 1u) % 10u);
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

float sceneDist(vec3 p) { return evalSceneSDF(p).dist; }

vec3 sceneNormal(vec3 p) {
    const float e = 0.0008;
    const vec2 k = vec2(1.0, -1.0);
    return normalize(
        k.xyy * sceneDist(p + k.xyy * e) +
        k.yyx * sceneDist(p + k.yyx * e) +
        k.yxy * sceneDist(p + k.yxy * e) +
        k.xxx * sceneDist(p + k.xxx * e)
    );
}

// ---- camera-orientation axis gizmo (corner overlay) ------------------------
float _seg(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    float t = clamp(dot(pa, ba) / max(dot(ba, ba), 1e-5), 0.0, 1.0);
    return length(pa - ba * t);
}

// X/Y/Z stroke glyphs in a [-1,1] box; returns distance to the nearest stroke.
float _glyph(vec2 q, int letter) {
    if (letter == 0) {        // X
        return min(_seg(q, vec2(-0.55, -0.8), vec2(0.55, 0.8)),
                   _seg(q, vec2(-0.55, 0.8), vec2(0.55, -0.8)));
    } else if (letter == 1) { // Y
        return min(min(_seg(q, vec2(-0.55, 0.8), vec2(0.0, 0.05)),
                       _seg(q, vec2(0.55, 0.8), vec2(0.0, 0.05))),
                   _seg(q, vec2(0.0, 0.05), vec2(0.0, -0.8)));
    }
    return min(min(_seg(q, vec2(-0.55, 0.8), vec2(0.55, 0.8)),   // Z
                   _seg(q, vec2(0.55, 0.8), vec2(-0.55, -0.8))),
               _seg(q, vec2(-0.55, -0.8), vec2(0.55, -0.8)));
}

vec3 axisGizmo(vec2 frag_px, vec3 base) {
    // bottom-left corner; Vulkan gl_FragCoord is top-left so flip y.
    vec2 p = vec2(frag_px.x, u_resolution.y - frag_px.y) - vec2(66.0, 66.0);
    if (max(abs(p.x), abs(p.y)) > 52.0) return base;
    const float LEN = 30.0;
    vec3 axes[3] = vec3[](vec3(1, 0, 0), vec3(0, 1, 0), vec3(0, 0, 1));
    vec3 cols[3] = vec3[](vec3(0.95, 0.30, 0.25),
                          vec3(0.30, 0.90, 0.40),
                          vec3(0.35, 0.55, 1.00));
    vec3 col = base;
    for (int i = 0; i < 3; i++) {
        vec2 sd = vec2(dot(axes[i], u_camera_right),
                       dot(axes[i], u_camera_up)) * LEN;
        float t = clamp(dot(p, sd) / max(dot(sd, sd), 1e-4), 0.0, 1.0);
        col = mix(col, cols[i], 1.0 - smoothstep(1.1, 2.4, length(p - sd * t)));
        // X/Y/Z letter just beyond the positive tip
        vec2 tip = sd + normalize(sd + vec2(1e-4)) * 10.0;
        float gd = _glyph((p - tip) / 5.5, i);
        col = mix(col, cols[i], 1.0 - smoothstep(0.16, 0.40, gd));
    }
    return col;
}

void main() {
    vec2 pix = gl_FragCoord.xy;
    vec2 screen_uv = (pix - 0.5 * u_resolution) / max(u_resolution.y, 1.0);
    // Vulkan gl_FragCoord origin is top-left; flip Y so "up" is up.
    screen_uv.y = -screen_uv.y;

    vec3 forward = normalize(u_camera_target - u_camera_position);
    vec3 right = normalize(u_camera_right);
    vec3 up = normalize(u_camera_up);
    vec3 ray_direction = normalize(
        2.0 * screen_uv.x * right + 2.0 * screen_uv.y * up + u_focal_length * forward
    );

    float travel = 0.0;
    bool hit = false;
    vec3 point = u_camera_position;
    for (int step = 0; step < 160; ++step) {
        point = u_camera_position + ray_direction * travel;
        float d = sceneDist(point);
        if (d < 0.0008) { hit = true; break; }
        travel += max(d, 0.0002);
        if (travel > 100.0) break;
    }

    vec3 color = gridBackground(u_camera_position, ray_direction, u_background_color);
    int front_owner = 0;
    if (hit) {
        Sample s = evalSceneSDF(point);
        front_owner = int(s.owner_id);
        vec3 normal = sceneNormal(point);
        vec3 light_direction = normalize(vec3(0.7, 1.0, 0.45));
        float diffuse = max(dot(normal, light_direction), 0.0);
        float rim = pow(1.0 - abs(dot(normal, -ray_direction)), 2.5);
        vec3 material = objectPalette(s.owner_id);
        vec3 surface = material * (0.24 + 0.76 * diffuse) + material * (0.25 * rim);
        float surface_opacity = u_surface_opacity;
        if (u_selected_object_id > 0 && int(s.owner_id) == u_selected_object_id) {
            surface = mix(surface, vec3(1.0, 0.82, 0.25), 0.45) + 0.15 * rim;
            // Keep the selected object clearly visible even at low global opacity.
            surface_opacity = max(surface_opacity, 0.85);
        }
        // Blend the surface over whatever is behind it (grid/background) so low
        // opacity reads as transparency, not an opaque background-colored blob.
        color = mix(color, pow(surface, vec3(0.86)), surface_opacity);
    }

    // X-ray highlight: the first march stops at the nearest surface, so a
    // selected object behind another is never reached. When the front-most hit
    // isn't the selected object, march again (stepping by |distance| so it
    // passes through occluders) until the selected object's surface is found,
    // and ghost it in so the selection is always visible.
    if (u_selected_object_id > 0 && front_owner != u_selected_object_id) {
        float xt = 0.0;
        for (int i = 0; i < 96; ++i) {
            vec3 xp = u_camera_position + ray_direction * xt;
            Sample xs = evalSceneSDF(xp);
            if (abs(xs.dist) < 0.0015
                    && int(xs.owner_id) == u_selected_object_id) {
                vec3 xn = sceneNormal(xp);
                float xd = max(dot(xn, normalize(vec3(0.7, 1.0, 0.45))), 0.0);
                vec3 hl = vec3(1.0, 0.84, 0.34) * (0.62 + 0.55 * xd);
                color = mix(color, hl, 0.74);
                break;
            }
            xt += max(abs(xs.dist), 0.002);
            if (xt > 100.0) break;
        }
    }

    color = axisGizmo(gl_FragCoord.xy, color);
    frag_color = vec4(color, 1.0);
}
