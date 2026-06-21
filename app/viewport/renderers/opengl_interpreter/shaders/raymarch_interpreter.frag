// Interpreter raymarch fragment shader (design §7.1, §13.6 step 2).
//
// Concatenated after the generated #defines, the FEATURE_* block and the
// assembled interpreter chunks (sdf_core + optional features). Runs the
// sphere-tracing loop against evalSceneSDF and reproduces the scene shader's
// in-shader visuals (grid, selection + boundary highlight, opacity) so the
// interpreter reaches parity with the codegen scene pass — no separate grid pass
// exists in sdf mode. render() already populates every uniform below (guarded by
// "if name in program"), so they wire up for free.

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
uniform int u_selected_boundary_owner_ids[16];
uniform int u_selected_boundary_node_indices[16];
uniform int u_selected_boundary_whole_flags[16];
uniform vec3 u_selected_boundary_normals[16];

const vec3 PREVIEW_COLOR = vec3(0.15, 0.92, 1.0);

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

#ifdef FEATURE_CULL
uniform bool u_cull_enabled;  // when true, march the world-grid cull path
#endif

float sceneDist(vec3 p) {
#ifdef FEATURE_CULL
    if (u_cull_enabled) return irCullDist(p);
#endif
    return evalSceneSDF(p).dist;
}

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

// ---- grid (in-shader; sdf mode has no separate grid pass) ------------------
float gridLineCoverage(float coordinate, float spacing) {
    float distance_to_line = abs(fract(coordinate / spacing + 0.5) - 0.5) * spacing;
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
    if (!u_show_grid || abs(direction_axis) < 0.000001) return background;
    float travel = -origin_axis / direction_axis;
    if (travel <= 0.0) return background;

    vec3 world_point = u_camera_position + ray_direction * travel;
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

// ---- surface shading with selection + boundary highlight -------------------
// Uses the interpreter's threaded owner_id directly (no sceneObjectId needed),
// and irNodeSDF(node_index, p) for per-node boundary surface tests.
vec3 shadePreviewLayerSurface(vec3 point, vec3 ray_direction) {
    vec3 normal = sceneNormal(point);
    vec3 light_direction = normalize(vec3(0.7, 1.0, 0.45));
    float diffuse = max(dot(normal, light_direction), 0.0);
    float rim = pow(1.0 - abs(dot(normal, -ray_direction)), 0.55);
    vec3 color = PREVIEW_COLOR * (0.34 + 0.66 * diffuse)
        + PREVIEW_COLOR * (0.45 * rim) + vec3(0.04, 0.10, 0.11);
    return pow(color, vec3(0.86));
}

vec3 shadeSceneSurface(Sample hit, vec3 point, vec3 ray_direction) {
    vec3 normal = sceneNormal(point);
    vec3 light_direction = normalize(vec3(0.7, 1.0, 0.45));
    float diffuse = max(dot(normal, light_direction), 0.0);
    float rim = pow(1.0 - abs(dot(normal, -ray_direction)), 2.5);
    vec3 material = objectPalette(hit.owner_id);
    vec3 surface_color = material * (0.24 + 0.76 * diffuse);
    surface_color += material * (0.25 * rim);

    bool boundary_hovered = false;
    bool boundary_selected = false;
    float hover_normal_length = length(u_boundary_hover_normal);
    if (
        u_boundary_selection_active
        && u_boundary_hover_node_index >= 0
        && abs(irNodeSDF(uint(u_boundary_hover_node_index), point).dist) < 0.004
        && (hover_normal_length <= 0.0001
            || dot(normal, normalize(u_boundary_hover_normal)) > 0.88)
    ) {
        boundary_hovered = true;
    }
    for (int index = 0; index < 16; ++index) {
        if (index >= u_selected_boundary_count) break;
        int selected_node_index = u_selected_boundary_node_indices[index];
        if (selected_node_index < 0) continue;
        if (abs(irNodeSDF(uint(selected_node_index), point).dist) >= 0.004) continue;
        float selected_normal_length = length(u_selected_boundary_normals[index]);
        if (
            u_selected_boundary_whole_flags[index] != 0
            || (selected_normal_length > 0.0001
                && dot(normal, normalize(u_selected_boundary_normals[index])) > 0.88)
        ) {
            boundary_selected = true;
            break;
        }
    }
    if (u_scene_selected_object_id > 0 && int(hit.owner_id) == u_scene_selected_object_id) {
        vec3 highlight = vec3(1.0, 0.94, 0.18);
        surface_color = mix(surface_color, highlight, 0.72);
        surface_color += highlight * (0.20 + 0.45 * rim);
    }
    if (boundary_selected) {
        vec3 selected_boundary = vec3(0.25, 0.72, 1.0);
        surface_color = mix(surface_color, selected_boundary, 0.50);
        surface_color += selected_boundary * (0.12 + 0.25 * rim);
    }
    if (boundary_hovered) {
        vec3 hover_boundary = vec3(1.0, 1.0, 0.92);
        surface_color = mix(surface_color, hover_boundary, 0.92);
        surface_color += vec3(1.0, 0.52, 0.08) * (0.45 + 0.55 * rim);
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
        2.0 * screen_uv.x * right + 2.0 * screen_uv.y * up + u_focal_length * forward
    );

    bool hit = false;
    vec3 point = u_camera_position;
    Sample hit_sample;
#ifdef FEATURE_CULL
    if (u_cull_enabled) {
        hit_sample = irMarchCulled(u_camera_position, ray_direction, point, hit);
    } else
#endif
    {
        float travel = 0.0;
        for (int step = 0; step < 160; ++step) {
            point = u_camera_position + ray_direction * travel;
            float d = sceneDist(point);
            if (d < 0.0008) { hit = true; break; }
            travel += max(d, 0.0002);
            if (travel > 100.0) break;
        }
        if (hit) hit_sample = evalSceneSDF(point);
    }

    vec3 background = gridBackground(ray_direction, u_background_color);
    if (u_render_preview_layer) {
        if (!hit) discard;
        frag_color = vec4(shadePreviewLayerSurface(point, ray_direction), 0.72);
        return;
    }
    vec3 color = background;
    if (hit) {
        color = mix(background, shadeSceneSurface(hit_sample, point, ray_direction),
                    u_surface_opacity);
    }
    frag_color = vec4(color, 1.0);
}
