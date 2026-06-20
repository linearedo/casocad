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
    if (!u_show_grid || abs(direction_axis) < 0.000001) return background;
    float travel = -origin_axis / direction_axis;
    if (travel <= 0.0) return background;

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

float box2SDF(vec2 q, vec2 half_size) {
    vec2 d = abs(q) - half_size;
    return length(max(d, vec2(0.0))) + min(max(d.x, d.y), 0.0);
}

float box3SDF(vec3 q, vec3 half_size) {
    vec3 d = abs(q) - half_size;
    return length(max(d, vec3(0.0))) + min(max(d.x, max(d.y, d.z)), 0.0);
}

vec3 shadePreviewLayerSurface(vec3 point, vec3 ray_direction) {
    vec3 normal = sceneNormal(point);
    vec3 light_direction = normalize(vec3(0.7, 1.0, 0.45));
    float diffuse = max(dot(normal, light_direction), 0.0);
    float rim = pow(1.0 - abs(dot(normal, -ray_direction)), 0.55);
    vec3 color =
        PREVIEW_COLOR * (0.34 + 0.66 * diffuse)
        + PREVIEW_COLOR * (0.45 * rim)
        + vec3(0.04, 0.10, 0.11);
    return pow(color, vec3(0.86));
}

vec3 shadeSceneSurface(vec3 point, vec3 ray_direction) {
    vec3 normal = sceneNormal(point);
    vec3 light_direction = normalize(vec3(0.7, 1.0, 0.45));
    float diffuse = max(dot(normal, light_direction), 0.0);
    float rim = pow(1.0 - abs(dot(normal, -ray_direction)), 2.5);
    vec3 material = objectPalette(sceneObjectId(point));
    vec3 surface_color = material * (0.24 + 0.76 * diffuse);
    surface_color += material * (0.25 * rim);
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
        if (travel > 100.0) break;
    }

    vec3 background = gridBackground(ray_direction, u_background_color);
    vec3 color = background;
    if (u_render_preview_layer) {
        if (!hit) discard;
        frag_color = vec4(shadePreviewLayerSurface(point, ray_direction), 0.72);
        return;
    }
    if (hit) {
        color = mix(
            background,
            shadeSceneSurface(point, ray_direction),
            u_surface_opacity
        );
    }
    frag_color = vec4(color, 1.0);
}
