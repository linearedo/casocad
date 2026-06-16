#version 330

out vec4 frag_color;

uniform vec2 u_resolution;
uniform vec3 u_camera_position;
uniform vec3 u_camera_target;
uniform float u_focal_length;
uniform bool u_show_components;
uniform float u_surface_opacity;
uniform bool u_show_grid;
uniform float u_grid_spacing;
uniform int u_grid_plane;
uniform bool u_boundary_selection_active;
uniform int u_boundary_hover_owner_id;
uniform int u_boundary_hover_direction;
uniform int u_scene_hover_object_id;
uniform int u_scene_selected_object_id;
uniform int u_preview_kind;
uniform vec3 u_preview_start;
uniform vec3 u_preview_current;
uniform vec3 u_preview_move_delta;
const int MAX_SELECTED_BOUNDARY_OWNERS = 128;
uniform int u_selected_boundary_region_count;
uniform ivec2 u_selected_boundary_regions[MAX_SELECTED_BOUNDARY_OWNERS];
const vec3 SELECTION_OVERLAY_COLOR = vec3(0.15, 0.92, 1.0);
const float SELECTION_OVERLAY_ALPHA = 0.92;
const vec3 PREVIEW_COLOR = vec3(0.15, 0.92, 1.0);
const float PI = 3.14159265359;

/*__SCENE_SDF__*/

bool movePreviewActive() {
    return (
        u_scene_selected_object_id != 0
        && length(u_preview_move_delta) > 0.000001
    );
}

float renderSceneSDF(vec3 p) {
    if (movePreviewActive()) {
        return sceneMovedSDF(
            p,
            u_scene_selected_object_id,
            u_preview_move_delta
        );
    }
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
        vec3 sample_point = movePreviewActive()
            ? hit_point - u_preview_move_delta
            : hit_point;
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
    vec3 sample_point = movePreviewActive() ? p - u_preview_move_delta : p;
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

float createPreviewSDF(vec3 p) {
    vec3 center = 0.5 * (u_preview_start + u_preview_current);
    vec3 local = p - center;
    vec2 drag = u_preview_current.xy - u_preview_start.xy;
    float extent_x = max(abs(drag.x) * 0.5, 0.05);
    float extent_y = max(abs(drag.y) * 0.5, 0.05);
    float radius = max(length(drag) * 0.5, 0.05);

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
        profile = length(local.xy) - radius;
    } else if (u_preview_kind == 3) {
        profile = box2SDF(local.xy, vec2(extent_x, extent_y));
    } else if (u_preview_kind == 4) {
        profile = box2SDF(local.xy, vec2(max(extent_x, extent_y)));
    } else if (u_preview_kind == 5) {
        vec2 half_size = vec2(extent_x, extent_y);
        profile = roundedBox2SDF(
            local.xy,
            half_size,
            max(0.01, min(half_size.x, half_size.y) * 0.2)
        );
    } else if (u_preview_kind == 6) {
        vec2 axes = vec2(extent_x, extent_y);
        profile = (length(local.xy / axes) - 1.0) * min(axes.x, axes.y);
    } else if (u_preview_kind == 7) {
        profile = regularPolygonSDF(local.xy, radius);
    }
    if (u_preview_kind >= 2 && u_preview_kind <= 7) {
        return max(profile, abs(local.z) - 0.002);
    }
    if (u_preview_kind == 8) {
        return length(local) - radius;
    }
    if (u_preview_kind == 9) {
        return box3SDF(local, vec3(extent_x, extent_y, max(extent_x, extent_y)));
    }
    if (u_preview_kind == 10) {
        vec2 d = abs(vec2(length(local.xy), local.z))
            - vec2(radius, max(extent_x, extent_y));
        return min(max(d.x, d.y), 0.0) + length(max(d, vec2(0.0)));
    }
    if (u_preview_kind == 11) {
        return length(vec2(length(local.xy) - radius, local.z))
            - max(radius * 0.25, 0.02);
    }
    return 1000000.0;
}

bool createPreviewActive() {
    return u_preview_kind > 0;
}

float previewSDF(vec3 p) {
    float distance_to_preview = 1000000.0;
    if (createPreviewActive()) {
        distance_to_preview = min(distance_to_preview, createPreviewSDF(p));
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
    if (u_boundary_hover_direction == 0) return normal.x < -0.95;
    if (u_boundary_hover_direction == 1) return normal.x > 0.95;
    if (u_boundary_hover_direction == 2) return normal.y < -0.95;
    if (u_boundary_hover_direction == 3) return normal.y > 0.95;
    if (u_boundary_hover_direction == 4) return normal.z < -0.95;
    return normal.z > 0.95;
}

int boundaryDirectionBit(vec3 normal) {
    vec3 absolute_normal = abs(normal);
    if (
        absolute_normal.x < 0.95
        && absolute_normal.y < 0.95
        && absolute_normal.z < 0.95
    ) {
        return 0;
    }
    if (
        absolute_normal.x >= absolute_normal.y
        && absolute_normal.x >= absolute_normal.z
    ) {
        return normal.x < 0.0 ? 1 : 2;
    }
    if (absolute_normal.y >= absolute_normal.z) {
        return normal.y < 0.0 ? 4 : 8;
    }
    return normal.z < 0.0 ? 16 : 32;
}

bool selectedBoundaryRegionMatches(int boundary_owner_id, vec3 normal) {
    int direction_bit = boundaryDirectionBit(normal);
    for (int index = 0; index < MAX_SELECTED_BOUNDARY_OWNERS; ++index) {
        if (index >= u_selected_boundary_region_count) {
            break;
        }
        ivec2 selector = u_selected_boundary_regions[index];
        if (
            selector.x == boundary_owner_id
            && (
                selector.y == 63
                || (
                    direction_bit != 0
                    && (selector.y & direction_bit) != 0
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
    vec3 world_up = (
        abs(dot(forward, vec3(0.0, 0.0, 1.0))) > 0.995
        ? vec3(0.0, 1.0, 0.0)
        : vec3(0.0, 0.0, 1.0)
    );
    vec3 right = normalize(cross(forward, world_up));
    vec3 up = normalize(cross(right, forward));
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

    vec3 background_top = vec3(0.07, 0.10, 0.16);
    vec3 background_bottom = vec3(0.018, 0.023, 0.033);
    vec3 background = mix(
        background_bottom,
        background_top,
        clamp(gl_FragCoord.y / max(u_resolution.y, 1.0), 0.0, 1.0)
    );
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

    if (createPreviewActive() || movePreviewActive()) {
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
