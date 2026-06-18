#version 330

out vec4 frag_color;

uniform vec2 u_resolution;
uniform vec3 u_camera_position;
uniform vec3 u_camera_target;
uniform vec3 u_camera_right;
uniform vec3 u_camera_up;
uniform float u_focal_length;
uniform float u_grid_spacing;
uniform int u_grid_plane;
uniform vec3 u_background_color;

float lineCoverage(float coordinate, float spacing) {
    float distance_to_line =
        abs(fract(coordinate / spacing + 0.5) - 0.5) * spacing;
    float pixel_width = max(fwidth(coordinate), 0.000001);
    return 1.0 - smoothstep(pixel_width * 0.55, pixel_width * 1.45, distance_to_line);
}

float axisCoverage(float coordinate) {
    float pixel_width = max(fwidth(coordinate), 0.000001);
    return 1.0 - smoothstep(pixel_width * 0.55, pixel_width * 1.45, abs(coordinate));
}

vec3 gridBackground(vec3 ray_direction, vec3 background) {
    int normal_axis = u_grid_plane == 2 ? 0 : (u_grid_plane == 1 ? 1 : 2);
    float origin_axis = u_camera_position[normal_axis];
    float direction_axis = ray_direction[normal_axis];
    if (abs(direction_axis) < 0.000001) {
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
        lineCoverage(point.x, u_grid_spacing),
        lineCoverage(point.y, u_grid_spacing)
    );
    float major = max(
        lineCoverage(point.x, u_grid_spacing * 5.0),
        lineCoverage(point.y, u_grid_spacing * 5.0)
    );
    float x_axis = axisCoverage(point.y);
    float y_axis = axisCoverage(point.x);
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
    vec3 background = u_background_color;
    frag_color = vec4(gridBackground(ray_direction, background), 1.0);
}
