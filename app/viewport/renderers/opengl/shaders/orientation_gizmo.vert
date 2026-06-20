#version 330

in vec3 in_position;
in vec3 in_color;
out vec3 v_color;

uniform mat3 u_view_rotation;
uniform vec2 u_origin;
uniform vec2 u_scale;
uniform float u_point_size;

void main() {
    vec3 camera_axis = u_view_rotation * in_position;
    vec2 position = u_origin + camera_axis.xy * u_scale;
    gl_Position = vec4(position, 0.0, 1.0);
    gl_PointSize = u_point_size;
    v_color = in_color;
}
