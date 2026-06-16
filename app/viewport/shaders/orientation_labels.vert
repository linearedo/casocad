#version 330

in vec3 in_anchor;
in vec2 in_offset;
in vec3 in_color;

out vec3 v_color;

uniform mat3 u_view_rotation;
uniform vec2 u_origin;
uniform vec2 u_scale;
uniform vec2 u_label_scale;

void main() {
    vec3 camera_anchor = u_view_rotation * in_anchor;
    vec2 position =
        u_origin + camera_anchor.xy * u_scale + in_offset * u_label_scale;
    gl_Position = vec4(position, 0.0, 1.0);
    v_color = in_color;
}
