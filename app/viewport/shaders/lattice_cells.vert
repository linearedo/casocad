#version 330

in vec3 in_offset;
in vec3 in_center;
in vec3 in_color;

out vec3 v_color;

uniform mat4 u_view_projection;
uniform float u_cell_size;

void main() {
    vec3 position = in_center + in_offset * u_cell_size;
    gl_Position = u_view_projection * vec4(position, 1.0);
    v_color = in_color;
}
