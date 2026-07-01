#version 450
layout(location = 0) in vec3 v_color;
layout(location = 0) out vec4 frag_color;
layout(std140, binding = 0) uniform SurfaceUBO {
    mat4 mvp;
    float opacity;
};
void main() {
    frag_color = vec4(v_color, clamp(opacity, 0.0, 1.0));
}
