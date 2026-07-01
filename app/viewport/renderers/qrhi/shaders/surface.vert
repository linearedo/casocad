#version 450
layout(location = 0) in vec3 in_position;
layout(location = 1) in vec3 in_normal;
layout(location = 2) in vec3 in_color;
layout(location = 0) out vec3 v_color;
layout(std140, binding = 0) uniform SurfaceUBO {
    mat4 mvp;
    float opacity;
};
vec3 safeNormal(vec3 value) {
    float len2 = dot(value, value);
    if (!(len2 > 1.0e-12) || !(len2 < 1.0e12)) {
        return vec3(0.0, 0.0, 1.0);
    }
    return value * inversesqrt(len2);
}
void main() {
    vec3 n = safeNormal(in_normal);
    vec3 light = normalize(vec3(0.35, 0.45, 0.82));
    float diffuse = abs(dot(n, light)) * 0.45 + 0.55;
    v_color = clamp(in_color * diffuse, 0.0, 1.0);
    gl_Position = mvp * vec4(in_position, 1.0);
}
