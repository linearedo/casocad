#version 450
layout(location = 0) in vec3 in_a;
layout(location = 1) in vec3 in_b;
layout(location = 2) in vec3 in_col;
layout(location = 3) in vec2 in_param;
layout(location = 0) out vec3 v_col;
layout(std140, binding = 0) uniform LineUBO {
    vec3 cam_pos;
    vec3 cam_right;
    vec3 cam_up;
    vec3 cam_target;
    float focal;
    float aspect;
    vec2 res;
    float half_px;
    float clip_y_sign;
};
vec4 project(vec3 P) {
    vec3 fwd = normalize(cam_target - cam_pos);
    vec3 r = normalize(cam_right);
    vec3 u = normalize(cam_up);
    vec3 v = P - cam_pos;
    return vec4(focal * dot(v, r) * aspect, -focal * dot(v, u),
                0.0, max(dot(v, fwd), 1e-4));
}
void main() {
    vec4 ca = project(in_a);
    vec4 cb = project(in_b);
    vec2 sa = (ca.xy / ca.w * 0.5 + 0.5) * res;
    vec2 sb = (cb.xy / cb.w * 0.5 + 0.5) * res;
    vec2 dir = sb - sa;
    float len = length(dir);
    dir = len > 1e-5 ? dir / len : vec2(1.0, 0.0);
    vec2 nrm = vec2(-dir.y, dir.x);
    vec4 cthis = in_param.x < 0.5 ? ca : cb;
    vec2 sthis = in_param.x < 0.5 ? sa : sb;
    vec2 soff = sthis + nrm * in_param.y * half_px;
    vec2 ndc = soff / res * 2.0 - 1.0;
    gl_Position = vec4(ndc.x * cthis.w, ndc.y * cthis.w * clip_y_sign,
                       cthis.z, cthis.w);
    v_col = in_col;
}
