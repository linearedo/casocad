// Surface pass: per-vertex Lambert shading baked into color (port of
// surface.vert / surface.frag).

struct SurfaceUniforms {
    mvp: mat4x4<f32>,
    opacity: f32,
    _pad0: f32,
    _pad1: f32,
    _pad2: f32,
};

@group(0) @binding(0) var<uniform> ubo: SurfaceUniforms;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
};

fn safe_normal(value: vec3<f32>) -> vec3<f32> {
    let len2 = dot(value, value);
    if (!(len2 > 1.0e-12) || !(len2 < 1.0e12)) {
        return vec3<f32>(0.0, 0.0, 1.0);
    }
    return value * inverseSqrt(len2);
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let n = safe_normal(input.normal);
    let light = normalize(vec3<f32>(0.35, 0.45, 0.82));
    let diffuse = abs(dot(n, light)) * 0.45 + 0.55;
    out.color = clamp(input.color * diffuse, vec3<f32>(0.0), vec3<f32>(1.0));
    out.clip_position = ubo.mvp * vec4<f32>(input.position, 1.0);
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, clamp(ubo.opacity, 0.0, 1.0));
}
