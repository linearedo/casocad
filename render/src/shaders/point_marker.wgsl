// Point markers as sphere impostors: one camera-facing quad per point
// (instanced triangle strip), shaded as a little ball in the fragment
// stage. Shares the line uniforms/bind group; point_radius_px rides the
// slot that is padding for line.wgsl. Alpha is pinned to 1.0 — see
// design_docs/mesh_preview_opacity_independence.md.

struct PointUniforms {
    camera_position: vec3<f32>,
    focal: f32,
    camera_right: vec3<f32>,
    aspect: f32,
    camera_up: vec3<f32>,
    half_px: f32,
    camera_target: vec3<f32>,
    clip_y_sign: f32,
    resolution: vec2<f32>,
    point_radius_px: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> ubo: PointUniforms;

struct VertexInput {
    @builtin(vertex_index) vertex_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) color: vec3<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
    @location(1) uv: vec2<f32>,
};

// Depth matching OrbitCamera::matrix (camera.rs): same near/far envelope
// derived from camera_position/camera_target, so mesh preview elements can
// be depth-tested against surface-pipeline geometry (see
// MESH_PREVIEW_OCCLUDE_OPACITY in renderer.rs).
fn project(p: vec3<f32>) -> vec4<f32> {
    let fwd = normalize(ubo.camera_target - ubo.camera_position);
    let r = normalize(ubo.camera_right);
    let u = normalize(ubo.camera_up);
    let v = p - ubo.camera_position;
    let clip_w = max(dot(v, fwd), 1.0e-4);
    let distance = max(length(ubo.camera_target - ubo.camera_position), 0.1);
    let near = max(distance / 1000.0, 0.001);
    let far = max(distance * 100.0, 100.0);
    let depth_scale = far / (far - near);
    let depth_bias = -(far * near) / (far - near);
    return vec4<f32>(
        ubo.focal * dot(v, r) * ubo.aspect,
        -ubo.focal * dot(v, u),
        depth_scale * clip_w + depth_bias,
        clip_w,
    );
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    var out: VertexOutput;
    let uv = vec2<f32>(
        f32(input.vertex_index & 1u) * 2.0 - 1.0,
        f32(input.vertex_index >> 1u) * 2.0 - 1.0,
    );
    let center = project(input.position);
    let screen = (center.xy / center.w * 0.5 + vec2<f32>(0.5)) * ubo.resolution;
    let soff = screen + uv * ubo.point_radius_px;
    let ndc = soff / ubo.resolution * 2.0 - vec2<f32>(1.0);
    out.clip_position = vec4<f32>(
        ndc.x * center.w,
        ndc.y * center.w * ubo.clip_y_sign,
        center.z,
        center.w,
    );
    out.color = input.color;
    out.uv = uv;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let r2 = dot(input.uv, input.uv);
    if (r2 > 1.0) {
        discard;
    }
    // Fake view-space sphere normal; same light/diffuse as surface.wgsl.
    let n = vec3<f32>(input.uv, sqrt(1.0 - r2));
    let light = normalize(vec3<f32>(0.35, 0.45, 0.82));
    let diffuse = abs(dot(n, light)) * 0.45 + 0.55;
    let shaded = clamp(input.color * diffuse, vec3<f32>(0.0), vec3<f32>(1.0));
    return vec4<f32>(shaded, 1.0);
}
