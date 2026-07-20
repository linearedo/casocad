// Screen-space thick lines (port of line.vert / line.frag). Each segment is
// six vertices carrying both endpoints plus (endpoint_sel, side); the vertex
// shader projects both endpoints and offsets perpendicular in pixel space.
// clip_y_sign is -1 for wgpu (screen y-down -> NDC y-up).

struct LineUniforms {
    camera_position: vec3<f32>,
    focal: f32,
    camera_right: vec3<f32>,
    aspect: f32,
    camera_up: vec3<f32>,
    half_px: f32,
    camera_target: vec3<f32>,
    clip_y_sign: f32,
    resolution: vec2<f32>,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> ubo: LineUniforms;

struct VertexInput {
    @location(0) point_a: vec3<f32>,
    @location(1) point_b: vec3<f32>,
    @location(2) color: vec3<f32>,
    @location(3) param: vec2<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec3<f32>,
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
    let ca = project(input.point_a);
    let cb = project(input.point_b);
    let sa = (ca.xy / ca.w * 0.5 + vec2<f32>(0.5)) * ubo.resolution;
    let sb = (cb.xy / cb.w * 0.5 + vec2<f32>(0.5)) * ubo.resolution;
    var dir = sb - sa;
    let len = length(dir);
    if (len > 1.0e-5) {
        dir = dir / len;
    } else {
        dir = vec2<f32>(1.0, 0.0);
    }
    let nrm = vec2<f32>(-dir.y, dir.x);
    var cthis = cb;
    var sthis = sb;
    if (input.param.x < 0.5) {
        cthis = ca;
        sthis = sa;
    }
    let soff = sthis + nrm * input.param.y * ubo.half_px;
    let ndc = soff / ubo.resolution * 2.0 - vec2<f32>(1.0);
    out.clip_position = vec4<f32>(
        ndc.x * cthis.w,
        ndc.y * cthis.w * ubo.clip_y_sign,
        cthis.z,
        cthis.w,
    );
    out.color = input.color;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(input.color, 1.0);
}
