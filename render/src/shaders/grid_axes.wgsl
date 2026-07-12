// Fullscreen analytic grid + world-axes pass (port of fullscreen.vert +
// grid_axes.frag). Framebuffer y is down in wgpu, matching the
// `u_fb_y_up == 0` path of the original.

struct GridUniforms {
    camera_position: vec3<f32>,
    focal_length: f32,
    camera_target: vec3<f32>,
    max_ray_distance: f32,
    camera_right: vec3<f32>,
    grid_spacing: f32,
    camera_up: vec3<f32>,
    show_grid: f32,
    background_color: vec3<f32>,
    grid_plane: f32,
    resolution: vec2<f32>,
    _pad0: f32,
    _pad1: f32,
};

@group(0) @binding(0) var<uniform> ubo: GridUniforms;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> VertexOutput {
    // Fullscreen triangle.
    var out: VertexOutput;
    let x = f32(i32(vertex_index & 1u) * 4 - 1);
    let y = f32(i32(vertex_index >> 1u) * 4 - 1);
    out.clip_position = vec4<f32>(x, y, 0.0, 1.0);
    return out;
}

// No early returns before fwidth(): WGSL requires derivative calls in
// uniform control flow, so validity is folded into the final mix weight.
fn grid_color(ro: vec3<f32>, rd: vec3<f32>, mt: f32, col: vec3<f32>, s: f32) -> vec3<f32> {
    var n = vec3<f32>(0.0, 0.0, 1.0);
    if (ubo.grid_plane > 0.5 && ubo.grid_plane < 1.5) {
        n = vec3<f32>(0.0, 1.0, 0.0);
    } else if (ubo.grid_plane >= 1.5) {
        n = vec3<f32>(1.0, 0.0, 0.0);
    }
    let den = dot(rd, n);
    let den_ok = abs(den) >= 1.0e-6;
    let safe_den = select(1.0, den, den_ok);
    let tt = -dot(ro, n) / safe_den;
    let hit = ubo.show_grid >= 0.5 && den_ok && tt > 0.0 && tt < mt;
    let p = ro + rd * tt;
    var g = p.xy;
    if (ubo.grid_plane > 0.5 && ubo.grid_plane < 1.5) {
        g = p.xz;
    } else if (ubo.grid_plane >= 1.5) {
        g = p.yz;
    }
    let w = fwidth(g);
    let a = abs(fract(g / ubo.grid_spacing + 0.5) - 0.5) * ubo.grid_spacing;
    let line = 1.0 - smoothstep(0.0, max(max(w.x, w.y), 1.0e-5) * 1.5, min(a.x, a.y));
    // Fade over a distance proportional to the cell size (1 m baseline).
    let ft = tt / max(ubo.grid_spacing, 1.0);
    let fade = clamp(1.0 / (1.0 + ft * ft * 0.002), 0.0, 1.0);
    let weight = line * s * fade * select(0.0, 1.0, hit);
    return mix(col, vec3<f32>(0.62, 0.75, 0.92), weight);
}

// One world axis drawn with screen-space-constant width via ray-line distance.
fn axis_line(
    ro: vec3<f32>,
    rd: vec3<f32>,
    mt: f32,
    col: vec3<f32>,
    axis: vec3<f32>,
    acol: vec3<f32>,
) -> vec3<f32> {
    if (ubo.show_grid < 0.5) {
        return col;
    }
    let b = dot(rd, axis);
    let den = 1.0 - b * b;
    if (abs(den) < 1.0e-6) {
        return col;
    }
    let d = dot(rd, ro);
    let e = dot(axis, ro);
    let t = (b * e - d) / den;
    if (t <= 0.0 || t >= mt) {
        return col;
    }
    let s = (e - b * d) / den;
    let pr = ro + rd * t;
    let pa = axis * s;
    let dist = length(pr - pa);
    let wpp = t * 2.0 / (ubo.focal_length * max(ubo.resolution.y, 1.0));
    let px = dist / max(wpp, 1.0e-9);
    let linev = 1.0 - smoothstep(0.9, 2.2, px);
    let ft = t / max(ubo.grid_spacing, 1.0);
    let fade = clamp(1.0 / (1.0 + ft * ft * 0.0008), 0.0, 1.0);
    return mix(col, acol, linev * fade);
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let px = input.clip_position.xy;
    var uv = (px - 0.5 * ubo.resolution) / max(ubo.resolution.y, 1.0);
    uv.y = -uv.y; // framebuffer y-down
    let fwd = normalize(ubo.camera_target - ubo.camera_position);
    let rd = normalize(
        2.0 * uv.x * normalize(ubo.camera_right)
            + 2.0 * uv.y * normalize(ubo.camera_up)
            + ubo.focal_length * fwd,
    );
    var col = grid_color(ubo.camera_position, rd, ubo.max_ray_distance, ubo.background_color, 0.6);
    col = axis_line(ubo.camera_position, rd, ubo.max_ray_distance, col,
                    vec3<f32>(1.0, 0.0, 0.0), vec3<f32>(1.00, 0.34, 0.25));
    col = axis_line(ubo.camera_position, rd, ubo.max_ray_distance, col,
                    vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(0.33, 0.92, 0.41));
    col = axis_line(ubo.camera_position, rd, ubo.max_ray_distance, col,
                    vec3<f32>(0.0, 0.0, 1.0), vec3<f32>(0.36, 0.57, 1.00));
    return vec4<f32>(col, 1.0);
}
