//! Analytic primitive surface builders, the per-object surface cache, and the
//! dispatcher between the two rendering strategies. Ported from
//! `app/viewport/surface_builder.py`.
//!
//! ```text
//! primitive/sweep leaf      -> analytic mesh (this module)
//! sharp boolean (meshable)  -> Strategy A, exact clip  (clipping)
//! anything else / field SDF -> Strategy B, dual contour (contouring)
//! ```

use std::collections::HashMap;
use std::f64::consts::PI;

use caso_kernel::sdf::node::{Node, Shape};
use caso_kernel::sdf::placed::PlacedSdf2D;
use caso_kernel::sdf::primitives_3d::{
    Box3, BoxFrame, CappedCone, Cone, Cylinder, Pyramid, Sphere, Torus,
};
use caso_kernel::sdf::solid_from_2d::Revolve;
use caso_kernel::sdf::tubes::{CapStyle, PolylineTube, QuadraticBezierTube};
use caso_kernel::vec3::{vec3, Vec3};
use caso_kernel::Frame;

use crate::clipping::{clip_surface, OperandMesh};
use crate::contouring::contour_surface;
use crate::geomops::{mesh_normals, normalize_or_z, wire_indices_from_triangles};
use crate::profiles2d::{
    f32_bounds, placed_1d_line, placed_2d_outline, placed_polyline_1d, profile_outline,
};
use crate::types::{
    empty_surface, failed_surface, object_color, SurfaceStatus, ViewportSurface,
    ViewportSurfaceKey, ViewportSurfaceScene,
};

const MAX_REVOLVE_VIEWPORT_RESOLUTION: u32 = 48;

// ---------------------------------------------------------------------------
// Mesh accumulation helpers

#[derive(Default)]
struct MeshAccum {
    vertices: Vec<Vec3>,
    normals: Vec<Vec3>,
    indices: Vec<u32>,
}

impl MeshAccum {
    fn push(&mut self, vertex: Vec3, normal: Vec3) -> u32 {
        self.vertices.push(vertex);
        self.normals.push(normal);
        (self.vertices.len() - 1) as u32
    }
}

fn surface_from_accum(
    node: &Node,
    key: ViewportSurfaceKey,
    color: [f32; 3],
    accum: MeshAccum,
    with_normals: bool,
    build_wire: bool,
) -> ViewportSurface {
    let vertices: Vec<[f32; 3]> = accum
        .vertices
        .iter()
        .map(|v| [v.x as f32, v.y as f32, v.z as f32])
        .collect();
    let normals: Vec<[f32; 3]> = if with_normals {
        accum
            .normals
            .iter()
            .map(|n| [n.x as f32, n.y as f32, n.z as f32])
            .collect()
    } else {
        mesh_normals(&vertices, &accum.indices)
    };
    let wire = if build_wire {
        wire_indices_from_triangles(&accum.indices)
    } else {
        Vec::new()
    };
    let (bounds_min, bounds_max) = f32_bounds(&vertices);
    ViewportSurface {
        key,
        object_kind: node.kind().to_string(),
        status: SurfaceStatus::Ready,
        vertices,
        normals,
        indices: accum.indices,
        wire_indices: wire,
        color,
        bounds_min,
        bounds_max,
        message: String::new(),
    }
}

// ---------------------------------------------------------------------------
// 3D primitive meshes

fn sphere_surface(node: &Node, shape: &Sphere, key: ViewportSurfaceKey, color: [f32; 3]) -> ViewportSurface {
    let segments = ((key.resolution as usize) * 2).max(64);
    let rings = (key.resolution as usize).max(32);
    let mut accum = MeshAccum::default();
    for ring in 0..=rings {
        let phi = PI * ring as f64 / rings as f64;
        let (sin_phi, cos_phi) = (phi.sin(), phi.cos());
        for segment in 0..segments {
            let theta = 2.0 * PI * segment as f64 / segments as f64;
            let normal = vec3(theta.cos() * sin_phi, theta.sin() * sin_phi, cos_phi);
            accum.push(shape.center + normal * shape.radius, normal);
        }
    }
    for ring in 0..rings {
        for segment in 0..segments {
            let next_segment = (segment + 1) % segments;
            let a = (ring * segments + segment) as u32;
            let b = (ring * segments + next_segment) as u32;
            let c = ((ring + 1) * segments + next_segment) as u32;
            let d = ((ring + 1) * segments + segment) as u32;
            if ring > 0 {
                accum.indices.extend_from_slice(&[a, b, d]);
            }
            if ring < rings - 1 {
                accum.indices.extend_from_slice(&[b, c, d]);
            }
        }
    }
    surface_from_accum(node, key, color, accum, true, true)
}

type FaceSpec = ([i32; 3], [usize; 2], f64);
const FACE_SPECS: [FaceSpec; 6] = [
    ([1, 0, 0], [1, 2], 1.0),
    ([-1, 0, 0], [1, 2], -1.0),
    ([0, 1, 0], [0, 2], 1.0),
    ([0, -1, 0], [0, 2], -1.0),
    ([0, 0, 1], [0, 1], 1.0),
    ([0, 0, -1], [0, 1], -1.0),
];

fn append_oriented_box(accum: &mut MeshAccum, center: Vec3, frame: &Frame, half: Vec3) {
    let axes = [frame.u, frame.v, frame.w];
    let half_axes = [half.x, half.y, half.z];
    for (normal_signs, tangent_axes, side) in FACE_SPECS {
        let mut normal = Vec3::ZERO;
        let mut fixed_axis = 0usize;
        for (axis_index, sign) in normal_signs.iter().enumerate() {
            if *sign != 0 {
                normal += axes[axis_index] * (*sign as f64);
                fixed_axis = axis_index;
            }
        }
        let [axis_a, axis_b] = tangent_axes;
        let mut local = [[0.0f64; 3]; 4];
        for corner in &mut local {
            corner[fixed_axis] = side * half_axes[fixed_axis];
        }
        let signs_a = [-1.0, 1.0, 1.0, -1.0];
        let signs_b = [-1.0, -1.0, 1.0, 1.0];
        for corner in 0..4 {
            local[corner][axis_a] = signs_a[corner] * half_axes[axis_a];
            local[corner][axis_b] = signs_b[corner] * half_axes[axis_b];
        }
        let base = accum.vertices.len() as u32;
        for corner in local {
            let world =
                center + axes[0] * corner[0] + axes[1] * corner[1] + axes[2] * corner[2];
            accum.push(world, normal);
        }
        if side > 0.0 {
            accum
                .indices
                .extend_from_slice(&[base, base + 1, base + 2, base, base + 2, base + 3]);
        } else {
            accum
                .indices
                .extend_from_slice(&[base, base + 2, base + 1, base, base + 3, base + 2]);
        }
    }
}

fn box_surface(node: &Node, shape: &Box3, key: ViewportSurfaceKey, color: [f32; 3]) -> ViewportSurface {
    let mut accum = MeshAccum::default();
    append_oriented_box(&mut accum, shape.center, &shape.frame, shape.half_size);
    surface_from_accum(node, key, color, accum, true, true)
}

fn box_frame_surface(
    node: &Node,
    shape: &BoxFrame,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> ViewportSurface {
    let half = shape.half_size;
    let radius = (shape.thickness * 0.5).min(half.x.min(half.y).min(half.z));
    let mut accum = MeshAccum::default();
    let axes = [shape.frame.u, shape.frame.v, shape.frame.w];
    let half_axes = [half.x, half.y, half.z];
    for axis_index in 0..3usize {
        let tangent_axes: Vec<usize> = (0..3).filter(|index| *index != axis_index).collect();
        let mut beam_half = [radius, radius, radius];
        beam_half[axis_index] = half_axes[axis_index];
        for sign_a in [-1.0, 1.0] {
            for sign_b in [-1.0, 1.0] {
                let mut offset = [0.0f64; 3];
                offset[tangent_axes[0]] = sign_a * (half_axes[tangent_axes[0]] - radius);
                offset[tangent_axes[1]] = sign_b * (half_axes[tangent_axes[1]] - radius);
                let beam_center = shape.center
                    + axes[0] * offset[0]
                    + axes[1] * offset[1]
                    + axes[2] * offset[2];
                append_oriented_box(
                    &mut accum,
                    beam_center,
                    &shape.frame,
                    vec3(beam_half[0], beam_half[1], beam_half[2]),
                );
            }
        }
    }
    surface_from_accum(node, key, color, accum, true, true)
}

fn cylinder_surface(
    node: &Node,
    shape: &Cylinder,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> ViewportSurface {
    frustum_like_surface(
        node,
        key,
        color,
        shape.center,
        &shape.frame,
        shape.radius,
        shape.radius,
        shape.half_height,
        true,
    )
}

fn capped_cone_surface(
    node: &Node,
    shape: &CappedCone,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> ViewportSurface {
    frustum_like_surface(
        node,
        key,
        color,
        shape.center,
        &shape.frame,
        shape.radius_a,
        shape.radius_b,
        shape.half_height,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn frustum_like_surface(
    node: &Node,
    key: ViewportSurfaceKey,
    color: [f32; 3],
    center: Vec3,
    frame: &Frame,
    bottom_radius: f64,
    top_radius: f64,
    half_height: f64,
    cylinder_normals: bool,
) -> ViewportSurface {
    let segments = ((key.resolution as usize) * 2).max(64);
    let (u, v, w) = (frame.u, frame.v, frame.w);
    let mut accum = MeshAccum::default();
    let slope = (bottom_radius - top_radius) / (2.0 * half_height).max(1.0e-12);
    for (z_sign, radius) in [(-1.0, bottom_radius), (1.0, top_radius)] {
        for segment in 0..segments {
            let theta = 2.0 * PI * segment as f64 / segments as f64;
            let radial = u * theta.cos() + v * theta.sin();
            let normal = if cylinder_normals {
                radial
            } else {
                normalize_or_z(radial + w * slope)
            };
            accum.push(center + radial * radius + w * (z_sign * half_height), normal);
        }
    }
    let top_center = accum.push(center + w * half_height, w);
    let bottom_center = accum.push(center - w * half_height, -w);
    let top_ring = accum.vertices.len() as u32;
    for segment in 0..segments {
        let theta = 2.0 * PI * segment as f64 / segments as f64;
        let radial = u * theta.cos() + v * theta.sin();
        accum.push(center + radial * top_radius + w * half_height, w);
    }
    let bottom_ring = accum.vertices.len() as u32;
    for segment in 0..segments {
        let theta = 2.0 * PI * segment as f64 / segments as f64;
        let radial = u * theta.cos() + v * theta.sin();
        accum.push(center + radial * bottom_radius - w * half_height, -w);
    }
    for segment in 0..segments as u32 {
        let next_segment = (segment + 1) % segments as u32;
        let bottom_a = segment;
        let bottom_b = next_segment;
        let top_a = segments as u32 + segment;
        let top_b = segments as u32 + next_segment;
        accum
            .indices
            .extend_from_slice(&[bottom_a, bottom_b, top_b, bottom_a, top_b, top_a]);
        accum
            .indices
            .extend_from_slice(&[top_center, top_ring + segment, top_ring + next_segment]);
        accum.indices.extend_from_slice(&[
            bottom_center,
            bottom_ring + next_segment,
            bottom_ring + segment,
        ]);
    }
    surface_from_accum(node, key, color, accum, true, true)
}

fn cone_surface(node: &Node, shape: &Cone, key: ViewportSurfaceKey, color: [f32; 3]) -> ViewportSurface {
    let segments = ((key.resolution as usize) * 2).max(64);
    let (u, v, w) = (shape.frame.u, shape.frame.v, shape.frame.w);
    let mut accum = MeshAccum::default();
    for segment in 0..segments {
        let theta = 2.0 * PI * segment as f64 / segments as f64;
        let radial = u * theta.cos() + v * theta.sin();
        let normal = normalize_or_z(radial + w * (shape.radius / (2.0 * shape.half_height)));
        accum.push(
            shape.center + radial * shape.radius - w * shape.half_height,
            normal,
        );
    }
    let apex = accum.push(shape.center + w * shape.half_height, w);
    let bottom_center = accum.push(shape.center - w * shape.half_height, -w);
    let bottom_ring = accum.vertices.len() as u32;
    for segment in 0..segments {
        let theta = 2.0 * PI * segment as f64 / segments as f64;
        let radial = u * theta.cos() + v * theta.sin();
        accum.push(shape.center + radial * shape.radius - w * shape.half_height, -w);
    }
    for segment in 0..segments as u32 {
        let next_segment = (segment + 1) % segments as u32;
        accum.indices.extend_from_slice(&[segment, next_segment, apex]);
        accum.indices.extend_from_slice(&[
            bottom_center,
            bottom_ring + next_segment,
            bottom_ring + segment,
        ]);
    }
    surface_from_accum(node, key, color, accum, true, true)
}

fn pyramid_surface(
    node: &Node,
    shape: &Pyramid,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> ViewportSurface {
    let axes = [shape.frame.u, shape.frame.v, shape.frame.w];
    let half = shape.base_half_size;
    let height = shape.half_height;
    let local = [
        [-half, -half, -height],
        [half, -half, -height],
        [half, half, -height],
        [-half, half, -height],
        [0.0, 0.0, height],
    ];
    let world: Vec<Vec3> = local
        .iter()
        .map(|point| shape.center + axes[0] * point[0] + axes[1] * point[1] + axes[2] * point[2])
        .collect();
    let faces = [
        [0usize, 2, 1],
        [0, 3, 2],
        [0, 1, 4],
        [1, 2, 4],
        [2, 3, 4],
        [3, 0, 4],
    ];
    let mut accum = MeshAccum::default();
    for face in faces {
        let base = accum.vertices.len() as u32;
        for index in face {
            accum.push(world[index], Vec3::ZERO);
        }
        accum.indices.extend_from_slice(&[base, base + 1, base + 2]);
    }
    surface_from_accum(node, key, color, accum, false, true)
}

fn torus_surface(node: &Node, shape: &Torus, key: ViewportSurfaceKey, color: [f32; 3]) -> ViewportSurface {
    let major_segments = ((key.resolution as usize) * 3).max(96);
    let minor_segments = (key.resolution as usize).max(32);
    let (u, v, w) = (shape.frame.u, shape.frame.v, shape.frame.w);
    let mut accum = MeshAccum::default();
    for major in 0..major_segments {
        let theta = 2.0 * PI * major as f64 / major_segments as f64;
        let radial = u * theta.cos() + v * theta.sin();
        let ring_center = shape.center + radial * shape.major_radius;
        for minor in 0..minor_segments {
            let phi = 2.0 * PI * minor as f64 / minor_segments as f64;
            let normal = radial * phi.cos() + w * phi.sin();
            accum.push(ring_center + normal * shape.minor_radius, normal);
        }
    }
    for major in 0..major_segments {
        let next_major = (major + 1) % major_segments;
        for minor in 0..minor_segments {
            let next_minor = (minor + 1) % minor_segments;
            let a = (major * minor_segments + minor) as u32;
            let b = (next_major * minor_segments + minor) as u32;
            let c = (next_major * minor_segments + next_minor) as u32;
            let d = (major * minor_segments + next_minor) as u32;
            accum.indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    surface_from_accum(node, key, color, accum, true, true)
}

// ---------------------------------------------------------------------------
// Tubes

fn drop_duplicate_points_3d(points: &[Vec3]) -> Vec<Vec3> {
    let mut deduped: Vec<Vec3> = Vec::with_capacity(points.len());
    for point in points {
        match deduped.last() {
            Some(last) if (*point - *last).length() <= 1.0e-9 => {}
            _ => deduped.push(*point),
        }
    }
    deduped
}

fn perpendicular_axis(tangent: Vec3) -> Vec3 {
    let candidates = [Vec3::X, Vec3::Y, Vec3::Z];
    let reference = candidates
        .into_iter()
        .min_by(|a, b| {
            a.dot(tangent)
                .abs()
                .partial_cmp(&b.dot(tangent).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .expect("three candidates");
    normalize_or_z(tangent.cross(reference))
}

fn tube_frames(points: &[Vec3]) -> Vec<(Vec3, Vec3, Vec3)> {
    let mut frames = Vec::with_capacity(points.len());
    let mut previous_normal: Option<Vec3> = None;
    for (index, point) in points.iter().enumerate() {
        let tangent = if index == 0 {
            normalize_or_z(points[1] - *point)
        } else if index == points.len() - 1 {
            normalize_or_z(*point - points[index - 1])
        } else {
            normalize_or_z(points[index + 1] - points[index - 1])
        };
        let normal = match previous_normal {
            None => perpendicular_axis(tangent),
            Some(previous) => {
                let projected = previous - tangent * previous.dot(tangent);
                if projected.length() <= 1.0e-12 {
                    perpendicular_axis(tangent)
                } else {
                    normalize_or_z(projected)
                }
            }
        };
        let binormal = normalize_or_z(tangent.cross(normal));
        let normal = normalize_or_z(binormal.cross(tangent));
        previous_normal = Some(normal);
        frames.push((normal, binormal, tangent));
    }
    frames
}

fn append_tube_cap(
    accum: &mut MeshAccum,
    center: Vec3,
    frame: (Vec3, Vec3, Vec3),
    radius: f64,
    ring_segments: usize,
    ring_start: Option<u32>,
    flip: bool,
) {
    let (normal, binormal, tangent) = frame;
    let cap_normal = if flip { -tangent } else { tangent };
    let center_index = accum.push(center, cap_normal);
    let cap_ring = accum.vertices.len() as u32;
    for segment in 0..ring_segments {
        let vertex = match ring_start {
            Some(start) => accum.vertices[(start + segment as u32) as usize],
            None => {
                let theta = 2.0 * PI * segment as f64 / ring_segments as f64;
                center + (normal * theta.cos() + binormal * theta.sin()) * radius
            }
        };
        accum.push(vertex, cap_normal);
    }
    for segment in 0..ring_segments as u32 {
        let next_segment = (segment + 1) % ring_segments as u32;
        if flip {
            accum
                .indices
                .extend_from_slice(&[center_index, cap_ring + next_segment, cap_ring + segment]);
        } else {
            accum
                .indices
                .extend_from_slice(&[center_index, cap_ring + segment, cap_ring + next_segment]);
        }
    }
}

fn tube_centerline_surface(
    node: &Node,
    key: ViewportSurfaceKey,
    color: [f32; 3],
    points: &[Vec3],
    radius: f64,
) -> ViewportSurface {
    let ring_segments = (key.resolution as usize).max(32);
    let frames = tube_frames(points);
    let mut accum = MeshAccum::default();
    for (point, (normal, binormal, _tangent)) in points.iter().zip(frames.iter()) {
        for segment in 0..ring_segments {
            let theta = 2.0 * PI * segment as f64 / ring_segments as f64;
            let radial = *normal * theta.cos() + *binormal * theta.sin();
            accum.push(*point + radial * radius, radial);
        }
    }
    for ring in 0..points.len() - 1 {
        let ring_base = (ring * ring_segments) as u32;
        let next_base = ((ring + 1) * ring_segments) as u32;
        for segment in 0..ring_segments as u32 {
            let next_segment = (segment + 1) % ring_segments as u32;
            let a = ring_base + segment;
            let b = ring_base + next_segment;
            let c = next_base + next_segment;
            let d = next_base + segment;
            accum.indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    append_tube_cap(&mut accum, points[0], frames[0], radius, ring_segments, None, true);
    let end_ring = ((points.len() - 1) * ring_segments) as u32;
    append_tube_cap(
        &mut accum,
        points[points.len() - 1],
        frames[frames.len() - 1],
        radius,
        ring_segments,
        Some(end_ring),
        false,
    );
    surface_from_accum(node, key, color, accum, true, true)
}

fn polyline_tube_surface(
    node: &Node,
    tube: &PolylineTube,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> Option<ViewportSurface> {
    if tube.inner_radius > 0.0 {
        return None;
    }
    let points = drop_duplicate_points_3d(&tube.points);
    if points.len() < 2 {
        return None;
    }
    Some(tube_centerline_surface(node, key, color, &points, tube.radius))
}

fn sample_quadratic_points_3d(points: &[Vec3], resolution: u32) -> Vec<Vec3> {
    let steps = ((resolution as usize) * 2).max(8);
    let mut sampled: Vec<Vec3> = Vec::new();
    let mut span_start = 0usize;
    while span_start + 2 <= points.len() - 1 {
        let a = points[span_start];
        let b = points[span_start + 1];
        let c = points[span_start + 2];
        for step in 0..=steps {
            if !sampled.is_empty() && step == 0 {
                continue;
            }
            let t = step as f64 / steps as f64;
            sampled.push(a * ((1.0 - t) * (1.0 - t)) + b * (2.0 * (1.0 - t) * t) + c * (t * t));
        }
        span_start += 2;
    }
    drop_duplicate_points_3d(&sampled)
}

fn bezier_tube_surface(
    node: &Node,
    tube: &QuadraticBezierTube,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> Option<ViewportSurface> {
    if tube.inner_radius > 0.0 {
        return None;
    }
    let points = sample_quadratic_points_3d(&tube.points, key.resolution);
    if points.len() < 2 {
        return None;
    }
    Some(tube_centerline_surface(node, key, color, &points, tube.radius))
}

// ---------------------------------------------------------------------------
// Extrude / Revolve

fn extrude_profile_surface(
    node: &Node,
    section: &PlacedSdf2D,
    height: f64,
    center_offset: f64,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> Option<ViewportSurface> {
    let outline = profile_outline(&section.profile, key.resolution);
    if outline.len() < 3 {
        return None;
    }
    let normal = section.normal();
    let bottom_offset = center_offset - height * 0.5;
    let top_offset = center_offset + height * 0.5;
    let bottom: Vec<Vec3> = outline
        .iter()
        .map(|point| {
            section.origin
                + section.axis_u * point[0]
                + section.axis_v * point[1]
                + normal * bottom_offset
        })
        .collect();
    let top: Vec<Vec3> = outline
        .iter()
        .map(|point| {
            section.origin
                + section.axis_u * point[0]
                + section.axis_v * point[1]
                + normal * top_offset
        })
        .collect();
    let count = outline.len() as u32;
    let mut accum = MeshAccum::default();
    for vertex in bottom.iter().chain(top.iter()) {
        accum.push(*vertex, Vec3::ZERO);
    }
    for index in 0..count {
        let next_index = (index + 1) % count;
        accum.indices.extend_from_slice(&[
            index,
            next_index,
            count + next_index,
            index,
            count + next_index,
            count + index,
        ]);
    }
    let mut bottom_mean = Vec3::ZERO;
    for vertex in &bottom {
        bottom_mean += *vertex;
    }
    bottom_mean = bottom_mean / bottom.len() as f64;
    let mut top_mean = Vec3::ZERO;
    for vertex in &top {
        top_mean += *vertex;
    }
    top_mean = top_mean / top.len() as f64;
    let bottom_center = accum.push(bottom_mean, Vec3::ZERO);
    let top_center = accum.push(top_mean, Vec3::ZERO);
    for index in 0..count {
        let next_index = (index + 1) % count;
        accum.indices.extend_from_slice(&[bottom_center, next_index, index]);
        accum
            .indices
            .extend_from_slice(&[top_center, count + index, count + next_index]);
    }
    Some(surface_from_accum(node, key, color, accum, false, true))
}

fn revolve_profile_surface(
    node: &Node,
    revolve: &Revolve,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> Option<ViewportSurface> {
    let section = revolve.section2d();
    let resolution = key.resolution.min(MAX_REVOLVE_VIEWPORT_RESOLUTION);
    let outline = profile_outline(&section.profile, resolution);
    if outline.len() < 3 {
        return None;
    }
    let frame = revolve.axis_frame().ok()?;
    let outline_points: Vec<Vec3> = outline
        .iter()
        .map(|point| section.origin + section.axis_u * point[0] + section.axis_v * point[1])
        .collect();
    let axial_values: Vec<f64> = outline_points
        .iter()
        .map(|point| (*point - frame.origin).dot(frame.axis))
        .collect();
    let radius_values: Vec<f64> = outline_points
        .iter()
        .zip(axial_values.iter())
        .map(|(point, axial)| ((*point - frame.origin) - frame.axis * *axial).length())
        .collect();
    let axial_points: Vec<Vec3> = axial_values
        .iter()
        .map(|axial| frame.origin + frame.axis * *axial)
        .collect();
    let angle = revolve.angle_degrees.to_radians();
    let closed = revolve.angle_degrees.abs() >= 360.0 - 1.0e-9;
    let mut segments = ((resolution as usize) * 3).max(32);
    if !closed {
        segments = ((segments as f64 * angle.abs() / (2.0 * PI)).ceil() as usize).max(4);
    }
    let mut vertices: Vec<Vec3> = Vec::new();
    let sweep_count = segments + if closed { 0 } else { 1 };
    for sweep_index in 0..sweep_count {
        let t = sweep_index as f64 / segments as f64;
        let theta = if closed { 2.0 * PI * t } else { angle * t };
        let radial = frame.radial * theta.cos() + frame.tangent * theta.sin();
        for (axial_point, radius) in axial_points.iter().zip(radius_values.iter()) {
            vertices.push(*axial_point + radial * *radius);
        }
    }
    let ring_count = if closed { segments } else { segments + 1 };
    let outline_count = outline.len();
    let mut indices: Vec<u32> = Vec::new();
    for sweep_index in 0..segments {
        let next_sweep = (sweep_index + 1) % ring_count;
        for index in 0..outline_count {
            let next_index = (index + 1) % outline_count;
            let a = (sweep_index * outline_count + index) as u32;
            let b = (next_sweep * outline_count + index) as u32;
            let c = (next_sweep * outline_count + next_index) as u32;
            let d = (sweep_index * outline_count + next_index) as u32;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    if !closed {
        append_revolve_cap(&mut indices, 0, outline_count, &mut vertices, false);
        append_revolve_cap(
            &mut indices,
            segments * outline_count,
            outline_count,
            &mut vertices,
            true,
        );
    }
    let (vertices, indices) = deduplicate_indexed_mesh(vertices, indices);
    let mut accum = MeshAccum::default();
    for vertex in vertices {
        accum.push(vertex, Vec3::ZERO);
    }
    accum.indices = indices;
    Some(surface_from_accum(node, key, color, accum, false, false))
}

fn append_revolve_cap(
    indices: &mut Vec<u32>,
    ring_start: usize,
    outline_count: usize,
    vertices: &mut Vec<Vec3>,
    flip: bool,
) {
    let mut center = Vec3::ZERO;
    for vertex in &vertices[ring_start..ring_start + outline_count] {
        center += *vertex;
    }
    center = center / outline_count as f64;
    let center_index = vertices.len() as u32;
    vertices.push(center);
    for index in 0..outline_count {
        let next_index = (index + 1) % outline_count;
        if flip {
            indices.extend_from_slice(&[
                center_index,
                (ring_start + next_index) as u32,
                (ring_start + index) as u32,
            ]);
        } else {
            indices.extend_from_slice(&[
                center_index,
                (ring_start + index) as u32,
                (ring_start + next_index) as u32,
            ]);
        }
    }
}

fn deduplicate_indexed_mesh(vertices: Vec<Vec3>, indices: Vec<u32>) -> (Vec<Vec3>, Vec<u32>) {
    if vertices.is_empty() || indices.is_empty() {
        return (vertices, indices);
    }
    // Weld by 12-decimal rounding.
    let round12 = |value: f64| (value * 1.0e12).round() / 1.0e12;
    let mut welded: Vec<Vec3> = Vec::new();
    let mut weld_map: HashMap<(u64, u64, u64), u32> = HashMap::new();
    let mut inverse: Vec<u32> = Vec::with_capacity(vertices.len());
    for vertex in &vertices {
        let bits = (
            round12(vertex.x).to_bits(),
            round12(vertex.y).to_bits(),
            round12(vertex.z).to_bits(),
        );
        let next = welded.len() as u32;
        let id = *weld_map.entry(bits).or_insert_with(|| {
            welded.push(vec3(round12(vertex.x), round12(vertex.y), round12(vertex.z)));
            next
        });
        inverse.push(id);
    }
    let mut seen_faces: HashMap<[u32; 3], ()> = HashMap::new();
    let mut cleaned: Vec<u32> = Vec::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        let a = inverse[tri[0] as usize];
        let b = inverse[tri[1] as usize];
        let c = inverse[tri[2] as usize];
        if a == b || a == c || b == c {
            continue;
        }
        let pa = welded[a as usize];
        let pb = welded[b as usize];
        let pc = welded[c as usize];
        if (pb - pa).cross(pc - pa).length() <= 1.0e-14 {
            continue;
        }
        let mut sorted = [a, b, c];
        sorted.sort_unstable();
        if seen_faces.insert(sorted, ()).is_some() {
            continue;
        }
        cleaned.extend_from_slice(&[a, b, c]);
    }
    (welded, cleaned)
}

// ---------------------------------------------------------------------------
// Dispatch

/// Analytic fast-path surface for a leaf, or None.
fn primitive_surface(
    node: &Node,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> Option<ViewportSurface> {
    match &node.shape {
        Shape::Sphere(shape) => Some(sphere_surface(node, shape, key, color)),
        Shape::Box3(shape) => Some(box_surface(node, shape, key, color)),
        Shape::BoxFrame(shape) => Some(box_frame_surface(node, shape, key, color)),
        Shape::Cylinder(shape) => Some(cylinder_surface(node, shape, key, color)),
        Shape::Cone(shape) => Some(cone_surface(node, shape, key, color)),
        Shape::CappedCone(shape) => Some(capped_cone_surface(node, shape, key, color)),
        Shape::Pyramid(shape) => Some(pyramid_surface(node, shape, key, color)),
        Shape::Torus(shape) => Some(torus_surface(node, shape, key, color)),
        Shape::PolylineTube(tube) => polyline_tube_surface(node, tube, key, color),
        Shape::QuadraticBezierTube(tube) => bezier_tube_surface(node, tube, key, color),
        Shape::Extrude(extrude) => extrude_profile_surface(
            node,
            extrude.section2d(),
            extrude.height,
            extrude.center_offset,
            key,
            color,
        ),
        Shape::Revolve(revolve) => revolve_profile_surface(node, revolve, key, color),
        _ => None,
    }
}

/// Provide a meshable leaf's analytic mesh arrays to the clip module.
fn operand_primitive_mesh(node: &Node, key: ViewportSurfaceKey) -> Option<OperandMesh> {
    let color = object_color(key.object_id);
    let surface = primitive_surface(node, key, color)?;
    if !surface.has_geometry() {
        return None;
    }
    Some(OperandMesh {
        vertices: surface
            .vertices
            .iter()
            .map(|v| vec3(v[0] as f64, v[1] as f64, v[2] as f64))
            .collect(),
        normals: surface
            .normals
            .iter()
            .map(|n| vec3(n[0] as f64, n[1] as f64, n[2] as f64))
            .collect(),
        triangles: surface
            .indices
            .chunks_exact(3)
            .map(|tri| [tri[0], tri[1], tri[2]])
            .collect(),
    })
}

/// The one routing point between rendering strategies.
pub fn build_viewport_surface(node: &Node, key: ViewportSurfaceKey) -> ViewportSurface {
    let color = object_color(key.object_id);
    if let Some(primitive) = primitive_surface(node, key, color) {
        return primitive;
    }
    if node.dimension() == 3 {
        let provider = |operand: &Node| operand_primitive_mesh(operand, key);
        if let Some(clipped) = clip_surface(node, key, color, &provider) {
            return clipped;
        }
        return match contour_surface(node, key, color) {
            Ok(surface) => surface,
            Err(message) => failed_surface(node, key, color, message),
        };
    }
    match &node.shape {
        Shape::PlacedSdf1D(placed) => placed_1d_line(node, placed, key, color),
        Shape::PlacedPolyline1D(placed) => placed_polyline_1d(node, placed, key, color),
        Shape::PlacedSdf2D(placed) => placed_2d_outline(node, placed, key, color),
        _ => empty_surface(node, key, color, "no viewport surface for dimension"),
    }
}

// ---------------------------------------------------------------------------
// Cache + scene build

/// Viewport-only CPU cache for generated draw surfaces. The key includes the
/// scene revision; reuse slots are bounded to one entry per live object.
#[derive(Default)]
pub struct ViewportSurfaceCache {
    pub resolution: u32,
    surfaces: HashMap<ViewportSurfaceKey, ViewportSurface>,
    latest_by_signature: HashMap<u32, (String, ViewportSurface)>,
    latest_by_translation: HashMap<u32, (String, ViewportSurface, Vec3)>,
}

/// Anchor point whose translation leaves the shape signature unchanged.
fn translation_anchor(node: &Node) -> Option<Vec3> {
    match &node.shape {
        Shape::Sphere(shape) => Some(shape.center),
        Shape::Box3(shape) => Some(shape.center),
        Shape::BoxFrame(shape) => Some(shape.center),
        Shape::CappedCone(shape) => Some(shape.center),
        Shape::Cone(shape) => Some(shape.center),
        Shape::Cylinder(shape) => Some(shape.center),
        Shape::Pyramid(shape) => Some(shape.center),
        Shape::Torus(shape) => Some(shape.center),
        Shape::PlacedSdf1D(placed) => Some(placed.origin),
        Shape::PlacedPolyline1D(placed) => Some(placed.origin),
        Shape::PlacedSdf2D(placed) => Some(placed.origin),
        Shape::PolylineTube(tube) => Some(tube.points[0]),
        Shape::QuadraticBezierTube(tube) => Some(tube.points[0]),
        Shape::Extrude(extrude) => Some(extrude.section2d().origin),
        Shape::Revolve(revolve) => Some(revolve.section2d().origin),
        Shape::Translate { offset, .. } => Some(*offset),
        _ => None,
    }
}

fn sig_float(value: f64) -> f64 {
    let rounded = (value * 1.0e12).round() / 1.0e12;
    if rounded == 0.0 {
        0.0
    } else {
        rounded
    }
}

fn sig_vec(value: Vec3) -> (f64, f64, f64) {
    (sig_float(value.x), sig_float(value.y), sig_float(value.z))
}

/// Signature invariant under pure translation of the anchor (Debug-format of
/// translation-independent parameters, the Rust analog of the Python tuples).
fn translation_shape_signature(node: &Node, anchor: Vec3) -> Option<String> {
    let relative = |point: Vec3| sig_vec(point - anchor);
    match &node.shape {
        Shape::Sphere(shape) => Some(format!("Sphere{:?}", sig_float(shape.radius))),
        Shape::Box3(shape) => Some(format!(
            "Box{:?}{:?}{:?}{:?}",
            sig_vec(shape.half_size),
            sig_vec(shape.frame.u),
            sig_vec(shape.frame.v),
            sig_vec(shape.frame.w)
        )),
        Shape::BoxFrame(shape) => Some(format!(
            "BoxFrame{:?}{:?}{:?}{:?}{:?}",
            sig_vec(shape.half_size),
            sig_float(shape.thickness),
            sig_vec(shape.frame.u),
            sig_vec(shape.frame.v),
            sig_vec(shape.frame.w)
        )),
        Shape::Cylinder(shape) => Some(format!(
            "Cylinder{:?}{:?}{:?}",
            sig_float(shape.radius),
            sig_float(shape.half_height),
            sig_vec(shape.frame.w)
        )),
        Shape::Cone(shape) => Some(format!(
            "Cone{:?}{:?}{:?}",
            sig_float(shape.radius),
            sig_float(shape.half_height),
            sig_vec(shape.frame.w)
        )),
        Shape::CappedCone(shape) => Some(format!(
            "CappedCone{:?}{:?}{:?}{:?}",
            sig_float(shape.radius_a),
            sig_float(shape.radius_b),
            sig_float(shape.half_height),
            sig_vec(shape.frame.w)
        )),
        Shape::Pyramid(shape) => Some(format!(
            "Pyramid{:?}{:?}{:?}",
            sig_float(shape.base_half_size),
            sig_float(shape.half_height),
            sig_vec(shape.frame.w)
        )),
        Shape::Torus(shape) => Some(format!(
            "Torus{:?}{:?}{:?}",
            sig_float(shape.major_radius),
            sig_float(shape.minor_radius),
            sig_vec(shape.frame.w)
        )),
        Shape::PlacedSdf1D(placed) => Some(format!(
            "PlacedSdf1D{:?}{:?}",
            placed.profile,
            sig_vec(placed.axis_u)
        )),
        Shape::PlacedPolyline1D(placed) => Some(format!(
            "PlacedPolyline1D{:?}{:?}{:?}",
            placed.profile,
            sig_vec(placed.axis_u),
            sig_vec(placed.axis_v)
        )),
        Shape::PlacedSdf2D(placed) => Some(format!(
            "PlacedSdf2D{:?}{:?}{:?}",
            placed.profile,
            sig_vec(placed.axis_u),
            sig_vec(placed.axis_v)
        )),
        Shape::PolylineTube(tube) => Some(format!(
            "PolylineTube{:?}{:?}{:?}{:?}",
            tube.points.iter().map(|p| relative(*p)).collect::<Vec<_>>(),
            sig_float(tube.radius),
            sig_float(tube.inner_radius),
            tube.caps == CapStyle::Flat
        )),
        Shape::QuadraticBezierTube(tube) => Some(format!(
            "QuadraticBezierTube{:?}{:?}{:?}{:?}",
            tube.points.iter().map(|p| relative(*p)).collect::<Vec<_>>(),
            sig_float(tube.radius),
            sig_float(tube.inner_radius),
            tube.caps == CapStyle::Flat
        )),
        Shape::Extrude(extrude) => {
            let section = &extrude.section;
            let section_sig = translation_shape_signature(section, anchor)?;
            Some(format!(
                "Extrude{section_sig}{:?}{:?}",
                sig_float(extrude.height),
                sig_float(extrude.center_offset)
            ))
        }
        Shape::Revolve(revolve) => {
            let section = &revolve.section;
            let section_origin = revolve.section2d().origin;
            let section_sig = translation_shape_signature(section, section_origin)?;
            Some(format!(
                "Revolve{section_sig}{:?}{:?}{:?}{:?}{:?}",
                revolve.axis.as_str(),
                revolve.axis_origin.map(|origin| sig_vec(origin - section_origin)),
                revolve.axis_direction.map(sig_vec),
                revolve.radial_direction.map(sig_vec),
                sig_float(revolve.angle_degrees)
            ))
        }
        Shape::Translate { child, .. } => Some(format!("Translate{child:?}")),
        _ => None,
    }
}

fn translated_surface(
    surface: &ViewportSurface,
    key: ViewportSurfaceKey,
    anchor: Vec3,
    previous_anchor: Vec3,
) -> ViewportSurface {
    let delta = anchor - previous_anchor;
    let delta_f32 = [delta.x as f32, delta.y as f32, delta.z as f32];
    let mut moved = surface.clone();
    moved.key = key;
    for vertex in &mut moved.vertices {
        for (component, delta_axis) in vertex.iter_mut().zip(delta_f32.iter()) {
            *component += *delta_axis;
        }
    }
    for (axis, delta_axis) in delta_f32.iter().enumerate() {
        moved.bounds_min[axis] += *delta_axis as f64;
        moved.bounds_max[axis] += *delta_axis as f64;
    }
    moved
}

impl ViewportSurfaceCache {
    pub fn new(resolution: u32) -> Self {
        Self {
            resolution,
            ..Default::default()
        }
    }

    pub fn get_or_build(&mut self, node: &Node, revision: u64) -> ViewportSurface {
        let key = ViewportSurfaceKey {
            object_id: node.object_id,
            scene_revision: revision,
            resolution: self.resolution,
        };
        let anchor = translation_anchor(node);
        let translation_signature =
            anchor.and_then(|anchor| translation_shape_signature(node, anchor));
        let signature = format!("{node:?}");
        if let Some(cached) = self.surfaces.get(&key) {
            return cached.clone();
        }
        if let Some((stored_signature, stored_surface)) =
            self.latest_by_signature.get(&key.object_id)
        {
            if *stored_signature == signature {
                let mut surface = stored_surface.clone();
                surface.key = key;
                self.store(key, signature, translation_signature, anchor, surface.clone());
                return surface;
            }
        }
        if let (Some(anchor_point), Some(translation_sig)) = (anchor, &translation_signature) {
            if let Some((stored_sig, stored_surface, stored_anchor)) =
                self.latest_by_translation.get(&key.object_id)
            {
                if stored_sig == translation_sig {
                    let surface =
                        translated_surface(stored_surface, key, anchor_point, *stored_anchor);
                    self.store(key, signature, translation_signature, anchor, surface.clone());
                    return surface;
                }
            }
        }
        let surface = build_viewport_surface(node, key);
        self.store(key, signature, translation_signature, anchor, surface.clone());
        surface
    }

    fn store(
        &mut self,
        key: ViewportSurfaceKey,
        signature: String,
        translation_signature: Option<String>,
        anchor: Option<Vec3>,
        surface: ViewportSurface,
    ) {
        self.surfaces.insert(key, surface.clone());
        self.latest_by_signature
            .insert(key.object_id, (signature, surface.clone()));
        if let (Some(translation_sig), Some(anchor_point)) = (translation_signature, anchor) {
            if surface.has_geometry() {
                self.latest_by_translation
                    .insert(key.object_id, (translation_sig, surface, anchor_point));
            }
        }
    }

    pub fn prune_before(&mut self, revision: u64) {
        self.surfaces.retain(|key, _| key.scene_revision >= revision);
    }

    pub fn prune_to_object_ids(&mut self, live: &[u32]) {
        self.latest_by_signature.retain(|id, _| live.contains(id));
        self.latest_by_translation.retain(|id, _| live.contains(id));
    }
}

/// Build the display-surface scene for a set of component nodes.
pub fn build_viewport_surface_scene(
    components: &[Node],
    revision: u64,
    cache: &mut ViewportSurfaceCache,
) -> ViewportSurfaceScene {
    // std::time::Instant is unsupported on wasm32; build timing is telemetry
    // only, so it reports 0 there.
    #[cfg(not(target_arch = "wasm32"))]
    let start = std::time::Instant::now();
    let primary_object_ids: Vec<u32> = components
        .iter()
        .filter(|component| component.object_id > 0)
        .map(|component| component.object_id)
        .collect();
    let live: Vec<&Node> = components
        .iter()
        .filter(|component| component.object_id > 0)
        .collect();
    let surfaces: Vec<ViewportSurface> = live
        .iter()
        .map(|component| cache.get_or_build(component, revision))
        .collect();
    cache.prune_before(revision.saturating_sub(2));
    let live_ids: Vec<u32> = live.iter().map(|component| component.object_id).collect();
    cache.prune_to_object_ids(&live_ids);
    #[cfg(not(target_arch = "wasm32"))]
    let build_ms = start.elapsed().as_secs_f64() * 1000.0;
    #[cfg(target_arch = "wasm32")]
    let build_ms = 0.0;
    ViewportSurfaceScene {
        revision,
        surfaces,
        build_ms,
        primary_object_ids,
    }
}
