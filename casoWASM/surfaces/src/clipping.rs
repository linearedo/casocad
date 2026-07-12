//! Strategy A — exact boolean rendering by SDF-clipping analytic meshes.
//! Ported from `app/viewport/surface_clipping.py`.
//!
//! The precise path for sharp SDF operators whose operands have an analytic
//! mesh: each operand's smooth mesh is clipped against the *other* operand's
//! exact SDF (marching triangles, root-found seams). Returns `None` when an
//! operand has no analytic mesh or the result fails the surface-error quality
//! gate — the dispatcher then falls back to dual contouring.

use std::collections::HashMap;

use caso_kernel::sdf::node::{Node, Shape};
use caso_kernel::vec3::{vec3, Vec3};

use crate::geomops::{
    orient_triangles, refine_edge_hermite, split_marked_triangles, wire_indices_from_triangles,
};
use crate::contouring::MAX_DUAL_CONTOUR_WIREFRAME_TRIANGLES;
use crate::types::{SurfaceStatus, ViewportSurface, ViewportSurfaceKey};

/// Analytic operand mesh: f64 vertices/normals plus triangle triples.
#[derive(Debug, Clone, Default)]
pub struct OperandMesh {
    pub vertices: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub triangles: Vec<[u32; 3]>,
}

/// Supplies the analytic mesh of a meshable primitive/sweep leaf, or None.
pub type OperandMeshProvider<'a> = &'a dyn Fn(&Node) -> Option<OperandMesh>;

const CLIP_OPERAND_SEED_DIVISIONS: f64 = 24.0;
const SEED_WELL_SHAPED_MIN_ANGLE: f64 = 15.0;
pub const CLIP_MAX_RELATIVE_SURFACE_ERROR: f64 = 1.0e-2;

fn sgn(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn normalize_or_z(vector: Vec3) -> Vec3 {
    let length = vector.length();
    if length <= 1.0e-12 {
        vec3(0.0, 0.0, 1.0)
    } else {
        vector / length
    }
}

fn node_extent(node: &Node) -> f64 {
    match node.bounding_box() {
        Ok(bounds) => (bounds.x_max - bounds.x_min)
            .max(bounds.y_max - bounds.y_min)
            .max(bounds.z_max - bounds.z_min)
            .max(1.0),
        Err(_) => 1.0,
    }
}

/// Clip a triangle mesh against an SDF half-space (marching triangles); cut
/// vertices are root-found exactly onto the clip's zero isosurface.
///
/// Contract (boundary-highlight overlays depend on it):
/// - the kept side is `value <= 0` (`keep_inside`) or `value >= 0`; the exact
///   zero set belongs to both sides;
/// - crossing-edge vertices are Hermite-root-found onto the clip's zero set
///   (`eps` is the gradient finite-difference step);
/// - output is deterministic in its inputs — the same mesh and clip with
///   opposite `keep_inside` produce bitwise-identical seam vertices, which is
///   what makes complementary inside/outside previews crack-free.
pub fn clip_mesh_to_sdf(
    mesh: &OperandMesh,
    clip: &Node,
    keep_inside: bool,
    eps: Vec3,
) -> OperandMesh {
    let sv = clip.eval(&mesh.vertices);
    let keep: Vec<bool> = sv
        .iter()
        .map(|value| if keep_inside { *value <= 0.0 } else { *value >= 0.0 })
        .collect();

    let mut vertices = mesh.vertices.clone();
    let mut normals = mesh.normals.clone();
    // Batch the crossing-edge cuts per edge slot like the Python code.
    let mut cut_index = vec![[None::<u32>; 3]; mesh.triangles.len()];
    for (slot, (i, j)) in [(0usize, 1usize), (1, 2), (2, 0)].iter().enumerate() {
        let mut edge_tris = Vec::new();
        let mut points_a = Vec::new();
        let mut points_b = Vec::new();
        let mut fa = Vec::new();
        let mut fb = Vec::new();
        let mut t_lin = Vec::new();
        for (tri_index, tri) in mesh.triangles.iter().enumerate() {
            let a = tri[*i] as usize;
            let b = tri[*j] as usize;
            if keep[a] != keep[b] {
                edge_tris.push(tri_index);
                points_a.push(mesh.vertices[a]);
                points_b.push(mesh.vertices[b]);
                fa.push(sv[a]);
                fb.push(sv[b]);
                let delta = sv[a] - sv[b];
                let t = if delta.abs() > 1.0e-12 {
                    (sv[a] / delta).clamp(0.0, 1.0)
                } else {
                    (sv[a]).clamp(0.0, 1.0)
                };
                t_lin.push((t, a, b));
            }
        }
        if edge_tris.is_empty() {
            continue;
        }
        let (points, _grads) = refine_edge_hermite(clip, &points_a, &points_b, &fa, &fb, eps, 4);
        for (local, tri_index) in edge_tris.iter().enumerate() {
            let (t, a, b) = t_lin[local];
            let normal = normalize_or_z(mesh.normals[a] + (mesh.normals[b] - mesh.normals[a]) * t);
            cut_index[*tri_index][slot] = Some(vertices.len() as u32);
            vertices.push(points[local]);
            normals.push(normal);
        }
    }

    let mut faces: Vec<[u32; 3]> = Vec::with_capacity(mesh.triangles.len());
    for (tri_index, tri) in mesh.triangles.iter().enumerate() {
        let [v0, v1, v2] = *tri;
        let k0 = keep[v0 as usize];
        let k1 = keep[v1 as usize];
        let k2 = keep[v2 as usize];
        let [c01, c12, c20] = cut_index[tri_index];
        match (k0, k1, k2) {
            (true, true, true) => faces.push([v0, v1, v2]),
            (true, false, false) =>

                faces.push([v0, c01.expect("cut"), c20.expect("cut")]),
            (false, true, false) => faces.push([v1, c12.expect("cut"), c01.expect("cut")]),
            (false, false, true) => faces.push([v2, c20.expect("cut"), c12.expect("cut")]),
            (true, true, false) => {
                faces.push([v0, v1, c12.expect("cut")]);
                faces.push([v0, c12.expect("cut"), c20.expect("cut")]);
            }
            (false, true, true) => {
                faces.push([v1, v2, c20.expect("cut")]);
                faces.push([v1, c20.expect("cut"), c01.expect("cut")]);
            }
            (true, false, true) => {
                faces.push([v2, v0, c01.expect("cut")]);
                faces.push([v2, c01.expect("cut"), c12.expect("cut")]);
            }
            (false, false, false) => {}
        }
    }
    OperandMesh {
        vertices,
        normals,
        triangles: faces,
    }
}

fn edge_lengths(mesh: &OperandMesh, tri: &[u32; 3]) -> [f64; 3] {
    let a = mesh.vertices[tri[0] as usize];
    let b = mesh.vertices[tri[1] as usize];
    let c = mesh.vertices[tri[2] as usize];
    [(b - a).length(), (c - b).length(), (a - c).length()]
}

/// Uniform 1->4 (red) subdivision until every edge is shorter than `max_edge`.
fn uniform_subdivide(mesh: &OperandMesh, max_edge: f64, max_passes: usize) -> OperandMesh {
    let mut mesh = mesh.clone();
    for _pass in 0..max_passes {
        let longest = mesh
            .triangles
            .iter()
            .map(|tri| {
                let lengths = edge_lengths(&mesh, tri);
                lengths[0].max(lengths[1]).max(lengths[2])
            })
            .fold(0.0f64, f64::max);
        if longest < max_edge {
            break;
        }
        let mut midpoint: HashMap<(u32, u32), u32> = HashMap::new();
        let mut vertices = mesh.vertices.clone();
        let mut normals = mesh.normals.clone();
        let mut resolve = |a: u32, b: u32,
                           vertices: &mut Vec<Vec3>,
                           normals: &mut Vec<Vec3>|
         -> u32 {
            let key = if a <= b { (a, b) } else { (b, a) };
            *midpoint.entry(key).or_insert_with(|| {
                let id = vertices.len() as u32;
                vertices.push((vertices[a as usize] + vertices[b as usize]) * 0.5);
                normals.push(normalize_or_z(normals[a as usize] + normals[b as usize]));
                id
            })
        };
        let mut triangles = Vec::with_capacity(mesh.triangles.len() * 4);
        for tri in &mesh.triangles {
            let [v0, v1, v2] = *tri;
            let m01 = resolve(v0, v1, &mut vertices, &mut normals);
            let m12 = resolve(v1, v2, &mut vertices, &mut normals);
            let m20 = resolve(v2, v0, &mut vertices, &mut normals);
            triangles.push([v0, m01, m20]);
            triangles.push([m01, v1, m12]);
            triangles.push([m20, m12, v2]);
            triangles.push([m01, m12, m20]);
        }
        mesh = OperandMesh {
            vertices,
            normals,
            triangles,
        };
    }
    mesh
}

fn triangle_min_angle(a: Vec3, b: Vec3, c: Vec3) -> f64 {
    let angle_at = |p: Vec3, q: Vec3, r: Vec3| -> f64 {
        let u = q - p;
        let w = r - p;
        let denom = (u.length() * w.length()).max(1.0e-30);
        (u.dot(w) / denom).clamp(-1.0, 1.0).acos().to_degrees()
    };
    angle_at(a, b, c).min(angle_at(b, c, a)).min(angle_at(c, a, b))
}

/// Seed a coarse flat-faced operand mesh so a curved cut through a big
/// undivided face has vertices to land on.
fn seed_operand_mesh(node: &Node, mesh: OperandMesh) -> OperandMesh {
    let extent = node_extent(node);
    let target = extent / CLIP_OPERAND_SEED_DIVISIONS;
    let needs_seed = mesh.triangles.iter().any(|tri| {
        let a = mesh.vertices[tri[0] as usize];
        let b = mesh.vertices[tri[1] as usize];
        let c = mesh.vertices[tri[2] as usize];
        let lengths = edge_lengths(&mesh, tri);
        let max_edge = lengths[0].max(lengths[1]).max(lengths[2]);
        triangle_min_angle(a, b, c) > SEED_WELL_SHAPED_MIN_ANGLE && max_edge > target
    });
    if !needs_seed {
        return mesh;
    }
    uniform_subdivide(&mesh, target, 8)
}

/// Refine mesh edges that are long AND near the `clip` cut (crack-free).
///
/// Only edges longer than `target_edge` and within `2·target_edge` of the
/// clip zero set are split, so cost is proportional to seam length; passes
/// are bounded by `max_passes`.
pub fn tessellate_for_clip(
    mesh: OperandMesh,
    clip: &Node,
    target_edge: f64,
    max_passes: usize,
) -> OperandMesh {
    let mut mesh = mesh;
    for _pass in 0..max_passes {
        let cv = clip.eval(&mesh.vertices);
        let band = 2.0 * target_edge;
        // Mark long near-cut edges.
        let mut long_edges: HashMap<(u32, u32), ()> = HashMap::new();
        for tri in &mesh.triangles {
            let lengths = edge_lengths(&mesh, tri);
            for (slot, (i, j)) in [(0usize, 1usize), (1, 2), (2, 0)].iter().enumerate() {
                let a = tri[*i];
                let b = tri[*j];
                let ca = cv[a as usize];
                let cb = cv[b as usize];
                let near = sgn(ca) != sgn(cb) || ca.abs().min(cb.abs()) < band;
                if near && lengths[slot] > target_edge {
                    let key = if a <= b { (a, b) } else { (b, a) };
                    long_edges.insert(key, ());
                }
            }
        }
        if long_edges.is_empty() {
            break;
        }
        let mut vertices = mesh.vertices.clone();
        let mut normals = mesh.normals.clone();
        let mut midpoint: HashMap<(u32, u32), u32> = HashMap::with_capacity(long_edges.len());
        for key in long_edges.keys() {
            let (a, b) = *key;
            let id = vertices.len() as u32;
            vertices.push((mesh.vertices[a as usize] + mesh.vertices[b as usize]) * 0.5);
            normals.push(normalize_or_z(
                mesh.normals[a as usize] + mesh.normals[b as usize],
            ));
            midpoint.insert(*key, id);
        }
        let lookup = |a: u32, b: u32| -> Option<u32> {
            let key = if a <= b { (a, b) } else { (b, a) };
            midpoint.get(&key).copied()
        };
        let m01: Vec<Option<u32>> = mesh
            .triangles
            .iter()
            .map(|tri| lookup(tri[0], tri[1]))
            .collect();
        let m12: Vec<Option<u32>> = mesh
            .triangles
            .iter()
            .map(|tri| lookup(tri[1], tri[2]))
            .collect();
        let m20: Vec<Option<u32>> = mesh
            .triangles
            .iter()
            .map(|tri| lookup(tri[2], tri[0]))
            .collect();
        let triangles = split_marked_triangles(&mesh.triangles, &m01, &m12, &m20);
        mesh = OperandMesh {
            vertices,
            normals,
            triangles,
        };
    }
    mesh
}

fn boolean_operands(node: &Node) -> Option<(&Node, &Node)> {
    match &node.shape {
        Shape::Union(op) | Shape::Intersection(op) | Shape::Difference(op) | Shape::Xor(op) => {
            Some((&op.left, &op.right))
        }
        _ => None,
    }
}

fn flip_mesh(mut mesh: OperandMesh) -> OperandMesh {
    for normal in &mut mesh.normals {
        *normal = -*normal;
    }
    for tri in &mut mesh.triangles {
        tri.swap(1, 2);
    }
    mesh
}

fn concat_meshes(parts: Vec<OperandMesh>) -> OperandMesh {
    let mut combined = OperandMesh::default();
    for part in parts {
        let offset = combined.vertices.len() as u32;
        combined.vertices.extend(part.vertices);
        combined.normals.extend(part.normals);
        combined.triangles.extend(
            part.triangles
                .into_iter()
                .map(|tri| [tri[0] + offset, tri[1] + offset, tri[2] + offset]),
        );
    }
    combined
}

pub fn clip_boolean_mesh_with_resolution(
    node: &Node,
    provider: OperandMeshProvider<'_>,
    resolution: u32,
) -> Option<OperandMesh> {
    let (left, right) = boolean_operands(node)?;
    let operand_l = clip_operand_mesh_res(left, provider, resolution)?;
    let operand_r = clip_operand_mesh_res(right, provider, resolution)?;

    // keep_inside per operand, and whether the right operand is inverted
    // (difference exposes B as an inner wall).
    let (keep_l, keep_r, flip_r, is_xor) = match &node.shape {
        Shape::Intersection(_) => (true, true, false, false),
        Shape::Union(_) => (false, false, false, false),
        Shape::Difference(_) => (false, true, true, false),
        Shape::Xor(_) => (false, false, false, true),
        _ => return None,
    };

    let extent = node_extent(node);
    let eps_value = (extent * 1.0e-4).max(1.0e-6);
    let eps = vec3(eps_value, eps_value, eps_value);
    let target = extent / (resolution as f64).max(8.0);
    let operand_l = tessellate_for_clip(operand_l, right, target, 9);
    let operand_r = tessellate_for_clip(operand_r, left, target, 9);

    let combined = if is_xor {
        let mut parts = Vec::new();
        for (operand, clip, keep_inside, flip) in [
            (&operand_l, right, false, false),
            (&operand_l, right, true, true),
            (&operand_r, left, false, false),
            (&operand_r, left, true, true),
        ] {
            let mut clipped = clip_mesh_to_sdf(operand, clip, keep_inside, eps);
            if clipped.triangles.is_empty() {
                continue;
            }
            if flip {
                clipped = flip_mesh(clipped);
            }
            parts.push(clipped);
        }
        if parts.is_empty() {
            return None;
        }
        concat_meshes(parts)
    } else {
        let clipped_l = clip_mesh_to_sdf(&operand_l, right, keep_l, eps);
        let mut clipped_r = clip_mesh_to_sdf(&operand_r, left, keep_r, eps);
        if flip_r {
            clipped_r = flip_mesh(clipped_r);
        }
        concat_meshes(vec![clipped_l, clipped_r])
    };
    if combined.triangles.is_empty() {
        return None;
    }

    // Orientation pass drops zero-area triangles; report "no clip" if it
    // degenerates entirely, so the caller falls back to dual contouring.
    let vertex_f32: Vec<[f32; 3]> = combined
        .vertices
        .iter()
        .map(|v| [v.x as f32, v.y as f32, v.z as f32])
        .collect();
    let normal_f32: Vec<[f32; 3]> = combined
        .normals
        .iter()
        .map(|n| [n.x as f32, n.y as f32, n.z as f32])
        .collect();
    let flat: Vec<u32> = combined.triangles.iter().flatten().copied().collect();
    let oriented = orient_triangles(&vertex_f32, &normal_f32, &flat);
    if oriented.is_empty() {
        return None;
    }
    // Drop discarded vertices, remapping indices.
    let mut remap: HashMap<u32, u32> = HashMap::new();
    let mut vertices = Vec::new();
    let mut normals = Vec::new();
    let mut triangles = Vec::with_capacity(oriented.len() / 3);
    for tri in oriented.chunks_exact(3) {
        let mut mapped = [0u32; 3];
        for (slot, index) in tri.iter().enumerate() {
            let next = vertices.len() as u32;
            let mapped_index = *remap.entry(*index).or_insert_with(|| {
                vertices.push(combined.vertices[*index as usize]);
                normals.push(combined.normals[*index as usize]);
                next
            });
            mapped[slot] = mapped_index;
        }
        triangles.push(mapped);
    }
    Some(OperandMesh {
        vertices,
        normals,
        triangles,
    })
}

fn clip_operand_mesh_res(
    node: &Node,
    provider: OperandMeshProvider<'_>,
    resolution: u32,
) -> Option<OperandMesh> {
    if boolean_operands(node).is_some() {
        return clip_boolean_mesh_with_resolution(node, provider, resolution);
    }
    let mesh = provider(node)?;
    Some(seed_operand_mesh(node, mesh))
}

/// Vertices of the clipped mesh must lie on the boolean surface (quality
/// gate — the only gate; no sliver/angle test by design).
fn clip_quality_ok(node: &Node, mesh: &OperandMesh) -> bool {
    if mesh.triangles.is_empty() {
        return false;
    }
    let extent = node_extent(node);
    let values = node.eval(&mesh.vertices);
    let surface_error = values.iter().fold(0.0f64, |acc, value| acc.max(value.abs()));
    surface_error <= CLIP_MAX_RELATIVE_SURFACE_ERROR * extent
}

/// Render a sharp boolean by clipping analytic operand meshes, or None.
pub fn clip_surface(
    node: &Node,
    key: ViewportSurfaceKey,
    color: [f32; 3],
    provider: OperandMeshProvider<'_>,
) -> Option<ViewportSurface> {
    let mesh = clip_boolean_mesh_with_resolution(node, provider, key.resolution)?;
    if !clip_quality_ok(node, &mesh) {
        return None;
    }
    let vertices: Vec<[f32; 3]> = mesh
        .vertices
        .iter()
        .map(|v| [v.x as f32, v.y as f32, v.z as f32])
        .collect();
    let normals: Vec<[f32; 3]> = mesh
        .normals
        .iter()
        .map(|n| [n.x as f32, n.y as f32, n.z as f32])
        .collect();
    let indices: Vec<u32> = mesh.triangles.iter().flatten().copied().collect();
    let triangle_count = indices.len() / 3;
    let wire = if triangle_count <= MAX_DUAL_CONTOUR_WIREFRAME_TRIANGLES {
        wire_indices_from_triangles(&indices)
    } else {
        Vec::new()
    };
    let mut bounds_min = [f64::INFINITY; 3];
    let mut bounds_max = [f64::NEG_INFINITY; 3];
    for vertex in &mesh.vertices {
        bounds_min[0] = bounds_min[0].min(vertex.x);
        bounds_min[1] = bounds_min[1].min(vertex.y);
        bounds_min[2] = bounds_min[2].min(vertex.z);
        bounds_max[0] = bounds_max[0].max(vertex.x);
        bounds_max[1] = bounds_max[1].max(vertex.y);
        bounds_max[2] = bounds_max[2].max(vertex.z);
    }
    Some(ViewportSurface {
        key,
        object_kind: node.kind().to_string(),
        status: SurfaceStatus::Ready,
        vertices,
        normals,
        indices,
        wire_indices: wire,
        color,
        bounds_min,
        bounds_max,
        message: "boolean rendered as SDF-clipped analytic meshes".to_string(),
    })
}
