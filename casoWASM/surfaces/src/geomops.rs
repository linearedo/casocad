//! Shared surface-geometry primitives used by both rendering strategies:
//! triangle orientation, normals, wireframe edges, SDF gradient/edge
//! root-finding, and edge-split subdivision. Ported from
//! `app/viewport/surface_geomops.py`.

use caso_kernel::sdf::node::Node;
use caso_kernel::vec3::{vec3, Vec3};

/// Edge-based 1/2/3-split subdivision (red-green, crack-free).
///
/// `mXY` is the inserted midpoint vertex id for edge (vX, vY) or `None`. The
/// split count per triangle selects the sub-triangulation; the shared
/// midpoint of a split edge is identical for both incident triangles, so no
/// T-junctions form.
pub fn split_marked_triangles(
    tris: &[[u32; 3]],
    m01: &[Option<u32>],
    m12: &[Option<u32>],
    m20: &[Option<u32>],
) -> Vec<[u32; 3]> {
    let mut out = Vec::with_capacity(tris.len() * 2);
    for (index, tri) in tris.iter().enumerate() {
        let [a, b, c] = *tri;
        match (m01[index], m12[index], m20[index]) {
            (None, None, None) => out.push([a, b, c]),
            (Some(p), Some(q), Some(r)) => {
                out.push([a, p, r]);
                out.push([p, b, q]);
                out.push([r, q, c]);
                out.push([p, q, r]);
            }
            (Some(p), None, None) => {
                out.push([a, p, c]);
                out.push([p, b, c]);
            }
            (None, Some(p), None) => {
                out.push([a, b, p]);
                out.push([a, p, c]);
            }
            (None, None, Some(p)) => {
                out.push([a, b, p]);
                out.push([b, c, p]);
            }
            (Some(p), Some(q), None) => {
                out.push([a, p, c]);
                out.push([p, b, q]);
                out.push([p, q, c]);
            }
            (None, Some(q), Some(r)) => {
                out.push([b, q, a]);
                out.push([q, c, r]);
                out.push([q, r, a]);
            }
            (Some(p), None, Some(r)) => {
                out.push([c, r, b]);
                out.push([r, a, p]);
                out.push([r, p, b]);
            }
        }
    }
    out
}

fn eval_batch(node: &Node, points: &[Vec3]) -> Vec<f64> {
    node.eval(points)
}

/// Central-difference gradient of the analytic SDF at scattered points.
/// Returns the raw (unnormalised) gradients.
pub fn analytic_gradient(node: &Node, points: &[Vec3], eps: Vec3) -> Vec<Vec3> {
    let mut samples = Vec::with_capacity(points.len() * 6);
    let offsets = [
        vec3(eps.x, 0.0, 0.0),
        vec3(-eps.x, 0.0, 0.0),
        vec3(0.0, eps.y, 0.0),
        vec3(0.0, -eps.y, 0.0),
        vec3(0.0, 0.0, eps.z),
        vec3(0.0, 0.0, -eps.z),
    ];
    for offset in offsets {
        for point in points {
            samples.push(*point + offset);
        }
    }
    let field = eval_batch(node, &samples);
    let n = points.len();
    (0..n)
        .map(|index| {
            vec3(
                (field[index] - field[n + index]) / (2.0 * eps.x),
                (field[2 * n + index] - field[3 * n + index]) / (2.0 * eps.y),
                (field[4 * n + index] - field[5 * n + index]) / (2.0 * eps.z),
            )
        })
        .collect()
}

/// Exact Hermite data for a batch of sign-crossing edges.
///
/// Finds the analytic zero of `node` along each edge with the Illinois
/// variant of regula falsi — always bracketed by the [a, b] sign change —
/// then samples the exact analytic gradient at the root. Returns
/// (points, raw gradients), accurate to SDF tolerance independent of grid
/// resolution.
pub fn refine_edge_hermite(
    node: &Node,
    point_a: &[Vec3],
    point_b: &[Vec3],
    fa: &[f64],
    fb: &[f64],
    eps: Vec3,
    iterations: usize,
) -> (Vec<Vec3>, Vec<Vec3>) {
    let n = point_a.len();
    if n == 0 {
        return (Vec::new(), Vec::new());
    }
    let direction: Vec<Vec3> = point_a
        .iter()
        .zip(point_b.iter())
        .map(|(a, b)| *b - *a)
        .collect();
    let mut lo = vec![0.0f64; n];
    let mut hi = vec![1.0f64; n];
    let mut f_lo = fa.to_vec();
    let mut f_hi = fb.to_vec();
    let mut t = vec![0.0f64; n];
    let mut kept_lo_last = vec![false; n];
    let mut kept_hi_last = vec![false; n];
    for _ in 0..iterations {
        let mut points = Vec::with_capacity(n);
        for index in 0..n {
            let denom = f_hi[index] - f_lo[index];
            t[index] = if denom.abs() > 1.0e-300 {
                lo[index] - f_lo[index] * (hi[index] - lo[index]) / denom
            } else {
                0.5 * (lo[index] + hi[index])
            };
            points.push(point_a[index] + direction[index] * t[index]);
        }
        let field = eval_batch(node, &points);
        for index in 0..n {
            let f = field[index];
            // Keep the half of the bracket that still straddles the sign change.
            let keep_lo = (f >= 0.0) == (f_hi[index] >= 0.0);
            if keep_lo {
                hi[index] = t[index];
                f_hi[index] = f;
            } else {
                lo[index] = t[index];
                f_lo[index] = f;
            }
            // Illinois: halve the stale endpoint when it is retained twice.
            if keep_lo && kept_lo_last[index] {
                f_lo[index] *= 0.5;
            }
            if !keep_lo && kept_hi_last[index] {
                f_hi[index] *= 0.5;
            }
            kept_lo_last[index] = keep_lo;
            kept_hi_last[index] = !keep_lo;
        }
    }
    let points: Vec<Vec3> = (0..n)
        .map(|index| point_a[index] + direction[index] * t[index])
        .collect();
    let gradients = analytic_gradient(node, &points, eps);
    (points, gradients)
}

/// Make winding consistent (against averaged vertex normals) and drop
/// degenerate triangles. Topology-preserving.
pub fn orient_triangles(
    vertices: &[[f32; 3]],
    normals: &[[f32; 3]],
    indices: &[u32],
) -> Vec<u32> {
    if indices.is_empty() {
        return Vec::new();
    }
    let mut out = Vec::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        let a = vertices[tri[0] as usize].map(f64::from);
        let b = vertices[tri[1] as usize].map(f64::from);
        let c = vertices[tri[2] as usize].map(f64::from);
        let ab = vec3(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
        let ac = vec3(c[0] - a[0], c[1] - a[1], c[2] - a[2]);
        let face = ab.cross(ac);
        if face.length() <= 1.0e-14 {
            continue;
        }
        let na = normals[tri[0] as usize].map(f64::from);
        let nb = normals[tri[1] as usize].map(f64::from);
        let nc = normals[tri[2] as usize].map(f64::from);
        let vnorm = vec3(
            (na[0] + nb[0] + nc[0]) / 3.0,
            (na[1] + nb[1] + nc[1]) / 3.0,
            (na[2] + nb[2] + nc[2]) / 3.0,
        );
        if face.dot(vnorm) < 0.0 {
            out.extend_from_slice(&[tri[0], tri[2], tri[1]]);
        } else {
            out.extend_from_slice(&[tri[0], tri[1], tri[2]]);
        }
    }
    out
}

/// Unique undirected edges of a triangle list, sorted, flattened.
pub fn wire_indices_from_triangles(indices: &[u32]) -> Vec<u32> {
    if indices.is_empty() {
        return Vec::new();
    }
    let mut edges: Vec<(u32, u32)> = Vec::with_capacity(indices.len());
    for tri in indices.chunks_exact(3) {
        for (a, b) in [(tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])] {
            edges.push(if a <= b { (a, b) } else { (b, a) });
        }
    }
    edges.sort_unstable();
    edges.dedup();
    edges.into_iter().flat_map(|(a, b)| [a, b]).collect()
}

/// Area-weighted-ish vertex normals accumulated from unit face normals,
/// with (0,0,1) fallback for isolated vertices.
pub fn mesh_normals(vertices: &[[f32; 3]], indices: &[u32]) -> Vec<[f32; 3]> {
    let mut normals = vec![[0.0f64; 3]; vertices.len()];
    for tri in indices.chunks_exact(3) {
        let a = vertices[tri[0] as usize].map(f64::from);
        let b = vertices[tri[1] as usize].map(f64::from);
        let c = vertices[tri[2] as usize].map(f64::from);
        let ab = vec3(b[0] - a[0], b[1] - a[1], b[2] - a[2]);
        let ac = vec3(c[0] - a[0], c[1] - a[1], c[2] - a[2]);
        let face = ab.cross(ac);
        let length = face.length();
        if length <= 1.0e-12 {
            continue;
        }
        let unit = face / length;
        for vertex in tri {
            let entry = &mut normals[*vertex as usize];
            entry[0] += unit.x;
            entry[1] += unit.y;
            entry[2] += unit.z;
        }
    }
    normals
        .into_iter()
        .map(|normal| {
            let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2])
                .sqrt();
            if length <= 1.0e-12 {
                [0.0, 0.0, 1.0]
            } else {
                [
                    (normal[0] / length) as f32,
                    (normal[1] / length) as f32,
                    (normal[2] / length) as f32,
                ]
            }
        })
        .collect()
}

pub fn normalize_or_z(vector: Vec3) -> Vec3 {
    let length = vector.length();
    if length <= 1.0e-12 || !length.is_finite() {
        return vec3(0.0, 0.0, 1.0);
    }
    vector / length
}
