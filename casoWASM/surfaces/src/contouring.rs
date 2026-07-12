//! Strategy B (fallback) — dual contouring of an arbitrary SDF.
//! Ported from `app/viewport/surface_contouring.py`.
//!
//! The universal isosurface extractor: renders any bounded 3D SDF from
//! `Node::eval` + `Node::bounding_box` alone. Dense grid at low resolutions;
//! at `resolution >= 96` a sparse narrow band (coarse grid locates the
//! surface, only crossing cells are subdivided) keeps cost on the O(n²)
//! surface band. Exact Hermite edge roots + sharp-feature QEF.

use std::collections::HashMap;

use caso_kernel::sdf::node::Node;
use caso_kernel::vec3::{vec3, Vec3};

use crate::geomops::{
    analytic_gradient, orient_triangles, refine_edge_hermite, wire_indices_from_triangles,
};
use crate::types::{
    empty_surface, SurfaceStatus, ViewportSurface, ViewportSurfaceKey,
};

pub const MAX_DUAL_CONTOUR_WIREFRAME_TRIANGLES: usize = 3000;
const QEF_SINGULAR_RATIO: f64 = 0.03;
const NARROW_BAND_MIN_RES: u32 = 96;
const NARROW_BAND_SUBDIV: i64 = 2;
const MIN_AXIS_CELLS: i64 = 8;
const MAX_AXIS_CELLS: i64 = 96;

const CORNER_OFFSETS: [[i64; 3]; 8] = [
    [0, 0, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 1, 0],
    [0, 0, 1],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
];

const CELL_EDGE_CORNERS: [(usize, usize); 12] = [
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (0, 2),
    (1, 3),
    (4, 6),
    (5, 7),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

pub fn contour_surface(
    node: &Node,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> Result<ViewportSurface, String> {
    if key.resolution >= NARROW_BAND_MIN_RES {
        let base_res = (key.resolution as i64 / NARROW_BAND_SUBDIV).max(MIN_AXIS_CELLS);
        return narrow_band_dual_contour(node, key, color, base_res, NARROW_BAND_SUBDIV);
    }
    dense_dual_contour(node, key, color)
}

fn sampling_bounds(node: &Node, resolution: i64) -> Result<(Vec3, Vec3), String> {
    let bounds = node
        .bounding_box()
        .map_err(|error| format!("viewport surface requires finite SDF bounds: {error}"))?;
    let mut mins = vec3(bounds.x_min, bounds.y_min, bounds.z_min);
    let mut maxs = vec3(bounds.x_max, bounds.y_max, bounds.z_max);
    for value in [mins.x, mins.y, mins.z, maxs.x, maxs.y, maxs.z] {
        if !value.is_finite() {
            return Err("viewport surface requires finite SDF bounds".to_string());
        }
    }
    let extents = (maxs - mins).max(Vec3::ZERO);
    let base = extents.x.max(extents.y).max(extents.z).max(1.0);
    let margin = (base / (resolution.max(1)) as f64)
        .max(base * 0.025)
        .max(1.0e-3);
    let pad = vec3(margin, margin, margin);
    let extra = vec3(
        if extents.x <= 1.0e-9 { margin } else { 0.0 },
        if extents.y <= 1.0e-9 { margin } else { 0.0 },
        if extents.z <= 1.0e-9 { margin } else { 0.0 },
    );
    mins = mins - pad - extra;
    maxs = maxs + pad + extra;
    Ok((mins, maxs))
}

fn axis_cell_counts(mins: Vec3, maxs: Vec3, resolution: i64) -> [i64; 3] {
    let extents = [
        (maxs.x - mins.x).max(1.0e-9),
        (maxs.y - mins.y).max(1.0e-9),
        (maxs.z - mins.z).max(1.0e-9),
    ];
    let largest = extents[0].max(extents[1]).max(extents[2]);
    extents.map(|extent| {
        let scaled = (resolution as f64 * extent / largest).ceil() as i64;
        scaled.clamp(MIN_AXIS_CELLS, MAX_AXIS_CELLS)
    })
}

fn linspace(minimum: f64, maximum: f64, count: usize) -> Vec<f64> {
    if count <= 1 {
        return vec![minimum];
    }
    let step = (maximum - minimum) / (count - 1) as f64;
    (0..count).map(|index| minimum + step * index as f64).collect()
}

/// One DC vertex + normal per crossing cell from exact Hermite data:
/// Illinois edge roots + Lindstrom/Schaefer truncated-eigen QEF.
fn solve_cells_hermite_qef(
    node: &Node,
    bases: &[Vec3],
    step: Vec3,
    corner_values: &[[f64; 8]],
) -> (Vec<Vec3>, Vec<Vec3>) {
    let cell_count = bases.len();
    let grad_eps = vec3(
        (step.x * 0.05).max(1.0e-6),
        (step.y * 0.05).max(1.0e-6),
        (step.z * 0.05).max(1.0e-6),
    );
    // Gather all crossing edges over all cells for one batched refine.
    let mut edge_cell = Vec::new();
    let mut edge_slot = Vec::new();
    let mut batch_a = Vec::new();
    let mut batch_b = Vec::new();
    let mut batch_fa = Vec::new();
    let mut batch_fb = Vec::new();
    for (cell, corner) in corner_values.iter().enumerate() {
        for (slot, (edge_a, edge_b)) in CELL_EDGE_CORNERS.iter().enumerate() {
            let fa = corner[*edge_a];
            let fb = corner[*edge_b];
            let crossing = ((fa <= 0.0 && fb >= 0.0) || (fb <= 0.0 && fa >= 0.0))
                && (fa - fb).abs() > 1.0e-12;
            if crossing {
                let offset_a = CORNER_OFFSETS[*edge_a];
                let offset_b = CORNER_OFFSETS[*edge_b];
                batch_a.push(
                    bases[cell]
                        + vec3(
                            offset_a[0] as f64 * step.x,
                            offset_a[1] as f64 * step.y,
                            offset_a[2] as f64 * step.z,
                        ),
                );
                batch_b.push(
                    bases[cell]
                        + vec3(
                            offset_b[0] as f64 * step.x,
                            offset_b[1] as f64 * step.y,
                            offset_b[2] as f64 * step.z,
                        ),
                );
                batch_fa.push(fa);
                batch_fb.push(fb);
                edge_cell.push(cell);
                edge_slot.push(slot);
            }
        }
    }
    let (refined_points, refined_grads) =
        refine_edge_hermite(node, &batch_a, &batch_b, &batch_fa, &batch_fb, grad_eps, 4);

    // Per-cell accumulation.
    let mut edge_points = vec![[Vec3::ZERO; 12]; cell_count];
    let mut edge_normals = vec![[Vec3::ZERO; 12]; cell_count];
    let mut edge_active = vec![[false; 12]; cell_count];
    let mut normal_valid = vec![[false; 12]; cell_count];
    for (index, cell) in edge_cell.iter().enumerate() {
        let slot = edge_slot[index];
        edge_points[*cell][slot] = refined_points[index];
        edge_active[*cell][slot] = true;
        let grad = refined_grads[index];
        let length = grad.length();
        if length > 1.0e-12 {
            edge_normals[*cell][slot] = grad / length;
            normal_valid[*cell][slot] = true;
        }
    }

    let mut vertices = Vec::with_capacity(cell_count);
    let mut normals = Vec::with_capacity(cell_count);
    for cell in 0..cell_count {
        let mut point_sum = Vec3::ZERO;
        let mut point_count = 0usize;
        for slot in 0..12 {
            if edge_active[cell][slot] {
                point_sum += edge_points[cell][slot];
                point_count += 1;
            }
        }
        let mass_point = if point_count > 0 {
            point_sum / point_count as f64
        } else {
            bases[cell] + step * 0.5
        };

        // QEF: A^T A and A^T b recentred at the mass point.
        let mut ata = [[0.0f64; 3]; 3];
        let mut atb = Vec3::ZERO;
        let mut qef_count = 0usize;
        for slot in 0..12 {
            if !normal_valid[cell][slot] {
                continue;
            }
            qef_count += 1;
            let n = edge_normals[cell][slot];
            let p = edge_points[cell][slot];
            let na = [n.x, n.y, n.z];
            for (i, ni) in na.iter().enumerate() {
                for (j, nj) in na.iter().enumerate() {
                    ata[i][j] += ni * nj;
                }
            }
            atb += n * n.dot(p);
        }
        let rhs = vec3(
            atb.x - (ata[0][0] * mass_point.x + ata[0][1] * mass_point.y + ata[0][2] * mass_point.z),
            atb.y - (ata[1][0] * mass_point.x + ata[1][1] * mass_point.y + ata[1][2] * mass_point.z),
            atb.z - (ata[2][0] * mass_point.x + ata[2][1] * mass_point.y + ata[2][2] * mass_point.z),
        );
        let (eigvals, eigvecs) = symmetric_eigen_3x3(&ata);
        let eig_max = eigvals[2].max(1.0e-30);
        let mut correction = Vec3::ZERO;
        for axis in 0..3 {
            if eigvals[axis] > QEF_SINGULAR_RATIO * eig_max {
                let axis_vec = eigvecs[axis];
                let projection = axis_vec.dot(rhs) / eigvals[axis];
                correction += axis_vec * projection;
            }
        }
        let mut candidate = mass_point + correction;
        if !(candidate.x.is_finite() && candidate.y.is_finite() && candidate.z.is_finite()) {
            candidate = mass_point;
        }
        // Clamp to the cell so a degenerate solve cannot spike outside it.
        let cell_min = bases[cell];
        let cell_max = bases[cell] + step;
        candidate = candidate.max(cell_min).min(cell_max);
        vertices.push(if qef_count > 0 { candidate } else { mass_point });

        let mut normal_sum = Vec3::ZERO;
        for slot in 0..12 {
            if normal_valid[cell][slot] {
                normal_sum += edge_normals[cell][slot];
            }
        }
        let length = normal_sum.length();
        normals.push(if length > 1.0e-12 {
            normal_sum / length
        } else {
            vec3(0.0, 0.0, 1.0)
        });
    }
    (vertices, normals)
}

/// Eigen-decomposition of a symmetric 3x3 matrix via cyclic Jacobi rotations.
/// Returns eigenvalues ascending and their unit eigenvectors (same order).
fn symmetric_eigen_3x3(matrix: &[[f64; 3]; 3]) -> ([f64; 3], [Vec3; 3]) {
    let mut a = *matrix;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _sweep in 0..24 {
        // Largest off-diagonal element.
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        if off < 1.0e-15 {
            break;
        }
        for (p, q) in [(0usize, 1usize), (0, 2), (1, 2)] {
            if a[p][q].abs() < 1.0e-30 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            let app = a[p][p];
            let aqq = a[q][q];
            let apq = a[p][q];
            a[p][p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
            a[q][q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
            a[p][q] = 0.0;
            a[q][p] = 0.0;
            let r = 3 - p - q;
            let arp = a[r][p];
            let arq = a[r][q];
            a[r][p] = c * arp - s * arq;
            a[p][r] = a[r][p];
            a[r][q] = s * arp + c * arq;
            a[q][r] = a[r][q];
            for row in &mut v {
                let vp = row[p];
                let vq = row[q];
                row[p] = c * vp - s * vq;
                row[q] = s * vp + c * vq;
            }
        }
    }
    let mut pairs: [(f64, Vec3); 3] = [
        (a[0][0], vec3(v[0][0], v[1][0], v[2][0])),
        (a[1][1], vec3(v[0][1], v[1][1], v[2][1])),
        (a[2][2], vec3(v[0][2], v[1][2], v[2][2])),
    ];
    pairs.sort_by(|left, right| left.0.partial_cmp(&right.0).unwrap_or(std::cmp::Ordering::Equal));
    (
        [pairs[0].0, pairs[1].0, pairs[2].0],
        [pairs[0].1, pairs[1].1, pairs[2].1],
    )
}

/// Analytic-gradient shading normals with grid-normal fallback.
fn analytic_vertex_normals(
    node: &Node,
    vertices: &[Vec3],
    step: Vec3,
    fallback: &[Vec3],
) -> Vec<[f32; 3]> {
    if vertices.is_empty() {
        return Vec::new();
    }
    let eps = vec3(
        (step.x * 0.25).max(1.0e-5),
        (step.y * 0.25).max(1.0e-5),
        (step.z * 0.25).max(1.0e-5),
    );
    let grads = analytic_gradient(node, vertices, eps);
    grads
        .iter()
        .zip(fallback.iter())
        .map(|(grad, fallback)| {
            let length = grad.length();
            let normal = if length > 1.0e-9 {
                *grad / length
            } else {
                *fallback
            };
            [normal.x as f32, normal.y as f32, normal.z as f32]
        })
        .collect()
}

struct QuadSink {
    indices: Vec<u32>,
}

impl QuadSink {
    #[allow(clippy::too_many_arguments)]
    fn emit(&mut self, a: i64, b: i64, c: i64, d: i64, fa: f64, fb: f64, flip: bool) {
        let crossing = ((fa <= 0.0 && fb >= 0.0) || (fb <= 0.0 && fa >= 0.0))
            && (fa - fb).abs() > 1.0e-12;
        if !crossing || a < 0 || b < 0 || c < 0 || d < 0 {
            return;
        }
        let (a, b, c, d) = (a as u32, b as u32, c as u32, d as u32);
        if flip {
            self.indices.extend_from_slice(&[a, c, b, a, d, c]);
        } else {
            self.indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn finish_surface(
    node: &Node,
    key: ViewportSurfaceKey,
    color: [f32; 3],
    mins: Vec3,
    maxs: Vec3,
    vertices: Vec<Vec3>,
    normals: Vec<Vec3>,
    step: Vec3,
    indices: Vec<u32>,
) -> ViewportSurface {
    let vertex_array: Vec<[f32; 3]> = vertices
        .iter()
        .map(|vertex| [vertex.x as f32, vertex.y as f32, vertex.z as f32])
        .collect();
    let normal_array = analytic_vertex_normals(node, &vertices, step, &normals);
    let index_array = orient_triangles(&vertex_array, &normal_array, &indices);
    let triangle_count = index_array.len() / 3;
    let wire = if triangle_count <= MAX_DUAL_CONTOUR_WIREFRAME_TRIANGLES {
        wire_indices_from_triangles(&index_array)
    } else {
        Vec::new()
    };
    let status = if index_array.is_empty() {
        SurfaceStatus::Empty
    } else {
        SurfaceStatus::Ready
    };
    let message = if index_array.is_empty() {
        "dual contour produced no faces".to_string()
    } else {
        String::new()
    };
    ViewportSurface {
        key,
        object_kind: node.kind().to_string(),
        status,
        vertices: vertex_array,
        normals: normal_array,
        indices: index_array,
        wire_indices: wire,
        color,
        alpha: 1.0,
        bounds_min: [mins.x, mins.y, mins.z],
        bounds_max: [maxs.x, maxs.y, maxs.z],
        message,
    }
}

fn dense_dual_contour(
    node: &Node,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> Result<ViewportSurface, String> {
    let (mins, maxs) = sampling_bounds(node, key.resolution as i64)?;
    let dims = axis_cell_counts(mins, maxs, key.resolution as i64);
    let (nx, ny, nz) = (dims[0] as usize, dims[1] as usize, dims[2] as usize);
    let xs = linspace(mins.x, maxs.x, nx + 1);
    let ys = linspace(mins.y, maxs.y, ny + 1);
    let zs = linspace(mins.z, maxs.z, nz + 1);
    let mut points = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    for x in &xs {
        for y in &ys {
            for z in &zs {
                points.push(vec3(*x, *y, *z));
            }
        }
    }
    let values = node.eval(&points);
    let value_at = |i: usize, j: usize, k: usize| values[(i * (ny + 1) + j) * (nz + 1) + k];
    let has_negative = values.iter().any(|value| *value <= 0.0);
    let has_positive = values.iter().any(|value| *value >= 0.0);
    if !(has_negative && has_positive) {
        return Ok(empty_surface(
            node,
            key,
            color,
            "no zero crossing in viewport bounds",
        ));
    }

    // Crossing cells + their corner values.
    let mut cell_vertex = vec![-1i64; nx * ny * nz];
    let cell_index = |i: usize, j: usize, k: usize| (i * ny + j) * nz + k;
    let mut bases = Vec::new();
    let mut corner_values = Vec::new();
    let mut cell_coords = Vec::new();
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
                let mut corners = [0.0f64; 8];
                for (slot, offset) in CORNER_OFFSETS.iter().enumerate() {
                    corners[slot] = value_at(
                        i + offset[0] as usize,
                        j + offset[1] as usize,
                        k + offset[2] as usize,
                    );
                }
                let value_min = corners.iter().cloned().fold(f64::INFINITY, f64::min);
                let value_max = corners.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                if value_min <= 0.0 && value_max >= 0.0 && (value_max - value_min) > 1.0e-12 {
                    cell_vertex[cell_index(i, j, k)] = bases.len() as i64;
                    bases.push(vec3(xs[i], ys[j], zs[k]));
                    corner_values.push(corners);
                    cell_coords.push((i, j, k));
                }
            }
        }
    }
    if bases.is_empty() {
        return Ok(empty_surface(node, key, color, "dual contour produced no cells"));
    }
    let step = vec3(xs[1] - xs[0], ys[1] - ys[0], zs[1] - zs[0]);
    let (vertices, normals) = solve_cells_hermite_qef(node, &bases, step, &corner_values);

    // Triangles from grid edges shared by four cells.
    let cv = |i: i64, j: i64, k: i64| -> i64 {
        if i < 0 || j < 0 || k < 0 || i >= nx as i64 || j >= ny as i64 || k >= nz as i64 {
            -1
        } else {
            cell_vertex[cell_index(i as usize, j as usize, k as usize)]
        }
    };
    let mut sink = QuadSink { indices: Vec::new() };
    // X-parallel edges.
    for i in 0..nx {
        for jj in 0..ny.saturating_sub(1) {
            for kk in 0..nz.saturating_sub(1) {
                let fa = value_at(i, jj + 1, kk + 1);
                let fb = value_at(i + 1, jj + 1, kk + 1);
                sink.emit(
                    cv(i as i64, jj as i64, kk as i64),
                    cv(i as i64, jj as i64 + 1, kk as i64),
                    cv(i as i64, jj as i64 + 1, kk as i64 + 1),
                    cv(i as i64, jj as i64, kk as i64 + 1),
                    fa,
                    fb,
                    fa > 0.0,
                );
            }
        }
    }
    // Y-parallel edges.
    for ii in 0..nx.saturating_sub(1) {
        for j in 0..ny {
            for kk in 0..nz.saturating_sub(1) {
                let fa = value_at(ii + 1, j, kk + 1);
                let fb = value_at(ii + 1, j + 1, kk + 1);
                sink.emit(
                    cv(ii as i64, j as i64, kk as i64),
                    cv(ii as i64 + 1, j as i64, kk as i64),
                    cv(ii as i64 + 1, j as i64, kk as i64 + 1),
                    cv(ii as i64, j as i64, kk as i64 + 1),
                    fa,
                    fb,
                    fa <= 0.0,
                );
            }
        }
    }
    // Z-parallel edges.
    for ii in 0..nx.saturating_sub(1) {
        for jj in 0..ny.saturating_sub(1) {
            for k in 0..nz {
                let fa = value_at(ii + 1, jj + 1, k);
                let fb = value_at(ii + 1, jj + 1, k + 1);
                sink.emit(
                    cv(ii as i64, jj as i64, k as i64),
                    cv(ii as i64 + 1, jj as i64, k as i64),
                    cv(ii as i64 + 1, jj as i64 + 1, k as i64),
                    cv(ii as i64, jj as i64 + 1, k as i64),
                    fa,
                    fb,
                    fa > 0.0,
                );
            }
        }
    }
    let _ = cell_coords;
    Ok(finish_surface(
        node, key, color, mins, maxs, vertices, normals, step, sink.indices,
    ))
}

const DC_EDGE_SPECS: [(usize, usize, [[i64; 3]; 4], bool); 3] = [
    (0, 1, [[0, -1, -1], [0, 0, -1], [0, 0, 0], [0, -1, 0]], true),
    (0, 2, [[-1, 0, -1], [0, 0, -1], [0, 0, 0], [-1, 0, 0]], false),
    (0, 4, [[-1, -1, 0], [0, -1, 0], [0, 0, 0], [-1, 0, 0]], true),
];

fn narrow_band_dual_contour(
    node: &Node,
    key: ViewportSurfaceKey,
    color: [f32; 3],
    base_res: i64,
    subdiv: i64,
) -> Result<ViewportSurface, String> {
    let (mins, maxs) = sampling_bounds(node, base_res)?;
    let cdims = axis_cell_counts(mins, maxs, base_res);
    let (cnx, cny, cnz) = (cdims[0] as usize, cdims[1] as usize, cdims[2] as usize);
    let cxs = linspace(mins.x, maxs.x, cnx + 1);
    let cys = linspace(mins.y, maxs.y, cny + 1);
    let czs = linspace(mins.z, maxs.z, cnz + 1);
    let mut coarse_points = Vec::with_capacity((cnx + 1) * (cny + 1) * (cnz + 1));
    for x in &cxs {
        for y in &cys {
            for z in &czs {
                coarse_points.push(vec3(*x, *y, *z));
            }
        }
    }
    let cvals = node.eval(&coarse_points);
    let cvalue_at = |i: usize, j: usize, k: usize| cvals[(i * (cny + 1) + j) * (cnz + 1) + k];
    if !(cvals.iter().any(|value| *value <= 0.0) && cvals.iter().any(|value| *value >= 0.0)) {
        return Ok(empty_surface(
            node,
            key,
            color,
            "no zero crossing in viewport bounds",
        ));
    }

    // Coarse crossing cells.
    let mut coarse_cells = Vec::new();
    for i in 0..cnx {
        for j in 0..cny {
            for k in 0..cnz {
                let mut value_min = f64::INFINITY;
                let mut value_max = f64::NEG_INFINITY;
                for offset in CORNER_OFFSETS {
                    let value = cvalue_at(
                        i + offset[0] as usize,
                        j + offset[1] as usize,
                        k + offset[2] as usize,
                    );
                    value_min = value_min.min(value);
                    value_max = value_max.max(value);
                }
                if value_min <= 0.0 && value_max >= 0.0 && (value_max - value_min) > 1.0e-12 {
                    coarse_cells.push([i as i64, j as i64, k as i64]);
                }
            }
        }
    }
    if coarse_cells.is_empty() {
        return Ok(empty_surface(node, key, color, "dual contour produced no cells"));
    }

    // Dilate by the 18-neighbourhood (|dx|+|dy|+|dz| <= 2).
    let mut dilated: Vec<[i64; 3]> = Vec::new();
    let mut seen_coarse: HashMap<i64, ()> = HashMap::new();
    for cell in &coarse_cells {
        for dx in -1..=1i64 {
            for dy in -1..=1i64 {
                for dz in -1..=1i64 {
                    if dx.abs() + dy.abs() + dz.abs() > 2 {
                        continue;
                    }
                    let candidate = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                    if candidate[0] < 0
                        || candidate[1] < 0
                        || candidate[2] < 0
                        || candidate[0] >= cdims[0]
                        || candidate[1] >= cdims[1]
                        || candidate[2] >= cdims[2]
                    {
                        continue;
                    }
                    let lid = (candidate[0] * cdims[1] + candidate[1]) * cdims[2] + candidate[2];
                    if seen_coarse.insert(lid, ()).is_none() {
                        dilated.push(candidate);
                    }
                }
            }
        }
    }

    // Fine cells inside the dilated band.
    let s = subdiv;
    let fdims = [cdims[0] * s, cdims[1] * s, cdims[2] * s];
    let mut fine_cells: Vec<[i64; 3]> = Vec::with_capacity(dilated.len() * (s * s * s) as usize);
    let mut seen_fine: HashMap<i64, ()> = HashMap::new();
    for cell in &dilated {
        for cx in 0..s {
            for cy in 0..s {
                for cz in 0..s {
                    let fine = [cell[0] * s + cx, cell[1] * s + cy, cell[2] * s + cz];
                    let flid = (fine[0] * fdims[1] + fine[1]) * fdims[2] + fine[2];
                    if seen_fine.insert(flid, ()).is_none() {
                        fine_cells.push(fine);
                    }
                }
            }
        }
    }

    // Evaluate the SDF once per unique fine corner point in the band.
    let py = fdims[1] + 1;
    let pz = fdims[2] + 1;
    let step_f = vec3(
        (maxs.x - mins.x) / fdims[0] as f64,
        (maxs.y - mins.y) / fdims[1] as f64,
        (maxs.z - mins.z) / fdims[2] as f64,
    );
    let mut corner_pid: Vec<[i64; 8]> = Vec::with_capacity(fine_cells.len());
    let mut pid_index: HashMap<i64, usize> = HashMap::new();
    let mut unique_points: Vec<Vec3> = Vec::new();
    for cell in &fine_cells {
        let mut pids = [0i64; 8];
        for (slot, offset) in CORNER_OFFSETS.iter().enumerate() {
            let point = [cell[0] + offset[0], cell[1] + offset[1], cell[2] + offset[2]];
            let pid = (point[0] * py + point[1]) * pz + point[2];
            pids[slot] = pid;
            pid_index.entry(pid).or_insert_with(|| {
                unique_points.push(vec3(
                    mins.x + point[0] as f64 * step_f.x,
                    mins.y + point[1] as f64 * step_f.y,
                    mins.z + point[2] as f64 * step_f.z,
                ));
                unique_points.len() - 1
            });
        }
        corner_pid.push(pids);
    }
    let field = node.eval(&unique_points);

    // Crossing fine cells.
    let mut cells: Vec<[i64; 3]> = Vec::new();
    let mut cell_corner_values: Vec<[f64; 8]> = Vec::new();
    for (cell, pids) in fine_cells.iter().zip(corner_pid.iter()) {
        let mut corners = [0.0f64; 8];
        for (slot, pid) in pids.iter().enumerate() {
            corners[slot] = field[pid_index[pid]];
        }
        let value_min = corners.iter().cloned().fold(f64::INFINITY, f64::min);
        let value_max = corners.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        if value_min <= 0.0 && value_max >= 0.0 && (value_max - value_min) > 1.0e-12 {
            cells.push(*cell);
            cell_corner_values.push(corners);
        }
    }
    if cells.is_empty() {
        return Ok(empty_surface(node, key, color, "dual contour produced no cells"));
    }
    let bases: Vec<Vec3> = cells
        .iter()
        .map(|cell| {
            vec3(
                mins.x + cell[0] as f64 * step_f.x,
                mins.y + cell[1] as f64 * step_f.y,
                mins.z + cell[2] as f64 * step_f.z,
            )
        })
        .collect();
    let (vertices, normals) = solve_cells_hermite_qef(node, &bases, step_f, &cell_corner_values);

    // Cell lid -> vertex id lookup.
    let mut vid_by_lid: HashMap<i64, i64> = HashMap::with_capacity(cells.len());
    for (vid, cell) in cells.iter().enumerate() {
        let lid = (cell[0] * fdims[1] + cell[1]) * fdims[2] + cell[2];
        vid_by_lid.insert(lid, vid as i64);
    }
    let lookup = |cell: [i64; 3]| -> i64 {
        if cell[0] < 0
            || cell[1] < 0
            || cell[2] < 0
            || cell[0] >= fdims[0]
            || cell[1] >= fdims[1]
            || cell[2] >= fdims[2]
        {
            return -1;
        }
        let lid = (cell[0] * fdims[1] + cell[1]) * fdims[2] + cell[2];
        *vid_by_lid.get(&lid).unwrap_or(&-1)
    };

    let mut sink = QuadSink { indices: Vec::new() };
    for (corner_a, corner_b, offsets, flip_positive) in DC_EDGE_SPECS {
        for (cell_index, cell) in cells.iter().enumerate() {
            let fa = cell_corner_values[cell_index][corner_a];
            let fb = cell_corner_values[cell_index][corner_b];
            let c0 = cell_corner_values[cell_index][0];
            let flip = if flip_positive { c0 > 0.0 } else { c0 <= 0.0 };
            let vids: Vec<i64> = offsets
                .iter()
                .map(|offset| {
                    lookup([
                        cell[0] + offset[0],
                        cell[1] + offset[1],
                        cell[2] + offset[2],
                    ])
                })
                .collect();
            sink.emit(vids[0], vids[1], vids[2], vids[3], fa, fb, flip);
        }
    }

    Ok(finish_surface(
        node, key, color, mins, maxs, vertices, normals, step_f, sink.indices,
    ))
}
