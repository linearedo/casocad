//! Shortest on-surface paths and knife-ghost construction for the boundary
//! cutter — ports of `core/boundary_paths.py` plus the knife builders from
//! the Python main window (`_straight_knife_handle`, point-cutter ghosts).
//!
//! Discrete curve-shortening constrained to the boundary: midpoint smoothing
//! pulls the polyline taut, Newton projection (`p -= f·∇f/|∇f|²`) pulls it
//! back onto the zero set, endpoints stay pinned. Everything is
//! scale-relative to the Domain bounding diagonal, never absolute meters.

use crate::error::{GeometryError, GeometryResult};
use crate::sdf::curtain::NormalCurtain;
use crate::sdf::node::{Node, Shape};
use crate::sdf::placed::PlacedSdf2D;
use crate::sdf::primitives_2d::Profile2D;
use crate::vec3::{vec3, Vec3};

pub const RELATIVE_SAMPLE_SPACING: f64 = 0.02;
pub const MINIMUM_SEGMENTS: usize = 8;
pub const MAXIMUM_SEGMENTS: usize = 96;
pub const SMOOTHING_ITERATIONS: usize = 48;
pub const PROJECTION_STEPS: usize = 4;

fn bounding_diagonal(root: &Node) -> f64 {
    let diagonal = root
        .bounding_box()
        .map(|bounds| bounds.diagonal())
        .unwrap_or(1.0);
    if !diagonal.is_finite() || diagonal <= 0.0 {
        1.0
    } else {
        diagonal
    }
}

fn gradient(root: &Node, point: Vec3, step: f64) -> Vec3 {
    vec3(
        root.eval_point(point + vec3(step, 0.0, 0.0))
            - root.eval_point(point - vec3(step, 0.0, 0.0)),
        root.eval_point(point + vec3(0.0, step, 0.0))
            - root.eval_point(point - vec3(0.0, step, 0.0)),
        root.eval_point(point + vec3(0.0, 0.0, step))
            - root.eval_point(point - vec3(0.0, 0.0, step)),
    ) * (1.0 / (2.0 * step))
}

/// Newton-project points onto the |f| = 0 boundary (`project_to_surface`).
pub fn project_to_surface(root: &Node, points: &mut [Vec3], step: f64, iterations: usize) {
    for _ in 0..iterations {
        for point in points.iter_mut() {
            let value = root.eval_point(*point);
            let grad = gradient(root, *point, step);
            let squared = grad.dot(grad).max(1.0e-18);
            *point = *point - grad * (value / squared);
        }
    }
}

fn resample_by_arclength(points: &[Vec3], segment_count: usize) -> Vec<Vec3> {
    let mut cumulative = Vec::with_capacity(points.len());
    cumulative.push(0.0);
    for pair in points.windows(2) {
        let last = *cumulative.last().expect("nonempty");
        cumulative.push(last + (pair[1] - pair[0]).length());
    }
    let total = *cumulative.last().expect("nonempty");
    if total <= 0.0 {
        return points.to_vec();
    }
    let mut result = Vec::with_capacity(segment_count + 1);
    for i in 0..=segment_count {
        let target = total * (i as f64) / (segment_count as f64);
        let index = cumulative
            .iter()
            .position(|value| *value >= target)
            .unwrap_or(cumulative.len() - 1)
            .max(1);
        let lower = cumulative[index - 1];
        let upper = cumulative[index];
        let alpha = if upper > lower {
            (target - lower) / (upper - lower)
        } else {
            0.0
        };
        result.push(points[index - 1] + (points[index] - points[index - 1]) * alpha);
    }
    result
}

/// Approximate shortest on-boundary path from `start` to `end`
/// (`surface_shortest_path`). Both endpoints must already lie on the
/// boundary; they are pinned exactly.
pub fn surface_shortest_path(root: &Node, start: Vec3, end: Vec3) -> GeometryResult<Vec<Vec3>> {
    let diagonal = bounding_diagonal(root);
    let chord = (end - start).length();
    if chord <= 1.0e-9 * diagonal {
        return Err(GeometryError::new("smooth polyline points must be distinct"));
    }
    let segment_count = ((chord / (RELATIVE_SAMPLE_SPACING * diagonal)).ceil() as usize)
        .clamp(MINIMUM_SEGMENTS, MAXIMUM_SEGMENTS);
    let mut path: Vec<Vec3> = (0..=segment_count)
        .map(|i| {
            let t = (i as f64) / (segment_count as f64);
            start * (1.0 - t) + end * t
        })
        .collect();
    let step = (diagonal * 1.0e-5).max(1.0e-9);
    project_to_surface(root, &mut path, step, PROJECTION_STEPS);
    path[0] = start;
    *path.last_mut().expect("nonempty") = end;
    for _ in 0..SMOOTHING_ITERATIONS {
        let snapshot = path.clone();
        for i in 1..path.len() - 1 {
            path[i] = snapshot[i - 1] * 0.25 + snapshot[i] * 0.5 + snapshot[i + 1] * 0.25;
        }
        project_to_surface(root, &mut path, step, 1);
        path[0] = start;
        *path.last_mut().expect("nonempty") = end;
    }
    let mut path = resample_by_arclength(&path, segment_count);
    project_to_surface(root, &mut path, step, 2);
    path[0] = start;
    *path.last_mut().expect("nonempty") = end;
    Ok(path)
}

/// Unit boundary normals along an on-surface path (`surface_path_normals`).
pub fn surface_path_normals(root: &Node, path: &[Vec3]) -> GeometryResult<Vec<Vec3>> {
    let diagonal = bounding_diagonal(root);
    let step = (diagonal * 1.0e-5).max(1.0e-9);
    let mut normals = Vec::with_capacity(path.len());
    let mut valid_any = false;
    for point in path {
        let grad = gradient(root, *point, step);
        let length = grad.length();
        if length > 1.0e-12 {
            normals.push(Some(grad * (1.0 / length)));
            valid_any = true;
        } else {
            normals.push(None);
        }
    }
    if !valid_any {
        return Err(GeometryError::new(
            "could not resolve surface normals along the path",
        ));
    }
    // Fill invalid entries from the nearest valid neighbor.
    let valid_indices: Vec<usize> = normals
        .iter()
        .enumerate()
        .filter_map(|(i, normal)| normal.map(|_| i))
        .collect();
    Ok(normals
        .iter()
        .enumerate()
        .map(|(i, normal)| {
            normal.unwrap_or_else(|| {
                let nearest = valid_indices
                    .iter()
                    .min_by_key(|j| i.abs_diff(**j))
                    .expect("has valid");
                normals[*nearest].expect("valid")
            })
        })
        .collect())
}

/// Chain shortest on-boundary paths through the clicked points and wrap them
/// in the NormalCurtain classification field (`smooth_polyline_knife`).
pub fn smooth_polyline_knife(root: &Node, clicked_points: &[Vec3]) -> GeometryResult<Node> {
    let threshold = 1.0e-9 * bounding_diagonal(root);
    let mut distinct: Vec<Vec3> = Vec::new();
    for point in clicked_points {
        if distinct
            .last()
            .map(|last| (*point - *last).length() > threshold)
            .unwrap_or(true)
        {
            distinct.push(*point);
        }
    }
    if distinct.len() < 2 {
        return Err(GeometryError::new(
            "smooth polyline needs at least two distinct points",
        ));
    }
    let mut path: Vec<Vec3> = Vec::new();
    for pair in distinct.windows(2) {
        let leg = surface_shortest_path(root, pair[0], pair[1])?;
        if path.is_empty() {
            path.extend(leg);
        } else {
            path.extend(leg.into_iter().skip(1));
        }
    }
    let normals = surface_path_normals(root, &path)?;
    let bounds = root.bounding_box()?;
    let extent = 4.0
        * (bounds.x_max - bounds.x_min)
            .max(bounds.y_max - bounds.y_min)
            .max(bounds.z_max - bounds.z_min)
            .max(1.0e-6);
    Ok(Node::new(
        "smooth_polyline_knife",
        Shape::NormalCurtain(NormalCurtain::new(path, normals, extent)?),
    ))
}

/// Half-plane ghost: a giant polygon on the plane spanned by the start-end
/// line and the boundary normal, covering one side of the line
/// (`_straight_knife_handle`).
pub fn straight_knife(root: &Node, start: Vec3, end: Vec3, normal: Vec3) -> GeometryResult<Node> {
    let normal_length = normal.length();
    if normal_length <= 1.0e-9 {
        return Err(GeometryError::new(
            "could not resolve a surface normal for the cutter",
        ));
    }
    let normal = normal * (1.0 / normal_length);
    let line = end - start;
    let line_length = line.length();
    if line_length <= 1.0e-9 {
        return Err(GeometryError::new("Planar segment cutter length must be nonzero."));
    }
    let line_axis = line * (1.0 / line_length);
    let mut side_axis = normal.cross(line_axis);
    let side_length = side_axis.length();
    if side_length <= 1.0e-9 {
        return Err(GeometryError::new(
            "Planar segment cutter must lie across the selected boundary.",
        ));
    }
    side_axis = side_axis * (1.0 / side_length);
    let bounds = root.bounding_box()?;
    let span = (bounds.x_max - bounds.x_min)
        .max(bounds.y_max - bounds.y_min)
        .max(bounds.z_max - bounds.z_min)
        .max(line_length)
        .max(1.0)
        * 2.0;
    let midpoint = (start + end) * 0.5;
    let origin = midpoint - line_axis * span;
    let profile = Profile2D::polygon(vec![
        [0.0, 0.0],
        [2.0 * span, 0.0],
        [2.0 * span, 2.0 * span],
        [0.0, 2.0 * span],
    ])?;
    Ok(Node::new(
        "segment_knife",
        Shape::PlacedSdf2D(PlacedSdf2D::new(
            profile,
            origin,
            line_axis,
            side_axis,
            Vec::new(),
        )?),
    ))
}
