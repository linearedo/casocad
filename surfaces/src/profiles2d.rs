//! 2D profile outlines and filled-surface builders (placed 1D/2D objects):
//! analytic outlines, marching-squares contouring with ring stitching,
//! ear-clip triangulation, and the sampled-cell fallback.
//! Ported from the 2D half of `app/viewport/surface_builder.py`.

use caso_kernel::sdf::node::Node;
use caso_kernel::sdf::placed::{PlacedPolyline1D, PlacedSdf1D, PlacedSdf2D};
use caso_kernel::sdf::primitives_2d::{Point2, Profile2D};
use caso_kernel::vec3::Vec3;

use crate::types::{
    empty_surface, SurfaceStatus, ViewportSurface, ViewportSurfaceKey,
};

const MAX_CONTOURED_2D_CELLS: u32 = 96;
const MAX_SAMPLED_2D_CELLS: u32 = 48;

// ---------------------------------------------------------------------------
// Analytic profile outlines

pub fn profile_outline(profile: &Profile2D, resolution: u32) -> Vec<Point2> {
    match profile {
        Profile2D::QuadraticBezierSurface { points } => {
            quadratic_surface_outline(points, resolution)
        }
        Profile2D::Circle { center, radius } => {
            ellipse_outline(*center, [*radius, *radius], resolution)
        }
        Profile2D::Ellipse { center, semi_axes } => ellipse_outline(*center, *semi_axes, resolution),
        Profile2D::RoundedRectangle {
            center,
            half_size,
            corner_radius,
        } => rounded_rectangle_outline(*center, *half_size, *corner_radius, resolution),
        Profile2D::Square { center, half_size } => {
            rectangle_outline(*center, [*half_size, *half_size])
        }
        Profile2D::Rectangle { center, half_size } => rectangle_outline(*center, *half_size),
        Profile2D::RegularPolygon {
            center,
            radius,
            side_count,
            rotation,
        } => (0..*side_count)
            .map(|index| {
                let angle =
                    rotation + index as f64 * 2.0 * std::f64::consts::PI / *side_count as f64;
                [center[0] + radius * angle.cos(), center[1] + radius * angle.sin()]
            })
            .collect(),
        Profile2D::Polygon { points } => closed_points(points),
        Profile2D::Offset { child, offset } => profile_outline(child, resolution)
            .into_iter()
            .map(|point| [point[0] + offset[0], point[1] + offset[1]])
            .collect(),
        _ => Vec::new(),
    }
}

fn ellipse_outline(center: Point2, semi_axes: Point2, resolution: u32) -> Vec<Point2> {
    let segments = (resolution as usize * 4).max(40);
    (0..segments)
        .map(|index| {
            let theta = 2.0 * std::f64::consts::PI * index as f64 / segments as f64;
            [
                center[0] + semi_axes[0] * theta.cos(),
                center[1] + semi_axes[1] * theta.sin(),
            ]
        })
        .collect()
}

fn rectangle_outline(center: Point2, half_size: Point2) -> Vec<Point2> {
    let [cu, cv] = center;
    let [hu, hv] = half_size;
    vec![
        [cu - hu, cv - hv],
        [cu + hu, cv - hv],
        [cu + hu, cv + hv],
        [cu - hu, cv + hv],
    ]
}

fn rounded_rectangle_outline(
    center: Point2,
    half_size: Point2,
    corner_radius: f64,
    resolution: u32,
) -> Vec<Point2> {
    let [cu, cv] = center;
    let inner_u = half_size[0] - corner_radius;
    let inner_v = half_size[1] - corner_radius;
    let arc_steps = ((resolution / 2) as usize).max(6);
    let pi = std::f64::consts::PI;
    let corners = [
        (cu + inner_u, cv - inner_v, -0.5 * pi, 0.0),
        (cu + inner_u, cv + inner_v, 0.0, 0.5 * pi),
        (cu - inner_u, cv + inner_v, 0.5 * pi, pi),
        (cu - inner_u, cv - inner_v, pi, 1.5 * pi),
    ];
    let mut outline = Vec::new();
    for (corner_u, corner_v, start, end) in corners {
        for step in 0..=arc_steps {
            if !outline.is_empty() && step == 0 {
                continue;
            }
            let theta = start + (end - start) * step as f64 / arc_steps as f64;
            outline.push([
                corner_u + corner_radius * theta.cos(),
                corner_v + corner_radius * theta.sin(),
            ]);
        }
    }
    outline
}

fn closed_points(points: &[Point2]) -> Vec<Point2> {
    let mut outline: Vec<Point2> = points.to_vec();
    if outline.len() >= 2 && outline[0] == outline[outline.len() - 1] {
        outline.pop();
    }
    outline
}

pub fn sample_quadratic_curve(points: &[Point2], resolution: u32) -> Vec<Point2> {
    let steps = (resolution as usize * 2).max(12);
    let mut sampled: Vec<Point2> = Vec::new();
    let mut span_start = 0usize;
    while span_start + 2 < points.len() + 1 && span_start + 2 <= points.len() - 1 {
        let a = points[span_start];
        let b = points[span_start + 1];
        let c = points[span_start + 2];
        for step in 0..=steps {
            if !sampled.is_empty() && step == 0 {
                continue;
            }
            let t = step as f64 / steps as f64;
            let w0 = (1.0 - t) * (1.0 - t);
            let w1 = 2.0 * (1.0 - t) * t;
            let w2 = t * t;
            sampled.push([
                w0 * a[0] + w1 * b[0] + w2 * c[0],
                w0 * a[1] + w1 * b[1] + w2 * c[1],
            ]);
        }
        span_start += 2;
    }
    sampled
}

fn quadratic_surface_outline(points: &[Point2], resolution: u32) -> Vec<Point2> {
    let steps = (resolution as usize * 2).max(12);
    let mut sampled = sample_quadratic_curve(points, resolution);
    if sampled.is_empty() {
        return sampled;
    }
    let first = sampled[0];
    let last = sampled[sampled.len() - 1];
    let dist = ((first[0] - last[0]).powi(2) + (first[1] - last[1]).powi(2)).sqrt();
    if dist > 1.0e-9 {
        for step in 1..=steps {
            let t = step as f64 / steps as f64;
            sampled.push([
                (1.0 - t) * last[0] + t * first[0],
                (1.0 - t) * last[1] + t * first[1],
            ]);
        }
    }
    drop_duplicate_points(sampled)
}

fn drop_duplicate_points(points: Vec<Point2>) -> Vec<Point2> {
    let mut deduped: Vec<Point2> = Vec::with_capacity(points.len());
    for point in points {
        match deduped.last() {
            None => deduped.push(point),
            Some(last) => {
                let dist = ((point[0] - last[0]).powi(2) + (point[1] - last[1]).powi(2)).sqrt();
                if dist > 1.0e-9 {
                    deduped.push(point);
                }
            }
        }
    }
    if deduped.len() > 1 {
        let first = deduped[0];
        let last = deduped[deduped.len() - 1];
        let dist = ((first[0] - last[0]).powi(2) + (first[1] - last[1]).powi(2)).sqrt();
        if dist <= 1.0e-9 {
            deduped.pop();
        }
    }
    deduped
}

// ---------------------------------------------------------------------------
// 2D polygon helpers

fn signed_area_2d(points: &[Point2]) -> f64 {
    let mut area = 0.0;
    for (index, point) in points.iter().enumerate() {
        let next = points[(index + 1) % points.len()];
        area += point[0] * next[1] - next[0] * point[1];
    }
    area * 0.5
}

fn cross_2d(previous: Point2, point: Point2, next: Point2) -> f64 {
    (point[0] - previous[0]) * (next[1] - point[1])
        - (point[1] - previous[1]) * (next[0] - point[0])
}

fn points_close_2d(first: Point2, second: Point2) -> bool {
    (first[0] - second[0]).abs() <= 1.0e-12 && (first[1] - second[1]).abs() <= 1.0e-12
}

fn point_in_polygon_2d(point: Point2, polygon: &[Point2]) -> bool {
    let mut inside = false;
    let [x, y] = point;
    let mut previous = polygon[polygon.len() - 1];
    for current in polygon {
        if (current[1] > y) != (previous[1] > y) {
            let slope = (previous[0] - current[0]) / (previous[1] - current[1]);
            let crossing_x = current[0] + (y - current[1]) * slope;
            if x < crossing_x {
                inside = !inside;
            }
        }
        previous = *current;
    }
    inside
}

fn point_in_triangle_2d(point: Point2, a: Point2, b: Point2, c: Point2) -> bool {
    cross_2d(a, b, point) >= -1.0e-12
        && cross_2d(b, c, point) >= -1.0e-12
        && cross_2d(c, a, point) >= -1.0e-12
}

fn polygon_is_convex(points: &[Point2]) -> bool {
    if points.len() <= 3 {
        return true;
    }
    for (index, point) in points.iter().enumerate() {
        let previous = points[(index + points.len() - 1) % points.len()];
        let next = points[(index + 1) % points.len()];
        if cross_2d(previous, *point, next) < -1.0e-12 {
            return false;
        }
    }
    true
}

/// Ear clipping with a degenerate-vertex sweep fallback (Python
/// `_triangulate_simple_polygon`).
pub fn triangulate_simple_polygon(points: &[Point2]) -> Vec<[usize; 3]> {
    if points.len() < 3 {
        return Vec::new();
    }
    let reversed_polygon: Vec<Point2>;
    let polygon: &[Point2] = if signed_area_2d(points) < 0.0 {
        reversed_polygon = points.iter().rev().copied().collect();
        &reversed_polygon
    } else {
        points
    };
    if polygon_is_convex(polygon) {
        return (1..polygon.len() - 1).map(|index| [0, index, index + 1]).collect();
    }
    let mut remaining: Vec<usize> = (0..polygon.len()).collect();
    let mut triangles = Vec::new();
    let mut guard = polygon.len() * polygon.len();
    while remaining.len() > 3 && guard > 0 {
        guard -= 1;
        let mut clipped = false;
        for position in 0..remaining.len() {
            let index = remaining[position];
            let previous = remaining[(position + remaining.len() - 1) % remaining.len()];
            let next_index = remaining[(position + 1) % remaining.len()];
            if cross_2d(polygon[previous], polygon[index], polygon[next_index]) <= 1.0e-12 {
                continue;
            }
            let contains = remaining.iter().any(|candidate| {
                *candidate != previous
                    && *candidate != index
                    && *candidate != next_index
                    && point_in_triangle_2d(
                        polygon[*candidate],
                        polygon[previous],
                        polygon[index],
                        polygon[next_index],
                    )
            });
            if contains {
                continue;
            }
            triangles.push([previous, index, next_index]);
            remaining.remove(position);
            clipped = true;
            break;
        }
        if clipped {
            continue;
        }
        for position in 0..remaining.len() {
            let index = remaining[position];
            let previous = remaining[(position + remaining.len() - 1) % remaining.len()];
            let next_index = remaining[(position + 1) % remaining.len()];
            if cross_2d(polygon[previous], polygon[index], polygon[next_index]).abs() <= 1.0e-12 {
                remaining.remove(position);
                clipped = true;
                break;
            }
        }
        if !clipped {
            return Vec::new();
        }
    }
    if remaining.len() == 3 {
        triangles.push([remaining[0], remaining[1], remaining[2]]);
    }
    // Note: when the input was reversed, triangle indices refer to the
    // reversed order; remap back to original indices.
    if signed_area_2d(points) < 0.0 {
        let count = points.len();
        triangles
            .iter()
            .map(|tri| tri.map(|index| count - 1 - index))
            .collect()
    } else {
        triangles
    }
}

// ---------------------------------------------------------------------------
// Marching squares + ring stitching

fn edge_has_crossing(first: f64, second: f64) -> bool {
    ((first <= 0.0 && 0.0 <= second) || (second <= 0.0 && 0.0 <= first))
        && (first - second).abs() > 1.0e-12
}

fn marching_square_pairs(
    mask: u8,
    center_inside: bool,
) -> &'static [(usize, usize)] {
    match mask {
        1 => &[(3, 0)],
        2 => &[(0, 1)],
        3 => &[(3, 1)],
        4 => &[(1, 2)],
        6 => &[(0, 2)],
        7 => &[(3, 2)],
        8 => &[(2, 3)],
        9 => &[(0, 2)],
        11 => &[(1, 2)],
        12 => &[(1, 3)],
        13 => &[(0, 1)],
        14 => &[(3, 0)],
        5 => {
            if center_inside {
                &[(0, 1), (2, 3)]
            } else {
                &[(3, 0), (1, 2)]
            }
        }
        10 => {
            if center_inside {
                &[(0, 3), (1, 2)]
            } else {
                &[(0, 1), (2, 3)]
            }
        }
        _ => &[],
    }
}

pub fn marching_squares_rings(
    values: &[f64],
    us: &[f64],
    vs: &[f64],
) -> Vec<Vec<Point2>> {
    let resolution = us.len() - 1;
    let value_at = |i: usize, j: usize| values[i * (resolution + 1) + j];
    let mut horizontal = vec![-1i64; resolution * (resolution + 1)];
    let mut vertical = vec![-1i64; (resolution + 1) * resolution];
    let mut vertices: Vec<Point2> = Vec::new();

    for i in 0..resolution {
        for j in 0..=resolution {
            let first = value_at(i, j);
            let second = value_at(i + 1, j);
            if !edge_has_crossing(first, second) {
                continue;
            }
            let t = (first / (first - second)).clamp(0.0, 1.0);
            horizontal[i * (resolution + 1) + j] = vertices.len() as i64;
            vertices.push([us[i] + t * (us[i + 1] - us[i]), vs[j]]);
        }
    }
    for i in 0..=resolution {
        for j in 0..resolution {
            let first = value_at(i, j);
            let second = value_at(i, j + 1);
            if !edge_has_crossing(first, second) {
                continue;
            }
            let t = (first / (first - second)).clamp(0.0, 1.0);
            vertical[i * resolution + j] = vertices.len() as i64;
            vertices.push([us[i], vs[j] + t * (vs[j + 1] - vs[j])]);
        }
    }

    let mut segments: Vec<(usize, usize)> = Vec::new();
    for i in 0..resolution {
        for j in 0..resolution {
            let mut mask = 0u8;
            if value_at(i, j) <= 0.0 {
                mask |= 1;
            }
            if value_at(i + 1, j) <= 0.0 {
                mask |= 2;
            }
            if value_at(i + 1, j + 1) <= 0.0 {
                mask |= 4;
            }
            if value_at(i, j + 1) <= 0.0 {
                mask |= 8;
            }
            if mask == 0 || mask == 15 {
                continue;
            }
            let center_inside = (value_at(i, j)
                + value_at(i + 1, j)
                + value_at(i, j + 1)
                + value_at(i + 1, j + 1))
                / 4.0
                <= 0.0;
            let edge_vertices = [
                horizontal[i * (resolution + 1) + j],
                vertical[(i + 1) * resolution + j],
                horizontal[i * (resolution + 1) + j + 1],
                vertical[i * resolution + j],
            ];
            for (first, second) in marching_square_pairs(mask, center_inside) {
                let first_vertex = edge_vertices[*first];
                let second_vertex = edge_vertices[*second];
                if first_vertex >= 0 && second_vertex >= 0 {
                    segments.push((first_vertex as usize, second_vertex as usize));
                }
            }
        }
    }
    stitch_contour_rings(&vertices, &segments)
}

fn stitch_contour_rings(
    vertices: &[Point2],
    segments: &[(usize, usize)],
) -> Vec<Vec<Point2>> {
    use std::collections::{BTreeMap, BTreeSet};
    let mut adjacency: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    let mut unused: BTreeSet<(usize, usize)> = BTreeSet::new();
    for (first, second) in segments {
        if first == second {
            continue;
        }
        let edge = (*first.min(second), *first.max(second));
        if unused.contains(&edge) {
            continue;
        }
        unused.insert(edge);
        adjacency.entry(*first).or_default().push(*second);
        adjacency.entry(*second).or_default().push(*first);
    }

    let mut rings: Vec<Vec<Point2>> = Vec::new();
    while let Some(edge) = unused.iter().next().copied() {
        let (first, second) = edge;
        unused.remove(&edge);
        let mut ring = vec![first, second];
        let mut previous = first;
        let mut current = second;
        let mut closed = false;
        for _ in 0..=segments.len() {
            let neighbors = adjacency.get(&current).cloned().unwrap_or_default();
            let candidates: Vec<usize> = neighbors
                .iter()
                .filter(|candidate| {
                    unused.contains(&(current.min(**candidate), current.max(**candidate)))
                })
                .copied()
                .collect();
            if candidates.is_empty() {
                closed = neighbors.contains(&ring[0]);
                break;
            }
            let next_vertex = if candidates.len() > 1 {
                candidates
                    .iter()
                    .find(|candidate| **candidate != previous)
                    .copied()
                    .unwrap_or(candidates[0])
            } else {
                candidates[0]
            };
            unused.remove(&(current.min(next_vertex), current.max(next_vertex)));
            if next_vertex == ring[0] {
                closed = true;
                break;
            }
            ring.push(next_vertex);
            previous = current;
            current = next_vertex;
        }
        if closed && ring.len() >= 3 {
            rings.push(ring.iter().map(|index| vertices[*index]).collect());
        }
    }
    rings.sort_by(|a, b| {
        signed_area_2d(b)
            .abs()
            .partial_cmp(&signed_area_2d(a).abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    rings
}

fn clean_polygon_ring(ring: &[Point2]) -> Vec<Point2> {
    if ring.len() < 3 {
        return ring.to_vec();
    }
    let mut cleaned: Vec<Point2> = Vec::with_capacity(ring.len());
    for point in ring {
        if let Some(last) = cleaned.last() {
            if points_close_2d(*last, *point) {
                continue;
            }
        }
        cleaned.push(*point);
    }
    if cleaned.len() > 1 && points_close_2d(cleaned[0], cleaned[cleaned.len() - 1]) {
        cleaned.pop();
    }
    let mut changed = true;
    while changed && cleaned.len() >= 3 {
        changed = false;
        let count = cleaned.len();
        let mut reduced = Vec::with_capacity(count);
        for (index, point) in cleaned.iter().enumerate() {
            let previous = cleaned[(index + count - 1) % count];
            let next = cleaned[(index + 1) % count];
            if cross_2d(previous, *point, next).abs() <= 1.0e-12 {
                changed = true;
                continue;
            }
            reduced.push(*point);
        }
        cleaned = reduced;
    }
    cleaned
}

fn contour_rings_have_holes(rings: &[Vec<Point2>]) -> bool {
    for (index, ring) in rings.iter().enumerate() {
        let probe = ring[0];
        let mut depth = 0;
        for (other_index, other) in rings.iter().enumerate() {
            if index == other_index {
                continue;
            }
            if point_in_polygon_2d(probe, other) {
                depth += 1;
            }
        }
        if depth % 2 == 1 {
            return true;
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Placed 2D / 1D builders

fn wire_surface_from_points(
    node: &Node,
    key: ViewportSurfaceKey,
    color: [f32; 3],
    vertices: Vec<Vec3>,
    segments: Vec<(u32, u32)>,
) -> ViewportSurface {
    let vertex_array: Vec<[f32; 3]> = vertices
        .iter()
        .map(|v| [v.x as f32, v.y as f32, v.z as f32])
        .collect();
    let normals = vec![[0.0, 0.0, 1.0]; vertex_array.len()];
    let wire: Vec<u32> = segments.iter().flat_map(|(a, b)| [*a, *b]).collect();
    let (bounds_min, bounds_max) = f32_bounds(&vertex_array);
    ViewportSurface {
        key,
        object_kind: node.kind().to_string(),
        status: SurfaceStatus::Outline,
        vertices: vertex_array,
        normals,
        indices: Vec::new(),
        wire_indices: wire,
        color,
        alpha: 1.0,
        bounds_min,
        bounds_max,
        message: "1D object rendered as line geometry".to_string(),
    }
}

pub(crate) fn f32_bounds(vertices: &[[f32; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut bounds_min = [f64::INFINITY; 3];
    let mut bounds_max = [f64::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            bounds_min[axis] = bounds_min[axis].min(vertex[axis] as f64);
            bounds_max[axis] = bounds_max[axis].max(vertex[axis] as f64);
        }
    }
    if vertices.is_empty() {
        return ([0.0; 3], [0.0; 3]);
    }
    (bounds_min, bounds_max)
}

pub fn placed_1d_line(
    node: &Node,
    placed: &PlacedSdf1D,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> ViewportSurface {
    let (t_min, t_max) = placed.profile.bounds();
    let resolution = ((key.resolution as usize) * 8).clamp(24, 256);
    let samples: Vec<f64> = (0..=resolution)
        .map(|index| t_min + (t_max - t_min) * index as f64 / resolution as f64)
        .collect();
    let inside: Vec<bool> = samples
        .windows(2)
        .map(|pair| placed.profile.eval((pair[0] + pair[1]) * 0.5) <= 0.0)
        .collect();
    let mut spans: Vec<(f64, f64)> = Vec::new();
    let mut start: Option<f64> = None;
    for (index, is_inside) in inside.iter().enumerate() {
        if *is_inside && start.is_none() {
            start = Some(samples[index]);
        }
        if let Some(begin) = start {
            if !*is_inside || index == inside.len() - 1 {
                let end = if !*is_inside {
                    samples[index]
                } else {
                    samples[index + 1]
                };
                if end > begin {
                    spans.push((begin, end));
                }
                start = None;
            }
        }
    }
    if spans.is_empty() {
        return empty_surface(node, key, color, "1D profile produced no line spans");
    }
    let mut vertices = Vec::with_capacity(spans.len() * 2);
    let mut segments = Vec::with_capacity(spans.len());
    for (start_t, end_t) in spans {
        let base = vertices.len() as u32;
        vertices.push(placed.origin + placed.axis_u * start_t);
        vertices.push(placed.origin + placed.axis_u * end_t);
        segments.push((base, base + 1));
    }
    wire_surface_from_points(node, key, color, vertices, segments)
}

pub fn placed_polyline_1d(
    node: &Node,
    placed: &PlacedPolyline1D,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> ViewportSurface {
    let local_points: Vec<Point2> = match &placed.profile {
        Profile2D::Polyline { points } => points.clone(),
        Profile2D::QuadraticBezierCurve { points } => sample_quadratic_curve(points, key.resolution),
        _ => return empty_surface(node, key, color, "unsupported 1D curve profile"),
    };
    if local_points.len() < 2 {
        return empty_surface(node, key, color, "1D curve has too few points");
    }
    let vertices: Vec<Vec3> = local_points
        .iter()
        .map(|point| placed.origin + placed.axis_u * point[0] + placed.axis_v * point[1])
        .collect();
    let segments: Vec<(u32, u32)> = (0..vertices.len() as u32 - 1)
        .map(|index| (index, index + 1))
        .collect();
    wire_surface_from_points(node, key, color, vertices, segments)
}

pub fn placed_2d_outline(
    node: &Node,
    placed: &PlacedSdf2D,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> ViewportSurface {
    if let Some(surface) = ordered_placed_2d_surface(node, placed, key, color) {
        return surface;
    }
    if let Some(surface) = contoured_placed_2d_surface(node, placed, key, color) {
        return surface;
    }
    sampled_placed_2d_surface(node, placed, key, color)
}

fn ordered_placed_2d_surface(
    node: &Node,
    placed: &PlacedSdf2D,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> Option<ViewportSurface> {
    let outline = profile_outline(&placed.profile, key.resolution);
    if outline.len() < 2 {
        return None;
    }
    let normal = placed.normal();
    let world: Vec<Vec3> = outline
        .iter()
        .map(|point| placed.origin + placed.axis_u * point[0] + placed.axis_v * point[1])
        .collect();
    let mut center = Vec3::ZERO;
    for point in &world {
        center += *point;
    }
    center = center / world.len() as f64;
    let mut vertices: Vec<[f32; 3]> = world
        .iter()
        .map(|v| [v.x as f32, v.y as f32, v.z as f32])
        .collect();
    vertices.push([center.x as f32, center.y as f32, center.z as f32]);
    let normal_f32 = [normal.x as f32, normal.y as f32, normal.z as f32];
    let normals = vec![normal_f32; vertices.len()];
    let count = outline.len() as u32;
    let wire: Vec<u32> = (0..count)
        .flat_map(|current| [current, (current + 1) % count])
        .collect();
    let center_index = count;
    let indices: Vec<u32> = (0..count)
        .flat_map(|current| [center_index, current, (current + 1) % count])
        .collect();
    let (bounds_min, bounds_max) = f32_bounds(&vertices);
    Some(ViewportSurface {
        key,
        object_kind: node.kind().to_string(),
        status: SurfaceStatus::Ready,
        vertices,
        normals,
        indices,
        wire_indices: wire,
        color,
        alpha: 1.0,
        bounds_min,
        bounds_max,
        message: "2D profile rendered as filled ordered surface".to_string(),
    })
}

fn contoured_placed_2d_surface(
    node: &Node,
    placed: &PlacedSdf2D,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> Option<ViewportSurface> {
    let (mut u_min, mut u_max, mut v_min, mut v_max) = placed.profile.bounds();
    let span = (u_max - u_min).max(v_max - v_min).max(1.0e-6);
    let pad = span * 0.025;
    u_min -= pad;
    u_max += pad;
    v_min -= pad;
    v_max += pad;
    let resolution = (key.resolution * 5).clamp(64, MAX_CONTOURED_2D_CELLS) as usize;
    let us: Vec<f64> = (0..=resolution)
        .map(|index| u_min + (u_max - u_min) * index as f64 / resolution as f64)
        .collect();
    let vs: Vec<f64> = (0..=resolution)
        .map(|index| v_min + (v_max - v_min) * index as f64 / resolution as f64)
        .collect();
    let mut values = Vec::with_capacity((resolution + 1) * (resolution + 1));
    for u in &us {
        for v in &vs {
            values.push(placed.profile.eval(*u, *v));
        }
    }
    let local_rings = marching_squares_rings(&values, &us, &vs);
    if local_rings.is_empty() {
        return None;
    }
    let mut cleaned_rings: Vec<Vec<Point2>> = Vec::new();
    for ring in &local_rings {
        let mut cleaned = clean_polygon_ring(ring);
        if cleaned.len() < 3 {
            continue;
        }
        if signed_area_2d(&cleaned) < 0.0 {
            cleaned.reverse();
        }
        cleaned_rings.push(cleaned);
    }
    if cleaned_rings.is_empty() || contour_rings_have_holes(&cleaned_rings) {
        return None;
    }

    let mut local_vertices: Vec<Point2> = Vec::new();
    let mut indices: Vec<u32> = Vec::new();
    let mut wire: Vec<u32> = Vec::new();
    for cleaned in &cleaned_rings {
        let triangles = triangulate_simple_polygon(cleaned);
        if triangles.is_empty() {
            continue;
        }
        let base = local_vertices.len() as u32;
        local_vertices.extend_from_slice(cleaned);
        for triangle in triangles {
            for index in triangle {
                indices.push(base + index as u32);
            }
        }
        let count = cleaned.len() as u32;
        for index in 0..count {
            wire.push(base + index);
            wire.push(base + (index + 1) % count);
        }
    }
    if local_vertices.is_empty() || indices.is_empty() {
        return None;
    }
    let normal = placed.normal();
    let vertices: Vec<[f32; 3]> = local_vertices
        .iter()
        .map(|point| {
            let world = placed.origin + placed.axis_u * point[0] + placed.axis_v * point[1];
            [world.x as f32, world.y as f32, world.z as f32]
        })
        .collect();
    let normal_f32 = [normal.x as f32, normal.y as f32, normal.z as f32];
    let normals = vec![normal_f32; vertices.len()];
    let (bounds_min, bounds_max) = f32_bounds(&vertices);
    Some(ViewportSurface {
        key,
        object_kind: node.kind().to_string(),
        status: SurfaceStatus::Ready,
        vertices,
        normals,
        indices,
        wire_indices: wire,
        color,
        alpha: 1.0,
        bounds_min,
        bounds_max,
        message: "2D profile rendered as contoured filled surface".to_string(),
    })
}

fn sampled_placed_2d_surface(
    node: &Node,
    placed: &PlacedSdf2D,
    key: ViewportSurfaceKey,
    color: [f32; 3],
) -> ViewportSurface {
    let (mut u_min, mut u_max, mut v_min, mut v_max) = placed.profile.bounds();
    let span = (u_max - u_min).max(v_max - v_min).max(1.0e-6);
    let pad = span * 0.025;
    u_min -= pad;
    u_max += pad;
    v_min -= pad;
    v_max += pad;
    let resolution = (key.resolution * 3).clamp(24, MAX_SAMPLED_2D_CELLS) as usize;
    let us: Vec<f64> = (0..=resolution)
        .map(|index| u_min + (u_max - u_min) * index as f64 / resolution as f64)
        .collect();
    let vs: Vec<f64> = (0..=resolution)
        .map(|index| v_min + (v_max - v_min) * index as f64 / resolution as f64)
        .collect();
    let mut inside = vec![false; resolution * resolution];
    let mut any_inside = false;
    for i in 0..resolution {
        for j in 0..resolution {
            let mid_u = (us[i] + us[i + 1]) * 0.5;
            let mid_v = (vs[j] + vs[j + 1]) * 0.5;
            if placed.profile.eval(mid_u, mid_v) <= 0.0 {
                inside[i * resolution + j] = true;
                any_inside = true;
            }
        }
    }
    if !any_inside {
        return empty_surface(node, key, color, "2D profile produced no filled cells");
    }
    let vertex_count_per_axis = resolution + 1;
    // Grid vertices in world space (only used ones are kept below).
    let mut used_map: std::collections::BTreeMap<usize, u32> = std::collections::BTreeMap::new();
    let mut vertices: Vec<[f32; 3]> = Vec::new();
    let mut resolve = |grid_index: usize,
                       vertices: &mut Vec<[f32; 3]>|
     -> u32 {
        *used_map.entry(grid_index).or_insert_with(|| {
            let i = grid_index / vertex_count_per_axis;
            let j = grid_index % vertex_count_per_axis;
            let world = placed.origin + placed.axis_u * us[i] + placed.axis_v * vs[j];
            vertices.push([world.x as f32, world.y as f32, world.z as f32]);
            (vertices.len() - 1) as u32
        })
    };
    let mut indices: Vec<u32> = Vec::new();
    for i in 0..resolution {
        for j in 0..resolution {
            if !inside[i * resolution + j] {
                continue;
            }
            let a = resolve(i * vertex_count_per_axis + j, &mut vertices);
            let b = resolve((i + 1) * vertex_count_per_axis + j, &mut vertices);
            let c = resolve((i + 1) * vertex_count_per_axis + j + 1, &mut vertices);
            let d = resolve(i * vertex_count_per_axis + j + 1, &mut vertices);
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    // Boundary wire: cell edges facing an outside cell or the grid border.
    let mut wire: Vec<u32> = Vec::new();
    for i in 0..resolution {
        for j in 0..resolution {
            if !inside[i * resolution + j] {
                continue;
            }
            let left = i == 0 || !inside[(i - 1) * resolution + j];
            let right = i == resolution - 1 || !inside[(i + 1) * resolution + j];
            let bottom = j == 0 || !inside[i * resolution + j - 1];
            let top = j == resolution - 1 || !inside[i * resolution + j + 1];
            let corner =
                |ii: usize, jj: usize| used_map.get(&(ii * vertex_count_per_axis + jj)).copied();
            let mut append = |a: Option<u32>, b: Option<u32>| {
                if let (Some(a), Some(b)) = (a, b) {
                    wire.push(a);
                    wire.push(b);
                }
            };
            if bottom {
                append(corner(i, j), corner(i + 1, j));
            }
            if right {
                append(corner(i + 1, j), corner(i + 1, j + 1));
            }
            if top {
                append(corner(i + 1, j + 1), corner(i, j + 1));
            }
            if left {
                append(corner(i, j + 1), corner(i, j));
            }
        }
    }
    let normal = placed.normal();
    let normal_f32 = [normal.x as f32, normal.y as f32, normal.z as f32];
    let normals = vec![normal_f32; vertices.len()];
    let (bounds_min, bounds_max) = f32_bounds(&vertices);
    ViewportSurface {
        key,
        object_kind: node.kind().to_string(),
        status: SurfaceStatus::Ready,
        vertices,
        normals,
        indices,
        wire_indices: wire,
        color,
        alpha: 1.0,
        bounds_min,
        bounds_max,
        message: "2D profile rendered as sampled filled surface".to_string(),
    }
}
