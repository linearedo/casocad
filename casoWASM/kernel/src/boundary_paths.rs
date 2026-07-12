//! Knife-ghost construction for the boundary cutter: the planar segment
//! knife and the point-collected planar stencils (polygon, quadratic bezier
//! surface). Everything is scale-relative to the Domain bounding diagonal,
//! never absolute meters.
//!
//! A smooth on-surface polyline knife (geodesic paths + NormalCurtain
//! classification) existed until 2026-07-12 and was removed as unproven; its
//! design record lives in design_docs/boundary_cutter_exactness.md for a
//! possible future reintroduction.

use crate::error::{GeometryError, GeometryResult};
use crate::sdf::node::{Node, Shape};
use crate::sdf::placed::PlacedSdf2D;
use crate::sdf::primitives_2d::Profile2D;
use crate::sdf::solid_from_2d::Extrude;
use crate::vec3::{vec3, Vec3};

/// Surface-normal alignment below which a planar knife (segment or point
/// stencil) on a visibly curved boundary warrants a curvature warning.
pub const KNIFE_CURVATURE_WARNING_ALIGNMENT: f64 = 0.95;
/// Mean-normal length below which an area knife has no stable orientation:
/// the clicked points wrap around the body (equator-like), so no meaningful
/// stencil plane exists.
pub const MEAN_NORMAL_MINIMUM: f64 = 0.1;

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

/// Orthonormal tangent pair with `u × v = n̂` (so the placed section's
/// workplane normal is exactly `n̂`).
fn orthonormal_tangents(normal: Vec3) -> (Vec3, Vec3) {
    let seed = if normal.x.abs() <= normal.y.abs() && normal.x.abs() <= normal.z.abs() {
        vec3(1.0, 0.0, 0.0)
    } else if normal.y.abs() <= normal.z.abs() {
        vec3(0.0, 1.0, 0.0)
    } else {
        vec3(0.0, 0.0, 1.0)
    };
    let mut axis_u = seed.cross(normal);
    axis_u = axis_u * (1.0 / axis_u.length());
    let axis_v = normal.cross(axis_u);
    (axis_u, axis_v)
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

/// Point-collected area knives (`polygon`, `quadratic_bezier_surface`):
/// a planar stencil on the plane fitted to the clicks (origin = centroid,
/// normal = mean click surface normal — order-independent), extruded
/// **one-sidedly** from just below the lowest click so the volume covers the
/// clicked sheet only — never the antipodal sheet of a closed surface
/// (design_docs/boundary_cutter_exactness.md).
///
/// Returns the ghost node and whether the clicked boundary is visibly curved
/// (normals disagree beyond `KNIFE_CURVATURE_WARNING_ALIGNMENT`).
pub fn stencil_knife(
    root: &Node,
    kind: &str,
    clicked_points: &[Vec3],
) -> GeometryResult<(Node, bool)> {
    if clicked_points.len() < 3 {
        return Err(GeometryError::new(format!(
            "{kind} requires at least three points"
        )));
    }
    if kind == "quadratic_bezier_surface" && clicked_points.len().is_multiple_of(2) {
        return Err(GeometryError::new(format!(
            "{kind} requires an odd point count: anchor, control, anchor"
        )));
    }
    let diagonal = bounding_diagonal(root);
    let step = (diagonal * 1.0e-5).max(1.0e-9);
    let mut unit_normals = Vec::with_capacity(clicked_points.len());
    for point in clicked_points {
        let grad = gradient(root, *point, step);
        let length = grad.length();
        if length <= 1.0e-12 {
            return Err(GeometryError::new(
                "could not resolve a surface normal for the cutter",
            ));
        }
        unit_normals.push(grad * (1.0 / length));
    }
    let count = clicked_points.len() as f64;
    let mean = unit_normals.iter().fold(Vec3::ZERO, |sum, n| sum + *n) * (1.0 / count);
    if mean.length() < MEAN_NORMAL_MINIMUM {
        return Err(GeometryError::new(
            "clicked points have opposing surface normals — no meaningful stencil \
             plane exists",
        ));
    }
    let normal = mean * (1.0 / mean.length());
    let curved = unit_normals
        .iter()
        .any(|n| n.dot(normal) < KNIFE_CURVATURE_WARNING_ALIGNMENT);
    let centroid = clicked_points.iter().fold(Vec3::ZERO, |sum, p| sum + *p) * (1.0 / count);
    let (axis_u, axis_v) = orthonormal_tangents(normal);
    let local: Vec<[f64; 2]> = clicked_points
        .iter()
        .map(|point| {
            let offset = *point - centroid;
            [offset.dot(axis_u), offset.dot(axis_v)]
        })
        .collect();
    let profile = match kind {
        "polygon" => Profile2D::polygon(local)?,
        "quadratic_bezier_surface" => Profile2D::quadratic_bezier_surface(local)?,
        other => {
            return Err(GeometryError::new(format!(
                "unknown stencil knife kind: {other}"
            )))
        }
    };
    // One-sided extrusion from just below the lowest click: the margin
    // absorbs the surface dipping below that click inside the footprint on
    // quasi-convex sheets (on a convex cap the interior bulges upward).
    let (u_min, u_max, v_min, v_max) = profile.bounds();
    let footprint = ((u_max - u_min).powi(2) + (v_max - v_min).powi(2)).sqrt();
    let margin = 0.05 * footprint + 1.0e-3 * diagonal;
    let lowest = clicked_points
        .iter()
        .map(|p| (*p - centroid).dot(normal))
        .fold(f64::INFINITY, f64::min);
    let section = Node::new(
        "stencil_knife_section",
        Shape::PlacedSdf2D(PlacedSdf2D::new(
            profile,
            centroid,
            axis_u,
            axis_v,
            Vec::new(),
        )?),
    );
    let height = 4.0 * diagonal.max(1.0e-6);
    let node = Node::new(
        "stencil_knife",
        Shape::Extrude(Extrude::new(
            section,
            height,
            lowest - margin + 0.5 * height,
        )?),
    );
    Ok((node, curved))
}
