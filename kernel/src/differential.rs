//! Batch differential queries on exact SDF nodes: normals, interior-seeded
//! Newton projection onto the boundary, curvature stencils.
//!
//! Interior-exactness contract (`design_docs/meshing_toolkit.md` §2): the
//! grammar guarantees a true distance only on the negative side of a field,
//! so no positive value is ever consumed as a distance. Projection refuses
//! positive starts outright.

use crate::boundary_ops::sdf_normal;
use crate::sdf::node::Node;
use crate::vec3::Vec3;

/// Normal-step factor relative to the field's bbox diagonal (first
/// derivative, f64 central-difference optimum near eps^(1/3)).
pub const NORMAL_STEP_RELATIVE: f64 = 1.0e-6;
/// Curvature-step factor relative to the bbox diagonal (second derivative,
/// optimum near eps^(1/4)).
pub const CURVATURE_STEP_RELATIVE: f64 = 1.0e-4;
/// On-the-wall acceptance band relative to the bbox diagonal.
pub const ZERO_BAND_RELATIVE: f64 = 1.0e-9;

const PROJECTION_ITERATIONS: usize = 24;
const BACKTRACK_HALVINGS: usize = 8;

/// (normal step, curvature step, zero band) scaled by the field's bbox
/// diagonal, clamped away from zero for degenerate fields.
pub fn differential_steps(field: &Node) -> (f64, f64, f64) {
    let diagonal = field
        .bounding_box()
        .map(|bounds| bounds.diagonal())
        .unwrap_or(1.0)
        .max(1.0e-9);
    (
        diagonal * NORMAL_STEP_RELATIVE,
        diagonal * CURVATURE_STEP_RELATIVE,
        diagonal * ZERO_BAND_RELATIVE,
    )
}

/// Outward gradient directions at many points (central differences).
pub fn batch_normals(field: &Node, points: &[Vec3], step: f64) -> Vec<Vec3> {
    points
        .iter()
        .map(|point| sdf_normal(field, *point, step))
        .collect()
}

/// The result of one Newton projection onto a field's zero set.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Projection {
    /// The best point reached (on the wall iff `converged`).
    pub point: Vec3,
    /// Field value at `point`.
    pub residual: f64,
    /// Total distance travelled from the start.
    pub distance_moved: f64,
    /// True iff `|residual| <= zero_band`.
    pub converged: bool,
}

/// Newton projection of an **interior** start point onto the field's zero
/// set. A start on the positive side is refused immediately
/// (`converged: false`, no iteration): outside magnitudes are lower bounds,
/// never distances. On the interior the full step is the exact correction;
/// backtracking only fires at C0 creases (equidistant seams), where honest
/// non-convergence beats returning a bad point.
pub fn project_to_zero_set(field: &Node, start: Vec3, normal_step: f64, zero_band: f64) -> Projection {
    let value = field.eval_point(start);
    if value > zero_band {
        // Positive start: refused (interior-exactness contract).
        return Projection {
            point: start,
            residual: value,
            distance_moved: 0.0,
            converged: false,
        };
    }
    project_with_gradient(field, start, zero_band, |point| {
        sdf_normal(field, point, normal_step)
    })
}

/// Projection onto a **leaf-primitive** field, which is exact on both sides
/// — the one case where a positive start is a true distance and may be
/// projected (used for `owner_sdf` layer seeding). Never call this with an
/// operator tree.
pub fn project_leaf_to_zero_set(
    field: &Node,
    start: Vec3,
    normal_step: f64,
    zero_band: f64,
) -> Projection {
    project_with_gradient(field, start, zero_band, |point| {
        sdf_normal(field, point, normal_step)
    })
}

/// In-plane variant for 2D placed domains: the start is dropped onto the
/// plane and the gradient re-projected onto it every iteration, so the
/// result cannot drift off-plane.
pub fn project_to_zero_set_in_plane(
    field: &Node,
    start: Vec3,
    origin: Vec3,
    normal: Vec3,
    normal_step: f64,
    zero_band: f64,
) -> Projection {
    let unit = normal * (1.0 / normal.length().max(1.0e-12));
    let offset = (start - origin).dot(unit);
    let planar_start = start - unit * offset;
    let value = field.eval_point(planar_start);
    if value > zero_band {
        // Positive start: refused (interior-exactness contract).
        return Projection {
            point: planar_start,
            residual: value,
            distance_moved: 0.0,
            converged: false,
        };
    }
    project_with_gradient(field, planar_start, zero_band, |point| {
        let gradient = sdf_normal(field, point, normal_step);
        let planar = gradient - unit * gradient.dot(unit);
        planar * (1.0 / planar.length().max(1.0e-12))
    })
}

fn project_with_gradient(
    field: &Node,
    start: Vec3,
    zero_band: f64,
    gradient_at: impl Fn(Vec3) -> Vec3,
) -> Projection {
    let mut point = start;
    let mut value = field.eval_point(point);
    for _ in 0..PROJECTION_ITERATIONS {
        if value.abs() <= zero_band {
            return Projection {
                point,
                residual: value,
                distance_moved: (point - start).length(),
                converged: true,
            };
        }
        let direction = gradient_at(point);
        let mut scale = 1.0;
        let mut improved = false;
        for _ in 0..=BACKTRACK_HALVINGS {
            let candidate = point - direction * (value * scale);
            let candidate_value = field.eval_point(candidate);
            if candidate_value.abs() < value.abs() {
                point = candidate;
                value = candidate_value;
                improved = true;
                break;
            }
            scale *= 0.5;
        }
        if !improved {
            break;
        }
    }
    Projection {
        point,
        residual: value,
        distance_moved: (point - start).length(),
        converged: value.abs() <= zero_band,
    }
}

/// Mean curvature of a distance field at a near-wall point reached from the
/// interior: `H = laplacian(f) / 2` (7-point stencil). At creases the value
/// is O(1/step) — read it as "refine here", not as a curvature.
pub fn mean_curvature(field: &Node, point: Vec3, step: f64) -> f64 {
    laplacian(field, point, step, &[Vec3::X, Vec3::Y, Vec3::Z]) / 2.0
}

/// In-plane curvature of a 2D placed field: `kappa = laplacian(f)` along the
/// two plane axes (5-point stencil).
pub fn curvature_2d(field: &Node, point: Vec3, axis_a: Vec3, axis_b: Vec3, step: f64) -> f64 {
    laplacian(field, point, step, &[axis_a, axis_b])
}

fn laplacian(field: &Node, point: Vec3, step: f64, axes: &[Vec3]) -> f64 {
    let center = field.eval_point(point);
    let mut sum = 0.0;
    for axis in axes {
        let offset = *axis * step;
        sum += field.eval_point(point + offset) + field.eval_point(point - offset) - 2.0 * center;
    }
    sum / (step * step)
}
