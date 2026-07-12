//! Side-of-curtain classification field for on-surface paths.
//!
//! Stores a dense path lying ON a Domain boundary plus the unit surface
//! normals there. Its classification field is the signed distance to the
//! ruled "curtain" swept from the path along the normals; the sign is which
//! side of the curtain a query point falls on. Used as a scene payload;
//! its former use as the smooth-polyline boundary knife was removed
//! 2026-07-12 (design_docs/boundary_cutter_exactness.md §6).
//! Ported from `core/sdf/curtain.py`.

use crate::bbox::BoundingBox3D;
use crate::error::{GeometryError, GeometryResult};
use crate::vec3::{vec3, Vec3};

#[derive(Debug, Clone, PartialEq)]
pub struct NormalCurtain {
    pub points: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub extent: f64,
    /// Per-segment unit binormals (cross(tangent, mean normal)), precomputed
    /// and validated at construction.
    binormals: Vec<Vec3>,
}

fn drop_duplicate_path_points(points: &[Vec3], normals: &[Vec3]) -> (Vec<Vec3>, Vec<Vec3>) {
    if points.is_empty() {
        return (Vec::new(), Vec::new());
    }
    let mut kept_points = vec![points[0]];
    let mut kept_normals = vec![normals[0]];
    for (point, normal) in points.iter().zip(normals.iter()).skip(1) {
        if (*point - *kept_points.last().expect("nonempty")).length() > 1.0e-12 {
            kept_points.push(*point);
            kept_normals.push(*normal);
        }
    }
    (kept_points, kept_normals)
}

fn unit_vectors(vectors: &[Vec3], message: &str) -> GeometryResult<Vec<Vec3>> {
    vectors
        .iter()
        .map(|vector| {
            let length = vector.length();
            if length <= 1.0e-12 {
                return Err(GeometryError::new(message));
            }
            Ok(*vector / length)
        })
        .collect()
}

fn segment_binormals(points: &[Vec3], normals: &[Vec3]) -> GeometryResult<Vec<Vec3>> {
    let count = points.len() - 1;
    let mut binormals = Vec::with_capacity(count);
    let mut valid = Vec::with_capacity(count);
    for index in 0..count {
        let tangent = points[index + 1] - points[index];
        let mean_normal = (normals[index] + normals[index + 1]) * 0.5;
        let binormal = tangent.cross(mean_normal);
        let length = binormal.length();
        if length > 1.0e-12 {
            binormals.push(binormal / length);
            valid.push(true);
        } else {
            binormals.push(binormal);
            valid.push(false);
        }
    }
    if !valid.iter().any(|flag| *flag) {
        return Err(GeometryError::new(
            "curtain path is parallel to its surface normals",
        ));
    }
    if valid.iter().all(|flag| *flag) {
        return Ok(binormals);
    }
    // A path leg momentarily parallel to its normal has no side of its own;
    // borrow the nearest well-defined segment's.
    let good: Vec<usize> = (0..count).filter(|index| valid[*index]).collect();
    for index in 0..count {
        if !valid[index] {
            let nearest = good
                .iter()
                .min_by_key(|candidate| index.abs_diff(**candidate))
                .expect("at least one valid binormal");
            binormals[index] = binormals[*nearest];
        }
    }
    Ok(binormals)
}

impl NormalCurtain {
    pub fn new(points: Vec<Vec3>, normals: Vec<Vec3>, extent: f64) -> GeometryResult<Self> {
        if points.len() != normals.len() {
            return Err(GeometryError::new("curtain needs one normal per path point"));
        }
        let (points, normals) = drop_duplicate_path_points(&points, &normals);
        if points.len() < 2 {
            return Err(GeometryError::new(
                "curtain requires at least two distinct path points",
            ));
        }
        if !extent.is_finite() || extent <= 0.0 {
            return Err(GeometryError::new("curtain extent must be finite and positive"));
        }
        let normals = unit_vectors(&normals, "curtain normals must be nonzero")?;
        let binormals = segment_binormals(&points, &normals)?;
        Ok(Self {
            points,
            normals,
            extent,
            binormals,
        })
    }

    pub fn eval_point(&self, p: Vec3) -> f64 {
        let mut best_distance_sq = f64::INFINITY;
        let mut best_rejection = Vec3::ZERO;
        let mut best_index = 0usize;
        for index in 0..self.points.len() - 1 {
            let tangent = self.points[index + 1] - self.points[index];
            let squared_length = tangent.length_squared();
            let offset = p - self.points[index];
            let along = (offset.dot(tangent) / squared_length).clamp(0.0, 1.0);
            let rejection = offset - tangent * along;
            let distance_sq = rejection.length_squared();
            if distance_sq < best_distance_sq {
                best_distance_sq = distance_sq;
                best_rejection = rejection;
                best_index = index;
            }
        }
        let side = best_rejection.dot(self.binormals[best_index]);
        let sign = if side < 0.0 { -1.0 } else { 1.0 };
        sign * best_distance_sq.sqrt()
    }

    pub fn bounding_box(&self) -> GeometryResult<BoundingBox3D> {
        let padding = vec3(self.extent, self.extent, self.extent);
        BoundingBox3D::from_points(
            self.points
                .iter()
                .flat_map(|point| [*point - padding, *point + padding]),
        )
    }
}
