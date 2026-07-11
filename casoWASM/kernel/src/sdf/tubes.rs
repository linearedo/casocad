//! Polyline and quadratic-Bezier tube primitives, ported from
//! `core/sdf/tubes.py`. All evaluation is pointwise: the numpy masks in the
//! original are just vectorized branching over the same scalar math.

use crate::bbox::BoundingBox3D;
use crate::error::{GeometryError, GeometryResult};
use crate::vec3::{vec3, Vec3};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CapStyle {
    Round,
    Flat,
}

impl CapStyle {
    pub fn parse(caps: &str) -> GeometryResult<Self> {
        match caps {
            "round" => Ok(Self::Round),
            "flat" => Ok(Self::Flat),
            _ => Err(GeometryError::new("tube caps must be 'round' or 'flat'")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Round => "round",
            Self::Flat => "flat",
        }
    }
}

fn validate_tube_radius(radius: f64, inner_radius: f64) -> GeometryResult<()> {
    if radius <= 0.0 || !radius.is_finite() {
        return Err(GeometryError::new("tube radius must be finite and positive"));
    }
    if inner_radius < 0.0 || !inner_radius.is_finite() {
        return Err(GeometryError::new(
            "tube inner radius must be finite and non-negative",
        ));
    }
    if inner_radius >= radius {
        return Err(GeometryError::new(
            "tube inner radius must be smaller than radius",
        ));
    }
    Ok(())
}

fn tube_signed_distance(centerline_distance: f64, radius: f64, inner_radius: f64) -> f64 {
    let outer = centerline_distance - radius;
    if inner_radius <= 0.0 {
        return outer;
    }
    outer.max(inner_radius - centerline_distance)
}

pub(crate) fn exact_extrusion(profile_distance: f64, axial: f64) -> f64 {
    let outside = (profile_distance.max(0.0).powi(2) + axial.max(0.0).powi(2)).sqrt();
    let inside = profile_distance.max(axial).min(0.0);
    outside + inside
}

pub(crate) fn segment_distance(p: Vec3, first: Vec3, second: Vec3) -> f64 {
    let ba = second - first;
    let denominator = ba.length_squared();
    if denominator <= 1.0e-24 {
        return (p - first).length();
    }
    let h = ((p - first).dot(ba) / denominator).clamp(0.0, 1.0);
    (p - first - ba * h).length()
}

fn flat_capped_segment_tube(p: Vec3, first: Vec3, second: Vec3, radius: f64) -> f64 {
    let ba = second - first;
    let length = ba.length();
    if length <= 1.0e-12 {
        return (p - first).length() - radius;
    }
    let direction = ba / length;
    let projection = (p - first).dot(direction);
    let radial = (p - first - direction * projection).length();
    let profile = radial - radius;
    let axial = (projection - 0.5 * length).abs() - 0.5 * length;
    exact_extrusion(profile, axial)
}

/// Exact distance from a point to one quadratic Bezier span (iq's closed-form
/// cubic solve), pointwise port of `_quadratic_bezier_distance_numpy`.
pub(crate) fn quadratic_bezier_distance(p: Vec3, start: Vec3, control: Vec3, end: Vec3) -> f64 {
    let a = control - start;
    let b = start - control * 2.0 + end;
    let c = a * 2.0;
    let b_dot_b = b.length_squared();
    if b_dot_b <= 1.0e-24 {
        return segment_distance(p, start, end);
    }
    let d = start - p;
    let kk = 1.0 / b_dot_b;
    let kx = kk * a.dot(b);
    let ky = kk * (2.0 * a.length_squared() + d.dot(b)) / 3.0;
    let kz = kk * d.dot(a);
    let pp = ky - kx * kx;
    let qq = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    let h = qq * qq + 4.0 * pp * pp * pp;
    let result = if h >= 0.0 {
        let h_root = h.max(0.0).sqrt();
        let x0 = 0.5 * (h_root - qq);
        let x1 = 0.5 * (-h_root - qq);
        let t = (x0.cbrt() + x1.cbrt() - kx).clamp(0.0, 1.0);
        let w = d + (c + b * t) * t;
        w.length_squared()
    } else {
        let z = (-pp).max(0.0).sqrt();
        let denominator = 2.0 * pp * z;
        let angle_argument = if denominator.abs() > 1.0e-24 {
            qq / denominator
        } else {
            0.0
        };
        let angle = angle_argument.clamp(-1.0, 1.0).acos() / 3.0;
        let m = angle.cos();
        let n = angle.sin() * 1.732050808;
        let t0 = ((m + m) * z - kx).clamp(0.0, 1.0);
        let t1 = ((-n - m) * z - kx).clamp(0.0, 1.0);
        let w0 = d + (c + b * t0) * t0;
        let w1 = d + (c + b * t1) * t1;
        w0.length_squared().min(w1.length_squared())
    };
    result.max(0.0).sqrt()
}

fn quadratic_bezier_spans(points: &[Vec3]) -> impl Iterator<Item = (Vec3, Vec3, Vec3)> + '_ {
    (0..points.len().saturating_sub(2))
        .step_by(2)
        .map(|index| (points[index], points[index + 1], points[index + 2]))
}

fn unit_vector(first: Vec3, second: Vec3) -> GeometryResult<Vec3> {
    let vector = second - first;
    let length = vector.length();
    if length <= 1.0e-12 || !length.is_finite() {
        return Err(GeometryError::new(
            "tube cap tangent must be finite and nonzero",
        ));
    }
    Ok(vector / length)
}

fn quadratic_bezier_endpoint_tangents(points: &[Vec3]) -> GeometryResult<(Vec3, Vec3)> {
    let start = points[0];
    let first_control = points[1];
    let first_end = points[2];
    let end = points[points.len() - 1];
    let last_control = points[points.len() - 2];
    let last_start = points[points.len() - 3];
    let start_tangent =
        unit_vector(start, first_control).or_else(|_| unit_vector(start, first_end))?;
    let end_tangent = unit_vector(last_control, end).or_else(|_| unit_vector(last_start, end))?;
    Ok((start_tangent, end_tangent))
}

fn points_bounds(points: &[Vec3], radius: f64) -> GeometryResult<BoundingBox3D> {
    let padding = vec3(radius, radius, radius);
    BoundingBox3D::from_points(
        points
            .iter()
            .flat_map(|p| [*p - padding, *p + padding]),
    )
}

#[derive(Debug, Clone, PartialEq)]
pub struct PolylineTube {
    pub points: Vec<Vec3>,
    pub radius: f64,
    pub inner_radius: f64,
    pub caps: CapStyle,
}

impl PolylineTube {
    pub fn new(
        points: Vec<Vec3>,
        radius: f64,
        inner_radius: f64,
        caps: CapStyle,
    ) -> GeometryResult<Self> {
        if points.len() < 2 {
            return Err(GeometryError::new(
                "polyline tube requires at least two points",
            ));
        }
        if points
            .windows(2)
            .all(|pair| (pair[1] - pair[0]).length() <= 1.0e-12)
        {
            return Err(GeometryError::new(
                "polyline tube requires at least one nonzero segment",
            ));
        }
        validate_tube_radius(radius, inner_radius)?;
        Ok(Self {
            points,
            radius,
            inner_radius,
            caps,
        })
    }

    pub fn eval_point(&self, p: Vec3) -> f64 {
        let centerline = self
            .points
            .windows(2)
            .map(|pair| segment_distance(p, pair[0], pair[1]))
            .fold(f64::INFINITY, f64::min);
        if self.caps == CapStyle::Round {
            return tube_signed_distance(centerline, self.radius, self.inner_radius);
        }
        let outer = self
            .points
            .windows(2)
            .map(|pair| flat_capped_segment_tube(p, pair[0], pair[1], self.radius))
            .fold(f64::INFINITY, f64::min);
        if self.inner_radius <= 0.0 {
            return outer;
        }
        outer.max(self.inner_radius - centerline)
    }

    pub fn bounding_box(&self) -> GeometryResult<BoundingBox3D> {
        points_bounds(&self.points, self.radius)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct QuadraticBezierTube {
    pub points: Vec<Vec3>,
    pub radius: f64,
    pub inner_radius: f64,
    pub caps: CapStyle,
}

impl QuadraticBezierTube {
    pub fn new(
        points: Vec<Vec3>,
        radius: f64,
        inner_radius: f64,
        caps: CapStyle,
    ) -> GeometryResult<Self> {
        if points.len() < 3 {
            return Err(GeometryError::new(
                "quadratic Bezier tube requires at least three points",
            ));
        }
        if points.len().is_multiple_of(2) {
            return Err(GeometryError::new(
                "quadratic Bezier tube requires an odd point count: anchor, control, anchor",
            ));
        }
        if quadratic_bezier_spans(&points).all(|(start, control, end)| {
            (control - start).length() <= 1.0e-12 && (end - start).length() <= 1.0e-12
        }) {
            return Err(GeometryError::new(
                "quadratic Bezier tube requires at least one nonzero span",
            ));
        }
        validate_tube_radius(radius, inner_radius)?;
        if caps == CapStyle::Flat {
            quadratic_bezier_endpoint_tangents(&points)?;
        }
        Ok(Self {
            points,
            radius,
            inner_radius,
            caps,
        })
    }

    /// Matches the Python `kind` property, which splits on span count.
    pub fn kind(&self) -> &'static str {
        if self.points.len() > 3 {
            "quadratic_bezier_polycurve_tube"
        } else {
            "quadratic_bezier_tube"
        }
    }

    pub fn eval_point(&self, p: Vec3) -> f64 {
        let centerline = quadratic_bezier_spans(&self.points)
            .map(|(start, control, end)| quadratic_bezier_distance(p, start, control, end))
            .fold(f64::INFINITY, f64::min);
        let mut outer = centerline - self.radius;
        if self.caps == CapStyle::Flat {
            let (start_tangent, end_tangent) = quadratic_bezier_endpoint_tangents(&self.points)
                .expect("validated at construction");
            let start = self.points[0];
            let end = self.points[self.points.len() - 1];
            let start_plane = (start - p).dot(start_tangent);
            let end_plane = (p - end).dot(end_tangent);
            outer = outer.max(start_plane).max(end_plane);
        }
        if self.inner_radius <= 0.0 {
            return outer;
        }
        outer.max(self.inner_radius - centerline)
    }

    pub fn bounding_box(&self) -> GeometryResult<BoundingBox3D> {
        points_bounds(&self.points, self.radius)
    }
}
