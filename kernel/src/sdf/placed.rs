//! Placed 1D/2D profiles: lower-dimensional exact SDFs embedded in 3D space
//! via a workplane/line frame. Ported from `core/sdf/placed_1d.py` and
//! `core/sdf/placed_2d.py`.

use crate::error::{GeometryError, GeometryResult};
use crate::frame::normalized;
use crate::sdf::node::Node;
use crate::sdf::primitives_1d::Profile1D;
use crate::sdf::primitives_2d::Profile2D;
use crate::vec3::Vec3;

fn workplane_axes(axis_u: Vec3, axis_v: Vec3) -> GeometryResult<(Vec3, Vec3)> {
    let u = normalized(axis_u).map_err(|_| GeometryError::new("workplane axes must be nonzero"))?;
    let v = normalized(axis_v).map_err(|_| GeometryError::new("workplane axes must be nonzero"))?;
    if u.dot(v).abs() > 1e-6 {
        return Err(GeometryError::new("workplane axes must be orthogonal"));
    }
    Ok((u, v))
}

/// A filled 2D profile placed on a 3D workplane. Its SDF is the profile
/// distance of the point's in-plane projection (constant along the normal).
#[derive(Debug, Clone, PartialEq)]
pub struct PlacedSdf2D {
    pub profile: Profile2D,
    pub origin: Vec3,
    pub axis_u: Vec3,
    pub axis_v: Vec3,
    pub sources: Vec<Node>,
}

impl PlacedSdf2D {
    pub fn new(
        profile: Profile2D,
        origin: Vec3,
        axis_u: Vec3,
        axis_v: Vec3,
        sources: Vec<Node>,
    ) -> GeometryResult<Self> {
        let (u, v) = workplane_axes(axis_u, axis_v)?;
        Ok(Self {
            profile,
            origin,
            axis_u: u,
            axis_v: v,
            sources,
        })
    }

    pub fn normal(&self) -> Vec3 {
        let normal = self.axis_u.cross(self.axis_v);
        normal / normal.length()
    }

    /// World point -> (u, v, signed plane offset).
    #[inline]
    pub fn project(&self, p: Vec3) -> (f64, f64, f64) {
        let r = p - self.origin;
        (r.dot(self.axis_u), r.dot(self.axis_v), r.dot(self.normal()))
    }

    #[inline]
    pub fn eval_point(&self, p: Vec3) -> f64 {
        let (u, v, _plane) = self.project(p);
        self.profile.eval(u, v)
    }

    pub fn is_coplanar_with(&self, other: &PlacedSdf2D, tolerance: f64) -> bool {
        let same_axes = (self.axis_u - other.axis_u).length() <= tolerance * 3.0_f64.sqrt()
            && (self.axis_v - other.axis_v).length() <= tolerance * 3.0_f64.sqrt();
        if !same_axes {
            return false;
        }
        let delta = other.origin - self.origin;
        delta.dot(self.normal()).abs() <= tolerance
    }

    pub fn shares_plane_with(&self, other: &PlacedSdf2D, tolerance: f64) -> bool {
        let normal_alignment = self.normal().dot(other.normal()).abs();
        if (1.0 - normal_alignment).abs() > tolerance {
            return false;
        }
        let delta = other.origin - self.origin;
        delta.dot(self.normal()).abs() <= tolerance
    }

    pub(crate) fn workplane_corners(&self) -> [Vec3; 4] {
        let (u_min, u_max, v_min, v_max) = self.profile.bounds();
        [
            self.origin + self.axis_u * u_min + self.axis_v * v_min,
            self.origin + self.axis_u * u_min + self.axis_v * v_max,
            self.origin + self.axis_u * u_max + self.axis_v * v_min,
            self.origin + self.axis_u * u_max + self.axis_v * v_max,
        ]
    }
}

/// A curve (open polyline / Bezier polycurve) placed on a 3D workplane,
/// acting as a 1D boundary object. The profile must be a curve profile.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacedPolyline1D {
    pub profile: Profile2D,
    pub origin: Vec3,
    pub axis_u: Vec3,
    pub axis_v: Vec3,
}

impl PlacedPolyline1D {
    pub fn new(
        profile: Profile2D,
        origin: Vec3,
        axis_u: Vec3,
        axis_v: Vec3,
    ) -> GeometryResult<Self> {
        if !matches!(
            profile,
            Profile2D::Polyline { .. } | Profile2D::QuadraticBezierCurve { .. }
        ) {
            return Err(GeometryError::new("PlacedPolyline1D requires a curve profile"));
        }
        let (u, v) = workplane_axes(axis_u, axis_v)?;
        Ok(Self {
            profile,
            origin,
            axis_u: u,
            axis_v: v,
        })
    }

    /// Kind string matching the Python property.
    pub fn kind(&self) -> &'static str {
        match &self.profile {
            Profile2D::QuadraticBezierCurve { points } => {
                if points.len() > 3 {
                    "placed_quadratic_bezier_polycurve_1d"
                } else {
                    "placed_quadratic_bezier_curve_1d"
                }
            }
            _ => "placed_polyline_1d",
        }
    }

    pub fn normal(&self) -> Vec3 {
        let normal = self.axis_u.cross(self.axis_v);
        normal / normal.length()
    }

    #[inline]
    pub fn project(&self, p: Vec3) -> (f64, f64, f64) {
        let r = p - self.origin;
        (r.dot(self.axis_u), r.dot(self.axis_v), r.dot(self.normal()))
    }

    #[inline]
    pub fn eval_point(&self, p: Vec3) -> f64 {
        let (u, v, _plane) = self.project(p);
        self.profile.eval(u, v)
    }

    pub fn contains_point(&self, p: Vec3, tolerance: f64) -> bool {
        let (u, v, plane) = self.project(p);
        plane.abs() <= tolerance && self.profile.eval(u, v) <= tolerance
    }

    pub(crate) fn workplane_corners(&self) -> [Vec3; 4] {
        let (u_min, u_max, v_min, v_max) = self.profile.bounds();
        [
            self.origin + self.axis_u * u_min + self.axis_v * v_min,
            self.origin + self.axis_u * u_min + self.axis_v * v_max,
            self.origin + self.axis_u * u_max + self.axis_v * v_min,
            self.origin + self.axis_u * u_max + self.axis_v * v_max,
        ]
    }
}

/// A filled 1D profile placed on a 3D line.
#[derive(Debug, Clone, PartialEq)]
pub struct PlacedSdf1D {
    pub profile: Profile1D,
    pub origin: Vec3,
    pub axis_u: Vec3,
    pub sources: Vec<Node>,
}

impl PlacedSdf1D {
    pub fn new(
        profile: Profile1D,
        origin: Vec3,
        axis_u: Vec3,
        sources: Vec<Node>,
    ) -> GeometryResult<Self> {
        let axis =
            normalized(axis_u).map_err(|_| GeometryError::new("line axis must be nonzero"))?;
        Ok(Self {
            profile,
            origin,
            axis_u: axis,
            sources,
        })
    }

    /// World point -> (line coordinate, radial distance from the line).
    #[inline]
    pub fn project(&self, p: Vec3) -> (f64, f64) {
        let r = p - self.origin;
        let coordinate = r.dot(self.axis_u);
        let perpendicular = r - self.axis_u * coordinate;
        (coordinate, perpendicular.length())
    }

    #[inline]
    pub fn eval_point(&self, p: Vec3) -> f64 {
        let (coordinate, _radial) = self.project(p);
        self.profile.eval(coordinate)
    }

    pub fn contains_point(&self, p: Vec3, tolerance: f64) -> bool {
        let (coordinate, radial) = self.project(p);
        radial <= tolerance && self.profile.eval(coordinate) <= tolerance
    }

    pub fn is_collinear_with(&self, other: &PlacedSdf1D, tolerance: f64) -> bool {
        let axis_delta = self.axis_u - other.axis_u;
        if axis_delta.x.abs() > tolerance
            || axis_delta.y.abs() > tolerance
            || axis_delta.z.abs() > tolerance
        {
            return false;
        }
        let delta = other.origin - self.origin;
        let perpendicular = delta - self.axis_u * delta.dot(self.axis_u);
        perpendicular.length() <= tolerance
    }
}
