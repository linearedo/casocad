//! Exact generators building 3D solids from placed 2D sections:
//! extrude and revolve. Ported from `core/sdf/solid_from_2d.py`.

use crate::bbox::BoundingBox3D;
use crate::error::{GeometryError, GeometryResult};
use crate::sdf::node::{Node, Shape};
use crate::sdf::placed::PlacedSdf2D;
use crate::sdf::tubes::exact_extrusion;
use crate::vec3::Vec3;

fn require_section(section: &Node) -> GeometryResult<&PlacedSdf2D> {
    match &section.shape {
        Shape::PlacedSdf2D(placed) => Ok(placed),
        _ => Err(GeometryError::new("generator section must be a placed 2D SDF")),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Extrude {
    pub section: Box<Node>,
    pub height: f64,
    pub center_offset: f64,
}

impl Extrude {
    pub fn new(section: Node, height: f64, center_offset: f64) -> GeometryResult<Self> {
        require_section(&section)?;
        if height <= 0.0 || !height.is_finite() {
            return Err(GeometryError::new(
                "extrude height must be finite and positive",
            ));
        }
        if !center_offset.is_finite() {
            return Err(GeometryError::new("extrude center offset must be finite"));
        }
        Ok(Self {
            section: Box::new(section),
            height,
            center_offset,
        })
    }

    pub fn section2d(&self) -> &PlacedSdf2D {
        match &self.section.shape {
            Shape::PlacedSdf2D(placed) => placed,
            _ => unreachable!("validated at construction"),
        }
    }

    #[inline]
    pub fn eval_point(&self, p: Vec3) -> f64 {
        let section = self.section2d();
        let (u, v, plane) = section.project(p);
        let profile_distance = section.profile.eval(u, v);
        let axial = (plane - self.center_offset).abs() - self.height * 0.5;
        exact_extrusion(profile_distance, axial)
    }

    pub fn bounding_box(&self) -> GeometryResult<BoundingBox3D> {
        let section = self.section2d();
        let (u_min, u_max, v_min, v_max) = section.profile.bounds();
        let normal = section.normal();
        let half = self.height * 0.5;
        let center = section.origin + normal * self.center_offset;
        let mut corners = Vec::with_capacity(8);
        for u in [u_min, u_max] {
            for v in [v_min, v_max] {
                for n in [-half, half] {
                    corners.push(center + section.axis_u * u + section.axis_v * v + normal * n);
                }
            }
        }
        BoundingBox3D::from_points(corners)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RevolveAxis {
    U,
    V,
}

impl RevolveAxis {
    pub fn parse(axis: &str) -> GeometryResult<Self> {
        match axis {
            "u" => Ok(Self::U),
            "v" => Ok(Self::V),
            _ => Err(GeometryError::new("revolve axis must be 'u' or 'v'")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::U => "u",
            Self::V => "v",
        }
    }
}

/// The revolve's orthonormal working frame: (origin, axis, radial, tangent).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RevolveFrame {
    pub origin: Vec3,
    pub axis: Vec3,
    pub radial: Vec3,
    pub tangent: Vec3,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Revolve {
    pub section: Box<Node>,
    pub axis: RevolveAxis,
    pub axis_origin: Option<Vec3>,
    pub axis_direction: Option<Vec3>,
    pub radial_direction: Option<Vec3>,
    pub angle_degrees: f64,
}

impl Revolve {
    pub fn new(
        section: Node,
        axis: RevolveAxis,
        axis_origin: Option<Vec3>,
        axis_direction: Option<Vec3>,
        radial_direction: Option<Vec3>,
        angle_degrees: f64,
    ) -> GeometryResult<Self> {
        require_section(&section)?;
        if !angle_degrees.is_finite() || angle_degrees.abs() <= 0.0 || angle_degrees.abs() > 360.0
        {
            return Err(GeometryError::new(
                "revolve angle magnitude must be finite and in (0, 360]",
            ));
        }
        let revolve = Self {
            section: Box::new(section),
            axis,
            axis_origin,
            axis_direction,
            radial_direction,
            angle_degrees,
        };
        revolve.axis_frame()?;
        Ok(revolve)
    }

    pub fn section2d(&self) -> &PlacedSdf2D {
        match &self.section.shape {
            Shape::PlacedSdf2D(placed) => placed,
            _ => unreachable!("validated at construction"),
        }
    }

    pub fn axis_frame(&self) -> GeometryResult<RevolveFrame> {
        let section = self.section2d();
        let origin = self.axis_origin.unwrap_or(section.origin);
        let axis = self.axis_direction.unwrap_or(match self.axis {
            RevolveAxis::U => section.axis_u,
            RevolveAxis::V => section.axis_v,
        });
        let axis_length = axis.length();
        if axis_length <= 1.0e-12 || !axis_length.is_finite() {
            return Err(GeometryError::new(
                "revolve axis direction must be finite and nonzero",
            ));
        }
        let axis = axis / axis_length;
        let mut radial = self.radial_direction.unwrap_or(match self.axis {
            RevolveAxis::U => section.axis_v,
            RevolveAxis::V => section.axis_u,
        });
        radial = radial - axis * radial.dot(axis);
        let radial_length = radial.length();
        if radial_length <= 1.0e-12 || !radial_length.is_finite() {
            return Err(GeometryError::new(
                "revolve radial direction must not be parallel to axis",
            ));
        }
        let radial = radial / radial_length;
        let tangent = axis.cross(radial);
        let tangent_length = tangent.length();
        if tangent_length <= 1.0e-12 || !tangent_length.is_finite() {
            return Err(GeometryError::new("revolve axis frame is degenerate"));
        }
        Ok(RevolveFrame {
            origin,
            axis,
            radial,
            tangent: tangent / tangent_length,
        })
    }

    /// Signed distance of the in-plane angular sector (partial revolutions).
    fn angular_sector_sdf(x: f64, y: f64, angle_degrees: f64) -> f64 {
        if angle_degrees.abs() >= 360.0 - 1.0e-9 {
            return -1.0e6;
        }
        let y = if angle_degrees < 0.0 { -y } else { y };
        let angle = angle_degrees.abs().to_radians();
        let radius = (x * x + y * y).sqrt();
        let mut theta = y.atan2(x);
        if theta < 0.0 {
            theta += 2.0 * std::f64::consts::PI;
        }
        let inside = theta <= angle;
        let ray_distance = |ray_x: f64, ray_y: f64| -> f64 {
            let projection = x * ray_x + y * ray_y;
            let cross = ray_x * y - ray_y * x;
            if projection >= 0.0 {
                cross.abs()
            } else {
                radius
            }
        };
        let start_distance = ray_distance(1.0, 0.0);
        let end_distance = ray_distance(angle.cos(), angle.sin());
        let distance = start_distance.min(end_distance);
        if inside {
            -distance
        } else {
            distance
        }
    }

    pub fn eval_point(&self, p: Vec3) -> f64 {
        let section = self.section2d();
        let frame = self.axis_frame().expect("validated at construction");
        let r = p - frame.origin;
        let axial = r.dot(frame.axis);
        let radial_x = r.dot(frame.radial);
        let radial_y = r.dot(frame.tangent);
        let radial = (radial_x * radial_x + radial_y * radial_y).max(0.0).sqrt();
        let sample = frame.origin + frame.axis * axial + frame.radial * radial;
        let (u, v, _plane) = section.project(sample);
        let profile = section.profile.eval(u, v);
        if self.angle_degrees.abs() >= 360.0 - 1.0e-9 {
            return profile;
        }
        let angular = Self::angular_sector_sdf(radial_x, radial_y, self.angle_degrees);
        let outside = (profile.max(0.0).powi(2) + angular.max(0.0).powi(2)).sqrt();
        let inside = profile.max(angular).min(0.0);
        outside + inside
    }

    pub fn bounding_box(&self) -> GeometryResult<BoundingBox3D> {
        let section = self.section2d();
        let frame = self.axis_frame()?;
        let corners = section.workplane_corners();
        let mut axial_min = f64::INFINITY;
        let mut axial_max = f64::NEG_INFINITY;
        let mut radius: f64 = 0.0;
        for corner in corners {
            let local = corner - frame.origin;
            let axial = local.dot(frame.axis);
            axial_min = axial_min.min(axial);
            axial_max = axial_max.max(axial);
            let radial_vector = local - frame.axis * axial;
            radius = radius.max(radial_vector.length());
        }
        let a = frame.origin + frame.axis * axial_min;
        let b = frame.origin + frame.axis * axial_max;
        let lower = a.min(b);
        let upper = a.max(b);
        BoundingBox3D::new(
            lower.x - radius,
            upper.x + radius,
            lower.y - radius,
            upper.y + radius,
            lower.z - radius,
            upper.z + radius,
        )
    }
}
