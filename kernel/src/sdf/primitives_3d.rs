//! Exact 3D SDF primitives, ported one-to-one from `core/sdf/primitives_3d.py`.

use crate::bbox::BoundingBox3D;
use crate::error::{GeometryError, GeometryResult};
use crate::frame::Frame;
use crate::vec3::{vec3, Vec3};

/// numpy's `np.sign`: -1, 0, or +1 (unlike `f64::signum`, which never yields 0).
#[inline]
pub(crate) fn py_sign(value: f64) -> f64 {
    if value > 0.0 {
        1.0
    } else if value < 0.0 {
        -1.0
    } else {
        0.0
    }
}

fn oriented_box_bounds(
    center: Vec3,
    frame: &Frame,
    half_size: Vec3,
) -> GeometryResult<BoundingBox3D> {
    if frame.is_identity() {
        return BoundingBox3D::new(
            center.x - half_size.x,
            center.x + half_size.x,
            center.y - half_size.y,
            center.y + half_size.y,
            center.z - half_size.z,
            center.z + half_size.z,
        );
    }
    let mut corners = Vec::with_capacity(8);
    for sign_u in [-1.0, 1.0] {
        for sign_v in [-1.0, 1.0] {
            for sign_w in [-1.0, 1.0] {
                corners.push(
                    center
                        + frame.u * (sign_u * half_size.x)
                        + frame.v * (sign_v * half_size.y)
                        + frame.w * (sign_w * half_size.z),
                );
            }
        }
    }
    BoundingBox3D::from_points(corners)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Sphere {
    pub center: Vec3,
    pub radius: f64,
}

impl Sphere {
    pub fn new(center: Vec3, radius: f64) -> GeometryResult<Self> {
        if radius <= 0.0 {
            return Err(GeometryError::new("sphere radius must be positive"));
        }
        Ok(Self { center, radius })
    }

    #[inline]
    pub fn eval_point(&self, p: Vec3) -> f64 {
        (p - self.center).length() - self.radius
    }

    pub fn bounding_box(&self) -> GeometryResult<BoundingBox3D> {
        let c = self.center;
        let r = self.radius;
        BoundingBox3D::new(c.x - r, c.x + r, c.y - r, c.y + r, c.z - r, c.z + r)
    }
}

/// Named `Box3` to avoid clashing with `std::boxed::Box`; serializes as "box".
#[derive(Debug, Clone, PartialEq)]
pub struct Box3 {
    pub center: Vec3,
    pub half_size: Vec3,
    pub frame: Frame,
}

#[inline]
fn box_distance(q: Vec3) -> f64 {
    let outside = vec3(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).length();
    let inside = q.x.max(q.y.max(q.z)).min(0.0);
    outside + inside
}

impl Box3 {
    pub fn new(center: Vec3, half_size: Vec3, frame: Frame) -> GeometryResult<Self> {
        if half_size.x <= 0.0 || half_size.y <= 0.0 || half_size.z <= 0.0 {
            return Err(GeometryError::new("box half sizes must be positive"));
        }
        Ok(Self {
            center,
            half_size,
            frame,
        })
    }

    #[inline]
    pub fn eval_point(&self, p: Vec3) -> f64 {
        let local = self.frame.to_local(p, self.center);
        let q = vec3(
            local.x.abs() - self.half_size.x,
            local.y.abs() - self.half_size.y,
            local.z.abs() - self.half_size.z,
        );
        box_distance(q)
    }

    pub fn bounding_box(&self) -> GeometryResult<BoundingBox3D> {
        oriented_box_bounds(self.center, &self.frame, self.half_size)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cylinder {
    pub center: Vec3,
    pub radius: f64,
    pub half_height: f64,
    pub frame: Frame,
}

impl Cylinder {
    pub fn new(center: Vec3, radius: f64, half_height: f64, frame: Frame) -> GeometryResult<Self> {
        if radius <= 0.0 || half_height <= 0.0 {
            return Err(GeometryError::new("cylinder dimensions must be positive"));
        }
        Ok(Self {
            center,
            radius,
            half_height,
            frame,
        })
    }

    #[inline]
    pub fn eval_point(&self, p: Vec3) -> f64 {
        let local = self.frame.to_local(p, self.center);
        let radial = (local.x * local.x + local.y * local.y).sqrt() - self.radius;
        let axial = local.z.abs() - self.half_height;
        let outside = (radial.max(0.0).powi(2) + axial.max(0.0).powi(2)).sqrt();
        let inside = radial.max(axial).min(0.0);
        outside + inside
    }

    pub fn bounding_box(&self) -> GeometryResult<BoundingBox3D> {
        oriented_box_bounds(
            self.center,
            &self.frame,
            vec3(self.radius, self.radius, self.half_height),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Cone {
    pub center: Vec3,
    pub radius: f64,
    pub half_height: f64,
    pub frame: Frame,
}

impl Cone {
    pub fn new(center: Vec3, radius: f64, half_height: f64, frame: Frame) -> GeometryResult<Self> {
        if radius <= 0.0 || half_height <= 0.0 {
            return Err(GeometryError::new("cone dimensions must be positive"));
        }
        Ok(Self {
            center,
            radius,
            half_height,
            frame,
        })
    }

    #[inline]
    pub fn eval_point(&self, p: Vec3) -> f64 {
        let local = self.frame.to_local(p, self.center);
        let wx = (local.x * local.x + local.y * local.y).sqrt();
        let wy = local.z - self.half_height;
        let qx = self.radius;
        let qy = -2.0 * self.half_height;
        let denominator = qx * qx + qy * qy;
        let h = ((wx * qx + wy * qy) / denominator).clamp(0.0, 1.0);
        let ax = wx - qx * h;
        let ay = wy - qy * h;
        let bx = wx - qx * (wx / qx).clamp(0.0, 1.0);
        let by = wy - qy;
        let d = (ax * ax + ay * ay).min(bx * bx + by * by);
        let s = (-(wx * qy - wy * qx)).max(-(wy - qy));
        d.sqrt() * py_sign(s)
    }

    pub fn bounding_box(&self) -> GeometryResult<BoundingBox3D> {
        oriented_box_bounds(
            self.center,
            &self.frame,
            vec3(self.radius, self.radius, self.half_height),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CappedCone {
    pub center: Vec3,
    pub radius_a: f64,
    pub radius_b: f64,
    pub half_height: f64,
    pub frame: Frame,
}

impl CappedCone {
    pub fn new(
        center: Vec3,
        radius_a: f64,
        radius_b: f64,
        half_height: f64,
        frame: Frame,
    ) -> GeometryResult<Self> {
        if radius_a <= 0.0 || radius_b <= 0.0 || half_height <= 0.0 {
            return Err(GeometryError::new(
                "capped cone dimensions must be positive",
            ));
        }
        Ok(Self {
            center,
            radius_a,
            radius_b,
            half_height,
            frame,
        })
    }

    #[inline]
    pub fn eval_point(&self, p: Vec3) -> f64 {
        let local = self.frame.to_local(p, self.center);
        let qx = (local.x * local.x + local.y * local.y).sqrt();
        let qy = local.z;
        let k1x = self.radius_b;
        let k1y = self.half_height;
        let k2x = self.radius_b - self.radius_a;
        let k2y = 2.0 * self.half_height;
        let cap_radius = if qy < 0.0 {
            self.radius_a
        } else {
            self.radius_b
        };
        let cax = qx - qx.min(cap_radius);
        let cay = qy.abs() - self.half_height;
        let dot_k2 = k2x * k2x + k2y * k2y;
        let projection = ((k1x - qx) * k2x + (k1y - qy) * k2y) / dot_k2;
        let f = projection.clamp(0.0, 1.0);
        let cbx = qx - k1x + k2x * f;
        let cby = qy - k1y + k2y * f;
        let sign = if cbx < 0.0 && cay < 0.0 { -1.0 } else { 1.0 };
        let distance_squared = (cax * cax + cay * cay).min(cbx * cbx + cby * cby);
        sign * distance_squared.sqrt()
    }

    pub fn bounding_box(&self) -> GeometryResult<BoundingBox3D> {
        let radius = self.radius_a.max(self.radius_b);
        oriented_box_bounds(
            self.center,
            &self.frame,
            vec3(radius, radius, self.half_height),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Pyramid {
    pub center: Vec3,
    pub base_half_size: f64,
    pub half_height: f64,
    pub frame: Frame,
}

impl Pyramid {
    pub fn new(
        center: Vec3,
        base_half_size: f64,
        half_height: f64,
        frame: Frame,
    ) -> GeometryResult<Self> {
        if base_half_size <= 0.0 || half_height <= 0.0 {
            return Err(GeometryError::new("pyramid dimensions must be positive"));
        }
        Ok(Self {
            center,
            base_half_size,
            half_height,
            frame,
        })
    }

    #[inline]
    pub fn eval_point(&self, p: Vec3) -> f64 {
        let local = self.frame.to_local(p, self.center);
        let scale = 2.0 * self.base_half_size;
        let mut px = (local.x / scale).abs();
        let py = (local.z + self.half_height) / scale;
        let mut pz = (local.y / scale).abs();
        if pz > px {
            std::mem::swap(&mut px, &mut pz);
        }
        px -= 0.5;
        pz -= 0.5;
        let h = 2.0 * self.half_height / scale;
        let m2 = h * h + 0.25;
        let qx = pz;
        let qy = h * py - 0.5 * px;
        let qz = h * px + 0.5 * py;
        let s = (-qx).max(0.0);
        let t = ((qy - 0.5 * pz) / (m2 + 0.25)).clamp(0.0, 1.0);
        let a = m2 * (qx + s) * (qx + s) + qy * qy;
        let b = m2 * (qx + 0.5 * t) * (qx + 0.5 * t) + (qy - m2 * t) * (qy - m2 * t);
        let d2 = if qy.min(-qx * m2 - qy * 0.5) > 0.0 {
            0.0
        } else {
            a.min(b)
        };
        scale * ((d2 + qz * qz) / m2).sqrt() * py_sign(qz.max(-py))
    }

    pub fn bounding_box(&self) -> GeometryResult<BoundingBox3D> {
        oriented_box_bounds(
            self.center,
            &self.frame,
            vec3(self.base_half_size, self.base_half_size, self.half_height),
        )
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoxFrame {
    pub center: Vec3,
    pub half_size: Vec3,
    pub thickness: f64,
    pub frame: Frame,
}

impl BoxFrame {
    pub fn new(
        center: Vec3,
        half_size: Vec3,
        thickness: f64,
        frame: Frame,
    ) -> GeometryResult<Self> {
        if half_size.x <= 0.0 || half_size.y <= 0.0 || half_size.z <= 0.0 || thickness <= 0.0 {
            return Err(GeometryError::new("box frame dimensions must be positive"));
        }
        Ok(Self {
            center,
            half_size,
            thickness,
            frame,
        })
    }

    #[inline]
    pub fn eval_point(&self, p: Vec3) -> f64 {
        let local = self.frame.to_local(p, self.center);
        let px = local.x.abs() - self.half_size.x;
        let py = local.y.abs() - self.half_size.y;
        let pz = local.z.abs() - self.half_size.z;
        let qx = (px + self.thickness).abs() - self.thickness;
        let qy = (py + self.thickness).abs() - self.thickness;
        let qz = (pz + self.thickness).abs() - self.thickness;
        box_distance(vec3(px, qy, qz))
            .min(box_distance(vec3(qx, py, qz)))
            .min(box_distance(vec3(qx, qy, pz)))
    }

    pub fn bounding_box(&self) -> GeometryResult<BoundingBox3D> {
        oriented_box_bounds(self.center, &self.frame, self.half_size)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Torus {
    pub center: Vec3,
    pub major_radius: f64,
    pub minor_radius: f64,
    pub frame: Frame,
}

impl Torus {
    pub fn new(
        center: Vec3,
        major_radius: f64,
        minor_radius: f64,
        frame: Frame,
    ) -> GeometryResult<Self> {
        if major_radius <= 0.0 || minor_radius <= 0.0 {
            return Err(GeometryError::new("torus radii must be positive"));
        }
        Ok(Self {
            center,
            major_radius,
            minor_radius,
            frame,
        })
    }

    #[inline]
    pub fn eval_point(&self, p: Vec3) -> f64 {
        let local = self.frame.to_local(p, self.center);
        let qx = (local.x * local.x + local.y * local.y).sqrt() - self.major_radius;
        (qx * qx + local.z * local.z).sqrt() - self.minor_radius
    }

    pub fn bounding_box(&self) -> GeometryResult<BoundingBox3D> {
        let outer = self.major_radius + self.minor_radius;
        oriented_box_bounds(
            self.center,
            &self.frame,
            vec3(outer, outer, self.minor_radius),
        )
    }
}
