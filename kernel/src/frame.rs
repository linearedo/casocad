use crate::error::{GeometryError, GeometryResult};
use crate::vec3::{vec3, Vec3};

/// Orthonormal orientation frame (axis_u, axis_v, axis_w) shared by every
/// oriented primitive. Construction normalizes each axis and refuses
/// non-orthogonal inputs, matching the Python `_orthonormal_frame` helper.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Frame {
    pub u: Vec3,
    pub v: Vec3,
    pub w: Vec3,
}

pub const IDENTITY_FRAME: Frame = Frame {
    u: vec3(1.0, 0.0, 0.0),
    v: vec3(0.0, 1.0, 0.0),
    w: vec3(0.0, 0.0, 1.0),
};

pub fn normalized(vector: Vec3) -> GeometryResult<Vec3> {
    let length = vector.length();
    if length <= 1e-12 {
        return Err(GeometryError::new("orientation axis must be nonzero"));
    }
    Ok(vector / length)
}

impl Frame {
    pub fn orthonormal(axis_u: Vec3, axis_v: Vec3, axis_w: Vec3) -> GeometryResult<Frame> {
        let u = normalized(axis_u)?;
        let v = normalized(axis_v)?;
        let w = normalized(axis_w)?;
        if u.dot(v).abs() > 1e-6 || u.dot(w).abs() > 1e-6 || v.dot(w).abs() > 1e-6 {
            return Err(GeometryError::new("orientation axes must be orthogonal"));
        }
        Ok(Frame { u, v, w })
    }

    pub fn is_identity(&self) -> bool {
        *self == IDENTITY_FRAME
    }

    /// World point -> local frame coordinates relative to `center`.
    #[inline]
    pub fn to_local(&self, point: Vec3, center: Vec3) -> Vec3 {
        let r = point - center;
        vec3(r.dot(self.u), r.dot(self.v), r.dot(self.w))
    }

    /// Local frame coordinates -> world point.
    #[inline]
    pub fn to_world(&self, local: Vec3, center: Vec3) -> Vec3 {
        center + self.u * local.x + self.v * local.y + self.w * local.z
    }
}

impl Default for Frame {
    fn default() -> Self {
        IDENTITY_FRAME
    }
}
