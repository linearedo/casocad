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

    /// Frame -> intrinsic Euler angles `[rx, ry, rz]` in degrees, under the
    /// convention `R = Rz(rz) · Ry(ry) · Rx(rx)` with `u`, `v`, `w` as the
    /// columns of `R`. At gimbal lock (`|u.z| ≈ 1`) `rz` is fixed to 0 and
    /// the remaining rotation is folded into `rx`.
    pub fn to_euler_degrees(&self) -> [f64; 3] {
        let sy = (-self.u.z).clamp(-1.0, 1.0);
        let ry = sy.asin();
        let (rx, rz) = if self.u.z.abs() >= 1.0 - 1e-9 {
            let rx = if self.u.z < 0.0 {
                self.v.x.atan2(self.v.y)
            } else {
                (-self.v.x).atan2(self.v.y)
            };
            (rx, 0.0)
        } else {
            (self.v.z.atan2(self.w.z), self.u.y.atan2(self.u.x))
        };
        [rx.to_degrees(), ry.to_degrees(), rz.to_degrees()]
    }

    /// Intrinsic Euler angles in degrees -> frame (`R = Rz · Ry · Rx`).
    /// The columns of a rotation matrix are orthonormal by construction, so
    /// the result always satisfies [`Frame::orthonormal`].
    pub fn from_euler_degrees(rx: f64, ry: f64, rz: f64) -> Frame {
        let (sa, ca) = rx.to_radians().sin_cos();
        let (sb, cb) = ry.to_radians().sin_cos();
        let (sg, cg) = rz.to_radians().sin_cos();
        Frame {
            u: vec3(cg * cb, sg * cb, -sb),
            v: vec3(cg * sb * sa - sg * ca, sg * sb * sa + cg * ca, cb * sa),
            w: vec3(cg * sb * ca + sg * sa, sg * sb * ca - cg * sa, cb * ca),
        }
    }
}

impl Default for Frame {
    fn default() -> Self {
        IDENTITY_FRAME
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_frames_close(a: &Frame, b: &Frame) {
        for (left, right) in [(a.u, b.u), (a.v, b.v), (a.w, b.w)] {
            assert!(
                (left - right).length() < 1e-9,
                "frames differ: {left:?} vs {right:?}"
            );
        }
    }

    #[test]
    fn identity_round_trip() {
        assert_eq!(IDENTITY_FRAME.to_euler_degrees(), [0.0, 0.0, 0.0]);
        assert_frames_close(&Frame::from_euler_degrees(0.0, 0.0, 0.0), &IDENTITY_FRAME);
    }

    #[test]
    fn single_axis_round_trip() {
        for angles in [[30.0, 0.0, 0.0], [0.0, 30.0, 0.0], [0.0, 0.0, 30.0]] {
            let frame = Frame::from_euler_degrees(angles[0], angles[1], angles[2]);
            let recovered = frame.to_euler_degrees();
            for (expected, actual) in angles.iter().zip(recovered.iter()) {
                assert!((expected - actual).abs() < 1e-9, "{angles:?} -> {recovered:?}");
            }
        }
    }

    #[test]
    fn combined_rotation_round_trip() {
        let angles = [30.0, -20.0, 45.0];
        let frame = Frame::from_euler_degrees(angles[0], angles[1], angles[2]);
        Frame::orthonormal(frame.u, frame.v, frame.w).expect("orthonormal");
        let recovered = frame.to_euler_degrees();
        for (expected, actual) in angles.iter().zip(recovered.iter()) {
            assert!((expected - actual).abs() < 1e-9, "{angles:?} -> {recovered:?}");
        }
    }

    #[test]
    fn gimbal_lock_reproduces_frame() {
        for ry in [90.0, -90.0] {
            let frame = Frame::from_euler_degrees(25.0, ry, 40.0);
            let [rx, recovered_ry, rz] = frame.to_euler_degrees();
            assert!((recovered_ry - ry).abs() < 1e-6);
            assert_eq!(rz, 0.0);
            assert_frames_close(&frame, &Frame::from_euler_degrees(rx, recovered_ry, rz));
        }
    }
}
