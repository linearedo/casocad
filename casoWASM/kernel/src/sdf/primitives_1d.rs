//! 1D SDF profiles, ported from `core/sdf/primitives_1d.py`.

use crate::error::{GeometryError, GeometryResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BooleanOp1D {
    Union,
    Intersection,
    Difference,
    Xor,
}

impl BooleanOp1D {
    pub fn parse(op: &str) -> GeometryResult<Self> {
        match op {
            "union" => Ok(Self::Union),
            "intersection" => Ok(Self::Intersection),
            "difference" => Ok(Self::Difference),
            "xor" => Ok(Self::Xor),
            other => Err(GeometryError::new(format!(
                "unsupported 1D boolean operation: {other}"
            ))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Union => "union",
            Self::Intersection => "intersection",
            Self::Difference => "difference",
            Self::Xor => "xor",
        }
    }
}

/// A local filled-segment signed distance over one parameter `t`.
#[derive(Debug, Clone, PartialEq)]
pub enum Profile1D {
    Segment {
        center: f64,
        half_length: f64,
    },
    Offset {
        child: Box<Profile1D>,
        offset: f64,
    },
    Binary {
        left: Box<Profile1D>,
        right: Box<Profile1D>,
        operation: BooleanOp1D,
        /// Legacy field: round-tripped in scene files, never used by eval.
        smoothing: f64,
    },
}

impl Profile1D {
    pub fn segment(center: f64, half_length: f64) -> GeometryResult<Self> {
        if half_length <= 0.0 {
            return Err(GeometryError::new("segment half length must be positive"));
        }
        Ok(Self::Segment {
            center,
            half_length,
        })
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Segment { .. } => "segment",
            Self::Offset { .. } => "offsetprofile1d",
            Self::Binary { .. } => "binaryprofile1d",
        }
    }

    pub fn eval(&self, t: f64) -> f64 {
        match self {
            Self::Segment {
                center,
                half_length,
            } => (t - center).abs() - half_length,
            Self::Offset { child, offset } => child.eval(t - offset),
            Self::Binary {
                left,
                right,
                operation,
                ..
            } => {
                let l = left.eval(t);
                let r = right.eval(t);
                match operation {
                    BooleanOp1D::Union => l.min(r),
                    BooleanOp1D::Intersection => l.max(r),
                    BooleanOp1D::Difference => l.max(-r),
                    BooleanOp1D::Xor => l.min(r).max(-l.max(r)),
                }
            }
        }
    }

    /// Finite local bounds: (t_min, t_max).
    pub fn bounds(&self) -> (f64, f64) {
        match self {
            Self::Segment {
                center,
                half_length,
            } => (center - half_length, center + half_length),
            Self::Offset { child, offset } => {
                let (minimum, maximum) = child.bounds();
                (minimum + offset, maximum + offset)
            }
            Self::Binary {
                left,
                right,
                operation,
                ..
            } => {
                let l = left.bounds();
                if *operation == BooleanOp1D::Difference {
                    return l;
                }
                let r = right.bounds();
                (l.0.min(r.0), l.1.max(r.1))
            }
        }
    }
}
