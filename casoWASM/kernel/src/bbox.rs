use crate::error::{GeometryError, GeometryResult};
use crate::vec3::Vec3;

/// Axis-aligned traversal bound. Not part of SDF semantics (spec: never treat
/// bounding boxes as geometry) — used for framing, disjointness fast paths,
/// and sampling extents only.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoundingBox3D {
    pub x_min: f64,
    pub x_max: f64,
    pub y_min: f64,
    pub y_max: f64,
    pub z_min: f64,
    pub z_max: f64,
}

impl BoundingBox3D {
    pub fn new(
        x_min: f64,
        x_max: f64,
        y_min: f64,
        y_max: f64,
        z_min: f64,
        z_max: f64,
    ) -> GeometryResult<Self> {
        if !(x_min <= x_max && y_min <= y_max && z_min <= z_max) {
            return Err(GeometryError::new(
                "bounding box minima must not exceed maxima",
            ));
        }
        Ok(Self {
            x_min,
            x_max,
            y_min,
            y_max,
            z_min,
            z_max,
        })
    }

    pub fn from_points<I: IntoIterator<Item = Vec3>>(points: I) -> GeometryResult<Self> {
        let mut iter = points.into_iter();
        let first = iter
            .next()
            .ok_or_else(|| GeometryError::new("bounding box requires at least one point"))?;
        let mut min = first;
        let mut max = first;
        for p in iter {
            min = min.min(p);
            max = max.max(p);
        }
        Self::new(min.x, max.x, min.y, max.y, min.z, max.z)
    }

    pub fn union(&self, other: &BoundingBox3D) -> BoundingBox3D {
        BoundingBox3D {
            x_min: self.x_min.min(other.x_min),
            x_max: self.x_max.max(other.x_max),
            y_min: self.y_min.min(other.y_min),
            y_max: self.y_max.max(other.y_max),
            z_min: self.z_min.min(other.z_min),
            z_max: self.z_max.max(other.z_max),
        }
    }

    /// Errors when the boxes do not overlap (mirrors the Python behavior).
    pub fn intersection(&self, other: &BoundingBox3D) -> GeometryResult<BoundingBox3D> {
        let values = BoundingBox3D {
            x_min: self.x_min.max(other.x_min),
            x_max: self.x_max.min(other.x_max),
            y_min: self.y_min.max(other.y_min),
            y_max: self.y_max.min(other.y_max),
            z_min: self.z_min.max(other.z_min),
            z_max: self.z_max.min(other.z_max),
        };
        if values.x_min > values.x_max || values.y_min > values.y_max || values.z_min > values.z_max
        {
            return Err(GeometryError::new("intersection has an empty bounding box"));
        }
        Ok(values)
    }

    pub fn center(&self) -> Vec3 {
        Vec3::new(
            (self.x_min + self.x_max) * 0.5,
            (self.y_min + self.y_max) * 0.5,
            (self.z_min + self.z_max) * 0.5,
        )
    }

    pub fn diagonal(&self) -> f64 {
        let dx = self.x_max - self.x_min;
        let dy = self.y_max - self.y_min;
        let dz = self.z_max - self.z_min;
        (dx * dx + dy * dy + dz * dz).sqrt()
    }
}
