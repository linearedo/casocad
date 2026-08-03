//! Typed, platform-neutral meshing controls.

use caso_kernel::meshing::MeshableDomains;
use caso_kernel::vec3::{vec3, Vec3};
#[derive(Debug, Clone, PartialEq)]
pub enum ControlRegion {
    Box { min: Vec3, max: Vec3 },
    Sphere { center: Vec3, radius: f64 },
    Cylinder { a: Vec3, b: Vec3, radius: f64 },
    PolylineTube { points: Vec<Vec3>, radius: f64 },
    Union(Box<Self>, Box<Self>),
    Intersection(Box<Self>, Box<Self>),
    Difference(Box<Self>, Box<Self>),
}

impl ControlRegion {
    pub fn box_region(min: Vec3, max: Vec3) -> Result<Self, String> {
        if min.x > max.x || min.y > max.y || min.z > max.z {
            return Err("control box minima must not exceed maxima".into());
        }
        Ok(Self::Box { min, max })
    }

    pub fn sphere(center: Vec3, radius: f64) -> Result<Self, String> {
        positive_finite(radius, "sphere radius")?;
        Ok(Self::Sphere { center, radius })
    }

    pub fn cylinder(a: Vec3, b: Vec3, radius: f64) -> Result<Self, String> {
        positive_finite(radius, "cylinder radius")?;
        if (b - a).length() <= f64::EPSILON {
            return Err("control cylinder endpoints must be distinct".into());
        }
        Ok(Self::Cylinder { a, b, radius })
    }

    pub fn polyline_tube(points: Vec<Vec3>, radius: f64) -> Result<Self, String> {
        positive_finite(radius, "polyline-tube radius")?;
        if points.len() < 2 {
            return Err("polyline_tube requires at least two points".into());
        }
        Ok(Self::PolylineTube { points, radius })
    }

    pub fn union(self, other: Self) -> Self {
        Self::Union(Box::new(self), Box::new(other))
    }

    pub fn intersection(self, other: Self) -> Self {
        Self::Intersection(Box::new(self), Box::new(other))
    }

    pub fn difference(self, other: Self) -> Self {
        Self::Difference(Box::new(self), Box::new(other))
    }

    pub fn sdf(&self, point: Vec3) -> f64 {
        match self {
            Self::Box { min, max } => {
                let center = (*min + *max) * 0.5;
                let half = (*max - *min) * 0.5;
                let q = vec3(
                    (point.x - center.x).abs() - half.x,
                    (point.y - center.y).abs() - half.y,
                    (point.z - center.z).abs() - half.z,
                );
                vec3(q.x.max(0.0), q.y.max(0.0), q.z.max(0.0)).length()
                    + q.x.max(q.y.max(q.z)).min(0.0)
            }
            Self::Sphere { center, radius } => (point - *center).length() - radius,
            Self::Cylinder { a, b, radius } => segment_distance(point, *a, *b) - radius,
            Self::PolylineTube { points, radius } => {
                points
                    .windows(2)
                    .map(|pair| segment_distance(point, pair[0], pair[1]))
                    .fold(f64::INFINITY, f64::min)
                    - radius
            }
            Self::Union(a, b) => a.sdf(point).min(b.sdf(point)),
            Self::Intersection(a, b) => a.sdf(point).max(b.sdf(point)),
            Self::Difference(a, b) => a.sdf(point).max(-b.sdf(point)),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct RefinementControl {
    pub domain: String,
    pub region: ControlRegion,
    pub size: f64,
    pub gradation: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryLayerControl {
    pub domain: String,
    pub boundary_region: String,
    pub hwall_n: f64,
    /// Soft tangential station-size target. Validity and element quality take
    /// precedence, so the mesher may locally split, merge, or move stations.
    pub hwall_t: f64,
    pub ratio: f64,
    pub thickness: f64,
    pub layers: usize,
}

impl BoundaryLayerControl {
    pub fn total_height(&self) -> f64 {
        if (self.ratio - 1.0).abs() < 1.0e-12 {
            self.hwall_n * self.layers as f64
        } else {
            self.hwall_n * (self.ratio.powi(self.layers as i32) - 1.0) / (self.ratio - 1.0)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ControlSet {
    pub target_size: Option<f64>,
    pub refinements: Vec<RefinementControl>,
    pub boundary_layers: Vec<BoundaryLayerControl>,
}

impl ControlSet {
    pub fn is_empty(&self) -> bool {
        self.target_size.is_none() && self.refinements.is_empty() && self.boundary_layers.is_empty()
    }

    pub fn target_size(&mut self, size: f64) -> Result<(), String> {
        positive_finite(size, "target size")?;
        if self.target_size.is_some() {
            return Err("controls.target_size(...) must be called exactly once".into());
        }
        self.target_size = Some(size);
        Ok(())
    }

    pub fn require_target_size(&self) -> Result<f64, String> {
        let size = self
            .target_size
            .ok_or_else(|| "controls.target_size(...) is required exactly once".to_string())?;
        positive_finite(size, "target size")?;
        Ok(size)
    }

    pub fn refinement(
        &mut self,
        domain: impl Into<String>,
        region: ControlRegion,
        size: f64,
        gradation: f64,
    ) -> Result<(), String> {
        positive_finite(size, "refinement size")?;
        if !gradation.is_finite() || gradation < 0.0 {
            return Err("refinement gradation must be finite and non-negative".into());
        }
        self.refinements.push(RefinementControl {
            domain: domain.into(),
            region,
            size,
            gradation,
        });
        Ok(())
    }

    pub fn refinement_box(
        &mut self,
        domain: impl Into<String>,
        min: Vec3,
        max: Vec3,
        size: f64,
        gradation: f64,
    ) -> Result<(), String> {
        self.refinement(
            domain,
            ControlRegion::box_region(min, max)?,
            size,
            gradation,
        )
    }

    pub fn boundary_layer(
        &mut self,
        domain: impl Into<String>,
        boundary_region: impl Into<String>,
        hwall_n: f64,
        hwall_t: f64,
        ratio: f64,
        thickness: f64,
    ) -> Result<(), String> {
        positive_finite(hwall_n, "boundary-layer hwall_n")?;
        positive_finite(hwall_t, "boundary-layer hwall_t")?;
        positive_finite(ratio, "boundary-layer ratio")?;
        positive_finite(thickness, "boundary-layer thickness")?;
        if ratio < 1.0 {
            return Err("boundary-layer ratio must be at least 1".into());
        }
        if thickness < hwall_n {
            return Err("boundary-layer thickness must fit at least hwall_n".into());
        }
        let mut layers = 0usize;
        let mut total = 0.0;
        let mut height = hwall_n;
        while total + height <= thickness {
            total += height;
            layers = layers
                .checked_add(1)
                .ok_or_else(|| "boundary-layer layer count overflowed".to_string())?;
            height *= ratio;
            if layers == 1_000_000 || !height.is_finite() {
                break;
            }
        }
        self.boundary_layers.push(BoundaryLayerControl {
            domain: domain.into(),
            boundary_region: boundary_region.into(),
            hwall_n,
            hwall_t,
            ratio,
            thickness,
            layers,
        });
        Ok(())
    }

    pub fn validate(&self, domains: &MeshableDomains) -> Result<(), String> {
        self.require_target_size()?;
        for name in self
            .refinements
            .iter()
            .map(|control| control.domain.as_str())
            .chain(
                self.boundary_layers
                    .iter()
                    .map(|control| control.domain.as_str()),
            )
        {
            domains.get(name).map_err(|error| error.to_string())?;
        }
        for layer in &self.boundary_layers {
            let domain = domains
                .get(&layer.domain)
                .map_err(|error| error.to_string())?;
            if !domain
                .boundary_regions
                .iter()
                .any(|region| region.name == layer.boundary_region)
            {
                return Err(format!(
                    "domain {:?} has no boundary region {:?}; available: {}",
                    layer.domain,
                    layer.boundary_region,
                    domain
                        .boundary_regions
                        .iter()
                        .map(|region| region.name.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
        }
        for (index, layer) in self.boundary_layers.iter().enumerate() {
            if self.boundary_layers[..index].iter().any(|other| {
                other.domain == layer.domain
                    && other.boundary_region == layer.boundary_region
                    && (other.hwall_n != layer.hwall_n
                        || other.ratio != layer.ratio
                        || other.layers != layer.layers)
            }) {
                return Err(format!(
                    "domain {:?} boundary region {:?} has incompatible normal layer controls",
                    layer.domain, layer.boundary_region
                ));
            }
        }
        Ok(())
    }

    pub fn size_at(&self, domain: &str, point: Vec3, background: f64) -> f64 {
        self.refinements
            .iter()
            .filter(|control| control.domain == domain)
            .fold(background, |size, control| {
                size.min(control.size + control.gradation * control.region.sdf(point).max(0.0))
            })
    }

    pub fn metadata(&self) -> serde_json::Value {
        serde_json::json!({
            "target_size": self.target_size,
            "refinement": self.refinements.iter().map(|control| serde_json::json!({
                "domain": control.domain,
                "region": region_metadata(&control.region),
                "size": control.size,
                "gradation": control.gradation,
            })).collect::<Vec<_>>(),
            "boundary_layer": self.boundary_layers.iter().map(|control| serde_json::json!({
                "domain": control.domain,
                "boundary_region": control.boundary_region,
                "hwall_n": control.hwall_n,
                "hwall_t": control.hwall_t,
                "ratio": control.ratio,
                "thickness": control.thickness,
                "derived_layers": control.layers,
                "actual_thickness": control.total_height(),
            })).collect::<Vec<_>>(),
        })
    }
}

fn region_metadata(region: &ControlRegion) -> serde_json::Value {
    let point = |value: Vec3| [value.x, value.y, value.z];
    match region {
        ControlRegion::Box { min, max } => {
            serde_json::json!({"box": {"min": point(*min), "max": point(*max)}})
        }
        ControlRegion::Sphere { center, radius } => {
            serde_json::json!({"sphere": {"center": point(*center), "radius": radius}})
        }
        ControlRegion::Cylinder { a, b, radius } => {
            serde_json::json!({"cylinder": {"a": point(*a), "b": point(*b), "radius": radius}})
        }
        ControlRegion::PolylineTube { points, radius } => serde_json::json!({
            "polyline_tube": {
                "points": points.iter().copied().map(point).collect::<Vec<_>>(),
                "radius": radius,
            }
        }),
        ControlRegion::Union(a, b) => {
            serde_json::json!({"union": [region_metadata(a), region_metadata(b)]})
        }
        ControlRegion::Intersection(a, b) => {
            serde_json::json!({"intersection": [region_metadata(a), region_metadata(b)]})
        }
        ControlRegion::Difference(a, b) => {
            serde_json::json!({"difference": [region_metadata(a), region_metadata(b)]})
        }
    }
}

fn positive_finite(value: f64, name: &str) -> Result<(), String> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(format!("{name} must be positive and finite"))
    }
}

fn segment_distance(point: Vec3, a: Vec3, b: Vec3) -> f64 {
    let ab = b - a;
    let t = ((point - a).dot(ab) / ab.dot(ab)).clamp(0.0, 1.0);
    (point - (a + ab * t)).length()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_at_is_domain_scoped_and_graded() {
        let mut controls = ControlSet::default();
        controls
            .refinement(
                "sea",
                ControlRegion::sphere(vec3(0.0, 0.0, 0.0), 1.0).unwrap(),
                0.1,
                0.2,
            )
            .unwrap();
        assert_eq!(controls.size_at("sea", vec3(0.0, 0.0, 0.0), 1.0), 0.1);
        assert_eq!(controls.size_at("pipe", vec3(0.0, 0.0, 0.0), 1.0), 1.0);
    }

    #[test]
    fn boundary_layer_derives_complete_layers_without_exceeding_thickness() {
        let mut controls = ControlSet::default();
        controls
            .boundary_layer("sea", "wall", 0.01, 0.05, 1.2, 0.05)
            .unwrap();
        let layer = &controls.boundary_layers[0];
        assert_eq!(layer.layers, 3);
        assert!((layer.total_height() - 0.0364).abs() < 1.0e-12);
        assert!(layer.total_height() + layer.hwall_n * layer.ratio.powi(3) > layer.thickness);

        assert!(controls
            .boundary_layer("sea", "wall", 0.02, 0.05, 1.2, 0.01)
            .is_err());
        assert!(controls
            .boundary_layer("sea", "wall", 0.01, 0.05, 0.9, 0.05)
            .is_err());
    }
}
