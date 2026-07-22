//! Domain-scoped meshing controls.  Regions are pure signed selectors: a
//! negative value means selected, so adding selector shapes does not change
//! the mesher.

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
            return Err("refinement box minima must not exceed maxima".into());
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
    pub first_height: f64,
    pub layers: usize,
    pub growth: f64,
}

impl BoundaryLayerControl {
    pub fn total_height(&self) -> f64 {
        if (self.growth - 1.0).abs() < 1.0e-12 {
            self.first_height * self.layers as f64
        } else {
            self.first_height * (self.growth.powi(self.layers as i32) - 1.0) / (self.growth - 1.0)
        }
    }
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct ControlSet {
    pub refinements: Vec<RefinementControl>,
    pub boundary_layers: Vec<BoundaryLayerControl>,
}

impl ControlSet {
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

    pub fn boundary_layer(
        &mut self,
        domain: impl Into<String>,
        boundary_region: impl Into<String>,
        first_height: f64,
        layers: usize,
        growth: f64,
    ) -> Result<(), String> {
        positive_finite(first_height, "boundary-layer first_height")?;
        positive_finite(growth, "boundary-layer growth")?;
        if layers == 0 {
            return Err("boundary-layer layers must be positive".into());
        }
        self.boundary_layers.push(BoundaryLayerControl {
            domain: domain.into(),
            boundary_region: boundary_region.into(),
            first_height,
            layers,
            growth,
        });
        Ok(())
    }

    pub fn validate(&self, domains: &MeshableDomains) -> Result<(), String> {
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
            let count = domains
                .names()
                .iter()
                .filter(|candidate| candidate.as_str() == name)
                .count();
            if count != 1 {
                return Err(if count == 0 {
                    format!(
                        "unknown meshing-control domain {name:?}; available: {}",
                        domains.names().join(", ")
                    )
                } else {
                    format!("ambiguous meshing-control domain {name:?}")
                });
            }
        }
        for layer in &self.boundary_layers {
            let domain = domains
                .get(&layer.domain)
                .map_err(|error| error.to_string())?;
            let matches = domain
                .boundary_regions
                .iter()
                .filter(|region| region.name == layer.boundary_region)
                .count();
            if matches != 1 {
                return Err(if matches == 0 {
                    format!(
                        "domain {:?} has no boundary region {:?}; available: {}",
                        layer.domain,
                        layer.boundary_region,
                        domain
                            .boundary_regions
                            .iter()
                            .map(|region| region.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    )
                } else {
                    format!(
                        "domain {:?} has ambiguous boundary region {:?}",
                        layer.domain, layer.boundary_region
                    )
                });
            }
        }
        for (index, layer) in self.boundary_layers.iter().enumerate() {
            if self.boundary_layers[..index].iter().any(|other| {
                other.domain == layer.domain
                    && other.boundary_region == layer.boundary_region
                    && other != layer
            }) {
                return Err(format!(
                    "domain {:?} boundary region {:?} has incompatible touching layer controls",
                    layer.domain, layer.boundary_region
                ));
            }
        }
        Ok(())
    }

    /// Smallest domain-scoped requested size, graded away from selectors.
    pub fn size_at(&self, domain: &str, point: Vec3, background: f64) -> f64 {
        self.refinements
            .iter()
            .filter(|control| control.domain == domain)
            .fold(background, |size, control| {
                size.min(control.size + control.gradation * control.region.sdf(point).max(0.0))
            })
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
    fn selectors_and_domain_scoping_choose_the_smallest_size() {
        let mut controls = ControlSet::default();
        controls
            .refinement(
                "sea",
                ControlRegion::sphere(vec3(0.0, 0.0, 0.0), 1.0).unwrap(),
                0.1,
                0.2,
            )
            .unwrap();
        controls
            .refinement(
                "sea",
                ControlRegion::box_region(vec3(-0.2, -0.2, -0.2), vec3(0.2, 0.2, 0.2)).unwrap(),
                0.05,
                0.1,
            )
            .unwrap();
        assert_eq!(controls.size_at("sea", vec3(0.0, 0.0, 0.0), 1.0), 0.05);
        assert_eq!(controls.size_at("pipe", vec3(0.0, 0.0, 0.0), 1.0), 1.0);
    }
}
