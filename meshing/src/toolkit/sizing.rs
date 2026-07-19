//! Analytic sizing field (`design_docs/meshing_toolkit.md` §8).
//!
//! Band distances go through `owner_sdf` — a leaf-primitive field, exact on
//! both sides — so the gradation bound holds analytically (Lipschitz by
//! construction, no smoothing grid). The optional curvature clamp only ever
//! evaluates interior points and projects them from the interior
//! (interior-exactness contract).

use caso_kernel::meshing::MeshableDomain;
use caso_kernel::vec3::Vec3;
use caso_kernel::{GeometryError, GeometryResult};

/// Refinement near one boundary region: `size` within `distance` of the
/// region's generating surface, growing at the gradation rate beyond it.
#[derive(Debug, Clone)]
pub struct SizingBand {
    pub region: String,
    pub distance: f64,
    pub size: f64,
}

#[derive(Debug, Clone)]
pub struct SizingSpec {
    /// Cell size far from every refinement source.
    pub background: f64,
    /// Hard floor for the returned size.
    pub min_size: f64,
    /// Maximum growth of size per unit of distance (Lipschitz bound).
    pub gradation: f64,
    pub bands: Vec<SizingBand>,
    /// Optional curvature clamp: size <= factor x radius of curvature at
    /// the nearest wall, evaluated for interior points near the boundary.
    pub curvature_factor: Option<f64>,
}

impl SizingSpec {
    /// Automatic defaults from the domain size: background = diagonal / 20,
    /// floor = diagonal x 1e-4, gradation 0.3, no bands, no curvature clamp.
    pub fn for_domain(domain: &MeshableDomain) -> Self {
        let diagonal = domain.bounds.diagonal().max(1.0e-9);
        Self {
            background: diagonal / 20.0,
            min_size: diagonal * 1.0e-4,
            gradation: 0.3,
            bands: Vec::new(),
            curvature_factor: None,
        }
    }
}

/// A validated, evaluable sizing field over one meshable domain.
#[derive(Debug, Clone)]
pub struct SizingField {
    domain: MeshableDomain,
    spec: SizingSpec,
    band_regions: Vec<usize>,
    diagonal: f64,
}

impl SizingField {
    /// Validates the spec: positive sizes, `min_size <= background`,
    /// non-negative gradation and distances, and every band region name
    /// resolves (the error lists the available names).
    pub fn new(domain: MeshableDomain, spec: SizingSpec) -> GeometryResult<Self> {
        if spec.background <= 0.0 || spec.min_size <= 0.0 {
            return Err(GeometryError::new("sizing sizes must be positive"));
        }
        if spec.min_size > spec.background {
            return Err(GeometryError::new(
                "sizing min_size must not exceed the background size",
            ));
        }
        if spec.gradation < 0.0 {
            return Err(GeometryError::new("sizing gradation must be non-negative"));
        }
        let mut band_regions = Vec::with_capacity(spec.bands.len());
        for band in &spec.bands {
            if band.size <= 0.0 || band.distance < 0.0 {
                return Err(GeometryError::new(
                    "sizing band sizes must be positive and distances non-negative",
                ));
            }
            let index = domain
                .boundary_regions
                .iter()
                .position(|region| region.name == band.region)
                .ok_or_else(|| {
                    let available: Vec<&str> = domain
                        .boundary_regions
                        .iter()
                        .map(|region| region.name.as_str())
                        .collect();
                    GeometryError::new(format!(
                        "unknown sizing band region {:?}; available: {}",
                        band.region,
                        available.join(", ")
                    ))
                })?;
            band_regions.push(index);
        }
        if spec.curvature_factor.is_some() && domain.dimension == 2 {
            domain.mesh_space()?; // fail construction, not evaluation
        }
        let diagonal = domain.bounds.diagonal().max(1.0e-9);
        Ok(Self {
            domain,
            spec,
            band_regions,
            diagonal,
        })
    }

    pub fn size_at(&self, point: Vec3) -> f64 {
        let mut size = self.spec.background;
        for (band, region_index) in self.spec.bands.iter().zip(&self.band_regions) {
            // owner_sdf is a leaf-primitive exact distance (both sides).
            let wall = self.domain.boundary_regions[*region_index]
                .owner_sdf(&[point])[0]
                .abs();
            let contribution = band.size + self.spec.gradation * (wall - band.distance).max(0.0);
            size = size.min(contribution);
        }
        if let Some(factor) = self.spec.curvature_factor {
            size = size.min(self.curvature_contribution(point, factor));
        }
        size.max(self.spec.min_size)
    }

    pub fn sizes(&self, points: &[Vec3]) -> Vec<f64> {
        points.iter().map(|point| self.size_at(*point)).collect()
    }

    /// Curvature clamp for an interior point near the wall: project it (an
    /// interior start — exact travel), take the curvature at the landed
    /// point, clamp size to `factor x radius`, graded back by the interior
    /// depth. Points outside, far from the wall, non-converged projections,
    /// and flat walls contribute nothing.
    fn curvature_contribution(&self, point: Vec3, factor: f64) -> f64 {
        let depth = self.domain.domain_sdf(&[point])[0];
        if depth >= 0.0 || -depth >= 2.0 * self.spec.background {
            return f64::INFINITY;
        }
        let Ok(projections) = self.domain.project_to_boundary(&[point]) else {
            return f64::INFINITY;
        };
        if !projections[0].converged {
            return f64::INFINITY;
        }
        let Ok(curvatures) = self.domain.curvature(&[projections[0].point]) else {
            return f64::INFINITY;
        };
        let kappa = curvatures[0].abs();
        if kappa <= 1.0 / self.diagonal {
            return f64::INFINITY; // flat wall
        }
        factor / kappa + self.spec.gradation * depth.abs()
    }
}
