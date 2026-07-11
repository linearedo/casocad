//! The public meshing API — ports of `core/meshing/api.py` and
//! `core/model.py::model_from_document`. "Meshing" here means FEA/CFD
//! discretization (never viewport surfaces).
//!
//! `compile_model` is the mesh-time hard gate: invalid role wiring,
//! generator precondition failures, or overlapping Domains are refused
//! before a mesher script receives field callables.

use crate::boundary::{BoundaryRegion, CutSide};
use crate::boundary_ops::{
    boundary_region_mask, cut_volume, find_node_by_object_id, surface_selector_volume,
};
use crate::error::{GeometryError, GeometryResult};
use crate::model::{compile_model, Model};
use crate::roles::{Domain, DomainKind};
use crate::scene::{SceneDocument, TagRef};
use crate::sdf::node::Node;
use crate::vec3::Vec3;
use crate::BoundingBox3D;

/// Derive a `Model` from explicitly declared document Domains. Free
/// top-level construction objects are not Domains by default.
pub fn model_from_document(document: &SceneDocument) -> GeometryResult<Model> {
    let mut domains = Vec::new();
    for (object_id, kind) in &document.domain_kinds {
        let Ok(region) = document.build_node(*object_id) else {
            continue;
        };
        let name = document.object(*object_id)?.name.clone();
        domains.push(Domain::new(name, *kind, region)?);
    }
    Model::new(domains)
}

/// The signed classification field of a region's knife chain: negative
/// exactly where every kept knife-half is satisfied (`_cut_chain_field`).
#[derive(Debug, Clone)]
pub struct SelectorField {
    /// (sign, volume): +1 keeps the inside of the volume, −1 the outside.
    parts: Vec<(f64, Node)>,
}

impl SelectorField {
    pub fn eval(&self, points: &[Vec3]) -> Vec<f64> {
        points
            .iter()
            .map(|point| {
                self.parts
                    .iter()
                    .map(|(sign, node)| sign * node.eval_point(*point))
                    .fold(f64::NEG_INFINITY, f64::max)
            })
            .collect()
    }
}

/// One boundary region, callable from mesher scripts (v2 §6): exact
/// membership (`contains` — the same classifier the viewport uses), the
/// owner's exact field, and the combined knife field.
#[derive(Debug, Clone)]
pub struct MeshableBoundaryRegion {
    pub name: String,
    pub tag: Option<String>,
    pub owner_object_id: u32,
    root: Node,
    region: BoundaryRegion,
    owner: Node,
    selector: Option<SelectorField>,
}

impl MeshableBoundaryRegion {
    /// Exact membership of world points — what is highlighted is what you get.
    pub fn contains(&self, points: &[Vec3]) -> GeometryResult<Vec<bool>> {
        boundary_region_mask(&self.root, &self.region, points, None)
    }

    /// The exact field of the region's generating surface (y+ layers,
    /// grading, refinement bands).
    pub fn owner_sdf(&self, points: &[Vec3]) -> Vec<f64> {
        points.iter().map(|point| self.owner.eval_point(*point)).collect()
    }

    /// Combined signed field of the cut chain (negative inside every kept
    /// knife-half); `None` for whole-surface regions.
    pub fn selector_sdf(&self, points: &[Vec3]) -> Option<Vec<f64>> {
        self.selector.as_ref().map(|field| field.eval(points))
    }
}

/// A boundary tag exposed to mesher scripts: name + signed field.
#[derive(Debug, Clone)]
pub struct MeshableBoundaryTag {
    pub name: String,
    field: SelectorField,
}

impl MeshableBoundaryTag {
    pub fn eval(&self, points: &[Vec3]) -> Vec<f64> {
        self.field.eval(points)
    }
}

#[derive(Debug, Clone)]
pub struct MeshableDomain {
    pub name: String,
    pub kind: DomainKind,
    pub dimension: u8,
    pub bounds: BoundingBox3D,
    region: Node,
    pub boundary_tags: Vec<MeshableBoundaryTag>,
    pub boundary_regions: Vec<MeshableBoundaryRegion>,
}

impl MeshableDomain {
    pub fn domain_sdf(&self, points: &[Vec3]) -> Vec<f64> {
        points
            .iter()
            .map(|point| self.region.eval_point(*point))
            .collect()
    }

    pub fn region_node(&self) -> &Node {
        &self.region
    }

    pub fn region_by_name(&self, name: &str) -> GeometryResult<&MeshableBoundaryRegion> {
        self.boundary_regions
            .iter()
            .find(|region| region.name == name)
            .ok_or_else(|| {
                let available: Vec<&str> = self
                    .boundary_regions
                    .iter()
                    .map(|region| region.name.as_str())
                    .collect();
                GeometryError::new(format!(
                    "unknown boundary region {name:?}; available: {}",
                    available.join(", ")
                ))
            })
    }
}

/// Name- (or unique-kind-) keyed collection of meshable domains.
#[derive(Debug, Clone, Default)]
pub struct MeshableDomains {
    items: Vec<MeshableDomain>,
}

impl MeshableDomains {
    pub fn new(items: Vec<MeshableDomain>) -> Self {
        Self { items }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = &MeshableDomain> {
        self.items.iter()
    }

    pub fn by_kind(&self, kind: DomainKind) -> Vec<&MeshableDomain> {
        self.items
            .iter()
            .filter(|domain| domain.kind == kind)
            .collect()
    }

    /// Lookup by name, or by domain kind when exactly one domain has it.
    pub fn get(&self, key: &str) -> GeometryResult<&MeshableDomain> {
        if let Some(domain) = self.items.iter().find(|domain| domain.name == key) {
            return Ok(domain);
        }
        if let Ok(kind) = DomainKind::parse(key) {
            let matches = self.by_kind(kind);
            if matches.len() == 1 {
                return Ok(matches[0]);
            }
            if matches.len() > 1 {
                let names: Vec<&str> =
                    matches.iter().map(|domain| domain.name.as_str()).collect();
                return Err(GeometryError::new(format!(
                    "domain kind {key:?} is ambiguous: {}",
                    names.join(", ")
                )));
            }
        }
        Err(GeometryError::new(format!(
            "unknown meshable domain {key:?}; available: {}",
            self.keys().join(", ")
        )))
    }

    /// Domain names plus the kinds that are unique to one domain.
    pub fn keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = self.items.iter().map(|domain| domain.name.clone()).collect();
        for kind in [DomainKind::Fluid, DomainKind::Solid] {
            if self.by_kind(kind).len() == 1 {
                keys.push(kind.as_str().to_string());
            }
        }
        keys
    }
}

fn cut_chain_field(root: &Node, region: &BoundaryRegion) -> GeometryResult<Option<SelectorField>> {
    if region.cuts.is_empty() {
        return Ok(None);
    }
    let mut parts = Vec::new();
    for cut in &region.cuts {
        let volume = cut_volume(root, cut)?;
        let sign = match cut.side {
            CutSide::Inside => 1.0,
            CutSide::Outside => -1.0,
        };
        parts.push((sign, volume));
    }
    Ok(Some(SelectorField { parts }))
}

/// Legacy single-selector field (`_boundary_region_callable`).
fn legacy_selector_field(
    document: &SceneDocument,
    root: &Node,
    region: &BoundaryRegion,
) -> Option<SelectorField> {
    let selector_id = region.selector_id.as_deref()?.strip_prefix("selector:")?;
    let selector_object_id: u32 = selector_id.parse().ok()?;
    let selector = document.build_node(selector_object_id).ok()?;
    let volume = surface_selector_volume(root, &selector).ok()??;
    let sign = if region.selector_side == CutSide::Outside {
        -1.0
    } else {
        1.0
    };
    Some(SelectorField {
        parts: vec![(sign, volume)],
    })
}

fn fluid_boundary_entries(
    document: &SceneDocument,
    root: &Node,
) -> (Vec<MeshableBoundaryTag>, Vec<MeshableBoundaryRegion>) {
    let mut tags = Vec::new();
    let mut regions = Vec::new();
    let Some(fluid) = &document.fluid_domain else {
        return (tags, regions);
    };
    for tag_ref in &fluid.tags {
        match tag_ref {
            TagRef::Region(region_id) => {
                let Some(region) = document
                    .boundary_regions
                    .iter()
                    .find(|region| region.object_id == *region_id)
                else {
                    continue;
                };
                let Some(owner) = find_node_by_object_id(root, region.owner_object_id) else {
                    continue;
                };
                let selector = cut_chain_field(root, region)
                    .ok()
                    .flatten()
                    .or_else(|| legacy_selector_field(document, root, region));
                if let Some(field) = &selector {
                    tags.push(MeshableBoundaryTag {
                        name: region.name.clone(),
                        field: field.clone(),
                    });
                }
                regions.push(MeshableBoundaryRegion {
                    name: region.name.clone(),
                    tag: region.tag.clone(),
                    owner_object_id: region.owner_object_id,
                    root: root.clone(),
                    region: region.clone(),
                    owner: owner.clone(),
                    selector,
                });
            }
            TagRef::Node(node_id) => {
                if let Ok(node) = document.build_node(*node_id) {
                    let name = document
                        .object(*node_id)
                        .map(|object| object.name.clone())
                        .unwrap_or_default();
                    tags.push(MeshableBoundaryTag {
                        name,
                        field: SelectorField {
                            parts: vec![(1.0, node)],
                        },
                    });
                }
            }
        }
    }
    (tags, regions)
}

/// Expose an exact-SDF document through the public meshing API
/// (`load_meshable_domains`, from an in-memory document).
pub fn meshable_domains_from_document(
    document: &SceneDocument,
) -> GeometryResult<MeshableDomains> {
    let model = model_from_document(document)?;
    compile_model(&model, 32)?;
    let fluid_root_id = document.fluid_domain.as_ref().map(|fluid| fluid.root);
    let mut items = Vec::new();
    for domain in &model.domains {
        let is_fluid_root = fluid_root_id.is_some_and(|root_id| {
            document
                .object(root_id)
                .map(|object| object.name == domain.name)
                .unwrap_or(false)
        });
        let (boundary_tags, boundary_regions) = if is_fluid_root {
            fluid_boundary_entries(document, &domain.region)
        } else {
            (Vec::new(), Vec::new())
        };
        items.push(MeshableDomain {
            name: domain.name.clone(),
            kind: domain.kind,
            dimension: domain.region.dimension(),
            bounds: domain.region.bounding_box()?,
            region: domain.region.clone(),
            boundary_tags,
            boundary_regions,
        });
    }
    Ok(MeshableDomains::new(items))
}

/// Load meshable domains from a saved casoCAD `scene.json` string.
pub fn load_meshable_domains_from_str(text: &str) -> GeometryResult<MeshableDomains> {
    let document = crate::serialization::load_scene_from_str(text)?;
    meshable_domains_from_document(&document)
}
