//! The public meshing API — ports of `core/meshing/api.py` and
//! `core/model.py::model_from_document`. "Meshing" here means FEA/CFD
//! discretization (never viewport surfaces).
//!
//! `compile_model` is the mesh-time hard gate: invalid role wiring,
//! generator precondition failures, or overlapping Domains are refused
//! before a mesher script receives field callables.

use crate::boundary::{BoundaryRegion, CutSide};
use crate::boundary_ops::{
    boundary_region_mask, cut_volume, evaluate_with_attribution, find_node_by_object_id,
    RELATIVE_SURFACE_TOLERANCE,
};
use crate::differential::{
    batch_normals, curvature_2d, differential_steps, mean_curvature, project_to_zero_set,
    project_to_zero_set_in_plane, Projection,
};
use crate::error::{GeometryError, GeometryResult};
use crate::model::{compile_model, Model};
use crate::roles::{Domain, DomainKind};
use crate::scene::{OperatorKind, SceneDocument, ScenePayload, TagRef};
use crate::sdf::node::{Node, Shape};
use crate::vec3::{vec3, Vec3};
use crate::BoundingBox3D;

/// Derive a `Model` from explicitly declared document Domains. Free
/// top-level construction objects are not Domains by default.
///
/// Domains may be nested (a subtracted solid stays a domain inside the
/// difference). Each domain's region is its world-embedded geometry MINUS
/// every domain marked strictly inside its subtree — an inner domain owns
/// its space, so `sea = difference(box, pipe)` with a fluid `gas` nested in
/// `pipe` meshes as `box − (pipe ∪ gas)` without any manual disjointness
/// work. Regions are derived fresh here on every call; nothing is stored.
pub fn model_from_document(document: &SceneDocument) -> GeometryResult<Model> {
    let mut domains = Vec::new();
    for (object_id, kind) in &document.domain_kinds {
        let Ok(region) = domain_region(document, *object_id) else {
            continue;
        };
        let name = document.object(*object_id)?.name.clone();
        domains.push(Domain::new(name, *kind, region)?);
    }
    Model::new(domains)
}

/// The meshing region of a marked object: its world-embedded geometry MINUS
/// every domain marked strictly inside its subtree.
///
/// Both sides are built from *additive bases* (see `additive_base`) so the
/// derived CSG stays inside the exactness grammar: subtracting an inner
/// domain back out of a difference is a set-level no-op
/// (`(A − B) ∪ B = A ∪ B`), so `sea = box − (ball−gas)` with `ball−gas` and
/// `gas` marked meshes as `box − (ball ∪ gas)` — exact primitives as
/// cutters instead of an inside-exact difference.
fn domain_region(document: &SceneDocument, object_id: u32) -> GeometryResult<Node> {
    let inner: Vec<u32> = document
        .domain_kinds
        .keys()
        .copied()
        .filter(|other| *other != object_id && document.contains(object_id, *other))
        .collect();
    let mut region = document.embedded_node(additive_base(document, object_id, &inner))?;
    if inner.is_empty() {
        return Ok(region);
    }
    let dimension = region.dimension();
    let cutter_ids: std::collections::BTreeSet<u32> = inner
        .iter()
        .map(|id| additive_base(document, *id, &inner))
        .collect();
    let mut cutter: Option<Node> = None;
    for id in cutter_ids {
        let node = document.embedded_node(id)?;
        if node.dimension() != dimension {
            continue;
        }
        cutter = Some(match cutter {
            None => node,
            Some(existing) => {
                let name = existing.name.clone();
                Node::new(name, Shape::union(existing, node)?)
            }
        });
    }
    if let Some(cutter) = cutter {
        let name = document.object(object_id)?.name.clone();
        region = Node::new(name, Shape::difference(region, cutter)?);
    }
    Ok(region)
}

/// Follow "additive base" links from `id`: through transform wrappers
/// (embedding re-applies them) and through Difference operators whose right
/// operand is itself a marked domain — that volume re-enters the region via
/// the cutter union, so the set is unchanged while the expression stays
/// exact.
fn additive_base(document: &SceneDocument, id: u32, marked: &[u32]) -> u32 {
    let mut current = id;
    loop {
        let Ok(object) = document.object(current) else {
            return current;
        };
        current = match &object.payload {
            ScenePayload::Operator {
                kind: OperatorKind::Difference,
                left,
                right,
            } if marked.contains(right) => *left,
            ScenePayload::Translate { child, .. }
            | ScenePayload::Rotate { child, .. }
            | ScenePayload::Scale { child, .. } => *child,
            _ => return current,
        };
    }
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
        points
            .iter()
            .map(|point| self.owner.eval_point(*point))
            .collect()
    }

    /// Combined signed field of the cut chain (negative inside every kept
    /// knife-half); `None` for whole-surface regions.
    pub fn selector_sdf(&self, points: &[Vec3]) -> Option<Vec<f64>> {
        self.selector.as_ref().map(|field| field.eval(points))
    }

    /// Outward normals of the domain boundary at these points (gradient of
    /// the domain root — correct on cut surfaces too).
    pub fn normals(&self, points: &[Vec3]) -> Vec<Vec3> {
        let (normal_step, _, _) = differential_steps(&self.root);
        batch_normals(&self.root, points, normal_step)
    }

    /// Newton projection onto the region's generating surface (the owner
    /// leaf — exact everywhere, so layer seeding may start on either side).
    pub fn project_to_owner(&self, points: &[Vec3]) -> Vec<Projection> {
        let (normal_step, _, zero_band) = differential_steps(&self.owner);
        points
            .iter()
            .map(|point| {
                crate::differential::project_leaf_to_zero_set(
                    &self.owner,
                    *point,
                    normal_step,
                    zero_band,
                )
            })
            .collect()
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

/// Total classification of one point against a domain boundary
/// (`design_docs/meshing_toolkit.md` §5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BoundaryClass {
    /// Band test `|f| <= tol` on the domain region field.
    pub on_boundary: bool,
    /// The controlling leaf (same owner attribution picking uses); lets a
    /// mesher form default patches for untagged boundary.
    pub owner_object_id: u32,
    /// The winning region (index into `boundary_regions`), `None` when no
    /// region claims the point.
    pub region_index: Option<usize>,
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

#[derive(Debug, Clone)]
pub struct MeshableDomainSpace {
    origin: Vec3,
    axis_a: Vec3,
    axis_b: Vec3,
    normal: Vec3,
    bounds: [f64; 4],
    region: Node,
}

impl MeshableDomainSpace {
    pub fn bounds(&self) -> [f64; 4] {
        self.bounds
    }

    pub fn point(&self, a: f64, b: f64) -> Vec3 {
        self.origin + self.axis_a * a + self.axis_b * b
    }

    pub fn coords(&self, point: Vec3) -> [f64; 3] {
        let offset = point - self.origin;
        [
            offset.dot(self.axis_a),
            offset.dot(self.axis_b),
            offset.dot(self.normal),
        ]
    }

    pub fn sdf(&self, a: f64, b: f64) -> f64 {
        self.region.eval_point(self.point(a, b))
    }
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

    pub fn mesh_space(&self) -> GeometryResult<MeshableDomainSpace> {
        if self.dimension != 2 {
            return Err(GeometryError::new(
                "mesh_space is only available for 2D meshable domains",
            ));
        }
        mesh_space_from_node(&self.region)
            .ok_or_else(|| GeometryError::new("2D mesh space requires a placed 2D domain"))
    }

    /// Outward boundary normals at these points (central differences on the
    /// region field).
    pub fn normals(&self, points: &[Vec3]) -> Vec<Vec3> {
        let (normal_step, _, _) = differential_steps(&self.region);
        batch_normals(&self.region, points, normal_step)
    }

    /// Newton projection of **interior** points onto the domain boundary.
    /// Positive-side starts are refused (`converged: false`, no iteration):
    /// only the interior distance is exact. 2D domains project in-plane.
    pub fn project_to_boundary(&self, points: &[Vec3]) -> GeometryResult<Vec<Projection>> {
        let (normal_step, _, zero_band) = differential_steps(&self.region);
        if self.dimension == 2 {
            let space = self.mesh_space()?;
            return Ok(points
                .iter()
                .map(|point| {
                    project_to_zero_set_in_plane(
                        &self.region,
                        *point,
                        space.origin,
                        space.normal,
                        normal_step,
                        zero_band,
                    )
                })
                .collect());
        }
        Ok(points
            .iter()
            .map(|point| project_to_zero_set(&self.region, *point, normal_step, zero_band))
            .collect())
    }

    /// Curvature of the boundary at near-wall points reached from the
    /// interior (project first): mean curvature `H` for 3D domains,
    /// in-plane `kappa` for 2D. At creases the stencil returns O(1/step) —
    /// meaningful as "refine here", not as a curvature.
    pub fn curvature(&self, points: &[Vec3]) -> GeometryResult<Vec<f64>> {
        let (_, curvature_step, _) = differential_steps(&self.region);
        if self.dimension == 2 {
            let space = self.mesh_space()?;
            return Ok(points
                .iter()
                .map(|point| {
                    curvature_2d(&self.region, *point, space.axis_a, space.axis_b, curvature_step)
                })
                .collect());
        }
        Ok(points
            .iter()
            .map(|point| mean_curvature(&self.region, *point, curvature_step))
            .collect())
    }

    /// The classification default band: `1e-3 x` bounds diagonal — the same
    /// scale-relative constant the viewport classifier uses. Wide enough to
    /// accept unprojected samples (a straight face's centroid on a curved
    /// wall, offset by the sagitta); callers classifying projected vertices
    /// pass the tight band explicitly instead.
    pub fn boundary_tolerance(&self) -> f64 {
        RELATIVE_SURFACE_TOLERANCE * self.bounds.diagonal()
    }

    /// All regions whose classifier accepts each point (multi-label view).
    /// `tolerance: None` means each region's own default, exactly as the
    /// viewport highlights — what is highlighted is what the mesher gets.
    pub fn regions_containing(
        &self,
        points: &[Vec3],
        tolerance: Option<f64>,
    ) -> GeometryResult<Vec<Vec<usize>>> {
        let mut matches = vec![Vec::new(); points.len()];
        for (index, entry) in self.boundary_regions.iter().enumerate() {
            let mask = boundary_region_mask(&entry.root, &entry.region, points, tolerance)?;
            for (hits, hit) in matches.iter_mut().zip(mask) {
                if hit {
                    hits.push(index);
                }
            }
        }
        Ok(matches)
    }

    /// Total boundary classification: for each point, whether it is on the
    /// domain boundary, which leaf owns it, and which region (if any) wins.
    ///
    /// Precedence among matching regions is the lexicographic maximum of
    /// `(cuts, patch_scoped, creation index)`: more knife cuts is more
    /// specific, a patch-/direction-scoped region beats a whole-surface one
    /// at equal cuts, and the final tie goes to the later-created region so
    /// newer refinements override older broad tags.
    pub fn classify_boundary(
        &self,
        points: &[Vec3],
        tolerance: Option<f64>,
    ) -> GeometryResult<Vec<BoundaryClass>> {
        let band = tolerance.unwrap_or_else(|| self.boundary_tolerance());
        let matches = self.regions_containing(points, tolerance)?;
        Ok(points
            .iter()
            .zip(matches)
            .map(|(point, hits)| {
                let (value, owner_object_id) = evaluate_with_attribution(&self.region, *point);
                let on_boundary = value.abs() <= band;
                let region_index = if on_boundary {
                    hits.into_iter().max_by_key(|index| {
                        let region = &self.boundary_regions[*index].region;
                        let patch_scoped =
                            region.patch_id.is_some() || region.outside_direction.is_some();
                        (region.cuts.len(), patch_scoped, *index)
                    })
                } else {
                    None
                };
                BoundaryClass {
                    on_boundary,
                    owner_object_id,
                    region_index,
                }
            })
            .collect())
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

fn mesh_space_from_node(node: &Node) -> Option<MeshableDomainSpace> {
    match &node.shape {
        Shape::PlacedSdf2D(placed) => {
            let normal = placed.normal();
            let (a_min, a_max, b_min, b_max) = placed.profile.bounds();
            Some(MeshableDomainSpace {
                origin: placed.origin,
                axis_a: placed.axis_u,
                axis_b: placed.axis_v,
                normal,
                bounds: [a_min, a_max, b_min, b_max],
                region: node.clone(),
            })
        }
        Shape::Translate { child, offset } => {
            let mut space = mesh_space_from_node(child)?;
            space.origin += *offset;
            space.region = node.clone();
            Some(space)
        }
        Shape::Scale { child, factor } => {
            let mut space = mesh_space_from_node(child)?;
            space.origin = space.origin * *factor;
            space.bounds = [
                space.bounds[0] * *factor,
                space.bounds[1] * *factor,
                space.bounds[2] * *factor,
                space.bounds[3] * *factor,
            ];
            space.region = node.clone();
            Some(space)
        }
        Shape::Rotate {
            child,
            axis,
            angle_degrees,
        } => {
            let mut space = mesh_space_from_node(child)?;
            space.origin = axis.rotate(space.origin, *angle_degrees);
            space.axis_a = axis.rotate(space.axis_a, *angle_degrees);
            space.axis_b = axis.rotate(space.axis_b, *angle_degrees);
            space.normal = space.axis_a.cross(space.axis_b);
            space.normal = space.normal / space.normal.length().max(1.0e-12);
            space.region = node.clone();
            Some(space)
        }
        Shape::Union(operands)
        | Shape::Intersection(operands)
        | Shape::Difference(operands)
        | Shape::Xor(operands) => {
            let mut space = mesh_space_from_node(&operands.left)?;
            if let Ok(bounds) = node.bounding_box() {
                space.bounds = projected_bounds(&bounds, space.origin, space.axis_a, space.axis_b);
            }
            space.region = node.clone();
            Some(space)
        }
        _ => None,
    }
}

fn projected_bounds(bounds: &BoundingBox3D, origin: Vec3, axis_a: Vec3, axis_b: Vec3) -> [f64; 4] {
    let mut a_min = f64::INFINITY;
    let mut a_max = f64::NEG_INFINITY;
    let mut b_min = f64::INFINITY;
    let mut b_max = f64::NEG_INFINITY;
    for x in [bounds.x_min, bounds.x_max] {
        for y in [bounds.y_min, bounds.y_max] {
            for z in [bounds.z_min, bounds.z_max] {
                let offset = vec3(x, y, z) - origin;
                let a = offset.dot(axis_a);
                let b = offset.dot(axis_b);
                a_min = a_min.min(a);
                a_max = a_max.max(a);
                b_min = b_min.min(b);
                b_max = b_max.max(b);
            }
        }
    }
    [a_min, a_max, b_min, b_max]
}

/// The exact shared wall between two directly nested marked domains
/// (`design_docs/meshing_toolkit.md` §6). The surface node is the inner
/// domain's embedded additive base — the *same* node the outer region was
/// cut with, so the interface IS the cut surface: one known exact field, no
/// numerical differencing (spec §8).
#[derive(Debug, Clone)]
pub struct MeshableInterface {
    /// The outer domain's name.
    pub domain_a: String,
    /// The inner domain's name.
    pub domain_b: String,
    /// The inner additive-base object generating the shared surface.
    pub owner_object_id: u32,
    surface: Node,
    side_a: Node,
    side_b: Node,
    surface_tolerance: f64,
    side_a_tolerance: f64,
    side_b_tolerance: f64,
}

impl MeshableInterface {
    /// The shared surface's exact field (the generating node's own SDF).
    pub fn surface_sdf(&self, points: &[Vec3]) -> Vec<f64> {
        points
            .iter()
            .map(|point| self.surface.eval_point(*point))
            .collect()
    }

    pub fn surface_node(&self) -> &Node {
        &self.surface
    }

    /// Membership of the shared wall: on the generating surface AND on both
    /// domains' boundaries (band tests) — this clips the generating node's
    /// zero set to the actual contact area.
    pub fn contains(&self, points: &[Vec3]) -> Vec<bool> {
        points
            .iter()
            .map(|point| {
                self.surface.eval_point(*point).abs() <= self.surface_tolerance
                    && self.side_a.eval_point(*point).abs() <= self.side_a_tolerance
                    && self.side_b.eval_point(*point).abs() <= self.side_b_tolerance
            })
            .collect()
    }

    /// Newton projection onto the shared wall through an **interior** field
    /// (interior-exactness contract): a start inside the inner domain
    /// projects through the inner region node, a start inside the outer
    /// domain through the outer region node, a start in neither is refused.
    /// Seed near the intended wall (the inner interior band `-h < f < 0`);
    /// `contains` verifies the landing.
    pub fn project(&self, points: &[Vec3]) -> Vec<Projection> {
        points
            .iter()
            .map(|point| {
                let side = if self.side_b.eval_point(*point) <= 0.0 {
                    Some(&self.side_b)
                } else if self.side_a.eval_point(*point) <= 0.0 {
                    Some(&self.side_a)
                } else {
                    None
                };
                match side {
                    Some(field) => {
                        let (normal_step, _, zero_band) = differential_steps(field);
                        project_to_zero_set(field, *point, normal_step, zero_band)
                    }
                    None => Projection {
                        point: *point,
                        residual: self.surface.eval_point(*point),
                        distance_moved: 0.0,
                        converged: false,
                    },
                }
            })
            .collect()
    }
}

/// Name- (or unique-kind-) keyed collection of meshable domains.
#[derive(Debug, Clone, Default)]
pub struct MeshableDomains {
    items: Vec<MeshableDomain>,
    interfaces: Vec<MeshableInterface>,
}

impl MeshableDomains {
    pub fn new(items: Vec<MeshableDomain>, interfaces: Vec<MeshableInterface>) -> Self {
        Self { items, interfaces }
    }

    /// All domain interfaces, for iteration; lookup goes through
    /// `interface_between` / `interfaces_of`.
    pub fn interfaces(&self) -> &[MeshableInterface] {
        &self.interfaces
    }

    /// Every interface one domain participates in.
    pub fn interfaces_of(&self, name: &str) -> Vec<&MeshableInterface> {
        self.interfaces
            .iter()
            .filter(|interface| interface.domain_a == name || interface.domain_b == name)
            .collect()
    }

    /// The interface between two domains, in either order.
    pub fn interface_between(&self, a: &str, b: &str) -> GeometryResult<&MeshableInterface> {
        self.interfaces
            .iter()
            .find(|interface| {
                (interface.domain_a == a && interface.domain_b == b)
                    || (interface.domain_a == b && interface.domain_b == a)
            })
            .ok_or_else(|| {
                let available: Vec<String> = self
                    .interfaces
                    .iter()
                    .map(|interface| format!("{}<->{}", interface.domain_a, interface.domain_b))
                    .collect();
                GeometryError::new(format!(
                    "unknown interface {a}<->{b}; available: {}",
                    if available.is_empty() {
                        "none".to_string()
                    } else {
                        available.join(", ")
                    }
                ))
            })
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
                let names: Vec<&str> = matches.iter().map(|domain| domain.name.as_str()).collect();
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
        let mut keys: Vec<String> = self
            .items
            .iter()
            .map(|domain| domain.name.clone())
            .collect();
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

/// Boundary entries of one non-fluid marked domain: its regions are
/// attached by their `domain_root` (the fluid domain keeps its tag-list
/// path below, which also carries `TagRef::Node` tag objects).
fn domain_boundary_entries(
    document: &SceneDocument,
    root: &Node,
    domain_root_id: u32,
) -> (Vec<MeshableBoundaryTag>, Vec<MeshableBoundaryRegion>) {
    let mut tags = Vec::new();
    let mut regions = Vec::new();
    for region in &document.boundary_regions {
        if document.region_domain_root(region) != Some(domain_root_id) {
            continue;
        }
        let Some(owner) = find_node_by_object_id(root, region.owner_object_id) else {
            continue;
        };
        let selector = cut_chain_field(root, region).ok().flatten();
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
    (tags, regions)
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
                let selector = cut_chain_field(root, region).ok().flatten();
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
pub fn meshable_domains_from_document(document: &SceneDocument) -> GeometryResult<MeshableDomains> {
    let model = model_from_document(document)?;
    compile_model(&model, 32)?;
    let fluid_root_id = document.fluid_domain.as_ref().map(|fluid| fluid.root);
    let mut items = Vec::new();
    for domain in &model.domains {
        let marked_id = document
            .domain_kinds
            .keys()
            .find(|id| {
                document
                    .object(**id)
                    .map(|object| object.name == domain.name)
                    .unwrap_or(false)
            })
            .copied();
        let (boundary_tags, boundary_regions) = match marked_id {
            Some(root_id) if fluid_root_id == Some(root_id) => {
                fluid_boundary_entries(document, &domain.region)
            }
            Some(root_id) => domain_boundary_entries(document, &domain.region, root_id),
            None => (Vec::new(), Vec::new()),
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
    let interfaces = derive_interfaces(document, &items);
    Ok(MeshableDomains::new(items, interfaces))
}

/// Interfaces between directly nested marked domains
/// (`design_docs/meshing_toolkit.md` §6): each marked domain pairs with its
/// nearest marked strict ancestor, and the shared surface is the inner
/// domain's embedded additive base — the identical node `domain_region`
/// subtracts from the outer region. Pairs of unequal dimension are skipped.
fn derive_interfaces(document: &SceneDocument, items: &[MeshableDomain]) -> Vec<MeshableInterface> {
    let marked: Vec<u32> = document.domain_kinds.keys().copied().collect();
    let domain_by_id = |id: u32| -> Option<&MeshableDomain> {
        let name = &document.object(id).ok()?.name;
        items.iter().find(|domain| &domain.name == name)
    };
    let mut interfaces = Vec::new();
    for inner_id in &marked {
        // Nearest marked strict ancestor: contains inner, and no other
        // marked domain sits between them.
        let Some(outer_id) = marked
            .iter()
            .filter(|outer| **outer != *inner_id && document.contains(**outer, *inner_id))
            .copied()
            .find(|outer| {
                !marked.iter().any(|middle| {
                    *middle != *outer
                        && *middle != *inner_id
                        && document.contains(*outer, *middle)
                        && document.contains(*middle, *inner_id)
                })
            })
        else {
            continue;
        };
        let (Some(outer), Some(inner)) = (domain_by_id(outer_id), domain_by_id(*inner_id)) else {
            continue;
        };
        if outer.dimension != inner.dimension {
            continue;
        }
        let inner_of_outer: Vec<u32> = marked
            .iter()
            .copied()
            .filter(|other| *other != outer_id && document.contains(outer_id, *other))
            .collect();
        let base_id = additive_base(document, *inner_id, &inner_of_outer);
        let Ok(surface) = document.embedded_node(base_id) else {
            continue;
        };
        let surface_tolerance = RELATIVE_SURFACE_TOLERANCE
            * surface
                .bounding_box()
                .map(|bounds| bounds.diagonal())
                .unwrap_or_else(|_| outer.bounds.diagonal());
        interfaces.push(MeshableInterface {
            domain_a: outer.name.clone(),
            domain_b: inner.name.clone(),
            owner_object_id: base_id,
            surface,
            side_a: outer.region.clone(),
            side_b: inner.region.clone(),
            surface_tolerance,
            side_a_tolerance: outer.boundary_tolerance(),
            side_b_tolerance: inner.boundary_tolerance(),
        });
    }
    interfaces
}

/// Load meshable domains from a saved casoCAD `scene.json` string.
pub fn load_meshable_domains_from_str(text: &str) -> GeometryResult<MeshableDomains> {
    let document = crate::serialization::load_scene_from_str(text)?;
    meshable_domains_from_document(&document)
}
