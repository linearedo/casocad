//! The editable scene document, ported from `core/scene.py`.
//!
//! The Python document is a DAG of shared node objects addressed by opaque
//! handles. The Rust port stores the graph explicitly: every object lives in
//! an id-keyed map and composite payloads reference children by `ObjectId`.
//! The stable `object_id` doubles as the editing handle. Kernel `Node` trees
//! (the authoritative evaluation form) are built on demand via [`SceneDocument::build_node`].
//! Undo snapshots are plain `Clone`s of the document value.

use std::collections::BTreeMap;

use crate::boundary::{BoundaryCut, BoundaryRegion, CutSide};
use crate::error::{GeometryError, GeometryResult};
use crate::frame::{Frame, IDENTITY_FRAME};
use crate::preconditions::revolve_violations;
use crate::roles::DomainKind;
use crate::sdf::curtain::NormalCurtain;
use crate::sdf::node::{Node, RotationAxis, Shape};
use crate::sdf::placed::{PlacedPolyline1D, PlacedSdf1D, PlacedSdf2D};
use crate::sdf::primitives_1d::{BooleanOp1D, Profile1D};
use crate::sdf::primitives_2d::Profile2D;
use crate::sdf::primitives_3d::{
    Box3, BoxFrame, CappedCone, Cone, Cylinder, Pyramid, Sphere, Torus,
};
use crate::sdf::solid_from_2d::{Extrude, Revolve, RevolveAxis};
use crate::sdf::tubes::{CapStyle, PolylineTube, QuadraticBezierTube};
use crate::vec3::{vec3, Vec3};

pub type ObjectId = u32;

pub const MAX_OBJECT_ID: ObjectId = 65_535;
pub const INTERNAL_BOUNDARY_SELECTOR_PREFIX: &str = "__boundary_selector_";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorKind {
    Union,
    Intersection,
    Difference,
    Xor,
}

impl OperatorKind {
    pub fn parse(operation: &str) -> GeometryResult<Self> {
        match operation {
            "union" => Ok(Self::Union),
            "intersection" => Ok(Self::Intersection),
            "difference" => Ok(Self::Difference),
            "xor" => Ok(Self::Xor),
            other => Err(GeometryError::new(format!("unknown SDF operation: {other}"))),
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

    fn as_boolean_op_1d(&self) -> BooleanOp1D {
        match self {
            Self::Union => BooleanOp1D::Union,
            Self::Intersection => BooleanOp1D::Intersection,
            Self::Difference => BooleanOp1D::Difference,
            Self::Xor => BooleanOp1D::Xor,
        }
    }
}

/// Payload of one scene object. Leaf shapes reuse the kernel structs;
/// composites reference their children by `ObjectId`.
#[derive(Debug, Clone, PartialEq)]
pub enum ScenePayload {
    Sphere(Sphere),
    Box3(Box3),
    Cylinder(Cylinder),
    Cone(Cone),
    CappedCone(CappedCone),
    Pyramid(Pyramid),
    BoxFrame(BoxFrame),
    Torus(Torus),
    PolylineTube(PolylineTube),
    QuadraticBezierTube(QuadraticBezierTube),
    NormalCurtain(NormalCurtain),
    Placed2D {
        profile: Profile2D,
        origin: Vec3,
        axis_u: Vec3,
        axis_v: Vec3,
        sources: Vec<ObjectId>,
    },
    PlacedPolyline1D {
        profile: Profile2D,
        origin: Vec3,
        axis_u: Vec3,
        axis_v: Vec3,
    },
    Placed1D {
        profile: Profile1D,
        origin: Vec3,
        axis_u: Vec3,
        sources: Vec<ObjectId>,
    },
    Operator {
        kind: OperatorKind,
        left: ObjectId,
        right: ObjectId,
    },
    Translate {
        child: ObjectId,
        offset: Vec3,
    },
    Rotate {
        child: ObjectId,
        axis: RotationAxis,
        angle_degrees: f64,
    },
    Scale {
        child: ObjectId,
        factor: f64,
    },
    Extrude {
        section: ObjectId,
        height: f64,
        center_offset: f64,
    },
    Revolve {
        section: ObjectId,
        axis: RevolveAxis,
        axis_origin: Option<Vec3>,
        axis_direction: Option<Vec3>,
        radial_direction: Option<Vec3>,
        angle_degrees: f64,
    },
}

impl ScenePayload {
    /// Child object ids, in the order the Python `children()` walks them.
    pub fn children(&self) -> Vec<ObjectId> {
        match self {
            Self::Operator { left, right, .. } => vec![*left, *right],
            Self::Translate { child, .. }
            | Self::Rotate { child, .. }
            | Self::Scale { child, .. } => vec![*child],
            Self::Extrude { section, .. } | Self::Revolve { section, .. } => vec![*section],
            Self::Placed2D { sources, .. } | Self::Placed1D { sources, .. } => sources.clone(),
            _ => Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct SceneObject {
    pub name: String,
    pub id: ObjectId,
    pub payload: ScenePayload,
}

/// A tag on the fluid domain: either a BoundaryRegion (by object_id) or a
/// placed 1D/2D scene node (by object id).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagRef {
    Region(ObjectId),
    Node(ObjectId),
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct FluidDomainRecord {
    pub root: ObjectId,
    pub tags: Vec<TagRef>,
    pub selectors: Vec<ObjectId>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SceneDocument {
    /// Every scene node by object id (includes nested nodes).
    pub objects: BTreeMap<ObjectId, SceneObject>,
    /// Top-level object ids in scene order (the `root_objects` of the file).
    pub roots: Vec<ObjectId>,
    pub boundary_regions: Vec<BoundaryRegion>,
    pub domain_kinds: BTreeMap<ObjectId, DomainKind>,
    pub fluid_domain: Option<FluidDomainRecord>,
    pub version: u64,
    next_object_id: ObjectId,
}

impl SceneDocument {
    pub fn new() -> Self {
        Self {
            next_object_id: 1,
            ..Default::default()
        }
    }

    /// The built-in von Kármán demo scene (Python `SceneDocument.default`).
    pub fn default_scene() -> GeometryResult<Self> {
        let mut document = Self::new();
        let outer = document.insert_object(
            "flow_volume",
            ScenePayload::Box3(Box3::new(Vec3::ZERO, vec3(1.6, 0.7, 0.45), IDENTITY_FRAME)?),
        )?;
        let obstacle = document.insert_object(
            "cylinder_obstacle",
            ScenePayload::Cylinder(Cylinder::new(Vec3::ZERO, 0.24, 0.55, IDENTITY_FRAME)?),
        )?;
        let root = document.insert_object(
            "von_karman_fluid",
            ScenePayload::Operator {
                kind: OperatorKind::Difference,
                left: outer,
                right: obstacle,
            },
        )?;
        document.roots = vec![root];
        let inlet_id = document.allocate_object_id()?;
        let mut inlet = BoundaryRegion::new("inlet", inlet_id, outer);
        inlet.outside_direction = Some(0);
        let outlet_id = document.allocate_object_id()?;
        let mut outlet = BoundaryRegion::new("outlet", outlet_id, outer);
        outlet.outside_direction = Some(1);
        document.boundary_regions = vec![inlet, outlet];
        document.fluid_domain = Some(FluidDomainRecord {
            root,
            tags: vec![TagRef::Region(inlet_id), TagRef::Region(outlet_id)],
            selectors: Vec::new(),
        });
        document.domain_kinds.insert(root, DomainKind::Fluid);
        Ok(document)
    }

    pub fn mark_changed(&mut self) {
        self.version += 1;
    }

    pub fn allocate_object_id(&mut self) -> GeometryResult<ObjectId> {
        let object_id = self.next_object_id;
        self.next_object_id += 1;
        if object_id > MAX_OBJECT_ID {
            return Err(GeometryError::new("maximum SDF object count exceeded"));
        }
        Ok(object_id)
    }

    /// Reserve ids so future allocations continue after `max_seen`.
    pub fn bump_next_object_id(&mut self, max_seen: ObjectId) {
        if self.next_object_id <= max_seen {
            self.next_object_id = max_seen + 1;
        }
    }

    /// Insert an object with a fresh id (not as a root).
    pub fn insert_object(
        &mut self,
        name: impl Into<String>,
        payload: ScenePayload,
    ) -> GeometryResult<ObjectId> {
        let id = self.allocate_object_id()?;
        self.objects.insert(
            id,
            SceneObject {
                name: name.into(),
                id,
                payload,
            },
        );
        Ok(id)
    }

    /// Insert an object with an explicit id (used by the loader).
    pub fn insert_object_with_id(
        &mut self,
        name: impl Into<String>,
        id: ObjectId,
        payload: ScenePayload,
    ) -> GeometryResult<ObjectId> {
        if id == 0 || id > MAX_OBJECT_ID {
            return Err(GeometryError::new("object_id must be in the range 1..65535"));
        }
        if self.objects.contains_key(&id) {
            return Err(GeometryError::new(format!("duplicate object_id {id}")));
        }
        self.objects.insert(
            id,
            SceneObject {
                name: name.into(),
                id,
                payload,
            },
        );
        self.bump_next_object_id(id);
        Ok(id)
    }

    pub fn object(&self, id: ObjectId) -> GeometryResult<&SceneObject> {
        self.objects
            .get(&id)
            .ok_or_else(|| GeometryError::new(format!("unknown scene object {id}")))
    }

    pub fn object_mut(&mut self, id: ObjectId) -> GeometryResult<&mut SceneObject> {
        self.objects
            .get_mut(&id)
            .ok_or_else(|| GeometryError::new(format!("unknown scene object {id}")))
    }

    pub fn is_internal_scene_node(name: &str) -> bool {
        name.starts_with(INTERNAL_BOUNDARY_SELECTOR_PREFIX)
    }

    /// (id, parent id) pairs: roots depth-first with DAG dedupe, then regions
    /// are reported by the separate `boundary_regions` list.
    pub fn walk(&self) -> Vec<(ObjectId, Option<ObjectId>)> {
        let mut seen = Vec::new();
        let mut order = Vec::new();
        for root in &self.roots {
            self.walk_node(*root, None, &mut seen, &mut order);
        }
        order
    }

    fn walk_node(
        &self,
        id: ObjectId,
        parent: Option<ObjectId>,
        seen: &mut Vec<ObjectId>,
        order: &mut Vec<(ObjectId, Option<ObjectId>)>,
    ) {
        if seen.contains(&id) {
            return;
        }
        seen.push(id);
        order.push((id, parent));
        if let Some(object) = self.objects.get(&id) {
            for child in object.payload.children() {
                self.walk_node(child, Some(id), seen, order);
            }
        }
    }

    /// All node ids reachable from the roots (the "live" SDF nodes).
    pub fn live_ids(&self) -> Vec<ObjectId> {
        self.walk().into_iter().map(|(id, _parent)| id).collect()
    }

    pub fn default_name(&self, kind: &str) -> String {
        let used: Vec<&str> = self
            .live_ids()
            .into_iter()
            .filter_map(|id| self.objects.get(&id).map(|object| object.name.as_str()))
            .chain(self.boundary_regions.iter().map(|region| region.name.as_str()))
            .collect();
        let mut index = 1usize;
        loop {
            let candidate = format!("{kind}_{index}");
            if !used.contains(&candidate.as_str()) {
                return candidate;
            }
            index += 1;
        }
    }

    /// Build the authoritative kernel `Node` tree for one object, resolving
    /// references (shared subgraphs are duplicated into the tree, keeping
    /// their object ids).
    pub fn build_node(&self, id: ObjectId) -> GeometryResult<Node> {
        self.build_node_guarded(id, &mut Vec::new())
    }

    fn build_node_guarded(&self, id: ObjectId, visiting: &mut Vec<ObjectId>) -> GeometryResult<Node> {
        if visiting.contains(&id) {
            return Err(GeometryError::new(format!(
                "scene graph cycle through object {id}"
            )));
        }
        visiting.push(id);
        let object = self.object(id)?;
        let shape = match &object.payload {
            ScenePayload::Sphere(shape) => Shape::Sphere(shape.clone()),
            ScenePayload::Box3(shape) => Shape::Box3(shape.clone()),
            ScenePayload::Cylinder(shape) => Shape::Cylinder(shape.clone()),
            ScenePayload::Cone(shape) => Shape::Cone(shape.clone()),
            ScenePayload::CappedCone(shape) => Shape::CappedCone(shape.clone()),
            ScenePayload::Pyramid(shape) => Shape::Pyramid(shape.clone()),
            ScenePayload::BoxFrame(shape) => Shape::BoxFrame(shape.clone()),
            ScenePayload::Torus(shape) => Shape::Torus(shape.clone()),
            ScenePayload::PolylineTube(shape) => Shape::PolylineTube(shape.clone()),
            ScenePayload::QuadraticBezierTube(shape) => Shape::QuadraticBezierTube(shape.clone()),
            ScenePayload::NormalCurtain(shape) => Shape::NormalCurtain(shape.clone()),
            ScenePayload::Placed2D {
                profile,
                origin,
                axis_u,
                axis_v,
                sources,
            } => {
                let mut source_nodes = Vec::with_capacity(sources.len());
                for source in sources {
                    source_nodes.push(self.build_node_guarded(*source, visiting)?);
                }
                Shape::PlacedSdf2D(PlacedSdf2D::new(
                    profile.clone(),
                    *origin,
                    *axis_u,
                    *axis_v,
                    source_nodes,
                )?)
            }
            ScenePayload::PlacedPolyline1D {
                profile,
                origin,
                axis_u,
                axis_v,
            } => Shape::PlacedPolyline1D(PlacedPolyline1D::new(
                profile.clone(),
                *origin,
                *axis_u,
                *axis_v,
            )?),
            ScenePayload::Placed1D {
                profile,
                origin,
                axis_u,
                sources,
            } => {
                let mut source_nodes = Vec::with_capacity(sources.len());
                for source in sources {
                    source_nodes.push(self.build_node_guarded(*source, visiting)?);
                }
                Shape::PlacedSdf1D(PlacedSdf1D::new(
                    profile.clone(),
                    *origin,
                    *axis_u,
                    source_nodes,
                )?)
            }
            ScenePayload::Operator { kind, left, right } => {
                let left = self.build_node_guarded(*left, visiting)?;
                let right = self.build_node_guarded(*right, visiting)?;
                match kind {
                    OperatorKind::Union => Shape::union(left, right)?,
                    OperatorKind::Intersection => Shape::intersection(left, right)?,
                    OperatorKind::Difference => Shape::difference(left, right)?,
                    OperatorKind::Xor => Shape::xor(left, right)?,
                }
            }
            ScenePayload::Translate { child, offset } => Shape::Translate {
                child: Box::new(self.build_node_guarded(*child, visiting)?),
                offset: *offset,
            },
            ScenePayload::Rotate {
                child,
                axis,
                angle_degrees,
            } => Shape::Rotate {
                child: Box::new(self.build_node_guarded(*child, visiting)?),
                axis: *axis,
                angle_degrees: *angle_degrees,
            },
            ScenePayload::Scale { child, factor } => {
                Shape::scale(self.build_node_guarded(*child, visiting)?, *factor)?
            }
            ScenePayload::Extrude {
                section,
                height,
                center_offset,
            } => Shape::Extrude(Extrude::new(
                self.build_node_guarded(*section, visiting)?,
                *height,
                *center_offset,
            )?),
            ScenePayload::Revolve {
                section,
                axis,
                axis_origin,
                axis_direction,
                radial_direction,
                angle_degrees,
            } => Shape::Revolve(Revolve::new(
                self.build_node_guarded(*section, visiting)?,
                *axis,
                *axis_origin,
                *axis_direction,
                *radial_direction,
                *angle_degrees,
            )?),
        };
        visiting.pop();
        Ok(Node::with_id(object.name.clone(), object.id, shape))
    }

    pub fn dimension_of(&self, id: ObjectId) -> GeometryResult<u8> {
        let object = self.object(id)?;
        Ok(match &object.payload {
            ScenePayload::Placed2D { .. } => 2,
            ScenePayload::PlacedPolyline1D { .. } | ScenePayload::Placed1D { .. } => 1,
            ScenePayload::Operator { left, .. } => self.dimension_of(*left)?,
            ScenePayload::Translate { child, .. }
            | ScenePayload::Rotate { child, .. }
            | ScenePayload::Scale { child, .. } => self.dimension_of(*child)?,
            _ => 3,
        })
    }

    /// Create a default-sized 3D primitive (sizes in `scale` scene units) and
    /// add it as a root. Mirrors Python `create_primitive` + `add_primitive`.
    pub fn add_primitive(&mut self, kind: &str, scale: f64) -> GeometryResult<ObjectId> {
        let name = self.default_name(kind);
        let payload = match kind {
            "sphere" => ScenePayload::Sphere(Sphere::new(Vec3::ZERO, 0.5 * scale)?),
            "box" => ScenePayload::Box3(Box3::new(
                Vec3::ZERO,
                vec3(0.5 * scale, 0.5 * scale, 0.5 * scale),
                IDENTITY_FRAME,
            )?),
            "cylinder" => {
                ScenePayload::Cylinder(Cylinder::new(Vec3::ZERO, 0.4 * scale, 0.6 * scale, IDENTITY_FRAME)?)
            }
            "capped_cone" => ScenePayload::CappedCone(CappedCone::new(
                Vec3::ZERO,
                0.45 * scale,
                0.25 * scale,
                0.6 * scale,
                IDENTITY_FRAME,
            )?),
            "cone" => ScenePayload::Cone(Cone::new(Vec3::ZERO, 0.45 * scale, 0.6 * scale, IDENTITY_FRAME)?),
            "pyramid" => ScenePayload::Pyramid(Pyramid::new(
                Vec3::ZERO,
                0.45 * scale,
                0.6 * scale,
                IDENTITY_FRAME,
            )?),
            "box_frame" => ScenePayload::BoxFrame(BoxFrame::new(
                Vec3::ZERO,
                vec3(0.5 * scale, 0.5 * scale, 0.5 * scale),
                0.08 * scale,
                IDENTITY_FRAME,
            )?),
            "torus" => ScenePayload::Torus(Torus::new(Vec3::ZERO, 0.5 * scale, 0.15 * scale, IDENTITY_FRAME)?),
            "polyline_tube" => ScenePayload::PolylineTube(PolylineTube::new(
                vec![
                    vec3(-0.75 * scale, 0.0, 0.0),
                    vec3(0.0, 0.5 * scale, 0.0),
                    vec3(0.75 * scale, 0.0, 0.0),
                ],
                0.12 * scale,
                0.0,
                CapStyle::Round,
            )?),
            "quadratic_bezier_tube" => ScenePayload::QuadraticBezierTube(QuadraticBezierTube::new(
                vec![
                    vec3(-0.75 * scale, 0.0, 0.0),
                    vec3(0.0, 0.55 * scale, 0.0),
                    vec3(0.75 * scale, 0.0, 0.0),
                ],
                0.12 * scale,
                0.0,
                CapStyle::Round,
            )?),
            other => {
                return Err(GeometryError::new(format!("unknown 3D primitive type: {other}")))
            }
        };
        let id = self.insert_object(name, payload)?;
        self.roots.push(id);
        self.mark_changed();
        Ok(id)
    }

    fn contains(&self, ancestor: ObjectId, descendant: ObjectId) -> bool {
        if ancestor == descendant {
            return true;
        }
        match self.objects.get(&ancestor) {
            Some(object) => object
                .payload
                .children()
                .iter()
                .any(|child| self.contains(*child, descendant)),
            None => false,
        }
    }

    /// Remove an id from the roots list, returning its former index (or the
    /// current roots length when it was nested, matching Python `_detach`).
    fn detach_root(&mut self, id: ObjectId) -> usize {
        match self.roots.iter().position(|root| *root == id) {
            Some(index) => {
                self.roots.remove(index);
                index
            }
            None => self.roots.len(),
        }
    }

    /// Combine two nodes with an SDF operator; 1D/2D operands merge into a
    /// single placed node with a Binary profile (Python `combine`).
    pub fn combine(
        &mut self,
        first: ObjectId,
        second: ObjectId,
        operation: &str,
    ) -> GeometryResult<ObjectId> {
        if first == second {
            return Err(GeometryError::new("select two different SDF nodes"));
        }
        if self.contains(first, second) || self.contains(second, first) {
            return Err(GeometryError::new(
                "an SDF cannot be combined with its own descendant",
            ));
        }
        if self.dimension_of(first)? != self.dimension_of(second)? {
            return Err(GeometryError::new(
                "boolean operands must have the same dimension",
            ));
        }
        let kind = OperatorKind::parse(operation)?;
        let default_name = self.default_name(operation);
        let replaces_fluid_root = self
            .fluid_domain
            .as_ref()
            .is_some_and(|fluid| fluid.root == first || fluid.root == second);

        let first_payload = self.object(first)?.payload.clone();
        let second_payload = self.object(second)?.payload.clone();
        let payload = match (&first_payload, &second_payload) {
            (
                ScenePayload::Placed1D {
                    profile: first_profile,
                    origin: first_origin,
                    axis_u: first_axis,
                    ..
                },
                ScenePayload::Placed1D {
                    profile: second_profile,
                    origin: second_origin,
                    ..
                },
            ) => {
                let first_node = self.build_node(first)?;
                let second_node = self.build_node(second)?;
                let (Shape::PlacedSdf1D(first_placed), Shape::PlacedSdf1D(second_placed)) =
                    (&first_node.shape, &second_node.shape)
                else {
                    unreachable!("payload matched Placed1D");
                };
                if !first_placed.is_collinear_with(second_placed, 1e-6) {
                    return Err(GeometryError::new("1D boolean operands must be collinear"));
                }
                let displacement = *second_origin - *first_origin;
                let second_offset = displacement.dot(*first_axis);
                ScenePayload::Placed1D {
                    profile: Profile1D::Binary {
                        left: Box::new(first_profile.clone()),
                        right: Box::new(Profile1D::Offset {
                            child: Box::new(second_profile.clone()),
                            offset: second_offset,
                        }),
                        operation: kind.as_boolean_op_1d(),
                        smoothing: 0.1,
                    },
                    origin: *first_origin,
                    axis_u: *first_axis,
                    sources: vec![first, second],
                }
            }
            (
                ScenePayload::Placed2D {
                    profile: first_profile,
                    origin: first_origin,
                    axis_u: first_axis_u,
                    axis_v: first_axis_v,
                    ..
                },
                ScenePayload::Placed2D {
                    profile: second_profile,
                    origin: second_origin,
                    ..
                },
            ) => {
                let first_node = self.build_node(first)?;
                let second_node = self.build_node(second)?;
                let (Shape::PlacedSdf2D(first_placed), Shape::PlacedSdf2D(second_placed)) =
                    (&first_node.shape, &second_node.shape)
                else {
                    unreachable!("payload matched Placed2D");
                };
                if !first_placed.is_coplanar_with(second_placed, 1e-6) {
                    return Err(GeometryError::new("2D boolean operands must be coplanar"));
                }
                let displacement = *second_origin - *first_origin;
                let second_offset = [
                    displacement.dot(*first_axis_u),
                    displacement.dot(*first_axis_v),
                ];
                ScenePayload::Placed2D {
                    profile: Profile2D::Binary {
                        left: Box::new(first_profile.clone()),
                        right: Box::new(Profile2D::Offset {
                            child: Box::new(second_profile.clone()),
                            offset: second_offset,
                        }),
                        operation: kind.as_boolean_op_1d(),
                        smoothing: 0.1,
                    },
                    origin: *first_origin,
                    axis_u: *first_axis_u,
                    axis_v: *first_axis_v,
                    sources: vec![first, second],
                }
            }
            _ if self.dimension_of(first)? == 3 => ScenePayload::Operator {
                kind,
                left: first,
                right: second,
            },
            _ => return Err(GeometryError::new("unsupported boolean operand types")),
        };

        let combined = self.insert_object(default_name, payload)?;
        let first_index = self.detach_root(first);
        let second_index = self.detach_root(second);
        let index = first_index.min(second_index).min(self.roots.len());
        self.roots.insert(index, combined);
        if replaces_fluid_root {
            if let Some(fluid) = self.fluid_domain.as_mut() {
                fluid.root = combined;
            }
            if let Some(fluid) = self.fluid_domain.clone() {
                self.domain_kinds.insert(fluid.root, DomainKind::Fluid);
            }
        }
        self.refresh_fluid_domain();
        self.mark_changed();
        Ok(combined)
    }

    /// Wrap a 3D node in a default transform (Python `wrap_transform`).
    pub fn wrap_transform(&mut self, id: ObjectId, transform: &str) -> GeometryResult<ObjectId> {
        if self.dimension_of(id)? != 3 {
            return Err(GeometryError::new(
                "edit the placed SDF origin and axes to transform 1D or 2D objects",
            ));
        }
        let name = self.default_name(transform);
        let payload = match transform {
            "translate" => ScenePayload::Translate {
                child: id,
                offset: vec3(0.1, 0.0, 0.0),
            },
            "rotate" => ScenePayload::Rotate {
                child: id,
                axis: RotationAxis::Y,
                angle_degrees: 15.0,
            },
            "scale" => ScenePayload::Scale {
                child: id,
                factor: 1.1,
            },
            other => return Err(GeometryError::new(format!("unknown transform: {other}"))),
        };
        let was_fluid_root = self
            .fluid_domain
            .as_ref()
            .is_some_and(|fluid| fluid.root == id);
        let wrapped = self.insert_object(name, payload)?;
        let index = self.detach_root(id);
        let index = index.min(self.roots.len());
        self.roots.insert(index, wrapped);
        if was_fluid_root {
            if let Some(fluid) = self.fluid_domain.as_mut() {
                fluid.root = wrapped;
            }
            self.domain_kinds.insert(wrapped, DomainKind::Fluid);
        }
        self.refresh_fluid_domain();
        self.mark_changed();
        Ok(wrapped)
    }

    /// Extrude or revolve one placed 2D section (Python `solid_from_2d`).
    #[allow(clippy::too_many_arguments)]
    pub fn solid_from_2d(
        &mut self,
        section: ObjectId,
        method: &str,
        signed_height: Option<f64>,
        revolve_axis: RevolveAxis,
        revolve_axis_origin: Option<Vec3>,
        revolve_axis_direction: Option<Vec3>,
        revolve_radial_direction: Option<Vec3>,
        revolve_angle_degrees: f64,
    ) -> GeometryResult<ObjectId> {
        if !matches!(self.object(section)?.payload, ScenePayload::Placed2D { .. }) {
            return Err(GeometryError::new("Solid From 2D requires placed 2D objects"));
        }
        let name = self.default_name(method);
        let payload = match method {
            "extrude" => {
                let height = signed_height.map_or(1.0, f64::abs);
                if height <= 0.0 || !height.is_finite() {
                    return Err(GeometryError::new(
                        "extrude height must be finite and positive",
                    ));
                }
                let center_offset = signed_height.map_or(0.0, |value| value * 0.5);
                ScenePayload::Extrude {
                    section,
                    height,
                    center_offset,
                }
            }
            "revolve" => ScenePayload::Revolve {
                section,
                axis: revolve_axis,
                axis_origin: revolve_axis_origin,
                axis_direction: revolve_axis_direction,
                radial_direction: revolve_radial_direction,
                angle_degrees: revolve_angle_degrees,
            },
            other => return Err(GeometryError::new(format!("invalid section count for {other}"))),
        };
        let replaces_fluid_root = self
            .fluid_domain
            .as_ref()
            .is_some_and(|fluid| fluid.root == section);
        let result = self.insert_object(name, payload)?;
        if method == "revolve" {
            let built = self.build_node(result)?;
            if let Shape::Revolve(revolve) = &built.shape {
                let issues = revolve_violations(&built.name, revolve);
                if !issues.is_empty() {
                    self.objects.remove(&result);
                    return Err(GeometryError::new(issues[0].clone()));
                }
            }
        }
        match self.roots.iter().position(|root| *root == section) {
            Some(index) => self.roots[index] = result,
            None => self.roots.push(result),
        }
        if replaces_fluid_root {
            if let Some(fluid) = self.fluid_domain.as_mut() {
                fluid.root = result;
            }
            self.domain_kinds.insert(result, DomainKind::Fluid);
        }
        self.refresh_fluid_domain();
        self.mark_changed();
        Ok(result)
    }

    pub fn set_domain_root(&mut self, id: ObjectId, kind: DomainKind) -> GeometryResult<()> {
        let dimension = self.dimension_of(id)?;
        if dimension != 2 && dimension != 3 {
            return Err(GeometryError::new("Domain root must be a 2D or 3D SDF"));
        }
        self.domain_kinds.insert(id, kind);
        if kind == DomainKind::Fluid {
            let previous = self.fluid_domain.take();
            self.fluid_domain = Some(FluidDomainRecord {
                root: id,
                tags: previous.as_ref().map(|fluid| fluid.tags.clone()).unwrap_or_default(),
                selectors: previous
                    .as_ref()
                    .map(|fluid| fluid.selectors.clone())
                    .unwrap_or_default(),
            });
        }
        self.mark_changed();
        Ok(())
    }

    pub fn unset_domain_root(&mut self, id: ObjectId) {
        self.domain_kinds.remove(&id);
        if self
            .fluid_domain
            .as_ref()
            .is_some_and(|fluid| fluid.root == id)
        {
            self.fluid_domain = None;
        }
        self.mark_changed();
    }

    /// Move an object by mutating its own placement when possible, otherwise
    /// wrapping it in a Translate (Python `move_object`).
    pub fn move_object(&mut self, id: ObjectId, delta: Vec3) -> GeometryResult<ObjectId> {
        let moved = self.translate_in_place(id, delta)?;
        if moved {
            self.mark_changed();
            return Ok(id);
        }
        let wrapped = self.wrap_transform(id, "translate")?;
        if let ScenePayload::Translate { offset, .. } = &mut self.object_mut(wrapped)?.payload {
            *offset = delta;
        }
        Ok(wrapped)
    }

    fn translate_in_place(&mut self, id: ObjectId, delta: Vec3) -> GeometryResult<bool> {
        let payload = &mut self.object_mut(id)?.payload;
        match payload {
            ScenePayload::Rotate { .. } | ScenePayload::Scale { .. } => Ok(false),
            ScenePayload::Translate { offset, .. } => {
                *offset += delta;
                Ok(true)
            }
            ScenePayload::Placed2D { origin, .. }
            | ScenePayload::PlacedPolyline1D { origin, .. }
            | ScenePayload::Placed1D { origin, .. } => {
                *origin += delta;
                Ok(true)
            }
            ScenePayload::Sphere(Sphere { center, .. })
            | ScenePayload::Box3(Box3 { center, .. })
            | ScenePayload::BoxFrame(BoxFrame { center, .. })
            | ScenePayload::CappedCone(CappedCone { center, .. })
            | ScenePayload::Cone(Cone { center, .. })
            | ScenePayload::Cylinder(Cylinder { center, .. })
            | ScenePayload::Pyramid(Pyramid { center, .. })
            | ScenePayload::Torus(Torus { center, .. }) => {
                *center += delta;
                Ok(true)
            }
            ScenePayload::PolylineTube(tube) => {
                for point in &mut tube.points {
                    *point += delta;
                }
                Ok(true)
            }
            ScenePayload::QuadraticBezierTube(tube) => {
                for point in &mut tube.points {
                    *point += delta;
                }
                Ok(true)
            }
            ScenePayload::Extrude { section, .. } => {
                let section = *section;
                self.translate_in_place(section, delta)
            }
            ScenePayload::Revolve {
                axis_origin,
                section,
                ..
            } => {
                if let Some(origin) = axis_origin {
                    *origin += delta;
                }
                let section = *section;
                self.translate_in_place(section, delta)
            }
            ScenePayload::NormalCurtain(_) | ScenePayload::Operator { .. } => {
                // Translate the subtree by translating every leaf placement
                // (Python `_translate_copy_in_place`); only a Rotate/Scale in
                // the subtree refuses and falls back to a Translate wrapper.
                let children = payload.children();
                for child in children {
                    if !self.translate_in_place(child, delta)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
        }
    }

    /// Rotate an object in place about `pivot` (Python `rotate_object`).
    pub fn rotate_object(
        &mut self,
        id: ObjectId,
        axis: RotationAxis,
        angle_degrees: f64,
        pivot: Option<Vec3>,
    ) -> GeometryResult<()> {
        if angle_degrees.abs() <= 1e-9 {
            return Ok(());
        }
        let pivot = match pivot {
            Some(pivot) => pivot,
            None => self.build_node(id)?.bounding_box()?.center(),
        };
        if self.rotate_in_place(id, axis, angle_degrees, pivot)? {
            self.mark_changed();
            Ok(())
        } else {
            Err(GeometryError::new(
                "only SDF objects with editable placement can be rotated",
            ))
        }
    }

    fn rotate_in_place(
        &mut self,
        id: ObjectId,
        axis: RotationAxis,
        angle_degrees: f64,
        pivot: Vec3,
    ) -> GeometryResult<bool> {
        let rotate_point = |point: Vec3| rotate_about(point - pivot, axis, angle_degrees) + pivot;
        let rotate_vector = |vector: Vec3| rotate_about(vector, axis, angle_degrees);
        let children: Vec<ObjectId>;
        {
            let payload = &mut self.object_mut(id)?.payload;
            match payload {
                ScenePayload::Translate { offset, child } => {
                    *offset = rotate_vector(*offset);
                    children = vec![*child];
                }
                ScenePayload::Scale { child, .. } | ScenePayload::Rotate { child, .. } => {
                    children = vec![*child];
                }
                ScenePayload::Placed2D {
                    origin,
                    axis_u,
                    axis_v,
                    sources,
                    ..
                } => {
                    *origin = rotate_point(*origin);
                    *axis_u = rotate_vector(*axis_u);
                    *axis_v = rotate_vector(*axis_v);
                    children = sources.clone();
                }
                ScenePayload::PlacedPolyline1D {
                    origin,
                    axis_u,
                    axis_v,
                    ..
                } => {
                    *origin = rotate_point(*origin);
                    *axis_u = rotate_vector(*axis_u);
                    *axis_v = rotate_vector(*axis_v);
                    children = Vec::new();
                }
                ScenePayload::Placed1D {
                    origin,
                    axis_u,
                    sources,
                    ..
                } => {
                    *origin = rotate_point(*origin);
                    *axis_u = rotate_vector(*axis_u);
                    children = sources.clone();
                }
                ScenePayload::Sphere(sphere) => {
                    sphere.center = rotate_point(sphere.center);
                    children = Vec::new();
                }
                ScenePayload::Box3(Box3 { center, frame, .. })
                | ScenePayload::BoxFrame(BoxFrame { center, frame, .. })
                | ScenePayload::CappedCone(CappedCone { center, frame, .. })
                | ScenePayload::Cone(Cone { center, frame, .. })
                | ScenePayload::Cylinder(Cylinder { center, frame, .. })
                | ScenePayload::Pyramid(Pyramid { center, frame, .. })
                | ScenePayload::Torus(Torus { center, frame, .. }) => {
                    *center = rotate_point(*center);
                    *frame = Frame {
                        u: rotate_vector(frame.u),
                        v: rotate_vector(frame.v),
                        w: rotate_vector(frame.w),
                    };
                    children = Vec::new();
                }
                ScenePayload::PolylineTube(tube) => {
                    for point in &mut tube.points {
                        *point = rotate_point(*point);
                    }
                    children = Vec::new();
                }
                ScenePayload::QuadraticBezierTube(tube) => {
                    for point in &mut tube.points {
                        *point = rotate_point(*point);
                    }
                    children = Vec::new();
                }
                ScenePayload::Extrude { section, .. } => {
                    children = vec![*section];
                }
                ScenePayload::Revolve {
                    axis_origin,
                    axis_direction,
                    radial_direction,
                    section,
                    ..
                } => {
                    if let Some(origin) = axis_origin {
                        *origin = rotate_point(*origin);
                    }
                    if let Some(direction) = axis_direction {
                        *direction = rotate_vector(*direction);
                    }
                    if let Some(direction) = radial_direction {
                        *direction = rotate_vector(*direction);
                    }
                    children = vec![*section];
                }
                ScenePayload::NormalCurtain(_) | ScenePayload::Operator { .. } => {
                    children = payload.children();
                    for child in children {
                        if !self.rotate_in_place(child, axis, angle_degrees, pivot)? {
                            return Ok(false);
                        }
                    }
                    return Ok(true);
                }
            }
        }
        for child in children {
            if !self.rotate_in_place(child, axis, angle_degrees, pivot)? {
                return Ok(false);
            }
        }
        Ok(true)
    }

    pub fn rename(&mut self, id: ObjectId, name: impl Into<String>) -> GeometryResult<()> {
        self.object_mut(id)?.name = name.into();
        self.mark_changed();
        Ok(())
    }

    /// Delete objects: boundary regions by object_id, SDF nodes by collapsing
    /// operators to the surviving operand and dropping transforms with their
    /// target (Python `delete_many`).
    pub fn delete_many(&mut self, ids: &[ObjectId]) -> usize {
        let region_ids: Vec<ObjectId> = self
            .boundary_regions
            .iter()
            .filter(|region| ids.contains(&region.object_id))
            .map(|region| region.object_id)
            .collect();
        let mut deleted = 0usize;
        if !region_ids.is_empty() {
            let before = self.boundary_regions.len();
            self.boundary_regions
                .retain(|region| !region_ids.contains(&region.object_id));
            deleted += before - self.boundary_regions.len();
        }
        let node_targets: Vec<ObjectId> = ids
            .iter()
            .copied()
            .filter(|id| self.objects.contains_key(id) && !region_ids.contains(id))
            .collect();
        if !node_targets.is_empty() {
            let roots = std::mem::take(&mut self.roots);
            let mut remaining = Vec::new();
            for root in roots {
                let (replacement, removed) = self.remove_targets_from(root, &node_targets);
                deleted += removed;
                if let Some(id) = replacement {
                    remaining.push(id);
                }
            }
            self.roots = remaining;
        }
        if deleted == 0 {
            return 0;
        }
        self.refresh_fluid_domain();
        self.mark_changed();
        deleted
    }

    pub fn delete(&mut self, id: ObjectId) -> usize {
        self.delete_many(&[id])
    }

    fn remove_targets_from(
        &mut self,
        current: ObjectId,
        targets: &[ObjectId],
    ) -> (Option<ObjectId>, usize) {
        if targets.contains(&current) {
            return (None, 1);
        }
        let payload = match self.objects.get(&current) {
            Some(object) => object.payload.clone(),
            None => return (None, 0),
        };
        match payload {
            ScenePayload::Translate { child, .. }
            | ScenePayload::Rotate { child, .. }
            | ScenePayload::Scale { child, .. } => {
                let (replacement, removed) = self.remove_targets_from(child, targets);
                if removed == 0 {
                    return (Some(current), 0);
                }
                match replacement {
                    None => (None, removed),
                    Some(new_child) => {
                        if let Ok(object) = self.object_mut(current) {
                            match &mut object.payload {
                                ScenePayload::Translate { child, .. }
                                | ScenePayload::Rotate { child, .. }
                                | ScenePayload::Scale { child, .. } => *child = new_child,
                                _ => {}
                            }
                        }
                        (Some(current), removed)
                    }
                }
            }
            ScenePayload::Operator { left, right, .. } => {
                let (left_replacement, left_removed) = self.remove_targets_from(left, targets);
                let (right_replacement, right_removed) = self.remove_targets_from(right, targets);
                let removed = left_removed + right_removed;
                if removed == 0 {
                    return (Some(current), 0);
                }
                match (left_replacement, right_replacement) {
                    (Some(new_left), Some(new_right)) => {
                        if let Ok(object) = self.object_mut(current) {
                            if let ScenePayload::Operator { left, right, .. } = &mut object.payload
                            {
                                *left = new_left;
                                *right = new_right;
                            }
                        }
                        (Some(current), removed)
                    }
                    (Some(survivor), None) | (None, Some(survivor)) => (Some(survivor), removed),
                    (None, None) => (None, removed),
                }
            }
            ScenePayload::Extrude { section, .. } | ScenePayload::Revolve { section, .. } => {
                let (replacement, removed) = self.remove_targets_from(section, targets);
                if removed == 0 {
                    return (Some(current), 0);
                }
                match replacement {
                    None => (None, removed),
                    Some(new_section) => {
                        if let Ok(object) = self.object_mut(current) {
                            match &mut object.payload {
                                ScenePayload::Extrude { section, .. }
                                | ScenePayload::Revolve { section, .. } => *section = new_section,
                                _ => {}
                            }
                        }
                        (Some(current), removed)
                    }
                }
            }
            _ => (Some(current), 0),
        }
    }

    /// Prune domain kinds, boundary regions, and the fluid record to nodes
    /// still reachable from the roots (Python `_refresh_fluid_domain`), and
    /// garbage-collect unreachable objects from the id map.
    pub fn refresh_fluid_domain(&mut self) {
        let live = self.live_ids();
        self.domain_kinds.retain(|id, _kind| live.contains(id));
        self.boundary_regions
            .retain(|region| live.contains(&region.owner_object_id));
        let region_ids: Vec<ObjectId> = self
            .boundary_regions
            .iter()
            .map(|region| region.object_id)
            .collect();
        if let Some(fluid) = self.fluid_domain.take() {
            if live.contains(&fluid.root) {
                let tags = fluid
                    .tags
                    .into_iter()
                    .filter(|tag| match tag {
                        TagRef::Region(id) => region_ids.contains(id),
                        TagRef::Node(id) => live.contains(id),
                    })
                    .collect();
                let selectors = fluid
                    .selectors
                    .into_iter()
                    .filter(|id| live.contains(id))
                    .collect();
                self.domain_kinds.insert(fluid.root, DomainKind::Fluid);
                self.fluid_domain = Some(FluidDomainRecord {
                    root: fluid.root,
                    tags,
                    selectors,
                });
            }
        }
        self.objects.retain(|id, _object| live.contains(id));
    }

    /// Undo snapshot: the document is a value, so a snapshot is a clone.
    pub fn snapshot(&self) -> SceneDocument {
        self.clone()
    }

    /// Create a primitive sized by a world-space drag (Python
    /// `add_primitive_from_drag`). `scale` (meters per working unit) only
    /// sets degenerate-drag minimums and parameter defaults.
    pub fn add_primitive_from_drag(
        &mut self,
        kind: &str,
        start: Vec3,
        end: Vec3,
        scale: f64,
    ) -> GeometryResult<ObjectId> {
        let minimum_half = 0.05 * scale;
        let center = (start + end) * 0.5;
        let (axis_a, axis_b) = drag_plane_axes(start, end);
        let axis_u = axis_unit(axis_a);
        let axis_v = axis_unit(axis_b);
        let extent_a = ((component(end, axis_a) - component(start, axis_a)).abs() * 0.5)
            .max(minimum_half);
        let extent_b = ((component(end, axis_b) - component(start, axis_b)).abs() * 0.5)
            .max(minimum_half);
        let planar = [
            component(end, axis_a) - component(start, axis_a),
            component(end, axis_b) - component(start, axis_b),
        ];
        let radius = ((planar[0] * planar[0] + planar[1] * planar[1]).sqrt() * 0.5)
            .max(minimum_half);

        let payload = match kind {
            "segment" | "interval" => {
                let direction = end - start;
                let length = direction.length();
                ScenePayload::Placed1D {
                    profile: Profile1D::segment(0.0, (0.5 * length).max(minimum_half))?,
                    origin: center,
                    axis_u: if length > 1e-12 {
                        direction * (1.0 / length)
                    } else {
                        vec3(1.0, 0.0, 0.0)
                    },
                    sources: Vec::new(),
                }
            }
            "polyline" => ScenePayload::PlacedPolyline1D {
                profile: Profile2D::polyline(vec![
                    [-extent_a, -extent_b],
                    [extent_a, extent_b],
                ])?,
                origin: center,
                axis_u,
                axis_v,
            },
            "quadratic_bezier_curve" => ScenePayload::PlacedPolyline1D {
                profile: Profile2D::quadratic_bezier_curve(vec![
                    [-extent_a, 0.0],
                    [0.0, extent_b],
                    [extent_a, 0.0],
                ])?,
                origin: center,
                axis_u,
                axis_v,
            },
            "quadratic_bezier_polycurve" => ScenePayload::PlacedPolyline1D {
                profile: Profile2D::quadratic_bezier_curve(vec![
                    [-extent_a, 0.0],
                    [-0.5 * extent_a, extent_b],
                    [0.0, 0.0],
                    [0.5 * extent_a, -extent_b],
                    [extent_a, 0.0],
                ])?,
                origin: center,
                axis_u,
                axis_v,
            },
            "polyline_tube" => ScenePayload::PolylineTube(PolylineTube::new(
                vec![start, end],
                0.12 * scale,
                0.0,
                CapStyle::Round,
            )?),
            "quadratic_bezier_tube" => {
                let mut control = center;
                set_component(&mut control, axis_b, component(center, axis_b) + extent_b);
                ScenePayload::QuadraticBezierTube(QuadraticBezierTube::new(
                    vec![start, control, end],
                    0.12 * scale,
                    0.0,
                    CapStyle::Round,
                )?)
            }
            "circle" | "rectangle" | "square" | "rounded_rectangle" | "ellipse"
            | "regular_polygon" | "polygon" => {
                let profile = match kind {
                    "circle" => Profile2D::circle([0.0, 0.0], radius)?,
                    "rectangle" => Profile2D::rectangle([0.0, 0.0], [extent_a, extent_b])?,
                    "square" => Profile2D::square([0.0, 0.0], extent_a.max(extent_b))?,
                    "rounded_rectangle" => Profile2D::rounded_rectangle(
                        [0.0, 0.0],
                        [extent_a, extent_b],
                        (0.01 * scale).max(extent_a.min(extent_b) * 0.2),
                    )?,
                    "ellipse" => Profile2D::ellipse([0.0, 0.0], [extent_a, extent_b])?,
                    "polygon" => Profile2D::polygon(vec![
                        [-extent_a, -extent_b],
                        [extent_a, -extent_b],
                        [extent_a, extent_b],
                        [-extent_a, extent_b],
                    ])?,
                    _ => Profile2D::regular_polygon([0.0, 0.0], radius, 6, 0.0)?,
                };
                ScenePayload::Placed2D {
                    profile,
                    origin: center,
                    axis_u,
                    axis_v,
                    sources: Vec::new(),
                }
            }
            _ => {
                // 3D primitives: defaults from `add_primitive`, then resize
                // from the drag (mirrors Python's create-then-mutate).
                let id = self.add_primitive(kind, scale)?;
                let delta = end - start;
                let box_delta = vec3(delta.x.abs(), delta.y.abs(), delta.z.abs());
                let radial_delta = (delta.x * delta.x + delta.y * delta.y).sqrt();
                let height_delta = delta.z.abs();
                let drag_half_height = if height_delta > 1e-9 {
                    (0.5 * height_delta).max(minimum_half)
                } else {
                    extent_a.max(extent_b)
                };
                let full_box_half = |fallback: f64| -> Vec3 {
                    let nonzero = [box_delta.x, box_delta.y, box_delta.z]
                        .iter()
                        .filter(|value| **value > 1e-9)
                        .count();
                    if nonzero == 3 {
                        vec3(
                            (0.5 * box_delta.x).max(minimum_half),
                            (0.5 * box_delta.y).max(minimum_half),
                            (0.5 * box_delta.z).max(minimum_half),
                        )
                    } else {
                        let mut half = vec3(fallback, fallback, fallback);
                        set_component(&mut half, axis_a, extent_a);
                        set_component(&mut half, axis_b, extent_b);
                        half
                    }
                };
                match &mut self.object_mut(id)?.payload {
                    ScenePayload::Sphere(sphere) => {
                        sphere.center = center;
                        sphere.radius = radius;
                    }
                    ScenePayload::Box3(shape) => {
                        shape.center = center;
                        shape.half_size = full_box_half(extent_a.max(extent_b));
                    }
                    ScenePayload::Cylinder(shape) => {
                        shape.center = center;
                        shape.radius = (0.5 * radial_delta).max(minimum_half);
                        shape.half_height = drag_half_height;
                    }
                    ScenePayload::CappedCone(shape) => {
                        shape.center = center;
                        shape.radius_a = (0.5 * radial_delta).max(minimum_half);
                        shape.radius_b = (shape.radius_a * 0.45).max(0.025 * scale);
                        shape.half_height = drag_half_height;
                    }
                    ScenePayload::Cone(shape) => {
                        shape.center = center;
                        shape.radius = (0.5 * radial_delta).max(minimum_half);
                        shape.half_height = drag_half_height;
                    }
                    ScenePayload::Pyramid(shape) => {
                        shape.center = center;
                        shape.base_half_size = extent_a.max(extent_b).max(minimum_half);
                        shape.half_height = if box_delta.z > 1e-9 {
                            (0.5 * box_delta.z).max(minimum_half)
                        } else {
                            extent_a.max(extent_b)
                        };
                    }
                    ScenePayload::BoxFrame(shape) => {
                        shape.center = center;
                        shape.half_size = full_box_half(extent_a.max(extent_b));
                        let smallest = shape
                            .half_size
                            .x
                            .min(shape.half_size.y)
                            .min(shape.half_size.z);
                        shape.thickness = (smallest * 0.14).max(0.015 * scale);
                    }
                    ScenePayload::Torus(shape) => {
                        shape.center = center;
                        shape.major_radius = radius;
                        shape.minor_radius = (radius * 0.25).max(0.02 * scale);
                    }
                    _ => {}
                }
                self.mark_changed();
                return Ok(id);
            }
        };
        let name = self.default_name(kind);
        let id = self.insert_object(name, payload)?;
        self.roots.push(id);
        self.mark_changed();
        Ok(id)
    }

    /// Place a point-defined shape from world-space points on a reference
    /// plane (Python `add_point_shape_from_world_points`).
    pub fn add_point_shape_from_world_points(
        &mut self,
        kind: &str,
        points: &[Vec3],
        reference_plane: &str,
    ) -> GeometryResult<ObjectId> {
        let (axis_u, axis_v) = reference_plane_axes(reference_plane)?;
        let minimum_points = if matches!(kind, "segment" | "interval" | "polyline" | "polyline_tube")
        {
            2
        } else {
            3
        };
        if points.len() < minimum_points {
            return Err(GeometryError::new(if minimum_points == 2 {
                format!("{kind} requires at least two points")
            } else {
                format!("{kind} requires at least three points")
            }));
        }
        if matches!(kind, "segment" | "interval") && points.len() != 2 {
            return Err(GeometryError::new("segment requires exactly two points"));
        }
        if kind == "quadratic_bezier_curve" && points.len() != 3 {
            return Err(GeometryError::new(
                "quadratic Bezier curve requires exactly three points",
            ));
        }
        if matches!(
            kind,
            "quadratic_bezier_polycurve" | "quadratic_bezier_tube" | "quadratic_bezier_surface"
        ) && points.len().is_multiple_of(2)
        {
            return Err(GeometryError::new(format!(
                "{kind} requires an odd point count: anchor, control, anchor"
            )));
        }
        let origin = points[0];
        let local: Vec<[f64; 2]> = points
            .iter()
            .map(|point| {
                let offset = *point - origin;
                [offset.dot(axis_u), offset.dot(axis_v)]
            })
            .collect();
        let payload = match kind {
            // Same Placed1D as the drag path: origin at the midpoint, the
            // profile spans half the point distance along axis_u.
            "segment" | "interval" => {
                let direction = points[1] - points[0];
                let length = direction.length();
                ScenePayload::Placed1D {
                    profile: Profile1D::segment(0.0, 0.5 * length)?,
                    origin: (points[0] + points[1]) * 0.5,
                    axis_u: if length > 1e-12 {
                        direction * (1.0 / length)
                    } else {
                        vec3(1.0, 0.0, 0.0)
                    },
                    sources: Vec::new(),
                }
            }
            "polyline" => ScenePayload::PlacedPolyline1D {
                profile: Profile2D::polyline(local)?,
                origin,
                axis_u,
                axis_v,
            },
            "quadratic_bezier_curve" | "quadratic_bezier_polycurve" => {
                ScenePayload::PlacedPolyline1D {
                    profile: Profile2D::quadratic_bezier_curve(local)?,
                    origin,
                    axis_u,
                    axis_v,
                }
            }
            "polyline_tube" => ScenePayload::PolylineTube(PolylineTube::new(
                points.to_vec(),
                0.12,
                0.0,
                CapStyle::Round,
            )?),
            "quadratic_bezier_tube" => ScenePayload::QuadraticBezierTube(
                QuadraticBezierTube::new(points.to_vec(), 0.12, 0.0, CapStyle::Round)?,
            ),
            "quadratic_bezier_surface" => ScenePayload::Placed2D {
                profile: Profile2D::quadratic_bezier_surface(local)?,
                origin,
                axis_u,
                axis_v,
                sources: Vec::new(),
            },
            "polygon" => ScenePayload::Placed2D {
                profile: Profile2D::polygon(local)?,
                origin,
                axis_u,
                axis_v,
                sources: Vec::new(),
            },
            other => {
                return Err(GeometryError::new(format!("unsupported point shape: {other}")))
            }
        };
        let name_key = match kind {
            "quadratic_bezier_polycurve" => "quadratic_bezier_curve",
            other => other,
        };
        let name = self.default_name(name_key);
        let id = self.insert_object(name, payload)?;
        self.roots.push(id);
        self.mark_changed();
        Ok(id)
    }

    /// Deep-copy the given nodes with fresh ids, offset them, and append as
    /// roots — Python `copy_nodes` + `paste_nodes` in one step. Selected
    /// nodes that are descendants of other selected nodes are skipped.
    pub fn duplicate_nodes(
        &mut self,
        ids: &[ObjectId],
        offset: Vec3,
    ) -> GeometryResult<Vec<ObjectId>> {
        let top: Vec<ObjectId> = ids
            .iter()
            .copied()
            .filter(|id| {
                !ids.iter()
                    .any(|other| *other != *id && self.contains(*other, *id))
            })
            .collect();
        let mut pasted = Vec::new();
        for id in top {
            let clone = self.clone_subtree(id)?;
            let name = format!("{} copy", self.object(clone)?.name);
            self.object_mut(clone)?.name = name.clone();
            self.roots.push(clone);
            // Non-translatable payloads get a Translate wrapper; the wrapper
            // is the pasted node then (Python names it like the clone).
            let moved = self.move_object(clone, offset)?;
            if moved != clone {
                self.object_mut(moved)?.name = name;
            }
            pasted.push(moved);
        }
        self.refresh_fluid_domain();
        self.mark_changed();
        Ok(pasted)
    }

    /// Create a BoundaryRegion tagging part of the FluidDomain boundary
    /// (Python `add_boundary_region`, 3D path).
    pub fn add_boundary_region(
        &mut self,
        owner_object_id: ObjectId,
        outside_direction: Option<u8>,
        patch_id: Option<&str>,
        patch_type: Option<&str>,
    ) -> GeometryResult<u32> {
        let fluid = self
            .fluid_domain
            .clone()
            .ok_or_else(|| GeometryError::new("select a FluidDomain root first"))?;
        let root = self.build_node(fluid.root)?;
        let owners = crate::boundary_ops::boundary_owner_ids(&root);
        if !owners.contains(&owner_object_id) {
            return Err(GeometryError::new(
                "selected object does not directly control the FluidDomain boundary",
            ));
        }
        let owner_name = self.object(owner_object_id)?.name.clone();
        let patch = patch_id.and_then(|wanted| {
            crate::boundary_ops::surface_patches_for_root(&root)
                .into_iter()
                .find(|patch| {
                    patch.owner_object_id == owner_object_id && patch.patch_id == wanted
                })
        });
        if patch_id.is_some() && patch.is_none() {
            return Err(GeometryError::new(
                "selected boundary patch is not part of the FluidDomain",
            ));
        }
        let name = match (patch_id, outside_direction) {
            (Some(patch), _) => format!("{owner_name} {patch}"),
            (None, Some(direction)) => format!("{owner_name} boundary {direction}"),
            (None, None) => format!("{owner_name} boundary"),
        };
        let object_id = self.allocate_object_id()?;
        let mut region = BoundaryRegion::new(name, object_id, owner_object_id);
        region.outside_direction =
            outside_direction.or(patch.as_ref().and_then(|p| p.outside_direction));
        region.patch_id = patch_id.map(str::to_string);
        region.patch_type = patch_type
            .map(str::to_string)
            .or(patch.as_ref().map(|p| p.patch_type.clone()));
        self.boundary_regions.push(region);
        if let Some(fluid) = self.fluid_domain.as_mut() {
            fluid.tags.push(TagRef::Region(object_id));
        }
        self.mark_changed();
        Ok(object_id)
    }

    /// Split a BoundaryRegion with a ghost knife (Python
    /// `split_boundary_region`, boundary_region_v2 §2). Children partition
    /// and REPLACE the parent; the ghost never becomes a scene object; a
    /// knife that misses either side refuses the split.
    pub fn split_boundary_region(
        &mut self,
        region_object_id: u32,
        ghost: &Node,
        validation_points: Option<&[Vec3]>,
    ) -> GeometryResult<(u32, u32)> {
        let fluid = self
            .fluid_domain
            .clone()
            .ok_or_else(|| GeometryError::new("select a FluidDomain root first"))?;
        let base = self
            .boundary_regions
            .iter()
            .find(|region| region.object_id == region_object_id)
            .cloned()
            .ok_or_else(|| {
                GeometryError::new("base boundary region is not part of this document")
            })?;
        if base.selector_start.is_some() {
            return Err(GeometryError::new(
                "interval-selector regions cannot be split; recreate them with the boundary cutter",
            ));
        }
        let mut base_cuts = base.cuts.clone();
        if base.selector_id.is_some() {
            let legacy = self.legacy_selector_cut(&base)?.ok_or_else(|| {
                GeometryError::new(
                    "this region references a selector that is no longer available",
                )
            })?;
            base_cuts.insert(0, legacy);
        }
        let root = self.build_node(fluid.root)?;
        let cut_index = base_cuts.len() + 1;
        let mut children = Vec::new();
        for side in [CutSide::Inside, CutSide::Outside] {
            let mut knife = ghost.clone();
            knife.object_id = 0;
            let mut child = BoundaryRegion::new(
                format!("{} / cut{cut_index} {}", base.name, side.as_str()),
                self.allocate_object_id()?,
                base.owner_object_id,
            );
            child.outside_direction = base.outside_direction;
            child.patch_id = base.patch_id.clone();
            child.patch_type = base.patch_type.clone();
            child.cuts = base_cuts
                .iter()
                .cloned()
                .chain([BoundaryCut { side, ghost: knife }])
                .collect();
            child.tag = base.tag.clone();
            children.push(child);
        }
        // Validate BEFORE mutating: both sides must select boundary points.
        let mut point_sets: Vec<(Vec<Vec3>, Option<f64>)> = Vec::new();
        if let Ok((samples, band)) = crate::boundary_ops::sample_boundary_points(&root, 24) {
            if !samples.is_empty() {
                point_sets.push((samples, Some(band)));
            }
        }
        let band = point_sets.first().and_then(|(_, band)| *band);
        if let Some(dense) = validation_points {
            if !dense.is_empty() {
                point_sets.push((dense.to_vec(), band));
            }
        }
        for child in &children {
            let populated = point_sets.iter().any(|(points, tolerance)| {
                crate::boundary_ops::boundary_region_mask(&root, child, points, *tolerance)
                    .map(|mask| mask.iter().any(|hit| *hit))
                    .unwrap_or(false)
            });
            if !point_sets.is_empty() && !populated {
                let side = child.cuts.last().expect("cut").side.as_str();
                return Err(GeometryError::new(format!(
                    "the knife does not cross the selected region (its '{side}' side selects \
                     no boundary points); nothing was cut"
                )));
            }
        }
        let child_ids = (children[0].object_id, children[1].object_id);
        self.boundary_regions
            .retain(|region| region.object_id != region_object_id);
        self.boundary_regions.extend(children);
        if let Some(fluid) = self.fluid_domain.as_mut() {
            fluid.tags.retain(|tag| *tag != TagRef::Region(region_object_id));
            fluid.tags.push(TagRef::Region(child_ids.0));
            fluid.tags.push(TagRef::Region(child_ids.1));
        }
        self.mark_changed();
        Ok(child_ids)
    }

    /// Convert a legacy single-selector reference into a cut-chain entry
    /// (Python `_legacy_selector_cut`).
    fn legacy_selector_cut(&self, region: &BoundaryRegion) -> GeometryResult<Option<BoundaryCut>> {
        let Some(selector_id) = region.selector_id.as_deref() else {
            return Ok(None);
        };
        let Some(id_text) = selector_id.strip_prefix("selector:") else {
            return Ok(None);
        };
        let Ok(selector_object_id) = id_text.parse::<ObjectId>() else {
            return Ok(None);
        };
        let Some(fluid) = &self.fluid_domain else {
            return Ok(None);
        };
        if !fluid.selectors.contains(&selector_object_id)
            && self.object(selector_object_id).is_err()
        {
            return Ok(None);
        }
        let Ok(mut knife) = self.build_node(selector_object_id) else {
            return Ok(None);
        };
        knife.object_id = 0;
        Ok(Some(BoundaryCut {
            side: region.selector_side,
            ghost: knife,
        }))
    }

    /// Copy a subtree from another document into this one with fresh ids,
    /// appended as a root (the paste half of Python copy/paste — the
    /// clipboard holds a document snapshot).
    pub fn import_subtree(
        &mut self,
        source: &SceneDocument,
        id: ObjectId,
        offset: Vec3,
    ) -> GeometryResult<ObjectId> {
        let imported = self.import_subtree_inner(source, id)?;
        let name = format!("{} copy", self.object(imported)?.name);
        self.object_mut(imported)?.name = name.clone();
        self.roots.push(imported);
        let moved = self.move_object(imported, offset)?;
        if moved != imported {
            self.object_mut(moved)?.name = name;
        }
        self.refresh_fluid_domain();
        self.mark_changed();
        Ok(moved)
    }

    fn import_subtree_inner(
        &mut self,
        source: &SceneDocument,
        id: ObjectId,
    ) -> GeometryResult<ObjectId> {
        let object = source.object(id)?.clone();
        let mut payload = object.payload;
        match &mut payload {
            ScenePayload::Operator { left, right, .. } => {
                *left = self.import_subtree_inner(source, *left)?;
                *right = self.import_subtree_inner(source, *right)?;
            }
            ScenePayload::Translate { child, .. }
            | ScenePayload::Rotate { child, .. }
            | ScenePayload::Scale { child, .. } => {
                *child = self.import_subtree_inner(source, *child)?;
            }
            ScenePayload::Extrude { section, .. } | ScenePayload::Revolve { section, .. } => {
                *section = self.import_subtree_inner(source, *section)?;
            }
            ScenePayload::Placed2D { sources, .. } | ScenePayload::Placed1D { sources, .. } => {
                for source_id in sources.iter_mut() {
                    *source_id = self.import_subtree_inner(source, *source_id)?;
                }
            }
            _ => {}
        }
        self.insert_object(object.name, payload)
    }

    /// Recursively deep-copy a subtree, allocating fresh ids (not attached
    /// to `roots`).
    fn clone_subtree(&mut self, id: ObjectId) -> GeometryResult<ObjectId> {
        let source = self.object(id)?.clone();
        let mut payload = source.payload;
        match &mut payload {
            ScenePayload::Operator { left, right, .. } => {
                *left = self.clone_subtree(*left)?;
                *right = self.clone_subtree(*right)?;
            }
            ScenePayload::Translate { child, .. }
            | ScenePayload::Rotate { child, .. }
            | ScenePayload::Scale { child, .. } => {
                *child = self.clone_subtree(*child)?;
            }
            ScenePayload::Extrude { section, .. } | ScenePayload::Revolve { section, .. } => {
                *section = self.clone_subtree(*section)?;
            }
            ScenePayload::Placed2D { sources, .. } | ScenePayload::Placed1D { sources, .. } => {
                for source_id in sources.iter_mut() {
                    *source_id = self.clone_subtree(*source_id)?;
                }
            }
            _ => {}
        }
        self.insert_object(source.name, payload)
    }
}

fn component(vector: Vec3, axis: usize) -> f64 {
    match axis {
        0 => vector.x,
        1 => vector.y,
        _ => vector.z,
    }
}

fn set_component(vector: &mut Vec3, axis: usize, value: f64) {
    match axis {
        0 => vector.x = value,
        1 => vector.y = value,
        _ => vector.z = value,
    }
}

fn axis_unit(axis: usize) -> Vec3 {
    match axis {
        0 => vec3(1.0, 0.0, 0.0),
        1 => vec3(0.0, 1.0, 0.0),
        _ => vec3(0.0, 0.0, 1.0),
    }
}

/// Axes of the plane a drag spans (Python `_drag_plane_axes`).
fn drag_plane_axes(start: Vec3, end: Vec3) -> (usize, usize) {
    let delta = [
        (end.x - start.x).abs(),
        (end.y - start.y).abs(),
        (end.z - start.z).abs(),
    ];
    let tolerance = 1e-9;
    if delta[2] <= tolerance {
        return (0, 1);
    }
    if delta[1] <= tolerance {
        return (0, 2);
    }
    if delta[0] <= tolerance {
        return (1, 2);
    }
    let mut order = [0usize, 1, 2];
    order.sort_by(|a, b| delta[*a].partial_cmp(&delta[*b]).expect("finite"));
    let mut top = [order[1], order[2]];
    top.sort();
    (top[0], top[1])
}

/// `REFERENCE_PLANE_AXES_3D` from `core/scene.py`.
fn reference_plane_axes(plane: &str) -> GeometryResult<(Vec3, Vec3)> {
    match plane {
        "xy" => Ok((vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0))),
        "xz" => Ok((vec3(1.0, 0.0, 0.0), vec3(0.0, 0.0, 1.0))),
        "yz" => Ok((vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0))),
        other => Err(GeometryError::new(format!("unknown reference plane: {other}"))),
    }
}

fn rotate_about(vector: Vec3, axis: RotationAxis, angle_degrees: f64) -> Vec3 {
    let angle = angle_degrees.to_radians();
    let c = angle.cos();
    let s = angle.sin();
    match axis {
        RotationAxis::X => vec3(vector.x, c * vector.y - s * vector.z, s * vector.y + c * vector.z),
        RotationAxis::Y => vec3(
            c * vector.x + s * vector.z,
            vector.y,
            -s * vector.x + c * vector.z,
        ),
        RotationAxis::Z => vec3(
            c * vector.x - s * vector.y,
            s * vector.x + c * vector.y,
            vector.z,
        ),
    }
}
