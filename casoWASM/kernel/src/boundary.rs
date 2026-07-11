//! Boundary regions: named subsets of a Domain boundary, identified by owner
//! surface + optional analytic patch scope + an ordered cut chain
//! (boundary_region_v2 §2). Ported from `core/boundary.py`.

use crate::error::{GeometryError, GeometryResult};
use crate::sdf::node::Node;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CutSide {
    Inside,
    Outside,
}

impl CutSide {
    pub fn parse(side: &str) -> GeometryResult<Self> {
        match side {
            "inside" => Ok(Self::Inside),
            "outside" => Ok(Self::Outside),
            _ => Err(GeometryError::new("cut side must be 'inside' or 'outside'")),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Inside => "inside",
            Self::Outside => "outside",
        }
    }
}

/// One knife-half in a region's cut chain (boundary_region_v2 §2).
///
/// `ghost` is detached selector geometry: never part of the scene graph,
/// never rendered; it is embedded in the region's serialized record. A 3D
/// ghost classifies by its own sign; a lower-dimensional ghost is extruded
/// through the scene at classification time.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryCut {
    pub side: CutSide,
    pub ghost: Node,
}

/// A named subset of a Domain boundary (boundary_region_v2 §2).
///
/// Identity = owner surface (provenance) + optional analytic patch scope +
/// the ordered chain of cuts that carved it. `tag` is an opaque physics label
/// the kernel never interprets. The `selector_*`/`outside_direction` fields
/// are the legacy single-selector schema, kept readable until every
/// creation/load path emits cut chains.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryRegion {
    pub name: String,
    pub object_id: u32,
    pub owner_object_id: u32,
    pub outside_direction: Option<u8>,
    pub patch_id: Option<String>,
    pub patch_type: Option<String>,
    pub selector_id: Option<String>,
    pub selector_type: Option<String>,
    pub selector_side: CutSide,
    pub selector_start: Option<f64>,
    pub selector_end: Option<f64>,
    pub cuts: Vec<BoundaryCut>,
    pub tag: Option<String>,
}

impl BoundaryRegion {
    pub fn new(name: impl Into<String>, object_id: u32, owner_object_id: u32) -> Self {
        Self {
            name: name.into(),
            object_id,
            owner_object_id,
            outside_direction: None,
            patch_id: None,
            patch_type: None,
            selector_id: None,
            selector_type: None,
            selector_side: CutSide::Inside,
            selector_start: None,
            selector_end: None,
            cuts: Vec::new(),
            tag: None,
        }
    }

    pub fn validate(&self) -> GeometryResult<()> {
        if let Some(direction) = self.outside_direction {
            if direction >= 6 {
                return Err(GeometryError::new(
                    "outside_direction must be in the range 0..5",
                ));
            }
        }
        if self.selector_start.is_some() != self.selector_end.is_some() {
            return Err(GeometryError::new(
                "selector_start and selector_end must be set together",
            ));
        }
        Ok(())
    }

    pub fn kind(&self) -> &'static str {
        "boundary_region"
    }

    pub fn dimension(&self) -> u8 {
        2
    }
}
