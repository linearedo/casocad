//! Exactness system for the exact-SDF CFD kernel, ported from
//! `core/sdf/roles.py`.
//!
//! casoCAD is a *safe geometry compiler*: the only expressible geometry is a
//! set of named, interior-exact-distance Domains (spec v2). Exactness is NOT
//! stored on nodes: each operator has fixed slots with required exactness,
//! and a node's result exactness is determined structurally by its top
//! operator. A leaf primitive (or an exact generator / transform thereof) is
//! exact on BOTH sides, so it can fill either slot — that is what lets the
//! same solid be a Domain root in one place and a subtraction cutter in
//! another.

use std::fmt;

use crate::error::GeometryError;
use crate::sdf::node::Node;

/// Which side of its boundary a field is exact on (spec §3).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Exactness {
    pub inside: bool,
    pub outside: bool,
}

impl Exactness {
    pub const NONE: Exactness = Exactness {
        inside: false,
        outside: false,
    };
    pub const SDF_INSIDE: Exactness = Exactness {
        inside: true,
        outside: false,
    };
    pub const SDF_OUTSIDE: Exactness = Exactness {
        inside: false,
        outside: true,
    };
    pub const SDF_BOTH: Exactness = Exactness {
        inside: true,
        outside: true,
    };

    /// `required in actual` in the Python Flag sense.
    pub fn contains(&self, required: Exactness) -> bool {
        (!required.inside || self.inside) && (!required.outside || self.outside)
    }

    pub fn label(&self) -> String {
        match (self.inside, self.outside) {
            (false, false) => "none".to_string(),
            (true, true) => "both".to_string(),
            (true, false) => "inside".to_string(),
            (false, true) => "outside".to_string(),
        }
    }
}

/// Physics tag on a top-level Domain (spec §2). Geometry rules are identical
/// for both kinds; the tag only selects the consuming solver downstream.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DomainKind {
    Fluid,
    Solid,
}

impl DomainKind {
    pub fn parse(kind: &str) -> Result<Self, GeometryError> {
        match kind {
            "fluid" => Ok(Self::Fluid),
            "solid" => Ok(Self::Solid),
            other => Err(GeometryError::new(format!("unknown domain kind {other:?}"))),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Fluid => "fluid",
            Self::Solid => "solid",
        }
    }
}

impl fmt::Display for DomainKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Operator slot signature: (left required, right required, result).
fn kind_signature(kind: &str) -> Option<(Exactness, Exactness, Exactness)> {
    match kind {
        "intersection" => Some((
            Exactness::SDF_INSIDE,
            Exactness::SDF_INSIDE,
            Exactness::SDF_INSIDE,
        )),
        "difference" => Some((
            Exactness::SDF_INSIDE,
            Exactness::SDF_OUTSIDE,
            Exactness::SDF_INSIDE,
        )),
        "union" => Some((
            Exactness::SDF_OUTSIDE,
            Exactness::SDF_OUTSIDE,
            Exactness::SDF_OUTSIDE,
        )),
        _ => None,
    }
}

/// XOR is intentionally not part of the exact compiler grammar: coincident or
/// touching boundaries can cancel, making the standard XOR SDF non-exact.
fn is_non_exact_operator(kind: &str) -> bool {
    kind == "xor"
}

/// Exactness-transparent unary transforms (isometry / uniform scale).
fn is_transform_kind(kind: &str) -> bool {
    matches!(kind, "translate" | "rotate" | "scale")
}

/// Exactness this node's result can provide (spec §4).
pub fn node_exactness(node: &Node) -> Exactness {
    let kind = node.kind();
    if is_non_exact_operator(kind) {
        return Exactness::NONE;
    }
    if let Some(signature) = kind_signature(kind) {
        return signature.2;
    }
    if is_transform_kind(kind) {
        let children = node.children();
        return match children.first() {
            Some(child) => node_exactness(child),
            None => Exactness::SDF_BOTH,
        };
    }
    Exactness::SDF_BOTH
}

fn node_label(node: &Node) -> String {
    format!("{} '{}'", node.kind(), node.name)
}

fn exactness_message(node: &Node, required: Exactness, actual: Exactness, context: &str) -> String {
    if node.kind() == "union" && required == Exactness::SDF_INSIDE {
        return format!(
            "Union {context} would lose exact interior distance. \
             Union preserves exact SDF distance outside the combined shapes, \
             but it does not generally preserve exact distance inside them. \
             A meshable Domain needs exact interior distance for meshing. \
             Use Difference/Intersection to build the Domain, or use this \
             Union only as a subtraction cutter."
        );
    }
    if required == Exactness::SDF_INSIDE {
        return format!(
            "{} {context} is not an exact interior distance field. A meshable \
             Domain needs exact interior SDF distance for meshing; this \
             expression provides {} exactness.",
            node_label(node),
            actual.label()
        );
    }
    if required == Exactness::SDF_OUTSIDE {
        return format!(
            "{} {context} is not an exact outside distance field. Subtraction \
             cutters must provide exact outside SDF distance; this expression \
             provides {} exactness.",
            node_label(node),
            actual.label()
        );
    }
    format!(
        "{} {context} does not provide the required {} exactness; it provides {} exactness.",
        node_label(node),
        required.label(),
        actual.label()
    )
}

fn non_exact_message(node: &Node) -> String {
    if node.kind() == "xor" {
        return format!(
            "XOR '{}' is not accepted for solver-ready Domains yet. casoCAD \
             does not currently prove the extra boundary-cancellation \
             conditions needed for XOR to preserve exact SDF distance. XOR is \
             still available for free SDF modeling, but not for compiled \
             meshing geometry.",
            node.name
        );
    }
    format!("{} is not part of the exact-SDF compiler grammar.", node_label(node))
}

/// Human-readable exactness violations in the subtree (empty = OK).
///
/// For every operator node, each operand must be able to fill its slot's
/// required exactness. Leaves fill either slot; a `union` result cannot fill
/// an inside-exact slot; etc. `required` applies to the subtree root
/// (defaults to inside-exact — the meshable-Domain requirement).
pub fn exactness_violations(node: &Node, required: Option<Exactness>) -> Vec<String> {
    let mut violations = Vec::new();
    let mut reported_non_exact: Vec<*const Node> = Vec::new();
    let root_exactness = node_exactness(node);
    if let Some(required) = required {
        if !root_exactness.contains(required) {
            if is_non_exact_operator(node.kind()) {
                violations.push(non_exact_message(node));
                reported_non_exact.push(node as *const Node);
            } else {
                violations.push(exactness_message(
                    node,
                    required,
                    root_exactness,
                    "cannot define a meshable Domain",
                ));
            }
        }
    }

    fn visit(
        n: &Node,
        violations: &mut Vec<String>,
        reported_non_exact: &mut Vec<*const Node>,
    ) {
        if let Some((want_left, want_right, _)) = kind_signature(n.kind()) {
            let children = n.children();
            if children.len() == 2 {
                for (child, want, slot) in [
                    (children[0], want_left, "left"),
                    (children[1], want_right, "right"),
                ] {
                    let child_exactness = node_exactness(child);
                    if !child_exactness.contains(want) {
                        let context = format!(
                            "cannot be used as the {slot} operand of {} '{}'",
                            n.kind(),
                            n.name
                        );
                        violations.push(exactness_message(child, want, child_exactness, &context));
                    }
                }
            }
        } else if is_non_exact_operator(n.kind())
            && !reported_non_exact.contains(&(n as *const Node))
        {
            violations.push(non_exact_message(n));
            reported_non_exact.push(n as *const Node);
        }
        for child in n.children() {
            visit(child, violations, reported_non_exact);
        }
    }

    visit(node, &mut violations, &mut reported_non_exact);
    violations
}

/// Errors if the subtree has any exactness violation (spec §4).
pub fn validate_exactness(node: &Node) -> Result<(), GeometryError> {
    let violations = exactness_violations(node, Some(Exactness::SDF_INSIDE));
    if violations.is_empty() {
        return Ok(());
    }
    Err(GeometryError::new(format!(
        "scene cannot be compiled as solver-ready exact geometry:\n  {}",
        violations.join("\n  ")
    )))
}

/// A named, exported top-level cell (spec §2). `region` is the inside-exact
/// SDF root; `kind` is the physics tag. Domain-level disjointness is a
/// Model-level check, not enforced here.
#[derive(Debug, Clone, PartialEq)]
pub struct Domain {
    pub name: String,
    pub kind: DomainKind,
    pub region: Node,
}

impl Domain {
    pub fn new(name: impl Into<String>, kind: DomainKind, region: Node) -> Result<Self, GeometryError> {
        let name = name.into();
        if name.is_empty() {
            return Err(GeometryError::new("Domain requires a non-empty name"));
        }
        Ok(Self { name, kind, region })
    }
}
