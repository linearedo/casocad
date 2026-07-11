//! The exact-geometry Model: a set of disjoint, named Domains (spec v2 §2,
//! §7), ported from `core/model.py`.
//!
//! Two invariants make a Model *compilable* (`compile_model`):
//!
//! 1. **Exactness grammar** — every Domain's region satisfies the
//!    slot-exactness grammar (`roles`, §4): enforced by construction.
//! 2. **Disjointness** — the Domains are mutually disjoint (§7). Unlike
//!    exactness, this is a *checked* invariant: overlap is a compile error.
//!    `overlap(A, B)` iff some point is interior to both.

use crate::error::{GeometryError, GeometryResult};
use crate::preconditions::precondition_violations;
use crate::roles::{exactness_violations, Domain, Exactness};

/// Default samples per axis for the disjointness probe: a dense-ish grid over
/// the bounding-box overlap region (an interval backstop is future work).
pub const DEFAULT_RESOLUTION: usize = 32;

/// A set of named Domains. Names must be unique; disjointness is checked at
/// compile time, not enforced here (§7).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Model {
    pub domains: Vec<Domain>,
}

impl Model {
    pub fn new(domains: Vec<Domain>) -> GeometryResult<Self> {
        for (index, domain) in domains.iter().enumerate() {
            if domains[..index].iter().any(|other| other.name == domain.name) {
                return Err(GeometryError::new(
                    "Domain names must be unique within a Model",
                ));
            }
        }
        Ok(Self { domains })
    }
}

fn linspace(minimum: f64, maximum: f64, count: usize) -> impl Iterator<Item = f64> {
    let step = if count > 1 {
        (maximum - minimum) / (count - 1) as f64
    } else {
        0.0
    };
    (0..count).map(move |index| minimum + step * index as f64)
}

/// True if Domains `a` and `b` share interior volume (§7).
///
/// Overlap can only occur inside both regions, hence inside the intersection
/// of their bounding boxes; disjoint boxes are the fast path. Otherwise a
/// grid is sampled in the overlap box and the domains overlap iff some sample
/// is interior to both. Domains that merely touch (share a boundary with
/// disjoint open interiors) are correctly reported as non-overlapping.
pub fn domains_overlap(a: &Domain, b: &Domain, resolution: usize) -> GeometryResult<bool> {
    if resolution < 2 {
        return Err(GeometryError::new("resolution must be at least 2"));
    }
    let box_a = a.region.bounding_box()?;
    let box_b = b.region.bounding_box()?;
    let overlap = match box_a.intersection(&box_b) {
        Ok(overlap) => overlap,
        // Disjoint bounding boxes -> regions cannot share interior.
        Err(_) => return Ok(false),
    };
    for x in linspace(overlap.x_min, overlap.x_max, resolution) {
        for y in linspace(overlap.y_min, overlap.y_max, resolution) {
            for z in linspace(overlap.z_min, overlap.z_max, resolution) {
                let point = crate::vec3::vec3(x, y, z);
                if a.region.eval_point(point) < 0.0 && b.region.eval_point(point) < 0.0 {
                    return Ok(true);
                }
            }
        }
    }
    Ok(false)
}

/// Human-readable overlap reports for every overlapping Domain pair
/// (empty = all disjoint).
pub fn disjointness_violations(model: &Model, resolution: usize) -> GeometryResult<Vec<String>> {
    let mut violations = Vec::new();
    for (index, a) in model.domains.iter().enumerate() {
        for b in &model.domains[index + 1..] {
            if domains_overlap(a, b, resolution)? {
                violations.push(format!(
                    "Domains '{}' and '{}' overlap (share interior volume); \
                     Domains must be disjoint",
                    a.name, b.name
                ));
            }
        }
    }
    Ok(violations)
}

/// Exactness-grammar violations across all Domains (empty = OK).
///
/// The cheap half of `compile_model` — a pure tree walk (§4) with no
/// disjointness sampling. Suitable for live per-edit diagnostics.
pub fn grammar_violations(model: &Model) -> Vec<String> {
    let mut violations = Vec::new();
    for domain in &model.domains {
        for issue in exactness_violations(&domain.region, Some(Exactness::SDF_INSIDE)) {
            violations.push(format!("Domain '{}': {}", domain.name, issue));
        }
    }
    violations
}

/// Validate a Model's compile-time invariants; error on the first failure.
///
/// Checks, in order: (1) every Domain region satisfies the exactness grammar
/// (§4); (2) generator/offset preconditions (§5, §6); (3) the Domains are
/// mutually disjoint (§7).
pub fn compile_model(model: &Model, resolution: usize) -> GeometryResult<()> {
    for domain in &model.domains {
        let exactness_issues = exactness_violations(&domain.region, Some(Exactness::SDF_INSIDE));
        if !exactness_issues.is_empty() {
            return Err(GeometryError::new(format!(
                "Domain '{}' cannot be compiled for meshing:\n  {}",
                domain.name,
                exactness_issues.join("\n  ")
            )));
        }
    }
    for domain in &model.domains {
        let precondition_issues = precondition_violations(&domain.region);
        if !precondition_issues.is_empty() {
            return Err(GeometryError::new(format!(
                "Domain '{}' violates a generator/offset precondition:\n  {}",
                domain.name,
                precondition_issues.join("\n  ")
            )));
        }
    }
    let overlaps = disjointness_violations(model, resolution)?;
    if !overlaps.is_empty() {
        return Err(GeometryError::new(overlaps.join("\n")));
    }
    Ok(())
}
