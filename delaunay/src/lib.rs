//! Deterministic, dependency-free Delaunay triangulation primitives.
//!
//! Coordinates are `f64`, connectivity is expressed with `usize`, and every
//! topological predicate is evaluated with expansion arithmetic.

mod delaunay_2d;
mod delaunay_3d;
mod expansion;
pub mod predicates;
mod recover_3d;
mod refine_2d;
mod refine_3d;

use std::fmt;

pub use delaunay_2d::triangulate_2d;
pub use delaunay_3d::tetrahedralize_3d;
pub use refine_2d::refine_2d;
pub use refine_3d::refine_3d;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintPolicy {
    Splittable,
    Fixed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SegmentConstraint {
    pub vertices: [usize; 2],
    pub policy: ConstraintPolicy,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FacetConstraint {
    pub vertices: [usize; 3],
    pub policy: ConstraintPolicy,
}

#[derive(Debug, Clone)]
pub struct Input2d {
    pub points: Vec<[f64; 2]>,
    pub constraints: Vec<SegmentConstraint>,
}

#[derive(Debug, Clone)]
pub struct Input3d {
    pub points: Vec<[f64; 3]>,
    pub segments: Vec<SegmentConstraint>,
    pub facets: Vec<FacetConstraint>,
}

#[derive(Debug, Clone, Copy)]
pub struct Limits {
    pub max_vertices: usize,
    pub max_cells: usize,
    pub max_iterations: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Self {
            max_vertices: 1_000_000,
            max_cells: 4_000_000,
            max_iterations: 100_000,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Refine2d {
    pub max_area: f64,
    pub min_angle_degrees: f64,
    pub limits: Limits,
}

#[derive(Debug, Clone, Copy)]
pub struct Refine3d {
    pub max_volume: f64,
    pub limits: Limits,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Triangulation2d {
    pub vertices: Vec<[f64; 2]>,
    pub triangles: Vec<[usize; 3]>,
    pub adjacency: Vec<[Option<usize>; 3]>,
    /// Recovered edge chains, one entry per input constraint.
    pub constraints: Vec<Vec<[usize; 2]>>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tetrahedralization3d {
    pub vertices: Vec<[f64; 3]>,
    pub tetrahedra: Vec<[usize; 4]>,
    pub adjacency: Vec<[Option<usize>; 4]>,
    /// Recovered edge chains, one entry per input segment.
    pub segments: Vec<Vec<[usize; 2]>>,
    /// Recovered triangular patches, one entry per input facet.
    pub facets: Vec<Vec<[usize; 3]>>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    NonFiniteCoordinate {
        vertex: usize,
    },
    DuplicateVertex {
        first: usize,
        second: usize,
    },
    DegenerateInput(&'static str),
    InvalidConstraint {
        constraint: usize,
        reason: &'static str,
    },
    CrossedConstraints {
        first: usize,
        second: usize,
    },
    NonManifoldSurface {
        edge: [usize; 2],
        incidence: usize,
    },
    UnrecoverableConstraint {
        constraint: usize,
        dimension: u8,
    },
    Cancelled,
    LimitExceeded(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonFiniteCoordinate { vertex } => {
                write!(formatter, "vertex {vertex} has a non-finite coordinate")
            }
            Self::DuplicateVertex { first, second } => {
                write!(formatter, "vertices {first} and {second} are coincident")
            }
            Self::DegenerateInput(reason) => formatter.write_str(reason),
            Self::InvalidConstraint { constraint, reason } => {
                write!(formatter, "constraint {constraint} is invalid: {reason}")
            }
            Self::CrossedConstraints { first, second } => {
                write!(formatter, "constraints {first} and {second} cross")
            }
            Self::NonManifoldSurface { edge, incidence } => write!(
                formatter,
                "surface edge {:?} has non-manifold incidence {incidence}",
                edge
            ),
            Self::UnrecoverableConstraint {
                constraint,
                dimension,
            } => write!(
                formatter,
                "could not recover {dimension}D constraint {constraint}"
            ),
            Self::Cancelled => formatter.write_str("Delaunay operation cancelled"),
            Self::LimitExceeded(resource) => {
                write!(formatter, "Delaunay {resource} limit exceeded")
            }
        }
    }
}

impl std::error::Error for Error {}

pub(crate) fn check_points<const N: usize>(points: &[[f64; N]]) -> Result<(), Error> {
    for (index, point) in points.iter().enumerate() {
        if point.iter().any(|value| !value.is_finite()) {
            return Err(Error::NonFiniteCoordinate { vertex: index });
        }
        if let Some(first) = points[..index].iter().position(|other| other == point) {
            return Err(Error::DuplicateVertex {
                first,
                second: index,
            });
        }
    }
    Ok(())
}

pub(crate) fn continued(check: &mut (impl FnMut() -> bool + ?Sized)) -> Result<(), Error> {
    check().then_some(()).ok_or(Error::Cancelled)
}
