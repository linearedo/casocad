use std::collections::{BTreeMap, BTreeSet};

use crate::delaunay_3d::{ordered_face, tetrahedron_faces, TetTopology};
use crate::predicates::{orient3d, Sign};
use crate::{
    continued, ConstraintPolicy, Error, Input3d, Limits, SegmentConstraint, Tetrahedralization3d,
};

#[derive(Clone, Copy)]
struct WorkSegment {
    vertices: [usize; 2],
    policy: ConstraintPolicy,
    owner: usize,
}

#[derive(Clone, Copy)]
struct WorkFacet {
    vertices: [usize; 3],
    policy: ConstraintPolicy,
    owner: usize,
}

pub(crate) fn validate_constraints_3d(input: &Input3d) -> Result<(), Error> {
    for (index, segment) in input.segments.iter().enumerate() {
        validate_segment(index, segment, input.points.len())?;
    }
    let mut incidence = BTreeMap::<[usize; 2], usize>::new();
    for (index, facet) in input.facets.iter().enumerate() {
        let [a, b, c] = facet.vertices;
        if a >= input.points.len()
            || b >= input.points.len()
            || c >= input.points.len()
            || a == b
            || b == c
            || c == a
        {
            return Err(Error::InvalidConstraint {
                constraint: index,
                reason: "facet vertices must be distinct existing vertices",
            });
        }
        if collinear3(input.points[a], input.points[b], input.points[c]) {
            return Err(Error::InvalidConstraint {
                constraint: index,
                reason: "facet is degenerate",
            });
        }
        for edge in [[a, b], [b, c], [c, a]] {
            *incidence.entry(ordered_edge(edge)).or_default() += 1;
        }
    }
    if !input.facets.is_empty() {
        if let Some((edge, count)) = incidence.into_iter().find(|(_, count)| *count != 2) {
            return Err(Error::NonManifoldSurface {
                edge,
                incidence: count,
            });
        }
    }
    Ok(())
}

fn validate_segment(index: usize, segment: &SegmentConstraint, points: usize) -> Result<(), Error> {
    let [a, b] = segment.vertices;
    if a >= points || b >= points || a == b {
        Err(Error::InvalidConstraint {
            constraint: index,
            reason: "segment endpoints must be distinct existing vertices",
        })
    } else {
        Ok(())
    }
}

pub(crate) fn recover_constraints(
    input: &Input3d,
    limits: Limits,
    check: &mut impl FnMut() -> bool,
    mut tetrahedralize: impl FnMut(&[[f64; 3]], &mut dyn FnMut() -> bool) -> Result<TetTopology, Error>,
) -> Result<Tetrahedralization3d, Error> {
    let mut points = input.points.clone();
    let mut segments = input
        .segments
        .iter()
        .enumerate()
        .map(|(owner, segment)| WorkSegment {
            vertices: segment.vertices,
            policy: segment.policy,
            owner,
        })
        .collect::<Vec<_>>();
    let mut facets = input
        .facets
        .iter()
        .enumerate()
        .map(|(owner, facet)| WorkFacet {
            vertices: facet.vertices,
            policy: facet.policy,
            owner,
        })
        .collect::<Vec<_>>();

    for iteration in 0..=limits.max_iterations {
        if iteration.is_multiple_of(16) {
            continued(check)?;
        }
        let (tetrahedra, adjacency) = tetrahedralize(&points, check)?;
        let edges = tetrahedra
            .iter()
            .flat_map(|tetrahedron| {
                [
                    [tetrahedron[0], tetrahedron[1]],
                    [tetrahedron[0], tetrahedron[2]],
                    [tetrahedron[0], tetrahedron[3]],
                    [tetrahedron[1], tetrahedron[2]],
                    [tetrahedron[1], tetrahedron[3]],
                    [tetrahedron[2], tetrahedron[3]],
                ]
                .map(ordered_edge)
            })
            .collect::<BTreeSet<_>>();
        let face_set = tetrahedra
            .iter()
            .flat_map(|tetrahedron| tetrahedron_faces(*tetrahedron).map(ordered_face))
            .collect::<BTreeSet<_>>();

        if let Some((index, segment)) = segments.iter().copied().enumerate().find(|(_, segment)| {
            segment_chain_3d(&points, segment.vertices)
                .iter()
                .any(|edge| !edges.contains(edge))
        }) {
            if segment.policy == ConstraintPolicy::Fixed {
                return Err(Error::UnrecoverableConstraint {
                    constraint: segment.owner,
                    dimension: 3,
                });
            }
            split_segment(&mut points, &mut segments, index, segment, limits)?;
            continue;
        }

        let patches = facets
            .iter()
            .map(|facet| facet_patch(&points, &face_set, facet.vertices))
            .collect::<Vec<_>>();
        if let Some(index) = patches.iter().position(Vec::is_empty) {
            let facet = facets[index];
            if facet.policy == ConstraintPolicy::Fixed {
                return Err(Error::UnrecoverableConstraint {
                    constraint: facet.owner,
                    dimension: 3,
                });
            }
            split_facet(&mut points, &mut facets, index, facet, limits)?;
            continue;
        }

        let mut recovered_segments = vec![Vec::new(); input.segments.len()];
        for segment in &segments {
            recovered_segments[segment.owner].extend(segment_chain_3d(&points, segment.vertices));
        }
        for recovered in &mut recovered_segments {
            recovered.sort_unstable();
            recovered.dedup();
        }
        let mut recovered_facets = vec![Vec::new(); input.facets.len()];
        for (facet, patch) in facets.iter().zip(patches) {
            recovered_facets[facet.owner].extend(patch);
        }
        for recovered in &mut recovered_facets {
            recovered.sort_unstable();
            recovered.dedup();
        }
        return Ok(Tetrahedralization3d {
            vertices: points,
            tetrahedra,
            adjacency,
            segments: recovered_segments,
            facets: recovered_facets,
        });
    }
    Err(Error::LimitExceeded("constraint recovery"))
}

fn split_segment(
    points: &mut Vec<[f64; 3]>,
    segments: &mut Vec<WorkSegment>,
    index: usize,
    segment: WorkSegment,
    limits: Limits,
) -> Result<(), Error> {
    if points.len() == limits.max_vertices {
        return Err(Error::LimitExceeded("vertex"));
    }
    let [a, b] = segment.vertices.map(|vertex| points[vertex]);
    let midpoint = [
        (a[0] + b[0]) * 0.5,
        (a[1] + b[1]) * 0.5,
        (a[2] + b[2]) * 0.5,
    ];
    if midpoint == a || midpoint == b {
        return Err(Error::UnrecoverableConstraint {
            constraint: segment.owner,
            dimension: 3,
        });
    }
    let middle = points.len();
    points.push(midpoint);
    segments.splice(
        index..=index,
        [
            WorkSegment {
                vertices: [segment.vertices[0], middle],
                ..segment
            },
            WorkSegment {
                vertices: [middle, segment.vertices[1]],
                ..segment
            },
        ],
    );
    Ok(())
}

fn split_facet(
    points: &mut Vec<[f64; 3]>,
    facets: &mut Vec<WorkFacet>,
    index: usize,
    facet: WorkFacet,
    limits: Limits,
) -> Result<(), Error> {
    if points.len() == limits.max_vertices {
        return Err(Error::LimitExceeded("vertex"));
    }
    let [a, b, c] = facet.vertices.map(|vertex| points[vertex]);
    let center = [
        (a[0] + b[0] + c[0]) / 3.0,
        (a[1] + b[1] + c[1]) / 3.0,
        (a[2] + b[2] + c[2]) / 3.0,
    ];
    if center == a || center == b || center == c {
        return Err(Error::UnrecoverableConstraint {
            constraint: facet.owner,
            dimension: 3,
        });
    }
    let middle = points.len();
    points.push(center);
    facets.splice(
        index..=index,
        [
            WorkFacet {
                vertices: [facet.vertices[0], facet.vertices[1], middle],
                ..facet
            },
            WorkFacet {
                vertices: [facet.vertices[1], facet.vertices[2], middle],
                ..facet
            },
            WorkFacet {
                vertices: [facet.vertices[2], facet.vertices[0], middle],
                ..facet
            },
        ],
    );
    Ok(())
}

fn facet_patch(
    points: &[[f64; 3]],
    faces: &BTreeSet<[usize; 3]>,
    facet: [usize; 3],
) -> Vec<[usize; 3]> {
    let [a, b, c] = facet.map(|vertex| points[vertex]);
    let mut patch = faces
        .iter()
        .copied()
        .filter(|face| {
            face.iter().all(|vertex| {
                orient3d(a, b, c, points[*vertex]) == Sign::Zero
                    && point_in_triangle_3d(points[*vertex], a, b, c)
            })
        })
        .collect::<Vec<_>>();
    let target_area = triangle_area(a, b, c);
    let patch_area = patch
        .iter()
        .map(|face| triangle_area(points[face[0]], points[face[1]], points[face[2]]))
        .sum::<f64>();
    if target_area == 0.0 || (patch_area - target_area).abs() > target_area * 1.0e-10 {
        patch.clear();
    }
    patch
}

fn point_in_triangle_3d(point: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> bool {
    let normal = cross(subtract(b, a), subtract(c, a));
    let axis = (0..3)
        .max_by(|left, right| normal[*left].abs().total_cmp(&normal[*right].abs()))
        .unwrap_or(2);
    let project = |value: [f64; 3]| match axis {
        0 => [value[1], value[2]],
        1 => [value[0], value[2]],
        _ => [value[0], value[1]],
    };
    let [p, a, b, c] = [point, a, b, c].map(project);
    let signs = [
        crate::predicates::orient2d(a, b, p),
        crate::predicates::orient2d(b, c, p),
        crate::predicates::orient2d(c, a, p),
    ];
    !signs.contains(&Sign::Positive) || !signs.contains(&Sign::Negative)
}

fn segment_chain_3d(points: &[[f64; 3]], segment: [usize; 2]) -> Vec<[usize; 2]> {
    let [a, b] = segment.map(|vertex| points[vertex]);
    let direction = subtract(b, a);
    let axis = (0..3)
        .max_by(|left, right| direction[*left].abs().total_cmp(&direction[*right].abs()))
        .unwrap_or(0);
    let mut vertices = points
        .iter()
        .enumerate()
        .filter_map(|(index, point)| {
            let offset = subtract(*point, a);
            let collinear = cross(direction, offset)
                .into_iter()
                .all(|value| value == 0.0);
            (collinear
                && (0..3).all(|coordinate| {
                    point[coordinate] >= a[coordinate].min(b[coordinate])
                        && point[coordinate] <= a[coordinate].max(b[coordinate])
                }))
            .then_some(index)
        })
        .collect::<Vec<_>>();
    vertices.sort_by(|left, right| {
        points[*left][axis]
            .total_cmp(&points[*right][axis])
            .then(left.cmp(right))
    });
    if points[segment[0]][axis] > points[segment[1]][axis] {
        vertices.reverse();
    }
    vertices
        .windows(2)
        .map(|pair| ordered_edge([pair[0], pair[1]]))
        .collect()
}

fn collinear3(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> bool {
    cross(subtract(b, a), subtract(c, a))
        .into_iter()
        .all(|value| value == 0.0)
}

fn triangle_area(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let cross = cross(subtract(b, a), subtract(c, a));
    0.5 * cross
        .into_iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt()
}

fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn ordered_edge(mut edge: [usize; 2]) -> [usize; 2] {
    edge.sort_unstable();
    edge
}
