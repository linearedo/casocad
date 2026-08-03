use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};

use crate::predicates::{incircle, orient2d, tie, Sign};
use crate::{check_points, continued, ConstraintPolicy, Error, Input2d, Limits, Triangulation2d};

#[derive(Clone, Copy)]
struct WorkSegment {
    vertices: [usize; 2],
    policy: ConstraintPolicy,
    owner: usize,
}

pub fn triangulate_2d(
    input: &Input2d,
    limits: Limits,
    mut check: impl FnMut() -> bool,
) -> Result<Triangulation2d, Error> {
    check_points(&input.points)?;
    if input.points.len() < 3 {
        return Err(Error::DegenerateInput(
            "2D triangulation requires at least three vertices",
        ));
    }
    if input.points.len() > limits.max_vertices {
        return Err(Error::LimitExceeded("vertex"));
    }
    validate_constraints(input)?;
    continued(&mut check)?;

    let mut points = input.points.clone();
    let mut segments = input
        .constraints
        .iter()
        .enumerate()
        .map(|(owner, constraint)| WorkSegment {
            vertices: constraint.vertices,
            policy: constraint.policy,
            owner,
        })
        .collect::<Vec<_>>();

    for recovery in 0..=limits.max_iterations {
        if recovery.is_multiple_of(32) {
            continued(&mut check)?;
        }
        let mut triangles = bowyer_watson(&points, limits, &mut check)?;
        let mut fixed = BTreeSet::new();
        let mut failed = None;
        for (index, segment) in segments.iter().enumerate() {
            for edge in segment_chain(&points, segment.vertices) {
                if !recover_edge(
                    &points,
                    &mut triangles,
                    edge,
                    &fixed,
                    limits.max_iterations,
                    &mut check,
                )? {
                    failed = Some(index);
                    break;
                }
                fixed.insert(ordered(edge));
            }
            if failed.is_some() {
                break;
            }
        }
        if let Some(index) = failed {
            let segment = segments[index];
            if segment.policy == ConstraintPolicy::Fixed {
                return Err(Error::UnrecoverableConstraint {
                    constraint: segment.owner,
                    dimension: 2,
                });
            }
            if points.len() == limits.max_vertices || recovery == limits.max_iterations {
                return Err(Error::LimitExceeded("constraint recovery"));
            }
            let [a, b] = segment.vertices.map(|vertex| points[vertex]);
            let midpoint = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
            if midpoint == a || midpoint == b {
                return Err(Error::UnrecoverableConstraint {
                    constraint: segment.owner,
                    dimension: 2,
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
            continue;
        }

        triangles.sort_unstable();
        let adjacency = triangle_adjacency(&triangles)?;
        let mut recovered = vec![Vec::new(); input.constraints.len()];
        for segment in &segments {
            recovered[segment.owner].extend(segment_chain(&points, segment.vertices));
        }
        for edges in &mut recovered {
            edges.sort_unstable();
            edges.dedup();
        }
        return Ok(Triangulation2d {
            vertices: points,
            triangles,
            adjacency,
            constraints: recovered,
        });
    }
    Err(Error::LimitExceeded("constraint recovery"))
}

fn validate_constraints(input: &Input2d) -> Result<(), Error> {
    for (index, constraint) in input.constraints.iter().enumerate() {
        let [a, b] = constraint.vertices;
        if a >= input.points.len() || b >= input.points.len() || a == b {
            return Err(Error::InvalidConstraint {
                constraint: index,
                reason: "segment endpoints must be distinct existing vertices",
            });
        }
    }
    for first in 0..input.constraints.len() {
        for second in first + 1..input.constraints.len() {
            let a = input.constraints[first].vertices;
            let b = input.constraints[second].vertices;
            if a.into_iter().any(|vertex| b.contains(&vertex)) {
                continue;
            }
            if segments_intersect(
                input.points[a[0]],
                input.points[a[1]],
                input.points[b[0]],
                input.points[b[1]],
            ) {
                return Err(Error::CrossedConstraints { first, second });
            }
        }
    }
    Ok(())
}

fn bowyer_watson(
    points: &[[f64; 2]],
    limits: Limits,
    check: &mut impl FnMut() -> bool,
) -> Result<Vec<[usize; 3]>, Error> {
    let (min, max) = bounds(points);
    let span = (max[0] - min[0]).max(max[1] - min[1]);
    if span == 0.0 || !span.is_finite() {
        return Err(Error::DegenerateInput(
            "2D vertices are collinear or unbounded",
        ));
    }
    let center = [(min[0] + max[0]) * 0.5, (min[1] + max[1]) * 0.5];
    let mut vertices = points.to_vec();
    let first_super = vertices.len();
    vertices.extend([
        [center[0] - 32.0 * span, center[1] - 16.0 * span],
        [center[0] + 32.0 * span, center[1] - 16.0 * span],
        [center[0], center[1] + 32.0 * span],
    ]);
    if vertices[first_super..]
        .iter()
        .flatten()
        .any(|value| !value.is_finite())
    {
        return Err(Error::DegenerateInput(
            "2D coordinate range cannot form a finite enclosing simplex",
        ));
    }
    let mut triangles = vec![[first_super, first_super + 1, first_super + 2]];
    let mut order = (0..points.len()).collect::<Vec<_>>();
    order.sort_by(|a, b| {
        points[*a][0]
            .total_cmp(&points[*b][0])
            .then(points[*a][1].total_cmp(&points[*b][1]))
            .then(a.cmp(b))
    });
    for (step, point) in order.into_iter().enumerate() {
        if step.is_multiple_of(128) {
            continued(check)?;
        }
        let mut bad = Vec::new();
        for (index, triangle) in triangles.iter().enumerate() {
            let [a, b, c] = triangle.map(|vertex| vertices[vertex]);
            let inside = match incircle(a, b, c, vertices[point]).ordering() {
                Ordering::Greater => true,
                Ordering::Equal => {
                    tie(&[triangle[0], triangle[1], triangle[2], point]) == Ordering::Greater
                }
                Ordering::Less => false,
            };
            if inside {
                bad.push(index);
            }
        }
        let mut boundary = BTreeMap::<[usize; 2], usize>::new();
        for &index in &bad {
            let triangle = triangles[index];
            for edge in triangle_edges(triangle) {
                *boundary.entry(ordered(edge)).or_default() += 1;
            }
        }
        let bad = bad.into_iter().collect::<BTreeSet<_>>();
        triangles = triangles
            .into_iter()
            .enumerate()
            .filter_map(|(index, triangle)| (!bad.contains(&index)).then_some(triangle))
            .collect();
        for (edge, incidence) in boundary {
            if incidence != 1 {
                continue;
            }
            if let Some(triangle) = oriented_triangle(edge[0], edge[1], point, &vertices) {
                triangles.push(triangle);
            }
        }
        if triangles.len() > limits.max_cells.saturating_mul(8).max(64) {
            return Err(Error::LimitExceeded("cell"));
        }
    }
    triangles.retain(|triangle| triangle.iter().all(|vertex| *vertex < first_super));
    triangles.sort_unstable();
    triangles.dedup();
    if triangles.is_empty() {
        return Err(Error::DegenerateInput("2D vertices are collinear"));
    }
    if triangles.len() > limits.max_cells {
        return Err(Error::LimitExceeded("cell"));
    }
    Ok(triangles)
}

fn recover_edge(
    points: &[[f64; 2]],
    triangles: &mut [[usize; 3]],
    target: [usize; 2],
    fixed: &BTreeSet<[usize; 2]>,
    max_iterations: usize,
    check: &mut impl FnMut() -> bool,
) -> Result<bool, Error> {
    let target = ordered(target);
    for iteration in 0..=max_iterations {
        if iteration.is_multiple_of(128) {
            continued(check)?;
        }
        let incidence = edge_incidence(triangles);
        if incidence.contains_key(&target) {
            return Ok(true);
        }
        let crossing = incidence.iter().find_map(|(edge, adjacent)| {
            (!fixed.contains(edge)
                && adjacent.len() == 2
                && segments_intersect(
                    points[target[0]],
                    points[target[1]],
                    points[edge[0]],
                    points[edge[1]],
                ))
            .then_some((*edge, adjacent.clone()))
        });
        let Some((edge, adjacent)) = crossing else {
            return Ok(false);
        };
        let first = adjacent[0];
        let second = adjacent[1];
        let opposite_a = triangles[first]
            .into_iter()
            .find(|vertex| !edge.contains(vertex))
            .expect("triangle opposite");
        let opposite_b = triangles[second]
            .into_iter()
            .find(|vertex| !edge.contains(vertex))
            .expect("triangle opposite");
        if edge_incidence(triangles).contains_key(&ordered([opposite_a, opposite_b])) {
            return Ok(false);
        }
        let Some(new_first) = oriented_triangle(opposite_a, opposite_b, edge[0], points) else {
            return Ok(false);
        };
        let Some(new_second) = oriented_triangle(opposite_b, opposite_a, edge[1], points) else {
            return Ok(false);
        };
        triangles[first] = new_first;
        triangles[second] = new_second;
    }
    Ok(false)
}

fn segment_chain(points: &[[f64; 2]], segment: [usize; 2]) -> Vec<[usize; 2]> {
    let [a, b] = segment.map(|vertex| points[vertex]);
    let direction = [b[0] - a[0], b[1] - a[1]];
    let axis = usize::from(direction[1].abs() > direction[0].abs());
    let mut vertices = points
        .iter()
        .enumerate()
        .filter_map(|(index, point)| {
            (orient2d(a, b, *point) == Sign::Zero
                && point[0] >= a[0].min(b[0])
                && point[0] <= a[0].max(b[0])
                && point[1] >= a[1].min(b[1])
                && point[1] <= a[1].max(b[1]))
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
        .map(|pair| ordered([pair[0], pair[1]]))
        .collect()
}

pub(crate) fn triangle_adjacency(
    triangles: &[[usize; 3]],
) -> Result<Vec<[Option<usize>; 3]>, Error> {
    let mut adjacency = vec![[None; 3]; triangles.len()];
    let mut edges = BTreeMap::<[usize; 2], (usize, usize)>::new();
    for (triangle_index, triangle) in triangles.iter().enumerate() {
        for edge_index in 0..3 {
            let edge = ordered([
                triangle[(edge_index + 1) % 3],
                triangle[(edge_index + 2) % 3],
            ]);
            if let Some((other_triangle, other_edge)) = edges.remove(&edge) {
                adjacency[triangle_index][edge_index] = Some(other_triangle);
                adjacency[other_triangle][other_edge] = Some(triangle_index);
            } else {
                edges.insert(edge, (triangle_index, edge_index));
            }
        }
    }
    Ok(adjacency)
}

fn edge_incidence(triangles: &[[usize; 3]]) -> BTreeMap<[usize; 2], Vec<usize>> {
    let mut result = BTreeMap::<[usize; 2], Vec<usize>>::new();
    for (index, triangle) in triangles.iter().enumerate() {
        for edge in triangle_edges(*triangle) {
            result.entry(ordered(edge)).or_default().push(index);
        }
    }
    result
}

fn triangle_edges(triangle: [usize; 3]) -> [[usize; 2]; 3] {
    [
        [triangle[0], triangle[1]],
        [triangle[1], triangle[2]],
        [triangle[2], triangle[0]],
    ]
}

fn oriented_triangle(a: usize, b: usize, c: usize, points: &[[f64; 2]]) -> Option<[usize; 3]> {
    match orient2d(points[a], points[b], points[c]) {
        Sign::Positive => Some([a, b, c]),
        Sign::Negative => Some([b, a, c]),
        Sign::Zero => None,
    }
}

fn segments_intersect(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> bool {
    let ab_c = orient2d(a, b, c).ordering();
    let ab_d = orient2d(a, b, d).ordering();
    let cd_a = orient2d(c, d, a).ordering();
    let cd_b = orient2d(c, d, b).ordering();
    (ab_c == Ordering::Equal && on_segment(a, b, c))
        || (ab_d == Ordering::Equal && on_segment(a, b, d))
        || (cd_a == Ordering::Equal && on_segment(c, d, a))
        || (cd_b == Ordering::Equal && on_segment(c, d, b))
        || (ab_c != ab_d && cd_a != cd_b)
}

fn on_segment(a: [f64; 2], b: [f64; 2], point: [f64; 2]) -> bool {
    point[0] >= a[0].min(b[0])
        && point[0] <= a[0].max(b[0])
        && point[1] >= a[1].min(b[1])
        && point[1] <= a[1].max(b[1])
}

fn ordered(mut edge: [usize; 2]) -> [usize; 2] {
    edge.sort_unstable();
    edge
}

fn bounds(points: &[[f64; 2]]) -> ([f64; 2], [f64; 2]) {
    points.iter().fold(
        ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]),
        |(mut min, mut max), point| {
            for axis in 0..2 {
                min[axis] = min[axis].min(point[axis]);
                max[axis] = max[axis].max(point[axis]);
            }
            (min, max)
        },
    )
}
