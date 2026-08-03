use caso_delaunay::{
    triangulate_2d, ConstraintPolicy, Error as DelaunayError, Input2d, Limits, SegmentConstraint,
    Triangulation2d,
};

use super::contour::PlanarConstraintGraph;
use super::*;

const MAX_REFINEMENT_STEPS: usize = 64;

pub(super) fn retriangulate(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    graph: &PlanarConstraintGraph,
    refine: bool,
) -> MeshResult<()> {
    let constraints = graph.constraints();
    triangulate_candidate(
        domain,
        space,
        context,
        candidate,
        &constraints,
        &BTreeSet::new(),
        refine,
    )
}

pub(super) fn triangulate_core(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    strip: &BoundaryLayerStrip,
    constraints: &BTreeSet<(PointKey, PointKey)>,
    _leaves: &BTreeMap<PointKey, Leaf>,
) -> MeshResult<Vec<Cell>> {
    let mut strip_incidence = BTreeMap::<(PointKey, PointKey), usize>::new();
    for cell in &strip.cells {
        for edge in 0..cell.points.len() {
            *strip_incidence
                .entry(ordered_pair(
                    cell.points[edge],
                    cell.points[(edge + 1) % cell.points.len()],
                ))
                .or_default() += 1;
        }
    }
    let excluded = strip_incidence
        .into_iter()
        .filter_map(|(edge, count)| (count == 1).then_some(edge))
        .collect::<BTreeSet<_>>();
    let before = candidate.cells.clone();
    triangulate_candidate(
        domain,
        space,
        context,
        candidate,
        constraints,
        &excluded,
        candidate.refine_layer_core,
    )?;
    if candidate.cells.is_empty() {
        candidate.cells = before;
        return Err(invalid_cdt(domain, "produced no protected core triangles"));
    }
    let mut cells = std::mem::take(&mut candidate.cells);
    cells.extend(strip.cells.iter().cloned());
    Ok(cells)
}

fn triangulate_candidate(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    constraints: &BTreeSet<(PointKey, PointKey)>,
    excluded: &BTreeSet<(PointKey, PointKey)>,
    _refine: bool,
) -> MeshResult<()> {
    context.check()?;
    validate_constraint_graph(domain, candidate, constraints)?;
    let constrained_points = constraints
        .iter()
        .flat_map(|&(a, b)| [a, b])
        .collect::<BTreeSet<_>>();
    let mut seen = BTreeMap::<[u64; 2], PointKey>::new();
    let mut keys = Vec::new();
    for (&key, point) in &candidate.points {
        let bits = point.uv.map(f64::to_bits);
        if let Some(previous) = seen.get(&bits) {
            if constrained_points.contains(&key)
                || constrained_points.contains(previous)
                || point.protected
                || candidate.points[previous].protected
            {
                return Err(invalid_cdt(
                    domain,
                    "contains coincident immutable vertices",
                ));
            }
            continue;
        }
        seen.insert(bits, key);
        keys.push(key);
    }
    let indices = keys
        .iter()
        .enumerate()
        .map(|(index, key)| (*key, index))
        .collect::<BTreeMap<_, _>>();
    let input = Input2d {
        points: keys.iter().map(|key| candidate.points[key].uv).collect(),
        constraints: constraints
            .iter()
            .map(|&(a, b)| {
                let vertices = [indices.get(&a), indices.get(&b)];
                let Some((&a, &b)) = vertices[0].zip(vertices[1]) else {
                    return Err(invalid_cdt(
                        domain,
                        "references a missing constraint vertex",
                    ));
                };
                Ok(SegmentConstraint {
                    vertices: [a, b],
                    policy: ConstraintPolicy::Fixed,
                })
            })
            .collect::<MeshResult<Vec<_>>>()?,
    };
    let max_cells = usize::try_from(context.limits.max_cells).unwrap_or(usize::MAX);
    let limits = Limits {
        max_vertices: max_cells.saturating_mul(2).max(input.points.len()),
        max_cells,
        max_iterations: MAX_REFINEMENT_STEPS,
    };
    let run = |candidate: &mut Candidate| {
        let result = triangulate_2d(&input, limits, || !context.job_control.is_cancelled());
        result.map_err(|error| map_error(domain, error, candidate))
    };
    let mesh = run(candidate)?;
    install_result(
        domain,
        space,
        context,
        candidate,
        &keys,
        constraints,
        excluded,
        mesh,
    )
}

#[allow(clippy::too_many_arguments)]
fn install_result(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    input_keys: &[PointKey],
    constraints: &BTreeSet<(PointKey, PointKey)>,
    excluded: &BTreeSet<(PointKey, PointKey)>,
    mesh: Triangulation2d,
) -> MeshResult<()> {
    if mesh.vertices.len() < input_keys.len() {
        return Err(invalid_cdt(domain, "lost input vertices"));
    }
    let mut keys = input_keys.to_vec();
    for &uv in &mesh.vertices[input_keys.len()..] {
        let key = PointKey::Inserted(candidate.next_inserted);
        candidate.next_inserted += 1;
        let world = space.point(uv[0], uv[1]);
        candidate.points.insert(
            key,
            Point {
                uv,
                world: world.to_array(),
                boundary: false,
                protected: false,
            },
        );
        keys.push(key);
    }
    let boundary_edges = constraints
        .iter()
        .filter(|&&(a, b)| candidate.points[&a].boundary && candidate.points[&b].boundary)
        .map(|&(a, b)| [candidate.points[&a].uv, candidate.points[&b].uv])
        .collect::<Vec<_>>();
    let excluded_edges = excluded
        .iter()
        .map(|&(a, b)| [candidate.points[&a].uv, candidate.points[&b].uv])
        .collect::<Vec<_>>();
    let mut cells = Vec::new();
    for (index, triangle) in mesh.triangles.into_iter().enumerate() {
        if index.is_multiple_of(512) {
            context.check()?;
        }
        let triangle = triangle.map(|vertex| keys[vertex]);
        let uv = triangle.map(|key| candidate.points[&key].uv);
        let center = [
            (uv[0][0] + uv[1][0] + uv[2][0]) / 3.0,
            (uv[0][1] + uv[1][1] + uv[2][1]) / 3.0,
        ];
        if !winding(center, &boundary_edges) || winding(center, &excluded_edges) {
            continue;
        }
        let world = space.point(center[0], center[1]);
        if domain.domain_sdf(&[world])[0] >= root_tolerance(domain, context.target_size) {
            continue;
        }
        let area = signed_area(triangle, &candidate.points);
        if area.abs() <= orientation_tolerance(context.target_size) {
            continue;
        }
        let mut cell = Cell::triangle(
            triangle,
            Leaf {
                level: 0,
                x: 0,
                y: 0,
            },
        );
        if area < 0.0 {
            cell.points.swap(1, 2);
        }
        cells.push(cell);
    }
    if cells.is_empty() {
        return Err(invalid_cdt(domain, "produced no interior triangles"));
    }
    candidate.cells = cells;
    let mut used = candidate
        .cells
        .iter()
        .flat_map(|cell| cell.points.iter().copied())
        .collect::<BTreeSet<_>>();
    used.extend(constraints.iter().flat_map(|&(a, b)| [a, b]));
    candidate.points.retain(|key, _| used.contains(key));
    candidate.construction_failures.clear();
    Ok(())
}

fn validate_constraint_graph(
    domain: &MeshableDomain,
    candidate: &Candidate,
    constraints: &BTreeSet<(PointKey, PointKey)>,
) -> MeshResult<()> {
    for &(a, b) in constraints {
        if a == b || !candidate.points.contains_key(&a) || !candidate.points.contains_key(&b) {
            return Err(invalid_cdt(domain, "contains an invalid constraint edge"));
        }
    }
    Ok(())
}

fn winding(point: [f64; 2], edges: &[[[f64; 2]; 2]]) -> bool {
    edges.iter().fold(false, |inside, [a, b]| {
        let crosses = (a[1] > point[1]) != (b[1] > point[1])
            && point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0];
        inside ^ crosses
    })
}

fn map_error(
    domain: &MeshableDomain,
    error: DelaunayError,
    candidate: &mut Candidate,
) -> MeshError {
    match error {
        DelaunayError::Cancelled => MeshError::Cancelled,
        DelaunayError::LimitExceeded(resource) => {
            candidate.layer_refinement_limit = Some(QualityTermination::MaxCells);
            MeshError::LimitExceeded(format!(
                "domain {:?} constrained Delaunay {resource} limit exceeded",
                domain.name
            ))
        }
        error => invalid_cdt(domain, &error.to_string()),
    }
}

fn invalid_cdt(domain: &MeshableDomain, reason: &str) -> MeshError {
    MeshError::InvalidInput(format!(
        "domain {:?} global constrained Delaunay triangulation {reason}",
        domain.name
    ))
}
