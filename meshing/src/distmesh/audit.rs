use super::*;

pub(super) fn validate_candidate(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
    assessment: &Assessment,
) -> MeshResult<()> {
    context.check()?;
    if candidate.cells.is_empty()
        || !assessment.refine.is_empty()
        || assessment.score.hard_invalid != 0
    {
        return Err(invalid(domain, "is not topology-valid"));
    }
    if crossing_cell_edges(candidate, &candidate.cells).is_some() {
        return Err(invalid(domain, "contains crossed cell edges"));
    }

    let mut incidence = BTreeMap::<(PointKey, PointKey), Vec<[PointKey; 2]>>::new();
    let mut cell_area = 0.0;
    for cell in &candidate.cells {
        let area = signed_area_polygon(&cell.points, &candidate.points);
        if area
            <= orientation_tolerance(maximum_edge_2d(
                &cell
                    .points
                    .iter()
                    .map(|point| candidate.points[point].world)
                    .collect::<Vec<_>>(),
            ))
        {
            return Err(invalid(domain, "contains a non-positive cell"));
        }
        cell_area += area;
        for edge in 0..cell.points.len() {
            let oriented = [
                cell.points[edge],
                cell.points[(edge + 1) % cell.points.len()],
            ];
            incidence
                .entry(ordered_pair(oriented[0], oriented[1]))
                .or_default()
                .push(oriented);
        }
    }
    if incidence
        .values()
        .any(|edges| edges.len() > 2 || edges.len() == 2 && edges[0] == edges[1])
    {
        return Err(invalid(domain, "has non-manifold edge incidence"));
    }
    for constraint in &candidate.protected_constraints {
        if !incidence.contains_key(constraint) {
            return Err(invalid(domain, "lost a protected constraint"));
        }
    }

    let boundary_area = assessment
        .boundary
        .iter()
        .map(|edge| {
            let a = candidate.points[&edge.points[0]].uv;
            let b = candidate.points[&edge.points[1]].uv;
            0.5 * (a[0] * b[1] - a[1] * b[0])
        })
        .sum::<f64>();
    let scale = domain.bounds.diagonal().powi(2).max(cell_area.abs());
    let area_residual = (cell_area - boundary_area).abs();
    let area_tolerance = scale * 1.0e-8;
    if area_residual > area_tolerance {
        return Err(invalid(
            domain,
            &format!(
                "does not close to the area of its boundary cycles (cells={cell_area:.12e}, boundary={boundary_area:.12e}, residual={area_residual:.12e}, tolerance={area_tolerance:.12e})"
            ),
        ));
    }
    let graph = contour::PlanarConstraintGraph::from_boundary(
        domain,
        context,
        candidate,
        &assessment.boundary,
    )?;
    if graph.contour_count() == 0
        || (!domain.boundary_regions.is_empty() && graph.owned_edge_count() == 0)
    {
        return Err(invalid(domain, "does not match its classified PSLG cycles"));
    }
    Ok(())
}

fn invalid(domain: &MeshableDomain, reason: &str) -> MeshError {
    MeshError::InvalidInput(format!(
        "domain {:?} final 2D mesh audit failed: {reason}",
        domain.name
    ))
}
