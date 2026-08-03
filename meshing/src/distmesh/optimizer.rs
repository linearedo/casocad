//! Queue-based quality optimization. Each phase clones at most one valid
//! snapshot, drains a complete deterministic queue, then accepts the sweep as
//! a unit only when all topology and quality guards pass.

use super::*;

const MAX_OUTLIER_PASSES: usize = 10;

pub(super) fn optimize(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &mut Assessment,
    statistics: &mut MeshingStatistics,
) -> MeshResult<()> {
    if let Some(termination) = candidate.layer_refinement_limit {
        record_quality_termination(statistics, termination);
    }
    if core_quality(domain, context, candidate)
        .worst_first
        .is_empty()
    {
        record_quality_termination(statistics, QualityTermination::Converged);
        return Ok(());
    }

    let mut outlier_passes = 0_usize;
    for _pass in 0..MAX_QUALITY_PASSES.saturating_sub(MAX_OUTLIER_PASSES) {
        context.check()?;
        if estimated_optimization_bytes(candidate) > MAX_OPTIMIZATION_BYTES {
            record_quality_termination(statistics, QualityTermination::MemoryBudget);
            return Ok(());
        }
        let mut changed = false;
        let mut max_cells_limited = false;

        let (accepted, limited) = size_sweep(domain, space, context, candidate, assessment)?;
        changed |= accepted;
        max_cells_limited |= limited;
        changed |= legalize_sweep(domain, space, context, candidate, assessment)?;
        changed |= smoothing_sweep(domain, space, context, candidate, assessment)?;
        if !changed && outlier_passes < MAX_OUTLIER_PASSES {
            let repaired = local_outlier_sweep(domain, space, context, candidate, assessment)?;
            changed |= repaired;
            outlier_passes += usize::from(repaired);
        }

        if !changed {
            record_quality_termination(
                statistics,
                if max_cells_limited || candidate.cells.len() as u64 >= context.limits.max_cells {
                    QualityTermination::MaxCells
                } else {
                    QualityTermination::Converged
                },
            );
            return Ok(());
        }
        statistics.quality_passes += 1;
    }
    while outlier_passes < MAX_OUTLIER_PASSES {
        context.check()?;
        if !local_outlier_sweep(domain, space, context, candidate, assessment)? {
            break;
        }
        outlier_passes += 1;
        statistics.quality_passes += 1;
    }
    record_quality_termination(statistics, QualityTermination::IterationLimit);
    Ok(())
}

pub(super) fn quality_gates_met(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
) -> bool {
    core_quality(domain, context, candidate).minimum_scaled_jacobian + 1.0e-12 >= QUALITY_TARGET
        && maximum_skewness(candidate) <= 0.60 + 1.0e-12
        && minimum_transition_scaled_jacobian(candidate) + 1.0e-12 >= 0.50
}

fn size_sweep(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &mut Assessment,
) -> MeshResult<(bool, bool)> {
    let mut split_queue = BTreeSet::new();
    let mut insert_queue = BTreeSet::new();
    let mut incidence = BTreeMap::<(PointKey, PointKey), Vec<usize>>::new();
    for (cell_index, cell) in candidate.cells.iter().enumerate() {
        for edge in 0..cell.points.len() {
            incidence
                .entry(ordered_pair(
                    cell.points[edge],
                    cell.points[(edge + 1) % cell.points.len()],
                ))
                .or_default()
                .push(cell_index);
        }
        if cell.protected || cell.points.len() != 3 {
            continue;
        }
        let mut lengths = Vec::with_capacity(3);
        for edge in 0..3 {
            let a = cell.points[edge];
            let b = cell.points[(edge + 1) % 3];
            let length = distance3(candidate.points[&a].world, candidate.points[&b].world);
            lengths.push((
                length,
                ordered_pair(a, b),
                edge_target(domain, context, candidate, a, b),
            ));
        }
        lengths.sort_by(|a, b| b.0.total_cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
        let excessive_growth = lengths[0].0 > LAYER_TRANSITION_GROWTH * lengths[2].0;
        let positions = cell
            .points
            .iter()
            .map(|key| candidate.points[key].world)
            .collect::<Vec<_>>();
        let quality =
            quality_score("tri3", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0);
        let skewness = quality_score("tri3", &positions, QualityMetric::Skewness).unwrap_or(1.0);
        if quality < QUALITY_TARGET
            || skewness > 0.60
            || lengths[0].0 > LAYER_TRANSITION_GROWTH * lengths[0].2
            || excessive_growth
        {
            if let Some(edge) = lengths
                .iter()
                .map(|entry| entry.1)
                .find(|edge| !candidate.protected_constraints.contains(edge))
            {
                split_queue.insert(edge);
            } else {
                let mut signature = cell.points.clone();
                signature.sort_unstable();
                insert_queue.insert(signature);
            }
        }
    }
    if split_queue.is_empty() && insert_queue.is_empty() {
        return Ok((false, false));
    }

    let mut trial = candidate.clone();
    let mut changed = false;
    let mut limited = false;
    let mut changed_cells = BTreeSet::<usize>::new();
    for (index, (a, b)) in split_queue.into_iter().enumerate() {
        if index.is_multiple_of(128) {
            context.check()?;
        }
        if trial.cells.len().saturating_add(2) as u64 > context.limits.max_cells {
            limited = true;
            break;
        }
        let Some(incident) = incidence.get(&(a, b)) else {
            continue;
        };
        if incident.iter().any(|cell| changed_cells.contains(cell)) {
            continue;
        }
        if split_if_locally_better(domain, space, context, &mut trial, a, b, incident)? {
            changed = true;
            changed_cells.extend(incident);
        }
    }
    let cell_indices = trial
        .cells
        .iter()
        .enumerate()
        .map(|(index, cell)| {
            let mut signature = cell.points.clone();
            signature.sort_unstable();
            (signature, index)
        })
        .collect::<BTreeMap<_, _>>();
    for (queue_index, signature) in insert_queue.into_iter().enumerate() {
        if queue_index.is_multiple_of(128) {
            context.check()?;
        }
        if trial.cells.len().saturating_add(2) as u64 > context.limits.max_cells {
            limited = true;
            break;
        }
        let Some(&cell) = cell_indices.get(&signature) else {
            continue;
        };
        let mut current = trial.cells[cell].points.clone();
        current.sort_unstable();
        if current == signature {
            changed |= apply_insert(domain, space, context, &mut trial, cell);
        }
    }
    if changed && accept_sweep(domain, space, context, candidate, &trial)? {
        *candidate = trial;
        *assessment = assess(domain, space, context, candidate)?;
        Ok((true, limited))
    } else {
        Ok((false, limited))
    }
}

#[derive(Clone, Copy)]
struct LocalQuality {
    minimum_scaled_jacobian: f64,
    maximum_skewness: f64,
    worst_distortion: f64,
}

fn split_if_locally_better(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    a: PointKey,
    b: PointKey,
    incident: &[usize],
) -> MeshResult<bool> {
    if incident.is_empty() || incident.len() > 2 {
        return Ok(false);
    }
    let before = local_quality(domain, context, candidate, incident);
    let old_cells = incident
        .iter()
        .map(|&index| (index, candidate.cells[index].clone()))
        .collect::<Vec<_>>();
    let old_len = candidate.cells.len();
    let old_next = candidate.next_inserted;
    if !apply_split(domain, space, context, candidate, a, b, incident)? {
        return Ok(false);
    }
    let affected = incident
        .iter()
        .copied()
        .chain(old_len..candidate.cells.len())
        .collect::<Vec<_>>();
    let after = local_quality(domain, context, candidate, &affected);
    let accepted = after.minimum_scaled_jacobian + 1.0e-12
        >= before.minimum_scaled_jacobian.min(QUALITY_TARGET)
        && after.maximum_skewness <= before.maximum_skewness.max(0.60) + 1.0e-12
        && after.worst_distortion
            < before.worst_distortion - 1.0e-12 * before.worst_distortion.max(1.0);
    if accepted {
        return Ok(true);
    }
    candidate.cells.truncate(old_len);
    for (index, cell) in old_cells {
        candidate.cells[index] = cell;
    }
    candidate.points.remove(&PointKey::Inserted(old_next));
    candidate.next_inserted = old_next;
    Ok(false)
}

fn local_quality(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
    cells: &[usize],
) -> LocalQuality {
    let mut result = LocalQuality {
        minimum_scaled_jacobian: 1.0,
        maximum_skewness: 0.0,
        worst_distortion: 0.0,
    };
    for &index in cells {
        let cell = &candidate.cells[index];
        let positions = cell
            .points
            .iter()
            .map(|key| candidate.points[key].world)
            .collect::<Vec<_>>();
        let scaled_jacobian =
            quality_score("tri3", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0);
        let skewness = quality_score("tri3", &positions, QualityMetric::Skewness).unwrap_or(1.0);
        let squared_log_size = (0..3)
            .map(|edge| {
                let a = cell.points[edge];
                let b = cell.points[(edge + 1) % 3];
                let length = distance3(candidate.points[&a].world, candidate.points[&b].world);
                (length / edge_target(domain, context, candidate, a, b))
                    .max(f64::MIN_POSITIVE)
                    .ln()
                    .powi(2)
            })
            .sum::<f64>()
            / 3.0;
        let distortion = (1.0 - scaled_jacobian)
            .max(skewness)
            .hypot(squared_log_size.sqrt());
        result.minimum_scaled_jacobian = result.minimum_scaled_jacobian.min(scaled_jacobian);
        result.maximum_skewness = result.maximum_skewness.max(skewness);
        result.worst_distortion = result.worst_distortion.max(distortion);
    }
    result
}

fn legalize_sweep(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &mut Assessment,
) -> MeshResult<bool> {
    let mut incidence = BTreeMap::<(PointKey, PointKey), Vec<usize>>::new();
    for (index, cell) in candidate.cells.iter().enumerate() {
        if cell.protected || cell.points.len() != 3 {
            continue;
        }
        for edge in 0..3 {
            incidence
                .entry(ordered_pair(cell.points[edge], cell.points[(edge + 1) % 3]))
                .or_default()
                .push(index);
        }
    }
    let edges = incidence
        .into_iter()
        .filter_map(|(edge, cells)| {
            (!candidate.protected_constraints.contains(&edge) && cells.len() == 2)
                .then(|| (edge, [cells[0], cells[1]]))
        })
        .collect::<Vec<_>>();
    let mut trial = candidate.clone();
    let mut changed = false;
    let mut changed_cells = BTreeSet::new();
    for (index, ((a, b), pair)) in edges.into_iter().enumerate() {
        if index.is_multiple_of(256) {
            context.check()?;
        }
        if pair.iter().any(|cell| changed_cells.contains(cell)) {
            continue;
        }
        if flip_pair_if_better(&mut trial, a, b, pair) {
            changed = true;
            changed_cells.extend(pair);
        }
    }
    if changed && accept_sweep(domain, space, context, candidate, &trial)? {
        *candidate = trial;
        *assessment = assess(domain, space, context, candidate)?;
        Ok(true)
    } else {
        Ok(false)
    }
}

fn smoothing_sweep(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &mut Assessment,
) -> MeshResult<bool> {
    let mut incidence = BTreeMap::<PointKey, Vec<usize>>::new();
    for (index, cell) in candidate.cells.iter().enumerate() {
        if cell.protected || cell.points.len() != 3 {
            continue;
        }
        for &key in &cell.points {
            incidence.entry(key).or_default().push(index);
        }
    }
    let mut updates = Vec::new();
    for (&key, point) in &candidate.points {
        if point.boundary || point.protected {
            continue;
        }
        let Some(incident) = incidence.get(&key) else {
            continue;
        };
        if incident.len() < 3 {
            continue;
        }
        let neighbors = incident
            .iter()
            .flat_map(|&cell| candidate.cells[cell].points.iter().copied())
            .filter(|other| *other != key)
            .collect::<BTreeSet<_>>();
        let laplacian = neighbors
            .iter()
            .map(|other| candidate.points[other].uv)
            .fold([0.0; 2], |sum, uv| [sum[0] + uv[0], sum[1] + uv[1]])
            .map(|value| value / neighbors.len() as f64);
        let odt = incident
            .iter()
            .map(|&cell| {
                candidate.cells[cell]
                    .points
                    .iter()
                    .map(|point| candidate.points[point].uv)
                    .fold([0.0; 2], |sum, uv| [sum[0] + uv[0], sum[1] + uv[1]])
                    .map(|value| value / 3.0)
            })
            .fold([0.0; 2], |sum, uv| [sum[0] + uv[0], sum[1] + uv[1]])
            .map(|value| value / incident.len() as f64);
        let target = [0.5 * (laplacian[0] + odt[0]), 0.5 * (laplacian[1] + odt[1])];
        updates.push((key, target));
    }
    if updates.is_empty() {
        return Ok(false);
    }
    for factor in [0.5, 0.25, 0.125, 0.0625] {
        let mut trial = candidate.clone();
        for &(key, target) in &updates {
            let old = candidate.points[&key];
            let uv = [
                old.uv[0] + factor * (target[0] - old.uv[0]),
                old.uv[1] + factor * (target[1] - old.uv[1]),
            ];
            let world = space.point(uv[0], uv[1]);
            if domain.domain_sdf(&[world])[0] < 0.0 {
                let point = trial.points.get_mut(&key).expect("smoothing point");
                point.uv = uv;
                point.world = world.to_array();
            }
        }
        if accept_sweep(domain, space, context, candidate, &trial)? {
            *candidate = trial;
            *assessment = assess(domain, space, context, candidate)?;
            return Ok(true);
        }
    }
    Ok(false)
}

fn local_outlier_sweep(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    assessment: &mut Assessment,
) -> MeshResult<bool> {
    let mut incidence = BTreeMap::<PointKey, Vec<usize>>::new();
    let mut queue = BTreeSet::new();
    let transition_edges = protected_quad_edges(candidate);
    for (index, cell) in candidate.cells.iter().enumerate() {
        if cell.protected || cell.points.len() != 3 {
            continue;
        }
        for &key in &cell.points {
            incidence.entry(key).or_default().push(index);
        }
        let positions = cell
            .points
            .iter()
            .map(|key| candidate.points[key].world)
            .collect::<Vec<_>>();
        let scaled_jacobian =
            quality_score("tri3", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0);
        let skewness = quality_score("tri3", &positions, QualityMetric::Skewness).unwrap_or(1.0);
        let required_quality = if triangle_touches_edges(cell, &transition_edges) {
            0.50
        } else {
            QUALITY_TARGET
        };
        if scaled_jacobian < required_quality || skewness > 0.60 {
            queue.extend(cell.points.iter().copied());
        }
    }
    queue.retain(|key| {
        candidate
            .points
            .get(key)
            .is_some_and(|point| !point.boundary && !point.protected)
    });
    if queue.is_empty() {
        return Ok(false);
    }

    let before = candidate.clone();
    let mut changed = false;
    for (queue_index, key) in queue.into_iter().enumerate() {
        if queue_index.is_multiple_of(128) {
            context.check()?;
        }
        let Some(incident) = incidence.get(&key) else {
            continue;
        };
        if incident.len() < 3 {
            continue;
        }
        let neighbors = incident
            .iter()
            .flat_map(|&cell| candidate.cells[cell].points.iter().copied())
            .filter(|other| *other != key)
            .collect::<BTreeSet<_>>();
        let laplacian = neighbors
            .iter()
            .map(|other| candidate.points[other].uv)
            .fold([0.0; 2], |sum, uv| [sum[0] + uv[0], sum[1] + uv[1]])
            .map(|value| value / neighbors.len() as f64);
        let odt = incident
            .iter()
            .map(|&cell| {
                candidate.cells[cell]
                    .points
                    .iter()
                    .map(|point| candidate.points[point].uv)
                    .fold([0.0; 2], |sum, uv| [sum[0] + uv[0], sum[1] + uv[1]])
                    .map(|value| value / 3.0)
            })
            .fold([0.0; 2], |sum, uv| [sum[0] + uv[0], sum[1] + uv[1]])
            .map(|value| value / incident.len() as f64);
        let old = candidate.points[&key];
        let required_quality = if incident
            .iter()
            .any(|&index| triangle_touches_edges(&candidate.cells[index], &transition_edges))
        {
            0.50
        } else {
            QUALITY_TARGET
        };
        let mut targets = vec![[0.5 * (laplacian[0] + odt[0]), 0.5 * (laplacian[1] + odt[1])]];
        for &cell_index in incident {
            let cell = &candidate.cells[cell_index];
            let positions = cell
                .points
                .iter()
                .map(|point| candidate.points[point].world)
                .collect::<Vec<_>>();
            let scaled_jacobian =
                quality_score("tri3", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0);
            let skewness =
                quality_score("tri3", &positions, QualityMetric::Skewness).unwrap_or(1.0);
            let cell_requirement = if triangle_touches_edges(cell, &transition_edges) {
                0.50
            } else {
                QUALITY_TARGET
            };
            if scaled_jacobian >= cell_requirement && skewness <= 0.60 {
                continue;
            }
            let others = cell
                .points
                .iter()
                .copied()
                .filter(|point| *point != key)
                .collect::<Vec<_>>();
            let [a, b] = [
                candidate.points[&others[0]].uv,
                candidate.points[&others[1]].uv,
            ];
            let edge = [b[0] - a[0], b[1] - a[1]];
            let length = edge[0].hypot(edge[1]);
            if length <= f64::EPSILON {
                continue;
            }
            let side = (edge[0] * (old.uv[1] - a[1]) - edge[1] * (old.uv[0] - a[0])).signum();
            let height = 0.5 * 3.0_f64.sqrt() * length * side;
            targets.push([
                0.5 * (a[0] + b[0]) - edge[1] * height / length,
                0.5 * (a[1] + b[1]) + edge[0] * height / length,
            ]);
        }
        let mut best_point = old;
        let mut best_quality = local_quality(domain, context, candidate, incident);
        for target in targets {
            for factor in [1.0, 0.75, 0.5, 0.25, 0.125, 0.0625] {
                let uv = [
                    old.uv[0] + factor * (target[0] - old.uv[0]),
                    old.uv[1] + factor * (target[1] - old.uv[1]),
                ];
                let world = space.point(uv[0], uv[1]);
                if domain.domain_sdf(&[world])[0] >= 0.0 {
                    continue;
                }
                let point = candidate.points.get_mut(&key).expect("outlier point");
                point.uv = uv;
                point.world = world.to_array();
                let quality = local_quality(domain, context, candidate, incident);
                if local_quality_is_better(quality, best_quality, required_quality) {
                    best_point = *candidate.points.get(&key).expect("outlier point");
                    best_quality = quality;
                }
            }
        }
        candidate.points.insert(key, best_point);
        changed |= best_point.world != old.world;
    }
    if changed && accept_sweep(domain, space, context, &before, candidate)? {
        *assessment = assess(domain, space, context, candidate)?;
        Ok(true)
    } else {
        *candidate = before;
        Ok(false)
    }
}

fn local_quality_is_better(
    after: LocalQuality,
    before: LocalQuality,
    required_quality: f64,
) -> bool {
    let violation = |quality: LocalQuality| {
        (required_quality - quality.minimum_scaled_jacobian).max(0.0)
            + (quality.maximum_skewness - 0.60).max(0.0)
    };
    let after_violation = violation(after);
    let before_violation = violation(before);
    after.minimum_scaled_jacobian + 1.0e-12 >= before.minimum_scaled_jacobian.min(required_quality)
        && after.maximum_skewness <= before.maximum_skewness.max(0.60) + 1.0e-12
        && (after_violation < before_violation - 1.0e-12
            || ((after_violation - before_violation).abs() <= 1.0e-12
                && (after.maximum_skewness < before.maximum_skewness - 1.0e-12
                    || ((after.maximum_skewness - before.maximum_skewness).abs() <= 1.0e-12
                        && after.worst_distortion < before.worst_distortion - 1.0e-12))))
}

fn protected_quad_edges(candidate: &Candidate) -> BTreeSet<(PointKey, PointKey)> {
    candidate
        .cells
        .iter()
        .filter(|cell| cell.protected && cell.points.len() == 4)
        .flat_map(|cell| {
            (0..4).map(|edge| {
                ordered_pair(
                    cell.points[edge],
                    cell.points[(edge + 1) % cell.points.len()],
                )
            })
        })
        .collect()
}

fn triangle_touches_edges(cell: &Cell, edges: &BTreeSet<(PointKey, PointKey)>) -> bool {
    cell.points.len() == 3
        && (0..3).any(|edge| {
            edges.contains(&ordered_pair(
                cell.points[edge],
                cell.points[(edge + 1) % cell.points.len()],
            ))
        })
}

fn minimum_transition_scaled_jacobian(candidate: &Candidate) -> f64 {
    let edges = protected_quad_edges(candidate);
    candidate
        .cells
        .iter()
        .filter(|cell| !cell.protected && triangle_touches_edges(cell, &edges))
        .map(|cell| {
            let positions = cell
                .points
                .iter()
                .map(|key| candidate.points[key].world)
                .collect::<Vec<_>>();
            quality_score("tri3", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0)
        })
        .fold(1.0, f64::min)
}

fn accept_sweep(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    before: &Candidate,
    after: &Candidate,
) -> MeshResult<bool> {
    let before_core = core_quality(domain, context, before);
    let after_core = core_quality(domain, context, after);
    let before_skewness = maximum_skewness(before);
    let after_skewness = maximum_skewness(after);
    let violation = |quality: &CoreQuality, skewness: f64, transition_quality: f64| {
        (QUALITY_TARGET - quality.minimum_scaled_jacobian).max(0.0)
            + (skewness - 0.60).max(0.0)
            + (0.50 - transition_quality).max(0.0)
    };
    let before_violation = violation(
        &before_core,
        before_skewness,
        minimum_transition_scaled_jacobian(before),
    );
    let after_violation = violation(
        &after_core,
        after_skewness,
        minimum_transition_scaled_jacobian(after),
    );
    let aggregate_improved =
        after_core.objective < before_core.objective - 1.0e-12 * before_core.objective.max(1.0);
    let hard_quality_improved = after_violation < before_violation - 1.0e-12;
    if (!hard_quality_improved && !aggregate_improved)
        || after_core.minimum_scaled_jacobian + 1.0e-12 < before_core.minimum_scaled_jacobian
        || after_skewness > before_skewness + 1.0e-12
        || !cap_quality_preserved(cap_quality(before).as_ref(), cap_quality(after).as_ref())
    {
        return Ok(false);
    }
    let assessment = assess(domain, space, context, after)?;
    Ok(assessment.refine.is_empty()
        && assessment.score.hard_invalid == 0
        && validate_planar_tile(after, &after.cells).is_ok())
}

fn maximum_skewness(candidate: &Candidate) -> f64 {
    candidate
        .cells
        .iter()
        .map(|cell| {
            let positions = cell
                .points
                .iter()
                .map(|key| candidate.points[key].world)
                .collect::<Vec<_>>();
            quality_score(cell.element_type(), &positions, QualityMetric::Skewness)
                .unwrap_or(f64::INFINITY)
        })
        .fold(0.0, f64::max)
}

fn edge_target(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
    a: PointKey,
    b: PointKey,
) -> f64 {
    candidate
        .layer_edge_targets
        .get(&ordered_pair(a, b))
        .copied()
        .unwrap_or_else(|| {
            let aw = candidate.points[&a].world;
            let bw = candidate.points[&b].world;
            let midpoint = midpoint3(aw, bw);
            local_target(
                candidate,
                context,
                &domain.name,
                Vec3::from_array(midpoint),
                0.5 * distance3(aw, bw),
                &[
                    Vec3::from_array(aw),
                    Vec3::from_array(bw),
                    Vec3::from_array(midpoint),
                ],
            )
        })
}

fn flip_pair_if_better(
    candidate: &mut Candidate,
    a: PointKey,
    b: PointKey,
    pair: [usize; 2],
) -> bool {
    let first = &candidate.cells[pair[0]];
    let second = &candidate.cells[pair[1]];
    if first.protected || second.protected || first.points.len() != 3 || second.points.len() != 3 {
        return false;
    }
    let Some(c) = first
        .points
        .iter()
        .copied()
        .find(|point| *point != a && *point != b)
    else {
        return false;
    };
    let Some(d) = second
        .points
        .iter()
        .copied()
        .find(|point| *point != a && *point != b)
    else {
        return false;
    };
    let before = pair_quality(
        [
            first.points.clone().try_into().expect("triangle"),
            second.points.clone().try_into().expect("triangle"),
        ],
        &candidate.points,
    );
    let mut replacements = [[c, d, a], [d, c, b]];
    orient_triangle(&mut replacements[0], &candidate.points);
    orient_triangle(&mut replacements[1], &candidate.points);
    let after = pair_quality(replacements, &candidate.points);
    after > before + 1.0e-12 && apply_flip(candidate, a, b)
}
