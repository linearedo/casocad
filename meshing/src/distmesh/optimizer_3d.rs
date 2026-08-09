use std::collections::BTreeSet;

use caso_delaunay::predicates::{orient3d, Sign};
use caso_kernel::meshing::MeshableDomain;
use caso_kernel::vec3::Vec3;

use super::distmesh_3d::VolumeMesh;
use crate::algorithm::{MeshingContext, MeshingStatistics, QualityTermination};
use crate::error::MeshResult;
use crate::quality::{quality_score, QualityMetric};

const MAX_PASSES: usize = 4;
const MAX_VERTICES_PER_PASS: usize = 512;

pub(super) fn optimize(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    mesh: &mut VolumeMesh,
    statistics: &mut MeshingStatistics,
) -> MeshResult<()> {
    let estimated = mesh.points.len().saturating_mul(96).saturating_add(
        mesh.cells
            .len()
            .saturating_mul(std::mem::size_of::<[usize; 4]>() * 4),
    );
    if estimated > super::MAX_OPTIMIZATION_BYTES {
        statistics.quality_termination = QualityTermination::MemoryBudget;
        return Ok(());
    }
    let mut protected = mesh.boundary_points.clone();
    protected.extend(mesh.prisms.iter().flatten().copied());
    protected.extend(mesh.pyramids.iter().flatten().copied());
    let mut incident = vec![Vec::new(); mesh.points.len()];
    let mut neighbors = vec![BTreeSet::new(); mesh.points.len()];
    for (cell_index, cell) in mesh.cells.iter().enumerate() {
        for &vertex in cell {
            incident[vertex].push(cell_index);
            neighbors[vertex].extend(cell.iter().copied().filter(|other| *other != vertex));
        }
    }
    let tolerance = domain.bounds.diagonal() * 1.0e-10 + f64::EPSILON;
    for pass in 0..MAX_PASSES {
        context.check()?;
        let mut changed = false;
        let mut attempted = 0usize;
        for vertex in 0..mesh.points.len() {
            if vertex.is_multiple_of(128) {
                context.check()?;
            }
            if protected.contains(&vertex)
                || incident[vertex].is_empty()
                || neighbors[vertex].is_empty()
            {
                continue;
            }
            let current = local_score(mesh, &incident[vertex]);
            if current.minimum_jacobian >= 0.40 && current.maximum_skewness <= 0.60 {
                continue;
            }
            if attempted == MAX_VERTICES_PER_PASS {
                break;
            }
            attempted += 1;
            let center = average(neighbors[vertex].iter().map(|other| mesh.points[*other]));
            let original = mesh.points[vertex];
            for amount in [0.35, 0.15] {
                let trial = lerp(original, center, amount);
                if domain.domain_sdf(&[Vec3::from_array(trial)])[0] > tolerance {
                    continue;
                }
                mesh.points[vertex] = trial;
                let valid = incident[vertex].iter().all(|cell| {
                    let points = mesh.cells[*cell].map(|point| mesh.points[point]);
                    orient3d(points[0], points[1], points[2], points[3]) == Sign::Positive
                        && quality_score("tet4", &points, QualityMetric::ScaledJacobian)
                            .is_some_and(|quality| quality > 0.0)
                });
                if valid && better(local_score(mesh, &incident[vertex]), current) {
                    changed = true;
                    break;
                }
                mesh.points[vertex] = original;
            }
        }
        statistics.quality_passes += 1;
        if !changed {
            statistics.quality_termination = QualityTermination::Converged;
            return Ok(());
        }
        if pass + 1 == MAX_PASSES {
            statistics.quality_termination = QualityTermination::IterationLimit;
        }
    }
    Ok(())
}

#[derive(Clone, Copy)]
struct Score {
    minimum_jacobian: f64,
    maximum_skewness: f64,
    mean_jacobian: f64,
}

fn local_score(mesh: &VolumeMesh, cells: &[usize]) -> Score {
    let mut minimum_jacobian = f64::INFINITY;
    let mut maximum_skewness: f64 = 0.0;
    let mut sum = 0.0;
    for &cell in cells {
        let points = mesh.cells[cell].map(|point| mesh.points[point]);
        let jacobian = quality_score("tet4", &points, QualityMetric::ScaledJacobian)
            .unwrap_or(f64::NEG_INFINITY);
        let skewness =
            quality_score("tet4", &points, QualityMetric::Skewness).unwrap_or(f64::INFINITY);
        minimum_jacobian = minimum_jacobian.min(jacobian);
        maximum_skewness = maximum_skewness.max(skewness);
        sum += jacobian;
    }
    Score {
        minimum_jacobian,
        maximum_skewness,
        mean_jacobian: sum / cells.len() as f64,
    }
}

fn better(candidate: Score, current: Score) -> bool {
    candidate.minimum_jacobian > current.minimum_jacobian + 1.0e-12
        || ((candidate.minimum_jacobian - current.minimum_jacobian).abs() <= 1.0e-12
            && (candidate.maximum_skewness < current.maximum_skewness - 1.0e-12
                || ((candidate.maximum_skewness - current.maximum_skewness).abs() <= 1.0e-12
                    && candidate.mean_jacobian > current.mean_jacobian + 1.0e-12)))
}

fn average(points: impl Iterator<Item = [f64; 3]>) -> [f64; 3] {
    let (sum, count) = points.fold(([0.0; 3], 0usize), |(mut sum, count), point| {
        for axis in 0..3 {
            sum[axis] += point[axis];
        }
        (sum, count + 1)
    });
    sum.map(|value| value / count as f64)
}

fn lerp(a: [f64; 3], b: [f64; 3], amount: f64) -> [f64; 3] {
    std::array::from_fn(|axis| a[axis] + (b[axis] - a[axis]) * amount)
}
