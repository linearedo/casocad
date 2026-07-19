//! Exact, tagged, closed, oriented boundary loops of a 2D meshable domain
//! (`design_docs/meshing_toolkit.md` §7).
//!
//! Arcs come from `surfaces::boundary_outline::curve_patch_arcs` — already
//! exactly on the outline with bisected junction endpoints — then get split
//! where the region classification changes, stitched into closed loops, and
//! oriented "material on the left" (outer loops CCW, holes CW in the
//! `mesh_space` chart).

use caso_kernel::boundary_ops::surface_patches_for_root;
use caso_kernel::meshing::{MeshableDomain, MeshableDomainSpace};
use caso_kernel::vec3::Vec3;
use caso_kernel::{GeometryError, GeometryResult};
use caso_surfaces::boundary_outline::curve_patch_arcs;

/// Endpoint weld band relative to the domain bbox diagonal: junction
/// vertices within one patch are bitwise-identical, across two patches the
/// independent bisections agree to the 1e-9 on-surface band.
const WELD_RELATIVE: f64 = 1.0e-7;
/// Interior probe offset for the orientation test, relative to the diagonal.
const ORIENT_PROBE_RELATIVE: f64 = 1.0e-6;
/// Classification band for loop vertices, relative to the diagonal. Loop
/// vertices lie exactly on the outline, so the tight band suffices — and it
/// is required: with the default (sagitta) band both halves of a knife-split
/// region match near the knife, and the label transition would land at the
/// band edge instead of on the knife's zero set, where the highlight seam
/// lies.
const LABEL_BAND_RELATIVE: f64 = 1.0e-9;
const REGION_BISECTION_ITERATIONS: usize = 48;

/// One run of boundary with a single classification: points exactly on the
/// outline curve, the owning patch, and the winning region (if any).
#[derive(Debug, Clone)]
pub struct BoundarySpan {
    pub points: Vec<Vec3>,
    pub owner_object_id: u32,
    pub patch_id: String,
    pub region_index: Option<usize>,
    pub region_name: Option<String>,
}

/// A closed chain of spans: consecutive spans share their junction vertex,
/// the last span ends on the first span's head.
#[derive(Debug, Clone)]
pub struct BoundaryLoop {
    pub spans: Vec<BoundarySpan>,
    /// `signed_area > 0` (CCW in chart coordinates): the outer loop.
    pub is_outer: bool,
    /// Shoelace area in `mesh_space` chart coordinates.
    pub signed_area: f64,
}

/// Closed, oriented, region-tagged exact boundary loops of a 2D domain.
pub fn boundary_loops(
    domain: &MeshableDomain,
    resolution: usize,
) -> GeometryResult<Vec<BoundaryLoop>> {
    if domain.dimension != 2 {
        return Err(GeometryError::new(
            "boundary_loops is only available for 2D meshable domains",
        ));
    }
    let space = domain.mesh_space()?;
    let root = domain.region_node();
    let diagonal = domain.bounds.diagonal().max(1.0e-9);
    let weld = diagonal * WELD_RELATIVE;
    let label_band = diagonal * LABEL_BAND_RELATIVE;

    let mut ring_chains: Vec<Vec<BoundarySpan>> = Vec::new();
    let mut open_spans: Vec<BoundarySpan> = Vec::new();
    for patch in surface_patches_for_root(root) {
        if patch.curve.is_none() {
            continue;
        }
        for arc in curve_patch_arcs(root, &patch, resolution) {
            if arc.len() < 2 {
                continue;
            }
            let closed =
                arc.len() >= 3 && (arc[0] - arc[arc.len() - 1]).length() <= weld;
            let spans = split_by_region(domain, &patch, &arc, label_band)?;
            if closed {
                ring_chains.push(merge_ring_seam(spans));
            } else {
                open_spans.extend(spans);
            }
        }
    }

    let mut chains = ring_chains;
    chains.extend(stitch_chains(open_spans, weld)?);
    Ok(chains
        .into_iter()
        .map(|spans| orient_chain(spans, &space, diagonal))
        .collect())
}

/// Split one arc where the winning region changes, bisecting each transition
/// so both resulting spans share an identical junction vertex. Each span is
/// labeled by the classification of its middle vertex.
fn split_by_region(
    domain: &MeshableDomain,
    patch: &caso_kernel::boundary_ops::BoundarySurfacePatch,
    arc: &[Vec3],
    label_band: f64,
) -> GeometryResult<Vec<BoundarySpan>> {
    let labels: Vec<Option<usize>> = domain
        .classify_boundary(arc, Some(label_band))?
        .into_iter()
        .map(|class| class.region_index)
        .collect();
    let mut spans = Vec::new();
    let mut points = vec![arc[0]];
    let mut point_labels = vec![labels[0]];
    for index in 0..arc.len() - 1 {
        if labels[index + 1] == labels[index] {
            points.push(arc[index + 1]);
            point_labels.push(labels[index + 1]);
            continue;
        }
        let junction = bisect_region_transition(
            domain,
            arc[index],
            arc[index + 1],
            labels[index],
            label_band,
        )?;
        points.push(junction);
        spans.push(make_span(
            domain,
            patch,
            std::mem::take(&mut points),
            middle_label(&point_labels),
        ));
        points = vec![junction, arc[index + 1]];
        point_labels = vec![labels[index + 1], labels[index + 1]];
    }
    spans.push(make_span(domain, patch, points, middle_label(&point_labels)));
    Ok(spans)
}

fn middle_label(labels: &[Option<usize>]) -> Option<usize> {
    labels[labels.len() / 2]
}

fn make_span(
    domain: &MeshableDomain,
    patch: &caso_kernel::boundary_ops::BoundarySurfacePatch,
    points: Vec<Vec3>,
    region_index: Option<usize>,
) -> BoundarySpan {
    BoundarySpan {
        points,
        owner_object_id: patch.owner_object_id,
        patch_id: patch.patch_id.clone(),
        region_index,
        region_name: region_index.map(|index| domain.boundary_regions[index].name.clone()),
    }
}

/// Deterministic parameter bisection of the classification transition on the
/// chord (p, q): both sides of a junction run this identical routine.
fn bisect_region_transition(
    domain: &MeshableDomain,
    p: Vec3,
    q: Vec3,
    from_label: Option<usize>,
    label_band: f64,
) -> GeometryResult<Vec3> {
    let (mut lo, mut hi) = (0.0f64, 1.0f64);
    for _ in 0..REGION_BISECTION_ITERATIONS {
        let mid = 0.5 * (lo + hi);
        let point = p + (q - p) * mid;
        let label = domain.classify_boundary(&[point], Some(label_band))?[0].region_index;
        if label == from_label {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    Ok(p + (q - p) * (0.5 * (lo + hi)))
}

/// A closed ring may start mid-region: when the seam's two spans carry the
/// same label, merge them across it so span boundaries are exactly the
/// region junctions.
fn merge_ring_seam(mut spans: Vec<BoundarySpan>) -> Vec<BoundarySpan> {
    if spans.len() < 2 {
        return spans;
    }
    if spans[0].region_index != spans[spans.len() - 1].region_index {
        return spans;
    }
    let first = spans.remove(0);
    let last = spans.last_mut().expect("at least one span remains");
    last.points.extend(first.points.into_iter().skip(1));
    spans
}

/// Deterministic greedy stitching of open spans into closed chains: start at
/// the lowest-index unused span, append the first unused span whose endpoint
/// lies within the weld band of the current tail (reversing it if needed),
/// welding to the existing tail coordinate (first-wins). An unclosable chain
/// is an error naming the patch — never a silent open chain.
fn stitch_chains(
    spans: Vec<BoundarySpan>,
    weld: f64,
) -> GeometryResult<Vec<Vec<BoundarySpan>>> {
    let mut used = vec![false; spans.len()];
    let mut chains = Vec::new();
    for start in 0..spans.len() {
        if used[start] {
            continue;
        }
        used[start] = true;
        let mut chain = vec![spans[start].clone()];
        loop {
            let tail = *chain
                .last()
                .expect("chain is never empty")
                .points
                .last()
                .expect("spans have >= 2 points");
            let head = chain[0].points[0];
            let chain_points: usize = chain.iter().map(|span| span.points.len()).sum();
            if chain_points > 3 && (tail - head).length() <= weld {
                let closing = chain
                    .last_mut()
                    .expect("chain is never empty")
                    .points
                    .last_mut()
                    .expect("spans have >= 2 points");
                *closing = head;
                break;
            }
            let next = (0..spans.len()).find_map(|index| {
                if used[index] {
                    return None;
                }
                let candidate = &spans[index];
                let candidate_head = candidate.points[0];
                let candidate_tail = candidate.points[candidate.points.len() - 1];
                if (candidate_head - tail).length() <= weld {
                    Some((index, false))
                } else if (candidate_tail - tail).length() <= weld {
                    Some((index, true))
                } else {
                    None
                }
            });
            let Some((index, reversed)) = next else {
                return Err(GeometryError::new(format!(
                    "open boundary chain: no arc continues from patch {:?} \
                     (outline pipeline gap larger than the weld band)",
                    chain.last().expect("chain is never empty").patch_id
                )));
            };
            used[index] = true;
            let mut span = spans[index].clone();
            if reversed {
                span.points.reverse();
            }
            span.points[0] = tail;
            chain.push(span);
        }
        chains.push(chain);
    }
    Ok(chains)
}

/// Orientation and area: "material on the left". The probe compares the
/// domain field a small step to the left and right of the longest segment's
/// midpoint (the deeper side is the material side — a sign comparison, no
/// positive magnitude is read as a distance).
fn orient_chain(
    mut spans: Vec<BoundarySpan>,
    space: &MeshableDomainSpace,
    diagonal: f64,
) -> BoundaryLoop {
    let mut chart: Vec<[f64; 2]> = Vec::new();
    for (span_index, span) in spans.iter().enumerate() {
        let skip = usize::from(span_index > 0);
        for point in span.points.iter().skip(skip) {
            let local = space.coords(*point);
            chart.push([local[0], local[1]]);
        }
    }
    // The ring closure duplicates the head; drop it for the shoelace walk.
    if chart.len() >= 2 {
        let first = chart[0];
        let last = chart[chart.len() - 1];
        if (first[0] - last[0]).abs() < f64::EPSILON && (first[1] - last[1]).abs() < f64::EPSILON {
            chart.pop();
        }
    }
    let mut signed_area = 0.0;
    for index in 0..chart.len() {
        let a = chart[index];
        let b = chart[(index + 1) % chart.len()];
        signed_area += a[0] * b[1] - b[0] * a[1];
    }
    signed_area *= 0.5;

    // Longest segment: most robust probe site (far from corners/junctions).
    let mut best = (0usize, 0.0f64);
    for index in 0..chart.len() {
        let a = chart[index];
        let b = chart[(index + 1) % chart.len()];
        let length = ((b[0] - a[0]).powi(2) + (b[1] - a[1]).powi(2)).sqrt();
        if length > best.1 {
            best = (index, length);
        }
    }
    let a = chart[best.0];
    let b = chart[(best.0 + 1) % chart.len()];
    let tangent = [(b[0] - a[0]) / best.1.max(1.0e-300), (b[1] - a[1]) / best.1.max(1.0e-300)];
    let left = [-tangent[1], tangent[0]];
    let mid = [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5];
    let probe = diagonal * ORIENT_PROBE_RELATIVE;
    let left_value = space.sdf(mid[0] + left[0] * probe, mid[1] + left[1] * probe);
    let right_value = space.sdf(mid[0] - left[0] * probe, mid[1] - left[1] * probe);
    if left_value > right_value {
        spans.reverse();
        for span in &mut spans {
            span.points.reverse();
        }
        signed_area = -signed_area;
    }
    BoundaryLoop {
        spans,
        is_outer: signed_area > 0.0,
        signed_area,
    }
}
