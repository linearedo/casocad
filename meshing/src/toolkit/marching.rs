//! Keyed boundary sampling for mesher scripts: pick a named boundary of a
//! 2D domain and march along it, returning ordered points that are EXACT
//! boundary-loop vertices (interior-exactness contract: an interpolated
//! chord point would be off the zero set, so interpolation never happens —
//! the sampler only selects among exact vertices, at approximately even
//! arc-length spacing).
//!
//! Boundary names, in lookup order:
//! - `"outer"` — the domain's outer boundary loop;
//! - a boundary-region name — the tagged piece ("inlet", "skin", …);
//! - a scene object's name — every piece of boundary whose outline that
//!   object contributes (the subtracted circle's hole, a rectangle's edges).
//!
//! `boundary_names` lists what a domain offers; unknown names error listing
//! the same set.

use caso_kernel::meshing::MeshableDomain;
use caso_kernel::vec3::Vec3;
use caso_kernel::{GeometryError, GeometryResult};

use super::loops2d::{boundary_loops, BoundaryLoop, BoundarySpan};

/// Internal loop resolution floor: enough exact vertices per arc that the
/// marching selection has vertices to spare between targets.
const MINIMUM_INTERNAL_RESOLUTION: usize = 48;

/// The boundary names a 2D domain can be sampled by: `"outer"` (when the
/// domain has an outer loop), every boundary-region name on the boundary,
/// and every contributing scene object's name. Ordered, deduplicated.
pub fn boundary_names(domain: &MeshableDomain) -> GeometryResult<Vec<String>> {
    let loops = boundary_loops(domain, MINIMUM_INTERNAL_RESOLUTION)?;
    Ok(names_of(&loops))
}

fn names_of(loops: &[BoundaryLoop]) -> Vec<String> {
    let mut names = Vec::new();
    if loops.iter().any(|chain| chain.is_outer) {
        names.push("outer".to_string());
    }
    let push_unique = |name: &str, names: &mut Vec<String>| {
        if !names.iter().any(|existing| existing == name) {
            names.push(name.to_string());
        }
    };
    for chain in loops {
        for span in &chain.spans {
            if let Some(region) = &span.region_name {
                push_unique(region, &mut names);
            }
        }
    }
    for chain in loops {
        for span in &chain.spans {
            push_unique(&span.owner_name, &mut names);
        }
    }
    names
}

/// March along the named boundary of a 2D domain and return `npoints`
/// ordered points, all exact boundary vertices, at approximately even
/// arc-length spacing. Closed boundaries do NOT repeat the head; open
/// pieces include both exact endpoints. Orientation follows the loops
/// (material on the left). When the boundary offers fewer exact vertices
/// than requested, all of them are returned.
pub fn boundary_marching_sample(
    domain: &MeshableDomain,
    name: &str,
    npoints: usize,
) -> GeometryResult<Vec<Vec3>> {
    if npoints < 2 {
        return Err(GeometryError::new(
            "boundary_marching_sample needs npoints >= 2",
        ));
    }
    let loops = boundary_loops(domain, npoints.max(MINIMUM_INTERNAL_RESOLUTION))?;
    let (points, closed) = select_chain(&loops, name)?;
    if closed && npoints < 3 {
        return Err(GeometryError::new(format!(
            "boundary {name:?} is a closed loop; sampling it needs npoints >= 3"
        )));
    }
    Ok(march(&points, closed, npoints))
}

/// Select the spans the name refers to and join them into one ordered
/// polyline. Returns (points, closed).
fn select_chain(loops: &[BoundaryLoop], name: &str) -> GeometryResult<(Vec<Vec3>, bool)> {
    if name == "outer" {
        let outers: Vec<&BoundaryLoop> =
            loops.iter().filter(|chain| chain.is_outer).collect();
        return match outers.len() {
            1 => Ok((join_spans(&outers[0].spans, true), true)),
            0 => Err(GeometryError::new(format!(
                "this domain has no outer boundary loop; available boundaries: {}",
                names_of(loops).join(", ")
            ))),
            count => Err(GeometryError::new(format!(
                "this domain has {count} outer boundary loops (disconnected \
                 domain); sample its pieces by object name instead"
            ))),
        };
    }
    let by_region = |span: &BoundarySpan| span.region_name.as_deref() == Some(name);
    let by_owner = |span: &BoundarySpan| span.owner_name == name;
    if loops.iter().flat_map(|chain| &chain.spans).any(by_region) {
        return joined_selection(loops, name, by_region);
    }
    if loops.iter().flat_map(|chain| &chain.spans).any(by_owner) {
        return joined_selection(loops, name, by_owner);
    }
    Err(GeometryError::new(format!(
        "unknown boundary {name:?}; available boundaries: {}",
        names_of(loops).join(", ")
    )))
}

/// Join every selected span into one connected polyline. Selected spans of
/// a closed chain may wrap across the chain's seam; the chain is rotated so
/// the run is contiguous. More than one disconnected run is an error.
fn joined_selection(
    loops: &[BoundaryLoop],
    name: &str,
    selected: impl Fn(&BoundarySpan) -> bool,
) -> GeometryResult<(Vec<Vec3>, bool)> {
    let mut runs: Vec<(Vec<&BoundarySpan>, bool)> = Vec::new();
    for chain in loops {
        let picked: Vec<bool> = chain.spans.iter().map(&selected).collect();
        if picked.iter().all(|hit| *hit) {
            // The whole closed chain belongs to the selection.
            runs.push((chain.spans.iter().collect(), true));
            continue;
        }
        // Contiguous runs, treating the chain as circular (a run may wrap
        // across the seam): start each run at a span whose predecessor is
        // not selected.
        let count = chain.spans.len();
        for start in 0..count {
            let previous = (start + count - 1) % count;
            if !picked[start] || picked[previous] {
                continue;
            }
            let mut run = Vec::new();
            let mut index = start;
            while picked[index] {
                run.push(&chain.spans[index]);
                index = (index + 1) % count;
                if index == start {
                    break;
                }
            }
            runs.push((run, false));
        }
    }
    match runs.len() {
        1 => {
            let (spans, closed) = runs.remove(0);
            Ok((join_spans_refs(&spans, closed), closed))
        }
        0 => Err(GeometryError::new(format!(
            "boundary {name:?} matched no spans; available boundaries: {}",
            names_of(loops).join(", ")
        ))),
        count => Err(GeometryError::new(format!(
            "boundary {name:?} is not a single connected curve ({count} \
             separate pieces); sample its pieces by region name instead"
        ))),
    }
}

fn join_spans(spans: &[BoundarySpan], closed: bool) -> Vec<Vec3> {
    let refs: Vec<&BoundarySpan> = spans.iter().collect();
    join_spans_refs(&refs, closed)
}

/// Concatenate span points into one polyline: consecutive spans share their
/// junction vertex (skip it), and a closed chain's final vertex repeats the
/// head (drop it).
fn join_spans_refs(spans: &[&BoundarySpan], closed: bool) -> Vec<Vec3> {
    let mut points = Vec::new();
    for (span_index, span) in spans.iter().enumerate() {
        let skip = usize::from(span_index > 0);
        points.extend(span.points.iter().skip(skip).copied());
    }
    if closed && points.len() >= 2 && points[0] == points[points.len() - 1] {
        points.pop();
    }
    points
}

/// Pick `npoints` existing vertices at approximately even arc-length
/// spacing — never interpolating (every returned point is an exact loop
/// vertex). Duplicate picks collapse, so fewer points than requested come
/// back only when the polyline itself has fewer vertices.
fn march(points: &[Vec3], closed: bool, npoints: usize) -> Vec<Vec3> {
    if points.len() <= npoints {
        return points.to_vec();
    }
    // Cumulative chord length at each vertex.
    let mut cumulative = Vec::with_capacity(points.len());
    let mut length = 0.0;
    cumulative.push(0.0);
    for pair in points.windows(2) {
        length += (pair[1] - pair[0]).length();
        cumulative.push(length);
    }
    let total = if closed {
        length + (points[0] - points[points.len() - 1]).length()
    } else {
        length
    };
    let steps = if closed { npoints } else { npoints - 1 };
    let mut picked = Vec::with_capacity(npoints);
    let mut cursor = 0usize;
    for step in 0..npoints {
        let target = total * (step as f64) / (steps as f64);
        while cursor + 1 < points.len() && cumulative[cursor + 1] <= target {
            cursor += 1;
        }
        // Nearest of the two bracketing vertices.
        let index = if cursor + 1 < points.len()
            && (cumulative[cursor + 1] - target) < (target - cumulative[cursor])
        {
            cursor + 1
        } else {
            cursor
        };
        if picked.last() != Some(&index) {
            picked.push(index);
        }
    }
    if !closed {
        // Both exact endpoints are part of an open sample.
        if picked.last() != Some(&(points.len() - 1)) {
            picked.push(points.len() - 1);
        }
    }
    picked.into_iter().map(|index| points[index]).collect()
}
