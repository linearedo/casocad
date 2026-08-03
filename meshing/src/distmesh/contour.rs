use super::*;

#[derive(Debug, Clone)]
pub(super) struct PlanarConstraintGraph {
    pub contours: Vec<Vec<PointKey>>,
    pub edges: Vec<GraphEdge>,
}

#[derive(Debug, Clone)]
pub(super) struct GraphEdge {
    pub points: [PointKey; 2],
    pub owner: Option<String>,
}

impl PlanarConstraintGraph {
    pub fn from_boundary(
        domain: &MeshableDomain,
        context: &MeshingContext<'_>,
        candidate: &Candidate,
        boundary: &[BoundaryEdge],
    ) -> MeshResult<Self> {
        if boundary.is_empty() {
            return Err(invalid_graph(domain, "has no closed boundary constraints"));
        }

        let mut adjacency = BTreeMap::<PointKey, Vec<(PointKey, usize)>>::new();
        let mut unique = BTreeSet::new();
        for (index, edge) in boundary.iter().enumerate() {
            context.check()?;
            let [a, b] = edge.points;
            if a == b || !unique.insert(ordered_pair(a, b)) {
                return Err(invalid_graph(
                    domain,
                    "contains a degenerate or duplicate boundary constraint",
                ));
            }
            adjacency.entry(a).or_default().push((b, index));
            adjacency.entry(b).or_default().push((a, index));
        }
        if adjacency.values().any(|neighbors| neighbors.len() != 2) {
            return Err(invalid_graph(domain, "has ambiguous boundary incidence"));
        }

        let mut unused = (0..boundary.len()).collect::<BTreeSet<_>>();
        let mut contours = Vec::new();
        while let Some(&first) = unused.first() {
            let [start, mut point] = boundary[first].points;
            unused.remove(&first);
            let mut contour = vec![start, point];
            while point != start {
                let (next, index) = adjacency[&point]
                    .iter()
                    .copied()
                    .find(|(_, index)| unused.contains(index))
                    .ok_or_else(|| {
                        invalid_graph(domain, "boundary contour terminates before closing")
                    })?;
                unused.remove(&index);
                point = next;
                contour.push(point);
                if contour.len() > boundary.len() + 1 {
                    return Err(invalid_graph(domain, "boundary contour does not close"));
                }
            }
            if contour.len() < 4
                || signed_area_polygon(&contour[..contour.len() - 1], &candidate.points).abs()
                    <= orientation_tolerance(context.target_size)
            {
                return Err(invalid_graph(domain, "boundary contour is degenerate"));
            }
            contours.push(simplify_contour(domain, context, candidate, contour));
        }

        let owner_by_edge = boundary
            .iter()
            .map(|edge| {
                (
                    ordered_pair(edge.points[0], edge.points[1]),
                    edge.owner.clone(),
                )
            })
            .collect::<BTreeMap<_, _>>();
        let edges = contours
            .iter()
            .flat_map(|contour| contour.windows(2))
            .map(|edge| GraphEdge {
                points: [edge[0], edge[1]],
                owner: owner_by_edge
                    .get(&ordered_pair(edge[0], edge[1]))
                    .cloned()
                    .flatten(),
            })
            .collect::<Vec<_>>();
        reject_crossings(domain, context, candidate, &edges)?;
        Ok(Self { contours, edges })
    }

    pub fn constraints(&self) -> BTreeSet<(PointKey, PointKey)> {
        self.edges
            .iter()
            .map(|edge| ordered_pair(edge.points[0], edge.points[1]))
            .collect()
    }

    pub fn contour_count(&self) -> usize {
        self.contours.len()
    }

    pub fn owned_edge_count(&self) -> usize {
        self.edges
            .iter()
            .filter(|edge| edge.owner.is_some())
            .count()
    }
}

fn simplify_contour(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
    contour: Vec<PointKey>,
) -> Vec<PointKey> {
    let mut vertices = contour[..contour.len() - 1].to_vec();
    let tolerance = chord_tolerance(domain, context.target_size);
    loop {
        let mut removed = false;
        for index in 0..vertices.len() {
            if vertices.len() <= 3 {
                break;
            }
            let previous = vertices[(index + vertices.len() - 1) % vertices.len()];
            let current = vertices[index];
            let next = vertices[(index + 1) % vertices.len()];
            // Core contour cleanup must not undo boundary-layer rediscretization.
            if candidate
                .layer_edge_targets
                .contains_key(&ordered_pair(previous, current))
                || candidate
                    .layer_edge_targets
                    .contains_key(&ordered_pair(current, next))
            {
                continue;
            }
            let a = Vec3::from_array(candidate.points[&previous].world);
            let point = Vec3::from_array(candidate.points[&current].world);
            let b = Vec3::from_array(candidate.points[&next].world);
            let midpoint = (a + b) * 0.5;
            let before = point - a;
            let after = b - point;
            let straight = before.length() > f64::EPSILON
                && after.length() > f64::EPSILON
                && before.dot(after) / (before.length() * after.length())
                    >= 0.923_879_532_511_286_7;
            if straight
                && (b - a).length() <= EDGE_RATIO_MAX * context.target_size
                && point_segment_distance(point, a, b) <= tolerance
                && domain.domain_sdf(&[midpoint])[0].abs() <= tolerance
            {
                vertices.remove(index);
                removed = true;
                break;
            }
        }
        if !removed {
            break;
        }
    }
    vertices.push(vertices[0]);
    vertices
}

pub(super) fn canonicalize_boundary_vertices(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    boundary: &[BoundaryEdge],
) -> MeshResult<bool> {
    let tolerance = root_tolerance(domain, context.target_size);
    let mut buckets = BTreeMap::<(i64, i64, i64), Vec<PointKey>>::new();
    let mut aliases = BTreeMap::<PointKey, PointKey>::new();
    let boundary_keys = boundary
        .iter()
        .flat_map(|edge| edge.points)
        .collect::<BTreeSet<_>>();

    for key in boundary_keys {
        context.check()?;
        let point = candidate.points[&key];
        let bucket = (
            (point.world[0] / tolerance).floor() as i64,
            (point.world[1] / tolerance).floor() as i64,
            (point.world[2] / tolerance).floor() as i64,
        );
        let mut representative = None;
        'nearby: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(found) = buckets
                        .get(&(bucket.0 + dx, bucket.1 + dy, bucket.2 + dz))
                        .and_then(|keys| {
                            keys.iter().copied().find(|other| {
                                distance3(point.world, candidate.points[other].world) <= tolerance
                            })
                        })
                    {
                        representative = Some(found);
                        break 'nearby;
                    }
                }
            }
        }
        if let Some(representative) = representative {
            if representative != key {
                aliases.insert(key, representative);
                if point.protected {
                    candidate
                        .points
                        .get_mut(&representative)
                        .expect("canonical boundary representative")
                        .protected = true;
                }
            }
        } else {
            buckets.entry(bucket).or_default().push(key);
        }
    }
    if aliases.is_empty() {
        return Ok(false);
    }

    for cell in &mut candidate.cells {
        for point in &mut cell.points {
            if let Some(representative) = aliases.get(point) {
                *point = *representative;
            }
        }
        if cell.points.iter().copied().collect::<BTreeSet<_>>().len() != cell.points.len() {
            return Err(invalid_graph(
                domain,
                "coincident boundary vertices collapse a mandatory cell",
            ));
        }
    }
    for key in aliases.keys() {
        candidate.points.remove(key);
    }
    Ok(true)
}

pub(super) fn project_to_owner(
    domain: &MeshableDomain,
    owner: Option<&str>,
    seed: Vec3,
    trust_distance: f64,
) -> MeshResult<Vec3> {
    let projection = if let Some(owner) = owner {
        domain
            .region_by_name(owner)
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?
            .project_to_owner(&[seed])[0]
    } else {
        domain
            .project_to_boundary(&[seed])
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?[0]
    };
    if projection.point.to_array().into_iter().all(f64::is_finite)
        && (owner.is_some()
            || (projection.converged && projection.distance_moved <= trust_distance))
    {
        return Ok(projection.point);
    }
    // Projection quality is best-effort. A trust-distance miss is not a
    // topological impossibility and must never turn an otherwise valid mesh
    // into a generation failure.
    Ok(seed)
}

pub(super) fn first_layer_transition(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    a: Vec3,
    b: Vec3,
) -> MeshResult<Option<(f64, String)>> {
    let edge_length = (b - a).length();
    let mut best = None::<(f64, String)>;
    for control in context
        .controls
        .boundary_layers
        .iter()
        .filter(|control| control.domain == domain.name)
    {
        let region = domain
            .region_by_name(&control.boundary_region)
            .map_err(|error| MeshError::InvalidInput(error.to_string()))?;
        let midpoint = (a + b) * 0.5;
        let owner_distance = region.owner_sdf(&[midpoint])[0].abs();
        let trust = edge_length
            .max(control.hwall_t)
            .min(domain.bounds.diagonal());
        if !owner_distance.is_finite() || owner_distance > trust {
            continue;
        }

        let points = (0..=16)
            .map(|sample| a + (b - a) * (sample as f64 / 16.0))
            .collect::<Vec<_>>();
        let parameters = if let Some(values) = region.selector_sdf(&points) {
            let mut roots = Vec::new();
            for interval in 0..16 {
                let left = values[interval];
                let right = values[interval + 1];
                if !left.is_finite() || !right.is_finite() || (left <= 0.0) == (right <= 0.0) {
                    continue;
                }
                let mut lo = interval as f64 / 16.0;
                let mut hi = (interval + 1) as f64 / 16.0;
                let left_inside = left <= 0.0;
                for _ in 0..56 {
                    let middle = 0.5 * (lo + hi);
                    let value = region
                        .selector_sdf(&[a + (b - a) * middle])
                        .expect("selector remains available")[0];
                    if (value <= 0.0) == left_inside {
                        lo = middle;
                    } else {
                        hi = middle;
                    }
                }
                roots.push(0.5 * (lo + hi));
            }
            roots
        } else {
            // Direction and CAD-patch transitions are already immutable
            // contour corners. Only selector fields can cross the interior
            // of a contour chord and therefore require an inserted station.
            Vec::new()
        };
        for parameter in parameters {
            if !(1.0e-6..=1.0 - 1.0e-6).contains(&parameter) {
                continue;
            }
            let entry = (parameter, control.boundary_region.clone());
            if best.as_ref().is_none_or(|current| parameter < current.0) {
                best = Some(entry);
            }
        }
    }
    Ok(best)
}

fn reject_crossings(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
    edges: &[GraphEdge],
) -> MeshResult<()> {
    let tolerance = root_tolerance(domain, context.target_size);
    for first in 0..edges.len() {
        if first.is_multiple_of(128) {
            context.check()?;
        }
        for second in first + 1..edges.len() {
            let a = edges[first].points;
            let b = edges[second].points;
            let shared = a.into_iter().filter(|point| b.contains(point)).count();
            if shared > 0 {
                continue;
            }
            let [a0, a1] = a.map(|key| candidate.points[&key].uv);
            let [b0, b1] = b.map(|key| candidate.points[&key].uv);
            let canonical = context.target_size * 0.10;
            if [a0, a1].into_iter().any(|a| {
                [b0, b1]
                    .into_iter()
                    .any(|b| (a[0] - b[0]).hypot(a[1] - b[1]) <= canonical)
            }) {
                continue;
            }
            if segments_conflict(a0, a1, b0, b1, tolerance) {
                return Err(invalid_graph(
                    domain,
                    &format!(
                        "contains crossed boundary constraints {a0:?}->{a1:?} and {b0:?}->{b1:?}"
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn segments_conflict(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2], tol: f64) -> bool {
    if segments_cross(a, b, c, d) {
        return true;
    }
    let length = (b[0] - a[0]).hypot(b[1] - a[1]);
    let area_tolerance = tol * length.max(f64::MIN_POSITIVE);
    let collinear =
        cross_2d(a, b, c).abs() <= area_tolerance && cross_2d(a, b, d).abs() <= area_tolerance;
    if !collinear {
        return false;
    }
    let axis = usize::from((b[1] - a[1]).abs() > (b[0] - a[0]).abs());
    let (a0, a1) = ordered_f64(a[axis], b[axis]);
    let (b0, b1) = ordered_f64(c[axis], d[axis]);
    a0.max(b0) < a1.min(b1) - tol
}

fn ordered_f64(a: f64, b: f64) -> (f64, f64) {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

fn invalid_graph(domain: &MeshableDomain, reason: &str) -> MeshError {
    MeshError::InvalidInput(format!(
        "domain {:?} planar constraint graph {reason}",
        domain.name
    ))
}
