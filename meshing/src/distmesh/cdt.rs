use std::collections::VecDeque;

use spade::handles::FixedVertexHandle;
use spade::{AngleLimit, RefinementParameters};

use super::contour::PlanarConstraintGraph;
use super::*;

const MINIMUM_ANGLE_DEGREES: f64 = 28.0;
const REFINEMENT_BATCH: usize = 512;
const MAX_REFINEMENT_BATCHES: usize = 64;

pub(super) fn retriangulate(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    graph: &PlanarConstraintGraph,
    refine: bool,
) -> MeshResult<()> {
    retriangulate_once(domain, space, context, candidate, graph, refine, 4)?;
    let used = candidate
        .cells
        .iter()
        .flat_map(|cell| cell.points.iter().copied())
        .collect::<BTreeSet<_>>();
    candidate.points.retain(|key, _| used.contains(key));
    Ok(())
}

fn retriangulate_once(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    graph: &PlanarConstraintGraph,
    refine: bool,
    repair_passes: u8,
) -> MeshResult<()> {
    let leaves = candidate
        .cells
        .iter()
        .flat_map(|cell| cell.points.iter().map(move |key| (*key, cell.leaf)))
        .collect::<BTreeMap<_, _>>();
    // A lattice sample can land numerically on a CAD corner while still
    // carrying an interior sign.  Keeping it beside the immutable contour
    // creates an unrefinable sliver, so canonicalize such seeds into the
    // boundary by omitting them from the global point load.
    let graph_constraints = graph.constraints();
    let graph_points = graph_constraints
        .iter()
        .flat_map(|&(a, b)| [a, b])
        .collect::<BTreeSet<_>>();
    let layer_constraint_points = graph_constraints
        .iter()
        .filter(|edge| candidate.layer_edge_targets.contains_key(edge))
        .flat_map(|&(a, b)| [a, b])
        .collect::<BTreeSet<_>>();
    let canonical_tolerance =
        root_tolerance(domain, context.target_size).max(context.target_size * 0.025);
    let mut representative_buckets = BTreeMap::<(i64, i64, i64), Vec<PointKey>>::new();
    let mut graph_aliases = BTreeMap::<PointKey, PointKey>::new();
    for &key in &graph_points {
        let point = candidate.points[&key].world;
        let bucket = (
            (point[0] / canonical_tolerance).floor() as i64,
            (point[1] / canonical_tolerance).floor() as i64,
            (point[2] / canonical_tolerance).floor() as i64,
        );
        // Boundary-layer stations are authored constraints, even when their
        // spacing is much smaller than the core target size.
        if layer_constraint_points.contains(&key) {
            representative_buckets.entry(bucket).or_default().push(key);
            continue;
        }
        let mut representative = None;
        'nearby_representative: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(found) = representative_buckets
                        .get(&(bucket.0 + dx, bucket.1 + dy, bucket.2 + dz))
                        .and_then(|keys| {
                            keys.iter().copied().find(|other| {
                                distance3(point, candidate.points[other].world)
                                    <= canonical_tolerance
                            })
                        })
                    {
                        representative = Some(found);
                        break 'nearby_representative;
                    }
                }
            }
        }
        if let Some(representative) = representative {
            graph_aliases.insert(key, representative);
        } else {
            representative_buckets.entry(bucket).or_default().push(key);
        }
    }
    let short_constraint_tolerance = context.target_size * 0.50;
    let chord_limit = chord_tolerance(domain, context.target_size);
    for edge in &graph.edges {
        let [a, b] = edge.points;
        // The core may coalesce short CAD chords, but never hwall_t edges.
        if candidate
            .layer_edge_targets
            .contains_key(&ordered_pair(a, b))
        {
            continue;
        }
        let a_root = resolved_alias(&graph_aliases, a);
        let b_root = resolved_alias(&graph_aliases, b);
        if a_root == b_root {
            continue;
        }
        let aw = candidate.points[&a_root].world;
        let bw = candidate.points[&b_root].world;
        if distance3(aw, bw) <= short_constraint_tolerance
            && domain.domain_sdf(&[Vec3::from_array(midpoint3(aw, bw))])[0].abs() <= chord_limit
        {
            let representative = a_root.min(b_root);
            let alias = a_root.max(b_root);
            let seed = Vec3::from_array(midpoint3(aw, bw));
            let projected = contour::project_to_owner(
                domain,
                edge.owner.as_deref(),
                seed,
                short_constraint_tolerance,
            )?;
            if projected.to_array().into_iter().all(f64::is_finite)
                && domain.domain_sdf(&[projected])[0].abs() <= chord_limit
                && (projected - seed).length() <= short_constraint_tolerance
            {
                let coords = space.coords(projected);
                let point = candidate
                    .points
                    .get_mut(&representative)
                    .expect("short constraint representative");
                point.uv = [coords[0], coords[1]];
                point.world = projected.to_array();
                point.boundary = true;
            }
            graph_aliases.insert(alias, representative);
            for value in graph_aliases.values_mut() {
                if *value == alias {
                    *value = representative;
                }
            }
        }
    }
    let seed_tolerance =
        (root_tolerance(domain, context.target_size) * 4.0).max(context.target_size * 0.20);
    let seed_keys = candidate
        .points
        .iter()
        .filter_map(|(&key, point)| {
            (!point.boundary && !graph_points.contains(&key)).then_some(key)
        })
        .collect::<Vec<_>>();
    let seed_values = domain.domain_sdf(
        &seed_keys
            .iter()
            .map(|key| Vec3::from_array(candidate.points[key].world))
            .collect::<Vec<_>>(),
    );
    let mut boundary_buckets = BTreeMap::<(i64, i64, i64), Vec<PointKey>>::new();
    for &key in &graph_points {
        let point = candidate.points[&key].world;
        boundary_buckets
            .entry((
                (point[0] / seed_tolerance).floor() as i64,
                (point[1] / seed_tolerance).floor() as i64,
                (point[2] / seed_tolerance).floor() as i64,
            ))
            .or_default()
            .push(key);
    }
    let mut rejected_seeds = seed_keys
        .into_iter()
        .zip(seed_values)
        .filter_map(|(key, value)| {
            let point = candidate.points[&key].world;
            let bucket = (
                (point[0] / seed_tolerance).floor() as i64,
                (point[1] / seed_tolerance).floor() as i64,
                (point[2] / seed_tolerance).floor() as i64,
            );
            let mut near_boundary_vertex = false;
            'nearby: for dx in -1..=1 {
                for dy in -1..=1 {
                    for dz in -1..=1 {
                        if boundary_buckets
                            .get(&(bucket.0 + dx, bucket.1 + dy, bucket.2 + dz))
                            .is_some_and(|keys| {
                                keys.iter().any(|other| {
                                    distance3(point, candidate.points[other].world)
                                        <= seed_tolerance
                                })
                            })
                        {
                            near_boundary_vertex = true;
                            break 'nearby;
                        }
                    }
                }
            }
            let near_boundary_edge = !near_boundary_vertex
                && value.abs() <= context.target_size * 0.01
                && graph_constraints.iter().any(|&(a, b)| {
                    point_segment_distance(
                        Vec3::from_array(point),
                        Vec3::from_array(candidate.points[&a].world),
                        Vec3::from_array(candidate.points[&b].world),
                    ) <= seed_tolerance
                });
            (value.abs() <= seed_tolerance || near_boundary_vertex || near_boundary_edge)
                .then_some(key)
        })
        .collect::<BTreeSet<_>>();
    let minimum_seed_spacing = context.target_size * 0.75;
    let mut spacing_buckets = BTreeMap::<(i64, i64, i64), Vec<PointKey>>::new();
    for &key in &graph_points {
        let point = candidate.points[&key].world;
        spacing_buckets
            .entry((
                (point[0] / minimum_seed_spacing).floor() as i64,
                (point[1] / minimum_seed_spacing).floor() as i64,
                (point[2] / minimum_seed_spacing).floor() as i64,
            ))
            .or_default()
            .push(key);
    }
    for &key in candidate.points.keys() {
        if graph_points.contains(&key)
            || candidate.points[&key].protected
            || rejected_seeds.contains(&key)
        {
            continue;
        }
        let point = candidate.points[&key].world;
        let bucket = (
            (point[0] / minimum_seed_spacing).floor() as i64,
            (point[1] / minimum_seed_spacing).floor() as i64,
            (point[2] / minimum_seed_spacing).floor() as i64,
        );
        let mut too_close = false;
        'nearby_seed: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if spacing_buckets
                        .get(&(bucket.0 + dx, bucket.1 + dy, bucket.2 + dz))
                        .is_some_and(|keys| {
                            keys.iter().any(|other| {
                                distance3(point, candidate.points[other].world)
                                    < minimum_seed_spacing
                            })
                        })
                    {
                        too_close = true;
                        break 'nearby_seed;
                    }
                }
            }
        }
        if too_close {
            rejected_seeds.insert(key);
        } else {
            spacing_buckets.entry(bucket).or_default().push(key);
        }
    }
    let keys = candidate
        .points
        .keys()
        .copied()
        .filter(|key| {
            !rejected_seeds.contains(key)
                && !graph_aliases.contains_key(key)
                && (!candidate.points[key].boundary || graph_points.contains(key))
        })
        .collect::<Vec<_>>();
    let mut indices = keys
        .iter()
        .enumerate()
        .map(|(index, key)| (*key, index))
        .collect::<BTreeMap<_, _>>();
    for (&alias, &representative) in &graph_aliases {
        indices.insert(alias, indices[&representative]);
    }
    let vertices = keys
        .iter()
        .map(|key| {
            let uv = candidate.points[key].uv;
            Point2::new(uv[0], uv[1])
        })
        .collect::<Vec<_>>();
    let constraints = graph_constraints
        .iter()
        .filter_map(|&(a, b)| {
            let edge = [indices[&a], indices[&b]];
            (edge[0] != edge[1]).then_some(edge)
        })
        .collect::<Vec<_>>();
    let mut conflicts = Vec::new();
    let mut triangulation =
        PointSpade::try_bulk_load_cdt(vertices, constraints, |edge| conflicts.push(edge)).map_err(
            |error| {
                MeshError::InvalidInput(format!(
                    "Spade rejected the global constraint graph for domain {:?}: {error:?}",
                    domain.name
                ))
            },
        )?;
    if triangulation.num_vertices() != keys.len() {
        return Err(invalid_cdt(
            domain,
            "contains coincident vertices after canonicalization",
        ));
    }
    if !conflicts.is_empty() {
        return Err(invalid_cdt(domain, "contains intersecting constraints"));
    }
    let (_, mut key_handles) = reordered_input_keys(domain, &triangulation, candidate, &keys)?;
    for (&alias, &representative) in &graph_aliases {
        key_handles.insert(alias, key_handles[&representative]);
    }
    verify_constraints(domain, &triangulation, &graph_constraints, &key_handles)?;

    let keep_constraint_edges = graph.edges.iter().any(|edge| {
        edge.points
            .iter()
            .any(|key| candidate.points[key].protected)
    });
    let _outer_faces = if refine {
        refine_in_batches(
            domain,
            context,
            candidate,
            &mut triangulation,
            keep_constraint_edges,
            true,
        )?
    } else {
        classify_outer_faces(&mut triangulation)
    };
    verify_constraints(domain, &triangulation, &graph_constraints, &key_handles)?;

    let input_by_position = keys
        .iter()
        .map(|&key| (point_bits(candidate.points[&key].uv), key))
        .collect::<BTreeMap<_, _>>();
    let mut handle_keys = Vec::with_capacity(triangulation.num_vertices());
    for index in 0..triangulation.num_vertices() {
        let handle = FixedVertexHandle::from_index(index);
        let position = triangulation.vertex(handle).position();
        if let Some(&key) = input_by_position.get(&point_bits([position.x, position.y])) {
            handle_keys.push(key);
            continue;
        }
        let world = space.point(position.x, position.y);
        let boundary = triangulation
            .vertex(handle)
            .out_edges()
            .any(|edge| edge.is_constraint_edge());
        let key = PointKey::Inserted(candidate.next_inserted);
        candidate.next_inserted += 1;
        candidate.points.insert(
            key,
            Point {
                uv: [position.x, position.y],
                world: world.to_array(),
                boundary,
                protected: false,
            },
        );
        handle_keys.push(key);
    }

    let cells = interior_cells(
        domain,
        context,
        candidate,
        &triangulation,
        &handle_keys,
        &leaves,
        &BTreeSet::new(),
    )?;
    if cells.is_empty() {
        return Err(invalid_cdt(domain, "produced no interior triangles"));
    }
    candidate.cells = cells;
    let removable = if repair_passes > 0 {
        bad_near_boundary_vertices(domain, context, candidate)
    } else {
        BTreeSet::new()
    };
    if !removable.is_empty() {
        for key in &removable {
            candidate.points.remove(key);
        }
        let mut retained = candidate
            .cells
            .iter()
            .flat_map(|cell| cell.points.iter().copied())
            .filter(|key| !removable.contains(key))
            .collect::<BTreeSet<_>>();
        retained.extend(graph_points);
        retained.extend(
            candidate
                .points
                .iter()
                .filter_map(|(&key, point)| point.protected.then_some(key)),
        );
        candidate.points.retain(|key, _| retained.contains(key));
        return retriangulate_with_repairs(
            domain,
            space,
            context,
            candidate,
            graph,
            true,
            repair_passes - 1,
        );
    }
    let repairs = if repair_passes > 0 {
        seed_bad_boundary_cavities(domain, space, context, candidate)
    } else {
        Vec::new()
    };
    if !repairs.is_empty() {
        let mut retained = candidate
            .cells
            .iter()
            .flat_map(|cell| cell.points.iter().copied())
            .collect::<BTreeSet<_>>();
        retained.extend(graph_points);
        retained.extend(
            candidate
                .points
                .iter()
                .filter_map(|(&key, point)| point.protected.then_some(key)),
        );
        candidate.points.retain(|key, _| retained.contains(key));
        let result = retriangulate_with_repairs(
            domain,
            space,
            context,
            candidate,
            graph,
            refine,
            repair_passes - 1,
        );
        for key in repairs {
            if let Some(point) = candidate.points.get_mut(&key) {
                point.protected = false;
            }
        }
        return result;
    }
    candidate.construction_failures.clear();
    let mut used = candidate
        .cells
        .iter()
        .flat_map(|cell| cell.points.iter().copied())
        .collect::<BTreeSet<_>>();
    used.extend(graph_points);
    candidate.points.retain(|key, _| used.contains(key));
    Ok(())
}

fn bad_near_boundary_vertices(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
) -> BTreeSet<PointKey> {
    if context
        .controls
        .boundary_layers
        .iter()
        .any(|control| control.domain == domain.name)
    {
        return BTreeSet::new();
    }
    let mut bad_regions = Vec::new();
    let mut forced = BTreeSet::new();
    for cell in &candidate.cells {
        if cell.protected || cell.points.len() != 3 {
            continue;
        }
        let positions = cell
            .points
            .iter()
            .map(|key| candidate.points[key].world)
            .collect::<Vec<_>>();
        for edge in 0..3 {
            if distance3(positions[edge], positions[(edge + 1) % 3]) + 1.0e-12 * context.target_size
                >= EDGE_RATIO_MIN * context.target_size
            {
                continue;
            }
            let mut endpoints = [cell.points[edge], cell.points[(edge + 1) % 3]]
                .into_iter()
                .filter(|key| {
                    let point = candidate.points[key];
                    !point.boundary && !point.protected && !forced.contains(key)
                })
                .collect::<Vec<_>>();
            endpoints.sort_unstable();
            if let Some(key) = endpoints.pop() {
                forced.insert(key);
            }
        }
        let quality =
            quality_score("tri3", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0);
        let skewness = quality_score("tri3", &positions, QualityMetric::Skewness).unwrap_or(1.0);
        if quality + 1.0e-12 >= QUALITY_TARGET && skewness <= 0.60 + 1.0e-12 {
            continue;
        }
        bad_regions.push(centroid_slice(&positions));
        let (edge, length) = (0..3)
            .map(|edge| (edge, distance3(positions[edge], positions[(edge + 1) % 3])))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("triangle edge");
        if length < context.target_size * 0.80 {
            let mut endpoints = [cell.points[edge], cell.points[(edge + 1) % 3]]
                .into_iter()
                .filter(|key| {
                    let point = candidate.points[key];
                    !point.boundary && !point.protected
                })
                .collect::<Vec<_>>();
            endpoints.sort_unstable();
            if let Some(key) = endpoints.pop() {
                forced.insert(key);
            }
        }
    }
    if bad_regions.is_empty() {
        return forced;
    }
    let movable = candidate
        .points
        .iter()
        .filter_map(|(&key, point)| {
            (!point.boundary && !point.protected).then_some((key, point.world))
        })
        .collect::<Vec<_>>();
    let values = domain.domain_sdf(
        &movable
            .iter()
            .map(|(_, world)| Vec3::from_array(*world))
            .collect::<Vec<_>>(),
    );
    forced.extend(
        movable
            .into_iter()
            .zip(values)
            .filter_map(|((key, world), value)| {
                (value.is_finite()
                    && value.abs() < context.target_size
                    && bad_regions
                        .iter()
                        .any(|center| distance3(*center, world) < context.target_size * 2.0))
                .then_some(key)
            })
            .collect::<BTreeSet<_>>(),
    );
    forced
}

fn retriangulate_with_repairs(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    graph: &PlanarConstraintGraph,
    refine: bool,
    repair_passes: u8,
) -> MeshResult<()> {
    retriangulate_once(
        domain,
        space,
        context,
        candidate,
        graph,
        refine,
        repair_passes,
    )
}

fn seed_bad_boundary_cavities(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
) -> Vec<PointKey> {
    if context
        .controls
        .boundary_layers
        .iter()
        .any(|control| control.domain == domain.name)
    {
        return Vec::new();
    }
    let mut seeds = Vec::new();
    let mut relocations = BTreeMap::new();
    for cell in &candidate.cells {
        if cell.points.len() != 3
            || cell.protected
            || cell
                .points
                .iter()
                .any(|key| candidate.points[key].protected)
            || cell
                .points
                .iter()
                .filter(|key| candidate.points[key].boundary)
                .count()
                < 2
        {
            continue;
        }
        let positions = cell
            .points
            .iter()
            .map(|key| candidate.points[key].world)
            .collect::<Vec<_>>();
        let quality =
            quality_score("tri3", &positions, QualityMetric::ScaledJacobian).unwrap_or(0.0);
        let skewness = quality_score("tri3", &positions, QualityMetric::Skewness).unwrap_or(1.0);
        if quality + 1.0e-12 >= QUALITY_TARGET && skewness <= 0.60 + 1.0e-12 {
            continue;
        }
        let (edge, length) = (0..3)
            .filter(|edge| {
                candidate.points[&cell.points[*edge]].boundary
                    && candidate.points[&cell.points[(*edge + 1) % 3]].boundary
            })
            .map(|edge| (edge, distance3(positions[edge], positions[(edge + 1) % 3])))
            .min_by(|a, b| a.1.total_cmp(&b.1))
            .expect("triangle edge");
        let a = candidate.points[&cell.points[edge]].uv;
        let b = candidate.points[&cell.points[(edge + 1) % 3]].uv;
        let midpoint = [0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])];
        let tangent = [b[0] - a[0], b[1] - a[1]];
        if length <= f64::EPSILON {
            continue;
        }
        let normal = [-tangent[1] / length, tangent[0] / length];
        let distance = 0.5 * 3.0_f64.sqrt() * length;
        let Some((uv, world, _)) = [1.0, -1.0]
            .into_iter()
            .map(|sign| {
                let uv = [
                    midpoint[0] + sign * normal[0] * distance,
                    midpoint[1] + sign * normal[1] * distance,
                ];
                let world = space.point(uv[0], uv[1]);
                let value = domain.domain_sdf(&[world])[0];
                (uv, world, value)
            })
            .filter(|(_, _, value)| value.is_finite() && *value < 0.0)
            .min_by(|a, b| a.2.total_cmp(&b.2))
        else {
            continue;
        };
        if candidate.points.values().any(|point| {
            distance3(point.world, world.to_array()) <= root_tolerance(domain, context.target_size)
        }) || seeds.iter().any(|seed: &Point| {
            distance3(seed.world, world.to_array()) <= root_tolerance(domain, context.target_size)
        }) {
            continue;
        }
        let point = Point {
            uv,
            world: world.to_array(),
            boundary: false,
            protected: true,
        };
        let opposite = cell.points[(edge + 2) % 3];
        if !candidate.points[&opposite].boundary && !candidate.points[&opposite].protected {
            relocations.entry(opposite).or_insert(point);
        } else {
            seeds.push(point);
        }
        if seeds.len() + relocations.len() == REFINEMENT_BATCH {
            break;
        }
    }
    let mut keys = Vec::with_capacity(seeds.len() + relocations.len());
    for (key, point) in relocations {
        candidate.points.insert(key, point);
        keys.push(key);
    }
    for point in seeds {
        let key = PointKey::Inserted(candidate.next_inserted);
        candidate.next_inserted += 1;
        candidate.points.insert(key, point);
        keys.push(key);
    }
    keys
}

pub(super) fn triangulate_core(
    domain: &MeshableDomain,
    space: &MeshableDomainSpace,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    strip: &BoundaryLayerStrip,
    constraints: &BTreeSet<(PointKey, PointKey)>,
    leaves: &BTreeMap<PointKey, Leaf>,
) -> MeshResult<Vec<Cell>> {
    let keys = candidate.points.keys().copied().collect::<Vec<_>>();
    let indices = keys
        .iter()
        .enumerate()
        .map(|(index, key)| (*key, index))
        .collect::<BTreeMap<_, _>>();
    let vertices = keys
        .iter()
        .map(|key| {
            let uv = candidate.points[key].uv;
            Point2::new(uv[0], uv[1])
        })
        .collect::<Vec<_>>();
    let edges = constraints
        .iter()
        .map(|&(a, b)| [indices[&a], indices[&b]])
        .collect::<Vec<_>>();
    let mut conflicts = Vec::new();
    let mut triangulation =
        PointSpade::try_bulk_load_cdt(vertices, edges, |edge| conflicts.push(edge)).map_err(
            |error| {
                MeshError::InvalidInput(format!(
                    "Spade rejected the protected core graph for domain {:?}: {error:?}",
                    domain.name
                ))
            },
        )?;
    if triangulation.num_vertices() != keys.len() {
        return Err(invalid_cdt(
            domain,
            "contains coincident protected-core vertices",
        ));
    }
    if !conflicts.is_empty() {
        return Err(invalid_cdt(
            domain,
            "contains crossed protected-core constraints",
        ));
    }
    let (input_handle_keys, key_handles) =
        reordered_input_keys(domain, &triangulation, candidate, &keys)?;
    verify_constraints(domain, &triangulation, constraints, &key_handles)?;
    let handle_keys = if candidate.refine_layer_core {
        let _outer_faces =
            refine_in_batches(domain, context, candidate, &mut triangulation, true, false)?;
        verify_constraints(domain, &triangulation, constraints, &key_handles)?;
        let input_by_position = input_handle_keys
            .iter()
            .map(|&key| (point_bits(candidate.points[&key].uv), key))
            .collect::<BTreeMap<_, _>>();
        let mut refined_keys = Vec::with_capacity(triangulation.num_vertices());
        for index in 0..triangulation.num_vertices() {
            let handle = FixedVertexHandle::from_index(index);
            let position = triangulation.vertex(handle).position();
            if let Some(&key) = input_by_position.get(&point_bits([position.x, position.y])) {
                refined_keys.push(key);
                continue;
            }
            let world = space.point(position.x, position.y);
            let key = PointKey::Inserted(candidate.next_inserted);
            candidate.next_inserted += 1;
            candidate.points.insert(
                key,
                Point {
                    uv: [position.x, position.y],
                    world: world.to_array(),
                    boundary: false,
                    protected: false,
                },
            );
            refined_keys.push(key);
        }
        refined_keys
    } else {
        input_handle_keys
    };

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
    let strip_boundary = strip_incidence
        .into_iter()
        .filter_map(|(edge, count)| (count == 1).then_some(edge))
        .collect::<BTreeSet<_>>();
    let mut cells = interior_cells(
        domain,
        context,
        candidate,
        &triangulation,
        &handle_keys,
        leaves,
        &strip_boundary,
    )?;
    if cells.is_empty() {
        return Err(invalid_cdt(domain, "produced no protected core triangles"));
    }
    cells.extend(strip.cells.iter().cloned());
    Ok(cells)
}

pub(super) fn interior_cells(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &Candidate,
    triangulation: &PointSpade,
    handle_keys: &[PointKey],
    leaves: &BTreeMap<PointKey, Leaf>,
    excluded_region_boundary: &BTreeSet<(PointKey, PointKey)>,
) -> MeshResult<Vec<Cell>> {
    let mut faces = Vec::<[PointKey; 3]>::new();
    let mut handles = Vec::<[usize; 3]>::new();
    for (index, face) in triangulation.inner_faces().enumerate() {
        if index.is_multiple_of(512) {
            context.check()?;
        }
        let face_handles = face.vertices().map(|vertex| vertex.fix().index());
        handles.push(face_handles);
        faces.push(face_handles.map(|handle| handle_keys[handle]));
    }

    let mut incidence = BTreeMap::<(usize, usize), Vec<usize>>::new();
    for (face, points) in handles.iter().enumerate() {
        for edge in 0..3 {
            let a = points[edge];
            let b = points[(edge + 1) % 3];
            incidence.entry(ordered_pair(a, b)).or_default().push(face);
        }
    }
    if incidence.values().any(|faces| faces.len() > 2) {
        return Err(invalid_cdt(domain, "produced non-manifold face incidence"));
    }

    let is_domain_barrier = |edge: (usize, usize)| {
        let a = FixedVertexHandle::from_index(edge.0);
        let b = FixedVertexHandle::from_index(edge.1);
        triangulation.exists_constraint(a, b)
            && candidate.points[&handle_keys[edge.0]].boundary
            && candidate.points[&handle_keys[edge.1]].boundary
    };
    let is_excluded_barrier = |edge: (usize, usize)| {
        excluded_region_boundary.contains(&ordered_pair(handle_keys[edge.0], handle_keys[edge.1]))
    };
    let domain_barrier_count = incidence
        .keys()
        .filter(|&&edge| is_domain_barrier(edge))
        .count();
    let mut domain_barrier_degree = BTreeMap::<usize, usize>::new();
    for &edge in incidence.keys().filter(|&&edge| is_domain_barrier(edge)) {
        *domain_barrier_degree.entry(edge.0).or_default() += 1;
        *domain_barrier_degree.entry(edge.1).or_default() += 1;
    }
    let odd_domain_barrier_vertices = domain_barrier_degree
        .values()
        .filter(|degree| **degree % 2 != 0)
        .count();
    let domain_parity = flood_winding(
        domain,
        candidate,
        handle_keys,
        &handles,
        &incidence,
        is_domain_barrier,
        true,
    )?;
    let excluded_parity = flood_winding(
        domain,
        candidate,
        handle_keys,
        &handles,
        &incidence,
        is_excluded_barrier,
        false,
    )?;

    let mut result = Vec::new();
    for (index, points) in faces.into_iter().enumerate() {
        if !domain_parity[index] || excluded_parity[index] {
            continue;
        }
        let area = signed_area(points, &candidate.points);
        if area.abs() <= orientation_tolerance(context.target_size) {
            continue;
        }
        let leaf = points
            .iter()
            .find_map(|key| leaves.get(key))
            .copied()
            .unwrap_or(Leaf {
                level: 0,
                x: 0,
                y: 0,
            });
        let mut cell = Cell::triangle(points, leaf);
        if area < 0.0 {
            cell.points.swap(1, 2);
        }
        result.push(cell);
    }
    if result.is_empty() {
        let first_negative = handles.iter().find_map(|face| {
            let points = face.map(|handle| candidate.points[&handle_keys[handle]]);
            let world = (Vec3::from_array(points[0].world)
                + Vec3::from_array(points[1].world)
                + Vec3::from_array(points[2].world))
                / 3.0;
            (domain.domain_sdf(&[world])[0] < 0.0).then_some([
                (points[0].uv[0] + points[1].uv[0] + points[2].uv[0]) / 3.0,
                (points[0].uv[1] + points[1].uv[1] + points[2].uv[1]) / 3.0,
            ])
        });
        let negative_centroids = handles
            .iter()
            .filter(|face| {
                let points = face
                    .map(|handle| Vec3::from_array(candidate.points[&handle_keys[handle]].world));
                domain.domain_sdf(&[(points[0] + points[1] + points[2]) / 3.0])[0] < 0.0
            })
            .count();
        return Err(invalid_cdt(
            domain,
            &format!(
                "classified no interior faces ({} faces, {domain_barrier_count} domain barriers, {odd_domain_barrier_vertices} odd barrier vertices, {} inside labels, {negative_centroids} negative-SDF face centroids, first={first_negative:?})",
                handles.len(),
                domain_parity.iter().filter(|inside| **inside).count(),
            ),
        ));
    }
    Ok(result)
}

fn refine_in_batches(
    domain: &MeshableDomain,
    context: &MeshingContext<'_>,
    candidate: &mut Candidate,
    triangulation: &mut PointSpade,
    keep_constraint_edges: bool,
    exclude_outer_faces: bool,
) -> MeshResult<BTreeSet<usize>> {
    let max_cells = usize::try_from(context.limits.max_cells).unwrap_or(usize::MAX);
    let max_area = 0.25 * 3.0_f64.sqrt() * context.target_size.powi(2);
    // A positive floor is essential around acute immutable PSLG corners:
    // refining all the way to root-finding epsilon creates numerically
    // coincident Steiner points without improving the input angle.
    let min_area = root_tolerance(domain, context.target_size)
        .powi(2)
        .max(max_area * 1.0e-6);
    for batch in 0..MAX_REFINEMENT_BATCHES {
        context.check()?;
        let face_headroom = max_cells.saturating_sub(triangulation.num_inner_faces());
        let additions = REFINEMENT_BATCH.min(face_headroom / 2);
        if additions == 0 {
            candidate.layer_refinement_limit = Some(QualityTermination::MaxCells);
            return Ok(classify_outer_faces(triangulation));
        }
        if estimated_cdt_bytes(triangulation).saturating_mul(2) > MAX_OPTIMIZATION_BYTES {
            candidate.layer_refinement_limit = Some(QualityTermination::MemoryBudget);
            return Ok(classify_outer_faces(triangulation));
        }
        let parameters = RefinementParameters::new()
            .exclude_outer_faces(exclude_outer_faces)
            .with_angle_limit(AngleLimit::from_deg(MINIMUM_ANGLE_DEGREES))
            .with_min_required_area(min_area)
            .with_max_allowed_area(max_area)
            .with_max_additional_vertices(additions);
        let parameters = if keep_constraint_edges {
            parameters.keep_constraint_edges()
        } else {
            parameters
        };
        let result = triangulation.refine(parameters);
        let excluded = result
            .excluded_faces
            .into_iter()
            .map(|face| face.index())
            .collect::<BTreeSet<_>>();
        if result.refinement_complete {
            return Ok(excluded);
        }
        if batch + 1 == MAX_REFINEMENT_BATCHES {
            candidate.layer_refinement_limit = Some(QualityTermination::IterationLimit);
        }
    }
    Ok(classify_outer_faces(triangulation))
}

fn classify_outer_faces(triangulation: &mut PointSpade) -> BTreeSet<usize> {
    triangulation
        .refine(
            RefinementParameters::new()
                .exclude_outer_faces(true)
                .keep_constraint_edges()
                .with_max_additional_vertices(0),
        )
        .excluded_faces
        .into_iter()
        .map(|face| face.index())
        .collect()
}

/// Flood each region separated by constraints, then classify the region once
/// by winding against the immutable constraint graph.  Classifying connected
/// regions avoids the contradictory hull seeds that arise when a curved or
/// concave contour shares only part of a Delaunay hull face.
fn flood_winding(
    domain: &MeshableDomain,
    candidate: &Candidate,
    handle_keys: &[PointKey],
    handles: &[[usize; 3]],
    incidence: &BTreeMap<(usize, usize), Vec<usize>>,
    is_barrier: impl Fn((usize, usize)) -> bool,
    classify_with_domain: bool,
) -> MeshResult<Vec<bool>> {
    let mut adjacency = vec![Vec::<usize>::new(); handles.len()];
    let mut barriers = Vec::new();
    for (&edge, adjacent) in incidence {
        if is_barrier(edge) {
            barriers.push(edge);
        } else if let [first, second] = adjacent.as_slice() {
            adjacency[*first].push(*second);
            adjacency[*second].push(*first);
        }
    }

    let winding_at = |point: [f64; 2]| {
        barriers.iter().fold(false, |inside, &(a, b)| {
            let a = candidate.points[&handle_keys[a]].uv;
            let b = candidate.points[&handle_keys[b]].uv;
            let crosses = (a[1] > point[1]) != (b[1] > point[1])
                && point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0];
            inside ^ crosses
        })
    };
    if classify_with_domain {
        return Ok(handles
            .iter()
            .map(|face| {
                let triangle = face.map(|handle| candidate.points[&handle_keys[handle]].uv);
                winding_at([
                    (triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3.0,
                    (triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3.0,
                ])
            })
            .collect());
    }

    let mut labels = vec![None; handles.len()];
    for seed in 0..handles.len() {
        if labels[seed].is_some() {
            continue;
        }
        let triangle = handles[seed].map(|handle| candidate.points[&handle_keys[handle]].uv);
        let point = [
            (triangle[0][0] + triangle[1][0] + triangle[2][0]) / 3.0,
            (triangle[0][1] + triangle[1][1] + triangle[2][1]) / 3.0,
        ];
        let inside = winding_at(point);

        let mut queue = VecDeque::from([seed]);
        labels[seed] = Some(inside);
        while let Some(face) = queue.pop_front() {
            for &other in &adjacency[face] {
                if labels[other].is_none() {
                    labels[other] = Some(inside);
                    queue.push_back(other);
                }
            }
        }
    }
    labels
        .into_iter()
        .map(|label| label.ok_or_else(|| invalid_cdt(domain, "contains an unreachable face")))
        .collect()
}

fn verify_constraints(
    domain: &MeshableDomain,
    triangulation: &PointSpade,
    constraints: &BTreeSet<(PointKey, PointKey)>,
    indices: &BTreeMap<PointKey, usize>,
) -> MeshResult<()> {
    for &(a, b) in constraints {
        let from = FixedVertexHandle::from_index(indices[&a]);
        let to = FixedVertexHandle::from_index(indices[&b]);
        if !constraint_path_exists(triangulation, from, to) {
            return Err(invalid_cdt(
                domain,
                "did not preserve every requested constraint",
            ));
        }
    }
    Ok(())
}

fn reordered_input_keys(
    domain: &MeshableDomain,
    triangulation: &PointSpade,
    candidate: &Candidate,
    keys: &[PointKey],
) -> MeshResult<(Vec<PointKey>, BTreeMap<PointKey, usize>)> {
    let input_by_position = keys
        .iter()
        .map(|&key| (point_bits(candidate.points[&key].uv), key))
        .collect::<BTreeMap<_, _>>();
    let mut handle_keys = Vec::with_capacity(triangulation.num_vertices());
    let mut key_handles = BTreeMap::new();
    for index in 0..triangulation.num_vertices() {
        let position = triangulation
            .vertex(FixedVertexHandle::from_index(index))
            .position();
        let key = input_by_position
            .get(&point_bits([position.x, position.y]))
            .copied()
            .ok_or_else(|| invalid_cdt(domain, "changed an input vertex during bulk loading"))?;
        handle_keys.push(key);
        key_handles.insert(key, index);
    }
    if key_handles.len() != keys.len() {
        return Err(invalid_cdt(
            domain,
            "could not map every bulk-loaded input vertex",
        ));
    }
    Ok((handle_keys, key_handles))
}

fn point_bits(point: [f64; 2]) -> [u64; 2] {
    point.map(f64::to_bits)
}

fn resolved_alias(aliases: &BTreeMap<PointKey, PointKey>, mut key: PointKey) -> PointKey {
    while let Some(&next) = aliases.get(&key) {
        key = next;
    }
    key
}

fn constraint_path_exists(
    triangulation: &PointSpade,
    from: FixedVertexHandle,
    to: FixedVertexHandle,
) -> bool {
    let start = triangulation.vertex(from).position();
    let end = triangulation.vertex(to).position();
    let direction = [end.x - start.x, end.y - start.y];
    let length2 = direction[0] * direction[0] + direction[1] * direction[1];
    let tolerance = f64::EPSILON * length2.sqrt().max(1.0) * 256.0;
    let mut pending = vec![from];
    let mut visited = BTreeSet::from([from]);
    while let Some(vertex) = pending.pop() {
        if vertex == to {
            return true;
        }
        for edge in triangulation
            .vertex(vertex)
            .out_edges()
            .filter(|edge| edge.is_constraint_edge())
        {
            let next = edge.to().fix();
            if visited.contains(&next) {
                continue;
            }
            let point = edge.to().position();
            let offset = [point.x - start.x, point.y - start.y];
            let cross = direction[0] * offset[1] - direction[1] * offset[0];
            let dot = direction[0] * offset[0] + direction[1] * offset[1];
            if cross.abs() <= tolerance && dot >= -tolerance && dot <= length2 + tolerance {
                visited.insert(next);
                pending.push(next);
            }
        }
    }
    false
}

fn estimated_cdt_bytes(triangulation: &PointSpade) -> usize {
    triangulation
        .num_vertices()
        .saturating_mul(160)
        .saturating_add(triangulation.num_inner_faces().saturating_mul(96))
}

fn invalid_cdt(domain: &MeshableDomain, reason: &str) -> MeshError {
    MeshError::InvalidInput(format!(
        "domain {:?} global constrained Delaunay triangulation {reason}",
        domain.name
    ))
}
