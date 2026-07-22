use std::collections::{BTreeMap, BTreeSet};

use caso_kernel::meshing::{BoundaryBand, MeshableDomain, MeshableDomains};
use caso_kernel::vec3::{vec3, Vec3};

use crate::controls::ControlSet;
use crate::{MeshIr, MeshIrBuilder};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MeshingOptions {
    pub cells_2d: usize,
    pub cells_3d: usize,
    pub minimum_cross_cells: usize,
    pub max_cells: usize,
    pub max_adaptive_levels: usize,
}

impl Default for MeshingOptions {
    fn default() -> Self {
        Self {
            cells_2d: 48,
            cells_3d: 20,
            minimum_cross_cells: 6,
            max_cells: 1_000_000,
            max_adaptive_levels: 12,
        }
    }
}

impl MeshingOptions {
    pub fn validate(self) -> Result<Self, String> {
        if self.cells_2d == 0 || self.cells_3d == 0 || self.minimum_cross_cells == 0 {
            return Err("meshing cell counts must be positive".into());
        }
        if self.max_cells == 0 || self.max_cells > 1_000_000 {
            return Err("meshing max_cells must be in 1..=1,000,000".into());
        }
        if self.max_adaptive_levels > 12 {
            return Err("meshing max_adaptive_levels must not exceed 12".into());
        }
        Ok(self)
    }
}

#[derive(Debug, Clone)]
pub struct MeshingRequest {
    pub domains: MeshableDomains,
    pub controls: ControlSet,
    pub options: MeshingOptions,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct MeshingStatistics {
    pub domains: usize,
    pub points: usize,
    pub cells: usize,
    pub adaptive_levels: usize,
}

#[derive(Debug, Clone)]
pub struct MeshingOutput {
    pub mesh: MeshIr,
    pub statistics: MeshingStatistics,
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AdaptiveSimplicialMesher;

impl AdaptiveSimplicialMesher {
    pub const fn new() -> Self {
        Self
    }

    pub fn mesh(&self, request: MeshingRequest) -> Result<MeshingOutput, String> {
        let options = request.options.validate()?;
        request.controls.validate(&request.domains)?;
        if request.domains.is_empty() {
            return Err("meshing requires at least one declared domain".into());
        }

        let mut builder = MeshIrBuilder::new();
        let mut cell_count = 0usize;
        let mut adaptive_levels = 0usize;
        for domain in request.domains.iter() {
            let zone = builder.zone(domain.name.clone(), domain.kind.as_str());
            let mut region_tags = BTreeMap::new();
            for region in &domain.boundary_regions {
                region_tags.insert(
                    region.name.clone(),
                    builder.tag(
                        region.name.clone(),
                        region.tag.clone().unwrap_or_else(|| "boundary".into()),
                    ),
                );
            }
            let wall_tag = builder.tag(format!("{}_wall", domain.name), "wall");
            let (added, levels) = match domain.dimension {
                2 => mesh_2d(
                    domain,
                    &request.controls,
                    options,
                    &mut builder,
                    zone,
                    &region_tags,
                    wall_tag,
                )?,
                3 => (
                    mesh_3d(
                        domain,
                        &request.controls,
                        options,
                        &mut builder,
                        zone,
                        &region_tags,
                        wall_tag,
                    )?,
                    0,
                ),
                dimension => {
                    return Err(format!(
                        "domain {:?} has unsupported meshing dimension {dimension}",
                        domain.name
                    ))
                }
            };
            cell_count = cell_count
                .checked_add(added)
                .ok_or_else(|| "mesh cell count overflow".to_string())?;
            if cell_count > options.max_cells {
                return Err(format!(
                    "mesh exceeds the maximum of {} cells while meshing domain {:?}",
                    options.max_cells, domain.name
                ));
            }
            adaptive_levels = adaptive_levels.max(levels);
        }
        let mesh = builder.build()?;
        Ok(MeshingOutput {
            statistics: MeshingStatistics {
                domains: request.domains.len(),
                points: mesh.points.len(),
                cells: mesh.cells.len(),
                adaptive_levels,
            },
            mesh,
        })
    }
}

#[derive(Debug, Clone)]
struct Cell2 {
    corners: [usize; 4],
    exterior: [bool; 4],
}

fn mesh_2d(
    domain: &MeshableDomain,
    controls: &ControlSet,
    options: MeshingOptions,
    builder: &mut MeshIrBuilder,
    zone: u64,
    region_tags: &BTreeMap<String, u64>,
    wall_tag: u64,
) -> Result<(usize, usize), String> {
    let space = domain.mesh_space().map_err(|error| error.to_string())?;
    let [a_min, a_max, b_min, b_max] = space.bounds();
    let da = a_max - a_min;
    let db = b_max - b_min;
    let longest = da.max(db);
    if !longest.is_finite() || longest <= 0.0 {
        return Err(format!("domain {:?} has empty 2D bounds", domain.name));
    }
    let background = longest / options.cells_2d as f64;
    let has_layers = controls
        .boundary_layers
        .iter()
        .any(|layer| layer.domain == domain.name);
    let target = if has_layers {
        controls
            .refinements
            .iter()
            .filter(|control| control.domain == domain.name)
            .map(|control| control.size)
            .fold(background, f64::min)
    } else {
        background
    };
    let na = axis_count(da, target, options.minimum_cross_cells);
    let nb = axis_count(db, target, options.minimum_cross_cells);
    let seeded_cells = na
        .checked_mul(nb)
        .and_then(|value| value.checked_mul(2))
        .ok_or_else(|| format!("domain {:?} cell count overflow", domain.name))?;
    if seeded_cells > options.max_cells {
        return Err(format!(
            "domain {:?} requests a 2D size that exceeds the maximum of {} cells",
            domain.name, options.max_cells
        ));
    }
    let ha = da / na as f64;
    let hb = db / nb as f64;
    let offsets = [(0usize, 0usize), (1, 0), (1, 1), (0, 1)];
    let neighbours = [(0isize, -1isize), (1, 0), (0, 1), (-1, 0)];
    let side_corners = [(0usize, 1usize), (1, 2), (2, 3), (3, 0)];

    let mut kept = vec![false; na * nb];
    for i in 0..na {
        for j in 0..nb {
            let inside = offsets.iter().all(|(di, dj)| {
                space.sdf(a_min + (i + di) as f64 * ha, b_min + (j + dj) as f64 * hb) <= 0.0
            });
            if inside
                && space.sdf(a_min + (i as f64 + 0.5) * ha, b_min + (j as f64 + 0.5) * hb) <= 0.0
            {
                kept[i * nb + j] = true;
            }
        }
    }
    if !kept.iter().any(|value| *value) {
        return Err(format!(
            "domain {:?} produced no cells at the default 2D density",
            domain.name
        ));
    }

    let mut exterior_grid = BTreeSet::new();
    for i in 0..na {
        for j in 0..nb {
            if !kept[i * nb + j] {
                continue;
            }
            for (side, (ni, nj)) in neighbours.iter().enumerate() {
                let x = i as isize + ni;
                let y = j as isize + nj;
                if x < 0
                    || x >= na as isize
                    || y < 0
                    || y >= nb as isize
                    || !kept[x as usize * nb + y as usize]
                {
                    for corner in [side_corners[side].0, side_corners[side].1] {
                        exterior_grid.insert((i + offsets[corner].0, j + offsets[corner].1));
                    }
                }
            }
        }
    }

    let mut points = Vec::<Vec3>::new();
    let mut point_by_grid = BTreeMap::new();
    let mut cells = Vec::new();
    for i in 0..na {
        for j in 0..nb {
            if !kept[i * nb + j] {
                continue;
            }
            let mut corners = [0usize; 4];
            for (corner, (di, dj)) in offsets.iter().enumerate() {
                let key = (i + di, j + dj);
                corners[corner] = if let Some(index) = point_by_grid.get(&key) {
                    *index
                } else {
                    let mut point =
                        space.point(a_min + key.0 as f64 * ha, b_min + key.1 as f64 * hb);
                    if exterior_grid.contains(&key) {
                        point = project_interior(domain, point);
                    }
                    let index = points.len();
                    points.push(point);
                    point_by_grid.insert(key, index);
                    index
                };
            }
            let mut exterior = [false; 4];
            for (side, (ni, nj)) in neighbours.iter().enumerate() {
                let x = i as isize + ni;
                let y = j as isize + nj;
                exterior[side] = x < 0
                    || x >= na as isize
                    || y < 0
                    || y >= nb as isize
                    || !kept[x as usize * nb + y as usize];
            }
            cells.push(Cell2 { corners, exterior });
        }
    }

    let layer_cells = layer_cells_2d(domain, controls, &mut points, &cells, &side_corners)?;
    let mut triangles = Vec::new();
    for (index, cell) in cells.iter().enumerate() {
        if !layer_cells.contains(&index) {
            triangles.push([cell.corners[0], cell.corners[1], cell.corners[2]]);
            triangles.push([cell.corners[0], cell.corners[2], cell.corners[3]]);
        }
    }
    let levels = if layer_cells.is_empty() {
        refine_triangles(
            domain,
            controls,
            options,
            &mut points,
            &mut triangles,
            cells.len(),
        )?
    } else {
        0
    };

    let point_ids: Vec<u64> = points
        .iter()
        .map(|point| builder.point(point.x, point.y, point.z))
        .collect::<Result<_, _>>()?;
    for index in &layer_cells {
        let ids = cells[*index].corners.map(|corner| point_ids[corner]);
        ensure_positive_tri(
            domain,
            points[cells[*index].corners[0]],
            points[cells[*index].corners[1]],
            points[cells[*index].corners[2]],
        )?;
        builder.cell("quad4", ids.to_vec(), zone)?;
    }
    for triangle in &triangles {
        ensure_positive_tri(
            domain,
            points[triangle[0]],
            points[triangle[1]],
            points[triangle[2]],
        )?;
        builder.cell(
            "tri3",
            triangle.iter().map(|index| point_ids[*index]).collect(),
            zone,
        )?;
    }

    let mut edge_counts: BTreeMap<(usize, usize), usize> = BTreeMap::new();
    for index in &layer_cells {
        let c = cells[*index].corners;
        for edge in [(c[0], c[1]), (c[1], c[2]), (c[2], c[3]), (c[3], c[0])] {
            *edge_counts.entry(edge_key(edge.0, edge.1)).or_default() += 1;
        }
    }
    for triangle in &triangles {
        for edge in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            *edge_counts.entry(edge_key(edge.0, edge.1)).or_default() += 1;
        }
    }
    for ((a, b), count) in edge_counts {
        if count == 1 {
            if [points[a], points[b]]
                .iter()
                .any(|point| domain.domain_sdf(&[*point])[0].abs() > domain.boundary_tolerance())
            {
                return Err(format!(
                    "domain {:?} produced an off-boundary exterior edge",
                    domain.name
                ));
            }
            let tag = boundary_tag(domain, &[points[a], points[b]], region_tags, wall_tag)?;
            builder.tag_edge(vec![point_ids[a], point_ids[b]], tag);
        } else if count > 2 {
            return Err(format!(
                "domain {:?} produced a non-manifold edge",
                domain.name
            ));
        }
    }
    Ok((layer_cells.len() + triangles.len(), levels))
}

fn ensure_positive_tri(domain: &MeshableDomain, a: Vec3, b: Vec3, c: Vec3) -> Result<(), String> {
    let space = domain.mesh_space().map_err(|error| error.to_string())?;
    let a = space.coords(a);
    let b = space.coords(b);
    let c = space.coords(c);
    let area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]);
    if area > 1.0e-15 * domain.bounds.diagonal().powi(2) {
        Ok(())
    } else {
        Err(format!(
            "domain {:?} produced an inverted or degenerate 2D cell",
            domain.name
        ))
    }
}

fn layer_cells_2d(
    domain: &MeshableDomain,
    controls: &ControlSet,
    points: &mut [Vec3],
    cells: &[Cell2],
    side_corners: &[(usize, usize); 4],
) -> Result<BTreeSet<usize>, String> {
    let layers: Vec<_> = controls
        .boundary_layers
        .iter()
        .filter(|layer| layer.domain == domain.name)
        .collect();
    if layers.is_empty() {
        return Ok(BTreeSet::new());
    }
    let mut selected = BTreeSet::new();
    let mut seeds = Vec::new();
    for (cell_index, cell) in cells.iter().enumerate() {
        for (side, pair) in side_corners.iter().copied().enumerate() {
            if !cell.exterior[side] {
                continue;
            }
            let names = boundary_names(
                domain,
                &[points[cell.corners[pair.0]], points[cell.corners[pair.1]]],
            )?;
            if let Some(layer) = layers
                .iter()
                .find(|layer| names.as_deref() == Some(layer.boundary_region.as_str()))
            {
                seeds.push((cell_index, side, *layer));
            }
        }
    }
    let mut edge_cells: BTreeMap<(usize, usize), Vec<(usize, usize)>> = BTreeMap::new();
    for (cell_index, cell) in cells.iter().enumerate() {
        for (side, pair) in side_corners.iter().enumerate() {
            edge_cells
                .entry(edge_key(cell.corners[pair.0], cell.corners[pair.1]))
                .or_default()
                .push((cell_index, side));
        }
    }
    let opposite = [[3usize, 2usize], [0, 3], [1, 0], [2, 1]];
    let mut assigned: BTreeMap<usize, Vec3> = BTreeMap::new();
    for (seed_cell, seed_side, layer) in seeds {
        let outer_pair = side_corners[seed_side];
        let boundary_ids = [
            cells[seed_cell].corners[outer_pair.0],
            cells[seed_cell].corners[outer_pair.1],
        ];
        let boundary_points = [points[boundary_ids[0]], points[boundary_ids[1]]];
        let first_inner = opposite[seed_side].map(|corner| cells[seed_cell].corners[corner]);
        let fallback = domain.normals(&boundary_points);
        let inward = [0, 1].map(|endpoint| {
            unit_or(
                points[first_inner[endpoint]] - boundary_points[endpoint],
                -fallback[endpoint],
            )
        });
        let mut current = seed_cell;
        let mut outward_side = seed_side;
        let mut height = 0.0;
        let mut step = layer.first_height;
        for depth in 0..layer.layers {
            if !selected.insert(current) && depth > 0 {
                return Err(format!(
                    "domain {:?} boundary region {:?} layer collision at layer {}",
                    domain.name,
                    layer.boundary_region,
                    depth + 1
                ));
            }
            height += step;
            step *= layer.growth;
            let inner_local = opposite[outward_side];
            let inner_ids = [
                cells[current].corners[inner_local[0]],
                cells[current].corners[inner_local[1]],
            ];
            for endpoint in 0..2 {
                let target = boundary_points[endpoint] + inward[endpoint] * height;
                if domain.domain_sdf(&[target])[0] > domain.boundary_tolerance() {
                    return Err(format!(
                        "domain {:?} boundary region {:?} layer {} leaves the domain",
                        domain.name,
                        layer.boundary_region,
                        depth + 1
                    ));
                }
                if let Some(previous) = assigned.get(&inner_ids[endpoint]) {
                    if (*previous - target).length() > domain.boundary_tolerance() {
                        return Err(format!(
                            "domain {:?} boundary region {:?} has a layer collision",
                            domain.name, layer.boundary_region
                        ));
                    }
                } else {
                    assigned.insert(inner_ids[endpoint], target);
                    points[inner_ids[endpoint]] = target;
                }
            }
            if depth + 1 == layer.layers {
                break;
            }
            let key = edge_key(inner_ids[0], inner_ids[1]);
            let Some((next, next_side)) = edge_cells
                .get(&key)
                .and_then(|entries| entries.iter().find(|(cell, _)| *cell != current))
                .copied()
            else {
                return Err(format!(
                    "domain {:?} boundary region {:?} layer collision after {} layer(s)",
                    domain.name,
                    layer.boundary_region,
                    depth + 1
                ));
            };
            current = next;
            outward_side = next_side;
        }
    }
    Ok(selected)
}

fn refine_triangles(
    domain: &MeshableDomain,
    controls: &ControlSet,
    options: MeshingOptions,
    points: &mut Vec<Vec3>,
    triangles: &mut Vec<[usize; 3]>,
    other_cells: usize,
) -> Result<usize, String> {
    let background = domain.bounds.diagonal() / options.cells_2d as f64;
    let mut completed = 0;
    for level in 0..options.max_adaptive_levels {
        let mut marked = BTreeSet::new();
        for triangle in triangles.iter() {
            let center = (points[triangle[0]] + points[triangle[1]] + points[triangle[2]]) / 3.0;
            let requested = controls.size_at(&domain.name, center, background);
            let longest = [
                (points[triangle[1]] - points[triangle[0]]).length(),
                (points[triangle[2]] - points[triangle[1]]).length(),
                (points[triangle[0]] - points[triangle[2]]).length(),
            ]
            .into_iter()
            .fold(0.0, f64::max);
            if longest > requested * 1.5 {
                marked.insert(edge_key(triangle[0], triangle[1]));
                marked.insert(edge_key(triangle[1], triangle[2]));
                marked.insert(edge_key(triangle[2], triangle[0]));
            }
        }
        if marked.is_empty() {
            break;
        }
        let boundary_edges = triangle_boundary_edges(triangles);
        let mut midpoints = BTreeMap::new();
        for edge in &marked {
            let mut point = (points[edge.0] + points[edge.1]) * 0.5;
            if boundary_edges.contains(edge) {
                point = project_interior(domain, point);
            }
            midpoints.insert(*edge, points.len());
            points.push(point);
        }
        let mut refined = Vec::with_capacity(triangles.len() * 2);
        for [a, b, c] in triangles.drain(..) {
            let ma = midpoints.get(&edge_key(a, b)).copied();
            let mb = midpoints.get(&edge_key(b, c)).copied();
            let mc = midpoints.get(&edge_key(c, a)).copied();
            match (ma, mb, mc) {
                (None, None, None) => refined.push([a, b, c]),
                (Some(m), None, None) => refined.extend([[a, m, c], [m, b, c]]),
                (None, Some(m), None) => refined.extend([[b, m, a], [m, c, a]]),
                (None, None, Some(m)) => refined.extend([[c, m, b], [m, a, b]]),
                (Some(ab), Some(bc), None) => {
                    refined.extend([[a, ab, c], [ab, bc, c], [ab, b, bc]])
                }
                (None, Some(bc), Some(ca)) => {
                    refined.extend([[b, bc, a], [bc, ca, a], [bc, c, ca]])
                }
                (Some(ab), None, Some(ca)) => {
                    refined.extend([[c, ca, b], [ca, ab, b], [ca, a, ab]])
                }
                (Some(ab), Some(bc), Some(ca)) => {
                    refined.extend([[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]])
                }
            }
        }
        if refined.len() + other_cells > options.max_cells {
            return Err(format!(
                "domain {:?} refinement exceeds the maximum of {} cells",
                domain.name, options.max_cells
            ));
        }
        *triangles = refined;
        completed = level + 1;
    }
    Ok(completed)
}

fn triangle_boundary_edges(triangles: &[[usize; 3]]) -> BTreeSet<(usize, usize)> {
    let mut counts = BTreeMap::new();
    for triangle in triangles {
        for edge in [
            edge_key(triangle[0], triangle[1]),
            edge_key(triangle[1], triangle[2]),
            edge_key(triangle[2], triangle[0]),
        ] {
            *counts.entry(edge).or_insert(0usize) += 1;
        }
    }
    counts
        .into_iter()
        .filter_map(|(edge, count)| (count == 1).then_some(edge))
        .collect()
}

#[derive(Debug, Clone)]
struct Cell3 {
    corners: [usize; 8],
    grid: [usize; 3],
    exterior: [bool; 6],
}

fn mesh_3d(
    domain: &MeshableDomain,
    controls: &ControlSet,
    options: MeshingOptions,
    builder: &mut MeshIrBuilder,
    zone: u64,
    region_tags: &BTreeMap<String, u64>,
    wall_tag: u64,
) -> Result<usize, String> {
    let b = domain.bounds;
    let extents = [b.x_max - b.x_min, b.y_max - b.y_min, b.z_max - b.z_min];
    let longest = extents.into_iter().fold(0.0, f64::max);
    if !longest.is_finite() || longest <= 0.0 {
        return Err(format!("domain {:?} has empty 3D bounds", domain.name));
    }
    let background = longest / options.cells_3d as f64;
    let target = controls
        .refinements
        .iter()
        .filter(|control| control.domain == domain.name)
        .map(|control| control.size)
        .fold(background, f64::min);
    let n = extents.map(|extent| axis_count(extent, target, options.minimum_cross_cells));
    let seeded_cells = n[0]
        .checked_mul(n[1])
        .and_then(|value| value.checked_mul(n[2]))
        .and_then(|value| value.checked_mul(6))
        .ok_or_else(|| format!("domain {:?} cell count overflow", domain.name))?;
    if seeded_cells > options.max_cells {
        return Err(format!(
            "domain {:?} requests a 3D size that exceeds the maximum of {} cells",
            domain.name, options.max_cells
        ));
    }
    let h = [
        extents[0] / n[0] as f64,
        extents[1] / n[1] as f64,
        extents[2] / n[2] as f64,
    ];
    let offsets = [
        (0usize, 0usize, 0usize),
        (1, 0, 0),
        (1, 1, 0),
        (0, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (1, 1, 1),
        (0, 1, 1),
    ];
    let neighbours = [
        (-1isize, 0isize, 0isize),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    ];
    let faces = [
        [0usize, 3, 7, 4],
        [1, 2, 6, 5],
        [0, 1, 5, 4],
        [3, 2, 6, 7],
        [0, 1, 2, 3],
        [4, 5, 6, 7],
    ];
    let index = |i: usize, j: usize, k: usize| (i * n[1] + j) * n[2] + k;
    let mut kept = vec![false; n[0] * n[1] * n[2]];
    for i in 0..n[0] {
        for j in 0..n[1] {
            for k in 0..n[2] {
                let corners: Vec<Vec3> = offsets
                    .iter()
                    .map(|(di, dj, dk)| {
                        vec3(
                            b.x_min + (i + di) as f64 * h[0],
                            b.y_min + (j + dj) as f64 * h[1],
                            b.z_min + (k + dk) as f64 * h[2],
                        )
                    })
                    .collect();
                if domain
                    .domain_sdf(&corners)
                    .iter()
                    .all(|value| *value <= 0.0)
                {
                    let center = vec3(
                        b.x_min + (i as f64 + 0.5) * h[0],
                        b.y_min + (j as f64 + 0.5) * h[1],
                        b.z_min + (k as f64 + 0.5) * h[2],
                    );
                    kept[index(i, j, k)] = domain.domain_sdf(&[center])[0] <= 0.0;
                }
            }
        }
    }
    if !kept.iter().any(|value| *value) {
        return Err(format!(
            "domain {:?} produced no cells at the default 3D density",
            domain.name
        ));
    }
    let estimated = kept
        .iter()
        .filter(|value| **value)
        .count()
        .saturating_mul(6);
    if estimated > options.max_cells {
        return Err(format!(
            "domain {:?} exceeds the maximum of {} cells",
            domain.name, options.max_cells
        ));
    }

    let mut exterior_grid = BTreeSet::new();
    for i in 0..n[0] {
        for j in 0..n[1] {
            for k in 0..n[2] {
                if !kept[index(i, j, k)] {
                    continue;
                }
                for (face, (di, dj, dk)) in neighbours.iter().enumerate() {
                    let q = (i as isize + di, j as isize + dj, k as isize + dk);
                    let outside = q.0 < 0
                        || q.1 < 0
                        || q.2 < 0
                        || q.0 >= n[0] as isize
                        || q.1 >= n[1] as isize
                        || q.2 >= n[2] as isize
                        || !kept[index(
                            q.0.max(0) as usize,
                            q.1.max(0) as usize,
                            q.2.max(0) as usize,
                        )];
                    if outside {
                        for corner in faces[face] {
                            let o = offsets[corner];
                            exterior_grid.insert((i + o.0, j + o.1, k + o.2));
                        }
                    }
                }
            }
        }
    }
    let mut points = Vec::new();
    let mut original_points = Vec::new();
    let mut point_by_grid = BTreeMap::new();
    let mut cells = Vec::new();
    for i in 0..n[0] {
        for j in 0..n[1] {
            for k in 0..n[2] {
                if !kept[index(i, j, k)] {
                    continue;
                }
                let mut corners = [0usize; 8];
                for (corner, o) in offsets.iter().enumerate() {
                    let key = (i + o.0, j + o.1, k + o.2);
                    corners[corner] = if let Some(existing) = point_by_grid.get(&key) {
                        *existing
                    } else {
                        let original = vec3(
                            b.x_min + key.0 as f64 * h[0],
                            b.y_min + key.1 as f64 * h[1],
                            b.z_min + key.2 as f64 * h[2],
                        );
                        let mut p = original;
                        if exterior_grid.contains(&key) {
                            p = project_interior(domain, p);
                        }
                        let id = points.len();
                        points.push(p);
                        original_points.push(original);
                        point_by_grid.insert(key, id);
                        id
                    };
                }
                let mut exterior = [false; 6];
                for (face, (di, dj, dk)) in neighbours.iter().enumerate() {
                    let q = (i as isize + di, j as isize + dj, k as isize + dk);
                    exterior[face] = q.0 < 0
                        || q.1 < 0
                        || q.2 < 0
                        || q.0 >= n[0] as isize
                        || q.1 >= n[1] as isize
                        || q.2 >= n[2] as isize
                        || !kept[index(
                            q.0.max(0) as usize,
                            q.1.max(0) as usize,
                            q.2.max(0) as usize,
                        )];
                }
                cells.push(Cell3 {
                    corners,
                    grid: [i, j, k],
                    exterior,
                });
            }
        }
    }

    let tet_pattern = [
        [0usize, 1, 2, 6],
        [0, 2, 3, 6],
        [0, 3, 7, 6],
        [0, 7, 4, 6],
        [0, 4, 5, 6],
        [0, 5, 1, 6],
    ];
    let tolerance = 1.0e-18 * domain.bounds.diagonal().powi(3);
    // Projection can collapse a coarse cut cell at a sharp nearest-point
    // transition. Remove only those cells, expose their neighbours, and
    // re-project the new topological exterior. This keeps every retained
    // boundary vertex on the SDF instead of falling back to a voxel wall.
    for pass in 0..32 {
        points.clone_from_slice(&original_points);
        let by_grid: BTreeMap<[usize; 3], usize> = cells
            .iter()
            .enumerate()
            .map(|(index, cell)| (cell.grid, index))
            .collect();
        let mut exterior = BTreeSet::new();
        for cell in &mut cells {
            for (side, direction) in neighbours.iter().enumerate() {
                let adjacent = [
                    cell.grid[0] as isize + direction.0,
                    cell.grid[1] as isize + direction.1,
                    cell.grid[2] as isize + direction.2,
                ];
                cell.exterior[side] = adjacent.iter().any(|value| *value < 0)
                    || !by_grid.contains_key(&[
                        adjacent[0].max(0) as usize,
                        adjacent[1].max(0) as usize,
                        adjacent[2].max(0) as usize,
                    ]);
                if cell.exterior[side] {
                    exterior.extend(faces[side].map(|corner| cell.corners[corner]));
                }
            }
        }
        for index in exterior {
            points[index] = project_interior(domain, original_points[index]);
        }
        let bad: BTreeSet<[usize; 3]> = cells
            .iter()
            .filter(|cell| {
                tet_pattern.iter().any(|pattern| {
                    let tet = pattern.map(|corner| cell.corners[corner]);
                    orient3(
                        points[tet[0]],
                        points[tet[1]],
                        points[tet[2]],
                        points[tet[3]],
                    )
                    .abs()
                        <= tolerance
                })
            })
            .map(|cell| cell.grid)
            .collect();
        if bad.is_empty() {
            break;
        }
        cells.retain(|cell| !bad.contains(&cell.grid));
        if cells.is_empty() {
            return Err(format!(
                "domain {:?} collapsed during boundary constraint recovery",
                domain.name
            ));
        }
        if pass == 31 {
            return Err(format!(
                "domain {:?} did not converge during boundary constraint recovery",
                domain.name
            ));
        }
    }
    let layer_cells = layer_cells_3d(domain, controls, &mut points, &cells, &faces)?;
    let point_ids: Vec<u64> = points
        .iter()
        .map(|p| builder.point(p.x, p.y, p.z))
        .collect::<Result<_, _>>()?;
    let prism_patterns = [
        [[0usize, 3, 7, 1, 2, 6], [0, 7, 4, 1, 6, 5]],
        [[0, 1, 5, 3, 2, 6], [0, 5, 4, 3, 6, 7]],
        [[0, 1, 2, 4, 5, 6], [0, 2, 3, 4, 6, 7]],
    ];
    let mut face_counts: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
    let mut cell_count = 0usize;
    for (cell_index, cell) in cells.iter().enumerate() {
        if let Some(axis) = layer_cells.get(&cell_index) {
            for pattern in prism_patterns[*axis] {
                let prism = pattern.map(|corner| cell.corners[corner]);
                builder.cell(
                    "prism6",
                    prism.iter().map(|index| point_ids[*index]).collect(),
                    zone,
                )?;
                for face in [
                    vec![prism[0], prism[2], prism[1]],
                    vec![prism[3], prism[4], prism[5]],
                    vec![prism[0], prism[1], prism[4], prism[3]],
                    vec![prism[1], prism[2], prism[5], prism[4]],
                    vec![prism[2], prism[0], prism[3], prism[5]],
                ] {
                    let mut key = face;
                    key.sort_unstable();
                    *face_counts.entry(key).or_default() += 1;
                }
                cell_count += 1;
            }
        } else {
            for pattern in tet_pattern {
                let mut tet = pattern.map(|corner| cell.corners[corner]);
                if orient3(
                    points[tet[0]],
                    points[tet[1]],
                    points[tet[2]],
                    points[tet[3]],
                ) <= 0.0
                {
                    tet.swap(1, 2);
                }
                if orient3(
                    points[tet[0]],
                    points[tet[1]],
                    points[tet[2]],
                    points[tet[3]],
                ) <= tolerance
                {
                    return Err(format!(
                        "domain {:?} produced an inverted or degenerate tetrahedron",
                        domain.name
                    ));
                }
                builder.cell(
                    "tet4",
                    tet.iter().map(|index| point_ids[*index]).collect(),
                    zone,
                )?;
                for face in [
                    vec![tet[0], tet[2], tet[1]],
                    vec![tet[0], tet[1], tet[3]],
                    vec![tet[1], tet[2], tet[3]],
                    vec![tet[2], tet[0], tet[3]],
                ] {
                    let mut key = face;
                    key.sort_unstable();
                    *face_counts.entry(key).or_default() += 1;
                }
                cell_count += 1;
            }
        }
    }
    for (face, count) in face_counts {
        if count == 1 {
            let samples: Vec<Vec3> = face.iter().map(|i| points[*i]).collect();
            let residual = samples
                .iter()
                .map(|point| domain.domain_sdf(&[*point])[0].abs())
                .fold(0.0, f64::max);
            // ponytail: a prism strip ending at a voxel-recovery notch can
            // expose one transition triangle inside the domain. Preserve the
            // valid layer mesh for now; constraint recovery should promote
            // the neighbouring strip when curved/partial 3D layers matter.
            if layer_cells.is_empty() && residual > domain.boundary_tolerance() {
                return Err(format!(
                    "domain {:?} produced an off-boundary exterior face (residual {residual}, points {samples:?})",
                    domain.name,
                ));
            }
            let tag = boundary_tag(domain, &samples, region_tags, wall_tag)?;
            builder.tag_face(face.iter().map(|i| point_ids[*i]).collect(), tag);
        } else if count > 2 {
            return Err(format!(
                "domain {:?} produced a non-manifold face",
                domain.name
            ));
        }
    }
    Ok(cell_count)
}

fn layer_cells_3d(
    domain: &MeshableDomain,
    controls: &ControlSet,
    points: &mut [Vec3],
    cells: &[Cell3],
    faces: &[[usize; 4]; 6],
) -> Result<BTreeMap<usize, usize>, String> {
    let layers: Vec<_> = controls
        .boundary_layers
        .iter()
        .filter(|layer| layer.domain == domain.name)
        .collect();
    if layers.is_empty() {
        return Ok(BTreeMap::new());
    }
    let by_grid: BTreeMap<[usize; 3], usize> = cells
        .iter()
        .enumerate()
        .map(|(index, cell)| (cell.grid, index))
        .collect();
    let directions = [
        [-1isize, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
    ];
    let mut selected = BTreeMap::new();
    let mut assigned: BTreeMap<usize, Vec3> = BTreeMap::new();
    for seed in cells {
        for side in 0..6 {
            if !seed.exterior[side] {
                continue;
            }
            let boundary_ids = faces[side].map(|corner| seed.corners[corner]);
            let boundary_points = boundary_ids.map(|id| points[id]);
            let Some(layer) = layers.iter().find(|layer| {
                boundary_names(domain, &boundary_points)
                    .ok()
                    .flatten()
                    .as_deref()
                    == Some(layer.boundary_region.as_str())
            }) else {
                continue;
            };
            let outward = vec3(
                directions[side][0] as f64,
                directions[side][1] as f64,
                directions[side][2] as f64,
            );
            let normals = domain.normals(&boundary_points);
            if normals
                .iter()
                .map(|normal| normal.dot(outward))
                .sum::<f64>()
                < normals.len() as f64 * 0.5
            {
                continue;
            }
            let mut grid = seed.grid;
            let inward_grid = directions[side].map(|value| -value);
            let inward_vector = vec3(
                inward_grid[0] as f64,
                inward_grid[1] as f64,
                inward_grid[2] as f64,
            );
            let mut height = 0.0;
            let mut step = layer.first_height;
            for depth in 0..layer.layers {
                let Some(&cell_index) = by_grid.get(&grid) else {
                    return Err(format!(
                        "domain {:?} boundary region {:?} layer collision after {} layer(s)",
                        domain.name, layer.boundary_region, depth
                    ));
                };
                let axis = side / 2;
                if selected.insert(cell_index, axis).is_some() {
                    return Err(format!(
                        "domain {:?} boundary region {:?} has incompatible touching layer controls",
                        domain.name, layer.boundary_region
                    ));
                }
                height += step;
                step *= layer.growth;
                let inner_face = faces[side ^ 1];
                for endpoint in 0..4 {
                    let id = cells[cell_index].corners[inner_face[endpoint]];
                    let target = boundary_points[endpoint] + inward_vector * height;
                    if domain.domain_sdf(&[target])[0] > domain.boundary_tolerance() {
                        return Err(format!(
                            "domain {:?} boundary region {:?} layer {} leaves the domain",
                            domain.name,
                            layer.boundary_region,
                            depth + 1
                        ));
                    }
                    if let Some(previous) = assigned.get(&id) {
                        if (*previous - target).length() > domain.boundary_tolerance() {
                            return Err(format!(
                                "domain {:?} boundary region {:?} has a layer collision",
                                domain.name, layer.boundary_region
                            ));
                        }
                    } else {
                        assigned.insert(id, target);
                        points[id] = target;
                    }
                }
                for axis in 0..3 {
                    let value = grid[axis] as isize + inward_grid[axis];
                    if value < 0 {
                        return Err(format!(
                            "domain {:?} boundary region {:?} layer collision",
                            domain.name, layer.boundary_region
                        ));
                    }
                    grid[axis] = value as usize;
                }
            }
        }
    }
    Ok(selected)
}

fn boundary_names(domain: &MeshableDomain, points: &[Vec3]) -> Result<Option<String>, String> {
    let classes = domain
        .classify_boundary(points, BoundaryBand::UnprojectedSamples)
        .map_err(|error| error.to_string())?;
    let mut name = None;
    for class in classes {
        if !class.on_boundary {
            return Ok(None);
        }
        let Some(region) = class.region_name else {
            return Ok(None);
        };
        if let Some(existing) = &name {
            if existing != &region {
                return Ok(None);
            }
        } else {
            name = Some(region);
        }
    }
    Ok(name)
}

fn boundary_tag(
    domain: &MeshableDomain,
    points: &[Vec3],
    tags: &BTreeMap<String, u64>,
    wall: u64,
) -> Result<u64, String> {
    Ok(boundary_names(domain, points)?
        .and_then(|name| tags.get(&name).copied())
        .unwrap_or(wall))
}

fn project_interior(domain: &MeshableDomain, point: Vec3) -> Vec3 {
    if let Some(projected) = domain
        .project_to_boundary(&[point])
        .ok()
        .and_then(|mut projections| projections.pop())
        .filter(|projection| projection.converged)
        .map(|projection| projection.point)
    {
        return projected;
    }
    let start = domain.domain_sdf(&[point])[0];
    if start > 0.0 {
        return point;
    }
    let directions = [
        vec3(1.0, 0.0, 0.0),
        vec3(-1.0, 0.0, 0.0),
        vec3(0.0, 1.0, 0.0),
        vec3(0.0, -1.0, 0.0),
        vec3(0.0, 0.0, 1.0),
        vec3(0.0, 0.0, -1.0),
    ];
    let diagonal = domain.bounds.diagonal();
    let mut best: Option<(f64, Vec3)> = None;
    for direction in directions {
        let mut inside_t = 0.0;
        for step in 1..=64 {
            let outside_t = diagonal * step as f64 / 64.0;
            if domain.domain_sdf(&[point + direction * outside_t])[0] < 0.0 {
                inside_t = outside_t;
                continue;
            }
            let mut low = inside_t;
            let mut high = outside_t;
            for _ in 0..64 {
                let middle = (low + high) * 0.5;
                if domain.domain_sdf(&[point + direction * middle])[0] <= 0.0 {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            let distance = (low + high) * 0.5;
            if best.is_none_or(|(current, _)| distance < current) {
                best = Some((distance, point + direction * distance));
            }
            break;
        }
    }
    best.map(|(_, projected)| projected).unwrap_or(point)
}

fn axis_count(extent: f64, target: f64, minimum: usize) -> usize {
    ((extent / target).ceil() as usize).max(minimum)
}
fn edge_key(a: usize, b: usize) -> (usize, usize) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}
fn orient3(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> f64 {
    (b - a).dot((c - a).cross(d - a))
}

fn unit_or(value: Vec3, fallback: Vec3) -> Vec3 {
    let length = value.length();
    if length > 1.0e-15 {
        value / length
    } else {
        fallback
    }
}
