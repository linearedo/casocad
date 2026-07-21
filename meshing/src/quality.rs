//! Runtime mesh-quality analysis. Scores are normalized to `0.0 = bad` and
//! `1.0 = ideal`; unsupported topology is represented by `None`.

use std::collections::BTreeMap;

use crate::{MeshCell, MeshFace, MeshIr};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum QualityMetric {
    ScaledJacobian,
    Skewness,
    AspectRatio,
    Compactness,
    Orthogonality,
}

impl QualityMetric {
    pub const ALL: [Self; 5] = [
        Self::ScaledJacobian,
        Self::Skewness,
        Self::AspectRatio,
        Self::Compactness,
        Self::Orthogonality,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::ScaledJacobian => "Scaled Jacobian",
            Self::Skewness => "Skewness",
            Self::AspectRatio => "Aspect Ratio",
            Self::Compactness => "Compactness",
            Self::Orthogonality => "Orthogonality",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CellQuality {
    pub cell_id: u64,
    pub score: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct QualityReport {
    pub metric: QualityMetric,
    pub top_dimension: Option<u8>,
    pub cells: Vec<CellQuality>,
    pub unsupported_count: usize,
}

impl QualityReport {
    pub fn score(&self, cell_id: u64) -> Option<f64> {
        self.cells
            .iter()
            .find(|cell| cell.cell_id == cell_id)
            .and_then(|cell| cell.score)
    }
}

/// Analyze only cells in the mesh's highest dimension. The caller owns
/// revision-based caching; this function is deterministic and has no state.
pub fn analyze(mesh: &MeshIr, metric: QualityMetric) -> QualityReport {
    let top_dimension = mesh
        .cells
        .iter()
        .filter_map(|cell| dimension(&cell.type_name))
        .max();
    let points: BTreeMap<u64, V> = mesh
        .points
        .iter()
        .map(|point| (point.id, V(point.position)))
        .collect();
    let cells_by_id: BTreeMap<u64, &MeshCell> =
        mesh.cells.iter().map(|cell| (cell.id, cell)).collect();
    let faces_by_id: BTreeMap<u64, &MeshFace> =
        mesh.faces.iter().map(|face| (face.id, face)).collect();

    let cells: Vec<CellQuality> = mesh
        .cells
        .iter()
        .filter(|cell| dimension(&cell.type_name) == top_dimension)
        .map(|cell| {
            let corners = corner_ids(cell)
                .and_then(|ids| ids.iter().map(|id| points.get(id).copied()).collect());
            let score = corners.and_then(|corners: Vec<V>| match metric {
                QualityMetric::ScaledJacobian => scaled_jacobian(cell, &corners),
                QualityMetric::Skewness => skewness(cell, &corners, &faces_by_id, &points),
                QualityMetric::AspectRatio => aspect_ratio(cell, &corners),
                QualityMetric::Compactness => compactness(cell, &corners, &faces_by_id, &points),
                QualityMetric::Orthogonality => {
                    orthogonality(cell, &corners, mesh, &cells_by_id, &faces_by_id, &points)
                }
            });
            CellQuality {
                cell_id: cell.id,
                score: score.map(unit),
            }
        })
        .collect();
    let unsupported_count = cells.iter().filter(|cell| cell.score.is_none()).count();
    QualityReport {
        metric,
        top_dimension,
        cells,
        unsupported_count,
    }
}

fn dimension(type_name: &str) -> Option<u8> {
    Some(match type_name {
        "point1" => 0,
        "edge2" | "edge3" => 1,
        "tri3" | "tri6" | "quad4" | "quad8" | "quad9" | "polygon" => 2,
        "tet4" | "tet10" | "hex8" | "hex20" | "hex27" | "prism6" | "prism15" | "pyramid5"
        | "pyramid13" | "polyhedron" => 3,
        _ => return None,
    })
}

fn corner_ids(cell: &MeshCell) -> Option<&[u64]> {
    let count = match cell.type_name.as_str() {
        "tri3" | "tri6" => 3,
        "quad4" | "quad8" | "quad9" => 4,
        "tet4" | "tet10" => 4,
        "hex8" | "hex20" | "hex27" => 8,
        "prism6" | "prism15" => 6,
        "pyramid5" | "pyramid13" => 5,
        "polygon" => cell.point_ids.len(),
        "polyhedron" => cell.point_ids.len(),
        _ => return None,
    };
    cell.point_ids.get(..count)
}

fn scaled_jacobian(cell: &MeshCell, p: &[V]) -> Option<f64> {
    match cell.type_name.as_str() {
        "tri3" | "tri6" => scaled_jacobian_2d(p, 3, 2.0 / 3.0_f64.sqrt()),
        "quad4" | "quad8" | "quad9" => scaled_jacobian_2d(p, 4, 1.0),
        "tet4" | "tet10" => corner_jacobians(
            p,
            &[(0, 1, 2, 3), (1, 0, 3, 2), (2, 0, 1, 3), (3, 0, 2, 1)],
            2.0_f64.sqrt(),
        ),
        "hex8" | "hex20" | "hex27" => corner_jacobians(
            p,
            &[
                (0, 1, 3, 4),
                (1, 2, 0, 5),
                (2, 3, 1, 6),
                (3, 0, 2, 7),
                (4, 7, 5, 0),
                (5, 4, 6, 1),
                (6, 5, 7, 2),
                (7, 6, 4, 3),
            ],
            1.0,
        ),
        "prism6" | "prism15" => corner_jacobians(
            p,
            &[
                (0, 1, 2, 3),
                (1, 2, 0, 4),
                (2, 0, 1, 5),
                (3, 5, 4, 0),
                (4, 3, 5, 1),
                (5, 4, 3, 2),
            ],
            2.0 / 3.0_f64.sqrt(),
        ),
        "pyramid5" | "pyramid13" => {
            let base = corner_jacobians(
                p,
                &[(0, 1, 3, 4), (1, 2, 0, 4), (2, 3, 1, 4), (3, 0, 2, 4)],
                2.0_f64.sqrt(),
            )?;
            let apex = [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]
                .into_iter()
                .map(|(a, b, c)| normalized_det(p[a] - p[4], p[b] - p[4], p[c] - p[4]).abs())
                .fold(f64::INFINITY, f64::min)
                * 2.0_f64.sqrt();
            Some(base.min(apex))
        }
        _ => None,
    }
}

fn scaled_jacobian_2d(p: &[V], count: usize, normalization: f64) -> Option<f64> {
    if p.len() != count {
        return None;
    }
    let normal = polygon_normal(p);
    let normal_length = normal.length();
    if normal_length <= EPS {
        return Some(0.0);
    }
    let normal = normal / normal_length;
    let mut result = f64::INFINITY;
    for i in 0..count {
        let incoming = p[(i + count - 1) % count] - p[i];
        let outgoing = p[(i + 1) % count] - p[i];
        let denominator = incoming.length() * outgoing.length();
        if denominator <= EPS {
            return Some(0.0);
        }
        // Incoming is reversed relative to the usual previous->current edge.
        result = result.min(-incoming.cross(outgoing).dot(normal) / denominator);
    }
    Some((result * normalization).max(0.0))
}

fn corner_jacobians(p: &[V], corners: &[(usize, usize, usize, usize)], scale: f64) -> Option<f64> {
    if corners.iter().any(|indices| {
        [indices.0, indices.1, indices.2, indices.3]
            .into_iter()
            .any(|i| i >= p.len())
    }) {
        return None;
    }
    let minimum = corners
        .iter()
        .map(|&(o, a, b, c)| normalized_det(p[a] - p[o], p[b] - p[o], p[c] - p[o]))
        .fold(f64::INFINITY, f64::min);
    Some((minimum * scale).max(0.0))
}

fn skewness(
    cell: &MeshCell,
    p: &[V],
    faces: &BTreeMap<u64, &MeshFace>,
    points: &BTreeMap<u64, V>,
) -> Option<f64> {
    match dimension(&cell.type_name)? {
        2 => polygon_skewness(p),
        3 => cell_face_points(cell, faces, points)?
            .iter()
            .map(|face| polygon_skewness(face))
            .collect::<Option<Vec<_>>>()
            .map(|scores| scores.into_iter().fold(1.0, f64::min)),
        _ => None,
    }
}

fn polygon_skewness(p: &[V]) -> Option<f64> {
    if p.len() < 3 {
        return None;
    }
    let ideal = std::f64::consts::PI * (p.len() as f64 - 2.0) / p.len() as f64;
    let normal = polygon_normal(p);
    if normal.length() <= EPS {
        return Some(0.0);
    }
    let mut min_angle = f64::INFINITY;
    let mut max_angle: f64 = 0.0;
    for i in 0..p.len() {
        let a = p[(i + p.len() - 1) % p.len()] - p[i];
        let b = p[(i + 1) % p.len()] - p[i];
        let lengths = a.length() * b.length();
        if lengths <= EPS {
            return Some(0.0);
        }
        let angle = (a.dot(b) / lengths).clamp(-1.0, 1.0).acos();
        min_angle = min_angle.min(angle);
        max_angle = max_angle.max(angle);
    }
    let skew =
        ((max_angle - ideal) / (std::f64::consts::PI - ideal)).max((ideal - min_angle) / ideal);
    Some(1.0 - skew.max(0.0))
}

fn aspect_ratio(cell: &MeshCell, p: &[V]) -> Option<f64> {
    let pairs = edge_pairs(&cell.type_name, p.len())?;
    let lengths: Vec<f64> = pairs.iter().map(|&(a, b)| (p[b] - p[a]).length()).collect();
    let minimum = lengths.iter().copied().fold(f64::INFINITY, f64::min);
    let maximum = lengths.iter().copied().fold(0.0, f64::max);
    Some(if minimum <= EPS || maximum <= EPS {
        0.0
    } else {
        minimum / maximum
    })
}

fn compactness(
    cell: &MeshCell,
    p: &[V],
    faces: &BTreeMap<u64, &MeshFace>,
    points: &BTreeMap<u64, V>,
) -> Option<f64> {
    match dimension(&cell.type_name)? {
        2 => {
            let area = polygon_area(p);
            let perimeter: f64 = (0..p.len())
                .map(|i| (p[(i + 1) % p.len()] - p[i]).length())
                .sum();
            if area <= EPS || perimeter <= EPS {
                return Some(0.0);
            }
            let n = p.len() as f64;
            Some(4.0 * n * (std::f64::consts::PI / n).tan() * area / perimeter.powi(2))
        }
        3 => {
            let face_points = cell_face_points(cell, faces, points)?;
            let center = centroid(p);
            let surface: f64 = face_points.iter().map(|face| polygon_area(face)).sum();
            let volume: f64 = face_points
                .iter()
                .map(|face| face_volume(center, face))
                .sum();
            if volume <= EPS || surface <= EPS {
                return Some(0.0);
            }
            let reference = match cell.type_name.as_str() {
                "tet4" | "tet10" => (2.0_f64.sqrt() / 12.0) / 3.0_f64.sqrt().powf(1.5),
                "hex8" | "hex20" | "hex27" => 1.0 / 6.0_f64.powf(1.5),
                "prism6" | "prism15" => {
                    let v = 3.0_f64.sqrt() / 4.0;
                    let s = 3.0_f64.sqrt() / 2.0 + 3.0;
                    v / s.powf(1.5)
                }
                "pyramid5" | "pyramid13" => {
                    let h = 0.5_f64.sqrt();
                    (h / 3.0) / (1.0 + 4.0 * (h * h + 0.25).sqrt() / 2.0).powf(1.5)
                }
                "polyhedron" => return None,
                _ => return None,
            };
            Some(volume / surface.powf(1.5) / reference)
        }
        _ => None,
    }
}

fn orthogonality(
    cell: &MeshCell,
    p: &[V],
    mesh: &MeshIr,
    cells: &BTreeMap<u64, &MeshCell>,
    faces: &BTreeMap<u64, &MeshFace>,
    points: &BTreeMap<u64, V>,
) -> Option<f64> {
    let center = centroid(p);
    match dimension(&cell.type_name)? {
        2 => {
            let edges: Vec<_> = mesh
                .edges
                .iter()
                .filter(|edge| cell.edge_ids.contains(&edge.id))
                .collect();
            if edges.is_empty() {
                return None;
            }
            let Some(plane) = polygon_normal(p).normalized() else {
                return Some(0.0);
            };
            let mut result: f64 = 1.0;
            for edge in edges {
                let edge_points: Vec<V> = edge
                    .point_ids
                    .iter()
                    .filter_map(|id| points.get(id).copied())
                    .collect();
                if edge_points.len() < 2 {
                    return None;
                }
                let Some(tangent) = (edge_points[1] - edge_points[0]).normalized() else {
                    return Some(0.0);
                };
                let Some(normal) = tangent.cross(plane).normalized() else {
                    return Some(0.0);
                };
                let face_center = centroid(&edge_points);
                let target = adjacent_center(
                    cell.id,
                    edge.owner_cell_id,
                    edge.neighbor_cell_id,
                    cells,
                    points,
                )
                .unwrap_or(face_center)
                    - center;
                let Some(target) = target.normalized() else {
                    return Some(0.0);
                };
                result = result.min(normal.dot(target).abs());
            }
            Some(result)
        }
        3 => {
            let mut result: f64 = 1.0;
            if cell.face_ids.is_empty() {
                return None;
            }
            for face_id in &cell.face_ids {
                let face = *faces.get(face_id)?;
                let face_points: Vec<V> = face
                    .point_ids
                    .iter()
                    .filter_map(|id| points.get(id).copied())
                    .collect();
                if face_points.len() != face.point_ids.len() {
                    return None;
                }
                let Some(normal) = polygon_normal(&face_points).normalized() else {
                    return Some(0.0);
                };
                let face_center = centroid(&face_points);
                let target = adjacent_center(
                    cell.id,
                    face.owner_cell_id,
                    face.neighbor_cell_id,
                    cells,
                    points,
                )
                .unwrap_or(face_center)
                    - center;
                let Some(target) = target.normalized() else {
                    return Some(0.0);
                };
                result = result.min(normal.dot(target).abs());
            }
            Some(result)
        }
        _ => None,
    }
}

fn adjacent_center(
    cell_id: u64,
    owner: Option<u64>,
    neighbor: Option<u64>,
    cells: &BTreeMap<u64, &MeshCell>,
    points: &BTreeMap<u64, V>,
) -> Option<V> {
    let other = if owner == Some(cell_id) {
        neighbor
    } else if neighbor == Some(cell_id) {
        owner
    } else {
        None
    }?;
    let cell = cells.get(&other)?;
    let p: Vec<V> = corner_ids(cell)?
        .iter()
        .filter_map(|id| points.get(id).copied())
        .collect();
    (p.len() == corner_ids(cell)?.len()).then(|| centroid(&p))
}

fn cell_face_points(
    cell: &MeshCell,
    faces: &BTreeMap<u64, &MeshFace>,
    points: &BTreeMap<u64, V>,
) -> Option<Vec<Vec<V>>> {
    if cell.face_ids.is_empty() {
        return None;
    }
    cell.face_ids
        .iter()
        .map(|id| {
            let face = faces.get(id)?;
            let ids = face_corner_ids(&face.type_name, &face.point_ids)?;
            let p: Vec<V> = ids
                .iter()
                .filter_map(|id| points.get(id).copied())
                .collect();
            (p.len() == ids.len()).then_some(p)
        })
        .collect()
}

fn face_corner_ids<'a>(type_name: &str, ids: &'a [u64]) -> Option<&'a [u64]> {
    let count = match type_name {
        "tri3" | "tri6" => 3,
        "quad4" | "quad8" | "quad9" => 4,
        "polygon" => ids.len(),
        _ => return None,
    };
    ids.get(..count)
}

fn edge_pairs(type_name: &str, count: usize) -> Option<Vec<(usize, usize)>> {
    let pairs = match type_name {
        "tri3" | "tri6" | "quad4" | "quad8" | "quad9" | "polygon" => {
            (0..count).map(|i| (i, (i + 1) % count)).collect()
        }
        "tet4" | "tet10" => vec![(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)],
        "hex8" | "hex20" | "hex27" => vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ],
        "prism6" | "prism15" => vec![
            (0, 1),
            (1, 2),
            (2, 0),
            (3, 4),
            (4, 5),
            (5, 3),
            (0, 3),
            (1, 4),
            (2, 5),
        ],
        "pyramid5" | "pyramid13" => vec![
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 4),
            (1, 4),
            (2, 4),
            (3, 4),
        ],
        _ => return None,
    };
    Some(pairs)
}

fn polygon_normal(p: &[V]) -> V {
    let mut normal = V::ZERO;
    for i in 0..p.len() {
        normal = normal + p[i].cross(p[(i + 1) % p.len()]);
    }
    normal
}

fn polygon_area(p: &[V]) -> f64 {
    if p.len() < 3 {
        return 0.0;
    }
    let origin = p[0];
    (1..p.len() - 1)
        .map(|i| (p[i] - origin).cross(p[i + 1] - origin).length() * 0.5)
        .sum()
}

fn face_volume(center: V, face: &[V]) -> f64 {
    if face.len() < 3 {
        return 0.0;
    }
    (1..face.len() - 1)
        .map(|i| {
            ((face[0] - center).dot((face[i] - center).cross(face[i + 1] - center))).abs() / 6.0
        })
        .sum()
}

fn centroid(p: &[V]) -> V {
    p.iter().copied().fold(V::ZERO, |sum, value| sum + value) / p.len() as f64
}

fn normalized_det(a: V, b: V, c: V) -> f64 {
    let denominator = a.length() * b.length() * c.length();
    if denominator <= EPS {
        0.0
    } else {
        a.dot(b.cross(c)) / denominator
    }
}

fn unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

const EPS: f64 = 1.0e-14;

#[derive(Debug, Clone, Copy)]
struct V([f64; 3]);

impl V {
    const ZERO: Self = Self([0.0; 3]);
    fn dot(self, other: Self) -> f64 {
        self.0[0] * other.0[0] + self.0[1] * other.0[1] + self.0[2] * other.0[2]
    }
    fn cross(self, other: Self) -> Self {
        Self([
            self.0[1] * other.0[2] - self.0[2] * other.0[1],
            self.0[2] * other.0[0] - self.0[0] * other.0[2],
            self.0[0] * other.0[1] - self.0[1] * other.0[0],
        ])
    }
    fn length(self) -> f64 {
        self.dot(self).sqrt()
    }
    fn normalized(self) -> Option<Self> {
        let length = self.length();
        (length > EPS).then(|| self / length)
    }
}

impl std::ops::Add for V {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self([
            self.0[0] + rhs.0[0],
            self.0[1] + rhs.0[1],
            self.0[2] + rhs.0[2],
        ])
    }
}
impl std::ops::Sub for V {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self([
            self.0[0] - rhs.0[0],
            self.0[1] - rhs.0[1],
            self.0[2] - rhs.0[2],
        ])
    }
}
impl std::ops::Div<f64> for V {
    type Output = Self;
    fn div(self, rhs: f64) -> Self {
        Self([self.0[0] / rhs, self.0[1] / rhs, self.0[2] / rhs])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MeshIrBuilder;

    fn one_cell(type_name: &str, coordinates: &[[f64; 3]]) -> MeshIr {
        let mut builder = MeshIrBuilder::new();
        let zone = builder.zone("zone", "fluid");
        let ids = coordinates
            .iter()
            .map(|p| builder.point(p[0], p[1], p[2]).unwrap())
            .collect();
        builder.cell(type_name, ids, zone).unwrap();
        builder.build().unwrap()
    }

    fn score(mesh: &MeshIr, metric: QualityMetric) -> Option<f64> {
        analyze(mesh, metric).cells[0].score
    }

    #[test]
    fn ideal_linear_families_are_normalized() {
        let h = 3.0_f64.sqrt() / 2.0;
        let tet_h = (2.0_f64 / 3.0).sqrt();
        let meshes = [
            one_cell("tri3", &[[0., 0., 0.], [1., 0., 0.], [0.5, h, 0.]]),
            one_cell(
                "quad4",
                &[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
            ),
            one_cell(
                "tet4",
                &[
                    [0., 0., 0.],
                    [1., 0., 0.],
                    [0.5, h, 0.],
                    [0.5, h / 3., tet_h],
                ],
            ),
            one_cell(
                "hex8",
                &[
                    [0., 0., 0.],
                    [1., 0., 0.],
                    [1., 1., 0.],
                    [0., 1., 0.],
                    [0., 0., 1.],
                    [1., 0., 1.],
                    [1., 1., 1.],
                    [0., 1., 1.],
                ],
            ),
            one_cell(
                "prism6",
                &[
                    [0., 0., 0.],
                    [1., 0., 0.],
                    [0.5, h, 0.],
                    [0., 0., 1.],
                    [1., 0., 1.],
                    [0.5, h, 1.],
                ],
            ),
            one_cell(
                "pyramid5",
                &[
                    [0., 0., 0.],
                    [1., 0., 0.],
                    [1., 1., 0.],
                    [0., 1., 0.],
                    [0.5, 0.5, 0.5_f64.sqrt()],
                ],
            ),
        ];
        for mesh in meshes {
            for metric in [
                QualityMetric::ScaledJacobian,
                QualityMetric::Skewness,
                QualityMetric::AspectRatio,
                QualityMetric::Compactness,
            ] {
                let value = score(&mesh, metric).expect("supported");
                assert!(value > 0.999, "{}: {value}", metric.label());
            }
        }
    }

    #[test]
    fn degeneracy_is_zero_and_unsupported_is_na() {
        let flat = one_cell("tri3", &[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]);
        assert_eq!(score(&flat, QualityMetric::ScaledJacobian), Some(0.0));
        let polygon = one_cell(
            "polygon",
            &[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
        );
        assert_eq!(score(&polygon, QualityMetric::ScaledJacobian), None);
    }

    #[test]
    fn higher_order_metrics_use_corner_nodes_only() {
        let mut p = vec![[0., 0., 0.], [1., 0., 0.], [0.5, 3.0_f64.sqrt() / 2.0, 0.]];
        p.extend([[50., 20., 2.], [-20., 9., 4.], [7., -30., 3.]]);
        let mesh = one_cell("tri6", &p);
        assert!(score(&mesh, QualityMetric::ScaledJacobian).unwrap() > 0.999);
    }

    #[test]
    fn scores_are_scale_and_rotation_invariant() {
        let base = [[0., 0., 0.], [2., 0., 0.], [1.2, 1., 0.]];
        let transformed = base.map(|[x, y, z]| [-3.0 * y + 4.0, 3.0 * x - 8.0, 3.0 * z + 2.0]);
        for metric in QualityMetric::ALL {
            let a = score(&one_cell("tri3", &base), metric);
            let b = score(&one_cell("tri3", &transformed), metric);
            match (a, b) {
                (Some(a), Some(b)) => assert!((a - b).abs() < 1.0e-12),
                (a, b) => assert_eq!(a, b),
            }
        }
    }

    #[test]
    fn inverted_tet_and_degenerate_supported_families_score_zero() {
        let inverted = one_cell(
            "tet4",
            &[[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.]],
        );
        assert_eq!(score(&inverted, QualityMetric::ScaledJacobian), Some(0.0));

        for (family, count) in [
            ("tri3", 3),
            ("quad4", 4),
            ("tet4", 4),
            ("hex8", 8),
            ("prism6", 6),
            ("pyramid5", 5),
        ] {
            let coordinates = vec![[0.0; 3]; count];
            let mesh = one_cell(family, &coordinates);
            for metric in QualityMetric::ALL {
                assert_eq!(
                    score(&mesh, metric),
                    Some(0.0),
                    "{family} {}",
                    metric.label()
                );
            }
        }
    }

    fn adjacent_quads(skew: f64) -> MeshIr {
        let mut builder = MeshIrBuilder::new();
        let zone = builder.zone("zone", "fluid");
        let p0 = builder.point(0., 0., 0.).unwrap();
        let p1 = builder.point(1., 0., 0.).unwrap();
        let p2 = builder.point(1., 1., 0.).unwrap();
        let p3 = builder.point(0., 1., 0.).unwrap();
        let p4 = builder.point(2., skew, 0.).unwrap();
        let p5 = builder.point(2., 1. + skew, 0.).unwrap();
        builder.cell("quad4", vec![p0, p1, p2, p3], zone).unwrap();
        builder.cell("quad4", vec![p1, p4, p5, p2], zone).unwrap();
        builder.build().unwrap()
    }

    #[test]
    fn orthogonality_detects_skewed_neighbors() {
        let aligned = analyze(&adjacent_quads(0.0), QualityMetric::Orthogonality);
        let skewed = analyze(&adjacent_quads(0.8), QualityMetric::Orthogonality);
        assert!(aligned.cells.iter().all(|cell| cell.score.unwrap() > 0.999));
        assert!(skewed.cells[0].score.unwrap() < aligned.cells[0].score.unwrap());
    }

    #[test]
    fn analysis_uses_only_highest_dimensional_cells() {
        let mut builder = MeshIrBuilder::new();
        let zone = builder.zone("zone", "fluid");
        let ids: Vec<u64> = [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
            .into_iter()
            .map(|p| builder.point(p[0], p[1], p[2]).unwrap())
            .collect();
        builder.cell("tri3", ids[..3].to_vec(), zone).unwrap();
        let tet = builder.cell("tet4", ids, zone).unwrap();
        let report = analyze(&builder.build().unwrap(), QualityMetric::AspectRatio);
        assert_eq!(report.top_dimension, Some(3));
        assert_eq!(report.cells.len(), 1);
        assert_eq!(report.cells[0].cell_id, tet);
    }

    #[test]
    fn malformed_references_are_na_and_counted() {
        let mut mesh = one_cell("tri3", &[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]);
        mesh.cells[0].point_ids[2] = u64::MAX;
        let report = analyze(&mesh, QualityMetric::AspectRatio);
        assert_eq!(report.cells[0].score, None);
        assert_eq!(report.unsupported_count, 1);
    }
}
