//! On-demand mesh quality. Unsupported element/metric combinations are
//! represented by `None`; finalized Arrow files are never mutated.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
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

    pub fn parse(value: &str) -> Option<Self> {
        Some(match value {
            "scaled_jacobian" => Self::ScaledJacobian,
            "skewness" => Self::Skewness,
            "aspect_ratio" => Self::AspectRatio,
            "compactness" => Self::Compactness,
            "orthogonality" => Self::Orthogonality,
            _ => return None,
        })
    }
}

pub fn quality_score(
    element_type: &str,
    points: &[[f64; 3]],
    metric: QualityMetric,
) -> Option<f64> {
    quality_score_with_neighbors(element_type, points, metric, &BTreeMap::new())
}

/// Score a cell while optionally using adjacent cell centers keyed by the
/// canonical point-ID signature of each edge (2D) or face (3D).
///
/// The public `quality_score` wrapper treats every side as a boundary. Query
/// execution supplies neighbors for exact Arrow topology when Orthogonality
/// is requested.
pub fn quality_score_with_neighbors(
    element_type: &str,
    points: &[[f64; 3]],
    metric: QualityMetric,
    neighbors: &BTreeMap<Vec<u64>, [f64; 3]>,
) -> Option<f64> {
    let corners = corner_points(element_type, points)?;
    let raw = match metric {
        QualityMetric::ScaledJacobian => scaled_jacobian(element_type, corners),
        QualityMetric::Skewness => skewness(element_type, corners),
        QualityMetric::AspectRatio => aspect_ratio(element_type, corners),
        QualityMetric::Compactness => compactness(element_type, corners),
        QualityMetric::Orthogonality => orthogonality(element_type, corners, None, neighbors),
    }?;
    Some(unit(raw))
}

/// Topology-aware variant used by Arrow queries. `point_ids` must correspond
/// to `points`; only corner nodes participate in the calculation.
pub(crate) fn quality_score_exact(
    element_type: &str,
    point_ids: &[u64],
    points: &[[f64; 3]],
    metric: QualityMetric,
    neighbor_centers: &BTreeMap<Vec<u64>, [f64; 3]>,
) -> Option<f64> {
    let count = match element_type {
        "polygon" => points.len(),
        _ => corner_count(element_type)?,
    };
    let corners = points.get(..count)?;
    let ids = point_ids.get(..count)?;
    let raw = match metric {
        QualityMetric::ScaledJacobian => scaled_jacobian(element_type, corners),
        QualityMetric::Skewness => skewness(element_type, corners),
        QualityMetric::AspectRatio => aspect_ratio(element_type, corners),
        QualityMetric::Compactness => compactness(element_type, corners),
        QualityMetric::Orthogonality => {
            orthogonality(element_type, corners, Some(ids), neighbor_centers)
        }
    }?;
    Some(unit(raw))
}

pub(crate) fn polyhedron_quality_score(
    points: &[[f64; 3]],
    faces: &[(Vec<u64>, Vec<[f64; 3]>)],
    metric: QualityMetric,
    neighbor_centers: &BTreeMap<Vec<u64>, [f64; 3]>,
) -> Option<f64> {
    let p = points.iter().copied().map(V).collect::<Vec<_>>();
    let raw = match metric {
        QualityMetric::Skewness => faces
            .iter()
            .map(|(_, face)| polygon_skewness(&face.iter().copied().map(V).collect::<Vec<_>>()))
            .collect::<Option<Vec<_>>>()?
            .into_iter()
            .reduce(f64::min)?,
        QualityMetric::Orthogonality => {
            if p.is_empty() || faces.is_empty() {
                return None;
            }
            let center = centroid(&p);
            let mut result = 1.0_f64;
            for (ids, face) in faces {
                let face = face.iter().copied().map(V).collect::<Vec<_>>();
                let normal = polygon_normal(&face).normalized()?;
                let mut signature = ids.clone();
                signature.sort_unstable();
                let target = neighbor_centers
                    .get(&signature)
                    .copied()
                    .map(V)
                    .unwrap_or_else(|| centroid(&face))
                    - center;
                result = result.min(normal.dot(target.normalized()?).abs());
            }
            result
        }
        QualityMetric::ScaledJacobian | QualityMetric::AspectRatio | QualityMetric::Compactness => {
            return None
        }
    };
    Some(unit(raw))
}

pub(crate) fn corner_count(element_type: &str) -> Option<usize> {
    Some(match element_type {
        "tri3" | "tri6" => 3,
        "quad4" | "quad8" | "quad9" => 4,
        "tet4" | "tet10" => 4,
        "hex8" | "hex20" | "hex27" => 8,
        "prism6" | "prism15" => 6,
        "pyramid5" | "pyramid13" => 5,
        "polygon" | "polyhedron" => return None,
        _ => return None,
    })
}

pub(crate) fn side_indices(element_type: &str, corner_len: usize) -> Option<Vec<Vec<usize>>> {
    Some(match element_type {
        "tri3" | "tri6" | "quad4" | "quad8" | "quad9" | "polygon" => (0..corner_len)
            .map(|index| vec![index, (index + 1) % corner_len])
            .collect(),
        "tet4" | "tet10" => vec![vec![0, 2, 1], vec![0, 1, 3], vec![1, 2, 3], vec![2, 0, 3]],
        "hex8" | "hex20" | "hex27" => vec![
            vec![0, 3, 2, 1],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
        ],
        "prism6" | "prism15" => vec![
            vec![0, 2, 1],
            vec![3, 4, 5],
            vec![0, 1, 4, 3],
            vec![1, 2, 5, 4],
            vec![2, 0, 3, 5],
        ],
        "pyramid5" | "pyramid13" => vec![
            vec![0, 3, 2, 1],
            vec![0, 1, 4],
            vec![1, 2, 4],
            vec![2, 3, 4],
            vec![3, 0, 4],
        ],
        "polyhedron" => return None,
        _ => return None,
    })
}

fn corner_points<'a>(element_type: &str, points: &'a [[f64; 3]]) -> Option<&'a [[f64; 3]]> {
    let count = match element_type {
        "polygon" => points.len(),
        _ => corner_count(element_type)?,
    };
    points.get(..count)
}

#[derive(Clone, Copy)]
struct V([f64; 3]);

impl V {
    const ZERO: Self = Self([0.0; 3]);

    fn dot(self, other: Self) -> f64 {
        (0..3).map(|axis| self.0[axis] * other.0[axis]).sum()
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

    fn add(self, other: Self) -> Self {
        Self(std::array::from_fn(|axis| self.0[axis] + other.0[axis]))
    }
}

impl std::ops::Sub for V {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(std::array::from_fn(|axis| self.0[axis] - other.0[axis]))
    }
}

impl std::ops::Div<f64> for V {
    type Output = Self;

    fn div(self, divisor: f64) -> Self {
        Self(self.0.map(|value| value / divisor))
    }
}

fn scaled_jacobian(element_type: &str, points: &[[f64; 3]]) -> Option<f64> {
    let p = points.iter().copied().map(V).collect::<Vec<_>>();
    match element_type {
        "tri3" | "tri6" => scaled_jacobian_2d(&p, 2.0 / 3.0_f64.sqrt()),
        "quad4" | "quad8" | "quad9" => scaled_jacobian_2d(&p, 1.0),
        "tet4" | "tet10" => corner_jacobians(
            &p,
            &[(0, 1, 2, 3), (1, 0, 3, 2), (2, 0, 1, 3), (3, 0, 2, 1)],
            2.0_f64.sqrt(),
        ),
        "hex8" | "hex20" | "hex27" => corner_jacobians(
            &p,
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
            &p,
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
                &p,
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

fn scaled_jacobian_2d(points: &[V], normalization: f64) -> Option<f64> {
    if points.len() < 3 {
        return None;
    }
    let normal = polygon_normal(points);
    let normal_length = normal.length();
    if normal_length <= EPS {
        return Some(0.0);
    }
    let normal = normal / normal_length;
    let mut minimum = f64::INFINITY;
    for index in 0..points.len() {
        let incoming = points[(index + points.len() - 1) % points.len()] - points[index];
        let outgoing = points[(index + 1) % points.len()] - points[index];
        let denominator = incoming.length() * outgoing.length();
        if denominator <= EPS {
            return Some(0.0);
        }
        minimum = minimum.min(-incoming.cross(outgoing).dot(normal) / denominator);
    }
    Some((minimum * normalization).max(0.0))
}

fn corner_jacobians(
    points: &[V],
    corners: &[(usize, usize, usize, usize)],
    scale: f64,
) -> Option<f64> {
    if corners
        .iter()
        .any(|&(o, a, b, c)| [o, a, b, c].into_iter().any(|index| index >= points.len()))
    {
        return None;
    }
    Some(
        corners
            .iter()
            .map(|&(o, a, b, c)| {
                normalized_det(
                    points[a] - points[o],
                    points[b] - points[o],
                    points[c] - points[o],
                )
            })
            .fold(f64::INFINITY, f64::min)
            .mul_add(scale, 0.0)
            .max(0.0),
    )
}

fn skewness(element_type: &str, points: &[[f64; 3]]) -> Option<f64> {
    let p = points.iter().copied().map(V).collect::<Vec<_>>();
    match element_dimension_for_quality(element_type)? {
        2 => polygon_skewness(&p),
        3 => side_indices(element_type, p.len())?
            .iter()
            .map(|side| polygon_skewness(&side.iter().map(|&index| p[index]).collect::<Vec<_>>()))
            .collect::<Option<Vec<_>>>()
            .and_then(|scores| scores.into_iter().reduce(f64::min)),
        _ => None,
    }
}

fn polygon_skewness(points: &[V]) -> Option<f64> {
    if points.len() < 3 {
        return None;
    }
    if polygon_normal(points).length() <= EPS {
        return Some(0.0);
    }
    let ideal = std::f64::consts::PI * (points.len() as f64 - 2.0) / points.len() as f64;
    let mut minimum = f64::INFINITY;
    let mut maximum = 0.0_f64;
    for index in 0..points.len() {
        let a = points[(index + points.len() - 1) % points.len()] - points[index];
        let b = points[(index + 1) % points.len()] - points[index];
        let denominator = a.length() * b.length();
        if denominator <= EPS {
            return Some(0.0);
        }
        let angle = (a.dot(b) / denominator).clamp(-1.0, 1.0).acos();
        minimum = minimum.min(angle);
        maximum = maximum.max(angle);
    }
    Some(
        1.0 - ((maximum - ideal) / (std::f64::consts::PI - ideal))
            .max((ideal - minimum) / ideal)
            .max(0.0),
    )
}

fn aspect_ratio(element_type: &str, points: &[[f64; 3]]) -> Option<f64> {
    let pairs = edge_pairs(element_type, points.len())?;
    let p = points.iter().copied().map(V).collect::<Vec<_>>();
    let lengths = pairs
        .iter()
        .map(|&(a, b)| (p[b] - p[a]).length())
        .collect::<Vec<_>>();
    let minimum = lengths.iter().copied().reduce(f64::min)?;
    let maximum = lengths.iter().copied().reduce(f64::max)?;
    Some(if minimum <= EPS || maximum <= EPS {
        0.0
    } else {
        minimum / maximum
    })
}

fn compactness(element_type: &str, points: &[[f64; 3]]) -> Option<f64> {
    let p = points.iter().copied().map(V).collect::<Vec<_>>();
    match element_dimension_for_quality(element_type)? {
        2 => {
            let area = polygon_area(&p);
            let perimeter = (0..p.len())
                .map(|index| (p[(index + 1) % p.len()] - p[index]).length())
                .sum::<f64>();
            if area <= EPS || perimeter <= EPS {
                return Some(0.0);
            }
            let count = p.len() as f64;
            Some(4.0 * count * (std::f64::consts::PI / count).tan() * area / perimeter.powi(2))
        }
        3 => {
            if element_type == "polyhedron" {
                return None;
            }
            let faces = side_indices(element_type, p.len())?
                .into_iter()
                .map(|side| side.into_iter().map(|index| p[index]).collect::<Vec<_>>())
                .collect::<Vec<_>>();
            let center = centroid(&p);
            let surface = faces.iter().map(|face| polygon_area(face)).sum::<f64>();
            let volume = faces
                .iter()
                .map(|face| face_volume(center, face))
                .sum::<f64>();
            if volume <= EPS || surface <= EPS {
                return Some(0.0);
            }
            let reference = match element_type {
                "tet4" | "tet10" => (2.0_f64.sqrt() / 12.0) / 3.0_f64.sqrt().powf(1.5),
                "hex8" | "hex20" | "hex27" => 1.0 / 6.0_f64.powf(1.5),
                "prism6" | "prism15" => {
                    let volume = 3.0_f64.sqrt() / 4.0;
                    let surface = 3.0_f64.sqrt() / 2.0 + 3.0;
                    volume / surface.powf(1.5)
                }
                "pyramid5" | "pyramid13" => {
                    let height = 0.5_f64.sqrt();
                    (height / 3.0) / (1.0 + 4.0 * (height * height + 0.25).sqrt() / 2.0).powf(1.5)
                }
                _ => return None,
            };
            Some(volume / surface.powf(1.5) / reference)
        }
        _ => None,
    }
}

fn orthogonality(
    element_type: &str,
    points: &[[f64; 3]],
    point_ids: Option<&[u64]>,
    neighbors: &BTreeMap<Vec<u64>, [f64; 3]>,
) -> Option<f64> {
    let dimension = element_dimension_for_quality(element_type)?;
    if !matches!(dimension, 2 | 3) {
        return None;
    }
    let p = points.iter().copied().map(V).collect::<Vec<_>>();
    let center = centroid(&p);
    let plane = (dimension == 2)
        .then(|| polygon_normal(&p).normalized())
        .flatten();
    if dimension == 2 && plane.is_none() {
        return Some(0.0);
    }
    let sides = side_indices(element_type, p.len())?;
    let mut result = 1.0_f64;
    for side in sides {
        let side_points = side.iter().map(|&index| p[index]).collect::<Vec<_>>();
        let side_center = centroid(&side_points);
        let normal = if dimension == 2 {
            let tangent = (side_points[1] - side_points[0]).normalized();
            tangent.and_then(|tangent| tangent.cross(plane?).normalized())
        } else {
            polygon_normal(&side_points).normalized()
        };
        let Some(normal) = normal else {
            return Some(0.0);
        };
        let signature = point_ids.map(|ids| {
            let mut signature = side.iter().map(|&index| ids[index]).collect::<Vec<_>>();
            signature.sort_unstable();
            signature
        });
        let target = signature
            .as_ref()
            .and_then(|signature| neighbors.get(signature))
            .copied()
            .map(V)
            .unwrap_or(side_center)
            - center;
        let Some(target) = target.normalized() else {
            return Some(0.0);
        };
        result = result.min(normal.dot(target).abs());
    }
    Some(match element_type {
        // The stored cell center is the corner centroid; normalize the
        // regular square-pyramid reference to the common 1.0 convention.
        "pyramid5" | "pyramid13" => result / (2.0 * 2.0_f64.sqrt() / 3.0),
        _ => result,
    })
}

fn edge_pairs(element_type: &str, count: usize) -> Option<Vec<(usize, usize)>> {
    Some(match element_type {
        "tri3" | "tri6" | "quad4" | "quad8" | "quad9" | "polygon" => (0..count)
            .map(|index| (index, (index + 1) % count))
            .collect(),
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
    })
}

fn polygon_normal(points: &[V]) -> V {
    let mut normal = V::ZERO;
    for index in 0..points.len() {
        normal = normal + points[index].cross(points[(index + 1) % points.len()]);
    }
    normal
}

fn polygon_area(points: &[V]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }
    let origin = points[0];
    (1..points.len() - 1)
        .map(|index| {
            (points[index] - origin)
                .cross(points[index + 1] - origin)
                .length()
                * 0.5
        })
        .sum()
}

fn face_volume(center: V, face: &[V]) -> f64 {
    if face.len() < 3 {
        return 0.0;
    }
    (1..face.len() - 1)
        .map(|index| {
            ((face[0] - center).dot((face[index] - center).cross(face[index + 1] - center))).abs()
                / 6.0
        })
        .sum()
}

fn centroid(points: &[V]) -> V {
    points
        .iter()
        .copied()
        .fold(V::ZERO, |sum, point| sum + point)
        / points.len() as f64
}

fn normalized_det(a: V, b: V, c: V) -> f64 {
    let denominator = a.length() * b.length() * c.length();
    if denominator <= EPS {
        0.0
    } else {
        a.dot(b.cross(c)) / denominator
    }
}

fn element_dimension_for_quality(element_type: &str) -> Option<u8> {
    Some(match element_type {
        "tri3" | "tri6" | "quad4" | "quad8" | "quad9" | "polygon" => 2,
        "tet4" | "tet10" | "hex8" | "hex20" | "hex27" | "prism6" | "prism15" | "pyramid5"
        | "pyramid13" | "polyhedron" => 3,
        _ => return None,
    })
}

fn unit(value: f64) -> f64 {
    if value.is_finite() {
        value.clamp(0.0, 1.0)
    } else {
        0.0
    }
}

const EPS: f64 = 1.0e-14;

#[cfg(test)]
mod tests {
    use super::*;

    fn score(element_type: &str, points: &[[f64; 3]], metric: QualityMetric) -> Option<f64> {
        quality_score(element_type, points, metric)
    }

    #[test]
    fn ideal_linear_families_are_normalized() {
        let h = 3.0_f64.sqrt() / 2.0;
        let tet_h = (2.0_f64 / 3.0).sqrt();
        let families: [(&str, &[[f64; 3]]); 6] = [
            ("tri3", &[[0., 0., 0.], [1., 0., 0.], [0.5, h, 0.]]),
            (
                "quad4",
                &[[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]],
            ),
            (
                "tet4",
                &[
                    [0., 0., 0.],
                    [1., 0., 0.],
                    [0.5, h, 0.],
                    [0.5, h / 3., tet_h],
                ],
            ),
            (
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
            (
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
            (
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
        for (element_type, points) in families {
            for metric in QualityMetric::ALL {
                assert!(
                    score(element_type, points, metric).is_some_and(|value| value > 0.999),
                    "{element_type} {}: {:?}",
                    metric.label(),
                    score(element_type, points, metric)
                );
            }
        }
    }

    #[test]
    fn degeneracy_inversion_unsupported_and_higher_order_are_distinct() {
        let flat = [[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]];
        assert_eq!(
            score("tri3", &flat, QualityMetric::ScaledJacobian),
            Some(0.0)
        );
        let inverted = [[0., 0., 0.], [0., 1., 0.], [1., 0., 0.], [0., 0., 1.]];
        assert_eq!(
            score("tet4", &inverted, QualityMetric::ScaledJacobian),
            Some(0.0)
        );
        assert_eq!(
            score("polyhedron", &inverted, QualityMetric::AspectRatio),
            None
        );

        let mut higher = vec![[0., 0., 0.], [1., 0., 0.], [0.5, 3.0_f64.sqrt() / 2.0, 0.]];
        higher.extend([[50., 20., 2.], [-20., 9., 4.], [7., -30., 3.]]);
        assert!(score("tri6", &higher, QualityMetric::ScaledJacobian)
            .is_some_and(|value| value > 0.999));
    }

    #[test]
    fn scores_are_scale_rotation_and_translation_invariant() {
        let base = [[0., 0., 0.], [2., 0., 0.], [1.2, 1., 0.]];
        let transformed = base.map(|[x, y, z]| [-3.0 * y + 4.0, 3.0 * x - 8.0, 3.0 * z + 2.0]);
        for metric in QualityMetric::ALL {
            let a = score("tri3", &base, metric);
            let b = score("tri3", &transformed, metric);
            match (a, b) {
                (Some(a), Some(b)) => assert!((a - b).abs() < 1.0e-12),
                (a, b) => assert_eq!(a, b),
            }
        }
    }

    #[test]
    fn topology_aware_orthogonality_detects_a_skewed_neighbor() {
        let points = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]];
        let ids = [1, 2, 3, 4];
        let mut aligned = BTreeMap::new();
        aligned.insert(vec![2, 3], [1.5, 0.5, 0.0]);
        let mut skewed = BTreeMap::new();
        skewed.insert(vec![2, 3], [1.5, 1.3, 0.0]);
        let aligned = quality_score_exact(
            "quad4",
            &ids,
            &points,
            QualityMetric::Orthogonality,
            &aligned,
        )
        .unwrap();
        let skewed = quality_score_exact(
            "quad4",
            &ids,
            &points,
            QualityMetric::Orthogonality,
            &skewed,
        )
        .unwrap();
        assert!(aligned > 0.999);
        assert!(skewed < aligned);
    }

    #[test]
    fn generic_polygon_and_topologized_polyhedron_keep_supported_metrics() {
        let polygon = [[0., 0., 0.], [1., 0., 0.], [1., 1., 0.], [0., 1., 0.]];
        assert!(quality_score_exact(
            "polygon",
            &[1, 2, 3, 4],
            &polygon,
            QualityMetric::Compactness,
            &BTreeMap::new(),
        )
        .is_some_and(|value| value > 0.999));
        assert_eq!(
            quality_score_exact(
                "polygon",
                &[1, 2, 3, 4],
                &polygon,
                QualityMetric::ScaledJacobian,
                &BTreeMap::new(),
            ),
            None
        );

        let cube = [
            [0., 0., 0.],
            [1., 0., 0.],
            [1., 1., 0.],
            [0., 1., 0.],
            [0., 0., 1.],
            [1., 0., 1.],
            [1., 1., 1.],
            [0., 1., 1.],
        ];
        let faces = [
            vec![0, 3, 2, 1],
            vec![4, 5, 6, 7],
            vec![0, 1, 5, 4],
            vec![1, 2, 6, 5],
            vec![2, 3, 7, 6],
            vec![3, 0, 4, 7],
        ]
        .into_iter()
        .map(|indices| {
            (
                indices.iter().map(|index| *index as u64 + 1).collect(),
                indices.into_iter().map(|index| cube[index]).collect(),
            )
        })
        .collect::<Vec<_>>();
        for metric in [QualityMetric::Skewness, QualityMetric::Orthogonality] {
            assert!(
                polyhedron_quality_score(&cube, &faces, metric, &BTreeMap::new())
                    .is_some_and(|value| value > 0.999)
            );
        }
        assert_eq!(
            polyhedron_quality_score(&cube, &faces, QualityMetric::Compactness, &BTreeMap::new()),
            None
        );
    }
}
