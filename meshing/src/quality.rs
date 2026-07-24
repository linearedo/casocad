//! On-demand mesh quality. Unsupported element/metric combinations are
//! represented by `None`; finalized Arrow files are never mutated.

use std::collections::BTreeMap;
use std::sync::Arc;

use arrow_array::{Array, Float64Array, LargeListArray, StringArray, UInt64Array};

use crate::{MeshError, MeshFile, MeshResult, RowKind};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
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

#[derive(Debug, Clone, PartialEq)]
pub struct CellQuality {
    pub cell_id: u64,
    pub score: Option<f64>,
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct QualityStatistics {
    pub supported: u64,
    pub unsupported: u64,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub mean: Option<f64>,
}

#[derive(Debug, Clone)]
struct CacheEntry {
    tick: u64,
    bytes: usize,
    values: Arc<Vec<CellQuality>>,
}

#[derive(Debug)]
pub struct QualityService {
    budget_bytes: usize,
    used_bytes: usize,
    tick: u64,
    cache: BTreeMap<(u64, QualityMetric), CacheEntry>,
}

impl QualityService {
    pub fn new(budget_bytes: usize) -> Self {
        Self {
            budget_bytes,
            used_bytes: 0,
            tick: 0,
            cache: BTreeMap::new(),
        }
    }

    pub fn tile_scores(
        &mut self,
        file: &MeshFile,
        tile_id: u64,
        metric: QualityMetric,
    ) -> MeshResult<Arc<Vec<CellQuality>>> {
        self.tick = self.tick.wrapping_add(1);
        if let Some(entry) = self.cache.get_mut(&(tile_id, metric)) {
            entry.tick = self.tick;
            return Ok(entry.values.clone());
        }
        let points = tile_points(file, tile_id)?;
        let mut values = Vec::new();
        for entry in file.tile_batches(tile_id, RowKind::Cell) {
            let batch = file.batch_view(entry.batch_index)?;
            let ids = column::<UInt64Array>(batch.record_batch(), "entity_id")?;
            let types = column::<StringArray>(batch.record_batch(), "element_type")?;
            let connectivity = column::<LargeListArray>(batch.record_batch(), "point_ids")?;
            for row in 0..batch.len() {
                let point_ids = connectivity.value(row);
                let point_ids = point_ids
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| {
                        MeshError::InvalidFile("point_ids must be LargeList<u64>".into())
                    })?;
                let geometry: Option<Vec<[f64; 3]>> = point_ids
                    .values()
                    .iter()
                    .map(|id| points.get(id).copied())
                    .collect();
                values.push(CellQuality {
                    cell_id: ids.value(row),
                    score: geometry
                        .as_deref()
                        .and_then(|points| quality_score(types.value(row), points, metric)),
                });
            }
        }
        let values = Arc::new(values);
        let bytes = values.len() * std::mem::size_of::<CellQuality>();
        self.evict_for(bytes);
        if bytes <= self.budget_bytes {
            self.used_bytes += bytes;
            self.cache.insert(
                (tile_id, metric),
                CacheEntry {
                    tick: self.tick,
                    bytes,
                    values: values.clone(),
                },
            );
        }
        Ok(values)
    }

    pub fn global_statistics(
        &mut self,
        file: &MeshFile,
        metric: QualityMetric,
    ) -> MeshResult<QualityStatistics> {
        let mut accumulator = Accumulator::default();
        let tiles: std::collections::BTreeSet<u64> = file
            .entity_batches(RowKind::Cell)
            .filter_map(|entry| entry.spatial_node_id)
            .collect();
        for tile in tiles {
            for value in self.tile_scores(file, tile, metric)?.iter() {
                accumulator.push(value.score);
            }
        }
        Ok(accumulator.finish())
    }

    fn evict_for(&mut self, incoming: usize) {
        while self.used_bytes.saturating_add(incoming) > self.budget_bytes {
            let Some(key) = self
                .cache
                .iter()
                .min_by_key(|(_, entry)| entry.tick)
                .map(|(key, _)| *key)
            else {
                break;
            };
            if let Some(entry) = self.cache.remove(&key) {
                self.used_bytes = self.used_bytes.saturating_sub(entry.bytes);
            }
        }
    }
}

impl Default for QualityService {
    fn default() -> Self {
        Self::new(64 * 1024 * 1024)
    }
}

#[derive(Default)]
struct Accumulator {
    count: u64,
    unsupported: u64,
    mean: f64,
    minimum: f64,
    maximum: f64,
}

impl Accumulator {
    fn push(&mut self, value: Option<f64>) {
        let Some(value) = value else {
            self.unsupported += 1;
            return;
        };
        if self.count == 0 {
            self.minimum = value;
            self.maximum = value;
            self.mean = value;
            self.count = 1;
            return;
        }
        self.count += 1;
        self.minimum = self.minimum.min(value);
        self.maximum = self.maximum.max(value);
        self.mean += (value - self.mean) / self.count as f64;
    }

    fn finish(self) -> QualityStatistics {
        QualityStatistics {
            supported: self.count,
            unsupported: self.unsupported,
            minimum: (self.count > 0).then_some(self.minimum),
            maximum: (self.count > 0).then_some(self.maximum),
            mean: (self.count > 0).then_some(self.mean),
        }
    }
}

pub fn quality_score(
    element_type: &str,
    points: &[[f64; 3]],
    metric: QualityMetric,
) -> Option<f64> {
    let value = match (element_type, metric) {
        ("tri3" | "tri6", QualityMetric::ScaledJacobian) => triangle_scaled_jacobian(points)?,
        ("tri3" | "tri6", QualityMetric::Skewness | QualityMetric::Orthogonality) => {
            triangle_skewness(points)?
        }
        ("tri3" | "tri6", QualityMetric::AspectRatio) => edge_ratio(points, 3)?,
        ("tri3" | "tri6", QualityMetric::Compactness) => triangle_compactness(points)?,
        ("quad4" | "quad8" | "quad9", QualityMetric::ScaledJacobian) => {
            polygon_scaled_jacobian(points, 4, 1.0)?
        }
        ("quad4" | "quad8" | "quad9", QualityMetric::Skewness) => quad_skewness(points)?,
        ("quad4" | "quad8" | "quad9", QualityMetric::AspectRatio) => edge_ratio(points, 4)?,
        ("quad4" | "quad8" | "quad9", QualityMetric::Compactness) => {
            polygon_compactness(points, 4)?
        }
        ("quad4" | "quad8" | "quad9", QualityMetric::Orthogonality) => quad_skewness(points)?,
        ("tet4" | "tet10", QualityMetric::ScaledJacobian) => tetra_scaled_jacobian(points)?,
        ("tet4" | "tet10", QualityMetric::AspectRatio) => tetra_edge_ratio(points)?,
        ("tet4" | "tet10", QualityMetric::Compactness) => tetra_compactness(points)?,
        _ => return None,
    };
    Some(value.clamp(0.0, 1.0))
}

fn tile_points(file: &MeshFile, tile_id: u64) -> MeshResult<BTreeMap<u64, [f64; 3]>> {
    let mut result = BTreeMap::new();
    for entry in file.tile_batches(tile_id, RowKind::Point) {
        let batch = file.batch_view(entry.batch_index)?;
        let ids = column::<UInt64Array>(batch.record_batch(), "entity_id")?;
        let x = column::<Float64Array>(batch.record_batch(), "x")?;
        let y = column::<Float64Array>(batch.record_batch(), "y")?;
        let z = column::<Float64Array>(batch.record_batch(), "z")?;
        for row in 0..batch.len() {
            result.insert(ids.value(row), [x.value(row), y.value(row), z.value(row)]);
        }
    }
    Ok(result)
}

fn column<'a, T: 'static>(batch: &'a arrow_array::RecordBatch, name: &str) -> MeshResult<&'a T> {
    batch
        .column_by_name(name)
        .and_then(|array| array.as_any().downcast_ref())
        .ok_or_else(|| MeshError::InvalidFile(format!("invalid Arrow column {name:?}")))
}

#[derive(Clone, Copy)]
struct V([f64; 3]);

impl V {
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
}

impl std::ops::Sub for V {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        Self(std::array::from_fn(|axis| self.0[axis] - other.0[axis]))
    }
}

fn triangle_scaled_jacobian(points: &[[f64; 3]]) -> Option<f64> {
    polygon_scaled_jacobian(points, 3, 2.0 / 3.0_f64.sqrt())
}

fn polygon_scaled_jacobian(points: &[[f64; 3]], count: usize, normalization: f64) -> Option<f64> {
    let points: Vec<V> = points.get(..count)?.iter().copied().map(V).collect();
    let normal = (points[1] - points[0]).cross(points[2] - points[0]);
    let normal_length = normal.length();
    if normal_length <= f64::EPSILON {
        return Some(0.0);
    }
    let mut minimum = f64::INFINITY;
    for index in 0..count {
        let incoming = points[(index + count - 1) % count] - points[index];
        let outgoing = points[(index + 1) % count] - points[index];
        let denominator = incoming.length() * outgoing.length() * normal_length;
        if denominator <= f64::EPSILON {
            return Some(0.0);
        }
        minimum = minimum.min(-incoming.cross(outgoing).dot(normal) / denominator);
    }
    Some(minimum * normalization)
}

fn edge_ratio(points: &[[f64; 3]], count: usize) -> Option<f64> {
    let points = points.get(..count)?;
    let lengths: Vec<f64> = (0..count)
        .map(|index| (V(points[(index + 1) % count]) - V(points[index])).length())
        .collect();
    let minimum = lengths.iter().copied().reduce(f64::min)?;
    let maximum = lengths.iter().copied().reduce(f64::max)?;
    (maximum > f64::EPSILON).then_some(minimum / maximum)
}

fn triangle_compactness(points: &[[f64; 3]]) -> Option<f64> {
    let points = points.get(..3)?;
    let area = (V(points[1]) - V(points[0]))
        .cross(V(points[2]) - V(points[0]))
        .length()
        * 0.5;
    let sum_squared: f64 = (0..3)
        .map(|index| {
            (V(points[(index + 1) % 3]) - V(points[index]))
                .length()
                .powi(2)
        })
        .sum();
    (sum_squared > f64::EPSILON).then_some(4.0 * 3.0_f64.sqrt() * area / sum_squared)
}

fn polygon_compactness(points: &[[f64; 3]], count: usize) -> Option<f64> {
    let points = points.get(..count)?;
    let area: f64 = (1..count - 1)
        .map(|index| {
            (V(points[index]) - V(points[0]))
                .cross(V(points[index + 1]) - V(points[0]))
                .length()
                * 0.5
        })
        .sum();
    let perimeter: f64 = (0..count)
        .map(|index| (V(points[(index + 1) % count]) - V(points[index])).length())
        .sum();
    (perimeter > f64::EPSILON).then_some(4.0 * std::f64::consts::PI * area / perimeter.powi(2))
}

fn triangle_skewness(points: &[[f64; 3]]) -> Option<f64> {
    angle_score(points, 3, 60.0_f64.to_radians())
}

fn quad_skewness(points: &[[f64; 3]]) -> Option<f64> {
    angle_score(points, 4, 90.0_f64.to_radians())
}

fn angle_score(points: &[[f64; 3]], count: usize, ideal: f64) -> Option<f64> {
    let points = points.get(..count)?;
    let mut worst = 0.0_f64;
    for index in 0..count {
        let a = V(points[(index + count - 1) % count]) - V(points[index]);
        let b = V(points[(index + 1) % count]) - V(points[index]);
        let denominator = a.length() * b.length();
        if denominator <= f64::EPSILON {
            return Some(0.0);
        }
        let angle = (a.dot(b) / denominator).clamp(-1.0, 1.0).acos();
        worst = worst.max((angle - ideal).abs() / ideal.max(std::f64::consts::PI - ideal));
    }
    Some(1.0 - worst)
}

fn tetra_scaled_jacobian(points: &[[f64; 3]]) -> Option<f64> {
    let p = points.get(..4)?;
    let corners = [(0, 1, 2, 3), (1, 0, 3, 2), (2, 0, 1, 3), (3, 0, 2, 1)];
    corners
        .into_iter()
        .map(|(o, a, b, c)| {
            let x = V(p[a]) - V(p[o]);
            let y = V(p[b]) - V(p[o]);
            let z = V(p[c]) - V(p[o]);
            let denominator = x.length() * y.length() * z.length();
            if denominator <= f64::EPSILON {
                0.0
            } else {
                x.cross(y).dot(z) / denominator * 2.0_f64.sqrt()
            }
        })
        .reduce(f64::min)
}

fn tetra_edge_ratio(points: &[[f64; 3]]) -> Option<f64> {
    let p = points.get(..4)?;
    let lengths: Vec<_> = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        .into_iter()
        .map(|(a, b)| (V(p[a]) - V(p[b])).length())
        .collect();
    let minimum = lengths.iter().copied().reduce(f64::min)?;
    let maximum = lengths.iter().copied().reduce(f64::max)?;
    (maximum > f64::EPSILON).then_some(minimum / maximum)
}

fn tetra_compactness(points: &[[f64; 3]]) -> Option<f64> {
    let p = points.get(..4)?;
    let volume = ((V(p[1]) - V(p[0]))
        .cross(V(p[2]) - V(p[0]))
        .dot(V(p[3]) - V(p[0])))
    .abs()
        / 6.0;
    let sum_edges: f64 = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
        .into_iter()
        .map(|(a, b)| (V(p[a]) - V(p[b])).length().powi(2))
        .sum();
    (sum_edges > f64::EPSILON).then_some(12.0 * (3.0 * volume).powf(2.0 / 3.0) / sum_edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_triangle_quad_and_tetrahedron_are_supported() {
        let triangle = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 2.0, 0.0],
        ];
        let quad = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
        ];
        let tetrahedron = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 2.0, 0.0],
            [0.5, 3.0_f64.sqrt() / 6.0, (2.0 / 3.0_f64).sqrt()],
        ];
        assert!(
            quality_score("tri3", &triangle, QualityMetric::ScaledJacobian)
                .is_some_and(|score| (score - 1.0).abs() < 1.0e-12)
        );
        assert!(quality_score("quad4", &quad, QualityMetric::ScaledJacobian)
            .is_some_and(|score| (score - 1.0).abs() < 1.0e-12));
        assert!(
            quality_score("tet4", &tetrahedron, QualityMetric::AspectRatio)
                .is_some_and(|score| (score - 1.0).abs() < 1.0e-12)
        );
        assert_eq!(
            quality_score("hex8", &[], QualityMetric::ScaledJacobian),
            None
        );
    }
}
