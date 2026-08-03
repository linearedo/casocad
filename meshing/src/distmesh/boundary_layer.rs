//! Boundary-layer station scoring lives here; normal construction remains in
//! the parent module so its established row heights and connectivity stay unchanged.

use std::cmp::Ordering;

use super::*;

#[derive(Debug, Clone, Copy)]
pub(super) struct StripScore {
    valid: bool,
    worst_skewness: f64,
    percentile_skewness: f64,
    minimum_scaled_jacobian: f64,
    tangential_error: f64,
}

impl StripScore {
    pub fn invalid() -> Self {
        Self {
            valid: false,
            worst_skewness: f64::INFINITY,
            percentile_skewness: f64::INFINITY,
            minimum_scaled_jacobian: f64::NEG_INFINITY,
            tangential_error: f64::INFINITY,
        }
    }

    pub fn is_valid(self) -> bool {
        self.valid
    }

    pub fn from_quads(quads: &[Vec<[f64; 3]>], tangential_error: f64) -> Self {
        let mut skewness = Vec::with_capacity(quads.len());
        let mut minimum_scaled_jacobian: f64 = 1.0;
        let mut valid = !quads.is_empty();
        for quad in quads {
            let jacobian = quality_score("quad4", quad, QualityMetric::ScaledJacobian)
                .unwrap_or(f64::NEG_INFINITY);
            let skew =
                quality_score("quad4", quad, QualityMetric::Skewness).unwrap_or(f64::INFINITY);
            valid &= jacobian > VALID_QUALITY && skew.is_finite();
            minimum_scaled_jacobian = minimum_scaled_jacobian.min(jacobian);
            skewness.push(skew);
        }
        skewness.sort_by(f64::total_cmp);
        let percentile = skewness
            .get((skewness.len().saturating_sub(1) * 99) / 100)
            .copied()
            .unwrap_or(f64::INFINITY);
        Self {
            valid,
            worst_skewness: skewness.last().copied().unwrap_or(f64::INFINITY),
            percentile_skewness: percentile,
            minimum_scaled_jacobian,
            tangential_error,
        }
    }

    pub fn better_than(self, other: Self) -> bool {
        self.valid
            .cmp(&other.valid)
            .then_with(|| other.worst_skewness.total_cmp(&self.worst_skewness))
            .then_with(|| {
                other
                    .percentile_skewness
                    .total_cmp(&self.percentile_skewness)
            })
            .then_with(|| {
                self.minimum_scaled_jacobian
                    .total_cmp(&other.minimum_scaled_jacobian)
            })
            .then_with(|| other.tangential_error.total_cmp(&self.tangential_error))
            == Ordering::Greater
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strip_quality_precedes_tangential_size_error() {
        let lower_skew = StripScore {
            valid: true,
            worst_skewness: 0.2,
            percentile_skewness: 0.15,
            minimum_scaled_jacobian: 0.8,
            tangential_error: 0.4,
        };
        let closer_size = StripScore {
            valid: true,
            worst_skewness: 0.3,
            percentile_skewness: 0.2,
            minimum_scaled_jacobian: 0.9,
            tangential_error: 0.0,
        };
        assert!(lower_skew.better_than(closer_size));
        assert!(!closer_size.better_than(lower_skew));
    }
}
