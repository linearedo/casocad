//! Exact signs for the predicates used by Delaunay topology.

use std::cmp::Ordering;

use crate::expansion::{add, determinant, difference, multiply, sign, Expansion};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sign {
    Negative,
    Zero,
    Positive,
}

impl Sign {
    pub(crate) fn ordering(self) -> Ordering {
        match self {
            Self::Negative => Ordering::Less,
            Self::Zero => Ordering::Equal,
            Self::Positive => Ordering::Greater,
        }
    }
}

fn result(value: &[f64]) -> Sign {
    match sign(value) {
        Ordering::Less => Sign::Negative,
        Ordering::Equal => Sign::Zero,
        Ordering::Greater => Sign::Positive,
    }
}

fn filtered(matrix: &[Vec<f64>]) -> Option<Sign> {
    let determinant = determinant_f64(matrix);
    let permanent = permanent_f64(matrix);
    let error = f64::EPSILON * 512.0 * permanent;
    (determinant.is_finite() && determinant.abs() > error).then_some({
        if determinant < 0.0 {
            Sign::Negative
        } else {
            Sign::Positive
        }
    })
}

fn determinant_f64(matrix: &[Vec<f64>]) -> f64 {
    match matrix.len() {
        0 => 1.0,
        1 => matrix[0][0],
        size => (0..size)
            .map(|column| {
                let minor = matrix[1..]
                    .iter()
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .filter_map(|(index, value)| (index != column).then_some(*value))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let term = matrix[0][column] * determinant_f64(&minor);
                if column % 2 == 0 {
                    term
                } else {
                    -term
                }
            })
            .sum(),
    }
}

fn permanent_f64(matrix: &[Vec<f64>]) -> f64 {
    match matrix.len() {
        0 => 1.0,
        1 => matrix[0][0].abs(),
        size => (0..size)
            .map(|column| {
                let minor = matrix[1..]
                    .iter()
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .filter_map(|(index, value)| (index != column).then_some(value.abs()))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                matrix[0][column].abs() * permanent_f64(&minor)
            })
            .sum(),
    }
}

fn lift(values: &[Expansion]) -> Expansion {
    values
        .iter()
        .fold(vec![0.0], |sum, value| add(&sum, &multiply(value, value)))
}

pub fn orient2d(a: [f64; 2], b: [f64; 2], c: [f64; 2]) -> Sign {
    let ax = a[0] - c[0];
    let ay = a[1] - c[1];
    let bx = b[0] - c[0];
    let by = b[1] - c[1];
    let approximate_det = ax * by - ay * bx;
    let error = f64::EPSILON * 8.0 * (ax * by).abs().max((ay * bx).abs());
    if approximate_det.is_finite() && approximate_det.abs() > error {
        return if approximate_det < 0.0 {
            Sign::Negative
        } else {
            Sign::Positive
        };
    }
    let ax = difference(a[0], c[0]);
    let ay = difference(a[1], c[1]);
    let bx = difference(b[0], c[0]);
    let by = difference(b[1], c[1]);
    result(&determinant(&[vec![ax, ay], vec![bx, by]]))
}

pub fn incircle(a: [f64; 2], b: [f64; 2], c: [f64; 2], d: [f64; 2]) -> Sign {
    let adx = a[0] - d[0];
    let ady = a[1] - d[1];
    let bdx = b[0] - d[0];
    let bdy = b[1] - d[1];
    let cdx = c[0] - d[0];
    let cdy = c[1] - d[1];
    let alift = adx * adx + ady * ady;
    let blift = bdx * bdx + bdy * bdy;
    let clift = cdx * cdx + cdy * cdy;
    let bcdet = bdx * cdy - bdy * cdx;
    let cadet = cdx * ady - cdy * adx;
    let abdet = adx * bdy - ady * bdx;
    let approximate_det = alift * bcdet + blift * cadet + clift * abdet;
    let permanent = alift * ((bdx * cdy).abs() + (bdy * cdx).abs())
        + blift * ((cdx * ady).abs() + (cdy * adx).abs())
        + clift * ((adx * bdy).abs() + (ady * bdx).abs());
    let error = f64::EPSILON * 32.0 * permanent;
    if approximate_det.is_finite() && approximate_det.abs() > error {
        return if approximate_det < 0.0 {
            Sign::Negative
        } else {
            Sign::Positive
        };
    }
    let rows = [a, b, c]
        .map(|point| {
            let values = [difference(point[0], d[0]), difference(point[1], d[1])];
            vec![values[0].clone(), values[1].clone(), lift(&values)]
        })
        .to_vec();
    result(&determinant(&rows))
}

pub fn orient3d(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]) -> Sign {
    let approximate = [a, b, c]
        .map(|point| {
            point
                .into_iter()
                .zip(d)
                .map(|(value, origin)| value - origin)
                .collect::<Vec<_>>()
        })
        .to_vec();
    if let Some(sign) = filtered(&approximate) {
        return sign;
    }
    let rows = [a, b, c]
        .map(|point| {
            point
                .into_iter()
                .zip(d)
                .map(|(value, origin)| difference(value, origin))
                .collect::<Vec<_>>()
        })
        .to_vec();
    result(&determinant(&rows))
}

pub fn insphere(a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3], e: [f64; 3]) -> Sign {
    let approximate = [a, b, c, d]
        .map(|point| {
            let values = point
                .into_iter()
                .zip(e)
                .map(|(value, origin)| value - origin)
                .collect::<Vec<_>>();
            vec![
                values[0],
                values[1],
                values[2],
                values.iter().map(|value| value * value).sum(),
            ]
        })
        .to_vec();
    if let Some(sign) = filtered(&approximate) {
        return sign;
    }
    let rows = [a, b, c, d]
        .map(|point| {
            let values = point
                .into_iter()
                .zip(e)
                .map(|(value, origin)| difference(value, origin))
                .collect::<Vec<_>>();
            vec![
                values[0].clone(),
                values[1].clone(),
                values[2].clone(),
                lift(&values),
            ]
        })
        .to_vec();
    result(&determinant(&rows))
}

/// Deterministic symbolic ordering for an exact-zero predicate. This does not
/// alter coordinates; it only chooses one of otherwise equivalent topologies.
pub(crate) fn tie(ids: &[usize]) -> Ordering {
    let inversions = ids
        .iter()
        .enumerate()
        .map(|(index, id)| ids[index + 1..].iter().filter(|other| *other < id).count())
        .sum::<usize>();
    if inversions % 2 == 0 {
        Ordering::Greater
    } else {
        Ordering::Less
    }
}
