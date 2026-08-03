//! Small expansion-arithmetic kernel following the error-free transforms
//! described by Shewchuk.  Operations normalize their result so the sign of
//! the last component is the exact sign of the represented real value.

const SPLITTER: f64 = 134_217_729.0;

pub(crate) type Expansion = Vec<f64>;

fn two_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let b_virtual = x - a;
    let a_virtual = x - b_virtual;
    let b_roundoff = b - b_virtual;
    let a_roundoff = a - a_virtual;
    (a_roundoff + b_roundoff, x)
}

fn two_diff(a: f64, b: f64) -> (f64, f64) {
    let x = a - b;
    let b_virtual = a - x;
    let a_virtual = x + b_virtual;
    let b_roundoff = b_virtual - b;
    let a_roundoff = a - a_virtual;
    (a_roundoff + b_roundoff, x)
}

fn split(value: f64) -> (f64, f64) {
    let c = SPLITTER * value;
    let high = c - (c - value);
    (high, value - high)
}

fn two_product(a: f64, b: f64) -> (f64, f64) {
    let x = a * b;
    let (a_high, a_low) = split(a);
    let (b_high, b_low) = split(b);
    let error = a_low * b_low - (((x - a_high * b_high) - a_low * b_high) - a_high * b_low);
    (error, x)
}

fn grow(expansion: &[f64], value: f64) -> Expansion {
    let mut result = Vec::with_capacity(expansion.len() + 1);
    let mut sum = value;
    for &component in expansion {
        let (roundoff, next) = two_sum(sum, component);
        if roundoff != 0.0 {
            result.push(roundoff);
        }
        sum = next;
    }
    if sum != 0.0 || result.is_empty() {
        result.push(sum);
    }
    result
}

fn normalize(terms: impl IntoIterator<Item = f64>) -> Expansion {
    let mut terms = terms
        .into_iter()
        .filter(|term| *term != 0.0)
        .collect::<Vec<_>>();
    terms.sort_by(|a, b| a.abs().total_cmp(&b.abs()));
    let mut result = Vec::new();
    for term in terms {
        result = grow(&result, term);
    }
    if result.is_empty() {
        result.push(0.0);
    }
    result
}

pub(crate) fn scalar(value: f64) -> Expansion {
    vec![value]
}

pub(crate) fn difference(a: f64, b: f64) -> Expansion {
    let (low, high) = two_diff(a, b);
    normalize([low, high])
}

pub(crate) fn negate(value: &[f64]) -> Expansion {
    value.iter().map(|component| -*component).collect()
}

pub(crate) fn add(a: &[f64], b: &[f64]) -> Expansion {
    normalize(a.iter().chain(b).copied())
}

pub(crate) fn subtract(a: &[f64], b: &[f64]) -> Expansion {
    add(a, &negate(b))
}

pub(crate) fn multiply(a: &[f64], b: &[f64]) -> Expansion {
    let mut terms = Vec::with_capacity(a.len() * b.len() * 2);
    for &left in a {
        for &right in b {
            let (low, high) = two_product(left, right);
            terms.extend([low, high]);
        }
    }
    normalize(terms)
}

pub(crate) fn sign(value: &[f64]) -> std::cmp::Ordering {
    value
        .iter()
        .rev()
        .find(|component| **component != 0.0)
        .map_or(std::cmp::Ordering::Equal, |component| {
            component.total_cmp(&0.0)
        })
}

pub(crate) fn determinant(matrix: &[Vec<Expansion>]) -> Expansion {
    match matrix.len() {
        0 => scalar(1.0),
        1 => matrix[0][0].clone(),
        size => {
            let mut result = scalar(0.0);
            for column in 0..size {
                let minor = matrix[1..]
                    .iter()
                    .map(|row| {
                        row.iter()
                            .enumerate()
                            .filter_map(|(index, value)| (index != column).then_some(value.clone()))
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>();
                let term = multiply(&matrix[0][column], &determinant(&minor));
                result = if column % 2 == 0 {
                    add(&result, &term)
                } else {
                    subtract(&result, &term)
                };
            }
            result
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cancellation_keeps_the_low_component() {
        let a = add(&scalar(1.0e16), &scalar(1.0));
        let value = subtract(&a, &scalar(1.0e16));
        assert_eq!(sign(&value), std::cmp::Ordering::Greater);
    }
}
