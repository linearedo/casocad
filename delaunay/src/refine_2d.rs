use crate::{continued, triangulate_2d, Error, Input2d, Refine2d, Triangulation2d};

pub fn refine_2d(
    input: &Input2d,
    options: Refine2d,
    mut check: impl FnMut() -> bool,
) -> Result<Triangulation2d, Error> {
    if !options.max_area.is_finite()
        || options.max_area <= 0.0
        || !options.min_angle_degrees.is_finite()
        || !(0.0..60.0).contains(&options.min_angle_degrees)
    {
        return Err(Error::DegenerateInput("invalid 2D refinement target"));
    }
    let mut working = input.clone();
    let mut result = triangulate_2d(&working, options.limits, &mut check)?;
    for iteration in 0..options.limits.max_iterations {
        if iteration.is_multiple_of(64) {
            continued(&mut check)?;
        }
        let bad = result.triangles.iter().copied().find(|triangle| {
            triangle_area(*triangle, &result.vertices) > options.max_area
                || minimum_angle(*triangle, &result.vertices) < options.min_angle_degrees
        });
        let Some(triangle) = bad else {
            return Ok(result);
        };
        if result.vertices.len() == options.limits.max_vertices {
            return Err(Error::LimitExceeded("vertex"));
        }
        let center = triangle.map(|vertex| result.vertices[vertex]);
        let center = [
            (center[0][0] + center[1][0] + center[2][0]) / 3.0,
            (center[0][1] + center[1][1] + center[2][1]) / 3.0,
        ];
        if result.vertices.contains(&center) {
            return Ok(result);
        }
        working.points = result.vertices;
        working.points.push(center);
        result = triangulate_2d(&working, options.limits, &mut check)?;
    }
    Ok(result)
}

fn triangle_area(triangle: [usize; 3], points: &[[f64; 2]]) -> f64 {
    let [a, b, c] = triangle.map(|vertex| points[vertex]);
    ((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])).abs() * 0.5
}

fn minimum_angle(triangle: [usize; 3], points: &[[f64; 2]]) -> f64 {
    let p = triangle.map(|vertex| points[vertex]);
    let lengths = [
        distance(p[1], p[2]),
        distance(p[2], p[0]),
        distance(p[0], p[1]),
    ];
    (0..3)
        .map(|index| {
            let a = lengths[(index + 1) % 3];
            let b = lengths[(index + 2) % 3];
            let opposite = lengths[index];
            ((a * a + b * b - opposite * opposite) / (2.0 * a * b))
                .clamp(-1.0, 1.0)
                .acos()
                .to_degrees()
        })
        .fold(f64::INFINITY, f64::min)
}

fn distance(a: [f64; 2], b: [f64; 2]) -> f64 {
    ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2)).sqrt()
}
