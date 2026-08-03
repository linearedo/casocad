use crate::{continued, tetrahedralize_3d, Error, Input3d, Refine3d, Tetrahedralization3d};

pub fn refine_3d(
    input: &Input3d,
    options: Refine3d,
    mut check: impl FnMut() -> bool,
) -> Result<Tetrahedralization3d, Error> {
    if !options.max_volume.is_finite() || options.max_volume <= 0.0 {
        return Err(Error::DegenerateInput("invalid 3D refinement target"));
    }
    let mut working = input.clone();
    let mut result = tetrahedralize_3d(&working, options.limits, &mut check)?;
    for iteration in 0..options.limits.max_iterations {
        if iteration.is_multiple_of(32) {
            continued(&mut check)?;
        }
        let bad = result
            .tetrahedra
            .iter()
            .copied()
            .find(|tetrahedron| volume(*tetrahedron, &result.vertices) > options.max_volume);
        let Some(tetrahedron) = bad else {
            return Ok(result);
        };
        if result.vertices.len() == options.limits.max_vertices {
            return Err(Error::LimitExceeded("vertex"));
        }
        let points = tetrahedron.map(|vertex| result.vertices[vertex]);
        let center = [
            points.iter().map(|point| point[0]).sum::<f64>() * 0.25,
            points.iter().map(|point| point[1]).sum::<f64>() * 0.25,
            points.iter().map(|point| point[2]).sum::<f64>() * 0.25,
        ];
        if result.vertices.contains(&center) {
            return Ok(result);
        }
        working.points = result.vertices;
        working.points.push(center);
        result = tetrahedralize_3d(&working, options.limits, &mut check)?;
    }
    Ok(result)
}

fn volume(tetrahedron: [usize; 4], points: &[[f64; 3]]) -> f64 {
    let [a, b, c, d] = tetrahedron.map(|vertex| points[vertex]);
    let ab = subtract(b, a);
    let ac = subtract(c, a);
    let ad = subtract(d, a);
    dot(ab, cross(ac, ad)).abs() / 6.0
}

fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}
