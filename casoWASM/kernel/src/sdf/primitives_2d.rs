//! 2D SDF profiles, ported one-to-one from `core/sdf/primitives_2d.py`.
//! Evaluation is pointwise over local workplane coordinates (u, v); the numpy
//! masks in the original are vectorized branching over the same scalar math.

use crate::error::{GeometryError, GeometryResult};
use crate::sdf::primitives_1d::BooleanOp1D;
use crate::sdf::primitives_3d::py_sign;

pub type Point2 = [f64; 2];

fn segment_distance(u: f64, v: f64, first: Point2, second: Point2) -> f64 {
    let [ax, ay] = first;
    let [bx, by] = second;
    let bax = bx - ax;
    let bay = by - ay;
    let denominator = bax * bax + bay * bay;
    if denominator <= 1e-24 {
        return ((u - ax).powi(2) + (v - ay).powi(2)).sqrt();
    }
    let h = (((u - ax) * bax + (v - ay) * bay) / denominator).clamp(0.0, 1.0);
    let dx = u - ax - h * bax;
    let dy = v - ay - h * bay;
    (dx * dx + dy * dy).sqrt()
}

/// Exact distance to one quadratic Bezier span (closed-form cubic solve).
fn quadratic_bezier_distance(u: f64, v: f64, start: Point2, control: Point2, end: Point2) -> f64 {
    let [ax, ay] = start;
    let [bx, by] = control;
    let [cx, cy] = end;
    let a_x = bx - ax;
    let a_y = by - ay;
    let b_x = ax - 2.0 * bx + cx;
    let b_y = ay - 2.0 * by + cy;
    let c_x = 2.0 * a_x;
    let c_y = 2.0 * a_y;
    let b_dot_b = b_x * b_x + b_y * b_y;
    if b_dot_b <= 1.0e-24 {
        return segment_distance(u, v, start, end);
    }
    let d_x = ax - u;
    let d_y = ay - v;
    let kk = 1.0 / b_dot_b;
    let kx = kk * (a_x * b_x + a_y * b_y);
    let ky = kk * (2.0 * (a_x * a_x + a_y * a_y) + d_x * b_x + d_y * b_y) / 3.0;
    let kz = kk * (d_x * a_x + d_y * a_y);
    let p = ky - kx * kx;
    let q = kx * (2.0 * kx * kx - 3.0 * ky) + kz;
    let h = q * q + 4.0 * p * p * p;
    let result = if h >= 0.0 {
        let h_root = h.max(0.0).sqrt();
        let x0 = 0.5 * (h_root - q);
        let x1 = 0.5 * (-h_root - q);
        let t = (x0.cbrt() + x1.cbrt() - kx).clamp(0.0, 1.0);
        let w_x = d_x + (c_x + b_x * t) * t;
        let w_y = d_y + (c_y + b_y * t) * t;
        w_x * w_x + w_y * w_y
    } else {
        let z = (-p).max(0.0).sqrt();
        let denominator = 2.0 * p * z;
        let angle_argument = if denominator.abs() > 1.0e-24 {
            q / denominator
        } else {
            0.0
        };
        let angle = angle_argument.clamp(-1.0, 1.0).acos() / 3.0;
        let m = angle.cos();
        let n = angle.sin() * 1.732050808;
        let t0 = ((m + m) * z - kx).clamp(0.0, 1.0);
        let t1 = ((-n - m) * z - kx).clamp(0.0, 1.0);
        let w0_x = d_x + (c_x + b_x * t0) * t0;
        let w0_y = d_y + (c_y + b_y * t0) * t0;
        let w1_x = d_x + (c_x + b_x * t1) * t1;
        let w1_y = d_y + (c_y + b_y * t1) * t1;
        (w0_x * w0_x + w0_y * w0_y).min(w1_x * w1_x + w1_y * w1_y)
    };
    result.max(0.0).sqrt()
}

/// Parity of horizontal-ray crossings against one quadratic Bezier span.
fn quadratic_bezier_ray_crossings(
    u: f64,
    v: f64,
    start: Point2,
    control: Point2,
    end: Point2,
) -> bool {
    let [ax, ay] = start;
    let [bx, by] = control;
    let [cx, cy] = end;
    let qa = ay - 2.0 * by + cy;
    let qb = 2.0 * (by - ay);
    let qc = ay - v;
    let x_at = |t: f64| (1.0 - t) * (1.0 - t) * ax + 2.0 * (1.0 - t) * t * bx + t * t * cx;
    let mut crossings = false;
    if qa.abs() <= 1.0e-12 {
        if qb.abs() <= 1.0e-12 {
            return false;
        }
        let t = -qc / qb;
        if (0.0..1.0).contains(&t) && x_at(t) > u {
            crossings = !crossings;
        }
        return crossings;
    }
    let discriminant = qb * qb - 4.0 * qa * qc;
    if discriminant > 1.0e-12 {
        let root = discriminant.max(0.0).sqrt();
        let t0 = (-qb - root) / (2.0 * qa);
        let t1 = (-qb + root) / (2.0 * qa);
        if (0.0..1.0).contains(&t0) && x_at(t0) > u {
            crossings = !crossings;
        }
        if (0.0..1.0).contains(&t1) && x_at(t1) > u {
            crossings = !crossings;
        }
    }
    crossings
}

/// Parity of horizontal-ray crossings against one straight segment.
fn segment_ray_crossing(u: f64, v: f64, first: Point2, second: Point2) -> bool {
    let [ax, ay] = first;
    let [bx, by] = second;
    let active = (ay > v) != (by > v);
    if !active {
        return false;
    }
    let intersection = (bx - ax) * (v - ay) / (by - ay) + ax;
    u < intersection
}

fn ellipse_distance(u: f64, v: f64, center: Point2, semi_axes: Point2) -> f64 {
    let [cu, cv] = center;
    let [au, av] = semi_axes;
    if (au - av).abs() <= 1.0e-12 {
        return ((u - cu).powi(2) + (v - cv).powi(2)).sqrt() - au;
    }
    let x = (u - cu).abs();
    let y = (v - cv).abs();
    let (px, py, ax, ay) = if x > y { (y, x, av, au) } else { (x, y, au, av) };
    let length_delta = ay * ay - ax * ax;
    let m = ax * px / length_delta;
    let n = ay * py / length_delta;
    let m2 = m * m;
    let n2 = n * n;
    let c = (m2 + n2 - 1.0) / 3.0;
    let c3 = c * c * c;
    let q = c3 + m2 * n2 * 2.0;
    let d = c3 + m2 * n2;
    let g = m + m * n2;
    let co = if d < 0.0 {
        let h = (q / c3).clamp(-1.0, 1.0).acos() / 3.0;
        let s = h.cos();
        let t = h.sin() * 3.0_f64.sqrt();
        let rx = (-c * (s + t + 2.0) + m2).max(0.0).sqrt();
        let ry = (-c * (s - t + 2.0) + m2).max(0.0).sqrt();
        let denominator = (rx * ry).max(1.0e-24);
        (ry + py_sign(length_delta) * rx + g.abs() / denominator - m) * 0.5
    } else {
        let h = 2.0 * m * n * d.max(0.0).sqrt();
        let s = (q + h).cbrt();
        let uu = (q - h).cbrt();
        let rx = -s - uu - c * 4.0 + 2.0 * m2;
        let ry = (s - uu) * 3.0_f64.sqrt();
        let rm = (rx * rx + ry * ry).sqrt();
        (ry / (rm - rx).max(1.0e-24).sqrt() + 2.0 * g / rm.max(1.0e-24) - m) * 0.5
    };
    let co = co.clamp(0.0, 1.0);
    let closest_x = ax * co;
    let closest_y = ay * (1.0 - co * co).max(0.0).sqrt();
    let distance = ((closest_x - px).powi(2) + (closest_y - py).powi(2)).sqrt();
    distance * py_sign(py - closest_y)
}

fn quadratic_bezier_spans(points: &[Point2]) -> impl Iterator<Item = (Point2, Point2, Point2)> + '_ {
    (0..points.len().saturating_sub(2))
        .step_by(2)
        .map(|index| (points[index], points[index + 1], points[index + 2]))
}

fn quadratic_bezier_surface_closed(points: &[Point2]) -> bool {
    let first = points[0];
    let last = points[points.len() - 1];
    let dx = first[0] - last[0];
    let dy = first[1] - last[1];
    (dx * dx + dy * dy).sqrt() <= 1.0e-12
}

fn points_bounds_2d(points: &[Point2]) -> (f64, f64, f64, f64) {
    let mut u_min = f64::INFINITY;
    let mut u_max = f64::NEG_INFINITY;
    let mut v_min = f64::INFINITY;
    let mut v_max = f64::NEG_INFINITY;
    for [u, v] in points {
        u_min = u_min.min(*u);
        u_max = u_max.max(*u);
        v_min = v_min.min(*v);
        v_max = v_max.max(*v);
    }
    (u_min, u_max, v_min, v_max)
}

fn validate_odd_bezier_points(points: &[Point2], what: &str) -> GeometryResult<()> {
    if points.len() < 3 {
        return Err(GeometryError::new(format!(
            "quadratic Bezier {what} requires at least three points"
        )));
    }
    if points.len().is_multiple_of(2) {
        return Err(GeometryError::new(format!(
            "quadratic Bezier {what} requires an odd point count: anchor, control, anchor"
        )));
    }
    let degenerate = quadratic_bezier_spans(points).all(|(start, control, end)| {
        let cd = ((control[0] - start[0]).powi(2) + (control[1] - start[1]).powi(2)).sqrt();
        let ed = ((end[0] - start[0]).powi(2) + (end[1] - start[1]).powi(2)).sqrt();
        cd <= 1e-12 && ed <= 1e-12
    });
    if degenerate {
        return Err(GeometryError::new(format!(
            "quadratic Bezier {what} requires at least one nonzero span"
        )));
    }
    Ok(())
}

/// A local filled-region signed distance over workplane coordinates (u, v).
#[derive(Debug, Clone, PartialEq)]
pub enum Profile2D {
    Polyline {
        points: Vec<Point2>,
    },
    /// Open Bezier polycurve; `kind()` splits on span count like Python.
    QuadraticBezierCurve {
        points: Vec<Point2>,
    },
    QuadraticBezierSurface {
        points: Vec<Point2>,
    },
    Polygon {
        points: Vec<Point2>,
    },
    Circle {
        center: Point2,
        radius: f64,
    },
    Rectangle {
        center: Point2,
        half_size: Point2,
    },
    Square {
        center: Point2,
        half_size: f64,
    },
    RoundedRectangle {
        center: Point2,
        half_size: Point2,
        corner_radius: f64,
    },
    Ellipse {
        center: Point2,
        semi_axes: Point2,
    },
    RegularPolygon {
        center: Point2,
        radius: f64,
        side_count: u32,
        rotation: f64,
    },
    Offset {
        child: Box<Profile2D>,
        offset: Point2,
    },
    DistanceOffset {
        child: Box<Profile2D>,
        offset: f64,
    },
    Binary {
        left: Box<Profile2D>,
        right: Box<Profile2D>,
        operation: BooleanOp1D,
        /// Legacy field: round-tripped in scene files, never used by eval.
        smoothing: f64,
    },
}

impl Profile2D {
    pub fn polyline(points: Vec<Point2>) -> GeometryResult<Self> {
        if points.len() < 2 {
            return Err(GeometryError::new("polyline requires at least two points"));
        }
        let degenerate = points.windows(2).all(|pair| {
            let dx = pair[1][0] - pair[0][0];
            let dy = pair[1][1] - pair[0][1];
            (dx * dx + dy * dy).sqrt() <= 1e-12
        });
        if degenerate {
            return Err(GeometryError::new(
                "polyline requires at least one nonzero segment",
            ));
        }
        Ok(Self::Polyline { points })
    }

    pub fn quadratic_bezier_curve(points: Vec<Point2>) -> GeometryResult<Self> {
        validate_odd_bezier_points(&points, "curve")?;
        Ok(Self::QuadraticBezierCurve { points })
    }

    pub fn quadratic_bezier_surface(points: Vec<Point2>) -> GeometryResult<Self> {
        validate_odd_bezier_points(&points, "surface")?;
        Ok(Self::QuadraticBezierSurface { points })
    }

    pub fn polygon(mut points: Vec<Point2>) -> GeometryResult<Self> {
        if points.len() >= 2 && points[0] == points[points.len() - 1] {
            points.pop();
        }
        if points.len() < 3 {
            return Err(GeometryError::new("polygon requires at least three points"));
        }
        Ok(Self::Polygon { points })
    }

    pub fn circle(center: Point2, radius: f64) -> GeometryResult<Self> {
        if radius <= 0.0 {
            return Err(GeometryError::new("circle radius must be positive"));
        }
        Ok(Self::Circle { center, radius })
    }

    pub fn rectangle(center: Point2, half_size: Point2) -> GeometryResult<Self> {
        if half_size[0] <= 0.0 || half_size[1] <= 0.0 {
            return Err(GeometryError::new("rectangle half sizes must be positive"));
        }
        Ok(Self::Rectangle { center, half_size })
    }

    pub fn square(center: Point2, half_size: f64) -> GeometryResult<Self> {
        if half_size <= 0.0 {
            return Err(GeometryError::new("square half size must be positive"));
        }
        Ok(Self::Square { center, half_size })
    }

    pub fn rounded_rectangle(
        center: Point2,
        half_size: Point2,
        corner_radius: f64,
    ) -> GeometryResult<Self> {
        if half_size[0] <= 0.0 || half_size[1] <= 0.0 {
            return Err(GeometryError::new("rectangle half sizes must be positive"));
        }
        if corner_radius <= 0.0 {
            return Err(GeometryError::new("corner radius must be positive"));
        }
        if corner_radius > half_size[0].min(half_size[1]) {
            return Err(GeometryError::new("corner radius exceeds rectangle half size"));
        }
        Ok(Self::RoundedRectangle {
            center,
            half_size,
            corner_radius,
        })
    }

    pub fn ellipse(center: Point2, semi_axes: Point2) -> GeometryResult<Self> {
        if semi_axes[0] <= 0.0 || semi_axes[1] <= 0.0 {
            return Err(GeometryError::new("ellipse semi-axes must be positive"));
        }
        Ok(Self::Ellipse { center, semi_axes })
    }

    pub fn regular_polygon(
        center: Point2,
        radius: f64,
        side_count: u32,
        rotation: f64,
    ) -> GeometryResult<Self> {
        if radius <= 0.0 {
            return Err(GeometryError::new("polygon radius must be positive"));
        }
        if side_count < 3 {
            return Err(GeometryError::new("polygon requires at least three sides"));
        }
        Ok(Self::RegularPolygon {
            center,
            radius,
            side_count,
            rotation,
        })
    }

    pub fn distance_offset(child: Profile2D, offset: f64) -> GeometryResult<Self> {
        if !offset.is_finite() {
            return Err(GeometryError::new("distance offset must be finite"));
        }
        Ok(Self::DistanceOffset {
            child: Box::new(child),
            offset,
        })
    }

    /// Kind string matching the Python `Profile2D.kind` properties.
    pub fn kind(&self) -> &'static str {
        match self {
            Self::Polyline { .. } => "polyline",
            Self::QuadraticBezierCurve { points } => {
                if points.len() > 3 {
                    "quadratic_bezier_polycurve"
                } else {
                    "quadratic_bezier_curve"
                }
            }
            Self::QuadraticBezierSurface { .. } => "quadratic_bezier_surface",
            Self::Polygon { .. } => "polygon",
            Self::Circle { .. } => "circleprofile",
            Self::Rectangle { .. } => "rectangleprofile",
            Self::Square { .. } => "squareprofile",
            Self::RoundedRectangle { .. } => "roundedrectangleprofile",
            Self::Ellipse { .. } => "ellipseprofile",
            Self::RegularPolygon { .. } => "regularpolygonprofile",
            Self::Offset { .. } => "offsetprofile",
            Self::DistanceOffset { .. } => "distanceoffsetprofile",
            Self::Binary { .. } => "binaryprofile",
        }
    }

    pub fn eval(&self, u: f64, v: f64) -> f64 {
        match self {
            Self::Polyline { points } => points
                .windows(2)
                .map(|pair| segment_distance(u, v, pair[0], pair[1]))
                .fold(f64::INFINITY, f64::min),
            Self::QuadraticBezierCurve { points } => quadratic_bezier_spans(points)
                .map(|(start, control, end)| quadratic_bezier_distance(u, v, start, control, end))
                .fold(f64::INFINITY, f64::min),
            Self::QuadraticBezierSurface { points } => {
                let mut distance = quadratic_bezier_spans(points)
                    .map(|(start, control, end)| {
                        quadratic_bezier_distance(u, v, start, control, end)
                    })
                    .fold(f64::INFINITY, f64::min);
                let closed = quadratic_bezier_surface_closed(points);
                if !closed {
                    distance = distance.min(segment_distance(
                        u,
                        v,
                        points[points.len() - 1],
                        points[0],
                    ));
                }
                let mut inside = false;
                for (start, control, end) in quadratic_bezier_spans(points) {
                    inside ^= quadratic_bezier_ray_crossings(u, v, start, control, end);
                }
                if !closed {
                    inside ^= segment_ray_crossing(u, v, points[points.len() - 1], points[0]);
                }
                if inside {
                    -distance
                } else {
                    distance
                }
            }
            Self::Polygon { points } => {
                eval_polygon(u, v, points.len(), |index| points[index])
            }
            Self::Circle { center, radius } => {
                ((u - center[0]).powi(2) + (v - center[1]).powi(2)).sqrt() - radius
            }
            Self::Rectangle { center, half_size } => {
                rectangle_distance(u, v, *center, *half_size)
            }
            Self::Square { center, half_size } => {
                rectangle_distance(u, v, *center, [*half_size, *half_size])
            }
            Self::RoundedRectangle {
                center,
                half_size,
                corner_radius,
            } => {
                let inner = [half_size[0] - corner_radius, half_size[1] - corner_radius];
                rectangle_distance(u, v, *center, inner) - corner_radius
            }
            Self::Ellipse { center, semi_axes } => ellipse_distance(u, v, *center, *semi_axes),
            Self::RegularPolygon {
                center,
                radius,
                side_count,
                rotation,
            } => eval_polygon(u, v, *side_count as usize, |index| {
                regular_polygon_vertex(*center, *radius, *side_count, *rotation, index)
            }),
            Self::Offset { child, offset } => child.eval(u - offset[0], v - offset[1]),
            Self::DistanceOffset { child, offset } => child.eval(u, v) - offset,
            Self::Binary {
                left,
                right,
                operation,
                ..
            } => {
                let l = left.eval(u, v);
                let r = right.eval(u, v);
                match operation {
                    BooleanOp1D::Union => l.min(r),
                    BooleanOp1D::Intersection => l.max(r),
                    BooleanOp1D::Difference => l.max(-r),
                    BooleanOp1D::Xor => l.min(r).max(-l.max(r)),
                }
            }
        }
    }

    /// Finite local bounds: (u_min, u_max, v_min, v_max).
    pub fn bounds(&self) -> (f64, f64, f64, f64) {
        match self {
            Self::Polyline { points }
            | Self::QuadraticBezierCurve { points }
            | Self::QuadraticBezierSurface { points }
            | Self::Polygon { points } => points_bounds_2d(points),
            Self::Circle { center, radius } => (
                center[0] - radius,
                center[0] + radius,
                center[1] - radius,
                center[1] + radius,
            ),
            Self::Rectangle { center, half_size }
            | Self::RoundedRectangle {
                center, half_size, ..
            } => (
                center[0] - half_size[0],
                center[0] + half_size[0],
                center[1] - half_size[1],
                center[1] + half_size[1],
            ),
            Self::Square { center, half_size } => (
                center[0] - half_size,
                center[0] + half_size,
                center[1] - half_size,
                center[1] + half_size,
            ),
            Self::Ellipse { center, semi_axes } => (
                center[0] - semi_axes[0],
                center[0] + semi_axes[0],
                center[1] - semi_axes[1],
                center[1] + semi_axes[1],
            ),
            Self::RegularPolygon {
                center,
                radius,
                side_count,
                rotation,
            } => {
                let vertices: Vec<Point2> = (0..*side_count as usize)
                    .map(|index| {
                        regular_polygon_vertex(*center, *radius, *side_count, *rotation, index)
                    })
                    .collect();
                points_bounds_2d(&vertices)
            }
            Self::Offset { child, offset } => {
                let (u_min, u_max, v_min, v_max) = child.bounds();
                (
                    u_min + offset[0],
                    u_max + offset[0],
                    v_min + offset[1],
                    v_max + offset[1],
                )
            }
            Self::DistanceOffset { child, offset } => {
                let (u_min, u_max, v_min, v_max) = child.bounds();
                let padding = offset.abs();
                (
                    u_min - padding,
                    u_max + padding,
                    v_min - padding,
                    v_max + padding,
                )
            }
            Self::Binary {
                left,
                right,
                operation,
                ..
            } => {
                let l = left.bounds();
                if *operation == BooleanOp1D::Difference {
                    return l;
                }
                let r = right.bounds();
                (
                    l.0.min(r.0),
                    l.1.max(r.1),
                    l.2.min(r.2),
                    l.3.max(r.3),
                )
            }
        }
    }
}

fn rectangle_distance(u: f64, v: f64, center: Point2, half_size: Point2) -> f64 {
    let qu = (u - center[0]).abs() - half_size[0];
    let qv = (v - center[1]).abs() - half_size[1];
    let outside = (qu.max(0.0).powi(2) + qv.max(0.0).powi(2)).sqrt();
    let inside = qu.max(qv).min(0.0);
    outside + inside
}

fn regular_polygon_vertex(
    center: Point2,
    radius: f64,
    side_count: u32,
    rotation: f64,
    index: usize,
) -> Point2 {
    let angle = rotation + index as f64 * 2.0 * std::f64::consts::PI / side_count as f64;
    [
        center[0] + radius * angle.cos(),
        center[1] + radius * angle.sin(),
    ]
}

/// Filled polygon: min distance to the closed outline, sign from ray parity.
fn eval_polygon(u: f64, v: f64, count: usize, vertex: impl Fn(usize) -> Point2) -> f64 {
    let mut distance = f64::INFINITY;
    let mut inside = false;
    for index in 0..count {
        let first = vertex(index);
        let second = vertex((index + 1) % count);
        distance = distance.min(segment_distance(u, v, first, second));
        inside ^= segment_ray_crossing(u, v, first, second);
    }
    if inside {
        -distance
    } else {
        distance
    }
}
