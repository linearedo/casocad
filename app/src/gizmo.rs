//! Transform gizmo for the Move and Rotate tools: screen-space overlay
//! handles (translate arrows / rotate rings) painted over the viewport,
//! with the drag math kept as pure functions.

use caso_kernel::sdf::node::RotationAxis;
use caso_kernel::vec3::{vec3, Vec3};
use caso_render::OrbitCamera;
use eframe::egui;

use crate::tools::project_to_screen;

/// Target on-screen radius of the gizmo, in egui points.
const GIZMO_SCREEN_RADIUS: f32 = 90.0;
/// Handle pick radius in egui points.
const HANDLE_PICK_RADIUS: f32 = 8.0;
const RING_SEGMENTS: usize = 48;

const AXES: [(RotationAxis, Vec3); 3] = [
    (RotationAxis::X, vec3(1.0, 0.0, 0.0)),
    (RotationAxis::Y, vec3(0.0, 1.0, 0.0)),
    (RotationAxis::Z, vec3(0.0, 0.0, 1.0)),
];

pub fn axis_direction(axis: RotationAxis) -> Vec3 {
    match axis {
        RotationAxis::X => vec3(1.0, 0.0, 0.0),
        RotationAxis::Y => vec3(0.0, 1.0, 0.0),
        RotationAxis::Z => vec3(0.0, 0.0, 1.0),
    }
}

fn axis_color(axis: RotationAxis, emphasized: bool) -> egui::Color32 {
    // Match the viewport's grid-axis colors (X red, Y green, Z blue).
    let base = match axis {
        RotationAxis::X => egui::Color32::from_rgb(232, 84, 84),
        RotationAxis::Y => egui::Color32::from_rgb(84, 208, 96),
        RotationAxis::Z => egui::Color32::from_rgb(84, 140, 255),
    };
    if emphasized {
        egui::Color32::from_rgb(
            (base.r() as u16 + 60).min(255) as u8,
            (base.g() as u16 + 60).min(255) as u8,
            (base.b() as u16 + 60).min(255) as u8,
        )
    } else {
        base
    }
}

/// The two in-plane basis vectors of the ring around `axis`.
fn ring_basis(axis: RotationAxis) -> (Vec3, Vec3) {
    match axis {
        RotationAxis::X => (vec3(0.0, 1.0, 0.0), vec3(0.0, 0.0, 1.0)),
        RotationAxis::Y => (vec3(0.0, 0.0, 1.0), vec3(1.0, 0.0, 0.0)),
        RotationAxis::Z => (vec3(1.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0)),
    }
}

/// Distance from a point to a screen-space segment (egui points).
pub fn point_segment_distance(point: egui::Pos2, a: egui::Pos2, b: egui::Pos2) -> f32 {
    let segment = b - a;
    let length_squared = segment.length_sq();
    if length_squared <= f32::EPSILON {
        return (point - a).length();
    }
    let t = ((point - a).dot(segment) / length_squared).clamp(0.0, 1.0);
    (point - (a + segment * t)).length()
}

/// World-space gizmo radius that projects to `GIZMO_SCREEN_RADIUS` points —
/// the exact inverse of `tools::project_to_screen`.
pub fn gizmo_world_radius(
    camera: &OrbitCamera,
    pivot: Vec3,
    rect: egui::Rect,
    pixels_per_point: f32,
) -> Option<f64> {
    let basis = camera.basis();
    let depth = (pivot - basis.position).dot(basis.forward);
    if depth <= 1.0e-9 {
        return None;
    }
    let height = (rect.height() as f64) * pixels_per_point as f64;
    let pixels = (GIZMO_SCREEN_RADIUS * pixels_per_point) as f64;
    Some(pixels * 2.0 * depth / (height.max(1.0) * camera.focal))
}

/// Parameter along the axis line `pivot + t·axis` closest to the mouse ray;
/// None when the ray runs nearly parallel to the axis.
pub fn axis_drag_parameter(
    ray_origin: Vec3,
    ray_direction: Vec3,
    pivot: Vec3,
    axis: Vec3,
) -> Option<f64> {
    let alignment = ray_direction.dot(axis);
    let denominator = 1.0 - alignment * alignment;
    if denominator.abs() < 1.0e-6 {
        return None;
    }
    let offset = ray_origin - pivot;
    let t = (offset.dot(axis) - alignment * offset.dot(ray_direction)) / denominator;
    Some(t)
}

/// Angle (radians) of the mouse ray's hit point in the ring plane around
/// `axis` at `pivot`; None when the ray runs nearly parallel to the plane.
pub fn ring_drag_angle(
    ray_origin: Vec3,
    ray_direction: Vec3,
    pivot: Vec3,
    axis: RotationAxis,
) -> Option<f64> {
    let normal = axis_direction(axis);
    let steepness = ray_direction.dot(normal);
    if steepness.abs() < 1.0e-6 {
        return None;
    }
    let t = (pivot - ray_origin).dot(normal) / steepness;
    if t <= 0.0 {
        return None;
    }
    let hit = ray_origin + ray_direction * t;
    let (u, v) = ring_basis(axis);
    let radial = hit - pivot;
    Some(radial.dot(v).atan2(radial.dot(u)))
}

/// Smallest signed angle difference (radians), wrap-normalized to ±π.
pub fn wrap_angle(delta: f64) -> f64 {
    let mut wrapped = delta % std::f64::consts::TAU;
    if wrapped > std::f64::consts::PI {
        wrapped -= std::f64::consts::TAU;
    } else if wrapped < -std::f64::consts::PI {
        wrapped += std::f64::consts::TAU;
    }
    wrapped
}

/// Screen-projected polyline of the ring around `axis` (None points culled).
fn ring_points(
    camera: &OrbitCamera,
    pivot: Vec3,
    radius: f64,
    axis: RotationAxis,
    rect: egui::Rect,
    pixels_per_point: f32,
) -> Vec<egui::Pos2> {
    let (u, v) = ring_basis(axis);
    let mut points = Vec::with_capacity(RING_SEGMENTS + 1);
    for step in 0..=RING_SEGMENTS {
        let angle = step as f64 / RING_SEGMENTS as f64 * std::f64::consts::TAU;
        let world = pivot + (u * angle.cos() + v * angle.sin()) * radius;
        if let Some(pos) = project_to_screen(camera, world, rect, pixels_per_point) {
            points.push(pos);
        }
    }
    points
}

/// The arrow (Move) or ring (Rotate) handle under the cursor, nearest first.
pub fn hit_test(
    kind: GizmoKind,
    cursor: egui::Pos2,
    camera: &OrbitCamera,
    pivot: Vec3,
    rect: egui::Rect,
    pixels_per_point: f32,
) -> Option<RotationAxis> {
    let radius = gizmo_world_radius(camera, pivot, rect, pixels_per_point)?;
    let mut nearest: Option<(f32, RotationAxis)> = None;
    let mut consider = |distance: f32, axis: RotationAxis| {
        if distance <= HANDLE_PICK_RADIUS && nearest.is_none_or(|(best, _)| distance < best) {
            nearest = Some((distance, axis));
        }
    };
    match kind {
        GizmoKind::Move => {
            let anchor = project_to_screen(camera, pivot, rect, pixels_per_point)?;
            for (axis, direction) in AXES {
                let tip_world = pivot + direction * radius;
                if let Some(tip) = project_to_screen(camera, tip_world, rect, pixels_per_point) {
                    consider(point_segment_distance(cursor, anchor, tip), axis);
                }
            }
        }
        GizmoKind::Rotate => {
            for (axis, _) in AXES {
                let points = ring_points(camera, pivot, radius, axis, rect, pixels_per_point);
                for pair in points.windows(2) {
                    consider(point_segment_distance(cursor, pair[0], pair[1]), axis);
                }
            }
        }
    }
    nearest.map(|(_, axis)| axis)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GizmoKind {
    Move,
    Rotate,
}

/// Paint the gizmo; `emphasized` (hovered or dragged) handles draw thicker.
pub fn paint(
    kind: GizmoKind,
    painter: &egui::Painter,
    camera: &OrbitCamera,
    pivot: Vec3,
    rect: egui::Rect,
    pixels_per_point: f32,
    emphasized: Option<RotationAxis>,
) {
    let Some(radius) = gizmo_world_radius(camera, pivot, rect, pixels_per_point) else {
        return;
    };
    let Some(anchor) = project_to_screen(camera, pivot, rect, pixels_per_point) else {
        return;
    };
    match kind {
        GizmoKind::Move => {
            for (axis, direction) in AXES {
                let active = emphasized == Some(axis);
                let color = axis_color(axis, active);
                let tip_world = pivot + direction * radius;
                let Some(tip) = project_to_screen(camera, tip_world, rect, pixels_per_point) else {
                    continue;
                };
                let width = if active { 3.5 } else { 2.0 };
                painter.line_segment([anchor, tip], egui::Stroke::new(width, color));
                painter.circle_filled(tip, if active { 6.0 } else { 4.5 }, color);
            }
            painter.circle_filled(anchor, 3.0, egui::Color32::from_gray(230));
        }
        GizmoKind::Rotate => {
            for (axis, _) in AXES {
                let active = emphasized == Some(axis);
                let color = axis_color(axis, active);
                let width = if active { 3.5 } else { 2.0 };
                let points = ring_points(camera, pivot, radius, axis, rect, pixels_per_point);
                for pair in points.windows(2) {
                    painter.line_segment([pair[0], pair[1]], egui::Stroke::new(width, color));
                }
            }
            painter.circle_filled(anchor, 3.0, egui::Color32::from_gray(230));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn axis_parameter_recovers_points_on_the_axis() {
        // Ray from above looking straight down at (2, 0, 0) on the X axis.
        let origin = vec3(2.0, 0.0, 5.0);
        let direction = vec3(0.0, 0.0, -1.0);
        let t = axis_drag_parameter(origin, direction, Vec3::ZERO, vec3(1.0, 0.0, 0.0))
            .expect("perpendicular ray");
        assert!((t - 2.0).abs() < 1e-12);
        // Ray parallel to the axis has no stable parameter.
        assert!(
            axis_drag_parameter(origin, vec3(1.0, 0.0, 0.0), Vec3::ZERO, vec3(1.0, 0.0, 0.0))
                .is_none()
        );
    }

    #[test]
    fn ring_angle_tracks_the_hit_direction() {
        // Ray down onto the Z ring plane at (1, 1, 0) → 45 degrees.
        let origin = vec3(1.0, 1.0, 5.0);
        let direction = vec3(0.0, 0.0, -1.0);
        let angle =
            ring_drag_angle(origin, direction, Vec3::ZERO, RotationAxis::Z).expect("plane hit");
        assert!((angle - std::f64::consts::FRAC_PI_4).abs() < 1e-12);
        // Grazing ray is rejected.
        assert!(
            ring_drag_angle(origin, vec3(1.0, 0.0, 0.0), Vec3::ZERO, RotationAxis::Z).is_none()
        );
    }

    #[test]
    fn wrap_angle_normalizes_across_the_seam() {
        let almost_tau = std::f64::consts::TAU - 0.1;
        assert!((wrap_angle(almost_tau) - (-0.1)).abs() < 1e-12);
        assert!((wrap_angle(-almost_tau) - 0.1).abs() < 1e-12);
        assert!((wrap_angle(0.25) - 0.25).abs() < 1e-12);
    }

    #[test]
    fn gizmo_radius_projects_to_the_target_screen_size() {
        let camera = OrbitCamera::default();
        let rect = egui::Rect::from_min_size(egui::pos2(0.0, 0.0), egui::vec2(800.0, 600.0));
        let pivot = camera.target;
        let radius = gizmo_world_radius(&camera, pivot, rect, 1.0).expect("radius");
        // A point radius away in the camera's right direction should land
        // GIZMO_SCREEN_RADIUS points from the projected pivot.
        let basis = camera.basis();
        let anchor = project_to_screen(&camera, pivot, rect, 1.0).expect("anchor");
        let tip = project_to_screen(&camera, pivot + basis.right * radius, rect, 1.0).expect("tip");
        let spread = (tip - anchor).length();
        assert!(
            (spread - GIZMO_SCREEN_RADIUS).abs() < 1.0,
            "spread = {spread}"
        );
    }
}
