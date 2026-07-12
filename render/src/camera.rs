//! Orbit camera for the viewport, ported from `app/viewport/camera.py` plus
//! the projection matrix from the QRhi surface renderer (adapted to wgpu's
//! y-up NDC and 0..1 clip depth). The CPU `screen_ray` reproduces the GPU
//! ray exactly — keeping them bit-identical is a design invariant.

use caso_kernel::bbox::BoundingBox3D;
use caso_kernel::vec3::{vec3, Vec3};

/// The familiar startup framing: 6 m away from a 1 m grid.
pub const DEFAULT_VIEW_DISTANCE: f64 = 6.0;
const WORLD_UP: Vec3 = vec3(0.0, 0.0, 1.0);
const FALLBACK_UP: Vec3 = vec3(0.0, 1.0, 0.0);

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    pub target: Vec3,
    pub distance: f64,
    pub yaw: f64,
    pub pitch: f64,
    pub focal: f64,
    /// Meters per working unit; zoom floors/ceilings scale with it.
    pub view_scale: f64,
}

impl Default for OrbitCamera {
    fn default() -> Self {
        Self {
            target: Vec3::ZERO,
            distance: DEFAULT_VIEW_DISTANCE,
            yaw: 35.0_f64.to_radians(),
            pitch: 28.0_f64.to_radians(),
            focal: 1.5,
            view_scale: 1.0,
        }
    }
}

pub struct CameraBasis {
    pub position: Vec3,
    pub forward: Vec3,
    pub right: Vec3,
    pub up: Vec3,
}

impl OrbitCamera {
    pub fn position(&self) -> Vec3 {
        let cos_pitch = self.pitch.cos();
        let offset = vec3(
            cos_pitch * self.yaw.cos(),
            cos_pitch * self.yaw.sin(),
            self.pitch.sin(),
        );
        self.target + offset * self.distance
    }

    /// (position, forward, right, up), matching the shader convention.
    pub fn basis(&self) -> CameraBasis {
        let position = self.position();
        let mut forward = self.target - position;
        forward = forward / forward.length().max(1.0e-9);
        let world_up = if forward.dot(WORLD_UP).abs() > 0.99 {
            FALLBACK_UP
        } else {
            WORLD_UP
        };
        let mut right = forward.cross(world_up);
        right = right / right.length().max(1.0e-9);
        let up = right.cross(forward);
        CameraBasis {
            position,
            forward,
            right,
            up,
        }
    }

    pub fn orbit(&mut self, delta_x: f64, delta_y: f64) {
        self.yaw -= delta_x * 0.01;
        self.pitch = (self.pitch + delta_y * 0.01).clamp(-1.5, 1.5);
    }

    /// Distance envelope, widened so working scale and meter scale stay
    /// reachable.
    pub fn zoom_limits(&self) -> (f64, f64) {
        (
            0.5 * self.view_scale.min(1.0),
            200.0 * self.view_scale.max(1.0),
        )
    }

    pub fn zoom_by(&mut self, wheel_delta: f64) {
        let (minimum, maximum) = self.zoom_limits();
        self.distance = (self.distance * (-wheel_delta * 0.0012).exp()).clamp(minimum, maximum);
    }

    pub fn fly_step(&self) -> f64 {
        (self.distance * 0.06).max(0.05 * self.view_scale)
    }

    /// Pan the target in the camera plane by a screen-pixel delta.
    pub fn pan(&mut self, delta_x: f64, delta_y: f64, viewport_height: f64) {
        let basis = self.basis();
        let world_per_pixel = 2.0 * self.distance / (self.focal * viewport_height.max(1.0));
        self.target =
            self.target - basis.right * (delta_x * world_per_pixel) + basis.up * (delta_y * world_per_pixel);
    }

    pub fn frame_target(&mut self, target: Vec3, distance: f64) {
        self.target = target;
        self.distance = distance;
    }

    pub fn frame_box(&mut self, bounds: &BoundingBox3D) {
        let center = bounds.center();
        let extent = (bounds.x_max - bounds.x_min)
            .max(bounds.y_max - bounds.y_min)
            .max(bounds.z_max - bounds.z_min)
            .max(1.0e-3);
        self.target = center;
        self.distance = (extent * 1.6).max(self.view_scale.min(1.0));
    }

    /// Reproduce the startup framing at the working scale, keeping the target.
    pub fn reframe_to_working_scale(&mut self) {
        self.distance = DEFAULT_VIEW_DISTANCE * self.view_scale;
    }

    /// (origin, direction) of the camera ray through screen (x, y), matching
    /// the grid shader exactly.
    pub fn screen_ray(&self, x: f64, y: f64, width: f64, height: f64) -> (Vec3, Vec3) {
        let basis = self.basis();
        let w = width.max(1.0);
        let h = height.max(1.0);
        let suvx = (x - 0.5 * w) / h;
        let suvy = -((y - 0.5 * h) / h);
        let mut direction =
            basis.right * (2.0 * suvx) + basis.up * (2.0 * suvy) + basis.forward * self.focal;
        direction = direction / direction.length().max(1.0e-9);
        (basis.position, direction)
    }

    /// Column-major projection*view matrix for wgpu (y-up NDC, 0..1 depth).
    pub fn matrix(&self, width: u32, height: u32) -> [f32; 16] {
        let basis = self.basis();
        let eye = basis.position;
        let distance = (eye - self.target).length().max(0.1);
        let near = (distance / 1000.0).max(0.001);
        let far = (distance * 100.0).max(100.0);
        let aspect_scale = height as f64 / (width as f64).max(1.0);
        let depth_scale = far / (far - near);
        let depth_bias = -(far * near) / (far - near);

        let row0 = basis.right * (self.focal * aspect_scale);
        let row1 = basis.up * self.focal;
        let row2 = basis.forward * depth_scale;
        let row3 = basis.forward;
        let t0 = -row0.dot(eye);
        let t1 = -row1.dot(eye);
        let t2 = depth_bias - depth_scale * basis.forward.dot(eye);
        let t3 = -basis.forward.dot(eye);
        // Rows above are the matrix rows; emit column-major for WGSL.
        [
            row0.x as f32, row1.x as f32, row2.x as f32, row3.x as f32,
            row0.y as f32, row1.y as f32, row2.y as f32, row3.y as f32,
            row0.z as f32, row1.z as f32, row2.z as f32, row3.z as f32,
            t0 as f32, t1 as f32, t2 as f32, t3 as f32,
        ]
    }
}
