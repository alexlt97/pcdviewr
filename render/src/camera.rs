//! Simple orbit camera for 3D point cloud viewing.
//!
//! Controls rotation and zoom via mouse input.
//! Uses nalgebra for all vector/matrix math.

use nalgebra::{Matrix4, Point3, Vector3};

/// Orbit camera state.
///
/// The camera orbits around a center point at a given distance,
/// controlled by spherical coordinates (yaw, pitch).
#[derive(Debug, Clone)]
pub struct Camera {
    /// Center point the camera orbits around
    pub center: Point3<f32>,
    /// Distance from center to camera
    pub distance: f32,
    /// Horizontal rotation angle (radians)
    pub yaw: f32,
    /// Vertical rotation angle (radians)
    pub pitch: f32,
}

impl Camera {
    /// Create a new camera looking at the origin from a default position.
    pub fn new() -> Self {
        Self {
            center: Point3::origin(),
            distance: 5.0,
            yaw: std::f32::consts::FRAC_PI_4,   // 45 degrees
            pitch: -std::f32::consts::FRAC_PI_6, // -30 degrees (looking slightly down)
        }
    }

    /// Create a camera positioned at the origin, looking toward the cloud center.
    pub fn from_bounds(min: [f32; 3], max: [f32; 3]) -> Self {
        let min_v = Vector3::from(min);
        let max_v = Vector3::from(max);
        let center = (min_v + max_v) / 2.0;
        let extent = max_v - min_v;
        let distance = extent.max() * 2.0;

        // Place camera eye at the origin, orbiting around the cloud center.
        // eye = center + distance * direction, so direction = -center / |center|
        let center_dist = center.norm();
        if center_dist < 0.001 {
            // Cloud center is near origin, fall back to default view
            Self {
                center: Point3::from(center),
                distance,
                yaw: std::f32::consts::FRAC_PI_4,
                pitch: -std::f32::consts::FRAC_PI_6,
            }
        } else {
            let dir = -center / center_dist;
            let yaw = dir.x.atan2(dir.z);
            let pitch = dir.y.asin();

            Self {
                center: Point3::from(center),
                distance: center_dist,
                yaw,
                pitch,
            }
        }
    }

    /// Update camera from mouse drag (orbit).
    pub fn orbit(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.005;
        self.yaw += dx * sensitivity;
        self.pitch -= dy * sensitivity;

        // Clamp pitch to avoid gimbal lock
        self.pitch = self.pitch.clamp(
            -std::f32::consts::FRAC_PI_2 + 0.01,
            std::f32::consts::FRAC_PI_2 - 0.01,
        );
    }

    /// Zoom the camera (scroll wheel).
    pub fn zoom(&mut self, delta: f32) {
        let factor = 1.0 - delta * 0.1;
        self.distance *= factor;
        self.distance = self.distance.max(0.1);
    }

    /// Pan the camera center (keyboard-style, distance-based speed).
    /// `dx` is horizontal pan, `dy` is vertical pan (typically ±1.0).
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let pan_speed = self.distance * 0.01;
        let right = Vector3::new(-self.yaw.sin(), 0.0, self.yaw.cos());
        let up = Vector3::y();
        self.center += (right * dx + up * dy) * pan_speed;
    }

    /// Pan the camera center (mouse-style, fixed sensitivity).
    /// `dx` and `dy` are raw pixel deltas from mouse motion.
    /// Note: screen Y is inverted (down = positive), so we negate dy.
    pub fn pan_mouse(&mut self, dx: f32, dy: f32) {
        let sensitivity = 0.005 * self.distance;
        let right = Vector3::new(-self.yaw.sin(), 0.0, self.yaw.cos());
        let up = Vector3::y();
        // Negate dy because screen Y goes down but world Y goes up
        self.center += (right * dx + up * -dy) * sensitivity;
    }

    /// Compute the camera position from spherical coordinates.
    pub fn position(&self) -> Point3<f32> {
        let cos_pitch = self.pitch.cos();
        let offset = Vector3::new(
            self.distance * cos_pitch * self.yaw.sin(),
            self.distance * self.pitch.sin(),
            self.distance * cos_pitch * self.yaw.cos(),
        );
        self.center + offset
    }

    /// Compute a look-at view matrix as a column-major 4x4 (16 floats).
    pub fn view_matrix(&self) -> [f32; 16] {
        let eye = self.position();
        let view = Matrix4::look_at_rh(&eye, &self.center, &Vector3::y());
        view.as_slice().try_into().unwrap()
    }
}
