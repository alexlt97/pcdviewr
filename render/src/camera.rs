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
            yaw: std::f32::consts::FRAC_PI_4,    // 45 degrees
            pitch: -std::f32::consts::FRAC_PI_6, // -30 degrees (looking slightly down)
        }
    }

    /// Create a camera framing the cloud bounds.
    pub fn from_bounds(min: [f32; 3], max: [f32; 3]) -> Self {
        let center = (Vector3::from(min) + Vector3::from(max)) * 0.5;
        let radius = (Vector3::from(max) - Vector3::from(min)).norm() * 0.5;
        Self {
            center: Point3::from(center),
            distance: (radius / (22.5_f32.to_radians().sin()) * 1.2).max(0.1),
            ..Self::new()
        }
    }

    /// Rotate the view while keeping the eye fixed.
    pub fn look(&mut self, dx: f32, dy: f32) {
        let eye = self.position();
        self.orbit(dx, dy);
        self.center += eye - self.position();
    }

    fn right(&self) -> Vector3<f32> {
        Vector3::new(self.yaw.cos(), 0.0, -self.yaw.sin())
    }

    /// Move in camera-relative directions, in world units.
    pub fn move_local(&mut self, right: f32, up: f32, forward: f32, amount: f32) {
        let direction = self.right() * right
            + Vector3::y() * up
            + (self.center - self.position()).normalize() * forward;
        if direction.norm_squared() > 0.0 {
            self.center += direction.normalize() * amount;
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
        let factor = (-delta * 0.1).clamp(-2.0, 2.0).exp();
        self.distance *= factor;
        self.distance = self.distance.clamp(0.001, 1.0e9);
    }

    /// Pan the camera center (keyboard-style, distance-based speed).
    /// `dx` is horizontal pan, `dy` is vertical pan (typically ±1.0).
    pub fn pan(&mut self, dx: f32, dy: f32) {
        let pan_speed = self.distance * 0.01;
        let right = self.right();
        let up = Vector3::y();
        self.center += (right * dx + up * dy) * pan_speed;
    }

    /// Zoom by a pinch scale factor (two-finger pinch on touch screens).
    /// `factor` > 1.0 zooms in, < 1.0 zooms out.
    /// The raw pixel ratio is damped to avoid the cloud disappearing on small
    /// accidental movements (lerp 15% toward the raw ratio per frame).
    pub fn zoom_pinch(&mut self, factor: f32) {
        // Damp: blend 15% of the raw delta per event so large jumps are smoothed.
        let damped = 1.0 + (factor - 1.0) * 0.15;
        // Hard clamp so a single event never changes distance by more than 10%.
        let clamped = damped.clamp(0.90, 1.10);
        self.distance /= clamped;
        self.distance = self.distance.clamp(0.001, 1.0e9);
    }

    /// Fly the camera forward/backward along its look direction.
    /// Positive `delta` moves into the cloud (toward where the camera is pointing).
    /// Speed scales with current distance so it feels natural at any zoom level.
    pub fn fly_forward(&mut self, delta: f32) {
        let speed = self.distance * 0.008;
        let cos_pitch = self.pitch.cos();
        // Forward vector: direction from camera toward center.
        let forward = Vector3::new(
            -cos_pitch * self.yaw.sin(),
            -self.pitch.sin(),
            -cos_pitch * self.yaw.cos(),
        );
        self.center += forward * delta * speed;
    }

    /// Strafe the camera left/right (perpendicular to look direction, on the XZ plane).
    /// Positive `delta` moves right.
    pub fn strafe(&mut self, delta: f32) {
        let speed = self.distance * 0.008;
        // Right vector is perpendicular to forward on the XZ plane.
        let right = self.right();
        self.center += right * delta * speed;
    }

    /// Pan the camera center (mouse-style, fixed sensitivity).
    /// `dx` and `dy` are raw pixel deltas from mouse motion.
    /// Note: screen Y is inverted (down = positive), so we negate dy.
    pub fn pan_mouse(&mut self, dx: f32, dy: f32) {
        self.pan_pixels(dx, dy, 720.0);
    }

    pub fn pan_pixels(&mut self, dx: f32, dy: f32, height: f32) {
        let sensitivity = 2.0 * self.distance * 22.5_f32.to_radians().tan() / height.max(1.0);
        let right = self.right();
        let forward = (self.center - self.position()).normalize();
        let up = right.cross(&forward);
        self.center += (-right * dx + up * dy) * sensitivity;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strafe_is_perpendicular_to_view_at_several_angles() {
        for yaw in [0.0, 0.7, 1.5, 3.0] {
            let mut camera = Camera {
                yaw,
                ..Camera::new()
            };
            let forward = (camera.center - camera.position()).normalize();
            let before = camera.center;
            camera.move_local(1.0, 0.0, 0.0, 2.0);
            let movement = camera.center - before;
            assert!(movement.dot(&forward).abs() < 1e-5);
            assert!((movement.norm() - 2.0).abs() < 1e-5);
        }
    }

    #[test]
    fn fly_look_keeps_eye_fixed() {
        let mut camera = Camera::new();
        let eye = camera.position();
        camera.look(100.0, 50.0);
        assert!((camera.position() - eye).norm() < 1e-5);
    }

    #[test]
    fn degenerate_bounds_and_large_zoom_stay_finite() {
        let mut camera = Camera::from_bounds([0.0; 3], [0.0; 3]);
        assert!(camera.view_matrix().iter().all(|v| v.is_finite()));
        camera.zoom(1000.0);
        assert!(camera.distance > 0.0);
        camera.zoom(-1000.0);
        assert!(camera.distance.is_finite());
    }

    #[test]
    fn dragging_right_moves_scene_right() {
        let mut camera = Camera::new();
        let before = camera.center;
        camera.pan_pixels(100.0, 0.0, 720.0);
        assert!((camera.center - before).dot(&camera.right()) < 0.0);
    }

    #[test]
    fn diagonal_motion_has_same_speed() {
        let mut camera = Camera::new();
        let before = camera.center;
        camera.move_local(1.0, 1.0, 1.0, 3.0);
        assert!(((camera.center - before).norm() - 3.0).abs() < 1e-5);
    }
}
