//! 6DOF pose + head tracking (`Pose` / `AngularVelocity` / `HeadTracker`).

use crate::mat4::Mat4;
use crate::quat::Quat;
use crate::vec3::Vec3;

// 6DOF Pose & Head Tracking
// ---------------------------------------------------------------------------

/// 6 Degrees-of-Freedom pose (position + orientation).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub position: Vec3,
    pub orientation: Quat,
}

impl Default for Pose {
    fn default() -> Self {
        Self {
            position: Vec3::ZERO,
            orientation: Quat::IDENTITY,
        }
    }
}

impl Pose {
    #[must_use]
    pub const fn new(position: Vec3, orientation: Quat) -> Self {
        Self {
            position,
            orientation,
        }
    }

    /// Convert pose to a 4x4 view matrix (inverse of the pose transform).
    #[must_use]
    pub fn to_view_matrix(self) -> Mat4 {
        let inv_rot = self.orientation.conjugate();
        let inv_pos = inv_rot.rotate_vector(self.position.to_array());
        let rot = inv_rot.to_rotation_matrix();
        Mat4 {
            data: [
                rot[0],
                rot[1],
                rot[2],
                0.0,
                rot[3],
                rot[4],
                rot[5],
                0.0,
                rot[6],
                rot[7],
                rot[8],
                0.0,
                -inv_pos[0],
                -inv_pos[1],
                -inv_pos[2],
                1.0,
            ],
        }
    }
}

/// Angular velocity for gyroscope integration.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AngularVelocity {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl AngularVelocity {
    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
}

/// Head tracker with 6DOF pose, integrating gyroscope data.
#[derive(Debug, Clone)]
pub struct HeadTracker {
    pub pose: Pose,
    pub linear_velocity: Vec3,
}

impl Default for HeadTracker {
    fn default() -> Self {
        Self::new()
    }
}

impl HeadTracker {
    #[must_use]
    pub fn new() -> Self {
        Self {
            pose: Pose::default(),
            linear_velocity: Vec3::ZERO,
        }
    }

    /// Integrate angular velocity over `dt` seconds (quaternion integration).
    pub fn integrate_rotation(&mut self, gyro: AngularVelocity, dt: f32) {
        let mag = gyro
            .x
            .mul_add(gyro.x, gyro.y.mul_add(gyro.y, gyro.z * gyro.z))
            .sqrt();
        if mag < 1e-10 {
            return;
        }
        let axis = [gyro.x / mag, gyro.y / mag, gyro.z / mag];
        let angle = mag * dt;
        let delta = Quat::from_axis_angle(axis, angle);
        self.pose.orientation = (self.pose.orientation * delta).normalized();
    }

    /// Integrate linear acceleration over `dt` (position update).
    pub fn integrate_position(&mut self, acceleration: Vec3, dt: f32) {
        self.linear_velocity = self.linear_velocity + acceleration.scale(dt);
        self.pose.position = self.pose.position + self.linear_velocity.scale(dt);
    }

    /// Predict pose at a future time given current velocities.
    #[must_use]
    pub fn predict(&self, gyro: AngularVelocity, dt: f32) -> Pose {
        let mut predicted = self.clone();
        predicted.integrate_rotation(gyro, dt);
        predicted.pose.position = predicted.pose.position + predicted.linear_velocity.scale(dt);
        predicted.pose
    }
}
