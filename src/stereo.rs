//! Stereo rendering (`Eye` / `StereoConfig` — IPD + eye matrices).

use crate::mat4::Mat4;
use crate::pose::Pose;
use crate::vec3::Vec3;

// Stereo Rendering
// ---------------------------------------------------------------------------

/// Eye identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Eye {
    Left,
    Right,
}

/// Stereo rendering configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StereoConfig {
    /// Inter-pupillary distance in meters.
    pub ipd: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
    /// Aspect ratio (width / height) per eye.
    pub aspect: f32,
    /// Near clip plane.
    pub near: f32,
    /// Far clip plane.
    pub far: f32,
}

impl Default for StereoConfig {
    fn default() -> Self {
        Self {
            ipd: 0.063,
            fov_y: 100.0_f32.to_radians(),
            aspect: 1.0,
            near: 0.01,
            far: 1000.0,
        }
    }
}

impl StereoConfig {
    /// Get the eye offset vector for a given eye.
    #[must_use]
    pub fn eye_offset(&self, eye: Eye) -> Vec3 {
        let half_ipd = self.ipd * 0.5;
        match eye {
            Eye::Left => Vec3::new(-half_ipd, 0.0, 0.0),
            Eye::Right => Vec3::new(half_ipd, 0.0, 0.0),
        }
    }

    /// Compute the view matrix for one eye given the head pose.
    #[must_use]
    pub fn eye_view_matrix(&self, head_pose: Pose, eye: Eye) -> Mat4 {
        let offset = self.eye_offset(eye);
        let rotated_offset = head_pose.orientation.rotate_vector(offset.to_array());
        let eye_pos =
            head_pose.position + Vec3::new(rotated_offset[0], rotated_offset[1], rotated_offset[2]);
        let eye_pose = Pose::new(eye_pos, head_pose.orientation);
        eye_pose.to_view_matrix()
    }

    /// Compute the projection matrix for one eye.
    #[must_use]
    pub fn eye_projection_matrix(&self, _eye: Eye) -> Mat4 {
        Mat4::perspective(self.fov_y, self.aspect, self.near, self.far)
    }

    /// Compute the combined view-projection matrix for one eye.
    #[must_use]
    pub fn eye_view_projection(&self, head_pose: Pose, eye: Eye) -> Mat4 {
        let view = self.eye_view_matrix(head_pose, eye);
        let proj = self.eye_projection_matrix(eye);
        proj * view
    }
}
