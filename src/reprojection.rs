//! Reprojection (`ReprojectionMode` / `Reprojection` — ASW / ATW).

use crate::mat4::Mat4;
use crate::pose::Pose;
use crate::quat::Quat;
use crate::vec3::Vec3;

// Reprojection (ASW / ATW)
// ---------------------------------------------------------------------------

/// Reprojection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReprojectionMode {
    /// No reprojection.
    None,
    /// Asynchronous Time Warp -- rotational only.
    Atw,
    /// Asynchronous Space Warp -- rotational + positional.
    Asw,
}

/// Reprojection system state.
#[derive(Debug, Clone)]
pub struct Reprojection {
    pub mode: ReprojectionMode,
    /// The pose at which the last frame was rendered.
    pub rendered_pose: Pose,
    /// Target framerate.
    pub target_fps: f32,
    /// Whether reprojection was activated last frame.
    pub active: bool,
    /// Number of consecutive reprojected frames.
    pub reprojected_count: u32,
}

impl Default for Reprojection {
    fn default() -> Self {
        Self {
            mode: ReprojectionMode::Atw,
            rendered_pose: Pose::default(),
            target_fps: 90.0,
            active: false,
            reprojected_count: 0,
        }
    }
}

impl Reprojection {
    /// Maximum allowed consecutive reprojected frames before fallback.
    const MAX_REPROJECTED: u32 = 5;

    #[must_use]
    pub fn new(mode: ReprojectionMode, target_fps: f32) -> Self {
        Self {
            mode,
            rendered_pose: Pose::default(),
            target_fps,
            active: false,
            reprojected_count: 0,
        }
    }

    /// Called when a frame is successfully rendered in time.
    pub const fn on_frame_rendered(&mut self, pose: Pose) {
        self.rendered_pose = pose;
        self.active = false;
        self.reprojected_count = 0;
    }

    /// Called when a frame misses the deadline, triggering reprojection.
    /// Returns the corrective rotation delta to apply.
    #[must_use]
    pub fn reproject(&mut self, current_pose: Pose) -> Option<Quat> {
        if self.mode == ReprojectionMode::None {
            return None;
        }
        if self.reprojected_count >= Self::MAX_REPROJECTED {
            return None;
        }
        self.active = true;
        self.reprojected_count += 1;

        // Compute rotational delta between rendered and current pose
        let delta = current_pose.orientation * self.rendered_pose.orientation.inverse();
        Some(delta.normalized())
    }

    /// Compute the positional correction for ASW mode.
    #[must_use]
    pub fn positional_correction(&self, current_pose: Pose) -> Vec3 {
        if self.mode != ReprojectionMode::Asw {
            return Vec3::ZERO;
        }
        current_pose.position - self.rendered_pose.position
    }

    /// Apply the reprojection warp to a view matrix.
    #[must_use]
    pub fn warp_view_matrix(&mut self, original_view: Mat4, current_pose: Pose) -> Mat4 {
        self.reproject(current_pose)
            .map_or(original_view, |rot_delta| {
                let correction = Mat4::from_quat_translation(rot_delta, Vec3::ZERO);
                let pos_delta = self.positional_correction(current_pose);
                let translation = Mat4::translation(pos_delta.x, pos_delta.y, pos_delta.z);
                (translation * correction) * original_view
            })
    }

    /// Get the frame time budget in seconds.
    #[must_use]
    pub fn frame_budget(&self) -> f32 {
        1.0 / self.target_fps
    }

    /// Check if reprojection limit has been reached.
    #[must_use]
    pub const fn is_at_limit(&self) -> bool {
        self.reprojected_count >= Self::MAX_REPROJECTED
    }
}
