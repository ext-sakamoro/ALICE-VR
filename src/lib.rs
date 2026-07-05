//! ALICE-VR: Pure Rust VR runtime.
//!
//! Provides head tracking (6DOF, quaternion integration), lens distortion
//! correction (barrel distortion, chromatic aberration), controller input
//! (buttons, triggers, thumbstick, haptics), stereo rendering (IPD, eye
//! matrices), reprojection (ASW/ATW), comfort metrics (FPS, latency), and
//! guardian/boundary system.

#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(
    clippy::module_name_repetitions,
    clippy::similar_names,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::missing_errors_doc,
    clippy::missing_panics_doc,
    clippy::must_use_candidate,
    clippy::wildcard_imports,
    clippy::doc_markdown,
    clippy::too_many_lines,
    clippy::cast_lossless,
    clippy::suboptimal_flops
)]

pub mod comfort;
pub mod controller;
pub mod guardian;
pub mod lens;
pub mod mat4;
pub mod pose;
pub mod prelude;
pub mod quat;
pub mod reprojection;
pub mod stereo;
pub mod vec3;

#[cfg(test)]
mod integration_tests;

// Backward-compat re-exports.
pub use crate::comfort::*;
pub use crate::controller::*;
pub use crate::guardian::*;
pub use crate::lens::*;
pub use crate::mat4::*;
pub use crate::pose::*;
pub use crate::quat::*;
pub use crate::reprojection::*;
pub use crate::stereo::*;
pub use crate::vec3::*;
