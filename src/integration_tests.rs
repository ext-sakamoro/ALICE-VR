//! Integration tests spanning multiple modules.

#![allow(
    clippy::float_cmp,
    clippy::unreadable_literal,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::too_many_lines,
    clippy::needless_range_loop,
    clippy::explicit_iter_loop,
    clippy::bool_to_int_with_if,
    clippy::approx_constant,
    clippy::cast_lossless,
    clippy::redundant_clone,
    clippy::format_collect,
    clippy::similar_names,
    clippy::needless_collect,
    clippy::iter_cloned_collect,
    clippy::suboptimal_flops
)]

use crate::comfort::*;
use crate::controller::*;
use crate::guardian::*;
use crate::lens::*;
use crate::mat4::*;
use crate::pose::*;
use crate::quat::*;
use crate::reprojection::*;
use crate::stereo::*;
use crate::vec3::*;
use core::f32::consts::PI;

const EPSILON: f32 = 1e-4;

fn approx_eq(a: f32, b: f32) -> bool {
    (a - b).abs() < EPSILON
}

// ---- Quaternion tests ----

#[test]
fn quat_identity() {
    let q = Quat::IDENTITY;
    assert!(approx_eq(q.magnitude(), 1.0));
}

#[test]
fn quat_default_is_identity() {
    let q = Quat::default();
    assert_eq!(q, Quat::IDENTITY);
}

#[test]
fn quat_from_axis_angle_zero() {
    let q = Quat::from_axis_angle([0.0, 1.0, 0.0], 0.0);
    assert!(approx_eq(q.w, 1.0));
    assert!(approx_eq(q.x, 0.0));
}

#[test]
fn quat_from_axis_angle_90_y() {
    let q = Quat::from_axis_angle([0.0, 1.0, 0.0], PI / 2.0);
    assert!(approx_eq(q.magnitude(), 1.0));
}

#[test]
fn quat_conjugate() {
    let q = Quat::new(1.0, 2.0, 3.0, 4.0);
    let c = q.conjugate();
    assert_eq!(c.w, 1.0);
    assert_eq!(c.x, -2.0);
    assert_eq!(c.y, -3.0);
    assert_eq!(c.z, -4.0);
}

#[test]
fn quat_inverse_identity() {
    let inv = Quat::IDENTITY.inverse();
    assert!(approx_eq(inv.w, 1.0));
    assert!(approx_eq(inv.x, 0.0));
}

#[test]
fn quat_mul_identity() {
    let q = Quat::from_axis_angle([0.0, 1.0, 0.0], 0.5);
    let result = q * Quat::IDENTITY;
    assert!(approx_eq(result.w, q.w));
    assert!(approx_eq(result.x, q.x));
}

#[test]
fn quat_mul_inverse_gives_identity() {
    let q = Quat::from_axis_angle([0.0, 1.0, 0.0], 1.0).normalized();
    let result = (q * q.inverse()).normalized();
    assert!(approx_eq(result.w, 1.0));
}

#[test]
fn quat_rotate_vector_identity() {
    let v = Quat::IDENTITY.rotate_vector([1.0, 2.0, 3.0]);
    assert!(approx_eq(v[0], 1.0));
    assert!(approx_eq(v[1], 2.0));
    assert!(approx_eq(v[2], 3.0));
}

#[test]
fn quat_rotate_vector_90_y() {
    let q = Quat::from_axis_angle([0.0, 1.0, 0.0], PI / 2.0);
    let v = q.rotate_vector([1.0, 0.0, 0.0]);
    assert!(approx_eq(v[0], 0.0));
    assert!(approx_eq(v[2], -1.0));
}

#[test]
fn quat_slerp_endpoints() {
    let a = Quat::IDENTITY;
    let b = Quat::from_axis_angle([0.0, 1.0, 0.0], PI / 2.0);
    let s0 = a.slerp(b, 0.0);
    assert!(approx_eq(s0.w, a.w));
    let s1 = a.slerp(b, 1.0);
    assert!(approx_eq(s1.w, b.w));
}

#[test]
fn quat_slerp_midpoint() {
    let a = Quat::IDENTITY;
    let b = Quat::from_axis_angle([0.0, 1.0, 0.0], PI / 2.0);
    let mid = a.slerp(b, 0.5);
    assert!(approx_eq(mid.magnitude(), 1.0));
}

#[test]
fn quat_normalized_already_unit() {
    let q = Quat::IDENTITY.normalized();
    assert!(approx_eq(q.magnitude(), 1.0));
}

#[test]
fn quat_normalized_non_unit() {
    let q = Quat::new(2.0, 0.0, 0.0, 0.0).normalized();
    assert!(approx_eq(q.w, 1.0));
}

#[test]
fn quat_from_euler_zero() {
    let q = Quat::from_euler(0.0, 0.0, 0.0);
    assert!(approx_eq(q.w, 1.0));
}

#[test]
fn quat_to_rotation_matrix_identity() {
    let m = Quat::IDENTITY.to_rotation_matrix();
    assert!(approx_eq(m[0], 1.0));
    assert!(approx_eq(m[4], 1.0));
    assert!(approx_eq(m[8], 1.0));
}

#[test]
fn quat_normalized_zero_returns_identity() {
    let q = Quat::new(0.0, 0.0, 0.0, 0.0).normalized();
    assert_eq!(q, Quat::IDENTITY);
}

// ---- Vec3 tests ----

#[test]
fn vec3_zero() {
    let v = Vec3::ZERO;
    assert_eq!(v.length(), 0.0);
}

#[test]
fn vec3_add() {
    let a = Vec3::new(1.0, 2.0, 3.0);
    let b = Vec3::new(4.0, 5.0, 6.0);
    let c = a + b;
    assert_eq!(c.x, 5.0);
    assert_eq!(c.y, 7.0);
    assert_eq!(c.z, 9.0);
}

#[test]
fn vec3_sub() {
    let a = Vec3::new(4.0, 5.0, 6.0);
    let b = Vec3::new(1.0, 2.0, 3.0);
    let c = a - b;
    assert_eq!(c.x, 3.0);
}

#[test]
fn vec3_scale() {
    let v = Vec3::new(1.0, 2.0, 3.0).scale(2.0);
    assert_eq!(v.x, 2.0);
    assert_eq!(v.y, 4.0);
}

#[test]
fn vec3_dot() {
    let a = Vec3::new(1.0, 0.0, 0.0);
    let b = Vec3::new(0.0, 1.0, 0.0);
    assert_eq!(a.dot(b), 0.0);
}

#[test]
fn vec3_cross() {
    let a = Vec3::new(1.0, 0.0, 0.0);
    let b = Vec3::new(0.0, 1.0, 0.0);
    let c = a.cross(b);
    assert!(approx_eq(c.z, 1.0));
}

#[test]
fn vec3_normalized() {
    let v = Vec3::new(3.0, 4.0, 0.0).normalized();
    assert!(approx_eq(v.length(), 1.0));
}

#[test]
fn vec3_distance() {
    let a = Vec3::new(0.0, 0.0, 0.0);
    let b = Vec3::new(3.0, 4.0, 0.0);
    assert!(approx_eq(a.distance(b), 5.0));
}

#[test]
fn vec3_to_array() {
    let v = Vec3::new(1.0, 2.0, 3.0);
    assert_eq!(v.to_array(), [1.0, 2.0, 3.0]);
}

#[test]
fn vec3_normalized_zero() {
    let v = Vec3::ZERO.normalized();
    assert_eq!(v, Vec3::ZERO);
}

// ---- Mat4 tests ----

#[test]
fn mat4_identity() {
    let m = Mat4::IDENTITY;
    assert_eq!(m.get(0, 0), 1.0);
    assert_eq!(m.get(1, 1), 1.0);
    assert_eq!(m.get(0, 1), 0.0);
}

#[test]
fn mat4_translation() {
    let m = Mat4::translation(1.0, 2.0, 3.0);
    assert_eq!(m.get(0, 3), 1.0);
    assert_eq!(m.get(1, 3), 2.0);
    assert_eq!(m.get(2, 3), 3.0);
}

#[test]
fn mat4_mul_identity() {
    let m = Mat4::translation(5.0, 6.0, 7.0);
    let result = m * Mat4::IDENTITY;
    assert!(approx_eq(result.get(0, 3), 5.0));
}

#[test]
fn mat4_perspective_non_zero() {
    let m = Mat4::perspective(PI / 2.0, 1.0, 0.1, 100.0);
    assert!(m.get(0, 0) > 0.0);
    assert!(m.get(1, 1) > 0.0);
}

#[test]
fn mat4_from_quat_translation() {
    let m = Mat4::from_quat_translation(Quat::IDENTITY, Vec3::new(1.0, 2.0, 3.0));
    assert!(approx_eq(m.get(0, 3), 1.0));
    assert!(approx_eq(m.get(1, 3), 2.0));
}

#[test]
fn mat4_default_is_identity() {
    assert_eq!(Mat4::default(), Mat4::IDENTITY);
}

// ---- Pose & HeadTracker tests ----

#[test]
fn pose_default() {
    let p = Pose::default();
    assert_eq!(p.position, Vec3::ZERO);
    assert_eq!(p.orientation, Quat::IDENTITY);
}

#[test]
fn pose_to_view_matrix_identity() {
    let p = Pose::default();
    let v = p.to_view_matrix();
    assert!(approx_eq(v.get(0, 0), 1.0));
}

#[test]
fn head_tracker_default() {
    let ht = HeadTracker::new();
    assert_eq!(ht.pose, Pose::default());
}

#[test]
fn head_tracker_integrate_rotation() {
    let mut ht = HeadTracker::new();
    ht.integrate_rotation(AngularVelocity::new(0.0, 1.0, 0.0), 0.1);
    assert!(ht.pose.orientation != Quat::IDENTITY);
}

#[test]
fn head_tracker_integrate_zero_gyro() {
    let mut ht = HeadTracker::new();
    ht.integrate_rotation(AngularVelocity::new(0.0, 0.0, 0.0), 0.1);
    assert_eq!(ht.pose.orientation, Quat::IDENTITY);
}

#[test]
fn head_tracker_integrate_position() {
    let mut ht = HeadTracker::new();
    ht.integrate_position(Vec3::new(1.0, 0.0, 0.0), 1.0);
    assert!(approx_eq(ht.linear_velocity.x, 1.0));
    assert!(approx_eq(ht.pose.position.x, 1.0));
}

#[test]
fn head_tracker_predict() {
    let ht = HeadTracker::new();
    let predicted = ht.predict(AngularVelocity::new(0.0, 1.0, 0.0), 0.016);
    assert!(predicted.orientation != Quat::IDENTITY);
}

#[test]
fn head_tracker_default_trait() {
    let ht = HeadTracker::default();
    assert_eq!(ht.pose, Pose::default());
}

// ---- Lens Distortion tests ----

#[test]
fn lens_default() {
    let lens = LensDistortion::default();
    assert!(approx_eq(lens.k[0], 1.0));
}

#[test]
fn lens_distort_center() {
    let lens = LensDistortion::default();
    let d = lens.distort([0.0, 0.0]);
    assert!(approx_eq(d[0], 0.0));
    assert!(approx_eq(d[1], 0.0));
}

#[test]
fn lens_undistort_center() {
    let lens = LensDistortion::default();
    let u = lens.undistort([0.0, 0.0]);
    assert!(approx_eq(u[0], 0.0));
}

#[test]
fn lens_distort_undistort_roundtrip() {
    let lens = LensDistortion::default();
    let original = [0.3, 0.4];
    let distorted = lens.distort(original);
    let undistorted = lens.undistort(distorted);
    assert!(approx_eq(undistorted[0], original[0]));
    assert!(approx_eq(undistorted[1], original[1]));
}

#[test]
fn lens_chromatic_center() {
    let lens = LensDistortion::default();
    let ca = lens.distort_chromatic([0.0, 0.0]);
    for ch in &ca {
        assert!(approx_eq(ch[0], 0.0));
        assert!(approx_eq(ch[1], 0.0));
    }
}

#[test]
fn lens_chromatic_channels_differ() {
    let lens = LensDistortion::default();
    let ca = lens.distort_chromatic([0.5, 0.5]);
    assert!((ca[0][0] - ca[2][0]).abs() > 1e-6);
}

#[test]
fn lens_distortion_mesh_size() {
    let lens = LensDistortion::default();
    let (positions, uvs) = lens.generate_distortion_mesh(4);
    assert_eq!(positions.len(), 16);
    assert_eq!(uvs.len(), 16);
}

#[test]
fn lens_distortion_mesh_corners() {
    let lens = LensDistortion::default();
    let (positions, _) = lens.generate_distortion_mesh(2);
    assert!(approx_eq(positions[0][0], -1.0));
    assert!(approx_eq(positions[0][1], -1.0));
}

// ---- Controller tests ----

#[test]
fn button_flags_none() {
    assert!(ButtonFlags::NONE.is_empty());
}

#[test]
fn button_flags_contains() {
    let flags = ButtonFlags::A.union(ButtonFlags::B);
    assert!(flags.contains(ButtonFlags::A));
    assert!(flags.contains(ButtonFlags::B));
    assert!(!flags.contains(ButtonFlags::X));
}

#[test]
fn button_flags_intersection() {
    let a = ButtonFlags::A.union(ButtonFlags::B);
    let b = ButtonFlags::B.union(ButtonFlags::X);
    let c = a.intersection(b);
    assert!(c.contains(ButtonFlags::B));
    assert!(!c.contains(ButtonFlags::A));
}

#[test]
fn thumbstick_deadzone_below() {
    let ts = ThumbstickState::new(0.05, 0.05);
    let filtered = ts.with_deadzone(0.15);
    assert!(approx_eq(filtered.x, 0.0));
    assert!(approx_eq(filtered.y, 0.0));
}

#[test]
fn thumbstick_deadzone_above() {
    let ts = ThumbstickState::new(0.8, 0.0);
    let filtered = ts.with_deadzone(0.1);
    assert!(filtered.x > 0.0);
}

#[test]
fn thumbstick_magnitude() {
    let ts = ThumbstickState::new(3.0, 4.0);
    assert!(approx_eq(ts.magnitude(), 5.0));
}

#[test]
fn thumbstick_angle() {
    let ts = ThumbstickState::new(1.0, 0.0);
    assert!(approx_eq(ts.angle(), 0.0));
}

#[test]
fn haptic_pulse_active() {
    let p = HapticPulse::new(0.5, 100.0, 1.0);
    assert!(p.is_active(0.0));
    assert!(p.is_active(0.49));
    assert!(!p.is_active(0.5));
}

#[test]
fn haptic_pulse_amplitude_fade() {
    let p = HapticPulse::new(1.0, 100.0, 1.0);
    let a0 = p.amplitude_at(0.0);
    let a_half = p.amplitude_at(0.5);
    assert!(a0 > a_half);
}

#[test]
fn haptic_pulse_clamp() {
    let p = HapticPulse::new(1.0, 100.0, 2.0);
    assert!(approx_eq(p.amplitude, 1.0));
}

#[test]
fn controller_state_new() {
    let cs = ControllerState::new(Hand::Left);
    assert_eq!(cs.hand, Hand::Left);
    assert!(cs.buttons.is_empty());
}

#[test]
fn controller_update_buttons_pressed() {
    let mut cs = ControllerState::new(Hand::Right);
    cs.update_buttons(ButtonFlags::A);
    assert!(cs.buttons_pressed.contains(ButtonFlags::A));
    assert!(cs.buttons_released.is_empty());
}

#[test]
fn controller_update_buttons_released() {
    let mut cs = ControllerState::new(Hand::Right);
    cs.update_buttons(ButtonFlags::A);
    cs.update_buttons(ButtonFlags::NONE);
    assert!(cs.buttons_released.contains(ButtonFlags::A));
}

#[test]
fn controller_trigger() {
    let mut cs = ControllerState::new(Hand::Left);
    cs.trigger = 0.8;
    assert!(cs.is_trigger_pulled(0.5));
    assert!(!cs.is_trigger_pulled(0.9));
}

#[test]
fn controller_grip() {
    let mut cs = ControllerState::new(Hand::Left);
    cs.grip = 0.7;
    assert!(cs.is_grip_held(0.5));
    assert!(!cs.is_grip_held(0.8));
}

// ---- Haptic engine tests ----

#[test]
fn haptic_engine_default() {
    let he = HapticEngine::default();
    assert!(he.left_pulse.is_none());
    assert!(he.right_pulse.is_none());
}

#[test]
fn haptic_engine_submit_left() {
    let mut he = HapticEngine::new();
    he.submit(Hand::Left, HapticPulse::new(0.5, 100.0, 0.8));
    assert!(he.left_pulse.is_some());
}

#[test]
fn haptic_engine_update() {
    let mut he = HapticEngine::new();
    he.submit(Hand::Left, HapticPulse::new(0.1, 100.0, 1.0));
    let (left, right) = he.update(0.05);
    assert!(left > 0.0);
    assert!(approx_eq(right, 0.0));
}

#[test]
fn haptic_engine_expires() {
    let mut he = HapticEngine::new();
    he.submit(Hand::Right, HapticPulse::new(0.1, 100.0, 1.0));
    he.update(0.2);
    assert!(he.right_pulse.is_none());
}

#[test]
fn haptic_engine_cancel_all() {
    let mut he = HapticEngine::new();
    he.submit(Hand::Left, HapticPulse::new(1.0, 100.0, 1.0));
    he.submit(Hand::Right, HapticPulse::new(1.0, 100.0, 1.0));
    he.cancel_all();
    assert!(he.left_pulse.is_none());
    assert!(he.right_pulse.is_none());
}

// ---- Stereo tests ----

#[test]
fn stereo_config_default() {
    let sc = StereoConfig::default();
    assert!(approx_eq(sc.ipd, 0.063));
}

#[test]
fn stereo_eye_offset_left() {
    let sc = StereoConfig::default();
    let offset = sc.eye_offset(Eye::Left);
    assert!(offset.x < 0.0);
}

#[test]
fn stereo_eye_offset_right() {
    let sc = StereoConfig::default();
    let offset = sc.eye_offset(Eye::Right);
    assert!(offset.x > 0.0);
}

#[test]
fn stereo_eye_offset_symmetric() {
    let sc = StereoConfig::default();
    let left = sc.eye_offset(Eye::Left);
    let right = sc.eye_offset(Eye::Right);
    assert!(approx_eq(left.x, -right.x));
}

#[test]
fn stereo_eye_view_matrix() {
    let sc = StereoConfig::default();
    let pose = Pose::default();
    let left = sc.eye_view_matrix(pose, Eye::Left);
    let right = sc.eye_view_matrix(pose, Eye::Right);
    assert!((left.get(0, 3) - right.get(0, 3)).abs() > 0.01);
}

#[test]
fn stereo_projection_matrix() {
    let sc = StereoConfig::default();
    let proj = sc.eye_projection_matrix(Eye::Left);
    assert!(proj.get(0, 0) > 0.0);
}

#[test]
fn stereo_view_projection() {
    let sc = StereoConfig::default();
    let pose = Pose::default();
    let vp = sc.eye_view_projection(pose, Eye::Left);
    assert!(vp.get(0, 0).is_finite());
}

// ---- Reprojection tests ----

#[test]
fn reprojection_default() {
    let rp = Reprojection::default();
    assert_eq!(rp.mode, ReprojectionMode::Atw);
    assert!(!rp.active);
}

#[test]
fn reprojection_none_mode() {
    let mut rp = Reprojection::new(ReprojectionMode::None, 90.0);
    let result = rp.reproject(Pose::default());
    assert!(result.is_none());
}

#[test]
fn reprojection_atw_returns_delta() {
    let mut rp = Reprojection::new(ReprojectionMode::Atw, 90.0);
    rp.on_frame_rendered(Pose::default());
    let new_pose = Pose::new(Vec3::ZERO, Quat::from_axis_angle([0.0, 1.0, 0.0], 0.01));
    let result = rp.reproject(new_pose);
    assert!(result.is_some());
    assert!(rp.active);
}

#[test]
fn reprojection_max_limit() {
    let mut rp = Reprojection::new(ReprojectionMode::Atw, 90.0);
    for _ in 0..5 {
        let _ = rp.reproject(Pose::default());
    }
    assert!(rp.is_at_limit());
    assert!(rp.reproject(Pose::default()).is_none());
}

#[test]
fn reprojection_reset_on_render() {
    let mut rp = Reprojection::new(ReprojectionMode::Atw, 90.0);
    let _ = rp.reproject(Pose::default());
    rp.on_frame_rendered(Pose::default());
    assert!(!rp.active);
    assert_eq!(rp.reprojected_count, 0);
}

#[test]
fn reprojection_frame_budget() {
    let rp = Reprojection::new(ReprojectionMode::Atw, 90.0);
    assert!(approx_eq(rp.frame_budget(), 1.0 / 90.0));
}

#[test]
fn reprojection_positional_correction_atw() {
    let rp = Reprojection::new(ReprojectionMode::Atw, 90.0);
    let correction = rp.positional_correction(Pose::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY));
    assert_eq!(correction, Vec3::ZERO);
}

#[test]
fn reprojection_positional_correction_asw() {
    let mut rp = Reprojection::new(ReprojectionMode::Asw, 90.0);
    rp.on_frame_rendered(Pose::default());
    let correction = rp.positional_correction(Pose::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY));
    assert!(approx_eq(correction.x, 1.0));
}

#[test]
fn reprojection_warp_view_matrix() {
    let mut rp = Reprojection::new(ReprojectionMode::Atw, 90.0);
    rp.on_frame_rendered(Pose::default());
    let view = Mat4::IDENTITY;
    let new_pose = Pose::new(Vec3::ZERO, Quat::from_axis_angle([0.0, 1.0, 0.0], 0.01));
    let warped = rp.warp_view_matrix(view, new_pose);
    assert!(warped != Mat4::IDENTITY);
}

// ---- Comfort metrics tests ----

#[test]
fn comfort_metrics_empty() {
    let cm = ComfortMetrics::new(60, 90.0);
    assert!(approx_eq(cm.fps(), 0.0));
    assert_eq!(cm.total_frames(), 0);
}

#[test]
fn comfort_metrics_record() {
    let mut cm = ComfortMetrics::new(60, 90.0);
    cm.record_frame(1.0 / 90.0);
    assert!(cm.fps() > 80.0);
}

#[test]
fn comfort_metrics_average() {
    let mut cm = ComfortMetrics::new(10, 90.0);
    for _ in 0..10 {
        cm.record_frame(0.011);
    }
    assert!(approx_eq(cm.average_frame_time(), 0.011));
}

#[test]
fn comfort_metrics_dropped() {
    let mut cm = ComfortMetrics::new(10, 90.0);
    cm.record_frame(0.1);
    assert_eq!(cm.dropped_frames, 1);
}

#[test]
fn comfort_metrics_percentile_99() {
    let mut cm = ComfortMetrics::new(100, 90.0);
    for i in 0..100 {
        cm.record_frame(0.01 + i as f32 * 0.0001);
    }
    let p99 = cm.percentile_99();
    assert!(p99 > 0.01);
}

#[test]
fn comfort_metrics_max() {
    let mut cm = ComfortMetrics::new(10, 90.0);
    cm.record_frame(0.01);
    cm.record_frame(0.05);
    cm.record_frame(0.02);
    assert!(approx_eq(cm.max_frame_time(), 0.05));
}

#[test]
fn comfort_metrics_min() {
    let mut cm = ComfortMetrics::new(10, 90.0);
    cm.record_frame(0.05);
    cm.record_frame(0.01);
    cm.record_frame(0.02);
    assert!(approx_eq(cm.min_frame_time(), 0.01));
}

#[test]
fn comfort_metrics_min_empty() {
    let cm = ComfortMetrics::new(10, 90.0);
    assert_eq!(cm.min_frame_time(), 0.0);
}

#[test]
fn comfort_metrics_latency() {
    let mut cm = ComfortMetrics::new(10, 90.0);
    cm.record_frame(1.0 / 90.0);
    let latency = cm.estimated_latency();
    assert!(latency > 1.0 / 90.0);
}

#[test]
fn comfort_metrics_score_good() {
    let mut cm = ComfortMetrics::new(10, 90.0);
    for _ in 0..10 {
        cm.record_frame(1.0 / 90.0);
    }
    assert!(cm.comfort_score() > 0.9);
    assert!(cm.is_comfortable());
}

#[test]
fn comfort_metrics_score_bad() {
    let mut cm = ComfortMetrics::new(10, 90.0);
    for _ in 0..10 {
        cm.record_frame(1.0 / 30.0);
    }
    assert!(cm.comfort_score() < 0.85);
    assert!(!cm.is_comfortable());
}

#[test]
fn comfort_metrics_reset() {
    let mut cm = ComfortMetrics::new(10, 90.0);
    cm.record_frame(0.01);
    cm.reset();
    assert_eq!(cm.total_frames(), 0);
    assert_eq!(cm.dropped_frames, 0);
}

#[test]
fn comfort_metrics_wraparound() {
    let mut cm = ComfortMetrics::new(5, 90.0);
    for _ in 0..10 {
        cm.record_frame(1.0 / 90.0);
    }
    assert_eq!(cm.total_frames(), 5);
}

// ---- Guardian tests ----

#[test]
fn guardian_rectangular() {
    let g = GuardianBoundary::rectangular(3.0, 3.0);
    assert_eq!(g.vertices.len(), 4);
}

#[test]
fn guardian_circular() {
    let g = GuardianBoundary::circular(2.0, 32);
    assert_eq!(g.vertices.len(), 32);
}

#[test]
fn guardian_is_inside_center() {
    let g = GuardianBoundary::rectangular(4.0, 4.0);
    assert!(g.is_inside(Vec3::new(0.0, 1.0, 0.0)));
}

#[test]
fn guardian_is_outside() {
    let g = GuardianBoundary::rectangular(2.0, 2.0);
    assert!(!g.is_inside(Vec3::new(5.0, 1.0, 0.0)));
}

#[test]
fn guardian_is_outside_ceiling() {
    let g = GuardianBoundary::rectangular(4.0, 4.0);
    assert!(!g.is_inside(Vec3::new(0.0, 10.0, 0.0)));
}

#[test]
fn guardian_is_outside_floor() {
    let g = GuardianBoundary::rectangular(4.0, 4.0);
    assert!(!g.is_inside(Vec3::new(0.0, -1.0, 0.0)));
}

#[test]
fn guardian_proximity_center() {
    let g = GuardianBoundary::rectangular(4.0, 4.0);
    let prox = g.proximity(Vec3::new(0.0, 1.0, 0.0));
    assert!(approx_eq(prox, 0.0));
}

#[test]
fn guardian_proximity_near_edge() {
    let g = GuardianBoundary::rectangular(4.0, 4.0);
    let prox = g.proximity(Vec3::new(1.85, 1.0, 0.0));
    assert!(prox > 0.0);
}

#[test]
fn guardian_signed_distance_inside() {
    let g = GuardianBoundary::rectangular(4.0, 4.0);
    let d = g.signed_distance([0.0, 0.0]);
    assert!(d < 0.0);
}

#[test]
fn guardian_signed_distance_outside() {
    let g = GuardianBoundary::rectangular(2.0, 2.0);
    let d = g.signed_distance([5.0, 0.0]);
    assert!(d > 0.0);
}

#[test]
fn guardian_closest_point() {
    let g = GuardianBoundary::rectangular(4.0, 4.0);
    let cp = g.closest_point([3.0, 0.0]);
    assert!(approx_eq(cp[0], 2.0));
}

#[test]
fn guardian_area_rectangle() {
    let g = GuardianBoundary::rectangular(4.0, 3.0);
    assert!(approx_eq(g.area(), 12.0));
}

#[test]
fn guardian_perimeter_rectangle() {
    let g = GuardianBoundary::rectangular(4.0, 3.0);
    assert!(approx_eq(g.perimeter(), 14.0));
}

#[test]
fn guardian_area_small_polygon() {
    let g = GuardianBoundary {
        vertices: vec![[0.0, 0.0], [1.0, 0.0]],
        warning_distance: 0.5,
        hard_distance: 0.1,
        floor_y: 0.0,
        ceiling_y: 3.0,
    };
    assert_eq!(g.area(), 0.0);
}

#[test]
fn guardian_signed_distance_few_vertices() {
    let g = GuardianBoundary {
        vertices: vec![[0.0, 0.0]],
        warning_distance: 0.5,
        hard_distance: 0.1,
        floor_y: 0.0,
        ceiling_y: 3.0,
    };
    assert_eq!(g.signed_distance([0.0, 0.0]), f32::MAX);
}

#[test]
fn guardian_closest_point_single_vertex() {
    let g = GuardianBoundary {
        vertices: vec![[1.0, 1.0]],
        warning_distance: 0.5,
        hard_distance: 0.1,
        floor_y: 0.0,
        ceiling_y: 3.0,
    };
    let cp = g.closest_point([5.0, 5.0]);
    assert_eq!(cp, [5.0, 5.0]);
}

// ---- Angular velocity tests ----

#[test]
fn angular_velocity_default() {
    let av = AngularVelocity::default();
    assert_eq!(av.x, 0.0);
    assert_eq!(av.y, 0.0);
    assert_eq!(av.z, 0.0);
}

#[test]
fn angular_velocity_new() {
    let av = AngularVelocity::new(1.0, 2.0, 3.0);
    assert_eq!(av.x, 1.0);
    assert_eq!(av.y, 2.0);
    assert_eq!(av.z, 3.0);
}

// ---- ThumbstickState default ----

#[test]
fn thumbstick_default() {
    let ts = ThumbstickState::default();
    assert_eq!(ts.x, 0.0);
    assert_eq!(ts.y, 0.0);
}

// ---- Haptic amplitude at expired ----

#[test]
fn haptic_amplitude_expired() {
    let p = HapticPulse::new(0.5, 100.0, 1.0);
    assert!(approx_eq(p.amplitude_at(1.0), 0.0));
}

// ---- Pose new constructor ----

#[test]
fn pose_new() {
    let p = Pose::new(Vec3::new(1.0, 2.0, 3.0), Quat::IDENTITY);
    assert!(approx_eq(p.position.x, 1.0));
}

// ---- Quat inverse non-unit ----

#[test]
fn quat_inverse_non_unit() {
    let q = Quat::new(2.0, 0.0, 0.0, 0.0);
    let inv = q.inverse();
    let result = q * inv;
    assert!(approx_eq(result.w, 1.0));
}
