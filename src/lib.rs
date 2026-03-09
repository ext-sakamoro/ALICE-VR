#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::similar_names)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_precision_loss)]

//! ALICE-VR: Pure Rust VR runtime.
//!
//! Provides head tracking (6DOF, quaternion integration), lens distortion
//! correction (barrel distortion, chromatic aberration), controller input
//! (buttons, triggers, thumbstick, haptics), stereo rendering (IPD, eye
//! matrices), reprojection (ASW/ATW), comfort metrics (FPS, latency), and
//! guardian/boundary system.

use core::f32::consts::PI;

// ---------------------------------------------------------------------------
// Quaternion
// ---------------------------------------------------------------------------

/// Unit quaternion for 3D rotation.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quat {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl core::ops::Mul for Quat {
    type Output = Self;

    /// Hamilton product.
    fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w.mul_add(
                rhs.w,
                self.x
                    .mul_add(-rhs.x, self.y.mul_add(-rhs.y, -self.z * rhs.z)),
            ),
            x: self.w.mul_add(
                rhs.x,
                self.x
                    .mul_add(rhs.w, self.y.mul_add(rhs.z, -self.z * rhs.y)),
            ),
            y: self.w.mul_add(
                rhs.y,
                self.x
                    .mul_add(-rhs.z, self.y.mul_add(rhs.w, self.z * rhs.x)),
            ),
            z: self.w.mul_add(
                rhs.z,
                self.x
                    .mul_add(rhs.y, self.y.mul_add(-rhs.x, self.z * rhs.w)),
            ),
        }
    }
}

impl Quat {
    pub const IDENTITY: Self = Self {
        w: 1.0,
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[must_use]
    pub const fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    /// Create from axis (must be unit length) and angle in radians.
    #[must_use]
    pub fn from_axis_angle(axis: [f32; 3], angle: f32) -> Self {
        let half = angle * 0.5;
        let s = half.sin();
        let c = half.cos();
        Self {
            w: c,
            x: axis[0] * s,
            y: axis[1] * s,
            z: axis[2] * s,
        }
    }

    /// Create from Euler angles (yaw, pitch, roll) in radians.
    #[must_use]
    pub fn from_euler(yaw: f32, pitch: f32, roll: f32) -> Self {
        let (sy, cy) = (yaw * 0.5).sin_cos();
        let (sp, cp) = (pitch * 0.5).sin_cos();
        let (sr, cr) = (roll * 0.5).sin_cos();
        Self {
            w: (cr * cp).mul_add(cy, sr * sp * sy),
            x: (sr * cp).mul_add(cy, -(cr * sp * sy)),
            y: (cr * sp).mul_add(cy, sr * cp * sy),
            z: (cr * cp).mul_add(sy, -(sr * sp * cy)),
        }
    }

    #[must_use]
    pub fn magnitude(self) -> f32 {
        self.w
            .mul_add(
                self.w,
                self.x
                    .mul_add(self.x, self.y.mul_add(self.y, self.z * self.z)),
            )
            .sqrt()
    }

    #[must_use]
    pub fn normalized(self) -> Self {
        let m = self.magnitude();
        if m < 1e-10 {
            return Self::IDENTITY;
        }
        let inv = 1.0 / m;
        Self {
            w: self.w * inv,
            x: self.x * inv,
            y: self.y * inv,
            z: self.z * inv,
        }
    }

    #[must_use]
    pub fn conjugate(self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    #[must_use]
    pub fn inverse(self) -> Self {
        let mag_sq = self.w.mul_add(
            self.w,
            self.x
                .mul_add(self.x, self.y.mul_add(self.y, self.z * self.z)),
        );
        if mag_sq < 1e-10 {
            return Self::IDENTITY;
        }
        let inv = 1.0 / mag_sq;
        Self {
            w: self.w * inv,
            x: -self.x * inv,
            y: -self.y * inv,
            z: -self.z * inv,
        }
    }

    /// Rotate a 3D vector by this quaternion.
    #[must_use]
    pub fn rotate_vector(self, v: [f32; 3]) -> [f32; 3] {
        let qv = Self::new(0.0, v[0], v[1], v[2]);
        let result = (self * qv) * self.conjugate();
        [result.x, result.y, result.z]
    }

    /// Spherical linear interpolation.
    #[must_use]
    pub fn slerp(self, other: Self, t: f32) -> Self {
        let mut dot = self.w.mul_add(
            other.w,
            self.x
                .mul_add(other.x, self.y.mul_add(other.y, self.z * other.z)),
        );
        let mut b = other;
        if dot < 0.0 {
            dot = -dot;
            b = Self::new(-b.w, -b.x, -b.y, -b.z);
        }
        if dot > 0.9995 {
            return Self::new(
                t.mul_add(b.w - self.w, self.w),
                t.mul_add(b.x - self.x, self.x),
                t.mul_add(b.y - self.y, self.y),
                t.mul_add(b.z - self.z, self.z),
            )
            .normalized();
        }
        let theta = dot.acos();
        let sin_theta = theta.sin();
        let wa = ((1.0 - t) * theta).sin() / sin_theta;
        let wb = (t * theta).sin() / sin_theta;
        Self::new(
            wa.mul_add(self.w, wb * b.w),
            wa.mul_add(self.x, wb * b.x),
            wa.mul_add(self.y, wb * b.y),
            wa.mul_add(self.z, wb * b.z),
        )
    }

    /// Convert to 3x3 rotation matrix (column-major).
    #[must_use]
    pub fn to_rotation_matrix(self) -> [f32; 9] {
        let (w, x, y, z) = (self.w, self.x, self.y, self.z);
        let x2 = x + x;
        let y2 = y + y;
        let z2 = z + z;
        let xx = x * x2;
        let xy = x * y2;
        let xz = x * z2;
        let yy = y * y2;
        let yz = y * z2;
        let zz = z * z2;
        let wx = w * x2;
        let wy = w * y2;
        let wz = w * z2;
        [
            1.0 - (yy + zz),
            xy + wz,
            xz - wy,
            xy - wz,
            1.0 - (xx + zz),
            yz + wx,
            xz + wy,
            yz - wx,
            1.0 - (xx + yy),
        ]
    }
}

// ---------------------------------------------------------------------------
// Vec3
// ---------------------------------------------------------------------------

/// Simple 3D vector.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl core::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}

impl core::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}

impl Vec3 {
    pub const ZERO: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
    };

    #[must_use]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }

    #[must_use]
    pub fn length(self) -> f32 {
        self.x
            .mul_add(self.x, self.y.mul_add(self.y, self.z * self.z))
            .sqrt()
    }

    #[must_use]
    pub fn scale(self, s: f32) -> Self {
        Self::new(self.x * s, self.y * s, self.z * s)
    }

    #[must_use]
    pub fn dot(self, rhs: Self) -> f32 {
        self.x.mul_add(rhs.x, self.y.mul_add(rhs.y, self.z * rhs.z))
    }

    #[must_use]
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y.mul_add(rhs.z, -(self.z * rhs.y)),
            self.z.mul_add(rhs.x, -(self.x * rhs.z)),
            self.x.mul_add(rhs.y, -(self.y * rhs.x)),
        )
    }

    #[must_use]
    pub fn normalized(self) -> Self {
        let len = self.length();
        if len < 1e-10 {
            return Self::ZERO;
        }
        self.scale(1.0 / len)
    }

    /// Distance to another point.
    #[must_use]
    pub fn distance(self, other: Self) -> f32 {
        (self - other).length()
    }

    /// Convert to array.
    #[must_use]
    pub const fn to_array(self) -> [f32; 3] {
        [self.x, self.y, self.z]
    }
}

// ---------------------------------------------------------------------------
// 4x4 Matrix
// ---------------------------------------------------------------------------

/// 4x4 column-major matrix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat4 {
    pub data: [f32; 16],
}

impl Default for Mat4 {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl core::ops::Mul for Mat4 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self {
        let mut out = [0.0_f32; 16];
        for col in 0..4 {
            for row in 0..4 {
                let mut sum = 0.0;
                for k in 0..4 {
                    sum = self.data[k * 4 + row].mul_add(rhs.data[col * 4 + k], sum);
                }
                out[col * 4 + row] = sum;
            }
        }
        Self { data: out }
    }
}

impl Mat4 {
    pub const IDENTITY: Self = Self {
        data: [
            1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0,
        ],
    };

    /// Create a translation matrix.
    #[must_use]
    pub const fn translation(x: f32, y: f32, z: f32) -> Self {
        Self {
            data: [
                1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, x, y, z, 1.0,
            ],
        }
    }

    /// Create a perspective projection matrix.
    #[must_use]
    pub fn perspective(fov_y_rad: f32, aspect: f32, near: f32, far: f32) -> Self {
        let f = 1.0 / (fov_y_rad * 0.5).tan();
        let nf = 1.0 / (near - far);
        let mut m = Self { data: [0.0; 16] };
        m.data[0] = f / aspect;
        m.data[5] = f;
        m.data[10] = (far + near) * nf;
        m.data[11] = -1.0;
        m.data[14] = 2.0 * far * near * nf;
        m
    }

    /// Create from rotation quaternion and translation.
    #[must_use]
    pub fn from_quat_translation(q: Quat, t: Vec3) -> Self {
        let rot = q.to_rotation_matrix();
        Self {
            data: [
                rot[0], rot[1], rot[2], 0.0, rot[3], rot[4], rot[5], 0.0, rot[6], rot[7], rot[8],
                0.0, t.x, t.y, t.z, 1.0,
            ],
        }
    }

    /// Get element at (row, col).
    #[must_use]
    pub const fn get(self, row: usize, col: usize) -> f32 {
        self.data[col * 4 + row]
    }
}

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Lens Distortion Correction
// ---------------------------------------------------------------------------

/// Lens distortion model parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LensDistortion {
    /// Barrel distortion coefficients (k1, k2, k3).
    pub k: [f32; 3],
    /// Chromatic aberration offsets for R, G, B channels.
    pub chromatic_aberration: [f32; 3],
    /// Lens center offset from screen center (normalized).
    pub center_offset: [f32; 2],
}

impl Default for LensDistortion {
    fn default() -> Self {
        Self {
            k: [1.0, 0.22, 0.24],
            chromatic_aberration: [0.995, 1.0, 1.005],
            center_offset: [0.0, 0.0],
        }
    }
}

impl LensDistortion {
    /// Apply barrel distortion to a normalized coordinate.
    /// Input `uv` is in `[-1, 1]` range centered on the lens.
    #[must_use]
    pub fn distort(&self, uv: [f32; 2]) -> [f32; 2] {
        let dx = uv[0] - self.center_offset[0];
        let dy = uv[1] - self.center_offset[1];
        let r2 = dx.mul_add(dx, dy * dy);
        let r4 = r2 * r2;
        let factor = self.k[1].mul_add(r2, self.k[2].mul_add(r4, self.k[0]));
        [
            dx.mul_add(factor, self.center_offset[0]),
            dy.mul_add(factor, self.center_offset[1]),
        ]
    }

    /// Apply inverse (undistort) using iterative Newton's method.
    #[must_use]
    pub fn undistort(&self, distorted: [f32; 2]) -> [f32; 2] {
        let mut uv = distorted;
        for _ in 0..8 {
            let d = self.distort(uv);
            uv[0] += distorted[0] - d[0];
            uv[1] += distorted[1] - d[1];
        }
        uv
    }

    /// Compute distorted UVs per color channel (chromatic aberration).
    #[must_use]
    pub fn distort_chromatic(&self, uv: [f32; 2]) -> [[f32; 2]; 3] {
        let mut result = [[0.0_f32; 2]; 3];
        for (i, ca_scale) in self.chromatic_aberration.iter().enumerate() {
            let scaled = [uv[0] * ca_scale, uv[1] * ca_scale];
            result[i] = self.distort(scaled);
        }
        result
    }

    /// Generate a distortion mesh grid for efficient GPU correction.
    /// Returns (positions, uvs) arrays for a grid of `resolution x resolution`.
    #[must_use]
    pub fn generate_distortion_mesh(&self, resolution: u32) -> (Vec<[f32; 2]>, Vec<[f32; 2]>) {
        let res = resolution as usize;
        let count = res * res;
        let mut positions = Vec::with_capacity(count);
        let mut uvs = Vec::with_capacity(count);
        let inv = 1.0 / (resolution - 1) as f32;
        for y in 0..res {
            for x in 0..res {
                let u = (x as f32 * inv).mul_add(2.0, -1.0);
                let v = (y as f32 * inv).mul_add(2.0, -1.0);
                positions.push([u, v]);
                uvs.push(self.distort([u, v]));
            }
        }
        (positions, uvs)
    }
}

// ---------------------------------------------------------------------------
// Controller Input
// ---------------------------------------------------------------------------

/// Button bit flags for VR controllers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ButtonFlags(pub u32);

impl ButtonFlags {
    pub const NONE: Self = Self(0);
    pub const A: Self = Self(1 << 0);
    pub const B: Self = Self(1 << 1);
    pub const X: Self = Self(1 << 2);
    pub const Y: Self = Self(1 << 3);
    pub const MENU: Self = Self(1 << 4);
    pub const SYSTEM: Self = Self(1 << 5);
    pub const TRIGGER: Self = Self(1 << 6);
    pub const GRIP: Self = Self(1 << 7);
    pub const THUMBSTICK_CLICK: Self = Self(1 << 8);
    pub const THUMBSTICK_TOUCH: Self = Self(1 << 9);
    pub const A_TOUCH: Self = Self(1 << 10);
    pub const B_TOUCH: Self = Self(1 << 11);
    pub const TRIGGER_TOUCH: Self = Self(1 << 12);

    #[must_use]
    pub const fn contains(self, flag: Self) -> bool {
        (self.0 & flag.0) == flag.0
    }

    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    #[must_use]
    pub const fn intersection(self, other: Self) -> Self {
        Self(self.0 & other.0)
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

/// Thumbstick axis state.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ThumbstickState {
    pub x: f32,
    pub y: f32,
}

impl ThumbstickState {
    #[must_use]
    pub const fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    /// Apply a circular deadzone.
    #[must_use]
    pub fn with_deadzone(self, deadzone: f32) -> Self {
        let mag = self.x.hypot(self.y);
        if mag < deadzone {
            return Self::new(0.0, 0.0);
        }
        let scale = (mag - deadzone) / (1.0 - deadzone) / mag;
        Self::new(self.x * scale, self.y * scale)
    }

    /// Magnitude of the thumbstick deflection.
    #[must_use]
    pub fn magnitude(self) -> f32 {
        self.x.hypot(self.y)
    }

    /// Angle in radians (atan2).
    #[must_use]
    pub fn angle(self) -> f32 {
        self.y.atan2(self.x)
    }
}

/// Haptic feedback request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HapticPulse {
    /// Duration in seconds.
    pub duration: f32,
    /// Frequency in Hz (0 = default).
    pub frequency: f32,
    /// Amplitude 0.0 to 1.0.
    pub amplitude: f32,
}

impl HapticPulse {
    #[must_use]
    pub const fn new(duration: f32, frequency: f32, amplitude: f32) -> Self {
        Self {
            duration,
            frequency,
            amplitude: if amplitude < 0.0 {
                0.0
            } else if amplitude > 1.0 {
                1.0
            } else {
                amplitude
            },
        }
    }

    /// Check if the pulse is still active given elapsed time.
    #[must_use]
    pub fn is_active(&self, elapsed: f32) -> bool {
        elapsed < self.duration
    }

    /// Get amplitude at a given time (with optional fade-out).
    #[must_use]
    pub fn amplitude_at(&self, elapsed: f32) -> f32 {
        if elapsed >= self.duration {
            return 0.0;
        }
        let fade = 1.0 - (elapsed / self.duration);
        self.amplitude * fade
    }
}

/// Handedness.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Hand {
    Left,
    Right,
}

/// Full state of one VR controller.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ControllerState {
    pub hand: Hand,
    pub pose: Pose,
    pub buttons: ButtonFlags,
    pub buttons_pressed: ButtonFlags,
    pub buttons_released: ButtonFlags,
    pub trigger: f32,
    pub grip: f32,
    pub thumbstick: ThumbstickState,
}

impl ControllerState {
    #[must_use]
    pub fn new(hand: Hand) -> Self {
        Self {
            hand,
            pose: Pose::default(),
            buttons: ButtonFlags::NONE,
            buttons_pressed: ButtonFlags::NONE,
            buttons_released: ButtonFlags::NONE,
            trigger: 0.0,
            grip: 0.0,
            thumbstick: ThumbstickState::default(),
        }
    }

    /// Update button state, computing pressed/released edges.
    pub const fn update_buttons(&mut self, new_buttons: ButtonFlags) {
        self.buttons_pressed = ButtonFlags(new_buttons.0 & !self.buttons.0);
        self.buttons_released = ButtonFlags(self.buttons.0 & !new_buttons.0);
        self.buttons = new_buttons;
    }

    /// Check if trigger is pulled past threshold.
    #[must_use]
    pub fn is_trigger_pulled(&self, threshold: f32) -> bool {
        self.trigger >= threshold
    }

    /// Check if grip is held past threshold.
    #[must_use]
    pub fn is_grip_held(&self, threshold: f32) -> bool {
        self.grip >= threshold
    }
}

/// Haptic engine that manages scheduled pulses per hand.
#[derive(Debug, Clone)]
pub struct HapticEngine {
    pub left_pulse: Option<(HapticPulse, f32)>,
    pub right_pulse: Option<(HapticPulse, f32)>,
}

impl Default for HapticEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl HapticEngine {
    #[must_use]
    pub const fn new() -> Self {
        Self {
            left_pulse: None,
            right_pulse: None,
        }
    }

    /// Submit a haptic pulse for a given hand.
    pub const fn submit(&mut self, hand: Hand, pulse: HapticPulse) {
        match hand {
            Hand::Left => self.left_pulse = Some((pulse, 0.0)),
            Hand::Right => self.right_pulse = Some((pulse, 0.0)),
        }
    }

    /// Advance time and return current amplitudes (left, right).
    pub fn update(&mut self, dt: f32) -> (f32, f32) {
        let left = Self::update_pulse(&mut self.left_pulse, dt);
        let right = Self::update_pulse(&mut self.right_pulse, dt);
        (left, right)
    }

    fn update_pulse(slot: &mut Option<(HapticPulse, f32)>, dt: f32) -> f32 {
        if let Some((pulse, elapsed)) = slot {
            *elapsed += dt;
            let amp = pulse.amplitude_at(*elapsed);
            if !pulse.is_active(*elapsed) {
                *slot = None;
            }
            amp
        } else {
            0.0
        }
    }

    /// Cancel all haptics.
    pub const fn cancel_all(&mut self) {
        self.left_pulse = None;
        self.right_pulse = None;
    }
}

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
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

// ---------------------------------------------------------------------------
// Comfort Metrics
// ---------------------------------------------------------------------------

/// Rolling statistics for comfort monitoring.
#[derive(Debug, Clone)]
pub struct ComfortMetrics {
    /// Ring buffer of frame times in seconds.
    frame_times: Vec<f32>,
    /// Current write index.
    index: usize,
    /// Number of samples collected.
    count: usize,
    /// Buffer capacity.
    capacity: usize,
    /// Target FPS.
    pub target_fps: f32,
    /// Total dropped frames.
    pub dropped_frames: u64,
}

impl ComfortMetrics {
    /// Create with a given window size and target FPS.
    #[must_use]
    pub fn new(window_size: usize, target_fps: f32) -> Self {
        Self {
            frame_times: vec![0.0; window_size],
            index: 0,
            count: 0,
            capacity: window_size,
            target_fps,
            dropped_frames: 0,
        }
    }

    /// Record a frame time in seconds.
    pub fn record_frame(&mut self, dt: f32) {
        self.frame_times[self.index] = dt;
        self.index = (self.index + 1) % self.capacity;
        if self.count < self.capacity {
            self.count += 1;
        }
        let budget = 1.0 / self.target_fps;
        if dt > budget * 1.5 {
            self.dropped_frames += 1;
        }
    }

    /// Average frame time over the window.
    #[must_use]
    pub fn average_frame_time(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let sum: f32 = self.frame_times[..self.count].iter().sum();
        sum / self.count as f32
    }

    /// Current average FPS.
    #[must_use]
    pub fn fps(&self) -> f32 {
        let avg = self.average_frame_time();
        if avg < 1e-10 {
            return 0.0;
        }
        1.0 / avg
    }

    /// 99th percentile frame time (approximate).
    #[must_use]
    pub fn percentile_99(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        let mut sorted: Vec<f32> = self.frame_times[..self.count].to_vec();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap_or(core::cmp::Ordering::Equal));
        let idx = ((self.count as f32) * 0.99) as usize;
        sorted[idx.min(self.count - 1)]
    }

    /// Maximum frame time in the window.
    #[must_use]
    pub fn max_frame_time(&self) -> f32 {
        self.frame_times[..self.count]
            .iter()
            .copied()
            .fold(0.0_f32, f32::max)
    }

    /// Minimum frame time in the window.
    #[must_use]
    pub fn min_frame_time(&self) -> f32 {
        if self.count == 0 {
            return 0.0;
        }
        self.frame_times[..self.count]
            .iter()
            .copied()
            .fold(f32::MAX, f32::min)
    }

    /// Motion-to-photon latency estimate (`frame_time` + half vsync).
    #[must_use]
    pub fn estimated_latency(&self) -> f32 {
        let avg = self.average_frame_time();
        let vsync = 1.0 / self.target_fps;
        avg + vsync * 0.5
    }

    /// Comfort score: 1.0 = perfect, 0.0 = terrible.
    #[must_use]
    pub fn comfort_score(&self) -> f32 {
        let fps = self.fps();
        if fps < 1.0 {
            return 0.0;
        }
        let ratio = (fps / self.target_fps).min(1.0);
        let consistency = if self.count > 1 {
            let avg = self.average_frame_time();
            let max = self.max_frame_time();
            if max > 1e-10 {
                (avg / max).min(1.0)
            } else {
                1.0
            }
        } else {
            1.0
        };
        ratio.mul_add(0.7, consistency * 0.3)
    }

    /// Whether the system is in a comfortable state.
    #[must_use]
    pub fn is_comfortable(&self) -> bool {
        self.comfort_score() > 0.85
    }

    /// Reset all metrics.
    pub fn reset(&mut self) {
        self.frame_times.fill(0.0);
        self.index = 0;
        self.count = 0;
        self.dropped_frames = 0;
    }

    /// Total number of frames recorded.
    #[must_use]
    pub const fn total_frames(&self) -> usize {
        self.count
    }
}

// ---------------------------------------------------------------------------
// Guardian / Boundary System
// ---------------------------------------------------------------------------

/// Helper: compute the squared distance from point to a line segment,
/// and the projection parameter t. Also returns the winding test data.
fn segment_point_dist_sq(
    seg_a: [f32; 2],
    seg_b: [f32; 2],
    point: [f32; 2],
) -> (f32, f32, f32, f32) {
    let ex = seg_b[0] - seg_a[0];
    let ey = seg_b[1] - seg_a[1];
    let wx = point[0] - seg_a[0];
    let wy = point[1] - seg_a[1];
    let edge_len_sq = ex.mul_add(ex, ey * ey);
    let t = if edge_len_sq < 1e-10 {
        0.0
    } else {
        (wx.mul_add(ex, wy * ey) / edge_len_sq).clamp(0.0, 1.0)
    };
    let dx = ex.mul_add(-t, wx);
    let dy = ey.mul_add(-t, wy);
    let dist_sq = dx.mul_add(dx, dy * dy);
    let cross = ex.mul_add(wy, -(ey * wx));
    (dist_sq, t, cross, 0.0)
}

/// A 2D boundary polygon (in XZ plane, Y is up).
#[derive(Debug, Clone)]
pub struct GuardianBoundary {
    /// Boundary vertices in XZ plane (closed polygon).
    pub vertices: Vec<[f32; 2]>,
    /// Warning distance (start showing boundary).
    pub warning_distance: f32,
    /// Hard boundary distance (full visual).
    pub hard_distance: f32,
    /// Floor height (Y).
    pub floor_y: f32,
    /// Ceiling height (Y).
    pub ceiling_y: f32,
}

impl GuardianBoundary {
    /// Create a rectangular boundary centered at origin.
    #[must_use]
    pub fn rectangular(width: f32, depth: f32) -> Self {
        let hw = width * 0.5;
        let hd = depth * 0.5;
        Self {
            vertices: vec![[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]],
            warning_distance: 0.5,
            hard_distance: 0.1,
            floor_y: 0.0,
            ceiling_y: 3.0,
        }
    }

    /// Create a circular boundary.
    #[must_use]
    pub fn circular(radius: f32, segments: u32) -> Self {
        let mut vertices = Vec::with_capacity(segments as usize);
        for i in 0..segments {
            let angle = 2.0 * PI * i as f32 / segments as f32;
            vertices.push([angle.cos() * radius, angle.sin() * radius]);
        }
        Self {
            vertices,
            warning_distance: 0.5,
            hard_distance: 0.1,
            floor_y: 0.0,
            ceiling_y: 3.0,
        }
    }

    /// Signed distance from a point (XZ) to the boundary polygon.
    /// Negative = inside, positive = outside.
    #[must_use]
    pub fn signed_distance(&self, point: [f32; 2]) -> f32 {
        if self.vertices.len() < 3 {
            return f32::MAX;
        }
        let n = self.vertices.len();
        let mut min_dist_sq = f32::MAX;
        let mut sign = 1.0_f32;

        for i in 0..n {
            let j = (i + 1) % n;
            let (dist_sq, _t, cross, _) =
                segment_point_dist_sq(self.vertices[i], self.vertices[j], point);
            min_dist_sq = min_dist_sq.min(dist_sq);

            // Winding number test
            let c1 = self.vertices[i][1] <= point[1];
            let c2 = self.vertices[j][1] <= point[1];
            if c1 != c2 && ((c2 && cross > 0.0) || (!c2 && cross < 0.0)) {
                sign = -sign;
            }
        }

        sign * min_dist_sq.sqrt()
    }

    /// Check if a position is within the boundary.
    #[must_use]
    pub fn is_inside(&self, position: Vec3) -> bool {
        if position.y < self.floor_y || position.y > self.ceiling_y {
            return false;
        }
        self.signed_distance([position.x, position.z]) < 0.0
    }

    /// Get the proximity level: 0.0 = safe, 1.0 = at boundary.
    #[must_use]
    pub fn proximity(&self, position: Vec3) -> f32 {
        let dist = -self.signed_distance([position.x, position.z]);
        if dist <= self.hard_distance {
            return 1.0;
        }
        if dist >= self.warning_distance {
            return 0.0;
        }
        1.0 - (dist - self.hard_distance) / (self.warning_distance - self.hard_distance)
    }

    /// Find the closest point on the boundary to a given XZ point.
    #[must_use]
    pub fn closest_point(&self, point: [f32; 2]) -> [f32; 2] {
        if self.vertices.len() < 2 {
            return point;
        }
        let n = self.vertices.len();
        let mut best = point;
        let mut best_dist = f32::MAX;
        for i in 0..n {
            let j = (i + 1) % n;
            let (dist_sq, t, _, _) =
                segment_point_dist_sq(self.vertices[i], self.vertices[j], point);
            if dist_sq < best_dist {
                best_dist = dist_sq;
                let ex = self.vertices[j][0] - self.vertices[i][0];
                let ey = self.vertices[j][1] - self.vertices[i][1];
                best = [
                    ex.mul_add(t, self.vertices[i][0]),
                    ey.mul_add(t, self.vertices[i][1]),
                ];
            }
        }
        best
    }

    /// Area of the boundary polygon.
    #[must_use]
    pub fn area(&self) -> f32 {
        let n = self.vertices.len();
        if n < 3 {
            return 0.0;
        }
        let mut sum = 0.0_f32;
        for i in 0..n {
            let j = (i + 1) % n;
            sum += self.vertices[i][0].mul_add(
                self.vertices[j][1],
                -(self.vertices[j][0] * self.vertices[i][1]),
            );
        }
        sum.abs() * 0.5
    }

    /// Perimeter of the boundary polygon.
    #[must_use]
    pub fn perimeter(&self) -> f32 {
        let n = self.vertices.len();
        if n < 2 {
            return 0.0;
        }
        let mut total = 0.0_f32;
        for i in 0..n {
            let j = (i + 1) % n;
            let dx = self.vertices[j][0] - self.vertices[i][0];
            let dy = self.vertices[j][1] - self.vertices[i][1];
            total += dx.hypot(dy);
        }
        total
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
        let correction =
            rp.positional_correction(Pose::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY));
        assert_eq!(correction, Vec3::ZERO);
    }

    #[test]
    fn reprojection_positional_correction_asw() {
        let mut rp = Reprojection::new(ReprojectionMode::Asw, 90.0);
        rp.on_frame_rendered(Pose::default());
        let correction =
            rp.positional_correction(Pose::new(Vec3::new(1.0, 0.0, 0.0), Quat::IDENTITY));
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
}
