//! Controller input (`ButtonFlags` / `ThumbstickState` / `HapticPulse` / `Hand` / `ControllerState` / `HapticEngine`).

use crate::pose::Pose;

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
