//! `Mat4` — 4x4 matrix.

use crate::quat::Quat;
use crate::vec3::Vec3;

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
