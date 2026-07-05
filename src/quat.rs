//! `Quat` — quaternion (rotation representation).

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
