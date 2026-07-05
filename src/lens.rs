//! Lens distortion correction (`LensDistortion`).

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
