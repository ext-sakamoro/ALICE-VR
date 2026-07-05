//! Guardian / boundary system (`GuardianBoundary`).

use crate::vec3::Vec3;
use core::f32::consts::PI;

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
