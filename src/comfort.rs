//! Comfort metrics (`ComfortMetrics` — FPS / latency).

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
