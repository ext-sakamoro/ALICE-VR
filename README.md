**English** | [日本語](README_JP.md)

# ALICE-VR

VR runtime for the A.L.I.C.E. ecosystem. Provides head tracking, lens distortion, controller input, stereo rendering, and comfort systems in pure Rust.

## Features

- **Head Tracking** — 6DOF tracking with quaternion integration, Euler angle conversion
- **Lens Distortion** — Barrel distortion correction, chromatic aberration compensation
- **Controller Input** — Button/trigger state, thumbstick axes, haptic feedback
- **Stereo Rendering** — IPD-aware eye matrices, asymmetric frustum projection
- **Reprojection** — Asynchronous Spacewarp (ASW) and Asynchronous Timewarp (ATW)
- **Comfort Metrics** — FPS monitoring, motion-to-photon latency tracking
- **Guardian System** — Boundary definition, proximity detection, fade warnings

## Architecture

```
VR Runtime
  │
  ├── Quat / Vec3       — Quaternion math, 3D vectors
  ├── HeadTracker        — 6DOF pose integration
  ├── LensDistortion     — Barrel distortion, chromatic aberration
  ├── Controller         — Input state, haptics
  ├── StereoRenderer     — Eye matrices, projection
  ├── Reprojection       — ASW / ATW frame synthesis
  ├── ComfortMonitor     — FPS, latency metrics
  └── Guardian           — Boundary system
```

## Usage

```rust
use alice_vr::Quat;

let rotation = Quat::from_axis_angle([0.0, 1.0, 0.0], 1.57);
let combined = rotation * Quat::IDENTITY;
```

## License

MIT OR Apache-2.0
