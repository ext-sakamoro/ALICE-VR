[English](README.md) | **日本語**

# ALICE-VR

A.L.I.C.E. エコシステム向けVRランタイム。ヘッドトラッキング、レンズ歪み補正、コントローラ入力、ステレオレンダリング、快適性システムを純Rustで提供。

## 機能

- **ヘッドトラッキング** — クォータニオン積分による6DOFトラッキング、オイラー角変換
- **レンズ歪み補正** — バレルディストーション補正、色収差補償
- **コントローラ入力** — ボタン/トリガー状態、サムスティック軸、ハプティクスフィードバック
- **ステレオレンダリング** — IPD対応アイマトリクス、非対称錐台プロジェクション
- **リプロジェクション** — ASW（非同期スペースワープ）/ ATW（非同期タイムワープ）
- **快適性メトリクス** — FPS監視、モーション・トゥ・フォトンレイテンシ追跡
- **ガーディアンシステム** — 境界定義、近接検出、フェード警告

## アーキテクチャ

```
VRランタイム
  │
  ├── Quat / Vec3       — クォータニオン演算、3Dベクトル
  ├── HeadTracker        — 6DOF姿勢積分
  ├── LensDistortion     — バレルディストーション、色収差
  ├── Controller         — 入力状態、ハプティクス
  ├── StereoRenderer     — アイマトリクス、プロジェクション
  ├── Reprojection       — ASW / ATW フレーム合成
  ├── ComfortMonitor     — FPS、レイテンシメトリクス
  └── Guardian           — 境界システム
```

## 使用例

```rust
use alice_vr::Quat;

let rotation = Quat::from_axis_angle([0.0, 1.0, 0.0], 1.57);
let combined = rotation * Quat::IDENTITY;
```

## ライセンス

MIT OR Apache-2.0
