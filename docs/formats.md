# Data formats & conventions

These are the contracts between the three parts of the project:

* `apps/iphone/` (ARKit capture app) **writes** an ARKit capture (§2).
* The Python stack (`capture/`, `core/`, `engine/`, `models/`, `eval/`) **reads** captures and logs, and **writes** scene checkpoints (§3) and rollouts (§4).
* `apps/macbook/` (native Metal viewer / trajectory editor) **reads** scene checkpoints and rollouts, and **writes** edited scene graphs (§3.3).

## 1. Conventions

| Thing | Convention |
| --- | --- |
| Units | meters, seconds, radians |
| World frame (sim) | right-handed, **z-up**. For driving scenes, +x = initial ego heading, +y = left |
| Camera frame (sim) | **OpenCV**: +x right, +y down, +z forward (optical axis) |
| Poses | `camera_to_world` / `actor_to_world`, 4x4 row-major nested lists, or `pos[3]` + `quat[4]` |
| Quaternions | **w, x, y, z** (Hamilton), not necessarily normalized on disk |
| Time | seconds, float64 in JSON; `t = 0` is the first frame of the log |
| Image coords | pixel centers at integer + 0.5; intrinsics `fx, fy, cx, cy` in pixels of the stored image |

## 2. ARKit capture (`apps/iphone` → `capture/arkit.py`)

A capture is a directory (or a `.zip` of that directory) named `capture_YYYYMMDD_HHMMSS/`:

```text
capture_20261124_101500/
├── capture.json
├── images/000000.jpg ...        # ARFrame.capturedImage, sensor (landscape) orientation, JPEG q≈0.9
├── depth/000000.png ...         # optional, LiDAR sceneDepth, 16-bit grayscale PNG, millimeters, 0 = invalid
├── confidence/000000.png ...    # optional, 8-bit PNG, ARConfidenceLevel 0/1/2
└── points.ply                   # optional, accumulated ARKit rawFeaturePoints (world frame), ascii PLY: x y z
```

`capture.json` (ARKit-native conventions — the loader converts them):

```json
{
  "format": "m4dgs-arkit-capture",
  "version": 1,
  "device_model": "iPhone17,3",
  "system_version": "18.2",
  "created_at": "2026-11-24T10:15:00Z",
  "world_convention": "arkit",          // ARKit world: gravity-aligned, +y up, right-handed
  "camera_convention": "arkit",         // ARKit camera: +x right, +y up, +z backward (toward viewer), landscape-right sensor image
  "image_width": 1920, "image_height": 1440,
  "depth_width": 256, "depth_height": 192,   // omitted / null if no LiDAR
  "has_lidar": true,
  "frames": [
    {
      "index": 0,
      "timestamp": 0.0,                        // seconds since first frame (ARFrame.timestamp - t0)
      "image": "images/000000.jpg",
      "depth": "depth/000000.png",             // or null
      "confidence": "confidence/000000.png",   // or null
      "camera_to_world": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],   // ARFrame.camera.transform, row-major
      "intrinsics": [[1450.0,0,960.0],[0,1450.0,720.0],[0,0,1]],      // scaled to image_width x image_height
      "tracking_state": "normal",              // "normal" | "limited:<reason>" | "notAvailable"
      "exposure_duration": 0.008,
      "motion_blur_score": 0.0                 // optional, angular speed rad/s at capture time
    }
  ]
}
```

`camera_to_world` is exactly `ARFrame.camera.transform` (simd column-major → written row-major). The loader
converts ARKit camera axes to OpenCV with `diag(1,-1,-1)` on the right, and ARKit world (+y up) to sim world (+z up)
with the rotation `x_sim = -z_ark, y_sim = -x_ark, z_sim = y_ark`.

## 3. Scene checkpoint (`*.safetensors`)

A standard [safetensors](https://github.com/huggingface/safetensors) file: 8-byte little-endian header length, a JSON
header, then raw little-endian tensor data. All tensors are `F32`.

### 3.1 Tensors — one group per scene-graph node

For node name `<n>` (e.g. `background`, `actor_2`), with `N` Gaussians and SH degree `d` (`K = (d+1)^2`):

| Key | Shape | Meaning |
| --- | --- | --- |
| `<n>.means` | (N,3) | mean, in **node-local** frame (world for background) |
| `<n>.log_scales` | (N,3) | log of per-axis std-dev |
| `<n>.quats` | (N,4) | rotation, wxyz, normalize before use |
| `<n>.opacity_logits` | (N,1) | `alpha = sigmoid(x)` |
| `<n>.sh` | (N,K,3) | SH coefficients; `rgb = 0.5 + Σ_k SH_k(dir) · sh[k]`, clamp ≥ 0 (3DGS convention, `C0 = 0.28209479`) |
| `<n>.velocity` | (N,3) | linear velocity in node-local frame (m/s) |
| `<n>.t_center` | (N,1) | temporal center τ (s) |
| `<n>.t_log_scale` | (N,1) | log temporal std-dev σ_t (s) |

4D evaluation of one Gaussian at time `t`:

```text
μ_local(t) = means + velocity · (t − τ)
α(t)       = sigmoid(opacity_logit) · exp(−½ ((t − τ)/exp(t_log_scale))²)
node pose  = T_node(t)  (identity for background; interpolated track for actors)
μ_world    = R_node(t) μ_local(t) + p_node(t)
q_world    = q_node(t) ⊗ q
```

SH view direction is `normalize(μ_world − camera_center)` rotated into the node-local frame (`R_node(t)^T · dir`).

### 3.2 Metadata

The safetensors `__metadata__` map contains `"format": "m4dgs-scene"`, `"version": "1"`, and `"scene_graph"`: a JSON string:

```json
{
  "version": 1,
  "sh_degree": 1,
  "time_range": [0.0, 8.0],
  "nodes": [
    { "name": "background", "kind": "background" },
    { "name": "actor_2", "kind": "actor", "category": "vehicle.car",
      "size": [4.5, 1.9, 1.6],                      // box l, w, h in node frame (x fwd, y left, z up), box centered at origin
      "track": [ { "t": 0.0, "pos": [12.0, 3.5, 0.8], "quat": [1,0,0,0] }, ... ] }
  ],
  "ego": {
    "size": [4.5, 1.9, 1.6],
    "camera_to_ego": [[...4x4 row-major...]],      // OpenCV camera in ego frame
    "camera": { "width": 512, "height": 288, "fx": 300.0, "fy": 300.0, "cx": 256.0, "cy": 144.0 },
    "track": [ { "t": 0.0, "pos": [0,0,0], "quat": [1,0,0,0] }, ... ]   // logged ego pose (ego_to_world)
  }
}
```

Track interpolation: linear in `pos`, slerp in `quat`, clamped (constant) outside the track's time range.

Optional fields (readers must tolerate their absence):

* top-level `"background_color": [r, g, b]` (0–1) — colour composited behind the Gaussians (`rgb + (1 − α)·bg`). Default black.
* per-actor `"visible_range": [t0, t1]` — the actor exists only for `t0 ≤ t ≤ t1` (its Gaussians are skipped outside). Default: always visible.

### 3.3 Edited scene graph sidecar

The Mac app never rewrites tensors. It saves edits as `<scene>.edits.json` next to the checkpoint containing a full
replacement `scene_graph` object (same schema as §3.2). The Python loader applies it automatically if present
(`--edits` to point elsewhere, `--no-edits` to ignore).

## 4. Rollout log (`*.rollout.json`, written by `eval/closed_loop`)

```json
{
  "format": "m4dgs-rollout", "version": 1,
  "scene": "checkpoints/scene_01.safetensors",
  "policy": "pilotnet",
  "dt": 0.1,
  "perturbations": ["actor_2:lateral_offset=-1.8m,t=2.5s"],
  "actor_tracks": { "actor_2": [ { "t": 0.0, "pos": [..], "quat": [..] } ] },   // tracks as simulated (after perturbation)
  "steps": [
    { "t": 0.0, "ego": { "pos": [0,0,0], "quat": [1,0,0,0] }, "speed": 8.0,
      "action": { "steer": 0.0, "accel": 0.0 },
      "latency_ms": { "raster": 3.1, "refine": 18.0, "policy": 1.2 },
      "events": [] }                                         // e.g. "collision:actor_2", "offroad"
  ],
  "metrics": { "collision": false, "min_ttc": 3.2, "max_lateral_error": 0.4, "route_completion": 1.0 }
}
```

## 5. Generic log (`log.json`, written by converters / the synthetic generator, read by `capture/dataset.py`)

All poses in the sim conventions of §1 (z-up world, OpenCV cameras).

```json
{
  "format": "m4dgs-log", "version": 1, "source": "synthetic | nuscenes | waymo",
  "frames": [ { "t": 0.0, "image": "images/000000.png", "depth": "depth/000000.npy",       // depth optional (.npy metres or 16-bit PNG mm)
                "camera_to_world": [[...4x4...]], "intrinsics": [[fx,0,cx],[0,fy,cy],[0,0,1]], "width": 512, "height": 288 } ],
  "ego_track": [ { "t": 0.0, "pos": [..], "quat": [..] } ],   // optional (handheld captures: derived from cameras)
  "camera_to_ego": [[...4x4...]],                             // optional
  "actors": [ { "name": "actor_1", "category": "vehicle.car", "size": [l,w,h], "track": [...], "visible_range": [t0,t1] } ],
  "points": "points.npy",                                     // optional (M,7) float32: x y z r g b t_observed (t may be NaN)
  "background_color": [r,g,b]                                  // optional
}
```

## 6. Live link (`apps/iphone` ⇄ `apps/macbook`, Wi-Fi)

The capture app streams keyframes to M4DGS Studio while recording. The Mac re-assembles **exactly the §2 capture
directory**, so everything downstream (`train_4dgs.py`) is unchanged. The phone still writes its local copy as well.

**Discovery / transport.** The Mac app advertises a Bonjour service `_m4dgs-live._tcp` (instance name = the Mac's
name) with an `NWListener`; the phone browses with `NWBrowser`, the user picks a Mac, and it opens one TCP connection
per capture. iOS needs `NSLocalNetworkUsageDescription` and `NSBonjourServices = ["_m4dgs-live._tcp"]`.

**Framing.** Every message, both directions:

```text
offset 0   4 bytes  magic "M4DG"
offset 4   1 byte   type
offset 5   3 bytes  reserved (0)
offset 8   uint32   header_len  (big-endian)
offset 12  uint32   payload_len (big-endian)
offset 16  header_len bytes   UTF-8 JSON header
           payload_len bytes  binary payload
```

| type | dir | header | payload |
| --- | --- | --- | --- |
| 1 `hello` | phone→Mac | `{"protocol":1,"capture_name":"capture_YYYYMMDD_HHMMSS","device_model","system_version","image_width","image_height","depth_width","depth_height","has_lidar"}` | — |
| 2 `frame` | phone→Mac | the §2 frame object (`index`, `timestamp`, `image`, `depth`, `confidence`, `camera_to_world`, `intrinsics`, `tracking_state`, `exposure_duration`, `motion_blur_score`) plus `"image_bytes"`, `"depth_bytes"`, `"confidence_bytes"` (0 when absent) | JPEG ‖ depth PNG ‖ confidence PNG, concatenated in that order — byte-identical to the files written locally |
| 3 `points` | phone→Mac | `{"count": n}` | the `points.ply` file bytes |
| 4 `end` | phone→Mac | the complete `capture.json` document | — |
| 5 `status` | Mac→phone | `{"frames_received": n, "bytes_received": b}` (after every frame) | — |
| 6 `cancel` | phone→Mac | `{}` — recording discarded; Mac deletes the partial directory | — |
| 7 `saved` | Mac→phone | `{"path": "<absolute path on the Mac>", "frames": n}` after `end` is written | — |

The Mac writes files to `<captures folder>/<capture_name>/` as frames arrive (paths taken from the frame header) and
writes `capture.json` + `points.ply` on `end`. A connection that drops before `end` leaves the directory with a
`capture.partial.json` (hello + frames so far) so it can still be inspected; it is not loadable as a capture.
