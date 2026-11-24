# M4DGS Studio (macOS)

Native SwiftUI + Metal viewer and programmatic trajectory editor for `metal-4dgs-sim` 4DGS scene
checkpoints (`*.safetensors`, [docs/formats.md](../../docs/formats.md) §3) and closed-loop rollouts
(`*.rollout.json`, §4). macOS 15+, Apple Silicon.

* Real-time 4D Gaussian rendering at the current time *t* (per-node track interpolation, per-Gaussian
  velocity and temporal opacity, SH degree 0–3, EWA splatting, GPU depth sort).
* Ego camera (logged ego track ∘ `camera_to_ego`, scene intrinsics) and a free orbit/fly camera.
* Timeline with play/pause/scrub/speed, overlays for actor boxes, actor tracks and the ego track.
* Trajectory editor (lateral / longitudinal offset, time shift, speed scale, smooth cut-in), saved as
  `<scene>.edits.json` for the Python stack.
* Rollout playback with speed / steer / accel, events (collisions highlighted), metrics and per-stage latency.
* Headless CLI: render a frame to PNG and benchmark GPU time.
* **Live Capture**: receives a capture streamed from the iPhone app over Wi-Fi (§6 live link), shows it
  arriving, and reconstructs it with the Python pipeline in one click (see [Live capture](#live-capture-iphone--mac)).

## Build & run

```bash
cd apps/macbook
swift build                 # debug build
swift test                  # unit tests (core math + GPU sort/projection tests)
./build_app.sh              # release build wrapped into build/M4DGS Studio.app

# test fixture (numpy only): ~228k Gaussians + a matching rollout
../../.venv/bin/python Tools/make_test_scene.py            # → Fixtures/test_scene.safetensors, .rollout.json
../../.venv/bin/python Tools/make_test_scene.py --gaussians 290000 --out Fixtures/test_scene_300k.safetensors

# GUI (optionally open a scene and a rollout at launch)
swift run M4DGSStudio Fixtures/test_scene.safetensors --rollout Fixtures/test_scene.rollout.json
open "build/M4DGS Studio.app" --args "$PWD/Fixtures/test_scene.safetensors"
```

Shaders are embedded as a Swift string (`Sources/M4DGSRender/Shaders.swift`) and compiled at launch with
`MTLDevice.makeLibrary(source:options:)`, so the offline Metal toolchain (`xcrun metal`) is not needed.

### Headless rendering / benchmark

```bash
.build/release/M4DGSStudio --render Fixtures/test_scene.safetensors --time 2.0 --out frame.png --bench 200
```

| Flag | Meaning |
| --- | --- |
| `--render <scene>` | scene checkpoint (applies `<scene>.edits.json` if present) |
| `--time T` | scene time in seconds (default: start of `time_range`) |
| `--out file.png` | write the ego view (rgba16Float target → 8-bit PNG) |
| `--width W --height H` | target size (default: the ego camera's native size, e.g. 512×288) |
| `--bench N` | N timed frames: P50/P90/min GPU time (`gpuEndTime − gpuStartTime`) for whole frames, a per-stage breakdown, and sustained back-to-back throughput |
| `--overlays` | draw actor boxes / tracks / ego track |
| `--orbit yaw,pitch,dist` | free camera (degrees, metres) around a point 8 m ahead of the ego instead of the ego camera |
| `--rollout file.rollout.json` | ego from rollout steps, actors from `actor_tracks` |
| `--no-edits`, `--edits file` | ignore the sidecar / use another one |
| `--perturb "actor_1:lateral_offset=3.5m,t=1.0s;…"` | apply edits from perturbation strings (same syntax as `eval/closed_loop`) |
| `--save-edits` | with `--perturb`: write the result as `<scene>.edits.json` |

## Live capture (iPhone → Mac)

The Mac side of the live link in [docs/formats.md](../../docs/formats.md) §6. **Live → Live Capture…** (⌘L, or the
toolbar button; `--live` opens it at launch) starts an `NWListener` that advertises Bonjour `_m4dgs-live._tcp` under
this Mac's name. In M4DGS Capture on the phone, pick the Mac and record. The panel shows:

* listening state, port and Bonjour name (the switch stops/starts the receiver), and the captures folder,
* the connected device (`hello`), a live thumbnail of the latest JPEG, frames / bytes received, and a top-down
  plot of the camera centres so far (ARKit x → right, z ↓, orange arrow = current heading, scale bar in metres),
* finished captures (plus **Add Folder…** for any existing capture directory) and **Reconstruct**.

Each connection is one capture, written to `<captures folder>/<capture_name>/` exactly as §2 while frames
arrive (JPEG/PNG bytes verbatim, paths from the frame header). The Mac replies `status` after every frame and
`saved` after `end` (`capture.json` written verbatim + `points.ply`). `cancel` deletes the directory; a dropped
connection leaves `capture.partial.json` (`{"format":"m4dgs-arkit-capture-partial","hello":…,"frames":[…]}`,
also refreshed every 10 frames) so the files can be inspected — it is not loadable as a capture. An existing
directory of the same name is never overwritten (`_1`, `_2`, … suffix; `saved.path` says where it went).

Input is treated as untrusted: `capture_name` must be a plain directory name (no `/`, `\`, `..`, leading `.`,
control characters, ≤128 chars); frame paths must be exactly `images|depth|confidence/<name>` with
`[A-Za-z0-9_.-]` names (anything else — `..`, absolute paths, extra components, unicode slashes — is rejected,
not rewritten, and the connection is closed); the same check applies to paths inside the final `capture.json`.
Headers are capped at 32 MiB, payloads at 512 MiB, a capture at 100k frames / 64 GiB; the three `*_bytes`
must add up to `payload_len`; bad magic is detected from the first 4 bytes.

**Reconstruct** runs, from the repo root,
`<repo>/.venv/bin/python <repo>/scripts/train_4dgs.py --data_dir <capture> --iterations N --output <repo>/checkpoints/<capture_name>.safetensors`
(N defaults to 7000), streams stdout/stderr into the log view, shows progress / loss / PSNR / Gaussian count parsed
from `[train] it …` lines, **Cancel** sends SIGTERM (SIGKILL after 5 s), and on success opens the checkpoint in
the viewer.

**Settings** (⌘,), stored in UserDefaults (domain `dev.m4dgs.studio`, shared by the .app and the bare binary;
empty = default): *repo root* — default: nearest ancestor of the executable, else of the working directory,
containing `scripts/train_4dgs.py`; *captures folder* — default `<repo>/data/live`, or `~/Documents/M4DGS
Captures` without a repo; *iterations*. Any of them can be overridden for one run with Cocoa argument syntax,
e.g. `open "build/M4DGS Studio.app" --args --live -capturesFolder /tmp/live`.

The first time the .app listens, macOS asks to allow **local network** access (and, if the application firewall
is on, incoming connections) — allow both, or the phone will not see the Mac. `build_app.sh` adds
`NSLocalNetworkUsageDescription` / `NSBonjourServices` to the Info.plist.

### Headless receiver / reconstruction

```bash
.build/debug/M4DGSStudio --receive [--out DIR] [--once] [--port P] [--name "Bonjour name"]
.build/debug/M4DGSStudio --reconstruct <capture dir> [--iterations N] [--output ckpt.safetensors] [--repo DIR]
```

`--receive` logs every event and prints `LISTENING_PORT=…` / `SAVED_CAPTURE=…`; `--once` exits after the first
connection ends (0 if it was saved). Ctrl-C closes open connections (leaving `capture.partial.json`).
`--reconstruct` is the same job the Reconstruct button runs; it prints `CHECKPOINT=…` on success.

### Testing without a phone: `Tools/fake_iphone_stream.py`

```bash
.build/debug/M4DGSStudio --receive --out /tmp/live --once &
../../.venv/bin/python Tools/fake_iphone_stream.py                  # Bonjour discovery (dns-sd -B / -L), 30 synthetic frames
../../.venv/bin/python Tools/fake_iphone_stream.py --host 127.0.0.1 --port 50123 --save-local /tmp/phone_copy/cap
../../.venv/bin/python Tools/fake_iphone_stream.py --replay ../../data/iphone/capture_20261124_101500
```

Without `--replay` it synthesizes an ARKit capture from `data/sample_scene` (`--frames`, default 30): JPEG q90
images, poses converted from sim/OpenCV back to ARKit (`ARKIT_WORLD_TO_SIM.T @ c2w_sim @ ARKIT_CAM_TO_CV`, the
inverse of `capture/arkit.py`), 256-wide 16-bit depth + confidence PNGs and a `points.ply` (20k points of
`points.npy`). Writes are split into random chunks of ≤ `--chunk-max` bytes that straddle message boundaries.
`--drop-after K` / `--cancel-after K` exercise the partial / cancel paths; `--save-local` writes the phone's
copy so it can be `diff -r`'d against the received directory. Exit status 0 iff `saved` arrived and every
frame got a `status`.

Verified on this Mac (M4): the received directory is byte-identical to the sender's local copy;
`load_arkit_capture` reads 30 frames and every `camera_to_world` equals the `log.json` source exactly; a
300-iteration `--reconstruct` (≈17 s, held-out PSNR 20.8 dB) produced a checkpoint that `--render` loads and
renders. Not yet verified with the real iPhone app over Wi-Fi.

## Controls

| Input | Action |
| --- | --- |
| drag | orbit (switches from ego to free camera, starting at the ego view) |
| ⌥/⇧-drag, right-drag | pan |
| scroll / pinch | zoom |
| W A S D / Q E (⇧ = faster) | move the free camera's target (fly) |
| click | select the actor under the cursor |
| space, ← / → | play/pause, step one frame (0.1 s, or the rollout `dt`) |
| 1 / 2, ⌘1 / ⌘2 | ego / free camera |
| F | toggle "follow ego" for the free camera |
| ⌘O, ⇧⌘O, ⌘S | open scene, open rollout, save edits |

The HUD shows the camera, time, the GPU time of the frame (P50 over ~0.25 s, from the command buffer's
`gpuStartTime`/`gpuEndTime`), visible/total splats and fps.

## Editing workflow

1. Open a scene. If `<scene>.edits.json` exists next to it, it is applied (the sidebar shows **Edits: …**).
2. Select an actor in the node list (or click it in the viewport). The editor applies, on top of that track:
   * **Lateral offset** (m, along the actor's own left axis at each instant), **Longitudinal** (m, forward axis),
   * **Time shift** (s, positive = arrives later) and **Speed scale** (×, same path, about the track's first key),
   * **Cut-in**: an extra lateral offset blended in with a smootherstep over `[start, start + duration]`;
     *Align heading to path* yaws the actor along the lateral motion.
   The edited track is shown live (yellow for the selected actor, the pre-edit track in grey).
   Non-identity edits are resampled every 0.1 s over the scene `time_range`, so the lerp/slerp readers use
   reproduces the edit exactly; times outside the source track clamp, as in §3.2.
3. **Save** (⌘S) writes `<scene>.edits.json` (`scene_01.safetensors` → `scene_01.edits.json`; an existing
   `scene_01.safetensors.edits.json` is also recognised and written back in place).
   **Reset actor / Reset all** drop the in-app edits; **Revert** reloads the checkpoint *without* the sidecar
   (the file is left on disk; saving after a revert overwrites it).

### How edits flow back to Python

The app never rewrites tensors. `<scene>.edits.json` is a **full replacement `scene_graph`** (same schema as
§3.2): the scene graph the app loaded, with the `track` arrays of edited actors replaced. Every other field is
copied verbatim from the loaded JSON, including fields the app doesn't interpret (`background_color`,
`visible_range`, `category`, `size`, `ego`, unknown keys). The Python loader picks the sidecar up
automatically (`--edits` to point elsewhere, `--no-edits` to ignore it), so `run_sim.py --scene
checkpoints/scene_01.safetensors` replays the edited scenario. The edit sliders themselves are not persisted:
reopening a scene shows the saved tracks as the new baseline.

## Rendering (docs/formats.md §3.1)

Per frame, one command buffer:

1. **preprocess** (compute, 1 thread per Gaussian): `μ_local(t) = means + velocity·(t − τ)`,
   `α(t) = sigmoid(logit)·exp(−½((t−τ)/σ_t)²)`; node pose from the interpolated track (lerp / slerp / clamp,
   evaluated on the CPU per node); nodes outside `visible_range` are skipped; `q_world = q_node ⊗ q`,
   `Σ = R S Sᵀ Rᵀ`; EWA projection `Σ2D = J W Σ Wᵀ Jᵀ + 0.3 I` (OpenCV intrinsics; x/z, y/z clamped
   to the frustum + 15% as in 3DGS); eigen-decomposition → oriented quad at 3σ; SH colour with the view
   direction rotated into the node frame, `rgb = max(0.5 + Σ SH·coeffs, 0)`; visible splats are compacted
   with an atomic counter and get key `0xFFFFFFFF − float_bits(depth)`.
2. **sort** (compute): stable LSD radix sort, 4 × 8-bit passes (tile histogram → single-threadgroup scan →
   SIMD-ballot ranked scatter), ascending key = back-to-front. Indirect draw arguments come from the counter.
3. **raster** (render): non-instanced indexed quads (4 vertices / 6 indices per splat, splat = `vertex_id/4`),
   Gaussian falloff `exp(−½|uv|²)` in the fragment shader, premultiplied *over* blending into a target
   cleared to `background_color` (default black), i.e. `rgb + (1 − α)·bg`.
4. **overlays**: world-space line segments, near-plane clipped, as screen-space quads.

Ego camera: `camera_to_world = ego_to_world(t) · camera_to_ego`, intrinsics from `scene_graph.ego.camera`,
letterboxed into the view (scissored). World is z-up; the free camera uses an OpenCV look-at (x right, y down).

### Performance (Apple M4, 512×288, release build)

`--bench 500`, fixture at t = 2.0 s, GPU time per frame = `gpuEndTime − gpuStartTime`:

| Scene | Visible splats | P50 frame | preprocess / sort / raster (P50) |
| --- | --- | --- | --- |
| 228,475 Gaussians, idle GPU | 160k | **2.21 ms** (P90 2.71) | 0.33 / 0.18 / 1.68 ms |
| 228,475 Gaussians, GPU shared with MLX training jobs | 160k | 3.4–4.7 ms (min 2.2), sustained 3.2–4.2 ms/frame | 0.5 / 0.5 / 3.2 ms |
| 288,582 Gaussians, GPU shared with MLX training jobs | 204k | 4.96–5.70 ms (min 2.7), sustained 4.1–5.1 ms/frame | 0.63 / 0.66 / 3.8 ms |

The min times (≈2.2 ms / 2.7 ms) match the idle-GPU P50 and are the best estimate of uncontended cost at
~300k; an idle-GPU P50 at 288k was not captured. Re-run `--bench` on an idle machine for clean numbers.

The fixture has ~64× overdraw (≈9.5 M fragments per frame). Raster cost was dominated by per-primitive overhead
of instanced 4-vertex strips (zero-area quads still cost 1.6 ms). Switching to non-instanced indexed quads
cut the frame from 4.8 ms to 2.2 ms. Packing the splat record to 24 bytes (half precision) measured no gain
and was not kept.

## Code layout

```text
Package.swift
Sources/M4DGSCore/      pure Swift, unit-tested
  SafeTensors.swift     safetensors reader / writer (F32; F16/BF16/F64 converted)
  SceneMath.swift       Quat (Hamilton wxyz, slerp), Pose, Track interpolation, SH eval, Gaussian4D (CPU reference)
  Camera.swift          intrinsics, OpenCV camera, ego camera, orbit camera
  SceneGraph.swift      lenient scene_graph parsing + raw JSON preservation for .edits.json
  SceneData.swift       GaussianCloud packing, LoadedScene (+ sidecar), SceneFrame (poses / visibility at t)
  TrackEditor.swift     TrackEdit + resampling editor, perturbation-string parser
  Rollout.swift         rollout log parsing
  Overlays.swift        line-segment builders (boxes, tracks, frustum)
  LiveLink.swift        §6 wire codec (incremental decoder, limits), path sanitizers, train-log parser
  LiveCaptureSession.swift  receiver state machine: writes the §2 directory, status/saved replies, cancel, partial
Sources/M4DGSRender/
  Shaders.swift         Metal source (preprocess, radix sort, raster, lines)
  Renderer.swift        SplatRenderer: buffers, pipelines, encoding, benchmark, readback
  FrameBuilder.swift    scene state + camera + overlays → RenderFrame; PNG writer
Sources/M4DGSStudio/
  main.swift            entry: headless CLI if --render, else SwiftUI app
  StudioApp.swift       App, menus, launch arguments
  AppModel.swift        observable app state (scene, edits, playback, rollout, picking)
  MetalViewport.swift   MTKView + input handling + per-frame render / GPU timing
  Views.swift           sidebar (nodes, editor), timeline, HUD, inspector (view options, rollout panel)
  Headless.swift        --render / --bench CLI
  HeadlessLive.swift    --receive / --reconstruct CLI
  LiveReceiver.swift    NWListener + Bonjour, one LiveCaptureSession per TCP connection
  LiveCaptureModel.swift, LiveCaptureView.swift   Live Capture window + Settings
  ReconstructJob.swift  train_4dgs.py child process (line streaming, progress, cancel)
  StudioSettings.swift  repo root / captures folder / iterations (UserDefaults + auto-detection)
Tests/                  XCTest: safetensors round-trip, slerp, track clamping, SH, 4D eval, camera
                        orientation, edits (axes, time remap, cut-in, field preservation), rollout parsing,
                        GPU sort order, single-Gaussian projection, compositing, temporal opacity / visible_range,
                        live link (wire layout, round trip, split-packet reassembly, bad magic / oversize,
                        malicious paths, full capture / drop / cancel / duplicate name, train-log parsing)
Tools/make_test_scene.py  synthetic z-up road scene + rollout (numpy only)
Tools/fake_iphone_stream.py  live-link sender stand-in (numpy + opencv)
build_app.sh            wraps the binary into "M4DGS Studio.app" (Info.plist, ad-hoc signed)
```

## Limitations

* The renderer re-sorts every frame (no temporal coherence); fine at these sizes.
* The 0.3 px low-pass is applied without the Mip-Splatting opacity compensation (matches original 3DGS).
* Overlay lines are drawn on top of the splats (no depth test).
* Edit parameters are not stored in `.edits.json` (the schema is a plain scene graph); only the resulting tracks are.
