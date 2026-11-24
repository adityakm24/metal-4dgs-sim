# metal-4dgs-sim

Metal/MLX 4D Gaussian Splatting + flow-matching closed-loop neural simulator for adversarial policy
stress-testing on Apple Silicon. The design document is [project-plan.md](project-plan.md); this README covers
what is built and how to use it.

```text
apps/iphone  ──(ARKit capture .zip, AirDrop)──►  capture/  ──►  core/reconstruct  ──►  scene .safetensors
                                                                                           │       ▲
                                                 apps/macbook (Metal viewer + trajectory editor)  ─┘ .edits.json
                                                                                           ▼
          engine/metal_splat (Metal kernels) ─► models/refiner (2-step flow matching) ─► eval/closed_loop policy
                                     ▲                                                      │ steer / accel
                                     └──────────── bicycle model ego-pose update ◄──────────┘
```

## Repository layout

| Path | What it is |
| --- | --- |
| `apps/iphone/` | **iPhone app** (SwiftUI + ARKit): posed RGB + LiDAR depth + VIO points capture, zip export. See its README. |
| `apps/macbook/` | **MacBook app** (SwiftUI + Metal): real-time 4DGS viewer, timeline, trajectory editor, rollout playback. See its README. |
| `capture/` | Dataset loaders: ARKit captures, nuScenes (raw JSON tables), Waymo converter (experimental), generic log, synthetic world |
| `core/reconstruct/` | 4DGS optimisation: init (LiDAR / VIO points / Depth Anything V2), Adam, L1+SSIM+depth, densify/prune |
| `engine/metal_splat/` | Differentiable tile rasterizer: custom Metal forward/backward kernels (`shaders/*.metal`) on MLX unified-memory arrays |
| `engine/scene_graph/` | Scene graph (background + rigid actor nodes), 4D Gaussians `(μ, Σ, α, SH, v, τ, σ_t)`, tracks, trajectory editor, box segmentation |
| `models/refiner/` | Flow-matching refiner: UNet with a ControlNet-style zero-init adapter on RGB/depth/alpha/flow |
| `eval/closed_loop/` | Rollout engine, bicycle model, policies (expert, PilotNet, openpilot adapter), perturbations, red-teamer |
| `eval/metrics/` | Depth AbsRel, temporal warping error, depth reprojection, collision/TTC/route metrics |
| `ui/` | 3-panel telemetry (raw splat + depth/flow insets │ refined │ BEV + signals + latency), mp4 recording |
| `scripts/` | Build, sample-scene generation, training, red-teaming, DAgger fine-tuning, benchmarks |
| `docs/formats.md` | File-format contracts between the iPhone app, Python stack and Mac app |

## Quickstart

```bash
python3.12 -m venv .venv && source .venv/bin/activate      # Python 3.11+
pip install -r requirements.txt
./scripts/build_metal.sh                                     # JIT-compile + test the Metal kernels

# 1. a synthetic ground-truth driving world rendered into a posed RGB+depth log (or use your own capture)
python scripts/make_sample_scene.py --out data/sample_scene

# 2. reconstruct the 4D scene graph
python scripts/train_4dgs.py --data_dir data/sample_scene --iterations 7000 --output checkpoints/scene_01.safetensors

# 3. train the 2-step flow-matching refiner on extrapolated (±2.5 m) views
python scripts/train_refiner.py --scene checkpoints/scene_01.safetensors \
    --gt_scene data/sample_scene/gt_scene.safetensors --output checkpoints/flow_refiner.safetensors

# 4. train the compact visuomotor policy by imitation inside the simulator
python scripts/train_policy.py --scene checkpoints/scene_01.safetensors --refiner checkpoints/flow_refiner.safetensors

# 5. closed-loop run with a cut-in injected at t = 2.5 s, 3-panel video + rollout log
python eval/closed_loop/run_sim.py --scene checkpoints/scene_01.safetensors \
    --refiner checkpoints/flow_refiner.safetensors --policy pilotnet --policy_ckpt checkpoints/pilotnet.safetensors \
    --perturbation "actor_2:lateral_offset=-1.8m,t=2.5s" --record demo_out.mp4

# 6. red-team the policy, then fine-tune on its failures (counterfactual intervention rate)
python scripts/redteam.py --scene checkpoints/scene_01.safetensors --refiner checkpoints/flow_refiner.safetensors
python scripts/finetune_policy.py --scene checkpoints/scene_01.safetensors --refiner checkpoints/flow_refiner.safetensors

# benchmarks / fidelity
python scripts/benchmark.py --scene checkpoints/scene_01.safetensors --refiner checkpoints/flow_refiner.safetensors \
    --policy pilotnet --policy_ckpt checkpoints/pilotnet.safetensors
python scripts/eval_fidelity.py --scene checkpoints/scene_01.safetensors \
    --gt_scene data/sample_scene/gt_scene.safetensors --refiner checkpoints/flow_refiner.safetensors
```

Real data: `--data_dir` also accepts an iPhone capture folder or `.zip` from `apps/iphone`, or a nuScenes root
(`--nuscenes_scene scene-0061`). `--mono_depth` initialises handheld captures without LiDAR from Depth Anything V2,
aligned to ARKit's metric VIO points (needs `pip install transformers torch pillow`).

### Perturbation language

`target:kind=value[,param=value]`, `;`-separated or repeated `--perturbation` flags:

| Spec | Effect |
| --- | --- |
| `actor_2:lateral_offset=-1.8m,t=2.5s[,ramp=1.5s][,until=6s]` | shift along the actor's left axis (negative = towards its right) |
| `actor_2:cut_in=-3.5m,t=2s` | same with a sharp 1 s ramp |
| `actor_1:hard_brake=8,t=3s` | brake to a stop at 8 m/s² |
| `actor_1:velocity_spike=-5m/s,t=2s,duration=1s` | sudden speed change |
| `actor_3:speed_scale=1.5` · `time_shift=-1s` · `longitudinal_offset=5m` · `remove=1` | retiming / removal |
| `obstacle_1:spawn=30m,t=2s,lateral=0m` | static box spawned in front of the ego |
| `camera:occlusion=0.35,t=3s,duration=1s` | synthetic lens occlusion |
| `ego:lateral_offset=1m` · `ego:yaw=5deg` · `ego:speed=12` | ego initial-state perturbation |

Trajectory edits made in the MacBook app are saved as `<scene>.edits.json` and picked up automatically by every
Python entry point (`--no_edits` to ignore).

## Status

Everything below was measured on an Apple M4 (16 GB) using the bundled synthetic sample scene (`make_sample_scene.py`:
a procedurally generated Gaussian street with 3 moving cars, rendered into 101 posed RGB+depth frames). The real-data
paths (iPhone capture, nuScenes, Waymo) are implemented but have **not** been run on real recordings yet.

### Roadmap vs. project plan

| Plan item | State |
| --- | --- |
| Custom Metal 4DGS forward/backward rasterizer | ✅ `engine/metal_splat` — forward matches a brute-force reference; backward matches MLX autodiff of a dense reference to 1e-3 |
| iPhone ARKit VIO capture + Depth Anything V2 init | ✅ app builds and runs in the Simulator; loader round-trip tested. ⚠️ Not yet run on a physical iPhone; Depth Anything path not exercised (optional deps) |
| nuScenes / Waymo loaders | ✅ nuScenes JSON loader written; ⚠️ untested without the dataset. Waymo converter is experimental (Linux-only deps) |
| MLX 2-step flow-matching refiner | ✅ trained on paired extrapolated-view renders; gains are modest (below) |
| Closed-loop policy + 3-panel telemetry | ✅ expert, PilotNet (MLX) and red-teamer working end-to-end. ⚠️ openpilot `supercombo.onnx` adapter is untested (no weights bundled, needs onnxruntime) |
| Metrics | ✅ depth AbsRel, E_warp, depth reprojection, collision/TTC/route. ❌ FVD not implemented (needs pretrained I3D) |
| MacBook app (viewer + trajectory editor) | ✅ `swift build`/`swift test` pass, renders checkpoints identically to the Python renderer |

### Reconstruction (`train_4dgs.py`, 7000 its, 512×288)
141k Gaussians, **44.1 dB PSNR on held-out frames**, 7 min (≈59 ms/iteration).

### Fidelity at shifted viewpoints (`eval_fidelity.py`, vs ground truth, 15 frames)

| Lateral shift | PSNR raw | PSNR refined | Depth AbsRel (4DGS) | E_warp raw / refined |
| --- | --- | --- | --- | --- |
| 0 m (logged path) | 45.5 | 44.7 | 0.013 | 0.0115 / 0.0121 |
| +1.0 m | 33.7 | 34.3 | 0.017 | 0.0116 / 0.0122 |
| +2.5 m | 28.3 | 29.0 | 0.030 | 0.0138 / 0.0144 |
| −2.5 m | 28.8 | 29.6 | 0.026 | 0.0131 / 0.0138 |

The refiner gains +0.6–0.8 dB at extrapolated views and costs 0.8 dB on the logged path. It slightly increases
temporal warping error, and it doesn't remove large disocclusion smears. The metric depth it is conditioned on comes
from the 4DGS buffer unchanged. A bigger model, longer training, or a latent-diffusion prior are the obvious next steps.

### Closed loop & red-teaming (PilotNet on refined frames, 10 Hz)

* Nominal and the plan's `actor_2:lateral_offset=-1.8m,t=2.5s` cut-in: no collision, max lateral error 0.10 m.
* `redteam.py` (CEM, 132 rollouts): **22 failures (17%)**: lead-car hard brakes 11/36, spawned obstacles 11/12;
  cut-ins 0/36, velocity spikes 0/36, lens occlusion 0/12.
* `finetune_policy.py` (DAgger on those failures): intervention rate on the failure scenarios **100% → 0%** (these
  were also the fine-tuning scenarios), on 20 fresh held-out adversarial scenarios **10% → 0%** (small sample).

### Latency @ 512×288 (P50)

| Stage | Measured | Plan target |
| --- | --- | --- |
| Rasterization, MacBook app (native Metal, forward-only, GPU time, 228k Gaussians) | **2.2 ms** | < 4.5 ms ✅ |
| Rasterization, Python/MLX differentiable path (RGB + depth + flow, 141–197k Gaussians) | 9.7 ms idle machine; 10–14 ms under load | < 4.5 ms ❌ |
| Refiner (2 Euler steps, fp16, 1.3 M params) | 22–25 ms | < 22 ms ≈ |
| Policy (PilotNet, MLX) | 1.0 ms | < 4 ms ✅ |
| Total closed-loop step (Python) | 33 ms idle-ish; 40 ms under load | ~30.5 ms ≈ |

The Python path still pays for MLX graph construction and a 64-bit `argsort` per frame. The native app shows the kernels
themselves fit the budget. Int8 is not implemented: MLX quantises only Linear layers and the refiner is convolutional.

