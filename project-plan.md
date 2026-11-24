# metal-4dgs-sim

**Metal-accelerated 4D Gaussian Splatting + Flow-Matching closed-loop neural simulator for adversarial driving and robotics policy stress-testing.**

`metal-4dgs-sim` turns passive, real-world RGB recordings into interactive, metric 4D environments. It decouples explicit 3D scene geometry from photorealistic neural rendering—allowing you to programmatically edit dynamic agent trajectories, synthesize geometrically consistent counterfactual views, and stress-test end-to-end driving or visuomotor policies in a closed loop on Apple Silicon.

---

## Why `metal-4dgs-sim`?

Current simulation paradigms for embodied AI suffer from a fundamental trade-off:
* **Pure Video Diffusion / World Models** synthesize photorealistic futures but hallucinate 3D geometry—lane lines warp, distances drift, and physics break under out-of-distribution actions.
* **Traditional Graphics Engines (CARLA / Unreal)** guarantee metric physics but suffer from a massive sim-to-real visual domain gap and require manual artist asset creation.
* **Passive 3DGS Log Replay** preserves real-world visuals but cannot synthesize novel dynamic interactions or clean up severe viewpoint extrapolation artifacts.

`metal-4dgs-sim` bridges this gap by anchoring a lightweight **Flow-Matching / Latent Diffusion visual refiner** onto an explicit, editable **4D Gaussian Splatting (4DGS) scene graph**—running natively via **Metal / MLX** unified memory with zero host-device copies.

---

## Architecture

```text
[iPhone 16 RGB + ARKit VIO] ──┐
                              ├──► [4DGS Scene Graph] ──► (Trajectory / Ego Perturbation)
[nuScenes / Waymo Logs] ──────┘            │
                                           ▼
                            [Metal Splat Rasterizer (<5ms)]
                                           │
                          (Noisy RGB + Metric Depth + Optical Flow)
                                           ▼
                         [MLX Flow-Matching Visual Refiner]
                                           │
                              (Photorealistic Novel View)
                                           ▼
                      [End-to-End Driving / Robotics Policy] ──► (Steering / Action) ──┐
                                                                                       │
                      ◄─────────────────── (Closed-Loop Ego Pose Update) ──────────────┘

```

### Core Components

1. **Pure-Vision 4D Reconstruction (`core/reconstruct`)**
* Reconstructs static backgrounds and segmented dynamic actors into 4D Gaussians $(\mu, \Sigma, \alpha, \text{SH}, v)$ using posed RGB frames from a standard iPhone 16 (ARKit 6-DoF VIO + feed-forward monocular depth initialization) or public driving logs (nuScenes / Waymo Open).


2. **Native Metal / MLX Splat Rasterizer (`engine/metal_splat`)**
* Custom Metal compute kernels and zero-copy MLX tensor bindings for real-time novel-view projection, depth rendering, and dynamic actor rigid-body/velocity perturbations.


3. **Geometry-Grounded Neural Refiner (`models/refiner`)**
* A distilled ControlNet / Flow-Matching adapter conditioned on raw 3DGS renders, metric depth buffers, and optical flow. Eliminates splat holes, floaters, and missing shadows during large lateral camera shifts ($\pm 2.5\text{m}$) without altering underlying metric geometry.


4. **Closed-Loop Adversarial Harness (`eval/closed_loop`)**
* Runs open-source end-to-end policies (e.g., Openpilot driving policy / visuomotor diffusion policies) inside the simulator. Includes a programmatic red-teamer to inject long-tail edge cases (sudden cut-ins, velocity spikes, synthetic occlusions) and log policy failure modes.



---

## Target Performance (Apple M4 Unified Memory @ 512×288)

| Pipeline Stage | Backend | Target Latency (P50) | Notes |
| --- | --- | --- | --- |
| **4DGS Novel-View Rasterization** | Custom Metal / MLX | `< 4.5 ms` | Zero-copy unified memory buffer |
| **Neural Visual Refinement** | MLX (FP16 / INT8) | `< 22.0 ms` | 2-step Flow-Matching / ControlNet adapter |
| **Policy Forward Pass** | MLX / ONNX Runtime | `< 4.0 ms` | Openpilot / compact visuomotor policy |
| **Total Closed-Loop Step** | **End-to-End** | **`~30.5 ms` (>30 FPS)** | Interactive real-time simulation |

---

## Repository Structure

```text
metal-4dgs-sim/
├── capture/                 # iPhone 16 ARKit VIO + RGB exporter & dataset loaders (nuScenes/Waymo)
├── engine/
│   ├── metal_splat/         # Custom Metal shaders for differentiable 4DGS rasterization
│   └── scene_graph/         # Dynamic actor segmentation & programmatic trajectory editor
├── models/
│   └── refiner/             # MLX Flow-Matching / ControlNet artifact & relighting refiner
├── eval/
│   ├── closed_loop/         # Policy rollout engine & closed-loop ego-motion integrator
│   └── metrics/             # Depth reprojection error, FVD, warping consistency, collision rate
├── ui/                      # 3-panel live telemetry & split-screen visualizer
└── scripts/                 # Reconstruction, adapter fine-tuning, and benchmark runners

```

---

## Quickstart

### Prerequisites

* macOS 15+ on Apple Silicon (M3 / M4 series recommended)
* Xcode Command Line Tools (for Metal shader compilation)
* Python 3.11+ with `mlx`, `numpy`, `opencv-python`
* *(Optional)* iPhone 16 (iOS 18+) for custom indoor/tabletop ARKit RGB capture

### 1. Build Metal Kernels & Install Dependencies

```bash
git clone https://github.com/adityakm24/metal-4dgs-sim.git
cd metal-4dgs-sim

# Compile custom Metal 4DGS rasterization shaders
./scripts/build_metal.sh

# Set up MLX environment
pip install -r requirements.txt

```

### 2. Reconstruct a 4D Scene

Reconstruct from an iPhone ARKit capture folder or a pre-processed nuScenes sequence:

```bash
python scripts/train_4dgs.py \
  --data_dir data/sample_scene \
  --iterations 7000 \
  --output checkpoints/scene_01.safetensors

```

### 3. Run Closed-Loop Adversarial Simulation

Launch the 3-panel simulator with a sudden cut-in perturbation injected at $t = 2.5\text{s}$:

```bash
python eval/closed_loop/run_sim.py \
  --scene checkpoints/scene_01.safetensors \
  --refiner checkpoints/flow_refiner_int8.safetensors \
  --policy openpilot \
  --perturbation "actor_2:lateral_offset=-1.8m,t=2.5s" \
  --record demo_out.mp4

```

---

## Evaluation & Benchmarks

`metal-4dgs-sim` tracks both **geometric fidelity** and **closed-loop control safety**:

* **Metric Depth Consistency ($\text{AbsRel}$):** Measures how faithfully the neural refiner preserves the exact metric depth rendered by the 4DGS scene graph across shifted viewpoints.
* **Temporal Warping Error ($E_{\text{warp}}$):** Evaluates optical-flow consistency across consecutive generated frames during ego-vehicle steering maneuvers.
* **Counterfactual Intervention Rate:** Quantifies policy collision rates before and after fine-tuning on synthetic edge-case rollouts generated by the simulator.

---

## Roadmap

* [x] Architecture & Metal zero-copy unified memory pipeline design
* [ ] Custom Metal 4DGS forward/backward rasterizer
* [ ] iPhone 16 ARKit VIO + Depth Anything V2 initialization pipeline
* [ ] MLX 2-step Flow-Matching refiner trained on paired degraded-splat / ground-truth frames
* [ ] Closed-loop Openpilot & visuomotor policy integration + 3-panel telemetry dashboard

---

## License

MIT License. See `LICENSE` for details.
