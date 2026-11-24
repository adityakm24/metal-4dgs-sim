#!/usr/bin/env bash
# End-to-end demo on the synthetic sample scene: data -> 4DGS -> refiner -> policy -> closed loop -> red team.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python3 )}
ITERS=${ITERS:-7000}
mkdir -p checkpoints outputs
[ -f data/sample_scene/log.json ] || $PY scripts/make_sample_scene.py --out data/sample_scene
[ -f checkpoints/scene_01.safetensors ] || $PY scripts/train_4dgs.py --data_dir data/sample_scene --iterations "$ITERS" --output checkpoints/scene_01.safetensors
[ -f checkpoints/flow_refiner.safetensors ] || $PY scripts/train_refiner.py --scene checkpoints/scene_01.safetensors --gt_scene data/sample_scene/gt_scene.safetensors --output checkpoints/flow_refiner.safetensors
[ -f checkpoints/pilotnet.safetensors ] || $PY scripts/train_policy.py --scene checkpoints/scene_01.safetensors --refiner checkpoints/flow_refiner.safetensors
$PY eval/closed_loop/run_sim.py --scene checkpoints/scene_01.safetensors --refiner checkpoints/flow_refiner.safetensors \
  --policy pilotnet --policy_ckpt checkpoints/pilotnet.safetensors --perturbation "actor_2:lateral_offset=-1.8m,t=2.5s" --record outputs/demo_out.mp4
$PY scripts/redteam.py --scene checkpoints/scene_01.safetensors --refiner checkpoints/flow_refiner.safetensors
$PY scripts/finetune_policy.py --scene checkpoints/scene_01.safetensors --refiner checkpoints/flow_refiner.safetensors
