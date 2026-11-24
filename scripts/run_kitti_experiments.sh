#!/usr/bin/env bash
# Real-data experiments on KITTI raw 2011_09_26_drive_0005 (after train_4dgs.py produced checkpoints/kitti_0005.safetensors).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-.venv/bin/python}
S=checkpoints/kitti_0005.safetensors
R=checkpoints/kitti_refiner.safetensors
[ -f $R ] || $PY scripts/train_refiner.py --scene $S --data_dir data/kitti_raw --steps 2000 --crop 128 256 --output $R
[ -f checkpoints/kitti_pilotnet.safetensors ] || $PY scripts/train_policy.py --scene $S --refiner $R --episodes 40 --epochs 30 \
    --data outputs/kitti_bc_data.npz --output checkpoints/kitti_pilotnet.safetensors
$PY eval/closed_loop/run_sim.py --scene $S --refiner $R --policy pilotnet --policy_ckpt checkpoints/kitti_pilotnet.safetensors \
    --perturbation "actor_1:hard_brake=6,t=3s" --record outputs/kitti_hard_brake.mp4
$PY eval/closed_loop/run_sim.py --scene $S --refiner $R --policy pilotnet --policy_ckpt checkpoints/kitti_pilotnet.safetensors \
    --record outputs/kitti_nominal.mp4
$PY scripts/redteam.py --scene $S --refiner $R --policy pilotnet --policy_ckpt checkpoints/kitti_pilotnet.safetensors \
    --families cut_in,hard_brake,obstacle,occlusion --population 6 --iterations 2 --log outputs/kitti_redteam.jsonl
$PY scripts/finetune_policy.py --scene $S --refiner $R --policy_ckpt checkpoints/kitti_pilotnet.safetensors \
    --redteam_log outputs/kitti_redteam.jsonl --bc_data outputs/kitti_bc_data.npz --output checkpoints/kitti_pilotnet_dagger.safetensors
