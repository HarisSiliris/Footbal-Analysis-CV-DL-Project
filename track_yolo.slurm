#!/bin/bash
#SBATCH --job-name=yolo_track
#SBATCH --partition=GPUQ
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source .venv/bin/activate
mkdir -p logs runs/track

python scripts/track_yolo.py \
  --source /cluster/home/charisis/football-analysis/datasets/football_coco/images \
  --model /cluster/home/charisis/football-analysis/runs/detect/train6/weights/best.pt \
  --output /cluster/home/charisis/football-analysis/runs/track/track_results.json \
  --names player,referee,ball \
  --imgsz 1280 \
  --device 0
