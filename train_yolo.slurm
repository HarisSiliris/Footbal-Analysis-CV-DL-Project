#!/bin/bash
#SBATCH --job-name=yolo_train
#SBATCH --partition=GPUQ
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
source .venv/bin/activate
mkdir -p logs
python scripts/train_yolo.py \
--data /cluster/home/charisis/football-analysis/datasets/football_coco/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch 8 \
  --device 0
