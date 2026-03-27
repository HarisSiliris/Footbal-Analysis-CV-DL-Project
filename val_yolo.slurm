#!/bin/bash
#SBATCH --job-name=yolo_val
#SBATCH --partition=GPUQ
#SBATCH --constraint=v100
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
source .venv/bin/activate

yolo detect val \
  model=/cluster/home/charisis/football-analysis/runs/detect/train6/weights/best.pt \
  data=/cluster/home/charisis/football-analysis/datasets/football_coco/data.yaml
