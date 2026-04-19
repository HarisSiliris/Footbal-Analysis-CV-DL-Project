This repository is for academic purposes as part of `TDT4265` at `NTNU`.

# Footbal-Analysis-CV-DL-Project

This project applies deep learning to football match footage in order to detect and track key objects in the scene. The current pipeline focuses on three classes:

- `player`
- `referee`
- `ball`

The end goal is to support downstream football analysis by building a reliable object detection and tracking workflow.

## Project Overview

The implemented workflow is:

`CVAT annotations -> COCO conversion -> train/val/test split -> YOLO dataset preparation -> YOLOv8 detection -> DeepSORT tracking -> evaluation and visualization`

The repository currently contains:

- dataset conversion scripts
- exploratory data analysis utilities
- training and validation entry points
- tracking and tracking-evaluation scripts
- a notebook for tracking visualization
- Slurm job files for running on IDUN

## Dataset Summary

The working dataset is a local football dataset in the original `Football2025` / `CVAT` format, converted into `COCO` and then `YOLO` format for training.

Current processed dataset summary:

- `6664` labeled frames
- `155998` annotations
- classes: `player`, `referee`, `ball`
- split sizes: `5331` train, `666` val, `667` test

The dataset is not distributed with this repository. Re-distribution of the dataset is prohibited.

## Model Choices

Detection:

- `YOLOv8s`
- pretrained initialization from `yolov8s.pt`
- trained at `1280x1280` for `100` epochs with batch size `8`

Tracking:

- `DeepSORT` via `deep-sort-realtime`
- tracking is run on top of the detector outputs to maintain identities across frames

## Repository Structure

- `scripts/`
  Main utilities for dataset preparation, training, validation, tracking, evaluation, and overlay rendering.
- `notebooks/`
  Notebook for visual inspection of tracking output.
- `train_yolo.slurm`, `val_yolo.slurm`, `track_yolo.slurm`
  Slurm entry points for running on NTNU IDUN.
- `requirements.txt`
  Python dependencies.
- `runs/`, `results/`, `logs/`
  Generated artifacts. These are not intended for version control.

## Requirements

Python `3.11` is recommended for local GPU training.

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install CUDA-enabled PyTorch if you want local GPU training. Example for CUDA `12.1`:

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

If you do not have a GPU or CUDA, use `--device cpu` instead, although this will be slower.

## Reproducible Workflow

If your dataset is still in the original `Football2025` / `CVAT` structure, run the scripts in this order.

### 1. Convert CVAT to COCO

```bash
python scripts/cvat_to_coco.py --data /path/to/Football2025 --output /path/to/output_coco
```

This creates:

- extracted image frames
- a COCO-style `annotations.json`

### 2. Create Train/Val/Test Splits

```bash
python scripts/create_coco_splits.py --coco /path/to/output_coco/annotations.json --out /path/to/output_coco/splits
```

This creates:

- `train.json`
- `val.json`
- `test.json`
- optional split id files

### 3. Prepare YOLO Dataset Files

```bash
python scripts/prepare_yolo_from_coco.py --coco /path/to/output_coco/annotations.json --out /path/to/your/yolo_dataset --splits /path/to/output_coco/splits
```

This creates:

- YOLO label files
- `splits/train.txt`, `splits/val.txt`, `splits/test.txt`
- `data.yaml`

### 4. Train the Detector

```bash
python scripts/train_yolo.py \
  --data /path/to/your/yolo_dataset/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch 8 \
  --device 0
```

### 5. Validate the Detector

```bash
python scripts/val_yolo.py \
  --model runs/detect/train*/weights/best.pt \
  --data /path/to/your/data.yaml \
  --imgsz 1280 \
  --batch 8 \
  --device 0
```

### 6. Run Tracking

```bash
python scripts/track_yolo.py \
  --source /path/to/your/image_folder \
  --model runs/detect/train*/weights/best.pt \
  --output runs/track/track_results.json \
  --names player,referee,ball \
  --imgsz 1280 \
  --device 0
```

### 7. Evaluate Tracking

```bash
python scripts/eval_tracking.py \
  --gt /path/to/your/datasets/football_coco/annotations.json \
  --pred runs/track/track_results.json \
  --iou-threshold 0.5
```

### 8. Render an Overlay Video

```bash
python scripts/render_tracking_video.py \
  --track-json runs/track/track_results.json \
  --image-dir /path/to/your/images \
  --output runs/track/tracking_overlay.mp4 \
  --fps 25 \
  --start-frame 1 \
  --duration 10
```

## IDUN Usage

These steps assume the dataset is mounted and you are running on NTNU IDUN.

Create a virtual environment:

```bash
cd /cluster/home/$USER/football-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run jobs:

```bash
sbatch train_yolo.slurm
sbatch val_yolo.slurm
sbatch track_yolo.slurm
```

Monitor jobs:

```bash
squeue -u $USER
tail -f logs/yolo_train-<jobid>.out
tail -f logs/yolo_val-<jobid>.out
```

## Important Paths

Make sure these paths are correct for your environment:

- `datasets/football_coco/data.yaml`
- `datasets/football_coco/splits/train.txt`
- `datasets/football_coco/splits/val.txt`
- `datasets/football_coco/splits/test.txt`

## Outputs

Generated artifacts are typically written to:

- `runs/detect/train*`
- `runs/detect/val*`
- `runs/track/`
- `results/`
- `logs/`

These are artifacts and are typically not committed to git.

## Included Scripts

- `scripts/cvat_to_coco.py`
  Convert CVAT/XML football annotations to COCO.
- `scripts/create_coco_splits.py`
  Split COCO annotations into train, val, and test sets.
- `scripts/prepare_yolo_from_coco.py`
  Convert COCO annotations to YOLO labels and `data.yaml`.
- `scripts/eda_coco.py`
  Generate EDA plots and summary statistics from COCO annotations.
- `scripts/dataset_analysis.py`
  Print dataset statistics and save annotated example images.
- `scripts/train_yolo.py`
  Main detection training entry point.
- `scripts/val_yolo.py`
  Main validation entry point.
- `scripts/track_yolo.py`
  Run YOLO detection plus DeepSORT tracking.
- `scripts/eval_tracking.py`
  Evaluate tracking results against ground truth.
- `scripts/render_tracking_video.py`
  Render tracking boxes and IDs into an MP4 overlay video.

## Notes

- The dataset itself is intentionally excluded from the repository.
- Trained weights and generated artifacts are also excluded from version control.
- The visualization notebook is available at `notebooks/visualize_tracking.ipynb`.
