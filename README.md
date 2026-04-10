This repository is for academic purposes as part of TDT4265 (NTNU).

# Footbal-Analysis-CV-DL-Project
The focus of this academic project is to harness the power of deep learning to analyze football match footage, extracting valuable insights for RBK (or their opponents). The main tasks here will be to detect and track the players / referee and ball.

**Project Scope**
This project covers object detection and tracking. Current training and validation scripts focus on detection for three classes: player, referee, ball.

**Dataset Access**
The dataset is not distributed with this repository. Re-distribution of the dataset is prohibited.

**Requirements**
Python 3.11 is recommended for local GPU training.
Create a virtual environment and install dependencies from `requirements.txt`.

**Local Training (No Cluster)**
Use this if you have an NVIDIA GPU and CUDA-enabled PyTorch installed.

Create venv and install deps:
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Install CUDA-enabled PyTorch (example for CUDA 12.1):
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Train:
```bash
python scripts/train_yolo.py \
  --data /path/to/your/data.yaml \
  --model yolov8s.pt \
  --epochs 100 \
  --imgsz 1280 \
  --batch 8 \
  --device 0
```

Validate:
```bash
python scripts/val_yolo.py \
  --model runs/detect/train*/weights/best.pt \
  --data /path/to/your/data.yaml \
  --imgsz 1280 \
  --batch 8 \
  --device 0
```

If you do not have a GPU or CUDA, set `--device cpu` (slower).

**IDUN Training (Cluster)**
These steps assume you have access to NTNU IDUN and the dataset is mounted.

Create a venv on IDUN:
```bash
cd /cluster/home/$USER/football-analysis
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run training (Slurm):
```bash
sbatch train_yolo.slurm
```

Run validation (Slurm):
```bash
sbatch val_yolo.slurm
```

Monitor jobs:
```bash
squeue -u $USER
tail -f logs/yolo_train-<jobid>.out
tail -f logs/yolo_val-<jobid>.out
```

**Data Paths**
Make sure `datasets/football_coco/data.yaml` points to your local dataset path.
Make sure `datasets/football_coco/splits/train.txt` and `val.txt` contain valid local image paths.

**Dataset formatting**
If the cluster dataset is still in the original `Football2025`/CVAT structure, run the scripts in this order:

1. `scripts/cvat_to_coco.py`
   - Convert CVAT/XML annotations in `Football2025/*` to COCO-style `annotations.json`.
   - Write image files and the COCO annotation JSON.
2. `scripts/create_coco_splits.py`
   - Split COCO data into `train`, `val`, and `test` subsets.
   - Produce `train.json`, `val.json`, `test.json`, and optional `*_ids.txt` files.
3. `scripts/prepare_yolo_from_coco.py`
   - Convert COCO annotations into YOLO label files.
   - Create `splits/train.txt`, `splits/val.txt`, and `splits/test.txt`.
   - Write `data.yaml` for YOLO training.
4. `scripts/train_yolo.py`
   - Use the generated `data.yaml` and labels to train the detector.

Example commands:
```bash
python scripts/cvat_to_coco.py --data /path/to/Football2025 --output /path/to/output_coco
python scripts/create_coco_splits.py --coco /path/to/output_coco/annotations.json --out /path/to/output_coco/splits
python scripts/prepare_yolo_from_coco.py --coco /path/to/output_coco/annotations.json --out /path/to/your/yolo_dataset --splits /path/to/output_coco/splits
python scripts/train_yolo.py --data /path/to/your/yolo_dataset/data.yaml --model yolov8s.pt --epochs 100 --imgsz 1280 --batch 8 --device 0
```

**Outputs**
Training outputs are saved under `runs/detect/train*`.
Validation outputs are saved under `runs/detect/val*`.
These are artifacts and typically not committed to git.

**Scripts**
`scripts/train_yolo.py` is the main training entry point.
`scripts/val_yolo.py` is the main validation entry point.
Additional dataset utilities live in `scripts/`.
