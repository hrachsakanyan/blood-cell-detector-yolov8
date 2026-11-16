"""Train YOLOv8n on the BCCD dataset (RBC, WBC, Platelets).

Usage:
    python train.py --data path/to/data.yaml
    python train.py --data path/to/data.yaml --epochs 30 --imgsz 416

The dataset must be in YOLO format with a data.yaml file.
The best weights are copied to models/best.pt for app.py.
"""

import argparse
import shutil
from pathlib import Path

import torch
from ultralytics import YOLO

MODELS_DIR = Path(__file__).parent / "models"


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8n on BCCD")
    parser.add_argument("--data", required=True, help="Path to the dataset data.yaml")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()

    data = Path(args.data)
    if not data.is_file():
        raise SystemExit(f"Dataset config not found: {data}")

    device = 0 if torch.cuda.is_available() else "cpu"
    print(f"Training on: {'GPU' if device == 0 else 'CPU'}")

    model = YOLO("yolov8n.pt")  # pretrained COCO weights, downloaded on first run
    results = model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=2,
        project="runs",
        name="bccd",
        exist_ok=True,
    )

    best = Path(results.save_dir) / "weights" / "best.pt"
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy(best, MODELS_DIR / "best.pt")
    print(f"Saved best weights to {MODELS_DIR / 'best.pt'}")


if __name__ == "__main__":
    main()
