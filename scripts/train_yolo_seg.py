from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLO segmentation model for mosaic vs uncensored regions.")
    parser.add_argument("--data", required=True, help="Path to Ultralytics data.yaml")
    parser.add_argument("--weights", default="yolov8m-seg.pt", help="Base YOLO segmentation weights")
    parser.add_argument("--out-dir", default="runs/yolo", help="Output project directory")
    parser.add_argument("--name", default="mosaic_uncensored_yolo", help="Run name")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=12)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    from ultralytics import YOLO

    data = Path(args.data)
    if not data.exists():
        raise FileNotFoundError(data)

    model = YOLO(args.weights)
    model.train(
        data=str(data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        save=True,
        save_period=10,
        cache=False,
        project=args.out_dir,
        name=args.name,
        exist_ok=True,
        plots=True,
        resume=args.resume,
        lr0=args.lr0,
        mixup=0.15,
        copy_paste=0.5,
        hsv_v=0.6,
        hsv_s=0.7,
        degrees=10,
        translate=0.15,
        scale=0.5,
        close_mosaic=10,
    )


if __name__ == "__main__":
    main()
