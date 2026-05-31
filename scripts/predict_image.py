from __future__ import annotations

import argparse
import json
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASS_TO_VERDICT = {0: "mosaic", 1: "uncensored"}


def collect_images(paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out.extend(sorted(x for x in p.iterdir() if x.suffix.lower() in IMG_EXTS))
        elif p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run image-level prediction with trained YOLO weights.")
    parser.add_argument("images", nargs="+", help="Image files or directories")
    parser.add_argument("--weights", required=True, help="Path to trained YOLO weights")
    parser.add_argument("--conf", type=float, default=0.50)
    parser.add_argument("--conf-mosaic", type=float, default=0.50)
    parser.add_argument("--conf-uncensored", type=float, default=0.70)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--out", default=None, help="Write JSON output to file")
    args = parser.parse_args()

    images = collect_images(args.images)
    if not images:
        raise SystemExit("no images found")

    from ultralytics import YOLO

    model = YOLO(args.weights)
    kwargs = {"conf": args.conf, "verbose": False}
    if args.imgsz:
        kwargs["imgsz"] = args.imgsz

    results = []
    for start in range(0, len(images), args.batch):
        batch = images[start:start + args.batch]
        preds = model.predict([str(p) for p in batch], **kwargs)
        for image_path, pred in zip(batch, preds):
            best = None
            if pred.boxes is not None and len(pred.boxes) > 0:
                classes = pred.boxes.cls.cpu().tolist()
                scores = pred.boxes.conf.cpu().tolist()
                for cls_raw, score_raw in zip(classes, scores):
                    cls = int(cls_raw)
                    score = float(score_raw)
                    if cls == 0 and score < args.conf_mosaic:
                        continue
                    if cls == 1 and score < args.conf_uncensored:
                        continue
                    if cls not in CLASS_TO_VERDICT:
                        continue
                    if best is None or score > best[1]:
                        best = (cls, score)
            if best is None:
                results.append({"image": str(image_path), "verdict": "none", "score": 0.0, "class_id": None})
            else:
                cls, score = best
                results.append({
                    "image": str(image_path),
                    "verdict": CLASS_TO_VERDICT[cls],
                    "score": round(score, 4),
                    "class_id": cls,
                })

    text = json.dumps(results, ensure_ascii=False, indent=2)
    if args.out:
        Path(args.out).write_text(text)
    else:
        print(text)


if __name__ == "__main__":
    main()
