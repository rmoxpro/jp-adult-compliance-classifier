from __future__ import annotations

import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VALID_CLASSES = {0, 1}


def parse_simple_yaml(path: Path) -> dict:
    data: dict[str, object] = {}
    names: dict[int, str] = {}
    in_names = False
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        if line.startswith("names:"):
            in_names = True
            continue
        if in_names and line.startswith("  "):
            key, value = line.strip().split(":", 1)
            names[int(key)] = value.strip().strip('\'"')
            continue
        in_names = False
        if ":" in line:
            key, value = line.split(":", 1)
            data[key.strip()] = value.strip().strip('\'"')
    data["names"] = names
    return data


def validate_label_file(path: Path) -> list[str]:
    errors: list[str] = []
    text = path.read_text().strip()
    if not text:
        return errors
    for line_no, line in enumerate(text.splitlines(), 1):
        parts = line.split()
        if len(parts) < 7:
            errors.append(f"{path}:{line_no}: segmentation label needs class + at least 3 points")
            continue
        try:
            cls = int(float(parts[0]))
            coords = [float(x) for x in parts[1:]]
        except ValueError:
            errors.append(f"{path}:{line_no}: non-numeric label value")
            continue
        if cls not in VALID_CLASSES:
            errors.append(f"{path}:{line_no}: invalid class {cls}; expected 0 or 1")
        if len(coords) % 2 != 0:
            errors.append(f"{path}:{line_no}: odd number of polygon coordinates")
        bad = [x for x in coords if x < 0.0 or x > 1.0]
        if bad:
            errors.append(f"{path}:{line_no}: coordinates must be normalized to [0, 1]")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate an Ultralytics YOLO segmentation dataset.")
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--max-errors", type=int, default=50)
    args = parser.parse_args()

    data_yaml = Path(args.data).expanduser().resolve()
    cfg = parse_simple_yaml(data_yaml)
    base = Path(str(cfg.get("path", data_yaml.parent))).expanduser()
    if not base.is_absolute():
        base = (data_yaml.parent / base).resolve()

    errors: list[str] = []
    total_images = 0
    total_labels = 0
    for split in ("train", "val", "test"):
        image_rel = cfg.get(split)
        if not image_rel:
            if split != "test":
                errors.append(f"missing {split}: entry in data.yaml")
            continue
        image_dir = base / str(image_rel)
        label_dir = base / str(image_rel).replace("images", "labels", 1)
        if not image_dir.is_dir():
            errors.append(f"missing image directory: {image_dir}")
            continue
        if not label_dir.is_dir():
            errors.append(f"missing label directory: {label_dir}")
            continue
        images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS)
        total_images += len(images)
        print(f"{split}: {len(images)} images")
        for img in images:
            label = label_dir / f"{img.stem}.txt"
            if not label.exists():
                errors.append(f"missing label for {img}")
                continue
            total_labels += 1
            errors.extend(validate_label_file(label))
            if len(errors) >= args.max_errors:
                break

    print(f"images={total_images} labels={total_labels}")
    if errors:
        print(f"errors={len(errors)}")
        for err in errors[: args.max_errors]:
            print(f"ERROR: {err}")
        raise SystemExit(1)
    print("OK")


if __name__ == "__main__":
    main()
