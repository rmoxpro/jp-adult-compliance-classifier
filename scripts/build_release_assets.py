from __future__ import annotations

import argparse
import hashlib
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GitHub Release assets for model-weight distribution.")
    parser.add_argument("--yolo-weight", required=True, help="Path to the recommended YOLO best.pt")
    parser.add_argument("--stage2-weight", required=True, help="Path to the recommended Stage2 best.pt")
    parser.add_argument("--out-dir", default="dist")
    parser.add_argument("--version", default="v0.1.0")
    args = parser.parse_args()

    yolo_src = Path(args.yolo_weight).expanduser().resolve()
    stage2_src = Path(args.stage2_weight).expanduser().resolve()
    if not yolo_src.exists():
        raise FileNotFoundError(yolo_src)
    if not stage2_src.exists():
        raise FileNotFoundError(stage2_src)

    out_dir = Path(args.out_dir).expanduser().resolve()
    build_dir = out_dir / f"jp-adult-compliance-model-{args.version}"
    weights_dir = build_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    yolo_dst = weights_dir / "yolo_best.pt"
    stage2_dst = weights_dir / "stage2_twostream_best.pt"
    yolo_dst.write_bytes(yolo_src.read_bytes())
    stage2_dst.write_bytes(stage2_src.read_bytes())

    files = [yolo_dst, stage2_dst]
    checksums = {str(p.relative_to(build_dir)): sha256_file(p) for p in files}
    manifest = {
        "version": args.version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Japanese adult-content compliance review assistance",
        "not_legal_advice": True,
        "requires_human_review": True,
        "files": {
            "weights/yolo_best.pt": {
                "role": "YOLO segmentation detector",
                "classes": {"0": "mosaic_nsfw", "1": "nude_genital"},
                "sha256": checksums["weights/yolo_best.pt"],
                "size_bytes": yolo_dst.stat().st_size,
            },
            "weights/stage2_twostream_best.pt": {
                "role": "Optional Stage 2 two-stream classifier",
                "classes": ["mosaic", "other", "uncensored"],
                "sha256": checksums["weights/stage2_twostream_best.pt"],
                "size_bytes": stage2_dst.stat().st_size,
            },
        },
    }
    write_text(build_dir / "release_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    write_text(build_dir / "SHA256SUMS", "".join(f"{digest}  {name}\n" for name, digest in checksums.items()))
    readme = (
        f"# JP Adult Compliance Model Weights {args.version}\n\n"
        "This package contains trained weights for the JP Adult Content Compliance Classifier.\n\n"
        "Files:\n\n"
        "- weights/yolo_best.pt: YOLO segmentation detector\n"
        "- weights/stage2_twostream_best.pt: optional Stage 2 false-positive reduction classifier\n"
        "- release_manifest.json: metadata and class names\n"
        "- SHA256SUMS: checksums\n\n"
        "These weights are intended for compliance support and human moderation assistance. "
        "They do not determine legality and are not legal advice.\n"
    )
    write_text(build_dir / "README_WEIGHTS.md", readme)

    zip_path = out_dir / f"jp-adult-compliance-model-{args.version}.zip"
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(build_dir.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(build_dir))

    print(zip_path)
    print(f"size_bytes={zip_path.stat().st_size}")
    print(f"sha256={sha256_file(zip_path)}")


if __name__ == "__main__":
    main()
