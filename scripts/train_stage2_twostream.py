from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from mosaic_uncensored_classifier.twostream import (
    CLASSES,
    CROP_INPUT_SIZE,
    CTX_INPUT_SIZE,
    FocalLoss,
    TwoStreamClassifier,
    TwoStreamManifestDataset,
    build_transforms,
)


def split_manifest(manifest: list[dict], seed: int, val_ratio: float) -> tuple[list[dict], list[dict]]:
    rng = random.Random(seed)
    indices = list(range(len(manifest)))
    rng.shuffle(indices)
    val_size = max(1, int(len(indices) * val_ratio))
    val_idx = set(indices[:val_size])
    train_entries = [manifest[i] for i in indices if i not in val_idx]
    val_entries = [manifest[i] for i in indices if i in val_idx]
    return train_entries, val_entries


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the optional Stage 2 two-stream classifier.")
    parser.add_argument("--pairs-dir", required=True, help="Directory containing manifest.json plus crop/frame images")
    parser.add_argument("--out-dir", default="runs/stage2")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--focal-loss", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=4.0)
    parser.add_argument("--oversample-hardneg-factor", type=float, default=100.0)
    parser.add_argument("--draw-bbox", action="store_true")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    pairs_dir = Path(args.pairs_dir)
    manifest_path = pairs_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if not manifest:
        raise ValueError("manifest.json is empty")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    print(f"all pairs: {len(manifest)}")
    print(f"labels: {dict(Counter(m["label"] for m in manifest))}")
    print(f"hard negatives: {sum(1 for m in manifest if m.get("is_hardneg"))}")

    train_entries, val_entries = split_manifest(manifest, args.seed, args.val_ratio)
    print(f"train={len(train_entries)} val={len(val_entries)}")

    crop_train_tf, crop_val_tf, ctx_train_tf, ctx_val_tf = build_transforms()
    train_ds = TwoStreamManifestDataset(train_entries, pairs_dir, crop_train_tf, ctx_train_tf, args.draw_bbox)
    val_ds = TwoStreamManifestDataset(val_entries, pairs_dir, crop_val_tf, ctx_val_tf, args.draw_bbox)

    counts = Counter(m["label"] for m in train_entries)
    sample_weights = []
    for item in train_entries:
        weight = 1.0 / counts[item["label"]]
        if item.get("is_hardneg"):
            weight *= args.oversample_hardneg_factor
        sample_weights.append(weight)

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, sampler=sampler, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = TwoStreamClassifier(num_classes=len(CLASSES)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = FocalLoss(gamma=args.focal_gamma) if args.focal_loss else nn.CrossEntropyLoss()

    best_acc = 0.0
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        start = time.time()
        for crop, ctx, y in train_loader:
            crop, ctx, y = crop.to(device), ctx.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(crop, ctx)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * y.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += y.size(0)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        cm = torch.zeros(len(CLASSES), len(CLASSES), dtype=torch.int64)
        with torch.no_grad():
            for crop, ctx, y in val_loader:
                crop, ctx, y = crop.to(device), ctx.to(device), y.to(device)
                logits = model(crop, ctx)
                loss = criterion(logits, y)
                pred = logits.argmax(1)
                val_loss += loss.item() * y.size(0)
                val_correct += (pred == y).sum().item()
                val_total += y.size(0)
                for t, p in zip(y.cpu(), pred.cpu()):
                    cm[t, p] += 1
        scheduler.step()

        train_acc = train_correct / max(train_total, 1)
        val_acc = val_correct / max(val_total, 1)
        print(
            f"Epoch {epoch + 1}/{args.epochs}: "
            f"train_loss={train_loss / max(train_total, 1):.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss / max(val_total, 1):.4f} val_acc={val_acc:.4f} "
            f"time={time.time() - start:.0f}s"
        )
        for i, name in enumerate(CLASSES):
            print(f"  {name:12} | {cm[i].tolist()}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": list(CLASSES),
                "epoch": epoch + 1,
                "val_acc": val_acc,
                "arch": "twostream_b3_b0",
                "input_sizes": {"crop": CROP_INPUT_SIZE, "context": CTX_INPUT_SIZE},
                "draw_bbox": args.draw_bbox,
            }, out_dir / "best.pt")
            print(f"  best updated: {val_acc:.4f}")

    print(f"done. best val_acc={best_acc:.4f}; saved to {out_dir / 'best.pt'}")


if __name__ == "__main__":
    main()
