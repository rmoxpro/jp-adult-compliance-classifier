# Dataset Format

This project does not ship any adult media, annotations, or trained weights. To train a model, prepare a lawful local dataset in one of the formats below.

## YOLO Segmentation Dataset

Use the standard Ultralytics layout:

```text
datasets/my_dataset/
  data.yaml
  images/
    train/
    val/
    test/
  labels/
    train/
    val/
    test/
```

`data.yaml`:

```yaml
path: /absolute/path/to/datasets/my_dataset
train: images/train
val: images/val
test: images/test

names:
  0: mosaic_nsfw
  1: nude_genital
```

Each image file should have a matching `.txt` label file with YOLO polygon segmentation labels:

```text
<class_id> <x1> <y1> <x2> <y2> ...
```

Coordinates are normalized to `[0, 1]`. Class IDs are:

- `0`: mosaic-obscured adult region
- `1`: uncensored explicit adult region

Images with no target objects may have an empty label file.

## Stage 2 Pair Dataset

Stage 2 is optional. It re-checks detector crops to reduce false positives.

```text
stage2_pairs/
  manifest.json
  crops/
    ...
  frames/
    ...
```

`manifest.json` is an array:

```json
[
  {
    "crop_path": "crops/example_crop.jpg",
    "frame_path": "frames/example_frame.jpg",
    "label": "mosaic",
    "bbox_norm": [0.1, 0.2, 0.4, 0.6],
    "is_hardneg": false
  }
]
```

Allowed labels:

- `mosaic`
- `uncensored`
- `other`

`bbox_norm` is optional but recommended when using `--draw-bbox`. It is `[x1, y1, x2, y2]`, normalized to `[0, 1]` in full-frame coordinates.

## Before Publishing Anything

Do not commit:

- images, thumbnails, extracted frames, or videos
- label files derived from private datasets if they reveal private filenames or sources
- trained weights
- database dumps
- logs containing URLs, filenames, or source metadata
