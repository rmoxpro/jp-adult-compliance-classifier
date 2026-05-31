# Usage

## 1. Install

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## 2. Validate Your Dataset

```bash
python scripts/validate_yolo_dataset.py --data /path/to/data.yaml
```

This checks that split directories exist, images have label files, label class IDs are valid, and normalized coordinates are in range.

## 3. Train YOLO Segmentation

```bash
python scripts/train_yolo_seg.py \
  --data /path/to/data.yaml \
  --out-dir runs/yolo \
  --name yolo_mosaic_uncensored \
  --epochs 100 \
  --imgsz 640 \
  --batch 12
```

The trained weights are saved under `runs/yolo/<name>/weights/`.

## 4. Optional Stage 2 Training

```bash
python scripts/train_stage2_twostream.py \
  --pairs-dir /path/to/stage2_pairs \
  --out-dir runs/stage2 \
  --epochs 15 \
  --batch 32 \
  --focal-loss \
  --draw-bbox
```

## 5. Run Image Prediction with Released Weights

Download and unpack the model-weight release asset:

```bash
curl -L -o jp-adult-compliance-model-v0.1.0.zip \
  https://github.com/rmoxpro/jp-adult-compliance-classifier/releases/download/v0.1.0/jp-adult-compliance-model-v0.1.0.zip
unzip jp-adult-compliance-model-v0.1.0.zip -d jp-adult-compliance-model-v0.1.0
```

Run prediction with the YOLO detector:

```bash
python scripts/predict_image.py \
  --weights jp-adult-compliance-model-v0.1.0/weights/yolo_best.pt \
  image1.jpg image2.jpg \
  --out results.json
```

The output is JSON with one verdict per image: `mosaic`, `uncensored`, or `none`. Treat `uncensored` as a human-review signal, not an automatic legal decision.

You can also use your own trained YOLO weights:

```bash
python scripts/predict_image.py \
  --weights runs/yolo/yolo_mosaic_uncensored/weights/best.pt \
  image1.jpg image2.jpg \
  --out results.json
```
