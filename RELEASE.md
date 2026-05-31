# Release Guide

This project keeps source code and model weights separate.

- Git repository: source code, docs, scripts, examples
- GitHub Release assets: trained model weights and checksums

## 1. Create the Release Asset Zip

From the parent project workspace:

```bash
python open_source/mosaic-uncensored-classifier/scripts/build_release_assets.py \
  --yolo-weight mosaic_dataset/training_run_ab/ab_1778334426/weights/best.pt \
  --stage2-weight mosaic_dataset/classifier_run_v8_2stream/best.pt \
  --out-dir open_source/mosaic-uncensored-classifier/dist \
  --version v0.1.0
```

This creates:

```text
dist/jp-adult-compliance-model-v0.1.0.zip
```

The zip contains:

```text
weights/yolo_best.pt
weights/stage2_twostream_best.pt
release_manifest.json
README_WEIGHTS.md
SHA256SUMS
```

## 2. Push Source Code

Create a new GitHub repository, then from this directory:

```bash
git init
git add .
git commit -m "Initial public release"
git branch -M main
git remote add origin git@github.com:<OWNER>/<REPO>.git
git push -u origin main
```

## 3. Create GitHub Release

In GitHub UI:

1. Open the repository.
2. Go to Releases.
3. Draft a new release.
4. Tag: `v0.1.0`.
5. Title: `v0.1.0`.
6. Attach `dist/jp-adult-compliance-model-v0.1.0.zip`.
7. Paste the release notes below.

Suggested release notes:

```markdown
Initial public release.

This release includes source code for training and inference plus a separate model-weight package for Japanese adult-content compliance review workflows.

The model package is intended for human moderation assistance only. It is not legal advice and does not determine legality by itself.

Assets:
- `jp-adult-compliance-model-v0.1.0.zip`
```

## 4. Do Not Commit Release Assets

`dist/`, `*.zip`, and model weight files are ignored by `.gitignore`. Keep the source repository lightweight and attach weights only to Releases.
