# 日本向け成人コンテンツ・コンプライアンス分類器

日本国内向けの成人コンテンツ運用において、モザイク処理漏れや明確な無修正リスクを検出し、人間のレビューを支援するためのコンピュータビジョン学習・推論ツールです。

このプロジェクトは **コンプライアンス支援、公開前チェック、人間レビューの補助** を目的としています。法的判断そのものを行うものではなく、法律助言でもありません。最終判断には、適切な法務確認または編集・審査担当者によるレビューが必要です。

このリポジトリには、**成人画像、動画、非公開アノテーション、学習済み重み、DBダンプ、スクレイピングツールは含まれていません**。学習には、各自が適法に管理できるローカルデータセットを用意してください。

## 想定用途

- 公開前のモザイク処理漏れ検出
- 明確な無修正リスクの人間レビュー対象化
- 日本向け成人コンテンツ運用における社内コンプライアンス支援
- 適法に管理されたデータセットによる成人コンテンツ・モデレーション研究

## 対象外・禁止用途

このプロジェクトを、成人コンテンツの探索、収集、ランキング、推薦、配布のために使用しないでください。性的搾取、非同意の私的画像、未成年関連、嫌がらせ、プラットフォームのモデレーション回避にも使用しないでください。このモデルはレビュー補助であり、コンテンツ発見ツールではありません。

## 含まれるもの

- データセット形式のドキュメント
- YOLO segmentationデータセット検証スクリプト
- 2クラスYOLO segmentation学習スクリプト
  - `mosaic_nsfw`
  - `nude_genital`
- 学習済みYOLO重みを使う画像推論CLI
- 任意のStage 2 two-stream分類器学習コード
  - crop branch: EfficientNet-B3
  - full-frame context branch: EfficientNet-B0
- 設定・manifest形式のみの小さなサンプル

## 含まれないもの

- 成人画像・動画
- 学習済みモデル重み
- 非公開manifestやアノテーション
- URL、取得元メタデータ、DBダンプ、ログ
- スクレイパーやデータ収集ツール

## モデル重み

Gitリポジトリ本体には学習済み重みを保存していません。公式重みはGitHub Release assetsで公開されています。

公開済みRelease asset:

```text
https://github.com/rmoxpro/jp-adult-compliance-classifier/releases/download/v0.1.0/jp-adult-compliance-model-v0.1.0.zip
```

```text
jp-adult-compliance-model-v0.1.0.zip
  weights/yolo_best.pt
  weights/stage2_twostream_best.pt
  release_manifest.json
  README_WEIGHTS.md
  SHA256SUMS
```

SHA256:

```text
706ac5fa74a10150b7ecde7264ccc73135bb7366d32358ca7402f9e5e4f4e1d2
```

重みの内容とRelease運用の詳細は [RELEASE.md](RELEASE.md) を参照してください。

## 評価結果

公開配布している推奨重みの内部評価では、YOLO検出器は mAP50 約 `0.94`、Stage 2 two-stream分類器は validation accuracy 約 `99.3%` でした。

これらは非公開の内部評価セットでの参考値です。公開データセットによる再現値ではなく、実運用での精度は対象ドメイン、画質、ラベル基準、しきい値に依存します。最終判断には必ず人間レビューが必要です。

## クイックスタート

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

YOLOデータセットを検証します。

```bash
python scripts/validate_yolo_dataset.py --data /path/to/data.yaml
```

YOLO segmentationを学習します。

```bash
python scripts/train_yolo_seg.py \
  --data /path/to/data.yaml \
  --out-dir runs/yolo \
  --name jp_compliance_yolo \
  --epochs 100 \
  --imgsz 640 \
  --batch 12
```

学習済み重みで画像推論します。

```bash
python scripts/predict_image.py \
  --weights runs/yolo/jp_compliance_yolo/weights/best.pt \
  image1.jpg image2.jpg \
  --out results.json
```

任意でStage 2分類器を学習します。

```bash
python scripts/train_stage2_twostream.py \
  --pairs-dir /path/to/stage2_pairs \
  --out-dir runs/stage2 \
  --epochs 15 \
  --batch 32 \
  --focal-loss \
  --draw-bbox
```

## データセット概要

YOLO学習には標準的なUltralytics形式を使います。

```text
datasets/my_dataset/
  data.yaml
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
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

Stage 2では、crop画像とfull-frame画像のペアを記述したローカル `manifest.json` を使います。詳細は [docs/DATASET.md](docs/DATASET.md) を参照してください。

## 公開前チェック

fork、release、model cardなどを公開する前に、以下が含まれていないことを確認してください。

- 画像、サムネイル、抽出フレーム、動画
- 非公開ファイル名や取得元を含むラベルファイル
- 意図せず含まれた学習済み重み
- DBダンプ
- URL、ファイル名、ユーザー名、取得元メタデータを含むログ

同梱の `.gitignore` は一般的な危険ファイルを除外しますが、公開前に必ず `git status` と実ファイル一覧を手動確認してください。

## 限界

- このプロジェクトはコンテンツの適法性を判断しません。
- 推論結果は確率的で、誤判定があります。
- 公開・削除・制裁などの判断には人間レビューが必要です。
- 精度は、適法に用意されたデータセットの品質、ラベル一貫性、運用対象ドメインに強く依存します。

## ライセンス

MIT License. 詳細は [LICENSE](LICENSE) を参照してください。

---

# English: JP Adult Content Compliance Classifier

Computer vision training and inference tools for detecting potentially non-compliant explicit adult regions in Japanese content moderation workflows.

This project is intended for **compliance support, pre-publication review, and human moderation assistance**. It is not legal advice, does not determine legality by itself, and should not be used as a substitute for qualified legal or editorial review.

This repository intentionally contains **no adult images, no videos, no private annotations, no trained weights, no database dumps, and no scraping tools**. Bring your own lawful dataset and check your local laws before collecting, storing, training on, or distributing adult-content data.

## Intended Use

- Detecting possible mosaic-processing failures before publication
- Flagging explicit adult-region risk for human review
- Building internal compliance workflows for Japanese adult-content operations
- Researching adult-content moderation classifiers with lawful, locally managed datasets

## Out of Scope / Prohibited Use

Do not use this project to discover, collect, rank, recommend, or distribute explicit adult material. Do not use it for sexual exploitation, non-consensual intimate imagery, minors, harassment, or evading platform moderation. The model is designed as a review aid, not as a content discovery tool.

## What Is Included

- Dataset format documentation
- YOLO segmentation dataset validator
- YOLO segmentation training script for two classes:
  - `mosaic_nsfw`
  - `nude_genital`
- Image prediction CLI for trained YOLO weights
- Optional Stage 2 two-stream classifier training code:
  - crop branch: EfficientNet-B3
  - full-frame context branch: EfficientNet-B0
- Tiny non-image examples for config and manifest formats

## What Is Not Included

- Adult images or videos
- Trained model weights
- Private manifests or annotation exports
- URLs, source metadata, database dumps, or logs
- Scrapers or dataset collection tools

## Model Weights

The Git repository does not store trained weights. Official weights are published as GitHub Release assets.

Published release asset:

```text
https://github.com/rmoxpro/jp-adult-compliance-classifier/releases/download/v0.1.0/jp-adult-compliance-model-v0.1.0.zip
```

```text
jp-adult-compliance-model-v0.1.0.zip
  weights/yolo_best.pt
  weights/stage2_twostream_best.pt
  release_manifest.json
  README_WEIGHTS.md
  SHA256SUMS
```

SHA256:

```text
706ac5fa74a10150b7ecde7264ccc73135bb7366d32358ca7402f9e5e4f4e1d2
```

See [RELEASE.md](RELEASE.md) for release details.

## Documentation

- [Dataset format](docs/DATASET.md)
- [Usage](docs/USAGE.md)

## Evaluation

On internal evaluation data, the recommended released weights achieved approximately `0.94` YOLO mAP50 and approximately `99.3%` Stage 2 validation accuracy.

These are reference metrics from private internal evaluation sets, not reproducible public benchmark results. Real-world performance depends on the operating domain, image quality, labeling policy, and thresholds. Human review is required for final decisions.

## Quick Start

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Validate your YOLO dataset:

```bash
python scripts/validate_yolo_dataset.py --data /path/to/data.yaml
```

Train YOLO segmentation:

```bash
python scripts/train_yolo_seg.py \
  --data /path/to/data.yaml \
  --out-dir runs/yolo \
  --name jp_compliance_yolo \
  --epochs 100 \
  --imgsz 640 \
  --batch 12
```

Run image prediction with your trained weights:

```bash
python scripts/predict_image.py \
  --weights runs/yolo/jp_compliance_yolo/weights/best.pt \
  image1.jpg image2.jpg \
  --out results.json
```

Optional Stage 2 training:

```bash
python scripts/train_stage2_twostream.py \
  --pairs-dir /path/to/stage2_pairs \
  --out-dir runs/stage2 \
  --epochs 15 \
  --batch 32 \
  --focal-loss \
  --draw-bbox
```

## Dataset Overview

YOLO training uses standard Ultralytics layout:

```text
datasets/my_dataset/
  data.yaml
  images/train/
  images/val/
  images/test/
  labels/train/
  labels/val/
  labels/test/
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

Stage 2 expects a local `manifest.json` with crop/full-frame pairs. See [docs/DATASET.md](docs/DATASET.md).

## Publishing Safety Checklist

Before publishing a fork, release, or model card, verify that you are not committing:

- images, thumbnails, extracted frames, or videos
- label files with private filenames or source identifiers
- trained weights unless you intentionally release them under a separate policy
- database dumps
- logs containing URLs, filenames, usernames, or source metadata

The included `.gitignore` blocks common sensitive paths and file types, but you should still inspect `git status` and the final repository contents manually.

## Limitations

- This project does not determine whether content is legal.
- Predictions are probabilistic and can be wrong.
- Human review is required for any enforcement or publication decision.
- Accuracy depends heavily on lawful dataset quality, label consistency, and operating domain.

## License

MIT License. See [LICENSE](LICENSE).
