# Model Card: JP Adult Content Compliance Classifier

This model package is intended for detecting potentially non-compliant explicit adult regions in Japanese content moderation workflows. It is designed to support human review of mosaic-processing failures and explicit-region risk.

## Intended Use

- Pre-publication review support
- Adult-content compliance workflow support
- Flagging possible mosaic-processing failures for human review
- Internal moderation research with lawful datasets

## Not Intended For

- Discovering, collecting, ranking, recommending, or distributing adult content
- Sexual exploitation, non-consensual intimate imagery, minors, harassment, or moderation evasion
- Automated legal decisions without human review

## Model Files

Recommended release package:

- `weights/yolo_best.pt`: YOLO segmentation detector for candidate regions
- `weights/stage2_twostream_best.pt`: optional Stage 2 two-stream classifier for false-positive reduction
- `release_manifest.json`: file metadata and expected class names
- `SHA256SUMS`: checksums for release integrity

## Classes

YOLO detector:

- `0`: `mosaic_nsfw`
- `1`: `nude_genital`

Stage 2 classifier:

- `mosaic`
- `other`
- `uncensored`

## Evaluation Metrics

Internal evaluation for the recommended released weights:

- YOLO detector: approximately `0.94` mAP50 on internal validation/test data
- Stage 2 two-stream classifier: `0.9927` validation accuracy

These metrics are from private internal evaluation sets. They are not public benchmark results and should not be interpreted as a guarantee of performance in other domains.

## License

The source code and the released model weights are distributed under the MIT License. See `LICENSE` in the repository for the license text.

## Limitations

- The model does not determine legality.
- The model can produce false positives and false negatives.
- Human review is required before publication, enforcement, or legal decisions.
- Performance depends on the dataset domain, labeling policy, image quality, resolution, and moderation thresholds.

## Safety

The release package should not include adult images, videos, private annotations, logs, URLs, or database dumps. Model weights should be distributed separately from source code, preferably as GitHub Release assets or a gated model repository.
