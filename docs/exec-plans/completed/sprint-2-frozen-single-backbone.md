# Sprint 2 Frozen Transformer Single-Backbone Execution Plan

## Objective

Implement frozen transformer feature extraction and single-backbone MLP baselines for HAM10000 using the canonical Sprint 1 lesion-aware split.

Sprint 2 answers one question:

> How strong are Vanilla ViT, Swin Transformer, and DeiT III-Small as frozen single-backbone feature extractors on the HAM10000 benchmark dermoscopic image classification task?

## Context

Sprint 1 is complete and provides the fixed evaluation base:

- Train: 7,008 images / 5,233 lesions
- Validation: 1,504 images / 1,117 lesions
- Test: 1,503 images / 1,120 lesions
- Cross-split lesion leakage: 0
- All seven classes present in every split

The earlier `dl-assignment` Sprint 2 flow is the main workflow reference: deterministic image dataset, frozen backbone wrapper, feature cache with alignment metadata, train-only class weighting, MLP training, validation macro-F1 checkpoint selection, metrics, per-class reports, confusion matrices, prediction dumps, and aggregate report assets.

## Implementation Scope

- Add transformer backbone wrappers under `src/dl_final/models/`.
- Use `timm` model IDs:
  - `vit_base_patch16_224.augreg_in21k_ft_in1k`
  - `swin_tiny_patch4_window7_224.ms_in1k`
  - `deit3_small_patch16_224.fb_in1k`
- Remove/bypass classifier heads through `num_classes=0` and `global_pool`.
- Use deterministic ImageNet-style preprocessing at `224x224`.
- Cache train and validation features as the required Sprint 2 modeling inputs.
- Permit test feature cache generation for future audit only, but do not train, select, or rank models by test metrics in Sprint 2.
- Train one MLP per backbone from cached train features.
- Select the best MLP checkpoint by validation macro-F1.
- Export validation metrics and artifacts for each single-backbone run.
- Write aggregate Sprint 2 report assets and `docs/report_notes/sprint2_frozen_single.md`.

## Non-Goals

- No concatenation fusion.
- No weighted fusion.
- No pairwise or three-backbone comparison models.
- No transformer fine-tuning.
- No test-set model selection.
- No notebook work unless a thin launcher is later needed.
- No clinical diagnosis framing.

## Decisions To Record

- Exact model IDs and library.
- ViT/DeiT pooling policy: use timm `num_classes=0, global_pool="token"` so the classifier-free CLS-token representation is the canonical feature vector.
- Swin pooling policy: use timm `num_classes=0, global_pool="avg"` so the classifier-free pooled final-stage representation is the canonical feature vector.
- Feature cache format: `.pt` tensor payload plus per-split CSV manifest and per-backbone JSON manifest.
- MLP baseline policy: one modest MLP recipe shared across backbones, validation macro-F1 checkpoint selection.
- Class imbalance strategy: train-only class-weighted cross entropy by default; weighted sampling remains an extension.
- Test usage policy: test caches may be materialized for final audit readiness, but Sprint 2 reports validation metrics only.

## Expected Artifacts

Tracked:

```text
src/dl_final/data/datasets.py
src/dl_final/data/transforms.py
src/dl_final/models/backbones.py
src/dl_final/models/feature_extractors.py
src/dl_final/models/mlp.py
src/dl_final/features/cache.py
src/dl_final/features/extract.py
src/dl_final/evaluation/metrics.py
src/dl_final/evaluation/reports.py
src/dl_final/training/loops.py
src/dl_final/training/early_stopping.py
src/dl_final/training/optim.py
scripts/extract_features.py
scripts/train_mlp.py
tests/test_sprint2_features.py
docs/report_notes/sprint2_frozen_single.md
```

Generated and Git-ignored:

```text
artifacts/features/ham10000/frozen/{vit_b16,swin_tiny,deit3_small}/
artifacts/runs/s2_frozen_*_mlp_seed42/
artifacts/report_assets/tables/single_backbone_frozen_results.csv
artifacts/report_assets/tables/single_backbone_frozen_per_class_metrics.csv
artifacts/report_assets/figures/frozen_single_backbone_macro_f1.svg
```

## Verification Gates

- `uv run` environment resolves Sprint 2 dependencies.
- Unit/smoke tests pass.
- Transformer forward smoke test passes with pretrained downloads disabled.
- All frozen model parameters have `requires_grad=False`.
- Feature output is 2D and has the expected dimension.
- Feature cache rows match split CSV row order.
- Feature cache contains no NaN or Inf values.
- Train/validation row counts match Sprint 1 split counts for full extraction, or prefix alignment is explicitly marked for smoke caches.
- Confusion matrix label order matches configured class order.
- Prediction dump contains one probability column per class.
- MLP checkpoint selection uses validation macro-F1.
- Generated feature caches, checkpoints, predictions, and run artifacts remain ignored by Git.

## Risks And Fallbacks

| Risk | Fallback |
|---|---|
| Pretrained weight download is unavailable. | Keep `--no-pretrained` smoke path for code verification; defer full extraction until weights are available. |
| Full image feature extraction is slow locally. | Use cache-once workflow; run full extraction on a GPU machine or Colab with the same scripts. |
| DeiT III model ID changes or is unavailable in installed `timm`. | Fail fast with supported-model listing and record a replacement decision before changing scope. |
| MLP overfits validation quickly. | Keep early stopping, dropout, weight decay, and class-weighted CE; deeper search remains outside Sprint 2 baseline. |
| Minority-class metrics are unstable. | Report support counts and use macro-F1/per-class metrics instead of accuracy-only interpretation. |

## Report Note Template

```text
Question:
Recipe:
Fixed controls:
Validation results:
Interpretation:
Evidence strength:
Report decision:
Limitations:
```

## First Implementation Step

Add the reusable Sprint 2 implementation surface:

1. HAM10000 split-backed image dataset and deterministic transform.
2. Transformer backbone wrapper smoke-tested without pretrained weights.
3. Feature cache read/write and alignment checks.
4. MLP training/evaluation artifacts on cached features.

After this surface passes smoke tests, run extraction/training commands with the real split files.
