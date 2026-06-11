# Sprint 4 Fine-Tuned Transformer Feature Execution Plan

## Objective

Implement controlled partial fine-tuning for the forward transformer set and convert the selected
validation checkpoints back into cached feature extractors.

Sprint 4 answers:

> Do fine-tuned transformer features improve HAM10000 benchmark dermoscopic image classification
> validation macro-F1 over frozen transformer features, and does feature fusion still help after
> partial fine-tuning?

## Context Read

Sprint 1 is complete: canonical lesion-aware split has `7008` train images, `1504` validation
images, `1503` test images, all seven classes present, and zero cross-split lesion leakage.

Sprint 2 is complete: frozen single-backbone validation macro-F1 is ViT `0.6924`, Swin `0.6115`,
DeiT III-Small `0.5017`, and BEiT-Base screening `0.4759`.

Sprint 3 is complete: the best modest frozen fusion result is `ViT + Swin + BEiT concat` at
validation macro-F1 `0.6988`. The E2b MLP capacity diagnostic produced a stronger frozen baseline:
`ViT + Swin concat + deep_reg MLP` at validation macro-F1 `0.7262`. Sprint 4 results must be
compared against both levels, not only the modest E2 matrix.

The old `dl-assignment` Sprint 4 flow provides the engineering pattern: fine-tune an image
classifier checkpoint, strip the temporary head, extract fine-tuned feature caches, then reuse the
cached-feature MLP/fusion matrix. This repo must adapt the pattern to transformers and must not
compute Sprint 4 test metrics.

## Implementation Scope

- Add transformer fine-tuning support under `src/dl_final/`.
- Add a reproducible launcher at `scripts/finetune_backbone.py`.
- Use the forward backbone set:
  - `vit_b16`
  - `swin_tiny`
  - `beit_base`
- Keep `deit3_small` as screened/planned baseline only; do not fine-tune it unless a later decision
  supersedes D021.
- Use a temporary 7-class classification head for image-level fine-tuning.
- Select checkpoints by validation macro-F1 only.
- Extract train/validation fine-tuned feature caches under
  `artifacts/features/ham10000/finetuned/<backbone>/`.
- Keep the cache format compatible with Sprint 2 frozen caches.
- Reuse `scripts/train_mlp.py --feature-source finetuned` for fine-tuned single-backbone MLP runs.
- Reuse `scripts/run_fusion_matrix.py --feature-source finetuned` for representative fine-tuned
  concat/weighted fusion runs.
- Update experiment registry, decisions, commands, configs, tests, and the Sprint 4 report note.

## Non-Goals

- Do not compute test metrics in Sprint 4.
- Do not use test split for checkpoint, hyperparameter, fusion, or backbone selection.
- Do not fine-tune DeiT III-Small in the canonical Sprint 4 compute budget.
- Do not implement full end-to-end transformer fusion over raw images.
- Do not add notebooks unless a thin Colab launcher is explicitly needed later.
- Do not claim clinical diagnosis or deployment usefulness.

## Exact Fine-Tuning Policy

Common policy:

- Pretrained timm model with `num_classes=7`.
- Image size `224`.
- Train transform: resize, light geometric augmentation, color jitter, tensor conversion, ImageNet
  normalization.
- Validation/extraction transform: deterministic resize and ImageNet normalization.
- Loss: train-only class-weighted cross entropy by default.
- Optimizer: AdamW.
- Scheduler: cosine annealing over the configured epoch cap.
- Checkpoint selection: best validation macro-F1.
- Default full-run epoch cap: `8` epochs, patience `3`.
- Default full-run batch size: `16` for Colab safety.
- Default LR: backbone `1e-5`, head `1e-4`, weight decay `1e-4`.
- Mixed precision: enabled on CUDA/MPS when available, optional off switch for smoke tests.

Per-backbone policy:

| Backbone | Temporary head | Trainable transformer blocks | Trainable normalization/head |
|---|---|---|---|
| `vit_b16` | timm classifier head | last 2 `blocks` | `norm`, `head` |
| `swin_tiny` | timm classifier head | last Swin stage (`layers[-1]`) | `norm`, `head` |
| `beit_base` | timm classifier head | last 2 `blocks` | `fc_norm` or `norm` if present, `head` |

## Freeze/Unfreeze Policy

- All model parameters are set `requires_grad=False` first.
- Only the policy-listed final blocks/stage and temporary head/norm parameters are set trainable.
- A trainability summary must be written with total parameters, trainable parameters, frozen
  parameters, trainable ratio, policy name, and trainable parameter name prefixes.
- Unit tests must verify representative trainable and frozen parameter names for ViT, Swin, and
  BEiT with `pretrained=False`.

## Feature Extraction Policy After Fine-Tuning

- Load the selected checkpoint into the same timm architecture with `num_classes=7`.
- Replace/reset the classifier for feature extraction with the Sprint 2 policy:
  - ViT: classifier-free CLS token, `global_pool="token"`, feature dim `768`.
  - Swin: classifier-free average pooled final-stage representation, `global_pool="avg"`,
    feature dim `768`.
  - BEiT: classifier-free average pooled patch-token representation, `global_pool="avg"`,
    feature dim `768`.
- Save cache payloads exactly like frozen caches: `.pt`, split manifest CSV, and backbone
  `manifest.json`.
- Cache metadata must include `feature_source=finetuned`, checkpoint path, checkpoint metadata,
  checkpoint selection metric, unfreeze policy, and fine-tuning config.
- Only train and validation caches are canonical Sprint 4 outputs. Test cache extraction remains a
  Sprint 5/final-audit operation.

## Downstream MLP/Fusion Matrix Policy

- Single-backbone fine-tuned MLP:
  - Run `scripts/train_mlp.py --feature-source finetuned --backbones vit_b16 swin_tiny beit_base`.
  - Use the same train-only StandardScaler and class-weighted MLP discipline as frozen baselines.
- Representative fine-tuned fusion:
  - Minimal canonical shortlist: `vit_b16+swin_tiny` and `vit_b16+swin_tiny+beit_base`.
  - Run `concat`, `weighted_learned_512`, and optionally `weighted_pca_384` if compute permits.
  - Start with the same modest MLP recipe for comparability.
  - If results are near frozen baselines, optionally run the E2b-style stronger MLP variants as a
    diagnostic, clearly separated from the canonical modest matrix.

## Evaluation Discipline

- Primary metric: validation macro-F1.
- Required validation metrics: accuracy, macro precision, macro recall, macro-F1, weighted-F1,
  per-class precision/recall/F1, confusion matrix.
- Checkpoint selection uses validation macro-F1 only.
- MLP/fusion selection uses validation macro-F1 only.
- Confusion matrix label order stays `akiec`, `bcc`, `bkl`, `df`, `nv`, `mel`, `vasc`.
- Per-class gains must be interpreted with support counts visible.

## Test Usage Policy

Sprint 4 must not compute test metrics. Test split may remain present in data files but must not be
loaded by the fine-tuning training loop or downstream Sprint 4 MLP/fusion commands. Final audit is
reserved for Sprint 5 after validation-selected candidates are pre-registered.

## Verification Gates

- `PYTHONPATH=src uv run ruff check .` passes.
- Relevant pytest suite passes with `uv run`.
- Freeze/unfreeze policy checks pass for all three forward backbones.
- One local smoke command can fine-tune a tiny prefix for one epoch with `--no-pretrained` and
  produce a checkpoint plus train/validation finetuned caches.
- Fine-tuned cache output shape matches expected feature dimensions.
- Feature cache row alignment checks cover `sample_id`, `image_id`, `lesion_id`, label, and split
  row order.
- Canonical full cache row counts must be train `7008`, validation `1504`.
- Prediction dumps from downstream MLP/fusion runs must contain `1504` validation rows.
- Feature tensors and prediction probabilities contain no NaN/Inf.
- Scaler fit policy remains train-only.
- Checkpoint metadata records validation macro-F1 selection.
- Generated checkpoints, feature caches, predictions, and run folders are Git-ignored.

## Expected Artifacts

Tracked:

```text
scripts/finetune_backbone.py
src/dl_final/training/finetune.py
src/dl_final/models/backbones.py
src/dl_final/data/transforms.py
tests/test_sprint4_finetune.py
configs/experiments/finetune_backbones.yaml
configs/experiments/finetuned_feature_matrix.yaml
docs/EXPERIMENT_REGISTRY.md
docs/DECISIONS.md
docs/COMMANDS.md
docs/report_notes/sprint4_finetuned_features.md
```

Generated and Git-ignored:

```text
artifacts/checkpoints/ham10000/finetuned/{vit_b16,swin_tiny,beit_base}/
artifacts/features/ham10000/finetuned/{vit_b16,swin_tiny,beit_base}/
artifacts/runs/*_s4_finetune_*_seed42/
artifacts/runs/*_s2_finetuned_*_none_mlp*_seed42/
artifacts/runs/*_s3_finetuned_*_concat_mlp*_seed42/
artifacts/report_assets/tables/single_backbone_finetuned_results.csv
artifacts/report_assets/tables/finetuned_fusion_results.csv
```

## Risks And Fallbacks

| Risk | Fallback |
|---|---|
| Colab OOM on BEiT. | Batch size `8`, mixed precision, freeze more blocks, one backbone at a time. |
| Fine-tuning overfits. | Lower backbone LR, increase weight decay/dropout downstream, keep negative result. |
| Fine-tuning underperforms frozen ViT/Swin. | Report as controlled negative result; discuss overfitting, LR sensitivity, insufficient unfreezing, and limited domain adaptation. |
| Weighted fusion underperforms concat. | Keep as diagnostic; do not interpret learned weights as direct backbone quality ranking. |
| Full Sprint 4 compute is unavailable locally. | Verify with smoke tests locally and run full `finetune_backbone.py` commands on Colab using `MyDrive/dl-final-artifact/`. |

## Report Note Template

```text
Question:
Fine-tuning policy:
Fixed controls:
Validation results:
Frozen baselines:
Per-class behavior:
Runtime and compute:
Interpretation:
Evidence strength:
Report decision:
Limitations:
```

## First Implementation Step

Add transformer fine-tuning helpers:

1. classification backbone construction and partial unfreeze policy;
2. train/validation image classifier loop selected by validation macro-F1;
3. classifier-free feature extraction from the selected checkpoint;
4. tests for trainability, checkpoint metadata, cache shape, and cache alignment.

Then run a one-epoch `--no-pretrained --limit-per-split` smoke command before any full Colab run.

## Completion Note

Completed on 2026-06-11 with a full Colab/GPU validation-only run over `vit_b16`, `swin_tiny`, and
`beit_base`.

Produced local generated artifacts under:

```text
artifacts/checkpoints/ham10000/finetuned/
artifacts/features/ham10000/finetuned/
artifacts/runs/s4_finetune_*_seed42/
artifacts/runs/*_s2_finetuned_*_none_mlp_seed42/
artifacts/runs/*_s3_finetuned_*_mlp_s4_*_seed42/
artifacts/report_assets/tables/*finetuned*.csv
artifacts/report_assets/figures/finetuned_*.png
```

Validation macro-F1 summary:

| Configuration | Feature source | Fusion | Validation macro-F1 |
|---|---|---|---:|
| ViT | finetuned | none | `0.6876` |
| Swin | finetuned | none | `0.6517` |
| BEiT | finetuned | none | `0.5181` |
| ViT + Swin | finetuned | concat | `0.7161` |
| ViT + Swin + BEiT | finetuned | concat | `0.7298` |

Best validation-only result: `vit_b16+swin_tiny+beit_base concat` at `0.7298` validation macro-F1.
This exceeded the modest frozen triple concat baseline (`0.6988`) and narrowly exceeded the E2b
stronger-MLP frozen ViT+Swin diagnostic baseline (`0.7262`). Test metrics were not computed.

Post-download local verification confirmed train/validation cache shapes `(7008, 768)` and
`(1504, 768)` for all three backbones, split row-order alignment, finite feature tensors, 12
fine-tuned prediction dumps with 1,504 validation rows, run configs with
`test_policy=not_used_in_sprint4`,
and Git-ignored generated artifacts.
