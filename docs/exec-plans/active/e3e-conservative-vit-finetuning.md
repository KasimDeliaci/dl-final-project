# E3e Execution Plan: Conservative ViT Fine-Tuning Diagnostic

Status: active.

## Question

The canonical partial fine-tuning run slightly reduced ViT single-backbone validation macro-F1
relative to the frozen ViT baseline (`0.6924 -> 0.6876`). E3e asks whether this drop was caused by
an overly aggressive ViT adaptation policy rather than by fine-tuning being generally unhelpful for
HAM10000 benchmark dermoscopic image classification.

## Hypothesis

Lowering the ViT backbone learning rate and/or reducing trainable transformer depth may preserve
the strong frozen ViT representation while allowing limited domain-specific adaptation. If the
canonical fine-tuned ViT drop came from over-adaptation, a conservative policy should improve the
ViT single-backbone MLP and may improve downstream ViT+Swin+BEiT fusion.

## Literature Motivation

Medical-image transfer learning studies report that fine-tuning strategy effectiveness varies by
architecture and modality, and that partial unfreezing, adaptive learning rates, and regularization
are meaningful axes rather than interchangeable implementation details. Vision Transformer work
also motivates parameter-limited adaptation: fine-tuning only selected ViT components can reduce
compute and memory while still adapting to downstream classification tasks.

Relevant sources:

- Davila et al. 2024, "Comparison of fine-tuning strategies for transfer learning in medical image
  classification": https://arxiv.org/abs/2406.10050
- Touvron et al. 2022, "Three things everyone should know about Vision Transformers":
  https://arxiv.org/abs/2203.09795

## Implementation Scope

E3e adds a narrow Colab-run fine-tuning diagnostic over ViT only:

1. `vit_b16_last2_lr5e-6`
   - Same trainable depth as canonical E3 ViT.
   - Backbone LR reduced from `1e-5` to `5e-6`.
   - Tests whether LR alone caused representation degradation.
2. `vit_b16_last1_lr5e-6`
   - Only the final ViT transformer block plus norm/head is trainable.
   - Backbone LR `5e-6`.
   - Tests whether opening two blocks was too aggressive.

Both conditions keep the canonical split, image size, train transforms, class weighting, head LR,
epochs, patience, feature extraction format, and validation macro-F1 checkpoint selection.

## Non-Goals

- No test split usage.
- No Swin or BEiT re-fine-tuning.
- No class-balanced focal loss, balanced sampler, SMOTE, new augmentation, TTA, LoRA, adapters, or
  cross-attention.
- No replacement of canonical `finetuned` artifacts.
- No final model selection from a single seed without downstream comparison.

## Freeze/Unfreeze Policy

Canonical E3 control:

- `vit_b16`: `last_2_blocks`, backbone LR `1e-5`, head LR `1e-4`.

E3e conditions:

- `vit_b16_last2_lr5e-6`: `last_2_blocks`, backbone LR `5e-6`, head LR `1e-4`.
- `vit_b16_last1_lr5e-6`: `last_1_block`, backbone LR `5e-6`, head LR `1e-4`.

All earlier ViT blocks remain frozen. Checkpoint metadata must record trainable prefixes, trainable
parameter count, learning rates, and validation-selected epoch.

## Feature Extraction Policy

After each selected checkpoint, extract train and validation feature caches only:

- `artifacts/features/ham10000/finetuned_vit_last2_lr5e6/vit_b16/`
- `artifacts/features/ham10000/finetuned_vit_last1_lr5e6/vit_b16/`

The cache format must remain Sprint 2 compatible: split `.pt` payloads, split CSV manifest, and
backbone-level `manifest.json`. Required counts:

- train: `7008`
- validation: `1504`

The test split is not loaded for E3e.

## Downstream Matrix Policy

Run validation-only downstream checks after Colab feature extraction:

1. ViT single-backbone MLP for each E3e feature source.
2. Mixed ViT+Swin+BEiT concat fusion for each E3e ViT source, using canonical E3 Swin/BEiT caches:
   - `finetuned_vit_last2_lr5e6_plus_s4_swin_beit`
   - `finetuned_vit_last1_lr5e6_plus_s4_swin_beit`
3. If either mixed concat result is competitive with the E3d metadata-conditioned controls, rerun
   the lightweight E3d operators on that mixed source as a validation-only follow-up:
   - `triple_metadata_film`
   - `triple_metadata_gated_backbone`

Do not broaden to full pairwise matrices unless the ViT single result or triple concat clearly
supports doing so.

## Evaluation Discipline

Primary metric:

- validation macro-F1.

Secondary metrics:

- validation accuracy,
- weighted-F1,
- per-class F1,
- training/validation curves,
- runtime,
- trainable parameter ratio.

Comparison anchors:

- frozen ViT single: `0.6924`.
- canonical fine-tuned ViT single: `0.6876`.
- E3b fine-tuned ViT+Swin+BEiT concat mean: `0.7246 ± 0.0143`.
- E3c raw metadata concat mean: `0.7278 ± 0.0058`.
- E3d FiLM mean: `0.7358 ± 0.0152`.
- E3d gated mean: `0.7347 ± 0.0112`.

## Validation-Only Selection

Checkpoint selection uses validation macro-F1 only. Downstream selection and escalation also use
validation macro-F1 only. Test metrics are not computed.

## Colab and Drive Policy

The Colab runner must clone/pull GitHub before running so Colab sees the latest scripts and configs.

Drive output namespace:

```text
/content/drive/MyDrive/dl-final-artifact/e3e_conservative_vit/
```

Local repo artifact namespaces:

```text
artifacts/checkpoints/ham10000/e3e_vit_last2_lr5e6/
artifacts/checkpoints/ham10000/e3e_vit_last1_lr5e6/
artifacts/features/ham10000/finetuned_vit_last2_lr5e6/
artifacts/features/ham10000/finetuned_vit_last1_lr5e6/
artifacts/features/ham10000/finetuned_vit_last2_lr5e6_plus_s4_swin_beit/
artifacts/features/ham10000/finetuned_vit_last1_lr5e6_plus_s4_swin_beit/
artifacts/runs/*e3e*
```

The notebook may restore canonical Sprint 4 Swin/BEiT feature caches from
`/content/drive/MyDrive/dl-final-artifact/artifacts/features/ham10000/finetuned/` but must not write
over that canonical source.

## Verification Gates

- `PYTHONPATH=src uv run ruff check .`
- Relevant pytest suite passes before pushing.
- Smoke run with `--limit-per-split 2` and `--no-pretrained`.
- Full split guard confirms train `7008`, val `1504`, test `1503`.
- Fine-tuned cache row counts: train `7008`, val `1504`.
- ViT feature dimension: `768`.
- No NaN/Inf in feature caches.
- Prediction dumps have `1504` rows and seven probability columns.
- Checkpoint metadata records `selection_metric=validation_macro_f1`.
- E3e run configs record `test_policy=not_loaded_or_used_in_e3e`.
- Generated checkpoints, features, runs, predictions, and report assets remain Git-ignored.

## Expected Artifacts

Colab/GPU artifacts:

```text
artifacts/checkpoints/ham10000/e3e_vit_last2_lr5e6/vit_b16/best.pt
artifacts/checkpoints/ham10000/e3e_vit_last1_lr5e6/vit_b16/best.pt
artifacts/features/ham10000/finetuned_vit_last2_lr5e6/vit_b16/manifest.json
artifacts/features/ham10000/finetuned_vit_last1_lr5e6/vit_b16/manifest.json
artifacts/runs/e3e_finetune_vit_last2_lr5e6_seed42/
artifacts/runs/e3e_finetune_vit_last1_lr5e6_seed42/
```

Downstream artifacts:

```text
artifacts/runs/*_s2_finetuned_vit_last*_vit_none_mlp_e3e_*_seed42/
artifacts/runs/*_s3_finetuned_vit_last*_plus_s4_swin_beit_vit-swin-beit_concat_mlp_e3e_*_seed42/
```

Report note to create after results:

```text
docs/report_notes/e3e_conservative_vit_finetuning.md
```

## Risks and Fallbacks

| Risk | Fallback |
| --- | --- |
| Lower LR under-adapts and remains below frozen ViT. | Report as evidence that conservative LR alone did not recover ViT; do not broaden the search. |
| Last-one-block policy improves single ViT but not fusion. | Report representation-level improvement but keep final fusion story anchored to stronger E3d controls. |
| Colab runtime is insufficient. | Run only `last1_lr5e-6` first; it is the cleaner over-adaptation probe. |
| Canonical Swin/BEiT caches are missing in Colab. | Restore them from Drive; if unavailable, stop and sync artifacts rather than recomputing or mixing stale caches. |
| Results are worse. | Keep as a negative ablation explaining ViT fine-tuning sensitivity; do not hide the run. |

## Report Note Template

```markdown
# E3e Conservative ViT Fine-Tuning Diagnostic

## Question
Did the canonical ViT partial fine-tuning drop come from an overly aggressive adaptation policy?

## Protocol
Validation-only. Fixed lesion-aware split. Test set not loaded. Two ViT-only Colab conditions:
last-2-block LR 5e-6 and last-1-block LR 5e-6.

## Results
| Condition | ViT single macro-F1 | Triple concat macro-F1 | Notes |

## Interpretation
Use cautious language: conservative fine-tuning [did/did not] recover the frozen ViT baseline.
Discuss LR/unfreeze sensitivity, overfitting risk, and downstream fusion impact.

## Limitations
Single seed, validation-only, one backbone, no test conclusion.
```

## First Implementation Step

Add E3e configs and Colab runner, push to GitHub, then open the notebook in Colab from Safari.
