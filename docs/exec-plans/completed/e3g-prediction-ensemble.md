# E3g Execution Plan: Validation-Only Prediction Ensemble

Status: completed.

## Question

E3d ve E3f sonucunda en guclu adaylar validation macro-F1 acisindan birbirine cok yakindir:

- E3d all-fine-tuned FiLM: `0.7358 ± 0.0152`
- E3d all-fine-tuned gated: `0.7347 ± 0.0112`
- E3f mixed frozen-ViT + fine-tuned Swin/BEiT gated: `0.7361 ± 0.0100`

Bu modellerin per-class davranisi ayni degildir. E3g su soruyu test eder:

> Mevcut validation prediction dump'lari uzerinde probability averaging yaparak seed variance ve
> model-family tradeoff'larini azaltip validation macro-F1'i artirabilir miyiz?

## Literature Motivation

Skin lesion classification challenge methods commonly use prediction ensembling, multi-crop/test
time augmentation, metadata, and multiple model families to improve robustness. This is directly
relevant here because E3d/E3f produce near-tied validation means but different per-class behavior.

Relevant sources:

- Gessert et al. 2019, ISIC 2019 winner-style method with ensembles, metadata, resolution/cropping
  variants, and balancing strategies: https://arxiv.org/abs/1910.03910
- Perez et al. 2018, skin lesion augmentation/TTA-style prediction averaging for ISIC validation and
  test predictions: https://www.ic.unicamp.br/~sandra/pdf/papers/perez_ISIC18.pdf
- Ahammed et al. 2025, metadata plus dermoscopic image fusion for skin lesion classification:
  https://openaccess.thecvf.com/content/CVPR2025W/MULA2025/papers/Ahammed_Skin_Lesion_Classification_Using_Dermoscopic_Images_and_Clinical_Metadata_Insights_CVPRW_2025_paper.pdf
- Tschandl et al. 2018, HAM10000 dataset paper documenting class imbalance and benchmark context:
  https://www.nature.com/articles/sdata2018161

## Implementation Scope

E3g is a local, validation-only, post-hoc prediction ensemble diagnostic. It does not train or
fine-tune any model.

Allowed inputs:

```text
artifacts/runs/*e3d_metadata_fusion*/predictions.csv
artifacts/runs/*e3f_mixed_adaptation*/predictions.csv
```

Optional control inputs, if available and aligned:

```text
artifacts/runs/*s4b_multiseed_cpu*/predictions.csv
artifacts/runs/*e3c_metadata*/predictions.csv
```

Required probability columns:

```text
prob_akiec
prob_bcc
prob_bkl
prob_df
prob_nv
prob_mel
prob_vasc
```

## Non-Goals

- No test split usage.
- No training or fine-tuning.
- No TTA feature extraction.
- No unrestricted validation weight search.
- No class-specific threshold/bias tuning in the primary result.
- No claiming clinical diagnosis performance.

## Ensemble Policy

Primary ensembles are deterministic and low-overfit:

| Ensemble ID | Members | Weighting | Purpose |
|---|---|---|---|
| `e3d_film_seed_avg` | E3d FiLM seeds `7,13,42,101,202` | equal seed average | Reduce seed variance of current E3d best family. |
| `e3d_gated_seed_avg` | E3d gated seeds `7,13,42,101,202` | equal seed average | Compare gated all-fine-tuned family. |
| `e3f_gated_seed_avg` | E3f mixed gated seeds `7,13,42,101,202` | equal seed average | Reduce seed variance of current E3f best family. |
| `e3d_film_plus_e3f_gated_equal` | above two seed-averaged families | `0.5/0.5` | Combine best FiLM and mixed gated families. |
| `e3d_gated_plus_e3f_gated_equal` | above two gated families | `0.5/0.5` | Test source change under same operator family. |
| `top3_family_equal` | E3d FiLM, E3d gated, E3f gated seed averages | `1/3` each | Conservative family-level ensemble. |

Secondary diagnostic, only after primary equal-weight results:

| Ensemble ID | Weighting | Constraint |
|---|---|---|
| `top2_grid_weighted_diagnostic` | grid over E3d FiLM vs E3f gated | weights in `{0.25, 0.5, 0.75}` only |
| `top3_grid_weighted_diagnostic` | small simplex grid over E3d FiLM/E3d gated/E3f gated | weights in `{0, 0.25, 0.5, 0.75, 1}`, sum `1`, at most 15 candidates |

Weighted diagnostics must be reported separately from primary equal-weight ensembles because they
use validation labels for model-choice pressure.

## Evaluation Discipline

Primary metric:

- validation macro-F1.

Secondary metrics:

- accuracy,
- weighted-F1,
- macro precision,
- macro recall,
- per-class precision/recall/F1,
- confusion matrix,
- corrected-vs-broken counts versus strongest single family,
- member agreement/error overlap.

Selection rule:

1. Prefer primary equal-weight ensembles if they improve over E3d/E3f best validation means.
2. Use weighted diagnostics only as exploratory evidence.
3. Do not use test split for any ensemble selection.
4. If best improvement is smaller than about `0.002` macro-F1, report as practical tie rather than
   meaningful score gain.

## Alignment and Probability Checks

All member prediction dumps must match exactly on:

```text
sample_id
image_id
lesion_id
split
true_label
```

Checks:

- each member has `1504` validation rows,
- all seven probability columns exist,
- probabilities are finite,
- each row probability sum is close to `1.0`,
- class label order is fixed,
- no `test` rows appear.

## Expected Artifacts

```text
artifacts/runs/e3g_prediction_ensemble/
artifacts/runs/e3g_prediction_ensemble/run_config.json
artifacts/runs/e3g_prediction_ensemble/ensemble_members.csv
artifacts/runs/e3g_prediction_ensemble/ensemble_results.csv
artifacts/runs/e3g_prediction_ensemble/ensemble_per_class_metrics.csv
artifacts/runs/e3g_prediction_ensemble/ensemble_confusion_matrices.json
artifacts/runs/e3g_prediction_ensemble/ensemble_predictions_<ensemble_id>.csv
artifacts/runs/e3g_prediction_ensemble/error_overlap_summary.csv
artifacts/report_assets/tables/e3g_prediction_ensemble_results.csv
artifacts/report_assets/tables/e3g_prediction_ensemble_per_class_metrics.csv
artifacts/report_assets/tables/e3g_prediction_ensemble_vs_controls.csv
artifacts/report_assets/figures/e3g_prediction_ensemble_macro_f1.png
docs/report_notes/e3g_prediction_ensemble.md
```

## Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Equal-weight ensembles do not improve validation macro-F1. | Report as evidence that current top families share too many errors; proceed to TTA/multi-view as next score-oriented experiment. |
| Weighted grid improves but only by tiny margin. | Keep as diagnostic, not primary result, due validation overfit risk. |
| Ensemble improves macro-F1 by sacrificing a clinically relevant minority class. | Report per-class tradeoff; do not select solely on mean macro-F1. |
| Prediction dumps are missing or row order differs. | Stop and regenerate affected run artifacts; do not align by implicit row index. |

## First Implementation Step

Add `scripts/ensemble_predictions.py` that loads named run groups, verifies row alignment and
probability schema, computes the primary equal-weight ensembles, then writes standard validation
metrics and report assets. Run only the primary equal-weight ensembles first.
