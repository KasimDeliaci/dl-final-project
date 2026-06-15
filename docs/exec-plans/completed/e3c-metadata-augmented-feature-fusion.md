# E3c Metadata-Augmented Cached Feature Fusion Execution Plan

## Objective

Evaluate whether HAM10000 benchmark metadata improves downstream validation macro-F1 when added
to the strongest cached transformer feature configurations.

E3c answers:

> Do age, sex, and lesion localization provide complementary signal beyond fine-tuned transformer
> image features for HAM10000 benchmark dermoscopic image classification?

This is a validation-only diagnostic over existing cached features. It is not a new fine-tuning
run, not a test-set audit, and not a clinical decision-support claim.

## Context

E3 completed partial fine-tuning for `vit_b16`, `swin_tiny`, and `beit_base`, then extracted
train/validation feature caches. The best single-seed fine-tuned downstream result was
`vit_b16+swin_tiny+beit_base concat` at validation macro-F1 `0.7298`.

E3b repeated the top cached-feature MLP candidates over seeds `7`, `13`, `42`, `101`, and `202`.
The fine-tuned triple concat condition had the highest mean validation macro-F1
(`0.7246 +/- 0.0143`), but the margin over the strongest frozen diagnostic remains small enough
that further claims should be cautious.

HAM10000 split files already contain metadata columns:

- `age`
- `sex`
- `localization`

Canonical split row counts remain train `7008`, validation `1504`, and test `1503`. E3c must load
only train and validation rows.

## Literature Motivation

Recent skin-lesion classification work commonly studies image+metadata fusion, with structured
metadata such as age, sex, and anatomical site processed by a small dense network or encoded as a
feature vector before fusion with image embeddings. This supports E3c as a reasonable benchmark
diagnostic, but not as evidence of clinical deployment readiness.

Useful framing:

- Metadata may help classes whose prevalence or visual ambiguity is correlated with age or
  anatomical location.
- Metadata can also encode dataset-specific correlations, so any gain must be discussed as
  benchmark behavior and validated under the same lesion-aware split discipline.
- Negative or neutral results are still informative: strong transformer image features may already
  capture most separable signal available under this split.

## Implementation Scope

- Add metadata preprocessing logic under `src/dl_final/features/`.
- Add or extend a reproducible training entry point for metadata-augmented cached-feature MLP runs.
- Reuse existing cached feature loading, row-alignment checks, metrics, confusion matrix, prediction
  dump, and report asset conventions.
- Run a small, pre-registered condition set over multiple seeds.
- Update experiment registry, decisions, commands, and report notes after implementation.

Expected implementation files:

```text
src/dl_final/features/metadata.py
scripts/train_metadata_augmented_mlp.py
scripts/summarize_e3c_metadata.py
tests/test_metadata_features.py
tests/test_e3c_metadata_augmented.py
docs/report_notes/e3c_metadata_augmented_features.md
```

If the existing `train_mlp.py`/`run_fusion_matrix.py` can be extended cleanly without confusing the
image-only artifact contract, prefer reuse. If that creates brittle conditionals, keep E3c in a
separate script and share core utilities from `src/dl_final/`.

## Non-Goals

- Do not compute test metrics.
- Do not use test split for metadata imputation, category discovery, scaling, model selection, or
  diagnostics.
- Do not use `dx`, `dx_type`, `dataset`, `image_path`, `image_id`, `sample_id`, or `lesion_id` as
  model input.
- Do not rerun transformer fine-tuning.
- Do not introduce probability-level late fusion or learned validation meta-ensembles in this
  experiment.
- Do not add notebooks unless a thin Colab/local launcher becomes necessary.
- Do not claim clinical diagnosis, screening utility, or deployment readiness.

## Metadata Policy

Allowed input fields:

| Field | Type | Policy |
|---|---|---|
| `age` | numeric | Fit train median imputer and train-only scaler. |
| `sex` | categorical | Fill missing/blank as `unknown`; fit train categories only. |
| `localization` | categorical | Fill missing/blank as `unknown`; fit train categories only. |

Encoding policy:

- Fit preprocessing on train split only.
- Apply the fitted preprocessing object to validation.
- Record train median age, age scaler statistics, categorical vocabularies, unknown handling, and
  output dimension in metadata preprocessing artifacts.
- Use deterministic column order:
  - `age_scaled`
  - one-hot `sex` categories in train vocabulary order
  - one-hot `localization` categories in train vocabulary order
- Handle unseen validation categories by mapping to `unknown` if present in the train vocabulary;
  otherwise produce an all-zero category block and record the count.
- Do not fit any metadata preprocessing from validation or test rows.

## Model Conditions

Run all conditions with seeds `7`, `13`, `42`, `101`, and `202` on CPU unless runtime becomes
unreasonable.

Canonical E3c conditions:

| Condition ID | Image feature source | Backbones | Fusion | Metadata |
|---|---|---|---|---|
| `metadata_only` | none | none | none | age + sex + localization |
| `ft_vit_swin_concat_plus_metadata` | `finetuned` | `vit_b16+swin_tiny` | concat | yes |
| `ft_vit_swin_beit_concat_plus_metadata` | `finetuned` | `vit_b16+swin_tiny+beit_base` | concat | yes |

Image-only controls:

- Reuse E3b CPU multi-seed summaries for:
  - `ft_vit_swin_concat`
  - `ft_vit_swin_beit_concat`

Optional diagnostic only if the canonical run is stable:

- `ft_vit_plus_metadata`, to test whether metadata changes the interpretation of the weaker
  fine-tuned ViT single-backbone result.

## Training Policy

- Use the same class order and metric code as previous MLP/fusion runs.
- Use train-only class weights.
- Use train-only scaler for image features exactly as existing cached-feature MLP runs do.
- Concatenate metadata features after image feature scaling for `plus_metadata` conditions.
- For metadata-only, train the same MLP classifier family on the metadata vector without image
  features.
- Keep the MLP recipe close to E3b comparability unless a smoke run shows severe instability.
- Checkpoint selection uses validation macro-F1 only.
- Save the selected MLP checkpoint, run config, metrics, per-class metrics, confusion matrix,
  prediction dump, training history, runtime metadata, and preprocessing metadata for every seed.

## Evaluation Discipline

- Primary metric: validation macro-F1.
- Secondary metrics: accuracy, macro precision, macro recall, weighted-F1, per-class precision,
  per-class recall, per-class F1.
- Report mean, std, min, max validation macro-F1 over seeds.
- Compare plus-metadata conditions against image-only E3b controls, not only against single-seed E3.
- Confusion matrix label order remains:
  - `akiec`
  - `bcc`
  - `bkl`
  - `df`
  - `nv`
  - `mel`
  - `vasc`
- Interpret gains per class with support counts visible.
- If metadata improves validation macro-F1 but hurts minority-class F1, the conclusion should be
  mixed rather than positive.

## Test Usage Policy

E3c must not load or transform `data/splits/test.csv`. Test metadata must not be used to fit
category vocabularies, imputers, scalers, or model parameters. Test metrics remain reserved for the
final audit after validation-selected candidates are pre-registered.

## Verification Gates

- `PYTHONPATH=src uv run ruff check .` passes.
- Relevant pytest suite passes with `uv run`.
- Metadata preprocessing tests prove train-only fit:
  - train median age is used for validation missing ages,
  - train category vocabulary is fixed,
  - validation-only categories do not expand the feature dimension.
- Cache alignment checks cover `sample_id`, `image_id`, `lesion_id`, label, split, and row order.
- Feature matrix row counts are train `7008`, validation `1504`.
- Metadata matrix row counts are train `7008`, validation `1504`.
- Combined feature dimensions match image dimension plus metadata dimension.
- No NaN/Inf in metadata features, image features, combined features, logits, or probabilities.
- Prediction dumps contain exactly `1504` validation rows.
- Confusion matrix label order is fixed.
- Generated artifacts remain Git-ignored.

## Expected Artifacts

Tracked:

```text
src/dl_final/features/metadata.py
scripts/train_metadata_augmented_mlp.py
scripts/summarize_e3c_metadata.py
tests/test_metadata_features.py
tests/test_e3c_metadata_augmented.py
docs/EXPERIMENT_REGISTRY.md
docs/DECISIONS.md
docs/COMMANDS.md
docs/report_notes/e3c_metadata_augmented_features.md
```

Generated and Git-ignored:

```text
artifacts/runs/*_e3c_metadata_*_seed*/
artifacts/report_assets/tables/e3c_metadata_augmented_results.csv
artifacts/report_assets/tables/e3c_metadata_augmented_summary.csv
artifacts/report_assets/tables/e3c_metadata_augmented_per_class_metrics.csv
artifacts/report_assets/tables/e3c_metadata_per_class_delta_vs_image_only.csv
artifacts/report_assets/tables/e3c_metadata_vs_image_only_validation.csv
artifacts/report_assets/figures/e3c_metadata_augmented_macro_f1.png
```

Each run directory should include:

```text
run_config.json
metrics_summary.csv
metrics_summary.json
per_class_metrics.csv
confusion_matrix.csv
confusion_matrix.png
predictions.csv
training_history.csv
checkpoint_metadata.json
model.pt
scaler_stats.npz
metadata_preprocessing.json
runtime_metadata.json
```

## Risks And Fallbacks

| Risk | Mitigation |
|---|---|
| Metadata improves accuracy but not macro-F1. | Keep macro-F1 primary and discuss class imbalance. |
| Metadata-only performs suspiciously well. | Audit leakage fields and verify only `age`, `sex`, `localization` were used. |
| Plus-metadata overfits validation. | Use multi-seed mean/std and compare per-class behavior; do not add more variants. |
| Age missing values affect minority classes. | Record missing counts and train-median imputation policy. |
| Localization encodes dataset-specific bias. | Frame as benchmark metadata fusion, not clinical generalization. |
| Implementation complicates existing MLP scripts. | Use a separate E3c script while keeping shared logic in `src/dl_final/`. |

## Report Note Template

```md
# Metadata-Augmented Fine-Tuned Feature Diagnostic

## Question

Does adding HAM10000 benchmark metadata (`age`, `sex`, `localization`) to cached fine-tuned
transformer features improve validation macro-F1 over image-only feature transfer?

## Protocol

- Split: canonical lesion-aware train/validation split.
- Test usage: not used.
- Image features: fine-tuned transformer caches.
- Metadata fields: age, sex, localization only.
- Metadata preprocessing: train-only imputation, scaling, and categorical vocabulary fitting.
- Seeds: 7, 13, 42, 101, 202.
- Selection metric: validation macro-F1.

## Results

| Condition | Mean macro-F1 | Std | Min | Max | Interpretation |
|---|---:|---:|---:|---:|---|

## Per-Class Behavior

Discuss which classes gained or lost F1 after adding metadata, with support counts.

## Interpretation

Use cautious benchmark language. If metadata helps, state that structured metadata may provide
complementary benchmark signal to fine-tuned transformer features. If it does not help, state that
the adapted image representations already captured most useful signal under this split, or that
metadata correlations were not reliable enough to improve macro-F1.

## Limitations

- Validation-only result.
- Metadata may encode dataset-specific correlations.
- No clinical diagnosis or deployment claim.
```

## First Implementation Step

Implement `src/dl_final/features/metadata.py` with a train-only `MetadataPreprocessor`, then add
unit tests for imputation, category handling, output dimension stability, no NaN/Inf, and validation
not changing fitted state. After that, wire the preprocessor into a small metadata-only smoke run
before combining metadata with cached transformer features.

## Completion Summary

E3c was implemented as a validation-only metadata-augmented cached-feature diagnostic. It added:

```text
src/dl_final/features/metadata.py
scripts/train_metadata_augmented_mlp.py
scripts/summarize_e3c_metadata.py
tests/test_metadata_features.py
tests/test_e3c_metadata_augmented.py
docs/report_notes/e3c_metadata_augmented_features.md
```

Completed canonical conditions over seeds `7`, `13`, `42`, `101`, and `202`:

| Condition | Mean validation macro-F1 | Std | Min | Max |
|---|---:|---:|---:|---:|
| `ft_vit_swin_beit_concat_plus_metadata` | `0.7278` | `0.0058` | `0.7213` | `0.7363` |
| `ft_vit_swin_concat_plus_metadata` | `0.7230` | `0.0138` | `0.7082` | `0.7376` |
| `metadata_only` | `0.2202` | `0.0077` | `0.2093` | `0.2297` |

Image-only E3b controls:

| Condition | Mean validation macro-F1 | Std |
|---|---:|---:|
| `finetuned_vit_swin_beit_concat` | `0.7246` | `0.0143` |
| `finetuned_vit_swin_concat` | `0.7160` | `0.0085` |

Interpretation: metadata produced a small validation macro-F1 gain over image-only fine-tuned
feature fusion, but per-class behavior was mixed. The result supports cautious benchmark language:
structured metadata may add small complementary signal to fine-tuned transformer features. It does
not support a clinical or deployment claim.

Verification completed:

- E3c artifact integrity check passed for 15 run directories.
- Prediction dumps contain 1,504 validation rows each.
- Metadata preprocessing artifacts record train-only fit and 19 output dimensions.
- `tests/test_metadata_features.py` and `tests/test_e3c_metadata_augmented.py` passed.
- Full ruff and relevant pytest suite are recorded in the final implementation report.
