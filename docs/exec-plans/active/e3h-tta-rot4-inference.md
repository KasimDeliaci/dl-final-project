# E3h Execution Plan: Validation-Only Rot4 Test-Time Augmentation

Status: active.

## Question

E3g sonucunda en guclu dusuk-overfit validation adayi `top3_family_equal` prediction ensemble oldu:

- validation macro-F1: `0.7665`
- accuracy: `0.8564`
- weighted-F1: `0.8576`

E3h su soruyu test eder:

> Yeni egitim yapmadan, ayni validation split uzerinde deterministic right-angle rotation TTA ile
> mevcut metadata-conditioned family tahminleri daha stabil hale gelip validation macro-F1'i
> artiriyor mu?

## Literature and Prior-Project Motivation

Skin lesion classification literature commonly uses deterministic test-time augmentation and
probability averaging as an inference-time robustness method. Cino et al. 2025 explicitly studies
TTA for skin lesion classification and reports small but measurable balanced multi-class accuracy
improvements. Perez et al. 2018 discusses augmentation and TTA-style prediction averaging in skin
lesion analysis. ISIC-style ensemble work also commonly uses rotations and flips during inference.

Relevant sources:

- Cino et al. 2025, *Skin Lesion Classification Through Test Time Augmentation and Explainable
  Artificial Intelligence*: https://www.mdpi.com/2313-433X/11/1/15
- Perez et al. 2018, *Data Augmentation for Skin Lesion Analysis*:
  https://www.ic.unicamp.br/~sandra/pdf/papers/perez_ISIC18.pdf
- Sabanci ISIC ensemble example using rotation/flip TTA:
  https://research.sabanciuniv.edu/40320/1/SIU_ISIC_SKIN_LESION_CLASSIFICATION.pdf

The old `dl-assignment` project also found that deterministic TTA could improve the best
validation-gated model, while broader train-time augmentation and larger TTA view sets were not
automatically better. The key lesson for this repo is to keep E3h pre-registered, small, and
validation-only.

## Implementation Scope

E3h is an inference-only validation diagnostic over already trained E3d/E3f MLP families.

Primary candidate:

```text
top3_family_equal + tta_rot4
```

Member families:

```text
E3d all-fine-tuned FiLM seeds: 7, 13, 42, 101, 202
E3d all-fine-tuned metadata-gated seeds: 7, 13, 42, 101, 202
E3f mixed frozen-ViT + fine-tuned Swin/BEiT metadata-gated seeds: 7, 13, 42, 101, 202
```

TTA policy:

```text
tta_rot4 = identity + rot90 + rot180 + rot270
```

Aggregation policy:

1. For each model seed and each TTA view, run normal inference and compute class probabilities.
2. Average probabilities across the four views for that model seed.
3. Average probabilities across seeds within each family.
4. Average the three family probabilities with equal weights for `top3_family_equal_tta_rot4`.

This follows the literature and the old project: average probabilities, not predicted labels.

## Non-Goals

- No test split loading or test metric computation.
- No new backbone fine-tuning.
- No new downstream MLP training.
- No random augmentation.
- No crop, affine warp, color jitter, blur/sharpen, hair removal, or color normalization search.
- No `tta_flip4`, `tta_d4_8`, or unrestricted TTA policy grid in the primary experiment.
- No validation-tuned ensemble weights in the primary result.
- No class-specific threshold tuning.
- No clinical diagnosis claims.

## Technical Policy

E3h should not overwrite existing feature caches or run artifacts.

TTA inference may use temporary view feature caches or streamed features, but the reportable
prediction rule is probability averaging. If temporary caches are used, place them under a separate
gitignored source:

```text
artifacts/features/ham10000/e3h_tta_rot4/<view>/<source>/<backbone>/
```

Preferred implementation for reproducibility:

1. Extract validation features for each required backbone/source/view once.
2. Verify each view cache has `1504` validation rows and exact split alignment.
3. Reuse saved run artifacts for each E3d/E3f seed:
   - `model.pt`
   - `config_resolved.yaml`
   - `scaler_stats.npz`
   - `preprocessing_metadata.json`
4. Apply the same train-fitted scaler statistics stored with each run.
5. Apply the same metadata preprocessing artifacts stored with each run.
6. Compute per-view probabilities and average them.

Identity sanity check:

- Recomputed identity probabilities should reproduce existing validation metrics closely enough to
  trust the new inference path.
- If identity differs materially from the recorded E3d/E3f prediction dumps, stop and debug before
  running `rot90/rot180/rot270`.

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
- corrected-vs-broken counts against no-TTA `top3_family_equal`,
- runtime multiplier.

Primary comparison:

```text
E3g top3_family_equal: 0.7665 validation macro-F1
E3h top3_family_equal_tta_rot4: measured validation macro-F1
```

Interpretation rule:

| Validation macro-F1 gain over E3g | Interpretation |
|---:|---|
| `< +0.002` | practical tie or noise-level |
| `+0.002` to `< +0.005` | suggestive only |
| `>= +0.005` | meaningful validation improvement if per-class behavior is acceptable |

Do not promote E3h solely if the gain comes from one low-support class with broad degradation
elsewhere.

## Test Usage Policy

Test split must not be loaded, transformed, or evaluated in E3h.

If E3h becomes the final validation-selected candidate, the test split is reserved for the later E4
final audit after the full selection rule is frozen.

## Verification Gates

Required before accepting E3h results:

- TTA policy expands exactly to `identity`, `rot90`, `rot180`, `rot270`.
- All prediction dumps contain exactly `1504` validation rows.
- No prediction dump contains test rows.
- Each row aligns by `sample_id`, `image_id`, `lesion_id`, `split`, and `true_label`.
- Probabilities are finite and row-normalized.
- Confusion matrix label order is fixed:
  `akiec`, `bcc`, `bkl`, `df`, `nv`, `mel`, `vasc`.
- Identity sanity check passes or any mismatch is explained before using TTA results.
- Runtime metadata records device, batch size, number of views, and elapsed seconds.
- Generated feature caches, prediction dumps, checkpoints, and report assets remain gitignored.

## Expected Artifacts

```text
artifacts/runs/e3h_tta_rot4/
artifacts/runs/e3h_tta_rot4/run_config.json
artifacts/runs/e3h_tta_rot4/tta_policy.json
artifacts/runs/e3h_tta_rot4/tta_identity_sanity.csv
artifacts/runs/e3h_tta_rot4/tta_model_results.csv
artifacts/runs/e3h_tta_rot4/tta_family_results.csv
artifacts/runs/e3h_tta_rot4/tta_ensemble_results.csv
artifacts/runs/e3h_tta_rot4/tta_per_class_metrics.csv
artifacts/runs/e3h_tta_rot4/tta_corrected_broken.csv
artifacts/runs/e3h_tta_rot4/predictions_top3_family_equal_tta_rot4.csv
artifacts/report_assets/tables/e3h_tta_rot4_results.csv
artifacts/report_assets/tables/e3h_tta_rot4_per_class_metrics.csv
artifacts/report_assets/tables/e3h_tta_rot4_vs_e3g.csv
artifacts/report_assets/figures/e3h_tta_rot4_macro_f1.png
artifacts/report_assets/figures/e3h_tta_rot4_per_class_f1_delta.png
artifacts/report_assets/figures/e3h_tta_rot4_confusion_matrix.png
docs/report_notes/e3h_tta_rot4.md
```

## Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Identity inference does not reproduce existing E3d/E3f metrics. | Stop and debug scaler/model/metadata reconstruction before any TTA interpretation. |
| TTA improves only one low-support class but hurts broader performance. | Report as per-class tradeoff, not as final improvement. |
| TTA does not improve over E3g. | Report as evidence that current ensemble is already stable to right-angle orientation changes. |
| Runtime is high locally. | Run validation-only extraction/inference on Colab, keeping outputs under a separate Drive artifact folder. |
| Temporary view caches are large. | Keep them gitignored and optionally delete after prediction dumps and report assets are generated. |

## Report Note Template

```text
E3h evaluated deterministic four-view right-angle rotation test-time augmentation as an
inference-only extension of the strongest validation-selected prediction ensemble. No model weights,
feature-cache training protocol, metadata preprocessing, or class thresholds were changed. The test
split was not loaded. Probabilities were averaged across TTA views, then across seeds and model
families using the same equal-family rule as E3g.

The main comparison is between no-TTA E3g `top3_family_equal` and E3h
`top3_family_equal_tta_rot4`. Any improvement should be interpreted as inference-time prediction
stabilization, not as a new representation-learning result.
```

## First Implementation Step

Add a small reusable TTA helper under `src/dl_final/evaluation/tta.py` with:

- `expand_tta_policy("tta_rot4")`,
- probability averaging with shape/normalization checks,
- alignment verification for prediction frames,
- identity sanity-check helpers.

Then implement `scripts/evaluate_tta_rot4.py` to run a smoke validation pass on a small prefix before
full validation inference.

## Implementation Status - 2026-06-15

Implemented:

- `src/dl_final/evaluation/tta.py` with deterministic policy expansion, probability averaging,
  probability validation, probability-column helpers, and prediction alignment checks.
- `scripts/evaluate_tta_rot4.py` for E3h inference-only validation evaluation over E3d/E3f
  metadata-conditioned model families.
- `tests/test_e3h_tta.py` for E3h helper behavior.
- `notebooks/06_e3h_tta_rot4.ipynb` as a thin Colab runner that restores required Drive artifacts,
  accepts an optional `MyDrive/dl-final-artifact/e3h_tta_rot4/e3h_tta_inputs.tar` fallback bundle,
  runs smoke/full E3h commands, verifies outputs, displays report assets, and syncs E3h results
  under `MyDrive/dl-final-artifact/e3h_tta_rot4/`.
- `docs/COMMANDS.md` E3h smoke, full Colab, and artifact-integrity commands.

Local verification:

- `PYTHONPATH=src uv run ruff check .` passed.
- `PYTHONPATH=src uv run pytest tests/test_e3h_tta.py` passed.
- Regression suite passed:
  `tests/test_e3h_tta.py`, `tests/test_sprint2_features.py`, `tests/test_sprint3_fusion.py`,
  `tests/test_representation_complementarity.py`, and `tests/test_sprint4_finetune.py`.
- Local smoke command completed with `--max-samples 8`, producing gitignored smoke artifacts under
  `artifacts/runs/e3h_tta_rot4_smoke_n8/` and `artifacts/report_assets/*e3h_tta_rot4*_smoke.*`.

Full validation status:

- Full local validation was attempted but stopped because local MPS/CPU inference was too slow, and
  `batch-size 256` was killed by memory pressure (`exit 137`).
- Full `1504`-row E3h validation should be run on Colab GPU using the notebook/command above.
  The notebook defaults to `E3H_BATCH_SIZE = 128` to avoid T4 out-of-memory failures; `192` or
  `256` can be tried only when Colab assigns a larger-memory GPU.
- E3h remains active until full validation outputs are produced and the report note is written.
