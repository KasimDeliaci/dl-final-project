# E3i Execution Plan: Simple Fusion Rot4 TTA Diagnostic

## Purpose

E3h applied `tta_rot4` directly to the strongest E3g metadata-conditioned prediction ensemble and
reduced validation macro-F1. E3i tests the narrower follow-up question:

> Does deterministic rot4 TTA help simpler image-only cached-feature MLP/fusion models before
> metadata-conditioned seed/family ensembling is applied?

This mirrors the older `dl-assignment` lesson more closely: TTA was evaluated on a selected simple
fine-tuned/weighted model and only then considered for final audit. E3i remains validation-only.

## Implementation Scope

- New inference-only runner: `scripts/evaluate_simple_tta_rot4.py`.
- Colab launcher: `notebooks/07_e3i_simple_tta_rot4.ipynb`.
- Candidate simple models:
  - fine-tuned `vit_b16+swin_tiny+beit_base` concat seed 42,
  - fine-tuned `vit_b16+swin_tiny+beit_base` weighted learned 512 seed 42,
  - fine-tuned `vit_b16+swin_tiny` concat seed 42.
- TTA policy:
  - `tta_rot4 = identity + rot90 + rot180 + rot270`.
- Outputs are isolated under:
  - `artifacts/runs/e3i_simple_tta_rot4/`,
  - `artifacts/report_assets/tables/e3i_simple_tta_rot4*.csv`,
  - `artifacts/report_assets/figures/e3i_simple_tta_rot4*.png`,
  - Drive mirror: `MyDrive/dl-final-artifact/e3i_simple_tta_rot4/`.

## Non-Goals

- No new backbone fine-tuning.
- No new downstream MLP training.
- No test split loading, transformation, evaluation, or model selection.
- No flip/D4/color/crop TTA policy search.
- No `weighted_pca_384` TTA unless future runs persist fitted PCA components. Re-fitting PCA for
  rotated validation features would change preprocessing and is not a valid identity-preserving
  inference-only comparison.
- No validation-tuned TTA weights.

## Evaluation Discipline

- Identity view must reproduce the stored no-TTA prediction dump within tolerance. Smoke runs may
  bypass strict tolerance because they evaluate only a prefix.
- Feature cache alignment is checked by `sample_id`, `image_id`, `lesion_id`, split, label, and row
  order.
- Saved train-fitted scaler statistics from each run are reused for every TTA view.
- Probability rows must be finite and normalized.
- The main reported unit is each simple model's no-TTA vs rot4 validation macro-F1 delta.
- A simple equal average across the selected simple models is written only as a diagnostic ensemble,
  not as a new primary model-selection rule.

## Verification Gates

- `PYTHONPATH=src uv run ruff check .`
- `PYTHONPATH=src uv run pytest tests/test_e3i_simple_tta.py tests/test_e3h_tta.py`
- Colab smoke run with `--max-samples 8`.
- Full Colab run produces 1504 validation rows.
- `run_config.json` records `test_policy = not_loaded_or_used_in_e3i`.
- `tta_identity_sanity.csv` is inspected before interpreting TTA results.
- Report assets and figures are generated and synced to Drive.

## Expected Artifacts

```text
artifacts/runs/e3i_simple_tta_rot4/
artifacts/runs/e3i_simple_tta_rot4/run_config.json
artifacts/runs/e3i_simple_tta_rot4/tta_policy.json
artifacts/runs/e3i_simple_tta_rot4/tta_view_results.csv
artifacts/runs/e3i_simple_tta_rot4/tta_run_results.csv
artifacts/runs/e3i_simple_tta_rot4/tta_ensemble_results.csv
artifacts/runs/e3i_simple_tta_rot4/tta_identity_sanity.csv
artifacts/runs/e3i_simple_tta_rot4/tta_per_class_delta.csv
artifacts/runs/e3i_simple_tta_rot4/tta_corrected_broken.csv
artifacts/runs/e3i_simple_tta_rot4/predictions_*_tta_rot4.csv
artifacts/report_assets/tables/e3i_simple_tta_rot4_*.csv
artifacts/report_assets/figures/e3i_simple_tta_rot4_*.png
```

## Risks And Fallbacks

| Risk | Fallback |
|---|---|
| Rotated views reduce macro-F1 again. | Report as evidence that current transformer feature pipeline is not rotation-invariant under this policy; do not promote TTA. |
| Only weighted learned benefits. | Treat as model-specific evidence, not broad TTA evidence; consider limited multi-seed follow-up. |
| Identity sanity fails. | Stop interpretation and debug model/scaler/feature source restoration before using TTA rows. |
| Colab Drive tree is incomplete. | Restore optional `e3i_simple_tta_inputs.tar` or sync the listed run/checkpoint/cache inputs to Drive. |

## Report Note Template

E3i evaluated deterministic four-view right-angle rotation test-time augmentation on simpler
image-only cached-feature fusion models. Unlike the previous metadata-conditioned ensemble TTA
diagnostic, this analysis applies TTA before model-family ensembling and keeps the comparison at the
individual fusion-run level. The test split was not loaded. The main comparison is no-TTA identity
validation macro-F1 versus rot4-averaged validation macro-F1 for concat and learned weighted fusion.
If a gain is observed, it should be described as model-specific inference-time stabilization. If no
gain is observed, it supports the conclusion that right-angle rotation averaging is not beneficial
for the current transformer feature transfer setup, despite being useful in the earlier CNN project.

## First Implementation Step

Implement `scripts/evaluate_simple_tta_rot4.py` by reusing the existing TTA feature extraction and
probability validation conventions, but reconstructing only the simple `FeatureMLP` and
`WeightedLearnedFusionMLP` models from their saved run configs.
