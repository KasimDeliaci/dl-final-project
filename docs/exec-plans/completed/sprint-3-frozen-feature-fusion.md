# Sprint 3 Frozen Feature Fusion Execution Plan

## Objective

Implement and run validation-only frozen feature fusion experiments over the canonical Sprint 2 transformer feature caches.

Sprint 3 answers one question:

> Do fused frozen transformer features improve validation macro-F1 beyond the strongest single-backbone frozen baseline, Vanilla ViT at `0.6924`?

If fusion does not improve over ViT, the result remains reportable as evidence about limited feature complementarity, redundant representations, high-dimensional overfitting, projection bottlenecks, PCA compression loss, or limited classifier capacity under this protocol.

## Implementation Scope

- Use existing Sprint 2 train/validation feature caches for:
  - `vit_b16`, feature dim `768`
  - `swin_tiny`, feature dim `768`
  - `deit3_small`, feature dim `384`
- Add fusion modules under `src/dl_final/models/fusion.py`.
- Add a reproducible matrix runner at `scripts/run_fusion_matrix.py`.
- Run the full 12-run validation matrix:
  - 4 concat runs
  - 4 weighted learned 512 runs
  - 4 weighted PCA 384 runs
- Reuse the existing MLP training discipline:
  - train-only scaler,
  - train-only class weights,
  - AdamW,
  - dropout/weight decay,
  - early stopping,
  - checkpoint selection by validation macro-F1.
- Export standard artifacts for each run: config, metrics, per-class metrics, confusion matrix, prediction dump, training history, runtime metadata, and method-specific fusion metadata.
- Export aggregate report assets comparing Sprint 3 fusion runs against Sprint 2 single-backbone validation baselines.
- Update experiment registry, decisions, commands, and Sprint 3 report note.

## Non-Goals

- No new raw-image feature extraction.
- No transformer fine-tuning.
- No LDA.
- No PCA for concat.
- No test metrics in Sprint 3, even if test caches exist.
- No test-set model, hyperparameter, checkpoint, or fusion-weight selection.
- No clinical diagnosis framing.
- No notebook implementation unless a future thin launcher is needed.

## Exact Fusion Methods

### `concat`

Frozen feature blocks are aligned row-by-row and concatenated without projection or PCA.

Expected dimensions:

- `vit_b16 + swin_tiny`: `1536`
- `vit_b16 + deit3_small`: `1152`
- `swin_tiny + deit3_small`: `1152`
- `vit_b16 + swin_tiny + deit3_small`: `1920`

### `weighted_learned_512`

Each backbone feature block is first standardized with a train-only scaler, then passed through a trainable linear projection into a shared 512-dimensional space. Global trainable backbone logits are softmax-normalized, and the projected blocks are combined by weighted sum. The fused 512-dimensional vector is classified by the MLP.

The learned softmax weights are exported as run artifacts. These weights are not interpreted as a direct backbone-quality ranking because they are optimized jointly with projection layers and the classifier.

### `weighted_pca_384`

Each backbone feature block is first standardized with a train-only scaler. Then PCA is fit separately for each backbone block using train split features only and transformed into a shared 384-dimensional latent space. Global trainable softmax weights combine the PCA blocks by weighted sum, and the fused 384-dimensional vector is classified by the MLP.

PCA metadata must record train-only fit policy, original dimensions, output dimensions, explained variance ratios, and split transform counts.

## PCA Policy

- PCA is used only for `weighted_pca_384`.
- PCA is never applied to concat.
- PCA is unsupervised; labels are not used for fitting.
- PCA is fit on train split only.
- Validation features are transformed with train-fitted PCA objects.
- PCA output dimension is `384`.
- For DeiT III-Small, `384 -> 384` is still represented as PCA train-only projection so the policy is uniform and auditable.

## Weighted Learned Projection Policy

- Projection dimension is fixed at `512`.
- Projection layers are trainable.
- Backbone weights are global parameters, softmax-normalized during forward pass.
- Fusion output feeds the same MLP classifier pattern as Sprint 2.
- Weight artifacts include raw logits, normalized weights, projection dimension, and backbone order.

## Evaluation Discipline

- Primary metric: validation macro-F1.
- Required metrics: accuracy, macro precision, macro recall, macro-F1, weighted-F1, per-class precision/recall/F1, confusion matrix.
- Selection rule: checkpoint and run ranking use validation macro-F1 only.
- Test split remains untouched for metrics in Sprint 3; final test audit is reserved for Sprint 5.
- Confusion matrix label order remains `akiec`, `bcc`, `bkl`, `df`, `nv`, `mel`, `vasc`.

## Verification Gates

- Unit/smoke tests pass with `uv run`.
- Concat output shapes match expected dimensions.
- Weighted learned output shape is `(batch, 512)`.
- Weighted PCA output shape is `(batch, 384)`.
- Softmax-normalized fusion weights sum to `1`.
- PCA train-only fit metadata is written.
- Feature row alignment checks cover `sample_id`, `image_id`, `lesion_id`, label, and split row order.
- Feature tensors and transformed fusion tensors contain no NaN or Inf.
- Train row count is `7008`; validation row count is `1504`.
- Prediction dumps contain `1504` validation rows.
- Generated caches, checkpoints, predictions, and run artifacts are ignored by Git.
- `PYTHONPATH=src uv run ruff check .` passes.
- Relevant pytest suite passes.

## Expected Artifacts

Tracked:

```text
src/dl_final/models/fusion.py
scripts/run_fusion_matrix.py
tests/test_sprint3_fusion.py
docs/exec-plans/active/sprint-3-frozen-feature-fusion.md
docs/report_notes/sprint3_frozen_fusion.md
docs/EXPERIMENT_REGISTRY.md
docs/DECISIONS.md
docs/COMMANDS.md
```

Generated and Git-ignored:

```text
artifacts/runs/*_s3_frozen_*_concat_mlp_seed42/
artifacts/runs/*_s3_frozen_*_weightedlearned512_mlp_seed42/
artifacts/runs/*_s3_frozen_*_weightedpca384_mlp_seed42/
artifacts/report_assets/tables/frozen_fusion_results.csv
artifacts/report_assets/tables/frozen_fusion_per_class_metrics.csv
artifacts/report_assets/tables/frozen_fusion_weight_summary.csv
artifacts/report_assets/tables/frozen_fusion_vs_single_validation.csv
artifacts/report_assets/figures/frozen_fusion_macro_f1.png
```

## Risks And Fallbacks

| Risk | Fallback |
|---|---|
| Fusion does not improve over ViT. | Report as negative/limited-complementarity evidence; discuss redundancy, overfitting, projection bottleneck, PCA information loss, and classifier capacity. |
| Concat overfits due to high dimensionality. | Keep the result; do not add PCA to concat. Discuss high-dimensional overfitting risk. |
| Weighted learned underperforms. | Keep learned weights as diagnostic artifacts and discuss projection bottleneck. |
| Weighted PCA underperforms. | Report as PCA compression diagnostic, not as canonical weighted fusion replacement. |
| Full 12-run matrix is slow locally. | Keep runner resumable by method/backbone filters; run smoke subset first, then full matrix. |

## Report Note Template

```text
Question:
Recipe:
Fixed controls:
Validation results:
Best fusion vs ViT single-backbone baseline:
Fusion method interpretation:
Per-class observations:
Evidence strength:
Report decision:
Limitations:
```

## First Implementation Step

Add `src/dl_final/models/fusion.py` and `tests/test_sprint3_fusion.py` for concat, weighted learned 512, weighted PCA 384 shape checks, softmax normalization, expected dimensions, PCA metadata, and cache-alignment failure behavior. Then add the matrix runner and verify it first on one limited run before executing the full 12-run matrix.

## Final Outcome

Sprint 3 completed locally on 2026-06-10.

Implemented:

- `src/dl_final/models/fusion.py`
- `scripts/run_fusion_matrix.py`
- `tests/test_sprint3_fusion.py`
- `docs/report_notes/sprint3_frozen_fusion.md`
- Sprint 3 updates in `docs/EXPERIMENT_REGISTRY.md`, `docs/DECISIONS.md`, and `docs/COMMANDS.md`

Generated 12 validation-only fusion runs:

- 4 `concat`
- 4 `weighted_learned_512`
- 4 `weighted_pca_384`

Best validation result:

- `vit_b16 + swin_tiny`, `concat`
- validation macro-F1 `0.6947`
- ViT single-backbone baseline validation macro-F1 `0.6924`
- gain `+0.0023`

Interpretation:

The gain over the strongest single-backbone baseline is positive but small. This should be reported as limited evidence that ViT and Swin may contain complementary frozen feature information under concat fusion, not as a broad claim that fusion is reliably superior.

Verification completed:

- Unit/smoke tests passed.
- Full 12-run matrix completed.
- Prediction dump row count was `1504` for every run.
- Train/validation row count checks passed: `7008` / `1504`.
- Weighted fusion softmax weights summed to 1.
- PCA metadata records train-only fit and no label usage.
- Generated artifacts are ignored by Git.

BEiT-expanded E2 matrix:

- `vit_b16 + beit_base`, `swin_tiny + beit_base`, and `vit_b16 + swin_tiny + beit_base` were run as E2 validation-only expanded fusion conditions after the planned ViT/Swin/DeiT matrix.
- Total E2 fusion runs after expansion: `21`.
- Best BEiT-expanded result: `vit_b16 + swin_tiny + beit_base` concat validation macro-F1 `0.6988`.
- BEiT pairwise concat results did not exceed ViT single: `vit_b16 + beit_base` `0.6556`, `swin_tiny + beit_base` `0.6381`.
- The BEiT triple result slightly exceeds both ViT single `0.6924` and planned ViT + Swin concat `0.6947`.
- Representation similarity diagnostic was computed on validation features with train-only scaling. BEiT had lower similarity to ViT (`0.4393`) and Swin (`0.2874`) than the canonical ViT + Swin pair (`0.5942`), supporting the interpretation that BEiT may add complementary representation structure despite weak standalone performance.
- Forward decision after E2: Sprint 4 should use `vit_b16`, `swin_tiny`, and `beit_base`; `deit3_small` remains a planned/screened baseline and is not carried into the fine-tuning compute budget.

E2b MLP capacity diagnostic:

- A focused validation-only MLP capacity probe was run after E2.
- Best E2b result: `vit_b16 + swin_tiny` concat with deep-regularized MLP, validation macro-F1 `0.7262`.
- Best BEiT triple E2b result: `vit_b16 + swin_tiny + beit_base` concat with wide-regularized MLP, validation macro-F1 `0.7159`.
- Best DeiT triple E2b result did not exceed its baseline materially (`0.6863`).
- Interpretation: BEiT remains the stronger third-backbone candidate than DeiT, but the strongest frozen concat validation result under a larger MLP is the ViT+Swin pair.
