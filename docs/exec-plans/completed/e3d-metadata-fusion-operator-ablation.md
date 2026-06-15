# E3d Metadata Fusion Operator Ablation Execution Plan

## Objective

Evaluate whether lightweight metadata-aware fusion operators improve over simple metadata
concatenation when applied to fixed fine-tuned transformer feature caches.

E3d answers:

> Given that E3c showed a small validation macro-F1 gain from appending HAM10000 metadata to
> fine-tuned transformer features, does a controlled metadata-conditioned fusion operator provide a
> stronger or more stable validation signal than raw concatenation?

This is a validation-only ablation. It does not repeat transformer fine-tuning, does not extract new
image features, and does not use the test split.

## Context

E3c used only simple concatenation:

```text
[scaled ViT features, scaled Swin features, scaled BEiT features, metadata vector]
```

E3c result over seeds `7`, `13`, `42`, `101`, `202`:

| Condition | Mean validation macro-F1 | Std | Min | Max |
|---|---:|---:|---:|---:|
| Fine-tuned ViT+Swin+BEiT concat + metadata | `0.7278` | `0.0058` | `0.7213` | `0.7363` |
| Fine-tuned ViT+Swin concat + metadata | `0.7230` | `0.0138` | `0.7082` | `0.7376` |
| Metadata-only MLP | `0.2202` | `0.0077` | `0.2093` | `0.2297` |

Image-only E3b controls:

| Condition | Mean validation macro-F1 | Std |
|---|---:|---:|
| Fine-tuned ViT+Swin+BEiT concat | `0.7246` | `0.0143` |
| Fine-tuned ViT+Swin concat | `0.7160` | `0.0085` |

E3c therefore suggests a small metadata benefit, but it does not answer whether metadata is best
used as a raw appended feature vector or as a conditioning signal over image representations.

## Literature Motivation

Multimodal skin-lesion literature supports testing fusion operators beyond raw concatenation, but
the design should stay proportional to this repo's benchmark scope.

- Frontiers 2025 compares multimodal fusion strategies on HAM10000, including simple
  concatenation, weighted concatenation, self-attention, and cross-attention fusion. This motivates
  operator-level ablation, but the repo should not add a large cross-attention architecture just to
  chase a validation bump.
  <https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608837/full>
- CVPRW 2025 reports that clinical metadata such as age, sex, and anatomical site can improve
  feature clustering and class separability relative to image-only models. This supports the E3c
  metadata signal and motivates asking how the metadata should enter the classifier.
  <https://openaccess.thecvf.com/content/CVPR2025W/MULA2025/html/Ahammed_Skin_Lesion_Classification_Using_Dermoscopic_Images_and_Clinical_Metadata_Insights_CVPRW_2025_paper.html>
- MetaNet/ISBI-style metadata fusion uses metadata to control the importance of visual features,
  rather than only appending metadata to a classifier input. This directly motivates a small
  metadata-gated feature operator.
  <https://jiaxinzhuang.github.io/pdf/metanet.pdf>
- Cross-attention multimodal skin-lesion studies show that metadata-image interaction can improve
  performance, but they introduce heavier architectures and different deployment assumptions. E3d
  borrows the interaction idea while keeping the implementation at cached-feature MLP scale.
  <https://pubmed.ncbi.nlm.nih.gov/40561935/>

## Implementation Scope

- Add lightweight metadata-aware fusion models under `src/dl_final/models/metadata_fusion.py`.
- Add a reproducible E3d runner, likely `scripts/train_metadata_fusion_operator.py`.
- Reuse `src/dl_final/features/metadata.py` from E3c for train-only metadata preprocessing.
- Reuse fixed fine-tuned feature caches:
  - `vit_b16`
  - `swin_tiny`
  - `beit_base`
- Run only the triple-backbone condition for the canonical E3d ablation.
- Compare new operators against existing E3c `concat + metadata` and E3b image-only controls.
- Update registry, decisions, commands, and report notes after implementation.

Expected implementation files:

```text
src/dl_final/models/metadata_fusion.py
scripts/train_metadata_fusion_operator.py
scripts/summarize_e3d_metadata_fusion.py
tests/test_metadata_fusion_models.py
tests/test_e3d_metadata_fusion.py
docs/report_notes/e3d_metadata_fusion_operator_ablation.md
```

## Non-Goals

- Do not compute test metrics.
- Do not load or transform `data/splits/test.csv`.
- Do not repeat transformer fine-tuning.
- Do not extract feature-level TTA caches in E3d.
- Do not add self-attention or cross-attention transformer blocks.
- Do not add class-aware loss, balanced sampler, train-time augmentation, or deeper unfreezing.
- Do not use `dx`, `dx_type`, `dataset`, `image_id`, `sample_id`, `lesion_id`, or `image_path` as
  model input.
- Do not treat learned gates as a direct backbone quality ranking.

## Conditions

Run all new E3d conditions with seeds `7`, `13`, `42`, `101`, and `202`.

Canonical E3d operator ablation:

| Condition ID | Image features | Metadata | Operator | Role |
|---|---|---|---|---|
| `triple_concat_metadata` | ViT+Swin+BEiT | yes | raw concat | Existing E3c control; reuse summary, rerun only if needed. |
| `triple_metadata_gated_backbone` | ViT+Swin+BEiT | yes | metadata predicts per-backbone gates | Main operator ablation. |
| `triple_metadata_film` | ViT+Swin+BEiT | yes | metadata predicts bounded scale/shift over image features | Interaction ablation. |
| `triple_metadata_two_branch` | ViT+Swin+BEiT | yes | image branch + metadata branch fused at hidden layer | Late hidden fusion ablation. |

Reference controls:

- E3b `finetuned_vit_swin_beit_concat` image-only mean/std.
- E3c `ft_vit_swin_beit_concat_plus_metadata` raw concat mean/std.
- E3c `metadata_only` mean/std.

Optional only if canonical run finishes cleanly:

- Repeat the best E3d operator on `vit_b16+swin_tiny` pair to check whether BEiT remains useful
  under metadata-conditioned fusion. This optional run should be clearly marked diagnostic.

## Operator Definitions

Input preprocessing shared by all E3d operators:

- Scale each image feature block with train-only `StandardScaler`, as in E3b/E3c.
- Fit metadata preprocessing on train only, as in E3c.
- Class weights are computed from train labels only.
- Label order remains `akiec`, `bcc`, `bkl`, `df`, `nv`, `mel`, `vasc`.

### `triple_metadata_gated_backbone`

Purpose: test whether metadata can modulate the relative contribution of ViT, Swin, and BEiT per
sample without discarding full feature vectors.

Proposed architecture:

```text
metadata vector -> small MLP -> 3 sigmoid gates
scaled block_i = image_block_i * gate_i
classifier input = concat(scaled ViT, scaled Swin, scaled BEiT, metadata vector)
classifier = FeatureMLP-compatible MLP
```

Artifact requirement:

- Save mean gate by backbone.
- Save mean gate by true class.
- Save gate standard deviation by backbone.

Interpretation rule:

- Gate values are model internals, not direct backbone quality scores.
- Class-level gate differences can be discussed only as diagnostic behavior.

### `triple_metadata_film`

Purpose: test whether metadata should condition the image feature vector through bounded affine
modulation rather than simple append.

Conservative architecture:

```text
metadata vector -> small MLP -> gamma, beta for image feature vector
gamma, beta bounded with tanh and small scale
conditioned image = image * (1 + 0.1 * gamma) + 0.1 * beta
classifier input = concat(conditioned image, metadata vector)
classifier = FeatureMLP-compatible MLP
```

The bounded modulation prevents the metadata branch from completely rewriting image features in a
small validation-only experiment.

### `triple_metadata_two_branch`

Purpose: test whether metadata works better as a separate branch fused with an image hidden
representation rather than appended at raw input.

Proposed architecture:

```text
image concat -> image MLP branch
metadata vector -> metadata MLP branch
concat(image hidden, metadata hidden) -> classifier head
```

This is the simplest hidden-level fusion control and should stay small to avoid overfitting.

## Training Policy

- Device: CPU by default for comparability with E3b/E3c downstream diagnostics.
- Seeds: `7`, `13`, `42`, `101`, `202`.
- Batch size: `128`.
- Epoch cap: `30`.
- Early stopping patience: `6`.
- Optimizer: AdamW.
- Learning rate: start with `1e-3`.
- Weight decay: start with `1e-4`.
- Dropout: start with `0.3`.
- Use train-only class-weighted cross entropy.
- Selection metric: validation macro-F1.

If one operator is clearly unstable during smoke testing, reduce its learning rate or increase
dropout before the canonical run and record the change in the plan/report note.

## Evaluation Discipline

- Primary metric: validation macro-F1 mean over seeds.
- Secondary metrics: accuracy, macro precision, macro recall, weighted-F1, per-class precision,
  recall, and F1.
- Report mean, std, min, max across seeds.
- Compare against E3c raw concat + metadata, not only against image-only E3b.
- Per-class results must include support counts.
- If an operator improves mean macro-F1 but worsens `mel` or multiple minority classes, report the
  result as mixed.

## Test Usage Policy

E3d must not use the test split. Test metrics remain reserved for final audit after validation
candidate selection. E3d may inform final candidate choice only through validation macro-F1,
multi-seed stability, and per-class behavior.

## Expected Artifacts

Tracked:

```text
src/dl_final/models/metadata_fusion.py
scripts/train_metadata_fusion_operator.py
scripts/summarize_e3d_metadata_fusion.py
tests/test_metadata_fusion_models.py
tests/test_e3d_metadata_fusion.py
docs/EXPERIMENT_REGISTRY.md
docs/DECISIONS.md
docs/COMMANDS.md
docs/report_notes/e3d_metadata_fusion_operator_ablation.md
```

Generated and Git-ignored:

```text
artifacts/runs/*_e3d_metadata_fusion_*_seed*/
artifacts/report_assets/tables/e3d_metadata_fusion_operator_results.csv
artifacts/report_assets/tables/e3d_metadata_fusion_operator_summary.csv
artifacts/report_assets/tables/e3d_metadata_fusion_operator_per_class_metrics.csv
artifacts/report_assets/tables/e3d_metadata_fusion_per_class_delta_vs_e3c.csv
artifacts/report_assets/tables/e3d_metadata_fusion_vs_e3c_validation.csv
artifacts/report_assets/tables/e3d_metadata_gate_summary.csv
artifacts/report_assets/figures/e3d_metadata_fusion_operator_macro_f1.png
```

Each run should include:

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
metadata_fusion_metadata.json
runtime_metadata.json
```

## Verification Gates

- `PYTHONPATH=src uv run ruff check .` passes.
- Relevant pytest suite passes with `uv run`.
- Model shape tests cover all E3d operators.
- Metadata preprocessing remains train-only.
- Image scaler remains train-only per backbone block.
- Feature cache alignment checks cover `sample_id`, `image_id`, `lesion_id`, label, split, and row
  order.
- Train row count is `7008`; validation row count is `1504`.
- Prediction dumps contain exactly `1504` validation rows.
- No NaN/Inf in image features, metadata features, logits, probabilities, or gate values.
- Gate summaries, if produced, have one row per backbone and optional per-class rows.
- Generated artifacts remain Git-ignored.

## Risks And Fallbacks

| Risk | Mitigation |
|---|---|
| Operators overfit validation more than raw concat. | Keep models small, use five seeds, and report negative/mixed results. |
| Gates appear interpretable but are unstable. | Treat gate summaries as diagnostic only, not causal backbone rankings. |
| FiLM modulation destabilizes training. | Bound modulation and fall back to gated/two-branch results if unstable. |
| Mean macro-F1 improves but `mel` worsens. | Report class-dependent tradeoff; do not call it a uniform improvement. |
| Implementation starts duplicating E3c logic. | Share preprocessing helpers where practical, keep E3d script scoped. |

## Report Note Template

```md
# Metadata Fusion Operator Ablation

## Question

Does metadata-conditioned fusion improve over raw metadata concatenation for fixed fine-tuned
transformer features?

## Protocol

- Split: canonical lesion-aware train/validation split.
- Test usage: not used.
- Image features: fixed fine-tuned ViT, Swin, and BEiT caches.
- Metadata: age, sex, localization only.
- Preprocessing: train-only image scaling and train-only metadata preprocessing.
- Seeds: 7, 13, 42, 101, 202.
- Selection metric: validation macro-F1.

## Results

| Operator | Mean macro-F1 | Std | Min | Max | Interpretation |
|---|---:|---:|---:|---:|---|

## Per-Class Behavior

Discuss deltas against E3c raw concat + metadata, with support counts.

## Interpretation

If a metadata-conditioned operator improves over raw concat, describe it as limited validation
evidence that metadata is more useful when it modulates image representations rather than only being
appended. If it does not improve, state that simple concat was sufficient under this fixed
cached-feature protocol.

## Limitations

- Validation-only.
- Fixed cached features; no end-to-end multimodal fine-tuning.
- Metadata may encode benchmark-specific correlations.
- No clinical diagnosis or deployment claim.
```

## First Implementation Step

Implement `src/dl_final/models/metadata_fusion.py` with small PyTorch modules for
`MetadataGatedBackboneMLP`, `MetadataFiLMMLP`, and `MetadataTwoBranchMLP`. Add shape, finite-output,
gate-range, and gradient smoke tests before wiring the training script.

## Completion Summary

E3d was implemented as a validation-only metadata fusion operator ablation over fixed fine-tuned
`vit_b16+swin_tiny+beit_base` feature caches. It added:

```text
src/dl_final/models/metadata_fusion.py
scripts/train_metadata_fusion_operator.py
scripts/summarize_e3d_metadata_fusion.py
tests/test_metadata_fusion_models.py
tests/test_e3d_metadata_fusion.py
docs/report_notes/e3d_metadata_fusion_operator_ablation.md
```

Completed canonical operators over seeds `7`, `13`, `42`, `101`, and `202`:

| Condition | Mean validation macro-F1 | Std | Min | Max |
|---|---:|---:|---:|---:|
| `triple_metadata_film` | `0.7358` | `0.0152` | `0.7158` | `0.7529` |
| `triple_metadata_gated_backbone` | `0.7347` | `0.0112` | `0.7189` | `0.7453` |
| `triple_metadata_two_branch` | `0.7328` | `0.0103` | `0.7207` | `0.7450` |

Controls:

| Condition | Mean validation macro-F1 | Std |
|---|---:|---:|
| E3c raw concat + metadata | `0.7278` | `0.0058` |
| E3b image-only fine-tuned triple concat | `0.7246` | `0.0143` |

Interpretation: all three metadata-conditioned operators improved mean validation macro-F1 over raw
metadata concatenation. The result supports limited validation evidence that structured benchmark
metadata is more useful when it conditions fine-tuned transformer representations than when only
appended to the MLP input. Per-class behavior is still mixed, so the result remains diagnostic.

Verification completed:

- E3d artifact integrity check passed for 15 run directories.
- Prediction dumps contain 1,504 validation rows each.
- Metadata preprocessing artifacts record train-only fit and 19 output dimensions.
- Metadata fusion metadata records validation macro-F1 selection.
- Gate summaries were produced for metadata-gated backbone runs.
- E3d model and runner tests passed.
- Full ruff and relevant pytest suite are recorded in the final implementation report.
