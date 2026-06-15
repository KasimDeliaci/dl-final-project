# E3f Execution Plan: Frozen ViT + Fine-Tuned Swin/BEiT Mixed Adaptation

Status: completed.

## Question

ViT single-backbone fine-tuning frozen ViT baseline'ini gecemedigi halde Swin ve BEiT single
fine-tuning kazanc gosterdi. E3f su soruyu test eder:

> ViT'i frozen bir strong control olarak tutup, sadece fine-tuning'den fayda goren Swin ve BEiT
> feature'larini adapted kullanmak validation macro-F1'i artirir mi?

## Hypothesis

Frozen ViT representation HAM10000 benchmark splitinde zaten gucludur. Swin ve BEiT ise frozen
halde daha zayif oldugu icin partial fine-tuning'den daha fazla fayda gormustur. Bu nedenle
`frozen ViT + fine-tuned Swin + fine-tuned BEiT` mixed feature source'u, all-fine-tuned ViT/Swin/BEiT
source'undan daha dengeli olabilir.

## Implementation Scope

No new backbone training is required. E3f only recombines existing validated feature caches:

| Backbone | Cache source |
|---|---|
| `vit_b16` | `artifacts/features/ham10000/frozen/vit_b16/` |
| `swin_tiny` | `artifacts/features/ham10000/finetuned/swin_tiny/` |
| `beit_base` | `artifacts/features/ham10000/finetuned/beit_base/` |

Mixed source namespace:

```text
artifacts/features/ham10000/frozen_vit_finetuned_swin_beit/
```

## Non-Goals

- No test split usage.
- No new transformer fine-tuning.
- No new backbone.
- No TTA.
- No final model selection from test metrics.
- No overwrite of canonical `frozen/` or `finetuned/` caches.

## Downstream Matrix Policy

Run validation-only downstream checks over seeds `7`, `13`, `42`, `101`, and `202`:

1. Image-only concat:
   - `frozen_vit_finetuned_swin_beit`
   - backbones: `vit_b16`, `swin_tiny`, `beit_base`
   - fusion method: `concat`
2. Metadata-conditioned operators over the same mixed source:
   - `triple_metadata_film`
   - `triple_metadata_gated_backbone`

FiLM is included because E3d's best mean validation macro-F1 came from bounded FiLM-style metadata
conditioning. The gated operator is included as a close comparator because it was nearly tied with
FiLM in E3d and provides gate diagnostics.

## Evaluation Discipline

Primary metric:

- validation macro-F1 mean over seeds.

Secondary metrics:

- validation macro-F1 std/min/max,
- accuracy mean,
- weighted-F1 mean,
- per-class F1 mean,
- prediction row counts,
- runtime.

Comparison anchors:

- all-fine-tuned E3b ViT+Swin+BEiT concat: `0.7246 ± 0.0143`,
- E3c all-fine-tuned concat + metadata: `0.7278 ± 0.0058`,
- E3d all-fine-tuned FiLM: `0.7358 ± 0.0152`,
- E3d all-fine-tuned gated: `0.7347 ± 0.0112`,
- frozen E2b ViT+Swin concat deep_reg: `0.7077 ± 0.0124` multi-seed CPU diagnostic,
- frozen ViT single: `0.6924`,
- canonical fine-tuned ViT single: `0.6876`.

## Validation-Only Selection

All decisions use validation macro-F1 only. The test split is not loaded or transformed.

## Verification Gates

- Mixed source contains exactly train `7008` and validation `1504` rows per backbone.
- Cache alignment is verified by `sample_id`, `image_id`, `lesion_id`, label, and split row order.
- Feature dim is `768` for all three backbones.
- No NaN/Inf in source caches.
- Downstream prediction dumps have `1504` rows.
- StandardScaler fit only on train cache blocks.
- Metadata preprocessing fit only on train split.
- Confusion matrix label order is fixed.
- Generated artifacts remain Git-ignored.
- `PYTHONPATH=src uv run ruff check .`
- Relevant pytest suite passes.

## Expected Artifacts

```text
artifacts/features/ham10000/frozen_vit_finetuned_swin_beit/{vit_b16,swin_tiny,beit_base}/
artifacts/runs/*e3f_mixed_adaptation*
artifacts/report_assets/tables/e3f_mixed_adaptation_results.csv
artifacts/report_assets/tables/e3f_mixed_adaptation_summary.csv
artifacts/report_assets/tables/e3f_mixed_adaptation_per_class_metrics.csv
artifacts/report_assets/tables/e3f_mixed_adaptation_vs_controls.csv
artifacts/report_assets/figures/e3f_mixed_adaptation_macro_f1.png
docs/report_notes/e3f_mixed_frozen_vit_finetuned_swin_beit.md
```

## Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Mixed source underperforms all-fine-tuned triple. | Report as evidence that fine-tuned ViT still contributes in fusion despite weaker single-backbone result. |
| Mixed source improves image-only concat but not FiLM/gated. | Report metadata conditioning as source-sensitive rather than universally helpful. |
| FiLM/gated overfit validation. | Keep multi-seed std/min/max visible and compare to image-only mixed concat. |
| Cache mismatch. | Stop; do not run downstream experiments until row order alignment is fixed. |

## First Implementation Step

Add a small reproducible script to build mixed feature sources from existing cache directories, then
create `frozen_vit_finetuned_swin_beit` and run the validation-only downstream checks.
