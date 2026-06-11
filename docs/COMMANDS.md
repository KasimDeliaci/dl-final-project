# Commands

Bu dosya reproducible komutları kaydeder. Sprint 1 için dependency yüzeyi bilinçli olarak küçüktür: `pandas` ve `pillow`.

Preferred local runner:

```bash
uv run python
```

Codex bundled runtime fallback:

```bash
export CODEX_PYTHON=/Users/arcustin2/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3
```

## Local Commands

Sprint 1 unit tests:

```bash
PYTHONPATH=src uv run python -m unittest tests/test_dataset_sprint1.py

# fallback
PYTHONPATH=src "$CODEX_PYTHON" -m unittest tests/test_dataset_sprint1.py
```

HAM10000 metadata/image audit:

```bash
PYTHONPATH=src uv run python scripts/audit_ham10000.py --config configs/dataset/selected_dataset.yaml

# fallback
PYTHONPATH=src "$CODEX_PYTHON" scripts/audit_ham10000.py --config configs/dataset/selected_dataset.yaml
```

Creates:

```text
data/processed/ham10000_audited_metadata.csv
artifacts/logs/dataset_audit.json
artifacts/report_assets/tables/class_distribution.csv
artifacts/report_assets/tables/lesion_class_distribution.csv
artifacts/report_assets/figures/class_distribution.svg
```

Canonical lesion-aware split:

```bash
PYTHONPATH=src uv run python scripts/make_lesion_split.py --config configs/dataset/selected_dataset.yaml

# fallback
PYTHONPATH=src "$CODEX_PYTHON" scripts/make_lesion_split.py --config configs/dataset/selected_dataset.yaml
```

Creates:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
artifacts/logs/split_manifest.json
artifacts/report_assets/tables/split_summary.csv
artifacts/report_assets/tables/split_class_distribution.csv
artifacts/report_assets/tables/lesion_leakage_audit.csv
artifacts/report_assets/figures/split_class_distribution.svg
```

Split image-open smoke check:

```bash
PYTHONPATH=src uv run python scripts/smoke_dataloader.py --config configs/dataset/selected_dataset.yaml --max-samples 8

# fallback
PYTHONPATH=src "$CODEX_PYTHON" scripts/smoke_dataloader.py --config configs/dataset/selected_dataset.yaml --max-samples 8
```

Creates:

```text
artifacts/logs/dataloader_smoke.json
```

These commands do not train models and do not use the test split for model selection.

## Colab Commands

## Sprint 2 Commands

Sprint 2 unit and smoke tests:

```bash
PYTHONPATH=src uv run python -m pytest tests/test_dataset_sprint1.py tests/test_sprint2_features.py
```

Frozen transformer feature extraction smoke run without pretrained downloads:

```bash
PYTHONPATH=src uv run python scripts/extract_features.py \
  --config configs/dataset/selected_dataset.yaml \
  --backbones vit_b16 swin_tiny deit3_small \
  --splits train val \
  --limit-per-split 2 \
  --batch-size 1 \
  --num-workers 0 \
  --no-pretrained \
  --output-root artifacts/features_smoke
```

Full frozen train/validation feature extraction:

```bash
PYTHONPATH=src uv run python scripts/extract_features.py \
  --config configs/dataset/selected_dataset.yaml \
  --backbones vit_b16 swin_tiny deit3_small \
  --splits train val \
  --batch-size 32 \
  --num-workers 2
```

Optional test cache generation for final-audit readiness only:

```bash
PYTHONPATH=src uv run python scripts/extract_features.py \
  --config configs/dataset/selected_dataset.yaml \
  --backbones vit_b16 swin_tiny deit3_small \
  --splits test \
  --batch-size 32 \
  --num-workers 2
```

Train validation-selected single-backbone MLP baselines:

```bash
PYTHONPATH=src uv run python scripts/train_mlp.py \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --backbones vit_b16 swin_tiny deit3_small
```

Sprint 2 MLP training reads train/validation caches only. Test metrics are not produced in Sprint 2 and must not be used for model selection.

BEiT candidate screening for the third-backbone slot:

```bash
PYTHONPATH=src uv run python scripts/extract_features.py \
  --config configs/dataset/selected_dataset.yaml \
  --backbones beit_base \
  --splits train val \
  --batch-size 32 \
  --num-workers 2

PYTHONPATH=src uv run python scripts/train_mlp.py \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --backbones beit_base \
  --run-tag beit_screen
```

This is still validation-only screening. Do not compute or compare test metrics here.

Creates:

```text
artifacts/features/ham10000/frozen/{vit_b16,swin_tiny,deit3_small}/
artifacts/runs/*_s2_frozen_*_none_mlp*_seed42/
artifacts/report_assets/tables/single_backbone_frozen_results.csv
artifacts/report_assets/tables/single_backbone_frozen_per_class_metrics.csv
artifacts/report_assets/figures/frozen_single_backbone_macro_f1.png
```

## Sprint 3 Commands

Sprint 3 unit and smoke tests:

```bash
PYTHONPATH=src uv run python -m pytest tests/test_sprint2_features.py tests/test_sprint3_fusion.py
```

One-run real-cache smoke test:

```bash
PYTHONPATH=src uv run python scripts/run_fusion_matrix.py \
  --backbones vit_b16 swin_tiny \
  --fusion-methods concat \
  --max-runs 1 \
  --epochs 1 \
  --batch-size 256 \
  --device cpu \
  --run-root artifacts/runs_smoke \
  --tables-dir artifacts/report_assets_smoke/tables \
  --figures-dir artifacts/report_assets_smoke/figures
```

Full validation-only frozen fusion matrix:

```bash
PYTHONPATH=src uv run python scripts/run_fusion_matrix.py \
  --batch-size 128 \
  --device cpu
```

BEiT-expanded E2 matrix after the planned ViT/Swin/DeiT matrix:

```bash
PYTHONPATH=src uv run python scripts/run_fusion_matrix.py \
  --only-combination vit_b16 beit_base \
  --fusion-methods concat weighted_learned_512 weighted_pca_384 \
  --batch-size 128 \
  --device cpu \
  --run-tag beit_matrix

PYTHONPATH=src uv run python scripts/run_fusion_matrix.py \
  --only-combination swin_tiny beit_base \
  --fusion-methods concat weighted_learned_512 weighted_pca_384 \
  --batch-size 128 \
  --device cpu \
  --run-tag beit_matrix

PYTHONPATH=src uv run python scripts/run_fusion_matrix.py \
  --only-combination vit_b16 swin_tiny beit_base \
  --fusion-methods concat weighted_learned_512 weighted_pca_384 \
  --batch-size 128 \
  --device cpu \
  --run-tag beit_matrix
```

Representation similarity diagnostic for E2:

```bash
PYTHONPATH=src uv run python scripts/analyze_representation_similarity.py \
  --backbones vit_b16 swin_tiny deit3_small beit_base \
  --split val \
  --max-samples 1504
```

Creates:

```text
artifacts/report_assets/tables/frozen_representation_similarity_val.csv
artifacts/report_assets/tables/frozen_fusion_complementarity_val.csv
artifacts/report_assets/tables/frozen_representation_similarity_val.json
artifacts/report_assets/figures/frozen_representation_similarity_val.png
```

E2b MLP capacity diagnostic:

```bash
# ViT single stronger MLP probes
PYTHONPATH=src uv run python scripts/train_mlp.py \
  --backbones vit_b16 \
  --hidden-dims 1024 512 256 \
  --dropout 0.4 \
  --learning-rate 0.0007 \
  --weight-decay 0.0001 \
  --epochs 50 \
  --early-stopping-patience 8 \
  --run-tag e2b_wide

PYTHONPATH=src uv run python scripts/train_mlp.py \
  --backbones vit_b16 \
  --hidden-dims 1024 512 256 \
  --dropout 0.5 \
  --learning-rate 0.0005 \
  --weight-decay 0.0003 \
  --epochs 50 \
  --early-stopping-patience 8 \
  --run-tag e2b_wide_reg

PYTHONPATH=src uv run python scripts/train_mlp.py \
  --backbones vit_b16 \
  --hidden-dims 2048 1024 512 \
  --dropout 0.5 \
  --learning-rate 0.0003 \
  --weight-decay 0.0005 \
  --epochs 60 \
  --early-stopping-patience 10 \
  --run-tag e2b_deep_reg
```

The same three MLP variants were run for representative concat fusion conditions:

```text
vit_b16 + swin_tiny
vit_b16 + swin_tiny + deit3_small
vit_b16 + swin_tiny + beit_base
```

Creates:

```text
artifacts/report_assets/tables/e2b_mlp_capacity_diagnostic.csv
```

Quality checks:

```bash
PYTHONPATH=src uv run ruff check .
PYTHONPATH=src uv run python -m pytest tests/test_dataset_sprint1.py tests/test_sprint2_features.py tests/test_sprint3_fusion.py tests/test_representation_complementarity.py
```

Creates:

```text
artifacts/runs/*_s3_frozen_*_concat_mlp_seed42/
artifacts/runs/*_s3_frozen_*_weightedlearned512_mlp_seed42/
artifacts/runs/*_s3_frozen_*_weightedpca384_mlp_seed42/
artifacts/runs/s3_frozen_fusion_manifest.json
artifacts/report_assets/tables/frozen_fusion_results.csv
artifacts/report_assets/tables/frozen_fusion_per_class_metrics.csv
artifacts/report_assets/tables/frozen_fusion_weight_summary.csv
artifacts/report_assets/tables/frozen_fusion_vs_single_validation.csv
artifacts/report_assets/figures/frozen_fusion_macro_f1.png
```

Sprint 3 reads train/validation caches only and does not compute test metrics. Model, checkpoint, fusion method, and fusion-weight interpretation use validation macro-F1.

Sprint 4 forward backbone set after E2:

```text
vit_b16
swin_tiny
beit_base
```

`deit3_small` remains a planned/screened baseline from Sprint 2-3 and is not carried into the fine-tuning compute budget unless a later decision supersedes D021.

## Sprint 4 Commands

Sprint 4 implementation and policy tests:

```bash
PYTHONPATH=src uv run python -m pytest \
  tests/test_sprint2_features.py \
  tests/test_sprint3_fusion.py \
  tests/test_sprint4_finetune.py
```

Local one-backbone smoke run without pretrained downloads:

```bash
PYTHONPATH=src uv run python scripts/finetune_backbone.py \
  --config configs/experiments/finetune_backbones.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --backbones vit_b16 \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --limit-per-split 2 \
  --no-pretrained \
  --no-mixed-precision \
  --checkpoint-dir artifacts/checkpoints_smoke/ham10000/finetuned \
  --feature-root artifacts/features_smoke \
  --run-root artifacts/runs_smoke
```

Full Colab/GPU fine-tuning and fine-tuned train/validation cache extraction:

```bash
PYTHONPATH=src uv run python scripts/finetune_backbone.py \
  --config configs/experiments/finetune_backbones.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --backbones vit_b16 swin_tiny beit_base \
  --batch-size 16 \
  --num-workers 2
```

Fine-tuned single-backbone MLP validation runs:

```bash
PYTHONPATH=src uv run python scripts/train_mlp.py \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned \
  --backbones vit_b16 swin_tiny beit_base \
  --batch-size 128
```

Representative fine-tuned fusion matrix:

```bash
PYTHONPATH=src uv run python scripts/run_fusion_matrix.py \
  --feature-source finetuned \
  --only-combination vit_b16 swin_tiny \
  --fusion-methods concat weighted_learned_512 weighted_pca_384 \
  --batch-size 128 \
  --run-tag s4_pair

PYTHONPATH=src uv run python scripts/run_fusion_matrix.py \
  --feature-source finetuned \
  --only-combination vit_b16 swin_tiny beit_base \
  --fusion-methods concat weighted_learned_512 weighted_pca_384 \
  --batch-size 128 \
  --run-tag s4_triple
```

Optional E2b-style fine-tuned concat capacity diagnostic:

```bash
PYTHONPATH=src uv run python scripts/run_fusion_matrix.py \
  --feature-source finetuned \
  --only-combination vit_b16 swin_tiny \
  --fusion-methods concat \
  --hidden-dims 2048 1024 512 \
  --dropout 0.5 \
  --learning-rate 0.0003 \
  --weight-decay 0.0005 \
  --epochs 60 \
  --early-stopping-patience 10 \
  --batch-size 128 \
  --run-tag s4_deep_reg
```

Quality checks:

```bash
PYTHONPATH=src uv run ruff check .
PYTHONPATH=src uv run python -m pytest \
  tests/test_dataset_sprint1.py \
  tests/test_sprint2_features.py \
  tests/test_sprint3_fusion.py \
  tests/test_representation_complementarity.py \
  tests/test_sprint4_finetune.py
```

Sprint 4 commands read train/validation only and do not compute test metrics. Full fine-tuning can
be run in Colab with Drive root:

```text
MyDrive/dl-final-artifact/
```

## Colab Commands

GPU gerektiren full transformer extraction ve fine-tuning işleri Sprint 2+ sırasında Colab'de çalıştırılabilir. Büyük artifact akışı için Drive root:

```text
MyDrive/dl-final-artifact/
```

## Report Commands

Rapor/sunum build ve figure generation komutları final report aşamasında eklenecektir.
