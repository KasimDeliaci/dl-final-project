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

## Colab Commands

GPU gerektiren full transformer extraction ve fine-tuning işleri Sprint 2+ sırasında Colab'de çalıştırılabilir. Büyük artifact akışı için Drive root:

```text
MyDrive/dl-final-artifact/
```

## Report Commands

Rapor/sunum build ve figure generation komutları final report aşamasında eklenecektir.
