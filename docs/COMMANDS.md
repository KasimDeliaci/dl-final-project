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

GPU gerektiren fine-tuning ve image-level transformer işleri Sprint 2+ sırasında eklenecektir. Büyük artifact akışı için Drive root:

```text
MyDrive/dl-final-artifact/
```

## Report Commands

Rapor/sunum build ve figure generation komutları final report aşamasında eklenecektir.
