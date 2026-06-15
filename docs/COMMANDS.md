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

Sprint 4b / E3b downstream MLP multi-seed robustness diagnostic over fixed cached features:

```bash
for seed in 7 13 42 101 202; do
  PYTHONPATH=src uv run python scripts/run_fusion_matrix.py \
    --feature-source finetuned \
    --only-combination vit_b16 swin_tiny beit_base \
    --fusion-methods concat \
    --batch-size 128 \
    --device cpu \
    --seed "$seed" \
    --experiment-id E3b \
    --run-tag s4b_multiseed_cpu_triple \
    --skip-export

  PYTHONPATH=src uv run python scripts/run_fusion_matrix.py \
    --feature-source finetuned \
    --only-combination vit_b16 swin_tiny \
    --fusion-methods concat \
    --batch-size 128 \
    --device cpu \
    --seed "$seed" \
    --experiment-id E3b \
    --run-tag s4b_multiseed_cpu_pair \
    --skip-export

  PYTHONPATH=src uv run python scripts/train_mlp.py \
    --feature-source finetuned \
    --backbones vit_b16 \
    --batch-size 128 \
    --device cpu \
    --seed "$seed" \
    --experiment-id E3b \
    --run-tag s4b_multiseed_cpu_vit \
    --tables-dir artifacts/report_assets_s4b/tables \
    --figures-dir artifacts/report_assets_s4b/figures

  PYTHONPATH=src uv run python scripts/run_fusion_matrix.py \
    --feature-source frozen \
    --only-combination vit_b16 swin_tiny \
    --fusion-methods concat \
    --hidden-dims 2048 1024 512 \
    --dropout 0.5 \
    --learning-rate 0.0003 \
    --weight-decay 0.0005 \
    --epochs 60 \
    --early-stopping-patience 10 \
    --batch-size 128 \
    --device cpu \
    --seed "$seed" \
    --experiment-id E3b \
    --run-tag s4b_multiseed_cpu_frozen_deep_reg \
    --skip-export
done

PYTHONPATH=src uv run python scripts/summarize_s4b_multiseed.py
```

Creates:

```text
artifacts/runs/*s4b_multiseed_cpu*/
artifacts/report_assets/tables/s4b_multiseed_cpu_diagnostic_results.csv
artifacts/report_assets/tables/s4b_multiseed_cpu_diagnostic_summary.csv
```

E3c metadata-augmented cached-feature diagnostic:

```bash
for seed in 7 13 42 101 202; do
  PYTHONPATH=src uv run python scripts/train_metadata_augmented_mlp.py \
    --condition metadata_only ft_vit_swin_concat_plus_metadata ft_vit_swin_beit_concat_plus_metadata \
    --device cpu \
    --batch-size 128 \
    --seed "$seed" \
    --run-tag e3c_metadata
done

PYTHONPATH=src uv run python scripts/summarize_e3c_metadata.py
```

Creates:

```text
artifacts/runs/*_e3c_metadata_*_seed*/
artifacts/report_assets/tables/e3c_metadata_augmented_results.csv
artifacts/report_assets/tables/e3c_metadata_augmented_summary.csv
artifacts/report_assets/tables/e3c_metadata_augmented_per_class_metrics.csv
artifacts/report_assets/tables/e3c_metadata_per_class_delta_vs_image_only.csv
artifacts/report_assets/tables/e3c_metadata_vs_image_only_validation.csv
artifacts/report_assets/figures/e3c_metadata_augmented_macro_f1.png
```

E3c artifact integrity check:

```bash
PYTHONPATH=src uv run python - <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd

runs = sorted(Path("artifacts/runs").glob("*e3c_metadata*/run_config.json"))
assert len(runs) == 15, len(runs)
for config_path in runs:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(config_path.parent / "predictions.csv")
    metadata = json.loads((config_path.parent / "metadata_preprocessing.json").read_text())
    runtime = json.loads((config_path.parent / "runtime_metadata.json").read_text())
    assert config["experiment_id"] == "E3c"
    assert config["test_policy"] == "not_loaded_or_used_in_e3c"
    assert runtime["test_rows_loaded"] == 0
    assert len(predictions) == 1504
    probability_columns = [col for col in predictions.columns if col.startswith("prob_")]
    assert len(probability_columns) == 7
    assert np.isfinite(predictions[probability_columns].to_numpy()).all()
    assert metadata["metadata_preprocessor"]["fit_split"] == "train"
    assert metadata["metadata_preprocessor"]["output_dim"] == 19
print("E3c artifact integrity check passed.")
PY
```

E3d metadata fusion operator ablation:

```bash
for seed in 7 13 42 101 202; do
  PYTHONPATH=src uv run python scripts/train_metadata_fusion_operator.py \
    --operators triple_metadata_gated_backbone triple_metadata_film triple_metadata_two_branch \
    --device cpu \
    --batch-size 128 \
    --seed "$seed" \
    --run-tag e3d_metadata_fusion
done

PYTHONPATH=src uv run python scripts/summarize_e3d_metadata_fusion.py
```

Creates:

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

E3d artifact integrity check:

```bash
PYTHONPATH=src uv run python - <<'PY'
from pathlib import Path
import json
import numpy as np
import pandas as pd

runs = sorted(Path("artifacts/runs").glob("*e3d_metadata_fusion*/run_config.json"))
assert len(runs) == 15, len(runs)
for config_path in runs:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    predictions = pd.read_csv(config_path.parent / "predictions.csv")
    metadata = json.loads((config_path.parent / "metadata_preprocessing.json").read_text())
    fusion = json.loads((config_path.parent / "metadata_fusion_metadata.json").read_text())
    runtime = json.loads((config_path.parent / "runtime_metadata.json").read_text())
    assert config["experiment_id"] == "E3d"
    assert config["test_policy"] == "not_loaded_or_used_in_e3d"
    assert runtime["test_rows_loaded"] == 0
    assert len(predictions) == 1504
    probability_columns = [col for col in predictions.columns if col.startswith("prob_")]
    assert len(probability_columns) == 7
    assert np.isfinite(predictions[probability_columns].to_numpy()).all()
    assert metadata["metadata_preprocessor"]["fit_split"] == "train"
    assert metadata["metadata_preprocessor"]["output_dim"] == 19
    assert fusion["selection_metric"] == "validation_macro_f1"
print("E3d artifact integrity check passed.")
PY
```

Quality checks:

```bash
PYTHONPATH=src uv run ruff check .
PYTHONPATH=src uv run python -m pytest \
  tests/test_dataset_sprint1.py \
  tests/test_sprint2_features.py \
  tests/test_sprint3_fusion.py \
  tests/test_representation_complementarity.py \
  tests/test_sprint4_finetune.py \
  tests/test_metadata_features.py \
  tests/test_e3c_metadata_augmented.py \
  tests/test_metadata_fusion_models.py \
  tests/test_e3d_metadata_fusion.py
```

## E3e Conservative ViT Fine-Tuning Diagnostic

Colab runner:

```text
notebooks/05_e3e_conservative_vit_finetuning.ipynb
```

Drive output namespace:

```text
MyDrive/dl-final-artifact/e3e_conservative_vit/
```

Local smoke run:

```bash
PYTHONPATH=src uv run python scripts/finetune_backbone.py \
  --config configs/experiments/e3e_vit_last1_lr5e6.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --backbones vit_b16 \
  --epochs 1 \
  --batch-size 1 \
  --num-workers 0 \
  --limit-per-split 2 \
  --no-pretrained \
  --no-mixed-precision \
  --checkpoint-dir artifacts/checkpoints_smoke/ham10000/e3e_vit_last1_lr5e6 \
  --feature-root artifacts/features_smoke \
  --run-root artifacts/runs_smoke
```

Colab full fine-tuning commands:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python scripts/finetune_backbone.py \
  --config configs/experiments/e3e_vit_last2_lr5e6.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --backbones vit_b16 \
  --batch-size 16 \
  --num-workers 2

PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python scripts/finetune_backbone.py \
  --config configs/experiments/e3e_vit_last1_lr5e6.yaml \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --backbones vit_b16 \
  --batch-size 16 \
  --num-workers 2
```

E3e ViT single-backbone MLP:

```bash
PYTHONPATH=src uv run python scripts/train_mlp.py \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_vit_last2_lr5e6 \
  --backbones vit_b16 \
  --batch-size 128 \
  --experiment-id E3e \
  --test-policy not_loaded_or_used_in_e3e \
  --run-tag e3e_vit_single

PYTHONPATH=src uv run python scripts/train_mlp.py \
  --dataset-config configs/dataset/selected_dataset.yaml \
  --feature-source finetuned_vit_last1_lr5e6 \
  --backbones vit_b16 \
  --batch-size 128 \
  --experiment-id E3e \
  --test-policy not_loaded_or_used_in_e3e \
  --run-tag e3e_vit_single
```

E3e mixed-source triple concat after copying canonical Swin/BEiT caches into the mixed source:

```bash
PYTHONPATH=src uv run python scripts/run_fusion_matrix.py \
  --feature-source finetuned_vit_last2_lr5e6_plus_s4_swin_beit \
  --only-combination vit_b16 swin_tiny beit_base \
  --fusion-methods concat \
  --batch-size 128 \
  --experiment-id E3e \
  --test-policy not_loaded_or_used_in_e3e \
  --run-tag e3e_mixed_triple

PYTHONPATH=src uv run python scripts/run_fusion_matrix.py \
  --feature-source finetuned_vit_last1_lr5e6_plus_s4_swin_beit \
  --only-combination vit_b16 swin_tiny beit_base \
  --fusion-methods concat \
  --batch-size 128 \
  --experiment-id E3e \
  --test-policy not_loaded_or_used_in_e3e \
  --run-tag e3e_mixed_triple
```

Optional metadata-conditioned follow-up, only if validation results justify it:

```bash
PYTHONPATH=src uv run python scripts/train_metadata_fusion_operator.py \
  --feature-source finetuned_vit_last1_lr5e6_plus_s4_swin_beit \
  --operators triple_metadata_film triple_metadata_gated_backbone \
  --device cpu \
  --batch-size 128 \
  --experiment-id E3e \
  --test-policy not_loaded_or_used_in_e3e \
  --run-tag e3e_metadata_followup
```

Sprint 4 commands read train/validation only and do not compute test metrics. Full fine-tuning can
be run in Colab with Drive root:

```text
MyDrive/dl-final-artifact/
```

Post-Colab artifact sync from a downloaded artifact bundle:

```bash
rsync -av /Users/arcustin2/Downloads/artifacts/ /Users/arcustin2/kasim/dl-final/artifacts/
```

Sprint 4 artifact integrity check:

```bash
PYTHONPATH=src uv run python - <<'PY'
from pathlib import Path
import json
import pandas as pd
import torch

from dl_final.features.cache import feature_cache_path, load_feature_cache, verify_cache_matches_split

root = Path("artifacts/features/ham10000/finetuned")
expected = {"train": 7008, "val": 1504}
for backbone in ["vit_b16", "swin_tiny", "beit_base"]:
    for split, rows in expected.items():
        cache = load_feature_cache(feature_cache_path(root / backbone, split))
        assert tuple(cache.features.shape) == (rows, 768)
        assert torch.isfinite(cache.features).all()
        assert cache.metadata["feature_source"] == "finetuned"
        verify_cache_matches_split(cache, Path("data/splits") / f"{split}.csv")

for path in sorted(Path("artifacts/runs").glob("*finetuned*/predictions.csv")):
    frame = pd.read_csv(path)
    assert len(frame) == 1504, path
    prob_cols = [col for col in frame.columns if col.startswith("prob_")]
    assert len(prob_cols) == 7, path
    assert frame[prob_cols].notna().all().all(), path
    run_config = json.loads(path.with_name("run_config.json").read_text())
    assert run_config["test_policy"] == "not_used_in_sprint4", path

for path in sorted(Path("artifacts/runs").glob("s4_finetune_*_seed42/predictions.csv")):
    frame = pd.read_csv(path)
    assert len(frame) == 1504, path
    prob_cols = [col for col in frame.columns if col.startswith("prob_")]
    assert len(prob_cols) == 7, path
    assert frame[prob_cols].notna().all().all(), path
    run_config = json.loads(path.with_name("run_config.json").read_text())
    assert run_config["test_policy"] == "not_used_in_sprint4", path

print("Sprint 4 artifact integrity check passed.")
PY
```

## E3f Mixed Frozen ViT + Fine-Tuned Swin/BEiT

Create the mixed feature source without overwriting canonical frozen or fine-tuned caches:

```bash
PYTHONPATH=src uv run python scripts/create_mixed_feature_source.py \
  --output-source frozen_vit_finetuned_swin_beit \
  --mapping vit_b16=frozen swin_tiny=finetuned beit_base=finetuned \
  --overwrite
```

Run image-only concat over five downstream MLP seeds:

```bash
for seed in 7 13 42 101 202; do
  PYTHONPATH=src uv run python scripts/run_fusion_matrix.py \
    --feature-source frozen_vit_finetuned_swin_beit \
    --only-combination vit_b16 swin_tiny beit_base \
    --fusion-methods concat \
    --device cpu \
    --batch-size 128 \
    --seed "$seed" \
    --experiment-id E3f \
    --test-policy not_loaded_or_used_in_e3f \
    --run-tag e3f_mixed_adaptation \
    --skip-export
done
```

Run metadata-conditioned FiLM and gated operators over the same seeds:

```bash
for seed in 7 13 42 101 202; do
  PYTHONPATH=src uv run python scripts/train_metadata_fusion_operator.py \
    --feature-source frozen_vit_finetuned_swin_beit \
    --operators triple_metadata_film triple_metadata_gated_backbone \
    --device cpu \
    --batch-size 128 \
    --seed "$seed" \
    --experiment-id E3f \
    --test-policy not_loaded_or_used_in_e3f \
    --run-tag e3f_mixed_adaptation
done
```

Summarize E3f:

```bash
PYTHONPATH=src uv run python scripts/summarize_e3f_mixed_adaptation.py
```

E3f artifact integrity check:

```bash
PYTHONPATH=src uv run python - <<'PY'
from pathlib import Path
import pandas as pd
import torch

from dl_final.features.cache import feature_cache_path, load_feature_cache

root = Path("artifacts/features/ham10000/frozen_vit_finetuned_swin_beit")
expected = {"train": 7008, "val": 1504}
for backbone in ["vit_b16", "swin_tiny", "beit_base"]:
    for split, rows in expected.items():
        cache = load_feature_cache(feature_cache_path(root / backbone, split))
        assert tuple(cache.features.shape) == (rows, 768), (backbone, split)
        assert torch.isfinite(cache.features).all(), (backbone, split)

paths = sorted(Path("artifacts/runs").glob("*e3f_mixed_adaptation*/predictions.csv"))
assert len(paths) == 15, len(paths)
for path in paths:
    frame = pd.read_csv(path)
    assert len(frame) == 1504, path
    prob_cols = [col for col in frame.columns if col.startswith("prob_")]
    assert len(prob_cols) == 7, path
    assert frame[prob_cols].notna().all().all(), path

print("E3f artifact integrity check passed.")
PY
```

## E3g Prediction-Level Ensemble

Build validation-only probability ensembles from existing E3d/E3f prediction dumps:

```bash
PYTHONPATH=src uv run python scripts/ensemble_predictions.py
```

To skip validation-weighted diagnostic grids and run only primary equal-weight ensembles:

```bash
PYTHONPATH=src uv run python scripts/ensemble_predictions.py \
  --skip-weighted-diagnostics
```

E3g artifact integrity check:

```bash
PYTHONPATH=src uv run python - <<'PY'
from pathlib import Path
import json
import pandas as pd

run = Path("artifacts/runs/e3g_prediction_ensemble")
config = json.loads((run / "run_config.json").read_text())
assert config["test_policy"] == "not_loaded_or_used_in_e3g"

results = pd.read_csv(run / "ensemble_results.csv")
assert "top3_family_equal" in set(results["ensemble_id"])

for path in sorted(run.glob("ensemble_predictions_*.csv")):
    frame = pd.read_csv(path)
    assert len(frame) == 1504, path
    prob_cols = [col for col in frame.columns if col.startswith("prob_")]
    assert len(prob_cols) == 7, path
    assert frame[prob_cols].notna().all().all(), path
    assert ((frame[prob_cols].sum(axis=1) - 1).abs() < 1e-4).all(), path

print("E3g artifact integrity check passed.")
PY
```

## E3h Rot4 TTA

E3h is a validation-only inference-time TTA diagnostic. Full validation should run on Colab GPU, not
on the local laptop.

Colab runner:

```text
notebooks/06_e3h_tta_rot4.ipynb
```

The notebook restores inputs from `MyDrive/dl-final-artifact/artifacts/`. If that tree is not
complete, it can also restore `MyDrive/dl-final-artifact/e3h_tta_rot4/e3h_tta_inputs.tar`, whose
archive root should contain `artifacts/...`. The notebook keeps a T4-safe default
`E3H_BATCH_SIZE = 128`; raise that single variable to `192` or `256` only if the assigned Colab GPU
has enough memory.

Local smoke only:

```bash
PYTHONPATH=src uv run python scripts/evaluate_tta_rot4.py \
  --max-samples 8 \
  --batch-size 4 \
  --num-workers 0 \
  --device auto \
  --no-mixed-precision
```

Full Colab command:

```bash
PYTHONUNBUFFERED=1 PYTHONPATH=src uv run python scripts/evaluate_tta_rot4.py \
  --batch-size 128 \
  --num-workers 2 \
  --device cuda \
  --identity-tolerance 1e-3
```

E3h artifact integrity check:

```bash
PYTHONPATH=src uv run python - <<'PY'
from pathlib import Path
import json
import pandas as pd

from dl_final.evaluation.tta import probabilities_from_frame

run = Path("artifacts/runs/e3h_tta_rot4")
config = json.loads((run / "run_config.json").read_text())
assert config["test_policy"] == "not_loaded_or_used_in_e3h"
assert config["views"] == ["identity", "rot90", "rot180", "rot270"]

pred = pd.read_csv(run / "predictions_top3_family_equal_tta_rot4.csv")
assert len(pred) == 1504, len(pred)
assert set(pred["split"].astype(str)) == {"val"}
_ = probabilities_from_frame(pred, ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"])

identity = pd.read_csv(run / "tta_identity_sanity.csv")
assert len(identity) == 15, len(identity)
assert identity["passed"].all()

results = pd.read_csv(run / "tta_ensemble_results.csv")
assert set(results["ensemble_id"]) == {"top3_family_equal_tta_rot4"}

print("E3h artifact integrity check passed.")
PY
```

## Colab Commands

GPU gerektiren full transformer extraction ve fine-tuning işleri Sprint 2+ sırasında Colab'de çalıştırılabilir. Büyük artifact akışı için Drive root:

```text
MyDrive/dl-final-artifact/
```

## Report Commands

Rapor/sunum build ve figure generation komutları final report aşamasında eklenecektir.
