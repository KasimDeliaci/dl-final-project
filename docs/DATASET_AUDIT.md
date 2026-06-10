# Dataset Audit

Bu dosya Sprint 1 dataset audit ve split doğrulama sonucunu kaydeder. HAM10000 burada klinik teşhis sistemi olarak değil, benchmark dermoscopic image classification veri seti olarak ele alınır.

## Dataset Identity

- Dataset: HAM10000
- Task: 7-class dermoscopic image classification benchmark
- Metadata file: `data/metadata/HAM10000_metadata.csv`
- Raw image root: `data/raw`
- Audited metadata output: `data/processed/ham10000_audited_metadata.csv`
- Number of metadata rows: 10,015
- Unique image IDs: 10,015
- Unique lesion IDs: 7,470
- Number of classes: 7
- Label column: `dx`
- Grouping column: `lesion_id`

## Audit Result

| Check | Result |
|---|---:|
| Metadata rows | 10,015 |
| Unique image IDs | 10,015 |
| Duplicate image IDs | 0 |
| Missing images | 0 |
| Unreferenced classification images | 0 |
| Missing labels | 0 |
| Missing lesion IDs | 0 |
| Unreadable sampled images | 0 |

The audit writes machine-readable evidence to:

```text
artifacts/logs/dataset_audit.json
artifacts/report_assets/tables/dataset_audit_summary.csv
artifacts/report_assets/tables/class_distribution.csv
artifacts/report_assets/tables/lesion_class_distribution.csv
```

## Class Distribution

| Class | Images | Percent |
|---|---:|---:|
| akiec | 327 | 3.2651 |
| bcc | 514 | 5.1323 |
| bkl | 1,099 | 10.9735 |
| df | 115 | 1.1483 |
| nv | 6,705 | 66.9496 |
| mel | 1,113 | 11.1133 |
| vasc | 142 | 1.4179 |

HAM10000 strongly favors `nv`; therefore accuracy alone is not a sufficient interpretation metric. Later model comparisons will report macro-F1, macro precision/recall, weighted-F1, per-class metrics, confusion matrices, and prediction dumps.

Report-ready figure:

```text
artifacts/report_assets/figures/class_distribution.svg
```

## Split Policy

Canonical split policy:

- Train: approximately 70%
- Validation: approximately 15%
- Test: approximately 15%
- Seed: 42
- Grouping: `lesion_id`
- Rule: no lesion ID may appear in more than one split.

The generated split files are:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

These files are generated artifacts and remain out of Git.

## Split Summary

| Split | Images | Lesions | Classes Present |
|---|---:|---:|---:|
| train | 7,008 | 5,233 | 7 |
| validation | 1,504 | 1,117 | 7 |
| test | 1,503 | 1,120 | 7 |

## Split Class Distribution

| Split | akiec | bcc | bkl | df | nv | mel | vasc |
|---|---:|---:|---:|---:|---:|---:|---:|
| train | 229 | 360 | 769 | 80 | 4,693 | 779 | 98 |
| validation | 49 | 77 | 165 | 18 | 1,006 | 167 | 22 |
| test | 49 | 77 | 165 | 17 | 1,006 | 167 | 22 |

Report-ready figure:

```text
artifacts/report_assets/figures/split_class_distribution.svg
```

## Leakage Audit

| Split Pair | Overlap Lesions |
|---|---:|
| train / validation | 0 |
| train / test | 0 |
| validation / test | 0 |

The leakage audit is also exported to:

```text
artifacts/report_assets/tables/lesion_leakage_audit.csv
artifacts/logs/split_manifest.json
```

## Smoke Check

`scripts/smoke_dataloader.py` opened eight sample images from each split and verified that each split contains all seven classes. Sample images were RGB-convertible with size `600x450`.

Smoke output:

```text
artifacts/logs/dataloader_smoke.json
```

## Preprocessing Notes

Sprint 1 does not define transformer preprocessing beyond verifying raw image readability and split integrity. Transformer-specific resize/crop and normalization policies will be fixed in Sprint 2 before feature extraction. The split produced here is the canonical split for frozen feature extraction, fusion, fine-tuning, and final audit unless a later decision explicitly supersedes it.

