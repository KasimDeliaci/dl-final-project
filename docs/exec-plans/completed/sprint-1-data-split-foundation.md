# Sprint 1 Dataset, Split, and Foundation Execution Plan

## Objective

Make the HAM10000 dataset foundation trustworthy before any transformer feature extraction, fusion experiment, fine-tuning, notebook runner, or training work begins.

Sprint 1 should answer one question:

> Can we create and verify a reproducible, lesion-aware HAM10000 train/validation/test setup that is strong enough to support all later transformer representation and fusion comparisons?

The sprint ends with a dataset audit, fixed split files, verification evidence, and a short report note. It does not end with a model.

## Context

This final project mirrors the discipline learned from the earlier CNN-based `dl-assignment` project, but the backbone family changes to transformer models:

- Vanilla ViT
- Swin Transformer
- DeiT III-Small

The later experiment story depends on three axes:

- transformer representation quality,
- feature fusion complementarity,
- frozen vs fine-tuned transfer learning.

Those axes are only meaningful if the dataset, split, class labels, and evaluation protocol are fixed first.

## Non-Goals

Do not do any of the following in Sprint 1:

- Do not implement transformer backbones.
- Do not extract features.
- Do not train MLP classifiers.
- Do not fine-tune models.
- Do not run fusion experiments.
- Do not use the test split for model selection.
- Do not add experimental branches that are not needed for dataset audit and split verification.

## Constraints

- Keep raw images, processed data, split outputs, feature caches, checkpoints, prediction dumps, and large artifacts out of Git.
- Use benchmark dermoscopic image classification language, not clinical diagnosis claims.
- Keep notebooks as thin launchers only; real logic belongs under `src/` and `scripts/`.
- Do not generate dummy split or fake audit results if HAM10000 files are absent.
- Fail clearly when required metadata or images are missing.
- Treat `docs/EXPERIMENT_REGISTRY.md` entry `E0 - Dataset Audit and Split` as the sprint-level experiment record.
- Preserve the validation/test distinction from the first split artifact onward.
- Use the new Drive artifact root `dl-final-artifact` for large files and Colab exchange, not the old `dl-midterm` location.

## Planned Workstreams

### 1. Dataset Contract

Confirm the canonical HAM10000 setup:

- Metadata file: `HAM10000_metadata.csv`.
- Task: seven-class image classification.
- Label column: `dx`.
- Image identity: `image_id`.
- Lesion grouping: `lesion_id`.
- Labels:
  - `akiec`
  - `bcc`
  - `bkl`
  - `df`
  - `mel`
  - `nv`
  - `vasc`

Document any source or local-path assumptions in `docs/DATASET_AUDIT.md` and `docs/COMMANDS.md`.

### 2. Repository and Environment Foundation

Create the minimum implementation surface needed for dataset work only:

- dataset config parsing,
- HAM10000 metadata normalization,
- image path resolution,
- split generation,
- leakage audit,
- dataloader smoke test,
- dataset/split report asset export.

The implementation should stay narrow. Transformer model code belongs to Sprint 2.

### 3. HAM10000 Metadata and Image Audit

Audit:

- metadata row count,
- unique `image_id` count,
- duplicate `image_id` values,
- missing labels,
- unknown labels outside the fixed seven-class mapping,
- missing `lesion_id` values,
- unique lesion count,
- image files present for every metadata row,
- image files not referenced by metadata,
- class distribution,
- lesion-level class distribution,
- basic image-open smoke check.

Output should distinguish blocking errors from warnings.

Blocking examples:

- metadata file missing,
- required columns missing,
- unknown label,
- large number of missing images,
- impossible image path mapping.

Warning examples:

- optional demographic metadata missing,
- lesion-aware split causing imperfect class ratio,
- extra image files not referenced by metadata.

### 4. Lesion-Aware Split Generation

Generate the canonical split only after the audit passes.

Target policy:

- Train/validation/test split near `70/15/15`.
- Group by `lesion_id`; no lesion can appear in more than one split.
- Preserve class ratios as much as feasible under group constraints.
- Fixed seed, initially `42`.

Expected split files:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

Each split row should carry at least:

```text
sample_id
image_id
lesion_id
label
split
image_path
```

If exact stratification is impossible because of minority-class lesion groups, document the compromise in `docs/DATASET_AUDIT.md` and `docs/DECISIONS.md`.

### 5. Verification and Smoke Tests

Add focused tests/smoke checks for:

- metadata normalization,
- unknown-label failure,
- missing-image detection,
- deterministic split reproduction,
- no cross-split lesion leakage,
- every image belongs to exactly one split,
- dataloader can load one batch per split,
- generated report asset paths are valid.

These checks are not optional; this sprint is about trust in the measurement base.

### 6. Artifact and Report Notes

Produce lightweight, report-ready outputs:

- class distribution table,
- split distribution table,
- leakage audit summary,
- missing image report if applicable,
- class distribution figure,
- split distribution figure,
- sprint report note.

The sprint report note should be written immediately after verification, not postponed to final report writing.

### 7. Local/Drive/Colab Flow

Document how large files move:

- Git tracks code, configs, docs, and lightweight placeholders.
- Local machine can hold raw HAM10000 files under ignored `data/`.
- Google Drive folder: `dl-final-artifact`.
- Colab should mount Drive and use `dl-final-artifact` for raw data, artifacts, caches, checkpoints, and run outputs.

No Colab notebook implementation is required in this plan, but Sprint 1 should leave path conventions clear enough that a thin runner can be added safely.

## Affected Files

Planned or likely files for Sprint 1 implementation:

```text
configs/dataset/selected_dataset.yaml
configs/default.yaml
docs/DATASET_AUDIT.md
docs/DECISIONS.md
docs/COMMANDS.md
docs/EXPERIMENT_REGISTRY.md
docs/report_notes/sprint1_dataset_split.md
src/
scripts/
tests/
data/splits/
artifacts/report_assets/
```

Exact source file names should be chosen during implementation in sympathy with the repo structure. The plan does not require copying the old `dl-assignment` package names verbatim.

## Verification Gates

Sprint 1 is complete only if all required gates pass:

- Dataset config parses.
- Metadata audit runs against the real local HAM10000 files.
- Required metadata columns are present.
- Fixed seven-class label mapping is validated.
- Every metadata row resolves to an image file, or missing files are explicitly reported and accepted.
- Split files are generated from audited metadata.
- Split generation is deterministic with the fixed seed.
- Every image appears in exactly one split.
- Cross-split lesion leakage is zero:
  - train vs validation,
  - train vs test,
  - validation vs test.
- Class distribution and split distribution are exported.
- Rare-class representation in validation/test is checked and documented.
- Dataloader smoke test loads one batch from train, validation, and test.
- No raw images, metadata CSV, generated splits, or report assets are accidentally staged for Git unless explicitly intended as lightweight placeholders.
- `docs/DATASET_AUDIT.md` includes source, class counts, split summary, leakage result, and non-clinical framing.
- `docs/report_notes/sprint1_dataset_split.md` summarizes what was done, what was fixed, and what later experiments can rely on.

## Planned Verification Commands

Exact commands may change during implementation, but Sprint 1 should end with commands similar to:

```bash
python -m pytest tests/test_dataset_sprint1.py
python scripts/audit_ham10000.py --config configs/dataset/selected_dataset.yaml
python scripts/make_lesion_split.py --config configs/dataset/selected_dataset.yaml
python scripts/smoke_dataloader.py --config configs/dataset/selected_dataset.yaml
```

If the repo adopts `uv`, these should be recorded with `uv run ...` in `docs/COMMANDS.md`.

## Expected Local Outputs

These outputs are generated artifacts and should generally remain Git-ignored:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
artifacts/report_assets/tables/class_distribution.csv
artifacts/report_assets/tables/split_class_distribution.csv
artifacts/report_assets/tables/lesion_leakage_audit.csv
artifacts/report_assets/figures/class_distribution.png
artifacts/report_assets/figures/split_class_distribution.png
```

If small final figures/tables are later copied into `reports/`, that should be a conscious report-asset decision.

## Interim Sprint Report

Create:

```text
docs/report_notes/sprint1_dataset_split.md
```

Minimum content:

```text
Question:
  Can HAM10000 be prepared as a leakage-safe 7-class benchmark split?

Recipe:
  Metadata/image audit, fixed label mapping, lesion-aware 70/15/15 split, leakage verification.

Fixed controls:
  Dataset version, label mapping, split seed, lesion grouping rule, primary metric policy.

Result:
  Fill after implementation with image count, lesion count, class distribution, split counts, leakage result.

Interpretation:
  Explain whether the split is trustworthy enough for transformer representation and fusion comparisons.

Evidence strength:
  List generated audit tables, split files, and smoke-test result.

Report decision:
  State whether this split becomes the canonical split for all following sprints.
```

Draft report sentence:

> HAM10000 metadata and image paths were validated, and a fixed lesion-aware train/validation/test split was created before model training. All images belonging to the same lesion were constrained to a single split, preventing cross-split lesion leakage in the benchmark evaluation.

Turkish report sentence:

> HAM10000 metadata ve görüntü yolları model eğitiminden önce doğrulanmış; aynı lezyona ait görüntülerin farklı splitlere sızmasını engelleyen sabit bir lesion-aware train/validation/test ayrımı oluşturulmuştur.

## Risks and Fallbacks

| Risk | Fallback |
|---|---|
| `lesion_id` grouping makes minority-class validation/test counts unstable. | Keep lesion grouping; document class-ratio compromise and consider rare-class preservation constraints. |
| Exact stratified grouped split is infeasible. | Use best-effort grouped stratification; if needed, grouped random fallback with explicit warning and decision note. |
| Metadata/images are missing locally. | Fail before split generation; document required local/Drive paths instead of creating dummy outputs. |
| Image path structure differs from old `dl-assignment`. | Isolate path resolution in dataset code and config; do not hardcode old repo paths. |
| Large artifacts are accidentally staged. | Run `git status --ignored` and keep `.gitignore` rules strict. |
| Sprint expands into model implementation. | Stop at dataset/split/dataloader foundation; move model work to Sprint 2. |

## Decision Records To Add During Sprint

Likely `docs/DECISIONS.md` entries:

- canonical HAM10000 local/source version,
- canonical lesion-aware split policy,
- split seed and target ratio,
- handling of imperfect rare-class stratification,
- metadata fields excluded from model input,
- Drive root: `dl-final-artifact`,
- Sprint 1 completion criteria.

## Completion Definition

Sprint 1 can move from `docs/exec-plans/active/` to `docs/exec-plans/completed/` only after:

1. The audit and split commands have run against the real local dataset.
2. Verification gates have passed or documented exceptions have been accepted.
3. `docs/DATASET_AUDIT.md` and `docs/report_notes/sprint1_dataset_split.md` are filled with actual results.
4. `docs/COMMANDS.md` contains reproducible Sprint 1 commands.
5. `docs/DECISIONS.md` records the split and artifact-flow decisions.
6. Git status confirms no large raw/generated files are staged.

## First Implementation Step After This Plan

Start by implementing the smallest dataset-audit path:

```text
config parse -> metadata read -> required column check -> label mapping check -> image path audit
```

Only after that passes should split generation be implemented.

## Final Outcome

Sprint 1 was implemented and verified against the real local HAM10000 dataset.

Completed:

- Lightweight `dl_final` data package for config loading, HAM10000 metadata audit, lesion-aware splitting, and split smoke checks.
- Audit script: `scripts/audit_ham10000.py`.
- Split script: `scripts/make_lesion_split.py`.
- Smoke script: `scripts/smoke_dataloader.py`.
- Unit tests: `tests/test_dataset_sprint1.py`.
- Dataset audit documentation: `docs/DATASET_AUDIT.md`.
- Sprint report note: `docs/report_notes/sprint1_dataset_split.md`.
- Reproducible command notes: `docs/COMMANDS.md`.

Verified:

```bash
PYTHONPATH=src /Users/arcustin2/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m unittest tests/test_dataset_sprint1.py
PYTHONPATH=src /Users/arcustin2/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/audit_ham10000.py --config configs/dataset/selected_dataset.yaml
PYTHONPATH=src /Users/arcustin2/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/make_lesion_split.py --config configs/dataset/selected_dataset.yaml
PYTHONPATH=src /Users/arcustin2/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/smoke_dataloader.py --config configs/dataset/selected_dataset.yaml --max-samples 8
```

Result summary:

- 10,015 metadata rows verified.
- 10,015 unique image IDs.
- 7,470 unique lesion IDs.
- 0 duplicate image IDs.
- 0 missing images.
- 0 missing labels.
- 0 missing lesion IDs.
- Canonical split:
  - train: 7,008 images, 5,233 lesions
  - validation: 1,504 images, 1,117 lesions
  - test: 1,503 images, 1,120 lesions
- All seven classes are present in every split.
- Cross-split lesion leakage is 0 for train/validation, train/test, and validation/test.

Generated local outputs remain intentionally Git-ignored:

```text
data/processed/ham10000_audited_metadata.csv
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
artifacts/logs/dataset_audit.json
artifacts/logs/split_manifest.json
artifacts/logs/dataloader_smoke.json
artifacts/report_assets/tables/*.csv
artifacts/report_assets/figures/*.svg
```
