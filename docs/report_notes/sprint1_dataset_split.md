# Sprint 1 Dataset Split Report Note

Question:

Can HAM10000 be prepared as a leakage-safe seven-class benchmark split for transformer feature extraction and fusion experiments?

Recipe:

Metadata/image audit, fixed seven-class label mapping, lesion-aware `70/15/15` split with seed `42`, leakage verification, class distribution export, and image-open smoke check.

Fixed controls:

- Dataset: HAM10000
- Label mapping: `akiec`, `bcc`, `bkl`, `df`, `nv`, `mel`, `vasc`
- Grouping rule: same `lesion_id` cannot cross train/validation/test
- Primary interpretation metric for later experiments: macro-F1
- Test policy: audit only, not model selection

Result:

- Metadata rows: 10,015
- Unique image IDs: 10,015
- Unique lesion IDs: 7,470
- Duplicate image IDs: 0
- Missing images: 0
- Missing labels: 0
- Missing lesion IDs: 0
- Split counts:
  - train: 7,008 images, 5,233 lesions
  - validation: 1,504 images, 1,117 lesions
  - test: 1,503 images, 1,120 lesions
- All seven classes are present in every split.
- Cross-split lesion leakage is zero for train/validation, train/test, and validation/test.

Interpretation:

The split is trustworthy enough to serve as the canonical evaluation base for Sprint 2 frozen transformer features, Sprint 3 feature fusion, Sprint 4 fine-tuning, and Sprint 5 final audit. The dataset remains highly imbalanced, especially because `nv` represents about 66.95% of all images, so later experiments must prioritize macro-F1 and per-class behavior over accuracy.

The most important methodological choice is lesion-level grouping. HAM10000 contains multiple images for some lesions; if images from the same lesion were allowed to appear in both train and validation/test, later transformer comparisons could overstate generalization by recognizing lesion-specific visual patterns rather than learning transferable class-discriminative representations. The zero-overlap leakage audit therefore makes the later backbone and fusion comparisons more defensible.

The split also preserves the seven-class task in every partition. This matters because the minority classes are small: `df` has only 115 total images and `vasc` has 142. The generated validation and test splits still contain all classes, with `df` represented by 18 validation images and 17 test images, and `vasc` represented by 22 validation and 22 test images. These supports are limited, so per-class metrics must be interpreted carefully; a few examples can move minority-class F1 noticeably.

The class distribution remains close to the full dataset distribution across train, validation, and test. This makes the split suitable for comparing Vanilla ViT, Swin Transformer, and DeiT III-Small under the same evaluation protocol. It does not remove the imbalance problem; instead, it exposes it consistently across all later experiments.

Evidence strength:

Generated evidence:

```text
data/processed/ham10000_audited_metadata.csv
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
artifacts/logs/dataset_audit.json
artifacts/logs/split_manifest.json
artifacts/logs/dataloader_smoke.json
artifacts/report_assets/tables/class_distribution.csv
artifacts/report_assets/tables/split_summary.csv
artifacts/report_assets/tables/split_class_distribution.csv
artifacts/report_assets/tables/lesion_leakage_audit.csv
artifacts/report_assets/figures/class_distribution.svg
artifacts/report_assets/figures/split_class_distribution.svg
```

Report decision:

Use this split as the canonical split for all following experiments unless a later explicit decision supersedes it. Do not compare future model results against different split policies as if they are directly equivalent.

Report-ready paragraph:

> Before any model training, the HAM10000 metadata and image paths were audited to establish a fixed seven-class benchmark setup. The audit verified 10,015 images, 10,015 unique image IDs, and 7,470 unique lesion IDs, with no missing images, duplicate image IDs, missing labels, or missing lesion IDs. A lesion-aware train/validation/test split was then created with seed 42, assigning all images from the same lesion to a single split. The resulting split contains 7,008 training images, 1,504 validation images, and 1,503 test images, with all seven classes represented in each split and zero cross-split lesion leakage.

Report-ready methodology note:

> Lesion-aware splitting was used because HAM10000 can contain multiple images of the same lesion. Preventing the same lesion from appearing in both training and evaluation splits reduces leakage risk and makes subsequent transformer backbone and feature-fusion comparisons more reliable. Since the dataset is strongly imbalanced, especially due to the majority `nv` class, later experiments use macro-F1 and per-class metrics as the main interpretation tools rather than relying only on accuracy.

Short presentation sentence:

> HAM10000 metadata and image paths were validated before model training, and a fixed lesion-aware train/validation/test split was created. All images belonging to the same lesion were constrained to a single split, yielding zero cross-split lesion leakage.

Limitations to mention later:

- The split is leakage-safe by `lesion_id`, but it is still a single split; multi-seed or cross-validation would be stronger if time permits.
- Minority-class validation/test support is limited, so per-class improvements for `df`, `vasc`, and `akiec` should be interpreted with support counts visible.
- This setup supports benchmark classification analysis only; it does not justify clinical deployment or diagnosis claims.
