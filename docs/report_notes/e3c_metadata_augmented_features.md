# Metadata-Augmented Fine-Tuned Feature Diagnostic

This note supports the final report section on structured metadata fusion for HAM10000 benchmark
dermoscopic image classification. It should not be written as a clinical diagnosis or deployment
claim.

## Question

Does adding HAM10000 benchmark metadata (`age`, `sex`, `localization`) to cached fine-tuned
transformer features improve validation macro-F1 over image-only feature transfer?

## Protocol

- Split: canonical lesion-aware train/validation split.
- Test usage: not used.
- Image features: fixed fine-tuned transformer caches from E3.
- Metadata fields: `age`, `sex`, `localization` only.
- Excluded fields: `dx`, `dx_type`, `dataset`, `image_id`, `sample_id`, `lesion_id`, `image_path`.
- Metadata preprocessing: train-only age median imputation, train-only age scaling, train-only
  categorical vocabularies for `sex` and `localization`.
- Seeds: `7`, `13`, `42`, `101`, `202`.
- Selection metric: validation macro-F1.
- Classifier: cached-feature MLP with the same class-weighted validation-selected discipline used
  for E3b.

The metadata vector has 19 dimensions:

- 1 scaled age feature,
- 3 train-fitted `sex` categories,
- 15 train-fitted `localization` categories.

## Literature Context

Multimodal skin-lesion classification work commonly combines image features with structured
clinical or benchmark metadata. Recent surveys and benchmark papers describe age, sex, and
anatomical site/location as common metadata inputs, often processed through a small dense network
or encoded as a vector before concatenation with image embeddings. In this project, the same idea is
used only as a controlled benchmark diagnostic over fixed transformer features.

Relevant references:

- Frontiers 2025 multimodal skin-lesion review/benchmark:
  https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608837/full
- CVPRW 2025 multimodal metadata study:
  https://openaccess.thecvf.com/content/CVPR2025W/MULA2025/papers/Ahammed_Skin_Lesion_Classification_Using_Dermoscopic_Images_and_Clinical_Metadata_Insights_CVPRW_2025_paper.pdf

## Multi-Seed Results

| Condition | Seeds | Mean validation macro-F1 | Std | Min | Max | Mean accuracy | Mean weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fine-tuned ViT+Swin+BEiT concat + metadata | 5 | `0.7278` | `0.0058` | `0.7213` | `0.7363` | `0.8278` | `0.8316` |
| Fine-tuned ViT+Swin concat + metadata | 5 | `0.7230` | `0.0138` | `0.7082` | `0.7376` | `0.8257` | `0.8298` |
| Metadata-only MLP | 5 | `0.2202` | `0.0077` | `0.2093` | `0.2297` | `0.3983` | `0.4738` |

Image-only E3b controls:

| Condition | Mean validation macro-F1 | Std | Min | Max |
|---|---:|---:|---:|---:|
| Fine-tuned ViT+Swin+BEiT concat | `0.7246` | `0.0143` | `0.7032` | `0.7413` |
| Fine-tuned ViT+Swin concat | `0.7160` | `0.0085` | `0.7070` | `0.7291` |
| Fine-tuned ViT single | `0.6801` | `0.0084` | `0.6722` | `0.6906` |

## Per-Class Behavior

Fine-tuned ViT+Swin+BEiT concat + metadata vs image-only triple concat:

| Label | Support | Image-only mean F1 | Metadata mean F1 | Delta |
|---|---:|---:|---:|---:|
| `akiec` | 49 | `0.6538` | `0.6820` | `+0.0282` |
| `bcc` | 77 | `0.6930` | `0.6957` | `+0.0028` |
| `bkl` | 165 | `0.7238` | `0.7076` | `-0.0163` |
| `df` | 18 | `0.7566` | `0.7599` | `+0.0032` |
| `mel` | 167 | `0.5714` | `0.5696` | `-0.0018` |
| `nv` | 1006 | `0.9130` | `0.9159` | `+0.0030` |
| `vasc` | 22 | `0.7605` | `0.7639` | `+0.0034` |

Fine-tuned ViT+Swin concat + metadata vs image-only pair concat:

| Label | Support | Image-only mean F1 | Metadata mean F1 | Delta |
|---|---:|---:|---:|---:|
| `akiec` | 49 | `0.6789` | `0.6830` | `+0.0041` |
| `bcc` | 77 | `0.7144` | `0.7157` | `+0.0013` |
| `bkl` | 165 | `0.6941` | `0.6967` | `+0.0026` |
| `df` | 18 | `0.7015` | `0.7254` | `+0.0239` |
| `mel` | 167 | `0.5666` | `0.5637` | `-0.0029` |
| `nv` | 1006 | `0.9091` | `0.9151` | `+0.0060` |
| `vasc` | 22 | `0.7474` | `0.7611` | `+0.0138` |

## Interpretation

Adding structured metadata produced a small validation macro-F1 gain over image-only fine-tuned
feature fusion in the multi-seed diagnostic. The strongest condition remained the triple
fine-tuned transformer concat configuration, now with metadata appended (`0.7278 ± 0.0058`).

The result is positive but modest. The metadata-only classifier performed poorly, so the gain is not
evidence that metadata alone carries sufficient class signal. Instead, the result suggests that
structured benchmark metadata may provide small complementary information when combined with
fine-tuned transformer representations.

Per-class behavior is mixed. Triple concat + metadata improved `akiec` most clearly, while `bkl`
decreased and `mel` was essentially unchanged. Pair concat + metadata improved `df`, `vasc`, and
`nv` modestly, with a small decrease for `mel`. This should be reported as class-dependent behavior,
not a uniform improvement.

## Limitations

- Validation-only diagnostic; test set was not used.
- Metadata may encode benchmark-specific correlations and may not generalize outside this split.
- Age, sex, and localization are useful structured inputs for a benchmark experiment, but this does
  not establish clinical diagnostic utility.
- The observed mean macro-F1 gains are small relative to seed variability across nearby candidates.

## Generated Artifacts

```text
artifacts/runs/*_e3c_metadata_*_seed*/
artifacts/report_assets/tables/e3c_metadata_augmented_results.csv
artifacts/report_assets/tables/e3c_metadata_augmented_summary.csv
artifacts/report_assets/tables/e3c_metadata_augmented_per_class_metrics.csv
artifacts/report_assets/tables/e3c_metadata_per_class_delta_vs_image_only.csv
artifacts/report_assets/tables/e3c_metadata_vs_image_only_validation.csv
artifacts/report_assets/figures/e3c_metadata_augmented_macro_f1.png
```

## Report-Ready Turkish Summary

Fine-tuned transformer feature'larına yaş, cinsiyet ve lezyon lokalizasyonu metadata'sı eklemek,
validation macro-F1 üzerinde küçük fakat tutarlı bir artış sağlamıştır. Beş seed ortalamasında
ViT+Swin+BEiT concat + metadata koşulu `0.7278 ± 0.0058` macro-F1 üretmiş; image-only fine-tuned
triple concat kontrolü `0.7246 ± 0.0143` seviyesinde kalmıştır. ViT+Swin concat + metadata koşulu
da image-only pair concat kontrolünü (`0.7160 ± 0.0085`) aşarak `0.7230 ± 0.0138` değerine
ulaşmıştır. Buna karşılık metadata-only MLP'nin düşük sonucu (`0.2202 ± 0.0077`), metadata'nın tek
başına yeterli olmadığını göstermektedir. Bu nedenle sonuç, HAM10000 benchmark bağlamında structured
metadata'nın fine-tuned transformer representation'lara küçük ve sınıf-bağımlı tamamlayıcı sinyal
ekleyebildiği şeklinde temkinli yorumlanmalıdır.
