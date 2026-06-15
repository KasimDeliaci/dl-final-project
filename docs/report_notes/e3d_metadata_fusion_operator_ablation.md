# Metadata Fusion Operator Ablation

This note supports the final report section on lightweight metadata-conditioned fusion over fixed
fine-tuned transformer features. It is a validation-only diagnostic and should not be framed as a
clinical diagnosis or deployment claim.

## Question

Does metadata-conditioned fusion improve over raw metadata concatenation for fixed fine-tuned ViT,
Swin, and BEiT feature caches?

## Protocol

- Split: canonical lesion-aware train/validation split.
- Test usage: not used.
- Image features: fixed fine-tuned `vit_b16`, `swin_tiny`, and `beit_base` feature caches.
- Metadata fields: `age`, `sex`, `localization` only.
- Excluded fields: `dx`, `dx_type`, `dataset`, `image_id`, `sample_id`, `lesion_id`, `image_path`.
- Image preprocessing: train-only `StandardScaler` per backbone block.
- Metadata preprocessing: train-only age median imputation, train-only age scaling, and train-only
  categorical vocabularies.
- Seeds: `7`, `13`, `42`, `101`, `202`.
- Selection metric: validation macro-F1.
- Test split: not loaded or transformed.

## Operator Definitions

`triple_metadata_film` uses a metadata MLP to generate bounded scale/shift terms for the concatenated
image feature vector. The modulation is conservative: image features are adjusted with a small
bounded affine transform before the metadata vector is appended to the classifier input.

`triple_metadata_gated_backbone` uses a metadata MLP to produce sample-level gates for ViT, Swin,
and BEiT feature blocks. The gated feature blocks are concatenated with the metadata vector before
classification.

`triple_metadata_two_branch` processes image features and metadata in separate MLP branches, then
fuses the hidden representations before the classifier head.

## Why This Is Still Fusion

E3d does not replace the transformer fusion story with metadata. The dominant signal still comes
from the fine-tuned ViT, Swin, and BEiT feature vectors. Metadata-only performance is weak
(`0.2202 ± 0.0077` validation macro-F1), so metadata alone is not a competitive classifier under
this protocol.

The role of metadata in E3d is conditional fusion:

```text
fine-tuned ViT features
fine-tuned Swin features        -> fused image representation -> metadata-conditioned classifier
fine-tuned BEiT features
age / sex / localization        -> conditioning signal
```

This distinction matters for report language. The result should be described as a fine-tuned
transformer feature fusion result with structured metadata conditioning, not as a tabular-metadata
model.

## FiLM Explanation

FiLM means Feature-wise Linear Modulation. In this experiment, the metadata branch predicts a small
feature-wise adjustment for the fused image representation:

```text
metadata -> small MLP -> gamma, beta
conditioned_image = image_features * (1 + 0.1 * gamma) + 0.1 * beta
classifier_input = concat(conditioned_image, metadata)
```

The `0.1` factor bounds the modulation so that metadata cannot overwrite the image representation.
This makes FiLM a conservative interaction operator: metadata can slightly emphasize or suppress
feature dimensions before classification, but the final decision remains image-feature dominated.

## Literature Context

Multimodal skin-lesion classification literature motivates testing metadata interaction operators
rather than only raw concatenation. Frontiers 2025 compares simple concatenation, weighted
concatenation, self-attention, and cross-attention fusion strategies on HAM10000. CVPRW 2025 reports
that age, sex, and anatomical site metadata can improve class separability relative to image-only
models. MetaNet-style metadata fusion uses metadata to control the importance of visual feature
channels, which directly motivates the gated operator in this ablation.

Relevant references:

- Frontiers 2025 multimodal fusion comparison:
  https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1608837/full
- CVPRW 2025 clinical metadata study:
  https://openaccess.thecvf.com/content/CVPR2025W/MULA2025/html/Ahammed_Skin_Lesion_Classification_Using_Dermoscopic_Images_and_Clinical_Metadata_Insights_CVPRW_2025_paper.html
- MetaNet metadata-image fusion:
  https://jiaxinzhuang.github.io/pdf/metanet.pdf

## Multi-Seed Results

| Condition | Seeds | Mean validation macro-F1 | Std | Min | Max | Mean accuracy | Mean weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Metadata FiLM | 5 | `0.7358` | `0.0152` | `0.7158` | `0.7529` | `0.8390` | `0.8418` |
| Metadata-gated backbone | 5 | `0.7347` | `0.0112` | `0.7189` | `0.7453` | `0.8356` | `0.8390` |
| Metadata two-branch | 5 | `0.7328` | `0.0103` | `0.7207` | `0.7450` | `0.8323` | `0.8366` |

Controls:

| Condition | Mean validation macro-F1 | Std | Min | Max |
|---|---:|---:|---:|---:|
| E3c raw concat + metadata | `0.7278` | `0.0058` | `0.7213` | `0.7363` |
| E3b image-only fine-tuned triple concat | `0.7246` | `0.0143` | `0.7032` | `0.7413` |
| E3c metadata-only MLP | `0.2202` | `0.0077` | `0.2093` | `0.2297` |

All three metadata-conditioned operators improved mean validation macro-F1 over raw metadata
concatenation. The largest mean score came from bounded FiLM-style conditioning (`0.7358`), while
metadata-gated backbone fusion had a similar mean and somewhat lower standard deviation.

The improvement should be read relative to both controls. Compared with E3b image-only fine-tuned
triple concat, FiLM increased mean validation macro-F1 by `+0.0112`. Compared with E3c raw concat +
metadata, FiLM increased mean validation macro-F1 by `+0.0080`. The result is meaningful as a
validation-stage ablation, but it is still not a final test-set conclusion.

## Per-Class Behavior vs E3c Raw Concat

Metadata FiLM:

| Label | Support | E3c raw concat F1 | E3d F1 | Delta |
|---|---:|---:|---:|---:|
| `akiec` | 49 | `0.6820` | `0.6708` | `-0.0112` |
| `bcc` | 77 | `0.6957` | `0.7178` | `+0.0221` |
| `bkl` | 165 | `0.7076` | `0.7117` | `+0.0042` |
| `df` | 18 | `0.7599` | `0.7702` | `+0.0103` |
| `mel` | 167 | `0.5696` | `0.6050` | `+0.0354` |
| `nv` | 1006 | `0.9159` | `0.9235` | `+0.0076` |
| `vasc` | 22 | `0.7639` | `0.7515` | `-0.0123` |

Metadata-gated backbone:

| Label | Support | E3c raw concat F1 | E3d F1 | Delta |
|---|---:|---:|---:|---:|
| `akiec` | 49 | `0.6820` | `0.7121` | `+0.0301` |
| `bcc` | 77 | `0.6957` | `0.7222` | `+0.0265` |
| `bkl` | 165 | `0.7076` | `0.7102` | `+0.0026` |
| `df` | 18 | `0.7599` | `0.7156` | `-0.0442` |
| `mel` | 167 | `0.5696` | `0.5899` | `+0.0203` |
| `nv` | 1006 | `0.9159` | `0.9202` | `+0.0043` |
| `vasc` | 22 | `0.7639` | `0.7725` | `+0.0087` |

Metadata two-branch:

| Label | Support | E3c raw concat F1 | E3d F1 | Delta |
|---|---:|---:|---:|---:|
| `akiec` | 49 | `0.6820` | `0.6825` | `+0.0005` |
| `bcc` | 77 | `0.6957` | `0.7189` | `+0.0232` |
| `bkl` | 165 | `0.7076` | `0.7144` | `+0.0069` |
| `df` | 18 | `0.7599` | `0.7852` | `+0.0254` |
| `mel` | 167 | `0.5696` | `0.5850` | `+0.0155` |
| `nv` | 1006 | `0.9159` | `0.9183` | `+0.0024` |
| `vasc` | 22 | `0.7639` | `0.7252` | `-0.0387` |

## Gate Diagnostic

The metadata-gated backbone model produced the following validation-wide mean gates over seeds:

| Backbone | Mean gate |
|---|---:|
| ViT | `0.5674` |
| Swin | `0.4461` |
| BEiT | `0.2944` |

These gate values are model internals. They should not be interpreted as a direct backbone quality
ranking. They only indicate how this specific metadata-conditioned classifier scaled feature blocks
under this training setup.

## Interpretation

E3d provides stronger validation evidence than E3c that metadata can be useful when it interacts
with image representations rather than only being appended as raw tabular features. The result is
still modest and class-dependent, but all three lightweight operators exceeded the raw concat +
metadata control in mean validation macro-F1.

FiLM is the highest-mean operator and is notable for improving `mel` F1 relative to raw concat. The
gated backbone operator is nearly tied in mean macro-F1 and improves `akiec`, `bcc`, `mel`, and
`vasc`, but it reduces `df`. The two-branch operator improves `df` and `mel`, but reduces `vasc`.
Therefore the correct report language is not that advanced metadata fusion uniformly improves all
classes, but that metadata-conditioned fusion can improve validation macro-F1 with class-dependent
tradeoffs.

The clean final-report claim is:

> The best validation-stage performance was obtained by combining fine-tuned ViT, Swin, and BEiT
> features and using structured metadata as a lightweight conditioning signal over the fused image
> representation.

The claim should not be:

> Metadata alone was sufficient, or metadata-conditioned fusion uniformly improved every class.

The first statement matches the controls and per-class behavior. The second would ignore the weak
metadata-only result and the class-dependent F1 deltas.

## Limitations

- Validation-only diagnostic; test set was not used.
- Fixed cached features; no end-to-end multimodal transformer fine-tuning.
- Metadata may encode benchmark-specific correlations.
- Operator differences are small enough that they should be discussed with seed variability.
- Low-support classes such as `df` and `vasc` have high per-class F1 uncertainty.
- No clinical diagnosis or deployment claim.

## Generated Artifacts

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

## Report-Ready Turkish Summary

Raw metadata concatenation sonrasında, metadata'nın image representation'ı doğrudan condition edip
etmediğini ölçmek için üç lightweight fusion operator'ı denenmiştir. Beş seed ortalamasında bounded
FiLM-style metadata conditioning `0.7358 ± 0.0152`, metadata-gated backbone fusion
`0.7347 ± 0.0112`, two-branch image/metadata fusion ise `0.7328 ± 0.0103` validation macro-F1
üretmiştir. Üç koşul da E3c raw concat + metadata kontrolünü (`0.7278 ± 0.0058`) ve image-only
fine-tuned triple concat kontrolünü (`0.7246 ± 0.0143`) aşmıştır. Bu sonuç, HAM10000 benchmark
bağlamında structured metadata'nın yalnızca input'a eklenmek yerine fine-tuned transformer
feature'larını condition ettiğinde daha güçlü validation sinyali verebildiğini göstermektedir. Ancak
per-class etkiler tek yönlü değildir; bu nedenle bulgu sınıf-bağımlı ve validation-only diagnostic
olarak yorumlanmalıdır.

Final rapor dilinde bu sonuç, "metadata modeli" olarak değil, "fine-tuned transformer feature fusion
üzerinde metadata-conditioned fusion" olarak sunulmalıdır. Ana ayırt edici sinyal ViT, Swin ve BEiT
temsillerinin birleşiminden gelmektedir; yaş, cinsiyet ve lokalizasyon bilgileri bu birleşik
representation'ı hafifçe modüle ederek ek katkı sağlamıştır. Metadata-only MLP'nin düşük sonucu bu
ayrımı destekler.
