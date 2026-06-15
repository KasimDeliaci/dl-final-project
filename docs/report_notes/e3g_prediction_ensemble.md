# E3g Prediction-Level Ensemble Diagnostic

Bu not, final raporda validation-only prediction ensemble sonucunu temkinli ve tekrar uretilebilir
sekilde kullanmak icin hazirlanmistir. E3g yeni model egitimi veya fine-tuning yapmaz; mevcut
validation prediction dump'lari uzerinde probability averaging uygular. Test split yuklenmemis ve
model secimi icin kullanilmamistir.

## Question

E3d ve E3f sonucunda en guclu validation adaylari birbirine cok yakindi, fakat per-class davranislari
ayni degildi:

- E3d all-fine-tuned FiLM: `0.7358 ± 0.0152`
- E3d all-fine-tuned gated: `0.7347 ± 0.0112`
- E3f mixed frozen-ViT + fine-tuned Swin/BEiT gated: `0.7361 ± 0.0100`

E3g, bu family'lerin validation probability output'larini birlestirerek seed variance ve hata
ortusmesini azaltip azaltmadigini test eder.

## Protocol

Input prediction families:

| Family | Source runs | Seeds |
|---|---|---|
| `e3d_film` | E3d all-fine-tuned FiLM | `7,13,42,101,202` |
| `e3d_gated` | E3d all-fine-tuned metadata-gated | `7,13,42,101,202` |
| `e3f_gated` | E3f mixed frozen ViT + fine-tuned Swin/BEiT metadata-gated | `7,13,42,101,202` |

Each family is first averaged across its five seeds. Family-level predictions are then combined with
fixed probability weights. Primary results use equal weights only. Weighted grids are reported as
diagnostic because their weights are selected from validation results.

Alignment checks:

- all prediction dumps have `1504` validation rows,
- all seven `prob_*` columns exist,
- probabilities are finite and sum to `1.0` per row,
- `sample_id`, `image_id`, `lesion_id`, `split`, and `true_label` align exactly,
- no test rows are loaded.

## Results

Primary equal-weight ensembles:

| Ensemble | Weighting | Validation macro-F1 | Accuracy | Weighted-F1 |
|---|---|---:|---:|---:|
| `top3_family_equal` | E3d FiLM + E3d gated + E3f gated, `1/3` each | `0.7665` | `0.8564` | `0.8576` |
| `e3d_film_plus_e3f_gated_equal` | E3d FiLM + E3f gated, `0.5/0.5` | `0.7587` | `0.8517` | `0.8533` |
| `e3d_film_seed_avg` | E3d FiLM seeds only | `0.7537` | `0.8491` | `0.8505` |
| `e3d_gated_plus_e3f_gated_equal` | E3d gated + E3f gated, `0.5/0.5` | `0.7463` | `0.8464` | `0.8481` |
| `e3d_gated_seed_avg` | E3d gated seeds only | `0.7413` | `0.8451` | `0.8468` |
| `e3f_gated_seed_avg` | E3f gated seeds only | `0.7393` | `0.8391` | `0.8420` |

Weighted diagnostic:

| Ensemble | Weighting | Validation macro-F1 | Accuracy | Weighted-F1 |
|---|---|---:|---:|---:|
| `top3_grid_film_0p5_gated_0p25_e3f_0p25` | E3d FiLM `0.50`, E3d gated `0.25`, E3f gated `0.25` | `0.7702` | `0.8557` | `0.8570` |
| `top2_grid_e3d_film_0p75_e3f_gated_0p25` | E3d FiLM `0.75`, E3f gated `0.25` | `0.7655` | `0.8590` | `0.8605` |

The primary equal-weight `top3_family_equal` ensemble is the strongest low-overfit E3g result. It is
substantially above the previous E3d/E3f individual-run means. The weighted grid diagnostic reaches
`0.7702`, but because it uses validation-label pressure for weight choice, it should be discussed as
exploratory evidence rather than the main selection result.

## Per-Class Behavior

Representative primary ensemble per-class F1:

| Label | `top3_family_equal` F1 | Support |
|---|---:|---:|
| `akiec` | `0.7527` | `49` |
| `bcc` | `0.7712` | `77` |
| `bkl` | `0.7251` | `165` |
| `df` | `0.7805` | `18` |
| `nv` | `0.9324` | `1006` |
| `mel` | `0.6279` | `167` |
| `vasc` | `0.7755` | `22` |

The ensemble improves macro-F1 by lifting several minority and medium-support classes at the same
time. Unlike E3f mixed gated, the gain is not driven only by `df`; `akiec`, `bcc`, `mel`, and `vasc`
also become stronger than several single-family averages. Low-support classes still require support
counts in the report.

## Error Overlap

Family error overlap is high but not complete:

| Family pair | Error Jaccard | Left errors | Right errors | Both wrong |
|---|---:|---:|---:|---:|
| E3d FiLM vs E3d gated | `0.7625` | `227` | `233` | `199` |
| E3d FiLM vs E3f gated | `0.6690` | `227` | `242` | `188` |
| E3d gated vs E3f gated | `0.7025` | `233` | `242` | `196` |

The lower overlap between E3d FiLM and E3f gated supports the intuition that source/operator
diversity helps the ensemble. The E3d FiLM and E3d gated pair is more redundant.

## Interpretation

E3g is currently the strongest validation-only result in the project. The primary equal-weight
family ensemble reaches `0.7665` validation macro-F1 without additional training, new data, test-set
access, or class-specific threshold tuning. This is consistent with skin lesion challenge practice,
where ensembles and prediction averaging commonly improve robustness.

The correct report language is:

- "prediction-level ensembling substantially improved validation macro-F1";
- "the result remains validation-only";
- "the weighted-grid result is diagnostic because it uses validation labels to choose weights";
- "test performance must be checked only once in the final audit after model selection is frozen."

Do not present E3g as a clinical conclusion. It is a HAM10000 benchmark dermoscopic image
classification validation result.

## Report-Ready Turkish Wording

Mevcut en guclu validation adaylari arasinda probability-level ensemble uygulanmistir. Bu asamada
yeni model egitilmemis, transformer feature'lari yeniden cikarilmamis ve test split
kullanilmamistir. E3d FiLM, E3d metadata-gated ve E3f mixed metadata-gated family'lerinin beser seed
prediction dump'lari once kendi iclerinde ortalanmis, ardindan family-level probability averaging
uygulanmistir.

En guclu primary sonuc, E3d FiLM, E3d gated ve E3f gated family'lerini esit agirlikla birlestiren
`top3_family_equal` ensemble olmustur. Bu yapi `0.7665` validation macro-F1, `0.8564` accuracy ve
`0.8576` weighted-F1 uretmistir. Bu deger, onceki E3d/E3f metadata-conditioned tek-family
ortalamalarinin belirgin uzerindedir. Kucuk weighted-grid diagnostic ise E3d FiLM'e `0.50`, E3d
gated'a `0.25`, E3f gated'a `0.25` agirlik verdiginde `0.7702` macro-F1 uretmistir; ancak bu agirlik
secimi validation sonuclarina dayandigi icin ana sonuc olarak degil, exploratory diagnostic olarak
raporlanmalidir.

Per-class analiz, ensemble kazancinin yalnizca tek bir dusuk destekli siniftan gelmedigini
gostermektedir. `top3_family_equal` ensemble `akiec`, `bcc`, `mel` ve `vasc` gibi macro-F1'i
etkileyen siniflarda da guclu degerler uretmistir. Buna ragmen `df` ve `vasc` gibi dusuk destekli
siniflar icin support count mutlaka raporda gorunur tutulmalidir.

Bu bulgu, prediction-level ensemble'in HAM10000 validation splitinde representation ve operator
cesitliliginden fayda sagladigini gosterir. Sonuc test set uzerinde henuz dogrulanmamistir; final
audit asamasinda tek seferlik test degerlendirmesi yapilmadan genelleme veya klinik performans
iddiasi kurulmamali.

## Artifacts

```text
artifacts/runs/e3g_prediction_ensemble/
artifacts/report_assets/tables/e3g_prediction_ensemble_results.csv
artifacts/report_assets/tables/e3g_prediction_ensemble_per_class_metrics.csv
artifacts/report_assets/tables/e3g_prediction_ensemble_vs_controls.csv
artifacts/report_assets/tables/e3g_prediction_ensemble_error_overlap.csv
artifacts/report_assets/tables/e3g_prediction_ensemble_corrected_broken.csv
artifacts/report_assets/figures/e3g_prediction_ensemble_macro_f1.png
artifacts/report_assets/figures/e3g_prediction_ensemble_per_class_f1.png
artifacts/report_assets/figures/e3g_prediction_ensemble_error_overlap.png
artifacts/report_assets/figures/e3g_top3_family_equal_confusion_matrix.png
artifacts/report_assets/figures/e3g_top3_grid_film_0p5_gated_0p25_e3f_0p25_confusion_matrix.png
```
