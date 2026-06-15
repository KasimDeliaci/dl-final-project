# E3f Mixed Frozen ViT + Fine-Tuned Swin/BEiT Adaptation

Bu not, backbone-level adaptation ablation sonucunu final raporda kullanmak icin hazirlanmistir.
Sonuclar validation-only degerlendirmedir; test split yuklenmemis ve model secimi icin
kullanilmamistir.

## Question

ViT single-backbone fine-tuning frozen ViT baseline'ini gecemedi, buna karsilik Swin ve BEiT
fine-tuning single-backbone sonuclarinda iyilesme sagladi. E3f su soruyu test eder:

> ViT frozen feature olarak korunup Swin ve BEiT fine-tuned feature olarak kullanilirsa, all-fine-tuned
> triple source'a gore daha iyi veya daha stabil validation macro-F1 elde edilir mi?

## Protocol

Yeni transformer fine-tuning yapilmadi. Mevcut cache'lerden mixed feature source olusturuldu:

| Backbone | Feature cache source |
|---|---|
| `vit_b16` | frozen |
| `swin_tiny` | fine-tuned |
| `beit_base` | fine-tuned |

Mixed source:

```text
artifacts/features/ham10000/frozen_vit_finetuned_swin_beit/
```

Validation-only downstream kosullari seeds `7`, `13`, `42`, `101`, `202` uzerinde calistirildi:

- image-only concat,
- metadata-gated backbone fusion,
- bounded FiLM-style metadata conditioning.

All preprocessing was fit on the train split only. Prediction dumps contain `1504` validation rows
per run. The test split was not loaded.

## Results

| Condition | Seeds | Mean macro-F1 | Std | Min | Max | Mean accuracy | Mean weighted-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Mixed frozen ViT + fine-tuned Swin/BEiT, metadata-gated | `7,13,42,101,202` | `0.7361` | `0.0100` | `0.7260` | `0.7481` | `0.8330` | `0.8368` |
| Mixed frozen ViT + fine-tuned Swin/BEiT, FiLM | `7,13,42,101,202` | `0.7267` | `0.0191` | `0.7016` | `0.7494` | `0.8281` | `0.8341` |
| Mixed frozen ViT + fine-tuned Swin/BEiT, concat | `7,13,42,101,202` | `0.7142` | `0.0126` | `0.6943` | `0.7275` | `0.8210` | `0.8251` |

Key controls:

| Control | Mean macro-F1 |
|---|---:|
| E3d all-fine-tuned FiLM | `0.7358 ± 0.0152` |
| E3d all-fine-tuned gated | `0.7347 ± 0.0112` |
| E3c all-fine-tuned concat + metadata | `0.7278 ± 0.0058` |
| E3b all-fine-tuned triple concat | `0.7246 ± 0.0143` |
| E2b frozen ViT+Swin concat deep_reg | `0.7077 ± 0.0124` |

## Interpretation

E3f gives useful but narrow evidence. Image-only mixed concat (`0.7142 ± 0.0126`) underperformed the
all-fine-tuned triple concat mean (`0.7246 ± 0.0143`). Therefore, simply replacing fine-tuned ViT
with frozen ViT does not improve the image-only fusion representation.

The metadata-gated mixed source reached the highest mean validation macro-F1 among the compared
multi-seed conditions (`0.7361 ± 0.0100`), slightly above E3d all-fine-tuned FiLM (`0.7358 ± 0.0152`)
and all-fine-tuned gated (`0.7347 ± 0.0112`). The margin over E3d FiLM is only about `+0.0003`
macro-F1, so this should be reported as a practical tie rather than a decisive new best model.

The result suggests that keeping ViT frozen can remain competitive when metadata-gated fusion is
used, but it does not show that frozen ViT is generally better than fine-tuned ViT. The source choice
interacts with the fusion operator.

## Per-Class Behavior

Compared with E3d all-fine-tuned gated fusion, E3f mixed gated improved mostly on `df` while reducing
several other classes:

| Label | E3f mixed gated F1 | E3d all-fine-tuned gated F1 | Delta |
|---|---:|---:|---:|
| `akiec` | `0.6537` | `0.7121` | `-0.0584` |
| `bcc` | `0.6939` | `0.7222` | `-0.0282` |
| `bkl` | `0.7056` | `0.7102` | `-0.0046` |
| `df` | `0.8449` | `0.7156` | `+0.1293` |
| `mel` | `0.5913` | `0.5899` | `+0.0014` |
| `nv` | `0.9209` | `0.9202` | `+0.0006` |
| `vasc` | `0.7426` | `0.7725` | `-0.0299` |

Because `df` has only `18` validation examples, the mean macro-F1 advantage should be interpreted
with support counts visible. This is class-level tradeoff evidence, not proof of broad superiority.

## Report-Ready Turkish Wording

Backbone-level adaptation etkisini ayirmak icin ek bir cached-feature ablation yapilmistir. Bu
kontrolde ViT feature'lari frozen cache'ten, Swin ve BEiT feature'lari ise fine-tuned cache'lerden
alinmistir. Boylece ViT'in frozen representation olarak korunmasinin, fine-tuning'den fayda goren
Swin/BEiT representation'lariyla birlikte daha dengeli bir fusion saglayip saglamadigi test
edilmistir.

Image-only concat kosulunda mixed source `0.7142 ± 0.0126` validation macro-F1 uretmis ve
all-fine-tuned triple concat ortalamasinin (`0.7246 ± 0.0143`) altinda kalmistir. Buna karsilik
metadata-gated operator ayni mixed source uzerinde `0.7361 ± 0.0100` validation macro-F1 uretmistir.
Bu deger E3d all-fine-tuned FiLM sonucuna (`0.7358 ± 0.0152`) cok yakindir ve sayisal olarak az
ustundedir. Ancak fark `+0.0003` macro-F1 duzeyinde oldugu icin bu sonuc pratik bir beraberlik
olarak yorumlanmalidir.

Per-class analiz, mixed gated kosulundaki kucuk mean macro-F1 avantajinin ozellikle `df` sinifindaki
artistan kaynaklandigini, buna karsilik `akiec`, `bcc` ve `vasc` tarafinda dusus oldugunu
gostermistir. Bu nedenle E3f sonucu "frozen ViT kesinlikle daha iyi" seklinde degil, "ViT'in frozen
kalmasi metadata-gated fusion ile rekabetci olabilir, ancak etki class-dependent ve validation-only"
seklinde raporlanmalidir.

## Artifacts

```text
artifacts/features/ham10000/frozen_vit_finetuned_swin_beit/
artifacts/report_assets/tables/e3f_mixed_adaptation_results.csv
artifacts/report_assets/tables/e3f_mixed_adaptation_summary.csv
artifacts/report_assets/tables/e3f_mixed_adaptation_per_class_metrics.csv
artifacts/report_assets/tables/e3f_mixed_adaptation_gate_summary.csv
artifacts/report_assets/tables/e3f_mixed_adaptation_vs_controls.csv
artifacts/report_assets/figures/e3f_mixed_adaptation_macro_f1.png
```
