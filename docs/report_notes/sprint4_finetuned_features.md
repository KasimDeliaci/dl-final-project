# Fine-Tuned Transformer Features Report Note

Bu not, final rapordaki "fine-tuned transformer features" ve "frozen vs fine-tuned feature transfer"
bölümlerine malzeme sağlamak için hazırlanmıştır. Akademik raporda iç proje yönetimi dili ana anlatı
olarak kullanılmamalıdır.

Çalışma klinik teşhis iddiası taşımaz. Tüm sonuçlar HAM10000 üzerinde benchmark dermoscopic image
classification bağlamında yorumlanmalıdır.

## Question

Kontrollü partial fine-tuning sonrası çıkarılan transformer feature'ları, frozen feature baseline'a
göre validation macro-F1 ve per-class behavior açısından iyileşme sağlıyor mu?

Alt sorular:

- Fine-tuned single backbones, frozen single baselines üstüne çıkıyor mu?
- Fine-tuned concat fusion, frozen fusion baselines üstüne çıkıyor mu?
- Fine-tuning sonrası BEiT üçüncü backbone olarak faydalı kalıyor mu?
- Fine-tuning gain hangi sınıflarda görülüyor?
- Fine-tuning compute/runtime maliyeti bu kazanımı makul kılıyor mu?

## Fine-Tuning Policy

Forward backbone seti:

```text
vit_b16
swin_tiny
beit_base
```

`deit3_small`, Sprint 2-3 içinde planned/screened baseline olarak kalır; Sprint 4 fine-tuning compute
budget'ına dahil edilmez.

Partial fine-tuning policy:

| Backbone | Trainable region | Temporary head | Feature extraction after selection |
|---|---|---|---|
| Vanilla ViT | last 2 transformer blocks + norm | 7-class head | CLS-token, 768 dim |
| Swin Transformer | last Swin stage + norm | 7-class head | avg pooled final-stage, 768 dim |
| BEiT-Base | last 2 transformer blocks + fc_norm/norm | 7-class head | avg pooled patch-token, 768 dim |

Checkpoint selection uses validation macro-F1 only. Test metrics are not computed in this stage.

## Fixed Controls

- Dataset: HAM10000 benchmark dermoscopic image classification.
- Split: fixed lesion-aware train/validation/test split from Sprint 1.
- Train rows: `7008`.
- Validation rows: `1504`.
- Test split: not used for Sprint 4 metrics or selection.
- Class order: `akiec`, `bcc`, `bkl`, `df`, `nv`, `mel`, `vasc`.
- Downstream classifier: cached-feature MLP with train-only StandardScaler and train-only class
  weights.
- Primary metric: validation macro-F1.

## Frozen Baselines To Compare Against

Single-backbone frozen baselines:

| Frozen baseline | Validation macro-F1 |
|---|---:|
| ViT | `0.6924` |
| Swin | `0.6115` |
| BEiT screening | `0.4759` |

Fusion baselines:

| Frozen fusion baseline | Validation macro-F1 | Role |
|---|---:|---|
| ViT + Swin + BEiT concat, modest MLP | `0.6988` | Best modest E2 fusion |
| ViT + Swin concat, deep_reg MLP | `0.7262` | E2b stronger-MLP diagnostic |

Fine-tuned results must be interpreted against both fusion baselines. Comparing only to the modest
E2 matrix would overstate fine-tuning gain because E2b showed that frozen fusion is classifier
capacity-sensitive.

## Expected Evidence

Required evidence for completion:

```text
artifacts/checkpoints/ham10000/finetuned/<backbone>/best.pt
artifacts/features/ham10000/finetuned/<backbone>/train.pt
artifacts/features/ham10000/finetuned/<backbone>/val.pt
artifacts/features/ham10000/finetuned/<backbone>/manifest.json
artifacts/runs/s4_finetune_<backbone_alias>_seed42/
artifacts/runs/*_s2_finetuned_*_none_mlp*_seed42/
artifacts/runs/*_s3_finetuned_*_mlp*_seed42/
artifacts/report_assets/tables/single_backbone_finetuned_results.csv
artifacts/report_assets/tables/finetuned_fusion_results.csv
```

Verification evidence should include:

- Freeze/unfreeze policy checks.
- Checkpoint selected by validation macro-F1.
- Fine-tuned feature cache row counts: train `7008`, validation `1504`.
- Feature cache alignment by `sample_id`, `image_id`, `lesion_id`, label, and split row order.
- No NaN/Inf in features or probabilities.
- Prediction dump row count `1504`.
- Confusion matrix label order fixed.
- Generated artifacts ignored by Git.

## Validation Results

Full Colab/GPU run completed on 2026-06-11. All values below are validation-only; the test split
was not used.

| Configuration | Feature source | Fusion | Validation macro-F1 | Accuracy | Macro precision | Macro recall | Weighted-F1 |
|---|---|---|---:|---:|---:|---:|---:|
| ViT | finetuned | none | `0.6876` | `0.7979` | `0.6473` | `0.7409` | `0.8063` |
| Swin | finetuned | none | `0.6517` | `0.7852` | `0.6244` | `0.6871` | `0.7930` |
| BEiT | finetuned | none | `0.5181` | `0.6915` | `0.4746` | `0.5938` | `0.7145` |
| ViT + Swin | finetuned | concat | `0.7161` | `0.8211` | `0.6734` | `0.7739` | `0.8273` |
| ViT + Swin + BEiT | finetuned | concat | `0.7298` | `0.8271` | `0.7004` | `0.7673` | `0.8325` |

Other representative fusion conditions:

| Configuration | Fusion | Validation macro-F1 |
|---|---|---:|
| ViT + Swin | weighted_learned_512 | `0.6908` |
| ViT + Swin | weighted_pca_384 | `0.6728` |
| ViT + Swin + BEiT | weighted_learned_512 | `0.6940` |
| ViT + Swin + BEiT | weighted_pca_384 | `0.6713` |

Temporary fine-tuning heads selected during image-level partial fine-tuning were weaker than the
downstream cached-feature MLP protocol: ViT `0.6299`, Swin `0.5907`, and BEiT `0.4136` validation
macro-F1. These heads are checkpoint-selection instruments, not the final downstream classifier.

## Baseline Comparison

Single-backbone fine-tuned features did not consistently improve over frozen single-backbone
baselines. ViT decreased slightly from frozen `0.6924` to fine-tuned `0.6876`. Swin improved from
`0.6115` to `0.6517`, and BEiT improved from `0.4759` to `0.5181`, but both remained below the
original frozen ViT baseline.

The ViT decrease should not be treated as an automatic implementation failure. The absolute drop is
small (`-0.0048` validation macro-F1), while ViT accuracy increased from `0.7872` to `0.7979` and
weighted-F1 increased from `0.7982` to `0.8063`. This pattern suggests a class-level tradeoff rather
than a global degradation: overall image-level performance improved slightly, but macro-F1 remained
sensitive to minority-class changes.

Concat fusion showed the strongest fine-tuned feature transfer result. `ViT + Swin + BEiT concat`
reached validation macro-F1 `0.7298`, above the modest frozen triple concat baseline `0.6988` and
slightly above the E2b stronger-MLP frozen ViT+Swin diagnostic baseline `0.7262`. The margin over
E2b is small (`+0.0036` macro-F1), so the result should be described as limited evidence that
domain-specific adaptation may improve representation quality, not as a decisive win. This is best
interpreted as a validation-stage near-tie with a small advantage for the fine-tuned triple concat
candidate.

BEiT remains most useful as a complementary fusion source. Fine-tuned BEiT single-backbone MLP
reached only `0.5181`, but adding BEiT to ViT+Swin concat increased validation macro-F1 from
`0.7161` to `0.7298`. This supports discussing BEiT as a complementary representation rather than a
strong standalone backbone under this protocol.

Weighted learned and weighted PCA fusion variants underperformed concat in this run. Learned fusion
weights should not be interpreted as a direct backbone quality ranking because projection, weighting,
and downstream classifier training are optimized jointly.

Recommended concise interpretation:

> Fine-tuning did not produce uniform single-backbone gains. The strongest evidence for useful
> fine-tuned representations appeared when the adapted ViT, Swin, and BEiT feature vectors were
> combined through concatenation. Because the improvement over the strongest frozen diagnostic
> baseline was only `+0.0036` validation macro-F1, this result should be treated as limited
> validation evidence rather than a decisive advantage.

## E3b Multi-Seed Robustness Diagnostic

After the single-seed Sprint 4 run, a CPU downstream MLP robustness diagnostic was run over fixed
cached features. No fine-tuning was repeated. The diagnostic changed only the downstream MLP seed and
used validation macro-F1 over seeds `7`, `13`, `42`, `101`, and `202`. Test metrics were not
computed.

| Condition | Seeds | Mean macro-F1 | Std | Min | Max | Mean accuracy | Mean weighted-F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| Fine-tuned ViT + Swin + BEiT concat | `7,13,42,101,202` | `0.7246` | `0.0143` | `0.7032` | `0.7413` | `0.8262` | `0.8305` |
| Fine-tuned ViT + Swin concat | `7,13,42,101,202` | `0.7160` | `0.0085` | `0.7070` | `0.7291` | `0.8205` | `0.8251` |
| Frozen ViT + Swin concat, deep_reg MLP | `7,13,42,101,202` | `0.7077` | `0.0124` | `0.6939` | `0.7262` | `0.8255` | `0.8301` |
| Fine-tuned ViT single | `7,13,42,101,202` | `0.6801` | `0.0084` | `0.6722` | `0.6906` | `0.7947` | `0.8039` |

The diagnostic supports the original ranking direction but changes the strength of the interpretation.
Fine-tuned `ViT + Swin + BEiT concat` remains the strongest mean validation macro-F1 condition, but
its standard deviation is non-trivial and one seed (`0.7032`) falls below some pair and frozen
diagnostic runs. Therefore, the correct wording is not that fine-tuned triple concat is decisively
superior, but that it is the best mean validation candidate among the checked cached-feature
conditions.

The E2b frozen `ViT + Swin concat deep_reg` seed-42 value (`0.7262`) was reproduced on CPU, but the
multi-seed mean was lower (`0.7077`). This means the earlier E2b baseline is a valid run, but it
appears to be near the high end of seed variability rather than a stable central estimate. The
fine-tuned triple concat result should still be compared against the seed-42 E2b value in the main
single-seed storyline, while E3b should be used as robustness context.

Fine-tuned ViT single averaged `0.6801`, below the frozen ViT single baseline `0.6924`. This
strengthens the conclusion that ViT single-backbone partial fine-tuning is a mixed or slightly
negative result under this protocol, even though accuracy and weighted-F1 can remain competitive.

Generated E3b diagnostic tables:

```text
artifacts/report_assets/tables/s4b_multiseed_cpu_diagnostic_results.csv
artifacts/report_assets/tables/s4b_multiseed_cpu_diagnostic_summary.csv
```

## Per-Class Behavior

Best validation-only configuration: `ViT + Swin + BEiT concat`.

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| akiec | `49` | `0.7447` | `0.7143` | `0.7292` |
| bcc | `77` | `0.6552` | `0.7403` | `0.6951` |
| bkl | `165` | `0.6702` | `0.7758` | `0.7191` |
| df | `18` | `0.7500` | `0.8333` | `0.7895` |
| nv | `1006` | `0.9458` | `0.8847` | `0.9142` |
| mel | `167` | `0.5372` | `0.6048` | `0.5690` |
| vasc | `22` | `0.6000` | `0.8182` | `0.6923` |

The largest apparent fusion benefit is visible on minority classes such as `df`, but the support is
only `18`, so this should be treated as high-variance evidence. `mel` remains the most important
class-level limitation among the higher-support non-`nv` classes.

ViT single-backbone fine-tuning illustrates the same issue. Fine-tuned ViT improved over frozen ViT
on `akiec`, `bkl`, `nv`, `mel`, and `vasc`, but its `df` F1 decreased from `0.7179` to `0.5455`.
Because `df` has only `18` validation examples, a small number of changed predictions can move
macro-F1 noticeably. This supports reporting the ViT single-backbone result as a mixed class-level
tradeoff rather than a clear failure of partial fine-tuning.

ViT per-class frozen-to-fine-tuned F1 deltas:

| Label | Support | Frozen ViT F1 | Fine-tuned ViT F1 | Delta |
|---|---:|---:|---:|---:|
| akiec | `49` | `0.5962` | `0.6931` | `+0.0969` |
| bcc | `77` | `0.6790` | `0.6585` | `-0.0205` |
| bkl | `165` | `0.6687` | `0.6852` | `+0.0165` |
| df | `18` | `0.7179` | `0.5455` | `-0.1725` |
| nv | `1006` | `0.8885` | `0.8921` | `+0.0036` |
| mel | `167` | `0.5061` | `0.5391` | `+0.0330` |
| vasc | `22` | `0.7907` | `0.8000` | `+0.0093` |

## Runtime And Artifacts

Fine-tuning runtime on the Colab GPU was approximately 7.8 minutes per backbone: ViT `473.6s`, Swin
`465.5s`, and BEiT `471.3s`. Downstream cached-feature MLP and fusion runs completed in seconds to
under 20 seconds per condition. This makes the partial fine-tuning cost moderate relative to the
small validation macro-F1 gain over the E2b frozen diagnostic.

Local artifact bundle copied from `/Users/arcustin2/Downloads/artifacts/` to repo-local
`artifacts/` on 2026-06-11. Key files:

```text
artifacts/checkpoints/ham10000/finetuned/{vit_b16,swin_tiny,beit_base}/best.pt
artifacts/features/ham10000/finetuned/{vit_b16,swin_tiny,beit_base}/{train.pt,val.pt,manifest.json}
artifacts/runs/s4_finetune_backbones_summary.json
artifacts/runs/s4_finetune_{vit,swin,beit}_seed42/
artifacts/runs/20260611_111312_s2_finetuned_vit_none_mlp_seed42/
artifacts/runs/20260611_111318_s2_finetuned_swin_none_mlp_seed42/
artifacts/runs/20260611_111326_s2_finetuned_beit_none_mlp_seed42/
artifacts/runs/20260611_111338_s3_finetuned_vit-swin_concat_mlp_s4_pair_seed42/
artifacts/runs/20260611_111408_s3_finetuned_vit-swin-beit_concat_mlp_s4_triple_seed42/
artifacts/report_assets/tables/single_backbone_finetuned_results.csv
artifacts/report_assets/tables/single_backbone_finetuned_per_class_metrics.csv
artifacts/report_assets/tables/finetuned_fusion_results.csv
artifacts/report_assets/tables/finetuned_fusion_per_class_metrics.csv
artifacts/report_assets/tables/finetuned_fusion_vs_single_validation.csv
artifacts/report_assets/figures/finetuned_single_backbone_macro_f1.png
artifacts/report_assets/figures/finetuned_fusion_macro_f1.png
```

## Verification Evidence

Local post-download verification confirmed:

- Fine-tuned feature cache shapes: train `(7008, 768)` and validation `(1504, 768)` for ViT, Swin,
  and BEiT.
- Feature cache manifests align with canonical split row order by `image_id`, `lesion_id`, label,
  and split.
- Feature tensors and prediction probabilities contain no NaN/Inf.
- All 12 fine-tuned prediction dumps contain `1504` validation rows.
- Run configs record `test_policy=not_used_in_sprint4`.
- Report tables and figures are present under `artifacts/report_assets/`.
- Generated checkpoints, feature caches, run artifacts, predictions, and report assets are ignored
  by Git.

## Limitations

- Results are expected to be single-seed unless a later multi-seed extension is added.
- Fine-tuning is partial, not full-model adaptation.
- Colab GPU runtime and memory may force batch-size or epoch adjustments.
- Validation macro-F1 is the selection signal; test audit is intentionally reserved for the final
  audit stage.
- Low-support classes such as `df` and `vasc` must be interpreted with support counts visible.
- The `0.7298` versus `0.7262` comparison is not a perfectly isolated representation comparison:
  the fine-tuned result uses three backbones with the modest MLP recipe, while the frozen diagnostic
  uses two frozen backbones with a stronger `deep_reg` MLP. The result identifies the current
  validation-best candidate but does not prove a large fine-tuning advantage.
- The fine-tuned checkpoint is selected using the temporary image-level head. This is a practical
  selection proxy, but it may not identify the exact epoch that maximizes downstream cached-feature
  MLP performance.
- Multiple validation comparisons have been run across backbone, fusion, and MLP variants. Final
  conclusions should acknowledge validation over-selection risk and reserve the test split for a
  single final audit.

## Robustness Checks

Completed:

1. E3b reran downstream cached-feature MLPs with multiple seeds for the top candidates:
   `ViT+Swin+BEiT finetuned concat`, `ViT+Swin finetuned concat`, fine-tuned ViT single, and the
   frozen E2b `ViT+Swin concat deep_reg` reference.

Remaining optional checks:

1. Add paired bootstrap confidence intervals for the macro-F1 difference between `0.7298` and
   `0.7262`.
2. Compare confusion matrices for frozen ViT vs fine-tuned ViT, fine-tuned ViT+Swin vs fine-tuned
   ViT+Swin+BEiT, and best fine-tuned fusion vs strongest frozen diagnostic.
3. Add a fixed-vs-broken validation analysis to count samples corrected by the triple fusion model
   and samples newly misclassified.
4. If compute permits, run an E2b-style stronger MLP diagnostic on the fine-tuned concat features to
   separate representation effects from downstream classifier capacity.

## Report-Ready Turkish Wording

The following paragraphs avoid internal project-management language and can be adapted directly for
the final academic report.

**Yöntem özeti:**

Bu aşamada seçilen üç transformer backbone için kontrollü partial fine-tuning uygulanmıştır. ViT ve
BEiT modellerinde son iki transformer bloğu, Swin Transformer'da ise son Swin stage'i ve sınıflandırma
başlığı eğitilebilir bırakılmış; önceki katmanlar dondurulmuştur. Checkpoint seçimi yalnız validation
macro-F1 ile yapılmış, test split herhangi bir model veya hiperparametre seçimi için
kullanılmamıştır. Seçilen checkpoint'lerden sonra sınıflandırma başlığı çıkarılarak train ve
validation splitleri için fine-tuned transformer feature cache'leri üretilmiş, downstream MLP ve
fusion deneyleri bu cache'ler üzerinden train-only StandardScaler ile çalıştırılmıştır.

**Bulgular:**

Fine-tuned single-backbone sonuçları frozen feature baselineları üzerinde tutarlı bir iyileşme
göstermemiştir. ViT validation macro-F1 değeri `0.6924`ten `0.6876`ya hafifçe düşerken, Swin
`0.6115`ten `0.6517`ye ve BEiT `0.4759`dan `0.5181`e yükselmiştir. Buna karşılık en güçlü
validation sonucu, fine-tuned ViT, Swin ve BEiT feature'larının concat fusion ile birleştirilmesiyle
elde edilmiştir. Bu yapı `0.7298` validation macro-F1, `0.8271` accuracy ve `0.8325` weighted-F1
değerlerine ulaşmıştır.

**Tartışma:**

Bu sonuçlar partial fine-tuning'in her transformer backbone için bağımsız olarak aynı ölçüde fayda
sağlamadığını göstermektedir. En güçlü sinyal, fine-tuned representation'ların concat fusion ile
birleştirildiği durumda ortaya çıkmıştır. Fine-tuned üçlü concat modeli, modest frozen üçlü concat
baseline'ını (`0.6988`) aşmış ve daha güçlü MLP kullanılan frozen ViT+Swin diagnostic baseline'ının
(`0.7262`) çok az üzerine çıkmıştır. Ancak ikinci karşılaştırmadaki fark yalnızca `+0.0036`
macro-F1 olduğu için bu sonuç, domain-specific adaptation'ın temsil kalitesini artırabileceğine dair
sınırlı validation evidence olarak yorumlanmalıdır; kesin veya büyük bir üstünlük iddiası
kurulmamalıdır.

Ek downstream multi-seed robustness analizinde, sabit cached feature'lar üzerinde beş farklı MLP
seed'i denenmiştir. Fine-tuned ViT+Swin+BEiT concat koşulu `0.7246 ± 0.0143` mean validation
macro-F1 ile en yüksek ortalamayı üretmiştir; fine-tuned ViT+Swin concat `0.7160 ± 0.0085`, frozen
ViT+Swin concat deep_reg ise `0.7077 ± 0.0124` ortalama vermiştir. Bu sonuç, üçlü fine-tuned concat
adayının ortalama olarak güçlü kaldığını, ancak validation seed variance nedeniyle sonucun yine
temkinli yorumlanması gerektiğini göstermektedir.

**BEiT yorumu:**

BEiT standalone feature kaynağı olarak zayıf kalmıştır; fine-tuned single-backbone MLP sonucu
`0.5181` macro-F1'dir. Buna rağmen BEiT'in ViT ve Swin feature'larına eklenmesi, fine-tuned
ViT+Swin concat sonucunu `0.7161`den `0.7298`e yükseltmiştir. Bu nedenle BEiT, bu protokolde güçlü
bir bağımsız backbone olarak değil, ViT ve Swin'e tamamlayıcı bilgi sağlayabilen bir representation
kaynağı olarak tartışılmalıdır.

**Sınırlılıklar:**

Sonuçlar validation-only ve single-seed olarak değerlendirilmiştir. Özellikle `df` ve `vasc` gibi
düşük destekli sınıflarda az sayıda prediction değişimi per-class F1 ve macro-F1 üzerinde belirgin
oynamalara neden olabilir. Bu nedenle en iyi validation adayının final değerlendirmesi, model seçimi
tamamlandıktan sonra test split üzerinde tek seferlik audit olarak yapılmalıdır. Sonuçlar yalnızca
HAM10000 benchmark dermoscopic image classification bağlamında raporlanmalı; klinik tanı, hasta
güvenliği veya deploy edilebilir medikal performans iddiası kurulmamalıdır.
