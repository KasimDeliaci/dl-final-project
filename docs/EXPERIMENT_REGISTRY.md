# Experiment Registry

Bu dosya, deney başlamadan önce hipotezi ve kontrol koşulunu sabitlemek için kullanılır. Kod veya notebook çalıştırılmadan önce ilgili deney satırı doldurulmalıdır.

## Template

```text
ID:
Status: planned | running | completed | dropped
Question:
Hypothesis:
Changed variable:
Fixed controls:
Selection rule:
Expected failure mode:
Required artifacts:
Report role:
```

## Initial Planned Experiments

### E0 - Dataset Audit and Split

Status: completed

Question: HAM10000 split'i leakage-safe ve class-aware biçimde kurulabiliyor mu?

Hypothesis: Lesion-aware stratified split, sınıf dağılımını makul düzeyde korurken aynı lesion'ın farklı split'lere sızmasını engeller.

Changed variable: Split generation policy.

Fixed controls: Dataset version and metadata source.

Selection rule: Bu model seçimi değil; split audit kabul kriteri leakage olmaması ve sınıf dağılımının raporlanabilir olmasıdır.

Expected failure mode: Minority classes nedeniyle exact stratification mümkün olmayabilir.

Required artifacts:

- split CSV files,
- class distribution table,
- lesion leakage audit,
- `docs/DATASET_AUDIT.md`.

Report role: Dataset preparation and leakage control.

Result note: Sprint 1 completed with 10,015 verified images, 7,470 unique lesion IDs, canonical lesion-aware split counts of 7,008 train / 1,504 validation / 1,503 test images, all seven classes present in every split, and zero cross-split lesion leakage.

### E1 - Frozen Single-Backbone Baselines

Status: completed

Question: Vanilla ViT, Swin Transformer ve DeiT III-Small frozen feature extractor olarak tek başına nasıl davranıyor?

Hypothesis: Farklı transformer mimarileri aynı dataset üzerinde farklı class-level güçlü/zayıf yönler gösterecektir.

Changed variable: Backbone architecture.

Fixed controls: Canonical lesion-aware split, deterministic 224x224 ImageNet preprocessing, frozen feature extraction, train-only StandardScaler, class-weighted MLP classifier recipe.

Selection rule: MLP checkpoint validation macro-F1 ile seçilir; test yalnız audit.

Expected failure mode: Feature dimension veya token pooling farkları nedeniyle MLP capacity farklı modellere eşit davranmayabilir.

Required artifacts:

- frozen feature cache,
- feature manifest,
- run config JSON,
- metrics summary,
- per-class metrics,
- confusion matrix,
- prediction dump,
- training history,
- runtime metadata.

Report role: Single-backbone representation quality baseline.

Result note: Sprint 2 completed with full train/validation frozen feature caches for Vanilla ViT, Swin Transformer, and DeiT III-Small. MLP checkpoints were selected by validation macro-F1 only. Validation macro-F1 results were ViT `0.6924`, Swin `0.6115`, and DeiT III-Small `0.5017`. BEiT-Base was later screened as a candidate third backbone and reached validation macro-F1 `0.4759`, so it was not selected over DeiT III-Small. Test metrics were not computed for model selection.

### E2 - Frozen Feature Fusion

Status: completed

Question: Transformer backbone feature'ları concat veya weighted fusion ile tamamlayıcı sinyal üretir mi?

Hypothesis: Pairwise veya three-backbone fusion, tek backbone'a göre macro-F1 artışı sağlayabilir; ancak bu artış feature complementarity'ye bağlıdır.

Changed variable: Backbone combination and fusion method: `concat`, `weighted_learned_512`, `weighted_pca_384`.

Fixed controls: Sprint 2 frozen feature caches, canonical lesion-aware split, train-only StandardScaler, class-weighted MLP recipe, validation macro-F1 selection, and no test metric use.

Selection rule: Fusion candidate selection validation macro-F1 ile yapılır.

Expected failure mode: Concatenation feature dimension'ı büyütüp overfitting yaratabilir; weighted learned fusion projection bottleneck oluşturabilir; weighted PCA compression information loss yaratabilir; zayıf veya redundant backbone'lar tamamlayıcı sinyali artırmayabilir.

Required artifacts:

- fusion metrics table,
- pairwise comparison table,
- per-class metrics,
- feature dimension log,
- prediction dump,
- confusion matrix,
- training history,
- learned fusion weights for weighted runs,
- PCA train-only metadata for `weighted_pca_384`.

Report role: Feature fusion and complementarity analysis.

Result note: Sprint 3 completed with a validation-only frozen fusion matrix over ViT, Swin Transformer, DeiT III-Small, and an E2 BEiT-expanded alternative. The planned ViT/Swin/DeiT matrix contained 12 runs. The BEiT-expanded matrix added 9 runs covering `vit_b16+beit_base`, `swin_tiny+beit_base`, and `vit_b16+swin_tiny+beit_base` with the same three fusion methods. Total E2 fusion runs: 21. The best planned ViT/Swin/DeiT run was `vit_b16+swin_tiny` with `concat`, reaching validation macro-F1 `0.6947`. The best BEiT-expanded run was `vit_b16+swin_tiny+beit_base` with `concat`, reaching validation macro-F1 `0.6988`. Both slightly exceeded the strongest Sprint 2 single-backbone baseline, ViT at `0.6924`, but the gains are small and should be framed as limited complementarity evidence rather than a decisive improvement. Test metrics were not computed.

BEiT-expanded E2 result: BEiT-Base was evaluated as an alternative third backbone inside E2 because its representation similarity diagnostic suggested lower similarity to ViT and Swin. BEiT pairwise runs did not exceed ViT single: `vit_b16+beit_base concat` reached `0.6556` macro-F1 and `swin_tiny+beit_base concat` reached `0.6381`. However, `vit_b16+swin_tiny+beit_base concat` reached `0.6988`, suggesting BEiT may add limited complementary signal when combined with both stronger backbones despite being weak as a single-backbone candidate.

Representation similarity diagnostic: Validation feature caches were analyzed with train-only scaling and sample-cosine RSA Pearson correlation. BEiT showed lower representation similarity with ViT (`0.4393`) and Swin (`0.2874`) than the canonical ViT+Swin pair (`0.5942`). The average pairwise representation complementarity for `vit_b16+swin_tiny+beit_base` was `0.5597`, higher than `vit_b16+swin_tiny` at `0.4058`. This supports a cautious interpretation that BEiT can be weak as a standalone classifier feature while still adding complementary structure in concat fusion. Test split was not used.

### E3 - Fine-Tuning Last Transformer Blocks

Status: completed

Question: Forward backbone setinde son transformer bloklarını fine-tune etmek HAM10000 temsillerini iyileştirir mi?

Hypothesis: Son transformer bloklarının kontrollü fine-tuning'i domain-specific dermoscopic representation kalitesini artırabilir; ancak küçük/dengesiz dataset nedeniyle overfitting riski yüksektir.

Changed variable: Transfer learning policy over the forward backbone set `vit_b16`, `swin_tiny`, `beit_base`.

Fixed controls: Canonical lesion-aware split, E2-selected forward backbone set, classifier evaluation protocol, validation-only checkpoint/model selection.

Selection rule: Fine-tuned checkpoint validation macro-F1 ile seçilir.

Expected failure mode: Çok agresif unfreeze veya yüksek LR minority-class precision/recall dengesini bozabilir.

Required artifacts:

- training log,
- validation curve,
- selected checkpoint metadata,
- fine-tuned feature cache,
- metrics summary,
- per-class metrics,
- prediction dump.

Report role: Transfer learning comparison.

Planning note: `deit3_small` remains a screened/planned baseline from E1/E2, but Sprint 4 fine-tuning scope uses `beit_base` as the third backbone because E2 validation fusion and representation similarity diagnostics favored BEiT complementarity.

Implementation plan: Sprint 4 is recorded in `docs/exec-plans/completed/sprint-4-finetuned-transformer-features.md`. Partial fine-tuning policy is `vit_b16` last 2 transformer blocks plus norm/head, `swin_tiny` last Swin stage plus norm/head, and `beit_base` last 2 transformer blocks plus fc_norm/norm/head. Canonical Sprint 4 cache extraction writes train and validation caches only under `artifacts/features/ham10000/finetuned/<backbone>/`. Test metrics are not computed in Sprint 4.

Result note: E3 completed as a validation-only Colab/GPU run over `vit_b16`, `swin_tiny`, and `beit_base`. Selected checkpoints were written under `artifacts/checkpoints/ham10000/finetuned/<backbone>/best.pt`, and fine-tuned train/validation feature caches were written under `artifacts/features/ham10000/finetuned/<backbone>/` with canonical row counts of 7,008 train and 1,504 validation rows. Single-backbone fine-tuned feature MLP validation macro-F1 values were ViT `0.6876`, Swin `0.6517`, and BEiT `0.5181`. Fine-tuned concat fusion improved to `0.7161` for `vit_b16+swin_tiny` and `0.7298` for `vit_b16+swin_tiny+beit_base`. The best fine-tuned result therefore exceeded the modest frozen triple concat baseline (`0.6988`) and narrowly exceeded the E2b stronger-MLP frozen ViT+Swin diagnostic baseline (`0.7262`). The gain over E2b is small, so it should be framed as limited evidence for domain-specific adaptation improving representation quality. Test metrics were not computed.

### E2b - MLP Capacity Diagnostic for Frozen Features

Status: completed

Question: Frozen feature fusion sonuçları mevcut modest MLP classifier kapasitesiyle mi sınırlı kalıyor?

Hypothesis: Daha geniş veya daha düzenlileştirilmiş MLP classifier, özellikle high-dimensional concat fusion koşullarında validation macro-F1'i artırabilir; ancak single-backbone ViT de benzer şekilde artarsa fusion complementarity yorumu değişmeyebilir.

Changed variable: MLP hidden dimensions, dropout, learning rate, weight decay, and early-stopping patience.

Fixed controls: Frozen feature caches, canonical lesion-aware split, train-only StandardScaler, train-only class-weighted cross entropy, selected representative feature configurations, validation-only selection, no test metrics.

Selection rule: Diagnostic variants validation macro-F1 ile karşılaştırılır. Test seti kullanılmaz.

Expected failure mode: Daha güçlü MLP high-dimensional concat features üzerinde overfit edebilir veya validation macro-F1'i artırmadan training instability yaratabilir.

Required artifacts:

- run configs,
- metrics summaries,
- per-class metrics,
- prediction dumps,
- training histories,
- diagnostic comparison table.

Report role: Classifier capacity sensitivity and robustness check for E2 fusion conclusions.

Result note: E2b completed as a validation-only MLP capacity diagnostic over representative frozen feature configurations. Stronger MLP variants did not improve ViT single-backbone validation macro-F1 over the original baseline (`0.6924`). They substantially improved `vit_b16+swin_tiny concat`, with the `deep_reg` variant reaching validation macro-F1 `0.7262`. `vit_b16+swin_tiny+beit_base concat` also improved, reaching `0.7159` with `wide_reg`, but did not exceed the stronger `vit_b16+swin_tiny concat` result. `vit_b16+swin_tiny+deit3_small concat` did not improve meaningfully over its baseline. This indicates that frozen fusion conclusions are sensitive to MLP capacity, and that BEiT remains a stronger third-backbone candidate than DeiT under this probe, while the best frozen concat configuration under stronger MLP is the ViT+Swin pair. Test metrics were not computed.

### E3b - Downstream MLP Multi-Seed Robustness Diagnostic

Status: completed

Question: Sprint 4'teki cached-feature MLP/fusion sonuçları downstream MLP initialization seed'ine ne kadar duyarlı?

Hypothesis: Fine-tuned concat fusion, tek seed sonucuna göre validation-best görünmektedir; ancak küçük marginler nedeniyle downstream MLP seed variance sonuç yorumunu etkileyebilir.

Changed variable: Downstream MLP random seed over cached features. Fine-tuned feature caches and frozen feature caches are fixed.

Fixed controls: Canonical train/validation split, cached feature tensors, train-only StandardScaler, train-only class weights, validation-only selection, no test metrics, CPU device for comparability with the original E2b frozen diagnostic.

Selection rule: Diagnostic only; final model selection is not changed by this run alone. Summary statistics are reported as mean/std/min/max validation macro-F1 over seeds `7`, `13`, `42`, `101`, and `202`.

Expected failure mode: Validation macro-F1 differences may be seed-sensitive, especially when comparing close candidates such as `0.7298` and `0.7262`.

Required artifacts:

- `artifacts/report_assets/tables/s4b_multiseed_cpu_diagnostic_results.csv`,
- `artifacts/report_assets/tables/s4b_multiseed_cpu_diagnostic_summary.csv`,
- per-run validation metrics, prediction dumps, training histories, and run configs under `artifacts/runs/*s4b_multiseed_cpu*`.

Report role: Robustness check for Sprint 4 fine-tuned feature conclusions.

Result note: E3b completed as a CPU validation-only downstream MLP robustness diagnostic. Over seeds `7,13,42,101,202`, fine-tuned `vit_b16+swin_tiny+beit_base concat` had the highest mean validation macro-F1 (`0.7246 ± 0.0143`, min `0.7032`, max `0.7413`). Fine-tuned `vit_b16+swin_tiny concat` averaged `0.7160 ± 0.0085`. Frozen `vit_b16+swin_tiny concat deep_reg` averaged `0.7077 ± 0.0124`; the original seed-42 E2b value `0.7262` was reproduced on CPU but appears to be near the high end of the observed seed range. Fine-tuned ViT single averaged `0.6801 ± 0.0084`, supporting the conclusion that ViT single-backbone fine-tuning is mixed and does not robustly exceed the frozen ViT baseline. Test metrics were not computed.

### E3c - Metadata-Augmented Cached Feature Fusion

Status: completed

Question: HAM10000 benchmark metadata (`age`, `sex`, `localization`) fine-tuned transformer feature fusion'a eklendiğinde validation macro-F1 ve per-class F1 iyileşiyor mu?

Hypothesis: Age, sex, and localization may provide complementary benchmark signal to image-only fine-tuned transformer features, especially for classes whose distribution is associated with anatomical site or age. However, metadata may also encode dataset-specific correlations and may not improve macro-F1 once strong transformer features are used.

Changed variable: Metadata input is added to fixed cached feature MLP/fusion conditions.

Fixed controls: Canonical lesion-aware train/validation split, fixed fine-tuned feature caches, train-only image scaler, train-only metadata preprocessing, train-only class weights, validation-only selection, no test metrics.

Selection rule: Diagnostic comparison uses mean/std/min/max validation macro-F1 over seeds `7`, `13`, `42`, `101`, and `202`. Final model selection is not changed by a single seed.

Expected failure mode: Metadata may improve majority-class or accuracy behavior while hurting macro-F1, or it may overfit validation through benchmark-specific correlations.

Required artifacts:

- run configs,
- metadata preprocessing metadata,
- metrics summaries,
- per-class metrics,
- confusion matrices,
- prediction dumps,
- training histories,
- multi-seed summary table,
- image-only vs image+metadata comparison table.

Report role: Multimodal feature-transfer diagnostic for structured benchmark metadata.

Implementation plan: E3c is recorded in `docs/exec-plans/completed/e3c-metadata-augmented-feature-fusion.md`. Allowed metadata fields are limited to `age`, `sex`, and `localization`; `dx`, `dx_type`, `dataset`, `image_id`, `sample_id`, and `lesion_id` are excluded from model input. Test split is not loaded or transformed.

Result note: E3c completed as a validation-only metadata-augmented cached-feature diagnostic. Over seeds `7,13,42,101,202`, fine-tuned `vit_b16+swin_tiny+beit_base concat + metadata` reached mean validation macro-F1 `0.7278 ± 0.0058`, compared with the image-only E3b triple concat mean `0.7246 ± 0.0143`. Fine-tuned `vit_b16+swin_tiny concat + metadata` reached `0.7230 ± 0.0138`, compared with the image-only E3b pair concat mean `0.7160 ± 0.0085`. The metadata-only MLP remained weak (`0.2202 ± 0.0077`), indicating that metadata is not independently sufficient. Per-class behavior was mixed: triple concat + metadata improved mean F1 most clearly for `akiec` but reduced `bkl`; pair concat + metadata improved `df`, `vasc`, and `nv` modestly while slightly reducing `mel`. The result should be framed as limited validation evidence that structured benchmark metadata can provide small complementary signal to fine-tuned transformer features, not as a clinical claim. Test metrics were not computed.

### E3d - Metadata Fusion Operator Ablation

Status: completed

Question: E3c'de küçük fayda veren metadata sinyali, raw concat yerine metadata-conditioned lightweight fusion operator'larıyla daha güçlü veya daha stabil validation macro-F1 üretir mi?

Hypothesis: Age, sex, and localization metadata may be more useful when used to modulate fine-tuned transformer image features than when only appended to the MLP input. However, small metadata-conditioned operators may overfit validation or shift gains away from important minority classes.

Changed variable: Metadata fusion operator over fixed fine-tuned ViT/Swin/BEiT cached features.

Fixed controls: Canonical train/validation split, fixed fine-tuned feature caches, train-only per-backbone image scaling, train-only metadata preprocessing, train-only class weights, validation-only selection, no test metrics, seeds `7`, `13`, `42`, `101`, and `202`.

Selection rule: Diagnostic comparison uses mean/std/min/max validation macro-F1 over seeds. E3d must be compared against E3c raw concat + metadata and E3b image-only controls.

Expected failure mode: Metadata-gated or FiLM-style operators may produce a small validation gain while hurting `mel` or other minority-class F1, or may be less stable than raw concat.

Required artifacts:

- run configs,
- metadata preprocessing metadata,
- metadata fusion metadata,
- optional gate summaries,
- metrics summaries,
- per-class metrics,
- confusion matrices,
- prediction dumps,
- training histories,
- operator summary table,
- comparison against E3c raw concat and E3b image-only controls.

Report role: Lightweight multimodal fusion operator ablation over fixed fine-tuned transformer features.

Implementation plan: E3d is recorded in `docs/exec-plans/completed/e3d-metadata-fusion-operator-ablation.md`. Canonical operators are metadata-gated backbone fusion, bounded FiLM-style metadata conditioning, and a two-branch image/metadata MLP. Test split is not loaded or transformed.

Result note: E3d completed as a validation-only metadata fusion operator ablation over fixed fine-tuned `vit_b16+swin_tiny+beit_base` features. Over seeds `7,13,42,101,202`, bounded FiLM-style metadata conditioning reached the highest mean validation macro-F1 (`0.7358 ± 0.0152`), followed by metadata-gated backbone fusion (`0.7347 ± 0.0112`) and two-branch image/metadata fusion (`0.7328 ± 0.0103`). All three operators exceeded the E3c raw concat + metadata control (`0.7278 ± 0.0058`) and the E3b image-only triple concat control (`0.7246 ± 0.0143`). Per-class behavior remained mixed: FiLM improved `mel` relative to E3c raw concat but reduced `akiec` and `vasc`; gated fusion improved `akiec`, `bcc`, `mel`, and `vasc` but reduced `df`; two-branch fusion improved `df` and `mel` but reduced `vasc`. The result supports limited validation evidence that metadata is more useful when it conditions image representations than when only appended, but it does not support a clinical claim or final test conclusion. Test metrics were not computed.

### E3e - Conservative ViT Fine-Tuning Diagnostic

Status: completed

Question: Canonical partial fine-tuning neden ViT single-backbone validation macro-F1'i frozen ViT baseline'a göre az düşürdü (`0.6924 -> 0.6876`)? Daha düşük backbone LR veya daha dar unfreeze policy ViT representation'ı koruyarak bu düşüşü azaltıyor mu?

Hypothesis: ViT'in canonical last-2-block fine-tuning koşulundaki düşüşü, HAM10000 benchmark dermoscopic image classification için over-adaptation veya learning-rate sensitivity kaynaklı olabilir. Daha konservatif `last_2_blocks + 5e-6` veya `last_1_block + 5e-6` koşulları frozen ViT representation kalitesini daha iyi koruyabilir.

Changed variable: Only ViT fine-tuning policy and backbone LR. Swin/BEiT checkpoints and caches are not re-fine-tuned.

Fixed controls: Canonical lesion-aware train/validation split, ViT model ID, image size, train transforms, class weighting, head LR `1e-4`, epochs `8`, early stopping patience `3`, validation macro-F1 checkpoint selection, no test metrics.

Selection rule: Compare E3e ViT single-backbone MLP against frozen ViT (`0.6924`) and canonical fine-tuned ViT (`0.6876`). Mixed ViT+Swin+BEiT concat uses validation macro-F1 only and is compared against E3b/E3c/E3d validation controls. No final selection is made from test metrics.

Expected failure mode: Lower LR may under-adapt, or last-one-block fine-tuning may improve ViT single but not downstream fusion. Negative results remain reportable as ViT fine-tuning sensitivity evidence.

Required artifacts:

- E3e configs,
- Colab runner,
- selected ViT checkpoints,
- train/validation feature caches,
- checkpoint metadata with trainable prefixes and learning rates,
- ViT single-backbone MLP runs,
- mixed ViT+Swin+BEiT concat runs,
- optional metadata-conditioned follow-up only if validation results justify it,
- training curves,
- per-class metrics,
- confusion matrices,
- prediction dumps,
- Drive sync under `MyDrive/dl-final-artifact/e3e_conservative_vit/`.

Report role: Focused fine-tuning sensitivity diagnostic for the strongest frozen transformer backbone.

Implementation plan: E3e is recorded in `docs/exec-plans/completed/e3e-conservative-vit-finetuning.md`. Colab launcher: `notebooks/05_e3e_conservative_vit_finetuning.ipynb`. Test split was not loaded or transformed for E3e.

Result note: E3e completed as a validation-only Colab diagnostic. Conservative ViT single-backbone features did not recover the frozen ViT baseline: `last_2_blocks + LR 5e-6` reached `0.6694` validation macro-F1 and `last_1_block + LR 5e-6` reached `0.6685`, both below frozen ViT (`0.6924`) and canonical fine-tuned ViT (`0.6876`). Mixed ViT+Swin+BEiT concat with canonical fine-tuned Swin/BEiT caches was stronger for the last-1 policy (`0.7259`) than the last-2 lower-LR policy (`0.7082`), but did not exceed the canonical seed-42 fine-tuned triple concat (`0.7298`) or the E3d metadata-conditioned validation means. E3e is therefore reported as a negative but informative fine-tuning sensitivity ablation, not as a replacement for the main fine-tuned fusion result. Test metrics were not computed.

### E3f - Frozen ViT + Fine-Tuned Swin/BEiT Mixed Adaptation

Status: completed

Question: ViT fine-tuning single-backbone macro-F1'i dusururken Swin ve BEiT fine-tuning kazanc gosterdi. ViT'i frozen strong control olarak tutup Swin/BEiT'i fine-tuned kullanmak, all-fine-tuned triple feature source'a gore daha iyi veya daha stabil validation macro-F1 uretir mi?

Hypothesis: Frozen ViT feature'lari zaten guclu oldugu icin ViT'i fine-tune etmek zorunlu olmayabilir. Buna karsilik Swin ve BEiT'in fine-tuned feature'lari frozen hallerine gore daha iyi oldugundan, mixed adaptation source (`frozen_vit_finetuned_swin_beit`) daha dengeli bir representation seti olusturabilir.

Changed variable: Backbone-level feature source assignment only. `vit_b16` uses frozen cache; `swin_tiny` and `beit_base` use canonical fine-tuned caches. No new transformer fine-tuning is run.

Fixed controls: Canonical lesion-aware train/validation split, train-only scaling, train-only metadata preprocessing, class weighting, validation macro-F1 selection, no test metrics.

Selection rule: Compare multi-seed validation macro-F1 for mixed image-only concat and mixed metadata-conditioned FiLM/gated operators against E3b/E3c/E3d all-fine-tuned controls. No test split usage.

Required artifacts:

- mixed feature source manifest,
- image-only concat MLP runs over seeds `7,13,42,101,202`,
- metadata FiLM and metadata-gated runs over the same seeds,
- summary tables,
- per-class metrics,
- prediction dumps,
- comparison against E3b/E3c/E3d controls.

Report role: Backbone-level adaptation ablation testing whether ViT should remain frozen while Swin/BEiT are adapted.

Implementation plan: E3f is recorded in `docs/exec-plans/completed/e3f-mixed-frozen-vit-finetuned-swin-beit.md`. Test split was not loaded or transformed for E3f.

Result note: E3f completed as a validation-only cached-feature ablation. The mixed image-only concat source (`frozen vit_b16 + fine-tuned swin_tiny + fine-tuned beit_base`) reached `0.7142 ± 0.0126` mean validation macro-F1, below the all-fine-tuned triple concat control (`0.7246 ± 0.0143`). Metadata-conditioned results were stronger: mixed FiLM reached `0.7267 ± 0.0191`, while mixed metadata-gated fusion reached `0.7361 ± 0.0100`. The mixed gated mean is numerically the highest current validation-only multi-seed mean and is essentially tied with E3d all-fine-tuned FiLM (`0.7358 ± 0.0152`). Per-class behavior is mixed: the mean gain is driven mainly by `df`, while `akiec`, `bcc`, and `vasc` decline relative to E3d all-fine-tuned gated. The result is reported as source/operator interaction evidence, not as a decisive final model conclusion. Test metrics were not computed.

### E3g - Prediction-Level Ensemble Diagnostic

Status: completed

Question: E3d and E3f top validation candidates are nearly tied but show different per-class behavior. Can validation-only probability averaging over existing prediction dumps reduce seed/model-family variance and improve validation macro-F1?

Hypothesis: Equal-weight probability averaging across top E3d/E3f families may improve validation macro-F1 by smoothing seed variance and combining complementary per-class behavior, especially between E3d FiLM and E3f mixed gated outputs.

Changed variable: Prediction-level aggregation only. No backbone, feature cache, metadata preprocessing, or MLP training is changed.

Fixed controls: Existing validation prediction dumps, fixed class order, fixed lesion-aware split, validation-only metrics, no test split usage.

Selection rule: Primary results are equal-weight seed/family ensembles. Small weighted-grid diagnostics may be run but must be reported separately because they use validation labels for model-choice pressure. If improvement is below about `0.002` macro-F1, report as practical tie.

Required artifacts:

- ensemble membership table,
- ensemble prediction dumps,
- validation metrics,
- per-class metrics,
- confusion matrices,
- error-overlap summary,
- comparison against E3d/E3f controls.

Report role: Post-hoc validation-only robustness and score-oriented diagnostic before considering heavier TTA/multi-view extraction.

Implementation plan: E3g is recorded in `docs/exec-plans/completed/e3g-prediction-ensemble.md`. Test split was not loaded or transformed for E3g.

Result note: E3g completed as a validation-only prediction ensemble diagnostic. The strongest primary equal-weight ensemble, `top3_family_equal`, averaged the seed-averaged E3d FiLM, E3d gated, and E3f mixed gated probability outputs with equal family weights and reached validation macro-F1 `0.7665`, accuracy `0.8564`, and weighted-F1 `0.8576`. This is the strongest low-overfit validation-only result so far. A small weighted-grid diagnostic reached `0.7702` macro-F1 with E3d FiLM `0.50`, E3d gated `0.25`, and E3f gated `0.25`, but it is reported as exploratory because the weights are selected under validation-label pressure. Error-overlap analysis showed incomplete family redundancy, especially between E3d FiLM and E3f gated (`0.6690` error Jaccard), supporting the ensemble gain. Test metrics were not computed.

### E3h - Rot4 Test-Time Augmentation Diagnostic

Status: active

Question: Can deterministic four-view right-angle rotation TTA improve the validation macro-F1 of
the E3g `top3_family_equal` prediction ensemble without training new models or using the test split?

Hypothesis: Probability averaging over `identity`, `rot90`, `rot180`, and `rot270` views may
stabilize predictions for the strongest metadata-conditioned E3d/E3f model families and improve
validation macro-F1 over the no-TTA E3g ensemble.

Changed variable: Inference-time deterministic TTA view averaging only.

Fixed controls: Existing lesion-aware split, existing E3d/E3f MLP weights, saved train-fitted
scaler statistics, saved metadata preprocessing, equal-family ensemble rule, validation-only
selection, no test split usage.

Selection rule: Primary result is `top3_family_equal_tta_rot4`, using equal probability averaging
across TTA views, seeds, and the three E3g families. A validation macro-F1 gain below `+0.002` over
E3g is a practical tie; `+0.002` to `< +0.005` is suggestive; `>= +0.005` is meaningful only if
per-class behavior is acceptable.

Required artifacts:

- TTA policy record,
- identity sanity-check table,
- validation prediction dumps,
- validation metrics,
- per-class metrics,
- confusion matrix,
- corrected-vs-broken analysis against E3g,
- runtime metadata,
- report-ready tables and figures.

Report role: Inference-time robustness diagnostic for the strongest validation-selected ensemble.
Not a new training or representation-learning result.

Implementation plan: E3h is recorded in `docs/exec-plans/active/e3h-tta-rot4-inference.md`.
Implementation support is in `src/dl_final/evaluation/tta.py` and
`scripts/evaluate_tta_rot4.py`. Test split must not be loaded or transformed for E3h.

### E3i - Simple Fusion Rot4 TTA Diagnostic

Status: active

Question: Does deterministic four-view right-angle rotation TTA improve simpler image-only
cached-feature MLP/fusion models before metadata-conditioned seed/family ensembling is applied?

Hypothesis: E3h may have applied TTA too late, after a strong probability ensemble had already
stabilized many errors. Applying the same fixed rot4 policy to simpler concat and learned-weighted
fusion models may recover the model-level TTA gain observed in the older CNN project.

Changed variable: Inference-time deterministic TTA view averaging over selected simple fusion
models only.

Fixed controls: Existing lesion-aware split, existing fine-tuned feature extractors, existing saved
MLP/fusion weights, saved train-fitted scaler statistics, validation-only comparison, no test split
usage.

Selection rule: E3i is diagnostic. The primary table compares each candidate run's stored no-TTA
validation macro-F1 against its rot4-averaged validation macro-F1. A simple equal average across
candidate runs may be reported as diagnostic context, but it is not a new validation-tuned ensemble
selection rule.

Candidate runs:

- fine-tuned `vit_b16+swin_tiny+beit_base` concat seed 42,
- fine-tuned `vit_b16+swin_tiny+beit_base` weighted learned 512 seed 42,
- fine-tuned `vit_b16+swin_tiny` concat seed 42.

Required artifacts:

- TTA policy record,
- identity sanity-check table,
- per-view validation metrics,
- per-run no-TTA vs rot4 comparison,
- validation prediction dumps,
- per-class delta table,
- corrected-vs-broken table,
- report-ready tables and figures.

Report role: Inference-time diagnostic to determine whether the negative E3h result is specific to
the already-ensembled metadata-conditioned model family.

Implementation plan: E3i is recorded in `docs/exec-plans/active/e3i-simple-fusion-tta.md`.
Implementation support is in `scripts/evaluate_simple_tta_rot4.py`. Test split must not be loaded
or transformed for E3i.

### E4 - Final Model Selection and Audit

Status: planned

Question: Final model hangi backbone/fusion/transfer-learning çizgisinden seçilmeli?

Hypothesis: Final model en yüksek test skoruna göre değil, validation discipline ile seçilen en güçlü ve açıklanabilir configuration olmalıdır.

Changed variable: Final selection among completed validated candidates.

Fixed controls: Test set remains untouched until audit.

Selection rule: Validation macro-F1 and stability of per-class behavior; test set only final audit.

Expected failure mode: Test audit gain küçük olabilir veya minority-class davranışı tek metrikte saklanabilir.

Required artifacts:

- final comparison table,
- final confusion matrix,
- final per-class F1 chart,
- prediction dump,
- report-ready decision note.

Report role: Final result and discussion anchor.
