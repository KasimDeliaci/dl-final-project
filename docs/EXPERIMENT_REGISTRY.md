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

Status: planned

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

Implementation plan: E3e is recorded in `docs/exec-plans/active/e3e-conservative-vit-finetuning.md`. Colab launcher: `notebooks/05_e3e_conservative_vit_finetuning.ipynb`. Test split is not loaded or transformed for E3e.

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
