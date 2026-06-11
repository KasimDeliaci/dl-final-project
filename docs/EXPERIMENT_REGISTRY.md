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

Status: planned

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

Implementation plan: Sprint 4 is registered in `docs/exec-plans/active/sprint-4-finetuned-transformer-features.md`. Partial fine-tuning policy is `vit_b16` last 2 transformer blocks plus norm/head, `swin_tiny` last Swin stage plus norm/head, and `beit_base` last 2 transformer blocks plus fc_norm/norm/head. Canonical Sprint 4 cache extraction writes train and validation caches only under `artifacts/features/ham10000/finetuned/<backbone>/`. Test metrics are not computed in Sprint 4.

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
