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

Status: planned

Question: ViT, Swin ve ek transformer backbone frozen feature extractor olarak tek başına nasıl davranıyor?

Hypothesis: Farklı transformer mimarileri aynı dataset üzerinde farklı class-level güçlü/zayıf yönler gösterecektir.

Changed variable: Backbone architecture.

Fixed controls: Split, preprocessing, frozen feature extraction, MLP classifier recipe.

Selection rule: MLP checkpoint validation macro-F1 ile seçilir; test yalnız audit.

Expected failure mode: Feature dimension veya token pooling farkları nedeniyle MLP capacity farklı modellere eşit davranmayabilir.

Required artifacts:

- frozen feature cache,
- run config JSON,
- metrics summary,
- per-class metrics,
- confusion matrix,
- prediction dump.

Report role: Single-backbone representation quality baseline.

### E2 - Frozen Feature Fusion

Status: planned

Question: Transformer backbone feature'ları concat veya weighted fusion ile tamamlayıcı sinyal üretir mi?

Hypothesis: Pairwise veya three-backbone fusion, tek backbone'a göre macro-F1 artışı sağlayabilir; ancak bu artış feature complementarity'ye bağlıdır.

Changed variable: Backbone combination and fusion method.

Fixed controls: Frozen feature caches, split, MLP recipe, evaluation protocol.

Selection rule: Fusion candidate selection validation macro-F1 ile yapılır.

Expected failure mode: Concatenation feature dimension'ı büyütüp overfitting yaratabilir; weighted fusion zayıf backbone'u fazla bastırabilir veya tamamlayıcı sinyali kaybedebilir.

Required artifacts:

- fusion metrics table,
- pairwise comparison table,
- per-class metrics,
- feature dimension log,
- prediction dump.

Report role: Feature fusion and complementarity analysis.

### E3 - Fine-Tuning Last Transformer Blocks

Status: planned

Question: Frozen feature extraction yerine son transformer bloklarını fine-tune etmek HAM10000 temsillerini iyileştirir mi?

Hypothesis: Son transformer bloklarının kontrollü fine-tuning'i domain-specific dermoscopic representation kalitesini artırabilir; ancak küçük/dengesiz dataset nedeniyle overfitting riski yüksektir.

Changed variable: Transfer learning policy.

Fixed controls: Split, backbone family, classifier evaluation protocol.

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
