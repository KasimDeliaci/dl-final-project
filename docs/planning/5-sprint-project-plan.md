# Five-Sprint Project Plan

Bu plan, `dl-final-projesi-2026.docx` gereksinimlerini HAM10000 üzerinde transformer tabanlı feature extraction ve feature fusion çalışmasına dönüştürmek için hazırlanmıştır. Amaç yalnızca üç transformer backbone'u çalıştırıp en yüksek accuracy'yi bulmak değildir; amaç lesion-aware, validation-disiplinli ve artifact-kanıtlı bir transformer representation ve feature complementarity çalışması üretmektir.

Bu belge overview seviyesindedir. Her sprint başlamadan önce `docs/exec-plans/active/` altında o sprint için daha dar, uygulanabilir execution plan açılmalıdır. Sprint tamamlanınca plan `docs/exec-plans/completed/` altına taşınmalı ve `docs/report_notes/` altında kısa sonuç notu yazılmalıdır.

## Project Contract

Sabit kapsam:

- Dataset: HAM10000, 7-class dermoscopic image classification benchmark.
- Backbone seti: Vanilla ViT, Swin Transformer, DeiT III-Small.
- Feature extraction: Her backbone'da classification head kaldırılarak feature vector alınır.
- Fusion: Concatenation fusion ve weighted fusion.
- Classifier: Single veya fused feature vector üzerinde MLP classifier.
- Karşılaştırmalar: Single-backbone, pairwise fusion ve three-backbone fusion.
- Transfer learning: Frozen feature extraction ve son transformer blokları fine-tuning karşılaştırması.
- Metrikler: Accuracy, precision, recall, F1-score; ana yorum metriği macro-F1.
- Evaluation discipline: Model, hyperparameter, checkpoint ve fusion weight seçimi validation üzerinden yapılır; test seti final audit veya açıkça işaretlenmiş diagnostic audit dışında kullanılmaz.
- Framing: Klinik teşhis iddiası yoktur; anlatı "benchmark dermoscopic image classification on HAM10000" çizgisinde kalır.

## Sprint Overview

1. **Sprint 1:** Dataset, split, repo/environment foundation ve araştırma çerçevesini kilitle.
2. **Sprint 2:** Frozen transformer feature extraction ve single-backbone MLP baselines üret.
3. **Sprint 3:** Frozen feature fusion deneyleriyle representation complementarity ölç.
4. **Sprint 4:** Son transformer blokları fine-tuning ve fine-tuned feature comparison yap.
5. **Sprint 5:** Validation'a göre final seçim, test audit, analiz, rapor ve sunum.

## Sprint 1 - Dataset, Split, Foundation, Research Framing

### Goal

Bu sprintin amacı model eğitmek değil, ölçüm zemininin güvenilir olduğunu kanıtlamaktır. HAM10000 metadata ve image mapping doğrulanır, lesion-aware train/validation/test split politikası kurulup denetlenir, evaluation protocol kilitlenir ve repo çalıştırılabilir hale getirilir.

Ana soru:

> Sonraki deneylerde görülen performans farkları gerçekten model/fusion farkı mı, yoksa split leakage, yanlış preprocessing, label mapping hatası veya artifact karmaşası mı?

### Main Workstreams

- Assignment contract ve scope'u kilitle: dataset, backbone seti, fusion yöntemleri, MLP classifier, metric seti ve test discipline.
- HAM10000 metadata audit yap: `image_id`, `lesion_id`, `dx`, image path eşleşmesi, duplicate/missing image, class distribution.
- 7-class label mapping'i sabitle: `akiec`, `bcc`, `bkl`, `df`, `mel`, `nv`, `vasc`.
- Lesion-aware split üret veya doğrula: aynı `lesion_id` birden fazla split'e girmemeli.
- Target split oranı: yaklaşık `70/15/15`; rare-class tradeoff varsa `docs/DATASET_AUDIT.md` içinde açıklanmalı.
- DataLoader smoke test hazırla: bir batch yüklenir, shape/label/transform doğrulanır.
- Thin notebook policy'yi koru: notebook sadece Colab launcher, core logic `src/` ve `scripts/` altında.

### Key Decisions

- Canonical split lesion-aware olacak; random image split ana sonuç olarak kullanılmayacak.
- Ana yorum metriği macro-F1 olacak.
- Metadata alanları model input'u olmayacak; yalnız audit ve error analysis için kullanılabilir.
- Transformer preprocessing başlangıçta pretrained weight'lerin beklediği image size ve normalization ile uyumlu olacak.
- Test seti model selection için kullanılmayacak.
- Run ID standardı baştan okunabilir tutulacak, örnek: `s2_frozen_vit_mlp_seed42`.

### Verification Gates

- Metadata read başarılı.
- Her metadata row için image path resolve ediliyor.
- Label mapping deterministic ve 7 class sabit.
- Her image yalnız tek split'te.
- Cross-split lesion leakage sıfır.
- Her split için class distribution raporlandı.
- Rare class'ların validation/testte tamamen kaybolmadığı doğrulandı veya compromise kaydedildi.
- Split script aynı seed ile aynı dosyaları üretiyor.
- DataLoader smoke test geçiyor.
- Artifact path'leri `.gitignore` ile Git dışında.
- `docs/DATASET_AUDIT.md`, `docs/DECISIONS.md` ve `docs/COMMANDS.md` güncellendi.

### Expected Artifacts

```text
configs/dataset/ham10000_lesion_seed42.yaml
data/splits/
docs/DATASET_AUDIT.md
docs/report_notes/sprint1_dataset_split.md
artifacts/report_assets/tables/class_distribution.csv
artifacts/report_assets/figures/class_distribution.png
```

Execution-level artifact listesi sprint planında netleştirilecektir.

### Interim Sprint Report

Hazır rapor/sunum cümlesi:

> HAM10000 metadata and image paths were validated, and a fixed lesion-aware train/validation/test split was created. All images belonging to the same lesion were assigned to a single split, yielding zero cross-split lesion leakage.

Türkçe rapor dili:

> Bu aşamada HAM10000 veri seti denetlenmiş ve aynı lezyona ait görüntülerin farklı splitlere sızmasını engelleyen lesion-aware bir train/validation/test ayrımı oluşturulmuştur.

### Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Rare class splitte kaybolur. | Lesion-level stratification iyileştirilir; gerekirse rare-class preservation constraint eklenir ve tradeoff belgelenir. |
| Image dosyaları metadata ile eşleşmez. | Eksik image listesi çıkarılır; azsa exclude edilir, fazlaysa dataset mapping/download düzeltilir. |
| Lesion-aware split class distribution'ı random split kadar dengeli olmaz. | Bu metodolojik güç olarak raporlanır; ana benchmark leakage-controlled kalır. |
| Scaffold büyür ama çalıştırılabilir olmaz. | Sprint 1 sonunda minimum audit, split ve smoke dataloader komutları çalışır olmalı. |

### Git / Documentation Hygiene

- Büyük data Git'e alınmaz.
- Sprint 1 komutları `docs/COMMANDS.md` içine yazılır.
- Split ve evaluation kararları `docs/DECISIONS.md` içine kaydedilir.
- Training experiment'leri Sprint 2 öncesinde `docs/EXPERIMENT_REGISTRY.md` içinde açılır.

## Sprint 2 - Frozen Transformer Features and Single-Backbone MLP Baselines

### Goal

Üç transformer backbone'un tek başına frozen representation kalitesini ölçmek.

Ana soru:

> Vanilla ViT, Swin Transformer ve DeiT III-Small, HAM10000 için classifier head kaldırıldığında ne kadar güçlü feature vector üretiyor?

Bu sprint fusion'a geçmeden önce zorunlu temel kanıtı üretir.

### Main Workstreams

- Ortak transformer backbone wrapper interface'i oluştur: `features = backbone(images)`.
- Her backbone için pretrained weights, classification head removal, eval mode, gradient kapatma ve feature shape manifest'i.
- ViT/DeiT için canonical token/pooling policy seç: başlangıçta tek policy seçilmeli, ablation minimum scope dışında kalmalı.
- Swin için pooled vector extraction policy belirle.
- Frozen feature cache üret: train/validation ve audit discipline uygunsa test cache. Test metric Sprint 5'e saklanır.
- Feature normalization policy belirle: train-only fit edilen scaler, her backbone feature block için ayrı uygulanabilir.
- Single-backbone MLP baseline'ları çalıştır: `frozen_vit_mlp`, `frozen_swin_mlp`, `frozen_deit3_small_mlp`.
- MLP tarafında küçük ve adil search: hidden size, dropout, LR, class-weighted CE, early stopping.
- Validation üzerinde metrics, per-class metrics, confusion matrix, prediction dump ve training curves üret.

### Key Decisions

- Her backbone için pooling/token policy.
- Feature normalization kullanılıp kullanılmayacağı.
- MLP capacity fairness: single ve fusion koşulları nasıl karşılaştırılabilir tutulacak?
- Class imbalance strategy: class-weighted CE mi, weighted sampler mı?
- Checkpoint selection: best validation macro-F1.
- Sprint 2'de test metriği raporlanmayacak.

### Verification Gates

- Her backbone forward smoke test geçti.
- Classification head kaldırıldı veya bypass edildi.
- Feature tensor row count split image count ile aynı.
- Feature cache NaN/Inf içermiyor.
- `image_id`, `lesion_id`, `split`, `label` ve feature row alignment doğrulandı.
- Scaler yalnız train split üzerinde fit edildi.
- MLP small-overfit sanity geçiyor.
- Confusion matrix label order sabit.
- Prediction dump her class için probability içeriyor.
- Her run artifact klasörü self-contained.

### Expected Artifacts

```text
configs/backbones/vit_b16.yaml
configs/backbones/swin_tiny.yaml
configs/backbones/deit3_small.yaml
configs/experiments/s2_*.yaml
artifacts/features/ham10000/frozen/
artifacts/runs/s2_frozen_*_mlp_seed42/
artifacts/report_assets/tables/single_backbone_frozen_results.csv
artifacts/report_assets/figures/frozen_single_backbone_macro_f1.png
docs/report_notes/sprint2_frozen_single.md
```

### Interim Sprint Report

Hazır rapor/sunum cümlesi:

> Frozen feature extraction was implemented for Vanilla ViT, Swin Transformer, and DeiT III-Small by removing each classification head and caching fixed feature vectors. Single-backbone MLP classifiers were trained using only train features and selected by validation macro-F1.

Türkçe yorum şablonu:

> Frozen özellik çıkarımı sonucunda transformer backbone'ların tek başına temsil kalitesi validation macro-F1 ve per-class metrics üzerinden karşılaştırılmıştır. Accuracy tek başına yorum metriği yapılmamış, sınıf dengesizliği nedeniyle macro-F1 ana metrik olarak kullanılmıştır.

### Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Transformer feature extraction yavaş olur. | Feature cache tek sefer üretilir; MLP/fusion tekrarları cache üzerinden yapılır. |
| Pooling/token policy belirsizliği. | Sprint 2'de tek canonical policy seçilir; CLS vs mean pooling ablation extension'a bırakılır. |
| MLP overfit eder. | Dropout, weight decay, early stopping ve küçük hidden size kullanılır. |
| Minority recall kötü kalır. | Class-weighted loss canonical yapılır; sampler veya focal loss extension olarak kalır. |
| Model API'leri farklı çıktı verir. | Wrapper'lar ortak smoke test ve manifest standardından geçirilir. |

### Git / Documentation Hygiene

- Her run öncesi `docs/EXPERIMENT_REGISTRY.md` kaydı açılır.
- Büyük feature cache ve run artifacts Git'e alınmaz.
- Config, scripts, tests ve report note commitlenir.

## Sprint 3 - Frozen Feature Fusion Experiments

### Goal

Feature fusion'ın gerçekten katkı sağlayıp sağlamadığını ölçmek.

Ana soru:

> Transformer backbone'ları birbirini tamamlayan temsiller mi öğreniyor, yoksa fusion sadece feature dimension artırıp overfitting mi yaratıyor?

### Main Workstreams

- Frozen feature fusion matrix'i pre-register et.
- Pairwise kombinasyonlar: ViT + Swin, ViT + DeiT III-Small, Swin + DeiT III-Small.
- Three-backbone kombinasyon: ViT + Swin + DeiT III-Small.
- Her kombinasyon için concatenation ve weighted fusion.
- Concatenation fusion: normalized feature blocks `image_id` alignment ile birleştirilir; final feature dimension manifest'e yazılır.
- Weighted fusion tanımını netleştir. Minimum viable öneri: feature-level block weighting before concatenation.
- Fusion weight selection validation üzerinden yapılır; test kullanılmaz.
- Complementarity diagnostics üret: per-class delta, confusion matrix değişimi, prediction disagreement, fusion gain vs single baseline.

### Key Decisions

- Canonical weighted fusion definition.
- Fusion weight search budget.
- MLP fairness policy.
- Primary fusion selection metric: validation macro-F1.
- Tie-breaker: minority-class recall, macro recall, overfitting gap ve model simplicity.
- Sprint 3'te test audit yok.

### Verification Gates

- Pairwise/triple feature alignment `image_id` üzerinden doğrulandı.
- Her fusion cache row count doğru.
- Weighted fusion weights yalnız validation ile seçildi.
- Scalers train-only fit edildi.
- MLP config budget single-backbone koşullarıyla karşılaştırılabilir.
- Her fusion run için validation prediction dump var.
- Confusion matrix ve per-class metrics üretildi.
- Fusion işe yaramasa bile artifact standardı tam.
- `docs/EXPERIMENT_REGISTRY.md` her fusion run için güncellendi.

### Expected Artifacts

```text
configs/experiments/s3_frozen_concat_*.yaml
configs/experiments/s3_frozen_weighted_*.yaml
artifacts/features/ham10000/frozen_fusion/
artifacts/runs/s3_frozen_*_seed42/
artifacts/report_assets/tables/fusion_vs_single_val_table.csv
artifacts/report_assets/tables/per_class_delta_table.csv
artifacts/report_assets/figures/macro_f1_frozen_single_vs_fusion.png
artifacts/report_assets/figures/per_class_f1_delta_fusion.png
docs/report_notes/sprint3_frozen_fusion.md
```

### Interim Sprint Report

Hazır rapor/sunum cümlesi:

> Frozen feature fusion experiments were conducted using pairwise and three-backbone combinations of Vanilla ViT, Swin Transformer, and DeiT III-Small. Concatenation and validation-selected weighted fusion were compared against the strongest single-backbone baseline.

Fusion işe yararsa:

> Pairwise or three-backbone fusion improved validation macro-F1 over the strongest single frozen backbone, suggesting partial complementarity between transformer representations.

Fusion işe yaramazsa:

> Fusion did not consistently improve validation macro-F1 over the strongest single backbone, suggesting redundant representations, high-dimensional overfitting risk, or insufficient fusion capacity under the current protocol.

### Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Fusion işe yaramaz. | Bu negative result olarak raporlanır; "more backbones is not automatically better" tartışılır. |
| Concatenation çok yüksek dimension üretir. | Daha küçük MLP, dropout, bottleneck veya projection extension düşünülür. |
| Weighted fusion belirsiz kalır. | Tanım `docs/DECISIONS.md` içinde açıkça yazılır. |
| Weight grid küçük kalır. | Küçük ama önceden tanımlı validation grid savunulur; test tuning yapılmaz. |
| Three-backbone pairwise'dan kötü çıkar. | Bu da rapor değeri taşır ve complementarity tartışmasına girer. |

### Git / Documentation Hygiene

- Fusion matrix başlamadan registry'ye girilir.
- `docs/DECISIONS.md` weighted fusion definition, weight selection policy ve MLP fairness policy içerir.
- Fused feature cache Git dışında kalır.

## Sprint 4 - Fine-Tuning Last Transformer Blocks and Fine-Tuned Feature Comparison

### Goal

Frozen representation yerine sınırlı domain adaptation yapılınca transformer feature kalitesi değişiyor mu sorusunu cevaplamak.

Ana sorular:

> Son transformer bloklarını fine-tune etmek, frozen pretrained features'a göre macro-F1 ve per-class recall açısından anlamlı iyileşme sağlıyor mu?

> Fine-tuning sonrası fusion hâlâ faydalı mı, yoksa tek bir güçlü fine-tuned backbone fusion ihtiyacını azaltıyor mu?

### Main Workstreams

- Her backbone için fine-tuning protocol tasarla: pretrained model, temporary 7-class head, son transformer blokları trainable, önceki bloklar frozen.
- Checkpoint validation macro-F1 ile seçilir.
- Seçilen checkpoint'ten classification head kaldırılır ve fine-tuned feature cache üretilir.
- Aynı MLP protocol ile fine-tuned single-backbone classifier eğitilir.
- Trainability audit kaydedilir: total/trainable parameters, unfrozen blocks, LR, batch size, image size, epochs, early stopping, best validation checkpoint.
- Fine-tuned single-backbone comparison yapılır.
- Compute izin verirse fine-tuned fusion shortlist çalıştırılır: best pair concat/weighted ve all-three concat/weighted.
- Frozen vs fine-tuned fairness korunur: aynı split, metric script, MLP protocol ve validation selection rule.

### Key Decisions

- Unfreeze depth: başlangıçta son bloklar; exact sayı sprint execution plan'da model API'ye göre kilitlenir.
- LR policy: backbone için küçük LR, temporary head için gerekirse farklı LR.
- Train augmentation: train split için kontrollü, validation/test deterministic.
- Class imbalance: class-weighted CE canonical tutulabilir.
- Fine-tuned fusion matrix full mü shortlist mi?
- Compute budget cap açık yazılmalı.

### Verification Gates

- Sadece hedef son bloklar ve temporary head trainable.
- Frozen blokların gradient almadığı doğrulandı.
- Checkpoint selection validation macro-F1 ile yapıldı.
- Early stopping test setini görmedi.
- Train/validation transforms ayrıldı.
- Fine-tuned feature cache manifest checkpoint hash veya checkpoint metadata içeriyor.
- Fine-tuned MLP protocol frozen MLP ile karşılaştırılabilir.
- Training curves overfitting açısından incelendi.
- Test evaluation Sprint 5'e kaldı.
- Failed/OOM run'lar registry'de dürüstçe işaretlendi.

### Expected Artifacts

```text
configs/experiments/s4_ft_*_last_blocks.yaml
configs/experiments/s4_ft_*_mlp.yaml
artifacts/checkpoints/ham10000/s4_ft_*/
artifacts/features/ham10000/finetuned/
artifacts/runs/s4_ft_*_seed42/
artifacts/report_assets/tables/frozen_vs_finetuned_single_val.csv
artifacts/report_assets/tables/frozen_vs_finetuned_fusion_val.csv
artifacts/report_assets/tables/trainable_params_table.csv
artifacts/report_assets/figures/frozen_vs_finetuned_macro_f1.png
artifacts/report_assets/figures/fine_tuning_training_curves.png
docs/report_notes/sprint4_finetuning.md
```

### Interim Sprint Report

Hazır rapor/sunum cümlesi:

> The last transformer blocks of each backbone were fine-tuned under the same lesion-aware split and selected by validation macro-F1. Fine-tuned checkpoints were then converted back into feature extractors, allowing a fair comparison between frozen and adapted transformer representations using the same MLP classifier protocol.

Olası sonuç dili:

> Fine-tuning improved validation macro-F1 for selected backbones but also increased training cost and overfitting risk, so frozen vs fine-tuned results were interpreted together with per-class metrics and runtime.

### Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Colab OOM. | Smaller batch, mixed precision, gradient accumulation, freeze more layers, one backbone at a time. |
| Fine-tuning çok uzun sürer. | Epoch cap, early stopping, son 1-2 block, full fusion yerine shortlist. |
| Fine-tuning overfit eder. | Lower LR, weight decay, dropout, less unfreezing. |
| Fine-tuned features frozen'dan kötü çıkar. | Negative result olarak raporlanır; pretrained transformers'ın frozen extractor gücü tartışılır. |
| DeiT III-Small API problemi. | Exact implementation source registry'de sabitlenir; wrapper izole düzeltilir. |

### Git / Documentation Hygiene

- Fine-tuning run'ları registry'ye başlamadan yazılır.
- OOM/failed run'lar silinmez; status ve sebep kaydedilir.
- `docs/DECISIONS.md` unfreeze depth, LR policy ve fine-tuned feature extraction policy içerir.
- Checkpoint ve feature caches Git'e alınmaz.

## Sprint 5 - Final Selection, Audit, Analysis, Report, Presentation

### Goal

Yeni model denemek değil; validation'a göre seçilmiş adayları final test audit'e sokmak, sonuçları bilimsel olarak analiz etmek ve teslim edilebilir rapor/sunum haline getirmek.

Ana soru:

> Tüm validation kanıtlarına göre en savunulabilir model/fusion/transfer learning sonucu hangisi, test audit bunu destekliyor mu, ve sonuçları nasıl dürüstçe tartışıyoruz?

### Main Workstreams

- Sprint 2-4 validation sonuçlarını tek master tabloda birleştir.
- Final selection test seti görülmeden yapılır ve `docs/DECISIONS.md` içine yazılır.
- Primary selection: validation macro-F1.
- Secondary diagnostics: macro recall, minority-class F1, per-class recall, overfitting gap, complexity, training time, validation curve stability.
- Test audit listesi testten önce kilitlenir.
- Önerilen audit seti: best frozen single, best frozen fusion, best fine-tuned single, best fine-tuned fusion, final validation-selected best.
- Test sonuçları görüldükten sonra yeni model seçimi yapılmaz.
- Final metrics: accuracy, macro precision/recall/F1, weighted-F1, per-class metrics, confusion matrix, prediction dump, runtime.
- Error analysis: class pair confusions, minority-class behavior, fusion class gains, validation/test consistency.
- Final report ve presentation hazırlanır.

### Key Decisions

- Final candidate list testten önce kilitlenir.
- Headline metric macro-F1.
- Test table scope: final model only mi, pre-registered diagnostic audit mi?
- Clinical claim yok.
- Kötü çıkan fusion/fine-tuning sonuçları saklanmaz; tartışılır.
- Submission package kapsamı netleşir.

### Verification Gates

- Final selection document testten önce commitlendi.
- Test evaluation script tek audit policy ile çalıştırıldı.
- Test metrics reproducible.
- Prediction dumps mevcut.
- Confusion matrix label order doğru.
- Validation ve test tabloları ayrı.
- Rapor içindeki her tablo/figürün kaynak artifact'i belli.
- `README.md` reproduction instructions içeriyor.
- Submission klasörü büyük checkpoint/feature cache/prediction dump dosyalarını yanlışlıkla içermiyor.

### Expected Artifacts

```text
artifacts/report_assets/tables/validation_master_table.csv
artifacts/report_assets/tables/test_audit_metrics.csv
artifacts/report_assets/tables/test_per_class_metrics.csv
artifacts/report_assets/figures/best_model_confusion_matrix.png
artifacts/report_assets/figures/per_class_f1_best_model.png
artifacts/report_assets/figures/frozen_vs_finetuned_summary.png
docs/report_notes/sprint5_final_audit.md
docs/report_notes/final_discussion_points.md
reports/final_report/
reports/presentation/
submission/
```

### Interim Sprint Report

Hazır rapor/sunum cümlesi:

> Final model selection was performed using validation macro-F1 before accessing test labels. A pre-registered set of validation-selected candidates was then evaluated on the held-out test split as a final audit.

Türkçe rapor dili:

> Final model seçimi test seti görülmeden, validation macro-F1 temel alınarak yapılmıştır. Test sonuçları yalnızca validation ile seçilmiş adayların son denetimi olarak raporlanmış; model veya fusion weight seçimi test performansına göre değiştirilmemiştir.

### Risks and Fallbacks

| Risk | Fallback |
|---|---|
| Test sonucu validation'dan düşük çıkar. | Sonuç saklanmaz; validation/test gap, class imbalance ve split variance tartışılır. |
| En iyi test sonucu validation'da seçilmeyen modelde görünür. | Headline model değiştirilmez; diagnostic olarak yazılır. |
| Rapor tablo listesine dönüşür. | Ana gövdeye 4-6 ana tablo/figür; detaylar appendix. |
| Sunum süresi yetmez. | Tek ana mesaj korunur: leakage-controlled HAM10000 split altında transformer feature extraction, fusion ve fine-tuning macro-F1/per-class metrics ile değerlendirildi. |

### Git / Documentation Hygiene

- Final report ile code aynı commit hash'e bağlanır.
- `docs/EXPERIMENT_REGISTRY.md` tüm run status'larını içerir.
- `docs/DECISIONS.md` final selection kararını içerir.
- `docs/COMMANDS.md` final reproduction commands içerir.
- Submission öncesi `git status` temiz olmalı.

## Minimum Viable Experiment Matrix

### Sprint 2 - Frozen Single-Backbone Baselines

| Run group | Method | Count |
|---|---:|---:|
| Vanilla ViT frozen features + MLP | single-backbone | 1 |
| Swin frozen features + MLP | single-backbone | 1 |
| DeiT III-Small frozen features + MLP | single-backbone | 1 |

Total: **3 run groups**.

### Sprint 3 - Frozen Fusion

| Run group | Fusion | Count |
|---|---:|---:|
| ViT + Swin | concat + weighted | 2 |
| ViT + DeiT III-Small | concat + weighted | 2 |
| Swin + DeiT III-Small | concat + weighted | 2 |
| ViT + Swin + DeiT III-Small | concat + weighted | 2 |

Total: **8 run groups**.

### Sprint 4 - Fine-Tuned Features

| Run group | Method | Count |
|---|---:|---:|
| Fine-tuned ViT last blocks + MLP | single-backbone | 1 |
| Fine-tuned Swin last blocks + MLP | single-backbone | 1 |
| Fine-tuned DeiT III-Small last blocks + MLP | single-backbone | 1 |
| Best fine-tuned pair fusion | concat + weighted | 2 |
| Fine-tuned all-three fusion | concat + weighted | 2 |

Total: **7 run groups**.

### Sprint 5 - Final Audit

Recommended pre-registered test audit set:

| Candidate | Why included? |
|---|---|
| Best frozen single | Single-backbone baseline |
| Best frozen fusion | Fusion effect without fine-tuning |
| Best fine-tuned single | Transfer learning effect |
| Best fine-tuned fusion | Combined effect |
| Final validation-selected best | Headline model |

Total validation-level scope: approximately **18 run groups**. This is dense but manageable because Sprint 2-3 primarily reuse cached features; Sprint 4 fine-tuning is the compute bottleneck.

## Publication-Oriented Extensions

Bunlar zorunlu değildir; zaman ve compute kalırsa projeyi paper yönüne yaklaştırabilir.

- Multi-seed robustness: top validation candidates için 3 seed, mean +/- std macro-F1.
- Confidence intervals: final audit için lesion-level bootstrap CI.
- Feature complementarity analysis: prediction disagreement, class-wise error overlap, cosine similarity, CKA-style representation similarity, UMAP/PCA.
- Pooling policy ablation: ViT/DeiT için CLS, mean pooling, CLS+mean concat.
- Projection-based weighted fusion: each backbone -> projection -> shared dimension -> learnable/gated weighted sum.
- Class imbalance ablation: unweighted CE, class-weighted CE, weighted sampler, focal loss.
- Diagnostic random image split vs lesion-aware split comparison, yalnız appendix/diagnostic olarak.
- Metadata-based error analysis: `dx_type`, `localization`, `dataset`, age/sex missingness; model input'u değil, yalnız diagnostic.

## Things To Avoid

- Test seti üzerinden model, checkpoint, hyperparameter veya fusion weight seçmek.
- Random image split'i headline yapmak.
- Aynı `lesion_id`'nin train ve testte bulunmasına izin vermek.
- Accuracy'yi tek ana metrik gibi sunmak.
- Macro-F1 düşükken modeli genel olarak başarılı ilan etmek.
- Klinik teşhis veya deployment iddiası kurmak.
- Feature cache alignment'ı `image_id` ile doğrulamamak.
- Train-only scaler/normalization kuralını bozmak.
- Validation transform ile train augmentation'ı karıştırmak.
- Fine-tuning'de hangi layer/block'un trainable olduğunu loglamamak.
- Fusion input dimension artışıyla gelen performans artışını doğrudan "daha iyi temsil" diye yorumlamak.
- Weighted fusion tanımını belirsiz bırakmak.
- Büyük checkpoint, feature cache, prediction dump veya raw data dosyalarını Git'e almak.
- Notebook içine core training logic gömmek.
- Run ID'siz artifact üretmek.
- Raporu yalnız sonuç tablosuna indirgemek.

## Reporting Plan

Final raporun ana hikayesi:

> Bu çalışma, HAM10000 benchmark üzerinde üç farklı transformer backbone'dan çıkarılan frozen ve fine-tuned feature representation'ların MLP classifier ile sınıflandırma performansını karşılaştırmış; concatenation ve weighted feature fusion yöntemlerinin single-backbone temsillere göre katkısını lesion-aware split ve validation-disiplinli model selection altında incelemiştir.

Ana rapor bölümleri:

1. Giriş: HAM10000 benchmark, transformer feature extraction ve fusion motivasyonu; klinik iddia yok.
2. Dataset and Split: metadata, 7 class, class imbalance, lesion-aware split, leakage prevention.
3. Method: ViT, Swin, DeiT III-Small, feature extraction policy, MLP, concat/weighted fusion, frozen/fine-tuned protocol.
4. Experimental Protocol: validation/test discipline, macro-F1, artifact/reproducibility rules.
5. Results: frozen single, frozen fusion, fine-tuned single/fusion, final test audit, per-class metrics.
6. Discussion: backbone strength, fusion effect, fine-tuning benefit/cost, class imbalance, limitations.
7. Conclusion: en savunulabilir sonuç, öğrenilenler, future work.

Gerekli ana tablolar:

- Dataset class distribution table.
- Split distribution table.
- Frozen single-backbone validation results.
- Frozen fusion validation results.
- Fine-tuned single/fusion validation results.
- Final test audit results.
- Per-class metrics for best model.
- Training time / trainable parameters table.

Gerekli ana görseller:

- Pipeline diagram.
- Class distribution plot.
- Experiment matrix diagram.
- Validation macro-F1 bar chart.
- Frozen vs fine-tuned comparison chart.
- Fusion gain chart.
- Best model confusion matrix.
- Per-class F1 bar chart.

## Scientific Framing

Bu proje şunu ölçer:

> Pretrained transformer backbone'ların HAM10000 görüntülerinden çıkardığı feature representation'lar, MLP classifier altında 7-class benchmark classification için ne kadar ayırt edici ve birbirini tamamlayıcıdır?

Bu proje şunları ölçmez:

- Klinik tanı güvenilirliği.
- Gerçek hastane deployment performansı.
- Dermatolog seviyesinde karar verme.
- Genel deri kanseri teşhis başarısı.

Fusion iyi çıkarsa "complementary information taşıyor olabilir" dili kullanılmalıdır. Fusion kötü çıkarsa redundant representations, high-dimensional overfitting, sınırlı MLP capacity veya basit weighted fusion grid gibi açıklamalar tartışılabilir. Kesin mimari üstünlük iddiasından kaçınılmalıdır.

Frozen vs fine-tuned comparison aynı split, preprocessing family, metric script, validation selection rule ve karşılaştırılabilir MLP budget ile yapılmalıdır. Fine-tuning daha fazla compute kullandığı için sonuçlar training time ve overfitting riskiyle birlikte yorumlanmalıdır.

## Recommended First Next Step

Model koduna başlamadan önce Sprint 1 execution plan'ı açılmalıdır:

```text
docs/exec-plans/active/SPRINT1_DATA_SPLIT_FOUNDATION.md
```

İlk uygulanabilir hedef:

> HAM10000 metadata audit + lesion-aware split generator + zero leakage verification.

Bu tamamlanmadan ViT/Swin/DeiT feature extraction veya training'e geçilmemelidir; projenin bilimsel savunulabilirliği modelden önce split ve evaluation discipline üzerine kurulacaktır.
