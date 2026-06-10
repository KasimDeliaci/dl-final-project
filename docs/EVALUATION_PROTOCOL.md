# Evaluation Protocol

Bu belge, HAM10000 transformer feature-fusion final projesinde kullanılacak değerlendirme kurallarını tanımlar.

## Primary Metric

Ana yorum metriği macro-F1'dır.

Neden: HAM10000 sınıf dağılımı dengesizdir. Accuracy çoğunluk sınıfı olan `nv` tarafına fazla duyarlı olabilir. Macro-F1 her sınıfa eşit ağırlık verdiği için minority-class davranışını daha görünür yapar.

## Required Metrics

Ödev gereği raporlanacak metrikler:

- accuracy
- precision
- recall
- F1-score

Bu projede ayrıca şunlar standart üretilecektir:

- macro precision
- macro recall
- macro F1
- weighted F1
- per-class precision/recall/F1
- confusion matrix

## Split Rules

Hedef split:

- stratified train/validation/test split,
- mümkünse yaklaşık `70/15/15`,
- HAM10000 metadata izin veriyorsa `lesion_id` bazında group-aware ayrım.

Leakage kuralı:

- Aynı lesion ID birden fazla split içinde yer almamalıdır.
- Eğer minority-class kısıtları nedeniyle ideal split mümkün olmazsa, compromise `docs/DATASET_AUDIT.md` içinde açıkça belgelenmelidir.

## Validation/Test Discipline

- Model, hyperparameter, fusion weight ve checkpoint seçimi validation veya validation-CV ile yapılır.
- Test seti yalnız final audit için kullanılır.
- Testte iyi görünen ama validation kuralıyla seçilmeyen model final sonuç olarak sunulmaz.
- Exploratory test-probe yapılırsa açıkça `diagnostic only` olarak işaretlenir ve final karar gerekçesi yapılmaz.

## Comparable Conditions

İki sonucu doğrudan karşılaştırmak için şu koşullar aynı olmalıdır:

- dataset version,
- split files,
- image preprocessing policy,
- backbone feature source (`frozen` veya `finetuned`),
- feature extraction token/pooling policy,
- MLP classifier recipe,
- metric calculation code,
- seed veya seed policy.

## Minimum Experiment Matrix

Frozen feature extraction:

- 3 single-backbone runs,
- 3 pairwise concatenation runs,
- 1 three-backbone concatenation run,
- 3 pairwise weighted fusion runs,
- 1 three-backbone weighted fusion run.

Fine-tuning:

- 3 single-backbone fine-tuned runs,
- 1 representative pairwise fusion run,
- 1 representative three-backbone fusion run.

Bu minimum matris, proje süresi ve Colab maliyetine göre daraltılabilir; daraltma yapılırsa karar `docs/DECISIONS.md` içine yazılmalıdır.

## Reporting Rules

Her run şu bilgileri taşımalıdır:

- run ID,
- backbone name,
- model family (`ViT`, `Swin`, `DeiT`, vb.),
- feature source (`frozen` veya `finetuned`),
- feature token/pooling policy,
- fusion method,
- feature dimension,
- MLP recipe,
- seed,
- split files,
- selected-by metric,
- runtime,
- primary metric,
- generated artifacts.

## Medical Framing

Bu proje klinik teşhis sistemi olarak sunulmayacaktır. Kullanılacak çerçeve:

> HAM10000 üzerinde benchmark dermoscopic image classification.

Klinik kullanıma hazır, teşhis güvenilirliği, hasta güvenliği veya deployment iddiası yapılmamalıdır.

