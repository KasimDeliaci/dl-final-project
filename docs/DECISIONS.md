# Decisions

Bu dosya proje boyunca verilen bilimsel ve mühendislik kararlarını kaydetmek için kullanılacaktır.

## D001 - Ayrı Proje Klasörü

Karar: Final projesi mevcut CNN tabanlı `dl-assignment` reposunun içine taşınmadı; `/Users/arcustin2/kasim/dl-final` altında ayrı başlatıldı.

Gerekçe: Eski projenin rapor, sunum, artifact ve path yapısını bozmadan temiz bir transformer projesi kurmak daha düşük risklidir. Eski projeden yalnız workflow dersleri ve deney disiplini taşınacaktır.

## D002 - Dataset Continuity

Karar: Final projesinde de HAM10000 kullanılacaktır.

Gerekçe: Assignment dataset seçimini serbest bırakmaktadır. Aynı dataset kullanmak önceki literatür ve split deneyiminden yararlanmayı sağlar; model ailesi değiştiği için proje yeni bilimsel soruyu yine karşılar.

## D003 - Primary Metric

Karar: Ana yorum metriği macro-F1 olacaktır.

Gerekçe: HAM10000 sınıf dengesizdir. Accuracy çoğunluk sınıfı performansını fazla yansıtabilir; macro-F1 minority-class davranışını daha görünür yapar.

## D004 - Initial Transformer Set

Karar: Zorunlu Vanilla ViT ve Swin Transformer'a ek olarak üçüncü backbone `DeiT III-Small` olarak seçilmiştir.

Gerekçe: HAM10000 üzerinde transformer benchmark literatüründe DeiT III ailesi için doğrudan performans sinyali BEiT'e göre daha nettir. `DeiT III-Small`, DeiT III ailesinin data-efficient ViT çizgisini temsil ederken Colab maliyetini `Base` varyanta göre daha yönetilebilir tutar. BEiT masked image modeling nedeniyle teorik olarak farklı bir pretraining çizgisi sunsa da bu proje kapsamında üçüncü backbone olarak kullanılmayacaktır.

## D005 - Assignment Constraint Matrix

Karar: Kod yazmadan önce ödevin zorunlu deney matrisi `docs/ASSIGNMENT_BRIEF.md` içinde sabitlenmiştir.

Gerekçe: Önceki CNN projesinde olduğu gibi, implementation başlamadan önce dataset, backbone seti, feature extraction, fusion yöntemleri, MLP classifier, frozen/fine-tuned karşılaştırması, metric seti ve validation/test disiplini net olmalıdır. Bu karar, daha sonra eklenecek kodun veya deneylerin ödev kapsamını genişletirken ana gereksinimleri belirsizleştirmesini engeller.

## D006 - Sprint 1 Runtime and Dependency Scope

Karar: Sprint 1 dataset audit ve split tooling'i, ağır ML dependency'leri eklemeden `pandas` ve `pillow` ile uygulanmıştır. Proje için minimal `pyproject.toml` eklenmiş ve preferred local runner `uv run python` olarak belirlenmiştir. Grouped lesion-aware split için sklearn yerine küçük deterministik greedy split algoritması kullanılmış; class distribution görselleri matplotlib yerine SVG olarak üretilmiştir.

Gerekçe: Sprint 1'in amacı model eğitimi değil veri/split güvenilirliğini kanıtlamaktır. Hafif dependency yüzeyi, Colab/local ayrımını sade tutar ve transformer implementation kararlarını Sprint 2'ye bırakır.

## D007 - Canonical Lesion-Aware Split

Karar: HAM10000 için canonical split `lesion_id` bazlı, seed `42` ile yaklaşık `70/15/15` olarak üretilmiştir: train 7,008 image / 5,233 lesion, validation 1,504 image / 1,117 lesion, test 1,503 image / 1,120 lesion.

Gerekçe: HAM10000 aynı lezyona ait birden fazla görüntü içerebilir. Aynı `lesion_id` değerinin farklı splitlerde bulunmasını engellemek, sonraki transformer feature extraction, fusion ve fine-tuning karşılaştırmalarının leakage'den etkilenmemesi için accuracy stabilitesinden daha önceliklidir.

## D008 - Drive Artifact Root

Karar: Bu proje için büyük artifact ve Colab exchange kökü `dl-final-artifact` olarak ayrılmıştır.

Gerekçe: Eski `dl-midterm` Drive alanı ile final projesinin raw data, checkpoint, feature cache ve run output dosyalarını karıştırmamak gerekir.

## D009 - Sprint 2 Transformer Backbone Implementation

Karar: Sprint 2 frozen feature extraction için `timm` kullanılacaktır. Exact model ID'leri:

- Vanilla ViT: `vit_base_patch16_224.augreg_in21k_ft_in1k`
- Swin Transformer: `swin_tiny_patch4_window7_224.ms_in1k`
- DeiT III-Small: `deit3_small_patch16_224.fb_in1k`

Gerekçe: `timm`, üç transformer ailesini tek feature extraction interface'iyle sağlar ve `num_classes=0` ile classifier head'i bypass etmeye izin verir. Bu, eski CNN projesindeki ortak backbone wrapper disiplinini transformer modellerine taşır.

## D010 - Sprint 2 Pooling And Token Policy

Karar: ViT ve DeiT III-Small için canonical frozen feature vector CLS-token representation olacaktır (`global_pool="token"`). Swin için canonical feature vector average pooled final-stage representation olacaktır (`global_pool="avg"`). Üç model de `224x224` ImageNet normalization ile çalıştırılacaktır.

Gerekçe: Sprint 2'nin amacı pooling ablation değil, üç backbone'un tek başına frozen representation kalitesini ölçmektir. Bu nedenle her backbone için tek, açıklanabilir ve `timm` tarafından desteklenen feature extraction noktası seçilmiştir.

## D011 - Sprint 2 Feature Cache And MLP Policy

Karar: Feature cache formatı split başına `.pt` tensor payload, split başına CSV manifest ve backbone başına JSON manifest olacaktır. Cache payload `sample_id`, `image_id`, `lesion_id`, `split`, label, feature tensor ve config metadata taşır. MLP baseline'ları train-only `StandardScaler`, class-weighted cross entropy, dropout/weight decay ve early stopping kullanır. Checkpoint seçimi validation macro-F1 ile yapılır.

Gerekçe: Cache manifestleri row alignment, NaN/Inf kontrolü ve future fusion hazırlığı için gereklidir. Train-only scaler ve train-only class weights validation/test leakage riskini azaltır. MLP kapasitesi modest tutulur, böylece sonuçlar classifier büyüklüğünden çok frozen feature kalitesini yansıtır.

## D012 - Sprint 2 Test Usage Policy

Karar: Sprint 2 train ve validation feature cache'lerini model eğitimi ve seçim için kullanır. Test feature cache'i audit hazırlığı için üretilebilir, ancak Sprint 2 MLP training script'i test cache okumaz ve test metric raporlamaz.

Gerekçe: Test seti yalnız Sprint 5 final audit veya açık diagnostic audit için kullanılmalıdır. Sprint 2 sonuçları validation macro-F1 ve per-class validation metrics üzerinden yorumlanacaktır.

## D013 - BEiT Candidate Screening

Karar: BEiT-Base, üçüncü backbone slotu için validation-only candidate screening olarak denenmiş; ancak canonical üçüncü backbone olarak seçilmemiştir. Canonical set `vit_b16`, `swin_tiny`, `deit3_small` olarak kalacaktır.

Gerekçe: BEiT-Base (`beit_base_patch16_224.in22k_ft_in22k_in1k`) aynı frozen feature + MLP recipe ile validation macro-F1 `0.4759` üretmiştir. DeiT III-Small aynı protokolde `0.5017` validation macro-F1 verdiği için daha güçlü üçüncü single-backbone baseline olarak kalır. Bu seçim validation metric üzerinden yapılmıştır; test seti kullanılmamıştır.
