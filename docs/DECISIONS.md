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

## D014 - Sprint 3 Concat Untouched Policy

Karar: Sprint 3 `concat` fusion run'larında backbone feature block'ları yalnız train-only StandardScaler ile normalize edilecek ve doğrudan yan yana eklenecektir. Concat üstüne PCA, LDA veya ek projection uygulanmayacaktır.

Gerekçe: Concat fusion'ın amacı feature bilgisini sıkıştırmadan, feature dimension artışının tek başına complementarity sağlayıp sağlamadığını ölçmektir. PCA veya projection eklemek concat koşulunu weighted/projection koşullarıyla karıştırır.

## D015 - Sprint 3 Weighted Learned 512 Policy

Karar: Canonical weighted fusion varyantı `weighted_learned_512` olarak tanımlanmıştır. Her backbone feature block'u trainable `Linear(input_dim -> 512)` projection'dan geçer; global backbone logits softmax ile normalize edilir; projected block'lar weighted sum ile birleştirilir ve MLP classifier'a verilir.

Gerekçe: Bu tanım eski `dl-assignment` Sprint 3 weighted fusion akışını transformer feature'larına taşır. Global learned weights raporlanabilir bir diagnostic sağlar, ancak projection ve classifier ile birlikte optimize edildiği için doğrudan backbone kalite sıralaması olarak yorumlanmayacaktır.

## D016 - Sprint 3 Weighted PCA 384 Policy

Karar: `weighted_pca_384` secondary diagnostic olarak eklenmiştir. PCA yalnız weighted fusion varyantında kullanılır; her backbone block'u train-only StandardScaler sonrası train split üzerinde fit edilen PCA ile 384 boyuta dönüştürülür. Validation'a aynı train-fitted PCA transform uygulanır. Labels PCA fit için kullanılmaz.

Gerekçe: DeiT III-Small feature dim'i 384 olduğu için ortak latent boyut 384 seçilmiştir. Bu varyant, trainable projection yerine unsupervised compression kullanıldığında weighted fusion'ın nasıl davrandığını ölçer. Concat'in yerine geçmez ve canonical weighted learned policy'yi değiştirmez.

## D017 - Sprint 3 Validation-Only Selection And Test Usage

Karar: Sprint 3'te model, checkpoint, fusion method ve learned weight yorumları validation macro-F1 üzerinden yapılır. Test metrics hesaplanmamıştır ve test seti Sprint 5 final audit'e bırakılmıştır.

Gerekçe: Fusion matrix, final model seçiminden önceki temsil/complementarity çalışmasıdır. Test sonuçlarını bu aşamada kullanmak final audit disiplinini zayıflatır.

## D018 - Sprint 3 Fusion Artifact Policy

Karar: Her Sprint 3 fusion run'ı standart run artifact bundle üretir: `run_config.json`, `metrics_summary.csv/json`, `per_class_metrics.csv`, `confusion_matrix.csv/png`, `predictions.csv`, `training_history.csv`, `checkpoint_metadata.json`, `model.pt`, `scaler_stats.npz`, `preprocessing_metadata.json`, `fusion_metadata.json`, `runtime_metadata.json` ve weighted run'lar için `fusion_weights.csv`.

Gerekçe: Fusion iyi veya kötü çıksa bile sonuçların raporda savunulabilmesi için row alignment, train-only scaler/PCA policy, learned weights, prediction dump ve confusion matrix gibi kanıtlar run ile birlikte kalmalıdır. Büyük run artifact'leri Git dışında tutulur.

## D019 - BEiT-Expanded E2 Fusion Matrix

Karar: BEiT-Base, Sprint 2 single-backbone screening'de canonical üçüncü backbone seçilmemiş olmasına rağmen E2 frozen fusion matrix içine expanded alternative olarak alınmıştır. BEiT-expanded E2 matrix, `vit_b16+beit_base`, `swin_tiny+beit_base` ve `vit_b16+swin_tiny+beit_base` kombinasyonlarını `concat`, `weighted_learned_512` ve `weighted_pca_384` method'larıyla validation-only çalıştırır.

Gerekçe: BEiT single-backbone screening'de zayıf kalmıştır (`0.4759` validation macro-F1), ancak masked image modeling pretraining çizgisi nedeniyle DeiT III-Small'a göre farklı bir tamamlayıcı sinyal taşıyabilir. Pairwise BEiT fusion koşulları ViT single baseline'ı geçmemiştir; buna rağmen `vit_b16+swin_tiny+beit_base concat` validation macro-F1 `0.6988` ile E2'nin en iyi validation sonucunu vermiştir. Bu sonuç, single-backbone strength ile fusion complementarity'nin aynı şey olmadığını tartışmak için kullanılacaktır. Test seti kullanılmamıştır.

## D020 - Representation Similarity Diagnostic Policy

Karar: E2 complementarity yorumu için representation similarity analizi validation split üzerinde yapılacaktır. Her backbone için StandardScaler yalnız train cache üzerinde fit edilir, validation feature'ları transform edilir, sample-by-sample cosine similarity matrisi çıkarılır ve backbone çiftleri bu similarity matrislerinin üst üçgenleri arasındaki Pearson correlation ile karşılaştırılır. Test split kullanılmaz.

Gerekçe: Backbone feature dimension'ları farklı olduğu için doğrudan feature correlation uygun değildir. Sample-similarity correlation, farklı boyutlardaki temsillerin örnekleri benzer şekilde gruplayıp gruplamadığını ölçer. Düşük representation similarity, daha yüksek complementarity sinyali olarak yorumlanabilir; ancak tek başına yüksek fusion performansı garanti etmez.

## D021 - Sprint 4 Forward Backbone Set

Karar: Sprint 4 ve ileri aşamalar için forward backbone seti `vit_b16`, `swin_tiny`, `beit_base` olarak güncellenmiştir. `deit3_small`, başlangıçta seçilmiş ve E2 planned matrix içinde değerlendirilmiş bir baseline olarak raporda kalır; ancak fine-tuning compute budget'ına taşınmaz.

Gerekçe: D004 ve D013 kararları, Sprint 2 single-backbone validation sonuçlarına göre DeiT III-Small'ı BEiT'e tercih etmişti (`0.5017` vs `0.4759`). E2 frozen fusion ve representation similarity analysis sonrasında daha güçlü bir complementarity kanıtı oluştu: `vit_b16+swin_tiny+beit_base concat` validation macro-F1 `0.6988` ile `vit_b16+swin_tiny` (`0.6947`) ve `vit_b16+swin_tiny+deit3_small` (`0.6863`) üstüne çıktı. BEiT pairwise koşulları ViT single baseline'ı geçmediği için yorum sınırlı tutulacaktır; ancak representation similarity, BEiT'in ViT ve Swin'e daha farklı feature geometry taşıdığını destekler. Bu karar validation-only evidence ile verilmiştir; test seti kullanılmamıştır.

## D022 - E2b MLP Capacity Diagnostic

Karar: E2 fusion sonuçlarının classifier capacity'ye duyarlılığını ölçmek için validation-only E2b diagnostic çalıştırılmıştır. Bu diagnostic ana E2 artifact tablolarına karıştırılmayacak, ayrı `e2b_mlp_capacity_diagnostic.csv` tablosunda raporlanacaktır.

Gerekçe: E2 ana matrix modest MLP recipe ile çalıştırılmıştır. Daha geniş MLP, high-dimensional concat feature'lar için daha uygun olabilir. E2b sonucunda `vit_b16+swin_tiny concat` deep-regularized MLP ile validation macro-F1 `0.7262` üretmiştir ve bu değer BEiT triple concat capacity variants üstündedir. Buna karşılık BEiT triple variants, DeiT triple variants'tan güçlü kalmıştır. Bu nedenle E2b, "BEiT üçüncü backbone olarak DeiT'ten daha tamamlayıcı" yorumunu korur; fakat "en iyi frozen fusion mutlaka three-backbone olmalıdır" yorumunu zayıflatır. Test seti kullanılmamıştır.
