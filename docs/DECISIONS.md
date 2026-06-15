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

## D023 - Sprint 4 Partial Fine-Tuning Policy

Karar: Sprint 4 fine-tuning, forward backbone seti `vit_b16`, `swin_tiny`, `beit_base` ile sınırlıdır. `vit_b16` ve `beit_base` için son 2 transformer block ile norm/head trainable yapılır. `swin_tiny` için son Swin stage (`layers[-1]`) ile norm/head trainable yapılır. Tüm önceki parametreler frozen kalır.

Gerekçe: Amaç full fine-tuning değil, kontrollü domain adaptation etkisini ölçmektir. Son blok/stage politikası Colab maliyetini ve overfitting riskini sınırlar, fakat pretrained representation'ın HAM10000 benchmark dermoscopic image classification görevine sınırlı uyarlanmasına izin verir.

## D024 - Sprint 4 Validation-Only Checkpoint Selection

Karar: Fine-tuned checkpoint seçimi validation macro-F1 ile yapılır. Fine-tuning loop'u Sprint 4 kapsamında test loader almaz ve test metrics üretmez.

Gerekçe: Test seti final audit için korunmalıdır. Checkpoint veya hyperparameter seçimini test sonucu ile yapmak, Sprint 5 final audit disiplinini bozar.

## D025 - Sprint 4 Fine-Tuned Feature Cache Policy

Karar: Fine-tuned checkpoint seçildikten sonra temporary classification head kaldırılır ve Sprint 2 ile aynı feature extraction noktaları kullanılarak `feature_source=finetuned` cache'leri üretilir. Canonical Sprint 4 cache path'i `artifacts/features/ham10000/finetuned/<backbone>/` olur ve train/validation splitleri içerir.

Gerekçe: Downstream MLP ve fusion script'lerinin aynı cache contract'ını kullanması frozen vs fine-tuned karşılaştırmasını daha temiz yapar. Cache metadata checkpoint path, checkpoint selection metric, unfreeze policy ve Sprint 4 test usage policy bilgisini taşır.

## D026 - Sprint 4 Downstream Fusion Comparison Policy

Karar: Fine-tuned single-backbone MLP sonuçları frozen single baselines ile; fine-tuned concat/weighted fusion sonuçları ise hem modest frozen fusion baseline (`ViT + Swin + BEiT concat`, `0.6988`) hem de E2b stronger-MLP frozen baseline (`ViT + Swin concat + deep_reg`, `0.7262`) ile karşılaştırılacaktır.

Gerekçe: E2b, frozen fusion'ın classifier capacity'ye duyarlı olduğunu göstermiştir. Fine-tuned sonuçları yalnız modest E2 baseline ile karşılaştırmak fine-tuning gain'i olduğundan güçlü gösterebilir. Bu nedenle E2b ayrı diagnostic baseline olarak görünür tutulmalıdır.

## D027 - Sprint 4 Colab Split And Artifact Guard

Karar: Sprint 4 Colab launcher akışı, full run öncesinde HAM10000 audit ve canonical lesion-aware split üretimini yeniden çalıştırır; split row count `train=7008`, `val=1504`, `test=1503` değilse çalışmayı durdurur. Fine-tuned cache'lerde eski split kaynaklı row count uyuşmazlığı bulunursa Sprint 4 local output'ları temizlenip yeniden üretilir. `scripts/finetune_backbone.py` full run sırasında train/validation row count guard uygular; smoke/limited run'lar `--limit-per-split` ile bu guard dışında tutulur.

Gerekçe: Colab/Drive state'i eski `dl-assignment` artifact bundle'larından veya stale split dosyalarından etkilenebilir. Sprint 4 fine-tuned feature cache'leri downstream MLP/fusion script'leri tarafından strict row alignment ile okunur; bu nedenle yanlış split üstüne üretilmiş cache'ler geçersizdir. Guard policy, Colab'de uzun fine-tuning başlamadan önce hatayı yakalamak ve yanlış artifact'lerin rapora karışmasını engellemek için gereklidir.

## D028 - Sprint 4 Validation Result Interpretation

Karar: Sprint 4 validation-only seçiminde en güçlü fine-tuned downstream sonuç `vit_b16+swin_tiny+beit_base concat` olmuştur (`0.7298` validation macro-F1). Bu sonuç, modest frozen triple concat baseline (`0.6988`) ve E2b stronger-MLP frozen ViT+Swin diagnostic baseline (`0.7262`) ile birlikte yorumlanacaktır. Test seti hâlâ kullanılmamıştır ve final audit'e bırakılmıştır.

Gerekçe: Fine-tuned single-backbone sonuçları karışıktır: ViT frozen baseline'ın biraz altında kalmış, Swin ve BEiT single-backbone olarak iyileşmiş fakat frozen ViT seviyesine ulaşmamıştır. Fine-tuned concat fusion'da BEiT eklemek `vit_b16+swin_tiny concat` sonucunu `0.7161`den `0.7298`e çıkarmıştır. Bu, BEiT'in standalone kalite yerine complementary representation kaynağı olarak tartışılmasını destekler. E2b'ye karşı fark küçük olduğu için sonuç temkinli şekilde "limited evidence" olarak raporlanacaktır.

## D029 - E3b Downstream Multi-Seed Diagnostic

Karar: Sprint 4 sonrası robustness kontrolü, full fine-tuning'i tekrar etmek yerine sabit frozen/fine-tuned feature cache'leri üzerinde downstream MLP seed'lerini değiştiren E3b diagnostic olarak yürütülmüştür. E3b CPU üzerinde seed `7`, `13`, `42`, `101`, `202` ile `vit_b16+swin_tiny+beit_base finetuned concat`, `vit_b16+swin_tiny finetuned concat`, `vit_b16 finetuned single` ve `vit_b16+swin_tiny frozen concat deep_reg` koşullarını tekrarlar. Test seti kullanılmaz.

Gerekçe: Sprint 4 ana sorusu feature transfer ve fusion karşılaştırmasıdır; checkpoint'leri yeniden fine-tune etmek pahalıdır ve farklı bir deney değişkeni ekler. Cached-feature MLP seed diagnostic ucuzdur ve `0.7298` ile `0.7262` gibi küçük validation farklarının downstream initialization'a ne kadar duyarlı olduğunu ölçer. CPU kullanımı, eski E2b frozen diagnostic ile cihaz farkını azaltır. Sonuçlar fine-tuned triple concat'in en yüksek mean validation macro-F1'i verdiğini (`0.7246 ± 0.0143`) ancak seed varyansının görünür olduğunu göstermiştir; bu nedenle E3b final seçim disiplinini değiştirmez, yorumun güven düzeyini kalibre eder.

## D030 - E3c Metadata-Augmented Feature Fusion Diagnostic

Karar: E3c, yeni transformer fine-tuning yapmadan sabit cached feature'lara HAM10000 benchmark metadata'sı ekleyen validation-only diagnostic olarak yürütülecektir. Model input metadata alanları yalnız `age`, `sex` ve `localization` ile sınırlıdır. `dx`, `dx_type`, `dataset`, `image_id`, `sample_id`, `lesion_id` ve path/source alanları model input'una dahil edilmeyecektir.

Gerekçe: E3/E3b sonuçları fine-tuned triple concat'in güçlü fakat yakın farkla önde olduğunu göstermiştir. Literatürde skin-lesion image feature'larının age, sex ve anatomical site metadata'sı ile birleştirilmesi yaygın bir multimodal benchmark yaklaşımıdır. Bu repo için en temiz sonraki soru, yeni backbone veya test-probe eklemeden structured metadata'nın fine-tuned transformer feature transfer'a tamamlayıcı sinyal sağlayıp sağlamadığını ölçmektir. Metadata preprocessing train-only fit edilecek, test split yüklenmeyecek ve sonuçlar validation macro-F1 ile multi-seed mean/std üzerinden raporlanacaktır.

## D031 - E3c Metadata Result Interpretation

Karar: E3c sonucu final selection'ı tek başına değiştiren kesin bir model üstünlüğü olarak değil, fine-tuned transformer feature fusion'a structured benchmark metadata'nın küçük ve sınıf-bağımlı tamamlayıcı sinyal ekleyebildiğini gösteren diagnostic evidence olarak yorumlanacaktır. En güçlü E3c koşulu `vit_b16+swin_tiny+beit_base concat + metadata` olmuştur (`0.7278 ± 0.0058` validation macro-F1), fakat image-only E3b triple concat mean'i (`0.7246 ± 0.0143`) ile fark küçüktür.

Gerekçe: Metadata-only MLP'nin düşük kalması (`0.2202 ± 0.0077`) metadata'nın tek başına yeterli olmadığını gösterir. Plus-metadata koşulları image-only kontrollere göre küçük mean macro-F1 artışı üretmiştir; ancak per-class etki karışıktır (`akiec` ve bazı küçük sınıflarda artış, `bkl`/`mel` tarafında küçük düşüşler). Bu nedenle rapor dili "metadata may provide small complementary benchmark signal" çizgisinde kalmalı, klinik genelleme veya deployment iddiası kurulmamalıdır. Test seti hâlâ kullanılmamıştır.

## D032 - E3d Lightweight Metadata Fusion Operator Scope

Karar: E3d, E3c'nin raw metadata concat sonucunu genişleten fakat end-to-end multimodal transformer mimarisi kurmayan lightweight cached-feature ablation olarak yürütülecektir. Canonical koşullar yalnız fine-tuned `vit_b16+swin_tiny+beit_base` triple feature seti üzerinde çalışır: metadata-gated backbone fusion, bounded FiLM-style metadata conditioning ve two-branch image/metadata MLP. E3c raw concat + metadata ve E3b image-only triple concat sonuçları kontrol olarak kullanılacaktır.

Gerekçe: Güncel multimodal skin-lesion literatürü metadata ile image representation'ı birleştirirken concat, weighted/gated fusion ve attention/cross-attention gibi operator'ları karşılaştırır. Bu repo için en uygun ek ablation, test-probe veya yeni Colab fine-tuning eklemeden metadata'nın image feature'ları yalnızca append etmek yerine modüle edip etmediğini ölçmektir. Büyük cross-attention mimarisi, balanced sampler, class-aware loss, train-time augmentation ve yeni fine-tuning bu kapsamın dışındadır. Test seti kullanılmayacak; sonuçlar validation macro-F1 mean/std ve per-class behavior ile raporlanacaktır.

## D033 - E3d Metadata Fusion Operator Result Interpretation

Karar: E3d sonucunda en yüksek mean validation macro-F1 bounded FiLM-style metadata conditioning ile elde edilmiştir (`0.7358 ± 0.0152`). Metadata-gated backbone fusion (`0.7347 ± 0.0112`) ve two-branch image/metadata fusion (`0.7328 ± 0.0103`) da E3c raw concat + metadata kontrolünü (`0.7278 ± 0.0058`) aşmıştır. Bu bulgu, structured metadata'nın yalnız input'a append edilmek yerine image representation'ı condition ettiğinde daha güçlü validation sinyali verebildiğini gösteren limited diagnostic evidence olarak yorumlanacaktır.

Gerekçe: Üç operator da mean macro-F1'i artırmıştır, fakat per-class davranış tek yönlü değildir. FiLM `mel` F1'i artırırken `akiec` ve `vasc` tarafında düşüş üretmiştir; gated fusion `akiec`, `bcc`, `mel` ve `vasc` için artış üretirken `df` düşmüştür; two-branch fusion `df` ve `mel` tarafında iyileşirken `vasc` düşmüştür. Bu nedenle E3d sonucu "advanced metadata fusion uniformly improves classification" şeklinde değil, "metadata-conditioned lightweight fusion can improve validation macro-F1 with class-dependent tradeoffs" şeklinde raporlanacaktır. Gated fusion'daki learned gate değerleri model içi diagnostic olarak kalır; backbone kalite sıralaması veya klinik açıklama olarak yorumlanmaz. Test seti hâlâ kullanılmamıştır.

## D034 - E3e Conservative ViT Fine-Tuning Scope

Karar: E3e, canonical Sprint 4 ViT fine-tuning düşüşünü inceleyen dar bir Colab diagnostic olacaktır. Yalnız `vit_b16` yeniden fine-tune edilir: `last_2_blocks + backbone LR 5e-6` ve `last_1_block + backbone LR 5e-6`. Canonical `finetuned` artifact'leri overwrite edilmez; E3e checkpoint ve feature cache'leri `e3e_vit_last2_lr5e6`, `e3e_vit_last1_lr5e6`, `finetuned_vit_last2_lr5e6` ve `finetuned_vit_last1_lr5e6` namespace'lerinde tutulur.

Gerekçe: Frozen ViT single validation macro-F1 `0.6924` iken canonical fine-tuned ViT single `0.6876` olmuştur. Bu küçük düşüş, fine-tuning'in tamamen faydasız olduğunu değil, ViT'in unfreeze depth ve backbone LR ayarlarına hassas olabileceğini gösterebilir. Medical-image transfer learning literatürü fine-tuning strategy'nin modality ve architecture'a bağlı olduğunu, ViT literatürü ise daha sınırlı parameter adaptation'ın geçerli bir compute/overfit kontrolü olduğunu destekler. E3e bu soruyu broad augmentation, class-aware loss, deeper unfreeze veya yeni backbone eklemeden test eder.

## D035 - E3e Validation and Colab Artifact Policy

Karar: E3e test split'i yüklemez, test metric üretmez ve final model seçimi yapmaz. Checkpoint seçimi validation macro-F1 ile yapılır. Downstream kontroller önce E3e ViT single MLP, sonra yeni ViT cache'i canonical Swin/BEiT cache'leriyle karıştıran validation-only triple concat koşusudur. Metadata-conditioned FiLM/gated follow-up yalnız validation sonuçları bunu makul kılarsa çalıştırılır.

Gerekçe: Eski `dl-assignment` lessons learned, son aşamada çok fazla training varyantı açmanın raporu dağıttığını gösterdi. E3e bu yüzden tek backbone, iki policy ve ayrı Drive namespace ile sınırlıdır. Colab çıktıları `/content/drive/MyDrive/dl-final-artifact/e3e_conservative_vit/` altına senkronize edilir; canonical Sprint 4 cache'leri yalnız okunur ve mixed-source kopyaları ayrı `finetuned_vit_*_plus_s4_swin_beit` klasörlerine yazılır.

## D036 - E3e Conservative ViT Result Interpretation

Karar: E3e sonucunda konservatif ViT fine-tuning politikaları ana fine-tuned feature fusion sonucunu değiştirmeyecektir. `last_2_blocks + backbone LR 5e-6` ViT single downstream MLP'de `0.6694`, `last_1_block + backbone LR 5e-6` ise `0.6685` validation macro-F1 üretmiştir; iki sonuç da frozen ViT (`0.6924`) ve canonical fine-tuned ViT (`0.6876`) altındadır. Mixed ViT+Swin+BEiT concat tarafında last-1 policy `0.7259` ile kabul edilebilir ama canonical triple concat (`0.7298`) ve E3d metadata-conditioned mean sonuçlarının gerisinde kalmıştır. Bu nedenle E3e final aday seçimini değiştirmez.

Gerekçe: Sonuç, ViT fine-tuning düşüşünün yalnızca fazla agresif unfreeze veya backbone LR kaynaklı olmadığını gösteren negative ablation olarak değerlidir. Daha konservatif adaptation single ViT feature kalitesini toparlamamış, fusion tarafında ise yalnızca ana E3b triple concat ortalamasına yakın bir sonuç üretmiştir. Bu bulgu raporda "ViT feature transfer is sensitive to fine-tuning policy and frozen ViT remains a strong control" şeklinde tartışılacaktır. Test seti kullanılmamıştır; E3e klinik veya final audit iddiası taşımaz.

## D037 - E3f Mixed Backbone Adaptation Scope

Karar: E3f, yeni backbone fine-tuning yapmadan `frozen vit_b16 + fine-tuned swin_tiny + fine-tuned beit_base` mixed feature source'unu test edecektir. Bu source image-only concat ve metadata-conditioned FiLM/gated operators ile seeds `7,13,42,101,202` uzerinde validation-only degerlendirilecektir. Test split yuklenmeyecek ve canonical `frozen/` veya `finetuned/` cache'leri overwrite edilmeyecektir.

Gerekçe: ViT single fine-tuning frozen ViT baseline'ini gecemedi; buna karsilik Swin ve BEiT single fine-tuning frozen hallerine gore iyilesti. Bu durum backbone bazinda ayni fine-tuning politikasini herkese uygulamak yerine, ViT'i frozen strong control olarak tutup Swin/BEiT'i adapted kullanmanin daha dogru bir ablation oldugunu gosterir. FiLM ve gated metadata operators E3d'de en guclu validation adaylari oldugu icin ayni mixed source uzerinde tekrar denenerek metadata-conditioned sonuc source-sensitive mi degil mi kontrol edilecektir.

## D038 - E3f Mixed Adaptation Result Interpretation

Karar: E3f sonucu final aday anlatiminda E3d ile pratik beraberlik olarak ele alinacaktir. Mixed image-only concat `0.7142 ± 0.0126` validation macro-F1 ile all-fine-tuned triple concat ortalamasinin (`0.7246 ± 0.0143`) altinda kalmistir. Mixed metadata-gated fusion ise `0.7361 ± 0.0100` ile numerik olarak en yuksek multi-seed validation mean'i uretmis, ancak E3d all-fine-tuned FiLM (`0.7358 ± 0.0152`) ile fark yalnizca `+0.0003` macro-F1 duzeyinde kalmistir.

Gerekçe: E3f, frozen ViT'in metadata-gated fusion ile rekabetci kalabilecegini gosterir; fakat image-only concat'te all-fine-tuned source gerisinde kaldigi icin "ViT'i frozen tutmak genel olarak daha iyidir" sonucunu desteklemez. Per-class analizde E3f gated sonucunun ortalama avantaji ozellikle dusuk destekli `df` sinifindan gelirken `akiec`, `bcc` ve `vasc` E3d gated'a gore dusmustur. Bu nedenle E3f raporda source/operator interaction diagnostic olarak kullanilacak, test split uzerinden final sonuc iddiasi kurulmayacaktir.

## D039 - E3g Prediction Ensemble Scope

Karar: E3g, mevcut E3d/E3f validation prediction dump'lari uzerinde post-hoc probability averaging deneyi olarak planlanacaktir. Primary ensemble kosullari equal-weight olacaktir: E3d FiLM seed-average, E3d gated seed-average, E3f mixed gated seed-average, E3d FiLM + E3f gated equal, E3d gated + E3f gated equal ve top-3 family equal. Weighted grid aramasi yapilirsa kucuk ve diagnostic olarak etiketlenecektir.

Gerekçe: E3d ve E3f en guclu adaylari validation macro-F1 acisindan pratik beraberliktedir, fakat per-class davranislari farklidir. Literature-backed ensemble/TTA pratikleri skin lesion classification challenge'larinda skor artisi icin yaygindir; ancak unrestricted validation weight search raporun guvenilirligini zedeler. Bu nedenle E3g once low-overfit equal-weight probability averaging ile sinirlandirilir. Test split kullanilmayacaktir.

## D040 - E3g Ensemble Result Interpretation

Karar: E3g sonucunda primary validation adayi `top3_family_equal` prediction ensemble olarak kaydedilecektir. Bu ensemble E3d FiLM, E3d gated ve E3f mixed gated family'lerinin seed-averaged probability output'larini esit agirlikla birlestirmis ve `0.7665` validation macro-F1, `0.8564` accuracy, `0.8576` weighted-F1 uretmistir. Weighted grid diagnostic'te `0.7702` macro-F1 elde edilmistir, ancak bu sonuc validation-label pressure nedeniyle ana model-selection sonucu olarak degil exploratory diagnostic olarak raporlanacaktir.

Gerekçe: Equal-weight ensemble yeni egitim, test access, class-specific threshold tuning veya unrestricted weight search kullanmadan onceki E3d/E3f validation sonuclarini belirgin sekilde asmistir. Family error-overlap tam olmadigi icin probability averaging farkli hata profillerinden faydalanmistir. Buna ragmen E3g halen validation-only bir sonuc oldugundan final test audit yapilmadan genelleme veya klinik performans iddiasi kurulmayacaktir.

## D041 - E3h Rot4 TTA Scope

Karar: E3h, E3g `top3_family_equal` ensemble'inin deterministic right-angle rotation TTA ile
iyilesip iyilesmedigini validation-only olarak test edecektir. Primary policy yalniz
`tta_rot4 = identity + rot90 + rot180 + rot270` olacaktir. `tta_flip4`, `tta_d4_8`, color/crop
augmentation, validation-tuned TTA weights, yeni backbone fine-tuning ve yeni downstream MLP egitimi
bu ilk E3h kapsaminda yapilmayacaktir.

Gerekçe: Eski `dl-assignment` projesinde validation-gated TTA faydali olmus, ancak daha fazla view
ve train-time augmentation otomatik olarak daha iyi sonuc vermemistir. Skin-lesion classification
literaturu de TTA ve probability averaging'i destekler, fakat bu adim policy search'e donerse
validation over-selection riski artar. Bu nedenle E3h tek, onceden sabitlenmis, geometry-safe
`rot4` policy ile sinirlandirilir. Test split E3h'de yuklenmeyecek; E3h final aday olursa test
yalniz E4 final audit icin kullanilacaktir.

## D042 - E3i Simple Fusion TTA Scope

Karar: E3i, E3h'deki negatif rot4 sonucunun zaten guclu bir metadata-conditioned probability
ensemble uzerinde TTA uygulanmasindan kaynaklanip kaynaklanmadigini test etmek icin daha sade
image-only cached-feature fusion modellerine odaklanacaktir. Kapsam yalniz fine-tuned
`vit_b16+swin_tiny+beit_base` concat seed 42, fine-tuned `vit_b16+swin_tiny+beit_base`
`weighted_learned_512` seed 42 ve fine-tuned `vit_b16+swin_tiny` concat seed 42 run'larini icerir.
Primary policy yine yalniz `tta_rot4 = identity + rot90 + rot180 + rot270` olacaktir.

Gerekçe: Eski `dl-assignment` projesinde TTA, final ensemble'a dogrudan eklenmeden once daha sade
fine-tuned/weighted model uzerinde validation-gated olarak fayda gostermisti. Bu projede E3h rot4
uygulamasi E3g top3 ensemble'ini dusurdugu icin en temiz takip deneyi TTA'yi model-family
ensembling oncesindeki sade fusion modellerinde test etmektir. `weighted_pca_384` E3i disinda
tutulur, cunku mevcut artifact'lerde fitted PCA component'leri saklanmadigindan rotated validation
feature'lari icin ayni preprocessing'i yeniden uygulamak mumkun degildir. Test split E3i'de
yuklenmeyecek ve validation-tuned TTA weight aramasi yapilmayacaktir.
