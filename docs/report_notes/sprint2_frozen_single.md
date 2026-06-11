# Frozen Transformer Feature Extraction and Single-Backbone Baselines

Bu not, final rapordaki "frozen feature extraction" ve "single-backbone baseline" bölümlerine doğrudan malzeme sağlamak için hazırlanmıştır. Akademik raporda iç proje yönetimi dili kullanılmamalı; bu aşama "frozen transformer feature extraction", "tek backbone temsilleri" ve "MLP classifier baseline" olarak anlatılmalıdır.

## Araştırma Sorusu

Vanilla ViT, Swin Transformer ve DeiT III-Small modelleri, classifier head kaldırılıp frozen feature extractor olarak kullanıldığında HAM10000 benchmark dermoscopic image classification görevi için ne kadar ayırt edici temsil üretmektedir?

Bu aşamanın amacı fusion veya fine-tuning ile en yüksek sonucu bulmak değildir. Amaç, daha karmaşık fusion deneylerine geçmeden önce her transformer backbone'un tek başına temsil kalitesini aynı split, aynı preprocessing, aynı classifier ve aynı model seçim kuralı altında ölçmektir.

## Deney Kurulumu

HAM10000 veri seti, önceki veri hazırlama aşamasında oluşturulan sabit lesion-aware split ile kullanılmıştır. Eğitim splitinde 7,008, validation splitinde 1,504 görüntü bulunmaktadır. Aynı lezyona ait görüntülerin farklı splitlere geçmesini engelleyen bu ayrım, backbone karşılaştırmalarında olası leakage etkisini azaltmak için korunmuştur. Test split bu aşamada kullanılmamıştır; model, checkpoint veya backbone tercihi validation macro-F1 üzerinden yapılmıştır.

Görüntüler her backbone için deterministik `224x224` yeniden boyutlandırma ve ImageNet normalization ile işlenmiştir. Bu aşamada data augmentation uygulanmamıştır; çünkü feature cache'lerin tekrarlanabilir ve backbone'lar arasında karşılaştırılabilir olması hedeflenmiştir. Backbone ağırlıkları dondurulmuş, classifier head bypass edilmiş ve her görüntü için sabit boyutlu feature vector üretilmiştir.

Feature extraction politikası şu şekilde sabitlenmiştir:

| Backbone | Model ID | Feature policy | Feature dim |
|---|---|---|---:|
| Vanilla ViT | `vit_base_patch16_224.augreg_in21k_ft_in1k` | CLS-token representation | 768 |
| Swin Transformer | `swin_tiny_patch4_window7_224.ms_in1k` | Average pooled final-stage representation | 768 |
| DeiT III-Small | `deit3_small_patch16_224.fb_in1k` | CLS-token representation | 384 |

Cache'lenen feature vector'lar üzerinde aynı MLP classifier kullanılmıştır. MLP yapısı `input_dim -> 512 -> 256 -> 7` şeklindedir; ara katmanlarda BatchNorm, ReLU ve dropout `0.3` kullanılmıştır. Feature normalization için `StandardScaler` yalnızca train feature cache üzerinde fit edilmiş, validation feature'larına aynı train istatistikleri uygulanmıştır. Loss fonksiyonu train splitinden hesaplanan class weight'lerle ağırlıklandırılmış cross entropy olarak kurulmuştur. Optimizer AdamW, learning rate `0.001`, weight decay `0.0001`, maksimum epoch sayısı `30` ve early stopping patience `6` olarak sabit tutulmuştur. En iyi checkpoint validation macro-F1 değerine göre seçilmiştir.

## Ana Sonuçlar

Validation sonuçları aşağıdaki gibidir:

| Backbone | Feature dim | Best epoch | Accuracy | Macro-F1 | Macro precision | Macro recall | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla ViT | 768 | 23 | 0.7872 | 0.6924 | 0.6739 | 0.7193 | 0.7982 |
| Swin Transformer | 768 | 19 | 0.7739 | 0.6115 | 0.5810 | 0.6585 | 0.7837 |
| DeiT III-Small | 384 | 13 | 0.6842 | 0.5017 | 0.4568 | 0.5903 | 0.7090 |

Bu sonuçlara göre frozen feature extraction koşulunda en güçlü tek backbone temsili Vanilla ViT tarafından üretilmiştir. ViT, validation macro-F1 değerinde Swin Transformer'dan yaklaşık `0.081`, DeiT III-Small'dan yaklaşık `0.191` puan daha yüksektir. Swin Transformer accuracy ve weighted-F1 bakımından ViT'e yakın görünse de macro-F1 farkı, sınıflar arası dengenin ViT lehine daha iyi olduğunu göstermektedir. DeiT III-Small ise bu MLP recipe ve frozen feature policy altında üçüncü sırada kalmıştır.

Accuracy tek başına yorum metriği olarak yeterli değildir. HAM10000 validation splitinde çoğunluk sınıfı olan `nv` yüksek support'a sahiptir; bu nedenle accuracy ve weighted-F1, minority-class davranışını maskeleyebilir. Bu yüzden ana yorum metriği macro-F1 olarak korunmuştur.

## Sınıf Bazlı Yorum

Per-class sonuçlar, backbone sıralamasının yalnız overall accuracy ile açıklanamayacağını göstermektedir. ViT özellikle `akiec`, `bcc`, `df` ve `vasc` gibi düşük veya orta support'lu sınıflarda daha dengeli F1 değerleri üretmiştir. `nv` sınıfı üç modelde de yüksek performans göstermektedir; ancak bu durum veri setindeki çoğunluk sınıf etkisini yansıttığı için modelin bütün sınıflarda eşit derecede güçlü olduğu anlamına gelmez.

Validation per-class F1 özeti:

| Backbone | akiec | bcc | bkl | df | nv | mel | vasc |
|---|---:|---:|---:|---:|---:|---:|---:|
| Vanilla ViT | 0.5962 | 0.6790 | 0.6687 | 0.7179 | 0.8885 | 0.5061 | 0.7907 |
| Swin Transformer | 0.5155 | 0.5912 | 0.6574 | 0.5200 | 0.8930 | 0.4649 | 0.6383 |
| DeiT III-Small | 0.4255 | 0.4172 | 0.5280 | 0.4138 | 0.8377 | 0.3899 | 0.5000 |

`mel` sınıfı üç model için de görece zor kalmıştır. ViT bu sınıfta da en iyi F1 değerini üretse de skor `0.5061` seviyesinde kalmaktadır. Bu durum, frozen single-backbone temsillerin sınıf ayrımında hâlâ sınırlı olduğunu ve sonraki feature fusion veya fine-tuning aşamalarının metodolojik gerekçesini güçlendirdiğini göstermektedir.

`df` ve `vasc` sınıflarında support sınırlı olduğu için F1 değerleri dikkatli yorumlanmalıdır. ViT'in bu sınıflarda yüksek skor üretmesi olumlu bir sinyaldir; ancak düşük örnek sayısı nedeniyle bu sonuçlar final model iddiası yerine validation-stage evidence olarak kullanılmalıdır.

## BEiT Candidate Screening

Üçüncü backbone seçimini kontrol etmek için BEiT-Base kısa bir validation-only candidate olarak denenmiştir. Kullanılan model `beit_base_patch16_224.in22k_ft_in22k_in1k` olmuştur. BEiT checkpoint'i average-pooled patch-token representation ile daha uyumlu olduğu için `global_pool=avg` kullanılmıştır.

BEiT-Base sonucu:

| Candidate | Feature dim | Best epoch | Accuracy | Macro-F1 | Macro precision | Macro recall | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| BEiT-Base | 768 | 19 | 0.6509 | 0.4759 | 0.4278 | 0.6071 | 0.6833 |

BEiT-Base, aynı MLP recipe altında DeiT III-Small'ın validation macro-F1 değerini geçememiştir (`0.4759` vs `0.5017`). Bu nedenle Sprint 2 single-backbone screening aşamasında üçüncü backbone DeiT III-Small olarak korunmuştur. Bu karar validation metric üzerinden verilmiştir; test split bu seçim için kullanılmamıştır.

## Önceki CNN Çalışmasıyla Karşılaştırmalı Yorum

Bu sonuçlar, önceki `dl-assignment` CNN projesindeki frozen feature extraction aşamasıyla karşılaştırıldığında önemli bir mimari fark gösterir. Önceki çalışmada ResNet50, EfficientNetB0 ve MobileNetV2 frozen feature extractor olarak kullanılmış; default class-weighted MLP koşulunda en güçlü single-backbone sonuç ResNet50 ile test macro-F1 `0.531` olmuştur. Aynı çalışmada MLP hyperparameter sensitivity denemeleri EfficientNetB0 için macro-F1'i `0.5605` seviyesine çıkarabilmiş, ancak bu sonuç validation-selected final model olarak değil diagnostic ablation olarak yorumlanmıştır.

Bu final projesinde ise frozen transformer temsilleri, daha feature fusion veya fine-tuning uygulanmadan daha yüksek bir başlangıç noktası üretmiştir. Vanilla ViT single-backbone validation macro-F1 `0.6924` değerine ulaşmıştır. Bu değer, eski CNN projesindeki en iyi frozen single-backbone skorundan (`0.531`) belirgin biçimde yüksektir ve hatta eski CNN projesindeki canonical fine-tuned three-backbone concat sonucuna (`0.706` macro-F1) oldukça yakındır. Bu karşılaştırma doğrudan "aynı test koşulu" gibi okunmamalıdır; eski notlardaki sayılar test macro-F1, bu projedeki sayılar ise validation macro-F1'dir. Yine de metodolojik olarak güçlü bir sinyal verir: transformer backbone'ları frozen halde bile HAM10000 benchmark dermoscopic image classification için CNN frozen feature'larına göre daha ayırt edici temsil üretmektedir.

Rapor dilinde bu fark şöyle çerçevelenebilir:

> Önceki CNN tabanlı çalışmada frozen feature extraction, fusion ve fine-tuning aşamaları için daha düşük bir başlangıç noktası oluşturmuştu; en güçlü default frozen single-backbone sonuç ResNet50 ile macro-F1 `0.531` seviyesindeydi. Bu projede ise Vanilla ViT, frozen single-backbone koşulunda validation macro-F1 `0.6924` değerine ulaşarak çok daha güçlü bir representation baseline sağlamıştır. Bu durum, transformer tabanlı pretrained representation'ların aynı benchmark üzerinde daha yüksek başlangıç temsil gücü sunduğunu göstermektedir. Sayısal karşılaştırma farklı validation/test protokolleri nedeniyle doğrudan leaderboard karşılaştırması olarak değil, mimari davranış farkı olarak yorumlanmalıdır.

## Rapor İçin Kullanılabilecek Metodoloji Paragrafı

> Frozen feature extraction aşamasında Vanilla ViT, Swin Transformer ve DeiT III-Small backbone'ları classifier head olmadan kullanılmıştır. Her görüntü deterministik `224x224` ImageNet preprocessing ile işlenmiş ve backbone ağırlıkları güncellenmeden sabit feature vector'lar çıkarılmıştır. ViT ve DeiT III-Small için CLS-token temsili, Swin Transformer için ise son aşamadan average pooled representation kullanılmıştır. Cache'lenen feature vector'lar üzerinde aynı MLP classifier eğitilmiş; scaler yalnız train split üzerinde fit edilmiş, class-weighted cross entropy kullanılmış ve checkpoint seçimi validation macro-F1 değerine göre yapılmıştır. Test seti bu aşamada model seçimi veya metrik raporlaması için kullanılmamıştır.

## Rapor İçin Kullanılabilecek Sonuç Paragrafı

> Frozen single-backbone sonuçlarında en yüksek validation macro-F1 Vanilla ViT ile elde edilmiştir (`0.6924`). Swin Transformer ikinci sırada yer almış (`0.6115`), DeiT III-Small ise aynı protokol altında daha düşük macro-F1 üretmiştir (`0.5017`). Accuracy ve weighted-F1 değerleri özellikle ViT ve Swin için birbirine yakın görünse de macro-F1 farkı, sınıf dengesizliği altında ViT temsilinin daha dengeli olduğunu göstermektedir. Bu nedenle sonraki fusion deneylerinde ViT en güçlü tek-backbone referans noktası olarak ele alınmıştır.

## Rapor İçin Kullanılabilecek Tartışma Paragrafı

> Per-class sonuçlar, frozen transformer feature'larının HAM10000 üzerindeki davranışının sınıflar arasında eşit dağılmadığını göstermektedir. `nv` sınıfında yüksek F1 değerleri elde edilirken, `mel` sınıfı tüm modeller için görece zor kalmıştır. ViT, düşük support'lu bazı sınıflarda daha yüksek F1 üretmiş olsa da `df` ve `vasc` gibi sınıflarda örnek sayısı sınırlı olduğundan bu bulgular dikkatli yorumlanmalıdır. Bu gözlem, accuracy odaklı bir yorum yerine macro-F1 ve per-class metrics odaklı değerlendirme yapılmasını desteklemektedir.

## Rapor Kararı

Vanilla ViT, frozen single-backbone baseline olarak en güçlü temsil kaynağıdır. Swin Transformer ve DeiT III-Small, tek başlarına ViT'i geçmemiş olsalar da farklı mimari aileleri temsil ettikleri için feature fusion aşamasında complementarity açısından değerlendirilmeye devam etmelidir. BEiT-Base candidate olarak denenmiş ancak DeiT III-Small'ı validation macro-F1 bakımından geçemediği için canonical backbone setine alınmamıştır.

Canonical frozen backbone seti:

```text
vit_b16
swin_tiny
deit3_small
```

## Evidence

Generated evidence:

```text
artifacts/features/ham10000/frozen/vit_b16/
artifacts/features/ham10000/frozen/swin_tiny/
artifacts/features/ham10000/frozen/deit3_small/
artifacts/features/ham10000/frozen/beit_base/
artifacts/runs/20260610_165135_s2_frozen_vit_none_mlp_seed42/
artifacts/runs/20260610_165153_s2_frozen_swin_none_mlp_seed42/
artifacts/runs/20260610_165206_s2_frozen_deit3s_none_mlp_seed42/
artifacts/runs/20260610_172510_s2_frozen_beit_none_mlp_beit_screen_seed42/
artifacts/report_assets/tables/single_backbone_frozen_results.csv
artifacts/report_assets/tables/single_backbone_frozen_per_class_metrics.csv
artifacts/report_assets/figures/frozen_single_backbone_macro_f1.png
```

Verification evidence:

- Full train/validation cache row counts match split counts.
- Cache row alignment against split CSVs passed.
- Feature tensors contain no NaN or Inf values.
- Prediction dumps contain 1,504 validation rows and one probability column per class.
- Checkpoint selection used validation macro-F1.
- Test split was not used for model selection.
- Generated feature caches, checkpoints, prediction dumps and run artifacts are Git-ignored.

## Limitations

- Sonuçlar tek seed (`42`) ile elde edilmiştir.
- Test metrics bu aşamada kasıtlı olarak hesaplanmamıştır.
- Pooling/token policy sabit tutulmuş, CLS-token vs mean-pooling ablation yapılmamıştır.
- MLP recipe backbone'lar arasında sabit tutulmuştur; daha geniş hyperparameter search ayrı bir ablation olarak ele alınmalıdır.
- Minority-class validation support sınırlıdır; özellikle `df` ve `vasc` sonuçları support sayıları görünür tutularak yorumlanmalıdır.
- Sonuçlar benchmark dermoscopic image classification kapsamındadır; klinik teşhis veya deployment iddiası üretmez.

## Post-E2 Update

Bu nottaki DeiT III-Small kararı, Sprint 2 single-backbone validation screening bağlamında doğrudur. E2 frozen fusion matrix ve representation similarity diagnostic sonrasında ileri aşamalar için üçüncü backbone BEiT-Base olarak güncellenmiştir. Bu değişiklik test seti kullanılmadan, validation-only fusion complementarity evidence ile yapılmıştır. DeiT III-Small raporda planned/screened baseline olarak kalır; Sprint 4 forward set `vit_b16`, `swin_tiny`, `beit_base` olacaktır.
