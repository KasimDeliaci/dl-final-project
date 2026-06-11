# Frozen Feature Fusion Report Note

Bu not, final rapordaki "frozen feature fusion" bölümüne doğrudan malzeme sağlamak için hazırlanmıştır. Akademik raporda iç proje yönetimi dili ana anlatı yapılmamalıdır. Bu aşama, "frozen transformer feature extraction", "single-backbone baseline", "feature fusion", "feature complementarity", "projection bottleneck", "PCA compression" ve "validation macro-F1" kavramlarıyla anlatılmalıdır.

Bu çalışma klinik teşhis iddiası taşımaz. Tüm sonuçlar HAM10000 üzerinde benchmark dermoscopic image classification bağlamında yorumlanmalıdır.

## Research Question

Bu aşamadaki ana araştırma sorusu şudur:

> Frozen transformer backbone feature'ları birleştirildiğinde, en güçlü single-backbone baseline olan ViT validation macro-F1 `0.6924` değerinin üstüne çıkılabiliyor mu?

İkinci soru ise daha açıklayıcıdır:

> Bir backbone tek başına zayıf olsa bile, diğer backbone'larla birleştirildiğinde tamamlayıcı temsil bilgisi sağlayabilir mi?

Bu ikinci soru özellikle BEiT-Base için önemlidir. BEiT, single-backbone validation macro-F1 bakımından DeiT III-Small'ın gerisinde kalmıştır (`0.4759` vs `0.5017`), ancak E2 frozen fusion matrix içinde ViT ve Swin ile birlikte kullanıldığında daha yüksek validation macro-F1 üretmiştir. Bu nedenle bu notun ana tartışma ekseni, single-backbone strength ile fusion complementarity'nin aynı şey olmadığıdır.

## Experimental Setup

Frozen feature fusion deneylerinde raw image üzerinden yeni feature extraction yapılmadı. Girdi olarak önceki frozen feature extraction aşamasında üretilen train ve validation cache dosyaları kullanıldı. Her run öncesinde ilgili backbone cache'leri şu alanlar üzerinden hizalandı:

- `sample_id`
- `image_id`
- `lesion_id`
- label index
- label name
- split row order

Bu hizalama zorunludur; çünkü fusion işlemi aynı görüntüye ait farklı backbone feature vector'larını aynı satırda birleştirir. Row order veya label alignment hatası, görünürde geçerli bir feature matrix üretse bile modelin yanlış örnekleri birleştirmesine neden olurdu.

Kullanılan sabit kontroller:

- Dataset: HAM10000 benchmark dermoscopic image classification.
- Split: fixed lesion-aware split.
- Train split: `7008` images.
- Validation split: `1504` images.
- Test split: kullanılmadı.
- Model/checkpoint/fusion method seçimi: validation macro-F1.
- Feature scaling: StandardScaler fit only on train cache.
- Class imbalance handling: train-only class-weighted cross entropy.
- Confusion matrix label order: `akiec`, `bcc`, `bkl`, `df`, `nv`, `mel`, `vasc`.

Test seti bu aşamada hiç kullanılmamıştır. Test audit, final model seçimi validation üzerinden tamamlandıktan sonra yapılmalıdır.

## Experiment Map

Bu rapor notunda dört deney katmanı birlikte ele alınır:

| Layer | Purpose | Scope | Selection signal |
|---|---|---|---|
| E1 reference | Single-backbone frozen representation strength | ViT, Swin, DeiT, BEiT screening | validation macro-F1 |
| E2 planned matrix | Planned frozen feature fusion | ViT/Swin/DeiT pairwise and triple fusion | validation macro-F1 |
| E2 expanded matrix | Alternative third-backbone check | BEiT pairwise and ViT+Swin+BEiT triple fusion | validation macro-F1 + representation similarity |
| E2b diagnostic | Classifier capacity sensitivity | Representative single/fusion concat configurations with stronger MLPs | validation macro-F1 |

Bu ayrım önemlidir. E2 ana matrix, feature fusion karşılaştırmasını modest MLP classifier ile yapar. E2b ise aynı feature cache'ler üzerinde classifier kapasitesi değiştiğinde sonuçların ne kadar hareket ettiğini ölçer. Bu nedenle E2b sonuçları ana E2 tablosuna karıştırılmamalı, "capacity diagnostic" olarak ayrı yorumlanmalıdır.

## Fusion Methods

Üç fusion method karşılaştırıldı.

### Concat

`concat` method'unda her backbone feature block'u train-only StandardScaler ile normalize edildi ve feature axis boyunca doğrudan birleştirildi. Bu yöntemde PCA, LDA veya trainable projection uygulanmadı.

Bu karar bilinçlidir. Concat koşulu, feature bilgisini sıkıştırmadan "daha fazla backbone temsilini yan yana koymak" fikrini test eder. Bu yüzden concat sonucunda performans düşerse olası açıklama high-dimensional overfitting veya redundant representations olabilir; performans artarsa bu, bilgi kaybı olmadan taşınabilen complementarity sinyaline işaret eder.

### Weighted Learned 512

`weighted_learned_512` method'unda her backbone feature block'u trainable `Linear(input_dim -> 512)` projection katmanından geçirildi. Daha sonra global trainable backbone logits softmax ile normalize edildi ve projected feature block'lar weighted sum ile birleştirildi. Ortaya çıkan 512-dimensional fused vector MLP classifier'a verildi.

Bu method, eski `dl-assignment` projesindeki weighted fusion yaklaşımının transformer feature'larına uyarlanmış halidir. Learned weights yorumlanabilir bir diagnostic artifact üretir; ancak bu weight'ler doğrudan "hangi backbone daha iyi" sıralaması değildir. Projection layers, softmax weights ve MLP classifier birlikte optimize edildiği için weight değerleri yalnızca bu model mimarisi içindeki katkı sinyali olarak okunmalıdır.

### Weighted PCA 384

`weighted_pca_384` method'u final projeye özel diagnostic/probe olarak eklendi. Her backbone feature block'u önce train-only StandardScaler ile normalize edildi. Daha sonra PCA yalnız train split üzerinde fit edildi ve validation feature'larına aynı PCA transform uygulandı. Ortak latent dimension `384` seçildi; çünkü DeiT III-Small feature dimension'ı `384` idi.

Bu method canonical weighted fusion'ın yerine geçmez. Amacı, trainable projection yerine unsupervised dimensionality reduction kullanıldığında weighted fusion'ın nasıl davrandığını görmektir. PCA label kullanmadığı için leakage üretmez; ancak sınıf ayrımını maksimize etmek için optimize edilmediğinden discriminative directions kaybedebilir.

## Planned ViT/Swin/DeiT Matrix

İlk E2 matrix, başlangıçta seçilen üç backbone üzerinden çalıştırıldı:

- Vanilla ViT
- Swin Transformer
- DeiT III-Small

Her pairwise ve three-backbone kombinasyon `concat`, `weighted_learned_512` ve `weighted_pca_384` ile çalıştırıldı. Toplam 12 validation-only run üretildi.

| Backbone combination | Fusion method | Feature dim | Accuracy | Macro-F1 | Macro precision | Macro recall | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ViT + Swin | concat | 1536 | 0.8178 | 0.6947 | 0.6578 | 0.7444 | 0.8244 |
| ViT + Swin + DeiT III-Small | concat | 1920 | 0.8225 | 0.6863 | 0.6618 | 0.7401 | 0.8281 |
| ViT + DeiT III-Small | concat | 1152 | 0.7812 | 0.6758 | 0.6395 | 0.7287 | 0.7940 |
| ViT + Swin + DeiT III-Small | weighted_learned_512 | 512 | 0.7793 | 0.6706 | 0.6487 | 0.7256 | 0.7946 |
| ViT + Swin | weighted_learned_512 | 512 | 0.7819 | 0.6516 | 0.5945 | 0.7415 | 0.7951 |
| ViT + DeiT III-Small | weighted_learned_512 | 512 | 0.7832 | 0.6489 | 0.6145 | 0.6986 | 0.7906 |
| ViT + DeiT III-Small | weighted_pca_384 | 384 | 0.7872 | 0.6419 | 0.6053 | 0.6937 | 0.7909 |
| ViT + Swin | weighted_pca_384 | 384 | 0.7733 | 0.6284 | 0.5955 | 0.6761 | 0.7832 |
| Swin + DeiT III-Small | concat | 1152 | 0.7706 | 0.6283 | 0.5826 | 0.6922 | 0.7831 |
| Swin + DeiT III-Small | weighted_pca_384 | 384 | 0.7540 | 0.5922 | 0.5676 | 0.6243 | 0.7627 |
| Swin + DeiT III-Small | weighted_learned_512 | 512 | 0.7434 | 0.5901 | 0.5485 | 0.6506 | 0.7602 |
| ViT + Swin + DeiT III-Small | weighted_pca_384 | 384 | 0.7354 | 0.5889 | 0.5471 | 0.6668 | 0.7533 |

Planned matrix içinde en iyi sonuç `ViT + Swin concat` ile elde edildi: validation macro-F1 `0.6947`. Bu değer ViT single-backbone baseline'ı olan `0.6924` değerinin yalnızca `+0.0023` üstündedir. Bu fark pozitif olmakla birlikte çok küçüktür. Bu nedenle planned matrix sonucu "fusion açık biçimde daha iyi" şeklinde değil, "ViT ve Swin arasında sınırlı tamamlayıcı bilgi olabilir" şeklinde temkinli yorumlanmalıdır.

DeiT III-Small'ın three-backbone concat koşuluna eklenmesi performansı artırmadı. `ViT + Swin + DeiT concat` macro-F1 `0.6863` değerinde kaldı ve `ViT + Swin concat` sonucunun altına düştü. Bu sonuç, daha fazla backbone eklemenin otomatik olarak daha iyi class-balanced performance üretmediğini gösterir.

## BEiT-Expanded E2 Matrix

Planned ViT/Swin/DeiT matrix sonrasında BEiT-Base E2 içine alternative third-backbone olarak eklendi. Bu kararın nedeni, BEiT'in masked image modeling pretraining çizgisi nedeniyle ViT/Swin/DeiT'ten farklı representation geometry sağlayabileceği hipotezidir.

BEiT-expanded matrix şu koşulları ekledi:

- `ViT + BEiT`
- `Swin + BEiT`
- `ViT + Swin + BEiT`

Her kombinasyon aynı üç fusion method ile çalıştırıldı. Toplam E2 fusion run sayısı böylece 21 oldu.

| Backbone combination | Fusion method | Feature dim | Accuracy | Macro-F1 | Macro precision | Macro recall | Weighted-F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| ViT + Swin + BEiT | concat | 2304 | 0.8125 | 0.6988 | 0.6674 | 0.7452 | 0.8177 |
| ViT + Swin + BEiT | weighted_learned_512 | 512 | 0.7999 | 0.6804 | 0.6427 | 0.7416 | 0.8091 |
| ViT + BEiT | concat | 1536 | 0.7673 | 0.6556 | 0.6142 | 0.7257 | 0.7836 |
| ViT + BEiT | weighted_pca_384 | 384 | 0.7886 | 0.6409 | 0.6074 | 0.6847 | 0.7932 |
| ViT + BEiT | weighted_learned_512 | 512 | 0.7660 | 0.6409 | 0.5946 | 0.7158 | 0.7814 |
| Swin + BEiT | concat | 1536 | 0.7633 | 0.6381 | 0.5994 | 0.7047 | 0.7810 |
| Swin + BEiT | weighted_learned_512 | 512 | 0.7473 | 0.6034 | 0.5592 | 0.6753 | 0.7653 |
| Swin + BEiT | weighted_pca_384 | 384 | 0.7281 | 0.5896 | 0.5461 | 0.6535 | 0.7456 |
| ViT + Swin + BEiT | weighted_pca_384 | 384 | 0.6975 | 0.5717 | 0.5185 | 0.6735 | 0.7217 |

BEiT pairwise koşulları ViT single-backbone baseline'ını geçmedi. `ViT + BEiT concat` macro-F1 `0.6556`, `Swin + BEiT concat` macro-F1 `0.6381` üretti. Bu sonuçlar BEiT'in tek başına veya sadece bir güçlü backbone ile birlikte yeterli olmadığını gösterir.

Buna karşın `ViT + Swin + BEiT concat` validation macro-F1 `0.6988` ile E2'nin en iyi sonucu oldu. Bu değer:

- ViT single baseline `0.6924` üstündedir.
- Planned `ViT + Swin concat` sonucu `0.6947` üstündedir.
- Planned `ViT + Swin + DeiT concat` sonucu `0.6863` üstündedir.

Bu bulgu, BEiT'in single-backbone olarak zayıf kalmasına rağmen ViT ve Swin ile birlikte kullanıldığında complementary signal sağlayabildiğini düşündürür. Ancak gain hâlâ küçüktür; bu nedenle sonuç "BEiT kesin olarak daha üstün" şeklinde değil, "BEiT, bu validation protocol altında daha tamamlayıcı üçüncü backbone adayıdır" şeklinde yazılmalıdır.

## Fusion Method Comparison

Genel olarak concat method'u weighted varyantlardan daha güçlü sonuçlar verdi. En iyi üç sonuç concat koşullarından geldi:

| Rank | Configuration | Fusion method | Validation macro-F1 |
|---:|---|---|---:|
| 1 | ViT + Swin + BEiT | concat | 0.6988 |
| 2 | ViT + Swin | concat | 0.6947 |
| 3 | ViT + Swin + DeiT III-Small | concat | 0.6863 |

Bu sonuç, frozen feature fusion için bilgi korumanın önemli olduğunu gösterir. Concat, her backbone feature block'unu ayrı ayrı korur ve sınıflandırıcıya daha geniş bir representation sunar. Buna karşılık weighted learned fusion, her backbone'u 512-dimensional ortak latent space'e indirir. Bu daha kompakt ve yorumlanabilir bir yapı sağlasa da projection bottleneck oluşturabilir.

Weighted PCA 384 varyantı en zayıf yöntemlerden biri oldu. Özellikle `ViT + Swin + BEiT weighted_pca_384` macro-F1 `0.5717` değerine düştü. Bu sonuç, unsupervised PCA compression'ın class-discriminative feature directions'ı korumak için yeterli olmayabileceğini gösterir. PCA variance preservation yapar; ancak HAM10000 gibi imbalanced ve fine-grained bir classification probleminde yüksek variance yönleri her zaman sınıf ayrımı için en faydalı yönler olmayabilir.

## MLP Capacity Diagnostic Addendum

E2 ana matrix modest MLP recipe ile çalıştırılmıştır. Daha sonra E2b diagnostic altında daha geniş ve daha regülarize MLP classifier'lar denenmiştir. Bu diagnostic, frozen concat fusion sonuçlarının classifier capacity'ye duyarlı olduğunu göstermiştir.

En önemli E2b sonucu:

```text
ViT + Swin concat + deep_reg MLP
validation macro-F1 = 0.7262
```

Bu değer, modest MLP ile elde edilen `ViT + Swin concat` sonucunun (`0.6947`) ve `ViT + Swin + BEiT concat` sonucunun (`0.6988`) belirgin biçimde üstündedir. `ViT + Swin + BEiT` de daha güçlü MLP ile iyileşmiştir (`0.7159`), ancak `ViT + Swin deep_reg` sonucunu geçememiştir.

Bu bulgu E2 yorumunu daha rafine hale getirir:

- BEiT, DeiT'e göre daha iyi üçüncü-backbone complementarity sinyali vermeye devam eder.
- Ancak en iyi frozen concat sonucu, daha güçlü MLP ile pairwise `ViT + Swin` koşulundan gelir.
- Bu nedenle fusion başarısızlığı veya başarısı yalnız representation complementarity ile açıklanmamalıdır; MLP classifier capacity de önemli bir kontrol değişkenidir.

Detaylı sonuçlar `docs/report_notes/e2b_mlp_capacity_diagnostic.md` içinde tutulmuştur.

E2b full validation table:

| Configuration | Fusion | MLP variant | Accuracy | Macro-F1 | Macro precision | Macro recall | Weighted-F1 | Best epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ViT | none | baseline | 0.7872 | 0.6924 | 0.6739 | 0.7193 | 0.7982 | 23 |
| ViT | none | wide | 0.7945 | 0.6813 | 0.6512 | 0.7200 | 0.8035 | 22 |
| ViT | none | wide_reg | 0.7866 | 0.6624 | 0.6179 | 0.7307 | 0.7979 | 31 |
| ViT | none | deep_reg | 0.7593 | 0.6438 | 0.5912 | 0.7416 | 0.7783 | 14 |
| ViT + Swin | concat | baseline | 0.8178 | 0.6947 | 0.6578 | 0.7444 | 0.8244 | 22 |
| ViT + Swin | concat | wide | 0.8152 | 0.6924 | 0.6545 | 0.7467 | 0.8206 | 29 |
| ViT + Swin | concat | wide_reg | 0.8118 | 0.6878 | 0.6584 | 0.7387 | 0.8197 | 43 |
| ViT + Swin | concat | deep_reg | 0.8305 | 0.7262 | 0.7157 | 0.7434 | 0.8337 | 49 |
| ViT + Swin + DeiT | concat | baseline | 0.8225 | 0.6863 | 0.6618 | 0.7401 | 0.8281 | 29 |
| ViT + Swin + DeiT | concat | wide | 0.8059 | 0.6863 | 0.6650 | 0.7269 | 0.8139 | 15 |
| ViT + Swin + DeiT | concat | wide_reg | 0.7706 | 0.6624 | 0.6122 | 0.7558 | 0.7866 | 13 |
| ViT + Swin + DeiT | concat | deep_reg | 0.7979 | 0.6755 | 0.6288 | 0.7443 | 0.8069 | 15 |
| ViT + Swin + BEiT | concat | baseline | 0.8125 | 0.6988 | 0.6674 | 0.7452 | 0.8177 | 23 |
| ViT + Swin + BEiT | concat | wide | 0.8105 | 0.7080 | 0.6775 | 0.7492 | 0.8191 | 20 |
| ViT + Swin + BEiT | concat | wide_reg | 0.8165 | 0.7159 | 0.6831 | 0.7623 | 0.8225 | 50 |
| ViT + Swin + BEiT | concat | deep_reg | 0.8324 | 0.7144 | 0.6891 | 0.7489 | 0.8342 | 44 |

E2b sonuçları iki ana mesaj verir. Birincisi, ViT single-backbone için daha büyük MLP daha iyi sonuç üretmemiştir; original baseline MLP hâlâ en iyi ViT single sonucudur. İkincisi, concat fusion koşulları MLP kapasitesinden belirgin biçimde etkilenmiştir. Özellikle `ViT + Swin concat`, deep-regularized MLP ile `0.7262` macro-F1 değerine ulaşarak tüm frozen validation sonuçları içinde en yüksek skoru üretmiştir.

Bu bulgu, "fusion katkı sağlamıyor" yorumunun fazla basit olacağını gösterir. Modest MLP altında fusion gain küçük görünürken, daha güçlü classifier altında ViT+Swin feature concat daha ayrıştırıcı hale gelmiştir. Dolayısıyla frozen fusion performansı yalnız backbone complementarity'ye değil, fused representation'ı işleyen classifier kapasitesine de bağlıdır.

## Representation Similarity Diagnostic

BEiT'in neden single-backbone olarak zayıf ama fusion içinde faydalı olabileceğini açıklamak için representation similarity diagnostic hesaplandı. Feature dimension'ları farklı olduğu için raw feature coordinates doğrudan karşılaştırılmadı. Bunun yerine her backbone için validation feature'ları train-only scaling sonrası sample-by-sample cosine similarity matrix'e dönüştürüldü. Daha sonra bu similarity matrix'lerin üst üçgenleri arasındaki Pearson correlation hesaplandı.

Bu yaklaşım şunu ölçer:

> İki backbone, validation örnekleri arasındaki benzerlik ilişkilerini aynı şekilde mi kuruyor?

Düşük representation similarity, iki backbone'un örnekleri farklı bir geometrik yapıda organize ettiğini gösterir. Bu durum potansiyel complementarity sinyali olabilir; ancak tek başına performans garantisi değildir.

| Backbone pair | Representation similarity | Representation complementarity |
|---|---:|---:|
| ViT + Swin | 0.5942 | 0.4058 |
| ViT + DeiT III-Small | 0.5273 | 0.4727 |
| ViT + BEiT | 0.4393 | 0.5607 |
| Swin + DeiT III-Small | 0.5460 | 0.4540 |
| Swin + BEiT | 0.2874 | 0.7126 |
| DeiT III-Small + BEiT | 0.2240 | 0.7760 |

BEiT'in ViT ve Swin ile representation similarity değerleri DeiT'e göre daha düşüktür. Özellikle `Swin + BEiT` similarity `0.2874` ile oldukça düşüktür. Bu, BEiT'in validation örnekleri arasındaki benzerlik ilişkilerini Swin'den oldukça farklı kurduğunu gösterir.

Average pairwise complementarity açısından:

- `ViT + Swin`: `0.4058`
- `ViT + Swin + DeiT III-Small`: `0.4442`
- `ViT + Swin + BEiT`: `0.5597`

Bu sonuç, BEiT'in neden triple concat içinde faydalı olabildiğini destekler. BEiT tek başına yüksek macro-F1 üretmemiştir; ancak ViT ve Swin'in zaten güçlü olduğu bir fusion setup içinde farklı representation geometry sağlayarak ek bilgi sunmuş olabilir.

Bu yorum dikkatli sınırlandırılmalıdır. Düşük similarity her zaman daha iyi fusion anlamına gelmez. Örneğin `Swin + BEiT` pairwise complementarity yüksek olsa da macro-F1 `0.6381` seviyesinde kalmıştır. Bu nedenle complementarity'nin faydalı olabilmesi için güçlü base representation'larla birlikte bilgi kaybı yaratmayan bir fusion operator'ü gerekir. Bu projede bu koşul en iyi `ViT + Swin + BEiT concat` ile sağlanmıştır.

## DeiT'ten BEiT'e Geçiş Kararı

Başlangıçta üçüncü backbone olarak DeiT III-Small seçildi. Bu karar Sprint 2 single-backbone validation sonuçlarıyla da desteklendi:

| Backbone | Validation macro-F1 |
|---|---:|
| ViT | 0.6924 |
| Swin | 0.6115 |
| DeiT III-Small | 0.5017 |
| BEiT-Base | 0.4759 |

Bu tabloya göre sadece single-backbone strength açısından bakıldığında DeiT, BEiT'ten daha makul üçüncü adaydı. Ancak E2 fusion sonuçları farklı bir tablo ortaya koydu. DeiT'in ViT ve Swin'e eklenmesi en iyi fusion sonucunu üretmedi; BEiT ise ViT ve Swin ile birlikte kullanıldığında E2'nin en yüksek validation macro-F1 değerine ulaştı.

Bu nedenle ileri aşamalarda kullanılacak backbone seti şu şekilde güncellendi:

```text
ViT
Swin Transformer
BEiT-Base
```

DeiT III-Small çalışmadan çıkarılmamıştır; raporda planned/screened baseline olarak kalır. Ancak fine-tuning compute budget'ı için üçüncü backbone olarak BEiT seçilmiştir. Bu karar test setine bakılarak verilmemiştir; validation-only fusion matrix ve representation similarity diagnostic sonuçlarına dayanmaktadır.

Rapor için ana yorum:

> DeiT III-Small, single-backbone validation performansı açısından BEiT-Base'ten daha güçlüydü. Ancak feature fusion deneyleri, single-backbone strength ile feature complementarity'nin aynı şey olmadığını gösterdi. BEiT-Base tek başına zayıf kalmasına rağmen ViT ve Swin ile birlikte concat fusion içinde daha tamamlayıcı bir temsil sağladı. Bu nedenle sonraki transfer learning aşamasında üçüncü backbone olarak BEiT-Base kullanıldı.

## Per-Class Observations

Planned matrix içindeki `ViT + Swin concat`, ViT single baseline'a göre bazı sınıflarda artış, bazı sınıflarda düşüş üretmiştir:

| Class | ViT F1 | ViT + Swin concat F1 | Delta |
|---|---:|---:|---:|
| `akiec` | 0.5962 | 0.5263 | -0.0698 |
| `bcc` | 0.6790 | 0.6707 | -0.0083 |
| `bkl` | 0.6687 | 0.7484 | +0.0797 |
| `df` | 0.7179 | 0.6829 | -0.0350 |
| `nv` | 0.8885 | 0.9078 | +0.0192 |
| `mel` | 0.5061 | 0.5815 | +0.0754 |
| `vasc` | 0.7907 | 0.7451 | -0.0456 |

Bu tablo, fusion gain'in tüm sınıflara eşit dağılmadığını gösterir. `bkl` ve `mel` sınıflarında artış gözlenirken `akiec`, `df` ve `vasc` sınıflarında düşüş görülmektedir. Bu nedenle macro-F1 artışı tek başına bütün sınıflar için iyileşme anlamına gelmez.

Bu tür per-class değişimler özellikle minority-class support düşük olduğunda dikkatli yorumlanmalıdır. HAM10000 validation splitinde bazı sınıfların support'u sınırlıdır; birkaç doğru veya yanlış örnek F1 değerini belirgin biçimde değiştirebilir.

## Önceki CNN Çalışmasıyla Karşılaştırmalı Yorum

Bu bölüm final raporun Discussion kısmında ayrı bir alt başlık olarak kullanılabilir. Amaç iki projeyi leaderboard gibi yarıştırmak değildir; çünkü eski `dl-assignment` notları ağırlıklı olarak test macro-F1 raporlarken bu projedeki E1/E2/E2b sonuçları validation-only model selection disiplinini korur ve test setini kullanmaz. Buna rağmen iki çalışma arasındaki davranış farkı, backbone ailesinin ve fusion ihtiyacının nasıl değiştiğini açıklamak için değerlidir.

Önceki CNN çalışmasında frozen single-backbone temsiller görece sınırlıydı. Default class-weighted MLP koşulunda en güçlü single-backbone ResNet50 olmuş ve macro-F1 `0.531` üretmiştir. Frozen feature fusion bu başlangıç noktasını anlamlı biçimde iyileştirmiştir: ResNet50 + EfficientNetB0 concat macro-F1 `0.595` sonucuna ulaşmış, yani default single-backbone baseline'a göre yaklaşık `+0.064` mutlak macro-F1 artışı sağlamıştır. Bu sonuç eski projede fusion'ın ana bilimsel katkılarından biriydi; farklı CNN backbone'ları birbirini tamamlayarak özellikle bazı minority ve orta-support sınıflarda daha dengeli sonuç üretmiştir.

Bu final projesinde tablo farklıdır. Frozen transformer aşamasında Vanilla ViT tek başına validation macro-F1 `0.6924` değerine ulaşmıştır. Bu değer, eski CNN çalışmasındaki frozen fusion sonucunun (`0.595`) oldukça üstündedir ve eski CNN fine-tuned three-backbone concat sonucuna (`0.706`) yakındır. Bu nedenle frozen transformer fusion aşamasında başlangıç baseline'ı çok daha güçlüdür; fusion'ın ekleyebileceği marj doğal olarak daralmıştır.

E2 modest MLP matrix altında fusion kazanımı çok sınırlı kalmıştır. Planned ViT/Swin/DeiT matrix'te en iyi sonuç `ViT + Swin concat` ile `0.6947` olmuş, ViT single baseline'a göre yalnızca `+0.0023` macro-F1 artış sağlamıştır. BEiT-expanded matrix'te `ViT + Swin + BEiT concat` `0.6988` ile en iyi modest-MLP sonucu üretmiştir; bu da ViT single baseline'a göre `+0.0064` artıştır. Bu, eski CNN projesindeki `+0.064` frozen fusion gain'e göre çok daha küçüktür.

E2b MLP capacity diagnostic bu yorumu daha nüanslı hale getirir. Daha güçlü ve daha regülarize MLP kullanıldığında `ViT + Swin concat + deep_reg MLP` validation macro-F1 `0.7262` değerine ulaşmıştır. Bu sonuç ViT single baseline'a göre yaklaşık `+0.0338` artış sağlar ve modest MLP altında görünmeyen bir fusion signal olduğunu gösterir. Buna rağmen eski CNN çalışmasından farklı olarak en iyi sonuç three-backbone fusion'dan değil, güçlü pairwise `ViT + Swin` concat temsilinden gelmiştir.

Karşılaştırmalı özet:

| Study stage | Best relevant model | Metric context | Macro-F1 | Main interpretation |
|---|---|---|---:|---|
| Previous CNN frozen single | ResNet50 | test macro-F1 | 0.531 | CNN frozen representation baseline was limited. |
| Previous CNN frozen fusion | ResNet50 + EfficientNetB0 concat | test macro-F1 | 0.595 | Fusion produced a clear gain over frozen single backbone. |
| Previous CNN fine-tuned fusion | ResNet50 + MobileNetV2 + EfficientNetB0 concat | test macro-F1 | 0.706 | Fine-tuning plus fusion substantially improved the CNN pipeline. |
| Current transformer frozen single | Vanilla ViT | validation macro-F1 | 0.6924 | Frozen transformer representation is already strong. |
| Current transformer fusion, modest MLP | ViT + Swin + BEiT concat | validation macro-F1 | 0.6988 | Fusion gain is positive but small under the baseline classifier. |
| Current transformer fusion, capacity diagnostic | ViT + Swin concat + deep_reg MLP | validation macro-F1 | 0.7262 | High-dimensional transformer fusion benefits from stronger MLP capacity. |

Rapor için ana yorum:

> Önceki CNN tabanlı çalışmada frozen feature fusion, zayıf single-backbone baseline'ı belirgin biçimde iyileştirmişti: ResNet50 + EfficientNetB0 concat, ResNet50 single baseline'a göre yaklaşık `+0.064` macro-F1 artışı sağlamıştı. Bu projede ise Vanilla ViT frozen single-backbone halinde `0.6924` validation macro-F1 değerine ulaşarak çok daha güçlü bir başlangıç noktası oluşturdu. Bu nedenle transformer fusion deneylerinde modest MLP altında ek kazanım küçük kaldı. Ancak E2b classifier-capacity diagnostic, yüksek boyutlu ViT + Swin concat temsilinin daha güçlü bir MLP ile `0.7262` macro-F1'e çıkabildiğini gösterdi. Bu sonuç, transformer temsillerinde fusion'ın tamamen faydasız olmadığını; fakat katkısının CNN projesindeki gibi doğrudan ve büyük değil, classifier capacity, representation redundancy ve fusion operator'ünün bilgi koruma kapasitesiyle daha yakından ilişkili olduğunu göstermektedir.

Bu bölümde özellikle şu sınır korunmalıdır: eski CNN çalışmasındaki test skorları ile bu projedeki validation skorları doğrudan aynı değerlendirme split'i gibi karşılaştırılmamalıdır. Karşılaştırmanın amacı, "transformer daha yüksek leaderboard skoru aldı" iddiası değil, frozen representation strength ve fusion gain davranışının iki backbone ailesinde farklı olduğunu göstermektir.

## Report-Ready Method Paragraph

> Frozen feature fusion aşamasında, önceki aşamada çıkarılan fixed transformer feature cache'leri kullanılmıştır. Her backbone feature block'u yalnız train split üzerinde fit edilen StandardScaler ile normalize edilmiş ve validation split'e aynı transform uygulanmıştır. Feature cache'leri `sample_id`, `image_id`, `lesion_id`, label ve split row order üzerinden doğrulanmıştır. Üç fusion yöntemi karşılaştırılmıştır: doğrudan concatenation, 512-dimensional trainable projection kullanan global softmax-weighted fusion ve 384-dimensional train-only PCA projection sonrası weighted fusion. Tüm modeller class-weighted MLP classifier ile eğitilmiş, checkpoint ve model seçimi validation macro-F1 üzerinden yapılmıştır. Test split bu aşamada kullanılmamıştır.

## Report-Ready Result Paragraph

> Planned ViT/Swin/DeiT fusion matrix içinde en yüksek validation macro-F1 `ViT + Swin concat` ile elde edilmiştir (`0.6947`). Bu sonuç en güçlü single-backbone baseline olan ViT'in macro-F1 değerini (`0.6924`) yalnızca küçük bir farkla geçmiştir. DeiT III-Small'ın üçüncü backbone olarak eklenmesi performansı artırmamış, `ViT + Swin + DeiT concat` `0.6863` macro-F1 değerinde kalmıştır. Daha sonra yapılan BEiT-expanded E2 matrix içinde `ViT + Swin + BEiT concat` `0.6988` validation macro-F1 ile en iyi sonucu üretmiştir. Bu bulgu, BEiT'in single-backbone olarak zayıf olmasına rağmen ViT ve Swin ile birlikte kullanıldığında tamamlayıcı temsil bilgisi sağlayabildiğini göstermektedir.

## Report-Ready Discussion Paragraph

> Sonuçlar, feature fusion performansının yalnızca tek backbone gücüyle açıklanamayacağını göstermektedir. DeiT III-Small, single-backbone screening aşamasında BEiT-Base'ten daha yüksek macro-F1 üretmiş olsa da, fusion aşamasında BEiT'in ViT ve Swin'e daha tamamlayıcı bir temsil sunduğu görülmüştür. Representation similarity diagnostic bu yorumu desteklemektedir: BEiT'in ViT ve Swin ile sample-similarity correlation değerleri DeiT'e göre daha düşüktür. Buna karşın BEiT pairwise fusion koşulları ViT baseline'ı geçmemiştir; en iyi sonuç yalnızca ViT ve Swin ile birlikte triple concat yapıldığında elde edilmiştir. Bu durum, complementarity'nin faydalı olabilmesi için hem güçlü kaynak temsillerin hem de bilgi kaybı yaratmayan fusion operator'ünün gerekli olduğunu göstermektedir.

## Evidence

Generated evidence:

```text
artifacts/runs/*_s3_frozen_*_seed42/
artifacts/runs/s3_frozen_fusion_manifest.json
artifacts/report_assets/tables/frozen_fusion_results.csv
artifacts/report_assets/tables/frozen_fusion_per_class_metrics.csv
artifacts/report_assets/tables/frozen_fusion_weight_summary.csv
artifacts/report_assets/tables/frozen_fusion_vs_single_validation.csv
artifacts/report_assets/tables/frozen_representation_similarity_val.csv
artifacts/report_assets/tables/frozen_fusion_complementarity_val.csv
artifacts/report_assets/tables/frozen_representation_similarity_val.json
artifacts/report_assets/figures/frozen_fusion_macro_f1.png
artifacts/report_assets/figures/frozen_representation_similarity_val.png
```

Verification evidence:

- E2 expanded fusion matrix completed with `21` validation-only fusion runs.
- Train row count matched `7008`.
- Validation row count matched `1504`.
- Every prediction dump contains `1504` validation rows.
- Feature cache alignment checks passed.
- Feature tensors and fusion tensors contained no NaN/Inf.
- Weighted fusion softmax weights summed to 1.
- PCA metadata records train-only fit and `uses_labels=false`.
- Representation similarity diagnostic used validation split, train-only scaling and no test labels.
- Confusion matrix label order remained fixed.
- Generated run artifacts and report assets are Git-ignored.

## Limitations

- Results are single-seed (`42`) validation results.
- Test metrics were intentionally not computed.
- The best gain over ViT single-backbone baseline is small.
- BEiT pairwise fusion did not beat ViT single; BEiT's benefit appeared only in the triple concat setting.
- Representation similarity is a post-hoc validation diagnostic, not a standalone model selection metric.
- Weighted fusion weights should not be interpreted as direct backbone quality rankings.
- PCA compression may discard class-discriminative information because it is unsupervised.
- Results support benchmark dermoscopic image classification analysis only; they do not support clinical diagnosis or deployment claims.

## Final Report Decision

Forward backbone set for the next transfer learning stage:

```text
vit_b16
swin_tiny
beit_base
```

DeiT III-Small remains part of the scientific story as a planned/screened third backbone and as evidence that stronger single-backbone validation performance does not necessarily imply stronger fusion complementarity.
