# Project Context

Bu proje, HAM10000 üzerinde çoklu transformer backbone feature extraction ve feature fusion çalışmasıdır.

Önceki dönem projesinde ResNet50, MobileNetV2 ve EfficientNetB0 ile CNN tabanlı özellik füzyonu denenmişti. Bu final projesinde aynı genel problem ve benzer MLP classifier fikri korunur; ancak temsil öğrenme tarafı vision transformer modellerine taşınır.

## Ödev Constraintleri

Bu projenin assignment tarafından zorunlu tuttuğu ana çizgi:

- Vanilla ViT, Swin Transformer ve ek transformer backbone kullanımı.
- Bu repo için ek backbone kararı: DeiT III-Small.
- Her backbone için classification head kaldırılarak feature vector extraction.
- Concatenation fusion ve weighted fusion.
- Single-backbone, two-backbone ve three-backbone karşılaştırmaları.
- Fused veya single-backbone feature vector üzerinde MLP classifier.
- Frozen feature extraction ile son transformer blokları fine-tuning karşılaştırması.
- Accuracy, precision, recall ve F1-score raporlaması.

## Araştırma Sorusu

HAM10000 gibi sınıf dengesizliği yüksek, orta ölçekli bir dermoskopik görüntü dataset'inde farklı ViT tabanlı backbone'ların çıkardığı temsiller:

1. tek başına ne kadar güçlüdür,
2. birbiriyle ne kadar tamamlayıcıdır,
3. concatenation ve weighted fusion ile ne kadar iyileşir,
4. frozen feature extraction yerine son transformer blokları fine-tune edildiğinde nasıl değişir?

## Başlangıç Backbone Seti

Zorunlu:

- Vanilla ViT
- Swin Transformer

Üçüncü backbone:

- DeiT III-Small

BEiT bu proje kapsamında kullanılmayacaktır. Gerekçe: BEiT masked image modeling ile farklı bir temsil çizgisi sunsa da HAM10000 özelinde DeiT III için daha doğrudan performans sinyali vardır; ayrıca DeiT III-Small Colab maliyeti açısından daha yönetilebilir bir üçüncü backbone sağlar.

## Ana Deney Ekseni

Deneyler üç ana eksende tutulacaktır:

1. **Representation quality:** Tek backbone frozen ve fine-tuned sonuçları.
2. **Feature complementarity:** Pairwise ve three-backbone concat/weighted fusion sonuçları.
3. **Transfer learning effect:** Frozen feature extraction ile son transformer blokları fine-tuning karşılaştırması.

Bu eksenlerin dışındaki fikirler ancak açık hipotez ve artifact standardı varsa yapılmalıdır.

## Kontrol İlkeleri

- Dataset split sabit kalmalıdır.
- Test seti seçim için kullanılmamalıdır.
- MLP classifier policy karşılaştırılabilir koşullarda sabit tutulmalıdır.
- Fusion sonuçları feature cache kaynağı karışmadan raporlanmalıdır.
- Fine-tuned ve frozen cache sonuçları aynıymış gibi karşılaştırılmamalıdır; feature source her tabloda açıkça belirtilmelidir.

## Beklenen Rapor Hikayesi

Final raporun ana hikayesi şu sırayla kurulacaktır:

1. Dataset ve leakage-safe split.
2. Transformer backbone'lardan frozen feature extraction.
3. Single-backbone baseline karşılaştırması.
4. Concatenation ve weighted fusion karşılaştırması.
5. Fine-tuning recipe ve frozen/fine-tuned farkı.
6. Final model seçimi ve test audit.
7. Neden-sonuç odaklı tartışma.
