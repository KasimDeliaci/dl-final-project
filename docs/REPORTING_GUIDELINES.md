# Reporting Guidelines

Bu proje rapor ve sunumunda ana ton şu olmalıdır:

> "X denendi, skor Y oldu" değil; "hangi reçeteyle denendi, hangi kontrol sabit tutuldu, neyi ölçmek istedik, sonuç neden böyle yorumlandı ve final karara neden girdi/girmedi?"

## Her Deney İçin Anlatı Şablonu

Her önemli deney şu sırayla anlatılmalıdır:

1. Hipotez
2. Değiştirilen parça
3. Sabit tutulan kontrol
4. Ölçülen sonuç
5. Neden böyle oldu?
6. Kanıt gücü
7. Final modele etkisi

## Transformer Projesine Özel Sorular

Rapor ve sunum şu sorulara net cevap vermelidir:

- Vanilla ViT, Swin ve ek transformer backbone hangi sınıflarda güçlü veya zayıf?
- CLS token veya pooling kararı feature kalitesini nasıl etkileyebilir?
- Concat fusion gerçekten tamamlayıcı bilgi topluyor mu, yoksa yalnız feature dimension'ı mı büyütüyor?
- Weighted fusion zayıf backbone'u bastırıp güçlü backbone'u öne çıkarıyor mu?
- Fine-tuning representation quality'yi artırıyor mu, yoksa overfitting mi yaratıyor?
- Final model seçimi test skoruna değil hangi validation kuralına dayanıyor?

## Görsel ve Tablo İlkeleri

Minimum rapor asset'leri:

- class distribution figure/table,
- single-backbone baseline table,
- fusion comparison table,
- frozen vs fine-tuned comparison table/figure,
- final confusion matrix,
- final per-class F1 chart,
- training curve, yalnız fine-tuning içeren önemli deneyler için.

Her tablo veya figür metinde referanslanmalıdır. Görsel yalnız dekorasyon için kullanılmamalıdır.

## Sunum Tonu

Sunumda her ana deney için konuşma cue'su:

```text
Kontrol: ne sabit kaldı?
Reçete: neyi değiştirdik?
Sonuç: ne ölçtük?
Karar: final çizgiye girdi mi, girmedi mi?
```

Bu yapı özellikle hoca sorularına karşı savunulabilirlik sağlar.

