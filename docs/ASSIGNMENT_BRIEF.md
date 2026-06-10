# Assignment Brief

Bu özet, `dl-final-projesi-2026.docx` dosyasındaki final proje gereksinimlerinden çıkarılmıştır. Kaynak belge korunur; bu dosya çalışma sırasında hızlı başvuru içindir.

## Amaç

Projede vision transformer tabanlı modeller kullanılarak:

- feature extraction yapılması,
- farklı transformer mimarilerinin öğrendiği temsillerin karşılaştırılması,
- feature fusion yöntemlerinin uygulanması,
- elde edilen özelliklerin MLP classifier ile değerlendirilmesi

beklenmektedir.

## Sabitlenmiş Ödev Constraintleri

Bu proje için kod yazmadan önce sabitlenen zorunlu çizgi:

- Dataset: HAM10000, public ve çok sınıflı dermoscopic image classification benchmark.
- Backbone seti: Vanilla ViT, Swin Transformer, DeiT III-Small.
- Feature extraction: Her backbone'dan feature vector alınacak; classification head kullanılmayacak.
- Fusion yöntemleri: Concatenation ve weighted fusion.
- Classifier: Single veya fused feature vector üzerinde MLP classifier.
- Karşılaştırmalar: Single-backbone, pairwise fusion ve three-backbone fusion.
- Transfer learning karşılaştırması: Frozen feature extraction ve son transformer blokları fine-tuning.
- Metrikler: Accuracy, precision, recall ve F1-score; ana yorum metriği macro-F1.
- Evaluation discipline: Model seçimi validation üzerinden yapılacak; test seti yalnız audit olarak kullanılacak.

Bu maddeler proje kapsamını tanımlar. Ek fikirler ancak bu matrisi bozmadığı ve `docs/EXPERIMENT_REGISTRY.md` içinde açık hipotezle kaydedildiği sürece yapılabilir.

## Dataset

Dataset seçimi gruba bırakılmıştır. Bu proje, önceki dönem projesiyle tutarlı kalmak için HAM10000 dermoskopik görüntü sınıflandırma dataset'i üzerinden ilerleyecektir.

Dataset için raporda:

- public erişim kaynağı,
- sınıf sayısı ve sınıf dağılımı,
- son yıllardaki literatür kullanımı,
- train/validation/test split politikası,
- leakage kontrolü

açıkça verilmelidir.

## Zorunlu Model Ailesi

Belgede iki model özellikle istenmiştir:

- Vanilla ViT
- Swin Transformer

Ek model kullanımı bonus getirebilir. Bu proje için üçüncü backbone olarak `DeiT III-Small` seçilmiştir. BEiT kapsam dışı bırakılmıştır; gerekçe `docs/DECISIONS.md` içinde kayıtlıdır.

## Zorunlu Feature Extraction

Her transformer backbone için:

- son classification head kaldırılmalı,
- ara katman veya son temsil katmanından feature vector elde edilmeli,
- feature dimension ve pooling/token seçimi raporda belirtilmelidir.

ViT tabanlı modellerde bu karar özellikle önemlidir: CLS token, global average pooled patch tokens veya model-specific pooled output farklı temsil davranışları üretebilir.

## Zorunlu Fusion

En az iki fusion yöntemi uygulanmalıdır:

- Concatenation
- Weighted fusion

Fusion yorumları yalnız skor üzerinden yapılmamalıdır. Hangi backbone temsillerinin tamamlayıcı olduğu, hangi fusion yönteminin neden fayda sağladığı veya sağlamadığı tartışılmalıdır.

## Zorunlu Classifier

Fused veya single-backbone feature vector'ler MLP classifier ile sınıflandırılacaktır.

Classifier raporunda:

- hidden layer yapısı,
- dropout veya regularization kullanımı,
- optimizer,
- learning rate,
- epoch/patience,
- class weighting kullanılıp kullanılmadığı

belirtilmelidir.

## Zorunlu Deneyler

Mutlaka karşılaştırılacak koşullar:

- single transformer backbone
- two-backbone fusion
- three-backbone fusion
- concatenation fusion
- weighted fusion
- frozen feature extraction
- fine-tuning of last transformer blocks

Her koşul aynı split ve aynı evaluation protocol ile karşılaştırılmalıdır.

## Minimum Deney Matrisi

Implementation başlamadan önce korunacak minimum deney matrisi:

```text
Backbone baselines:
  - ViT single
  - Swin single
  - DeiT III-Small single

Frozen feature fusion:
  - ViT + Swin
  - ViT + DeiT III-Small
  - Swin + DeiT III-Small
  - ViT + Swin + DeiT III-Small
  - Her kombinasyon için concat ve weighted fusion

Fine-tuning comparison:
  - Her backbone için son transformer blokları fine-tuning
  - Compute izin verirse temsilci pairwise ve three-backbone fusion

Final audit:
  - Validation ile seçilen final configuration
  - Test seti üzerinde tek final audit
```

Bu matris ileride Colab maliyeti nedeniyle daraltılırsa karar `docs/DECISIONS.md` içinde ayrıca kaydedilmelidir.

## Zorunlu Metrikler

Raporlanacak metrikler:

- accuracy
- precision
- recall
- F1-score

Bu projede sınıf dengesizliği nedeniyle ana yorum metriği macro-F1 olacaktır. Accuracy destekleyici metrik olarak raporlanacaktır.

## Tartışma Soruları

Raporun en kritik bölümü tartışmadır. Şu sorular açıkça cevaplanmalıdır:

- Hangi transformer backbone daha iyi özellik öğrenmiştir? Neden?
- Feature fusion performansı neden artırmış veya artırmamış olabilir?
- Hangi model/fusion/fine-tuning kombinasyonu en iyi sonucu vermiştir?
- Modellerin güçlü ve zayıf yönleri nelerdir?
- Frozen feature extraction ile fine-tuning arasında performans ve süre farkı nedir?

## Teslim

Teslim paketinde şunlar beklenir:

- kaynak kod,
- PDF rapor,
- sunum veya Youtube video linki,
- varsa eğitim logları ve grafikler.
