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
