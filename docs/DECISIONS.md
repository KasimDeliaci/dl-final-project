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
