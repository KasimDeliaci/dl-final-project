# Dataset Audit

Bu dosya Phase 0 sırasında doldurulacaktır. Şu anda dataset audit yapılmamıştır; aşağıdaki başlıklar audit tamamlandığında artifact ve tablo referanslarıyla güncellenmelidir.

## Dataset Identity

- Dataset:
- Source:
- Download/access path:
- Metadata file:
- Image directory:
- Number of images:
- Number of classes:

## Class Distribution

Raporlanacaklar:

- train/validation/test öncesi genel sınıf dağılımı,
- split sonrası sınıf dağılımı,
- minority-class support değerleri,
- accuracy ve macro-F1 yorumunu etkileyebilecek dengesizlikler.

## Leakage Audit

Kontrol edilecekler:

- `lesion_id` metadata içinde var mı?
- Aynı lesion ID birden fazla split'e düşüyor mu?
- Duplicate image veya aynı lesion'a ait varyantlar split'ler arasında sızıyor mu?

Kabul kriteri:

- Mümkünse hiçbir lesion ID train/validation/test arasında paylaşılmamalıdır.
- Eğer class support nedeniyle compromise gerekirse bu durum raporda açıkça belirtilmelidir.

## Split Policy

Hedef:

- Train: yaklaşık 70%
- Validation: yaklaşık 15%
- Test: yaklaşık 15%
- Stratified ve mümkünse group-aware.

## Preprocessing Notes

Transformer backbone'lar için preprocessing kararı ayrıca belgelenmelidir:

- image size,
- resize/crop policy,
- normalization statistics,
- train-time augmentation olup olmayacağı,
- frozen feature extraction ile fine-tuning preprocessing farkları.

## Audit Artifacts

Beklenen dosyalar:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
artifacts/figures/class_distribution.png
artifacts/logs/dataset_audit.json
```

