# Configs

Bu klasör ileride reproducible deney ayarları için kullanılacaktır.

Şu aşamada config dosyaları implementation değil, kararları ve beklenen alanları görünür kılan scaffold dosyalarıdır.

Beklenen roller:

- `default.yaml`: genel proje/runtime varsayımları.
- `dataset/selected_dataset.yaml`: HAM10000 dataset ve split politikası.
- `backbones/`: ViT/Swin/ek transformer adayları.
- `experiments/`: frozen, fusion ve fine-tuning deney matrisleri.
- `report_assets.yaml`: rapor/sunum asset path sözleşmesi.

