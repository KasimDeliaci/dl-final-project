# Artifacts Directory

Bu klasör generated artifact dosyaları için ayrılmıştır ve Git dışında tutulur.

Alt klasör rolleri:

- `raw/`: dış kaynaklardan gelen ham çalışma çıktıları veya bundle referansları
- `features/`: frozen ve fine-tuned feature cache dosyaları
- `predictions/`: per-sample prediction dump dosyaları
- `figures/`: rapor/sunum için üretilecek grafikler
- `logs/`: training ve evaluation logları
- `models/`: checkpoint ve model ağırlıkları

Artifact standardı için `docs/ARTIFACT_STANDARD.md` dosyasına bak.

