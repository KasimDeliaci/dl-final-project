# Scripts Directory

Bu klasör command-line runner script'leri için ayrılmıştır.

Sprint 1 script'leri:

- `audit_ham10000.py`: HAM10000 metadata/image audit ve class distribution artifact üretimi.
- `make_lesion_split.py`: canonical lesion-aware train/validation/test split üretimi.
- `smoke_dataloader.py`: split CSV'lerinden örnek image open smoke check.

Script'ler notebook içine gömülü logic yerine reproducible çalıştırma noktaları olmalıdır.

Sonraki script rolleri:

- feature extraction,
- MLP training/evaluation,
- fusion experiment runner,
- fine-tuning runner,
- report figure generation.
