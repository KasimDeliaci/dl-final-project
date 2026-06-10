# Initial Roadmap

Bu roadmap kod yazmadan önce proje akışını sabitlemek için hazırlanmıştır. Aşamalar gerekirse daraltılabilir; daraltma kararı `docs/DECISIONS.md` içine yazılmalıdır.

Her phase başlamadan önce `docs/exec-plans/active/` altında o phase'e ait kısa execution plan açılmalıdır. Phase tamamlandığında plan `docs/exec-plans/completed/` altına taşınmalı ve ilgili sonuç notu `docs/report_notes/` altında yazılmalıdır.

## Phase 0 - Setup and Dataset Audit

Amaç:

- HAM10000 metadata ve image path yapısını doğrulamak.
- Lesion-aware split üretmek veya mevcut split'i doğrulamak.
- Class distribution ve leakage audit artifact'lerini hazırlamak.

Çıktılar:

- `docs/DATASET_AUDIT.md`
- split CSV files
- class distribution figure/table

## Phase 1 - Frozen Transformer Features

Amaç:

- Vanilla ViT, Swin ve ek transformer backbone için frozen feature extraction yapmak.
- Her backbone'un single-model MLP performansını ölçmek.

Kontrol:

- Split aynı.
- MLP recipe aynı.
- Test yalnız audit.

Çıktılar:

- feature caches
- single-backbone baseline table
- per-class metrics
- prediction dumps

## Phase 2 - Frozen Feature Fusion

Amaç:

- Pairwise ve three-backbone concat fusion denemek.
- Pairwise ve three-backbone weighted fusion denemek.
- Fusion'ın gerçekten tamamlayıcı sinyal sağlayıp sağlamadığını yorumlamak.

Çıktılar:

- fusion comparison table
- feature dimension log
- per-class delta veya per-class F1 figure
- report notes

## Phase 3 - Fine-Tuning

Amaç:

- Son transformer bloklarını kontrollü biçimde fine-tune etmek.
- Frozen feature extraction ile fine-tuned representation arasındaki farkı ölçmek.

Kontrol:

- Aynı split.
- Validation macro-F1 checkpoint selection.
- Aynı final MLP evaluation protocol.

Çıktılar:

- training logs
- validation curves
- fine-tuned feature caches
- frozen vs fine-tuned comparison

## Phase 4 - Final Selection and Audit

Amaç:

- Final model/fusion/transfer-learning kararını validation discipline ile vermek.
- Test seti üzerinde tek final audit yapmak.
- Final confusion matrix ve per-class F1 yorumunu hazırlamak.

Çıktılar:

- final leaderboard
- final confusion matrix
- final per-class F1 chart
- final report decision note

## Phase 5 - Report and Presentation

Amaç:

- Raporu "reçete, kontrol, sonuç, yorum" formatında yazmak.
- Sunumu soru-cevap savunmasına uygun şekilde hazırlamak.

Çıktılar:

- PDF report
- presentation PDF
- video outline or speaking notes
