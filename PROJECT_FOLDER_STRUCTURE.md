# Project Folder Structure

Bu repository, HAM10000 üzerinde transformer tabanlı feature extraction ve feature fusion final projesi için doküman-öncelikli, script-driven bir araştırma workflow'u izler.

```text
Git repository
  code + configs + docs + report source + lightweight placeholders

Local / Colab storage
  HAM10000 images + checkpoints + feature caches + prediction dumps + large artifacts

Colab
  GPU-heavy fine-tuning runner

Markdown docs
  proje hafızası, deney registry'si, evaluation protocol, literature notes ve report evidence
```

## Main Rules

- Colab bir runner olarak kullanılacaktır; proje hafızası ve kararlar repo dokümanlarında tutulacaktır.
- Core logic ileride `src/` altında paketlenecektir.
- Entry point script'leri ileride `scripts/` altında tutulacaktır.
- Reproducible ayarlar `configs/` altında tutulacaktır.
- Büyük veri, checkpoint, feature cache ve prediction dump dosyaları Git'e alınmayacaktır.
- Önemli kararlar `docs/DECISIONS.md` içine yazılacaktır.
- Her önemli deney başlamadan önce `docs/EXPERIMENT_REGISTRY.md` içinde kayıt açılacaktır.

## Layout

```text
dl-final/
├── AGENTS.md
├── README.md
├── PROJECT_FOLDER_STRUCTURE.md
├── .gitignore
│
├── configs/
│   ├── default.yaml
│   ├── report_assets.yaml
│   ├── dataset/
│   ├── backbones/
│   └── experiments/
│
├── docs/
│   ├── ASSIGNMENT_BRIEF.md
│   ├── PROJECT_CONTEXT.md
│   ├── DATASET_AUDIT.md
│   ├── EVALUATION_PROTOCOL.md
│   ├── EXPERIMENT_REGISTRY.md
│   ├── ARTIFACT_STANDARD.md
│   ├── REPORTING_GUIDELINES.md
│   ├── DECISIONS.md
│   ├── COMMANDS.md
│   ├── LESSONS_LEARNED.md
│   ├── planning/
│   ├── exec-plans/
│   ├── literature/
│   └── report_notes/
│
├── notebooks/
├── scripts/
├── src/
├── tests/
├── data/
├── artifacts/
├── outputs/
├── reports/
│   ├── final_report/
│   └── presentation/
└── submission/
```

## Documentation Roles

- `docs/ASSIGNMENT_BRIEF.md`: final proje docx gereksinimlerinin çalışma özeti.
- `docs/PROJECT_CONTEXT.md`: güncel bilimsel/teknik bağlam.
- `docs/EVALUATION_PROTOCOL.md`: metrikler, split, leakage ve reporting kuralları.
- `docs/EXPERIMENT_REGISTRY.md`: deney başlamadan doldurulacak hipotez/kontrol kayıtları.
- `docs/ARTIFACT_STANDARD.md`: her run için üretilecek kanıt dosyaları.
- `docs/REPORTING_GUIDELINES.md`: rapor ve sunum anlatı tonu.
- `docs/exec-plans/`: task-level execution plan kayıtları.
- `docs/literature/`: paper registry, literature index ve tema notları.
- `docs/report_notes/`: deney sonrası kısa yorum notları.

