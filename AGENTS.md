# Agent Guide

Bu repository, Derin Öğrenme final projesi için ViT tabanlı feature extraction ve feature fusion çalışmasıdır. Önceki CNN tabanlı dönem projesinden öğrenilen workflow dersleri korunur; ancak implementasyon ve rapor anlatısı transformer backbone'larına göre yeniden kurulmalıdır.

## Start Here

- Assignment summary: `docs/ASSIGNMENT_BRIEF.md`
- Folder structure: `PROJECT_FOLDER_STRUCTURE.md`
- Project context: `docs/PROJECT_CONTEXT.md`
- Evaluation protocol: `docs/EVALUATION_PROTOCOL.md`
- Experiment registry: `docs/EXPERIMENT_REGISTRY.md`
- Artifact standard: `docs/ARTIFACT_STANDARD.md`
- Reporting guidelines: `docs/REPORTING_GUIDELINES.md`
- Lessons learned: `docs/LESSONS_LEARNED.md`
- Initial roadmap: `docs/planning/INITIAL_ROADMAP.md`
- Execution plans: `docs/exec-plans/`
- Literature scaffold: `docs/literature/`

## Working Rules

- Kod implementasyonuna başlamadan önce ilgili deney `docs/EXPERIMENT_REGISTRY.md` içinde tanımlanmalıdır.
- Substantial işler için `docs/exec-plans/active/` altında plan açılmalı, iş tamamlanınca `docs/exec-plans/completed/` altına taşınmalıdır.
- Test seti model seçimi için kullanılmamalıdır. Test sonuçları yalnız final audit veya diagnostic audit olarak etiketlenmelidir.
- Her önemli run için config, metrics, per-class metrics, prediction dump ve report note üretilmelidir.
- Dataset, checkpoint, feature cache, prediction dump ve büyük artifact dosyaları Git'e alınmamalıdır.
- HAM10000 klinik teşhis bağlamında sunulmamalıdır. Dil "benchmark dermoscopic image classification" çizgisinde kalmalıdır.
- Final rapor Türkçe yazılacaktır. Domain-standard terimler gerektiğinde İngilizce bırakılabilir: feature extraction, fine-tuning, backbone, MLP, fusion, TTA, validation/test split, macro-F1.
- Final akademik raporda "sprint" gibi iç proje yönetimi terimleri kullanılmamalıdır. Bunlar dataset preparation, frozen feature extraction, feature fusion, fine-tuning, ablation ve diagnostic analysis olarak çevrilmelidir.
- Notebooks ince launcher olarak kalmalıdır; gerçek logic ileride `src/` ve `scripts/` altında tutulmalıdır.
- Literatür çalışması yapılırsa ham synthesis `deep-research-report-literature.md`, düzenli registry/index ise `docs/literature/` altında tutulmalıdır.

## No-Code Scaffold Status

Bu ilk scaffold aşamasında kaynak kod eklenmemiştir. `src/`, `scripts/` ve `notebooks/` klasörleri yalnızca rol tanımlarıyla ayrılmıştır. İlk kod eklenmeden önce environment ve package kararları ayrıca verilmelidir.
