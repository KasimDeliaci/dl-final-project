# DL Final Project

Bu proje, HAM10000 dermoskopik görüntü sınıflandırma veri seti üzerinde transformer tabanlı özellik çıkarımı ve özellik füzyonu deneyleri için hazırlanmıştır.

Mevcut dönem projesindeki CNN tabanlı yaklaşımın genel deney disiplini korunur; ancak backbone ailesi CNN yerine ViT tabanlı modellerdir. Amaç yalnızca yüksek skor üretmek değil, farklı transformer temsillerinin ne öğrendiğini, feature fusion'ın ne zaman fayda sağladığını ve frozen feature extraction ile fine-tuning arasındaki farkı kontrollü biçimde açıklamaktır.

## Kapsam

Zorunlu çizgi:

- Vanilla ViT
- Swin Transformer
- DeiT III-Small
- Her backbone'dan feature vector extraction
- Concatenation fusion
- Weighted fusion
- MLP classifier
- Single-backbone, two-backbone ve three-backbone karşılaştırmaları
- Frozen feature extraction ve son transformer blokları fine-tuning karşılaştırması
- Accuracy, precision, recall ve F1-score raporlaması

## Başlangıç Kararı

Bu repo koddan önce deney düzenini sabitlemek için scaffold edilmiştir. İlk implementasyon başlamadan önce şu dosyalar okunmalıdır:

- `PROJECT_FOLDER_STRUCTURE.md`
- `docs/ASSIGNMENT_BRIEF.md`
- `docs/PROJECT_CONTEXT.md`
- `docs/EVALUATION_PROTOCOL.md`
- `docs/EXPERIMENT_REGISTRY.md`
- `docs/ARTIFACT_STANDARD.md`
- `docs/REPORTING_GUIDELINES.md`
- `docs/LESSONS_LEARNED.md`
- `docs/planning/5-sprint-project-plan.md`
- `docs/literature/README.md`

## Ana İlke

Her deney şu sorulara cevap vermeden başlamamalıdır:

1. Neyi ölçmek istiyoruz?
2. Hangi kontrol koşulu sabit kalacak?
3. Model veya hyperparameter seçimi hangi validation kuralıyla yapılacak?
4. Test seti yalnız audit olarak mı kullanılacak?
5. Rapor ve sunum için hangi artifact üretilecek?

Bu proje eski CNN çalışmasının kopyası olmayacak; aynı dataset ve benzer MLP/fusion hattı korunurken transformer backbone davranışı ayrı bir deney nesnesi olarak incelenecektir.

## Scaffold Notu

`configs/`, `docs/literature/`, `reports/`, `tests/`, `outputs/` ve `submission/` klasörleri eski dönem projesindeki yapıya benzer şekilde hazırlanmıştır. Bu aşamada bu dosyalar implementation değil, ileride yapılacak işleri düzenli tutacak placeholder ve çalışma sözleşmeleridir.
