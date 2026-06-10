# Artifact Standard

Bu proje, deney sonunda yalnız skor üretmemelidir. Her önemli run, rapor ve sunumda savunulabilir kanıt bırakmalıdır.

## Directory Roles

- `artifacts/raw/`: indirilen veya dışarıdan gelen ham veri referansları; mümkünse gerçek büyük veri Git'e alınmaz.
- `artifacts/features/`: frozen veya fine-tuned feature cache dosyaları.
- `artifacts/predictions/`: per-sample prediction dump dosyaları.
- `artifacts/figures/`: rapor ve sunum için üretilen grafikler.
- `artifacts/logs/`: training logs, run stdout, config snapshots.
- `artifacts/models/`: checkpoint ve model ağırlıkları.

Bu klasörler Git dışında tutulur. Rapor/sunumda kullanılacak küçük final figürleri ileride bilinçli olarak `reports/` altına kopyalanabilir.

## Required Run Files

Her önemli run şu dosyaları üretmelidir:

```text
run_config.json
metrics_summary.csv
per_class_metrics.csv
predictions.csv
confusion_matrix.png
report_note.md
```

Training içeren run'lar ayrıca şunları üretmelidir:

```text
training_log.csv
training_curves.png
checkpoint_metadata.json
```

## Prediction Dump Columns

`predictions.csv` minimum kolonları:

```text
sample_id
image_id
lesion_id
split
true_label
pred_label
correct
confidence
prob_<class_1>
prob_<class_2>
...
prob_<class_n>
```

Bu standart, daha sonra error overlap, model disagreement, confidence analysis ve per-class failure analysis yapılabilmesi için zorunludur.

## Report Note Template

Her run sonunda kısa not:

```text
Question:
Recipe:
Fixed controls:
Result:
Interpretation:
Evidence strength:
Report decision:
```

Bu not yazılmadan deney "tamamlandı" sayılmamalıdır.

