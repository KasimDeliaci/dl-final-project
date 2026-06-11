# Fine-Tuned Transformer Features Report Note

Bu not, final rapordaki "fine-tuned transformer features" ve "frozen vs fine-tuned feature transfer"
bölümlerine malzeme sağlamak için hazırlanmıştır. Akademik raporda iç proje yönetimi dili ana anlatı
olarak kullanılmamalıdır.

Çalışma klinik teşhis iddiası taşımaz. Tüm sonuçlar HAM10000 üzerinde benchmark dermoscopic image
classification bağlamında yorumlanmalıdır.

## Question

Kontrollü partial fine-tuning sonrası çıkarılan transformer feature'ları, frozen feature baseline'a
göre validation macro-F1 ve per-class behavior açısından iyileşme sağlıyor mu?

Alt sorular:

- Fine-tuned single backbones, frozen single baselines üstüne çıkıyor mu?
- Fine-tuned concat fusion, frozen fusion baselines üstüne çıkıyor mu?
- Fine-tuning sonrası BEiT üçüncü backbone olarak faydalı kalıyor mu?
- Fine-tuning gain hangi sınıflarda görülüyor?
- Fine-tuning compute/runtime maliyeti bu kazanımı makul kılıyor mu?

## Fine-Tuning Policy

Forward backbone seti:

```text
vit_b16
swin_tiny
beit_base
```

`deit3_small`, Sprint 2-3 içinde planned/screened baseline olarak kalır; Sprint 4 fine-tuning compute
budget'ına dahil edilmez.

Partial fine-tuning policy:

| Backbone | Trainable region | Temporary head | Feature extraction after selection |
|---|---|---|---|
| Vanilla ViT | last 2 transformer blocks + norm | 7-class head | CLS-token, 768 dim |
| Swin Transformer | last Swin stage + norm | 7-class head | avg pooled final-stage, 768 dim |
| BEiT-Base | last 2 transformer blocks + fc_norm/norm | 7-class head | avg pooled patch-token, 768 dim |

Checkpoint selection uses validation macro-F1 only. Test metrics are not computed in this stage.

## Fixed Controls

- Dataset: HAM10000 benchmark dermoscopic image classification.
- Split: fixed lesion-aware train/validation/test split from Sprint 1.
- Train rows: `7008`.
- Validation rows: `1504`.
- Test split: not used for Sprint 4 metrics or selection.
- Class order: `akiec`, `bcc`, `bkl`, `df`, `nv`, `mel`, `vasc`.
- Downstream classifier: cached-feature MLP with train-only StandardScaler and train-only class
  weights.
- Primary metric: validation macro-F1.

## Frozen Baselines To Compare Against

Single-backbone frozen baselines:

| Frozen baseline | Validation macro-F1 |
|---|---:|
| ViT | `0.6924` |
| Swin | `0.6115` |
| BEiT screening | `0.4759` |

Fusion baselines:

| Frozen fusion baseline | Validation macro-F1 | Role |
|---|---:|---|
| ViT + Swin + BEiT concat, modest MLP | `0.6988` | Best modest E2 fusion |
| ViT + Swin concat, deep_reg MLP | `0.7262` | E2b stronger-MLP diagnostic |

Fine-tuned results must be interpreted against both fusion baselines. Comparing only to the modest
E2 matrix would overstate fine-tuning gain because E2b showed that frozen fusion is classifier
capacity-sensitive.

## Expected Evidence

Required evidence for completion:

```text
artifacts/checkpoints/ham10000/finetuned/<backbone>/best.pt
artifacts/features/ham10000/finetuned/<backbone>/train.pt
artifacts/features/ham10000/finetuned/<backbone>/val.pt
artifacts/features/ham10000/finetuned/<backbone>/manifest.json
artifacts/runs/s4_finetune_<backbone_alias>_seed42/
artifacts/runs/*_s2_finetuned_*_none_mlp*_seed42/
artifacts/runs/*_s3_finetuned_*_mlp*_seed42/
artifacts/report_assets/tables/single_backbone_finetuned_results.csv
artifacts/report_assets/tables/finetuned_fusion_results.csv
```

Verification evidence should include:

- Freeze/unfreeze policy checks.
- Checkpoint selected by validation macro-F1.
- Fine-tuned feature cache row counts: train `7008`, validation `1504`.
- Feature cache alignment by `sample_id`, `image_id`, `lesion_id`, label, and split row order.
- No NaN/Inf in features or probabilities.
- Prediction dump row count `1504`.
- Confusion matrix label order fixed.
- Generated artifacts ignored by Git.

## Validation Results

Pending full Colab/GPU run.

Fill after completion:

| Configuration | Feature source | Fusion | Validation macro-F1 | Accuracy | Macro precision | Macro recall | Weighted-F1 |
|---|---|---|---:|---:|---:|---:|---:|
| ViT | finetuned | none | TBD | TBD | TBD | TBD | TBD |
| Swin | finetuned | none | TBD | TBD | TBD | TBD | TBD |
| BEiT | finetuned | none | TBD | TBD | TBD | TBD | TBD |
| ViT + Swin | finetuned | concat | TBD | TBD | TBD | TBD | TBD |
| ViT + Swin + BEiT | finetuned | concat | TBD | TBD | TBD | TBD | TBD |

## Interpretation Template

If fine-tuning improves validation macro-F1:

> Partial fine-tuning improved validation macro-F1 for selected transformer feature sources,
> suggesting that limited domain-specific adaptation may improve representation quality for
> HAM10000 benchmark dermoscopic image classification. The gain should be interpreted together with
> per-class metrics and runtime cost because the dataset is imbalanced and minority-class support is
> limited.

If fine-tuning underperforms frozen baselines:

> Fine-tuned transformer features did not consistently improve over frozen feature baselines under
> this controlled protocol. Possible explanations include overfitting, insufficient data, aggressive
> or insufficient unfreezing, learning-rate sensitivity, minority-class instability, or limited domain
> shift adaptation. This negative result remains reportable because the split, selection rule, and
> downstream MLP controls were kept fixed.

If BEiT remains useful only in fusion:

> BEiT should be discussed as a complementary representation source rather than a strong standalone
> backbone. Its value depends on whether ViT + Swin + BEiT fine-tuned concat improves over ViT + Swin
> fine-tuned concat and whether per-class gains justify the added compute.

## Limitations

- Results are expected to be single-seed unless a later multi-seed extension is added.
- Fine-tuning is partial, not full-model adaptation.
- Colab GPU runtime and memory may force batch-size or epoch adjustments.
- Validation macro-F1 is the selection signal; test audit is intentionally reserved for the final
  audit stage.
- Low-support classes such as `df` and `vasc` must be interpreted with support counts visible.
