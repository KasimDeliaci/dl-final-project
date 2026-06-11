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

Full Colab/GPU run completed on 2026-06-11. All values below are validation-only; the test split
was not used.

| Configuration | Feature source | Fusion | Validation macro-F1 | Accuracy | Macro precision | Macro recall | Weighted-F1 |
|---|---|---|---:|---:|---:|---:|---:|
| ViT | finetuned | none | `0.6876` | `0.7979` | `0.6473` | `0.7409` | `0.8063` |
| Swin | finetuned | none | `0.6517` | `0.7852` | `0.6244` | `0.6871` | `0.7930` |
| BEiT | finetuned | none | `0.5181` | `0.6915` | `0.4746` | `0.5938` | `0.7145` |
| ViT + Swin | finetuned | concat | `0.7161` | `0.8211` | `0.6734` | `0.7739` | `0.8273` |
| ViT + Swin + BEiT | finetuned | concat | `0.7298` | `0.8271` | `0.7004` | `0.7673` | `0.8325` |

Other representative fusion conditions:

| Configuration | Fusion | Validation macro-F1 |
|---|---|---:|
| ViT + Swin | weighted_learned_512 | `0.6908` |
| ViT + Swin | weighted_pca_384 | `0.6728` |
| ViT + Swin + BEiT | weighted_learned_512 | `0.6940` |
| ViT + Swin + BEiT | weighted_pca_384 | `0.6713` |

Temporary fine-tuning heads selected during image-level partial fine-tuning were weaker than the
downstream cached-feature MLP protocol: ViT `0.6299`, Swin `0.5907`, and BEiT `0.4136` validation
macro-F1. These heads are checkpoint-selection instruments, not the final downstream classifier.

## Baseline Comparison

Single-backbone fine-tuned features did not consistently improve over frozen single-backbone
baselines. ViT decreased slightly from frozen `0.6924` to fine-tuned `0.6876`. Swin improved from
`0.6115` to `0.6517`, and BEiT improved from `0.4759` to `0.5181`, but both remained below the
original frozen ViT baseline.

Concat fusion showed the strongest fine-tuned feature transfer result. `ViT + Swin + BEiT concat`
reached validation macro-F1 `0.7298`, above the modest frozen triple concat baseline `0.6988` and
slightly above the E2b stronger-MLP frozen ViT+Swin diagnostic baseline `0.7262`. The margin over
E2b is small (`+0.0036` macro-F1), so the result should be described as limited evidence that
domain-specific adaptation may improve representation quality, not as a decisive win.

BEiT remains most useful as a complementary fusion source. Fine-tuned BEiT single-backbone MLP
reached only `0.5181`, but adding BEiT to ViT+Swin concat increased validation macro-F1 from
`0.7161` to `0.7298`. This supports discussing BEiT as a complementary representation rather than a
strong standalone backbone under this protocol.

Weighted learned and weighted PCA fusion variants underperformed concat in this run. Learned fusion
weights should not be interpreted as a direct backbone quality ranking because projection, weighting,
and downstream classifier training are optimized jointly.

## Per-Class Behavior

Best validation-only configuration: `ViT + Swin + BEiT concat`.

| Label | Support | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| akiec | `49` | `0.7447` | `0.7143` | `0.7292` |
| bcc | `77` | `0.6552` | `0.7403` | `0.6951` |
| bkl | `165` | `0.6702` | `0.7758` | `0.7191` |
| df | `18` | `0.7500` | `0.8333` | `0.7895` |
| nv | `1006` | `0.9458` | `0.8847` | `0.9142` |
| mel | `167` | `0.5372` | `0.6048` | `0.5690` |
| vasc | `22` | `0.6000` | `0.8182` | `0.6923` |

The largest apparent fusion benefit is visible on minority classes such as `df`, but the support is
only `18`, so this should be treated as high-variance evidence. `mel` remains the most important
class-level limitation among the higher-support non-`nv` classes.

## Runtime And Artifacts

Fine-tuning runtime on the Colab GPU was approximately 7.8 minutes per backbone: ViT `473.6s`, Swin
`465.5s`, and BEiT `471.3s`. Downstream cached-feature MLP and fusion runs completed in seconds to
under 20 seconds per condition. This makes the partial fine-tuning cost moderate relative to the
small validation macro-F1 gain over the E2b frozen diagnostic.

Local artifact bundle copied from `/Users/arcustin2/Downloads/artifacts/` to repo-local
`artifacts/` on 2026-06-11. Key files:

```text
artifacts/checkpoints/ham10000/finetuned/{vit_b16,swin_tiny,beit_base}/best.pt
artifacts/features/ham10000/finetuned/{vit_b16,swin_tiny,beit_base}/{train.pt,val.pt,manifest.json}
artifacts/runs/s4_finetune_backbones_summary.json
artifacts/runs/s4_finetune_{vit,swin,beit}_seed42/
artifacts/runs/20260611_111312_s2_finetuned_vit_none_mlp_seed42/
artifacts/runs/20260611_111318_s2_finetuned_swin_none_mlp_seed42/
artifacts/runs/20260611_111326_s2_finetuned_beit_none_mlp_seed42/
artifacts/runs/20260611_111338_s3_finetuned_vit-swin_concat_mlp_s4_pair_seed42/
artifacts/runs/20260611_111408_s3_finetuned_vit-swin-beit_concat_mlp_s4_triple_seed42/
artifacts/report_assets/tables/single_backbone_finetuned_results.csv
artifacts/report_assets/tables/single_backbone_finetuned_per_class_metrics.csv
artifacts/report_assets/tables/finetuned_fusion_results.csv
artifacts/report_assets/tables/finetuned_fusion_per_class_metrics.csv
artifacts/report_assets/tables/finetuned_fusion_vs_single_validation.csv
artifacts/report_assets/figures/finetuned_single_backbone_macro_f1.png
artifacts/report_assets/figures/finetuned_fusion_macro_f1.png
```

## Verification Evidence

Local post-download verification confirmed:

- Fine-tuned feature cache shapes: train `(7008, 768)` and validation `(1504, 768)` for ViT, Swin,
  and BEiT.
- Feature cache manifests align with canonical split row order by `image_id`, `lesion_id`, label,
  and split.
- Feature tensors and prediction probabilities contain no NaN/Inf.
- All 12 fine-tuned prediction dumps contain `1504` validation rows.
- Run configs record `test_policy=not_used_in_sprint4`.
- Report tables and figures are present under `artifacts/report_assets/`.
- Generated checkpoints, feature caches, run artifacts, predictions, and report assets are ignored
  by Git.

## Limitations

- Results are expected to be single-seed unless a later multi-seed extension is added.
- Fine-tuning is partial, not full-model adaptation.
- Colab GPU runtime and memory may force batch-size or epoch adjustments.
- Validation macro-F1 is the selection signal; test audit is intentionally reserved for the final
  audit stage.
- Low-support classes such as `df` and `vasc` must be interpreted with support counts visible.
