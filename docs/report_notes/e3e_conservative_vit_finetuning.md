# E3e Conservative ViT Fine-Tuning Diagnostic

Bu not, final rapordaki fine-tuning sensitivity ve negative ablation tartismasi icin
hazirlanmistir. Sonuclar validation-only degerlendirmedir; test split yuklenmemis ve model secimi
icin kullanilmamistir.

## Question

Canonical partial fine-tuning kosulunda ViT single-backbone downstream MLP sonucu frozen ViT
baseline'a gore hafif dusmustu (`0.6924 -> 0.6876` validation macro-F1). E3e, bu dususun overly
aggressive adaptation kaynakli olup olmadigini test eder:

- Ayni son iki ViT transformer blogunu acip backbone LR'yi `1e-5` yerine `5e-6` yapmak temsili
  koruyor mu?
- Daha da konservatif olarak yalniz son ViT transformer blogunu acmak temsili koruyor mu?
- Bu yeni ViT feature'lari canonical fine-tuned Swin/BEiT feature'lariyla concat fusion yapildiginda
  ana fine-tuned fusion hikayesini guclendiriyor mu?

## Protocol

Dataset ve split sabit tutulmustur:

- HAM10000 benchmark dermoscopic image classification.
- Lesion-aware train/validation split.
- Train cache rows: `7008`.
- Validation cache rows: `1504`.
- Class order: `akiec`, `bcc`, `bkl`, `df`, `nv`, `mel`, `vasc`.

Iki ViT-only Colab kosulu calistirilmistir:

| Condition | Unfreeze policy | Backbone LR | Head LR | Checkpoint selection |
|---|---|---:|---:|---|
| `e3e_vit_last2_lr5e6` | last 2 ViT blocks + norm/head | `5e-6` | `1e-4` | validation macro-F1 |
| `e3e_vit_last1_lr5e6` | last 1 ViT block + norm/head | `5e-6` | `1e-4` | validation macro-F1 |

Her checkpoint'ten sonra train/validation feature cache uretildi. Downstream MLP kosullari
train-only StandardScaler ve train-only class weights ile calistirildi. Test split E3e icinde
kullanilmadi.

## Results

### Fine-tuned ViT single features

| Condition | Validation macro-F1 | Accuracy | Macro precision | Macro recall | Weighted-F1 | Best downstream epoch |
|---|---:|---:|---:|---:|---:|---:|
| Frozen ViT baseline | `0.6924` | - | - | - | - | - |
| Canonical fine-tuned ViT | `0.6876` | `0.7979` | `0.6473` | `0.7409` | `0.8063` | - |
| E3e ViT last-2, LR `5e-6` | `0.6694` | `0.7813` | `0.6137` | `0.7515` | `0.7944` | `11` |
| E3e ViT last-1, LR `5e-6` | `0.6685` | `0.7773` | `0.6247` | `0.7366` | `0.7911` | `9` |

The conservative policies did not recover the frozen ViT baseline. Both E3e ViT single-feature
conditions were below the canonical fine-tuned ViT single result and below the frozen ViT single
baseline.

### Mixed ViT+Swin+BEiT concat fusion

The mixed fusion conditions used the E3e ViT feature cache plus canonical Sprint 4 fine-tuned Swin
and BEiT feature caches. Only concat fusion was run, because the goal was to test whether the
conservative ViT feature improves the existing triple-fusion story.

| Condition | Validation macro-F1 | Accuracy | Macro precision | Macro recall | Weighted-F1 | Best downstream epoch |
|---|---:|---:|---:|---:|---:|---:|
| E3e last-2 ViT + canonical Swin/BEiT concat | `0.7082` | `0.8211` | `0.6696` | `0.7639` | `0.8273` | `13` |
| E3e last-1 ViT + canonical Swin/BEiT concat | `0.7259` | `0.8245` | `0.6924` | `0.7675` | `0.8302` | `13` |

Comparison anchors:

| Anchor | Validation macro-F1 |
|---|---:|
| E3b fine-tuned ViT+Swin+BEiT concat, multi-seed mean | `0.7246 ± 0.0143` |
| Canonical fine-tuned ViT+Swin+BEiT concat, seed 42 | `0.7298` |
| E3c raw metadata concat, multi-seed mean | `0.7278 ± 0.0058` |
| E3d bounded FiLM metadata conditioning, multi-seed mean | `0.7358 ± 0.0152` |

The last-1 conservative ViT mixed triple result (`0.7259`) is close to the E3b image-only triple
mean, but it does not exceed the canonical seed-42 triple concat result, the E3c metadata control,
or the E3d metadata-conditioned fusion operators. The last-2 lower-LR policy (`0.7082`) is clearly
weaker for triple fusion.

## Per-Class Behavior

For mixed triple concat, the last-1 policy improved macro-F1 over the last-2 lower-LR policy mainly
through `df`, `mel`, `vasc`, and slightly `nv`:

| Label | Last-2 LR `5e-6` F1 | Last-1 LR `5e-6` F1 | Delta |
|---|---:|---:|---:|
| `akiec` | `0.6415` | `0.6071` | `-0.0344` |
| `bcc` | `0.7075` | `0.6500` | `-0.0575` |
| `bkl` | `0.7252` | `0.7322` | `+0.0070` |
| `df` | `0.6522` | `0.8108` | `+0.1586` |
| `nv` | `0.9088` | `0.9117` | `+0.0029` |
| `mel` | `0.5769` | `0.5937` | `+0.0167` |
| `vasc` | `0.7451` | `0.7755` | `+0.0304` |

This class pattern is not uniformly better. The last-1 policy trades lower `akiec` and `bcc`
performance for stronger `df`, `mel`, and `vasc` behavior. Because `df` and `vasc` have low support,
these deltas should be reported with support counts visible and not overinterpreted.

## Interpretation

E3e is a negative but useful fine-tuning sensitivity diagnostic. Lowering the ViT backbone learning
rate and reducing trainable depth did not restore the frozen ViT single-backbone result. This makes
it less likely that the original ViT drop was caused only by opening too many blocks or using a
slightly high backbone LR.

The result is consistent with several possible explanations:

- the lower LR policies may under-adapt the ViT representation;
- validation macro-F1 checkpoint selection through the temporary image-level head may not select
  the best downstream cached-feature epoch;
- ViT feature transfer may already be strong in the frozen state, leaving little room for partial
  adaptation under this fixed protocol;
- minority-class instability can move macro-F1 even when accuracy and weighted-F1 remain competitive.

The strongest E3e mixed fusion condition (`last_1_block`, LR `5e-6`) remains usable as supporting
evidence that conservative ViT adaptation is not catastrophic in fusion, but it does not improve the
main validation-best story. The report should keep the primary fine-tuned/fusion interpretation
anchored to E3b/E3c/E3d rather than replacing it with E3e.

## Artifact Notes

Local artifact roots copied from Colab:

```text
artifacts/checkpoints/ham10000/e3e_vit_last2_lr5e6/
artifacts/checkpoints/ham10000/e3e_vit_last1_lr5e6/
artifacts/features/ham10000/finetuned_vit_last2_lr5e6/
artifacts/features/ham10000/finetuned_vit_last1_lr5e6/
artifacts/features/ham10000/finetuned_vit_last2_lr5e6_plus_s4_swin_beit/
artifacts/features/ham10000/finetuned_vit_last1_lr5e6_plus_s4_swin_beit/
artifacts/report_assets/tables/single_backbone_finetuned_vit_last*_lr5e6_results.csv
artifacts/report_assets/tables/finetuned_vit_last*_lr5e6_plus_s4_swin_beit_fusion_results.csv
artifacts/report_assets/figures/finetuned_vit_last*_lr5e6*_macro_f1.png
```

The first Drive download contained report tables, figures, checkpoints, and feature caches, but the
`artifacts/runs/*e3e*` directories were empty because the notebook's sync filter copied matched
directory names without their contents. The Colab notebook sync cell has been corrected to copy each
matched E3e run directory recursively. If complete run bundles are required, rerun only the final
sync cell while the Colab runtime still has the run directories, or rerun the downstream validation
cells and then the corrected sync cell.

## Report-Ready Turkish Wording

ViT icin daha konservatif partial fine-tuning kosullari da ayrica test edilmistir. Bu kontrolde
backbone learning rate `5e-6` seviyesine dusurulmus ve iki ayri unfreeze policy denenmistir: son iki
transformer blogunu acik birakan kosul ve yalniz son transformer blogunu acik birakan kosul. Bu
deneyde test split kullanilmamis, checkpoint ve downstream model secimi yalniz validation macro-F1
ile yapilmistir.

Sonuclar, ViT'teki hafif fine-tuning dususunun yalnizca fazla agresif unfreeze veya yuksek backbone
learning rate ile aciklanamayabilecegini gostermistir. Konservatif son-iki-blok kosulu `0.6694`,
son-bir-blok kosulu ise `0.6685` validation macro-F1 uretmis ve her ikisi de frozen ViT baseline'inin
(`0.6924`) ve canonical fine-tuned ViT sonucunun (`0.6876`) altinda kalmistir. Bu durum, ViT'in
frozen representation olarak zaten guclu oldugunu ve bu protokolde sinirli fine-tuning'in downstream
feature kalitesini garanti olarak artirmadigini dusundurmektedir.

Konservatif ViT feature'lari Swin ve BEiT fine-tuned feature'lariyla concat fusion icinde
degerlendirildiginde, son-bir-blok kosulu `0.7259` validation macro-F1 ile ana fine-tuned triple
concat ortalamasina yakin bir sonuc uretmistir; ancak canonical seed-42 triple concat (`0.7298`),
metadata-augmented concat (`0.7278 ± 0.0058`) ve metadata-conditioned fusion operator'larinin gerisine
dustugu icin ana sonuc olarak secilmemistir. Bu nedenle konservatif ViT fine-tuning sonucu, nihai
performans artisi olarak degil, ViT fine-tuning sensitivity ve sinirli/karisik adaptation etkisini
gosteren bir ablation olarak raporlanmalidir.
