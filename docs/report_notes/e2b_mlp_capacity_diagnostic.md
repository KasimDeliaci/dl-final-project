# MLP Capacity Diagnostic Report Note

Bu not, frozen feature fusion sonuçlarının MLP classifier kapasitesine ne kadar duyarlı olduğunu değerlendirmek için hazırlanmıştır. Ana E2 matrix modest MLP recipe ile çalıştırılmıştır. E2b diagnostic, aynı frozen feature cache'leri üzerinde daha geniş ve daha regülarize MLP classifier'lar denenirse validation macro-F1 değişiyor mu sorusunu yanıtlar.

Bu çalışma test setini kullanmaz. Tüm seçim ve yorum validation macro-F1 üzerinden yapılır.

## Question

Frozen feature fusion sonuçları mevcut MLP classifier kapasitesiyle sınırlı mı?

Daha somut olarak:

> High-dimensional concat fusion feature'ları, daha güçlü bir MLP classifier ile daha iyi ayrıştırılabilir mi?

Bu soru E2 yorumunu etkiler. Eğer daha güçlü MLP yalnız fusion koşullarını iyileştirirse, önceki düşük fusion gain'in bir kısmı classifier capacity bottleneck ile açıklanabilir. Eğer single-backbone ViT de aynı oranda iyileşirse, fusion complementarity iddiası zayıflar.

## Diagnostic Scope

E2b tüm fusion matrix'i yeniden çalıştırmaz. Bunun yerine rapor açısından en önemli temsilci koşulları seçer:

- `vit_b16` single-backbone baseline
- `vit_b16 + swin_tiny` concat
- `vit_b16 + swin_tiny + deit3_small` concat
- `vit_b16 + swin_tiny + beit_base` concat

Bu seçim bilinçlidir. E2 ana sonucu concat fusion tarafında geldiği için weighted fusion tekrar genişletilmedi. Amaç fusion method search yapmak değil, concat features üzerinde MLP capacity sensitivity ölçmektir.

## MLP Variants

Baseline MLP:

```text
hidden_dims = [512, 256]
dropout = 0.3
learning_rate = 0.001
weight_decay = 0.0001
early_stopping_patience = 6
```

E2b variants:

| Variant | Hidden dims | Dropout | LR | Weight decay | Max epochs | Patience |
|---|---:|---:|---:|---:|---:|---:|
| `wide` | 1024, 512, 256 | 0.4 | 0.0007 | 0.0001 | 50 | 8 |
| `wide_reg` | 1024, 512, 256 | 0.5 | 0.0005 | 0.0003 | 50 | 8 |
| `deep_reg` | 2048, 1024, 512 | 0.5 | 0.0003 | 0.0005 | 60 | 10 |

Scaler policy, class weighting, split files, feature cache source and validation-only selection rules were unchanged.

## Results

| Configuration | Fusion | MLP variant | Accuracy | Macro-F1 | Macro precision | Macro recall | Weighted-F1 | Best epoch |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ViT | none | baseline | 0.7872 | 0.6924 | 0.6739 | 0.7193 | 0.7982 | 23 |
| ViT | none | wide | 0.7945 | 0.6813 | 0.6512 | 0.7200 | 0.8035 | 22 |
| ViT | none | wide_reg | 0.7866 | 0.6624 | 0.6179 | 0.7307 | 0.7979 | 31 |
| ViT | none | deep_reg | 0.7593 | 0.6438 | 0.5912 | 0.7416 | 0.7783 | 14 |
| ViT + Swin | concat | baseline | 0.8178 | 0.6947 | 0.6578 | 0.7444 | 0.8244 | 22 |
| ViT + Swin | concat | wide | 0.8152 | 0.6924 | 0.6545 | 0.7467 | 0.8206 | 29 |
| ViT + Swin | concat | wide_reg | 0.8118 | 0.6878 | 0.6584 | 0.7387 | 0.8197 | 43 |
| ViT + Swin | concat | deep_reg | 0.8305 | 0.7262 | 0.7157 | 0.7434 | 0.8337 | 49 |
| ViT + Swin + DeiT | concat | baseline | 0.8225 | 0.6863 | 0.6618 | 0.7401 | 0.8281 | 29 |
| ViT + Swin + DeiT | concat | wide | 0.8059 | 0.6863 | 0.6650 | 0.7269 | 0.8139 | 15 |
| ViT + Swin + DeiT | concat | wide_reg | 0.7706 | 0.6624 | 0.6122 | 0.7558 | 0.7866 | 13 |
| ViT + Swin + DeiT | concat | deep_reg | 0.7979 | 0.6755 | 0.6288 | 0.7443 | 0.8069 | 15 |
| ViT + Swin + BEiT | concat | baseline | 0.8125 | 0.6988 | 0.6674 | 0.7452 | 0.8177 | 23 |
| ViT + Swin + BEiT | concat | wide | 0.8105 | 0.7080 | 0.6775 | 0.7492 | 0.8191 | 20 |
| ViT + Swin + BEiT | concat | wide_reg | 0.8165 | 0.7159 | 0.6831 | 0.7623 | 0.8225 | 50 |
| ViT + Swin + BEiT | concat | deep_reg | 0.8324 | 0.7144 | 0.6891 | 0.7489 | 0.8342 | 44 |

## Interpretation

E2b changes the strength of the frozen fusion conclusion.

The original E2 matrix suggested that `ViT + Swin + BEiT concat` was the best frozen fusion result (`0.6988`). After increasing MLP capacity, the strongest result became:

```text
ViT + Swin concat + deep_reg MLP
validation macro-F1 = 0.7262
```

This is a substantial gain over:

- ViT single baseline: `0.6924`
- original ViT + Swin concat: `0.6947`
- original ViT + Swin + BEiT concat: `0.6988`
- best BEiT triple capacity probe: `0.7159`

This indicates that at least part of the earlier weak fusion gain was caused by classifier capacity or regularization limits. The high-dimensional `ViT + Swin` concat representation appears to benefit from a deeper, more regularized MLP.

## Effect On BEiT vs DeiT Decision

E2b does not make DeiT stronger than BEiT as the third backbone. Under stronger MLP variants:

- Best `ViT + Swin + BEiT concat`: `0.7159`
- Best `ViT + Swin + DeiT concat`: `0.6863`

BEiT remains the better third-backbone candidate among the two. However, E2b also shows that the best frozen concat result is the pairwise `ViT + Swin` condition, not a three-backbone condition.

The resulting interpretation is:

> BEiT remains more complementary than DeiT as a third backbone, but adding a third backbone is not always better than giving a strong pairwise concat representation enough MLP capacity.

## Report Decision

For the report, E2 and E2b should be presented as two layers:

1. E2 modest-MLP matrix:
   - Best result: `ViT + Swin + BEiT concat`, macro-F1 `0.6988`.
   - Main interpretation: BEiT can be weak standalone but complementary in triple fusion.

2. E2b MLP capacity diagnostic:
   - Best result: `ViT + Swin concat + deep_reg MLP`, macro-F1 `0.7262`.
   - Main interpretation: frozen concat fusion is classifier-capacity sensitive; a stronger MLP can extract more signal from high-dimensional ViT+Swin features.

This does not use test metrics and should not be presented as final model selection. It is a validation-stage diagnostic that informs Sprint 4 planning.

## Placement In Final Report

Bu sonuçlar ana "frozen feature fusion" tablosundan sonra kısa bir classifier-capacity sensitivity alt bölümü olarak verilmelidir. Ana metinde tüm 16 satırlık tablo kullanılabilir; yer sıkıntısı varsa tablo appendix'e, ana metne ise şu özet taşınabilir:

| Finding | Evidence |
|---|---|
| ViT single daha büyük MLP ile iyileşmedi. | baseline `0.6924`, best E2b ViT `0.6813` |
| ViT+Swin concat daha güçlü MLP ile belirgin iyileşti. | baseline `0.6947`, deep_reg `0.7262` |
| BEiT triple daha güçlü MLP ile iyileşti ama ViT+Swin deep_reg'i geçmedi. | baseline `0.6988`, best `0.7159` |
| DeiT triple stronger MLP ile anlamlı iyileşmedi. | baseline `0.6863`, best `0.6863` |

Rapor cümlesi:

> MLP capacity diagnostic showed that the initial frozen fusion results were partly classifier-limited. A deeper regularized MLP substantially improved the ViT+Swin concat representation, while larger MLPs did not improve the ViT single-backbone baseline. This suggests that high-dimensional fused features can contain useful signal that the modest baseline classifier does not fully exploit.

## Sprint 4 Implication

Sprint 4 should still keep `vit_b16`, `swin_tiny`, and `beit_base` as the forward backbone set because BEiT remains more useful than DeiT as the third backbone candidate. However, the representative frozen fusion baseline to beat should include:

- `ViT + Swin concat + deep_reg MLP`
- `ViT + Swin + BEiT concat + wide_reg/deep_reg MLP`

This matters because the fine-tuning stage should not compare against an artificially weak frozen fusion classifier.

## Evidence

Generated table:

```text
artifacts/report_assets/tables/e2b_mlp_capacity_diagnostic.csv
```

Generated run artifacts:

```text
artifacts/runs/*_e2b_wide_seed42/
artifacts/runs/*_e2b_wide_reg_seed42/
artifacts/runs/*_e2b_deep_reg_seed42/
```

Verification:

- All E2b runs used train/validation caches only.
- Test metrics were not computed.
- Prediction dumps contain `1504` validation rows.
- Generated artifacts are Git-ignored.

## Limitations

- Single seed only.
- The MLP variants are a focused diagnostic, not an exhaustive hyperparameter search.
- Larger MLPs increase overfitting risk; validation improvements should be audited carefully in final selection.
- E2b changes validation-stage interpretation but does not replace the final test audit requirement.
