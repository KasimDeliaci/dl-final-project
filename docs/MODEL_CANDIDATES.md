# Model Candidates

Bu dosya transformer backbone kararlarını netleştirmek için tutulur. Bu aşamada implementation başlamamıştır; ancak üçüncü backbone kararı `DeiT III-Small` olarak kilitlenmiştir.

## Required Models

### Vanilla ViT

Role: Zorunlu baseline transformer.

Neden gerekli:

- Assignment açıkça Vanilla ViT istemektedir.
- Patch-token ve CLS-token temsili, transformer feature extraction için temel referans sağlar.

Kontrol edilecekler:

- pretrained weight erişimi,
- input image size,
- feature extraction için CLS token mı pooled output mu kullanılacağı,
- feature dimension,
- fine-tuning için açılacak son block sayısı.

### Swin Transformer

Role: Zorunlu hierarchical/window-based transformer.

Neden gerekli:

- Assignment açıkça Swin Transformer istemektedir.
- ViT'e göre hierarchical feature yapısı ve window attention kullanır; bu nedenle feature complementarity tartışması için anlamlıdır.

Kontrol edilecekler:

- pretrained weight erişimi,
- stage/block feature extraction API'si,
- pooled representation kararı,
- fine-tuning için açılacak son stage/block sayısı.

## Selected Third Model

### DeiT III-Small

Role: Ek puan ve üçüncü transformer temsili için seçilen backbone.

Neden seçildi:

- HAM10000 literatüründe DeiT III ailesi için doğrudan performans sinyali BEiT'e göre daha nettir.
- ViT ailesine yakın kalır, ancak data-efficient training fikriyle ayrı bir temsil çizgisi sunar.
- `Small` varyant, üçüncü backbone olarak Colab maliyetini yönetilebilir tutar.

Kontrol edilecekler:

- Kullanılacak library'de feature extraction API'si doğrulanmalıdır.
- Pretrained weight erişimi doğrulanmalıdır.
- Input size, feature extraction point ve feature dimension netleştirilmelidir.
- Vanilla ViT'e yakın davranırsa fusion complementarity sınırlı kalabilir; bu durum raporda ayrıca tartışılmalıdır.

## Screened But Excluded Candidate

### BEiT

Role: Üçüncü backbone slotu için kısa validation-only screening ile değerlendirildi, canonical set dışında bırakıldı.

Gerekçe:

- Masked image modeling pretraining nedeniyle temsil çeşitliliği açısından ilginçtir.
- Kullanılan aday: `beit_base_patch16_224.in22k_ft_in22k_in1k`.
- Feature extraction policy: average pooled patch-token representation, `num_classes=0`, `global_pool=avg`.
- Validation macro-F1: `0.4759`.
- DeiT III-Small aynı MLP recipe ile validation macro-F1 `0.5017` verdi.
- Bu nedenle canonical üçüncü backbone `DeiT III-Small` olarak kalır.
- Test seti bu seçim için kullanılmamıştır.

## Implementation Öncesi Doğrulama Tablosu

Sprint 2 implementation sırasında seçili backbone tablosu doldurulmuştur:

```text
Model: Vanilla ViT
Library/source: timm / vit_base_patch16_224.augreg_in21k_ft_in1k
Pretrained weights: ImageNet-21k pretraining, ImageNet-1k fine-tuning
Input size: 224
Feature extraction point: CLS token, classifier-free num_classes=0, global_pool=token
Feature dimension: 768
Frozen local feasibility: verified
Fine-tuning Colab feasibility: pending Sprint 4
Expected role in report: strongest Sprint 2 frozen single-backbone baseline

Model: Swin Transformer
Library/source: timm / swin_tiny_patch4_window7_224.ms_in1k
Pretrained weights: ImageNet-1k
Input size: 224
Feature extraction point: average pooled final-stage representation, classifier-free num_classes=0, global_pool=avg
Feature dimension: 768
Frozen local feasibility: verified
Fine-tuning Colab feasibility: pending Sprint 4
Expected role in report: hierarchical transformer baseline and fusion complementarity candidate

Model: DeiT III-Small
Library/source: timm / deit3_small_patch16_224.fb_in1k
Pretrained weights: ImageNet-1k
Input size: 224
Feature extraction point: CLS token, classifier-free num_classes=0, global_pool=token
Feature dimension: 384
Frozen local feasibility: verified
Fine-tuning Colab feasibility: pending Sprint 4
Expected role in report: third transformer baseline and fusion complementarity candidate

Model: BEiT-Base
Library/source: timm / beit_base_patch16_224.in22k_ft_in22k_in1k
Pretrained weights: ImageNet-22k pretraining/fine-tuning with ImageNet-1k fine-tuning
Input size: 224
Feature extraction point: average pooled patch tokens, classifier-free num_classes=0, global_pool=avg
Feature dimension: 768
Frozen local feasibility: verified
Fine-tuning Colab feasibility: not selected for canonical Sprint 4 scope
Expected role in report: screened candidate excluded from canonical third-backbone slot
```
