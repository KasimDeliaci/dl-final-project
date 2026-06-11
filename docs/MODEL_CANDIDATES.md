# Model Candidates

Bu dosya transformer backbone kararlarını netleştirmek için tutulur. Başlangıçta üçüncü backbone `DeiT III-Small` olarak seçilmişti. Sprint 2 single-backbone screening bu kararı destekledi; ancak E2 frozen fusion matrix ve validation representation similarity diagnostic sonrasında ileri aşamalar için üçüncü backbone `BEiT-Base` olarak güncellenmiştir.

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

## Forward Third Model After E2

### BEiT-Base

Role: Sprint 4 fine-tuning ve ileri fusion analysis için seçilen üçüncü backbone.

Neden seçildi:

- Sprint 2 single-backbone screening'de zayıf kaldı (`0.4759` validation macro-F1), ancak E2 fusion'da en iyi validation sonucu BEiT triple concat ile geldi: `vit_b16+swin_tiny+beit_base concat` validation macro-F1 `0.6988`.
- Bu sonuç ViT single baseline (`0.6924`), planned ViT+Swin concat (`0.6947`) ve ViT+Swin+DeiT concat (`0.6863`) üstündedir.
- Representation similarity diagnostic BEiT'in ViT ve Swin'den daha farklı feature geometry taşıdığını gösterdi:
  - ViT + BEiT similarity `0.4393`
  - Swin + BEiT similarity `0.2874`
  - ViT + Swin similarity `0.5942`
- Bu nedenle BEiT, tek başına zayıf olmasına rağmen feature complementarity açısından daha açıklayıcı üçüncü backbone haline geldi.

Kontrol edilecekler:

- Kullanılacak library'de feature extraction API'si doğrulanmalıdır.
- Pretrained weight erişimi doğrulanmalıdır.
- Input size, feature extraction point ve feature dimension netleştirilmelidir.
- Fine-tuning maliyeti DeiT III-Small'a göre daha yüksek olabilir; Sprint 4 planında compute riski açık tutulmalıdır.
- BEiT pairwise fusion ViT single baseline'ı geçmediği için iddia triple complementarity ile sınırlı tutulmalıdır.

## Screened But Not Forward

### DeiT III-Small

Role: Başlangıçta seçilen ve Sprint 2/Sprint 3 planned matrix içinde değerlendirilen üçüncü backbone; Sprint 4 forward setinden çıkarıldı.

Gerekçe:

- Sprint 2 single-backbone validation macro-F1: `0.5017`, BEiT single result `0.4759` üstündeydi.
- E2 planned matrix içinde ViT+Swin+DeiT concat validation macro-F1 `0.6863` üretti.
- BEiT-expanded E2 matrix içinde ViT+Swin+BEiT concat validation macro-F1 `0.6988` üretti.
- Representation similarity açısından BEiT, ViT/Swin'e daha düşük similarity ve daha yüksek complementarity gösterdi.
- Bu nedenle DeiT III-Small raporda planned/screened baseline olarak kalır, ancak Sprint 4 fine-tuning compute budget'ına taşınmaz.
- Test seti bu değişiklik için kullanılmamıştır.

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
Fine-tuning Colab feasibility: not selected for Sprint 4 forward scope
Expected role in report: planned third-backbone baseline; replaced after E2 fusion complementarity evidence

Model: BEiT-Base
Library/source: timm / beit_base_patch16_224.in22k_ft_in22k_in1k
Pretrained weights: ImageNet-22k pretraining/fine-tuning with ImageNet-1k fine-tuning
Input size: 224
Feature extraction point: average pooled patch tokens, classifier-free num_classes=0, global_pool=avg
Feature dimension: 768
Frozen local feasibility: verified
Fine-tuning Colab feasibility: selected for Sprint 4 forward scope
Expected role in report: third forward backbone selected by E2 fusion complementarity and representation similarity evidence
```
