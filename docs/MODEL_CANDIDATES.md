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

## Excluded Candidate

### BEiT

Role: Değerlendirildi, bu proje kapsamı dışında bırakıldı.

Gerekçe:

- Masked image modeling pretraining nedeniyle temsil çeşitliliği açısından ilginçtir.
- Ancak HAM10000 özelinde BEiT'in DeiT III'ten daha güçlü olduğuna dair doğrudan literatür sinyali zayıftır.
- Bu aşamada proje anlatısını ViT, Swin ve DeiT III-Small üçlüsüyle daha kontrollü tutmak tercih edilmiştir.

## Implementation Öncesi Doğrulama Tablosu

İlk implementation başlamadan önce seçili her backbone için şu tablo doldurulmalıdır:

```text
Model:
Library/source:
Pretrained weights:
Input size:
Feature extraction point:
Feature dimension:
Frozen local feasibility:
Fine-tuning Colab feasibility:
Expected role in report:
```
