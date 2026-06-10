# Execution Plans

Bu klasör, kod veya rapor üzerinde anlamlı değişiklik gerektiren işleri takip etmek için kullanılır.

## Akış

1. Yeni iş başlamadan önce `docs/exec-plans/active/` altında kısa bir plan dosyası oluştur.
2. Plan içinde hedef, kapsam, verification gate ve beklenen artifact'leri yaz.
3. İş tamamlanınca plan dosyasını `docs/exec-plans/completed/` altına taşı.
4. Deney sonucu ayrıca `docs/report_notes/` veya ilgili artifact klasöründe yorumlanmalıdır.

## Ne Zaman Kullanılır?

Exec-plan gerektiren işler:

- dataset audit,
- feature extraction pipeline,
- MLP training pipeline,
- fusion experiment matrix,
- fine-tuning workflow,
- report veya presentation scaffold,
- final audit ve artifact generation.

Küçük typo veya tek satırlık doküman düzeltmeleri için exec-plan gerekmez.

## Plan Template

```markdown
# Plan Title

## Goal

## Scope

## Non-Goals

## Steps

## Verification Gates

## Expected Artifacts

## Report/Sunum Note
```

