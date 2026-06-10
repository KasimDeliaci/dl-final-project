# Source Directory

Bu klasör reusable Python package kodu için ayrılmıştır.

Package adı:

```text
dl_final
```

Sprint 1 kapsamı:

- HAM10000 metadata audit helpers,
- image path resolution,
- lesion-aware split generation,
- split smoke checks,
- lightweight report asset generation.

Sonraki sorumluluklar:

- transformer feature extraction adapters,
- MLP classifier utilities,
- evaluation metrics,
- artifact writing helpers.

Core model/training logic notebook içine gömülmemeli; burada veya `scripts/` entrypoint'leri arkasında tutulmalıdır.
