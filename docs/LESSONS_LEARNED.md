# Lessons Learned

Bu final projesi, önceki CNN tabanlı dönem projesinden çıkarılan workflow dersleriyle başlatılmıştır.

## Ana Ders

Çok deney yapmak tek başına güçlü bilimsel anlatı üretmez. Güçlü anlatı için her deneyin hipotezi, kontrol koşulu, seçim kuralı ve artifact standardı baştan belirlenmelidir.

## Bu Projede Uygulanacak Kurallar

### 1. Deney registry'si zorunlu

Her deney `docs/EXPERIMENT_REGISTRY.md` içine yazılmadan başlatılmayacak.

### 2. Test seti seçim için kullanılmayacak

Validation ve test ayrımı en baştan korunacak. Test seti yalnız final audit veya açıkça etiketlenmiş diagnostic audit için kullanılacak.

### 3. Prediction dump standart olacak

Summary metrics yeterli sayılmayacak. Her önemli run per-sample prediction ve probability vector üretecek.

### 4. Colab/local artifact akışı net olacak

GPU-heavy fine-tuning Colab'de yapılabilir; ancak Colab çıktısı local projeye standart artifact bundle olarak dönecek.

### 5. Fusion denemelerinden önce temsil kalitesi okunacak

Yeni fusion denemesi yapmadan önce ilgili backbone feature'larının single-model gücü ve per-class davranışı kontrol edilecek.

### 6. Rapor notu deneyle birlikte yazılacak

Deney bittikten hemen sonra `report_note.md` veya `docs/report_notes/` altında kısa yorum yazılacak:

```text
Bunu neden denedik?
Neyi sabit tuttuk?
Neden final modele girdi veya girmedi?
```

### 7. Ana hikaye erken daraltılacak

Bu proje baştan üç ana hikayeye odaklanacak:

- transformer representation quality,
- feature fusion complementarity,
- frozen vs fine-tuned transfer learning.

Diğer fikirler ancak bu hikayeyi güçlendiriyorsa yapılacak.

