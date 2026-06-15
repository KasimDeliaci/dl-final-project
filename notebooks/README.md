# Notebooks Directory

Notebook'lar ince runner olarak kullanılacaktır.

Kurallar:

- Notebook içinde uzun business logic tutulmayacak.
- Colab notebook'ları repo clone/pull, Drive artifact restore ve script çağırma akışına odaklanacak.
- Notebook output'ları kalıcı kanıt sayılmayacak; önemli sonuçlar artifact dosyalarına yazılacak.

## Available Launchers

- `04_sprint4_finetuned_transformer_features.ipynb`: canonical Sprint 4 fine-tuned transformer
  feature extraction and downstream validation runner.
- `05_e3e_conservative_vit_finetuning.ipynb`: E3e conservative ViT fine-tuning diagnostic. This
  runner writes to separate E3e checkpoint, feature, run, and Drive namespaces and does not replace
  canonical Sprint 4 artifacts.
