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
- `06_e3h_tta_rot4.ipynb`: E3h validation-only rot4 TTA diagnostic. This runner restores existing
  E3d/E3f/E3g artifacts, runs inference-only probability averaging on Colab GPU, and syncs outputs
  under `MyDrive/dl-final-artifact/e3h_tta_rot4/`.
- `07_e3i_simple_tta_rot4.ipynb`: E3i validation-only rot4 TTA diagnostic for simpler image-only
  fusion models. This runner restores fine-tuned feature/checkpoint/run artifacts, runs
  inference-only TTA on Colab GPU, and syncs outputs under
  `MyDrive/dl-final-artifact/e3i_simple_tta_rot4/`.
