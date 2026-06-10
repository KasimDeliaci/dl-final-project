"""Sprint 2 frozen transformer feature and MLP tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl_final.data.datasets import HAM10000ImageDataset
from dl_final.data.transforms import IMAGENET_MEAN, IMAGENET_STD, build_feature_transform
from dl_final.evaluation.metrics import compute_classification_metrics
from dl_final.features.cache import (
    class_weights_from_cache,
    load_feature_cache,
    save_feature_cache,
    verify_cache_matches_split,
)
from dl_final.models.backbones import build_frozen_feature_extractor, expected_feature_dim
from dl_final.models.feature_extractors import count_trainable_parameters
from dl_final.models.mlp import FeatureMLP
from dl_final.training.loops import train_mlp_model
from dl_final.training.optim import build_optimizer

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]


def test_feature_transform_uses_imagenet_normalization() -> None:
    transform = build_feature_transform(224)

    assert tuple(transform.transforms[-1].mean) == IMAGENET_MEAN
    assert tuple(transform.transforms[-1].std) == IMAGENET_STD


def test_ham10000_image_dataset_reads_split_rows(tmp_path: Path) -> None:
    image_path = tmp_path / "ISIC_1.jpg"
    Image.new("RGB", (32, 24), color=(128, 64, 32)).save(image_path)
    split_csv = tmp_path / "train.csv"
    split_csv.write_text(
        "sample_id,image_id,lesion_id,label,split,image_path\n"
        f"ISIC_1,ISIC_1,HAM_1,nv,train,{image_path}\n",
        encoding="utf-8",
    )

    dataset = HAM10000ImageDataset(split_csv, CLASS_NAMES, transform=build_feature_transform(64))
    item = dataset[0]

    assert item["image"].shape == (3, 64, 64)
    assert int(item["label"]) == CLASS_NAMES.index("nv")
    assert item["lesion_id"] == "HAM_1"


def test_feature_cache_roundtrip_alignment_and_class_weights(tmp_path: Path) -> None:
    split_csv = tmp_path / "train.csv"
    split_csv.write_text(
        "sample_id,image_id,lesion_id,label,split,image_path\n"
        "s1,img1,l1,nv,train,/tmp/img1.jpg\n"
        "s2,img2,l2,mel,train,/tmp/img2.jpg\n",
        encoding="utf-8",
    )
    cache = save_feature_cache(
        tmp_path / "train.pt",
        features=torch.ones(2, 4),
        labels=torch.tensor([0, 1]),
        sample_ids=["s1", "s2"],
        image_ids=["img1", "img2"],
        lesion_ids=["l1", "l2"],
        label_names=["nv", "mel"],
        split_names=["train", "train"],
        split="train",
        backbone="vit_b16",
        class_names=["nv", "mel"],
        feature_source="frozen",
        seed=42,
    )

    loaded = load_feature_cache(cache.path)

    assert loaded.feature_dim == 4
    verify_cache_matches_split(loaded, split_csv)
    assert class_weights_from_cache(loaded, 2).tolist() == [1.0, 1.0]


def test_feature_cache_rejects_nan(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="NaN or Inf"):
        save_feature_cache(
            tmp_path / "train.pt",
            features=torch.tensor([[float("nan")]]),
            labels=torch.tensor([0]),
            sample_ids=["s1"],
            image_ids=["img1"],
            lesion_ids=["l1"],
            label_names=["nv"],
            split_names=["train"],
            split="train",
            backbone="vit_b16",
            class_names=["nv"],
            feature_source="frozen",
            seed=42,
        )


def test_frozen_transformer_backbone_output_dimensions_without_pretrained_weights() -> None:
    for backbone in ("vit_b16", "swin_tiny", "deit3_small", "beit_base"):
        model = build_frozen_feature_extractor(backbone, pretrained=False)
        output = model(torch.randn(1, 3, 224, 224))

        assert output.shape == (1, expected_feature_dim(backbone))
        assert count_trainable_parameters(model) == 0


def test_metrics_and_mlp_training_smoke() -> None:
    torch.manual_seed(7)
    features = torch.randn(24, 8)
    labels = torch.tensor([0, 1, 2, 3, 4, 5, 6, 0] * 3)
    train_loader = DataLoader(TensorDataset(features, labels), batch_size=8, shuffle=True)
    val_loader = DataLoader(TensorDataset(features, labels), batch_size=8, shuffle=False)
    model = FeatureMLP(input_dim=8, num_classes=7, hidden_dims=[16], dropout=0.0)
    optimizer = build_optimizer(
        model.parameters(),
        optimizer_name="adamw",
        learning_rate=1e-2,
        weight_decay=0.0,
    )
    criterion = nn.CrossEntropyLoss()

    model, history, best_metrics = train_mlp_model(
        model,
        train_loader,
        val_loader,
        class_names=CLASS_NAMES,
        device=torch.device("cpu"),
        epochs=2,
        criterion=criterion,
        optimizer=optimizer,
        early_stopping_patience=2,
    )
    metrics = compute_classification_metrics(labels.numpy(), labels.numpy(), CLASS_NAMES)

    assert len(history) >= 1
    assert "best_epoch" in best_metrics
    assert metrics["macro_f1"] == 1.0
