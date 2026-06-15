"""Smoke tests for E3c metadata-augmented training helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def _load_e3c_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "train_metadata_augmented_mlp.py"
    )
    spec = importlib.util.spec_from_file_location("train_metadata_augmented_mlp", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_condition_mapping_is_explicit() -> None:
    module = _load_e3c_module()

    assert module._backbones_for_condition("metadata_only") == []
    assert module._backbones_for_condition("ft_vit_swin_concat_plus_metadata") == [
        "vit_b16",
        "swin_tiny",
    ]
    assert module._backbones_for_condition("ft_vit_swin_beit_concat_plus_metadata") == [
        "vit_b16",
        "swin_tiny",
        "beit_base",
    ]


def test_scale_and_concat_uses_train_fit_per_backbone() -> None:
    module = _load_e3c_module()
    train_caches = [
        SimpleNamespace(
            features=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            backbone="vit_b16",
            feature_dim=2,
        ),
        SimpleNamespace(
            features=torch.tensor([[10.0], [20.0]]),
            backbone="swin_tiny",
            feature_dim=1,
        ),
    ]
    val_caches = [
        SimpleNamespace(features=torch.tensor([[5.0, 6.0]])),
        SimpleNamespace(features=torch.tensor([[30.0]])),
    ]

    train_features, val_features, metadata, scaler_stats = module.scale_and_concat(
        train_caches,
        val_caches,
    )

    assert train_features.shape == (2, 3)
    assert val_features.shape == (1, 3)
    assert np.isfinite(train_features).all()
    assert np.isfinite(val_features).all()
    assert metadata["scalers"][0]["fit_split"] == "train"
    assert sorted(scaler_stats) == [
        "swin_tiny_mean",
        "swin_tiny_scale",
        "swin_tiny_var",
        "vit_b16_mean",
        "vit_b16_scale",
        "vit_b16_var",
    ]
