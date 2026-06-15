"""Smoke tests for E3d metadata fusion runner helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import torch

from dl_final.models.metadata_fusion import (
    MetadataFiLMMLP,
    MetadataGatedBackboneMLP,
    MetadataTwoBranchMLP,
)


def _load_e3d_module():
    module_path = (
        Path(__file__).resolve().parents[1] / "scripts" / "train_metadata_fusion_operator.py"
    )
    spec = importlib.util.spec_from_file_location("train_metadata_fusion_operator", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_operator_model_maps_all_e3d_conditions() -> None:
    module = _load_e3d_module()
    kwargs = {
        "input_dims": [4, 3, 2],
        "image_dim": 9,
        "metadata_dim": 5,
        "num_classes": 7,
        "hidden_dims": [8, 4],
        "dropout": 0.1,
    }

    assert isinstance(
        module.build_operator_model("triple_metadata_gated_backbone", **kwargs),
        MetadataGatedBackboneMLP,
    )
    assert isinstance(
        module.build_operator_model("triple_metadata_film", **kwargs),
        MetadataFiLMMLP,
    )
    assert isinstance(
        module.build_operator_model("triple_metadata_two_branch", **kwargs),
        MetadataTwoBranchMLP,
    )


def test_gate_summary_writes_all_and_per_class_rows(tmp_path: Path) -> None:
    module = _load_e3d_module()
    model = MetadataGatedBackboneMLP(
        input_dims=[4, 3, 2],
        metadata_dim=5,
        num_classes=7,
        hidden_dims=[8],
    )
    features = np.random.default_rng(42).normal(size=(10, 14)).astype("float32")
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 0, 1, 2])

    wrote = module.maybe_write_gate_summary(
        model,
        tmp_path,
        features,
        y_true,
        ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"],
        backbones=["vit_b16", "swin_tiny", "beit_base"],
        device=torch.device("cpu"),
    )

    assert wrote is True
    rows = (tmp_path / "gate_summary.csv").read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1 + 3 * 8
