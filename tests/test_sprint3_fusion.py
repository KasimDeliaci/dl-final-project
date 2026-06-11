"""Sprint 3 fusion shape, PCA, and alignment tests."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

from dl_final.features.cache import save_feature_cache
from dl_final.models.fusion import (
    ConcatenationFusion,
    ProjectedWeightedFusion,
    WeightedLearnedFusionMLP,
    WeightedPCAFusionMLP,
    WeightedSumFusion,
    expected_concat_dim,
)


def _load_runner_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_fusion_matrix.py"
    spec = importlib.util.spec_from_file_location("run_fusion_matrix", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_concat_output_shape_and_expected_dimensions() -> None:
    fusion = ConcatenationFusion()
    output = fusion([torch.randn(4, 768), torch.randn(4, 384)])

    assert output.shape == (4, 1152)
    assert expected_concat_dim(["vit_b16", "swin_tiny"]) == 1536
    assert expected_concat_dim(["vit_b16", "deit3_small"]) == 1152
    assert expected_concat_dim(["swin_tiny", "deit3_small"]) == 1152
    assert expected_concat_dim(["vit_b16", "swin_tiny", "deit3_small"]) == 1920


def test_weighted_learned_projection_shape_and_softmax_sum() -> None:
    fusion = ProjectedWeightedFusion([768, 384], projection_dim=512)
    output = fusion([torch.randn(3, 768), torch.randn(3, 384)])
    weights = fusion.normalized_weights()

    assert output.shape == (3, 512)
    assert weights.shape == (2,)
    assert torch.isclose(weights.sum(), torch.tensor(1.0))


def test_weighted_learned_mlp_accepts_concatenated_features() -> None:
    model = WeightedLearnedFusionMLP([768, 768, 384], num_classes=7, projection_dim=512)
    logits = model(torch.randn(5, 1920))

    assert logits.shape == (5, 7)
    assert torch.isclose(model.normalized_weights().sum(), torch.tensor(1.0))


def test_weighted_pca_sum_shape_and_mlp_output() -> None:
    fusion = WeightedSumFusion(num_backbones=3, feature_dim=384)
    fused = fusion([torch.randn(2, 384), torch.randn(2, 384), torch.randn(2, 384)])
    model = WeightedPCAFusionMLP(num_backbones=3, feature_dim=384, num_classes=7)
    logits = model(torch.randn(2, 1152))

    assert fused.shape == (2, 384)
    assert logits.shape == (2, 7)
    assert torch.isclose(fusion.normalized_weights().sum(), torch.tensor(1.0))
    assert torch.isclose(model.normalized_weights().sum(), torch.tensor(1.0))


def test_fusion_run_matrix_expands_to_twelve_runs() -> None:
    runner = _load_runner_module()

    runs = runner.expand_fusion_run_matrix(
        fusion_methods=["concat", "weighted_learned_512", "weighted_pca_384"],
        backbones=["vit_b16", "swin_tiny", "deit3_small"],
    )

    assert len(runs) == 12
    assert {run["fusion_method"] for run in runs} == {
        "concat",
        "weighted_learned_512",
        "weighted_pca_384",
    }
    assert all(len(run["backbones"]) >= 2 for run in runs)


def test_fusion_run_matrix_can_select_one_exact_combination() -> None:
    runner = _load_runner_module()

    runs = runner.expand_fusion_run_matrix(
        fusion_methods=["concat", "weighted_learned_512", "weighted_pca_384"],
        backbones=["vit_b16", "swin_tiny", "beit_base"],
        only_combination=["vit_b16", "swin_tiny", "beit_base"],
    )

    assert len(runs) == 3
    assert all(run["backbones"] == ["vit_b16", "swin_tiny", "beit_base"] for run in runs)


def test_cache_alignment_detects_mismatched_order(tmp_path: Path) -> None:
    runner = _load_runner_module()
    first = save_feature_cache(
        tmp_path / "a.pt",
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
    second = save_feature_cache(
        tmp_path / "b.pt",
        features=torch.ones(2, 4),
        labels=torch.tensor([1, 0]),
        sample_ids=["s2", "s1"],
        image_ids=["img2", "img1"],
        lesion_ids=["l2", "l1"],
        label_names=["mel", "nv"],
        split_names=["train", "train"],
        split="train",
        backbone="swin_tiny",
        class_names=["nv", "mel"],
        feature_source="frozen",
        seed=42,
    )

    with pytest.raises(ValueError, match="sample_id order"):
        runner.verify_cache_alignment([first, second])


def test_pca_transform_records_train_only_metadata() -> None:
    runner = _load_runner_module()
    train = torch.randn(20, 8)
    val = torch.randn(6, 8)

    train_pca, val_pca, metadata = runner.fit_transform_pca_block(
        train,
        val,
        backbone="toy",
        output_dim=4,
        seed=42,
    )

    assert train_pca.shape == (20, 4)
    assert val_pca.shape == (6, 4)
    assert metadata["fit_split"] == "train"
    assert metadata["uses_labels"] is False
    assert metadata["input_dim"] == 8
    assert metadata["output_dim"] == 4
