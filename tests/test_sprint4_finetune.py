"""Sprint 4 transformer fine-tuning policy and fine-tuned cache tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd
import torch

from dl_final.features.cache import (
    load_feature_cache,
    save_feature_cache,
    verify_cache_matches_split,
)
from dl_final.models.backbones import (
    apply_finetuning_policy,
    build_classification_backbone,
    build_finetuned_feature_extractor,
    expected_feature_dim,
)


def _load_fusion_runner_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "run_fusion_matrix.py"
    spec = importlib.util.spec_from_file_location("run_fusion_matrix", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_transformer_finetuning_policy_unfreezes_only_target_regions() -> None:
    checks = {
        "vit_b16": {
            "policy": "last_2_blocks",
            "trainable_prefix": "blocks.11",
            "frozen_prefix": "blocks.0",
            "head_prefix": "head",
        },
        "swin_tiny": {
            "policy": "last_stage",
            "trainable_prefix": "layers.3",
            "frozen_prefix": "layers.0",
            "head_prefix": "head",
        },
        "beit_base": {
            "policy": "last_2_blocks",
            "trainable_prefix": "blocks.11",
            "frozen_prefix": "blocks.0",
            "head_prefix": "head",
        },
    }

    for backbone, expected in checks.items():
        model = build_classification_backbone(backbone, num_classes=7, pretrained=False)
        summary = apply_finetuning_policy(model, backbone, policy=expected["policy"])
        params = dict(model.named_parameters())
        trainable = {name for name, parameter in params.items() if parameter.requires_grad}

        assert summary["policy"] == expected["policy"]
        assert 0 < summary["trainable_params"] < summary["total_params"]
        assert any(name.startswith(expected["trainable_prefix"]) for name in trainable)
        assert any(name.startswith(expected["head_prefix"]) for name in trainable)
        assert all(
            not parameter.requires_grad
            for name, parameter in params.items()
            if name.startswith(expected["frozen_prefix"])
        )


def test_vit_last_1_block_policy_unfreezes_only_final_block() -> None:
    model = build_classification_backbone("vit_b16", num_classes=7, pretrained=False)
    summary = apply_finetuning_policy(model, "vit_b16", policy="last_1_block")
    params = dict(model.named_parameters())
    trainable = {name for name, parameter in params.items() if parameter.requires_grad}

    assert summary["policy"] == "last_1_block"
    assert any(name.startswith("blocks.11") for name in trainable)
    assert not any(name.startswith("blocks.10") for name in trainable)
    assert any(name.startswith("head") for name in trainable)
    assert all(
        not parameter.requires_grad
        for name, parameter in params.items()
        if name.startswith("blocks.0")
    )


def test_finetuned_checkpoint_feature_extractor_shape(tmp_path: Path) -> None:
    model = build_classification_backbone("vit_b16", num_classes=7, pretrained=False)
    checkpoint_path = tmp_path / "vit_best.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "checkpoint_metadata": {"selection_metric": "validation_macro_f1"},
        },
        checkpoint_path,
    )

    extractor = build_finetuned_feature_extractor(
        "vit_b16",
        checkpoint_path=str(checkpoint_path),
        num_classes=7,
    )
    output = extractor(torch.randn(2, 3, 224, 224))

    assert output.shape == (2, expected_feature_dim("vit_b16"))


def test_finetuned_feature_cache_shape_and_split_alignment(tmp_path: Path) -> None:
    split_csv = tmp_path / "train.csv"
    split_csv.write_text(
        "sample_id,image_id,lesion_id,label,split,image_path\n"
        "s1,img1,l1,nv,train,/tmp/img1.jpg\n"
        "s2,img2,l2,mel,train,/tmp/img2.jpg\n",
        encoding="utf-8",
    )
    cache = save_feature_cache(
        tmp_path / "train.pt",
        features=torch.ones(2, 768),
        labels=torch.tensor([0, 1]),
        sample_ids=["s1", "s2"],
        image_ids=["img1", "img2"],
        lesion_ids=["l1", "l2"],
        label_names=["nv", "mel"],
        split_names=["train", "train"],
        split="train",
        backbone="beit_base",
        class_names=["nv", "mel"],
        feature_source="finetuned",
        seed=42,
        config={"checkpoint_metadata": {"selection_metric": "validation_macro_f1"}},
    )

    loaded = load_feature_cache(cache.path)

    assert loaded.metadata["feature_source"] == "finetuned"
    assert loaded.features.shape == (2, 768)
    assert loaded.metadata["config"]["checkpoint_metadata"]["selection_metric"] == (
        "validation_macro_f1"
    )
    verify_cache_matches_split(loaded, split_csv)


def test_finetuned_fusion_runner_uses_e3_and_sprint4_test_policy() -> None:
    runner = _load_fusion_runner_module()
    runs = runner.expand_fusion_run_matrix(
        fusion_methods=["concat"],
        backbones=["vit_b16", "swin_tiny", "beit_base"],
        only_combination=["vit_b16", "swin_tiny", "beit_base"],
    )

    assert runs == [
        {
            "backbones": ["vit_b16", "swin_tiny", "beit_base"],
            "fusion_method": "concat",
            "fusion_input_dim": 2304,
        }
    ]
    assert runner._experiment_id_for_source("finetuned") == "E3"
    assert runner._test_policy_for_source("finetuned") == "not_used_in_sprint4"
    assert runner._test_policy_for_source("finetuned_vit_last1_lr5e6") == "not_used_in_sprint4"
    assert runner._test_policy_for_source(
        "finetuned_vit_last1_lr5e6",
        "not_loaded_or_used_in_e3e",
    ) == "not_loaded_or_used_in_e3e"


def test_fusion_export_includes_e3e_feature_sources(tmp_path: Path) -> None:
    runner = _load_fusion_runner_module()
    run_dir = tmp_path / "runs" / "e3e_run"
    run_dir.mkdir(parents=True)
    config = {
        "run_id": "e3e_run",
        "experiment_id": "E3e",
        "feature_source": "finetuned_vit_last1_lr5e6_plus_s4_swin_beit",
        "fusion_method": "concat",
        "backbone": "vit_b16+swin_tiny+beit_base",
        "backbones": ["vit_b16", "swin_tiny", "beit_base"],
    }
    (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    pd.DataFrame([{"macro_f1": 0.71, "accuracy": 0.8, "weighted_f1": 0.81}]).to_csv(
        run_dir / "metrics_summary.csv",
        index=False,
    )
    pd.DataFrame([{"label": "nv", "f1": 0.9}]).to_csv(
        run_dir / "per_class_metrics.csv",
        index=False,
    )

    exported = runner.export_fusion_report_assets(
        tmp_path / "runs",
        tmp_path / "tables",
        tmp_path / "figures",
        feature_source="finetuned_vit_last1_lr5e6_plus_s4_swin_beit",
    )

    results = pd.read_csv(exported["fusion_results"])
    assert results["experiment_id"].tolist() == ["E3e"]
    assert results["feature_source"].tolist() == [
        "finetuned_vit_last1_lr5e6_plus_s4_swin_beit"
    ]
    weights = pd.read_csv(exported["fusion_weight_summary"])
    assert weights.empty
    assert weights.columns.tolist() == ["run_id", "backbone", "weight", "weight_sum"]
