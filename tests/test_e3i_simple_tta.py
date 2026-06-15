"""E3i simple TTA runner tests."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/evaluate_simple_tta_rot4.py"
SPEC = importlib.util.spec_from_file_location("evaluate_simple_tta_rot4", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
e3i = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = e3i
SPEC.loader.exec_module(e3i)


def test_e3i_rejects_weighted_pca_without_saved_pca_components(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = _run_config(fusion_method="weighted_pca_384")
    (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="weighted_pca_384 is excluded"):
        e3i.load_run_specs([run_dir])


def test_e3i_builds_supported_simple_fusion_models() -> None:
    class_names = ["akiec", "bcc", "bkl"]

    concat = e3i.build_restored_model(_run_config(fusion_method="concat"), class_names)
    weighted = e3i.build_restored_model(
        _run_config(fusion_method="weighted_learned_512"),
        class_names,
    )

    assert concat is not None
    assert weighted is not None


def test_e3i_model_input_uses_saved_train_scaler_stats(tmp_path: Path) -> None:
    config = _run_config(fusion_method="concat")
    stats_path = tmp_path / "scaler_stats.npz"
    np.savez(
        stats_path,
        vit_b16_mean=np.array([1.0, 2.0], dtype="float32"),
        vit_b16_scale=np.array([2.0, 4.0], dtype="float32"),
        swin_tiny_mean=np.array([3.0, 4.0], dtype="float32"),
        swin_tiny_scale=np.array([1.0, 2.0], dtype="float32"),
    )
    frame = _frame()
    feature_bank = {
        ("finetuned", "vit_b16", "identity"): e3i.FeatureBlock(
            features=np.array([[3.0, 10.0]], dtype="float32"),
            labels=np.array([0], dtype="int64"),
            frame=frame,
        ),
        ("finetuned", "swin_tiny", "identity"): e3i.FeatureBlock(
            features=np.array([[5.0, 10.0]], dtype="float32"),
            labels=np.array([0], dtype="int64"),
            frame=frame,
        ),
    }

    features = e3i.build_model_input(
        config,
        stats_path,
        feature_bank=feature_bank,
        view="identity",
    )

    assert features.shape == (1, 4)
    assert np.allclose(features, np.array([[1.0, 2.0, 2.0, 3.0]], dtype="float32"))


def test_e3i_run_alias_is_filesystem_safe() -> None:
    config = _run_config(fusion_method="weighted_learned_512")
    config["backbone"] = "vit_b16+swin_tiny+beit_base"

    assert e3i.run_alias(config) == "vit_b16-swin_tiny-beit_base_weightedlearned512_seed42"


def _run_config(*, fusion_method: str) -> dict:
    return {
        "run_id": "example",
        "feature_source": "finetuned",
        "fusion_method": fusion_method,
        "backbone": "vit_b16+swin_tiny",
        "backbones": ["vit_b16", "swin_tiny"],
        "feature_dim": 4 if fusion_method == "concat" else 512,
        "fusion_input_dim": 4,
        "input_dims": {"vit_b16": 2, "swin_tiny": 2},
        "projection_dim": 512,
        "hidden_dims": [8],
        "dropout": 0.1,
        "seed": 42,
    }


def _frame():
    import pandas as pd

    return pd.DataFrame(
        {
            "sample_id": ["a"],
            "image_id": ["a"],
            "lesion_id": ["l"],
            "split": ["val"],
            "true_label": ["akiec"],
        }
    )
