"""Representation complementarity diagnostics tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import torch

from dl_final.evaluation.complementarity import (
    build_fusion_complementarity_summary,
    compute_representation_complementarity,
)
from dl_final.features.cache import save_feature_cache


def test_representation_complementarity_handles_different_feature_dims(tmp_path: Path) -> None:
    splits_dir = tmp_path / "splits"
    splits_dir.mkdir()
    split_text = (
        "sample_id,image_id,lesion_id,label,split,image_path\n"
        "s1,img1,l1,nv,{split},/tmp/img1.jpg\n"
        "s2,img2,l2,mel,{split},/tmp/img2.jpg\n"
        "s3,img3,l3,nv,{split},/tmp/img3.jpg\n"
        "s4,img4,l4,mel,{split},/tmp/img4.jpg\n"
    )
    (splits_dir / "train.csv").write_text(split_text.format(split="train"), encoding="utf-8")
    (splits_dir / "val.csv").write_text(split_text.format(split="val"), encoding="utf-8")
    root = tmp_path / "features" / "ham10000" / "frozen"
    labels = torch.tensor([0, 1, 0, 1])
    for backbone, dim in (("vit_b16", 5), ("beit_base", 3)):
        for split in ("train", "val"):
            save_feature_cache(
                root / backbone / f"{split}.pt",
                features=torch.randn(4, dim),
                labels=labels,
                sample_ids=["s1", "s2", "s3", "s4"],
                image_ids=["img1", "img2", "img3", "img4"],
                lesion_ids=["l1", "l2", "l3", "l4"],
                label_names=["nv", "mel", "nv", "mel"],
                split_names=[split] * 4,
                split=split,
                backbone=backbone,
                class_names=["nv", "mel"],
                feature_source="frozen",
                seed=42,
            )

    summary = compute_representation_complementarity(
        feature_root=tmp_path / "features",
        dataset_name="ham10000",
        feature_source="frozen",
        backbones=["vit_b16", "beit_base"],
        splits_dir=splits_dir,
        split="val",
        max_samples=4,
        seed=42,
    )

    assert len(summary) == 1
    assert summary["representation_similarity"].between(-1, 1).all()
    assert summary["method"].iloc[0] == "train_scaled_sample_cosine_rsa_pearson"


def test_representation_complementarity_refuses_test_split(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Test split is reserved"):
        compute_representation_complementarity(
            feature_root=tmp_path,
            dataset_name="ham10000",
            feature_source="frozen",
            backbones=["vit_b16", "swin_tiny"],
            splits_dir=tmp_path,
            split="test",
        )


def test_fusion_complementarity_summary_uses_pairwise_average() -> None:
    pairwise = pd.DataFrame(
        [
            {
                "left_backbone": "vit_b16",
                "right_backbone": "swin_tiny",
                "representation_similarity": 0.8,
                "representation_complementarity": 0.2,
            },
            {
                "left_backbone": "vit_b16",
                "right_backbone": "beit_base",
                "representation_similarity": 0.6,
                "representation_complementarity": 0.4,
            },
            {
                "left_backbone": "swin_tiny",
                "right_backbone": "beit_base",
                "representation_similarity": 0.7,
                "representation_complementarity": 0.3,
            },
        ]
    )
    results = pd.DataFrame(
        [
            {
                "run_id": "r1",
                "backbone": "vit_b16+swin_tiny+beit_base",
                "backbones": "['vit_b16', 'swin_tiny', 'beit_base']",
                "backbone_count": 3,
                "fusion_method": "concat",
                "macro_f1": 0.7,
            }
        ]
    )

    summary = build_fusion_complementarity_summary(results, pairwise, baseline_macro_f1=0.69)

    assert len(summary) == 1
    assert summary["avg_pairwise_representation_similarity"].iloc[0] == pytest.approx(0.7)
    assert summary["avg_pairwise_representation_complementarity"].iloc[0] == pytest.approx(0.3)
    assert summary["macro_f1_delta_vs_vit"].iloc[0] == pytest.approx(0.01)
