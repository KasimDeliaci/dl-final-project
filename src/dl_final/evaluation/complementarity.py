"""Representation complementarity analysis for cached transformer features."""

from __future__ import annotations

import ast
from collections.abc import Sequence
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.nn import functional as F

from dl_final.features.cache import (
    FeatureCache,
    cache_allows_prefix_split_verification,
    feature_cache_path,
    load_feature_cache,
    verify_cache_matches_split,
)


def compute_representation_complementarity(
    *,
    feature_root: str | Path,
    dataset_name: str,
    feature_source: str,
    backbones: Sequence[str],
    splits_dir: str | Path,
    split: str = "val",
    max_samples: int = 1504,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare backbone representations by sample-similarity correlation.

    Feature dimensions can differ across backbones. For each backbone, this function fits a
    StandardScaler on train features, transforms the requested split, builds a sample-by-sample
    cosine-similarity matrix, and correlates the upper triangles of those matrices.
    """

    if split == "test":
        raise ValueError("Test split is reserved for final audit; use train or val diagnostics.")
    if len(backbones) < 2:
        raise ValueError("At least two backbones are required for complementarity analysis.")

    train_caches: list[FeatureCache] = []
    split_caches: list[FeatureCache] = []
    for backbone in backbones:
        cache_dir = Path(feature_root) / dataset_name / feature_source / backbone
        train_cache = load_feature_cache(feature_cache_path(cache_dir, "train"))
        split_cache = load_feature_cache(feature_cache_path(cache_dir, split))
        verify_cache_matches_split(
            train_cache,
            Path(splits_dir) / "train.csv",
            allow_prefix=cache_allows_prefix_split_verification(train_cache),
        )
        verify_cache_matches_split(
            split_cache,
            Path(splits_dir) / f"{split}.csv",
            allow_prefix=cache_allows_prefix_split_verification(split_cache),
        )
        train_caches.append(train_cache)
        split_caches.append(split_cache)

    verify_cache_alignment(split_caches)
    sample_count = len(split_caches[0].labels)
    indices = _sample_indices(sample_count, max_samples=max_samples, seed=seed)
    similarity_vectors = {
        split_cache.backbone: _upper_triangle_cosine_vector(
            _scale_split_features(train_cache, split_cache)[indices]
        )
        for train_cache, split_cache in zip(train_caches, split_caches, strict=True)
    }

    rows: list[dict[str, float | int | str]] = []
    for left, right in combinations(backbones, 2):
        similarity = _safe_pearson(similarity_vectors[left], similarity_vectors[right])
        rows.append(
            {
                "split": split,
                "num_samples": int(len(indices)),
                "left_backbone": left,
                "right_backbone": right,
                "representation_similarity": similarity,
                "representation_complementarity": 1.0 - similarity,
                "method": "train_scaled_sample_cosine_rsa_pearson",
                "scaler_policy": "StandardScaler fit on train cache only",
                "test_policy": "test_not_used",
            }
        )
    return pd.DataFrame(rows)


def build_fusion_complementarity_summary(
    fusion_results: pd.DataFrame,
    pairwise_complementarity: pd.DataFrame,
    *,
    baseline_macro_f1: float = 0.6924,
) -> pd.DataFrame:
    """Attach average pairwise complementarity to each fusion result row."""

    pair_comp = {
        frozenset((row.left_backbone, row.right_backbone)): float(
            row.representation_complementarity
        )
        for row in pairwise_complementarity.itertuples(index=False)
    }
    pair_sim = {
        frozenset((row.left_backbone, row.right_backbone)): float(
            row.representation_similarity
        )
        for row in pairwise_complementarity.itertuples(index=False)
    }

    rows: list[dict[str, Any]] = []
    for row in fusion_results.itertuples(index=False):
        backbones = _parse_backbones(row.backbones)
        if len(backbones) < 2:
            continue
        pair_keys = [frozenset(pair) for pair in combinations(backbones, 2)]
        similarities = [pair_sim[key] for key in pair_keys if key in pair_sim]
        complementarities = [pair_comp[key] for key in pair_keys if key in pair_comp]
        if len(similarities) != len(pair_keys) or len(complementarities) != len(pair_keys):
            continue
        macro_f1 = float(row.macro_f1)
        rows.append(
            {
                "run_id": row.run_id,
                "backbone": row.backbone,
                "fusion_method": row.fusion_method,
                "backbone_count": int(row.backbone_count),
                "avg_pairwise_representation_similarity": float(np.mean(similarities)),
                "avg_pairwise_representation_complementarity": float(np.mean(complementarities)),
                "macro_f1": macro_f1,
                "macro_f1_delta_vs_vit": macro_f1 - baseline_macro_f1,
            }
        )
    return pd.DataFrame(rows).sort_values("macro_f1", ascending=False)


def save_representation_similarity_heatmap(
    pairwise: pd.DataFrame,
    path: str | Path,
) -> Path:
    """Save a pairwise representation-similarity heatmap."""

    import os

    os.environ["MPLBACKEND"] = "Agg"

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    backbones = sorted(set(pairwise["left_backbone"]) | set(pairwise["right_backbone"]))
    matrix = pd.DataFrame(np.eye(len(backbones)), index=backbones, columns=backbones)
    for row in pairwise.itertuples(index=False):
        matrix.loc[row.left_backbone, row.right_backbone] = row.representation_similarity
        matrix.loc[row.right_backbone, row.left_backbone] = row.representation_similarity

    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(matrix.values.astype(float), cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(len(backbones)), backbones, rotation=45, ha="right")
    ax.set_yticks(range(len(backbones)), backbones)
    ax.set_title("Validation representation similarity")
    for row_index in range(len(backbones)):
        for col_index in range(len(backbones)):
            ax.text(
                col_index,
                row_index,
                f"{matrix.iloc[row_index, col_index]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix.iloc[row_index, col_index] < 0.65 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def verify_cache_alignment(caches: Sequence[FeatureCache]) -> None:
    """Verify that feature caches have identical sample, lesion, label, and split order."""

    if len(caches) < 2:
        return
    reference = caches[0]
    fields = {
        "sample_id order": "sample_ids",
        "image_id order": "image_ids",
        "lesion_id order": "lesion_ids",
        "label order": "label_names",
        "split order": "split_names",
    }
    for cache in caches[1:]:
        for label, attr in fields.items():
            if getattr(cache, attr) != getattr(reference, attr):
                raise ValueError(
                    f"Cache {label} mismatch between {reference.backbone} and {cache.backbone}."
                )
        if not torch.equal(cache.labels, reference.labels):
            raise ValueError(
                f"Cache label index order mismatch between {reference.backbone} and "
                f"{cache.backbone}."
            )


def _scale_split_features(train_cache: FeatureCache, split_cache: FeatureCache) -> torch.Tensor:
    scaler = StandardScaler()
    scaler.fit(train_cache.features.numpy())
    scaled = scaler.transform(split_cache.features.numpy()).astype("float32")
    if not np.isfinite(scaled).all():
        raise ValueError(f"Scaled features contain NaN or Inf for {split_cache.backbone}.")
    return torch.from_numpy(scaled)


def _sample_indices(sample_count: int, *, max_samples: int, seed: int) -> torch.Tensor:
    if max_samples <= 0 or sample_count <= max_samples:
        return torch.arange(sample_count)
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(sample_count, generator=generator)[:max_samples].sort().values


def _upper_triangle_cosine_vector(features: torch.Tensor) -> np.ndarray:
    normalized = F.normalize(features.float(), p=2, dim=1)
    similarity = normalized @ normalized.T
    row_index, col_index = torch.triu_indices(
        similarity.shape[0],
        similarity.shape[1],
        offset=1,
    )
    return similarity[row_index, col_index].cpu().numpy()


def _safe_pearson(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError("Similarity vectors must have the same shape.")
    if np.isclose(left.std(), 0.0) or np.isclose(right.std(), 0.0):
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def _parse_backbones(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    parsed = ast.literal_eval(str(value))
    if not isinstance(parsed, list):
        raise ValueError(f"Expected a backbone list, got: {value!r}")
    return [str(item) for item in parsed]
