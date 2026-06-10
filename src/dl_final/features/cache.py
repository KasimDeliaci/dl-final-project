"""Feature-cache read/write and validation helpers."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class FeatureCache:
    path: Path
    features: torch.Tensor
    labels: torch.Tensor
    sample_ids: list[str]
    image_ids: list[str]
    lesion_ids: list[str]
    label_names: list[str]
    split_names: list[str]
    split: str
    backbone: str
    feature_dim: int
    metadata: dict[str, Any]


class FeatureDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Tensor dataset backed by one cached feature split."""

    def __init__(self, cache: FeatureCache) -> None:
        self.cache = cache
        if len(cache.features) != len(cache.labels):
            raise ValueError("Feature and label counts do not match.")

    def __len__(self) -> int:
        return len(self.cache.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cache.features[index].float(), self.cache.labels[index].long()


def backbone_cache_dir(
    output_root: str | Path,
    dataset_name: str,
    feature_source: str,
    backbone: str,
) -> Path:
    return Path(output_root) / dataset_name / feature_source / backbone


def feature_cache_path(cache_dir: str | Path, split: str) -> Path:
    return Path(cache_dir) / f"{split}.pt"


def save_feature_cache(
    path: str | Path,
    *,
    features: torch.Tensor,
    labels: torch.Tensor,
    sample_ids: list[str],
    image_ids: list[str],
    lesion_ids: list[str],
    label_names: list[str],
    split_names: list[str],
    split: str,
    backbone: str,
    class_names: list[str],
    feature_source: str,
    seed: int,
    config: dict[str, Any] | None = None,
) -> FeatureCache:
    """Save one split's features with alignment metadata."""

    cache_path = Path(path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    features = features.detach().cpu().float()
    labels = labels.detach().cpu().long()
    metadata: dict[str, Any] = {
        "split": split,
        "backbone": backbone,
        "feature_source": feature_source,
        "feature_dim": int(features.shape[1]),
        "num_samples": int(features.shape[0]),
        "class_names": list(class_names),
        "seed": int(seed),
        "config": config or {},
    }
    payload = {
        "features": features,
        "labels": labels,
        "sample_ids": sample_ids,
        "image_ids": image_ids,
        "lesion_ids": lesion_ids,
        "label_names": label_names,
        "split_names": split_names,
        "metadata": metadata,
    }
    _validate_payload(payload)
    torch.save(payload, cache_path)
    _write_split_manifest(cache_path.with_name(f"{split}_manifest.csv"), payload)
    return load_feature_cache(cache_path)


def load_feature_cache(path: str | Path, map_location: str | torch.device = "cpu") -> FeatureCache:
    cache_path = Path(path)
    if not cache_path.exists():
        raise FileNotFoundError(f"Feature cache does not exist: {cache_path}")
    try:
        payload = torch.load(cache_path, map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(cache_path, map_location=map_location)
    _validate_payload(payload)
    metadata = dict(payload["metadata"])
    return FeatureCache(
        path=cache_path,
        features=payload["features"],
        labels=payload["labels"],
        sample_ids=list(payload["sample_ids"]),
        image_ids=list(payload["image_ids"]),
        lesion_ids=list(payload["lesion_ids"]),
        label_names=list(payload["label_names"]),
        split_names=list(payload["split_names"]),
        split=str(metadata["split"]),
        backbone=str(metadata["backbone"]),
        feature_dim=int(metadata["feature_dim"]),
        metadata=metadata,
    )


def save_backbone_manifest(cache_dir: str | Path, caches: list[FeatureCache]) -> Path:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "backbone": caches[0].backbone if caches else None,
        "feature_source": caches[0].metadata.get("feature_source") if caches else None,
        "feature_dim": caches[0].feature_dim if caches else None,
        "splits": {
            cache.split: {
                "path": str(cache.path),
                "manifest_csv": str(cache.path.with_name(f"{cache.split}_manifest.csv")),
                "num_samples": int(cache.features.shape[0]),
                "feature_dim": cache.feature_dim,
                "has_nan": bool(torch.isnan(cache.features).any().item()),
                "has_inf": bool(torch.isinf(cache.features).any().item()),
            }
            for cache in caches
        },
    }
    path = cache_root / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return path


def verify_cache_matches_split(
    cache: FeatureCache,
    split_csv: str | Path,
    *,
    allow_prefix: bool = False,
) -> None:
    """Verify row-order alignment against the canonical split CSV."""

    split = pd.read_csv(split_csv)
    expected_image_ids = split["image_id"].astype(str).tolist()
    expected_labels = split["label"].astype(str).tolist()
    if "lesion_id" in split.columns:
        expected_lesion_ids = split["lesion_id"].astype(str).tolist()
    else:
        expected_lesion_ids = [""] * len(split)
    if allow_prefix:
        expected_image_ids = expected_image_ids[: len(cache.image_ids)]
        expected_labels = expected_labels[: len(cache.label_names)]
        expected_lesion_ids = expected_lesion_ids[: len(cache.lesion_ids)]
    if cache.image_ids != expected_image_ids:
        raise ValueError(f"Cache image_id order does not match split CSV: {split_csv}")
    if cache.label_names != expected_labels:
        raise ValueError(f"Cache label order does not match split CSV: {split_csv}")
    if cache.lesion_ids != expected_lesion_ids:
        raise ValueError(f"Cache lesion_id order does not match split CSV: {split_csv}")


def cache_allows_prefix_split_verification(cache: FeatureCache) -> bool:
    config = cache.metadata.get("config", {})
    return bool(isinstance(config, dict) and config.get("limit_per_split") is not None)


def class_weights_from_cache(cache: FeatureCache, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(cache.labels.long(), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    weights[counts == 0] = 0.0
    return weights


def _write_split_manifest(path: Path, payload: dict[str, Any]) -> None:
    frame = pd.DataFrame(
        {
            "row_index": range(len(payload["labels"])),
            "sample_id": payload["sample_ids"],
            "image_id": payload["image_ids"],
            "lesion_id": payload["lesion_ids"],
            "split": payload["split_names"],
            "label": payload["label_names"],
            "label_index": payload["labels"].tolist(),
            "backbone": payload["metadata"]["backbone"],
        }
    )
    frame.to_csv(path, index=False)


def _validate_payload(payload: dict[str, Any]) -> None:
    required = {
        "features",
        "labels",
        "sample_ids",
        "image_ids",
        "lesion_ids",
        "label_names",
        "split_names",
        "metadata",
    }
    missing = required - set(payload)
    if missing:
        raise ValueError(f"Feature cache is missing keys: {sorted(missing)}")
    features = payload["features"]
    labels = payload["labels"]
    if not isinstance(features, torch.Tensor) or features.ndim != 2:
        raise ValueError("Feature cache `features` must be a 2D torch.Tensor.")
    if not isinstance(labels, torch.Tensor) or labels.ndim != 1:
        raise ValueError("Feature cache `labels` must be a 1D torch.Tensor.")
    if torch.isnan(features).any() or torch.isinf(features).any():
        raise ValueError("Feature cache contains NaN or Inf values.")
    sample_count = int(features.shape[0])
    fields = ("sample_ids", "image_ids", "lesion_ids", "label_names", "split_names")
    for field in fields:
        if len(payload[field]) != sample_count:
            raise ValueError(f"Feature cache `{field}` length does not match feature rows.")
    if int(labels.shape[0]) != sample_count:
        raise ValueError("Feature and label counts do not match.")
    metadata = payload["metadata"]
    if int(metadata["feature_dim"]) != int(features.shape[1]):
        raise ValueError("Feature dimension metadata does not match feature tensor.")
