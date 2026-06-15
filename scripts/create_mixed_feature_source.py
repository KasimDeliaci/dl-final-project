"""Create a mixed feature-source namespace from existing backbone cache directories."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import torch

from dl_final.config import load_dataset_config
from dl_final.features.cache import (
    cache_allows_prefix_split_verification,
    feature_cache_path,
    load_feature_cache,
    verify_cache_matches_split,
)
from dl_final.models.backbones import expected_feature_dim

EXPECTED_SPLIT_COUNTS = {"train": 7008, "val": 1504}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--feature-root", default="artifacts/features")
    parser.add_argument("--output-source", required=True)
    parser.add_argument(
        "--mapping",
        nargs="+",
        required=True,
        metavar="BACKBONE=SOURCE",
        help="Backbone to source mapping, e.g. vit_b16=frozen swin_tiny=finetuned.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = load_dataset_config(args.dataset_config)["dataset"]
    dataset_name = str(dataset_config["name"])
    splits_dir = Path(dataset_config["splits_dir"])
    feature_root = Path(args.feature_root) / dataset_name
    mapping = parse_mapping(args.mapping)
    output_root = feature_root / args.output_source
    if output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"{output_root} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    reference_rows: dict[str, dict[str, list[Any]]] = {}
    mixed_manifest: dict[str, Any] = {
        "feature_source": args.output_source,
        "dataset": dataset_name,
        "selection_metric": "validation_macro_f1",
        "test_policy": "not_loaded_or_used_in_e3f",
        "backbones": {},
    }
    for backbone, source in mapping.items():
        source_dir = feature_root / source / backbone
        dest_dir = output_root / backbone
        validate_source(backbone, source_dir, splits_dir, reference_rows)
        shutil.copytree(source_dir, dest_dir)
        mixed_manifest["backbones"][backbone] = {
            "source": source,
            "source_dir": str(source_dir),
            "dest_dir": str(dest_dir),
            "feature_dim": expected_feature_dim(backbone),
            "splits": EXPECTED_SPLIT_COUNTS,
        }

    (output_root / "mixed_manifest.json").write_text(
        json.dumps(mixed_manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote mixed feature source: {output_root}")


def parse_mapping(values: list[str]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Invalid mapping {value!r}; expected BACKBONE=SOURCE.")
        backbone, source = value.split("=", 1)
        backbone = backbone.strip()
        source = source.strip()
        if not backbone or not source:
            raise ValueError(f"Invalid mapping {value!r}; expected BACKBONE=SOURCE.")
        mapping[backbone] = source
    if len(mapping) < 2:
        raise ValueError("At least two backbone mappings are required.")
    return mapping


def validate_source(
    backbone: str,
    source_dir: Path,
    splits_dir: Path,
    reference_rows: dict[str, dict[str, list[Any]]],
) -> None:
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source cache directory: {source_dir}")
    for split, expected_rows in EXPECTED_SPLIT_COUNTS.items():
        cache = load_feature_cache(feature_cache_path(source_dir, split))
        verify_cache_matches_split(
            cache,
            splits_dir / f"{split}.csv",
            allow_prefix=cache_allows_prefix_split_verification(cache),
        )
        if int(cache.features.shape[0]) != expected_rows:
            raise ValueError(
                f"{backbone} {split} cache has {int(cache.features.shape[0])} rows; "
                f"expected {expected_rows}."
            )
        if cache.feature_dim != expected_feature_dim(backbone):
            raise ValueError(
                f"{backbone} feature dim is {cache.feature_dim}; "
                f"expected {expected_feature_dim(backbone)}."
            )
        if not torch.isfinite(cache.features).all():
            raise ValueError(f"{backbone} {split} cache contains NaN or Inf values.")
        row_keys = {
            "sample_ids": cache.sample_ids,
            "image_ids": cache.image_ids,
            "lesion_ids": cache.lesion_ids,
            "label_names": cache.label_names,
            "split_names": cache.split_names,
            "labels": cache.labels.tolist(),
        }
        if split not in reference_rows:
            reference_rows[split] = row_keys
        elif row_keys != reference_rows[split]:
            raise ValueError(f"{backbone} {split} cache row order does not match reference.")


if __name__ == "__main__":
    main()
