"""Extract frozen transformer feature vectors from HAM10000 split CSVs."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dl_final.config import load_dataset_config
from dl_final.data.datasets import HAM10000ImageDataset
from dl_final.data.transforms import build_feature_transform
from dl_final.features.cache import backbone_cache_dir, save_backbone_manifest
from dl_final.features.extract import extract_and_cache_backbone
from dl_final.models.backbones import build_frozen_feature_extractor, supported_backbones


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--feature-source", default="frozen", choices=["frozen"])
    parser.add_argument("--backbones", nargs="+", default=supported_backbones())
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val"],
        choices=["train", "val", "test"],
    )
    parser.add_argument("--include-test", action="store_true")
    parser.add_argument("--output-root", default="artifacts/features")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--no-pretrained", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = load_dataset_config(args.config)["dataset"]
    default_config = load_dataset_config(args.default_config)
    runtime = default_config.get("runtime", {})
    seed = int(dataset_config.get("seed", runtime.get("seed", 42)))
    _seed_everything(seed)

    device_arg = args.device if args.device != "auto" else runtime.get("device", "auto")
    device = _resolve_device(device_arg)
    mixed_precision = (
        bool(runtime.get("mixed_precision", True))
        and not args.no_mixed_precision
        and device.type in {"cuda", "mps"}
    )
    splits = list(dict.fromkeys(args.splits + (["test"] if args.include_test else [])))
    class_names = list(dataset_config["class_names"])
    image_size = int(dataset_config.get("image_size") or 224)

    loaders = {
        split: _create_loader(
            split_csv=Path(dataset_config["splits_dir"]) / f"{split}.csv",
            class_names=class_names,
            image_size=image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            split_name=split,
            max_samples=args.limit_per_split,
        )
        for split in splits
    }

    for backbone in args.backbones:
        model = build_frozen_feature_extractor(backbone, pretrained=not args.no_pretrained)
        cache_dir = backbone_cache_dir(
            args.output_root,
            str(dataset_config["name"]),
            args.feature_source,
            backbone,
        )
        caches = extract_and_cache_backbone(
            model=model,
            backbone=backbone,
            loaders=loaders,
            output_dir=cache_dir,
            class_names=class_names,
            feature_source=args.feature_source,
            seed=seed,
            device=device,
            mixed_precision=mixed_precision,
            config={
                "dataset_config": str(Path(args.config)),
                "default_config": str(Path(args.default_config)),
                "batch_size": args.batch_size,
                "image_size": image_size,
                "pretrained": not args.no_pretrained,
                "limit_per_split": args.limit_per_split,
                "splits": splits,
            },
        )
        manifest = save_backbone_manifest(cache_dir, caches)
        print(f"Wrote {backbone} feature manifest: {manifest}")


def _create_loader(
    *,
    split_csv: Path,
    class_names: list[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    split_name: str,
    max_samples: int | None,
) -> DataLoader:
    dataset = HAM10000ImageDataset(
        split_csv,
        class_names=class_names,
        transform=build_feature_transform(image_size),
        split_name=split_name,
        max_samples=max_samples,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _resolve_device(value: str | None) -> torch.device:
    requested = str(value or "auto")
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(requested)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
