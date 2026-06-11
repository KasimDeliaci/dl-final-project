"""Fine-tune final transformer blocks and extract fine-tuned feature caches."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from dl_final.config import load_dataset_config
from dl_final.data.datasets import HAM10000ImageDataset
from dl_final.data.transforms import build_feature_transform, build_finetune_train_transform
from dl_final.features.cache import backbone_cache_dir
from dl_final.models.backbones import default_finetuning_policy, supported_backbones
from dl_final.training.finetune import extract_finetuned_feature_cache, finetune_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/finetune_backbones.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--backbones", nargs="+", default=None)
    parser.add_argument("--checkpoint-dir", default=None)
    parser.add_argument("--feature-root", default="artifacts/features")
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--backbone-learning-rate", type=float, default=None)
    parser.add_argument("--head-learning-rate", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--limit-per-split", type=int, default=None)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--skip-feature-extraction", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    finetune_config = load_dataset_config(args.config)["finetuning"]
    dataset_config = load_dataset_config(args.dataset_config)["dataset"]
    default_config = load_dataset_config(args.default_config)
    runtime_config = default_config.get("runtime", {})

    seed = int(finetune_config.get("seed", dataset_config.get("seed", 42)))
    _seed_everything(seed)

    device = _resolve_device(args.device or runtime_config.get("device", "auto"))
    mixed_precision = (
        bool(runtime_config.get("mixed_precision", True))
        and not args.no_mixed_precision
        and device.type in {"cuda", "mps"}
    )
    batch_size = int(args.batch_size or finetune_config.get("batch_size", 16))
    num_workers = int(
        args.num_workers if args.num_workers is not None else runtime_config.get("num_workers", 2)
    )
    epochs = int(args.epochs or finetune_config.get("epochs", 8))
    backbone_lr = float(
        args.backbone_learning_rate or finetune_config.get("backbone_learning_rate", 1e-5)
    )
    head_lr = float(args.head_learning_rate or finetune_config.get("head_learning_rate", 1e-4))
    weight_decay = float(args.weight_decay or finetune_config.get("weight_decay", 1e-4))
    patience = int(
        args.early_stopping_patience or finetune_config.get("early_stopping_patience", 3)
    )
    checkpoint_dir = Path(
        args.checkpoint_dir
        or finetune_config.get("checkpoint_dir", "artifacts/checkpoints/ham10000/finetuned")
    )
    backbones = _selected_backbones(args, finetune_config)
    class_names = list(dataset_config["class_names"])
    image_size = int(dataset_config.get("image_size") or finetune_config.get("image_size", 224))

    train_loader = _create_loader(
        split_csv=Path(dataset_config["splits_dir"]) / "train.csv",
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        split_name="train",
        max_samples=args.limit_per_split,
        train=True,
        seed=seed,
    )
    val_loader = _create_loader(
        split_csv=Path(dataset_config["splits_dir"]) / "val.csv",
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        split_name="val",
        max_samples=args.limit_per_split,
        train=False,
        seed=seed,
    )
    extraction_loaders = {
        "train": _create_loader(
            split_csv=Path(dataset_config["splits_dir"]) / "train.csv",
            class_names=class_names,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            split_name="train",
            max_samples=args.limit_per_split,
            train=False,
            seed=seed,
        ),
        "val": val_loader,
    }

    summary: list[dict[str, Any]] = []
    for backbone in backbones:
        policy = _policy_for_backbone(finetune_config, backbone)
        checkpoint_path = checkpoint_dir / backbone / "best.pt"
        checkpoint_metadata: dict[str, Any] = {}
        run_dir: Path | None = None
        if not args.skip_training:
            print(f"Fine-tuning {backbone} on {device}.")
            checkpoint_path, run_dir, checkpoint_metadata = finetune_backbone(
                backbone=backbone,
                train_loader=train_loader,
                val_loader=val_loader,
                class_names=class_names,
                device=device,
                checkpoint_dir=checkpoint_dir,
                seed=seed,
                epochs=epochs,
                backbone_learning_rate=backbone_lr,
                head_learning_rate=head_lr,
                weight_decay=weight_decay,
                early_stopping_patience=patience,
                policy=policy,
                mixed_precision=mixed_precision,
                pretrained=not args.no_pretrained,
                class_weighting=not args.no_class_weights,
                run_root=args.run_root,
                limit_per_split=args.limit_per_split,
                config={
                    "finetune_config": str(Path(args.config)),
                    "dataset_config": str(Path(args.dataset_config)),
                    "default_config": str(Path(args.default_config)),
                },
            )
            print(f"Wrote selected checkpoint: {checkpoint_path}")
        if not args.skip_feature_extraction:
            cache_dir = backbone_cache_dir(
                args.feature_root,
                str(dataset_config["name"]),
                "finetuned",
                backbone,
            )
            manifest_path = extract_finetuned_feature_cache(
                backbone=backbone,
                checkpoint_path=checkpoint_path,
                loaders=extraction_loaders,
                output_dir=cache_dir,
                class_names=class_names,
                seed=seed,
                device=device,
                mixed_precision=mixed_precision,
                config={
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_metadata": checkpoint_metadata,
                    "selection_metric": "validation_macro_f1",
                    "test_policy": "not_used_in_sprint4",
                    "feature_source": "finetuned",
                    "limit_per_split": args.limit_per_split,
                    "unfreeze_policy": policy,
                    "batch_size": batch_size,
                    "image_size": image_size,
                },
            )
            print(f"Wrote fine-tuned feature manifest: {manifest_path}")
        summary.append(
            {
                "backbone": backbone,
                "checkpoint_path": str(checkpoint_path),
                "run_dir": str(run_dir) if run_dir is not None else None,
                "feature_cache_dir": str(
                    backbone_cache_dir(
                        args.feature_root,
                        str(dataset_config["name"]),
                        "finetuned",
                        backbone,
                    )
                ),
            }
        )

    summary_path = Path(args.run_root) / "s4_finetune_backbones_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote fine-tuning summary: {summary_path}")


def _selected_backbones(args: argparse.Namespace, config: dict[str, Any]) -> list[str]:
    if args.backbones is not None:
        return list(args.backbones)
    return list(config.get("backbones", supported_backbones()))


def _policy_for_backbone(config: dict[str, Any], backbone: str) -> str:
    policies = config.get("unfreeze_policies", {})
    if isinstance(policies, dict) and backbone in policies:
        return str(policies[backbone])
    return default_finetuning_policy(backbone)


def _create_loader(
    *,
    split_csv: Path,
    class_names: list[str],
    image_size: int,
    batch_size: int,
    num_workers: int,
    split_name: str,
    max_samples: int | None,
    train: bool,
    seed: int,
) -> DataLoader:
    transform = (
        build_finetune_train_transform(image_size) if train else build_feature_transform(image_size)
    )
    dataset = HAM10000ImageDataset(
        split_csv,
        class_names=class_names,
        transform=transform,
        split_name=split_name,
        max_samples=max_samples,
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        generator=generator if train else None,
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
