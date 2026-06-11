"""Train validation-selected MLP baselines on cached frozen transformer features."""

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl_final.config import load_dataset_config
from dl_final.evaluation.reports import (
    export_single_backbone_report_assets,
    prediction_frame,
    write_run_artifacts,
)
from dl_final.features.cache import (
    cache_allows_prefix_split_verification,
    class_weights_from_cache,
    feature_cache_path,
    load_feature_cache,
    verify_cache_matches_split,
)
from dl_final.models.backbones import backbone_alias, supported_backbones
from dl_final.models.mlp import FeatureMLP
from dl_final.training.loops import evaluate_model, train_mlp_model
from dl_final.training.optim import build_optimizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--default-config", default="configs/default.yaml")
    parser.add_argument("--feature-root", default="artifacts/features")
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--tables-dir", default="artifacts/report_assets/tables")
    parser.add_argument("--figures-dir", default="artifacts/report_assets/figures")
    parser.add_argument("--feature-source", default="frozen")
    parser.add_argument("--backbones", nargs="+", default=supported_backbones())
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", default="adamw", choices=["adamw", "adam", "sgd"])
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[512, 256])
    parser.add_argument("--early-stopping-patience", type=int, default=6)
    parser.add_argument("--no-class-weights", action="store_true")
    parser.add_argument("--run-tag", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = load_dataset_config(args.dataset_config)["dataset"]
    seed = int(dataset_config.get("seed", 42))
    _seed_everything(seed)
    device = _resolve_device(args.device)

    for backbone in args.backbones:
        run_single_backbone(args, dataset_config, backbone, seed, device)

    exported = export_single_backbone_report_assets(
        args.run_root,
        args.tables_dir,
        args.figures_dir,
        feature_source=args.feature_source,
    )
    print("Exported report assets:")
    for name, path in exported.items():
        print(f"  {name}: {path}")


def run_single_backbone(
    args: argparse.Namespace,
    dataset_config: dict,
    backbone: str,
    seed: int,
    device: torch.device,
) -> Path:
    started = perf_counter()
    class_names = list(dataset_config["class_names"])
    cache_dir = (
        Path(args.feature_root) / str(dataset_config["name"]) / args.feature_source / backbone
    )
    train_cache = load_feature_cache(feature_cache_path(cache_dir, "train"))
    val_cache = load_feature_cache(feature_cache_path(cache_dir, "val"))
    for split, cache in (("train", train_cache), ("val", val_cache)):
        verify_cache_matches_split(
            cache,
            Path(dataset_config["splits_dir"]) / f"{split}.csv",
            allow_prefix=cache_allows_prefix_split_verification(cache),
        )

    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_cache.features.numpy()).astype("float32")
    val_features = scaler.transform(val_cache.features.numpy()).astype("float32")

    train_dataset = TensorDataset(
        torch.from_numpy(train_features),
        train_cache.labels.long(),
    )
    val_dataset = TensorDataset(
        torch.from_numpy(val_features),
        val_cache.labels.long(),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = FeatureMLP(
        input_dim=train_cache.feature_dim,
        num_classes=len(class_names),
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )
    weights = (
        class_weights_from_cache(train_cache, len(class_names)).to(device)
        if not args.no_class_weights
        else None
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = build_optimizer(
        model.parameters(),
        optimizer_name=args.optimizer,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        momentum=args.momentum,
    )
    model, history, best_metrics = train_mlp_model(
        model,
        train_loader,
        val_loader,
        class_names=class_names,
        device=device,
        epochs=args.epochs,
        criterion=criterion,
        optimizer=optimizer,
        early_stopping_patience=args.early_stopping_patience,
    )
    metrics, y_true, y_pred, probabilities = evaluate_model(
        model,
        val_loader,
        class_names=class_names,
        device=device,
    )
    metrics["best_epoch"] = best_metrics.get("best_epoch")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    tag = f"_{args.run_tag}" if args.run_tag else ""
    run_id = (
        f"{timestamp}_s2_{args.feature_source}_{backbone_alias(backbone)}"
        f"_none_mlp{tag}_seed{seed}"
    )
    run_dir = Path(args.run_root) / run_id
    runtime_seconds = round(perf_counter() - started, 4)
    run_config = {
        "run_id": run_id,
        "experiment_id": "E3" if args.feature_source == "finetuned" else "E1",
        "seed": seed,
        "dataset": dataset_config["name"],
        "feature_source": args.feature_source,
        "backbone": backbone,
        "backbones": [backbone],
        "fusion_method": "none",
        "feature_dim": train_cache.feature_dim,
        "scaler": "StandardScaler fit on train cache only",
        "class_weighting": not args.no_class_weights,
        "batch_size": args.batch_size,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hidden_dims": args.hidden_dims,
        "dropout": args.dropout,
        "early_stopping_patience": args.early_stopping_patience,
        "selection_metric": "validation_macro_f1",
        "test_policy": (
            "not_used_in_sprint4" if args.feature_source == "finetuned"
            else "not_used_in_sprint2_model_selection"
        ),
        "feature_cache_dir": str(cache_dir),
        "runtime_seconds": runtime_seconds,
    }
    write_run_artifacts(
        run_dir,
        run_config=run_config,
        history=history,
        metrics=metrics,
        predictions=prediction_frame(
            cache=val_cache,
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            class_names=class_names,
        ),
        class_names=class_names,
        backbone=backbone,
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    np.savez(
        run_dir / "scaler_stats.npz",
        mean=scaler.mean_,
        scale=scaler.scale_,
        var=scaler.var_,
    )
    print(f"Wrote MLP validation run: {run_dir}")
    return run_dir


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
