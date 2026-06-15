"""Train E3c metadata-augmented cached-feature MLP diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from time import perf_counter
from types import SimpleNamespace
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from dl_final.config import load_dataset_config
from dl_final.evaluation.reports import prediction_frame, write_run_artifacts
from dl_final.features.cache import (
    FeatureCache,
    cache_allows_prefix_split_verification,
    class_weights_from_cache,
    feature_cache_path,
    load_feature_cache,
    verify_cache_matches_split,
)
from dl_final.features.metadata import MetadataPreprocessor
from dl_final.models.backbones import expected_feature_dim
from dl_final.models.mlp import FeatureMLP
from dl_final.training.loops import evaluate_model, train_mlp_model
from dl_final.training.optim import build_optimizer

EXPECTED_SPLIT_COUNTS = {"train": 7008, "val": 1504}
E3C_CONDITIONS = (
    "metadata_only",
    "ft_vit_swin_concat_plus_metadata",
    "ft_vit_swin_beit_concat_plus_metadata",
    "ft_vit_plus_metadata",
)
CANONICAL_CONDITIONS = E3C_CONDITIONS[:3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--feature-root", default="artifacts/features")
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument(
        "--condition",
        choices=E3C_CONDITIONS,
        nargs="+",
        default=CANONICAL_CONDITIONS,
    )
    parser.add_argument("--device", default="cpu")
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--run-tag", default="e3c_metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = load_dataset_config(args.dataset_config)["dataset"]
    seed = int(args.seed if args.seed is not None else dataset_config.get("seed", 42))
    _seed_everything(seed)
    device = _resolve_device(args.device)
    for condition in args.condition:
        run_e3c_condition(
            condition,
            args=args,
            dataset_config=dataset_config,
            seed=seed,
            device=device,
        )


def run_e3c_condition(
    condition: str,
    *,
    args: argparse.Namespace,
    dataset_config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> Path:
    started = perf_counter()
    class_names = list(dataset_config["class_names"])
    splits_dir = Path(dataset_config["splits_dir"])
    train_split = pd.read_csv(splits_dir / "train.csv")
    val_split = pd.read_csv(splits_dir / "val.csv")
    backbones = _backbones_for_condition(condition)

    metadata_preprocessor = MetadataPreprocessor().fit(train_split)
    train_metadata_result = metadata_preprocessor.transform(train_split)
    val_metadata_result = metadata_preprocessor.transform(val_split)
    train_metadata = train_metadata_result.features
    val_metadata = val_metadata_result.features
    _assert_shape("train metadata", train_metadata, EXPECTED_SPLIT_COUNTS["train"])
    _assert_shape("validation metadata", val_metadata, EXPECTED_SPLIT_COUNTS["val"])

    if condition == "metadata_only":
        train_features = train_metadata
        val_features = val_metadata
        train_labels = _labels_from_split(train_split, class_names)
        val_labels = _labels_from_split(val_split, class_names)
        reference = _reference_from_split(val_split, val_labels)
        image_preprocessing_metadata: dict[str, Any] = {"image_features": None}
        scaler_stats: dict[str, np.ndarray] = {}
        feature_source = "metadata"
        fusion_method = "metadata_only"
        backbone_label = "metadata"
    else:
        train_caches, val_caches = load_aligned_caches(
            feature_root=args.feature_root,
            dataset_name=str(dataset_config["name"]),
            feature_source="finetuned",
            backbones=backbones,
            splits_dir=splits_dir,
        )
        train_image, val_image, image_preprocessing_metadata, scaler_stats = scale_and_concat(
            train_caches,
            val_caches,
        )
        train_features = np.concatenate([train_image, train_metadata], axis=1).astype("float32")
        val_features = np.concatenate([val_image, val_metadata], axis=1).astype("float32")
        train_labels = train_caches[0].labels.long()
        val_labels = val_caches[0].labels.long()
        reference = val_caches[0]
        feature_source = "finetuned"
        fusion_method = "concat_plus_metadata"
        backbone_label = "+".join(backbones)

    _assert_finite("train E3c features", train_features)
    _assert_finite("validation E3c features", val_features)
    train_dataset = TensorDataset(torch.from_numpy(train_features), train_labels)
    val_dataset = TensorDataset(torch.from_numpy(val_features), val_labels)
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    model = FeatureMLP(
        input_dim=int(train_features.shape[1]),
        num_classes=len(class_names),
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
    )
    weights = (
        _class_weights(train_labels, len(class_names)).to(device)
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
    run_id = f"{timestamp}_e3c_{condition}{tag}_seed{seed}"
    run_dir = Path(args.run_root) / run_id
    runtime_seconds = round(perf_counter() - started, 4)
    metadata_output_dim = int(train_metadata.shape[1])
    image_input_dim = int(train_features.shape[1] - metadata_output_dim)
    run_config = {
        "run_id": run_id,
        "experiment_id": "E3c",
        "condition": condition,
        "seed": seed,
        "dataset": dataset_config["name"],
        "feature_source": feature_source,
        "backbone": backbone_label,
        "backbones": backbones,
        "fusion_method": fusion_method,
        "feature_dim": int(train_features.shape[1]),
        "image_feature_dim": image_input_dim,
        "metadata_feature_dim": metadata_output_dim,
        "metadata_fields": ["age", "sex", "localization"],
        "metadata_policy": "train-only imputation, scaling, and categorical vocabulary",
        "excluded_metadata_fields": [
            "dx",
            "dx_type",
            "dataset",
            "image_id",
            "sample_id",
            "lesion_id",
            "image_path",
        ],
        "scaler": "image StandardScaler fit per backbone train cache; metadata fit on train split",
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
        "test_policy": "not_loaded_or_used_in_e3c",
        "runtime_seconds": runtime_seconds,
    }
    write_run_artifacts(
        run_dir,
        run_config=run_config,
        history=history,
        metrics=metrics,
        predictions=prediction_frame(
            cache=reference,
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            class_names=class_names,
        ),
        class_names=class_names,
        backbone=condition,
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    np.savez(run_dir / "scaler_stats.npz", **scaler_stats)
    metadata_artifact = {
        "metadata_preprocessor": metadata_preprocessor.to_metadata(),
        "train_transform": train_metadata_result.metadata,
        "validation_transform": val_metadata_result.metadata,
        "image_preprocessing": image_preprocessing_metadata,
    }
    (run_dir / "metadata_preprocessing.json").write_text(
        json.dumps(metadata_artifact, indent=2),
        encoding="utf-8",
    )
    (run_dir / "runtime_metadata.json").write_text(
        json.dumps(
            {
                "runtime_seconds": runtime_seconds,
                "device": str(device),
                "train_rows": int(train_features.shape[0]),
                "validation_rows": int(val_features.shape[0]),
                "prediction_rows": int(probabilities.shape[0]),
                "test_rows_loaded": 0,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote E3c metadata validation run: {run_dir}")
    return run_dir


def load_aligned_caches(
    *,
    feature_root: str | Path,
    dataset_name: str,
    feature_source: str,
    backbones: Sequence[str],
    splits_dir: Path,
) -> tuple[list[FeatureCache], list[FeatureCache]]:
    train_caches: list[FeatureCache] = []
    val_caches: list[FeatureCache] = []
    for backbone in backbones:
        cache_dir = Path(feature_root) / dataset_name / feature_source / backbone
        train_cache = load_feature_cache(feature_cache_path(cache_dir, "train"))
        val_cache = load_feature_cache(feature_cache_path(cache_dir, "val"))
        for split, cache in (("train", train_cache), ("val", val_cache)):
            verify_cache_matches_split(
                cache,
                splits_dir / f"{split}.csv",
                allow_prefix=cache_allows_prefix_split_verification(cache),
            )
            expected_count = EXPECTED_SPLIT_COUNTS[split]
            if int(cache.features.shape[0]) != expected_count:
                raise ValueError(
                    f"{backbone} {split} cache has {int(cache.features.shape[0])} rows; "
                    f"expected {expected_count}."
                )
            if cache.feature_dim != expected_feature_dim(backbone):
                raise ValueError(
                    f"{backbone} feature dim is {cache.feature_dim}; "
                    f"expected {expected_feature_dim(backbone)}."
                )
        train_caches.append(train_cache)
        val_caches.append(val_cache)
    verify_cache_alignment(train_caches)
    verify_cache_alignment(val_caches)
    return train_caches, val_caches


def verify_cache_alignment(caches: Sequence[FeatureCache]) -> None:
    if len(caches) < 2:
        return
    reference = caches[0]
    for cache in caches[1:]:
        if cache.sample_ids != reference.sample_ids:
            raise ValueError("Cache sample_id order mismatch.")
        if cache.image_ids != reference.image_ids:
            raise ValueError("Cache image_id order mismatch.")
        if cache.lesion_ids != reference.lesion_ids:
            raise ValueError("Cache lesion_id order mismatch.")
        if cache.label_names != reference.label_names:
            raise ValueError("Cache label order mismatch.")
        if cache.split_names != reference.split_names:
            raise ValueError("Cache split order mismatch.")
        if not torch.equal(cache.labels, reference.labels):
            raise ValueError("Cache label index order mismatch.")


def scale_and_concat(
    train_caches: Sequence[FeatureCache],
    val_caches: Sequence[FeatureCache],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    train_blocks: list[np.ndarray] = []
    val_blocks: list[np.ndarray] = []
    scaler_metadata: list[dict[str, Any]] = []
    scaler_stats: dict[str, np.ndarray] = {}
    for train_cache, val_cache in zip(train_caches, val_caches, strict=True):
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_cache.features.numpy()).astype("float32")
        val_features = scaler.transform(val_cache.features.numpy()).astype("float32")
        train_blocks.append(train_features)
        val_blocks.append(val_features)
        scaler_metadata.append(
            {
                "backbone": train_cache.backbone,
                "fit_split": "train",
                "input_dim": train_cache.feature_dim,
                "train_rows": int(train_features.shape[0]),
                "validation_rows": int(val_features.shape[0]),
            }
        )
        prefix = train_cache.backbone
        scaler_stats[f"{prefix}_mean"] = scaler.mean_
        scaler_stats[f"{prefix}_scale"] = scaler.scale_
        scaler_stats[f"{prefix}_var"] = scaler.var_
    return (
        np.concatenate(train_blocks, axis=1).astype("float32"),
        np.concatenate(val_blocks, axis=1).astype("float32"),
        {
            "scalers": scaler_metadata,
            "fusion_feature_policy": "scaled backbone blocks concatenated before metadata append",
        },
        scaler_stats,
    )


def _backbones_for_condition(condition: str) -> list[str]:
    if condition == "metadata_only":
        return []
    if condition == "ft_vit_swin_concat_plus_metadata":
        return ["vit_b16", "swin_tiny"]
    if condition == "ft_vit_swin_beit_concat_plus_metadata":
        return ["vit_b16", "swin_tiny", "beit_base"]
    if condition == "ft_vit_plus_metadata":
        return ["vit_b16"]
    raise ValueError(f"Unsupported E3c condition: {condition}")


def _labels_from_split(frame: pd.DataFrame, class_names: list[str]) -> torch.Tensor:
    label_to_index = {label: index for index, label in enumerate(class_names)}
    return torch.tensor([label_to_index[str(label)] for label in frame["label"]], dtype=torch.long)


def _reference_from_split(frame: pd.DataFrame, labels: torch.Tensor) -> SimpleNamespace:
    return SimpleNamespace(
        sample_ids=frame["sample_id"].astype(str).tolist(),
        image_ids=frame["image_id"].astype(str).tolist(),
        lesion_ids=frame["lesion_id"].astype(str).tolist(),
        label_names=frame["label"].astype(str).tolist(),
        split_names=frame["split"].astype(str).tolist(),
        labels=labels,
    )


def _class_weights(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    cache = SimpleNamespace(labels=labels)
    return class_weights_from_cache(cache, num_classes)


def _resolve_device(value: str | None) -> torch.device:
    requested = str(value or "cpu")
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


def _assert_shape(name: str, values: np.ndarray, expected_rows: int) -> None:
    if int(values.shape[0]) != expected_rows:
        raise ValueError(f"{name} has {int(values.shape[0])} rows; expected {expected_rows}.")


def _assert_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf values.")


if __name__ == "__main__":
    main()
