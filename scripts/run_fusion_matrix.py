"""Run Sprint 3 frozen transformer feature-fusion experiments."""

from __future__ import annotations

import argparse
import json
import os
import random
from collections.abc import Sequence
from datetime import datetime
from itertools import combinations
from pathlib import Path
from time import perf_counter
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
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
from dl_final.models.backbones import (
    backbone_alias,
    expected_feature_dim,
    supported_backbones,
)
from dl_final.models.fusion import (
    WeightedLearnedFusionMLP,
    WeightedPCAFusionMLP,
    expected_concat_dim,
)
from dl_final.models.mlp import FeatureMLP
from dl_final.training.loops import evaluate_model, train_mlp_model
from dl_final.training.optim import build_optimizer

FUSION_METHODS = ("concat", "weighted_learned_512", "weighted_pca_384")
PCA_OUTPUT_DIM = 384
WEIGHTED_LEARNED_PROJECTION_DIM = 512
EXPECTED_SPLIT_COUNTS = {"train": 7008, "val": 1504}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--feature-root", default="artifacts/features")
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--tables-dir", default="artifacts/report_assets/tables")
    parser.add_argument("--figures-dir", default="artifacts/report_assets/figures")
    parser.add_argument("--feature-source", default="frozen")
    parser.add_argument("--backbones", nargs="+", default=supported_backbones())
    parser.add_argument(
        "--only-combination",
        nargs="+",
        default=None,
        help="Run exactly this backbone combination instead of expanding all pairs/triples.",
    )
    parser.add_argument("--fusion-methods", nargs="+", choices=FUSION_METHODS, default=None)
    parser.add_argument("--max-runs", type=int, default=None)
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
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--experiment-id", default=None)
    parser.add_argument("--test-policy", default=None)
    parser.add_argument("--skip-export", action="store_true")
    return parser.parse_args()


def expand_fusion_run_matrix(
    *,
    fusion_methods: Sequence[str] | None = None,
    backbones: Sequence[str] | None = None,
    only_combination: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Expand pairwise and three-backbone fusion runs for the selected methods."""

    selected_backbones = list(backbones or supported_backbones())
    selected_methods = list(fusion_methods or FUSION_METHODS)
    if only_combination is not None:
        combo = list(only_combination)
        if len(combo) < 2:
            raise ValueError("--only-combination requires at least two backbones.")
        return [
            {
                "backbones": combo,
                "fusion_method": method,
                "fusion_input_dim": expected_concat_dim(combo),
            }
            for method in selected_methods
        ]
    runs: list[dict[str, Any]] = []
    for size in range(2, len(selected_backbones) + 1):
        for combo in combinations(selected_backbones, size):
            for method in selected_methods:
                runs.append(
                    {
                        "backbones": list(combo),
                        "fusion_method": method,
                        "fusion_input_dim": expected_concat_dim(combo),
                    }
                )
    return runs


def main() -> None:
    args = parse_args()
    dataset_config = load_dataset_config(args.dataset_config)["dataset"]
    seed = int(args.seed if args.seed is not None else dataset_config.get("seed", 42))
    _seed_everything(seed)
    device = _resolve_device(args.device)
    run_specs = expand_fusion_run_matrix(
        fusion_methods=args.fusion_methods,
        backbones=args.backbones,
        only_combination=args.only_combination,
    )
    if args.max_runs is not None:
        run_specs = run_specs[: args.max_runs]
    if not run_specs:
        raise ValueError("No fusion runs selected.")

    completed: list[dict[str, Any]] = []
    for spec in run_specs:
        print(f"Running Sprint 3 fusion: {'+'.join(spec['backbones'])} / {spec['fusion_method']}")
        run_dir = run_fusion_experiment(
            spec,
            args=args,
            dataset_config=dataset_config,
            seed=seed,
            device=device,
        )
        completed.append({**spec, "run_dir": str(run_dir)})

    manifest = {
        "experiment_id": _experiment_id_for_source(args.feature_source, args.experiment_id),
        "feature_source": args.feature_source,
        "seed": seed,
        "selection_metric": "validation_macro_f1",
        "test_policy": _test_policy_for_source(args.feature_source, args.test_policy),
        "completed_runs": completed,
    }
    manifest_name = f"{args.feature_source}_fusion_manifest.json"
    manifest_path = Path(args.run_root) / manifest_name
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if not args.skip_export:
        exported = export_fusion_report_assets(
            args.run_root,
            args.tables_dir,
            args.figures_dir,
            feature_source=args.feature_source,
        )
        print("Exported report assets:")
        for name, path in exported.items():
            print(f"  {name}: {path}")
    print(f"Wrote fusion manifest: {manifest_path}")


def run_fusion_experiment(
    spec: dict[str, Any],
    *,
    args: argparse.Namespace,
    dataset_config: dict[str, Any],
    seed: int,
    device: torch.device,
) -> Path:
    started = perf_counter()
    class_names = list(dataset_config["class_names"])
    backbones = list(spec["backbones"])
    fusion_method = str(spec["fusion_method"])
    train_caches, val_caches = load_aligned_caches(
        feature_root=args.feature_root,
        dataset_name=str(dataset_config["name"]),
        feature_source=args.feature_source,
        backbones=backbones,
        splits_dir=Path(dataset_config["splits_dir"]),
    )

    train_features, val_features, preprocessing_metadata, scaler_stats = build_fusion_features(
        train_caches,
        val_caches,
        fusion_method=fusion_method,
        seed=seed,
    )
    _assert_finite("train fusion features", train_features)
    _assert_finite("validation fusion features", val_features)

    train_dataset = TensorDataset(torch.from_numpy(train_features), train_caches[0].labels.long())
    val_dataset = TensorDataset(torch.from_numpy(val_features), val_caches[0].labels.long())
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    input_dims = [expected_feature_dim(backbone) for backbone in backbones]
    if fusion_method == "concat":
        model: nn.Module = FeatureMLP(
            input_dim=train_features.shape[1],
            num_classes=len(class_names),
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
        )
        fusion_output_dim = int(train_features.shape[1])
        projection_dim = None
    elif fusion_method == "weighted_learned_512":
        model = WeightedLearnedFusionMLP(
            input_dims=input_dims,
            num_classes=len(class_names),
            projection_dim=WEIGHTED_LEARNED_PROJECTION_DIM,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
        )
        fusion_output_dim = WEIGHTED_LEARNED_PROJECTION_DIM
        projection_dim = WEIGHTED_LEARNED_PROJECTION_DIM
    elif fusion_method == "weighted_pca_384":
        model = WeightedPCAFusionMLP(
            num_backbones=len(backbones),
            feature_dim=PCA_OUTPUT_DIM,
            num_classes=len(class_names),
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
        )
        fusion_output_dim = PCA_OUTPUT_DIM
        projection_dim = PCA_OUTPUT_DIM
    else:
        raise ValueError(f"Unsupported fusion method: {fusion_method}")

    weights = (
        class_weights_from_cache(train_caches[0], len(class_names)).to(device)
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
    combo_alias = "-".join(backbone_alias(backbone) for backbone in backbones)
    method_alias = _fusion_method_alias(fusion_method)
    tag = f"_{args.run_tag}" if args.run_tag else ""
    run_id = (
        f"{timestamp}_s3_{args.feature_source}_{combo_alias}_"
        f"{method_alias}_mlp{tag}_seed{seed}"
    )
    run_dir = Path(args.run_root) / run_id
    runtime_seconds = round(perf_counter() - started, 4)
    run_config = {
        "run_id": run_id,
        "experiment_id": _experiment_id_for_source(args.feature_source, args.experiment_id),
        "seed": seed,
        "dataset": dataset_config["name"],
        "feature_source": args.feature_source,
        "backbone": "+".join(backbones),
        "backbones": backbones,
        "backbone_count": len(backbones),
        "fusion_method": fusion_method,
        "feature_dim": fusion_output_dim,
        "fusion_input_dim": expected_concat_dim(backbones),
        "fusion_output_dim": fusion_output_dim,
        "input_dims": dict(zip(backbones, input_dims, strict=True)),
        "projection_dim": projection_dim,
        "pca_output_dim": PCA_OUTPUT_DIM if fusion_method == "weighted_pca_384" else None,
        "concat_pca_policy": "no_pca_for_concat",
        "scaler": "StandardScaler fit separately per backbone block on train cache only",
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
        "test_policy": _test_policy_for_source(args.feature_source, args.test_policy),
        "feature_cache_dirs": [
            str(Path(args.feature_root) / str(dataset_config["name"]) / args.feature_source / b)
            for b in backbones
        ],
        "runtime_seconds": runtime_seconds,
    }
    write_run_artifacts(
        run_dir,
        run_config=run_config,
        history=history,
        metrics=metrics,
        predictions=prediction_frame(
            cache=val_caches[0],
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            class_names=class_names,
        ),
        class_names=class_names,
        backbone="+".join(backbones),
    )
    torch.save(model.state_dict(), run_dir / "model.pt")
    np.savez(run_dir / "scaler_stats.npz", **scaler_stats)
    (run_dir / "preprocessing_metadata.json").write_text(
        json.dumps(preprocessing_metadata, indent=2),
        encoding="utf-8",
    )
    fusion_metadata = {
        "fusion_method": fusion_method,
        "backbones": backbones,
        "input_dims": dict(zip(backbones, input_dims, strict=True)),
        "fusion_input_dim": expected_concat_dim(backbones),
        "fusion_output_dim": fusion_output_dim,
        "selection_metric": "validation_macro_f1",
        "test_policy": _test_policy_for_source(args.feature_source, args.test_policy),
    }
    fusion_weights = extract_fusion_weights(model, backbones)
    if fusion_weights:
        fusion_metadata["learned_weights"] = fusion_weights
        pd.DataFrame(fusion_weights).to_csv(run_dir / "fusion_weights.csv", index=False)
    (run_dir / "fusion_metadata.json").write_text(
        json.dumps(fusion_metadata, indent=2),
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
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote Sprint 3 validation run: {run_dir}")
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
    """Verify that feature caches have identical sample, label, lesion, and split order."""

    if len(caches) < 2:
        return
    reference = caches[0]
    checks = {
        "sample_id order": "sample_ids",
        "image_id order": "image_ids",
        "lesion_id order": "lesion_ids",
        "label order": "label_names",
        "split order": "split_names",
    }
    for cache in caches[1:]:
        for label, attr in checks.items():
            if getattr(cache, attr) != getattr(reference, attr):
                raise ValueError(
                    f"Cache {label} mismatch between {reference.backbone} and {cache.backbone}."
                )
        if not torch.equal(cache.labels, reference.labels):
            raise ValueError(
                f"Cache label index order mismatch between {reference.backbone} and "
                f"{cache.backbone}."
            )


def build_fusion_features(
    train_caches: Sequence[FeatureCache],
    val_caches: Sequence[FeatureCache],
    *,
    fusion_method: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    train_scaled, val_scaled, scaler_metadata, scaler_stats = scale_feature_blocks(
        train_caches,
        val_caches,
    )
    if fusion_method in {"concat", "weighted_learned_512"}:
        train_features = np.concatenate(train_scaled, axis=1).astype("float32")
        val_features = np.concatenate(val_scaled, axis=1).astype("float32")
        metadata = {
            "scalers": scaler_metadata,
            "pca": None,
            "fusion_feature_policy": "scaled blocks concatenated without PCA",
        }
        return train_features, val_features, metadata, scaler_stats
    if fusion_method == "weighted_pca_384":
        train_blocks: list[np.ndarray] = []
        val_blocks: list[np.ndarray] = []
        pca_metadata: list[dict[str, Any]] = []
        for index, (train_block, val_block, cache) in enumerate(
            zip(train_scaled, val_scaled, train_caches, strict=True)
        ):
            train_pca, val_pca, block_metadata = fit_transform_pca_block(
                train_block,
                val_block,
                backbone=cache.backbone,
                output_dim=PCA_OUTPUT_DIM,
                seed=seed + index,
            )
            train_blocks.append(train_pca)
            val_blocks.append(val_pca)
            pca_metadata.append(block_metadata)
        train_features = np.concatenate(train_blocks, axis=1).astype("float32")
        val_features = np.concatenate(val_blocks, axis=1).astype("float32")
        metadata = {
            "scalers": scaler_metadata,
            "pca": pca_metadata,
            "fusion_feature_policy": "train-only PCA per scaled backbone block before weighted sum",
        }
        return train_features, val_features, metadata, scaler_stats
    raise ValueError(f"Unsupported fusion method: {fusion_method}")


def scale_feature_blocks(
    train_caches: Sequence[FeatureCache],
    val_caches: Sequence[FeatureCache],
) -> tuple[list[np.ndarray], list[np.ndarray], list[dict[str, Any]], dict[str, np.ndarray]]:
    train_scaled: list[np.ndarray] = []
    val_scaled: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    stats: dict[str, np.ndarray] = {}
    for train_cache, val_cache in zip(train_caches, val_caches, strict=True):
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_cache.features.numpy()).astype("float32")
        val_features = scaler.transform(val_cache.features.numpy()).astype("float32")
        train_scaled.append(train_features)
        val_scaled.append(val_features)
        metadata.append(
            {
                "backbone": train_cache.backbone,
                "fit_split": "train",
                "input_dim": train_cache.feature_dim,
                "train_rows": int(train_features.shape[0]),
                "validation_rows": int(val_features.shape[0]),
            }
        )
        prefix = train_cache.backbone
        stats[f"{prefix}_mean"] = scaler.mean_
        stats[f"{prefix}_scale"] = scaler.scale_
        stats[f"{prefix}_var"] = scaler.var_
    return train_scaled, val_scaled, metadata, stats


def fit_transform_pca_block(
    train_features: torch.Tensor | np.ndarray,
    val_features: torch.Tensor | np.ndarray,
    *,
    backbone: str,
    output_dim: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    train_array = _as_numpy(train_features)
    val_array = _as_numpy(val_features)
    if output_dim > min(train_array.shape):
        raise ValueError(
            f"PCA output_dim={output_dim} exceeds min(train shape)={min(train_array.shape)}."
        )
    solver = "full" if output_dim >= min(train_array.shape) else "randomized"
    pca = PCA(n_components=output_dim, svd_solver=solver, random_state=seed)
    train_pca = pca.fit_transform(train_array).astype("float32")
    val_pca = pca.transform(val_array).astype("float32")
    metadata = {
        "backbone": backbone,
        "fit_split": "train",
        "uses_labels": False,
        "input_dim": int(train_array.shape[1]),
        "output_dim": int(output_dim),
        "train_rows_fit": int(train_array.shape[0]),
        "validation_rows_transformed": int(val_array.shape[0]),
        "svd_solver": solver,
        "explained_variance_ratio_sum": float(pca.explained_variance_ratio_.sum()),
        "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
    }
    return train_pca, val_pca, metadata


def extract_fusion_weights(model: nn.Module, backbones: Sequence[str]) -> list[dict[str, Any]]:
    if not hasattr(model, "normalized_weights"):
        return []
    with torch.no_grad():
        weights = model.normalized_weights().detach().cpu().numpy()
    return [
        {"backbone": backbone, "weight": float(weight), "weight_sum": float(weights.sum())}
        for backbone, weight in zip(backbones, weights, strict=True)
    ]


def export_fusion_report_assets(
    run_root: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
    *,
    feature_source: str,
) -> dict[str, Path]:
    table_root = Path(tables_dir)
    figure_root = Path(figures_dir)
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    fusion_rows: list[dict[str, Any]] = []
    single_rows: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []
    weight_frames: list[pd.DataFrame] = []
    for config_path in sorted(Path(run_root).glob("*/run_config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("feature_source") != feature_source:
            continue
        if "_e2b_" in str(config.get("run_id", "")):
            continue
        metrics_path = config_path.parent / "metrics_summary.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
        row = {**config, **metrics}
        if config.get("fusion_method") != "none":
            fusion_rows.append(row)
            per_class_path = config_path.parent / "per_class_metrics.csv"
            if per_class_path.exists():
                per_class_frames.append(
                    pd.read_csv(per_class_path).assign(run_id=config["run_id"])
                )
            weights_path = config_path.parent / "fusion_weights.csv"
            if weights_path.exists():
                weight_frames.append(pd.read_csv(weights_path).assign(run_id=config["run_id"]))
        elif (
            config.get("experiment_id") == "E1"
            and config.get("backbone") in supported_backbones()
        ):
            single_rows.append(row)

    if not fusion_rows:
        raise FileNotFoundError("No Sprint 3 fusion run artifacts found.")

    fusion = pd.DataFrame(fusion_rows).sort_values("macro_f1", ascending=False)
    prefix = "frozen" if feature_source == "frozen" else feature_source
    fusion_results_path = table_root / f"{prefix}_fusion_results.csv"
    fusion.to_csv(fusion_results_path, index=False)

    per_class_path = table_root / f"{prefix}_fusion_per_class_metrics.csv"
    if per_class_frames:
        pd.concat(per_class_frames, ignore_index=True).to_csv(per_class_path, index=False)
    else:
        pd.DataFrame().to_csv(per_class_path, index=False)

    weight_summary_path = table_root / f"{prefix}_fusion_weight_summary.csv"
    if weight_frames:
        pd.concat(weight_frames, ignore_index=True).to_csv(weight_summary_path, index=False)
    else:
        pd.DataFrame().to_csv(weight_summary_path, index=False)

    comparison = fusion.copy()
    comparison.insert(0, "result_source", "fusion")
    if single_rows:
        singles = pd.DataFrame(single_rows)
        singles.insert(0, "result_source", "single_backbone")
        comparison = pd.concat([singles, comparison], ignore_index=True, sort=False)
    comparison = comparison.sort_values("macro_f1", ascending=False)
    comparison_path = table_root / f"{prefix}_fusion_vs_single_validation.csv"
    comparison.to_csv(comparison_path, index=False)

    plot_path = figure_root / f"{prefix}_fusion_macro_f1.png"
    save_fusion_macro_f1_plot(comparison, plot_path, feature_source=feature_source)
    return {
        "fusion_results": fusion_results_path,
        "fusion_per_class_metrics": per_class_path,
        "fusion_weight_summary": weight_summary_path,
        "fusion_vs_single_validation": comparison_path,
        "fusion_macro_f1_plot": plot_path,
    }


def save_fusion_macro_f1_plot(
    results: pd.DataFrame,
    output_path: str | Path,
    *,
    feature_source: str = "frozen",
) -> Path:
    output = Path(output_path)
    labels = []
    for _, row in results.head(16).iterrows():
        if row.get("result_source") == "single_backbone":
            labels.append(str(row.get("backbone")))
        else:
            labels.append(f"{row.get('backbone')} {row.get('fusion_method')}")
    values = results.head(16)["macro_f1"].astype(float).to_numpy()
    colors = [
        "#2563eb" if source == "fusion" else "#64748b"
        for source in results.head(16)["result_source"].tolist()
    ]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.bar(range(len(values)), values, color=colors)
    ax.axhline(0.6924, color="#dc2626", linestyle="--", linewidth=1.5, label="ViT baseline")
    ax.set_xticks(range(len(values)), labels, rotation=45, ha="right")
    ax.set_ylabel("Validation macro-F1")
    title_source = "Frozen" if feature_source == "frozen" else "Fine-tuned"
    ax.set_title(f"{title_source} feature fusion vs single-backbone validation macro-F1")
    ax.legend()
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _as_numpy(value: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype("float32")
    return value.astype("float32")


def _assert_finite(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or Inf values.")


def _fusion_method_alias(method: str) -> str:
    if method == "weighted_learned_512":
        return "weightedlearned512"
    if method == "weighted_pca_384":
        return "weightedpca384"
    return method


def _experiment_id_for_source(feature_source: str, override: str | None = None) -> str:
    if override:
        return override
    return "E3" if feature_source == "finetuned" else "E2"


def _test_policy_for_source(feature_source: str, override: str | None = None) -> str:
    if override:
        return override
    if feature_source.startswith("finetuned"):
        return "not_used_in_sprint4"
    return "not_used_in_sprint3"


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
