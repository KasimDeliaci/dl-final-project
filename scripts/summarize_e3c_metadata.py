"""Summarize E3c metadata-augmented cached-feature diagnostics."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--run-glob", default="*e3c_metadata*/run_config.json")
    parser.add_argument("--tables-dir", default="artifacts/report_assets/tables")
    parser.add_argument("--figures-dir", default="artifacts/report_assets/figures")
    parser.add_argument("--expected-runs", type=int, default=15)
    parser.add_argument(
        "--image-only-summary",
        default="artifacts/report_assets/tables/s4b_multiseed_cpu_diagnostic_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_configs = sorted(Path(args.run_root).glob(args.run_glob))
    rows = [_load_run_row(path) for path in run_configs]
    if args.expected_runs is not None and len(rows) != args.expected_runs:
        raise ValueError(f"Expected {args.expected_runs} E3c runs, found {len(rows)}.")
    results = pd.DataFrame(rows).sort_values(["condition", "seed"])

    table_root = Path(args.tables_dir)
    figure_root = Path(args.figures_dir)
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    results_path = table_root / "e3c_metadata_augmented_results.csv"
    results.to_csv(results_path, index=False)

    summary = summarize_results(results)
    summary_path = table_root / "e3c_metadata_augmented_summary.csv"
    summary.to_csv(summary_path, index=False)

    per_class = collect_per_class(run_configs)
    per_class_path = table_root / "e3c_metadata_augmented_per_class_metrics.csv"
    per_class.to_csv(per_class_path, index=False)

    per_class_delta = build_per_class_delta(per_class, Path(args.run_root))
    per_class_delta_path = table_root / "e3c_metadata_per_class_delta_vs_image_only.csv"
    per_class_delta.to_csv(per_class_delta_path, index=False)

    comparison = build_image_only_comparison(summary, Path(args.image_only_summary))
    comparison_path = table_root / "e3c_metadata_vs_image_only_validation.csv"
    comparison.to_csv(comparison_path, index=False)

    plot_path = figure_root / "e3c_metadata_augmented_macro_f1.png"
    save_summary_plot(comparison, plot_path)

    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {per_class_path}")
    print(f"Wrote {per_class_delta_path}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {plot_path}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("condition")
        .agg(
            n=("seed", "count"),
            seeds=("seed", _format_seeds),
            macro_f1_mean=("macro_f1", "mean"),
            macro_f1_std=("macro_f1", "std"),
            macro_f1_min=("macro_f1", "min"),
            macro_f1_max=("macro_f1", "max"),
            accuracy_mean=("accuracy", "mean"),
            weighted_f1_mean=("weighted_f1", "mean"),
            best_epoch_median=("best_epoch", "median"),
            runtime_seconds_mean=("runtime_seconds", "mean"),
            metadata_feature_dim=("metadata_feature_dim", "first"),
            image_feature_dim=("image_feature_dim", "first"),
        )
        .reset_index()
        .sort_values("macro_f1_mean", ascending=False)
    )


def collect_per_class(run_configs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for config_path in run_configs:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        per_class_path = config_path.parent / "per_class_metrics.csv"
        if per_class_path.exists():
            frames.append(
                pd.read_csv(per_class_path).assign(
                    run_id=config["run_id"],
                    condition=config["condition"],
                    seed=int(config["seed"]),
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_per_class_delta(metadata_per_class: pd.DataFrame, run_root: Path) -> pd.DataFrame:
    if metadata_per_class.empty:
        return pd.DataFrame()
    image_only = collect_image_only_e3b_per_class(run_root)
    if image_only.empty:
        return pd.DataFrame()
    pairs = {
        "ft_vit_swin_concat_plus_metadata": "finetuned_vit_swin_concat",
        "ft_vit_swin_beit_concat_plus_metadata": "finetuned_vit_swin_beit_concat",
    }
    metadata_summary = (
        metadata_per_class.groupby(["condition", "label"])
        .agg(metadata_f1_mean=("f1", "mean"), metadata_f1_std=("f1", "std"))
        .reset_index()
    )
    image_summary = (
        image_only.groupby(["condition", "label"])
        .agg(
            image_only_f1_mean=("f1", "mean"),
            image_only_f1_std=("f1", "std"),
            support=("support", "first"),
        )
        .reset_index()
    )
    rows: list[pd.DataFrame] = []
    for metadata_condition, image_condition in pairs.items():
        left = metadata_summary[metadata_summary["condition"] == metadata_condition]
        right = image_summary[image_summary["condition"] == image_condition]
        merged = left.merge(right, on="label", how="inner", suffixes=("", "_image"))
        merged["metadata_condition"] = metadata_condition
        merged["image_only_condition"] = image_condition
        merged["delta_f1_mean"] = merged["metadata_f1_mean"] - merged["image_only_f1_mean"]
        rows.append(merged)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)[
        [
            "metadata_condition",
            "image_only_condition",
            "label",
            "support",
            "image_only_f1_mean",
            "metadata_f1_mean",
            "delta_f1_mean",
            "image_only_f1_std",
            "metadata_f1_std",
        ]
    ]


def collect_image_only_e3b_per_class(run_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for config_path in sorted(run_root.glob("*s4b_multiseed_cpu*/run_config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        condition = _image_only_condition(config)
        if condition is None:
            continue
        per_class_path = config_path.parent / "per_class_metrics.csv"
        if per_class_path.exists():
            frames.append(pd.read_csv(per_class_path).assign(condition=condition))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_image_only_comparison(
    summary: pd.DataFrame,
    image_only_summary_path: Path,
) -> pd.DataFrame:
    metadata_rows = summary.copy()
    metadata_rows.insert(0, "result_source", "metadata_augmented")
    if not image_only_summary_path.exists():
        return metadata_rows
    image_only = pd.read_csv(image_only_summary_path)
    wanted = {
        "finetuned_vit_swin_concat",
        "finetuned_vit_swin_beit_concat",
        "finetuned_vit_single",
    }
    image_only = image_only[image_only["condition"].isin(wanted)].copy()
    image_only.insert(0, "result_source", "image_only_e3b")
    return pd.concat([metadata_rows, image_only], ignore_index=True, sort=False).sort_values(
        "macro_f1_mean",
        ascending=False,
    )


def save_summary_plot(comparison: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    ordered = comparison.sort_values("macro_f1_mean", ascending=False)
    labels = ordered["condition"].astype(str).tolist()
    values = ordered["macro_f1_mean"].astype(float).tolist()
    errors = ordered["macro_f1_std"].fillna(0.0).astype(float).tolist()
    colors = [
        "#0f766e" if source == "metadata_augmented" else "#64748b"
        for source in ordered["result_source"].tolist()
    ]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax.bar(range(len(values)), values, yerr=errors, color=colors, capsize=4)
    ax.set_xticks(range(len(values)), labels, rotation=45, ha="right")
    ax.set_ylabel("Validation macro-F1 mean over seeds")
    ax.set_title("E3c metadata-augmented features vs image-only controls")
    for index, value in enumerate(values):
        ax.text(index, value, f"{value:.3f}", ha="center", va="bottom", fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _load_run_row(run_config_path: Path) -> dict[str, Any]:
    run_dir = run_config_path.parent
    config = json.loads(run_config_path.read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    runtime_path = run_dir / "runtime_metadata.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    predictions_path = run_dir / "predictions.csv"
    prediction_rows = len(pd.read_csv(predictions_path)) if predictions_path.exists() else None
    return {
        "condition": config["condition"],
        "run_id": config["run_id"],
        "seed": int(config["seed"]),
        "experiment_id": config.get("experiment_id"),
        "feature_source": config.get("feature_source"),
        "backbone": config.get("backbone"),
        "backbones": "+".join(config.get("backbones", [])),
        "fusion_method": config.get("fusion_method"),
        "feature_dim": config.get("feature_dim"),
        "image_feature_dim": config.get("image_feature_dim"),
        "metadata_feature_dim": config.get("metadata_feature_dim"),
        "hidden_dims": str(config.get("hidden_dims")),
        "dropout": config.get("dropout"),
        "learning_rate": config.get("learning_rate"),
        "weight_decay": config.get("weight_decay"),
        "epochs": config.get("epochs"),
        "early_stopping_patience": config.get("early_stopping_patience"),
        "device": runtime.get("device"),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "best_epoch": metrics.get("best_epoch"),
        "runtime_seconds": config.get("runtime_seconds"),
        "prediction_rows": prediction_rows,
        "test_policy": config.get("test_policy"),
        "run_dir": str(run_dir),
    }


def _image_only_condition(config: dict[str, Any]) -> str | None:
    backbones = "+".join(config.get("backbones", [config.get("backbone", "")]))
    if (
        config.get("feature_source") == "finetuned"
        and config.get("fusion_method") == "concat"
        and backbones == "vit_b16+swin_tiny+beit_base"
    ):
        return "finetuned_vit_swin_beit_concat"
    if (
        config.get("feature_source") == "finetuned"
        and config.get("fusion_method") == "concat"
        and backbones == "vit_b16+swin_tiny"
    ):
        return "finetuned_vit_swin_concat"
    return None


def _format_seeds(seeds: pd.Series) -> str:
    return ",".join(str(int(seed)) for seed in sorted(seeds))


if __name__ == "__main__":
    main()
