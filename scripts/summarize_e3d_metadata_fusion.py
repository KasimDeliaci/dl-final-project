"""Summarize E3d metadata fusion operator ablations."""

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
    parser.add_argument("--run-glob", default="*e3d_metadata_fusion*/run_config.json")
    parser.add_argument("--tables-dir", default="artifacts/report_assets/tables")
    parser.add_argument("--figures-dir", default="artifacts/report_assets/figures")
    parser.add_argument("--expected-runs", type=int, default=15)
    parser.add_argument(
        "--e3c-summary",
        default="artifacts/report_assets/tables/e3c_metadata_augmented_summary.csv",
    )
    parser.add_argument(
        "--e3b-summary",
        default="artifacts/report_assets/tables/s4b_multiseed_cpu_diagnostic_summary.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_configs = sorted(Path(args.run_root).glob(args.run_glob))
    rows = [_load_run_row(path) for path in run_configs]
    if args.expected_runs is not None and len(rows) != args.expected_runs:
        raise ValueError(f"Expected {args.expected_runs} E3d runs, found {len(rows)}.")
    results = pd.DataFrame(rows).sort_values(["condition", "seed"])

    table_root = Path(args.tables_dir)
    figure_root = Path(args.figures_dir)
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    results_path = table_root / "e3d_metadata_fusion_operator_results.csv"
    results.to_csv(results_path, index=False)

    summary = summarize_results(results)
    summary_path = table_root / "e3d_metadata_fusion_operator_summary.csv"
    summary.to_csv(summary_path, index=False)

    per_class = collect_per_class(run_configs)
    per_class_path = table_root / "e3d_metadata_fusion_operator_per_class_metrics.csv"
    per_class.to_csv(per_class_path, index=False)

    per_class_delta = build_per_class_delta_vs_e3c(
        per_class,
        Path(args.tables_dir) / "e3c_metadata_augmented_per_class_metrics.csv",
    )
    per_class_delta_path = table_root / "e3d_metadata_fusion_per_class_delta_vs_e3c.csv"
    per_class_delta.to_csv(per_class_delta_path, index=False)

    comparison = build_control_comparison(
        summary,
        e3c_summary_path=Path(args.e3c_summary),
        e3b_summary_path=Path(args.e3b_summary),
    )
    comparison_path = table_root / "e3d_metadata_fusion_vs_e3c_validation.csv"
    comparison.to_csv(comparison_path, index=False)

    gate_summary = collect_gate_summaries(run_configs)
    gate_summary_path = table_root / "e3d_metadata_gate_summary.csv"
    gate_summary.to_csv(gate_summary_path, index=False)

    plot_path = figure_root / "e3d_metadata_fusion_operator_macro_f1.png"
    save_summary_plot(comparison, plot_path)

    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {per_class_path}")
    print(f"Wrote {per_class_delta_path}")
    print(f"Wrote {comparison_path}")
    print(f"Wrote {gate_summary_path}")
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


def collect_gate_summaries(run_configs: list[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for config_path in run_configs:
        config = json.loads(config_path.read_text(encoding="utf-8"))
        gate_path = config_path.parent / "gate_summary.csv"
        if gate_path.exists():
            frames.append(
                pd.read_csv(gate_path).assign(
                    run_id=config["run_id"],
                    condition=config["condition"],
                    seed=int(config["seed"]),
                )
            )
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_per_class_delta_vs_e3c(
    e3d_per_class: pd.DataFrame,
    e3c_per_class_path: Path,
) -> pd.DataFrame:
    if e3d_per_class.empty or not e3c_per_class_path.exists():
        return pd.DataFrame()
    e3c = pd.read_csv(e3c_per_class_path)
    e3c = e3c[e3c["condition"] == "ft_vit_swin_beit_concat_plus_metadata"]
    e3c_summary = (
        e3c.groupby("label")
        .agg(e3c_raw_concat_f1_mean=("f1", "mean"), support=("support", "first"))
        .reset_index()
    )
    e3d_summary = (
        e3d_per_class.groupby(["condition", "label"])
        .agg(e3d_f1_mean=("f1", "mean"), e3d_f1_std=("f1", "std"))
        .reset_index()
    )
    merged = e3d_summary.merge(e3c_summary, on="label", how="inner")
    merged["delta_vs_e3c_raw_concat"] = (
        merged["e3d_f1_mean"] - merged["e3c_raw_concat_f1_mean"]
    )
    return merged[
        [
            "condition",
            "label",
            "support",
            "e3c_raw_concat_f1_mean",
            "e3d_f1_mean",
            "delta_vs_e3c_raw_concat",
            "e3d_f1_std",
        ]
    ].sort_values(["condition", "label"])


def build_control_comparison(
    summary: pd.DataFrame,
    *,
    e3c_summary_path: Path,
    e3b_summary_path: Path,
) -> pd.DataFrame:
    rows = summary.copy()
    rows.insert(0, "result_source", "e3d_operator")
    controls: list[pd.DataFrame] = [rows]
    if e3c_summary_path.exists():
        e3c = pd.read_csv(e3c_summary_path)
        e3c = e3c[
            e3c["condition"].isin(
                {
                    "ft_vit_swin_beit_concat_plus_metadata",
                    "metadata_only",
                }
            )
        ].copy()
        e3c.insert(0, "result_source", "e3c_control")
        controls.append(e3c)
    if e3b_summary_path.exists():
        e3b = pd.read_csv(e3b_summary_path)
        e3b = e3b[e3b["condition"] == "finetuned_vit_swin_beit_concat"].copy()
        e3b.insert(0, "result_source", "e3b_image_only_control")
        controls.append(e3b)
    return pd.concat(controls, ignore_index=True, sort=False).sort_values(
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
        "#7c3aed" if source == "e3d_operator" else "#64748b"
        for source in ordered["result_source"].tolist()
    ]
    fig, ax = plt.subplots(figsize=(13, 6), constrained_layout=True)
    ax.bar(range(len(values)), values, yerr=errors, color=colors, capsize=4)
    ax.set_xticks(range(len(values)), labels, rotation=45, ha="right")
    ax.set_ylabel("Validation macro-F1 mean over seeds")
    ax.set_title("E3d metadata fusion operators vs E3c/E3b controls")
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
        "metadata_fusion_operator": config.get("metadata_fusion_operator"),
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


def _format_seeds(seeds: pd.Series) -> str:
    return ",".join(str(int(seed)) for seed in sorted(seeds))


if __name__ == "__main__":
    main()
