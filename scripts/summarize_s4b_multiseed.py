"""Summarize Sprint 4b/E3b cached-feature multi-seed diagnostic runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--run-glob", default="*s4b_multiseed_cpu*/run_config.json")
    parser.add_argument("--tables-dir", default="artifacts/report_assets/tables")
    parser.add_argument(
        "--results-name",
        default="s4b_multiseed_cpu_diagnostic_results.csv",
    )
    parser.add_argument(
        "--summary-name",
        default="s4b_multiseed_cpu_diagnostic_summary.csv",
    )
    parser.add_argument("--expected-runs", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [
        _load_run_row(path)
        for path in sorted(Path(args.run_root).glob(args.run_glob))
    ]
    if args.expected_runs is not None and len(rows) != args.expected_runs:
        raise ValueError(f"Expected {args.expected_runs} runs, found {len(rows)}.")

    results = pd.DataFrame(rows).sort_values(["condition", "seed"])
    output_dir = Path(args.tables_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / args.results_name
    results.to_csv(results_path, index=False)

    summary = (
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
        )
        .reset_index()
        .sort_values("macro_f1_mean", ascending=False)
    )
    summary_path = output_dir / args.summary_name
    summary.to_csv(summary_path, index=False)

    print(f"Wrote {results_path}")
    print(f"Wrote {summary_path}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def _load_run_row(run_config_path: Path) -> dict[str, Any]:
    run_dir = run_config_path.parent
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics_summary.json").read_text(encoding="utf-8"))
    runtime_path = run_dir / "runtime_metadata.json"
    runtime = json.loads(runtime_path.read_text(encoding="utf-8")) if runtime_path.exists() else {}
    backbones = "+".join(run_config.get("backbones", [run_config["backbone"]]))
    return {
        "condition": _condition_name(run_config, backbones),
        "run_id": run_config["run_id"],
        "seed": int(run_config["seed"]),
        "experiment_id": run_config.get("experiment_id"),
        "feature_source": run_config["feature_source"],
        "backbone": run_config["backbone"],
        "backbones": backbones,
        "fusion_method": run_config["fusion_method"],
        "hidden_dims": str(run_config.get("hidden_dims")),
        "dropout": run_config.get("dropout"),
        "learning_rate": run_config.get("learning_rate"),
        "weight_decay": run_config.get("weight_decay"),
        "epochs": run_config.get("epochs"),
        "early_stopping_patience": run_config.get("early_stopping_patience"),
        "device": runtime.get("device", "cpu-single-run"),
        "accuracy": metrics["accuracy"],
        "macro_precision": metrics["macro_precision"],
        "macro_recall": metrics["macro_recall"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "best_epoch": metrics.get("best_epoch"),
        "runtime_seconds": run_config.get("runtime_seconds"),
        "test_policy": run_config.get("test_policy"),
        "run_dir": str(run_dir),
    }


def _condition_name(run_config: dict[str, Any], backbones: str) -> str:
    feature_source = run_config["feature_source"]
    fusion_method = run_config["fusion_method"]
    backbone = run_config["backbone"]
    if (
        feature_source == "finetuned"
        and fusion_method == "concat"
        and backbones == "vit_b16+swin_tiny+beit_base"
    ):
        return "finetuned_vit_swin_beit_concat"
    if (
        feature_source == "finetuned"
        and fusion_method == "concat"
        and backbones == "vit_b16+swin_tiny"
    ):
        return "finetuned_vit_swin_concat"
    if feature_source == "finetuned" and fusion_method == "none" and backbone == "vit_b16":
        return "finetuned_vit_single"
    if (
        feature_source == "frozen"
        and fusion_method == "concat"
        and backbones == "vit_b16+swin_tiny"
    ):
        return "frozen_vit_swin_concat_deep_reg"
    return "other"


def _format_seeds(seeds: pd.Series) -> str:
    return ",".join(str(int(seed)) for seed in sorted(seeds))


if __name__ == "__main__":
    main()
