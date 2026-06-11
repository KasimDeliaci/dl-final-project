"""Run artifact writers for Sprint 2 frozen single-backbone baselines."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dl_final.evaluation.metrics import per_class_frame


def write_run_artifacts(
    run_dir: str | Path,
    *,
    run_config: dict[str, Any],
    history: pd.DataFrame,
    metrics: dict[str, Any],
    predictions: pd.DataFrame,
    class_names: list[str],
    backbone: str,
) -> None:
    """Write the standard artifact bundle for one validation-selected MLP run."""

    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    metrics_row = {
        key: value
        for key, value in metrics.items()
        if key not in {"per_class", "confusion_matrix"}
    }
    pd.DataFrame([metrics_row]).to_csv(output_dir / "metrics_summary.csv", index=False)
    (output_dir / "metrics_summary.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    per_class_frame(metrics, backbone=backbone).to_csv(
        output_dir / "per_class_metrics.csv",
        index=False,
    )
    pd.DataFrame(metrics["confusion_matrix"], index=class_names, columns=class_names).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    predictions.to_csv(output_dir / "predictions.csv", index=False)
    history.to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "checkpoint_metadata.json").write_text(
        json.dumps(
            {
                "selection_metric": "validation_macro_f1",
                "best_epoch": metrics.get("best_epoch"),
                "best_validation_macro_f1": metrics.get("macro_f1"),
                "checkpoint_file": "model.pt",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    save_confusion_matrix_plot(
        metrics["confusion_matrix"],
        class_names,
        output_dir / "confusion_matrix.png",
        title=f"{backbone} validation confusion matrix",
    )
    save_training_curves(
        history,
        output_dir / "training_curves.png",
        title=f"{backbone} MLP training history",
    )
    _write_run_report_note(output_dir / "report_note.md", run_config, metrics)


def prediction_frame(
    *,
    cache,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    rows = {
        "sample_id": cache.sample_ids,
        "image_id": cache.image_ids,
        "lesion_id": cache.lesion_ids,
        "split": cache.split_names,
        "true_label": [class_names[int(index)] for index in y_true],
        "pred_label": [class_names[int(index)] for index in y_pred],
        "correct": (y_true == y_pred).astype(bool).tolist(),
        "confidence": probabilities.max(axis=1).tolist() if len(probabilities) else [],
    }
    for index, label in enumerate(class_names):
        rows[f"prob_{label}"] = probabilities[:, index].tolist()
    return pd.DataFrame(rows)


def export_single_backbone_report_assets(
    run_root: str | Path,
    tables_dir: str | Path,
    figures_dir: str | Path,
    *,
    feature_source: str = "frozen",
) -> dict[str, Path]:
    """Aggregate validation metrics from Sprint 2 single-backbone runs."""

    rows: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []
    for config_path in sorted(Path(run_root).glob("*/run_config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("feature_source") != feature_source or config.get("fusion_method") != "none":
            continue
        if "_e2b_" in str(config.get("run_id", "")):
            continue
        metrics_path = config_path.parent / "metrics_summary.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path).iloc[0].to_dict()
        rows.append({**config, **metrics})
        per_class_path = config_path.parent / "per_class_metrics.csv"
        if per_class_path.exists():
            per_class_frames.append(pd.read_csv(per_class_path).assign(run_id=config["run_id"]))

    if not rows:
        raise FileNotFoundError(f"No {feature_source} single-backbone run artifacts found.")

    table_root = Path(tables_dir)
    figure_root = Path(figures_dir)
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame(rows).sort_values("backbone")
    results_path = table_root / f"single_backbone_{feature_source}_results.csv"
    results.to_csv(results_path, index=False)

    per_class_path = table_root / f"single_backbone_{feature_source}_per_class_metrics.csv"
    if per_class_frames:
        pd.concat(per_class_frames, ignore_index=True).to_csv(per_class_path, index=False)
    else:
        pd.DataFrame().to_csv(per_class_path, index=False)

    plot_path = figure_root / f"{feature_source}_single_backbone_macro_f1.png"
    save_macro_f1_plot(results, plot_path)
    return {
        "results_table": results_path,
        "per_class_table": per_class_path,
        "macro_f1_plot": plot_path,
    }


def save_confusion_matrix_plot(
    matrix: list[list[int]],
    class_names: list[str],
    output_path: str | Path,
    *,
    title: str,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    values = np.asarray(matrix)
    fig, ax = plt.subplots(figsize=(7, 6), constrained_layout=True)
    image = ax.imshow(values, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            ax.text(col, row, str(values[row, col]), ha="center", va="center", fontsize=8)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_training_curves(history: pd.DataFrame, output_path: str | Path, *, title: str) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].plot(history["epoch"], history["train_loss"], label="train")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[1].plot(history["epoch"], history["val_macro_f1"], label="val macro-F1")
    axes[1].plot(history["epoch"], history["val_weighted_f1"], label="val weighted-F1")
    axes[1].set_title("Validation metrics")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def save_macro_f1_plot(results: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ordered = results.sort_values("macro_f1", ascending=False)
    ax.bar(ordered["backbone"], ordered["macro_f1"], color="#2563eb")
    ax.set_ylim(0, max(float(ordered["macro_f1"].max()) * 1.15, 0.05))
    ax.set_ylabel("Validation macro-F1")
    ax.set_title("Frozen single-backbone validation macro-F1")
    for index, value in enumerate(ordered["macro_f1"]):
        ax.text(index, float(value), f"{float(value):.3f}", ha="center", va="bottom", fontsize=9)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _write_run_report_note(path: Path, run_config: dict[str, Any], metrics: dict[str, Any]) -> None:
    result_line = (
        f"Validation macro-F1 = {metrics['macro_f1']:.4f}; "
        f"accuracy = {metrics['accuracy']:.4f}; "
        f"best epoch = {metrics.get('best_epoch')}."
    )
    text = f"""Question:
How strong is {run_config["backbone"]} as a frozen single-backbone feature extractor?

Recipe:
Frozen timm transformer features, train-only StandardScaler, class-weighted MLP,
validation macro-F1 checkpoint selection.

Fixed controls:
Canonical HAM10000 lesion-aware split, deterministic 224x224 ImageNet preprocessing,
no fusion, no fine-tuning, no test-set model selection.

Result:
{result_line}

Interpretation:
This run contributes to the Sprint 2 single-backbone representation baseline table.
Accuracy is secondary because HAM10000 is imbalanced.

Evidence strength:
Run artifacts include config, metrics summary, per-class metrics, validation predictions,
confusion matrix, and training history.

Report decision:
Use as a validation-selected frozen single-backbone baseline, not as a final test result.
"""
    path.write_text(text, encoding="utf-8")
