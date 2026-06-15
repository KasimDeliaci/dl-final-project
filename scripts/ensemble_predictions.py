"""Build validation-only prediction ensembles from existing run prediction dumps."""

from __future__ import annotations

import argparse
import json
import os
from itertools import product
from pathlib import Path
from typing import Any

os.environ["MPLBACKEND"] = "Agg"

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dl_final.evaluation.metrics import compute_classification_metrics, per_class_frame
from dl_final.evaluation.reports import save_confusion_matrix_plot

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "nv", "mel", "vasc"]
PROBABILITY_COLUMNS = [f"prob_{label}" for label in CLASS_NAMES]
ALIGNMENT_COLUMNS = ["sample_id", "image_id", "lesion_id", "split", "true_label"]
EXPECTED_VALIDATION_ROWS = 1504
SEEDS = (7, 13, 42, 101, 202)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--output-run-dir", default="artifacts/runs/e3g_prediction_ensemble")
    parser.add_argument("--tables-dir", default="artifacts/report_assets/tables")
    parser.add_argument("--figures-dir", default="artifacts/report_assets/figures")
    parser.add_argument(
        "--e3d-summary",
        default="artifacts/report_assets/tables/e3d_metadata_fusion_operator_summary.csv",
    )
    parser.add_argument(
        "--e3f-summary",
        default="artifacts/report_assets/tables/e3f_mixed_adaptation_summary.csv",
    )
    parser.add_argument("--skip-weighted-diagnostics", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root)
    output_run_dir = Path(args.output_run_dir)
    output_run_dir.mkdir(parents=True, exist_ok=True)
    table_root = Path(args.tables_dir)
    figure_root = Path(args.figures_dir)
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    family_specs = {
        "e3d_film": {
            "experiment_id": "E3d",
            "condition": "triple_metadata_film",
            "run_name_contains": "e3d_metadata_fusion",
        },
        "e3d_gated": {
            "experiment_id": "E3d",
            "condition": "triple_metadata_gated_backbone",
            "run_name_contains": "e3d_metadata_fusion",
        },
        "e3f_gated": {
            "experiment_id": "E3f",
            "condition": "triple_metadata_gated_backbone",
            "run_name_contains": "e3f_mixed_adaptation",
        },
    }
    family_predictions: dict[str, FamilyPrediction] = {}
    member_rows: list[dict[str, Any]] = []
    reference: pd.DataFrame | None = None
    for family_id, spec in family_specs.items():
        members = load_family_members(run_root, family_id, spec)
        if len(members) != len(SEEDS):
            raise ValueError(f"{family_id} expected {len(SEEDS)} members, found {len(members)}.")
        for member in members:
            reference = verify_prediction_frame(member.predictions, reference, member.run_id)
            member_rows.append(member.to_row())
        family_predictions[family_id] = average_family_predictions(family_id, members)

    if reference is None:
        raise ValueError("No ensemble members found.")

    ensemble_specs = primary_ensemble_specs()
    if not args.skip_weighted_diagnostics:
        ensemble_specs.extend(weighted_diagnostic_specs())

    results: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []
    confusion_matrices: dict[str, list[list[int]]] = {}
    ensemble_predictions: dict[str, pd.DataFrame] = {}
    for spec in ensemble_specs:
        frame, row, metrics = evaluate_ensemble(spec, family_predictions, reference)
        results.append(row)
        per_class_frames.append(
            per_class_frame(metrics, backbone=spec["ensemble_id"]).assign(
                ensemble_id=spec["ensemble_id"],
                ensemble_type=spec["ensemble_type"],
            )
        )
        confusion_matrices[spec["ensemble_id"]] = metrics["confusion_matrix"]
        ensemble_predictions[spec["ensemble_id"]] = frame
        ensemble_path = output_run_dir / f"ensemble_predictions_{spec['ensemble_id']}.csv"
        frame.to_csv(ensemble_path, index=False)

    results_frame = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    per_class = pd.concat(per_class_frames, ignore_index=True)
    members = pd.DataFrame(member_rows).sort_values(["family_id", "seed"])
    error_overlap = build_error_overlap_summary(family_predictions, reference)
    corrected_broken = build_corrected_broken_summary(results_frame, ensemble_predictions)
    control_comparison = build_control_comparison(
        results_frame,
        e3d_summary_path=Path(args.e3d_summary),
        e3f_summary_path=Path(args.e3f_summary),
    )

    run_config = {
        "experiment_id": "E3g",
        "selection_metric": "validation_macro_f1",
        "test_policy": "not_loaded_or_used_in_e3g",
        "class_names": CLASS_NAMES,
        "alignment_columns": ALIGNMENT_COLUMNS,
        "probability_columns": PROBABILITY_COLUMNS,
        "primary_policy": "equal-weight probability averaging over seed-averaged families",
        "weighted_diagnostics": not args.skip_weighted_diagnostics,
        "seeds": list(SEEDS),
    }
    (output_run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    members.to_csv(output_run_dir / "ensemble_members.csv", index=False)
    results_frame.to_csv(output_run_dir / "ensemble_results.csv", index=False)
    per_class.to_csv(output_run_dir / "ensemble_per_class_metrics.csv", index=False)
    (output_run_dir / "ensemble_confusion_matrices.json").write_text(
        json.dumps(confusion_matrices, indent=2),
        encoding="utf-8",
    )
    error_overlap.to_csv(output_run_dir / "error_overlap_summary.csv", index=False)
    corrected_broken.to_csv(output_run_dir / "corrected_broken_summary.csv", index=False)

    results_frame.to_csv(table_root / "e3g_prediction_ensemble_results.csv", index=False)
    per_class.to_csv(table_root / "e3g_prediction_ensemble_per_class_metrics.csv", index=False)
    control_comparison.to_csv(
        table_root / "e3g_prediction_ensemble_vs_controls.csv",
        index=False,
    )
    error_overlap.to_csv(table_root / "e3g_prediction_ensemble_error_overlap.csv", index=False)
    corrected_broken.to_csv(
        table_root / "e3g_prediction_ensemble_corrected_broken.csv",
        index=False,
    )
    save_results_plot(control_comparison, figure_root / "e3g_prediction_ensemble_macro_f1.png")
    save_per_class_plot(
        per_class,
        figure_root / "e3g_prediction_ensemble_per_class_f1.png",
    )
    save_representative_confusion_plots(results_frame, confusion_matrices, figure_root)
    save_error_overlap_plot(
        error_overlap,
        figure_root / "e3g_prediction_ensemble_error_overlap.png",
    )
    print(f"Wrote E3g ensemble run: {output_run_dir}")
    print(results_frame.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


class PredictionMember:
    def __init__(
        self,
        *,
        family_id: str,
        run_id: str,
        seed: int,
        config_path: Path,
        predictions: pd.DataFrame,
    ) -> None:
        self.family_id = family_id
        self.run_id = run_id
        self.seed = seed
        self.config_path = config_path
        self.predictions = predictions

    def to_row(self) -> dict[str, Any]:
        return {
            "family_id": self.family_id,
            "run_id": self.run_id,
            "seed": self.seed,
            "run_dir": str(self.config_path.parent),
            "prediction_rows": int(len(self.predictions)),
        }


class FamilyPrediction:
    def __init__(
        self,
        *,
        family_id: str,
        probabilities: np.ndarray,
        member_run_ids: list[str],
    ) -> None:
        self.family_id = family_id
        self.probabilities = probabilities
        self.member_run_ids = member_run_ids


def load_family_members(
    run_root: Path,
    family_id: str,
    spec: dict[str, str],
) -> list[PredictionMember]:
    members: list[PredictionMember] = []
    for config_path in sorted(run_root.glob("*/run_config.json")):
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("experiment_id") != spec["experiment_id"]:
            continue
        if config.get("condition") != spec["condition"]:
            continue
        if spec["run_name_contains"] not in config_path.parent.name:
            continue
        predictions_path = config_path.parent / "predictions.csv"
        if not predictions_path.exists():
            raise FileNotFoundError(f"Missing predictions for {config_path.parent}")
        predictions = pd.read_csv(predictions_path)
        members.append(
            PredictionMember(
                family_id=family_id,
                run_id=str(config["run_id"]),
                seed=int(config["seed"]),
                config_path=config_path,
                predictions=predictions,
            )
        )
    seeds = sorted(member.seed for member in members)
    if seeds != list(SEEDS):
        raise ValueError(f"{family_id} seeds are {seeds}; expected {list(SEEDS)}.")
    return sorted(members, key=lambda member: member.seed)


def verify_prediction_frame(
    frame: pd.DataFrame,
    reference: pd.DataFrame | None,
    run_id: str,
) -> pd.DataFrame:
    missing = [column for column in ALIGNMENT_COLUMNS + PROBABILITY_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"{run_id} is missing columns: {missing}")
    if len(frame) != EXPECTED_VALIDATION_ROWS:
        raise ValueError(f"{run_id} has {len(frame)} rows; expected {EXPECTED_VALIDATION_ROWS}.")
    if set(frame["split"].astype(str).unique()) != {"val"}:
        raise ValueError(f"{run_id} contains non-validation rows.")
    probabilities = frame[PROBABILITY_COLUMNS].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all():
        raise ValueError(f"{run_id} contains NaN or Inf probabilities.")
    probability_sums = probabilities.sum(axis=1)
    if not np.allclose(probability_sums, 1.0, atol=1e-4):
        raise ValueError(f"{run_id} probability rows do not sum to 1.")
    current = frame[ALIGNMENT_COLUMNS].reset_index(drop=True)
    if reference is None:
        return current
    if not current.equals(reference):
        raise ValueError(f"{run_id} row alignment does not match reference.")
    return reference


def average_family_predictions(
    family_id: str,
    members: list[PredictionMember],
) -> FamilyPrediction:
    stacked = np.stack(
        [member.predictions[PROBABILITY_COLUMNS].to_numpy(dtype=float) for member in members],
        axis=0,
    )
    probabilities = stacked.mean(axis=0)
    return FamilyPrediction(
        family_id=family_id,
        probabilities=probabilities,
        member_run_ids=[member.run_id for member in members],
    )


def primary_ensemble_specs() -> list[dict[str, Any]]:
    return [
        {
            "ensemble_id": "e3d_film_seed_avg",
            "ensemble_type": "primary_equal_weight",
            "family_weights": {"e3d_film": 1.0},
        },
        {
            "ensemble_id": "e3d_gated_seed_avg",
            "ensemble_type": "primary_equal_weight",
            "family_weights": {"e3d_gated": 1.0},
        },
        {
            "ensemble_id": "e3f_gated_seed_avg",
            "ensemble_type": "primary_equal_weight",
            "family_weights": {"e3f_gated": 1.0},
        },
        {
            "ensemble_id": "e3d_film_plus_e3f_gated_equal",
            "ensemble_type": "primary_equal_weight",
            "family_weights": {"e3d_film": 0.5, "e3f_gated": 0.5},
        },
        {
            "ensemble_id": "e3d_gated_plus_e3f_gated_equal",
            "ensemble_type": "primary_equal_weight",
            "family_weights": {"e3d_gated": 0.5, "e3f_gated": 0.5},
        },
        {
            "ensemble_id": "top3_family_equal",
            "ensemble_type": "primary_equal_weight",
            "family_weights": {
                "e3d_film": 1.0 / 3.0,
                "e3d_gated": 1.0 / 3.0,
                "e3f_gated": 1.0 / 3.0,
            },
        },
    ]


def weighted_diagnostic_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for e3d_weight in (0.25, 0.5, 0.75):
        e3f_weight = 1.0 - e3d_weight
        specs.append(
            {
                "ensemble_id": (
                    f"top2_grid_e3d_film_{_weight_token(e3d_weight)}_"
                    f"e3f_gated_{_weight_token(e3f_weight)}"
                ),
                "ensemble_type": "weighted_diagnostic",
                "family_weights": {"e3d_film": e3d_weight, "e3f_gated": e3f_weight},
            }
        )
    for e3d_film, e3d_gated, e3f_gated in product(
        (0.0, 0.25, 0.5, 0.75, 1.0),
        repeat=3,
    ):
        if not np.isclose(e3d_film + e3d_gated + e3f_gated, 1.0):
            continue
        weights = {
            "e3d_film": e3d_film,
            "e3d_gated": e3d_gated,
            "e3f_gated": e3f_gated,
        }
        if any(value == 0.0 for value in weights.values()):
            continue
        specs.append(
            {
                "ensemble_id": (
                    "top3_grid_"
                    f"film_{_weight_token(e3d_film)}_"
                    f"gated_{_weight_token(e3d_gated)}_"
                    f"e3f_{_weight_token(e3f_gated)}"
                ),
                "ensemble_type": "weighted_diagnostic",
                "family_weights": {key: value for key, value in weights.items() if value > 0.0},
            }
        )
    deduped: dict[str, dict[str, Any]] = {}
    for spec in specs:
        deduped[spec["ensemble_id"]] = spec
    return list(deduped.values())


def evaluate_ensemble(
    spec: dict[str, Any],
    family_predictions: dict[str, FamilyPrediction],
    reference: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    probabilities = np.zeros((len(reference), len(CLASS_NAMES)), dtype=float)
    for family_id, weight in spec["family_weights"].items():
        probabilities += float(weight) * family_predictions[family_id].probabilities
    probability_sums = probabilities.sum(axis=1, keepdims=True)
    probabilities = probabilities / probability_sums
    y_true = reference["true_label"].map({label: index for index, label in enumerate(CLASS_NAMES)})
    if y_true.isna().any():
        raise ValueError("Unknown true_label found in predictions.")
    y_true_array = y_true.to_numpy(dtype=int)
    y_pred = probabilities.argmax(axis=1)
    metrics = compute_classification_metrics(y_true_array, y_pred, CLASS_NAMES)
    output = reference.copy()
    output["pred_label"] = [CLASS_NAMES[index] for index in y_pred]
    output["correct"] = (y_true_array == y_pred).astype(bool)
    output["confidence"] = probabilities.max(axis=1)
    for index, label in enumerate(CLASS_NAMES):
        output[f"prob_{label}"] = probabilities[:, index]
    row = {
        "ensemble_id": spec["ensemble_id"],
        "ensemble_type": spec["ensemble_type"],
        "family_weights": json.dumps(spec["family_weights"], sort_keys=True),
        "member_families": ",".join(spec["family_weights"].keys()),
        "prediction_rows": int(len(output)),
        "selection_metric": "validation_macro_f1",
        "test_policy": "not_loaded_or_used_in_e3g",
        **{
            key: value
            for key, value in metrics.items()
            if key not in {"per_class", "confusion_matrix"}
        },
    }
    return output, row, metrics


def build_error_overlap_summary(
    families: dict[str, FamilyPrediction],
    reference: pd.DataFrame,
) -> pd.DataFrame:
    y_true = reference["true_label"].map({label: index for index, label in enumerate(CLASS_NAMES)})
    y_true_array = y_true.to_numpy(dtype=int)
    family_errors: dict[str, np.ndarray] = {}
    for family_id, family in families.items():
        y_pred = family.probabilities.argmax(axis=1)
        family_errors[family_id] = y_pred != y_true_array
    rows: list[dict[str, Any]] = []
    family_ids = sorted(family_errors)
    for left_index, left in enumerate(family_ids):
        for right in family_ids[left_index + 1 :]:
            left_errors = family_errors[left]
            right_errors = family_errors[right]
            both_wrong = left_errors & right_errors
            either_wrong = left_errors | right_errors
            rows.append(
                {
                    "left_family": left,
                    "right_family": right,
                    "left_errors": int(left_errors.sum()),
                    "right_errors": int(right_errors.sum()),
                    "both_wrong": int(both_wrong.sum()),
                    "either_wrong": int(either_wrong.sum()),
                    "error_jaccard": float(both_wrong.sum() / either_wrong.sum())
                    if either_wrong.sum()
                    else 0.0,
                    "left_correct_right_wrong": int((~left_errors & right_errors).sum()),
                    "left_wrong_right_correct": int((left_errors & ~right_errors).sum()),
                }
            )
    return pd.DataFrame(rows)


def build_corrected_broken_summary(
    results: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    primary = results[results["ensemble_type"] == "primary_equal_weight"]
    baseline_id = str(primary.sort_values("macro_f1", ascending=False).iloc[0]["ensemble_id"])
    baseline_correct = predictions[baseline_id]["correct"].to_numpy(dtype=bool)
    rows: list[dict[str, Any]] = []
    for ensemble_id, frame in predictions.items():
        correct = frame["correct"].to_numpy(dtype=bool)
        rows.append(
            {
                "baseline_ensemble_id": baseline_id,
                "ensemble_id": ensemble_id,
                "baseline_correct": int(baseline_correct.sum()),
                "ensemble_correct": int(correct.sum()),
                "corrected_vs_baseline": int((~baseline_correct & correct).sum()),
                "broken_vs_baseline": int((baseline_correct & ~correct).sum()),
                "unchanged_correct": int((baseline_correct & correct).sum()),
                "unchanged_wrong": int((~baseline_correct & ~correct).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(["corrected_vs_baseline", "ensemble_id"], ascending=False)


def build_control_comparison(
    results: pd.DataFrame,
    *,
    e3d_summary_path: Path,
    e3f_summary_path: Path,
) -> pd.DataFrame:
    rows = results.rename(columns={"macro_f1": "macro_f1_mean"}).copy()
    rows.insert(0, "result_source", rows["ensemble_type"])
    controls: list[pd.DataFrame] = [rows]
    if e3d_summary_path.exists():
        e3d = pd.read_csv(e3d_summary_path)
        e3d = e3d[
            e3d["condition"].isin(
                {
                    "triple_metadata_film",
                    "triple_metadata_gated_backbone",
                }
            )
        ].copy()
        e3d["ensemble_id"] = e3d["condition"]
        e3d.insert(0, "result_source", "e3d_control")
        controls.append(e3d)
    if e3f_summary_path.exists():
        e3f = pd.read_csv(e3f_summary_path)
        e3f = e3f[
            e3f["condition"].isin(
                {
                    "mixed_frozen_vit_ft_swin_beit_metadata_gated",
                    "mixed_frozen_vit_ft_swin_beit_metadata_film",
                    "mixed_frozen_vit_ft_swin_beit_concat",
                }
            )
        ].copy()
        e3f["ensemble_id"] = e3f["condition"]
        e3f.insert(0, "result_source", "e3f_control")
        controls.append(e3f)
    return pd.concat(controls, ignore_index=True, sort=False).sort_values(
        "macro_f1_mean",
        ascending=False,
    )


def save_results_plot(comparison: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    selected = comparison[
        comparison["result_source"].isin(
            {"primary_equal_weight", "weighted_diagnostic", "e3d_control", "e3f_control"}
        )
    ].copy()
    selected = selected.sort_values("macro_f1_mean", ascending=True).tail(16)
    labels = selected["ensemble_id"].astype(str).tolist()
    values = selected["macro_f1_mean"].astype(float).tolist()
    colors = [
        "#0f766e"
        if source == "primary_equal_weight"
        else "#9333ea"
        if source == "weighted_diagnostic"
        else "#64748b"
        for source in selected["result_source"].tolist()
    ]
    fig, ax = plt.subplots(figsize=(13, 8), constrained_layout=True)
    ax.barh(range(len(values)), values, color=colors)
    ax.set_yticks(range(len(values)), labels)
    ax.set_xlabel("Validation macro-F1")
    ax.set_title("E3g prediction ensembles vs E3d/E3f controls")
    for index, value in enumerate(values):
        ax.text(value, index, f" {value:.3f}", va="center", fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def save_per_class_plot(per_class: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    wanted = [
        "e3d_film_seed_avg",
        "e3f_gated_seed_avg",
        "e3d_film_plus_e3f_gated_equal",
        "top3_family_equal",
    ]
    frame = per_class[per_class["ensemble_id"].isin(wanted)].copy()
    pivot = frame.pivot(index="label", columns="ensemble_id", values="f1").loc[CLASS_NAMES]
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    x = np.arange(len(CLASS_NAMES))
    width = 0.18
    for index, column in enumerate(pivot.columns):
        ax.bar(x + (index - 1.5) * width, pivot[column].to_numpy(), width=width, label=column)
    ax.set_xticks(x, CLASS_NAMES)
    ax.set_ylabel("Validation F1")
    ax.set_title("E3g per-class F1 for representative primary ensembles")
    ax.legend(fontsize=8)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def save_representative_confusion_plots(
    results: pd.DataFrame,
    confusion_matrices: dict[str, list[list[int]]],
    figure_root: Path,
) -> None:
    primary = results[results["ensemble_type"] == "primary_equal_weight"]
    best_primary = str(primary.sort_values("macro_f1", ascending=False).iloc[0]["ensemble_id"])
    best_overall = str(results.sort_values("macro_f1", ascending=False).iloc[0]["ensemble_id"])
    for ensemble_id in sorted({best_primary, best_overall}):
        save_confusion_matrix_plot(
            confusion_matrices[ensemble_id],
            CLASS_NAMES,
            figure_root / f"e3g_{ensemble_id}_confusion_matrix.png",
            title=f"E3g {ensemble_id} validation confusion matrix",
        )


def save_error_overlap_plot(overlap: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    families = sorted(set(overlap["left_family"]).union(set(overlap["right_family"])))
    matrix = pd.DataFrame(np.eye(len(families)), index=families, columns=families)
    for _, row in overlap.iterrows():
        left = str(row["left_family"])
        right = str(row["right_family"])
        value = float(row["error_jaccard"])
        matrix.loc[left, right] = value
        matrix.loc[right, left] = value
    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    image = ax.imshow(matrix.to_numpy(dtype=float), cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xticks(range(len(families)), families, rotation=45, ha="right")
    ax.set_yticks(range(len(families)), families)
    ax.set_title("E3g family error-overlap Jaccard")
    for row in range(len(families)):
        for col in range(len(families)):
            ax.text(
                col,
                row,
                f"{matrix.iloc[row, col]:.2f}",
                ha="center",
                va="center",
                color="white" if matrix.iloc[row, col] > 0.5 else "black",
            )
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def _weight_token(value: float) -> str:
    return str(value).replace(".", "p")


if __name__ == "__main__":
    main()
