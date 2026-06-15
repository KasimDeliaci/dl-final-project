"""Evaluate validation-only rot4 TTA on simple cached-feature fusion runs."""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from dataclasses import dataclass
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
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from tqdm.auto import tqdm

from dl_final.config import load_dataset_config
from dl_final.data.datasets import HAM10000ImageDataset
from dl_final.data.transforms import IMAGENET_MEAN, IMAGENET_STD
from dl_final.evaluation.metrics import compute_classification_metrics
from dl_final.evaluation.reports import save_confusion_matrix_plot
from dl_final.evaluation.tta import (
    average_probabilities,
    expand_tta_policy,
    probabilities_from_frame,
    verify_prediction_alignment,
)
from dl_final.features.cache import (
    feature_cache_path,
    load_feature_cache,
    verify_cache_matches_split,
)
from dl_final.models.backbones import build_finetuned_feature_extractor, expected_feature_dim
from dl_final.models.fusion import WeightedLearnedFusionMLP
from dl_final.models.mlp import FeatureMLP

EXPECTED_VAL_ROWS = 1504
DEFAULT_RUN_DIRS = (
    "artifacts/runs/20260611_111408_s3_finetuned_vit-swin-beit_concat_mlp_s4_triple_seed42",
    "artifacts/runs/20260611_111414_s3_finetuned_vit-swin-beit_weightedlearned512_mlp_s4_triple_seed42",
    "artifacts/runs/20260611_111338_s3_finetuned_vit-swin_concat_mlp_s4_pair_seed42",
)


@dataclass(frozen=True)
class FeatureBlock:
    features: np.ndarray
    labels: np.ndarray
    frame: pd.DataFrame


@dataclass(frozen=True)
class RunSpec:
    run_dir: Path
    run_config: dict[str, Any]
    alias: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--feature-root", default="artifacts/features")
    parser.add_argument("--checkpoint-root", default="artifacts/checkpoints/ham10000/finetuned")
    parser.add_argument("--run-dirs", nargs="+", default=list(DEFAULT_RUN_DIRS))
    parser.add_argument("--output-run-dir", default="artifacts/runs/e3i_simple_tta_rot4")
    parser.add_argument("--tables-dir", default="artifacts/report_assets/tables")
    parser.add_argument("--figures-dir", default="artifacts/report_assets/figures")
    parser.add_argument("--policy", default="tta_rot4", choices=["tta_rot4"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--no-mixed-precision", action="store_true")
    parser.add_argument("--identity-tolerance", type=float, default=1e-4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = perf_counter()
    dataset_config = load_dataset_config(args.dataset_config)["dataset"]
    seed = int(dataset_config.get("seed", 42))
    _seed_everything(seed)
    device = _resolve_device(args.device)
    class_names = list(dataset_config["class_names"])
    split_csv = Path(dataset_config["splits_dir"]) / "val.csv"
    views = expand_tta_policy(args.policy)
    mixed_precision = (
        not args.no_mixed_precision and device.type in {"cuda", "mps"} and args.max_samples is None
    )

    output_run_dir = Path(args.output_run_dir)
    if args.max_samples is not None:
        output_run_dir = output_run_dir.with_name(
            f"{output_run_dir.name}_smoke_n{args.max_samples}"
        )
    output_run_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = Path(args.tables_dir)
    figures_dir = Path(args.figures_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    run_specs = load_run_specs([Path(path) for path in args.run_dirs])
    feature_bank = build_feature_bank(
        run_specs,
        feature_root=Path(args.feature_root),
        checkpoint_root=Path(args.checkpoint_root),
        dataset_name=str(dataset_config["name"]),
        split_csv=split_csv,
        class_names=class_names,
        image_size=resolve_image_size(dataset_config),
        views=views,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        mixed_precision=mixed_precision,
    )
    reference = next(iter(feature_bank.values()))
    y_true = reference.labels
    reference_frame = reference.frame

    view_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    identity_rows: list[dict[str, Any]] = []
    per_class_frames: list[pd.DataFrame] = []
    corrected_frames: list[pd.DataFrame] = []
    tta_predictions: dict[str, pd.DataFrame] = {}
    baseline_probabilities: list[np.ndarray] = []
    tta_probabilities: list[np.ndarray] = []

    for spec in run_specs:
        model = build_restored_model(spec.run_config, class_names).to(device)
        model.load_state_dict(torch.load(spec.run_dir / "model.pt", map_location="cpu"))
        model.eval()
        probabilities_by_view: list[np.ndarray] = []
        for view in views:
            view_started = perf_counter()
            features = build_model_input(
                spec.run_config,
                spec.run_dir / "scaler_stats.npz",
                feature_bank=feature_bank,
                view=view,
            )
            probabilities = predict_probabilities(
                model,
                features,
                device=device,
                batch_size=args.batch_size,
            )
            probabilities_by_view.append(probabilities)
            metrics = compute_metrics_from_probabilities(
                probabilities,
                y_true=y_true,
                class_names=class_names,
            )
            view_rows.append(
                {
                    "run_alias": spec.alias,
                    "run_id": spec.run_config["run_id"],
                    "fusion_method": spec.run_config["fusion_method"],
                    "backbone": spec.run_config["backbone"],
                    "view": view,
                    "rows": int(len(y_true)),
                    "macro_f1": metrics["macro_f1"],
                    "accuracy": metrics["accuracy"],
                    "weighted_f1": metrics["weighted_f1"],
                    "runtime_seconds": round(perf_counter() - view_started, 4),
                }
            )
        model.to("cpu")
        del model
        release_torch_cache()

        recorded = pd.read_csv(spec.run_dir / "predictions.csv").head(len(y_true)).copy()
        recorded_probabilities = probabilities_from_frame(recorded, class_names)
        baseline_probabilities.append(recorded_probabilities)
        identity_rows.append(
            identity_sanity_row(
                spec,
                recorded_probabilities=recorded_probabilities,
                recomputed_probabilities=probabilities_by_view[0],
                y_true=y_true,
                class_names=class_names,
                tolerance=float(args.identity_tolerance),
            )
        )

        averaged = average_probabilities(probabilities_by_view)
        tta_probabilities.append(averaged)
        tta_metrics = compute_metrics_from_probabilities(
            averaged,
            y_true=y_true,
            class_names=class_names,
        )
        baseline_metrics = compute_metrics_from_probabilities(
            recorded_probabilities,
            y_true=y_true,
            class_names=class_names,
        )
        run_rows.append(
            comparison_row(
                spec,
                baseline_metrics=baseline_metrics,
                candidate_metrics=tta_metrics,
                rows=len(y_true),
            )
        )
        prediction = prediction_frame_from_probabilities(
            reference_frame,
            y_true=y_true,
            probabilities=averaged,
            class_names=class_names,
        )
        tta_predictions[spec.alias] = prediction
        prediction.to_csv(output_run_dir / f"predictions_{spec.alias}_tta_rot4.csv", index=False)
        per_class_frames.append(
            per_class_delta_frame(
                spec,
                baseline_probabilities=recorded_probabilities,
                candidate_probabilities=averaged,
                y_true=y_true,
                class_names=class_names,
            )
        )
        corrected_frames.append(
            corrected_broken_frame(
                spec,
                baseline=recorded,
                candidate=prediction,
                class_names=class_names,
            )
        )

    baseline_ensemble = average_probabilities(baseline_probabilities)
    tta_ensemble = average_probabilities(tta_probabilities)
    ensemble_baseline_metrics = compute_metrics_from_probabilities(
        baseline_ensemble,
        y_true=y_true,
        class_names=class_names,
    )
    ensemble_tta_metrics = compute_metrics_from_probabilities(
        tta_ensemble,
        y_true=y_true,
        class_names=class_names,
    )
    ensemble_predictions = prediction_frame_from_probabilities(
        reference_frame,
        y_true=y_true,
        probabilities=tta_ensemble,
        class_names=class_names,
    )
    ensemble_predictions.to_csv(
        output_run_dir / "predictions_simple_equal_tta_rot4.csv",
        index=False,
    )
    ensemble_rows = [
        {
            "ensemble_id": "simple_equal_tta_rot4",
            "baseline_ensemble_id": "simple_equal_no_tta",
            "rows": int(len(y_true)),
            "baseline_macro_f1": ensemble_baseline_metrics["macro_f1"],
            "tta_macro_f1": ensemble_tta_metrics["macro_f1"],
            "delta_macro_f1": ensemble_tta_metrics["macro_f1"]
            - ensemble_baseline_metrics["macro_f1"],
            "baseline_accuracy": ensemble_baseline_metrics["accuracy"],
            "tta_accuracy": ensemble_tta_metrics["accuracy"],
            "delta_accuracy": ensemble_tta_metrics["accuracy"]
            - ensemble_baseline_metrics["accuracy"],
            "baseline_weighted_f1": ensemble_baseline_metrics["weighted_f1"],
            "tta_weighted_f1": ensemble_tta_metrics["weighted_f1"],
            "delta_weighted_f1": ensemble_tta_metrics["weighted_f1"]
            - ensemble_baseline_metrics["weighted_f1"],
        }
    ]

    run_config = {
        "experiment_id": "E3i",
        "policy": args.policy,
        "views": views,
        "run_dirs": [str(spec.run_dir) for spec in run_specs],
        "run_ids": [str(spec.run_config["run_id"]) for spec in run_specs],
        "test_policy": "not_loaded_or_used_in_e3i",
        "selection_metric": "validation_macro_f1",
        "max_samples": args.max_samples,
        "rows": int(len(y_true)),
        "device": str(device),
        "batch_size": int(args.batch_size),
        "num_workers": int(args.num_workers),
        "mixed_precision": bool(mixed_precision),
        "runtime_seconds": round(perf_counter() - started, 4),
    }
    write_outputs(
        output_run_dir=output_run_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        run_config=run_config,
        view_results=pd.DataFrame(view_rows),
        run_results=pd.DataFrame(run_rows),
        ensemble_results=pd.DataFrame(ensemble_rows),
        identity_sanity=pd.DataFrame(identity_rows),
        per_class=pd.concat(per_class_frames, ignore_index=True),
        corrected_broken=pd.concat(corrected_frames, ignore_index=True),
        class_names=class_names,
        ensemble_predictions=ensemble_predictions,
    )
    print(f"Wrote E3i simple TTA artifacts: {output_run_dir}")


def resolve_image_size(dataset_config: dict[str, Any], *, default: int = 224) -> int:
    """Return the configured image size, treating YAML null as the default."""

    value = dataset_config.get("image_size")
    if value is None:
        return int(default)
    return int(value)


def load_run_specs(run_dirs: list[Path]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for run_dir in run_dirs:
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing run_config.json: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if config.get("feature_source") != "finetuned":
            raise ValueError(
                "E3i expects finetuned feature_source, "
                f"got {config.get('feature_source')}"
            )
        method = str(config.get("fusion_method"))
        if method not in {"concat", "weighted_learned_512"}:
            raise ValueError(
                f"E3i supports concat and weighted_learned_512 only; got {method!r}. "
                "weighted_pca_384 is excluded because saved artifacts do not contain fitted PCA "
                "components for rotated validation features."
            )
        specs.append(RunSpec(run_dir=run_dir, run_config=config, alias=run_alias(config)))
    if not specs:
        raise ValueError("No E3i run specs were selected.")
    return specs


def run_alias(config: dict[str, Any]) -> str:
    backbone = str(config["backbone"]).replace("+", "-")
    method = str(config["fusion_method"]).replace("_", "")
    seed = str(config.get("seed", "seed"))
    return sanitize_token(f"{backbone}_{method}_seed{seed}")


def sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-")


def build_feature_bank(
    run_specs: list[RunSpec],
    *,
    feature_root: Path,
    checkpoint_root: Path,
    dataset_name: str,
    split_csv: Path,
    class_names: list[str],
    image_size: int,
    views: list[str],
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    mixed_precision: bool,
) -> dict[tuple[str, str, str], FeatureBlock]:
    needed = sorted(
        {
            (str(spec.run_config["feature_source"]), str(backbone))
            for spec in run_specs
            for backbone in spec.run_config["backbones"]
        }
    )
    bank: dict[tuple[str, str, str], FeatureBlock] = {}
    for source, backbone in needed:
        for view in views:
            key = (source, backbone, view)
            if view == "identity":
                cache_dir = feature_root / dataset_name / source / backbone
                bank[key] = load_identity_feature_block(
                    cache_dir,
                    split_csv=split_csv,
                    max_samples=max_samples,
                )
                continue
            model = build_finetuned_feature_extractor(
                backbone,
                checkpoint_path=str(checkpoint_root / backbone / "best.pt"),
                num_classes=len(class_names),
                pretrained=False,
            )
            bank[key] = extract_view_feature_block(
                model,
                backbone=backbone,
                split_csv=split_csv,
                class_names=class_names,
                image_size=image_size,
                view=view,
                device=device,
                batch_size=batch_size,
                num_workers=num_workers,
                max_samples=max_samples,
                mixed_precision=mixed_precision,
            )
            model.to("cpu")
            del model
            release_torch_cache()
    verify_feature_bank_alignment(bank)
    return bank


def load_identity_feature_block(
    cache_dir: Path,
    *,
    split_csv: Path,
    max_samples: int | None,
) -> FeatureBlock:
    cache = load_feature_cache(feature_cache_path(cache_dir, "val"))
    verify_cache_matches_split(cache, split_csv, allow_prefix=False)
    if max_samples is not None:
        features = cache.features[:max_samples]
        labels = cache.labels[:max_samples]
        frame = feature_cache_frame(
            sample_ids=cache.sample_ids[:max_samples],
            image_ids=cache.image_ids[:max_samples],
            lesion_ids=cache.lesion_ids[:max_samples],
            split_names=cache.split_names[:max_samples],
            label_names=cache.label_names[:max_samples],
        )
    else:
        features = cache.features
        labels = cache.labels
        frame = feature_cache_frame(
            sample_ids=cache.sample_ids,
            image_ids=cache.image_ids,
            lesion_ids=cache.lesion_ids,
            split_names=cache.split_names,
            label_names=cache.label_names,
        )
    return FeatureBlock(
        features=features.numpy().astype("float32"),
        labels=labels.numpy().astype("int64"),
        frame=frame,
    )


def feature_cache_frame(
    *,
    sample_ids: list[str],
    image_ids: list[str],
    lesion_ids: list[str],
    split_names: list[str],
    label_names: list[str],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": sample_ids,
            "image_id": image_ids,
            "lesion_id": lesion_ids,
            "split": split_names,
            "true_label": label_names,
        }
    )


def extract_view_feature_block(
    model: nn.Module,
    *,
    backbone: str,
    split_csv: Path,
    class_names: list[str],
    image_size: int,
    view: str,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    mixed_precision: bool,
) -> FeatureBlock:
    expected_dim = expected_feature_dim(backbone)
    dataset = HAM10000ImageDataset(
        split_csv,
        class_names=class_names,
        transform=TTAViewTransform(image_size=image_size, view=view),
        split_name="val",
        max_samples=max_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    model.to(device)
    model.eval()
    feature_batches: list[torch.Tensor] = []
    label_batches: list[torch.Tensor] = []
    rows: list[dict[str, str]] = []
    autocast_enabled = mixed_precision and device.type in {"cuda", "mps"}
    device_type = device.type if device.type in {"cuda", "cpu", "mps"} else "cpu"
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"e3i:{backbone}:{view}"):
            images = batch["image"].to(device, non_blocking=True)
            with torch.amp.autocast(device_type=device_type, enabled=autocast_enabled):
                features = model(images)
            feature_batches.append(features.detach().cpu().float())
            label_batches.append(batch["label"].cpu().long())
            for index in range(len(batch["image_id"])):
                rows.append(
                    {
                        "sample_id": str(batch["sample_id"][index]),
                        "image_id": str(batch["image_id"][index]),
                        "lesion_id": str(batch["lesion_id"][index]),
                        "split": str(batch["split"][index]),
                        "true_label": str(batch["label_name"][index]),
                    }
                )
    features_tensor = torch.cat(feature_batches, dim=0)
    if int(features_tensor.shape[1]) != expected_dim:
        raise ValueError(
            f"{backbone} produced {int(features_tensor.shape[1])}; expected {expected_dim}."
        )
    features = features_tensor.numpy().astype("float32")
    if not np.isfinite(features).all():
        raise ValueError(f"{backbone} {view} features contain NaN or Inf.")
    return FeatureBlock(
        features=features,
        labels=torch.cat(label_batches, dim=0).numpy().astype("int64"),
        frame=pd.DataFrame(rows),
    )


class TTAViewTransform:
    """Apply one deterministic TTA view before canonical ImageNet preprocessing."""

    def __init__(self, *, image_size: int, view: str) -> None:
        self.image_size = int(image_size)
        self.view = view
        self.post = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size), antialias=True),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

    def __call__(self, image: Image.Image) -> torch.Tensor:
        if self.view == "identity":
            transformed = image
        elif self.view == "rot90":
            transformed = image.transpose(Image.Transpose.ROTATE_90)
        elif self.view == "rot180":
            transformed = image.transpose(Image.Transpose.ROTATE_180)
        elif self.view == "rot270":
            transformed = image.transpose(Image.Transpose.ROTATE_270)
        else:
            raise ValueError(f"Unsupported TTA view: {self.view}")
        return self.post(transformed)


def verify_feature_bank_alignment(bank: dict[tuple[str, str, str], FeatureBlock]) -> None:
    blocks = list(bank.values())
    if not blocks:
        raise ValueError("Feature bank is empty.")
    reference = blocks[0]
    for key, block in bank.items():
        verify_prediction_alignment([reference.frame, block.frame])
        if not np.array_equal(reference.labels, block.labels):
            raise ValueError(f"Feature block label mismatch: {key}")


def build_restored_model(run_config: dict[str, Any], class_names: list[str]) -> nn.Module:
    hidden_dims = list(run_config["hidden_dims"])
    dropout = float(run_config["dropout"])
    method = str(run_config["fusion_method"])
    if method == "concat":
        return FeatureMLP(
            input_dim=int(run_config["feature_dim"]),
            num_classes=len(class_names),
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    if method == "weighted_learned_512":
        input_dims = [
            int(run_config["input_dims"][backbone]) for backbone in run_config["backbones"]
        ]
        return WeightedLearnedFusionMLP(
            input_dims=input_dims,
            num_classes=len(class_names),
            projection_dim=int(run_config["projection_dim"]),
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    raise ValueError(f"Unsupported E3i fusion method: {method}")


def build_model_input(
    run_config: dict[str, Any],
    scaler_stats_path: Path,
    *,
    feature_bank: dict[tuple[str, str, str], FeatureBlock],
    view: str,
) -> np.ndarray:
    stats = np.load(scaler_stats_path)
    source = str(run_config["feature_source"])
    blocks: list[np.ndarray] = []
    for backbone in run_config["backbones"]:
        raw = feature_bank[(source, backbone, view)].features.astype("float32")
        mean = stats[f"{backbone}_mean"].astype("float32")
        scale = stats[f"{backbone}_scale"].astype("float32")
        blocks.append(((raw - mean) / np.clip(scale, a_min=1e-12, a_max=None)).astype("float32"))
    features = np.concatenate(blocks, axis=1).astype("float32")
    if int(features.shape[1]) != int(run_config["fusion_input_dim"]):
        raise ValueError(
            f"Model input dim {int(features.shape[1])} != "
            f"fusion_input_dim {run_config['fusion_input_dim']}."
        )
    if not np.isfinite(features).all():
        raise ValueError("Model input features contain NaN or Inf.")
    return features


def predict_probabilities(
    model: nn.Module,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    loader = DataLoader(
        TensorDataset(torch.from_numpy(features.astype("float32"))),
        batch_size=batch_size,
        shuffle=False,
    )
    batches: list[torch.Tensor] = []
    model.eval()
    with torch.no_grad():
        for (batch_features,) in loader:
            logits = model(batch_features.to(device))
            batches.append(torch.softmax(logits, dim=1).detach().cpu())
    probabilities = torch.cat(batches, dim=0).numpy().astype("float32")
    row_sums = np.clip(probabilities.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
    return (probabilities / row_sums).astype("float32")


def compute_metrics_from_probabilities(
    probabilities: np.ndarray,
    *,
    y_true: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    return compute_classification_metrics(y_true, probabilities.argmax(axis=1), class_names)


def identity_sanity_row(
    spec: RunSpec,
    *,
    recorded_probabilities: np.ndarray,
    recomputed_probabilities: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    tolerance: float,
) -> dict[str, Any]:
    recorded_metrics = compute_metrics_from_probabilities(
        recorded_probabilities,
        y_true=y_true,
        class_names=class_names,
    )
    recomputed_metrics = compute_metrics_from_probabilities(
        recomputed_probabilities,
        y_true=y_true,
        class_names=class_names,
    )
    max_delta = float(np.max(np.abs(recorded_probabilities - recomputed_probabilities)))
    passed = bool(max_delta <= tolerance or len(y_true) < EXPECTED_VAL_ROWS)
    return {
        "run_alias": spec.alias,
        "run_id": spec.run_config["run_id"],
        "fusion_method": spec.run_config["fusion_method"],
        "backbone": spec.run_config["backbone"],
        "rows": int(len(y_true)),
        "recorded_macro_f1": recorded_metrics["macro_f1"],
        "recomputed_identity_macro_f1": recomputed_metrics["macro_f1"],
        "macro_f1_delta": recomputed_metrics["macro_f1"] - recorded_metrics["macro_f1"],
        "max_abs_probability_delta": max_delta,
        "passed": passed,
        "status": "passed" if passed else "warning",
    }


def comparison_row(
    spec: RunSpec,
    *,
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    rows: int,
) -> dict[str, Any]:
    return {
        "run_alias": spec.alias,
        "run_id": spec.run_config["run_id"],
        "fusion_method": spec.run_config["fusion_method"],
        "backbone": spec.run_config["backbone"],
        "rows": int(rows),
        "baseline_macro_f1": baseline_metrics["macro_f1"],
        "tta_macro_f1": candidate_metrics["macro_f1"],
        "delta_macro_f1": candidate_metrics["macro_f1"] - baseline_metrics["macro_f1"],
        "baseline_accuracy": baseline_metrics["accuracy"],
        "tta_accuracy": candidate_metrics["accuracy"],
        "delta_accuracy": candidate_metrics["accuracy"] - baseline_metrics["accuracy"],
        "baseline_weighted_f1": baseline_metrics["weighted_f1"],
        "tta_weighted_f1": candidate_metrics["weighted_f1"],
        "delta_weighted_f1": candidate_metrics["weighted_f1"] - baseline_metrics["weighted_f1"],
    }


def prediction_frame_from_probabilities(
    reference_frame: pd.DataFrame,
    *,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    y_pred = probabilities.argmax(axis=1)
    rows = reference_frame.reset_index(drop=True).copy()
    rows["pred_label"] = [class_names[int(index)] for index in y_pred]
    rows["correct"] = (y_true == y_pred).astype(bool)
    rows["confidence"] = probabilities.max(axis=1)
    for index, label in enumerate(class_names):
        rows[f"prob_{label}"] = probabilities[:, index]
    return rows


def per_class_delta_frame(
    spec: RunSpec,
    *,
    baseline_probabilities: np.ndarray,
    candidate_probabilities: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    baseline = compute_metrics_from_probabilities(
        baseline_probabilities,
        y_true=y_true,
        class_names=class_names,
    )
    candidate = compute_metrics_from_probabilities(
        candidate_probabilities,
        y_true=y_true,
        class_names=class_names,
    )
    base = pd.DataFrame(baseline["per_class"]).rename(
        columns={
            "precision": "baseline_precision",
            "recall": "baseline_recall",
            "f1": "baseline_f1",
        }
    )
    cand = pd.DataFrame(candidate["per_class"]).rename(
        columns={"precision": "tta_precision", "recall": "tta_recall", "f1": "tta_f1"}
    )
    merged = base.merge(cand, on=["label", "support"], how="inner")
    merged.insert(0, "run_alias", spec.alias)
    merged.insert(1, "fusion_method", spec.run_config["fusion_method"])
    merged.insert(2, "backbone", spec.run_config["backbone"])
    merged["delta_f1"] = merged["tta_f1"] - merged["baseline_f1"]
    merged["delta_recall"] = merged["tta_recall"] - merged["baseline_recall"]
    merged["delta_precision"] = merged["tta_precision"] - merged["baseline_precision"]
    return merged


def corrected_broken_frame(
    spec: RunSpec,
    *,
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    class_names: list[str],
) -> pd.DataFrame:
    verify_prediction_alignment([baseline, candidate])
    rows: list[dict[str, Any]] = []
    for label in ["all", *class_names]:
        if label == "all":
            mask = np.ones(len(candidate), dtype=bool)
        else:
            mask = candidate["true_label"].astype(str).to_numpy() == label
        baseline_correct = baseline.loc[mask, "correct"].astype(bool).to_numpy()
        candidate_correct = candidate.loc[mask, "correct"].astype(bool).to_numpy()
        rows.append(
            {
                "run_alias": spec.alias,
                "fusion_method": spec.run_config["fusion_method"],
                "backbone": spec.run_config["backbone"],
                "label": label,
                "support": int(mask.sum()),
                "corrected_by_tta": int((~baseline_correct & candidate_correct).sum()),
                "broken_by_tta": int((baseline_correct & ~candidate_correct).sum()),
                "both_correct": int((baseline_correct & candidate_correct).sum()),
                "both_wrong": int((~baseline_correct & ~candidate_correct).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_outputs(
    *,
    output_run_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    run_config: dict[str, Any],
    view_results: pd.DataFrame,
    run_results: pd.DataFrame,
    ensemble_results: pd.DataFrame,
    identity_sanity: pd.DataFrame,
    per_class: pd.DataFrame,
    corrected_broken: pd.DataFrame,
    class_names: list[str],
    ensemble_predictions: pd.DataFrame,
) -> None:
    suffix = "_smoke" if run_config["max_samples"] is not None else ""
    (output_run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    (output_run_dir / "tta_policy.json").write_text(
        json.dumps({"policy": run_config["policy"], "views": run_config["views"]}, indent=2),
        encoding="utf-8",
    )
    view_results.to_csv(output_run_dir / "tta_view_results.csv", index=False)
    run_results.to_csv(output_run_dir / "tta_run_results.csv", index=False)
    ensemble_results.to_csv(output_run_dir / "tta_ensemble_results.csv", index=False)
    identity_sanity.to_csv(output_run_dir / "tta_identity_sanity.csv", index=False)
    per_class.to_csv(output_run_dir / "tta_per_class_delta.csv", index=False)
    corrected_broken.to_csv(output_run_dir / "tta_corrected_broken.csv", index=False)

    view_results.to_csv(tables_dir / f"e3i_simple_tta_rot4_view_results{suffix}.csv", index=False)
    run_results.to_csv(tables_dir / f"e3i_simple_tta_rot4_run_results{suffix}.csv", index=False)
    ensemble_results.to_csv(
        tables_dir / f"e3i_simple_tta_rot4_ensemble_results{suffix}.csv",
        index=False,
    )
    identity_sanity.to_csv(
        tables_dir / f"e3i_simple_tta_rot4_identity_sanity{suffix}.csv",
        index=False,
    )
    per_class.to_csv(
        tables_dir / f"e3i_simple_tta_rot4_per_class_delta{suffix}.csv",
        index=False,
    )
    corrected_broken.to_csv(
        tables_dir / f"e3i_simple_tta_rot4_corrected_broken{suffix}.csv",
        index=False,
    )

    save_macro_plot(run_results, figures_dir / f"e3i_simple_tta_rot4_macro_f1{suffix}.png")
    save_view_plot(view_results, figures_dir / f"e3i_simple_tta_rot4_view_macro_f1{suffix}.png")
    matrix = compute_classification_metrics(
        np.array([class_names.index(label) for label in ensemble_predictions["true_label"]]),
        np.array([class_names.index(label) for label in ensemble_predictions["pred_label"]]),
        class_names,
    )["confusion_matrix"]
    save_confusion_matrix_plot(
        matrix,
        class_names,
        figures_dir / f"e3i_simple_tta_rot4_ensemble_confusion_matrix{suffix}.png",
        title="E3i simple equal + rot4 validation confusion matrix",
    )


def save_macro_plot(results: pd.DataFrame, output_path: Path) -> None:
    labels = results["run_alias"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    ax.bar(
        x - width / 2,
        results["baseline_macro_f1"],
        width,
        label="identity/no TTA",
        color="#64748b",
    )
    ax.bar(x + width / 2, results["tta_macro_f1"], width, label="rot4 TTA", color="#0f766e")
    ymin_value = min(results["baseline_macro_f1"].min(), results["tta_macro_f1"].min())
    ymax_value = max(results["baseline_macro_f1"].max(), results["tta_macro_f1"].max())
    ymin = max(0.0, float(ymin_value) - 0.04)
    ymax = min(1.0, float(ymax_value) + 0.04)
    ax.set_ylim(ymin, ymax)
    ax.set_ylabel("Validation macro-F1")
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.legend()
    ax.set_title("E3i simple fusion rot4 TTA")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_view_plot(results: pd.DataFrame, output_path: Path) -> None:
    pivot = results.pivot(index="run_alias", columns="view", values="macro_f1")
    fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)
    pivot.plot(kind="bar", ax=ax)
    ax.set_ylabel("Validation macro-F1")
    ax.set_xlabel("")
    ax.set_title("E3i per-view macro-F1")
    ax.legend(title="view")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def release_torch_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


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
