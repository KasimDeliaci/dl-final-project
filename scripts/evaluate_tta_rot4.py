"""Evaluate E3h validation-only rot4 test-time augmentation."""

from __future__ import annotations

import argparse
import json
import os
import random
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
from PIL import Image, ImageOps
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
    FeatureCache,
    feature_cache_path,
    load_feature_cache,
    verify_cache_matches_split,
)
from dl_final.models.backbones import (
    build_finetuned_feature_extractor,
    build_frozen_feature_extractor,
    expected_feature_dim,
)
from dl_final.models.metadata_fusion import MetadataFiLMMLP, MetadataGatedBackboneMLP

TRIPLE_BACKBONES = ("vit_b16", "swin_tiny", "beit_base")
SEEDS = (7, 13, 42, 101, 202)
EXPECTED_VAL_ROWS = 1504
E3G_BASELINE_MACRO_F1 = 0.7665


@dataclass(frozen=True)
class RunMember:
    family: str
    run_dir: Path
    run_config: dict[str, Any]


@dataclass(frozen=True)
class FeatureBlock:
    features: np.ndarray
    labels: np.ndarray
    frame: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--feature-root", default="artifacts/features")
    parser.add_argument("--checkpoint-root", default="artifacts/checkpoints/ham10000/finetuned")
    parser.add_argument("--run-root", default="artifacts/runs")
    parser.add_argument("--output-run-dir", default="artifacts/runs/e3h_tta_rot4")
    parser.add_argument("--tables-dir", default="artifacts/report_assets/tables")
    parser.add_argument("--figures-dir", default="artifacts/report_assets/figures")
    parser.add_argument("--policy", default="tta_rot4", choices=["tta_rot4"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=64)
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
    val_frame = pd.read_csv(split_csv)
    if args.max_samples is not None:
        val_frame = val_frame.head(args.max_samples).copy()
    views = expand_tta_policy(args.policy)
    mixed_precision = (
        not args.no_mixed_precision
        and device.type in {"cuda", "mps"}
        and args.max_samples is None
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

    members = discover_members(Path(args.run_root))
    feature_bank = build_feature_bank(
        feature_root=Path(args.feature_root),
        checkpoint_root=Path(args.checkpoint_root),
        dataset_name=str(dataset_config["name"]),
        split_csv=split_csv,
        class_names=class_names,
        image_size=int(dataset_config.get("image_size") or 224),
        views=views,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_samples=args.max_samples,
        mixed_precision=mixed_precision,
    )
    reference_frame = next(iter(feature_bank.values())).frame
    y_true = next(iter(feature_bank.values())).labels

    model_rows: list[dict[str, Any]] = []
    family_probabilities: dict[str, np.ndarray] = {}
    identity_sanity_rows: list[dict[str, Any]] = []
    view_runtime_rows: list[dict[str, Any]] = []
    for family, family_members in members.items():
        seed_probabilities: list[np.ndarray] = []
        for member in family_members:
            seed_rows, view_probabilities, identity_row = evaluate_member_views(
                member,
                feature_bank=feature_bank,
                reference_frame=reference_frame,
                class_names=class_names,
                y_true=y_true,
                device=device,
                batch_size=args.batch_size,
                views=views,
                identity_tolerance=args.identity_tolerance,
            )
            model_rows.extend(seed_rows)
            identity_sanity_rows.append(identity_row)
            seed_probabilities.append(average_probabilities(view_probabilities))
            for row in seed_rows:
                view_runtime_rows.append(
                    {
                        "family": family,
                        "seed": int(member.run_config["seed"]),
                        "run_id": member.run_config["run_id"],
                        "view": row["view"],
                        "runtime_seconds": row["runtime_seconds"],
                    }
                )
        family_probabilities[family] = average_probabilities(seed_probabilities)

    family_rows, family_predictions = evaluate_family_outputs(
        family_probabilities,
        reference_frame=reference_frame,
        y_true=y_true,
        class_names=class_names,
    )
    ensemble_probabilities = average_probabilities(
        [
            family_probabilities["e3d_film"],
            family_probabilities["e3d_gated"],
            family_probabilities["e3f_mixed_gated"],
        ]
    )
    ensemble_id = "top3_family_equal_tta_rot4"
    ensemble_metrics = compute_metrics_from_probabilities(
        ensemble_probabilities,
        y_true=y_true,
        class_names=class_names,
    )
    ensemble_predictions = prediction_frame_from_probabilities(
        reference_frame,
        y_true=y_true,
        probabilities=ensemble_probabilities,
        class_names=class_names,
    )
    e3g_predictions = load_e3g_baseline_predictions(
        Path(args.run_root) / "e3g_prediction_ensemble",
        class_names=class_names,
        max_samples=args.max_samples,
    )
    corrected_broken = corrected_broken_frame(
        e3g_predictions,
        ensemble_predictions,
        class_names=class_names,
    )
    vs_e3g = build_vs_e3g_table(
        ensemble_metrics,
        e3g_predictions,
        y_true=y_true,
        class_names=class_names,
    )

    model_results = pd.DataFrame(model_rows)
    identity_sanity = pd.DataFrame(identity_sanity_rows)
    family_results = pd.DataFrame(family_rows)
    ensemble_results = pd.DataFrame(
        [
            {
                "ensemble_id": ensemble_id,
                "policy": args.policy,
                "views": "+".join(views),
                "rows": int(len(y_true)),
                "macro_f1": ensemble_metrics["macro_f1"],
                "accuracy": ensemble_metrics["accuracy"],
                "weighted_f1": ensemble_metrics["weighted_f1"],
                "macro_precision": ensemble_metrics["macro_precision"],
                "macro_recall": ensemble_metrics["macro_recall"],
                "delta_macro_f1_vs_e3g": float(
                    ensemble_metrics["macro_f1"] - vs_e3g.loc[0, "e3g_macro_f1"]
                ),
                "delta_macro_f1_vs_recorded_e3g": float(
                    ensemble_metrics["macro_f1"] - E3G_BASELINE_MACRO_F1
                ),
            }
        ]
    )
    per_class = pd.DataFrame(ensemble_metrics["per_class"]).assign(
        ensemble_id=ensemble_id,
        policy=args.policy,
    )
    per_class_delta = per_class_delta_vs_e3g(
        e3g_predictions,
        ensemble_predictions,
        class_names=class_names,
    )

    write_outputs(
        output_run_dir=output_run_dir,
        tables_dir=tables_dir,
        figures_dir=figures_dir,
        model_results=model_results,
        family_results=family_results,
        ensemble_results=ensemble_results,
        per_class=per_class,
        per_class_delta=per_class_delta,
        corrected_broken=corrected_broken,
        identity_sanity=identity_sanity,
        ensemble_predictions=ensemble_predictions,
        family_predictions=family_predictions,
        runtime_rows=pd.DataFrame(view_runtime_rows),
        run_config={
            "experiment_id": "E3h",
            "policy": args.policy,
            "views": views,
            "candidate": ensemble_id,
            "dataset": dataset_config["name"],
            "split": "val",
            "rows": int(len(y_true)),
            "max_samples": args.max_samples,
            "class_names": class_names,
            "member_families": sorted(members),
            "aggregation": "equal probability averaging across views, seeds, and families",
            "selection_metric": "validation_macro_f1",
            "test_policy": "not_loaded_or_used_in_e3h",
            "runtime_seconds": round(perf_counter() - started, 4),
            "device": str(device),
            "batch_size": args.batch_size,
            "mixed_precision": mixed_precision,
        },
        class_names=class_names,
    )

    print(f"Wrote E3h TTA artifacts: {output_run_dir}")
    print(ensemble_results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))


def discover_members(run_root: Path) -> dict[str, list[RunMember]]:
    specs = {
        "e3d_film": "*e3d_triple_metadata_film_e3d_metadata_fusion_seed*/run_config.json",
        "e3d_gated": (
            "*e3d_triple_metadata_gated_backbone_e3d_metadata_fusion_seed*/run_config.json"
        ),
        "e3f_mixed_gated": (
            "*e3d_triple_metadata_gated_backbone_e3f_mixed_adaptation_seed*/run_config.json"
        ),
    }
    members: dict[str, list[RunMember]] = {}
    for family, pattern in specs.items():
        family_members: list[RunMember] = []
        for config_path in sorted(run_root.glob(pattern)):
            config = json.loads(config_path.read_text(encoding="utf-8"))
            if int(config["seed"]) in SEEDS:
                family_members.append(
                    RunMember(family=family, run_dir=config_path.parent, run_config=config)
                )
        family_members = sorted(family_members, key=lambda member: int(member.run_config["seed"]))
        seeds = [int(member.run_config["seed"]) for member in family_members]
        if seeds != list(SEEDS):
            raise ValueError(f"{family} expected seeds {list(SEEDS)}, found {seeds}.")
        members[family] = family_members
    return members


def build_feature_bank(
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
    bank: dict[tuple[str, str, str], FeatureBlock] = {}
    source_backbones = {
        "finetuned": TRIPLE_BACKBONES,
        "frozen_vit_finetuned_swin_beit": TRIPLE_BACKBONES,
    }
    for source, backbones in source_backbones.items():
        for backbone in backbones:
            for view in views:
                key = (source, backbone, view)
                if source == "frozen_vit_finetuned_swin_beit" and backbone != "vit_b16":
                    bank[key] = bank[("finetuned", backbone, view)]
                    continue
                if view == "identity":
                    bank[key] = load_identity_feature_block(
                        feature_root / dataset_name / source / backbone,
                        split_csv=split_csv,
                        max_samples=max_samples,
                    )
                    continue
                model = build_tta_feature_extractor(
                    source=source,
                    backbone=backbone,
                    checkpoint_root=checkpoint_root,
                    class_names=class_names,
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
                release_torch_model(model)
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
        cache = slice_cache(cache, max_samples)
    frame = feature_cache_frame(cache)
    return FeatureBlock(
        features=cache.features.numpy().astype("float32"),
        labels=cache.labels.numpy().astype("int64"),
        frame=frame,
    )


def slice_cache(cache: FeatureCache, max_samples: int) -> FeatureCache:
    return FeatureCache(
        path=cache.path,
        features=cache.features[:max_samples],
        labels=cache.labels[:max_samples],
        sample_ids=cache.sample_ids[:max_samples],
        image_ids=cache.image_ids[:max_samples],
        lesion_ids=cache.lesion_ids[:max_samples],
        label_names=cache.label_names[:max_samples],
        split_names=cache.split_names[:max_samples],
        split=cache.split,
        backbone=cache.backbone,
        feature_dim=cache.feature_dim,
        metadata=cache.metadata,
    )


def feature_cache_frame(cache: FeatureCache) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": cache.sample_ids,
            "image_id": cache.image_ids,
            "lesion_id": cache.lesion_ids,
            "split": cache.split_names,
            "true_label": cache.label_names,
        }
    )


def build_tta_feature_extractor(
    *,
    source: str,
    backbone: str,
    checkpoint_root: Path,
    class_names: list[str],
) -> nn.Module:
    if source == "frozen_vit_finetuned_swin_beit" and backbone == "vit_b16":
        return build_frozen_feature_extractor(backbone, pretrained=True)
    checkpoint_path = checkpoint_root / backbone / "best.pt"
    return build_finetuned_feature_extractor(
        backbone,
        checkpoint_path=str(checkpoint_path),
        num_classes=len(class_names),
        pretrained=False,
    )


def release_torch_model(model: nn.Module) -> None:
    """Release accelerator memory after one view extraction."""

    model.to("cpu")
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


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
        for batch in tqdm(loader, desc=f"e3h:{backbone}:{view}"):
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
        elif self.view == "hflip":
            transformed = ImageOps.mirror(image)
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


def evaluate_member_views(
    member: RunMember,
    *,
    feature_bank: dict[tuple[str, str, str], FeatureBlock],
    reference_frame: pd.DataFrame,
    class_names: list[str],
    y_true: np.ndarray,
    device: torch.device,
    batch_size: int,
    views: list[str],
    identity_tolerance: float,
) -> tuple[list[dict[str, Any]], list[np.ndarray], dict[str, Any]]:
    run_config = member.run_config
    model = build_restored_model(run_config, class_names).to(device)
    model.load_state_dict(torch.load(member.run_dir / "model.pt", map_location="cpu"))
    model.eval()
    metadata = metadata_features_from_artifact(
        member.run_dir / "metadata_preprocessing.json",
        reference_frame=reference_frame,
        split_csv=Path("data/splits/val.csv"),
    )
    probabilities_by_view: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    for view in views:
        started = perf_counter()
        features = build_model_input(
            run_config,
            member.run_dir / "scaler_stats.npz",
            feature_bank=feature_bank,
            metadata=metadata,
            view=view,
        )
        probabilities = predict_probabilities(
            model,
            features,
            device=device,
            batch_size=batch_size,
        )
        probabilities_by_view.append(probabilities)
        metrics = compute_metrics_from_probabilities(
            probabilities,
            y_true=y_true,
            class_names=class_names,
        )
        rows.append(
            {
                "family": member.family,
                "seed": int(run_config["seed"]),
                "run_id": run_config["run_id"],
                "view": view,
                "rows": int(len(y_true)),
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "runtime_seconds": round(perf_counter() - started, 4),
            }
        )
    identity_row = identity_sanity_check(
        member,
        recomputed_probabilities=probabilities_by_view[0],
        y_true=y_true,
        class_names=class_names,
        tolerance=identity_tolerance,
    )
    return rows, probabilities_by_view, identity_row


def build_restored_model(run_config: dict[str, Any], class_names: list[str]) -> nn.Module:
    input_dims = [int(run_config["input_dims"][backbone]) for backbone in run_config["backbones"]]
    image_dim = int(run_config["image_feature_dim"])
    metadata_dim = int(run_config["metadata_feature_dim"])
    hidden_dims = list(run_config["hidden_dims"])
    dropout = float(run_config["dropout"])
    operator = str(run_config["metadata_fusion_operator"])
    if operator == "triple_metadata_film":
        return MetadataFiLMMLP(
            image_dim=image_dim,
            metadata_dim=metadata_dim,
            num_classes=len(class_names),
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    if operator == "triple_metadata_gated_backbone":
        return MetadataGatedBackboneMLP(
            input_dims=input_dims,
            metadata_dim=metadata_dim,
            num_classes=len(class_names),
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    raise ValueError(f"E3h only supports FiLM/gated operators, got {operator!r}.")


def metadata_features_from_artifact(
    artifact_path: Path,
    *,
    reference_frame: pd.DataFrame,
    split_csv: Path,
) -> np.ndarray:
    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    preprocessor = artifact["metadata_preprocessor"]
    split_frame = pd.read_csv(split_csv).head(len(reference_frame)).copy()
    split_image_ids = split_frame["image_id"].astype(str).tolist()
    reference_image_ids = reference_frame["image_id"].astype(str).tolist()
    if split_image_ids != reference_image_ids:
        raise ValueError("Metadata split frame does not align with reference image IDs.")
    age = pd.to_numeric(split_frame["age"], errors="coerce").fillna(
        float(preprocessor["age"]["median_imputation_value"])
    )
    age_scaled = (
        (age.to_numpy(dtype="float32") - float(preprocessor["age"]["mean"]))
        / float(preprocessor["age"]["scale"])
    ).reshape(-1, 1)
    sex = _normalize_category(split_frame["sex"])
    localization = _normalize_category(split_frame["localization"])
    sex_encoded = one_hot_with_vocab(sex, preprocessor["categorical"]["sex_categories"])
    localization_encoded = one_hot_with_vocab(
        localization,
        preprocessor["categorical"]["localization_categories"],
    )
    features = np.concatenate([age_scaled, sex_encoded, localization_encoded], axis=1).astype(
        "float32"
    )
    if int(features.shape[1]) != int(preprocessor["output_dim"]):
        raise ValueError("Metadata feature dimension mismatch.")
    if not np.isfinite(features).all():
        raise ValueError("Metadata features contain NaN or Inf.")
    return features


def _normalize_category(series: pd.Series) -> pd.Series:
    normalized = series.fillna("unknown").astype(str).str.strip().str.lower()
    return normalized.mask(normalized == "", "unknown")


def one_hot_with_vocab(values: pd.Series, categories: list[str]) -> np.ndarray:
    category_to_index = {category: index for index, category in enumerate(categories)}
    unknown_index = category_to_index.get("unknown")
    encoded = np.zeros((len(values), len(categories)), dtype="float32")
    for row_index, value in enumerate(values.tolist()):
        index = category_to_index.get(value, unknown_index)
        if index is not None:
            encoded[row_index, index] = 1.0
    return encoded


def build_model_input(
    run_config: dict[str, Any],
    scaler_stats_path: Path,
    *,
    feature_bank: dict[tuple[str, str, str], FeatureBlock],
    metadata: np.ndarray,
    view: str,
) -> np.ndarray:
    stats = np.load(scaler_stats_path)
    blocks: list[np.ndarray] = []
    source = str(run_config["feature_source"])
    for backbone in run_config["backbones"]:
        raw = feature_bank[(source, backbone, view)].features.astype("float32")
        mean = stats[f"{backbone}_mean"].astype("float32")
        scale = stats[f"{backbone}_scale"].astype("float32")
        blocks.append(((raw - mean) / np.clip(scale, a_min=1e-12, a_max=None)).astype("float32"))
    features = np.concatenate([*blocks, metadata], axis=1).astype("float32")
    if int(features.shape[1]) != int(run_config["feature_dim"]):
        raise ValueError(
            "Model input dim "
            f"{int(features.shape[1])} != run feature_dim {run_config['feature_dim']}."
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
    y_pred = probabilities.argmax(axis=1)
    return compute_classification_metrics(y_true, y_pred, class_names)


def identity_sanity_check(
    member: RunMember,
    *,
    recomputed_probabilities: np.ndarray,
    y_true: np.ndarray,
    class_names: list[str],
    tolerance: float,
) -> dict[str, Any]:
    predictions_path = member.run_dir / "predictions.csv"
    recorded_predictions = pd.read_csv(predictions_path).head(len(y_true)).copy()
    recorded_probabilities = probabilities_from_frame(recorded_predictions, class_names)
    recomputed_metrics = compute_metrics_from_probabilities(
        recomputed_probabilities,
        y_true=y_true,
        class_names=class_names,
    )
    recorded_metrics = json.loads((member.run_dir / "metrics_summary.json").read_text())
    max_abs_probability_delta = float(
        np.max(np.abs(recomputed_probabilities - recorded_probabilities))
    )
    macro_f1_delta = float(recomputed_metrics["macro_f1"] - recorded_metrics["macro_f1"])
    if len(y_true) == EXPECTED_VAL_ROWS and max_abs_probability_delta > tolerance:
        raise ValueError(
            f"Identity sanity failed for {member.run_config['run_id']}: "
            f"max probability delta {max_abs_probability_delta:.6f} > {tolerance:.6f}."
        )
    return {
        "family": member.family,
        "seed": int(member.run_config["seed"]),
        "run_id": member.run_config["run_id"],
        "rows": int(len(y_true)),
        "recorded_macro_f1": float(recorded_metrics["macro_f1"]),
        "recomputed_identity_macro_f1": float(recomputed_metrics["macro_f1"]),
        "macro_f1_delta": macro_f1_delta,
        "max_abs_probability_delta": max_abs_probability_delta,
        "passed": bool(max_abs_probability_delta <= tolerance or len(y_true) < EXPECTED_VAL_ROWS),
    }


def evaluate_family_outputs(
    family_probabilities: dict[str, np.ndarray],
    *,
    reference_frame: pd.DataFrame,
    y_true: np.ndarray,
    class_names: list[str],
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    predictions: dict[str, pd.DataFrame] = {}
    for family, probabilities in family_probabilities.items():
        metrics = compute_metrics_from_probabilities(
            probabilities,
            y_true=y_true,
            class_names=class_names,
        )
        rows.append(
            {
                "family": family,
                "rows": int(len(y_true)),
                "macro_f1": metrics["macro_f1"],
                "accuracy": metrics["accuracy"],
                "weighted_f1": metrics["weighted_f1"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
            }
        )
        predictions[family] = prediction_frame_from_probabilities(
            reference_frame,
            y_true=y_true,
            probabilities=probabilities,
            class_names=class_names,
        )
    return rows, predictions


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


def load_e3g_baseline_predictions(
    e3g_dir: Path,
    *,
    class_names: list[str],
    max_samples: int | None,
) -> pd.DataFrame:
    path = e3g_dir / "ensemble_predictions_top3_family_equal.csv"
    frame = pd.read_csv(path)
    if max_samples is not None:
        frame = frame.head(max_samples).copy()
    _ = probabilities_from_frame(frame, class_names)
    return frame


def corrected_broken_frame(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
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
                "label": label,
                "support": int(mask.sum()),
                "corrected_by_tta": int((~baseline_correct & candidate_correct).sum()),
                "broken_by_tta": int((baseline_correct & ~candidate_correct).sum()),
                "both_correct": int((baseline_correct & candidate_correct).sum()),
                "both_wrong": int((~baseline_correct & ~candidate_correct).sum()),
            }
        )
    return pd.DataFrame(rows)


def build_vs_e3g_table(
    candidate_metrics: dict[str, Any],
    e3g_predictions: pd.DataFrame,
    *,
    y_true: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    e3g_probabilities = probabilities_from_frame(e3g_predictions, class_names)
    e3g_metrics = compute_metrics_from_probabilities(
        e3g_probabilities,
        y_true=y_true,
        class_names=class_names,
    )
    return pd.DataFrame(
        [
            {
                "baseline": "e3g_top3_family_equal",
                "candidate": "e3h_top3_family_equal_tta_rot4",
                "e3g_macro_f1": e3g_metrics["macro_f1"],
                "e3h_macro_f1": candidate_metrics["macro_f1"],
                "delta_macro_f1": candidate_metrics["macro_f1"] - e3g_metrics["macro_f1"],
                "e3g_accuracy": e3g_metrics["accuracy"],
                "e3h_accuracy": candidate_metrics["accuracy"],
                "delta_accuracy": candidate_metrics["accuracy"] - e3g_metrics["accuracy"],
                "e3g_weighted_f1": e3g_metrics["weighted_f1"],
                "e3h_weighted_f1": candidate_metrics["weighted_f1"],
                "delta_weighted_f1": candidate_metrics["weighted_f1"]
                - e3g_metrics["weighted_f1"],
            }
        ]
    )


def per_class_delta_vs_e3g(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    *,
    class_names: list[str],
) -> pd.DataFrame:
    verify_prediction_alignment([baseline, candidate])
    y_true = np.array([class_names.index(label) for label in candidate["true_label"].astype(str)])
    baseline_pred = np.array(
        [class_names.index(label) for label in baseline["pred_label"].astype(str)]
    )
    candidate_pred = np.array(
        [class_names.index(label) for label in candidate["pred_label"].astype(str)]
    )
    baseline_metrics = compute_classification_metrics(y_true, baseline_pred, class_names)
    candidate_metrics = compute_classification_metrics(y_true, candidate_pred, class_names)
    base = pd.DataFrame(baseline_metrics["per_class"]).rename(
        columns={"precision": "e3g_precision", "recall": "e3g_recall", "f1": "e3g_f1"}
    )
    cand = pd.DataFrame(candidate_metrics["per_class"]).rename(
        columns={"precision": "e3h_precision", "recall": "e3h_recall", "f1": "e3h_f1"}
    )
    merged = base.merge(cand, on=["label", "support"], how="inner")
    merged["delta_f1"] = merged["e3h_f1"] - merged["e3g_f1"]
    merged["delta_recall"] = merged["e3h_recall"] - merged["e3g_recall"]
    merged["delta_precision"] = merged["e3h_precision"] - merged["e3g_precision"]
    return merged


def write_outputs(
    *,
    output_run_dir: Path,
    tables_dir: Path,
    figures_dir: Path,
    model_results: pd.DataFrame,
    family_results: pd.DataFrame,
    ensemble_results: pd.DataFrame,
    per_class: pd.DataFrame,
    per_class_delta: pd.DataFrame,
    corrected_broken: pd.DataFrame,
    identity_sanity: pd.DataFrame,
    ensemble_predictions: pd.DataFrame,
    family_predictions: dict[str, pd.DataFrame],
    runtime_rows: pd.DataFrame,
    run_config: dict[str, Any],
    class_names: list[str],
) -> None:
    output_run_dir.mkdir(parents=True, exist_ok=True)
    (output_run_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2),
        encoding="utf-8",
    )
    (output_run_dir / "tta_policy.json").write_text(
        json.dumps({"policy": run_config["policy"], "views": run_config["views"]}, indent=2),
        encoding="utf-8",
    )
    model_results.to_csv(output_run_dir / "tta_model_results.csv", index=False)
    family_results.to_csv(output_run_dir / "tta_family_results.csv", index=False)
    ensemble_results.to_csv(output_run_dir / "tta_ensemble_results.csv", index=False)
    per_class.to_csv(output_run_dir / "tta_per_class_metrics.csv", index=False)
    per_class_delta.to_csv(output_run_dir / "tta_per_class_delta_vs_e3g.csv", index=False)
    corrected_broken.to_csv(output_run_dir / "tta_corrected_broken.csv", index=False)
    identity_sanity.to_csv(output_run_dir / "tta_identity_sanity.csv", index=False)
    runtime_rows.to_csv(output_run_dir / "tta_runtime_by_view.csv", index=False)
    ensemble_predictions.to_csv(
        output_run_dir / "predictions_top3_family_equal_tta_rot4.csv",
        index=False,
    )
    for family, frame in family_predictions.items():
        frame.to_csv(output_run_dir / f"predictions_{family}_tta_rot4.csv", index=False)

    table_suffix = "_smoke" if run_config["max_samples"] is not None else ""
    ensemble_results.to_csv(tables_dir / f"e3h_tta_rot4_results{table_suffix}.csv", index=False)
    per_class.to_csv(tables_dir / f"e3h_tta_rot4_per_class_metrics{table_suffix}.csv", index=False)
    per_class_delta.to_csv(
        tables_dir / f"e3h_tta_rot4_per_class_delta_vs_e3g{table_suffix}.csv",
        index=False,
    )
    corrected_broken.to_csv(
        tables_dir / f"e3h_tta_rot4_corrected_broken{table_suffix}.csv",
        index=False,
    )
    identity_sanity.to_csv(
        tables_dir / f"e3h_tta_rot4_identity_sanity{table_suffix}.csv",
        index=False,
    )

    save_macro_plot(ensemble_results, figures_dir / f"e3h_tta_rot4_macro_f1{table_suffix}.png")
    save_per_class_delta_plot(
        per_class_delta,
        figures_dir / f"e3h_tta_rot4_per_class_f1_delta{table_suffix}.png",
    )
    matrix = compute_confusion_matrix_from_predictions(ensemble_predictions, class_names)
    save_confusion_matrix_plot(
        matrix,
        class_names,
        figures_dir / f"e3h_tta_rot4_confusion_matrix{table_suffix}.png",
        title="E3h top3 family equal + rot4 validation confusion matrix",
    )


def save_macro_plot(results: pd.DataFrame, output_path: Path) -> None:
    value = float(results.iloc[0]["macro_f1"])
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.bar(
        ["E3g no TTA", "E3h rot4 TTA"],
        [E3G_BASELINE_MACRO_F1, value],
        color=["#64748b", "#0f766e"],
    )
    ax.set_ylabel("Validation macro-F1")
    ax.set_ylim(
        max(0.0, min(E3G_BASELINE_MACRO_F1, value) - 0.03),
        min(1.0, max(E3G_BASELINE_MACRO_F1, value) + 0.03),
    )
    for index, score in enumerate([E3G_BASELINE_MACRO_F1, value]):
        ax.text(index, score, f"{score:.4f}", ha="center", va="bottom")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_per_class_delta_plot(delta: pd.DataFrame, output_path: Path) -> None:
    ordered = delta.sort_values("delta_f1")
    colors = ["#b91c1c" if value < 0 else "#0f766e" for value in ordered["delta_f1"]]
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.barh(ordered["label"], ordered["delta_f1"], color=colors)
    ax.axvline(0.0, color="#111827", linewidth=1)
    ax.set_xlabel("F1 delta vs E3g no-TTA")
    ax.set_title("E3h rot4 per-class F1 delta")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def compute_confusion_matrix_from_predictions(
    predictions: pd.DataFrame,
    class_names: list[str],
) -> list[list[int]]:
    y_true = np.array([class_names.index(label) for label in predictions["true_label"].astype(str)])
    y_pred = np.array([class_names.index(label) for label in predictions["pred_label"].astype(str)])
    return compute_classification_metrics(y_true, y_pred, class_names)["confusion_matrix"]


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
