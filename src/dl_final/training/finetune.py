"""Partial transformer fine-tuning and fine-tuned feature cache extraction."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dl_final.evaluation.metrics import compute_classification_metrics
from dl_final.evaluation.reports import prediction_frame, write_run_artifacts
from dl_final.features.cache import save_backbone_manifest
from dl_final.features.extract import extract_and_cache_backbone
from dl_final.models.backbones import (
    apply_finetuning_policy,
    backbone_alias,
    build_classification_backbone,
    build_finetuned_feature_extractor,
)
from dl_final.training.early_stopping import EarlyStopping


def labels_from_image_loader(loader: DataLoader) -> list[int]:
    """Read integer labels from a HAM10000 image dataloader's split frame."""

    dataset = loader.dataset
    frame = getattr(dataset, "frame", None)
    label_to_index = getattr(dataset, "label_to_index", None)
    if frame is None or label_to_index is None:
        raise ValueError("Expected HAM10000ImageDataset-backed dataloader.")
    return [int(label_to_index[str(label)]) for label in frame["label"].astype(str).tolist()]


def class_weights_from_labels(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = torch.bincount(torch.tensor(labels, dtype=torch.long), minlength=num_classes).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    weights[counts == 0] = 0.0
    return weights


def finetune_backbone(
    *,
    backbone: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_names: list[str],
    device: torch.device,
    checkpoint_dir: str | Path,
    seed: int,
    epochs: int,
    backbone_learning_rate: float,
    head_learning_rate: float,
    weight_decay: float,
    early_stopping_patience: int,
    policy: str | None,
    mixed_precision: bool,
    pretrained: bool = True,
    class_weighting: bool = True,
    run_root: str | Path = "artifacts/runs",
    limit_per_split: int | None = None,
    config: dict[str, Any] | None = None,
) -> tuple[Path, Path, dict[str, Any]]:
    """Fine-tune one transformer and save the best validation macro-F1 checkpoint."""

    model = build_classification_backbone(
        backbone,
        num_classes=len(class_names),
        pretrained=pretrained,
    )
    trainability = apply_finetuning_policy(model, backbone, policy=policy)
    model.to(device)

    weights = None
    if class_weighting:
        train_labels = labels_from_image_loader(train_loader)
        weights = class_weights_from_labels(train_labels, len(class_names))
        weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        _parameter_groups(
            model,
            backbone_learning_rate=backbone_learning_rate,
            head_learning_rate=head_learning_rate,
        ),
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))

    started = perf_counter()
    model, history, best_metrics = train_image_classifier(
        model,
        train_loader,
        val_loader,
        class_names=class_names,
        device=device,
        epochs=epochs,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        early_stopping_patience=early_stopping_patience,
        mixed_precision=mixed_precision,
    )
    runtime_seconds = round(perf_counter() - started, 4)
    metrics, y_true, y_pred, probabilities = evaluate_image_classifier(
        model,
        val_loader,
        class_names=class_names,
        device=device,
        criterion=criterion,
    )
    metrics["best_epoch"] = best_metrics.get("best_epoch")

    checkpoint_root = Path(checkpoint_dir) / backbone
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_root / "best.pt"
    checkpoint_metadata = {
        "experiment_id": "E3",
        "backbone": backbone,
        "seed": seed,
        "selection_metric": "validation_macro_f1",
        "best_validation_macro_f1": metrics["macro_f1"],
        "best_epoch": metrics.get("best_epoch"),
        "test_policy": "not_used_in_sprint4",
        "runtime_seconds": runtime_seconds,
        "unfreeze_policy": trainability,
        "class_weighting": class_weighting,
        "backbone_learning_rate": backbone_learning_rate,
        "head_learning_rate": head_learning_rate,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "early_stopping_patience": early_stopping_patience,
        "pretrained": pretrained,
        "limit_per_split": limit_per_split,
        "config": config or {},
    }
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "checkpoint_metadata": checkpoint_metadata,
        },
        checkpoint_path,
    )

    run_id = f"s4_finetune_{backbone_alias(backbone)}_seed{seed}"
    if limit_per_split is not None:
        run_id = f"{run_id}_limit{limit_per_split}"
    run_dir = Path(run_root) / run_id
    run_config = {
        "run_id": run_id,
        "experiment_id": "E3",
        "seed": seed,
        "feature_source": "finetuned",
        "backbone": backbone,
        "backbones": [backbone],
        "fusion_method": "finetune_head",
        "selection_metric": "validation_macro_f1",
        "test_policy": "not_used_in_sprint4",
        "checkpoint_path": str(checkpoint_path),
        "runtime_seconds": runtime_seconds,
        **checkpoint_metadata,
    }
    write_run_artifacts(
        run_dir,
        run_config=run_config,
        history=history,
        metrics=metrics,
        predictions=image_prediction_frame(
            val_loader,
            y_true=y_true,
            y_pred=y_pred,
            probabilities=probabilities,
            class_names=class_names,
        ),
        class_names=class_names,
        backbone=backbone,
    )
    (run_dir / "checkpoint_metadata.json").write_text(
        json.dumps(checkpoint_metadata, indent=2),
        encoding="utf-8",
    )
    return checkpoint_path, run_dir, checkpoint_metadata


def extract_finetuned_feature_cache(
    *,
    backbone: str,
    checkpoint_path: str | Path,
    loaders: dict[str, DataLoader],
    output_dir: str | Path,
    class_names: list[str],
    seed: int,
    device: torch.device,
    mixed_precision: bool,
    config: dict[str, Any] | None = None,
) -> Path:
    """Extract train/validation feature caches from a selected fine-tuned checkpoint."""

    model = build_finetuned_feature_extractor(
        backbone,
        checkpoint_path=str(checkpoint_path),
        num_classes=len(class_names),
        pretrained=False,
    )
    caches = extract_and_cache_backbone(
        model=model,
        backbone=backbone,
        loaders=loaders,
        output_dir=output_dir,
        class_names=class_names,
        feature_source="finetuned",
        seed=seed,
        device=device,
        mixed_precision=mixed_precision,
        config=config,
    )
    return save_backbone_manifest(output_dir, caches)


def train_image_classifier(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    class_names: list[str],
    device: torch.device,
    epochs: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    early_stopping_patience: int,
    mixed_precision: bool,
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any]]:
    """Train with early stopping and restore the best validation macro-F1 state."""

    stopper = EarlyStopping(patience=early_stopping_patience, mode="max")
    best_state = deepcopy(model.state_dict())
    best_metrics: dict[str, Any] = {}
    rows: list[dict[str, float | int | None]] = []
    device_type = device.type if device.type in {"cuda", "cpu", "mps"} else "cpu"
    autocast_enabled = mixed_precision and device.type in {"cuda", "mps"}
    scaler = torch.amp.GradScaler("cuda", enabled=mixed_precision and device.type == "cuda")

    for epoch in range(1, epochs + 1):
        train_loss, train_true, train_pred = _run_image_train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            device_type=device_type,
            autocast_enabled=autocast_enabled,
            scaler=scaler,
        )
        val_loss, val_true, val_pred, _ = predict_images_with_loss(
            model,
            val_loader,
            criterion,
            device,
        )
        train_metrics = compute_classification_metrics(train_true, train_pred, class_names)
        val_metrics = compute_classification_metrics(val_true, val_pred, class_names)
        rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "backbone_lr": float(optimizer.param_groups[0]["lr"]),
                "head_lr": float(optimizer.param_groups[-1]["lr"]),
            }
        )
        if stopper.best_score is None or val_metrics["macro_f1"] > stopper.best_score:
            best_state = deepcopy(model.state_dict())
            best_metrics = dict(val_metrics)
        if stopper.step(val_metrics["macro_f1"], epoch):
            break
        if scheduler is not None:
            scheduler.step()

    model.load_state_dict(best_state)
    best_metrics["best_epoch"] = stopper.best_epoch
    return model, pd.DataFrame(rows), best_metrics


@torch.no_grad()
def evaluate_image_classifier(
    model: nn.Module,
    loader: DataLoader,
    *,
    class_names: list[str],
    device: torch.device,
    criterion: nn.Module,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    _, true, pred, probabilities = predict_images_with_loss(model, loader, criterion, device)
    metrics = compute_classification_metrics(true, pred, class_names)
    return metrics, np.asarray(true), np.asarray(pred), probabilities


@torch.no_grad()
def predict_images_with_loss(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, list[int], list[int], np.ndarray]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    true: list[int] = []
    pred: list[int] = []
    probability_batches: list[torch.Tensor] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1).detach().cpu()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        true.extend(labels.detach().cpu().tolist())
        pred.extend(probabilities.argmax(dim=1).tolist())
        probability_batches.append(probabilities)
    probabilities_array = (
        torch.cat(probability_batches).numpy() if probability_batches else np.empty((0, 0))
    )
    return total_loss / max(total_samples, 1), true, pred, probabilities_array


def image_prediction_frame(
    loader: DataLoader,
    *,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    class_names: list[str],
) -> pd.DataFrame:
    dataset = loader.dataset
    frame = getattr(dataset, "frame", None)
    if frame is None:
        raise ValueError("Expected HAM10000ImageDataset-backed dataloader.")
    cache_like = type(
        "PredictionCache",
        (),
        {
            "sample_ids": frame.get("sample_id", frame["image_id"]).astype(str).tolist(),
            "image_ids": frame["image_id"].astype(str).tolist(),
            "lesion_ids": frame.get("lesion_id", pd.Series([""] * len(frame))).astype(str).tolist(),
            "split_names": frame.get("split", pd.Series(["val"] * len(frame))).astype(str).tolist(),
        },
    )()
    return prediction_frame(
        cache=cache_like,
        y_true=y_true,
        y_pred=y_pred,
        probabilities=probabilities,
        class_names=class_names,
    )


def _run_image_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    device_type: str,
    autocast_enabled: bool,
    scaler: torch.amp.GradScaler,
) -> tuple[float, list[int], list[int]]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    true: list[int] = []
    pred: list[int] = []
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device_type, enabled=autocast_enabled):
            logits = model(images)
            loss = criterion(logits, labels)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        true.extend(labels.detach().cpu().tolist())
        pred.extend(logits.detach().argmax(dim=1).cpu().tolist())
    return total_loss / max(total_samples, 1), true, pred


def _parameter_groups(
    model: nn.Module,
    *,
    backbone_learning_rate: float,
    head_learning_rate: float,
) -> list[dict[str, Any]]:
    head_keywords = (".head.", "head.", ".fc_norm.", "fc_norm.", ".norm.", "norm.")
    backbone_parameters: list[nn.Parameter] = []
    head_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if any(keyword in name for keyword in head_keywords):
            head_parameters.append(parameter)
        else:
            backbone_parameters.append(parameter)
    groups: list[dict[str, Any]] = []
    if backbone_parameters:
        groups.append({"params": backbone_parameters, "lr": backbone_learning_rate})
    if head_parameters:
        groups.append({"params": head_parameters, "lr": head_learning_rate})
    if not groups:
        raise ValueError("Fine-tuning policy produced no trainable parameters.")
    return groups
