"""Training and validation loops for cached-feature MLP baselines."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from dl_final.evaluation.metrics import compute_classification_metrics
from dl_final.training.early_stopping import EarlyStopping


def train_mlp_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    class_names: list[str],
    device: torch.device,
    epochs: int,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    early_stopping_patience: int,
) -> tuple[nn.Module, pd.DataFrame, dict[str, Any]]:
    """Train an MLP and restore the best validation macro-F1 state."""

    model.to(device)
    stopper = EarlyStopping(patience=early_stopping_patience, mode="max")
    best_state = deepcopy(model.state_dict())
    best_metrics: dict[str, Any] = {}
    rows: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        train_loss, train_true, train_pred = _run_train_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_true, val_pred, _ = predict_with_loss(model, val_loader, criterion, device)
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
            }
        )
        if stopper.best_score is None or val_metrics["macro_f1"] > stopper.best_score:
            best_state = deepcopy(model.state_dict())
            best_metrics = dict(val_metrics)
        if stopper.step(val_metrics["macro_f1"], epoch):
            break

    model.load_state_dict(best_state)
    best_metrics["best_epoch"] = stopper.best_epoch
    return model, pd.DataFrame(rows), best_metrics


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    *,
    class_names: list[str],
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Return metrics plus true labels, predictions, and probabilities."""

    y_true, y_pred, probabilities = predict(model, loader, device)
    metrics = compute_classification_metrics(y_true, y_pred, class_names)
    return metrics, y_true, y_pred, probabilities


@torch.no_grad()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    true_batches: list[torch.Tensor] = []
    pred_batches: list[torch.Tensor] = []
    prob_batches: list[torch.Tensor] = []
    for features, labels in loader:
        logits = model(features.to(device))
        probabilities = torch.softmax(logits, dim=1).detach().cpu()
        true_batches.append(labels.cpu())
        pred_batches.append(probabilities.argmax(dim=1))
        prob_batches.append(probabilities)
    return (
        torch.cat(true_batches).numpy(),
        torch.cat(pred_batches).numpy(),
        torch.cat(prob_batches).numpy(),
    )


@torch.no_grad()
def predict_with_loss(
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
    prob_batches: list[torch.Tensor] = []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        logits = model(features)
        loss = criterion(logits, labels)
        probabilities = torch.softmax(logits, dim=1).detach().cpu()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        true.extend(labels.cpu().tolist())
        pred.extend(probabilities.argmax(dim=1).tolist())
        prob_batches.append(probabilities)
    probabilities = torch.cat(prob_batches).numpy() if prob_batches else np.empty((0, 0))
    return total_loss / max(total_samples, 1), true, pred, probabilities


def _run_train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, list[int], list[int]]:
    model.train()
    total_loss = 0.0
    total_samples = 0
    true: list[int] = []
    pred: list[int] = []
    for features, labels in loader:
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(features)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = int(labels.shape[0])
        total_loss += float(loss.detach().cpu()) * batch_size
        total_samples += batch_size
        true.extend(labels.cpu().tolist())
        pred.extend(logits.argmax(dim=1).detach().cpu().tolist())
    return total_loss / max(total_samples, 1), true, pred
