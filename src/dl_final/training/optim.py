"""Optimizer construction helpers."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def build_optimizer(
    parameters: Iterable[torch.nn.Parameter],
    *,
    optimizer_name: str,
    learning_rate: float,
    weight_decay: float,
    momentum: float = 0.9,
) -> torch.optim.Optimizer:
    normalized = optimizer_name.lower()
    if normalized == "adamw":
        return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "adam":
        return torch.optim.Adam(parameters, lr=learning_rate, weight_decay=weight_decay)
    if normalized == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=learning_rate,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )
    raise ValueError("Unsupported optimizer. Expected one of: adamw, adam, sgd.")

