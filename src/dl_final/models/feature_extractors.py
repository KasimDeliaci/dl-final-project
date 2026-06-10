"""Classifier-free frozen feature-extractor wrappers."""

from __future__ import annotations

import torch
from torch import nn


class FrozenFeatureExtractor(nn.Module):
    """Thin wrapper that freezes a classifier-free model and returns 2D features."""

    def __init__(
        self,
        name: str,
        model: nn.Module,
        feature_dim: int,
        pooling_policy: str,
        model_id: str,
    ) -> None:
        super().__init__()
        self.name = name
        self.model = model
        self.feature_dim = int(feature_dim)
        self.pooling_policy = pooling_policy
        self.model_id = model_id
        self.freeze()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.model(images)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = torch.flatten(features, start_dim=1)
        return features.float()

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        self.eval()


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of trainable parameters left in a frozen extractor."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

