"""Feature-fusion modules for cached transformer representations."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from dl_final.models.backbones import expected_feature_dim
from dl_final.models.mlp import FeatureMLP


def expected_concat_dim(backbones: Sequence[str]) -> int:
    """Return the feature width produced by direct concatenation."""

    if len(backbones) < 2:
        raise ValueError("Fusion requires at least two backbones.")
    return sum(expected_feature_dim(backbone) for backbone in backbones)


class ConcatenationFusion(nn.Module):
    """Concatenate aligned feature tensors along the feature axis."""

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) < 2:
            raise ValueError("Concatenation fusion requires at least two feature tensors.")
        sample_counts = {int(feature.shape[0]) for feature in features}
        if len(sample_counts) != 1:
            raise ValueError("All feature tensors must have the same batch size.")
        return torch.cat(list(features), dim=1)


class ProjectedWeightedFusion(nn.Module):
    """Project feature blocks and combine them with global softmax weights."""

    def __init__(self, input_dims: Sequence[int], projection_dim: int = 512) -> None:
        super().__init__()
        if len(input_dims) < 2:
            raise ValueError("Weighted fusion requires at least two input dimensions.")
        if projection_dim <= 0:
            raise ValueError("projection_dim must be positive.")
        self.input_dims = tuple(int(dim) for dim in input_dims)
        self.projection_dim = int(projection_dim)
        self.projections = nn.ModuleList(
            nn.Linear(input_dim, self.projection_dim) for input_dim in self.input_dims
        )
        self.logits = nn.Parameter(torch.zeros(len(self.input_dims)))

    def normalized_weights(self) -> torch.Tensor:
        """Return global backbone weights after softmax normalization."""

        return torch.softmax(self.logits, dim=0)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != len(self.input_dims):
            raise ValueError(
                f"Expected {len(self.input_dims)} feature tensors, got {len(features)}."
            )
        projected: list[torch.Tensor] = []
        for index, (feature, projection, expected_dim) in enumerate(
            zip(features, self.projections, self.input_dims, strict=True)
        ):
            if int(feature.shape[1]) != expected_dim:
                raise ValueError(
                    f"Feature tensor {index} has dim {int(feature.shape[1])}, "
                    f"expected {expected_dim}."
                )
            projected.append(projection(feature))
        stacked = torch.stack(projected, dim=1)
        weights = self.normalized_weights().view(1, -1, 1)
        return (stacked * weights).sum(dim=1)


class WeightedSumFusion(nn.Module):
    """Combine same-width feature blocks with global softmax weights."""

    def __init__(self, num_backbones: int, feature_dim: int) -> None:
        super().__init__()
        if num_backbones < 2:
            raise ValueError("Weighted sum fusion requires at least two backbones.")
        if feature_dim <= 0:
            raise ValueError("feature_dim must be positive.")
        self.num_backbones = int(num_backbones)
        self.feature_dim = int(feature_dim)
        self.logits = nn.Parameter(torch.zeros(self.num_backbones))

    def normalized_weights(self) -> torch.Tensor:
        """Return global backbone weights after softmax normalization."""

        return torch.softmax(self.logits, dim=0)

    def forward(self, features: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(features) != self.num_backbones:
            raise ValueError(f"Expected {self.num_backbones} feature tensors, got {len(features)}.")
        for index, feature in enumerate(features):
            if int(feature.shape[1]) != self.feature_dim:
                raise ValueError(
                    f"Feature tensor {index} has dim {int(feature.shape[1])}, "
                    f"expected {self.feature_dim}."
                )
        stacked = torch.stack(list(features), dim=1)
        weights = self.normalized_weights().view(1, -1, 1)
        return (stacked * weights).sum(dim=1)


class WeightedLearnedFusionMLP(nn.Module):
    """Classifier for trainable projected weighted fusion over concatenated features."""

    def __init__(
        self,
        input_dims: Sequence[int],
        num_classes: int,
        *,
        projection_dim: int = 512,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.input_dims = tuple(int(dim) for dim in input_dims)
        self.fusion = ProjectedWeightedFusion(self.input_dims, projection_dim=projection_dim)
        self.classifier = FeatureMLP(
            input_dim=projection_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        parts = torch.split(features, self.input_dims, dim=1)
        fused = self.fusion(parts)
        return self.classifier(fused)

    def normalized_weights(self) -> torch.Tensor:
        """Return learned softmax weights from the fusion layer."""

        return self.fusion.normalized_weights()


class WeightedPCAFusionMLP(nn.Module):
    """Classifier for weighted fusion over precomputed same-width PCA blocks."""

    def __init__(
        self,
        *,
        num_backbones: int,
        feature_dim: int,
        num_classes: int,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_backbones = int(num_backbones)
        self.feature_dim = int(feature_dim)
        self.input_dims = tuple([self.feature_dim] * self.num_backbones)
        self.fusion = WeightedSumFusion(self.num_backbones, self.feature_dim)
        self.classifier = FeatureMLP(
            input_dim=self.feature_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        parts = torch.split(features, self.input_dims, dim=1)
        fused = self.fusion(parts)
        return self.classifier(fused)

    def normalized_weights(self) -> torch.Tensor:
        """Return learned softmax weights from the fusion layer."""

        return self.fusion.normalized_weights()
