"""Lightweight metadata-conditioned MLP classifiers for E3d ablations."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from dl_final.models.mlp import FeatureMLP


class MetadataGatedBackboneMLP(nn.Module):
    """Use metadata to produce sample-level gates for each image backbone block."""

    def __init__(
        self,
        *,
        input_dims: Sequence[int],
        metadata_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.3,
        gate_hidden_dim: int = 32,
    ) -> None:
        super().__init__()
        if not input_dims:
            raise ValueError("input_dims must contain at least one image feature block.")
        self.input_dims = tuple(int(dim) for dim in input_dims)
        self.image_dim = int(sum(self.input_dims))
        self.metadata_dim = int(metadata_dim)
        self.total_input_dim = self.image_dim + self.metadata_dim
        self.gate = nn.Sequential(
            nn.Linear(self.metadata_dim, int(gate_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(gate_hidden_dim), len(self.input_dims)),
            nn.Sigmoid(),
        )
        self.classifier = FeatureMLP(
            input_dim=self.total_input_dim,
            num_classes=num_classes,
            hidden_dims=list(hidden_dims),
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        image_features, metadata = self._split_input(features)
        gates = self.gate(metadata)
        gated_blocks = [
            block * gates[:, index : index + 1]
            for index, block in enumerate(torch.split(image_features, self.input_dims, dim=1))
        ]
        return self.classifier(torch.cat([*gated_blocks, metadata], dim=1))

    @torch.no_grad()
    def gate_values(self, features: torch.Tensor) -> torch.Tensor:
        """Return per-backbone gates for a batch of concatenated image+metadata features."""

        _, metadata = self._split_input(features)
        return self.gate(metadata)

    def _split_input(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[1] != self.total_input_dim:
            raise ValueError(
                f"Expected input_dim={self.total_input_dim}, got {int(features.shape[1])}."
            )
        return features[:, : self.image_dim], features[:, self.image_dim :]


class MetadataFiLMMLP(nn.Module):
    """Bounded FiLM-style metadata conditioning over concatenated image features."""

    def __init__(
        self,
        *,
        image_dim: int,
        metadata_dim: int,
        num_classes: int,
        hidden_dims: Sequence[int] = (512, 256),
        dropout: float = 0.3,
        film_hidden_dim: int = 64,
        modulation_scale: float = 0.1,
    ) -> None:
        super().__init__()
        self.image_dim = int(image_dim)
        self.metadata_dim = int(metadata_dim)
        self.total_input_dim = self.image_dim + self.metadata_dim
        self.modulation_scale = float(modulation_scale)
        self.film = nn.Sequential(
            nn.Linear(self.metadata_dim, int(film_hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Linear(int(film_hidden_dim), self.image_dim * 2),
        )
        self.classifier = FeatureMLP(
            input_dim=self.total_input_dim,
            num_classes=num_classes,
            hidden_dims=list(hidden_dims),
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        image_features, metadata = self._split_input(features)
        gamma_beta = torch.tanh(self.film(metadata))
        gamma, beta = torch.split(gamma_beta, self.image_dim, dim=1)
        conditioned = (
            image_features * (1.0 + self.modulation_scale * gamma)
            + self.modulation_scale * beta
        )
        return self.classifier(torch.cat([conditioned, metadata], dim=1))

    def _split_input(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[1] != self.total_input_dim:
            raise ValueError(
                f"Expected input_dim={self.total_input_dim}, got {int(features.shape[1])}."
            )
        return features[:, : self.image_dim], features[:, self.image_dim :]


class MetadataTwoBranchMLP(nn.Module):
    """Process image features and metadata in separate branches before hidden fusion."""

    def __init__(
        self,
        *,
        image_dim: int,
        metadata_dim: int,
        num_classes: int,
        image_hidden_dim: int = 512,
        metadata_hidden_dim: int = 64,
        fusion_hidden_dims: Sequence[int] = (256,),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.image_dim = int(image_dim)
        self.metadata_dim = int(metadata_dim)
        self.total_input_dim = self.image_dim + self.metadata_dim
        self.image_branch = _branch(self.image_dim, int(image_hidden_dim), dropout)
        self.metadata_branch = _branch(self.metadata_dim, int(metadata_hidden_dim), dropout)
        self.classifier = FeatureMLP(
            input_dim=int(image_hidden_dim) + int(metadata_hidden_dim),
            num_classes=num_classes,
            hidden_dims=list(fusion_hidden_dims),
            dropout=dropout,
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        image_features, metadata = self._split_input(features)
        image_hidden = self.image_branch(image_features)
        metadata_hidden = self.metadata_branch(metadata)
        return self.classifier(torch.cat([image_hidden, metadata_hidden], dim=1))

    def _split_input(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if features.shape[1] != self.total_input_dim:
            raise ValueError(
                f"Expected input_dim={self.total_input_dim}, got {int(features.shape[1])}."
            )
        return features[:, : self.image_dim], features[:, self.image_dim :]


def _branch(input_dim: int, output_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(int(input_dim), int(output_dim)),
        nn.BatchNorm1d(int(output_dim)),
        nn.ReLU(inplace=True),
        nn.Dropout(float(dropout)),
    )
