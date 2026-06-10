"""MLP classifiers for cached transformer feature vectors."""

from __future__ import annotations

from torch import nn


class FeatureMLP(nn.Module):
    """Small fully connected classifier for single-backbone frozen features."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dims: list[int] | tuple[int, ...] = (512, 256),
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        previous = int(input_dim)
        for hidden in hidden_dims:
            layers.extend(
                [
                    nn.Linear(previous, int(hidden)),
                    nn.BatchNorm1d(int(hidden)),
                    nn.ReLU(inplace=True),
                    nn.Dropout(float(dropout)),
                ]
            )
            previous = int(hidden)
        layers.append(nn.Linear(previous, int(num_classes)))
        self.net = nn.Sequential(*layers)

    def forward(self, features):
        return self.net(features)

