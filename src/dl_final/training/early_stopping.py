"""Patience-based early stopping."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EarlyStopping:
    patience: int
    mode: str = "max"
    min_delta: float = 0.0
    best_score: float | None = None
    best_epoch: int = 0
    bad_epochs: int = 0

    def step(self, score: float, epoch: int) -> bool:
        if self.best_score is None or self._is_improvement(score):
            self.best_score = float(score)
            self.best_epoch = int(epoch)
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "max":
            return score > self.best_score + self.min_delta
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        raise ValueError(f"Unsupported early-stopping mode: {self.mode}")

