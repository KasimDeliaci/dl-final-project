"""Deterministic test-time augmentation helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

TTA_VIEW_REGISTRY: dict[str, list[str]] = {
    "identity": ["identity"],
    "tta_rot4": ["identity", "rot90", "rot180", "rot270"],
}

ALIGNMENT_COLUMNS = ("sample_id", "image_id", "lesion_id", "split", "true_label")


@dataclass(frozen=True)
class AlignmentReport:
    """Summary of a prediction-frame alignment check."""

    row_count: int
    aligned_columns: tuple[str, ...] = ALIGNMENT_COLUMNS


def expand_tta_policy(policy_name: str) -> list[str]:
    """Return deterministic view names for a registered TTA policy."""

    try:
        return list(TTA_VIEW_REGISTRY[policy_name])
    except KeyError as exc:
        supported = ", ".join(sorted(TTA_VIEW_REGISTRY))
        raise ValueError(f"Unsupported TTA policy {policy_name!r}. Supported: {supported}") from exc


def average_probabilities(probabilities: list[np.ndarray | torch.Tensor]) -> np.ndarray:
    """Average class-probability matrices and preserve row normalization."""

    if not probabilities:
        raise ValueError("At least one probability matrix is required.")
    arrays = [_to_numpy(matrix) for matrix in probabilities]
    reference_shape = arrays[0].shape
    if len(reference_shape) != 2:
        raise ValueError("Probability matrices must be 2D.")
    for matrix in arrays:
        if matrix.shape != reference_shape:
            raise ValueError("All probability matrices must have the same shape.")
        assert_probability_matrix(matrix)
    averaged = np.stack(arrays, axis=0).mean(axis=0).astype("float64")
    row_sums = np.clip(averaged.sum(axis=1, keepdims=True), a_min=1e-12, a_max=None)
    return (averaged / row_sums).astype("float32")


def assert_probability_matrix(
    probabilities: np.ndarray,
    *,
    expected_rows: int | None = None,
    expected_classes: int | None = None,
    atol: float = 1e-4,
) -> None:
    """Validate finite, row-normalized class probabilities."""

    matrix = np.asarray(probabilities)
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D probability matrix, got shape {matrix.shape}.")
    if expected_rows is not None and int(matrix.shape[0]) != int(expected_rows):
        raise ValueError(f"Expected {expected_rows} rows, got {int(matrix.shape[0])}.")
    if expected_classes is not None and int(matrix.shape[1]) != int(expected_classes):
        raise ValueError(f"Expected {expected_classes} classes, got {int(matrix.shape[1])}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Probability matrix contains NaN or Inf values.")
    if (matrix < -atol).any():
        raise ValueError("Probability matrix contains negative values.")
    row_sums = matrix.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=atol):
        raise ValueError("Probability rows do not sum to 1.0.")


def verify_prediction_alignment(frames: list[pd.DataFrame]) -> AlignmentReport:
    """Verify that prediction frames are aligned on sample identity and label columns."""

    if not frames:
        raise ValueError("At least one prediction frame is required.")
    reference = frames[0].reset_index(drop=True)
    missing = [column for column in ALIGNMENT_COLUMNS if column not in reference.columns]
    if missing:
        raise ValueError(f"Reference frame is missing alignment columns: {missing}")
    for index, frame in enumerate(frames[1:], start=1):
        current = frame.reset_index(drop=True)
        missing = [column for column in ALIGNMENT_COLUMNS if column not in current.columns]
        if missing:
            raise ValueError(f"Frame {index} is missing alignment columns: {missing}")
        if len(current) != len(reference):
            raise ValueError(
                f"Frame {index} has {len(current)} rows; expected {len(reference)} rows."
            )
        for column in ALIGNMENT_COLUMNS:
            if current[column].astype(str).tolist() != reference[column].astype(str).tolist():
                raise ValueError(f"Frame {index} is not aligned on {column!r}.")
    return AlignmentReport(row_count=len(reference))


def probability_columns(class_names: list[str]) -> list[str]:
    """Return standard probability column names for a class order."""

    return [f"prob_{label}" for label in class_names]


def probabilities_from_frame(frame: pd.DataFrame, class_names: list[str]) -> np.ndarray:
    """Read class-probability columns from a prediction frame."""

    columns = probability_columns(class_names)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Prediction frame is missing probability columns: {missing}")
    probabilities = frame.loc[:, columns].to_numpy(dtype="float32")
    assert_probability_matrix(
        probabilities,
        expected_rows=len(frame),
        expected_classes=len(class_names),
    )
    return probabilities


def _to_numpy(matrix: np.ndarray | torch.Tensor) -> np.ndarray:
    if isinstance(matrix, torch.Tensor):
        return matrix.detach().cpu().numpy()
    return np.asarray(matrix)
