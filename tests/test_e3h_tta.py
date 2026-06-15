"""E3h TTA helper tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dl_final.evaluation.tta import (
    assert_probability_matrix,
    average_probabilities,
    expand_tta_policy,
    probabilities_from_frame,
    verify_prediction_alignment,
)


def test_tta_rot4_policy_is_pre_registered_and_deterministic() -> None:
    assert expand_tta_policy("identity") == ["identity"]
    assert expand_tta_policy("tta_rot4") == ["identity", "rot90", "rot180", "rot270"]


def test_average_probabilities_preserves_shape_and_row_normalization() -> None:
    first = np.array([[0.8, 0.2], [0.3, 0.7]], dtype="float32")
    second = np.array([[0.6, 0.4], [0.5, 0.5]], dtype="float32")

    averaged = average_probabilities([first, second])

    assert averaged.shape == (2, 2)
    assert np.allclose(averaged.sum(axis=1), 1.0)
    assert np.allclose(averaged, np.array([[0.7, 0.3], [0.4, 0.6]], dtype="float32"))


def test_probability_validation_rejects_unnormalized_rows() -> None:
    probabilities = np.array([[0.8, 0.4]], dtype="float32")

    with pytest.raises(ValueError, match="sum to 1.0"):
        assert_probability_matrix(probabilities)


def test_prediction_alignment_requires_exact_identity_columns() -> None:
    first = _prediction_frame(["a", "b"])
    second = _prediction_frame(["a", "c"])

    with pytest.raises(ValueError, match="sample_id"):
        verify_prediction_alignment([first, second])


def test_probabilities_from_frame_uses_fixed_class_order() -> None:
    frame = _prediction_frame(["a", "b"])
    frame["prob_akiec"] = [0.25, 0.75]
    frame["prob_bcc"] = [0.75, 0.25]

    probabilities = probabilities_from_frame(frame, ["akiec", "bcc"])

    assert probabilities.tolist() == [[0.25, 0.75], [0.75, 0.25]]


def _prediction_frame(image_ids: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": image_ids,
            "image_id": image_ids,
            "lesion_id": ["l1", "l2"],
            "split": ["val", "val"],
            "true_label": ["akiec", "bcc"],
        }
    )
