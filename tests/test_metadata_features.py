"""Tests for train-only HAM10000 metadata preprocessing."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dl_final.features.metadata import MetadataPreprocessor


def test_metadata_preprocessor_fits_train_only_state() -> None:
    train = pd.DataFrame(
        {
            "age": [10, 20, None, 40],
            "sex": ["male", "female", "unknown", "female"],
            "localization": ["back", "face", "unknown", "back"],
        }
    )
    val = pd.DataFrame(
        {
            "age": [None, 80],
            "sex": ["nonbinary", "female"],
            "localization": ["ear", "face"],
        }
    )

    preprocessor = MetadataPreprocessor().fit(train)
    train_result = preprocessor.transform(train)
    val_result = preprocessor.transform(val)

    assert preprocessor.age_median_ == 20.0
    assert train_result.features.shape == (4, len(preprocessor.column_names_))
    assert val_result.features.shape == (2, len(preprocessor.column_names_))
    assert np.isfinite(train_result.features).all()
    assert np.isfinite(val_result.features).all()
    assert "sex=nonbinary" not in preprocessor.column_names_
    assert "localization=ear" not in preprocessor.column_names_
    assert val_result.metadata["categorical"]["unseen_counts"] == {
        "sex": 1,
        "localization": 1,
    }


def test_validation_missing_category_does_not_expand_without_train_unknown() -> None:
    train = pd.DataFrame(
        {
            "age": [30, 50],
            "sex": ["male", "female"],
            "localization": ["back", "face"],
        }
    )
    val = pd.DataFrame(
        {
            "age": [None],
            "sex": [None],
            "localization": [None],
        }
    )

    preprocessor = MetadataPreprocessor().fit(train)
    before_columns = preprocessor.column_names_
    transformed = preprocessor.transform(val)

    assert preprocessor.column_names_ == before_columns
    assert transformed.features.shape == (1, len(before_columns))
    assert transformed.metadata["categorical"]["unseen_counts"] == {
        "sex": 1,
        "localization": 1,
    }


def test_metadata_preprocessor_rejects_missing_required_columns() -> None:
    frame = pd.DataFrame({"age": [10], "sex": ["male"]})

    with pytest.raises(ValueError, match="missing required fields"):
        MetadataPreprocessor().fit(frame)
