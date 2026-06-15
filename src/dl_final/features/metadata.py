"""Train-only HAM10000 metadata preprocessing for E3c diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

ALLOWED_METADATA_FIELDS = ("age", "sex", "localization")
FORBIDDEN_METADATA_FIELDS = (
    "dx",
    "dx_type",
    "dataset",
    "image_id",
    "sample_id",
    "lesion_id",
    "image_path",
)


@dataclass(frozen=True)
class MetadataTransformResult:
    """Transformed metadata matrix plus transform diagnostics."""

    features: np.ndarray
    metadata: dict[str, Any]


@dataclass
class MetadataPreprocessor:
    """Fit train-only preprocessing for HAM10000 age, sex, and localization."""

    age_median_: float | None = None
    age_mean_: float | None = None
    age_scale_: float | None = None
    age_var_: float | None = None
    sex_categories_: tuple[str, ...] = ()
    localization_categories_: tuple[str, ...] = ()
    column_names_: tuple[str, ...] = ()
    train_missing_counts_: dict[str, int] | None = None
    train_rows_: int | None = None

    def fit(self, frame: pd.DataFrame) -> MetadataPreprocessor:
        """Fit imputation, scaling, and categorical vocabularies from train rows only."""

        _require_allowed_columns(frame)
        age = _coerce_age(frame["age"])
        median = float(age.median(skipna=True))
        if np.isnan(median):
            median = 0.0
        age_imputed = age.fillna(median).to_numpy(dtype="float32").reshape(-1, 1)
        age_mean = float(age_imputed.mean())
        age_var = float(age_imputed.var())
        age_scale = float(np.sqrt(age_var)) if age_var > 0 else 1.0

        sex = _normalize_category(frame["sex"])
        localization = _normalize_category(frame["localization"])

        self.age_median_ = median
        self.age_mean_ = age_mean
        self.age_var_ = age_var
        self.age_scale_ = age_scale
        self.sex_categories_ = tuple(sorted(sex.unique().tolist()))
        self.localization_categories_ = tuple(sorted(localization.unique().tolist()))
        self.column_names_ = (
            ("age_scaled",)
            + tuple(f"sex={value}" for value in self.sex_categories_)
            + tuple(f"localization={value}" for value in self.localization_categories_)
        )
        self.train_missing_counts_ = {
            "age": int(age.isna().sum()),
            "sex": int((sex == "unknown").sum()),
            "localization": int((localization == "unknown").sum()),
        }
        self.train_rows_ = int(len(frame))
        return self

    def transform(self, frame: pd.DataFrame) -> MetadataTransformResult:
        """Transform rows without mutating fitted train-only state."""

        self._check_fitted()
        _require_allowed_columns(frame)
        age = _coerce_age(frame["age"]).fillna(float(self.age_median_))
        age_scaled = ((age.to_numpy(dtype="float32") - float(self.age_mean_)) / float(
            self.age_scale_
        )).reshape(-1, 1)

        sex = _normalize_category(frame["sex"])
        localization = _normalize_category(frame["localization"])
        sex_encoded, sex_unseen = _one_hot_with_train_vocabulary(
            sex,
            self.sex_categories_,
        )
        localization_encoded, localization_unseen = _one_hot_with_train_vocabulary(
            localization,
            self.localization_categories_,
        )

        features = np.concatenate(
            [age_scaled, sex_encoded, localization_encoded],
            axis=1,
        ).astype("float32")
        if not np.isfinite(features).all():
            raise ValueError("Metadata features contain NaN or Inf values.")
        metadata = {
            "fit_split": "train",
            "input_fields": list(ALLOWED_METADATA_FIELDS),
            "excluded_fields": list(FORBIDDEN_METADATA_FIELDS),
            "output_dim": int(features.shape[1]),
            "column_names": list(self.column_names_),
            "rows_transformed": int(features.shape[0]),
            "age": {
                "median_imputation_value": float(self.age_median_),
                "mean": float(self.age_mean_),
                "scale": float(self.age_scale_),
                "var": float(self.age_var_),
            },
            "categorical": {
                "sex_categories": list(self.sex_categories_),
                "localization_categories": list(self.localization_categories_),
                "unseen_counts": {
                    "sex": int(sex_unseen),
                    "localization": int(localization_unseen),
                },
            },
            "train_missing_counts": self.train_missing_counts_ or {},
            "train_rows_fit": int(self.train_rows_ or 0),
        }
        return MetadataTransformResult(features=features, metadata=metadata)

    def fit_transform(self, frame: pd.DataFrame) -> MetadataTransformResult:
        """Fit on the supplied train rows and transform them."""

        return self.fit(frame).transform(frame)

    def to_metadata(self) -> dict[str, Any]:
        """Return fitted preprocessing metadata for artifact writing."""

        self._check_fitted()
        return {
            "fit_split": "train",
            "input_fields": list(ALLOWED_METADATA_FIELDS),
            "excluded_fields": list(FORBIDDEN_METADATA_FIELDS),
            "output_dim": len(self.column_names_),
            "column_names": list(self.column_names_),
            "age": {
                "median_imputation_value": float(self.age_median_),
                "mean": float(self.age_mean_),
                "scale": float(self.age_scale_),
                "var": float(self.age_var_),
            },
            "categorical": {
                "sex_categories": list(self.sex_categories_),
                "localization_categories": list(self.localization_categories_),
            },
            "train_missing_counts": self.train_missing_counts_ or {},
            "train_rows_fit": int(self.train_rows_ or 0),
        }

    def _check_fitted(self) -> None:
        if self.age_median_ is None or not self.column_names_:
            raise RuntimeError("MetadataPreprocessor must be fit before transform.")


def metadata_frame_from_split(split_csv: str) -> pd.DataFrame:
    """Read a split CSV and return the row-aligned metadata input columns."""

    frame = pd.read_csv(split_csv)
    _require_allowed_columns(frame)
    return frame.loc[:, list(ALLOWED_METADATA_FIELDS)].copy()


def _require_allowed_columns(frame: pd.DataFrame) -> None:
    missing = [field for field in ALLOWED_METADATA_FIELDS if field not in frame.columns]
    if missing:
        raise ValueError(f"Metadata frame is missing required fields: {missing}")


def _coerce_age(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _normalize_category(series: pd.Series) -> pd.Series:
    normalized = series.fillna("unknown").astype(str).str.strip().str.lower()
    return normalized.mask(normalized == "", "unknown")


def _one_hot_with_train_vocabulary(
    values: pd.Series,
    categories: tuple[str, ...],
) -> tuple[np.ndarray, int]:
    category_to_index = {category: index for index, category in enumerate(categories)}
    unknown_index = category_to_index.get("unknown")
    encoded = np.zeros((len(values), len(categories)), dtype="float32")
    unseen_count = 0
    for row_index, value in enumerate(values.tolist()):
        index = category_to_index.get(value)
        if index is None:
            unseen_count += 1
            index = unknown_index
        if index is not None:
            encoded[row_index, index] = 1.0
    return encoded, unseen_count
