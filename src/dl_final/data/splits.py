"""Lesion-aware HAM10000 split generation and verification."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

SPLIT_NAMES = ("train", "val", "test")


@dataclass(frozen=True)
class SplitResult:
    splits: dict[str, pd.DataFrame]
    warnings: list[str]


def create_lesion_aware_splits(
    metadata: pd.DataFrame,
    class_names: list[str],
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
    seed: int = 42,
    label_col: str = "label",
    group_col: str = "lesion_id",
) -> SplitResult:
    validate_split_ratios(train_size, val_size, test_size)
    _require_columns(metadata, ["image_id", label_col, group_col])
    if metadata[group_col].isna().any() or metadata[group_col].astype(str).eq("").any():
        raise ValueError("Cannot create canonical split without complete lesion_id values.")

    ratios = {"train": train_size, "val": val_size, "test": test_size}
    group_table = _group_table(metadata, label_col, group_col)
    assignments = _assign_groups_greedily(group_table, class_names, ratios, seed)

    split_metadata = metadata.copy()
    split_metadata["split"] = split_metadata[group_col].map(assignments)
    if split_metadata["split"].isna().any():
        raise ValueError("Some lesion groups were not assigned to a split.")

    splits: dict[str, pd.DataFrame] = {}
    columns = _split_columns(split_metadata)
    for split_name in SPLIT_NAMES:
        split_frame = (
            split_metadata[split_metadata["split"] == split_name][columns]
            .sort_values("image_id")
            .reset_index(drop=True)
        )
        splits[split_name] = split_frame

    leaks = check_lesion_leakage(splits, group_col=group_col)
    if leaks:
        raise ValueError("Lesion leakage detected: " + "; ".join(leaks))

    warnings = split_distribution_warnings(splits, class_names)
    return SplitResult(splits=splits, warnings=warnings)


def write_split_csvs(splits: dict[str, pd.DataFrame], splits_dir: Path) -> None:
    splits_dir.mkdir(parents=True, exist_ok=True)
    for split_name in SPLIT_NAMES:
        splits[split_name].to_csv(splits_dir / f"{split_name}.csv", index=False)


def read_split_csvs(splits_dir: Path) -> dict[str, pd.DataFrame]:
    splits: dict[str, pd.DataFrame] = {}
    for split_name in SPLIT_NAMES:
        path = splits_dir / f"{split_name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Split file does not exist: {path}")
        splits[split_name] = pd.read_csv(path)
    return splits


def check_lesion_leakage(
    splits: dict[str, pd.DataFrame], group_col: str = "lesion_id"
) -> list[str]:
    group_sets = {
        name: set(frame[group_col].dropna().astype(str))
        for name, frame in splits.items()
        if group_col in frame.columns
    }
    leaks: list[str] = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = sorted(group_sets.get(left, set()) & group_sets.get(right, set()))
        if overlap:
            preview = ", ".join(overlap[:5])
            suffix = "..." if len(overlap) > 5 else ""
            leaks.append(f"{left}/{right} share {len(overlap)} lesions: {preview}{suffix}")
    return leaks


def split_distribution_table(
    splits: dict[str, pd.DataFrame], class_names: list[str]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    totals = {split_name: len(frame) for split_name, frame in splits.items()}
    for split_name in SPLIT_NAMES:
        counts = splits[split_name]["label"].value_counts().reindex(class_names, fill_value=0)
        total = totals[split_name]
        for label, count in counts.items():
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "count": int(count),
                    "percent_within_split": round(float(count) / total * 100, 4)
                    if total
                    else 0.0,
                }
            )
    return pd.DataFrame(rows)


def split_summary_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for split_name in SPLIT_NAMES:
        frame = splits[split_name]
        rows.append(
            {
                "split": split_name,
                "images": len(frame),
                "lesions": int(frame["lesion_id"].nunique()),
                "classes_present": int(frame["label"].nunique()),
            }
        )
    return pd.DataFrame(rows)


def leakage_audit_table(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        left_ids = set(splits[left]["lesion_id"].astype(str))
        right_ids = set(splits[right]["lesion_id"].astype(str))
        overlap = sorted(left_ids & right_ids)
        rows.append(
            {
                "left_split": left,
                "right_split": right,
                "overlap_lesions": len(overlap),
                "overlap_preview": ";".join(overlap[:10]),
            }
        )
    return pd.DataFrame(rows)


def split_distribution_warnings(
    splits: dict[str, pd.DataFrame], class_names: list[str]
) -> list[str]:
    warnings: list[str] = []
    for split_name, frame in splits.items():
        missing = sorted(set(class_names) - set(frame["label"].astype(str)))
        if missing:
            warnings.append(f"{split_name} split is missing classes: {missing}")
    return warnings


def validate_split_ratios(train_size: float, val_size: float, test_size: float) -> None:
    total = train_size + val_size + test_size
    if abs(total - 1.0) > 1e-9:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.4f}")
    if min(train_size, val_size, test_size) <= 0:
        raise ValueError("Split ratios must all be positive.")


def _group_table(metadata: pd.DataFrame, label_col: str, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for lesion_id, group in metadata.groupby(group_col, sort=False):
        label = str(group[label_col].mode().iat[0])
        rows.append(
            {
                "lesion_id": str(lesion_id),
                "label": label,
                "image_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _assign_groups_greedily(
    group_table: pd.DataFrame,
    class_names: list[str],
    ratios: dict[str, float],
    seed: int,
) -> dict[str, str]:
    rng = random.Random(seed)
    assignments: dict[str, str] = {}
    split_class_counts = {
        split_name: {label: 0 for label in class_names} for split_name in SPLIT_NAMES
    }

    for label in class_names:
        label_groups = group_table[group_table["label"] == label].to_dict("records")
        rng.shuffle(label_groups)
        label_groups.sort(key=lambda row: int(row["image_count"]), reverse=True)
        total_images = sum(int(row["image_count"]) for row in label_groups)
        targets = {split: total_images * ratios[split] for split in SPLIT_NAMES}

        for row in label_groups:
            split_name = min(
                SPLIT_NAMES,
                key=lambda split: (
                    split_class_counts[split][label] / targets[split]
                    if targets[split] > 0
                    else 0,
                    split_class_counts[split][label],
                ),
            )
            lesion_id = str(row["lesion_id"])
            assignments[lesion_id] = split_name
            split_class_counts[split_name][label] += int(row["image_count"])

    return assignments


def _split_columns(frame: pd.DataFrame) -> list[str]:
    preferred = ["sample_id", "image_id", "lesion_id", "label", "split", "image_path"]
    remaining = [
        col
        for col in ["dx", "dx_type", "age", "sex", "localization", "dataset"]
        if col in frame.columns
    ]
    return [col for col in preferred if col in frame.columns] + remaining


def _require_columns(frame: pd.DataFrame, columns: list[str]) -> None:
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

