"""HAM10000 metadata loading, image resolution, and audit helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")
REQUIRED_METADATA_COLUMNS = ("image_id", "dx")


@dataclass(frozen=True)
class DatasetAudit:
    metadata_path: Path
    raw_dir: Path
    image_rows: int
    unique_image_ids: int
    duplicate_image_ids: list[str]
    missing_images: list[str]
    unreferenced_images: list[str]
    missing_labels: int
    missing_lesion_ids: int
    unique_lesion_ids: int
    lesion_id_available: bool
    unreadable_images: list[str]
    class_distribution: pd.DataFrame
    lesion_class_distribution: pd.DataFrame

    @property
    def has_blocking_errors(self) -> bool:
        return bool(
            self.duplicate_image_ids
            or self.missing_images
            or self.missing_labels
            or self.missing_lesion_ids
            or self.unreadable_images
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata_path": str(self.metadata_path),
            "raw_dir": str(self.raw_dir),
            "image_rows": self.image_rows,
            "unique_image_ids": self.unique_image_ids,
            "duplicate_image_ids": self.duplicate_image_ids,
            "missing_images": self.missing_images,
            "unreferenced_images_count": len(self.unreferenced_images),
            "missing_labels": self.missing_labels,
            "missing_lesion_ids": self.missing_lesion_ids,
            "unique_lesion_ids": self.unique_lesion_ids,
            "lesion_id_available": self.lesion_id_available,
            "unreadable_images": self.unreadable_images,
            "has_blocking_errors": self.has_blocking_errors,
        }


def dataset_paths(dataset_config: dict[str, Any]) -> tuple[Path, Path, Path, Path]:
    metadata_dir = Path(str(dataset_config["metadata_dir"]))
    metadata_path = metadata_dir / str(dataset_config["metadata_filename"])
    raw_dir = Path(str(dataset_config["raw_dir"]))
    processed_dir = Path(str(dataset_config["processed_dir"]))
    splits_dir = Path(str(dataset_config["splits_dir"]))
    return metadata_path, raw_dir, processed_dir, splits_dir


def load_ham10000_metadata(metadata_path: Path, class_names: list[str]) -> pd.DataFrame:
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file does not exist: {metadata_path}")

    metadata = pd.read_csv(metadata_path)
    missing_columns = [col for col in REQUIRED_METADATA_COLUMNS if col not in metadata.columns]
    if missing_columns:
        raise ValueError(f"Metadata is missing required columns: {missing_columns}")

    normalized = metadata.copy()
    normalized["image_id"] = normalized["image_id"].astype(str)
    normalized["label"] = normalized["dx"].astype(str)
    if "lesion_id" in normalized.columns:
        normalized["lesion_id"] = normalized["lesion_id"].astype(str)
    else:
        normalized["lesion_id"] = ""

    unknown_labels = sorted(set(normalized["label"].dropna()) - set(class_names))
    if unknown_labels:
        raise ValueError(
            f"Metadata contains labels outside configured class_names: {unknown_labels}"
        )

    normalized.insert(0, "sample_id", normalized["image_id"])
    return normalized


def attach_image_paths(metadata: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    if not raw_dir.exists():
        raise FileNotFoundError(f"HAM10000 raw image directory does not exist: {raw_dir}")

    image_index = index_images(raw_dir)
    with_paths = metadata.copy()
    with_paths["image_path"] = with_paths["image_id"].map(image_index).fillna("")
    return with_paths


def index_images(raw_dir: Path) -> dict[str, str]:
    image_index: dict[str, str] = {}
    for extension in IMAGE_EXTENSIONS:
        for path in raw_dir.rglob(f"*{extension}"):
            if _is_segmentation_path(path):
                continue
            image_index.setdefault(path.stem, str(path))
        for path in raw_dir.rglob(f"*{extension.upper()}"):
            if _is_segmentation_path(path):
                continue
            image_index.setdefault(path.stem, str(path))
    return image_index


def _is_segmentation_path(path: Path) -> bool:
    return any("segment" in part.lower() for part in path.parts)


def audit_metadata(
    metadata: pd.DataFrame,
    metadata_path: Path,
    raw_dir: Path,
    class_names: list[str],
    image_open_sample: int = 16,
) -> DatasetAudit:
    duplicate_ids = (
        metadata.loc[metadata["image_id"].duplicated(keep=False), "image_id"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    missing_images = metadata.loc[metadata["image_path"].eq(""), "image_id"].astype(str).tolist()
    missing_labels = int(metadata["label"].isna().sum())
    missing_lesion_ids = int(metadata["lesion_id"].isna().sum() + metadata["lesion_id"].eq("").sum())
    unique_lesion_ids = int(metadata["lesion_id"].replace("", pd.NA).dropna().nunique())

    referenced = set(metadata["image_id"].astype(str))
    all_images = index_images(raw_dir)
    unreferenced_images = sorted(set(all_images) - referenced)

    class_distribution = class_distribution_table(metadata, class_names)
    lesion_class_distribution = lesion_class_distribution_table(metadata, class_names)
    unreadable_images = _sample_unreadable_images(metadata, image_open_sample)

    return DatasetAudit(
        metadata_path=metadata_path,
        raw_dir=raw_dir,
        image_rows=len(metadata),
        unique_image_ids=int(metadata["image_id"].nunique()),
        duplicate_image_ids=sorted(duplicate_ids),
        missing_images=missing_images,
        unreferenced_images=unreferenced_images,
        missing_labels=missing_labels,
        missing_lesion_ids=missing_lesion_ids,
        unique_lesion_ids=unique_lesion_ids,
        lesion_id_available=missing_lesion_ids == 0 and "lesion_id" in metadata.columns,
        unreadable_images=unreadable_images,
        class_distribution=class_distribution,
        lesion_class_distribution=lesion_class_distribution,
    )


def class_distribution_table(metadata: pd.DataFrame, class_names: list[str]) -> pd.DataFrame:
    counts = metadata["label"].value_counts().reindex(class_names, fill_value=0)
    total = int(counts.sum())
    frame = counts.rename_axis("label").reset_index(name="count")
    frame["percent"] = (frame["count"] / total * 100).round(4) if total else 0.0
    return frame


def lesion_class_distribution_table(
    metadata: pd.DataFrame, class_names: list[str]
) -> pd.DataFrame:
    lesion_labels = (
        metadata.groupby("lesion_id", dropna=False)["label"]
        .agg(lambda values: values.mode().iat[0])
        .reset_index()
    )
    counts = lesion_labels["label"].value_counts().reindex(class_names, fill_value=0)
    total = int(counts.sum())
    frame = counts.rename_axis("label").reset_index(name="lesion_count")
    frame["lesion_percent"] = (frame["lesion_count"] / total * 100).round(4) if total else 0.0
    return frame


def write_audit_tables(audit: DatasetAudit, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)
    audit.class_distribution.to_csv(tables_dir / "class_distribution.csv", index=False)
    audit.lesion_class_distribution.to_csv(
        tables_dir / "lesion_class_distribution.csv", index=False
    )
    if audit.missing_images:
        pd.DataFrame({"image_id": audit.missing_images}).to_csv(
            tables_dir / "missing_images.csv", index=False
        )
    if audit.unreferenced_images:
        pd.DataFrame({"image_id": audit.unreferenced_images}).to_csv(
            tables_dir / "unreferenced_images.csv", index=False
        )


def _sample_unreadable_images(metadata: pd.DataFrame, sample_size: int) -> list[str]:
    paths = [path for path in metadata["image_path"].astype(str).tolist() if path]
    sample = paths[:sample_size]
    unreadable: list[str] = []
    for path_text in sample:
        try:
            with Image.open(path_text) as image:
                image.verify()
        except Exception:
            unreadable.append(path_text)
    return unreadable
