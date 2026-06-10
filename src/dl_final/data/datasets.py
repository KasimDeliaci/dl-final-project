"""PyTorch datasets backed by the canonical HAM10000 split CSV files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset


class HAM10000ImageDataset(Dataset[dict[str, Any]]):
    """Image dataset for frozen feature extraction and later image-level training."""

    def __init__(
        self,
        split_csv: str | Path,
        class_names: list[str],
        transform: Any | None = None,
        split_name: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        self.split_csv = Path(split_csv)
        if not self.split_csv.exists():
            raise FileNotFoundError(f"Split CSV does not exist: {self.split_csv}")
        self.frame = pd.read_csv(self.split_csv)
        if max_samples is not None:
            self.frame = self.frame.head(max_samples).copy()
        self.class_names = list(class_names)
        self.label_to_index = {label: index for index, label in enumerate(self.class_names)}
        self.transform = transform
        self.split_name = split_name or self.split_csv.stem
        self._validate()

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")

        with Image.open(image_path) as image:
            image = image.convert("RGB")
            image_value = self.transform(image) if self.transform is not None else image

        label_name = str(row["label"])
        return {
            "image": image_value,
            "label": torch.tensor(self.label_to_index[label_name], dtype=torch.long),
            "label_name": label_name,
            "sample_id": str(row.get("sample_id", row["image_id"])),
            "image_id": str(row["image_id"]),
            "lesion_id": str(row.get("lesion_id", "")),
            "split": str(row.get("split", self.split_name)),
            "image_path": str(image_path),
        }

    def _validate(self) -> None:
        required = {"image_id", "label", "image_path"}
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Split CSV is missing required columns: {sorted(missing)}")
        unknown = sorted(set(self.frame["label"].astype(str)) - set(self.class_names))
        if unknown:
            raise ValueError(f"Split CSV contains labels outside class_names: {unknown}")

