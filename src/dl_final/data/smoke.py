"""Dataset smoke-check helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from PIL import Image


def smoke_check_split(split_path: Path, max_samples: int = 8) -> dict[str, object]:
    if not split_path.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")

    frame = pd.read_csv(split_path)
    required = {"image_id", "label", "image_path"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Split file is missing required columns: {sorted(missing)}")

    checked = 0
    sizes: list[tuple[int, int]] = []
    for row in frame.head(max_samples).itertuples(index=False):
        image_path = Path(str(getattr(row, "image_path")))
        if not image_path.exists():
            raise FileNotFoundError(f"Image file does not exist: {image_path}")
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            sizes.append(image.size)
        checked += 1

    return {
        "split_path": str(split_path),
        "rows": len(frame),
        "checked_images": checked,
        "sample_sizes": sizes,
        "labels_present": sorted(frame["label"].astype(str).unique().tolist()),
    }

