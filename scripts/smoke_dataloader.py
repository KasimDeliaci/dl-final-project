#!/usr/bin/env python3
"""Smoke-check generated split CSVs by opening sample images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dl_final.config import load_dataset_config
from dl_final.data.ham10000 import dataset_paths
from dl_final.data.smoke import smoke_check_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--max-samples", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(args.config)
    _, _, _, splits_dir = dataset_paths(config["dataset"])
    results = {}
    for split_name in ("train", "val", "test"):
        results[split_name] = smoke_check_split(
            splits_dir / f"{split_name}.csv",
            max_samples=args.max_samples,
        )

    output_path = Path("artifacts/logs/dataloader_smoke.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

