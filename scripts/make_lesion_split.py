#!/usr/bin/env python3
"""Create canonical lesion-aware HAM10000 train/validation/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dl_final.config import load_dataset_config
from dl_final.data.ham10000 import (
    attach_image_paths,
    audit_metadata,
    dataset_paths,
    load_ham10000_metadata,
)
from dl_final.data.splits import (
    create_lesion_aware_splits,
    leakage_audit_table,
    split_distribution_table,
    split_summary_table,
    write_split_csvs,
)
from dl_final.reporting import write_grouped_split_svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(args.config)
    dataset_config = config["dataset"]
    metadata_path, raw_dir, _, splits_dir = dataset_paths(dataset_config)
    if args.metadata_path:
        metadata_path = Path(args.metadata_path)
    if args.raw_dir:
        raw_dir = Path(args.raw_dir)

    class_names = list(dataset_config["class_names"])
    metadata = load_ham10000_metadata(metadata_path, class_names)
    metadata = attach_image_paths(metadata, raw_dir)
    audit = audit_metadata(metadata, metadata_path, raw_dir, class_names)
    if audit.has_blocking_errors and not args.allow_incomplete:
        raise SystemExit("Dataset audit found blocking errors; split files were not written.")

    split_config = dataset_config["split"]
    result = create_lesion_aware_splits(
        metadata,
        class_names,
        train_size=float(split_config["train"]),
        val_size=float(split_config["val"]),
        test_size=float(split_config["test"]),
        seed=int(dataset_config.get("seed", 42)),
    )
    write_split_csvs(result.splits, splits_dir)

    tables_dir = Path("artifacts/report_assets/tables")
    figures_dir = Path("artifacts/report_assets/figures")
    logs_dir = Path("artifacts/logs")
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    split_summary = split_summary_table(result.splits)
    split_distribution = split_distribution_table(result.splits, class_names)
    leakage_audit = leakage_audit_table(result.splits)
    split_summary.to_csv(tables_dir / "split_summary.csv", index=False)
    split_distribution.to_csv(tables_dir / "split_class_distribution.csv", index=False)
    leakage_audit.to_csv(tables_dir / "lesion_leakage_audit.csv", index=False)
    write_grouped_split_svg(split_distribution, figures_dir / "split_class_distribution.svg")

    manifest = {
        "seed": int(dataset_config.get("seed", 42)),
        "ratios": {
            "train": float(split_config["train"]),
            "val": float(split_config["val"]),
            "test": float(split_config["test"]),
        },
        "splits_dir": str(splits_dir),
        "warnings": result.warnings,
        "split_summary": split_summary.to_dict(orient="records"),
        "leakage_audit": leakage_audit.to_dict(orient="records"),
    }
    (logs_dir / "split_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Wrote splits to {splits_dir}")
    print(split_summary.to_string(index=False))
    print("Lesion leakage audit:")
    print(leakage_audit.to_string(index=False))
    for warning in result.warnings:
        print(f"WARNING: {warning}")


if __name__ == "__main__":
    main()

