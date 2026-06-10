#!/usr/bin/env python3
"""Audit HAM10000 metadata/images and export Sprint 1 report assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dl_final.config import load_dataset_config
from dl_final.data.ham10000 import (
    attach_image_paths,
    audit_metadata,
    dataset_paths,
    load_ham10000_metadata,
    write_audit_tables,
)
from dl_final.reporting import write_bar_svg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--metadata-path", default=None)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--allow-incomplete", action="store_true")
    parser.add_argument("--image-open-sample", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_dataset_config(args.config)
    dataset_config = config["dataset"]
    metadata_path, raw_dir, processed_dir, _ = dataset_paths(dataset_config)
    if args.metadata_path:
        metadata_path = Path(args.metadata_path)
    if args.raw_dir:
        raw_dir = Path(args.raw_dir)

    class_names = list(dataset_config["class_names"])
    metadata = load_ham10000_metadata(metadata_path, class_names)
    metadata = attach_image_paths(metadata, raw_dir)
    audit = audit_metadata(
        metadata,
        metadata_path,
        raw_dir,
        class_names,
        image_open_sample=args.image_open_sample,
    )

    processed_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = Path("artifacts/report_assets/tables")
    figures_dir = Path("artifacts/report_assets/figures")
    logs_dir = Path("artifacts/logs")
    figures_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    audited_metadata_path = processed_dir / "ham10000_audited_metadata.csv"
    metadata.to_csv(audited_metadata_path, index=False)
    write_audit_tables(audit, tables_dir)
    write_bar_svg(
        audit.class_distribution,
        "label",
        "count",
        figures_dir / "class_distribution.svg",
        "HAM10000 class distribution",
    )
    (logs_dir / "dataset_audit.json").write_text(
        json.dumps(audit.to_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame([audit.to_dict()]).to_csv(tables_dir / "dataset_audit_summary.csv", index=False)

    _print_summary(audit, audited_metadata_path)
    if audit.has_blocking_errors and not args.allow_incomplete:
        raise SystemExit("Dataset audit found blocking errors; splits should not be generated.")


def _print_summary(audit, audited_metadata_path: Path) -> None:
    print(f"Metadata: {audit.metadata_path}")
    print(f"Raw images: {audit.raw_dir}")
    print(f"Audited metadata: {audited_metadata_path}")
    print(f"Rows: {audit.image_rows}")
    print(f"Unique image IDs: {audit.unique_image_ids}")
    print(f"Duplicate image IDs: {len(audit.duplicate_image_ids)}")
    print(f"Missing images: {len(audit.missing_images)}")
    print(f"Unreferenced image files: {len(audit.unreferenced_images)}")
    print(f"Missing labels: {audit.missing_labels}")
    print(f"Missing lesion IDs: {audit.missing_lesion_ids}")
    print(f"Unique lesion IDs: {audit.unique_lesion_ids}")
    print(f"Unreadable sampled images: {len(audit.unreadable_images)}")
    print("Class distribution:")
    print(audit.class_distribution.to_string(index=False))


if __name__ == "__main__":
    main()

