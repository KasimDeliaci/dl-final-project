"""Analyze validation representation similarity across cached backbone features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from dl_final.config import load_dataset_config
from dl_final.evaluation.complementarity import (
    build_fusion_complementarity_summary,
    compute_representation_complementarity,
    save_representation_similarity_heatmap,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/dataset/selected_dataset.yaml")
    parser.add_argument("--feature-root", default="artifacts/features")
    parser.add_argument("--feature-source", default="frozen")
    parser.add_argument(
        "--backbones",
        nargs="+",
        default=["vit_b16", "swin_tiny", "deit3_small", "beit_base"],
    )
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--max-samples", type=int, default=1504)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tables-dir", default="artifacts/report_assets/tables")
    parser.add_argument("--figures-dir", default="artifacts/report_assets/figures")
    parser.add_argument(
        "--fusion-results",
        default="artifacts/report_assets/tables/frozen_fusion_results.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_config = load_dataset_config(args.dataset_config)["dataset"]
    table_root = Path(args.tables_dir)
    figure_root = Path(args.figures_dir)
    table_root.mkdir(parents=True, exist_ok=True)
    figure_root.mkdir(parents=True, exist_ok=True)

    pairwise = compute_representation_complementarity(
        feature_root=args.feature_root,
        dataset_name=str(dataset_config["name"]),
        feature_source=args.feature_source,
        backbones=args.backbones,
        splits_dir=dataset_config["splits_dir"],
        split=args.split,
        max_samples=args.max_samples,
        seed=args.seed,
    )
    pairwise_path = table_root / f"{args.feature_source}_representation_similarity_{args.split}.csv"
    pairwise.to_csv(pairwise_path, index=False)

    heatmap_path = figure_root / f"{args.feature_source}_representation_similarity_{args.split}.png"
    save_representation_similarity_heatmap(pairwise, heatmap_path)

    exported = {
        "pairwise_similarity": str(pairwise_path),
        "similarity_heatmap": str(heatmap_path),
    }
    fusion_results_path = Path(args.fusion_results)
    if fusion_results_path.exists():
        fusion_results = pd.read_csv(fusion_results_path)
        summary = build_fusion_complementarity_summary(fusion_results, pairwise)
        summary_path = (
            table_root / f"{args.feature_source}_fusion_complementarity_{args.split}.csv"
        )
        summary.to_csv(summary_path, index=False)
        exported["fusion_complementarity"] = str(summary_path)

    metadata_path = (
        table_root / f"{args.feature_source}_representation_similarity_{args.split}.json"
    )
    metadata_path.write_text(
        json.dumps(
            {
                "dataset": dataset_config["name"],
                "feature_source": args.feature_source,
                "split": args.split,
                "backbones": args.backbones,
                "max_samples": args.max_samples,
                "seed": args.seed,
                "method": "train_scaled_sample_cosine_rsa_pearson",
                "test_policy": "test_not_used",
                "exported": exported,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    exported["metadata"] = str(metadata_path)

    print("Exported representation similarity assets:")
    for name, path in exported.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
