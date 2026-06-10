"""Lightweight report asset helpers for Sprint 1."""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

import pandas as pd


def write_bar_svg(
    frame: pd.DataFrame,
    label_col: str,
    value_col: str,
    output_path: Path,
    title: str,
    color: str = "#2563eb",
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = frame[label_col].astype(str).tolist()
    values = [int(value) for value in frame[value_col].tolist()]
    max_value = max(values) if values else 1
    width = 900
    height = 520
    left = 70
    top = 70
    plot_width = 780
    plot_height = 330
    gap = 14
    bar_width = (plot_width - gap * (len(values) - 1)) / max(len(values), 1)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="32" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">{_escape(title)}</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1"/>',
    ]

    for index, (label, value) in enumerate(zip(labels, values, strict=True)):
        bar_height = 0 if max_value == 0 else value / max_value * plot_height
        x = left + index * (bar_width + gap)
        y = top + plot_height - bar_height
        parts.append(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width:.2f}" height="{bar_height:.2f}" fill="{color}"/>'
        )
        parts.append(
            f'<text x="{x + bar_width/2:.2f}" y="{y - 6:.2f}" text-anchor="middle" font-family="Arial" font-size="13">{value}</text>'
        )
        parts.append(
            f'<text x="{x + bar_width/2:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="Arial" font-size="14">{_escape(label)}</text>'
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def write_grouped_split_svg(distribution: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = distribution["label"].drop_duplicates().astype(str).tolist()
    splits = ["train", "val", "test"]
    colors = {"train": "#2563eb", "val": "#f59e0b", "test": "#10b981"}
    pivot = distribution.pivot(index="label", columns="split", values="count").fillna(0)
    max_value = int(pivot.max().max()) if not pivot.empty else 1

    width = 980
    height = 560
    left = 80
    top = 80
    plot_width = 830
    plot_height = 340
    group_gap = 20
    group_width = (plot_width - group_gap * (len(labels) - 1)) / max(len(labels), 1)
    bar_width = group_width / 3.4

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="490" y="34" text-anchor="middle" font-family="Arial" font-size="22" font-weight="700">HAM10000 split distribution by class</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1"/>',
    ]

    for label_index, label in enumerate(labels):
        group_x = left + label_index * (group_width + group_gap)
        for split_index, split_name in enumerate(splits):
            value = int(pivot.loc[label, split_name]) if split_name in pivot.columns else 0
            bar_height = 0 if max_value == 0 else value / max_value * plot_height
            x = group_x + split_index * bar_width
            y = top + plot_height - bar_height
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width - 2:.2f}" height="{bar_height:.2f}" fill="{colors[split_name]}"/>'
            )
        parts.append(
            f'<text x="{group_x + group_width/2:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="Arial" font-size="14">{_escape(label)}</text>'
        )

    legend_x = 700
    for index, split_name in enumerate(splits):
        y = 450 + index * 26
        parts.append(f'<rect x="{legend_x}" y="{y}" width="16" height="16" fill="{colors[split_name]}"/>')
        parts.append(
            f'<text x="{legend_x + 24}" y="{y + 13}" font-family="Arial" font-size="14">{split_name}</text>'
        )

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def _escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
