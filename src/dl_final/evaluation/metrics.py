"""Classification metrics for HAM10000 benchmark runs."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def compute_classification_metrics(
    y_true: list[int] | np.ndarray,
    y_pred: list[int] | np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    true = np.asarray(y_true)
    pred = np.asarray(y_pred)
    labels = list(range(len(class_names)))
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        true,
        pred,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted_precision, weighted_recall, weighted_f1, _ = precision_recall_fscore_support(
        true,
        pred,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    report = classification_report(
        true,
        pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    per_class = [
        {
            "label": label,
            "precision": float(report[label]["precision"]),
            "recall": float(report[label]["recall"]),
            "f1": float(report[label]["f1-score"]),
            "support": int(report[label]["support"]),
        }
        for label in class_names
    ]
    matrix = confusion_matrix(true, pred, labels=labels)
    return {
        "accuracy": float(accuracy_score(true, pred)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_precision": float(weighted_precision),
        "weighted_recall": float(weighted_recall),
        "weighted_f1": float(weighted_f1),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def per_class_frame(metrics: dict[str, Any], backbone: str) -> pd.DataFrame:
    frame = pd.DataFrame(metrics["per_class"])
    frame.insert(0, "backbone", backbone)
    return frame

