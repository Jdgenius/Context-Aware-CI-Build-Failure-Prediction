from __future__ import annotations

from typing import Any, Literal

from sklearn.ensemble import RandomForestClassifier


def create_random_forest_classifier(
    *,
    n_estimators: int = 300,
    max_depth: int | None = None,
    random_state: int = 0,
    class_weight: Literal["balanced", "balanced_subsample"] | dict[Any, float] | None = "balanced",
    n_jobs: int = 1,
):
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=n_jobs,
    )


def binary_classification_metrics(
    labels: list[int],
    predictions: list[int],
) -> dict[str, float]:
    if len(labels) != len(predictions):
        raise ValueError("labels and predictions must have the same length.")
    if not labels:
        return {
            "accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
        }

    true_positive = sum(
        1 for label, prediction in zip(labels, predictions, strict=True)
        if label == 1 and prediction == 1
    )
    true_negative = sum(
        1 for label, prediction in zip(labels, predictions, strict=True)
        if label == 0 and prediction == 0
    )
    false_positive = sum(
        1 for label, prediction in zip(labels, predictions, strict=True)
        if label == 0 and prediction == 1
    )
    false_negative = sum(
        1 for label, prediction in zip(labels, predictions, strict=True)
        if label == 1 and prediction == 0
    )

    accuracy = (true_positive + true_negative) / len(labels)
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = (
        true_positive / precision_denominator
        if precision_denominator
        else 0.0
    )
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    f1_denominator = precision + recall
    f1 = 2 * precision * recall / f1_denominator if f1_denominator else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


__all__ = [
    "binary_classification_metrics",
    "create_random_forest_classifier",
]
