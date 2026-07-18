from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch

from context_aware_ci_build_failure_prediction.models.baseline.baseline import (
    binary_classification_metrics,
    create_random_forest_classifier,
)
from context_aware_ci_build_failure_prediction.models.load_samples import (
    LoadedSampleTable,
    load_sample_splits,
)


LOGGER = logging.getLogger(__name__)


def train_random_forest_baseline(
    source_dir: str | Path,
    model_path: str | Path,
    *,
    num_samples: int | None = 300,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
    shuffle_splits: bool = True,
    shard_glob: str = "shard_*.pt",
    n_estimators: int = 300,
    max_depth: int | None = None,
    class_weight: Literal["balanced", "balanced_subsample"] | dict[Any, float] | None = "balanced",
    n_jobs: int = 1,
    table_path: str | Path | None = None,
    bar_graph_path: str | Path | None = None,
    show_bar_graph: bool = True,
    confusion_matrix_path: str | Path | None = None,
    show_confusion_matrix: bool = True,
) -> dict[str, Any]:
    LOGGER.info("Loading samples from %s", source_dir)
    splits = load_sample_splits(
        source_dir=source_dir,
        num_samples=num_samples,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        shuffle=shuffle_splits,
        shard_glob=shard_glob,
    )
    sample_count = (
        int(splits.train.labels.shape[0])
        + int(splits.validation.labels.shape[0])
        + int(splits.test.labels.shape[0])
    )
    if sample_count == 0:
        raise ValueError("No samples were loaded.")
    if splits.train.labels.shape[0] == 0:
        raise ValueError("No samples were assigned to the train split.")
    if splits.test.labels.shape[0] == 0:
        raise ValueError(
            "No samples were assigned to the test split. "
            "Use more repositories or lower --validation-fraction/--test-fraction."
        )

    LOGGER.info(
        "Loaded %s samples; train_samples=%s validation_samples=%s "
        "test_samples=%s feature_dim=%s",
        sample_count,
        splits.train.labels.shape[0],
        splits.validation.labels.shape[0],
        splits.test.labels.shape[0],
        splits.train.features.shape[1],
    )

    train_features = splits.train.features.numpy()
    train_labels = splits.train.labels.to(torch.int64).numpy()
    validation_features = splits.validation.features.numpy()
    validation_labels = splits.validation.labels.to(torch.int64).numpy()
    test_features = splits.test.features.numpy()
    test_labels = splits.test.labels.to(torch.int64).numpy()
    
    #Remove later
    #test_features = test_features[:10]
    #test_labels = test_labels[:10]

    model = create_random_forest_classifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        class_weight=class_weight,
        n_jobs=n_jobs,
    )
    LOGGER.info(
        "Training RandomForestClassifier n_estimators=%s max_depth=%s class_weight=%s",
        n_estimators,
        max_depth,
        class_weight,
    )
    model.fit(train_features, train_labels)

    validation_metrics = None
    if len(validation_labels) > 0:
        validation_predictions = model.predict(validation_features).astype(int).tolist()
        validation_metrics = binary_classification_metrics(
            validation_labels.astype(int).tolist(),
            validation_predictions,
        )

    test_predictions = model.predict(test_features).astype(int).tolist()
    test_labels_list = test_labels.astype(int).tolist()
    metrics = binary_classification_metrics(test_labels_list, test_predictions)
    confusion_matrix = build_binary_confusion_matrix(
        labels=test_labels_list,
        predictions=test_predictions,
    )
    LOGGER.info(
        "Test metrics: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )

    print("\nRandom forest test-set confusion matrix:")
    print(confusion_matrix_to_dataframe(confusion_matrix).to_string())
    resolved_confusion_matrix_path = plot_confusion_matrix(
        confusion_matrix=confusion_matrix,
        title="Random Forest Test Confusion Matrix",
        confusion_matrix_path=confusion_matrix_path,
        show=show_confusion_matrix,
    )

    table = build_test_result_table(
        samples=splits.test,
        predictions=test_predictions,
    )
    print("\nRandom forest test-set commit results:")
    print(table.to_string(index=False))

    if table_path is not None:
        table_path = Path(table_path)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_path, index=False)
        LOGGER.info("Saved test table to %s", table_path)

    resolved_bar_graph_path = plot_commit_accuracy_bar_graph(
        table=table,
        bar_graph_path=bar_graph_path,
        show=show_bar_graph,
    )

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model": model,
        "metrics": metrics,
        "training_config": {
            "source_dir": str(source_dir),
            "num_samples": num_samples,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "seed": seed,
            "shuffle_splits": shuffle_splits,
            "shard_glob": shard_glob,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "class_weight": class_weight,
            "n_jobs": n_jobs,
        },
        "validation_metrics": validation_metrics,
        "confusion_matrix": confusion_matrix,
        "feature_dim": int(splits.train.features.shape[1]),
    }
    with model_path.open("wb") as file:
        pickle.dump(checkpoint, file)
    LOGGER.info("Saved random forest checkpoint to %s", model_path)

    return {
        "model_path": str(model_path),
        "num_samples": sample_count,
        "train_samples": int(splits.train.labels.shape[0]),
        "validation_samples": int(splits.validation.labels.shape[0]),
        "test_samples": int(splits.test.labels.shape[0]),
        "train_repos": splits.train_repos,
        "validation_repos": splits.validation_repos,
        "test_repos": splits.test_repos,
        "validation_metrics": validation_metrics,
        "metrics": metrics,
        "confusion_matrix": confusion_matrix,
        "table_path": str(table_path) if table_path is not None else None,
        "confusion_matrix_path": (
            str(resolved_confusion_matrix_path)
            if resolved_confusion_matrix_path is not None
            else None
        ),
        "bar_graph_path": (
            str(resolved_bar_graph_path)
            if resolved_bar_graph_path is not None
            else None
        ),
    }


def build_test_result_table(
    *,
    samples: LoadedSampleTable,
    predictions: list[int],
) -> pd.DataFrame:
    rows = []
    for index, prediction in zip(range(len(predictions)), predictions, strict=True):
        label = int(samples.labels[index].item())
        rows.append(
            {
                "repo": samples.repos[index],
                "commit_sha": samples.commit_shas[index],
                "label": label,
                "prediction": int(prediction),
                "accuracy": int(label == int(prediction)),
            }
        )

    return pd.DataFrame(rows)


def build_binary_confusion_matrix(
    *,
    labels: list[int],
    predictions: list[int],
) -> dict[str, int | list[list[int]]]:
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
    true_positive = sum(
        1 for label, prediction in zip(labels, predictions, strict=True)
        if label == 1 and prediction == 1
    )
    return {
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
        "matrix": [
            [true_negative, false_positive],
            [false_negative, true_positive],
        ],
    }


def confusion_matrix_to_dataframe(
    confusion_matrix: dict[str, int | list[list[int]]],
) -> pd.DataFrame:
    matrix = confusion_matrix["matrix"]
    return pd.DataFrame(
        matrix,
        index=["actual_0", "actual_1"],
        columns=["predicted_0", "predicted_1"],
    )


def plot_confusion_matrix(
    *,
    confusion_matrix: dict[str, int | list[list[int]]],
    title: str,
    confusion_matrix_path: str | Path | None,
    show: bool,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for the confusion matrix plot."
        ) from exc

    matrix = confusion_matrix["matrix"]
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(title)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("Actual label")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(["0", "1"])
    axis.set_yticklabels(["0", "1"])

    for actual_index, row in enumerate(matrix):
        for predicted_index, value in enumerate(row):
            axis.text(
                predicted_index,
                actual_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > max(max(row) for row in matrix) / 2 else "black",
            )

    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()

    resolved_path = (
        Path(confusion_matrix_path) if confusion_matrix_path is not None else None
    )
    if resolved_path is not None:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(resolved_path, dpi=150, bbox_inches="tight")
        LOGGER.info("Saved confusion matrix plot to %s", resolved_path)

    if show:
        plt.show()

    plt.close(figure)
    return resolved_path


def plot_commit_accuracy_bar_graph(
    *,
    table: pd.DataFrame,
    bar_graph_path: str | Path | None,
    show: bool,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for the commit accuracy bar graph."
        ) from exc

    labels = [
        f"{row.repo}\n{str(row.commit_sha)[:8]}"
        for row in table.itertuples(index=False)
    ]
    values = table["accuracy"].tolist()
    width = max(10, min(32, len(values) * 0.45))
    figure, axis = plt.subplots(figsize=(width, 5))
    axis.bar(range(len(values)), values)
    axis.set_title("Random Forest Test Commit Accuracy")
    axis.set_ylabel("Accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(range(len(values)))
    axis.set_xticklabels(labels, rotation=70, ha="right", fontsize=8)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()

    resolved_path = Path(bar_graph_path) if bar_graph_path is not None else None
    if resolved_path is not None:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(resolved_path, dpi=150, bbox_inches="tight")
        LOGGER.info("Saved test-set bar graph to %s", resolved_path)

    if show:
        plt.show()

    plt.close(figure)
    return resolved_path


def configure_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the random forest baseline.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle-splits", action="store_true")
    parser.add_argument("--shard-glob", default="shard_*.pt")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--max-depth", type=int)
    parser.add_argument("--class-weight", default="balanced")
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--table-path")
    parser.add_argument("--bar-graph-path")
    parser.add_argument("--no-show-bar-graph", action="store_true")
    parser.add_argument("--confusion-matrix-path")
    parser.add_argument("--no-show-confusion-matrix", action="store_true")
    parser.add_argument("--summary-path")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    summary = train_random_forest_baseline(
        source_dir=args.source_dir,
        model_path=args.model_path,
        num_samples=args.num_samples,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        shuffle_splits=not args.no_shuffle_splits,
        shard_glob=args.shard_glob,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight=args.class_weight,
        n_jobs=args.n_jobs,
        table_path=args.table_path,
        bar_graph_path=args.bar_graph_path,
        show_bar_graph=not args.no_show_bar_graph,
        confusion_matrix_path=args.confusion_matrix_path,
        show_confusion_matrix=not args.no_show_confusion_matrix,
    )

    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
