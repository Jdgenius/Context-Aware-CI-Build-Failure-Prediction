from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch

from context_aware_ci_build_failure_prediction.models.baseline.baseline import (
    binary_classification_metrics,
    create_random_forest_classifier,
)
from context_aware_ci_build_failure_prediction.models.load_samples import load_sample_splits


LOGGER = logging.getLogger(__name__)
DEFAULT_SAMPLE_SIZES = (1000, 2500, 5000, 7500, 10000)


def run_random_forest_sample_sweep(
    source_dir: str | Path,
    *,
    sample_sizes: list[int] | tuple[int, ...] = DEFAULT_SAMPLE_SIZES,
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
) -> dict[str, Any]:
    if not sample_sizes:
        raise ValueError("sample_sizes must contain at least one value.")
    if any(sample_size <= 0 for sample_size in sample_sizes):
        raise ValueError("sample_sizes must all be positive.")

    LOGGER.info("Starting random forest sample sweep from %s", source_dir)
    results: list[dict[str, Any]] = []

    for sample_size in sample_sizes:
        LOGGER.info("Running sample sweep item requested_num_samples=%s", sample_size)
        splits = load_sample_splits(
            source_dir=source_dir,
            num_samples=sample_size,
            validation_fraction=validation_fraction,
            test_fraction=test_fraction,
            seed=seed,
            shuffle=shuffle_splits,
            shard_glob=shard_glob,
        )
        train_samples = int(splits.train.labels.shape[0])
        validation_samples = int(splits.validation.labels.shape[0])
        test_samples = int(splits.test.labels.shape[0])
        loaded_samples = train_samples + validation_samples + test_samples

        row: dict[str, Any] = {
            "requested_num_samples": sample_size,
            "loaded_samples": loaded_samples,
            "train_samples": train_samples,
            "validation_samples": validation_samples,
            "test_samples": test_samples,
            "train_repos": len(splits.train_repos),
            "validation_repos": len(splits.validation_repos),
            "test_repos": len(splits.test_repos),
            "test_accuracy": float("nan"),
            "test_precision": float("nan"),
            "test_recall": float("nan"),
            "test_f1": float("nan"),
        }

        if train_samples == 0 or test_samples == 0:
            LOGGER.warning(
                "Skipping requested_num_samples=%s because train_samples=%s "
                "test_samples=%s",
                sample_size,
                train_samples,
                test_samples,
            )
            results.append(row)
            continue

        model = create_random_forest_classifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=seed,
            class_weight=class_weight,
            n_jobs=n_jobs,
        )
        train_features = splits.train.features.numpy()
        train_labels = splits.train.labels.to(torch.int64).numpy()
        test_features = splits.test.features.numpy()
        test_labels = splits.test.labels.to(torch.int64).numpy()

        model.fit(train_features, train_labels)
        test_predictions = model.predict(test_features).astype(int).tolist()
        test_labels_list = test_labels.astype(int).tolist()
        metrics = binary_classification_metrics(test_labels_list, test_predictions)
        row.update(
            {
                "test_accuracy": metrics["accuracy"],
                "test_precision": metrics["precision"],
                "test_recall": metrics["recall"],
                "test_f1": metrics["f1"],
            }
        )
        LOGGER.info(
            "requested_num_samples=%s train_samples=%s test_samples=%s "
            "test_accuracy=%.4f",
            sample_size,
            train_samples,
            test_samples,
            metrics["accuracy"],
        )
        results.append(row)

    table = pd.DataFrame(results)
    print("\nRandom forest sample-sweep test accuracy:")
    print(table.to_string(index=False))

    resolved_table_path = Path(table_path) if table_path is not None else None
    if resolved_table_path is not None:
        resolved_table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(resolved_table_path, index=False)
        LOGGER.info("Saved sample sweep table to %s", resolved_table_path)

    resolved_bar_graph_path = plot_sample_sweep_accuracy_bar_graph(
        table=table,
        bar_graph_path=bar_graph_path,
        show=show_bar_graph,
    )

    return {
        "source_dir": str(source_dir),
        "sample_sizes": list(sample_sizes),
        "validation_fraction": validation_fraction,
        "test_fraction": test_fraction,
        "seed": seed,
        "shuffle_splits": shuffle_splits,
        "shard_glob": shard_glob,
        "n_estimators": n_estimators,
        "max_depth": max_depth,
        "class_weight": class_weight,
        "n_jobs": n_jobs,
        "results": results,
        "table_path": str(resolved_table_path) if resolved_table_path else None,
        "bar_graph_path": (
            str(resolved_bar_graph_path) if resolved_bar_graph_path else None
        ),
    }


def plot_sample_sweep_accuracy_bar_graph(
    *,
    table: pd.DataFrame,
    bar_graph_path: str | Path | None,
    show: bool,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for the sample sweep accuracy bar graph."
        ) from exc

    labels = [
        f"{sample_size:,}\ntrain={train_samples:,}"
        for sample_size, train_samples in zip(
            table["requested_num_samples"],
            table["train_samples"],
            strict=True,
        )
    ]
    values = table["test_accuracy"].tolist()

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(range(len(values)), values)
    axis.set_title("Random Forest Test Accuracy by Sample Budget")
    axis.set_xlabel("Requested samples and resulting train samples")
    axis.set_ylabel("Test accuracy")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(range(len(values)))
    axis.set_xticklabels(labels)
    axis.grid(True, axis="y", alpha=0.3)
    figure.tight_layout()

    resolved_path = Path(bar_graph_path) if bar_graph_path is not None else None
    if resolved_path is not None:
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(resolved_path, dpi=150, bbox_inches="tight")
        LOGGER.info("Saved sample sweep bar graph to %s", resolved_path)

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


def _parse_sample_sizes(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train random forests across sample budgets and plot test accuracy."
    )
    parser.add_argument("--source-dir", required=True)
    parser.add_argument(
        "--sample-sizes",
        default=",".join(str(value) for value in DEFAULT_SAMPLE_SIZES),
        help="Comma-separated sample budgets. Default: 1000,2500,5000,7500,10000.",
    )
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
    parser.add_argument("--summary-path")
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )
    args = parser.parse_args(argv)

    configure_logging(args.log_level)
    summary = run_random_forest_sample_sweep(
        source_dir=args.source_dir,
        sample_sizes=_parse_sample_sizes(args.sample_sizes),
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
    )

    if args.summary_path:
        summary_path = Path(args.summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())