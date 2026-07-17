from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from context_aware_ci_build_failure_prediction.models.baseline.baseline import (
    binary_classification_metrics,
    create_random_forest_classifier,
)
from context_aware_ci_build_failure_prediction.models.load_samples import (
    LoadedSampleTable,
    load_sample_table,
)


LOGGER = logging.getLogger(__name__)


def train_random_forest_baseline(
    source_dir: str | Path,
    model_path: str | Path,
    *,
    num_samples: int | None = 300,
    validation_fraction: float = 0.2,
    seed: int = 42,
    shard_glob: str = "shard_*.pt",
    n_estimators: int = 300,
    max_depth: int | None = None,
    class_weight: str | None = "balanced",
    n_jobs: int = 1,
    table_path: str | Path | None = None,
    bar_graph_path: str | Path | None = None,
    show_bar_graph: bool = True,
) -> dict[str, Any]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")

    LOGGER.info("Loading samples from %s", source_dir)
    samples = load_sample_table(
        source_dir=source_dir,
        num_samples=num_samples,
        shard_glob=shard_glob,
    )
    sample_count = int(samples.labels.shape[0])
    if sample_count == 0:
        raise ValueError("No samples were loaded.")

    train_indices, test_indices = make_attention_fusion_style_split(
        sample_count=sample_count,
        validation_fraction=validation_fraction,
        seed=seed,
    )
    LOGGER.info(
        "Loaded %s samples; train_samples=%s test_samples=%s feature_dim=%s",
        sample_count,
        len(train_indices),
        len(test_indices),
        samples.features.shape[1],
    )

    features = samples.features.numpy()
    labels = samples.labels.to(torch.int64).numpy()

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
    model.fit(features[train_indices], labels[train_indices])

    test_predictions = model.predict(features[test_indices]).astype(int).tolist()
    test_labels = labels[test_indices].astype(int).tolist()
    metrics = binary_classification_metrics(test_labels, test_predictions)
    LOGGER.info(
        "Test metrics: accuracy=%.4f precision=%.4f recall=%.4f f1=%.4f",
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
    )

    table = build_test_result_table(
        samples=samples,
        test_indices=test_indices,
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
            "seed": seed,
            "shard_glob": shard_glob,
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "class_weight": class_weight,
            "n_jobs": n_jobs,
        },
        "feature_dim": int(samples.features.shape[1]),
    }
    with model_path.open("wb") as file:
        pickle.dump(checkpoint, file)
    LOGGER.info("Saved random forest checkpoint to %s", model_path)

    return {
        "model_path": str(model_path),
        "num_samples": sample_count,
        "train_samples": len(train_indices),
        "test_samples": len(test_indices),
        "metrics": metrics,
        "table_path": str(table_path) if table_path is not None else None,
        "bar_graph_path": (
            str(resolved_bar_graph_path)
            if resolved_bar_graph_path is not None
            else None
        ),
    }


def make_attention_fusion_style_split(
    *,
    sample_count: int,
    validation_fraction: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    validation_size = int(round(sample_count * validation_fraction))
    if sample_count > 1:
        validation_size = min(max(validation_size, 1), sample_count - 1)
    else:
        validation_size = 0

    train_size = sample_count - validation_size
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(sample_count, generator=generator).tolist()
    return indices[:train_size], indices[train_size:]


def build_test_result_table(
    *,
    samples: LoadedSampleTable,
    test_indices: list[int],
    predictions: list[int],
) -> pd.DataFrame:
    rows = []
    for index, prediction in zip(test_indices, predictions, strict=True):
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
    parser.add_argument("--seed", type=int, default=42)
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
    summary = train_random_forest_baseline(
        source_dir=args.source_dir,
        model_path=args.model_path,
        num_samples=args.num_samples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
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
