from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from typing import Any

import torch

from context_aware_ci_build_failure_prediction.models.baseline.baseline import (
    binary_classification_metrics,
)
from context_aware_ci_build_failure_prediction.models.baseline.fit import (
    build_test_result_table,
    make_attention_fusion_style_split,
    plot_commit_accuracy_bar_graph,
)
from context_aware_ci_build_failure_prediction.models.load_samples import (
    load_sample_table,
)


def run_random_forest_inference(
    model_path: str | Path,
    source_dir: str | Path,
    *,
    num_samples: int | None = None,
    validation_fraction: float | None = None,
    seed: int | None = None,
    shard_glob: str | None = None,
    table_path: str | Path | None = None,
    bar_graph_path: str | Path | None = None,
    show_bar_graph: bool = True,
) -> dict[str, Any]:
    checkpoint = load_random_forest_checkpoint(model_path)
    training_config = checkpoint.get("training_config", {})
    resolved_num_samples = (
        num_samples if num_samples is not None else training_config.get("num_samples", 300)
    )
    resolved_validation_fraction = (
        validation_fraction
        if validation_fraction is not None
        else training_config.get("validation_fraction", 0.2)
    )
    resolved_seed = seed if seed is not None else training_config.get("seed", 42)
    resolved_shard_glob = shard_glob or training_config.get("shard_glob", "shard_*.pt")

    samples = load_sample_table(
        source_dir=source_dir,
        num_samples=resolved_num_samples,
        shard_glob=resolved_shard_glob,
    )
    sample_count = int(samples.labels.shape[0])
    _, test_indices = make_attention_fusion_style_split(
        sample_count=sample_count,
        validation_fraction=resolved_validation_fraction,
        seed=resolved_seed,
    )

    features = samples.features.numpy()
    labels = samples.labels.to(torch.int64).numpy()
    model = checkpoint["model"]
    predictions = model.predict(features[test_indices]).astype(int).tolist()
    test_labels = labels[test_indices].astype(int).tolist()
    metrics = binary_classification_metrics(test_labels, predictions)

    table = build_test_result_table(
        samples=samples,
        test_indices=test_indices,
        predictions=predictions,
    )
    print("\nRandom forest test-set commit results:")
    print(table.to_string(index=False))

    if table_path is not None:
        table_path = Path(table_path)
        table_path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(table_path, index=False)

    resolved_bar_graph_path = plot_commit_accuracy_bar_graph(
        table=table,
        bar_graph_path=bar_graph_path,
        show=show_bar_graph,
    )

    return {
        "model_path": str(model_path),
        "num_samples": sample_count,
        "test_samples": len(test_indices),
        "metrics": metrics,
        "table_path": str(table_path) if table_path is not None else None,
        "bar_graph_path": (
            str(resolved_bar_graph_path)
            if resolved_bar_graph_path is not None
            else None
        ),
    }


def load_random_forest_checkpoint(model_path: str | Path) -> dict[str, Any]:
    with Path(model_path).open("rb") as file:
        checkpoint = pickle.load(file)

    if "model" not in checkpoint:
        raise ValueError(f"Random forest checkpoint {model_path} is missing 'model'.")

    return checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run random forest baseline inference.")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--num-samples", type=int)
    parser.add_argument("--validation-fraction", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--shard-glob")
    parser.add_argument("--table-path")
    parser.add_argument("--bar-graph-path")
    parser.add_argument("--no-show-bar-graph", action="store_true")
    parser.add_argument("--summary-path")
    args = parser.parse_args(argv)

    summary = run_random_forest_inference(
        model_path=args.model_path,
        source_dir=args.source_dir,
        num_samples=args.num_samples,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
        shard_glob=args.shard_glob,
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
