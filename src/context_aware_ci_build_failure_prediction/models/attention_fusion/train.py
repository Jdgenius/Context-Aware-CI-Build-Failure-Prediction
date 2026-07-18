from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from context_aware_ci_build_failure_prediction.models.attention_fusion.model import (
    AttentionFusionClassifier,
)
from context_aware_ci_build_failure_prediction.models.load_samples import (
    load_sample_splits,
)


LOGGER = logging.getLogger(__name__)


def train_attention_fusion(
    source_dir: str | Path,
    checkpoint_path: str | Path,
    *,
    num_samples: int | None = 300,
    batch_size: int = 32,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
    shuffle_splits: bool = True,
    model_dim: int = 128,
    attention_dim: int = 64,
    classifier_hidden_dim: int = 128,
    dropout: float = 0.2,
    separate_projections: bool = True,
    device: str | None = None,
    shard_glob: str = "shard_*.pt",
    plot_curves: bool = True,
    curve_path: str | Path | None = None,
    show_curves: bool = True,
    confusion_matrix_path: str | Path | None = None,
    show_confusion_matrix: bool = True,
) -> dict[str, Any]:
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    LOGGER.info("Starting attention-fusion training")
    LOGGER.info(
        "Training configuration: source_dir=%s num_samples=%s batch_size=%s "
        "epochs=%s learning_rate=%s validation_fraction=%s test_fraction=%s "
        "seed=%s shuffle_splits=%s",
        source_dir,
        num_samples,
        batch_size,
        epochs,
        learning_rate,
        validation_fraction,
        test_fraction,
        seed,
        shuffle_splits,
    )

    _set_seed(seed)
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("Using device: %s", resolved_device)

    LOGGER.info("Loading samples from .pt shard files")
    splits = load_sample_splits(
        source_dir=source_dir,
        num_samples=num_samples,
        validation_fraction=validation_fraction,
        test_fraction=test_fraction,
        seed=seed,
        shuffle=shuffle_splits,
        shard_glob=shard_glob,
    )
    train_dataset = splits.train.to_tensor_dataset()
    validation_dataset = splits.validation.to_tensor_dataset()
    test_dataset = splits.test.to_tensor_dataset()
    sample_count = len(train_dataset) + len(validation_dataset) + len(test_dataset)
    if sample_count == 0:
        raise ValueError("No training samples were loaded.")
    if len(train_dataset) == 0:
        raise ValueError("No training samples were assigned to the train split.")

    embedding_dim = splits.embedding_dim
    LOGGER.info(
        "Loaded %s samples with embedding_dim=%s",
        sample_count,
        embedding_dim,
    )
    LOGGER.info(
        "Repo split: train_repos=%s validation_repos=%s test_repos=%s",
        len(splits.train_repos),
        len(splits.validation_repos),
        len(splits.test_repos),
    )
    LOGGER.info(
        "Dataset split: train_samples=%s validation_samples=%s test_samples=%s",
        len(train_dataset),
        len(validation_dataset),
        len(test_dataset),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_loader = (
        DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)
        if len(validation_dataset) > 0
        else None
    )
    test_loader = (
        DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        if len(test_dataset) > 0
        else None
    )

    model = AttentionFusionClassifier(
        embedding_dim=embedding_dim,
        model_dim=model_dim,
        attention_dim=attention_dim,
        classifier_hidden_dim=classifier_hidden_dim,
        dropout=dropout,
        separate_projections=separate_projections,
    ).to(resolved_device)
    LOGGER.info(
        "Initialized AttentionFusionClassifier: model_dim=%s attention_dim=%s "
        "classifier_hidden_dim=%s dropout=%s separate_projections=%s",
        model_dim,
        attention_dim,
        classifier_hidden_dim,
        dropout,
        separate_projections,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss()
    history: list[dict[str, float | int]] = []

    for epoch in range(1, epochs + 1):
        epoch_started_at = time.perf_counter()
        LOGGER.info("Epoch %s/%s started", epoch, epochs)
        train_metrics = _run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=resolved_device,
        )
        epoch_record: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_error": train_metrics["error"],
        }

        if validation_loader is not None:
            validation_metrics = evaluate_attention_fusion(
                model=model,
                loader=validation_loader,
                criterion=criterion,
                device=resolved_device,
            )
            epoch_record.update(
                {
                    "validation_loss": validation_metrics["loss"],
                    "validation_error": validation_metrics["error"],
                }
            )

        history.append(epoch_record)
        LOGGER.info(
            "%s elapsed_seconds=%.2f",
            _format_epoch_record(epoch_record),
            time.perf_counter() - epoch_started_at,
        )

    final_test_metrics = None
    confusion_matrix = None
    resolved_confusion_matrix_path = None
    if test_loader is not None:
        LOGGER.info("Evaluating final test confusion matrix")
        (
            final_test_metrics,
            test_labels,
            test_predictions,
        ) = evaluate_attention_fusion_with_predictions(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=resolved_device,
        )
        confusion_matrix = build_binary_confusion_matrix(
            labels=test_labels,
            predictions=test_predictions,
        )
        print("\nAttention fusion test confusion matrix:")
        print(format_confusion_matrix(confusion_matrix))
        resolved_confusion_matrix_path = plot_confusion_matrix(
            confusion_matrix=confusion_matrix,
            title="Attention Fusion Test Confusion Matrix",
            confusion_matrix_path=confusion_matrix_path,
            show=show_confusion_matrix,
        )
    else:
        LOGGER.info("Skipping confusion matrix because test split is empty")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": {
            "embedding_dim": embedding_dim,
            "model_dim": model_dim,
            "attention_dim": attention_dim,
            "classifier_hidden_dim": classifier_hidden_dim,
            "dropout": dropout,
            "separate_projections": separate_projections,
        },
        "training_config": {
            "source_dir": str(source_dir),
            "num_samples": num_samples,
            "batch_size": batch_size,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "validation_fraction": validation_fraction,
            "test_fraction": test_fraction,
            "seed": seed,
            "shuffle_splits": shuffle_splits,
            "shard_glob": shard_glob,
        },
        "history": history,
        "final_test_metrics": final_test_metrics,
        "confusion_matrix": confusion_matrix,
    }

    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Saving checkpoint to %s", checkpoint_path)
    torch.save(checkpoint, checkpoint_path)
    LOGGER.info("Checkpoint saved")

    resolved_curve_path = None
    if plot_curves:
        LOGGER.info("Plotting training curves")
        resolved_curve_path = plot_training_curves(
            history=history,
            curve_path=curve_path,
            show=show_curves,
        )
    else:
        LOGGER.info("Training curve plotting disabled")

    LOGGER.info("Training complete")
    return {
        "checkpoint_path": str(checkpoint_path),
        "num_samples": sample_count,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "test_samples": len(test_dataset),
        "train_repos": splits.train_repos,
        "validation_repos": splits.validation_repos,
        "test_repos": splits.test_repos,
        "embedding_dim": embedding_dim,
        "device": str(resolved_device),
        "curve_path": str(resolved_curve_path) if resolved_curve_path else None,
        "confusion_matrix_path": (
            str(resolved_confusion_matrix_path)
            if resolved_confusion_matrix_path is not None
            else None
        ),
        "final_test_metrics": final_test_metrics,
        "confusion_matrix": confusion_matrix,
        "history": history,
    }


def evaluate_attention_fusion(
    model: AttentionFusionClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    LOGGER.debug("Evaluating on %s batches", len(loader))
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    with torch.no_grad():
        for message, diff, context, labels in loader:
            message = message.to(device)
            diff = diff.to(device)
            context = context.to(device)
            labels = labels.to(device)

            logits = model(message, diff, context)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predictions == labels).sum().item())
            total_examples += batch_size

    return _metrics(total_loss, total_correct, total_examples)


def evaluate_attention_fusion_with_predictions(
    model: AttentionFusionClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[dict[str, float], list[int], list[int]]:
    model.eval()
    LOGGER.debug("Evaluating predictions on %s batches", len(loader))
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    all_labels: list[int] = []
    all_predictions: list[int] = []

    with torch.no_grad():
        for message, diff, context, labels in loader:
            message = message.to(device)
            diff = diff.to(device)
            context = context.to(device)
            labels = labels.to(device)

            logits = model(message, diff, context)
            loss = criterion(logits, labels)
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()

            batch_size = labels.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_correct += int((predictions == labels).sum().item())
            total_examples += batch_size
            all_labels.extend(labels.to(torch.int64).cpu().tolist())
            all_predictions.extend(predictions.to(torch.int64).cpu().tolist())

    return _metrics(total_loss, total_correct, total_examples), all_labels, all_predictions


def _run_epoch(
    model: AttentionFusionClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    model.train()
    LOGGER.debug("Training on %s batches", len(loader))
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for message, diff, context, labels in loader:
        message = message.to(device)
        diff = diff.to(device)
        context = context.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(message, diff, context)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        probabilities = torch.sigmoid(logits.detach())
        predictions = (probabilities >= 0.5).float()
        batch_size = labels.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_correct += int((predictions == labels).sum().item())
        total_examples += batch_size

    return _metrics(total_loss, total_correct, total_examples)


def plot_training_curves(
    history: list[dict[str, float | int]],
    *,
    curve_path: str | Path | None = None,
    show: bool = True,
) -> Path | None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required to plot training curves. "
            "Install it or call train_attention_fusion(..., plot_curves=False)."
        ) from exc

    if not history:
        raise ValueError("Cannot plot training curves without training history.")

    epochs = [int(record["epoch"]) for record in history]
    train_loss = [float(record["train_loss"]) for record in history]
    train_error = [float(record["train_error"]) for record in history]
    has_validation = "validation_loss" in history[0]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="train")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")

    axes[1].plot(epochs, train_error, marker="o", label="train")
    axes[1].set_title("Error")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Error rate")
    axes[1].set_ylim(0.0, 1.0)

    if has_validation:
        validation_loss = [float(record["validation_loss"]) for record in history]
        validation_error = [
            float(record["validation_error"]) for record in history
        ]
        axes[0].plot(epochs, validation_loss, marker="o", label="validation")
        axes[1].plot(epochs, validation_error, marker="o", label="validation")

    for axis in axes:
        axis.grid(True, alpha=0.3)
        axis.legend()

    figure.tight_layout()

    resolved_curve_path = Path(curve_path) if curve_path is not None else None
    if resolved_curve_path is not None:
        resolved_curve_path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(resolved_curve_path, dpi=150, bbox_inches="tight")
        LOGGER.info("Saved training curves to %s", resolved_curve_path)

    if show:
        LOGGER.info("Displaying training curves")
        plt.show()

    plt.close(figure)
    return resolved_curve_path


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


def format_confusion_matrix(
    confusion_matrix: dict[str, int | list[list[int]]],
) -> str:
    matrix = confusion_matrix["matrix"]
    return (
        "            predicted_0  predicted_1\n"
        f"actual_0    {matrix[0][0]:>11}  {matrix[0][1]:>11}\n"
        f"actual_1    {matrix[1][0]:>11}  {matrix[1][1]:>11}"
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

    max_value = max(max(row) for row in matrix)
    for actual_index, row in enumerate(matrix):
        for predicted_index, value in enumerate(row):
            axis.text(
                predicted_index,
                actual_index,
                str(value),
                ha="center",
                va="center",
                color="white" if max_value > 0 and value > max_value / 2 else "black",
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
        LOGGER.info("Displaying confusion matrix")
        plt.show()

    plt.close(figure)
    return resolved_path


def _metrics(total_loss: float, total_correct: int, total_examples: int) -> dict[str, float]:
    if total_examples == 0:
        return {"loss": float("nan"), "error": float("nan")}
    return {
        "loss": total_loss / total_examples,
        "error": 1.0 - (total_correct / total_examples),
    }


def _set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_epoch_record(record: dict[str, float | int]) -> str:
    parts = [
        f"epoch={int(record['epoch'])}",
        f"train_loss={record['train_loss']:.4f}",
        f"train_error={record['train_error']:.4f}",
    ]
    if "validation_loss" in record:
        parts.extend(
            [
                f"validation_loss={record['validation_loss']:.4f}",
                f"validation_error={record['validation_error']:.4f}",
            ]
        )
    return " ".join(parts)


def configure_logging(level: str = "INFO") -> None:
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Unsupported log level: {level}")

    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the attention-fusion classifier.")
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-shuffle-splits", action="store_true")
    parser.add_argument("--model-dim", type=int, default=128)
    parser.add_argument("--attention-dim", type=int, default=64)
    parser.add_argument("--classifier-hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--shared-projection", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--shard-glob", default="shard_*.pt")
    parser.add_argument("--curve-path")
    parser.add_argument("--no-curves", action="store_true")
    parser.add_argument("--no-show-curves", action="store_true")
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
    summary = train_attention_fusion(
        source_dir=args.source_dir,
        checkpoint_path=args.checkpoint_path,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        seed=args.seed,
        shuffle_splits=not args.no_shuffle_splits,
        model_dim=args.model_dim,
        attention_dim=args.attention_dim,
        classifier_hidden_dim=args.classifier_hidden_dim,
        dropout=args.dropout,
        separate_projections=not args.shared_projection,
        device=args.device,
        shard_glob=args.shard_glob,
        plot_curves=not args.no_curves,
        curve_path=args.curve_path,
        show_curves=not args.no_show_curves,
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
