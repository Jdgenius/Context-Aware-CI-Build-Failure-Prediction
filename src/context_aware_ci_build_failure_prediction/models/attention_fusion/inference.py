from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from context_aware_ci_build_failure_prediction.models.attention_fusion.model import (
    AttentionFusionClassifier,
)
from context_aware_ci_build_failure_prediction.models.load_samples import (
    load_training_pairs,
)


def load_attention_fusion_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | None = None,
) -> tuple[AttentionFusionClassifier, dict[str, Any], torch.device]:
    resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=resolved_device)

    if "model_config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(
            "Checkpoint must contain 'model_config' and 'model_state_dict'."
        )

    model = AttentionFusionClassifier(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(resolved_device)
    model.eval()

    return model, checkpoint, resolved_device


def predict_attention_fusion(
    checkpoint_path: str | Path,
    source_dir: str | Path,
    *,
    num_samples: int | None = 300,
    batch_size: int = 32,
    threshold: float = 0.5,
    device: str | None = None,
    include_attention: bool = True,
    shard_glob: str = "shard_*.pt",
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1].")

    model, checkpoint, resolved_device = load_attention_fusion_checkpoint(
        checkpoint_path,
        device=device,
    )
    dataset = load_training_pairs(
        source_dir=source_dir,
        num_samples=num_samples,
        shard_glob=shard_glob,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    probabilities: list[float] = []
    predictions: list[int] = []
    labels: list[float] = []
    attention_weights: list[list[float]] = []

    with torch.no_grad():
        for message, diff, context, batch_labels in loader:
            message = message.to(resolved_device)
            diff = diff.to(resolved_device)
            context = context.to(resolved_device)

            if include_attention:
                logits, attention = model(message, diff, context, return_attention=True)
                attention_weights.extend(attention.detach().cpu().tolist())
            else:
                logits = model(message, diff, context)

            batch_probabilities = torch.sigmoid(logits).detach().cpu()
            batch_predictions = (batch_probabilities >= threshold).to(torch.int64)

            probabilities.extend(float(value) for value in batch_probabilities.tolist())
            predictions.extend(int(value) for value in batch_predictions.tolist())
            labels.extend(float(value) for value in batch_labels.tolist())

    result: dict[str, Any] = {
        "checkpoint_path": str(checkpoint_path),
        "source_dir": str(source_dir),
        "num_samples": len(dataset),
        "threshold": threshold,
        "probabilities": probabilities,
        "predictions": predictions,
        "labels": labels,
        "device": str(resolved_device),
        "model_config": checkpoint["model_config"],
    }
    if include_attention:
        result["attention_weights"] = attention_weights
    if labels:
        correct_fraction = sum(
            int(prediction == int(label))
            for prediction, label in zip(predictions, labels, strict=True)
        ) / len(labels)
        result["error"] = 1.0 - correct_fraction

    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run attention-fusion inference.")
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--device")
    parser.add_argument("--shard-glob", default="shard_*.pt")
    parser.add_argument("--no-attention", action="store_true")
    parser.add_argument("--output-path")
    args = parser.parse_args(argv)

    result = predict_attention_fusion(
        checkpoint_path=args.checkpoint_path,
        source_dir=args.source_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        threshold=args.threshold,
        device=args.device,
        include_attention=not args.no_attention,
        shard_glob=args.shard_glob,
    )

    output = json.dumps(result, indent=2)
    if args.output_path:
        output_path = Path(args.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(output + "\n", encoding="utf-8")
    else:
        print(output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
