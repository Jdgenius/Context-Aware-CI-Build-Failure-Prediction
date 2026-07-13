from __future__ import annotations

import gc
import gzip
import json
import os
from collections.abc import Iterator
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Callable

import torch

from ..types import RawSample


FORMAT_VERSION = 1
REQUIRED_SHARD_KEYS = {
    "format_version",
    "shard_index",
    "num_samples",
    "sample_ids",
    "repos",
    "commit_shas",
    "build_ids",
    "labels",
    "message_embeddings",
    "diff_embeddings",
    "context_embeddings",
    "record_indices",
}


@dataclass
class EmbeddedSampleRecord:
    raw_sample: RawSample
    message_embedding: torch.Tensor
    diff_embedding: torch.Tensor
    context_embedding: torch.Tensor


class EmbeddingShardWriter:
    def __init__(
        self,
        output_dir: str = "./embedding_shards",
        shard_size: int = 5000,
        allow_overwrite: bool = False,
        on_shard_complete: Callable[[int, Path, Path, int], None] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.shard_size = shard_size
        self.allow_overwrite = allow_overwrite
        self.on_shard_complete = on_shard_complete
        self.buffer: list[EmbeddedSampleRecord] = []
        self.shard_index = 0

    def add(self, record: EmbeddedSampleRecord) -> None:
        if not isinstance(record, EmbeddedSampleRecord):
            raise TypeError(
                "EmbeddingShardWriter.add expected EmbeddedSampleRecord, "
                f"got {type(record).__name__}"
            )

        self.buffer.append(record)

        if len(self.buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return

        records = list(self.buffer)
        shard_index = self.shard_index
        tensor_path, sidecar_path = self._final_paths(shard_index)
        tensor_tmp_path, sidecar_tmp_path = self._temporary_paths(shard_index)

        self._ensure_output_paths_available(tensor_path, sidecar_path)

        payload, sidecar_records = build_shard_payload(
            records=records,
            shard_index=shard_index,
        )

        validate_shard_payload(payload, sidecar_records)

        promoted_tensor = False

        try:
            torch.save(payload, tensor_tmp_path)

            with gzip.open(sidecar_tmp_path, "wt", encoding="utf-8") as sidecar_file:
                for sidecar_record in sidecar_records:
                    sidecar_file.write(json.dumps(to_json_safe(sidecar_record), ensure_ascii=False))
                    sidecar_file.write("\n")

            os.replace(tensor_tmp_path, tensor_path)
            promoted_tensor = True
            os.replace(sidecar_tmp_path, sidecar_path)
        except Exception as exc:
            if promoted_tensor and tensor_path.exists():
                if sidecar_path.exists():
                    sidecar_path.unlink()
                tensor_path.unlink()
            self._cleanup_temporary_files(tensor_tmp_path, sidecar_tmp_path)
            raise RuntimeError(
                f"Failed to write paired shard {shard_index:05d}; "
                "buffer preserved and no completed pair was produced"
            ) from exc

        if self.on_shard_complete is not None:
            try:
                self.on_shard_complete(
                    shard_index,
                    tensor_path,
                    sidecar_path,
                    len(records),
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Completed shard pair {shard_index:05d}, but manifest update failed; "
                    "the completed pair was retained for recovery and the buffer was preserved"
                ) from exc

        self.buffer.clear()
        self.shard_index += 1
        print(f"Saved shard pair: {tensor_path} and {sidecar_path} with {len(records)} samples")
        gc.collect()

    def close(self) -> None:
        self.flush()

    def _final_paths(self, shard_index: int) -> tuple[Path, Path]:
        stem = f"shard_{shard_index:05d}"
        return (
            self.output_dir / f"{stem}.pt",
            self.output_dir / f"{stem}.text.jsonl.gz",
        )

    def _temporary_paths(self, shard_index: int) -> tuple[Path, Path]:
        stem = f"shard_{shard_index:05d}"
        return (
            self.output_dir / f"{stem}.pt.tmp",
            self.output_dir / f"{stem}.text.jsonl.gz.tmp",
        )

    def _ensure_output_paths_available(self, tensor_path: Path, sidecar_path: Path) -> None:
        tensor_exists = tensor_path.exists()
        sidecar_exists = sidecar_path.exists()

        if tensor_exists != sidecar_exists:
            raise FileExistsError(
                f"Inconsistent shard output state for {tensor_path.stem}: "
                f"tensor_exists={tensor_exists}, sidecar_exists={sidecar_exists}"
            )

        if (tensor_exists or sidecar_exists) and not self.allow_overwrite:
            raise FileExistsError(
                f"Refusing to overwrite existing shard pair: {tensor_path}, {sidecar_path}"
            )

    def _cleanup_temporary_files(self, *paths: Path) -> None:
        for path in paths:
            if path.exists():
                path.unlink()


def build_shard_payload(
    records: list[EmbeddedSampleRecord],
    shard_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not records:
        raise ValueError("Cannot build a shard from an empty buffer")

    message_embeddings = stack_embeddings(
        [record.message_embedding for record in records],
        "message_embeddings",
    )
    diff_embeddings = stack_embeddings(
        [record.diff_embedding for record in records],
        "diff_embeddings",
    )
    context_embeddings = stack_embeddings(
        [record.context_embedding for record in records],
        "context_embeddings",
    )

    sample_ids = [record.raw_sample.sample_id for record in records]
    repos = [record.raw_sample.repo for record in records]
    commit_shas = [record.raw_sample.commit_sha for record in records]
    build_ids = [record.raw_sample.build_id for record in records]
    labels = labels_to_tensor([record.raw_sample.label for record in records])
    record_indices = torch.arange(len(records), dtype=torch.long)

    sidecar_records = [
        build_sidecar_record(record=record, record_index=record_index)
        for record_index, record in enumerate(records)
    ]

    payload = {
        "format_version": FORMAT_VERSION,
        "shard_index": shard_index,
        "num_samples": len(records),
        "sample_ids": sample_ids,
        "repos": repos,
        "commit_shas": commit_shas,
        "build_ids": build_ids,
        "labels": labels,
        "message_embeddings": message_embeddings,
        "diff_embeddings": diff_embeddings,
        "context_embeddings": context_embeddings,
        "record_indices": record_indices,
    }

    return payload, sidecar_records


def stack_embeddings(embeddings: list[torch.Tensor], field_name: str) -> torch.Tensor:
    if not embeddings:
        raise ValueError(f"{field_name} cannot be empty")

    normalized = []
    expected_shape = None

    for index, embedding in enumerate(embeddings):
        if not isinstance(embedding, torch.Tensor):
            raise TypeError(f"{field_name}[{index}] must be a torch.Tensor")

        if embedding.ndim == 2 and embedding.shape[0] == 1:
            embedding = embedding.squeeze(0)

        if embedding.ndim != 1:
            raise ValueError(
                f"{field_name}[{index}] must be one-dimensional before stacking, "
                f"got shape {tuple(embedding.shape)}"
            )

        if expected_shape is None:
            expected_shape = tuple(embedding.shape)
        elif tuple(embedding.shape) != expected_shape:
            raise ValueError(
                f"Inconsistent {field_name} dimensions: expected {expected_shape}, "
                f"got {tuple(embedding.shape)} at index {index}"
            )

        normalized.append(embedding.detach().cpu())

    return torch.stack(normalized)


def labels_to_tensor(labels: list[Any]) -> torch.Tensor:
    values = []

    for label in labels:
        if label is None:
            values.append(float("nan"))
        elif isinstance(label, bool):
            values.append(float(int(label)))
        elif isinstance(label, (int, float)):
            values.append(float(label))
        else:
            try:
                values.append(float(label))
            except (TypeError, ValueError):
                values.append(float("nan"))

    return torch.tensor(values, dtype=torch.float32)


def build_sidecar_record(record: EmbeddedSampleRecord, record_index: int) -> dict[str, Any]:
    raw = record.raw_sample
    return {
        "format_version": FORMAT_VERSION,
        "sample_id": raw.sample_id,
        "record_index": record_index,
        "repo": raw.repo,
        "commit_sha": raw.commit_sha,
        "parent_commit_sha": raw.parent_commit_sha,
        "build_id": raw.build_id,
        "source_row_index": raw.source_row_index,
        "label": raw.label,
        "text": {
            "commit_message": raw.commit_message.text,
            "diff": raw.diff.text,
            "context": raw.context.text,
        },
        "provenance": {
            "commit_message": {
                "extraction": raw.commit_message.provenance,
                "tokenization": raw.commit_message.tokenization,
            },
            "diff": {
                "extraction": raw.diff.provenance,
                "tokenization": raw.diff.tokenization,
            },
            "context": {
                "extraction": raw.context.provenance,
                "tokenization": raw.context.tokenization,
            },
        },
    }


def validate_shard_payload(
    payload: dict[str, Any],
    sidecar_records: list[dict[str, Any]],
) -> None:
    missing_keys = REQUIRED_SHARD_KEYS - set(payload)
    if missing_keys:
        raise ValueError(f"Shard payload is missing required keys: {sorted(missing_keys)}")

    n = payload["num_samples"]
    if n <= 0:
        raise ValueError("Shard payload must contain at least one sample")

    aligned_list_fields = ["sample_ids", "repos", "commit_shas", "build_ids"]
    for field_name in aligned_list_fields:
        if len(payload[field_name]) != n:
            raise ValueError(f"{field_name} length does not match num_samples")

    if payload["labels"].shape[0] != n:
        raise ValueError("labels first dimension does not match num_samples")

    if len(sidecar_records) != n:
        raise ValueError("sidecar record count does not match num_samples")

    for field_name in ["message_embeddings", "diff_embeddings", "context_embeddings"]:
        tensor = payload[field_name]
        if tensor.shape[0] != n:
            raise ValueError(f"{field_name} first dimension does not match num_samples")
        if tensor.ndim != 2:
            raise ValueError(f"{field_name} must be two-dimensional")

    if not torch.equal(payload["record_indices"], torch.arange(n, dtype=torch.long)):
        raise ValueError("record_indices must equal torch.arange(num_samples)")

    sample_ids = payload["sample_ids"]
    if len(set(sample_ids)) != n:
        raise ValueError("Duplicate sample_id exists within shard")

    for index, sidecar_record in enumerate(sidecar_records):
        if sidecar_record["record_index"] != index:
            raise ValueError(f"sidecar record {index} has mismatched record_index")
        if sidecar_record["sample_id"] != sample_ids[index]:
            raise ValueError(f"sidecar record {index} sample_id does not match tensor payload")


def to_json_safe(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return to_json_safe(asdict(value))

    if isinstance(value, dict):
        return {str(key): to_json_safe(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [to_json_safe(item) for item in value]

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, torch.Tensor):
        if value.ndim == 0:
            return to_json_safe(value.item())
        return to_json_safe(value.tolist())

    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:
            pass

    return value


def load_embedding_shard(path: str | Path) -> dict[str, Any]:
    shard_path = Path(path)
    payload = torch.load(shard_path, map_location="cpu")

    if not isinstance(payload, dict):
        raise ValueError(f"Embedding shard {shard_path} did not contain a dictionary")

    validate_shard_payload(payload, _dummy_sidecar_records(payload))
    return payload


def _dummy_sidecar_records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"record_index": index, "sample_id": sample_id}
        for index, sample_id in enumerate(payload.get("sample_ids", []))
    ]


def iter_text_sidecar(path: str | Path) -> Iterator[dict[str, Any]]:
    sidecar_path = Path(path)

    with gzip.open(sidecar_path, "rt", encoding="utf-8") as sidecar_file:
        for line in sidecar_file:
            if line.strip():
                yield json.loads(line)
