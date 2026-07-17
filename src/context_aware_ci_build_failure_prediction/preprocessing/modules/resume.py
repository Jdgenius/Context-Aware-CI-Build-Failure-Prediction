from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .manifest import (
    ExistingOutputError,
    find_generated_output_paths,
    load_and_validate_manifest,
)
from .shard_writer import iter_text_sidecar, load_embedding_shard


@dataclass
class ResumeState:
    enabled: bool = False
    manifest: dict[str, Any] | None = None
    completed_sample_ids: set[str] = field(default_factory=set)
    completed_shards_at_start: int = 0
    completed_samples_at_start: int = 0
    starting_shard_index: int = 0
    removed_incomplete_paths: list[Path] = field(default_factory=list)

    @property
    def estimated_sample_id_memory_bytes(self) -> int:
        return sum(sys.getsizeof(sample_id) for sample_id in self.completed_sample_ids)


def prepare_resume_state(
    output_dir: str | Path,
    expected_dataset: dict[str, Any],
    expected_embedding: dict[str, Any],
    expected_preprocessing: dict[str, Any],
) -> ResumeState:
    output_path = Path(output_dir)
    manifest_path = output_path / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Cannot resume without an existing manifest.json")

    manifest = load_and_validate_manifest(manifest_path, verify_checksums=True)
    validate_manifest_compatibility(
        manifest=manifest,
        expected_dataset=expected_dataset,
        expected_embedding=expected_embedding,
        expected_preprocessing=expected_preprocessing,
    )
    removed_paths = remove_incomplete_resume_outputs(output_path, manifest)
    completed_sample_ids = recover_completed_sample_ids(output_path, manifest)
    shards = manifest["shards"]

    return ResumeState(
        enabled=True,
        manifest=manifest,
        completed_sample_ids=completed_sample_ids,
        completed_shards_at_start=len(shards),
        completed_samples_at_start=len(completed_sample_ids),
        starting_shard_index=(max((shard["shard_index"] for shard in shards), default=-1) + 1),
        removed_incomplete_paths=removed_paths,
    )


def validate_manifest_compatibility(
    manifest: dict[str, Any],
    expected_dataset: dict[str, Any],
    expected_embedding: dict[str, Any],
    expected_preprocessing: dict[str, Any],
) -> None:
    mismatches: list[str] = []
    compare_dict("dataset", manifest["dataset"], expected_dataset, mismatches)
    compare_dict("preprocessing", manifest["preprocessing"], expected_preprocessing, mismatches)

    embedding_fields = [
        "model_name",
        "embedding_dimension",
        "max_length",
        "pooling",
        "output_dtype",
    ]
    actual_embedding = {
        field: manifest["embedding"].get(field)
        for field in embedding_fields
    }
    compatible_embedding = {
        field: expected_embedding.get(field)
        for field in embedding_fields
    }
    compare_dict("embedding", actual_embedding, compatible_embedding, mismatches)

    if mismatches:
        detail = "\n".join(f"- {mismatch}" for mismatch in mismatches)
        raise ValueError(f"Cannot resume incompatible preprocessing run:\n{detail}")


def compare_dict(
    section: str,
    actual: dict[str, Any],
    expected: dict[str, Any],
    mismatches: list[str],
) -> None:
    keys = sorted(set(actual) | set(expected))
    for key in keys:
        if actual.get(key) != expected.get(key):
            mismatches.append(
                f"{section}.{key}: manifest={actual.get(key)!r}, current={expected.get(key)!r}"
            )


def remove_incomplete_resume_outputs(output_dir: Path, manifest: dict[str, Any]) -> list[Path]:
    completed_files = {
        entry["tensor_file"]
        for entry in manifest["shards"]
    } | {
        entry["text_file"]
        for entry in manifest["shards"]
    }
    generated_paths = find_generated_output_paths(output_dir)
    final_unlisted: dict[int, list[Path]] = {}
    removed_paths: list[Path] = []

    for path in generated_paths:
        if path.name == "manifest.json":
            continue
        if path.name.endswith(".tmp"):
            path.unlink()
            removed_paths.append(path)
            continue
        if path.name in completed_files:
            continue
        if path.name.startswith("shard_"):
            index = shard_index(path.name)
            if index is not None:
                final_unlisted.setdefault(index, []).append(path)

    for index, paths in sorted(final_unlisted.items()):
        names = {path.name for path in paths}
        expected_pair = {
            f"shard_{index:05d}.pt",
            f"shard_{index:05d}.text.jsonl.gz",
        }
        if names == expected_pair:
            raise ExistingOutputError(
                f"Unlisted completed-looking shard pair exists for index {index:05d}; "
                "refusing to trust it during resume"
            )
        for path in paths:
            path.unlink()
            removed_paths.append(path)

    return removed_paths


def recover_completed_sample_ids(output_dir: Path, manifest: dict[str, Any]) -> set[str]:
    completed_sample_ids: set[str] = set()

    for entry in manifest["shards"]:
        tensor_path = output_dir / entry["tensor_file"]
        text_path = output_dir / entry["text_file"]
        payload = load_embedding_shard(tensor_path)
        sidecar_records = list(iter_text_sidecar(text_path))
        expected_count = entry["num_samples"]

        if payload["shard_index"] != entry["shard_index"]:
            raise ValueError(f"Shard index mismatch for {entry['tensor_file']}")
        if payload["num_samples"] != expected_count:
            raise ValueError(f"Tensor sample count mismatch for shard {entry['shard_index']}")
        if len(sidecar_records) != expected_count:
            raise ValueError(f"Sidecar sample count mismatch for shard {entry['shard_index']}")

        sample_ids = payload["sample_ids"]
        for index, sidecar_record in enumerate(sidecar_records):
            sample_id = sample_ids[index]
            if sidecar_record.get("record_index") != index:
                raise ValueError(f"Sidecar record index mismatch in shard {entry['shard_index']}")
            if sidecar_record.get("sample_id") != sample_id:
                raise ValueError(f"Sidecar sample_id mismatch in shard {entry['shard_index']}")
            if sample_id in completed_sample_ids:
                raise ValueError(f"Duplicate sample_id exists across completed shards: {sample_id}")
            completed_sample_ids.add(sample_id)

    return completed_sample_ids


def shard_index(filename: str) -> int | None:
    try:
        return int(filename.split("_", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        return None
