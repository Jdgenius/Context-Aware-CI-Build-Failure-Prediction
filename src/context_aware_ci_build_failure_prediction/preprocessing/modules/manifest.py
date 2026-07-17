from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any


FORMAT_VERSION = 1
GENERATED_OUTPUT_NAMES = {"manifest.json", "manifest.json.tmp"}
GENERATED_OUTPUT_PATTERNS = [
    "shard_*.pt",
    "shard_*.pt.tmp",
    "shard_*.text.jsonl.gz",
    "shard_*.text.jsonl.gz.tmp",
]


class ExistingOutputError(RuntimeError):
    pass


class ManifestManager:
    def __init__(
        self,
        output_dir: str | Path,
        dataset: dict[str, Any],
        embedding: dict[str, Any],
        preprocessing: dict[str, Any],
        failed_sample_count: Callable[[], int] | None = None,
        existing_manifest: dict[str, Any] | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.manifest_path = self.output_dir / "manifest.json"
        self.failed_sample_count = failed_sample_count or (lambda: 0)
        if existing_manifest is None:
            self.manifest: dict[str, Any] = {
                "format_version": FORMAT_VERSION,
                "dataset": dataset,
                "embedding": embedding,
                "preprocessing": preprocessing,
                "shards": [],
                "totals": {
                    "successful_samples": 0,
                    "failed_samples": self.failed_sample_count(),
                    "num_shards": 0,
                },
                "runs": [],
            }
        else:
            self.manifest = deepcopy(existing_manifest)
            self.manifest.setdefault("runs", [])

    def record_completed_shard(
        self,
        shard_index: int,
        tensor_path: str | Path,
        text_path: str | Path,
        num_samples: int,
    ) -> None:
        tensor_path = Path(tensor_path)
        text_path = Path(text_path)

        if not tensor_path.exists() or not text_path.exists():
            raise FileNotFoundError(
                f"Cannot record shard {shard_index:05d}; paired files are incomplete"
            )

        next_manifest = deepcopy(self.manifest)
        next_manifest["shards"].append(
            {
                "shard_index": shard_index,
                "tensor_file": tensor_path.name,
                "text_file": text_path.name,
                "num_samples": num_samples,
                "tensor_sha256": sha256_file(tensor_path),
                "text_sha256": sha256_file(text_path),
            }
        )
        update_totals(next_manifest, self.failed_sample_count())
        write_manifest_atomic(self.manifest_path, next_manifest)
        self.manifest = next_manifest

    def finalize(self) -> None:
        next_manifest = deepcopy(self.manifest)
        update_totals(next_manifest, self.failed_sample_count())
        write_manifest_atomic(self.manifest_path, next_manifest)
        self.manifest = next_manifest

    def record_run(self, run: dict[str, Any]) -> None:
        next_manifest = deepcopy(self.manifest)
        next_manifest.setdefault("runs", []).append(run)
        update_totals(next_manifest, self.failed_sample_count())
        write_manifest_atomic(self.manifest_path, next_manifest)
        self.manifest = next_manifest


def build_dataset_metadata(
    source_csv: str | Path,
    source_csv_sha256: str | None,
    repo_col: str,
    commit_col: str,
    label_col: str,
    build_id_col: str | None,
    parent_commit_col: str | None,
    dataset_name: str = "travistorrent",
) -> dict[str, Any]:
    return {
        "name": dataset_name,
        "source_csv": portable_source_path(source_csv),
        "source_csv_sha256": source_csv_sha256,
        "repo_column": repo_col,
        "commit_column": commit_col,
        "label_column": label_col,
        "build_id_column": build_id_col,
        "parent_commit_column": parent_commit_col,
    }


def build_preprocessing_metadata(
    shard_size: int,
    raw_batch_size: int,
    embed_batch_size: int,
    max_diff_chars_per_file: int,
    max_total_diff_chars: int,
    max_changed_lines_per_file: int,
    max_context_chars_per_snippet: int,
    max_total_context_chars: int,
) -> dict[str, Any]:
    return {
        "shard_size": shard_size,
        "raw_batch_size": raw_batch_size,
        "embed_batch_size": embed_batch_size,
        "max_diff_chars_per_file": max_diff_chars_per_file,
        "max_total_diff_chars": max_total_diff_chars,
        "max_changed_lines_per_file": max_changed_lines_per_file,
        "max_context_chars_per_snippet": max_context_chars_per_snippet,
        "max_total_context_chars": max_total_context_chars,
    }


def prepare_output_dir(output_dir: str | Path, overwrite: bool = False) -> list[Path]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated_paths = find_generated_output_paths(output_path)

    if not generated_paths:
        return []

    if not overwrite:
        raise ExistingOutputError(describe_existing_outputs(generated_paths))

    removed_paths = []
    for path in generated_paths:
        if path.is_file():
            path.unlink()
            removed_paths.append(path)

    return removed_paths


def find_generated_output_paths(output_dir: Path) -> list[Path]:
    paths: set[Path] = set()
    for name in GENERATED_OUTPUT_NAMES:
        path = output_dir / name
        if path.exists():
            paths.add(path)

    for pattern in GENERATED_OUTPUT_PATTERNS:
        paths.update(path for path in output_dir.glob(pattern) if path.exists())

    return sorted(paths)


def describe_existing_outputs(paths: list[Path]) -> str:
    tmp_paths = [path.name for path in paths if path.name.endswith(".tmp")]
    manifest_paths = [path.name for path in paths if path.name.startswith("manifest")]
    final_shards = [
        path
        for path in paths
        if path.name.startswith("shard_") and not path.name.endswith(".tmp")
    ]

    tensor_indices = {_shard_index(path.name) for path in final_shards if path.name.endswith(".pt")}
    text_indices = {
        _shard_index(path.name)
        for path in final_shards
        if path.name.endswith(".text.jsonl.gz")
    }
    unpaired = sorted(
        index
        for index in tensor_indices ^ text_indices
        if index is not None
    )

    if tmp_paths:
        return f"Interrupted preprocessing output exists: {tmp_paths}"
    if unpaired:
        return f"Inconsistent unpaired shard output exists for indices: {unpaired}"
    if manifest_paths:
        return f"Existing manifest output exists: {manifest_paths}"
    return f"Existing completed shard output exists: {[path.name for path in final_shards]}"


def load_and_validate_manifest(
    path: str | Path,
    verify_checksums: bool = False,
) -> dict[str, Any]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    required_sections = {
        "format_version",
        "dataset",
        "embedding",
        "preprocessing",
        "shards",
        "totals",
    }
    missing_sections = required_sections - set(manifest)
    if missing_sections:
        raise ValueError(f"Manifest missing required sections: {sorted(missing_sections)}")
    if manifest["format_version"] != FORMAT_VERSION:
        raise ValueError("Unsupported manifest format_version")

    shards = manifest["shards"]
    shard_indices = [entry.get("shard_index") for entry in shards]
    if len(set(shard_indices)) != len(shard_indices):
        raise ValueError("Manifest contains duplicate shard indices")
    if shard_indices != list(range(len(shards))):
        raise ValueError("Manifest shard indices must be contiguous starting at zero")

    successful_samples = 0
    for entry in shards:
        expected_tensor = f"shard_{entry['shard_index']:05d}.pt"
        expected_text = f"shard_{entry['shard_index']:05d}.text.jsonl.gz"
        if entry.get("tensor_file") != expected_tensor or entry.get("text_file") != expected_text:
            raise ValueError(f"Manifest shard filenames do not match shard_index {entry['shard_index']}")

        tensor_path = manifest_path.parent / entry["tensor_file"]
        text_path = manifest_path.parent / entry["text_file"]
        if not tensor_path.exists() or not text_path.exists():
            raise FileNotFoundError(f"Manifest references missing shard pair {entry['shard_index']}")

        if verify_checksums:
            if sha256_file(tensor_path) != entry["tensor_sha256"]:
                raise ValueError(f"Tensor checksum mismatch for shard {entry['shard_index']}")
            if sha256_file(text_path) != entry["text_sha256"]:
                raise ValueError(f"Text checksum mismatch for shard {entry['shard_index']}")

        successful_samples += entry["num_samples"]

    totals = manifest["totals"]
    if totals.get("num_shards") != len(shards):
        raise ValueError("Manifest totals.num_shards does not match shard entries")
    if totals.get("successful_samples") != successful_samples:
        raise ValueError("Manifest successful sample total does not match shard entries")

    return manifest


def write_manifest_atomic(path: str | Path, manifest: dict[str, Any]) -> None:
    manifest_path = Path(path)
    tmp_path = manifest_path.with_name(f"{manifest_path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2, sort_keys=True)
        file.write("\n")
        file.flush()
        os.fsync(file.fileno())

    os.replace(tmp_path, manifest_path)


def update_totals(manifest: dict[str, Any], failed_samples: int) -> None:
    manifest["totals"] = {
        "successful_samples": sum(shard["num_samples"] for shard in manifest["shards"]),
        "failed_samples": failed_samples,
        "num_shards": len(manifest["shards"]),
    }


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def portable_source_path(path: str | Path) -> str:
    return Path(path).name


def _shard_index(filename: str) -> int | None:
    try:
        return int(filename.split("_", 1)[1].split(".", 1)[0])
    except (IndexError, ValueError):
        return None
