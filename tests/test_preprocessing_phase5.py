from __future__ import annotations

import json
import os
import uuid
from pathlib import Path

import pytest
import torch

from context_aware_ci_build_failure_prediction.preprocessing.modules.manifest import (
    ExistingOutputError,
    ManifestManager,
    build_dataset_metadata,
    build_preprocessing_metadata,
    load_and_validate_manifest,
    prepare_output_dir,
    sha256_file,
)
from context_aware_ci_build_failure_prediction.preprocessing.modules.shard_writer import (
    EmbeddedSampleRecord,
    EmbeddingShardWriter,
)
from context_aware_ci_build_failure_prediction.preprocessing.types import (
    RawSample,
    TextArtifact,
    TokenizationMetadata,
)


def workspace_output_dir() -> Path:
    output_dir = Path("embedding_shards_test") / "phase5" / uuid.uuid4().hex
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def artifact(name: str, text: str) -> TextArtifact:
    return TextArtifact(
        text=text,
        provenance={"source_type": name, "extraction_truncated": False},
        tokenization=TokenizationMetadata(
            token_count_before_truncation=len(text) + 2,
            retained_token_count=len(text) + 2,
            was_tokenizer_truncated=False,
        ),
    )


def sample_record(index: int, label: int | float = 0) -> EmbeddedSampleRecord:
    raw_sample = RawSample(
        sample_id=f"sha256:{index:064d}",
        source_row_index=100 + index,
        repo="owner/repo",
        commit_sha=f"commit-{index}",
        parent_commit_sha=f"parent-{index}",
        build_id=f"build-{index}",
        label=label,
        commit_message=artifact("commit_message", f"message text {index}"),
        diff=artifact("diff", f"diff text {index}"),
        context=artifact("context", f"context text {index}"),
    )
    return EmbeddedSampleRecord(
        raw_sample=raw_sample,
        message_embedding=torch.tensor([index, index + 0.1], dtype=torch.float32),
        diff_embedding=torch.tensor([index + 1, index + 1.1], dtype=torch.float32),
        context_embedding=torch.tensor([index + 2, index + 2.1], dtype=torch.float32),
    )


def manifest_manager(output_dir: Path, failed_count=lambda: 0) -> ManifestManager:
    return ManifestManager(
        output_dir=output_dir,
        dataset=build_dataset_metadata(
            source_csv="travistorrent.csv",
            source_csv_sha256="csvhash",
            repo_col="gh_project_name",
            commit_col="git_trigger_commit",
            label_col="tr_status",
            build_id_col="tr_build_id",
            parent_commit_col="git_prev_built_commit",
        ),
        embedding={
            "model_name": "fake-codebert",
            "embedding_dimension": 3,
            "max_length": 512,
            "pooling": "attention_mask_mean_pooling_last_hidden_state",
            "output_dtype": "torch.float16",
        },
        preprocessing=build_preprocessing_metadata(
            shard_size=2,
            raw_batch_size=4,
            embed_batch_size=8,
            max_diff_chars_per_file=20_000,
            max_total_diff_chars=100_000,
            max_changed_lines_per_file=20,
            max_context_chars_per_snippet=20_000,
            max_total_context_chars=150_000,
        ),
        failed_sample_count=failed_count,
    )


def write_one_manifested_shard(output_dir: Path, failed_count=lambda: 0) -> ManifestManager:
    manager = manifest_manager(output_dir, failed_count=failed_count)
    writer = EmbeddingShardWriter(
        output_dir=str(output_dir),
        shard_size=10,
        on_shard_complete=manager.record_completed_shard,
    )
    writer.add(sample_record(0, label=1))
    writer.add(sample_record(1, label=0))
    writer.close()
    manager.finalize()
    return manager


def test_manifest_creation_metadata_shards_and_totals():
    output_dir = workspace_output_dir()

    write_one_manifested_shard(output_dir)

    manifest = load_and_validate_manifest(output_dir / "manifest.json", verify_checksums=True)
    assert manifest["dataset"]["name"] == "travistorrent"
    assert manifest["dataset"]["source_csv"] == "travistorrent.csv"
    assert manifest["dataset"]["source_csv_sha256"] == "csvhash"
    assert manifest["dataset"]["repo_column"] == "gh_project_name"
    assert manifest["embedding"]["model_name"] == "fake-codebert"
    assert manifest["embedding"]["embedding_dimension"] == 3
    assert manifest["preprocessing"]["shard_size"] == 2
    assert manifest["preprocessing"]["embed_batch_size"] == 8
    assert manifest["totals"] == {
        "successful_samples": 2,
        "failed_samples": 0,
        "num_shards": 1,
    }

    shard = manifest["shards"][0]
    assert shard["tensor_file"] == "shard_00000.pt"
    assert shard["text_file"] == "shard_00000.text.jsonl.gz"
    assert shard["num_samples"] == 2
    assert shard["tensor_sha256"] == sha256_file(output_dir / shard["tensor_file"])
    assert shard["text_sha256"] == sha256_file(output_dir / shard["text_file"])


def test_manifest_checksum_validation_detects_modified_shard():
    output_dir = workspace_output_dir()
    write_one_manifested_shard(output_dir)

    with (output_dir / "shard_00000.pt").open("ab") as file:
        file.write(b"corruption")

    with pytest.raises(ValueError, match="Tensor checksum mismatch"):
        load_and_validate_manifest(output_dir / "manifest.json", verify_checksums=True)


def test_atomic_manifest_replacement_keeps_previous_manifest(monkeypatch):
    output_dir = workspace_output_dir()
    manager = manifest_manager(output_dir)
    manager.finalize()
    previous_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

    original_replace = os.replace

    def failing_replace(src, dst):
        if Path(dst).name == "manifest.json":
            raise OSError("replace failed")
        original_replace(src, dst)

    monkeypatch.setattr(os, "replace", failing_replace)
    writer = EmbeddingShardWriter(
        output_dir=str(output_dir),
        shard_size=10,
        on_shard_complete=manager.record_completed_shard,
    )
    writer.add(sample_record(0))

    with pytest.raises(RuntimeError, match="manifest update failed"):
        writer.flush()

    monkeypatch.setattr(os, "replace", original_replace)
    current_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert current_manifest == previous_manifest
    assert load_and_validate_manifest(output_dir / "manifest.json") == previous_manifest


def test_existing_output_protection_and_overwrite_cleanup():
    output_dir = workspace_output_dir()
    unrelated = output_dir / "keep.me"
    unrelated.write_text("do not delete", encoding="utf-8")
    (output_dir / "manifest.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ExistingOutputError, match="Existing manifest"):
        prepare_output_dir(output_dir, overwrite=False)

    removed = prepare_output_dir(output_dir, overwrite=True)
    assert [path.name for path in removed] == ["manifest.json"]
    assert unrelated.exists()

    (output_dir / "shard_00000.pt").write_bytes(b"orphan")
    with pytest.raises(ExistingOutputError, match="Inconsistent unpaired"):
        prepare_output_dir(output_dir, overwrite=False)

    prepare_output_dir(output_dir, overwrite=True)
    (output_dir / "shard_00000.pt.tmp").write_bytes(b"interrupted")
    with pytest.raises(ExistingOutputError, match="Interrupted preprocessing output"):
        prepare_output_dir(output_dir, overwrite=False)


def test_failure_totals_count_sample_failures_not_shard_write_failures():
    output_dir = workspace_output_dir()
    failed = {"count": 1}
    manager = write_one_manifested_shard(output_dir, failed_count=lambda: failed["count"])
    manifest = load_and_validate_manifest(output_dir / "manifest.json")
    assert manifest["totals"]["failed_samples"] == 1
    assert manifest["totals"]["successful_samples"] == 2

    def failing_record_completed_shard(*args, **kwargs):
        raise RuntimeError("manifest boom")

    writer = EmbeddingShardWriter(
        output_dir=str(workspace_output_dir()),
        shard_size=10,
        on_shard_complete=failing_record_completed_shard,
    )
    writer.add(sample_record(2))

    with pytest.raises(RuntimeError, match="manifest update failed"):
        writer.flush()

    failed["count"] = 1
    manager.finalize()
    manifest = load_and_validate_manifest(output_dir / "manifest.json")
    assert manifest["totals"]["failed_samples"] == 1


def test_manifest_validation_rejects_bad_schema_and_references():
    output_dir = workspace_output_dir()
    write_one_manifested_shard(output_dir)
    manifest_path = output_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    bad_path = output_dir / "missing-section.json"
    bad_path.write_text(json.dumps({"format_version": 1}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing required sections"):
        load_and_validate_manifest(bad_path)

    duplicate = dict(manifest)
    duplicate["shards"] = manifest["shards"] + [dict(manifest["shards"][0])]
    duplicate_path = output_dir / "duplicate.json"
    duplicate_path.write_text(json.dumps(duplicate), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate shard indices"):
        load_and_validate_manifest(duplicate_path)

    non_contiguous = dict(manifest)
    non_contiguous["shards"] = [dict(manifest["shards"][0], shard_index=1)]
    non_contiguous_path = output_dir / "non-contiguous.json"
    non_contiguous_path.write_text(json.dumps(non_contiguous), encoding="utf-8")
    with pytest.raises(ValueError, match="contiguous"):
        load_and_validate_manifest(non_contiguous_path)

    missing_file = dict(manifest)
    missing_file["shards"] = [dict(manifest["shards"][0], tensor_file="shard_00099.pt")]
    missing_file_path = output_dir / "missing-file.json"
    missing_file_path.write_text(json.dumps(missing_file), encoding="utf-8")
    with pytest.raises(ValueError, match="filenames"):
        load_and_validate_manifest(missing_file_path)

    bad_totals = dict(manifest)
    bad_totals["totals"] = dict(manifest["totals"], successful_samples=999)
    bad_totals_path = output_dir / "bad-totals.json"
    bad_totals_path.write_text(json.dumps(bad_totals), encoding="utf-8")
    with pytest.raises(ValueError, match="successful sample total"):
        load_and_validate_manifest(bad_totals_path)
