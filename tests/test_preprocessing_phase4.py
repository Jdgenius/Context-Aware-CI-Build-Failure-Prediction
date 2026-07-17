from __future__ import annotations

import gzip
import json
import os
import uuid
from pathlib import Path

import pytest
import torch

from context_aware_ci_build_failure_prediction.preprocessing.modules.shard_writer import (
    EmbeddedSampleRecord,
    EmbeddingShardWriter,
    build_shard_payload,
    iter_text_sidecar,
    labels_to_tensor,
    load_embedding_shard,
)
from context_aware_ci_build_failure_prediction.preprocessing.types import (
    RawSample,
    TextArtifact,
    TokenizationMetadata,
)


def workspace_output_dir() -> Path:
    output_dir = Path("embedding_shards_test") / "phase4" / uuid.uuid4().hex
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


def test_successful_paired_write_schema_and_alignment():
    output_dir = workspace_output_dir()
    writer = EmbeddingShardWriter(output_dir=str(output_dir), shard_size=10)
    records = [sample_record(0, label=1), sample_record(1, label=0)]

    for record in records:
        writer.add(record)
    writer.flush()

    tensor_path = output_dir / "shard_00000.pt"
    sidecar_path = output_dir / "shard_00000.text.jsonl.gz"
    assert tensor_path.exists()
    assert sidecar_path.exists()

    payload = load_embedding_shard(tensor_path)
    sidecar_records = list(iter_text_sidecar(sidecar_path))

    assert payload["format_version"] == 1
    assert payload["shard_index"] == 0
    assert payload["num_samples"] == 2
    assert payload["sample_ids"] == [record.raw_sample.sample_id for record in records]
    assert payload["repos"] == ["owner/repo", "owner/repo"]
    assert payload["commit_shas"] == ["commit-0", "commit-1"]
    assert payload["build_ids"] == ["build-0", "build-1"]
    assert torch.equal(payload["record_indices"], torch.arange(2))
    assert payload["message_embeddings"].shape == (2, 2)
    assert payload["diff_embeddings"].shape == (2, 2)
    assert payload["context_embeddings"].shape == (2, 2)
    assert torch.equal(payload["labels"], torch.tensor([1.0, 0.0]))

    assert len(sidecar_records) == 2
    for index, sidecar_record in enumerate(sidecar_records):
        assert sidecar_record["record_index"] == index
        assert sidecar_record["sample_id"] == payload["sample_ids"][index]
        assert sidecar_record["text"]["commit_message"] == records[index].raw_sample.commit_message.text
        assert sidecar_record["text"]["diff"] == records[index].raw_sample.diff.text
        assert sidecar_record["text"]["context"] == records[index].raw_sample.context.text
        assert sidecar_record["provenance"]["diff"]["extraction"]["source_type"] == "diff"
        assert sidecar_record["provenance"]["diff"]["tokenization"]["retained_token_count"]


def test_labels_to_tensor_maps_status_strings_to_binary_values():
    labels = labels_to_tensor(["passed", "failed", "errored", "canceled", 1, 0, None])

    assert torch.equal(labels[:6], torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]))
    assert torch.isnan(labels[6])


def test_automatic_shard_boundaries_preserve_global_order():
    output_dir = workspace_output_dir()
    writer = EmbeddingShardWriter(output_dir=str(output_dir), shard_size=2)

    for index in range(5):
        writer.add(sample_record(index, label=index))
    writer.close()

    all_sample_ids = []
    for shard_index, expected_count in [(0, 2), (1, 2), (2, 1)]:
        payload = load_embedding_shard(output_dir / f"shard_{shard_index:05d}.pt")
        sidecar_records = list(
            iter_text_sidecar(output_dir / f"shard_{shard_index:05d}.text.jsonl.gz")
        )
        assert payload["num_samples"] == expected_count
        assert [record["record_index"] for record in sidecar_records] == list(range(expected_count))
        all_sample_ids.extend(payload["sample_ids"])

    assert all_sample_ids == [f"sha256:{index:064d}" for index in range(5)]
    assert len(set(all_sample_ids)) == 5


def test_validation_rejects_duplicate_sample_ids_and_inconsistent_embedding_dimensions():
    output_dir = workspace_output_dir()
    duplicate_a = sample_record(1)
    duplicate_b = sample_record(2)
    duplicate_b.raw_sample.sample_id = duplicate_a.raw_sample.sample_id
    writer = EmbeddingShardWriter(output_dir=str(output_dir), shard_size=10)
    writer.add(duplicate_a)
    writer.add(duplicate_b)

    with pytest.raises(ValueError, match="Duplicate sample_id"):
        writer.flush()

    bad_dimension = sample_record(3)
    bad_dimension.diff_embedding = torch.tensor([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="Inconsistent diff_embeddings"):
        build_shard_payload([sample_record(4), bad_dimension], shard_index=0)


def test_existing_unpaired_and_overwrite_outputs_are_rejected():
    output_dir = workspace_output_dir()
    (output_dir / "shard_00000.pt").write_bytes(b"orphan")
    writer = EmbeddingShardWriter(output_dir=str(output_dir), shard_size=10)
    writer.add(sample_record(0))

    with pytest.raises(FileExistsError, match="Inconsistent shard output state"):
        writer.flush()

    output_dir = workspace_output_dir()
    (output_dir / "shard_00000.pt").write_bytes(b"exists")
    with gzip.open(output_dir / "shard_00000.text.jsonl.gz", "wt", encoding="utf-8") as file:
        file.write("{}\n")
    writer = EmbeddingShardWriter(output_dir=str(output_dir), shard_size=10)
    writer.add(sample_record(0))

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        writer.flush()


@pytest.mark.parametrize(
    "failure_point",
    ["tensor_tmp_write", "sidecar_tmp_write", "first_promotion", "second_promotion"],
)
def test_atomic_failure_preserves_buffer_and_avoids_completed_unpaired_outputs(monkeypatch, failure_point):
    output_dir = workspace_output_dir()
    writer = EmbeddingShardWriter(output_dir=str(output_dir), shard_size=10)
    writer.add(sample_record(0))

    original_torch_save = torch.save
    original_gzip_open = gzip.open
    original_replace = os.replace

    if failure_point == "tensor_tmp_write":
        def failing_save(*args, **kwargs):
            raise OSError("tensor tmp failed")

        monkeypatch.setattr(torch, "save", failing_save)

    if failure_point == "sidecar_tmp_write":
        class FailingSidecar:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback):
                return False

            def write(self, text):
                raise OSError("sidecar tmp failed")

        def fake_gzip_open(*args, **kwargs):
            return FailingSidecar()

        monkeypatch.setattr(gzip, "open", fake_gzip_open)

    if failure_point in {"first_promotion", "second_promotion"}:
        calls = {"count": 0}

        def failing_replace(src, dst):
            calls["count"] += 1
            if failure_point == "first_promotion" and calls["count"] == 1:
                raise OSError("first promotion failed")
            if failure_point == "second_promotion" and calls["count"] == 2:
                raise OSError("second promotion failed")
            original_replace(src, dst)

        monkeypatch.setattr(os, "replace", failing_replace)

    with pytest.raises(RuntimeError, match="Failed to write paired shard"):
        writer.flush()

    assert len(writer.buffer) == 1
    assert writer.shard_index == 0
    assert not (output_dir / "shard_00000.pt").exists()
    assert not (output_dir / "shard_00000.text.jsonl.gz").exists()

    monkeypatch.setattr(torch, "save", original_torch_save)
    monkeypatch.setattr(gzip, "open", original_gzip_open)
    monkeypatch.setattr(os, "replace", original_replace)


def test_loader_validates_schema_without_reading_sidecar():
    output_dir = workspace_output_dir()
    writer = EmbeddingShardWriter(output_dir=str(output_dir), shard_size=10)
    writer.add(sample_record(0))
    writer.flush()

    sidecar_path = output_dir / "shard_00000.text.jsonl.gz"
    sidecar_path.unlink()
    payload = load_embedding_shard(output_dir / "shard_00000.pt")
    assert payload["num_samples"] == 1

    malformed_path = output_dir / "malformed.pt"
    torch.save({"format_version": 1}, malformed_path)

    with pytest.raises(ValueError, match="missing required keys"):
        load_embedding_shard(malformed_path)
