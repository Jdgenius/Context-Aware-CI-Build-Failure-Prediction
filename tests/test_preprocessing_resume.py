from __future__ import annotations

import json
import uuid
from pathlib import Path

import pandas as pd
import pytest
import torch

from context_aware_ci_build_failure_prediction.preprocessing import cli
from context_aware_ci_build_failure_prediction.preprocessing import main as main_module
from context_aware_ci_build_failure_prediction.preprocessing.modules.manifest import (
    ExistingOutputError,
    ManifestManager,
    build_dataset_metadata,
    build_preprocessing_metadata,
    load_and_validate_manifest,
    sha256_file,
)
from context_aware_ci_build_failure_prediction.preprocessing.modules.resume import (
    prepare_resume_state,
)
from context_aware_ci_build_failure_prediction.preprocessing.modules.shard_writer import (
    EmbeddedSampleRecord,
    EmbeddingShardWriter,
    iter_text_sidecar,
    load_embedding_shard,
)
from context_aware_ci_build_failure_prediction.preprocessing.types import (
    RawSample,
    TextArtifact,
    make_sample_id,
)


def workspace_path(name: str) -> Path:
    path = Path("embedding_shards_test") / "resume" / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv(path: Path) -> pd.DataFrame:
    df = pd.DataFrame(
        [
            {
                "gh_project_name": "owner/complete",
                "git_trigger_commit": "commit-a",
                "tr_status": 1,
                "tr_build_id": "build-a",
                "git_prev_built_commit": "parent-a",
            },
            {
                "gh_project_name": "owner/partial",
                "git_trigger_commit": "commit-b",
                "tr_status": 0,
                "tr_build_id": "build-b",
                "git_prev_built_commit": "parent-b",
            },
            {
                "gh_project_name": "owner/partial",
                "git_trigger_commit": "commit-c",
                "tr_status": 1,
                "tr_build_id": "build-c",
                "git_prev_built_commit": "parent-c",
            },
        ]
    )
    df.to_csv(path, index=False)
    return df


def embedding_metadata() -> dict:
    return {
        "model_name": "fake-codebert",
        "embedding_dimension": 2,
        "max_length": 12,
        "pooling": "fake_pooling",
        "output_dtype": "torch.float32",
        "device": "cpu",
    }


def preprocessing_metadata(shard_size: int = 2) -> dict:
    return build_preprocessing_metadata(
        shard_size=shard_size,
        raw_batch_size=4,
        embed_batch_size=2,
        max_diff_chars_per_file=20_000,
        max_total_diff_chars=100_000,
        max_changed_lines_per_file=20,
        max_context_chars_per_snippet=20_000,
        max_total_context_chars=150_000,
    )


def dataset_metadata(csv_path: Path) -> dict:
    return build_dataset_metadata(
        source_csv=csv_path,
        source_csv_sha256=sha256_file(csv_path),
        repo_col="gh_project_name",
        commit_col="git_trigger_commit",
        label_col="tr_status",
        build_id_col="tr_build_id",
        parent_commit_col="git_prev_built_commit",
    )


def sample_id_for_row(row: pd.Series, source_row_index: int) -> str:
    return make_sample_id(
        repo=row["gh_project_name"],
        commit_sha=row["git_trigger_commit"],
        build_id=row["tr_build_id"],
        source_row_index=source_row_index,
    )


def sample_record(row: pd.Series, source_row_index: int) -> EmbeddedSampleRecord:
    raw_sample = RawSample(
        sample_id=sample_id_for_row(row, source_row_index),
        source_row_index=source_row_index,
        repo=row["gh_project_name"],
        commit_sha=row["git_trigger_commit"],
        parent_commit_sha=row["git_prev_built_commit"],
        build_id=row["tr_build_id"],
        label=row["tr_status"],
        commit_message=TextArtifact(text=f"message {source_row_index}", provenance={}),
        diff=TextArtifact(text=f"diff {source_row_index}", provenance={}),
        context=TextArtifact(text=f"context {source_row_index}", provenance={}),
    )
    vector = torch.tensor([source_row_index, source_row_index + 0.5], dtype=torch.float32)
    return EmbeddedSampleRecord(
        raw_sample=raw_sample,
        message_embedding=vector,
        diff_embedding=vector + 1,
        context_embedding=vector + 2,
    )


def write_manifested_shard(output_dir: Path, csv_path: Path, records: list[EmbeddedSampleRecord]) -> None:
    manager = ManifestManager(
        output_dir=output_dir,
        dataset=dataset_metadata(csv_path),
        embedding=embedding_metadata(),
        preprocessing=preprocessing_metadata(),
        failed_sample_count=lambda: 0,
    )
    writer = EmbeddingShardWriter(
        output_dir=str(output_dir),
        shard_size=10,
        on_shard_complete=manager.record_completed_shard,
    )
    for record in records:
        writer.add(record)
    writer.close()
    manager.finalize()


class FakeEmbedder:
    metadata = embedding_metadata()


def fake_process_one_repo_to_embeddings(**kwargs) -> None:
    repo_df = kwargs["repo_df"]
    writer = kwargs["writer"]
    for _, row in repo_df.iterrows():
        writer.add(sample_record(row, int(row["__source_row_index"])))


def test_resume_skips_completed_rows_and_continues_shard_indices(monkeypatch):
    output_dir = workspace_path("output")
    csv_path = workspace_path("csv") / "travistorrent.csv"
    df = write_csv(csv_path)
    write_manifested_shard(output_dir, csv_path, [sample_record(df.iloc[0], 0), sample_record(df.iloc[1], 1)])
    old_manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    processed_repos = []

    def recording_process_one_repo_to_embeddings(**kwargs) -> None:
        processed_repos.append((kwargs["repo_name"], len(kwargs["repo_df"])))
        fake_process_one_repo_to_embeddings(**kwargs)

    monkeypatch.setattr(main_module, "CodeBERTEmbedder", FakeEmbedder)
    monkeypatch.setattr(main_module, "process_one_repo_to_embeddings", recording_process_one_repo_to_embeddings)

    result = main_module.process_travistorrent_to_codebert_embeddings(
        travistorrent_csv_path=str(csv_path),
        output_dir=str(output_dir),
        temp_repo_root=str(workspace_path("temp")),
        failure_log_path=str(output_dir / "failures.jsonl"),
        shard_size=2,
        raw_batch_size=4,
        embed_batch_size=2,
        resume=True,
    )

    manifest = load_and_validate_manifest(output_dir / "manifest.json", verify_checksums=True)
    assert result["resume"] == {
        "enabled": True,
        "completed_samples_at_start": 2,
        "completed_shards_at_start": 1,
        "skipped_completed_samples": 2,
        "starting_shard_index": 1,
        "new_successful_samples": 1,
    }
    assert processed_repos == [("owner/partial", 1)]
    assert manifest["shards"][0] == old_manifest["shards"][0]
    assert [shard["shard_index"] for shard in manifest["shards"]] == [0, 1]
    assert manifest["totals"]["successful_samples"] == 3

    sample_ids = []
    for shard in manifest["shards"]:
        payload = load_embedding_shard(output_dir / shard["tensor_file"])
        sidecars = list(iter_text_sidecar(output_dir / shard["text_file"]))
        sample_ids.extend(payload["sample_ids"])
        assert payload["sample_ids"] == [record["sample_id"] for record in sidecars]
    assert len(sample_ids) == len(set(sample_ids))
    assert sample_id_for_row(df.iloc[2], 2) in sample_ids


def test_resume_refuses_incompatible_configuration():
    output_dir = workspace_path("output")
    csv_path = workspace_path("csv") / "travistorrent.csv"
    df = write_csv(csv_path)
    write_manifested_shard(output_dir, csv_path, [sample_record(df.iloc[0], 0)])

    bad_preprocessing = preprocessing_metadata(shard_size=99)
    with pytest.raises(ValueError, match="preprocessing.shard_size"):
        prepare_resume_state(
            output_dir=output_dir,
            expected_dataset=dataset_metadata(csv_path),
            expected_embedding=embedding_metadata(),
            expected_preprocessing=bad_preprocessing,
        )


def test_resume_removes_incomplete_files_but_refuses_unlisted_pairs():
    output_dir = workspace_path("output")
    csv_path = workspace_path("csv") / "travistorrent.csv"
    df = write_csv(csv_path)
    write_manifested_shard(output_dir, csv_path, [sample_record(df.iloc[0], 0)])

    (output_dir / "manifest.json.tmp").write_text("partial", encoding="utf-8")
    (output_dir / "shard_00001.pt").write_bytes(b"orphan")
    state = prepare_resume_state(
        output_dir=output_dir,
        expected_dataset=dataset_metadata(csv_path),
        expected_embedding=embedding_metadata(),
        expected_preprocessing=preprocessing_metadata(),
    )
    assert sorted(path.name for path in state.removed_incomplete_paths) == [
        "manifest.json.tmp",
        "shard_00001.pt",
    ]
    assert not (output_dir / "shard_00001.pt").exists()

    (output_dir / "shard_00002.pt").write_bytes(b"unlisted tensor")
    (output_dir / "shard_00002.text.jsonl.gz").write_bytes(b"unlisted sidecar")
    with pytest.raises(ExistingOutputError, match="Unlisted completed-looking shard pair"):
        prepare_resume_state(
            output_dir=output_dir,
            expected_dataset=dataset_metadata(csv_path),
            expected_embedding=embedding_metadata(),
            expected_preprocessing=preprocessing_metadata(),
        )


def test_resume_and_overwrite_conflict(monkeypatch):
    csv_path = workspace_path("csv") / "travistorrent.csv"
    write_csv(csv_path)
    called = False

    def fake_process(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "process_travistorrent_to_codebert_embeddings", fake_process)

    exit_code = cli.main(
        [
            "run",
            "--travistorrent-csv-path",
            str(csv_path),
            "--output-dir",
            str(workspace_path("output")),
            "--resume",
            "--overwrite",
        ]
    )

    assert exit_code == 1
    assert called is False


def test_cli_writes_resume_summary(monkeypatch):
    csv_path = workspace_path("csv") / "travistorrent.csv"
    write_csv(csv_path)
    output_dir = workspace_path("output")
    summary_path = workspace_path("summary") / "summary.json"
    captured = {}

    def fake_process(**kwargs):
        captured.update(kwargs)
        (output_dir / "manifest.json").write_text("{}", encoding="utf-8")
        return {
            "resume": {
                "enabled": True,
                "completed_samples_at_start": 2,
                "completed_shards_at_start": 1,
                "skipped_completed_samples": 2,
                "starting_shard_index": 1,
                "new_successful_samples": 1,
            }
        }

    monkeypatch.setattr(cli, "process_travistorrent_to_codebert_embeddings", fake_process)
    monkeypatch.setattr(
        cli,
        "load_and_validate_manifest",
        lambda path, verify_checksums=False: {
            "totals": {
                "successful_samples": 3,
                "failed_samples": 0,
                "num_shards": 2,
            }
        },
    )

    exit_code = cli.main(
        [
            "run",
            "--travistorrent-csv-path",
            str(csv_path),
            "--output-dir",
            str(output_dir),
            "--run-summary-path",
            str(summary_path),
            "--resume",
        ]
    )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert exit_code == 0
    assert captured["resume"] is True
    assert summary["resume"]["enabled"] is True
    assert summary["resume"]["new_successful_samples"] == 1
