from pathlib import Path

import pandas as pd
import pytest

from context_aware_ci_build_failure_prediction.preprocessing import main as main_module
from context_aware_ci_build_failure_prediction.preprocessing import process as process_module
from context_aware_ci_build_failure_prediction.preprocessing.types import (
    SOURCE_ROW_INDEX_COL,
    RawSample,
    TextArtifact,
    TokenizationMetadata,
    make_sample_id,
    normalize_optional_value,
)


def test_make_sample_id_is_deterministic():
    first = make_sample_id("owner/repo", "abc123", "42", 7)
    second = make_sample_id("owner/repo", "abc123", "42", 7)

    assert first == second
    assert first.startswith("sha256:")
    assert len(first) == len("sha256:") + 64


def test_make_sample_id_changes_with_source_row_index():
    first = make_sample_id("owner/repo", "abc123", "42", 7)
    second = make_sample_id("owner/repo", "abc123", "42", 8)

    assert first != second


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (float("nan"), None),
        (pd.NA, None),
        ("", None),
        ("   ", None),
        (" abc ", "abc"),
        (123, "123"),
        (123.0, "123"),
    ],
)
def test_normalize_optional_value(value, expected):
    assert normalize_optional_value(value) == expected


def test_add_source_row_index_preserves_csv_order():
    df = pd.DataFrame({"repo": ["b", "a"], "commit": ["2", "1"]})

    main_module.add_source_row_index(df)

    assert df[SOURCE_ROW_INDEX_COL].tolist() == [0, 1]


def test_missing_optional_columns_warn_once_and_do_not_fail(monkeypatch, capsys):
    df = pd.DataFrame(
        {
            "gh_project_name": ["owner/repo"],
            "git_trigger_commit": ["abc123"],
            "tr_status": ["passed"],
        }
    )

    class DummyWriter:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    monkeypatch.setattr(main_module.pd, "read_csv", lambda path: df)
    monkeypatch.setattr(main_module, "TempRepoManager", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "CodeBERTEmbedder", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "EmbeddingShardWriter", DummyWriter)
    monkeypatch.setattr(main_module, "JsonlLogger", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        main_module,
        "process_one_repo_to_embeddings",
        lambda *args, **kwargs: None,
    )

    main_module.process_travistorrent_to_codebert_embeddings(
        travistorrent_csv_path="unused.csv",
        max_repos=0,
        output_dir="embedding_shards_test/phase1-main",
        overwrite=True,
    )

    output = capsys.readouterr().out
    assert "Warning: build_id_col 'tr_build_id' not found." in output
    assert "Warning: parent_commit_col 'git_prev_built_commit' not found." in output
    assert output.count("build_id_col") == 1
    assert output.count("parent_commit_col") == 1


def test_top_level_preserves_source_row_index_through_grouping(monkeypatch):
    df = pd.DataFrame(
        {
            "gh_project_name": ["owner/repo"],
            "git_trigger_commit": ["abc123"],
            "tr_status": ["passed"],
        }
    )
    captured = {}

    class DummyWriter:
        def __init__(self, *args, **kwargs):
            pass

        def close(self):
            pass

    def capture_repo_df(*args, **kwargs):
        captured["repo_df"] = kwargs["repo_df"].copy()

    monkeypatch.setattr(main_module.pd, "read_csv", lambda path: df)
    monkeypatch.setattr(main_module, "TempRepoManager", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "CodeBERTEmbedder", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "EmbeddingShardWriter", DummyWriter)
    monkeypatch.setattr(main_module, "JsonlLogger", lambda *args, **kwargs: object())
    monkeypatch.setattr(main_module, "process_one_repo_to_embeddings", capture_repo_df)

    main_module.process_travistorrent_to_codebert_embeddings(
        "unused.csv",
        max_repos=1,
        output_dir="embedding_shards_test/phase1-main",
        overwrite=True,
    )

    assert captured["repo_df"][SOURCE_ROW_INDEX_COL].tolist() == [0]


def test_build_raw_sample_contains_identity_fields(monkeypatch):
    row = {
        SOURCE_ROW_INDEX_COL: 5,
        "gh_project_name": "owner/repo",
        "git_trigger_commit": "abc123",
        "tr_status": "failed",
        "tr_build_id": 99,
        "git_prev_built_commit": "parent456",
    }

    monkeypatch.setattr(
        process_module,
        "build_commit_message_artifact",
        lambda repo_path, commit: TextArtifact(
            text="message",
            provenance={"source_type": "commit_message"},
        ),
    )
    monkeypatch.setattr(process_module, "get_changed_files", lambda repo_path, commit: ["a.py"])
    monkeypatch.setattr(
        process_module,
        "build_diff_artifact",
        lambda repo_path, commit_sha, changed_files, **kwargs: TextArtifact(
            text="diff",
            provenance={"source_type": "diff"},
        ),
    )
    monkeypatch.setattr(
        process_module,
        "build_context_artifact",
        lambda repo_path, commit_sha, changed_files, **kwargs: TextArtifact(
            text="context",
            provenance={"source_type": "context"},
        ),
    )

    sample = process_module.build_raw_sample_from_row(row=row, repo_path=Path("."))

    assert isinstance(sample, RawSample)
    assert sample.sample_id == make_sample_id("owner/repo", "abc123", "99", 5)
    assert sample.source_row_index == 5
    assert sample.repo == "owner/repo"
    assert sample.commit_sha == "abc123"
    assert sample.parent_commit_sha == "parent456"
    assert sample.build_id == "99"
    assert sample.label == "failed"
    assert sample.commit_message == TextArtifact(
        text="message",
        provenance={"source_type": "commit_message"},
    )
    assert sample.diff == TextArtifact(text="diff", provenance={"source_type": "diff"})
    assert sample.context == TextArtifact(
        text="context",
        provenance={"source_type": "context"},
    )


def test_build_raw_sample_missing_optional_columns_become_none(monkeypatch):
    row = {
        SOURCE_ROW_INDEX_COL: 5,
        "gh_project_name": "owner/repo",
        "git_trigger_commit": "abc123",
        "tr_status": "failed",
    }

    monkeypatch.setattr(
        process_module,
        "build_commit_message_artifact",
        lambda repo_path, commit: TextArtifact(text="", provenance={}),
    )
    monkeypatch.setattr(process_module, "get_changed_files", lambda repo_path, commit: [])
    monkeypatch.setattr(
        process_module,
        "build_diff_artifact",
        lambda repo_path, commit_sha, changed_files, **kwargs: TextArtifact(text="", provenance={}),
    )
    monkeypatch.setattr(
        process_module,
        "build_context_artifact",
        lambda repo_path, commit_sha, changed_files, **kwargs: TextArtifact(text="", provenance={}),
    )

    sample = process_module.build_raw_sample_from_row(row=row, repo_path=Path("."))

    assert sample.build_id is None
    assert sample.parent_commit_sha is None


def test_embed_and_write_raw_batch_uses_existing_text_values():
    sample = RawSample(
        sample_id="sha256:test",
        source_row_index=1,
        repo="owner/repo",
        commit_sha="abc123",
        parent_commit_sha=None,
        build_id=None,
        label="passed",
        commit_message=TextArtifact(text="message text", provenance={}),
        diff=TextArtifact(text="diff text", provenance={}),
        context=TextArtifact(text="context text", provenance={}),
    )
    seen_batches = []

    class FakeEmbedder:
        def embed_texts_with_metadata(self, texts, batch_size):
            seen_batches.append(list(texts))
            metadata = [
                TokenizationMetadata(
                    token_count_before_truncation=len(text),
                    retained_token_count=len(text),
                    was_tokenizer_truncated=False,
                )
                for text in texts
            ]
            return [f"embedding:{texts[0]}"], metadata

    class FakeWriter:
        def __init__(self):
            self.records = []

        def add(self, record):
            self.records.append(record)

    writer = FakeWriter()

    process_module.embed_and_write_raw_batch(
        raw_buffer=[sample],
        embedder=FakeEmbedder(),
        writer=writer,
        embed_batch_size=8,
    )

    assert seen_batches == [["message text"], ["diff text"], ["context text"]]
    assert len(writer.records) == 1
    record = writer.records[0]
    assert record.raw_sample.repo == "owner/repo"
    assert record.raw_sample.commit_sha == "abc123"
    assert record.message_embedding == "embedding:message text"
    assert record.diff_embedding == "embedding:diff text"
    assert record.context_embedding == "embedding:context text"
    assert record.raw_sample.label == "passed"
