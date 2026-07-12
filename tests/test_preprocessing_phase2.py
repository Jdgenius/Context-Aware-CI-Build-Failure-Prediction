from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pytest

from context_aware_ci_build_failure_prediction.preprocessing import process as process_module
from context_aware_ci_build_failure_prediction.preprocessing.helpers.context_extraction import (
    build_context_artifact,
    build_context_string,
)
from context_aware_ci_build_failure_prediction.preprocessing.helpers.git_extraction import (
    build_commit_message_artifact,
    build_diff_artifact,
    build_diff_string,
    get_commit_message,
)
from context_aware_ci_build_failure_prediction.preprocessing.types import (
    SOURCE_ROW_INDEX_COL,
    TextArtifact,
)


def run_git(repo_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return result.stdout


def init_repo(tmp_path: Path) -> Path:
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    run_git(repo_path, "init")
    run_git(repo_path, "config", "user.email", "test@example.com")
    run_git(repo_path, "config", "user.name", "Test User")
    return repo_path


def commit_all(repo_path: Path, message: str) -> str:
    run_git(repo_path, "add", "-A")
    run_git(repo_path, "commit", "-m", message)
    return run_git(repo_path, "rev-parse", "HEAD").strip()


def write_text(repo_path: Path, relative_path: str, text: str) -> None:
    path = repo_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_bytes(repo_path: Path, relative_path: str, data: bytes) -> None:
    path = repo_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


@pytest.fixture
def workspace_tmp() -> Path:
    base_path = Path("embedding_shards_test") / "phase2-test-repos"
    base_path.mkdir(parents=True, exist_ok=True)
    repo_parent = base_path / uuid.uuid4().hex
    repo_parent.mkdir()
    return repo_parent


@pytest.fixture
def diff_repo(workspace_tmp: Path) -> tuple[Path, str, list[str]]:
    repo_path = init_repo(workspace_tmp)
    write_text(repo_path, "modified.txt", "\n".join(f"line {i}" for i in range(1, 13)) + "\n")
    write_text(repo_path, "deleted.txt", "delete me\n")
    write_text(repo_path, "rename_old.txt", "same contents\n")
    write_bytes(repo_path, "binary.bin", b"\x00\x01old-binary")
    commit_all(repo_path, "base")

    modified_lines = [f"line {i}" for i in range(1, 13)]
    modified_lines[1] = "line 2 changed"
    modified_lines[7] = "line 8 changed"
    write_text(repo_path, "modified.txt", "\n".join(modified_lines) + "\n")
    write_text(repo_path, "added.txt", "added one\nadded two\n")
    (repo_path / "deleted.txt").unlink()
    run_git(repo_path, "mv", "rename_old.txt", "renamed.txt")
    write_bytes(repo_path, "binary.bin", b"\x00\x02new-binary")
    commit_sha = commit_all(repo_path, "phase 2 diff commit")

    return repo_path, commit_sha, [
        "modified.txt",
        "added.txt",
        "deleted.txt",
        "renamed.txt",
        "binary.bin",
    ]


def assert_offsets_slice_text(text: str, item: dict) -> None:
    assert text[item["start_offset"]:item["end_offset"]]
    if "diff_start_offset" in item:
        assert text[item["diff_start_offset"]:item["diff_end_offset"]]
    if "context_start_offset" in item:
        assert text[item["context_start_offset"]:item["context_end_offset"]]


def test_commit_message_artifact_preserves_existing_text(diff_repo):
    repo_path, commit_sha, _ = diff_repo

    artifact = build_commit_message_artifact(repo_path, commit_sha)

    assert artifact.text == get_commit_message(repo_path, commit_sha)
    assert artifact.provenance == {
        "source_type": "commit_message",
        "original_char_count": len(artifact.text),
        "retained_char_count": len(artifact.text),
        "extraction_truncated": False,
    }


def test_diff_string_wrapper_matches_artifact_text(diff_repo):
    repo_path, commit_sha, changed_files = diff_repo

    assert build_diff_string(repo_path, commit_sha, changed_files) == build_diff_artifact(
        repo_path,
        commit_sha,
        changed_files,
    ).text


def test_diff_provenance_tracks_retained_files_and_offsets(diff_repo):
    repo_path, commit_sha, changed_files = diff_repo

    artifact = build_diff_artifact(repo_path, commit_sha, changed_files)
    provenance = artifact.provenance

    assert provenance["retained_file_order"] == changed_files
    assert provenance["files_found"] == len(changed_files)
    assert provenance["files_retained"] == len(changed_files)

    by_path = {file_metadata["path"]: file_metadata for file_metadata in provenance["files"]}
    for file_metadata in provenance["files"]:
        assert_offsets_slice_text(artifact.text, file_metadata)
        assert artifact.text[
            file_metadata["diff_start_offset"]:file_metadata["diff_end_offset"]
        ] == artifact.text[file_metadata["start_offset"]:file_metadata["end_offset"]].split(
            "\n<FIELD_SEP>\n",
            1,
        )[1]
        for hunk in file_metadata["hunk_ranges"] or []:
            assert_offsets_slice_text(artifact.text, hunk)
            assert artifact.text[hunk["start_offset"]:hunk["end_offset"]].startswith("@@")

    modified = by_path["modified.txt"]
    assert modified["change_type"] == "modified"
    assert len(modified["hunk_ranges"]) >= 2
    assert 2 in modified["added_line_numbers"]
    assert 8 in modified["added_line_numbers"]
    assert 2 in modified["removed_line_numbers"]
    assert 8 in modified["removed_line_numbers"]

    assert by_path["added.txt"]["change_type"] == "added"
    assert by_path["deleted.txt"]["change_type"] == "deleted"
    assert by_path["renamed.txt"]["change_type"] == "renamed"
    assert by_path["renamed.txt"]["old_path"] == "rename_old.txt"
    assert by_path["renamed.txt"]["new_path"] == "renamed.txt"
    assert by_path["binary.bin"]["binary"] is True
    assert by_path["binary.bin"]["hunk_ranges"] is None
    assert by_path["binary.bin"]["added_line_numbers"] is None
    assert by_path["binary.bin"]["removed_line_numbers"] is None


def test_diff_provenance_records_extraction_truncation(diff_repo):
    repo_path, commit_sha, changed_files = diff_repo

    artifact = build_diff_artifact(
        repo_path,
        commit_sha,
        changed_files,
        max_diff_chars_per_file=40,
    )

    assert artifact.provenance["extraction_truncated"] is True
    assert artifact.provenance["files"][0]["truncated"] is True
    assert artifact.provenance["files"][0]["retained_char_count"] == 40


@pytest.fixture
def context_repo(workspace_tmp: Path) -> tuple[Path, str, list[str]]:
    repo_path = init_repo(workspace_tmp)
    write_text(
        repo_path,
        "Example.java",
        "\n".join(
            [
                "public class Example {",
                "  public void target() {",
                "    int x = 1;",
                "    int y = 2;",
                "    System.out.println(x + y);",
                "  }",
                "",
                "  public void other() {",
                "    int z = 3;",
                "  }",
                "}",
            ]
        )
        + "\n",
    )
    write_text(repo_path, "plain.txt", "\n".join(f"plain {i}" for i in range(1, 8)) + "\n")
    commit_all(repo_path, "base")

    write_text(
        repo_path,
        "Example.java",
        "\n".join(
            [
                "public class Example {",
                "  public void target() {",
                "    int x = 10;",
                "    int y = 20;",
                "    System.out.println(x + y);",
                "  }",
                "",
                "  public void other() {",
                "    int z = 3;",
                "  }",
                "}",
            ]
        )
        + "\n",
    )
    write_text(repo_path, "plain.txt", "plain 1\nplain changed\nplain 3\nplain 4\nplain 5\n")
    commit_sha = commit_all(repo_path, "phase 2 context commit")
    return repo_path, commit_sha, ["Example.java", "plain.txt"]


def test_context_string_wrapper_matches_artifact_text(context_repo):
    repo_path, commit_sha, changed_files = context_repo

    assert build_context_string(repo_path, commit_sha, changed_files) == build_context_artifact(
        repo_path,
        commit_sha,
        changed_files,
    ).text


def test_context_provenance_merges_duplicate_regions_and_offsets(context_repo):
    repo_path, commit_sha, changed_files = context_repo

    artifact = build_context_artifact(repo_path, commit_sha, changed_files)
    regions = artifact.provenance["regions"]
    by_file = {region["file_path"]: region for region in regions}

    java_region = by_file["Example.java"]
    assert java_region["start_line"] == 2
    assert java_region["end_line"] == 6
    assert java_region["symbol_name"] == "target"
    assert java_region["symbol_type"] == "function"
    assert java_region["extraction_method"] == "brace_based_function"
    assert java_region["triggering_changed_lines"] == [3, 4]
    assert len(java_region["occurrences"]) == 2

    plain_region = by_file["plain.txt"]
    assert plain_region["symbol_name"] is None
    assert plain_region["symbol_type"] is None
    assert plain_region["extraction_method"] == "global_window"
    assert plain_region["start_line"] == 1
    assert plain_region["end_line"] == 5

    for region in regions:
        for occurrence in region["occurrences"]:
            assert_offsets_slice_text(artifact.text, occurrence)
            assert artifact.text[
                occurrence["context_start_offset"]:occurrence["context_end_offset"]
            ]


def test_context_provenance_records_truncation(context_repo):
    repo_path, commit_sha, changed_files = context_repo

    artifact = build_context_artifact(
        repo_path,
        commit_sha,
        changed_files,
        max_context_chars_per_snippet=12,
    )

    assert artifact.provenance["extraction_truncated"] is True
    assert any(region["truncated"] for region in artifact.provenance["regions"])


def test_raw_sample_uses_artifacts_without_changing_embedder_inputs(monkeypatch):
    row = {
        SOURCE_ROW_INDEX_COL: 0,
        "gh_project_name": "owner/repo",
        "git_trigger_commit": "abc123",
        "tr_status": "passed",
    }
    monkeypatch.setattr(
        process_module,
        "build_commit_message_artifact",
        lambda repo_path, commit: TextArtifact(
            text="message text",
            provenance={"source_type": "commit_message"},
        ),
    )
    monkeypatch.setattr(process_module, "get_changed_files", lambda repo_path, commit: ["a.py"])
    monkeypatch.setattr(
        process_module,
        "build_diff_artifact",
        lambda repo_path, commit_sha, changed_files: TextArtifact(
            text="diff text",
            provenance={"source_type": "diff"},
        ),
    )
    monkeypatch.setattr(
        process_module,
        "build_context_artifact",
        lambda repo_path, commit_sha, changed_files: TextArtifact(
            text="context text",
            provenance={"source_type": "context"},
        ),
    )

    sample = process_module.build_raw_sample_from_row(row, Path("."))
    seen_batches = []

    class FakeEmbedder:
        def embed_texts(self, texts, batch_size):
            seen_batches.append(list(texts))
            return [texts[0]]

    class FakeWriter:
        def add(self, record):
            pass

    process_module.embed_and_write_raw_batch([sample], FakeEmbedder(), FakeWriter())

    assert seen_batches == [["message text"], ["diff text"], ["context text"]]
    assert sample.commit_message.provenance["source_type"] == "commit_message"
    assert sample.diff.provenance["source_type"] == "diff"
    assert sample.context.provenance["source_type"] == "context"
