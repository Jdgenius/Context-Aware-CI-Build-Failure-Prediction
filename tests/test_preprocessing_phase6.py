from __future__ import annotations

import json
import subprocess
import uuid
from pathlib import Path

import pandas as pd
import pytest
import torch

from context_aware_ci_build_failure_prediction.preprocessing import main as main_module
from context_aware_ci_build_failure_prediction.preprocessing.helpers.context_extraction import (
    build_context_string,
)
from context_aware_ci_build_failure_prediction.preprocessing.helpers.git_extraction import (
    build_diff_string,
    get_changed_files,
    get_commit_message,
)
from context_aware_ci_build_failure_prediction.preprocessing.modules.manifest import (
    load_and_validate_manifest,
)
from context_aware_ci_build_failure_prediction.preprocessing.modules.shard_writer import (
    iter_text_sidecar,
    load_embedding_shard,
)
from context_aware_ci_build_failure_prediction.preprocessing.types import make_sample_id


def workspace_path(name: str) -> Path:
    path = Path("embedding_shards_test") / "phase6" / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_git(repo_path: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {repo_path}:\n{result.stderr}"
        )
    return result.stdout


def write_text(repo_path: Path, relative_path: str, text: str) -> None:
    path = repo_path / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def commit_all(repo_path: Path, message: str) -> str:
    run_git(repo_path, "add", "-A")
    run_git(repo_path, "commit", "-m", message)
    return run_git(repo_path, "rev-parse", "HEAD").strip()


def build_integration_repo() -> tuple[Path, list[str]]:
    repo_path = workspace_path("repo")
    run_git(repo_path, "init")
    run_git(repo_path, "config", "user.email", "test@example.com")
    run_git(repo_path, "config", "user.name", "Test User")

    write_text(
        repo_path,
        "src/Example.java",
        "\n".join(
            [
                "public class Example {",
                "  public void target() {",
                "    int x = 1;",
                "    int y = 2;",
                "    System.out.println(x + y);",
                "  }",
                "}",
            ]
        )
        + "\n",
    )
    write_text(repo_path, "src/delete_me.txt", "delete me\n")
    base_commit = commit_all(repo_path, "base commit")

    write_text(
        repo_path,
        "src/Example.java",
        "\n".join(
            [
                "public class Example {",
                "  public void target() {",
                "    int x = 10;",
                "    int y = 20;",
                "    System.out.println(x + y);",
                "  }",
                "}",
            ]
        )
        + "\n",
    )
    write_text(repo_path, "src/added.txt", "added file\n")
    modified_commit = commit_all(repo_path, "modified two hunks and added file")

    (repo_path / "src/delete_me.txt").unlink()
    deleted_commit = commit_all(repo_path, "deleted file")

    long_message = "long message " + ("token " * 80)
    write_text(repo_path, "src/long.txt", "long context\n" * 30)
    long_commit = commit_all(repo_path, long_message)

    return repo_path, [base_commit, modified_commit, deleted_commit, long_commit]


class LocalRepoManager:
    def __init__(self, repo_path: Path):
        self._repo_path = repo_path

    def partial_clone(self, repo_name: str) -> Path:
        return self._repo_path

    def fetch_commit(self, repo_path: Path, commit_sha: str) -> None:
        run_git(repo_path, "cat-file", "-e", f"{commit_sha}^{{commit}}")

    def delete_repo(self, repo_name: str) -> None:
        return None


class RecordingEmbedder:
    instances: list["RecordingEmbedder"] = []

    def __init__(self):
        self.calls: list[list[str]] = []
        self.metadata = {
            "model_name": "recording-codebert",
            "embedding_dimension": 4,
            "max_length": 12,
            "pooling": "fake_mean_pooling",
            "output_dtype": "torch.float32",
        }
        RecordingEmbedder.instances.append(self)

    def embed_texts_with_metadata(self, texts: list[str], batch_size: int = 32):
        from context_aware_ci_build_failure_prediction.preprocessing.types import (
            TokenizationMetadata,
        )

        self.calls.append(list(texts))
        embeddings = []
        metadata = []
        for index, text in enumerate(texts):
            token_count = len(text.split()) + 2
            retained = min(token_count, self.metadata["max_length"])
            embeddings.append(
                torch.tensor(
                    [
                        len(text),
                        retained,
                        len(self.calls),
                        index,
                    ],
                    dtype=torch.float32,
                )
            )
            metadata.append(
                TokenizationMetadata(
                    token_count_before_truncation=token_count,
                    retained_token_count=retained,
                    was_tokenizer_truncated=token_count > retained,
                )
            )
        return torch.stack(embeddings), metadata


def create_csv(csv_path: Path, repo_name: str, commits: list[str]) -> pd.DataFrame:
    base_commit, modified_commit, deleted_commit, long_commit = commits
    rows = [
        {
            "gh_project_name": repo_name,
            "git_trigger_commit": modified_commit,
            "tr_status": 1,
            "tr_build_id": "build-0",
            "git_prev_built_commit": base_commit,
        },
        {
            "gh_project_name": repo_name,
            "git_trigger_commit": "missing-commit",
            "tr_status": 0,
            "tr_build_id": "build-bad",
            "git_prev_built_commit": modified_commit,
        },
        {
            "gh_project_name": repo_name,
            "git_trigger_commit": deleted_commit,
            "tr_status": 0,
            "tr_build_id": "build-1",
            "git_prev_built_commit": modified_commit,
        },
        {
            "gh_project_name": repo_name,
            "git_trigger_commit": long_commit,
            "tr_status": 1,
            "tr_build_id": "build-2",
            "git_prev_built_commit": deleted_commit,
        },
        {
            "gh_project_name": repo_name,
            "git_trigger_commit": modified_commit,
            "tr_status": 1,
            "tr_build_id": "build-3",
            "git_prev_built_commit": base_commit,
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    return df


@pytest.fixture
def integration_run(monkeypatch):
    repo_path, commits = build_integration_repo()
    output_dir = workspace_path("output")
    csv_path = workspace_path("csv") / "travistorrent_fixture.csv"
    repo_name = "local/repo"
    df = create_csv(csv_path, repo_name, commits)
    RecordingEmbedder.instances.clear()

    monkeypatch.setattr(
        main_module,
        "TempRepoManager",
        lambda *args, **kwargs: LocalRepoManager(repo_path),
    )
    monkeypatch.setattr(main_module, "CodeBERTEmbedder", RecordingEmbedder)

    failure_log_path = output_dir / "failures.jsonl"
    main_module.process_travistorrent_to_codebert_embeddings(
        travistorrent_csv_path=str(csv_path),
        output_dir=str(output_dir),
        temp_repo_root=str(output_dir / "temp_repos"),
        failure_log_path=str(failure_log_path),
        shard_size=2,
        raw_batch_size=2,
        embed_batch_size=2,
        max_context_chars_per_snippet=120,
        overwrite=True,
    )

    return {
        "repo_path": repo_path,
        "commits": commits,
        "output_dir": output_dir,
        "csv_path": csv_path,
        "df": df,
        "repo_name": repo_name,
        "embedder": RecordingEmbedder.instances[0],
        "failure_log_path": failure_log_path,
    }


def load_all_outputs(output_dir: Path):
    manifest = load_and_validate_manifest(output_dir / "manifest.json", verify_checksums=True)
    payloads = [
        load_embedding_shard(output_dir / shard["tensor_file"])
        for shard in manifest["shards"]
    ]
    sidecars = [
        list(iter_text_sidecar(output_dir / shard["text_file"]))
        for shard in manifest["shards"]
    ]
    return manifest, payloads, sidecars


def flatten(list_of_lists):
    return [item for items in list_of_lists for item in items]


def successful_rows(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["git_trigger_commit"] != "missing-commit"].copy()


def test_end_to_end_outputs_alignment_manifest_and_failure_log(integration_run):
    output_dir = integration_run["output_dir"]
    df = integration_run["df"]
    manifest, payloads, sidecars = load_all_outputs(output_dir)

    assert manifest["totals"] == {
        "successful_samples": 4,
        "failed_samples": 1,
        "num_shards": 2,
    }
    assert [payload["num_samples"] for payload in payloads] == [2, 2]
    assert [record["record_index"] for record in sidecars[0]] == [0, 1]
    assert [record["record_index"] for record in sidecars[1]] == [0, 1]

    flat_sidecars = flatten(sidecars)
    flat_sample_ids = flatten([payload["sample_ids"] for payload in payloads])
    expected_sample_ids = [
        make_sample_id(
            repo=row["gh_project_name"],
            commit_sha=row["git_trigger_commit"],
            build_id=row["tr_build_id"],
            source_row_index=int(index),
        )
        for index, row in successful_rows(df).iterrows()
    ]

    assert flat_sample_ids == expected_sample_ids
    assert [record["sample_id"] for record in flat_sidecars] == expected_sample_ids
    assert len(set(flat_sample_ids)) == 4
    assert all(payload["message_embeddings"].shape[1] == 4 for payload in payloads)
    assert all(payload["diff_embeddings"].shape[1] == 4 for payload in payloads)
    assert all(payload["context_embeddings"].shape[1] == 4 for payload in payloads)
    assert flatten([payload["labels"].tolist() for payload in payloads]) == [1.0, 0.0, 1.0, 1.0]

    failed_sample_id = make_sample_id(
        repo=integration_run["repo_name"],
        commit_sha="missing-commit",
        build_id="build-bad",
        source_row_index=1,
    )
    assert failed_sample_id not in flat_sample_ids
    assert failed_sample_id not in [record["sample_id"] for record in flat_sidecars]

    failure_records = [
        json.loads(line)
        for line in integration_run["failure_log_path"].read_text(encoding="utf-8").splitlines()
    ]
    assert len(failure_records) == 1
    failure = failure_records[0]
    assert failure["stage"] == "sample_processing"
    assert failure["source_row_index"] == 1
    assert failure["sample_id"] == failed_sample_id
    assert failure["build_id"] == "build-bad"
    assert "diff --git" not in failure["error"]


def test_embedding_inputs_equal_compatibility_wrappers_and_exclude_metadata(integration_run):
    repo_path = integration_run["repo_path"]
    df = integration_run["df"]
    embedder = integration_run["embedder"]
    successful = successful_rows(df)

    expected_messages = []
    expected_diffs = []
    expected_contexts = []
    for _, row in successful.iterrows():
        commit = row["git_trigger_commit"]
        changed_files = get_changed_files(repo_path, commit)
        expected_messages.append(get_commit_message(repo_path, commit))
        expected_diffs.append(build_diff_string(repo_path, commit, changed_files))
        expected_contexts.append(
            build_context_string(
                repo_path,
                commit,
                changed_files,
                max_context_chars_per_snippet=120,
            )
        )

    captured_messages = embedder.calls[0] + embedder.calls[3]
    captured_diffs = embedder.calls[1] + embedder.calls[4]
    captured_contexts = embedder.calls[2] + embedder.calls[5]

    assert captured_messages == expected_messages
    assert captured_diffs == expected_diffs
    assert captured_contexts == expected_contexts
    for text in captured_messages + captured_diffs + captured_contexts:
        assert "provenance" not in text
        assert "token_count_before_truncation" not in text


def test_sidecar_text_and_metadata_align_with_tensor_rows(integration_run):
    repo_path = integration_run["repo_path"]
    manifest, payloads, sidecars = load_all_outputs(integration_run["output_dir"])
    flat_payload_sample_ids = flatten([payload["sample_ids"] for payload in payloads])
    flat_sidecars = flatten(sidecars)

    assert manifest["embedding"]["model_name"] == "recording-codebert"
    assert manifest["preprocessing"]["shard_size"] == 2
    assert manifest["preprocessing"]["max_context_chars_per_snippet"] == 120

    for record in flat_sidecars:
        assert record["sample_id"] in flat_payload_sample_ids
        assert record["provenance"]["commit_message"]["extraction"]
        assert record["provenance"]["commit_message"]["tokenization"]
        assert record["provenance"]["diff"]["extraction"]
        assert record["provenance"]["diff"]["tokenization"]
        assert record["provenance"]["context"]["extraction"]
        assert record["provenance"]["context"]["tokenization"]

        for file_metadata in record["provenance"]["diff"]["extraction"].get("files", []):
            start = file_metadata["diff_start_offset"]
            end = file_metadata["diff_end_offset"]
            assert record["text"]["diff"][start:end]

    tensor_paths = [integration_run["output_dir"] / shard["tensor_file"] for shard in manifest["shards"]]
    sidecar_paths = [integration_run["output_dir"] / shard["text_file"] for shard in manifest["shards"]]
    sidecar_paths[0].write_bytes(b"corrupt sidecar")
    loaded = load_embedding_shard(tensor_paths[0])
    assert loaded["num_samples"] == 2
    with pytest.raises(Exception):
        load_and_validate_manifest(integration_run["output_dir"] / "manifest.json", verify_checksums=True)


def test_phase6_storage_estimate_from_integration_output(integration_run):
    output_dir = integration_run["output_dir"]
    manifest = load_and_validate_manifest(output_dir / "manifest.json")
    successful = manifest["totals"]["successful_samples"]
    tensor_bytes = sum((output_dir / shard["tensor_file"]).stat().st_size for shard in manifest["shards"])
    sidecar_bytes = sum((output_dir / shard["text_file"]).stat().st_size for shard in manifest["shards"])

    avg_tensor = tensor_bytes / successful
    avg_sidecar = sidecar_bytes / successful
    projected_100k = (avg_tensor + avg_sidecar) * 100_000
    projected_1m = (avg_tensor + avg_sidecar) * 1_000_000

    assert avg_tensor > 0
    assert avg_sidecar > 0
    assert projected_100k > tensor_bytes + sidecar_bytes
    assert projected_1m == pytest.approx(projected_100k * 10)
