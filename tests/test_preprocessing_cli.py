from __future__ import annotations

import json
import uuid
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest

from context_aware_ci_build_failure_prediction.preprocessing import cli


def workspace_path(name: str) -> Path:
    path = Path("embedding_shards_test") / "cli" / f"{name}-{uuid.uuid4().hex}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_csv() -> Path:
    path = workspace_path("csv") / "input.csv"
    path.write_text("gh_project_name,git_trigger_commit,tr_status\nrepo,abc,passed\n", encoding="utf-8")
    return path


def test_run_cli_forwards_arguments_and_writes_success_summary(monkeypatch):
    csv_path = write_csv()
    output_dir = workspace_path("output")
    summary_path = workspace_path("summary") / "summary.json"
    captured = {}

    def fake_process(**kwargs):
        captured.update(kwargs)
        (output_dir / "manifest.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(cli, "process_travistorrent_to_codebert_embeddings", fake_process)
    monkeypatch.setattr(
        cli,
        "load_and_validate_manifest",
        lambda path, verify_checksums=False: {
            "totals": {
                "successful_samples": 3,
                "failed_samples": 1,
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
            "--temp-repo-root",
            str(workspace_path("temp")),
            "--failure-log-path",
            str(workspace_path("logs") / "failures.jsonl"),
            "--repo-col",
            "repo",
            "--commit-col",
            "commit",
            "--label-col",
            "label",
            "--build-id-col",
            "build",
            "--parent-commit-col",
            "parent",
            "--shard-size",
            "7",
            "--raw-batch-size",
            "3",
            "--embed-batch-size",
            "2",
            "--max-repos",
            "5",
            "--overwrite",
            "--run-summary-path",
            str(summary_path),
            "--repo-timing-log-path",
            str(workspace_path("timing") / "timing.jsonl"),
        ]
    )

    assert exit_code == 0
    assert captured["travistorrent_csv_path"] == str(csv_path)
    assert captured["repo_col"] == "repo"
    assert captured["commit_col"] == "commit"
    assert captured["label_col"] == "label"
    assert captured["build_id_col"] == "build"
    assert captured["parent_commit_col"] == "parent"
    assert captured["shard_size"] == 7
    assert captured["raw_batch_size"] == 3
    assert captured["embed_batch_size"] == 2
    assert captured["max_repos"] == 5
    assert captured["overwrite"] is True
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "succeeded"
    assert summary["results"]["successful_samples"] == 3
    assert summary["results"]["failed_samples"] == 1
    assert summary["results"]["num_shards"] == 2


def test_run_cli_writes_failed_summary_and_returns_nonzero(monkeypatch):
    csv_path = write_csv()
    summary_path = workspace_path("summary") / "summary.json"

    def fake_process(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "process_travistorrent_to_codebert_embeddings", fake_process)

    exit_code = cli.main(
        [
            "run",
            "--travistorrent-csv-path",
            str(csv_path),
            "--output-dir",
            str(workspace_path("output")),
            "--temp-repo-root",
            str(workspace_path("temp")),
            "--failure-log-path",
            str(workspace_path("logs") / "failures.jsonl"),
            "--run-summary-path",
            str(summary_path),
        ]
    )

    assert exit_code == 1
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["exception"]["type"] == "RuntimeError"
    assert summary["exception"]["message"] == "boom"


def test_run_cli_writes_interrupted_summary(monkeypatch):
    csv_path = write_csv()
    summary_path = workspace_path("summary") / "summary.json"

    def fake_process(**kwargs):
        raise KeyboardInterrupt("stop")

    monkeypatch.setattr(cli, "process_travistorrent_to_codebert_embeddings", fake_process)

    exit_code = cli.main(
        [
            "run",
            "--travistorrent-csv-path",
            str(csv_path),
            "--output-dir",
            str(workspace_path("output")),
            "--temp-repo-root",
            str(workspace_path("temp")),
            "--failure-log-path",
            str(workspace_path("logs") / "failures.jsonl"),
            "--run-summary-path",
            str(summary_path),
        ]
    )

    assert exit_code == 130
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary["status"] == "interrupted"
    assert summary["exception"]["type"] == "KeyboardInterrupt"


def test_input_path_validation_fails_before_preprocessing(monkeypatch):
    called = False

    def fake_process(**kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(cli, "process_travistorrent_to_codebert_embeddings", fake_process)

    exit_code = cli.main(
        [
            "run",
            "--travistorrent-csv-path",
            str(workspace_path("missing") / "missing.csv"),
            "--output-dir",
            str(workspace_path("output")),
        ]
    )

    assert exit_code == 1
    assert called is False


def test_environment_check_reports_disk_git_and_csv(monkeypatch, capsys):
    csv_path = write_csv()
    DiskUsage = namedtuple("usage", ["total", "used", "free"])
    monkeypatch.setattr(cli.shutil, "disk_usage", lambda path: DiskUsage(100, 40, 60))
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/git")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="git version 2.44.0\n"),
    )

    exit_code = cli.main(
        [
            "environment-check",
            "--travistorrent-csv-path",
            str(csv_path),
            "--output-dir",
            str(workspace_path("output")),
            "--temp-repo-root",
            str(workspace_path("temp")),
            "--json",
        ]
    )

    assert exit_code == 0
    report = json.loads(capsys.readouterr().out)
    assert report["git"]["available"] is True
    assert report["input_csv"]["readable"] is True
    assert report["disk"]["output_dir"]["free_bytes"] == 60


def test_environment_check_require_cuda_failure(monkeypatch):
    csv_path = write_csv()
    monkeypatch.setattr(cli.shutil, "which", lambda name: "/usr/bin/git")
    monkeypatch.setattr(
        cli.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="git version 2.44.0\n"),
    )
    monkeypatch.setattr(cli.torch.cuda, "is_available", lambda: False)

    exit_code = cli.main(
        [
            "environment-check",
            "--travistorrent-csv-path",
            str(csv_path),
            "--output-dir",
            str(workspace_path("output")),
            "--temp-repo-root",
            str(workspace_path("temp")),
            "--require-cuda",
            "--json",
        ]
    )

    assert exit_code == 1


def test_linux_absolute_paths_parse_and_forward(monkeypatch):
    captured = {}
    csv_path = write_csv().resolve()

    def fake_process(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(cli, "process_travistorrent_to_codebert_embeddings", fake_process)
    monkeypatch.setattr(cli, "ensure_parent_dirs", lambda args: None)

    exit_code = cli.main(
        [
            "run",
            "--travistorrent-csv-path",
            str(csv_path),
            "--output-dir",
            "/tmp/preprocessing-output",
            "--temp-repo-root",
            "/tmp/preprocessing-temp-repos",
            "--failure-log-path",
            "/tmp/preprocessing-logs/failures.jsonl",
            "--max-repos",
            "1",
        ]
    )

    assert exit_code == 0
    assert captured["output_dir"] == "/tmp/preprocessing-output"
    assert captured["temp_repo_root"] == "/tmp/preprocessing-temp-repos"
