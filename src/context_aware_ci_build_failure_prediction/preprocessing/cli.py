from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers

from .helpers.embedding import CodeBERTEmbedder
from .main import process_travistorrent_to_codebert_embeddings
from .modules.manifest import load_and_validate_manifest, write_manifest_atomic
from .modules.repo_manager import DEFAULT_COMMIT_COL, DEFAULT_LABEL_COL, DEFAULT_REPO_COL
from .types import DEFAULT_BUILD_ID_COL, DEFAULT_PARENT_COMMIT_COL


class GracefulTermination(KeyboardInterrupt):
    pass


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m context_aware_ci_build_failure_prediction.preprocessing.cli",
        description="Run explainability-ready TravisTorrent preprocessing.",
    )
    subparsers = parser.add_subparsers(dest="command")

    run_parser = subparsers.add_parser("run", help="Run preprocessing")
    add_common_paths(run_parser)
    add_preprocessing_args(run_parser)
    run_parser.add_argument("--run-summary-path", default=None)
    run_parser.add_argument("--repo-timing-log-path", default=None)

    env_parser = subparsers.add_parser("environment-check", help="Report environment readiness")
    add_common_paths(env_parser)
    env_parser.add_argument("--require-cuda", action="store_true")
    env_parser.add_argument("--json", action="store_true", help="Emit JSON only")

    add_common_paths(parser)
    add_preprocessing_args(parser)
    parser.add_argument("--run-summary-path", default=None)
    parser.add_argument("--repo-timing-log-path", default=None)
    return parser


def add_common_paths(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--travistorrent-csv-path", required=False)
    parser.add_argument("--output-dir", default="./embedding_shards")
    parser.add_argument("--temp-repo-root", default="./temp_repos")
    parser.add_argument("--failure-log-path", default="./logs/failures.jsonl")


def add_preprocessing_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo-col", default=DEFAULT_REPO_COL)
    parser.add_argument("--commit-col", default=DEFAULT_COMMIT_COL)
    parser.add_argument("--label-col", default=DEFAULT_LABEL_COL)
    parser.add_argument("--build-id-col", default=DEFAULT_BUILD_ID_COL)
    parser.add_argument("--parent-commit-col", default=DEFAULT_PARENT_COMMIT_COL)
    parser.add_argument("--shard-size", type=int, default=5000)
    parser.add_argument("--raw-batch-size", type=int, default=64)
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--max-diff-chars-per-file", type=int, default=20_000)
    parser.add_argument("--max-total-diff-chars", type=int, default=100_000)
    parser.add_argument("--max-changed-lines-per-file", type=int, default=20)
    parser.add_argument("--max-context-chars-per-snippet", type=int, default=20_000)
    parser.add_argument("--max-total-context-chars", type=int, default=150_000)
    parser.add_argument("--max-repos", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--resume", action="store_true")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    command = args.command or "run"

    if command == "environment-check":
        return environment_check_command(args)

    return run_command(args)


def environment_check_command(args: argparse.Namespace) -> int:
    report = collect_environment_report(
        travistorrent_csv_path=args.travistorrent_csv_path,
        output_dir=args.output_dir,
        temp_repo_root=args.temp_repo_root,
    )
    failures = []

    if not report["git"]["available"]:
        failures.append("Git is not installed or not on PATH")
    if args.travistorrent_csv_path and not report["input_csv"]["readable"]:
        failures.append(f"Input CSV is not readable: {args.travistorrent_csv_path}")
    if args.require_cuda and not report["cuda"]["available"]:
        failures.append("CUDA is required but unavailable to PyTorch")

    report["status"] = "failed" if failures else "ok"
    report["failures"] = failures

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print_environment_report(report)

    return 1 if failures else 0


def run_command(args: argparse.Namespace) -> int:
    if not args.travistorrent_csv_path:
        raise SystemExit("--travistorrent-csv-path is required")

    started_at = datetime.now(timezone.utc)
    start = time.monotonic()
    status = "failed"
    exception_info = None
    process_result: dict[str, Any] | None = None

    try:
        install_signal_handlers()
        if args.resume and args.overwrite:
            raise ValueError("--resume and --overwrite are mutually exclusive")
        validate_input_csv(args.travistorrent_csv_path)
        ensure_parent_dirs(args)
        process_result = process_travistorrent_to_codebert_embeddings(
            travistorrent_csv_path=args.travistorrent_csv_path,
            output_dir=args.output_dir,
            temp_repo_root=args.temp_repo_root,
            failure_log_path=args.failure_log_path,
            repo_col=args.repo_col,
            commit_col=args.commit_col,
            label_col=args.label_col,
            build_id_col=args.build_id_col,
            parent_commit_col=args.parent_commit_col,
            shard_size=args.shard_size,
            raw_batch_size=args.raw_batch_size,
            embed_batch_size=args.embed_batch_size,
            max_diff_chars_per_file=args.max_diff_chars_per_file,
            max_total_diff_chars=args.max_total_diff_chars,
            max_changed_lines_per_file=args.max_changed_lines_per_file,
            max_context_chars_per_snippet=args.max_context_chars_per_snippet,
            max_total_context_chars=args.max_total_context_chars,
            max_repos=args.max_repos,
            overwrite=args.overwrite,
            resume=args.resume,
            repo_timing_log_path=args.repo_timing_log_path,
        )
        status = "succeeded"
        return 0
    except KeyboardInterrupt as exc:
        status = "interrupted"
        exception_info = exception_payload(exc)
        traceback.print_exc()
        return 130
    except Exception as exc:
        exception_info = exception_payload(exc)
        traceback.print_exc()
        return 1
    finally:
        if args.run_summary_path:
            finished_at = datetime.now(timezone.utc)
            summary = build_run_summary(
                args=args,
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                elapsed_seconds=time.monotonic() - start,
                exception_info=exception_info,
                process_result=process_result,
            )
            write_manifest_atomic(args.run_summary_path, summary)


def validate_input_csv(path: str | Path) -> None:
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {csv_path}")
    with csv_path.open("rb"):
        pass


def ensure_parent_dirs(args: argparse.Namespace) -> None:
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.temp_repo_root).mkdir(parents=True, exist_ok=True)
    Path(args.failure_log_path).parent.mkdir(parents=True, exist_ok=True)
    if args.run_summary_path:
        Path(args.run_summary_path).parent.mkdir(parents=True, exist_ok=True)
    if args.repo_timing_log_path:
        Path(args.repo_timing_log_path).parent.mkdir(parents=True, exist_ok=True)


def collect_environment_report(
    travistorrent_csv_path: str | None,
    output_dir: str | Path,
    temp_repo_root: str | Path,
) -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    git_path = shutil.which("git")
    git_version = None
    if git_path:
        try:
            git_version = subprocess.run(
                ["git", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:
            git_version = None

    return {
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
        },
        "project": {
            "version": project_version(),
            "git_commit": current_git_commit(),
        },
        "torch": {
            "version": torch.__version__,
        },
        "transformers": {
            "version": transformers.__version__,
        },
        "cuda": {
            "available": cuda_available,
            "runtime_version": torch.version.cuda,
            "gpu_count": gpu_count,
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
        },
        "selected_preprocessing_device": "cuda" if cuda_available else "cpu",
        "disk": {
            "output_dir": disk_report(output_dir),
            "temp_repo_root": disk_report(temp_repo_root),
        },
        "git": {
            "available": git_path is not None,
            "path": git_path,
            "version": git_version,
        },
        "input_csv": input_csv_report(travistorrent_csv_path),
    }


def print_environment_report(report: dict[str, Any]) -> None:
    print(json.dumps(report, indent=2, sort_keys=True))


def build_run_summary(
    args: argparse.Namespace,
    status: str,
    started_at: datetime,
    finished_at: datetime,
    elapsed_seconds: float,
    exception_info: dict[str, str] | None,
    process_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_path = Path(args.output_dir) / "manifest.json"
    results = {
        "successful_samples": 0,
        "failed_samples": 0,
        "num_shards": 0,
        "successful_samples_per_second": None,
        "seconds_per_successful_sample": None,
    }
    if manifest_path.exists():
        manifest = load_and_validate_manifest(manifest_path, verify_checksums=False)
        totals = manifest["totals"]
        successful = totals["successful_samples"]
        results.update(totals)
        if successful:
            results["successful_samples_per_second"] = successful / elapsed_seconds
            results["seconds_per_successful_sample"] = elapsed_seconds / successful

    summary = {
        "status": status,
        "started_at_utc": isoformat_utc(started_at),
        "finished_at_utc": isoformat_utc(finished_at),
        "elapsed_seconds": elapsed_seconds,
        "environment": run_environment_summary(),
        "configuration": {
            "input_csv": args.travistorrent_csv_path,
            "output_dir": args.output_dir,
            "temp_repo_root": args.temp_repo_root,
            "max_repos": args.max_repos,
            "shard_size": args.shard_size,
            "raw_batch_size": args.raw_batch_size,
            "embed_batch_size": args.embed_batch_size,
        },
        "results": results,
    }
    if process_result and "resume" in process_result:
        summary["resume"] = process_result["resume"]
    if exception_info is not None:
        summary["exception"] = exception_info
    return summary


def run_environment_summary() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    gpu_count = torch.cuda.device_count() if cuda_available else 0
    return {
        "hostname": socket.gethostname(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cuda_available": cuda_available,
        "gpu_names": [torch.cuda.get_device_name(i) for i in range(gpu_count)],
    }


def input_csv_report(path: str | None) -> dict[str, Any]:
    if path is None:
        return {"path": None, "readable": False, "size_bytes": None}
    csv_path = Path(path)
    return {
        "path": str(csv_path),
        "readable": csv_path.is_file() and os.access(csv_path, os.R_OK),
        "size_bytes": csv_path.stat().st_size if csv_path.is_file() else None,
    }


def disk_report(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    probe = target if target.exists() else target.parent
    usage = shutil.disk_usage(probe)
    return {
        "path": str(target),
        "total_bytes": usage.total,
        "used_bytes": usage.used,
        "free_bytes": usage.free,
    }


def project_version() -> str | None:
    try:
        from importlib.metadata import version

        return version("context-aware-ci-build-failure-prediction")
    except Exception:
        return None


def current_git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return None


def exception_payload(exc: BaseException) -> dict[str, str]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
    }


def isoformat_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def install_signal_handlers() -> None:
    def handle_signal(signum, frame):
        raise GracefulTermination(f"Received signal {signum}")

    for signal_name in ("SIGTERM", "SIGINT"):
        if hasattr(signal, signal_name):
            signal.signal(getattr(signal, signal_name), handle_signal)


if __name__ == "__main__":
    raise SystemExit(main())
