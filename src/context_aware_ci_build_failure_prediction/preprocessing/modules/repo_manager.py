from __future__ import annotations

import gc
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Iterable

import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel


# ============================================================
# Configuration
# ============================================================

MODEL_NAME = "microsoft/codebert-base"

DIFF_FILE_SEP = "\n<DIFF_FILE_SEP>\n"
FIELD_SEP = "\n<FIELD_SEP>\n"
CONTEXT_SEP = "\n<CONTEXT_SEP>\n"

DEFAULT_REPO_COL = "gh_project_name"
DEFAULT_COMMIT_COL = "git_trigger_commit"

# Adjust this to your TravisTorrent label column.
# Common possibilities might be "tr_status", "tr_result", etc.
DEFAULT_LABEL_COL = "tr_status"


# ============================================================
# Git utilities
# ============================================================

class GitCommandError(RuntimeError):
    pass


def run_cmd(
    args: list[str],
    cwd: Path | None = None,
    timeout: int = 300
) -> str:
    result = subprocess.run(
        args,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout
    )

    if result.returncode != 0:
        raise GitCommandError(
            f"Command failed:\n{' '.join(args)}\n\nSTDERR:\n{result.stderr}"
        )

    return result.stdout


def safe_repo_dir_name(repo_name: str) -> str:
    return repo_name.replace("/", "__")


class TempRepoManager:
    """
    Creates one temporary partial clone per repo, processes it,
    then deletes it after the repo is done.

    This is intentionally not a permanent cache.
    """

    def __init__(self, temp_repo_root: str = "./temp_repos"):
        self.temp_repo_root = Path(temp_repo_root)
        self.temp_repo_root.mkdir(parents=True, exist_ok=True)

    def repo_path(self, repo_name: str) -> Path:
        return self.temp_repo_root / safe_repo_dir_name(repo_name)

    def repo_url(self, repo_name: str) -> str:
        return f"https://github.com/{repo_name}.git"

    def partial_clone(self, repo_name: str) -> Path:
        """
        Partial clone with blob filtering.

        This avoids downloading all file contents upfront.
        """
        repo_path = self.repo_path(repo_name)

        if repo_path.exists():
            shutil.rmtree(repo_path)

        url = self.repo_url(repo_name)

        print(f"\nPartial cloning {repo_name}")

        run_cmd(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                url,
                str(repo_path)
            ],
            timeout=900
        )

        return repo_path

    def delete_repo(self, repo_name: str) -> None:
        repo_path = self.repo_path(repo_name)

        if repo_path.exists():
            print(f"Deleting temporary repo: {repo_path}")
            shutil.rmtree(repo_path, ignore_errors=True)

    def commit_exists_locally(self, repo_path: Path, commit_sha: str) -> bool:
        try:
            run_cmd(
                ["git", "cat-file", "-e", f"{commit_sha}^{{commit}}"],
                cwd=repo_path,
                timeout=60
            )
            return True
        except Exception:
            return False

    def fetch_commit(self, repo_path: Path, commit_sha: str) -> None:
        """
        Try to fetch only the needed commit first.
        Fallback to broader fetch if direct fetch fails.
        """
        if self.commit_exists_locally(repo_path, commit_sha):
            return

        try:
            run_cmd(
                ["git", "fetch", "origin", commit_sha, "--depth=1"],
                cwd=repo_path,
                timeout=600
            )
        except Exception:
            print(f"Direct fetch failed for {commit_sha}. Trying broader fetch.")
            run_cmd(
                ["git", "fetch", "origin", "--tags", "--prune"],
                cwd=repo_path,
                timeout=1200
            )

        if not self.commit_exists_locally(repo_path, commit_sha):
            raise GitCommandError(f"Could not fetch commit {commit_sha}")