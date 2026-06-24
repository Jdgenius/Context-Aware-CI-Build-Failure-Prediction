from pathlib import Path

from ..modules.repo_manager import run_cmd, DIFF_FILE_SEP, FIELD_SEP

def get_commit_message(repo_path: Path, commit_sha: str) -> str:
    return run_cmd(
        ["git", "show", "-s", "--format=%B", commit_sha],
        cwd=repo_path
    ).strip()


def get_changed_files(repo_path: Path, commit_sha: str) -> list[str]:
    output = run_cmd(
        ["git", "show", "--name-only", "--format=", commit_sha],
        cwd=repo_path
    )

    return [line.strip() for line in output.splitlines() if line.strip()]


def get_file_diff(
    repo_path: Path,
    commit_sha: str,
    file_path: str,
    unified_lines: int = 0
) -> str:
    return run_cmd(
        [
            "git",
            "show",
            "--format=",
            f"--unified={unified_lines}",
            commit_sha,
            "--",
            file_path
        ],
        cwd=repo_path
    ).strip()


def get_file_content_after_commit(
    repo_path: Path,
    commit_sha: str,
    file_path: str
) -> str | None:
    """
    Returns None for deleted/binary/unavailable files.
    """
    try:
        return run_cmd(
            ["git", "show", f"{commit_sha}:{file_path}"],
            cwd=repo_path
        )
    except Exception:
        return None


def build_diff_string(
    repo_path: Path,
    commit_sha: str,
    changed_files: list[str],
    max_diff_chars_per_file: int = 20_000,
    max_total_diff_chars: int = 100_000
) -> str:
    snippets = []
    total_chars = 0

    for file_path in changed_files:
        try:
            diff_text = get_file_diff(repo_path, commit_sha, file_path, unified_lines=0)
        except Exception:
            continue

        if not diff_text.strip():
            continue

        diff_text = diff_text[:max_diff_chars_per_file]

        snippet = f"{file_path}{FIELD_SEP}{diff_text}"
        snippets.append(snippet)

        total_chars += len(snippet)

        if total_chars >= max_total_diff_chars:
            break

    return DIFF_FILE_SEP.join(snippets)