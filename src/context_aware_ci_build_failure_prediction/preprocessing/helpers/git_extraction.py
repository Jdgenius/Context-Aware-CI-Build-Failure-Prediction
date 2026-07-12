from pathlib import Path
import re
from typing import Any

from ..modules.repo_manager import run_cmd, DIFF_FILE_SEP, FIELD_SEP
from ..types import TextArtifact


HUNK_HEADER_RE = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)

def get_commit_message(repo_path: Path, commit_sha: str) -> str:
    return build_commit_message_artifact(repo_path, commit_sha).text


def build_commit_message_artifact(repo_path: Path, commit_sha: str) -> TextArtifact:
    message = run_cmd(
        ["git", "show", "-s", "--format=%B", commit_sha],
        cwd=repo_path
    ).strip()

    return TextArtifact(
        text=message,
        provenance={
            "source_type": "commit_message",
            "original_char_count": len(message),
            "retained_char_count": len(message),
            "extraction_truncated": False,
        },
    )


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
    return build_diff_artifact(
        repo_path=repo_path,
        commit_sha=commit_sha,
        changed_files=changed_files,
        max_diff_chars_per_file=max_diff_chars_per_file,
        max_total_diff_chars=max_total_diff_chars,
    ).text


def build_diff_artifact(
    repo_path: Path,
    commit_sha: str,
    changed_files: list[str],
    max_diff_chars_per_file: int = 20_000,
    max_total_diff_chars: int = 100_000,
) -> TextArtifact:
    snippets: list[str] = []
    retained_files: list[dict[str, Any]] = []
    total_chars = 0
    extraction_truncated = False
    change_metadata_by_path = _get_change_metadata_by_path(repo_path, commit_sha)

    for file_path in changed_files:
        try:
            diff_text = get_file_diff(repo_path, commit_sha, file_path, unified_lines=0)
        except Exception:
            continue

        if not diff_text.strip():
            continue

        original_char_count = len(diff_text)
        retained_diff_text = diff_text[:max_diff_chars_per_file]
        file_truncated = original_char_count > len(retained_diff_text)
        extraction_truncated = extraction_truncated or file_truncated

        snippet = f"{file_path}{FIELD_SEP}{retained_diff_text}"
        snippet_start_offset = _joined_next_offset(snippets, DIFF_FILE_SEP)
        diff_start_offset = snippet_start_offset + len(file_path) + len(FIELD_SEP)
        diff_end_offset = diff_start_offset + len(retained_diff_text)
        snippet_end_offset = snippet_start_offset + len(snippet)
        snippets.append(snippet)

        metadata = _parse_diff_metadata(
            file_path=file_path,
            diff_text=retained_diff_text,
            snippet_start_offset=snippet_start_offset,
            diff_start_offset=diff_start_offset,
            diff_end_offset=diff_end_offset,
            snippet_end_offset=snippet_end_offset,
        )
        metadata.update(change_metadata_by_path.get(file_path, {}))
        metadata.update(
            {
                "retained_path": file_path,
                "original_char_count": original_char_count,
                "retained_char_count": len(retained_diff_text),
                "truncated": file_truncated,
            }
        )
        retained_files.append(metadata)

        total_chars += len(snippet)

        if total_chars >= max_total_diff_chars:
            extraction_truncated = True
            break

    text = DIFF_FILE_SEP.join(snippets)
    return TextArtifact(
        text=text,
        provenance={
            "source_type": "diff",
            "files_found": len(changed_files),
            "files_retained": len(retained_files),
            "original_char_count": sum(
                file_metadata["original_char_count"] for file_metadata in retained_files
            ),
            "retained_char_count": len(text),
            "extraction_truncated": extraction_truncated,
            "retained_file_order": [file_metadata["retained_path"] for file_metadata in retained_files],
            "files": retained_files,
        },
    )


def _joined_next_offset(existing_snippets: list[str], separator: str) -> int:
    if not existing_snippets:
        return 0

    return sum(len(snippet) for snippet in existing_snippets) + (
        len(separator) * len(existing_snippets)
    )


def _get_change_metadata_by_path(repo_path: Path, commit_sha: str) -> dict[str, dict[str, Any]]:
    try:
        output = run_cmd(
            ["git", "show", "--name-status", "--format=", commit_sha],
            cwd=repo_path,
        )
    except Exception:
        return {}

    metadata_by_path: dict[str, dict[str, Any]] = {}

    for line in output.splitlines():
        if not line.strip():
            continue

        parts = line.split("\t")
        status = parts[0]

        if status.startswith("R") and len(parts) >= 3:
            old_path = parts[1]
            new_path = parts[2]
            metadata_by_path[new_path] = {
                "old_path": old_path,
                "new_path": new_path,
                "change_type": "renamed",
            }
        elif status == "A" and len(parts) >= 2:
            path = parts[1]
            metadata_by_path[path] = {"new_path": path, "change_type": "added"}
        elif status == "D" and len(parts) >= 2:
            path = parts[1]
            metadata_by_path[path] = {"old_path": path, "change_type": "deleted"}
        elif status == "M" and len(parts) >= 2:
            path = parts[1]
            metadata_by_path[path] = {
                "old_path": path,
                "new_path": path,
                "change_type": "modified",
            }

    return metadata_by_path


def _parse_diff_metadata(
    file_path: str,
    diff_text: str,
    snippet_start_offset: int,
    diff_start_offset: int,
    diff_end_offset: int,
    snippet_end_offset: int,
) -> dict[str, Any]:
    old_path: str | None = None
    new_path: str | None = None
    change_type: str | None = None
    is_binary = False

    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parsed_old_path, parsed_new_path = _parse_diff_git_paths(line)
            old_path = parsed_old_path
            new_path = parsed_new_path
        elif line.startswith("new file mode"):
            change_type = "added"
        elif line.startswith("deleted file mode"):
            change_type = "deleted"
        elif line.startswith("rename from "):
            old_path = line.removeprefix("rename from ").strip() or old_path
            change_type = "renamed"
        elif line.startswith("rename to "):
            new_path = line.removeprefix("rename to ").strip() or new_path
            change_type = "renamed"
        elif line.startswith("Binary files ") or line.startswith("GIT binary patch"):
            is_binary = True

    if change_type is None and (old_path is not None or new_path is not None):
        change_type = "modified"

    hunk_ranges = _parse_retained_hunks(diff_text, diff_start_offset)
    return {
        "old_path": old_path,
        "new_path": new_path,
        "change_type": change_type,
        "binary": is_binary,
        "retained_char_count": len(diff_text),
        "truncated": None,
        "hunk_ranges": None if is_binary and not hunk_ranges else hunk_ranges,
        "added_line_numbers": None
        if is_binary and not hunk_ranges
        else _flatten_hunk_numbers(hunk_ranges, "added_line_numbers"),
        "removed_line_numbers": None
        if is_binary and not hunk_ranges
        else _flatten_hunk_numbers(hunk_ranges, "removed_line_numbers"),
        "start_offset": snippet_start_offset,
        "end_offset": snippet_end_offset,
        "diff_start_offset": diff_start_offset,
        "diff_end_offset": diff_end_offset,
        "path": file_path,
    }


def _parse_diff_git_paths(line: str) -> tuple[str | None, str | None]:
    remainder = line.removeprefix("diff --git ")
    parts = remainder.split(" ")

    if len(parts) < 2:
        return None, None

    return _strip_git_prefix(parts[0]), _strip_git_prefix(parts[1])


def _strip_git_prefix(path: str) -> str:
    if path.startswith("a/") or path.startswith("b/"):
        return path[2:]

    return path


def _parse_retained_hunks(diff_text: str, diff_start_offset: int) -> list[dict[str, Any]]:
    hunks: list[dict[str, Any]] = []
    current_hunk: dict[str, Any] | None = None
    current_old_line: int | None = None
    current_new_line: int | None = None
    offset = diff_start_offset

    for line in diff_text.splitlines(keepends=True):
        stripped_line = line.rstrip("\r\n")
        match = HUNK_HEADER_RE.match(stripped_line)

        if match:
            if current_hunk is not None:
                current_hunk["end_offset"] = offset
                hunks.append(current_hunk)

            old_start = int(match.group("old_start"))
            new_start = int(match.group("new_start"))
            old_count = int(match.group("old_count") or "1")
            new_count = int(match.group("new_count") or "1")
            current_hunk = {
                "old_start": old_start,
                "old_count": old_count,
                "new_start": new_start,
                "new_count": new_count,
                "added_line_numbers": [],
                "removed_line_numbers": [],
                "start_offset": offset,
                "end_offset": None,
            }
            current_old_line = old_start
            current_new_line = new_start
            offset += len(line)
            continue

        if current_hunk is None:
            offset += len(line)
            continue

        if stripped_line.startswith("+++ ") or stripped_line.startswith("--- "):
            offset += len(line)
            continue

        if stripped_line.startswith("+"):
            current_hunk["added_line_numbers"].append(current_new_line)
            if current_new_line is not None:
                current_new_line += 1
        elif stripped_line.startswith("-"):
            current_hunk["removed_line_numbers"].append(current_old_line)
            if current_old_line is not None:
                current_old_line += 1
        else:
            if current_old_line is not None:
                current_old_line += 1
            if current_new_line is not None:
                current_new_line += 1

        offset += len(line)

    if current_hunk is not None:
        current_hunk["end_offset"] = offset
        hunks.append(current_hunk)

    return hunks


def _flatten_hunk_numbers(hunks: list[dict[str, Any]], key: str) -> list[int]:
    numbers: list[int] = []
    for hunk in hunks:
        numbers.extend(number for number in hunk[key] if number is not None)

    return numbers
