import re

from pathlib import Path

from .git_extraction import get_file_diff, get_file_content_after_commit
from ..modules.repo_manager import CONTEXT_SEP, FIELD_SEP

def extract_changed_new_line_numbers(diff_text: str) -> list[int]:
    """
    Extracts added/modified line numbers from the new side of a unified diff.
    """
    changed_lines = []
    current_new_line = None

    hunk_header_pattern = re.compile(
        r"@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@"
    )

    for line in diff_text.splitlines():
        match = hunk_header_pattern.match(line)

        if match:
            current_new_line = int(match.group(1))
            continue

        if current_new_line is None:
            continue

        if line.startswith("+++"):
            continue

        if line.startswith("+"):
            changed_lines.append(current_new_line)
            current_new_line += 1
        elif line.startswith("-"):
            continue
        else:
            current_new_line += 1

    return changed_lines


def extract_global_context(
    source_lines: list[str],
    changed_line_number: int,
    max_lines: int = 75
) -> str:
    index = changed_line_number - 1

    if index < 0 or index >= len(source_lines):
        return ""

    half_window = max_lines // 2
    start = max(0, index - half_window)
    end = min(len(source_lines), start + max_lines)

    return "\n".join(source_lines[start:end])


def find_brace_based_function_context(
    source_lines: list[str],
    changed_line_number: int,
    max_global_lines: int = 75,
    max_function_lines: int = 200
) -> str:
    """
    Heuristic function/method extractor for Java-like languages.
    For Ruby or hard cases, falls back to local/global window.
    """
    index = changed_line_number - 1

    if index < 0 or index >= len(source_lines):
        return ""

    function_signature_pattern = re.compile(
        r"""
        ^\s*
        (
            public|private|protected|static|final|synchronized|abstract|
            void|int|long|double|float|boolean|char|String|
            def|function|
            [A-Za-z_][A-Za-z0-9_<>\[\]]*
        )
        [\w<>\[\],\s]*          
        \s+
        [A-Za-z_][A-Za-z0-9_]*\s*
        \([^)]*\)
        \s*
        (\{|$)
        """,
        re.VERBOSE
    )

    function_start = None

    for i in range(index, max(-1, index - max_function_lines), -1):
        if function_signature_pattern.search(source_lines[i]):
            function_start = i
            break

    if function_start is None:
        return extract_global_context(
            source_lines,
            changed_line_number,
            max_lines=max_global_lines
        )

    brace_balance = 0
    seen_open_brace = False
    function_end = function_start

    for j in range(function_start, min(len(source_lines), function_start + max_function_lines)):
        line = source_lines[j]

        brace_balance += line.count("{")
        brace_balance -= line.count("}")

        if "{" in line:
            seen_open_brace = True

        if seen_open_brace and brace_balance <= 0:
            function_end = j
            break

    snippet = source_lines[function_start:function_end + 1]

    if not snippet:
        return extract_global_context(
            source_lines,
            changed_line_number,
            max_lines=max_global_lines
        )

    return "\n".join(snippet)


def build_context_string(
    repo_path: Path,
    commit_sha: str,
    changed_files: list[str],
    max_changed_lines_per_file: int = 20,
    max_context_chars_per_snippet: int = 20_000,
    max_total_context_chars: int = 150_000
) -> str:
    context_snippets = []
    total_chars = 0

    for file_path in changed_files:
        try:
            diff_text = get_file_diff(repo_path, commit_sha, file_path, unified_lines=0)
        except Exception:
            continue

        changed_line_numbers = extract_changed_new_line_numbers(diff_text)

        if not changed_line_numbers:
            continue

        # Limit extreme commits.
        changed_line_numbers = changed_line_numbers[:max_changed_lines_per_file]

        source_text = get_file_content_after_commit(repo_path, commit_sha, file_path)

        if source_text is None:
            continue

        source_lines = source_text.splitlines()

        for line_number in changed_line_numbers:
            context = find_brace_based_function_context(
                source_lines,
                line_number
            )

            if not context.strip():
                continue

            context = context[:max_context_chars_per_snippet]

            snippet = f"{file_path}:{line_number}{FIELD_SEP}{context}"
            context_snippets.append(snippet)

            total_chars += len(snippet)

            if total_chars >= max_total_context_chars:
                return CONTEXT_SEP.join(context_snippets)

    return CONTEXT_SEP.join(context_snippets)