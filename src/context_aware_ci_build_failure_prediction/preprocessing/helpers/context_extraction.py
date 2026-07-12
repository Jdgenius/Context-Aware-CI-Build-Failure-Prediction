import re

from pathlib import Path
from typing import Any

from .git_extraction import get_file_diff, get_file_content_after_commit
from ..modules.repo_manager import CONTEXT_SEP, FIELD_SEP
from ..types import TextArtifact

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


def extract_global_context_region(
    source_lines: list[str],
    changed_line_number: int,
    max_lines: int = 75
) -> dict[str, Any] | None:
    index = changed_line_number - 1

    if index < 0 or index >= len(source_lines):
        return None

    half_window = max_lines // 2
    start = max(0, index - half_window)
    end = min(len(source_lines), start + max_lines)
    text = "\n".join(source_lines[start:end])

    return {
        "text": text,
        "start_line": start + 1,
        "end_line": end,
        "symbol_name": None,
        "symbol_type": None,
        "extraction_method": "global_window",
    }


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


def find_brace_based_function_context_region(
    source_lines: list[str],
    changed_line_number: int,
    max_global_lines: int = 75,
    max_function_lines: int = 200
) -> dict[str, Any] | None:
    """
    Region-returning companion for find_brace_based_function_context.
    It intentionally follows the same heuristic path so emitted text stays unchanged.
    """
    index = changed_line_number - 1

    if index < 0 or index >= len(source_lines):
        return None

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
        (?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*
        \([^)]*\)
        \s*
        (\{|$)
        """,
        re.VERBOSE
    )

    function_start = None
    signature_match = None

    for i in range(index, max(-1, index - max_function_lines), -1):
        signature_match = function_signature_pattern.search(source_lines[i])
        if signature_match:
            function_start = i
            break

    if function_start is None:
        return extract_global_context_region(
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
        return extract_global_context_region(
            source_lines,
            changed_line_number,
            max_lines=max_global_lines
        )

    return {
        "text": "\n".join(snippet),
        "start_line": function_start + 1,
        "end_line": function_end + 1,
        "symbol_name": signature_match.group("name") if signature_match else None,
        "symbol_type": "function" if signature_match else None,
        "extraction_method": "brace_based_function",
    }


def build_context_string(
    repo_path: Path,
    commit_sha: str,
    changed_files: list[str],
    max_changed_lines_per_file: int = 20,
    max_context_chars_per_snippet: int = 20_000,
    max_total_context_chars: int = 150_000
) -> str:
    return build_context_artifact(
        repo_path=repo_path,
        commit_sha=commit_sha,
        changed_files=changed_files,
        max_changed_lines_per_file=max_changed_lines_per_file,
        max_context_chars_per_snippet=max_context_chars_per_snippet,
        max_total_context_chars=max_total_context_chars,
    ).text


def build_context_artifact(
    repo_path: Path,
    commit_sha: str,
    changed_files: list[str],
    max_changed_lines_per_file: int = 20,
    max_context_chars_per_snippet: int = 20_000,
    max_total_context_chars: int = 150_000
) -> TextArtifact:
    context_snippets: list[str] = []
    region_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    total_chars = 0
    extraction_truncated = False

    for file_path in changed_files:
        try:
            diff_text = get_file_diff(repo_path, commit_sha, file_path, unified_lines=0)
        except Exception:
            continue

        changed_line_numbers = extract_changed_new_line_numbers(diff_text)

        if not changed_line_numbers:
            continue

        # Limit extreme commits.
        if len(changed_line_numbers) > max_changed_lines_per_file:
            extraction_truncated = True
        changed_line_numbers = changed_line_numbers[:max_changed_lines_per_file]

        source_text = get_file_content_after_commit(repo_path, commit_sha, file_path)

        if source_text is None:
            continue

        source_lines = source_text.splitlines()

        for line_number in changed_line_numbers:
            region = find_brace_based_function_context_region(
                source_lines,
                line_number
            )

            if region is None:
                continue

            context = region["text"]

            if not context.strip():
                continue

            retained_context = context[:max_context_chars_per_snippet]
            snippet_truncated = len(context) > len(retained_context)
            extraction_truncated = extraction_truncated or snippet_truncated

            prefix = f"{file_path}:{line_number}"
            snippet = f"{prefix}{FIELD_SEP}{retained_context}"
            snippet_start_offset = _joined_next_offset(context_snippets, CONTEXT_SEP)
            context_start_offset = snippet_start_offset + len(prefix) + len(FIELD_SEP)
            context_end_offset = context_start_offset + len(retained_context)
            snippet_end_offset = snippet_start_offset + len(snippet)
            context_snippets.append(snippet)
            _merge_context_region(
                region_by_key=region_by_key,
                file_path=file_path,
                triggering_line=line_number,
                retained_context=retained_context,
                original_char_count=len(context),
                snippet_truncated=snippet_truncated,
                snippet_start_offset=snippet_start_offset,
                snippet_end_offset=snippet_end_offset,
                context_start_offset=context_start_offset,
                context_end_offset=context_end_offset,
                region=region,
            )

            total_chars += len(snippet)

            if total_chars >= max_total_context_chars:
                extraction_truncated = True
                return _context_artifact_from_parts(
                    context_snippets,
                    region_by_key,
                    extraction_truncated,
                )

    return _context_artifact_from_parts(
        context_snippets,
        region_by_key,
        extraction_truncated,
    )


def _joined_next_offset(existing_snippets: list[str], separator: str) -> int:
    if not existing_snippets:
        return 0

    return sum(len(snippet) for snippet in existing_snippets) + (
        len(separator) * len(existing_snippets)
    )


def _merge_context_region(
    region_by_key: dict[tuple[Any, ...], dict[str, Any]],
    file_path: str,
    triggering_line: int,
    retained_context: str,
    original_char_count: int,
    snippet_truncated: bool,
    snippet_start_offset: int,
    snippet_end_offset: int,
    context_start_offset: int,
    context_end_offset: int,
    region: dict[str, Any],
) -> None:
    key = (
        file_path,
        region["start_line"],
        region["end_line"],
        retained_context,
        region["extraction_method"],
    )
    occurrence = {
        "triggering_changed_line": triggering_line,
        "start_offset": snippet_start_offset,
        "end_offset": snippet_end_offset,
        "context_start_offset": context_start_offset,
        "context_end_offset": context_end_offset,
    }

    if key not in region_by_key:
        region_by_key[key] = {
            "file_path": file_path,
            "start_line": region["start_line"],
            "end_line": region["end_line"],
            "symbol_name": region["symbol_name"],
            "symbol_type": region["symbol_type"],
            "extraction_method": region["extraction_method"],
            "triggering_changed_lines": [triggering_line],
            "original_char_count": original_char_count,
            "retained_char_count": len(retained_context),
            "truncated": snippet_truncated,
            "occurrences": [occurrence],
        }
        return

    existing = region_by_key[key]
    existing["triggering_changed_lines"] = sorted(
        set(existing["triggering_changed_lines"] + [triggering_line])
    )
    existing["truncated"] = existing["truncated"] or snippet_truncated
    existing["occurrences"].append(occurrence)


def _context_artifact_from_parts(
    context_snippets: list[str],
    region_by_key: dict[tuple[Any, ...], dict[str, Any]],
    extraction_truncated: bool,
) -> TextArtifact:
    text = CONTEXT_SEP.join(context_snippets)
    regions = list(region_by_key.values())

    return TextArtifact(
        text=text,
        provenance={
            "source_type": "context",
            "regions_retained": len(regions),
            "original_char_count": sum(
                region["original_char_count"] for region in regions
            ),
            "retained_char_count": len(text),
            "extraction_truncated": extraction_truncated,
            "regions": regions,
        },
    )
