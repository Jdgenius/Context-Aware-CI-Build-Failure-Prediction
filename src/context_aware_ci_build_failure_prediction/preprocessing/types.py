from __future__ import annotations

import hashlib
import numbers
from dataclasses import dataclass
from typing import Any

import pandas as pd


SOURCE_ROW_INDEX_COL = "__source_row_index"
DEFAULT_BUILD_ID_COL = "tr_build_id"
DEFAULT_PARENT_COMMIT_COL = "git_prev_built_commit"


@dataclass(frozen=True)
class TokenizationMetadata:
    token_count_before_truncation: int
    retained_token_count: int
    was_tokenizer_truncated: bool


@dataclass
class TextArtifact:
    text: str
    provenance: dict[str, Any]
    tokenization: TokenizationMetadata | None = None


@dataclass
class RawSample:
    sample_id: str
    source_row_index: int
    repo: str
    commit_sha: str
    parent_commit_sha: str | None
    build_id: str | None
    label: Any
    commit_message: TextArtifact
    diff: TextArtifact
    context: TextArtifact


def normalize_optional_value(value: Any) -> str | None:
    if value is None:
        return None

    try:
        if pd.isna(value):
            return None
    except Exception:
        pass

    try:
        if value != value:
            return None
    except Exception:
        pass

    if isinstance(value, str):
        normalized = value.strip()
        return normalized or None

    if isinstance(value, numbers.Real) and not isinstance(value, bool):
        if float(value).is_integer():
            return str(int(float(value)))

    return str(value)


def make_sample_id(
    repo: str,
    commit_sha: str,
    build_id: str | None,
    source_row_index: int,
) -> str:
    normalized_build_id = build_id or ""
    canonical = "\n".join(
        [
            repo,
            commit_sha,
            normalized_build_id,
            str(source_row_index),
        ]
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"
