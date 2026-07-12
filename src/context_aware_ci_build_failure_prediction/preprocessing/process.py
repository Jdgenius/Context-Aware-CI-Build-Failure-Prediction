import pandas as pd
import torch
import gc

from typing import Any
from pathlib import Path
from tqdm import tqdm
from collections.abc import Mapping

from .modules.failure_logger import JsonlLogger
from .modules.shard_writer import EmbeddingShardWriter
from .modules.repo_manager import DEFAULT_REPO_COL, DEFAULT_COMMIT_COL, DEFAULT_LABEL_COL
from .modules.repo_manager import TempRepoManager
from .helpers.embedding import CodeBERTEmbedder
from .helpers.git_extraction import (
    build_commit_message_artifact,
    build_diff_artifact,
    get_changed_files,
)
from .helpers.context_extraction import build_context_artifact
from .types import (
    DEFAULT_BUILD_ID_COL,
    DEFAULT_PARENT_COMMIT_COL,
    SOURCE_ROW_INDEX_COL,
    RawSample,
    make_sample_id,
    normalize_optional_value,
)

def build_raw_sample_from_row(
    row: Mapping[str, Any],
    repo_path: Path,
    repo_col: str = DEFAULT_REPO_COL,
    commit_col: str = DEFAULT_COMMIT_COL,
    label_col: str = DEFAULT_LABEL_COL,
    build_id_col: str | None = DEFAULT_BUILD_ID_COL,
    parent_commit_col: str | None = DEFAULT_PARENT_COMMIT_COL,
) -> RawSample:
    repo_name = str(row[repo_col])
    commit_sha = str(row[commit_col])
    source_row_index = int(row[SOURCE_ROW_INDEX_COL])
    build_id = (
        normalize_optional_value(row.get(build_id_col))
        if build_id_col is not None
        else None
    )
    parent_commit_sha = (
        normalize_optional_value(row.get(parent_commit_col))
        if parent_commit_col is not None
        else None
    )
    sample_id = make_sample_id(
        repo=repo_name,
        commit_sha=commit_sha,
        build_id=build_id,
        source_row_index=source_row_index,
    )

    commit_message = build_commit_message_artifact(repo_path, commit_sha)
    changed_files = get_changed_files(repo_path, commit_sha)

    diff = build_diff_artifact(
        repo_path=repo_path,
        commit_sha=commit_sha,
        changed_files=changed_files
    )

    context = build_context_artifact(
        repo_path=repo_path,
        commit_sha=commit_sha,
        changed_files=changed_files
    )

    label = row.get(label_col)

    return RawSample(
        sample_id=sample_id,
        source_row_index=source_row_index,
        repo=repo_name,
        commit_sha=commit_sha,
        parent_commit_sha=parent_commit_sha,
        build_id=build_id,
        label=label,
        commit_message=commit_message,
        diff=diff,
        context=context,
    )

def embed_and_write_raw_batch(
    raw_buffer: list[RawSample],
    embedder: CodeBERTEmbedder,
    writer: EmbeddingShardWriter,
    embed_batch_size: int = 32
) -> None:
    messages = [r.commit_message.text or "" for r in raw_buffer]
    diffs = [r.diff.text or "" for r in raw_buffer]
    contexts = [r.context.text or "" for r in raw_buffer]

    message_embeddings, message_tokenization = embedder.embed_texts_with_metadata(
        messages,
        batch_size=embed_batch_size
    )

    diff_embeddings, diff_tokenization = embedder.embed_texts_with_metadata(
        diffs,
        batch_size=embed_batch_size
    )

    context_embeddings, context_tokenization = embedder.embed_texts_with_metadata(
        contexts,
        batch_size=embed_batch_size
    )

    for i, raw in enumerate(raw_buffer):
        raw.commit_message.tokenization = message_tokenization[i]
        raw.diff.tokenization = diff_tokenization[i]
        raw.context.tokenization = context_tokenization[i]
        writer.add({
            "repo": raw.repo,
            "commit_sha": raw.commit_sha,
            "message_embedding": message_embeddings[i],
            "diff_embedding": diff_embeddings[i],
            "context_embedding": context_embeddings[i],
            "label": raw.label,
        })

def process_one_repo_to_embeddings(
    repo_name: str,
    repo_df: pd.DataFrame,
    repo_manager: TempRepoManager,
    embedder: CodeBERTEmbedder,
    writer: EmbeddingShardWriter,
    failure_logger: JsonlLogger,
    repo_col: str = DEFAULT_REPO_COL,
    commit_col: str = DEFAULT_COMMIT_COL,
    label_col: str = DEFAULT_LABEL_COL,
    build_id_col: str | None = DEFAULT_BUILD_ID_COL,
    parent_commit_col: str | None = DEFAULT_PARENT_COMMIT_COL,
    embed_batch_size: int = 32,
    raw_batch_size: int = 64
) -> None:
    """
    Processes one repository:

    1. Partial clone repo.
    2. Fetch needed commits.
    3. Build raw text strings temporarily.
    4. Batch embed with CodeBERT.
    5. Save embeddings to shards.
    6. Delete cloned repo.
    """

    repo_path = None

    try:
        repo_path = repo_manager.partial_clone(repo_name)

        rows: list[dict[str, Any]] = [
            {str(k): v for k, v in row.to_dict().items()}
            for _, row in repo_df.iterrows()
        ]
        raw_buffer: list[RawSample] = []

        for row in tqdm(rows, desc=f"Extracting raw samples for {repo_name}"):
            commit_sha = row[commit_col]

            try:
                repo_manager.fetch_commit(repo_path, commit_sha)

                raw_sample = build_raw_sample_from_row(
                    row=row,
                    repo_path=repo_path,
                    repo_col=repo_col,
                    commit_col=commit_col,
                    label_col=label_col,
                    build_id_col=build_id_col,
                    parent_commit_col=parent_commit_col,
                )

                raw_buffer.append(raw_sample)

                if len(raw_buffer) >= raw_batch_size:
                    embed_and_write_raw_batch(
                        raw_buffer=raw_buffer,
                        embedder=embedder,
                        writer=writer,
                        embed_batch_size=embed_batch_size
                    )
                    raw_buffer.clear()
                    gc.collect()

            except Exception as e:
                failure_logger.write({
                    "stage": "sample_processing",
                    "repo": repo_name,
                    "commit_sha": commit_sha,
                    "error": str(e)
                })

        if raw_buffer:
            embed_and_write_raw_batch(
                raw_buffer=raw_buffer,
                embedder=embedder,
                writer=writer,
                embed_batch_size=embed_batch_size
            )
            raw_buffer.clear()
            gc.collect()

    except Exception as e:
        failure_logger.write({
            "stage": "repo_processing",
            "repo": repo_name,
            "error": str(e)
        })

    finally:
        # This is the key storage-saving step.
        repo_manager.delete_repo(repo_name)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        gc.collect()
