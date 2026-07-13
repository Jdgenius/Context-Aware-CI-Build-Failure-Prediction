import pandas as pd
import json
import time

from collections.abc import Iterator
from pathlib import Path
from .helpers.embedding import CodeBERTEmbedder
from .modules.repo_manager import DEFAULT_REPO_COL, DEFAULT_COMMIT_COL, DEFAULT_LABEL_COL, TempRepoManager
from .modules.shard_writer import EmbeddingShardWriter
from .modules.failure_logger import JsonlLogger
from .modules.manifest import (
    ManifestManager,
    build_dataset_metadata,
    build_preprocessing_metadata,
    prepare_output_dir,
    sha256_file,
)
from .process import process_one_repo_to_embeddings   
from .types import (
    DEFAULT_BUILD_ID_COL,
    DEFAULT_PARENT_COMMIT_COL,
    SOURCE_ROW_INDEX_COL,
)

def iter_repo_groups(
    df: pd.DataFrame,
    repo_col: str,
) -> Iterator[tuple[str, pd.DataFrame]]:
    for repo_name, repo_df in df.groupby(repo_col):
        if not isinstance(repo_name, str):
            raise TypeError(f"Expected repo name to be str, got {type(repo_name).__name__}")
        yield repo_name, repo_df

def add_source_row_index(df: pd.DataFrame) -> pd.DataFrame:
    df[SOURCE_ROW_INDEX_COL] = range(len(df))
    return df

def process_travistorrent_to_codebert_embeddings(
    travistorrent_csv_path: str,
    output_dir: str = "./embedding_shards",
    temp_repo_root: str = "./temp_repos",
    failure_log_path: str = "./logs/failures.jsonl",
    repo_col: str = DEFAULT_REPO_COL,
    commit_col: str = DEFAULT_COMMIT_COL,
    label_col: str = DEFAULT_LABEL_COL,
    build_id_col: str | None = DEFAULT_BUILD_ID_COL,
    parent_commit_col: str | None = DEFAULT_PARENT_COMMIT_COL,
    shard_size: int = 5000,
    raw_batch_size: int = 64,
    embed_batch_size: int = 32,
    max_diff_chars_per_file: int = 20_000,
    max_total_diff_chars: int = 100_000,
    max_changed_lines_per_file: int = 20,
    max_context_chars_per_snippet: int = 20_000,
    max_total_context_chars: int = 150_000,
    max_repos: int | None = None,
    overwrite: bool = False,
    repo_timing_log_path: str | None = None,
) -> None:
    """
    Main entry point.

    This assumes the CSV fits in memory. TravisTorrent metadata likely should.
    If not, you can later split by repo using chunks.
    """
    removed_paths = prepare_output_dir(output_dir, overwrite=overwrite)
    if removed_paths:
        print(
            "Removed generated preprocessing outputs: "
            f"{[path.name for path in removed_paths]}"
        )

    print("Reading CSV...")
    df = pd.read_csv(travistorrent_csv_path)
    add_source_row_index(df)
    source_csv_sha256 = (
        sha256_file(travistorrent_csv_path)
        if Path(travistorrent_csv_path).exists()
        else None
    )

    required_cols = [repo_col, commit_col]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(
                f"Column '{col}' not found. Available columns:\n{df.columns.tolist()}"
            )

    if label_col not in df.columns:
        print(
            f"Warning: label_col '{label_col}' not found. "
            f"Labels will be None."
        )

    for optional_name, optional_col in [
        ("build_id_col", build_id_col),
        ("parent_commit_col", parent_commit_col),
    ]:
        if optional_col is not None and optional_col not in df.columns:
            print(
                f"Warning: {optional_name} '{optional_col}' not found. "
                f"Values will be None."
            )

    print(f"Total samples in CSV: {len(df)}")
    print("Loading RepoManager, Embedder, and ShardWriter...")
    repo_manager = TempRepoManager(temp_repo_root=temp_repo_root)
    embedder = CodeBERTEmbedder()
    failed_sample_count = 0

    def increment_failed_sample_count() -> None:
        nonlocal failed_sample_count
        failed_sample_count += 1

    manifest_manager = ManifestManager(
        output_dir=output_dir,
        dataset=build_dataset_metadata(
            source_csv=travistorrent_csv_path,
            source_csv_sha256=source_csv_sha256,
            repo_col=repo_col,
            commit_col=commit_col,
            label_col=label_col,
            build_id_col=build_id_col,
            parent_commit_col=parent_commit_col,
        ),
        embedding=getattr(embedder, "metadata", {}),
        preprocessing=build_preprocessing_metadata(
            shard_size=shard_size,
            raw_batch_size=raw_batch_size,
            embed_batch_size=embed_batch_size,
            max_diff_chars_per_file=max_diff_chars_per_file,
            max_total_diff_chars=max_total_diff_chars,
            max_changed_lines_per_file=max_changed_lines_per_file,
            max_context_chars_per_snippet=max_context_chars_per_snippet,
            max_total_context_chars=max_total_context_chars,
        ),
        failed_sample_count=lambda: failed_sample_count,
    )
    writer = EmbeddingShardWriter(
        output_dir=output_dir,
        shard_size=shard_size,
        on_shard_complete=manifest_manager.record_completed_shard,
    )
    failure_logger = JsonlLogger(failure_log_path)
    if repo_timing_log_path is not None:
        Path(repo_timing_log_path).parent.mkdir(parents=True, exist_ok=True)


    df[repo_col] = df[repo_col].astype(str)
    df[commit_col] = df[commit_col].astype(str)
    grouped = iter_repo_groups(df, repo_col)

    if max_repos is not None:
        grouped = list(grouped)[:max_repos]

    try:
        for repo_name, repo_df in grouped:
            print(f"\nProcessing repo: {repo_name}")
            print(f"Samples: {len(repo_df)}")
            print("Processing repoto embeddings...")
            repo_started_at = time.monotonic()
            failed_before_repo = failed_sample_count
            process_one_repo_to_embeddings(
                repo_name=repo_name,
                repo_df=repo_df,
                repo_manager=repo_manager,
                embedder=embedder,
                writer=writer,
                failure_logger=failure_logger,
                repo_col=repo_col,
                commit_col=commit_col,
                label_col=label_col,
                build_id_col=build_id_col,
                parent_commit_col=parent_commit_col,
                max_diff_chars_per_file=max_diff_chars_per_file,
                max_total_diff_chars=max_total_diff_chars,
                max_changed_lines_per_file=max_changed_lines_per_file,
                max_context_chars_per_snippet=max_context_chars_per_snippet,
                max_total_context_chars=max_total_context_chars,
                embed_batch_size=embed_batch_size,
                raw_batch_size=raw_batch_size,
                on_sample_failure=increment_failed_sample_count,
            )
            if repo_timing_log_path is not None:
                failed_in_repo = failed_sample_count - failed_before_repo
                attempted_samples = len(repo_df)
                write_repo_timing_record(
                    path=repo_timing_log_path,
                    record={
                        "repo": repo_name,
                        "elapsed_seconds": time.monotonic() - repo_started_at,
                        "attempted_samples": attempted_samples,
                        "successful_samples": attempted_samples - failed_in_repo,
                        "failed_samples": failed_in_repo,
                    },
                )

    finally:
        writer.close()
        manifest_manager.finalize()


def write_repo_timing_record(path: str, record: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")
