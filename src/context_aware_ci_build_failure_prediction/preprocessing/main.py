import pandas as pd

from collections.abc import Iterator
from .helpers.embedding import CodeBERTEmbedder
from .modules.repo_manager import DEFAULT_REPO_COL, DEFAULT_COMMIT_COL, DEFAULT_LABEL_COL, TempRepoManager
from .modules.shard_writer import EmbeddingShardWriter
from .modules.failure_logger import JsonlLogger
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
    max_repos: int | None = None
) -> None:
    """
    Main entry point.

    This assumes the CSV fits in memory. TravisTorrent metadata likely should.
    If not, you can later split by repo using chunks.
    """
    print("Reading CSV...")
    df = pd.read_csv(travistorrent_csv_path)
    add_source_row_index(df)

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
    writer = EmbeddingShardWriter(
        output_dir=output_dir,
        shard_size=shard_size
    )
    failure_logger = JsonlLogger(failure_log_path)


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
                embed_batch_size=embed_batch_size,
                raw_batch_size=raw_batch_size
            )

    finally:
        writer.close()
