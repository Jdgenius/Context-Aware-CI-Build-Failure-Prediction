from .main import process_travistorrent_to_codebert_embeddings

process_travistorrent_to_codebert_embeddings(
    travistorrent_csv_path="final-2017-01-25.csv",
    output_dir="./embedding_shards_test",
    temp_repo_root="./temp_repos_test",
    failure_log_path="./logs/failures_test.jsonl",
    repo_col="gh_project_name",
    commit_col="git_trigger_commit",
    label_col="tr_status",
    shard_size=100,
    raw_batch_size=8,
    embed_batch_size=8,
    max_repos=2
)