# AWS GPU Preprocessing Run Guide

This guide runs the explainability-ready TravisTorrent preprocessing pipeline on a headless Linux GPU instance. It assumes the Phase 6 pipeline is already committed.

## 1. Environment

Use a Linux GPU image with NVIDIA drivers and Python 3.14 available. The project currently declares:

```text
requires-python = ">=3.14,<3.15"
```

If the base image does not provide Python 3.14, install it with a tool such as `pyenv` before running Poetry.

## 2. Clone

```bash
git clone <repo-url>
cd Context-Aware-CI-Build-Failure-Prediction
```

## 3. Install Poetry

```bash
python -m pip install --user pipx
python -m pipx ensurepath
pipx install poetry
```

Restart the shell if `poetry` is not on `PATH`.

## 4. Install Dependencies

```bash
poetry install
```

The lock file includes Linux CUDA-related PyTorch packages. Do not replace the installed PyTorch build with a CPU-only wheel.

## 5. Validate CUDA and Disk

```bash
poetry run python -m context_aware_ci_build_failure_prediction.preprocessing.cli environment-check \
  --travistorrent-csv-path /data/input/final-2017-01-25.csv \
  --output-dir /data/output/embedding_shards \
  --temp-repo-root /scratch/temp_repos \
  --require-cuda
```

For Hugging Face model downloads, use standard environment variables if needed:

```bash
export HF_HOME=/data/hf_cache
export TRANSFORMERS_CACHE=/data/hf_cache/transformers
```

Do not hard-code credentials or tokens in commands.

## 6. Place Data

Copy the TravisTorrent CSV to persistent storage, for example:

```text
/data/input/final-2017-01-25.csv
```

Keep final outputs on persistent storage, not only EC2 instance-store or temporary scratch disks.

## 7. Choose Output and Temp Directories

Recommended layout:

```text
/data/output/embedding_shards      final shard pairs and manifest
/scratch/temp_repos                temporary cloned repositories
/data/output/logs                  failure logs and run summaries
```

EC2 instance-store data is not necessarily persistent. Copy important outputs before stopping or terminating the instance.

## 8. One-Repository Verification

```bash
poetry run python -m context_aware_ci_build_failure_prediction.preprocessing.cli run \
  --travistorrent-csv-path /data/input/final-2017-01-25.csv \
  --output-dir /data/output/embedding_shards_smoke \
  --temp-repo-root /scratch/temp_repos_smoke \
  --failure-log-path /data/output/logs/failures_smoke.jsonl \
  --run-summary-path /data/output/logs/run_summary_smoke.json \
  --repo-timing-log-path /data/output/logs/repo_timing_smoke.jsonl \
  --shard-size 25 \
  --raw-batch-size 4 \
  --embed-batch-size 4 \
  --max-repos 1
```

Inspect:

```text
/data/output/embedding_shards_smoke/manifest.json
/data/output/logs/run_summary_smoke.json
/data/output/logs/failures_smoke.jsonl
```

## 9. Ten-Repository Benchmark

```bash
poetry run python -m context_aware_ci_build_failure_prediction.preprocessing.cli run \
  --travistorrent-csv-path /data/input/final-2017-01-25.csv \
  --output-dir /data/output/embedding_shards_10repo \
  --temp-repo-root /scratch/temp_repos_10repo \
  --failure-log-path /data/output/logs/failures_10repo.jsonl \
  --run-summary-path /data/output/logs/run_summary_10repo.json \
  --repo-timing-log-path /data/output/logs/repo_timing_10repo.jsonl \
  --shard-size 5000 \
  --raw-batch-size 64 \
  --embed-batch-size 32 \
  --max-repos 10
```

## 10. Larger Representative Benchmark

After the 10-repository benchmark succeeds:

```bash
poetry run python -m context_aware_ci_build_failure_prediction.preprocessing.cli run \
  --travistorrent-csv-path /data/input/final-2017-01-25.csv \
  --output-dir /data/output/embedding_shards_100repo \
  --temp-repo-root /scratch/temp_repos_100repo \
  --failure-log-path /data/output/logs/failures_100repo.jsonl \
  --run-summary-path /data/output/logs/run_summary_100repo.json \
  --repo-timing-log-path /data/output/logs/repo_timing_100repo.jsonl \
  --shard-size 5000 \
  --raw-batch-size 64 \
  --embed-batch-size 32 \
  --max-repos 100
```

## 11. Full Run

Only after representative benchmarks succeed:

```bash
poetry run python -m context_aware_ci_build_failure_prediction.preprocessing.cli run \
  --travistorrent-csv-path /data/input/final-2017-01-25.csv \
  --output-dir /data/output/embedding_shards \
  --temp-repo-root /scratch/temp_repos \
  --failure-log-path /data/output/logs/failures.jsonl \
  --run-summary-path /data/output/logs/run_summary.json \
  --repo-timing-log-path /data/output/logs/repo_timing.jsonl \
  --shard-size 5000 \
  --raw-batch-size 64 \
  --embed-batch-size 32
```

## 12. Outputs

Final preprocessing outputs:

```text
embedding_shards/
  manifest.json
  shard_00000.pt
  shard_00000.text.jsonl.gz
  ...
```

Operational logs:

```text
failures.jsonl
run_summary.json
repo_timing.jsonl
```

## 13. Stopping Safely

`Ctrl+C` or a normal termination signal should stop with a non-success status and attempt to write a run summary. Completed shard pairs remain protected by atomic write logic.

Full automatic resume is not implemented. For long runs, reliability still requires either deterministic partitioning into independent jobs or a later resume implementation.

## 14. Persistence Warning

Before stopping or terminating the instance, copy final outputs from instance-local storage to persistent storage such as EBS or S3. Do not keep important final outputs only in `/tmp`, `/scratch`, or instance-store volumes.
